"""
03_severity_engine.py
=======================
OmniCataract-X v3.0 — Phase 3: The Severity Engine.

Run this AFTER Phase 2's exit milestone has passed (a trained, converged
CataractDetectorDualHead checkpoint must exist).

WHAT THIS FILE COVERS (maps to the plan's Task 3.1 - 3.3):
  3.1  Freeze the Phase 2 backbone. Train a SupCon projection head on top
       of frozen features so cataract/normal embeddings organize into
       separable clusters. Extract final embeddings for all images.
  3.2  Generate the severity proxy from TWO independent signals:
         Proxy A: L2 distance from the "Normal" embedding centroid
         Proxy B: raw pathology-head logit magnitude (from Phase 2's model)
       Cross-validate them via Spearman correlation — report honestly,
       do not hide a weak correlation.
  3.3  Train the Severity Probe MLP (Huber loss, on frozen/detached
       features) against the normalized Proxy A target. Define
       Mild/Moderate/Severe as tertiles. Produce face-validity material
       for a manual (non-clinical) sanity check.

WHY EVERYTHING HERE OPERATES ON A FROZEN BACKBONE:
This is the single most important design decision carried over from the
plan. Severity has no real clinical ground truth — it is a self-generated
proxy. If gradients from severity training were allowed to reach the
shared backbone, a noisy proxy signal could quietly degrade the Phase 2
detection model that IS backed by real labels. Freezing the backbone
before any of this file's training happens makes that impossible by
construction, not by discipline.

EXIT MILESTONE (do not proceed to Phase 4 until this passes):
  SupCon embedding space visibly separates classes (checked via the
  centroid-distance distribution, not just assumed); both proxies computed
  and cross-correlated (Spearman rho reported, whatever its value is);
  Severity Probe trained and face-validity material generated for review.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Phase 2's model class is imported by loading that module directly, since
# these files are numbered (not valid Python module names) rather than
# packaged — see load_phase2_module() below.
import importlib.util


# ==========================================================================
# CONFIG
# ==========================================================================
DRIVE_ROOT = "/content/drive/MyDrive/OmniCataract-X"
CHECKPOINT_DIR_PHASE2 = f"{DRIVE_ROOT}/checkpoints/phase2"
CHECKPOINT_DIR_PHASE3 = f"{DRIVE_ROOT}/checkpoints/phase3"
RESULTS_DIR = f"{DRIVE_ROOT}/results/phase3"

SUPCON_PROJECTION_HIDDEN = 256
SUPCON_PROJECTION_OUT = 128
SUPCON_TEMPERATURE = 0.07
SUPCON_EPOCHS = 15
SUPCON_LR = 1e-3

SEVERITY_HIDDEN = 512
SEVERITY_EPOCHS = 40
SEVERITY_LR = 1e-3
HUBER_DELTA = 1.0

MIN_POSITIVES_PER_BATCH_WARNING = 2  # WHY: SupCon needs >=2 same-class samples per batch to form any positive pair


# ==========================================================================
# UTILITY — load Phase 2's model class without renaming that file
# ==========================================================================
def load_phase2_module(path: str = "02_core_detection_model.py"):
    """
    WHY this exists: the project's files are deliberately numbered
    (01_, 02_, 03_...) so a beginner can see execution order at a glance
    in a file browser — but numbered filenames aren't valid Python module
    names for a plain `import`. This loads the file directly by path
    instead, which is the standard workaround and keeps the numbering.
    """
    spec = importlib.util.spec_from_file_location("phase2_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ==========================================================================
# TASK 3.1 — SUPCON TRAINING ON FROZEN BACKBONE
# ==========================================================================
def freeze_backbone(model: nn.Module) -> None:
    """
    WHY: this is the stop-gradient boundary described throughout the plan.
    Called once, before any Phase 3 training starts. Everything downstream
    in this file (SupCon projection head, Severity Probe) trains on TOP OF
    these frozen features, never through them.
    """
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.backbone.eval()  # WHY also set eval mode: freezes BatchNorm/Dropout
                            # running statistics too, not just gradient flow.
    n_frozen = sum(p.numel() for p in model.backbone.parameters())
    print(f"[ok] Backbone frozen: {n_frozen:,} parameters will not receive gradients "
          f"for the remainder of Phase 3.")


class SupConProjectionHead(nn.Module):
    """
    Projects the frozen backbone's pooled features into a lower-dimensional
    space specifically shaped by the contrastive loss, then L2-normalizes
    so cosine similarity (used inside SupConLoss) is well-behaved.

    WHY a separate projection head rather than using raw backbone features
    directly: this is standard contrastive-learning practice (SimCLR,
    SupCon) — the projection head absorbs some of the contrastive-specific
    geometry, keeping the backbone's raw features more general-purpose.
    """

    def __init__(self, input_dim: int, hidden_dim: int = SUPCON_PROJECTION_HIDDEN,
                 output_dim: int = SUPCON_PROJECTION_OUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.net(features)
        return F.normalize(projected, p=2, dim=1)  # L2 normalize -> unit sphere


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020), operating directly
    on a batch of L2-normalized embeddings and their class labels — every
    same-label pair in the batch is a positive, every different-label pair
    is a negative. This is the "SupCon-Out" formulation, chosen because it
    works with an arbitrary batch composition rather than requiring two
    separately-augmented views per anchor image.

    WHY this matters for a small-batch, single-GPU setting (documented as a
    known limitation in the plan): with batch_size=16-32 on a T4, you may
    get very few positive pairs per batch if the cataract/normal class
    ratio is imbalanced. This class warns (not crashes) when that happens,
    since a batch with too few positives just contributes a near-zero,
    uninformative gradient rather than being a bug.
    """

    def __init__(self, temperature: float = SUPCON_TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        batch_size = embeddings.shape[0]

        labels = labels.contiguous().view(-1, 1)
        # positive mask: 1 where same label (including self, removed below)
        positive_mask = torch.eq(labels, labels.T).float().to(device)

        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        # WHY subtract the row max before exp: standard log-sum-exp numerical
        # stability trick — without it, exp() can overflow for large
        # similarity values, silently producing NaN losses.
        sim_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - sim_max.detach()

        # WHY exclude self-similarity: a sample should never count as its
        # own positive pair, or the loss trivially minimizes on the
        # diagonal and learns nothing about actual class structure.
        self_mask = torch.eye(batch_size, device=device)
        logits_mask = 1.0 - self_mask
        positive_mask = positive_mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        n_positives_per_sample = positive_mask.sum(dim=1)
        if (n_positives_per_sample < 1).any():
            n_zero_positive = int((n_positives_per_sample < 1).sum().item())
            if n_zero_positive > batch_size * 0.3:
                print(f"[warn] {n_zero_positive}/{batch_size} samples in this batch have "
                      f"ZERO same-class partners — SupCon gradient will be weak/uninformative "
                      f"for them. Consider a larger batch size or class-balanced sampling.")
        # WHY the clamp: avoid division by zero for samples with no positive
        # partner in this particular batch — their contribution becomes 0
        # rather than NaN.
        safe_denominator = torch.where(
            n_positives_per_sample < 1e-6,
            torch.ones_like(n_positives_per_sample),
            n_positives_per_sample,
        )

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / safe_denominator
        loss = -mean_log_prob_pos.mean()
        return loss


def train_supcon(model: nn.Module, projection_head: SupConProjectionHead,
                  dataloader: DataLoader, device: str,
                  epochs: int = SUPCON_EPOCHS, lr: float = SUPCON_LR) -> dict:
    """
    WHY only projection_head.parameters() go to the optimizer: the backbone
    is frozen (see freeze_backbone). Training only the projection head on
    top of fixed features is what keeps this phase fully decoupled from
    Phase 2's detection model.
    """
    model.to(device)
    model.eval()  # backbone + Phase-2 heads all in eval mode — only the new head trains
    projection_head.to(device)
    projection_head.train()

    optimizer = torch.optim.Adam(projection_head.parameters(), lr=lr)
    loss_fn = SupConLoss()

    history = []
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():  # WHY no_grad here specifically: backbone
                                    # forward must not build a graph at all,
                                    # not just have requires_grad=False on its
                                    # params — this saves memory too.
                outputs = model(images)
                features = outputs["features"]

            projected = projection_head(features)
            loss = loss_fn(projected, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "supcon_loss": avg_loss})
        print(f"[epoch {epoch}] SupCon loss: {avg_loss:.4f}")

    return {"history": history}


@torch.no_grad()
def extract_embeddings(model: nn.Module, projection_head: SupConProjectionHead,
                        dataloader: DataLoader, device: str) -> dict:
    """
    WHY this returns THREE things (projected embeddings, raw features, and
    detection logits) in one pass rather than three separate functions:
    running inference over the full dataset is the expensive part; doing it
    once and returning everything Phase 3.2 needs avoids three redundant
    forward passes over potentially thousands of images.
    """
    model.eval()
    projection_head.eval()

    all_projected, all_raw_features, all_det_logits, all_labels = [], [], [], []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        projected = projection_head(outputs["features"])

        all_projected.append(projected.cpu().numpy())
        all_raw_features.append(outputs["features"].cpu().numpy())
        all_det_logits.append(outputs["detection_logits"].cpu().numpy())
        all_labels.append(labels.numpy())

    return {
        "projected_embeddings": np.concatenate(all_projected, axis=0),
        "raw_features": np.concatenate(all_raw_features, axis=0),
        "detection_logits": np.concatenate(all_det_logits, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
    }


# ==========================================================================
# TASK 3.2 — SEVERITY PROXY GENERATION
# ==========================================================================
def compute_normal_centroid(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """WHY centroid, not medoid or any fancier estimator: the plan
    specifically calls for the mathematical center of the Normal cluster —
    simple, reproducible, and easy to explain/defend in a write-up."""
    normal_embeddings = embeddings[labels == 0]
    if len(normal_embeddings) == 0:
        raise ValueError("No 'Normal' (label=0) samples found — cannot compute a centroid. "
                          "Check your label encoding (0=normal, 1=cataract expected).")
    centroid = normal_embeddings.mean(axis=0)
    print(f"[ok] Normal centroid computed from {len(normal_embeddings)} normal-class embeddings.")
    return centroid


def compute_proxy_a_distance(embeddings: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Proxy A: raw (un-normalized) L2 distance from the Normal centroid, per image."""
    distances = np.linalg.norm(embeddings - centroid[None, :], axis=1)
    return distances


def compute_proxy_b_logit_magnitude(detection_logits: np.ndarray) -> np.ndarray:
    """
    Proxy B: the raw pre-sigmoid detection logit itself. A higher logit
    means the Phase 2 model is more confident this is a cataract case —
    used here as a second, architecturally independent severity-like
    signal (it comes from the supervised detection head, not the
    unsupervised SupCon embedding space).
    """
    return detection_logits


def cross_validate_proxies(proxy_a: np.ndarray, proxy_b: np.ndarray) -> dict:
    """
    WHY this function exists and is called unconditionally (not optionally):
    reporting this correlation honestly — including if it's weak — is what
    makes the severity claim defensible. A silently-omitted correlation
    check is exactly the kind of thing a careful reviewer (or your own
    future self) should be suspicious of.
    """
    rho, p_value = spearmanr(proxy_a, proxy_b)
    print(f"[ok] Spearman correlation between Proxy A (centroid distance) and "
          f"Proxy B (detection logit): rho={rho:.4f}, p={p_value:.4g}")

    if rho < 0.3:
        print("[REPORT HONESTLY] Correlation is weak (rho < 0.3). The two proxies "
              "may be capturing different things, or one/both may be unreliable. "
              "State this plainly in any write-up rather than picking whichever "
              "proxy looks better after the fact.")
    elif rho < 0.6:
        print("[note] Correlation is moderate. Reasonable but not strong evidence "
              "the two proxies agree — report the exact rho value, not just 'they agree'.")
    else:
        print("[ok] Correlation is reasonably strong — supports treating the proxy "
              "as capturing a coherent underlying signal, though this is still not "
              "clinical validation.")

    return {"spearman_rho": float(rho), "p_value": float(p_value)}


def normalize_proxy_within_positive(proxy_values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    WHY normalize using ONLY cataract-positive cases' min/max: normal cases
    should already sit near the low end of the distance scale by
    construction (they ARE the centroid the distance is measured from).
    Including them in the min/max fit would compress the useful dynamic
    range among the positive cases, which is where severity actually needs
    to be discriminative. Values are clipped to [0,1] after the affine
    transform since a normal-case value can occasionally fall outside the
    positive-only range.
    """
    positive_values = proxy_values[labels == 1]
    if len(positive_values) < 2:
        raise ValueError("Fewer than 2 positive cases — cannot compute a meaningful "
                          "normalization range. Check your label encoding and data.")

    p_min, p_max = positive_values.min(), positive_values.max()
    if p_max - p_min < 1e-8:
        print("[warn] Positive-case proxy values have near-zero spread — "
              "normalization will be degenerate (all values collapse near 0 or 1). "
              "This likely indicates the SupCon embedding space did not separate "
              "severity-relevant structure within the positive class.")
        p_max = p_min + 1e-8  # avoid a division-by-zero crash; the warning above is the real signal

    normalized = (proxy_values - p_min) / (p_max - p_min)
    normalized = np.clip(normalized, 0.0, 1.0)
    return normalized


# ==========================================================================
# TASK 3.3 — SEVERITY PROBE MLP
# ==========================================================================
class SeverityProbe(nn.Module):
    """
    Small MLP trained on top of FROZEN, DETACHED backbone features to
    predict the continuous severity proxy score.

    WHY input_dim defaults to the backbone's raw feature dimension (768 for
    ConvNeXt-Tiny) rather than the SupCon projection's 128-dim output: the
    raw features carry more information; the SupCon projection space is
    specifically shaped for the contrastive objective and is a narrower,
    more specialized view. Using raw features for the probe, while still
    relying on the SupCon-derived proxy as the *training target*, gives the
    probe the richest input while keeping the target grounded in the
    class-organized embedding geometry.
    """

    def __init__(self, input_dim: int, hidden_dim: int = SEVERITY_HIDDEN, dropout_p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # WHY detach() here, defensively, even though callers are expected
        # to pass already-frozen features: this is the stop-gradient
        # boundary made structurally impossible to violate by accident,
        # not just documented as a convention.
        features = features.detach()
        raw_output = self.net(features)
        return torch.sigmoid(raw_output).squeeze(-1)


def train_severity_probe(raw_features: np.ndarray, severity_targets: np.ndarray,
                          device: str, epochs: int = SEVERITY_EPOCHS,
                          lr: float = SEVERITY_LR, batch_size: int = 32,
                          val_fraction: float = 0.15, seed: int = 42) -> dict:
    """
    WHY this trains from pre-extracted numpy arrays rather than re-running
    the backbone: the backbone is frozen and features were already
    extracted once in extract_embeddings(). Re-running the full model
    forward pass again here would be wasted compute for zero benefit.
    """
    rng = np.random.RandomState(seed)
    n = len(raw_features)
    idx = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train = torch.tensor(raw_features[train_idx], dtype=torch.float32)
    y_train = torch.tensor(severity_targets[train_idx], dtype=torch.float32)
    X_val = torch.tensor(raw_features[val_idx], dtype=torch.float32)
    y_val = torch.tensor(severity_targets[val_idx], dtype=torch.float32)

    probe = SeverityProbe(input_dim=raw_features.shape[1]).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    huber = nn.HuberLoss(delta=HUBER_DELTA)

    history = []
    for epoch in range(epochs):
        probe.train()
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_train), batch_size):
            batch_idx = perm[i:i + batch_size]
            xb = X_train[batch_idx].to(device)
            yb = y_train[batch_idx].to(device)

            pred = probe(xb)
            loss = huber(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        probe.eval()
        with torch.no_grad():
            val_pred = probe(X_val.to(device)).cpu().numpy()
        val_correlation = np.corrcoef(val_pred, y_val.numpy())[0, 1] if len(val_pred) > 1 else float("nan")

        history.append({
            "epoch": epoch,
            "train_huber_loss": epoch_loss / max(n_batches, 1),
            "val_correlation": val_correlation,
        })
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"[epoch {epoch}] train_huber_loss={epoch_loss / max(n_batches, 1):.4f} "
                  f"val_correlation={val_correlation:.4f}")

    return {"probe": probe, "history": history, "val_idx": val_idx, "train_idx": train_idx}


def compute_tertile_thresholds(severity_scores: np.ndarray, labels: np.ndarray) -> dict:
    """
    WHY tertiles specifically, and WHY computed only on positive cases:
    the plan defines Mild/Moderate/Severe as statistical percentile cutoffs
    on the score distribution AMONG CATARACT-POSITIVE cases — this is
    explicitly not a clinical grading boundary, and the thresholds should
    say so wherever they're used downstream (app UI, README).
    """
    positive_scores = severity_scores[labels == 1]
    if len(positive_scores) < 3:
        raise ValueError("Need at least 3 positive cases to compute tertiles meaningfully.")

    mild_upper = float(np.percentile(positive_scores, 33.33))
    moderate_upper = float(np.percentile(positive_scores, 66.67))

    print(f"[ok] Tertile thresholds computed from {len(positive_scores)} positive cases: "
          f"Mild <= {mild_upper:.4f} < Moderate <= {moderate_upper:.4f} < Severe")

    return {"mild_upper": mild_upper, "moderate_upper": moderate_upper}


def assign_severity_grade(score: float, thresholds: dict) -> str:
    if score <= thresholds["mild_upper"]:
        return "Mild"
    elif score <= thresholds["moderate_upper"]:
        return "Moderate"
    else:
        return "Severe"


# ==========================================================================
# FACE-VALIDITY CHECK (qualitative, non-clinical — explicitly labeled as such)
# ==========================================================================
def generate_face_validity_grid(images: np.ndarray, severity_scores: np.ndarray,
                                 labels: np.ndarray, save_path: str,
                                 n_per_group: int = 10) -> None:
    """
    WHY this produces a saved image grid rather than an interactive display:
    Colab sessions are not always interactive when this runs (e.g. as part
    of a longer unattended script), and a saved file is something you can
    review later, attach to a write-up, or show someone else for a second
    opinion — all things an inline plt.show() can't do.

    THIS IS NOT CLINICAL VALIDATION. It exists purely so you, a non-expert,
    can visually sanity-check that "higher severity score" roughly tracks
    "looks more opaque to the eye" — a coherence check, not a diagnostic one.
    """
    positive_mask = labels == 1
    positive_scores = severity_scores[positive_mask]
    positive_images = images[positive_mask]

    order = np.argsort(positive_scores)
    n = len(order)
    if n < n_per_group * 3:
        n_per_group = max(1, n // 3)
        print(f"[warn] Not enough positive cases for {10}-per-group; using {n_per_group} instead.")

    low_idx = order[:n_per_group]
    mid_idx = order[n // 2 - n_per_group // 2: n // 2 + n_per_group // 2]
    high_idx = order[-n_per_group:]

    fig, axes = plt.subplots(3, n_per_group, figsize=(n_per_group * 1.6, 5))
    for row, (group_idx, group_name) in enumerate(
        zip([low_idx, mid_idx, high_idx], ["LOW severity", "MID severity", "HIGH severity"])
    ):
        for col, idx in enumerate(group_idx):
            ax = axes[row, col] if n_per_group > 1 else axes[row]
            img = positive_images[idx]
            if img.ndim == 3 and img.shape[0] == 3:  # CHW -> HWC for display
                img = np.transpose(img, (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            ax.imshow(img)
            ax.set_title(f"{positive_scores[idx]:.2f}", fontsize=7)
            ax.axis("off")
        axes[row, 0].set_ylabel(group_name, fontsize=8) if n_per_group > 1 else None

    fig.suptitle("Face-Validity Check (NOT clinical validation) — "
                  "do higher scores look more opaque?", fontsize=10)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] Face-validity grid saved to {save_path}. Manually review this — "
          f"it is a sanity check, not proof of clinical accuracy.")


# ==========================================================================
# SAVE / LOAD RESULTS
# ==========================================================================
def save_severity_results(image_ids, embeddings_dict: dict, proxy_a, proxy_b,
                           proxy_a_normalized, severity_scores, thresholds: dict,
                           correlation_result: dict, results_dir: str = RESULTS_DIR) -> None:
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "image_id": image_ids,
        "label": embeddings_dict["labels"],
        "proxy_a_raw_distance": proxy_a,
        "proxy_b_detection_logit": proxy_b,
        "proxy_a_normalized": proxy_a_normalized,
        "severity_score": severity_scores,
    })
    df["severity_grade"] = df["severity_score"].apply(lambda s: assign_severity_grade(s, thresholds))

    csv_path = Path(results_dir, "severity_proxies.csv")
    df.to_csv(csv_path, index=False)
    print(f"[ok] Severity proxy table saved to {csv_path}")

    summary = {
        "thresholds": thresholds,
        "cross_validation": correlation_result,
        "n_images": len(df),
        "n_positive": int((df["label"] == 1).sum()),
    }
    with open(Path(results_dir, "severity_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ok] Severity summary (thresholds + correlation) saved to "
          f"{Path(results_dir, 'severity_summary.json')}")


# ==========================================================================
# CLI ENTRY POINT
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="OmniCataract-X Phase 3: Severity Engine")
    parser.add_argument("--smoke-test", action="store_true",
                         help="Run a tiny end-to-end check with random data, no real checkpoint needed.")
    parser.add_argument("--phase2-checkpoint", type=str, default=f"{CHECKPOINT_DIR_PHASE2}/best.pt")
    args = parser.parse_args()

    if args.smoke_test:
        print("=" * 70)
        print("PHASE 3 SMOKE TEST (random data, verifies pipeline + math only)")
        print("=" * 70)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[info] Using device: {device}")

        phase2 = load_phase2_module()
        model = phase2.CataractDetectorDualHead(pretrained=False)
        freeze_backbone(model)

        projection_head = SupConProjectionHead(input_dim=model.feature_dim)

        from torch.utils.data import TensorDataset
        n = 64
        fake_images = torch.randn(n, 3, 224, 224)
        fake_labels = torch.randint(0, 2, (n,)).float()
        ds = TensorDataset(fake_images, fake_labels)
        loader = DataLoader(ds, batch_size=16)

        train_supcon(model, projection_head, loader, device, epochs=2)

        emb = extract_embeddings(model, projection_head, loader, device)
        centroid = compute_normal_centroid(emb["projected_embeddings"], emb["labels"])
        proxy_a = compute_proxy_a_distance(emb["projected_embeddings"], centroid)
        proxy_b = compute_proxy_b_logit_magnitude(emb["detection_logits"])

        corr = cross_validate_proxies(proxy_a, proxy_b)
        proxy_a_norm = normalize_proxy_within_positive(proxy_a, emb["labels"])

        probe_result = train_severity_probe(emb["raw_features"], proxy_a_norm, device, epochs=3)

        with torch.no_grad():
            all_scores = probe_result["probe"](
                torch.tensor(emb["raw_features"], dtype=torch.float32).to(device)
            ).cpu().numpy()

        thresholds = compute_tertile_thresholds(all_scores, emb["labels"])

        generate_face_validity_grid(
            fake_images.numpy(), all_scores, emb["labels"],
            save_path="/tmp/face_validity_smoke_test.png", n_per_group=3
        )

        print(f"[ok] Smoke test complete. Spearman rho: {corr['spearman_rho']:.4f}, "
              f"thresholds: {thresholds}")
        print("=" * 70)
    else:
        print("This script's functions are intended to be called from a notebook "
              "cell after loading your trained Phase 2 checkpoint, e.g.:\n\n"
              "  phase2 = load_phase2_module()\n"
              "  model = phase2.CataractDetectorDualHead()\n"
              "  # ... load Phase 2 checkpoint weights into model ...\n"
              "  freeze_backbone(model)\n"
              "  projection_head = SupConProjectionHead(input_dim=model.feature_dim)\n"
              "  train_supcon(model, projection_head, train_loader, device='cuda')\n\n"
              "Run with --smoke-test to verify the pipeline works before that.")


if __name__ == "__main__":
    main()
