"""
02_core_detection_model.py
============================
OmniCataract-X v3.0 — Phase 2: Core Detection & Disentanglement.

Run this AFTER Phase 1's exit milestone has passed.

WHAT THIS FILE COVERS (maps to the plan's Task 2.1 - 2.4):
  2.1  ConvNeXt-Tiny backbone with dual heads (Detection + Acquisition Quality)
       + gradient checkpointing for VRAM savings
  2.2  Homoscedastic Uncertainty Weighting loss (Kendall et al.) with
       sigma L2 regularization
  2.3  AMP training loop with gradient accumulation, cosine LR schedule,
       rolling checkpointing, W&B logging (optional/defensive), early stopping
  2.4  Evaluation: AUC/sensitivity/specificity/precision/F1 with bootstrapped
       95% confidence intervals, ROC/PR curve plotting

EXIT MILESTONE (do not proceed to Phase 3 until this passes):
  Detection + Quality heads trained jointly; uncertainty weighting confirmed
  active via gradient-norm/sigma logs (neither task's sigma has silently
  exploded to "give up" on it); validation AUC reported WITH a 95% CI
  (there is no hardcoded accuracy target to hit — report the real number).
"""

import os
import json
import time
import shutil
import argparse
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import timm

from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================================================
# CONFIG
# ==========================================================================
DRIVE_ROOT = "/content/drive/MyDrive/OmniCataract-X"
CHECKPOINT_DIR = f"{DRIVE_ROOT}/checkpoints/phase2"
RESULTS_DIR = f"{DRIVE_ROOT}/results/phase2"
LOGS_DIR = f"{DRIVE_ROOT}/logs/phase2"

FEATURE_DIM = 768        # ConvNeXt-Tiny's pooled feature dimension
HEAD_HIDDEN_DIM = 256
DROPOUT_P = 0.3

DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_EPOCHS = 50
DEFAULT_ACCUM_STEPS = 4       # simulates batch_size * 4 effective batch
SIGMA_L2_LAMBDA = 0.01
LABEL_SMOOTH_POS = 0.9
LABEL_SMOOTH_NEG = 0.1
EARLY_STOP_PATIENCE = 10
KEEP_LAST_N_CHECKPOINTS = 3


# ==========================================================================
# TASK 2.1 — MODEL ARCHITECTURE
# ==========================================================================
class CataractDetectorDualHead(nn.Module):
    """
    ConvNeXt-Tiny backbone with two independent heads:
      - Head A (Detection): P(cataract)
      - Head B (Acquisition Quality): P(good image quality)

    WHY dual heads share one backbone: forcing both tasks through the same
    feature extractor produces richer, more general features than training
    two separate networks would — this is the entire point of the
    Multi-Task Learning design (see the Phase 2 plan document, Task 2.1).

    WHY the backbone returns pooled features directly: `num_classes=0` in
    timm strips the original classification head AND applies global average
    pooling by default (global_pool='avg'), so we get a (B, FEATURE_DIM)
    tensor ready to feed into our own heads — no separate GAP layer needed.

    The forward() method returns a dict including the pooled features
    themselves (not just the two logits) because Phase 3's SupCon training
    and Phase 4's distillation both need access to these intermediate
    features — designing that access path in now avoids reworking this
    class later.
    """

    def __init__(self, backbone_name: str = "convnext_tiny", pretrained: bool = True,
                 feature_dim: int = FEATURE_DIM, hidden_dim: int = HEAD_HIDDEN_DIM,
                 dropout_p: float = DROPOUT_P):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # WHY verify feature_dim rather than trust the constant: different
        # timm versions / backbone variants can have different output dims.
        # Failing loudly here beats a silent shape mismatch three layers down.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            actual_dim = self.backbone(dummy).shape[-1]
        if actual_dim != feature_dim:
            print(f"[warn] Backbone '{backbone_name}' outputs {actual_dim}-dim features, "
                  f"not the expected {feature_dim}. Adjusting head input size automatically.")
            feature_dim = actual_dim
        self.feature_dim = feature_dim

        self.detection_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 1),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 1),
        )

        self._grad_checkpointing_enabled = False

    def enable_gradient_checkpointing(self) -> bool:
        """
        WHY: on a 16GB T4, gradient checkpointing is the difference between
        batch size 8 and batch size 32 for ConvNeXt-Tiny. It trades ~20-30%
        slower training for substantially lower peak VRAM — call this if
        you hit CUDA OOM at your target batch size, not by default, since
        the speed cost is real and shouldn't be paid unless needed.

        timm's ConvNeXt implementation supports `set_grad_checkpointing()`
        natively. We check for it defensively rather than assuming every
        backbone variant exposes it.
        """
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(enable=True)
            self._grad_checkpointing_enabled = True
            print("[ok] Gradient checkpointing enabled on backbone.")
            return True
        else:
            print(f"[warn] Backbone does not expose set_grad_checkpointing(). "
                  f"Skipping — if you hit OOM, reduce batch size instead.")
            return False

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)  # (B, feature_dim)
        detection_logits = self.detection_head(features).squeeze(-1)  # (B,)
        quality_logits = self.quality_head(features).squeeze(-1)      # (B,)
        return {
            "detection_logits": detection_logits,
            "quality_logits": quality_logits,
            "features": features,
        }


# ==========================================================================
# TASK 2.2 — HOMOSCEDASTIC UNCERTAINTY WEIGHTING LOSS
# ==========================================================================
def _smooth_binary_labels(labels: torch.Tensor,
                           pos: float = LABEL_SMOOTH_POS,
                           neg: float = LABEL_SMOOTH_NEG) -> torch.Tensor:
    """
    WHY: BCEWithLogitsLoss has no native `label_smoothing` argument (unlike
    CrossEntropyLoss), so we apply it manually to the targets before the
    loss call. label=1 -> pos (0.9), label=0 -> neg (0.1). This softens the
    model's confidence and is what makes later calibration (temperature
    scaling) work properly instead of fighting an already-overconfident model.
    """
    return labels * (pos - neg) + neg


class HomoscedasticMTLLoss(nn.Module):
    """
    Kendall et al.-style homoscedastic uncertainty weighting, combining the
    Detection loss and Quality loss into one automatically-balanced total.

    WHY this exists instead of fixed loss weights: hand-picking a fixed
    weight between two losses of different scales/difficulty is guesswork
    that has to be re-tuned by hand. This lets the network learn its own
    per-task weighting via two learnable parameters (log_var_1, log_var_2).

    Formula (as specified in the plan):
        L_total = (1/(2*sigma1^2))*L_det + (1/(2*sigma2^2))*L_qual
                  + log(sigma1) + log(sigma2)
                  + lambda*(sigma1^2 + sigma2^2)
        where sigma_i = exp(log_var_i)

    WHY the L2 term on sigma: without it, the network can trivially
    minimize loss by inflating sigma on the harder task indefinitely,
    which is mathematically equivalent to giving up on that task rather
    than actually learning it. This is a documented failure mode of plain
    Kendall-style weighting — the L2 penalty keeps sigma bounded.

    IMPORTANT: log_var_1 and log_var_2 are `nn.Parameter`s, meaning they
    must be included in the optimizer's parameter list alongside the
    model's own parameters (see build_optimizer() below) — otherwise
    they never update and the "automatic balancing" silently does nothing.
    """

    def __init__(self, sigma_l2_lambda: float = SIGMA_L2_LAMBDA):
        super().__init__()
        self.log_var_1 = nn.Parameter(torch.zeros(1))  # detection
        self.log_var_2 = nn.Parameter(torch.zeros(1))  # quality
        self.sigma_l2_lambda = sigma_l2_lambda

    def forward(self, detection_logits: torch.Tensor, detection_labels: torch.Tensor,
                quality_logits: torch.Tensor, quality_labels: torch.Tensor) -> dict:

        smoothed_det_labels = _smooth_binary_labels(detection_labels)
        l_det = F.binary_cross_entropy_with_logits(detection_logits, smoothed_det_labels)

        # WHY no smoothing on the quality label: quality is a more
        # objective/measurable signal than "is this a cataract" (which has
        # real clinical ambiguity); smoothing it isn't warranted the same way.
        l_qual = F.binary_cross_entropy_with_logits(quality_logits, quality_labels)

        sigma1 = torch.exp(self.log_var_1)
        sigma2 = torch.exp(self.log_var_2)

        weighted_det = (1.0 / (2.0 * sigma1 ** 2)) * l_det
        weighted_qual = (1.0 / (2.0 * sigma2 ** 2)) * l_qual
        reg_term = torch.log(sigma1) + torch.log(sigma2)
        sigma_l2 = self.sigma_l2_lambda * (sigma1 ** 2 + sigma2 ** 2)

        total = (weighted_det + weighted_qual + reg_term + sigma_l2).squeeze()

        return {
            "total_loss": total,
            "l_detection_raw": l_det.detach(),
            "l_quality_raw": l_qual.detach(),
            "sigma1": sigma1.detach().item(),
            "sigma2": sigma2.detach().item(),
        }

    def get_current_sigmas(self) -> dict:
        """WHY: called every epoch for logging — this is the evidence that
        neither task is being silently abandoned (see EXIT MILESTONE)."""
        return {
            "sigma_detection": torch.exp(self.log_var_1).item(),
            "sigma_quality": torch.exp(self.log_var_2).item(),
        }


def build_optimizer(model: nn.Module, loss_fn: HomoscedasticMTLLoss,
                     lr: float = DEFAULT_LR, weight_decay: float = DEFAULT_WEIGHT_DECAY) -> torch.optim.Optimizer:
    """
    WHY loss_fn.parameters() must be included: log_var_1/log_var_2 are the
    whole mechanism behind automatic task balancing. Forgetting to include
    them here is a subtle, easy-to-miss bug — the model would still train,
    but the uncertainty weighting would silently stay at its initial values
    forever. This function exists specifically to make that impossible to
    get wrong by accident.
    """
    params = list(model.parameters()) + list(loss_fn.parameters())
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


# ==========================================================================
# TASK 2.3 — TRAINING LOOP: AMP, GRADIENT ACCUMULATION, CHECKPOINTING
# ==========================================================================
def compute_gradient_norm(model: nn.Module) -> float:
    """WHY: logged every epoch to confirm gradients are flowing sanely
    (not vanishing, not exploding) — a cheap, high-value diagnostic."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


class EarlyStopping:
    """WHY: with no fixed epoch count guaranteed to be optimal, stop
    training once validation AUC stops improving for `patience` epochs,
    rather than wasting Colab GPU-hours on a plateaued model."""

    def __init__(self, patience: int = EARLY_STOP_PATIENCE, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return True  # is_best

        improved = (score > self.best_score) if self.mode == "max" else (score < self.best_score)
        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


def save_checkpoint(model: nn.Module, loss_fn: HomoscedasticMTLLoss,
                     optimizer: torch.optim.Optimizer, scheduler,
                     epoch: int, val_auc: float, is_best: bool,
                     checkpoint_dir: str = CHECKPOINT_DIR) -> str:
    """
    WHY save loss_fn's state too: log_var_1/log_var_2 are learned
    parameters. Resuming training without them would silently reset
    uncertainty weighting back to its initial (untrained) state.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss_fn_state_dict": loss_fn.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_auc": val_auc,
    }

    epoch_path = Path(checkpoint_dir, f"epoch_{epoch:03d}.pt")
    torch.save(state, epoch_path)
    print(f"[ok] Checkpoint saved: {epoch_path} (val_auc={val_auc:.4f})")

    if is_best:
        best_path = Path(checkpoint_dir, "best.pt")
        torch.save(state, best_path)
        print(f"[ok] New best checkpoint: {best_path} (val_auc={val_auc:.4f})")

    cleanup_old_checkpoints(checkpoint_dir)
    return str(epoch_path)


def cleanup_old_checkpoints(checkpoint_dir: str = CHECKPOINT_DIR,
                             keep_last_n: int = KEEP_LAST_N_CHECKPOINTS) -> None:
    """
    WHY: Google Drive's free tier is 15GB. Every epoch checkpoint for a
    ConvNeXt-Tiny model is tens of MB; left unchecked across 50+ epochs
    this fills the quota and a future checkpoint save fails silently at
    the worst possible time. `best.pt` is never deleted by this function.
    """
    epoch_checkpoints = sorted(Path(checkpoint_dir).glob("epoch_*.pt"),
                                key=lambda p: p.stat().st_mtime)
    n_to_delete = len(epoch_checkpoints) - keep_last_n
    if n_to_delete > 0:
        for old_ckpt in epoch_checkpoints[:n_to_delete]:
            old_ckpt.unlink()
            print(f"[ok] Rolling cleanup: removed old checkpoint {old_ckpt.name}")


def load_checkpoint(model: nn.Module, loss_fn: HomoscedasticMTLLoss,
                     optimizer: torch.optim.Optimizer, scheduler,
                     checkpoint_path: str, device: str = "cpu") -> int:
    """Returns the epoch number to resume FROM (i.e. the next epoch to run)."""
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    loss_fn.load_state_dict(state["loss_fn_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler and state.get("scheduler_state_dict"):
        scheduler.load_state_dict(state["scheduler_state_dict"])
    print(f"[ok] Resumed from checkpoint at epoch {state['epoch']} "
          f"(val_auc was {state['val_auc']:.4f})")
    return state["epoch"] + 1


def init_wandb_defensive(config: dict, project: str = "omnicataract-x"):
    """
    WHY defensive: W&B requires an internet connection and a logged-in
    account. Training should not hard-crash if W&B is unavailable
    (offline testing, no API key set yet, etc.) — it should degrade to
    local console logging instead.
    """
    try:
        import wandb
        run = wandb.init(project=project, config=config, reinit=True)
        print(f"[ok] W&B logging active: {run.url}")
        return wandb
    except Exception as e:
        print(f"[info] W&B unavailable ({e}). Falling back to console-only logging.")
        return None


def train_one_epoch(model: nn.Module, loss_fn: HomoscedasticMTLLoss,
                     dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                     scaler: torch.cuda.amp.GradScaler, device: str,
                     accum_steps: int = DEFAULT_ACCUM_STEPS) -> dict:
    """
    WHY gradient accumulation: on a T4 you may only fit batch_size=32
    physically in VRAM, but a larger *effective* batch size (here,
    32*4=128) tends to give more stable gradients, especially once
    SupCon enters the picture in Phase 3. This simulates that larger
    batch without needing the VRAM for it, at the cost of accum_steps
    forward passes before each optimizer step.
    """
    model.train()
    loss_fn.train()

    total_loss, total_det_loss, total_qual_loss = 0.0, 0.0, 0.0
    n_batches = 0
    grad_norms = []  # WHY a list, not a single value: gradients get cleared by
                      # zero_grad() at every accumulation boundary, so the norm
                      # MUST be captured right before that happens, not after
                      # the loop ends (see the bug this replaced — capturing it
                      # post-loop silently reads 0.0 on every single epoch).
    optimizer.zero_grad()

    for i, (images, det_labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        det_labels = det_labels.to(device, non_blocking=True)

        # NOTE: quality labels are a placeholder here — Phase 2's real
        # training script expects a dataset that provides BOTH detection
        # and quality labels per image. See NOTE_ON_QUALITY_LABELS below
        # for how the plan expects this to be sourced.
        qual_labels = det_labels.new_ones(det_labels.shape)  # placeholder, see note below

        use_amp = device == "cuda"
        with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            outputs = model(images)
            loss_dict = loss_fn(
                outputs["detection_logits"], det_labels,
                outputs["quality_logits"], qual_labels,
            )
            loss = loss_dict["total_loss"] / accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accum_steps == 0:
            if use_amp:
                # WHY unscale before computing the norm: AMP's GradScaler
                # multiplies gradients by a scale factor internally: reading
                # the norm without unscaling first would report a wildly
                # inflated, meaningless number.
                scaler.unscale_(optimizer)
            grad_norms.append(compute_gradient_norm(model))

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss_dict["total_loss"].item()
        total_det_loss += loss_dict["l_detection_raw"].item()
        total_qual_loss += loss_dict["l_quality_raw"].item()
        n_batches += 1

    grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
    sigmas = loss_fn.get_current_sigmas()

    return {
        "train_loss": total_loss / max(n_batches, 1),
        "train_det_loss": total_det_loss / max(n_batches, 1),
        "train_qual_loss": total_qual_loss / max(n_batches, 1),
        "grad_norm": grad_norm,
        **sigmas,
    }


# NOTE_ON_QUALITY_LABELS:
# The plan's public datasets (ODIR-5K, RFMiD) do not ship a ground-truth
# "image quality" label. Two honest options, either is acceptable — pick
# one and document it in your write-up:
#   (a) Use a no-reference IQA proxy (e.g. BRISQUE score, thresholded)
#       computed offline as a weak-supervision quality label.
#   (b) Train Head B in a genuinely unsupervised/self-supervised way and
#       validate it post-hoc (this is what the plan's Day-17-equivalent
#       "Acquisition Channel Validation" step is for).
# This training loop is written so that whichever labeling strategy you
# choose just needs to populate a `quality_labels` tensor per batch —
# replace the placeholder line above with a real data source before a
# full training run. Left as a placeholder here intentionally so Phase 2's
# exit milestone (pipeline + loss math correctness) can be verified before
# that separate labeling decision is finalized.


@torch.no_grad()
def validate_one_epoch(model: nn.Module, loss_fn: HomoscedasticMTLLoss,
                        dataloader: DataLoader, device: str) -> dict:
    model.eval()
    loss_fn.eval()

    all_det_logits, all_det_labels = [], []
    total_loss = 0.0
    n_batches = 0

    for images, det_labels in dataloader:
        images = images.to(device, non_blocking=True)
        det_labels_dev = det_labels.to(device, non_blocking=True)
        qual_labels = det_labels_dev.new_ones(det_labels_dev.shape)  # see NOTE_ON_QUALITY_LABELS

        outputs = model(images)
        loss_dict = loss_fn(
            outputs["detection_logits"], det_labels_dev,
            outputs["quality_logits"], qual_labels,
        )

        total_loss += loss_dict["total_loss"].item()
        n_batches += 1
        all_det_logits.append(outputs["detection_logits"].cpu())
        all_det_labels.append(det_labels.cpu())

    logits = torch.cat(all_det_logits).numpy()
    labels = torch.cat(all_det_labels).numpy()
    probs = 1 / (1 + np.exp(-logits))  # sigmoid

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        # WHY this can happen: a validation batch/split with only one
        # class present makes AUC undefined. Report clearly rather than
        # crashing, especially useful during small smoke-test runs.
        print("[warn] AUC undefined for this validation set (likely only one class present).")
        auc = float("nan")

    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_auc": auc,
        "val_probs": probs,
        "val_labels": labels,
    }


def run_training(model: nn.Module, loss_fn: HomoscedasticMTLLoss,
                  train_loader: DataLoader, val_loader: DataLoader,
                  device: str, epochs: int = DEFAULT_EPOCHS,
                  lr: float = DEFAULT_LR, accum_steps: int = DEFAULT_ACCUM_STEPS,
                  use_wandb: bool = True, resume_from: str = None) -> dict:

    model.to(device)
    loss_fn.to(device)

    optimizer = build_optimizer(model, loss_fn, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # WHY the try/except: torch>=2.4 moved GradScaler to torch.amp and
    # deprecated the torch.cuda.amp path; older torch versions (this repo
    # pins 2.1.0, but Colab's preinstalled version can differ) only have
    # the old path. Supporting both avoids a hard version pin becoming a
    # silent breakage point.
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    early_stopper = EarlyStopping(patience=EARLY_STOP_PATIENCE, mode="max")

    start_epoch = 0
    if resume_from:
        start_epoch = load_checkpoint(model, loss_fn, optimizer, scheduler, resume_from, device)

    wandb_run = init_wandb_defensive(
        config={"lr": lr, "epochs": epochs, "accum_steps": accum_steps}
    ) if use_wandb else None

    history = []
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_metrics = train_one_epoch(model, loss_fn, train_loader, optimizer, scaler, device, accum_steps)
        val_metrics = validate_one_epoch(model, loss_fn, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        epoch_log = {
            "epoch": epoch,
            **train_metrics,
            "val_loss": val_metrics["val_loss"],
            "val_auc": val_metrics["val_auc"],
            "elapsed_sec": elapsed,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(epoch_log)

        print(f"[epoch {epoch}] train_loss={train_metrics['train_loss']:.4f} "
              f"val_auc={val_metrics['val_auc']:.4f} "
              f"sigma_det={train_metrics['sigma_detection']:.3f} "
              f"sigma_qual={train_metrics['sigma_quality']:.3f} "
              f"grad_norm={train_metrics['grad_norm']:.3f} "
              f"({elapsed:.1f}s)")

        if wandb_run:
            wandb_run.log(epoch_log)

        is_best = early_stopper.step(val_metrics["val_auc"])
        save_checkpoint(model, loss_fn, optimizer, scheduler, epoch, val_metrics["val_auc"], is_best)

        if early_stopper.should_stop:
            print(f"[ok] Early stopping triggered at epoch {epoch} "
                  f"(no val_auc improvement for {EARLY_STOP_PATIENCE} epochs).")
            break

    return {"history": history, "final_val_metrics": val_metrics}


# ==========================================================================
# TASK 2.4 — EVALUATION: METRICS + BOOTSTRAPPED CONFIDENCE INTERVALS
# ==========================================================================
def compute_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> dict:
    """
    WHY threshold=0.5 as the default here specifically: Phase 2's job is to
    prove the pipeline and loss math are correct. Real threshold CALIBRATION
    (picking the operating point from the validation ROC curve, per the
    plan's Section 11.2) is a Phase 3/evaluation-stage concern — don't
    conflate "does the model train correctly" with "what's the best
    clinical operating point," they're different questions.
    """
    y_pred = (y_probs >= threshold).astype(int)

    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def bootstrap_confidence_interval(y_true: np.ndarray, y_probs: np.ndarray,
                                   metric_name: str = "auc",
                                   n_resamples: int = 1000,
                                   ci: float = 0.95,
                                   seed: int = 42) -> dict:
    """
    WHY bootstrapping specifically (rather than a closed-form CI formula):
    it makes no distributional assumptions about the metric, which matters
    for something like AUC where a normal approximation can be misleading
    at small sample sizes — exactly the regime a Colab-Free single-GPU
    project is likely to be in for at least some splits.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_resamples):
        idx = rng.randint(0, n, n)
        y_true_bs, y_probs_bs = y_true[idx], y_probs[idx]

        if len(np.unique(y_true_bs)) < 2:
            continue  # WHY: AUC undefined for a resample with only one class present

        m = compute_metrics(y_true_bs, y_probs_bs)
        scores.append(m[metric_name])

    if len(scores) == 0:
        return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "n_valid_resamples": 0}

    scores = np.array(scores)
    alpha = (1 - ci) / 2
    lower = np.percentile(scores, 100 * alpha)
    upper = np.percentile(scores, 100 * (1 - alpha))

    return {
        "mean": float(np.mean(scores)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_valid_resamples": len(scores),
    }


def full_evaluation_report(y_true: np.ndarray, y_probs: np.ndarray,
                            n_resamples: int = 1000) -> pd.DataFrame:
    """WHY a single entry point: this is what gets called at every
    evaluation checkpoint (Phase 2 validation, later the RFMiD pseudo-external
    test, later the ablation table) so the reporting format stays identical
    and comparable across all of them."""
    point_estimates = compute_metrics(y_true, y_probs)

    rows = []
    for metric_name, point_value in point_estimates.items():
        ci = bootstrap_confidence_interval(y_true, y_probs, metric_name, n_resamples)
        rows.append({
            "metric": metric_name,
            "point_estimate": point_value,
            "ci_lower_95": ci["ci_lower"],
            "ci_upper_95": ci["ci_upper"],
            "n_valid_resamples": ci["n_valid_resamples"],
        })

    return pd.DataFrame(rows)


def plot_roc_curve(y_true: np.ndarray, y_probs: np.ndarray,
                    save_path: str = None, n_bootstrap_curves: int = 100) -> None:
    """WHY plot bootstrapped curves faintly in the background: a single ROC
    line looks more certain than it is; showing the bootstrap spread is a
    more honest visualization of how stable the curve really is."""
    fig, ax = plt.subplots(figsize=(6, 6))

    rng = np.random.RandomState(42)
    n = len(y_true)
    for _ in range(n_bootstrap_curves):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        fpr_bs, tpr_bs, _ = roc_curve(y_true[idx], y_probs[idx])
        ax.plot(fpr_bs, tpr_bs, color="steelblue", alpha=0.03)

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    ax.plot(fpr, tpr, color="#1b2a4a", linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (with bootstrap variability band)")
    ax.legend()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[ok] ROC curve saved to {save_path}")
    plt.close(fig)


def plot_pr_curve(y_true: np.ndarray, y_probs: np.ndarray, save_path: str = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ax.plot(recall, precision, color="#0f9b8e", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[ok] PR curve saved to {save_path}")
    plt.close(fig)


def save_metrics_csv(report_df: pd.DataFrame, results_dir: str = RESULTS_DIR,
                      filename: str = "phase2_validation_metrics.csv") -> None:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    path = Path(results_dir, filename)
    report_df.to_csv(path, index=False)
    print(f"[ok] Metrics saved to {path}")


# ==========================================================================
# CLI ENTRY POINT
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="OmniCataract-X Phase 2: Core Detection Model")
    parser.add_argument("--smoke-test", action="store_true",
                         help="Run a tiny end-to-end check with random data, no real dataset needed.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    if args.smoke_test:
        print("=" * 70)
        print("PHASE 2 SMOKE TEST (random data, verifies pipeline + loss math only)")
        print("=" * 70)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[info] Using device: {device}")

        model = CataractDetectorDualHead(pretrained=False)  # WHY pretrained=False here: smoke test only, avoids a slow weights download
        loss_fn = HomoscedasticMTLLoss()

        from torch.utils.data import TensorDataset
        n = 32
        fake_images = torch.randn(n, 3, 224, 224)
        fake_labels = torch.randint(0, 2, (n,)).float()
        ds = TensorDataset(fake_images, fake_labels)
        loader = DataLoader(ds, batch_size=8)

        result = run_training(model, loss_fn, loader, loader, device,
                               epochs=2, accum_steps=1, use_wandb=not args.no_wandb)
        print(f"[ok] Smoke test complete. Final val_auc: {result['final_val_metrics']['val_auc']:.4f}")
        print("=" * 70)
    else:
        print("This script's main() is intended to be imported and called from a "
              "training notebook cell with your real DataLoaders from Phase 1, e.g.:\n\n"
              "  from importlib import import_module\n"
              "  phase2 = import_module('02_core_detection_model')\n"
              "  model = phase2.CataractDetectorDualHead()\n"
              "  loss_fn = phase2.HomoscedasticMTLLoss()\n"
              "  result = phase2.run_training(model, loss_fn, train_loader, val_loader, device='cuda')\n\n"
              "Run with --smoke-test to verify the pipeline works before that.")


if __name__ == "__main__":
    main()
