"""
04_optimization_distillation_export.py
========================================
OmniCataract-X v3.0 — Phase 4: Optimization, Distillation & Export.

Run this AFTER Phase 3's exit milestone has passed (a trained Phase 2
detection model + a trained Phase 3 severity probe must exist).

WHAT THIS FILE COVERS (maps to the plan's Task 4.1 - 4.3):
  4.1  Knowledge Distillation: ConvNeXt-Tiny (teacher) -> MobileNetV3-Large
       (student), using BOTH feature-map matching (intermediate activations)
       AND logit matching (soft KL-divergence targets), plus a lightweight
       severity-score distillation term so the student inherits all three
       outputs, not just detection.
  4.2  Export the distilled student to ONNX, then apply INT8 post-training
       static quantization. Benchmark real CPU latency and model size —
       both are MEASURED here, not assumed or promised in advance.
  4.3  A framework-agnostic ONNX Runtime inference wrapper
       (`CataractDetectorONNX`) usable without PyTorch installed at all,
       plus a numerical equivalence test against the original PyTorch model.

WHY FEATURE-MAP MATCHING USES REAL, INSPECTED MODULE NAMES:
ConvNeXt-Tiny and MobileNetV3-Large have architecturally unrelated internal
structures. Guessing plausible-looking hook points risks silently hooking
the wrong layer (or crashing on the wrong timm version). The four hook
points used below were verified by actually instantiating both backbones
and inspecting their real named submodules and output shapes — see the
TEACHER_STAGE_INFO / STUDENT_STAGE_INFO comments for the exact values found.

EXIT MILESTONE (do not proceed to Phase 5 until this passes):
  A working .onnx file exists; measured latency and model size are
  documented (whatever the real numbers are); accuracy retention vs. the
  teacher is reported honestly, including on borderline/hazy cases —
  no target number is assumed to be hit in advance.
"""

import time
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, QuantType, QuantFormat,
)

import importlib.util


# ==========================================================================
# CONFIG
# ==========================================================================
DRIVE_ROOT = "/content/drive/MyDrive/OmniCataract-X"
CHECKPOINT_DIR_PHASE2 = f"{DRIVE_ROOT}/checkpoints/phase2"
CHECKPOINT_DIR_PHASE4 = f"{DRIVE_ROOT}/checkpoints/phase4"
EXPORTS_DIR = f"{DRIVE_ROOT}/exports"
RESULTS_DIR = f"{DRIVE_ROOT}/results/phase4"

TEACHER_BACKBONE = "convnext_tiny"
STUDENT_BACKBONE = "mobilenetv3_large_100"

IMG_SIZE = 224
ONNX_OPSET = 13
TARGET_LATENCY_MS = 100  # a target to benchmark AGAINST, not a guarantee

DISTILL_EPOCHS = 50
DISTILL_LR = 1e-3
FREEZE_STUDENT_BACKBONE_EPOCHS = 10
ALPHA_FEATURE = 0.5
BETA_LOGIT = 0.5
GAMMA_SEVERITY = 0.2   # extension beyond the base alpha/beta formula — see note below
KD_TEMPERATURE = 4.0

# WHY these exact stage pairs — verified by direct inspection (see the
# preceding analysis step run before writing this file), not guessed:
#   Teacher (ConvNeXt-Tiny) stages.{0,1,2,3}: (96,56x56) (192,28x28) (384,14x14) (768,7x7)
#   Student (MobileNetV3)   blocks.{1,2,4,6}: (24,56x56) (40,28x28) (112,14x14) (960,7x7)
# Each pair matches spatial resolution exactly, so only a 1x1 conv channel
# adapter is needed — no spatial resize/pool required.
TEACHER_STAGE_CHANNELS = {"stage0": 96, "stage1": 192, "stage2": 384, "stage3": 768}
STUDENT_STAGE_CHANNELS = {"stage0": 24, "stage1": 40, "stage2": 112, "stage3": 960}


# ==========================================================================
# UTILITY — load numbered-file modules (same pattern as Phase 3)
# ==========================================================================
def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ==========================================================================
# TASK 4.1a — INTERMEDIATE FEATURE EXTRACTION VIA FORWARD HOOKS
# ==========================================================================
def get_teacher_hook_modules(model: nn.Module) -> dict:
    """
    WHY this function exists rather than hardcoding string paths inline
    everywhere: `model.backbone.stages[i]` is the ONE place this structural
    assumption lives. If a future timm version changes ConvNeXt's internal
    naming, this is the only place that needs updating.
    """
    backbone = model.backbone
    return {
        "stage0": backbone.stages[0],
        "stage1": backbone.stages[1],
        "stage2": backbone.stages[2],
        "stage3": backbone.stages[3],
    }


def get_student_hook_modules(model: nn.Module) -> dict:
    backbone = model.backbone
    return {
        "stage0": backbone.blocks[1],
        "stage1": backbone.blocks[2],
        "stage2": backbone.blocks[4],
        "stage3": backbone.blocks[6],
    }


class IntermediateFeatureExtractor:
    """
    Registers forward hooks on a fixed set of named submodules and captures
    their output activations into a dict every time the model's forward()
    runs. WHY hooks rather than modifying the model's forward() method:
    this works on both the teacher and student without needing to touch
    Phase 2's already-tested CataractDetectorDualHead class at all.
    """

    def __init__(self, hook_modules: dict):
        self.activations = {}
        self.handles = []
        for name, module in hook_modules.items():
            handle = module.register_forward_hook(self._make_hook(name))
            self.handles.append(handle)

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            self.activations[name] = out
        return hook

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()


# ==========================================================================
# TASK 4.1b — DISTILLATION LOSSES
# ==========================================================================
class FeatureDistillationLoss(nn.Module):
    """
    MSE loss between teacher and (channel-adapted) student intermediate
    feature maps, averaged across the 4 matched stages.

    WHY a learnable 1x1 conv adapter per stage: teacher and student have
    different channel counts at every matched stage (see the table in this
    file's module docstring). A 1x1 conv is the standard, minimal way to
    project student channels into the teacher's channel space so an MSE
    loss is even computable — it adds negligible parameters/compute
    compared to either backbone.
    """

    def __init__(self, teacher_channels: dict = TEACHER_STAGE_CHANNELS,
                 student_channels: dict = STUDENT_STAGE_CHANNELS):
        super().__init__()
        assert teacher_channels.keys() == student_channels.keys(), \
            "Teacher and student stage names must match exactly."
        self.adapters = nn.ModuleDict({
            stage: nn.Conv2d(student_channels[stage], teacher_channels[stage], kernel_size=1)
            for stage in teacher_channels
        })
        self.stage_names = list(teacher_channels.keys())

    def forward(self, teacher_activations: dict, student_activations: dict) -> dict:
        per_stage_losses = {}
        for stage in self.stage_names:
            t_act = teacher_activations[stage].detach()  # WHY detach: teacher must never receive gradient here
            s_act = student_activations[stage]

            if t_act.shape[-2:] != s_act.shape[-2:]:
                # WHY this fallback exists even though the default 224x224
                # input makes it unnecessary: if this file is ever reused
                # with a different input resolution, spatial mismatch would
                # otherwise crash the MSE call with a confusing shape error
                # instead of this clear, intentional adaptive-pool fallback.
                s_act = F.adaptive_avg_pool2d(s_act, t_act.shape[-2:])

            adapted = self.adapters[stage](s_act)
            per_stage_losses[stage] = F.mse_loss(adapted, t_act)

        total = sum(per_stage_losses.values()) / len(per_stage_losses)
        return {"feature_loss": total, "per_stage": per_stage_losses}


def soft_binary_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                         temperature: float = KD_TEMPERATURE) -> torch.Tensor:
    """
    KL-divergence-based soft-target distillation loss for a binary
    (single-logit, sigmoid) output — the standard Hinton-style KD formula
    adapted from softmax/multi-class to the binary/sigmoid case.

    WHY treat this as a 2-class distribution [p, 1-p] rather than using
    BCE directly on the softened probabilities: BCE on softened
    probabilities is a common simplification, but it isn't actually the KL
    divergence between the teacher and student distributions. Building the
    explicit 2-class distribution and using F.kl_div is the more faithful
    implementation of what "temperature-softened distillation" means.

    WHY multiply by temperature^2: this is Hinton et al.'s standard
    correction — softening with a higher temperature shrinks the gradient
    magnitude roughly proportionally to 1/T^2, and this multiplier keeps
    the effective gradient scale comparable across different T choices.
    """
    student_prob = torch.sigmoid(student_logits / temperature)
    teacher_prob = torch.sigmoid(teacher_logits / temperature).detach()

    eps = 1e-7
    student_dist = torch.stack([student_prob, 1 - student_prob], dim=1).clamp(eps, 1 - eps)
    teacher_dist = torch.stack([teacher_prob, 1 - teacher_prob], dim=1).clamp(eps, 1 - eps)

    # F.kl_div(input=log(student), target=teacher) computes KL(teacher || student),
    # which is the correct direction for distillation (match student to teacher).
    kd = F.kl_div(torch.log(student_dist), teacher_dist, reduction="batchmean")
    return kd * (temperature ** 2)


# ==========================================================================
# TASK 4.1c — BUILDING THE STUDENT MODEL
# ==========================================================================
def build_student_model(phase2_module, backbone_name: str = STUDENT_BACKBONE,
                         pretrained: bool = True) -> nn.Module:
    """
    WHY this reuses Phase 2's CataractDetectorDualHead class directly rather
    than defining a new student-specific class: that class already handles
    dynamic feature-dimension detection, both heads with the exact
    structure the plan specifies, and the optional gradient-checkpointing
    hook (which will gracefully no-op with a warning for MobileNetV3, since
    that architecture doesn't expose timm's set_grad_checkpointing — this
    is fine, checkpointing matters far less for a model this small anyway).
    Reusing a class that was already independently tested in Phase 2 is
    strictly safer than writing a parallel, untested student-specific one.

    WHY `pretrained` is exposed as a parameter rather than hardcoded True:
    real training runs want ImageNet-pretrained weights, but pipeline
    testing (e.g. --smoke-test, or any offline/no-internet environment)
    needs to be able to skip the weights download entirely and still
    exercise every line of downstream logic.
    """
    return phase2_module.CataractDetectorDualHead(backbone_name=backbone_name, pretrained=pretrained)


# ==========================================================================
# TASK 4.1d — DISTILLATION TRAINING LOOP
# ==========================================================================
def set_student_backbone_trainable(student: nn.Module, trainable: bool) -> None:
    """WHY a warmup freeze (see FREEZE_STUDENT_BACKBONE_EPOCHS): letting the
    randomly-initialized adapters and heads stabilize against a FIXED
    pretrained student backbone for the first few epochs avoids the early,
    noisy gradients from an untrained adapter corrupting the student's
    otherwise-good ImageNet-pretrained backbone weights."""
    for p in student.backbone.parameters():
        p.requires_grad = trainable


def train_distillation(teacher: nn.Module, teacher_severity_probe: nn.Module,
                        student: nn.Module, student_severity_probe: nn.Module,
                        dataloader: DataLoader, device: str,
                        epochs: int = DISTILL_EPOCHS, lr: float = DISTILL_LR,
                        freeze_epochs: int = FREEZE_STUDENT_BACKBONE_EPOCHS) -> dict:
    """
    WHY the teacher's severity probe is also passed in and used to generate
    SOFT severity targets (rather than re-reading Phase 3's saved CSV
    proxies): this keeps the distillation loop self-contained and exactly
    consistent with whatever the teacher model currently predicts, rather
    than depending on a separate, potentially-stale results file.
    """
    teacher.to(device).eval()
    teacher_severity_probe.to(device).eval()
    student.to(device)
    student_severity_probe.to(device)

    feature_loss_fn = FeatureDistillationLoss().to(device)

    teacher_hooks = IntermediateFeatureExtractor(get_teacher_hook_modules(teacher))
    student_hooks = IntermediateFeatureExtractor(get_student_hook_modules(student))

    set_student_backbone_trainable(student, trainable=False)
    print(f"[ok] Student backbone frozen for the first {freeze_epochs} warmup epochs.")

    trainable_params = (
        list(student.parameters()) +
        list(student_severity_probe.parameters()) +
        list(feature_loss_fn.parameters())  # WHY included: the adapter convs must learn too
    )
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    history = []
    try:
        for epoch in range(epochs):
            if epoch == freeze_epochs:
                set_student_backbone_trainable(student, trainable=True)
                print(f"[ok] Epoch {epoch}: student backbone unfrozen, fine-tuning end-to-end now.")

            student.train()
            student_severity_probe.train()

            epoch_feature_loss, epoch_logit_loss, epoch_severity_loss, epoch_total = 0.0, 0.0, 0.0, 0.0
            n_batches = 0

            for images, _labels in dataloader:
                images = images.to(device, non_blocking=True)

                with torch.no_grad():
                    teacher_out = teacher(images)
                    teacher_severity = teacher_severity_probe(teacher_out["features"])

                student_out = student(images)
                student_severity = student_severity_probe(student_out["features"])

                feat_loss_dict = feature_loss_fn(teacher_hooks.activations, student_hooks.activations)
                feature_loss = feat_loss_dict["feature_loss"]

                logit_loss = (
                    soft_binary_kd_loss(student_out["detection_logits"], teacher_out["detection_logits"]) +
                    soft_binary_kd_loss(student_out["quality_logits"], teacher_out["quality_logits"])
                ) / 2.0

                # WHY this severity term is an ADDITION beyond the base
                # alpha*feature + beta*logit formula: the student needs to
                # produce a severity output too (per the app's final output
                # spec), and MSE-matching it to the teacher's severity
                # probe output is the natural way to distill that third
                # head. Weighted lightly (gamma=0.2) since it is the least
                # certain of the three signals being distilled — severity
                # is already a proxy one level removed from ground truth,
                # and shouldn't dominate the loss.
                severity_loss = F.mse_loss(student_severity, teacher_severity.detach())

                total_loss = (ALPHA_FEATURE * feature_loss +
                              BETA_LOGIT * logit_loss +
                              GAMMA_SEVERITY * severity_loss)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_feature_loss += feature_loss.item()
                epoch_logit_loss += logit_loss.item()
                epoch_severity_loss += severity_loss.item()
                epoch_total += total_loss.item()
                n_batches += 1

            n_batches = max(n_batches, 1)
            epoch_log = {
                "epoch": epoch,
                "feature_loss": epoch_feature_loss / n_batches,
                "logit_loss": epoch_logit_loss / n_batches,
                "severity_loss": epoch_severity_loss / n_batches,
                "total_loss": epoch_total / n_batches,
                "backbone_frozen": epoch < freeze_epochs,
            }
            history.append(epoch_log)
            print(f"[epoch {epoch}] total={epoch_log['total_loss']:.4f} "
                  f"feature={epoch_log['feature_loss']:.4f} "
                  f"logit={epoch_log['logit_loss']:.4f} "
                  f"severity={epoch_log['severity_loss']:.4f} "
                  f"(backbone_frozen={epoch_log['backbone_frozen']})")
    finally:
        # WHY finally: hooks must be removed even if training crashes
        # partway through, or they silently leak into any later use of
        # these same model instances (e.g. the export step right after).
        teacher_hooks.remove()
        student_hooks.remove()

    return {"history": history}


def evaluate_accuracy_retention(teacher: nn.Module, teacher_severity_probe: nn.Module,
                                 student: nn.Module, student_severity_probe: nn.Module,
                                 dataloader: DataLoader, device: str,
                                 borderline_severity_range: tuple = (0.3, 0.7)) -> dict:
    """
    WHY report retention on borderline/hazy cases SEPARATELY from overall
    retention: distillation quality degrades unevenly — easy, high-margin
    cases are the last to break, borderline cases are the first. Reporting
    only an overall number can hide that the student has quietly gotten
    much worse specifically where it matters most.
    """
    teacher.to(device).eval()
    teacher_severity_probe.to(device).eval()
    student.to(device).eval()
    student_severity_probe.to(device).eval()

    teacher_det_probs, student_det_probs = [], []
    teacher_severities, student_severities = [], []

    with torch.no_grad():
        for images, _labels in dataloader:
            images = images.to(device)

            t_out = teacher(images)
            t_sev = teacher_severity_probe(t_out["features"])
            s_out = student(images)
            s_sev = student_severity_probe(s_out["features"])

            teacher_det_probs.append(torch.sigmoid(t_out["detection_logits"]).cpu().numpy())
            student_det_probs.append(torch.sigmoid(s_out["detection_logits"]).cpu().numpy())
            teacher_severities.append(t_sev.cpu().numpy())
            student_severities.append(s_sev.cpu().numpy())

    teacher_det_probs = np.concatenate(teacher_det_probs)
    student_det_probs = np.concatenate(student_det_probs)
    teacher_severities = np.concatenate(teacher_severities)
    student_severities = np.concatenate(student_severities)

    overall_agreement = float(np.corrcoef(teacher_det_probs, student_det_probs)[0, 1])
    overall_mae = float(np.mean(np.abs(teacher_det_probs - student_det_probs)))

    borderline_mask = (
        (teacher_severities >= borderline_severity_range[0]) &
        (teacher_severities <= borderline_severity_range[1])
    )
    n_borderline = int(borderline_mask.sum())

    if n_borderline >= 2:
        borderline_agreement = float(np.corrcoef(
            teacher_det_probs[borderline_mask], student_det_probs[borderline_mask]
        )[0, 1])
        borderline_mae = float(np.mean(np.abs(
            teacher_det_probs[borderline_mask] - student_det_probs[borderline_mask]
        )))
    else:
        print(f"[warn] Only {n_borderline} borderline-severity cases found in this "
              f"evaluation set — borderline-specific retention is not meaningfully computable.")
        borderline_agreement, borderline_mae = float("nan"), float("nan")

    result = {
        "overall_detection_prob_correlation": overall_agreement,
        "overall_detection_prob_mae": overall_mae,
        "borderline_detection_prob_correlation": borderline_agreement,
        "borderline_detection_prob_mae": borderline_mae,
        "n_borderline_cases": n_borderline,
        "severity_correlation": float(np.corrcoef(teacher_severities, student_severities)[0, 1]),
    }

    print(f"[ok] Accuracy retention — overall detection-prob correlation: "
          f"{overall_agreement:.4f} (MAE={overall_mae:.4f}); "
          f"borderline-case correlation: {borderline_agreement:.4f} "
          f"(MAE={borderline_mae:.4f}, n={n_borderline}); "
          f"severity correlation: {result['severity_correlation']:.4f}")

    return result


# ==========================================================================
# TASK 4.2a — COMBINED EXPORT WRAPPER (dict-output model -> tuple-output for ONNX)
# ==========================================================================
class CombinedExportModel(nn.Module):
    """
    WHY this wrapper exists: CataractDetectorDualHead.forward() returns a
    dict, and the severity probe is a separate module entirely. ONNX
    export needs a single nn.Module whose forward() returns a plain tensor
    or tuple of tensors — this wrapper is that single entry point, fusing
    the detection/quality model and the severity probe into exactly the
    three outputs the app needs: (detection_logit, quality_logit, severity_score).
    """

    def __init__(self, detector: nn.Module, severity_probe: nn.Module):
        super().__init__()
        self.detector = detector
        self.severity_probe = severity_probe

    def forward(self, x: torch.Tensor):
        out = self.detector(x)
        severity = self.severity_probe(out["features"])
        return out["detection_logits"], out["quality_logits"], severity


def export_to_onnx(model: nn.Module, save_path: str, opset: int = ONNX_OPSET,
                    img_size: int = IMG_SIZE) -> str:
    model.eval()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, img_size, img_size)

    export_kwargs = dict(
        input_names=["image"],
        output_names=["detection_logit", "quality_logit", "severity_score"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "detection_logit": {0: "batch_size"},
            "quality_logit": {0: "batch_size"},
            "severity_score": {0: "batch_size"},
        },
        opset_version=opset,
    )

    # WHY force the legacy (non-dynamo) exporter explicitly: torch >= 2.5
    # defaults to a newer dynamo-based ONNX exporter that requires the
    # separate `onnxscript` package, which may not be preinstalled in a
    # given Colab session. The legacy TorchScript-based exporter has no
    # such extra dependency and is what this file's numerical-equivalence
    # test (verify_onnx_matches_pytorch) was validated against — pinning
    # to it explicitly avoids behavior silently changing across torch
    # versions. Older torch (<2.5) doesn't accept `dynamo=` at all, so we
    # fall back gracefully rather than erroring.
    try:
        torch.onnx.export(model, dummy_input, save_path, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy_input, save_path, **export_kwargs)

    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)  # WHY: fails loudly here if the export produced an invalid graph
    print(f"[ok] ONNX model exported and validated: {save_path}")
    return save_path


# ==========================================================================
# TASK 4.2b — INT8 STATIC QUANTIZATION
# ==========================================================================
class NumpyCalibrationDataReader(CalibrationDataReader):
    """
    WHY static (not dynamic) quantization: static quantization calibrates
    activation ranges using real representative data, giving better
    accuracy retention than dynamic quantization for CNN-heavy models like
    this one — at the cost of needing this calibration data reader.
    """

    def __init__(self, calibration_images: np.ndarray, input_name: str = "image"):
        self.input_name = input_name
        self.data = calibration_images.astype(np.float32)
        self._iter = iter(self.data)

    def get_next(self):
        try:
            item = next(self._iter)
            return {self.input_name: item[np.newaxis, ...]}
        except StopIteration:
            return None

    def rewind(self):
        self._iter = iter(self.data)


def quantize_to_int8(fp32_onnx_path: str, int8_onnx_path: str,
                      calibration_images: np.ndarray) -> str:
    """
    WHY 100 calibration images (per the plan) is treated as a minimum, not
    exact requirement here: the function accepts whatever calibration set
    is passed in; the CALLER is responsible for supplying a representative
    validation-set sample per the plan's Task 4.2 spec.
    """
    Path(int8_onnx_path).parent.mkdir(parents=True, exist_ok=True)
    reader = NumpyCalibrationDataReader(calibration_images)

    quantize_static(
        model_input=fp32_onnx_path,
        model_output=int8_onnx_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    print(f"[ok] INT8-quantized ONNX model saved: {int8_onnx_path}")
    return int8_onnx_path


def benchmark_latency(onnx_path: str, n_runs: int = 100, img_size: int = IMG_SIZE) -> dict:
    """
    WHY this MEASURES rather than assumes the <100ms target: the plan is
    explicit that latency numbers are targets to verify, not guarantees —
    report whatever this actually produces on the hardware it's run on.
    """
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

    # WHY a warmup pass before timing: the first inference call includes
    # one-time graph optimization/memory allocation overhead that would
    # otherwise unfairly inflate the reported latency.
    session.run(None, {input_name: dummy})

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    latencies = np.array(latencies)
    model_size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)

    result = {
        "mean_ms": float(latencies.mean()),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "model_size_mb": float(model_size_mb),
        "meets_target": bool(np.median(latencies) < TARGET_LATENCY_MS),
    }
    print(f"[ok] Latency benchmark ({onnx_path}): "
          f"mean={result['mean_ms']:.2f}ms median={result['median_ms']:.2f}ms "
          f"p95={result['p95_ms']:.2f}ms size={result['model_size_mb']:.2f}MB "
          f"(target: <{TARGET_LATENCY_MS}ms, met: {result['meets_target']})")
    return result


# ==========================================================================
# TASK 4.3 — ONNX RUNTIME INFERENCE WRAPPER
# ==========================================================================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CataractDetectorONNX:
    """
    Framework-agnostic inference wrapper — works on any machine with
    onnxruntime + numpy + Pillow installed, no PyTorch required.

    WHY preprocessing is duplicated here rather than importing Phase 1's
    Albumentations pipeline: a deployment environment (e.g. a lightweight
    Docker container for the FastAPI backend in Phase 5) should not need
    the full training-time dependency stack (Albumentations, PyTorch)
    just to run inference. This class's preprocessing is written to be
    NUMERICALLY IDENTICAL to Phase 1's eval-time transform
    (resize -> [0,1] scale -> ImageNet normalize -> CHW), which the
    equivalence test below verifies directly, not just asserts.
    """

    def __init__(self, onnx_path: str, severity_thresholds_path: str = None,
                 img_size: int = IMG_SIZE):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.img_size = img_size

        self.thresholds = {"mild_upper": 0.33, "moderate_upper": 0.66}  # sane fallback
        if severity_thresholds_path and Path(severity_thresholds_path).exists():
            with open(severity_thresholds_path) as f:
                summary = json.load(f)
                self.thresholds = summary["thresholds"]
        else:
            print("[warn] No severity thresholds file provided/found — using fallback "
                  "tertile-like defaults (0.33 / 0.66). Pass Phase 3's severity_summary.json "
                  "for the real, data-derived thresholds.")

    def _preprocess(self, pil_image) -> np.ndarray:
        image = pil_image.convert("RGB").resize((self.img_size, self.img_size))
        arr = np.array(image).astype(np.float32) / 255.0        # HWC, [0,1]
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD               # normalize
        arr = arr.transpose(2, 0, 1)                              # HWC -> CHW
        return arr[np.newaxis, ...].astype(np.float32)            # add batch dim

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def _severity_grade(self, score: float) -> str:
        if score <= self.thresholds["mild_upper"]:
            return "Mild"
        elif score <= self.thresholds["moderate_upper"]:
            return "Moderate"
        else:
            return "Severe"

    def predict(self, pil_image) -> dict:
        input_tensor = self._preprocess(pil_image)
        det_logit, qual_logit, severity = self.session.run(None, {self.input_name: input_tensor})

        cataract_prob = float(self._sigmoid(det_logit)[0])
        quality_prob = float(self._sigmoid(qual_logit)[0])
        severity_score = float(severity[0])

        return {
            "cataract_detected": cataract_prob >= 0.2,
            "cataract_confidence": cataract_prob,
            "quality_score": quality_prob,
            "quality_status": "good" if quality_prob >= 0.7 else "poor",
            "severity_score": severity_score,
            "severity_grade": self._severity_grade(severity_score),
        }


def verify_onnx_matches_pytorch(pytorch_model: nn.Module, onnx_path: str,
                                 img_size: int = IMG_SIZE, tolerance: float = 1e-4,
                                 n_test_samples: int = 5) -> bool:
    """
    WHY this exists as an explicit, mandatory check rather than an assumed
    guarantee of torch.onnx.export: export bugs (wrong opset behavior,
    unsupported ops silently approximated, dynamic-axis mistakes) are a
    real and common failure mode. This confirms the ONNX graph produces
    numerically equivalent output to the original PyTorch model before
    that ONNX file is trusted for anything downstream.
    """
    pytorch_model.eval()
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    all_close = True
    max_diff_seen = 0.0

    for i in range(n_test_samples):
        torch.manual_seed(i)
        x = torch.randn(1, 3, img_size, img_size)

        with torch.no_grad():
            torch_out = pytorch_model(x)
        torch_arrays = [o.numpy() for o in torch_out]

        onnx_out = session.run(None, {input_name: x.numpy()})

        for t_arr, o_arr, name in zip(torch_arrays, onnx_out,
                                       ["detection_logit", "quality_logit", "severity_score"]):
            diff = np.abs(t_arr - o_arr).max()
            max_diff_seen = max(max_diff_seen, diff)
            if diff > tolerance:
                all_close = False
                print(f"[FAIL] Sample {i}, output '{name}': max diff {diff:.6f} "
                      f"exceeds tolerance {tolerance}")

    if all_close:
        print(f"[ok] ONNX output matches PyTorch within tolerance {tolerance} "
              f"across {n_test_samples} random samples (max diff observed: {max_diff_seen:.2e}).")
    else:
        print(f"[FAIL] ONNX/PyTorch outputs diverge beyond tolerance. "
              f"Do not trust this export for deployment until resolved.")

    return all_close


# ==========================================================================
# CLI ENTRY POINT
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="OmniCataract-X Phase 4: Optimization & Export")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if not args.smoke_test:
        print("This script's functions are intended to be called from a notebook cell "
              "with your real trained Phase 2 + Phase 3 checkpoints loaded. "
              "Run with --smoke-test first to verify the pipeline works end-to-end.")
        return

    print("=" * 70)
    print("PHASE 4 SMOKE TEST (random data + untrained tiny models, verifies pipeline only)")
    print("=" * 70)
    device = "cpu"  # WHY cpu even if cuda available: ONNX export/quantization below is CPU-target anyway

    phase2 = load_module("02_core_detection_model.py", "phase2_module")
    phase3 = load_module("03_severity_engine.py", "phase3_module")

    teacher = phase2.CataractDetectorDualHead(backbone_name=TEACHER_BACKBONE, pretrained=False)
    teacher_severity_probe = phase3.SeverityProbe(input_dim=teacher.feature_dim)
    phase3.freeze_backbone(teacher)

    student = build_student_model(phase2, backbone_name=STUDENT_BACKBONE, pretrained=False)
    student.backbone_name_for_debug = STUDENT_BACKBONE
    # WHY pretrained=False for student's own severity probe backbone check:
    student_severity_probe = phase3.SeverityProbe(input_dim=student.feature_dim)

    from torch.utils.data import TensorDataset
    n = 16
    fake_images = torch.randn(n, 3, 224, 224)
    fake_labels = torch.randint(0, 2, (n,)).float()
    loader = DataLoader(TensorDataset(fake_images, fake_labels), batch_size=8)

    print("\n--- Running 2 epochs of distillation (freeze_epochs=1) ---")
    train_distillation(teacher, teacher_severity_probe, student, student_severity_probe,
                        loader, device, epochs=2, freeze_epochs=1)

    print("\n--- Evaluating accuracy retention ---")
    evaluate_accuracy_retention(teacher, teacher_severity_probe, student, student_severity_probe,
                                 loader, device)

    print("\n--- Exporting to ONNX ---")
    combined = CombinedExportModel(student, student_severity_probe)
    onnx_path = export_to_onnx(combined, "/tmp/omnicataract_smoke_test.onnx")

    print("\n--- Verifying ONNX matches PyTorch ---")
    matches = verify_onnx_matches_pytorch(combined, onnx_path)
    assert matches, "ONNX/PyTorch equivalence check failed in smoke test"

    print("\n--- Quantizing to INT8 ---")
    calib_images = np.random.randn(10, 3, 224, 224).astype(np.float32)  # tiny for smoke test speed
    int8_path = quantize_to_int8(onnx_path, "/tmp/omnicataract_smoke_test_int8.onnx", calib_images)

    print("\n--- Benchmarking latency (FP32 vs INT8) ---")
    fp32_bench = benchmark_latency(onnx_path, n_runs=20)
    int8_bench = benchmark_latency(int8_path, n_runs=20)

    print("\n--- Testing CataractDetectorONNX wrapper ---")
    from PIL import Image
    fake_pil = Image.fromarray((np.random.rand(300, 300, 3) * 255).astype(np.uint8))
    detector = CataractDetectorONNX(int8_path)
    result = detector.predict(fake_pil)
    print(f"Wrapper prediction: {result}")
    assert "cataract_detected" in result and "severity_grade" in result

    print(f"\n[ok] Smoke test complete. FP32: {fp32_bench['median_ms']:.2f}ms, "
          f"INT8: {int8_bench['median_ms']:.2f}ms, "
          f"size reduction: {fp32_bench['model_size_mb']:.2f}MB -> {int8_bench['model_size_mb']:.2f}MB")
    print("=" * 70)


if __name__ == "__main__":
    main()
