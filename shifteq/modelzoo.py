"""Model loading and head selection, with hard guards against layer fusion.

Why fusion is a correctness issue here, not a performance detail
----------------------------------------------------------------
Two independent reasons, both fatal to this study:

1. `ultralytics.nn.tasks.BaseModel.fuse()` contains

       if isinstance(m, Detect) and getattr(m, "end2end", False):
           m.fuse()          # -> Detect.fuse(): self.cv2 = self.cv3 = None

   i.e. fusing a model whose head is in end-to-end mode **deletes the one-to-many
   branch in place**. The whole premise of the core ablation is that both aggregation
   heads live on one checkpoint with identical backbone/neck weights. Fuse once with
   end2end=True and the NMS head is gone from that process; a later
   `end2end=False` pass then dies in `Detect._inference` on an empty prediction dict.

   `AutoBackend` defaults to `fuse=True`, and both `BasePredictor.setup_model` and
   `BaseValidator.__call__` go through it. So *the default Ultralytics inference path
   destroys the ablation*. That is what this module exists to prevent.

2. Conv+BN folding is not numerically neutral. It replaces two ops by one with
   pre-multiplied weights, changing the low-order bits of every activation. This study
   measures how a 1-px translation perturbs detector output; a numerics change of the
   same order as the effect under study is a confound we simply refuse to accept.

Everything here therefore loads with `fuse=False` and asserts it stayed that way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from .utils import LOGGER, state_dict_fingerprint

HeadMode = Literal["e2e", "nms"]

# Preprocessing family. YOLO models letterbox (aspect preserved + gray pad);
# RT-DETR's dataset overrides load_image(rect_mode=False), i.e. it *stretches* to a
# square. De-shifting has to invert whichever one was used.
PreprocMode = Literal["letterbox", "stretch"]


@dataclass
class ModelSpec:
    name: str
    weights: str
    preproc: PreprocMode = "letterbox"
    imgsz: int = 640
    stride: int = 32
    #: which aggregation heads this checkpoint can serve
    heads: tuple[HeadMode, ...] = ("e2e", "nms")
    #: published COCO val mAP50-95 per head, for the Phase-0 gate (None = unknown)
    published_map: dict[str, float] = field(default_factory=dict)
    #: where each published number comes from, and how tight a match to expect. Numbers
    #: re-evaluated by Ultralytics on their own checkpoint should land inside 0.1 AP;
    #: numbers quoted from an upstream paper's own evaluation framework need not.
    published_source: str = "Ultralytics model card (their own re-evaluation)"
    #: Gate-A tolerance in AP. Loosened only where the published number was produced by a
    #: different codebase, and never silently -- see PROTOCOL.md.
    gate_tol: float = 0.1


# Published numbers: Ultralytics YOLO26 model card / arXiv:2606.03748 Table 7.
#   "mAP50-95"      -> one-to-many head + NMS
#   "mAP50-95 (e2e)"-> one-to-one head, NMS-free
MODELS: dict[str, ModelSpec] = {
    "yolo26n": ModelSpec(
        name="yolo26n",
        weights="yolo26n.pt",
        heads=("e2e", "nms"),
        published_map={"nms": 40.9, "e2e": 40.1},
    ),
    "yolo26s": ModelSpec(
        name="yolo26s", weights="yolo26s.pt", heads=("e2e", "nms"),
        published_map={"nms": 48.6, "e2e": 47.8},
    ),
    # Cross-checks. YOLOv10 also carries a dual head, so it gives a second within-model
    # head ablation (only its e2e number is published). v8/11 are one-to-many only.
    "yolov10n": ModelSpec(
        name="yolov10n", weights="yolov10n.pt", heads=("e2e", "nms"),
        published_map={"e2e": 38.5},
        published_source="YOLOv10 (arXiv:2405.14458) Tab. 1, one-to-one head. The "
                         "one-to-many branch ships in the checkpoint but was never "
                         "tabulated, so it is measured here without a target.",
    ),
    "yolov8n": ModelSpec(name="yolov8n", weights="yolov8n.pt", heads=("nms",),
                         published_map={"nms": 37.3}),
    "yolo11n": ModelSpec(name="yolo11n", weights="yolo11n.pt", heads=("nms",),
                         published_map={"nms": 39.5}),
    # RT-DETR is the one arm whose target was not produced by this codebase's ecosystem:
    # 53.0 is the RT-DETR-R50 number from the original paper's PaddlePaddle evaluation,
    # which Ultralytics quotes for their PyTorch port rather than re-measuring. Measured
    # here at 52.77 with Ultralytics' own validator, and the residual does not move with
    # conf or max_det (results/phase0/rtdetr_protocol_probe.json), so it is a port gap,
    # not a harness bug. The gate is therefore a sanity check -- a preprocessing mistake
    # would cost whole AP points, not a quarter of one -- rather than a reproduction.
    "rtdetr-l": ModelSpec(
        name="rtdetr-l", weights="rtdetr-l.pt", preproc="stretch",
        heads=("e2e",), published_map={"e2e": 53.0}, gate_tol=0.3,
        published_source="RT-DETR (arXiv:2304.08069) R50, evaluated in PaddlePaddle by the "
                         "original authors -- a cross-framework comparison, not an "
                         "Ultralytics re-evaluation.",
    ),
}


class FusionError(RuntimeError):
    pass


def resolve_device(device) -> torch.device:
    """Accept both conventions. Ultralytics writes GPUs as "0" / "0,1"; torch.device
    rejects those, and `load_checkpoint` passes whatever it is straight to `.to()`."""
    if isinstance(device, torch.device):
        return device
    try:
        return torch.device(str(device))
    except (RuntimeError, ValueError):
        from ultralytics.utils.torch_utils import select_device

        return select_device(str(device), verbose=False)


def fix_rtdetr_anchor_cache(net: nn.Module) -> bool:
    """Work around an Ultralytics 8.4.x device bug in `RTDETRDecoder`.

    The decoder caches its anchor grid and validity mask as *plain attributes*, keyed on the
    feature-map shapes:

        if self.dynamic or self.shapes != shapes:
            self.anchors, self.valid_mask = self._generate_anchors(..., device=feats.device)
        features = self.enc_output(self.valid_mask * feats)

    `nn.Module.to()` moves parameters and registered buffers; it does not touch plain
    attributes. Ultralytics runs a CPU forward at 640x640 inside
    `get_flops_with_torch_profiler` while reporting the model summary, *before* the model
    reaches the GPU. That populates the cache on CPU with exactly the shapes validation will
    use, so the regeneration branch is skipped and the first CUDA forward dies with
    "Expected all tensors to be on the same device". Verified against rtdetr-l.pt on
    ultralytics 8.4.120 / torch 2.13.

    Setting `dynamic` makes the decoder rebuild the grid on every forward from
    `feats.device`. It is a cache-invalidation flag, not a numerics change: the regenerated
    tensors are bit-identical to the ones the cache was supposed to hold, and the rebuild is
    a few hundred microseconds against a 400 ms forward pass. Returns True if a decoder was
    found, so callers can record that the workaround applied.
    """
    from ultralytics.nn.modules.head import RTDETRDecoder

    found = False
    for m in net.modules():
        if isinstance(m, RTDETRDecoder):
            m.dynamic = True
            m.shapes = []
            found = True
    return found


def _iter_detect_heads(net: nn.Module):
    from ultralytics.nn.modules.head import Detect

    for m in net.modules():
        if isinstance(m, Detect):
            yield m


def available_heads(net: nn.Module) -> tuple[HeadMode, ...]:
    """Which aggregation paths this loaded module can actually serve."""
    heads = [m for m in _iter_detect_heads(net)]
    if not heads:
        return ("e2e",)  # RT-DETR and friends: set prediction is the only path
    h = heads[-1]
    out: list[HeadMode] = []
    if hasattr(h, "one2one_cv2"):
        out.append("e2e")
    if getattr(h, "cv2", None) is not None and getattr(h, "cv3", None) is not None:
        out.append("nms")
    return tuple(out)


def assert_unfused(net: nn.Module, require_both_heads: bool = True) -> dict:
    """Fail loudly if anything in the graph has been folded or amputated.

    Returns a small dict of evidence that callers should persist into their run
    manifest, so a reviewer can check after the fact that no run was fused.
    """
    n_bn = sum(isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)) for m in net.modules())
    if n_bn < 10:
        raise FusionError(
            f"only {n_bn} BatchNorm layers left in the graph -- this model has been fused. "
            "Reload with fuse=False (see shifteq.modelzoo.load_detector)."
        )

    evidence = {"n_batchnorm": n_bn, "heads": []}
    for m in _iter_detect_heads(net):
        has_o2m = getattr(m, "cv2", None) is not None and getattr(m, "cv3", None) is not None
        has_o2o = hasattr(m, "one2one_cv2")
        evidence["heads"].append(
            {"type": type(m).__name__, "one2many_present": bool(has_o2m), "one2one_present": bool(has_o2o)}
        )
        if require_both_heads and not has_o2m:
            raise FusionError(
                "Detect.cv2/cv3 are None: the one-to-many branch was removed by Detect.fuse(). "
                "This checkpoint instance can no longer serve the NMS path."
            )
    return evidence


def load_detector(
    spec: ModelSpec | str,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[nn.Module, ModelSpec]:
    """Load an Ultralytics detector as a bare `nn.Module`, unfused, in eval mode.

    We deliberately bypass `YOLO()`/`AutoBackend`/`BasePredictor` and call
    `load_checkpoint(..., fuse=False)` directly, because those wrappers all fuse.
    """
    from ultralytics.nn.tasks import load_checkpoint

    if isinstance(spec, str):
        if spec not in MODELS:
            raise KeyError(f"unknown model '{spec}'; known: {sorted(MODELS)}")
        spec = MODELS[spec]

    device = resolve_device(device)
    net, _ckpt = load_checkpoint(spec.weights, device=device, fuse=False)
    net = net.to(device=device, dtype=dtype).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    if fix_rtdetr_anchor_cache(net):
        LOGGER.info(f"{spec.name}: RTDETRDecoder anchor cache set to rebuild per forward "
                    "(ultralytics device-placement workaround)")

    assert_unfused(net, require_both_heads=False)
    have = available_heads(net)
    missing = [h for h in spec.heads if h not in have]
    if missing:
        LOGGER.warning(
            f"{spec.name}: declared heads {spec.heads} but the checkpoint only serves {have}; "
            f"{missing} will fail at set_head()."
        )
    LOGGER.info(f"loaded {spec.name} from {spec.weights} on {device} "
                f"(unfused, {dtype}, heads={have})")
    return net, spec


def set_head(net: nn.Module, mode: HeadMode, max_det: int = 300, agnostic_nms: bool = False) -> None:
    """Select the aggregation path without touching a single weight.

    `DetectionModel.end2end` is a property whose setter forwards to
    `set_head_attr(end2end=...)`, which sets `Detect._end2end`. Flipping it is a pure
    control-flow change inside `Detect.forward`: with True the head runs `one2one_*`
    and its internal top-k `postprocess`; with False it runs `cv2/cv3` and leaves NMS
    to the caller. Backbone and neck are byte-identical across the two runs -- which is
    exactly the property that makes the head the only free variable in the ablation.
    """
    if mode not in ("e2e", "nms"):
        raise ValueError(mode)
    want = mode == "e2e"
    have = available_heads(net)
    if mode not in have:
        raise ValueError(
            f"this checkpoint cannot serve the '{mode}' path (available: {have}). "
            + ("The one-to-many branch is missing -- most likely the model was fused "
               "while end2end was True, which calls Detect.fuse() and sets cv2 = cv3 = None."
               if mode == "nms" else "No one2one_cv2 branch in the head.")
        )
    if not hasattr(net, "end2end"):
        # Architectures with a single aggregation path (RT-DETR): nothing to switch, and
        # asking for their one path is not an error.
        LOGGER.info(f"{type(net).__name__} has no end2end switch; '{mode}' is its only path")
        return
    net.end2end = want
    if want:
        # Mirrors BaseValidator.__call__: the head's own top-k needs to know max_det.
        net.set_head_attr(max_det=max_det, agnostic_nms=agnostic_nms)
    # Re-assert: nothing above may have amputated a branch.
    assert_unfused(net, require_both_heads=False)


def head_is(net: nn.Module) -> HeadMode:
    """Which path a forward pass will take right now."""
    if not hasattr(net, "end2end"):
        return "e2e" if available_heads(net) == ("e2e",) else "nms"
    return "e2e" if net.end2end else "nms"


def weight_fingerprint(net: nn.Module) -> str:
    return state_dict_fingerprint(net)


def resolve_weights(spec: ModelSpec) -> Path:
    """Materialise the checkpoint locally and return its path (for hashing).

    `check_file` only *searches*; released weights come from the assets release, which is
    what `attempt_download_asset` knows how to fetch.
    """
    from ultralytics.utils.downloads import attempt_download_asset

    return Path(attempt_download_asset(spec.weights))
