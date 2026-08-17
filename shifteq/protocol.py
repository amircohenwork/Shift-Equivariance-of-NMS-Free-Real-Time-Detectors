"""The frozen protocol: every constant this study depends on, in one place.

`PROTOCOL.md` is the human-readable pre-registration. This module is its machine-readable
twin, and the two are checked against each other by `tests/test_protocol_frozen.py`. Nothing
elsewhere in the package may hard-code one of these values; they are imported from here so
that changing a threshold is a visible, single-line, reviewable act rather than a default
argument quietly drifting.

`protocol_hash()` fingerprints the whole set. Every run manifest records it. If a number in
a results file was produced under a different protocol than the one you are reading about,
the hashes will not match and you will know.

Amendment policy: bump `VERSION`, add a dated entry to `CHANGELOG` below, and state in the
paper which version produced which table. Do not edit a frozen value in place.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

VERSION = "1.1.0"

CHANGELOG = [
    ("1.0.0", "2026-08-15", "Initial freeze. Protocol under which all results in "
                            "results/ and README.md were produced."),
    ("1.1.0", "2026-08-17", "Added CROSS_CHECK and PADDING. Both variations were already "
                            "declared as intended in v1.0.0 (PROTOCOL.md section 9); this "
                            "version specifies how they are analysed. Written after the "
                            "yolo26n full-validation results were known and before any "
                            "cross-model or padding sweep had been run. No v1.0.0 value "
                            "changed, so every yolo26n number stands unaltered."),
]

#: Hashes of superseded versions, so a run manifest recorded under an older protocol stays
#: traceable after an amendment. `protocol_hash()` only ever describes the current version.
PRIOR_HASHES = {
    "1.0.0": "7daf22d45cd62116",
}


# --------------------------------------------------------------------------------------
# 1. Shift protocol
# --------------------------------------------------------------------------------------
SHIFT = {
    # "grow": letterbox to `imgsz`, then embed that tile in a canvas of
    #         align(imgsz + max_shift, stride). Network input is LARGER than imgsz.
    #         Preserves full letterbox resolution and byte-identity with the Ultralytics
    #         val dataloader, at the cost of showing the network a size it was not
    #         validated at (full val, delta=0: +0.029 AP e2e / +0.068 nms; the +0.225
    #         quoted from gate B6's 500-image probe did not survive the full set).
    # "fit":  letterbox to (imgsz - max_shift), then embed in a canvas of exactly `imgsz`.
    #         Network input matches the validated size; costs a little resolution.
    #         Closer to Manfredi & Wang in spirit -- the shift budget comes out of the
    #         image scale rather than out of the input size.
    "embed_mode": "grow",
    "imgsz": 640,
    "stride": 32,
    # Canvas/letterbox fill. 114 is the Ultralytics letterbox gray. Manfredi & Wang used
    # black (0). -1 = replicate, -2 = reflect (Kayhan & van Gemert ablation).
    "pad_value": 114,
    # Shifts are (dy, dx), non-negative, because the canvas grows toward the bottom-right.
    # delta=(0,0) places the tile at the top-left corner.
    "shift_order": "(dy, dx)",
    "shift_range": "[0, max_shift]^2, integer pixels only (no interpolation)",
    # Ground truth is NEVER shifted. Predictions are mapped back to original-image
    # coordinates and evaluated against unmodified COCO annotations.
    "gt_transformed": False,
}


# --------------------------------------------------------------------------------------
# 2. Inference / post-processing (identical for both heads -- this is the ablation)
# --------------------------------------------------------------------------------------
INFERENCE = {
    "conf_thres": 0.001,     # Ultralytics val default
    "iou_thres": 0.7,        # NMS IoU, one-to-many path only
    "max_det": 300,          # verified neutral between heads (scripts/01b)
    "multi_label": True,     # Ultralytics val default
    "agnostic_nms": False,
    "dtype": "float32",
    "tf32": False,           # ON by default on Ampere; moves confidences by up to 0.137
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "fuse": False,           # fusing an end2end head deletes the one-to-many branch
    "nms_kernel": "torchvision.ops.nms",  # must be in sys.modules or Ultralytics silently
                                          # falls back to a different, slower kernel
    "letterbox_rect": False,  # square letterbox; the published table uses rect=True, and
                              # 01d quantifies the gap (+0.21 e2e, +0.05 nms)
}


# --------------------------------------------------------------------------------------
# 3. COCO detection AP (standard, restated so nothing is implicit)
# --------------------------------------------------------------------------------------
COCO_AP = {
    "iou_thresholds": "0.50:0.05:0.95 (10)",
    "recall_thresholds": "0.00:0.01:1.00 (101)",
    "max_dets": 100,          # COCOeval maxDets[-1], applied per (image, category)
    "area_ranges": {"all": [0, 1e10], "small": [0, 1024],
                    "medium": [1024, 9216], "large": [9216, 1e10]},
    "crowd_handling": "iscrowd GT are ignored; IoU against them is "
                      "intersection-over-detection-area",
    "empty_category": "category with GT but no detections contributes AP 0; "
                      "category with no GT is excluded",
    "tie_break": "(category asc, score desc, image asc, shift asc) -- a total order, so "
                 "rankings are reproducible; pycocotools leaves ties to input order "
                 "(difference <1e-9)",
    "reference": "faster-coco-eval, agreement verified to 1.1e-16 (gate B4)",
}


# --------------------------------------------------------------------------------------
# 4. Oracle envelope (Manfredi & Wang Algorithm 1)
# --------------------------------------------------------------------------------------
ORACLE = {
    "max_shift": 1,           # the ONLY value comparable to their Table 1
    "step": 1,
    "iterations_K": 1,        # paper: further iterations move AP <1%
    # Sec. 3.1: "In practice, we use AP50 as the metric". AP(0.50:0.95) is read off AT the
    # assignment AP50 chose -- it is not a separate AP-driven search.
    "objective": "AP50",
    "objective_area": "all",
    "init": "delta*_i = (0,0) for every image",
    "image_order": "dataset order (sorted COCO image_id)",
    "n_images": 5000,         # full COCO val2017
}


# --------------------------------------------------------------------------------------
# 5. Uniform-shift sweep
# --------------------------------------------------------------------------------------
SWEEP = {
    "max_shift": 31,          # one full stride minus one pixel
    "step": 2,                # underestimates the AP range by ~18% vs step 1 (validated,
                              # scripts/05 --step 1 --tag dense). Quote as a lower bound.
    "n_images": 1000,
    "subset_seed": 0,
    "subset_rule": "numpy default_rng(seed).choice without replacement, then sorted",
}


# --------------------------------------------------------------------------------------
# 6. Instance-level matching -- the definition flicker / jitter / SI all rest on
# --------------------------------------------------------------------------------------
MATCHING = {
    "score_threshold_tau": 0.25,   # Ultralytics predict-time default. At COCO's 0.001
                                   # almost nothing ever drops out and flicker is
                                   # meaningless, so this is a real, declared choice.
    "iou_threshold": 0.5,
    "algorithm": "greedy over detections in descending score order",
    "assignment": "one-to-one: each GT may be claimed by at most one detection per shift",
    "candidate_rule": "highest-IoU GT among those not yet claimed and with IoU >= 0.5",
    "iou_tie_break": "lowest GT index (numpy argmax)",
    "score_sort": "stable (mergesort) on -score",
    "two_passes": "class-aware (requires predicted class == GT class) drives detection and "
                  "jitter; class-agnostic (ignores class) drives class-flip",
    "crowd_gt": "matched using intersection-over-detection-area, but excluded from all "
                "reported aggregates",
    "unmatched_gt": "counted as not-detected at that shift; contributes SI = 0",
    "no_hungarian": "deliberately greedy, matching COCOeval's own rule rather than an "
                    "optimal assignment",
}


# --------------------------------------------------------------------------------------
# 7. Stability Index (2-D adaptation of arXiv:2407.04305)
# --------------------------------------------------------------------------------------
STABILITY_INDEX = {
    "formula": "SI2D = SI_c * (SI_l + SI_e) / 2",
    "SI_c": "max(0, 1 - |c1 - c2| / (c^0.99 - c^0.01))",
    "SI_l": "IoU of boxes sharing the GT extent, differing only in centre",
    "SI_e": "IoU of boxes sharing the GT centre, differing only in extent",
    "SI_h": "omitted -- no 2-D analogue of heading",
    "pivot": "the ground-truth box (the 3-D paper's geometric mean over two frames "
             "collapses to this when the object itself does not move)",
    "confidence_percentiles": "c^0.01 and c^0.99 taken from the delta=(0,0) pass and "
                              "frozen for the whole grid",
    "missing_in_one_member": "SI = 0 (a missed detection is maximal instability)",
    "pairing_mode": "ref",   # every delta against delta=(0,0); "adjacent" also available
}


# --------------------------------------------------------------------------------------
# 8. Shift test-time augmentation
# --------------------------------------------------------------------------------------
TTA = {
    "merge_nms_iou": 0.7,
    "max_det": 300,
    "strategies": ["nms", "topk", "vote"],
    "vote_rule": "NMS, then keep detections backed by >= half the shifts at IoU >= 0.7",
    "headline_comparison": "nms vs topk on identical detections -- the cost of staying "
                           "NMS-free under shift-TTA",
}


# --------------------------------------------------------------------------------------
# 10. Cross-model check                                                    (added v1.1.0)
# --------------------------------------------------------------------------------------
CROSS_CHECK = {
    "primary_model": "yolo26n",
    # The only other released checkpoint carrying a one-to-one and a one-to-many branch
    # trained together, hence the one arm that can replicate the core ablation.
    "replication_model": "yolov10n",
    "universality_models": ["yolov8n", "yolo11n", "rtdetr-l"],
    "n_images": 5000,
    "max_shift": 63,
    "step": 1,
    # Declared in advance: the curve on which yolo26n showed its largest head effect.
    # Everything else is secondary and reported as a count, not as ten separate claims.
    "primary_curve": "x/AP_small",
    "test": "paired sign-flip dispersion test across delta, absolute deviations",
    "alpha": 0.05,
    "permutations": 20000,
    "failed_replication_means": "primary curve not significant, or significant with the "
                                "opposite sign",
    "cross_model_dispersion_reported_as": ["AP points", "coefficient of variation"],
    "sign_agreement_is": "descriptive concordance, not a p-value (the curves are not "
                         "independent of one another)",
    "gate_a_tolerance_ap": 0.1,
    "gate_a_exceptions": {
        # Declared, not applied silently. See modelzoo.MODELS["rtdetr-l"].
        "rtdetr-l": "0.3 -- the published 53.0 is the original paper's PaddlePaddle "
                    "evaluation quoted for a PyTorch port, not an Ultralytics "
                    "re-measurement; no absolute-AP claim is made from this arm",
    },
}

# --------------------------------------------------------------------------------------
# 11. Padding ablation                                                     (added v1.1.0)
# --------------------------------------------------------------------------------------
PADDING = {
    "hypothesis": "Kayhan & van Gemert (CVPR'20): a constant border is an implicit "
                  "absolute-position cue, which competes with aliasing as the explanation "
                  "for the stride-periodic response",
    "prediction_if_aliasing": "stride periodicity survives a non-constant border",
    "prediction_if_border_cue": "stride periodicity weakens or disappears",
    # The grouping IS the design: 0 controls for the constant's *value*, {-1,-2} remove
    # its *constant-ness*. Comparing 114 against 0 alone decides nothing.
    "constant_border_values": [114, 0],
    "content_border_values": [-1, -2],
    "value_meanings": {"114": "gray (frozen protocol)", "0": "black (Manfredi & Wang)",
                       "-1": "replicate", "-2": "reflect"},
    "sets_both_borders": "letterbox bars at cache build time AND canvas frame at embed time",
    "n_images": 1000,
    "baseline": 114,
    "comparison": "relative dispersion (the padding moves the AP level, so absolute points "
                  "would conflate 'wobbles less' with 'scores lower')",
    # Pre-declared so a small difference cannot be read as a null result after the fact.
    "underpowered_if_group_delta_within": 0.10,
    "escalate_to_n_images": 5000,
}


ALL: dict[str, Any] = {
    "version": VERSION,
    "shift": SHIFT,
    "inference": INFERENCE,
    "coco_ap": COCO_AP,
    "oracle": ORACLE,
    "sweep": SWEEP,
    "matching": MATCHING,
    "stability_index": STABILITY_INDEX,
    "tta": TTA,
    "cross_check": CROSS_CHECK,
    "padding": PADDING,
}


def protocol_hash() -> str:
    """Stable fingerprint of the frozen protocol. Recorded in every run manifest."""
    return hashlib.sha256(
        json.dumps(ALL, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def describe() -> str:
    lines = [f"shifteq protocol v{VERSION}  (hash {protocol_hash()})"]
    for section, body in ALL.items():
        if section == "version":
            continue
        lines.append(f"\n[{section}]")
        for k, v in body.items():
            lines.append(f"  {k:26s} {v}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
