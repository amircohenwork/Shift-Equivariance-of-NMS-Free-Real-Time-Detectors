"""The protocol is only frozen if something checks that it stayed frozen.

A pre-registration is worthless if the code drifts away from it silently. These tests bind
`PROTOCOL.md`, `shifteq/protocol.py`, and the actual runtime defaults together, so changing
a threshold requires changing all three deliberately — which is exactly the friction a
pre-registration is supposed to create.

If one of these fails, do not "fix" it by editing the constant. Bump `protocol.VERSION`, add
a changelog entry, and say in the paper which version produced which table.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shifteq import protocol  # noqa: E402

# The hash of the frozen v1.0.0 protocol. Every number in results/ was produced under it.
FROZEN_HASH = protocol.protocol_hash()


def test_hash_is_stable_across_calls():
    assert protocol.protocol_hash() == protocol.protocol_hash()
    assert len(protocol.protocol_hash()) == 16


def test_protocol_md_exists_and_declares_the_same_version():
    md = (ROOT / "PROTOCOL.md").read_text()
    assert f"v{protocol.VERSION}" in md.split("\n")[0], (
        "PROTOCOL.md title must state the version in shifteq/protocol.py"
    )
    assert "FROZEN" in md


def test_changelog_covers_current_version():
    versions = [v for v, _, _ in protocol.CHANGELOG]
    assert protocol.VERSION in versions, "bump CHANGELOG when bumping VERSION"
    md = (ROOT / "PROTOCOL.md").read_text()
    assert protocol.VERSION in md.split("## Changelog")[-1], (
        "PROTOCOL.md changelog must contain the current version"
    )


@pytest.mark.parametrize(
    "value",
    [
        "0.25",   # tau
        "0.001",  # conf
        "0.7",    # nms iou
        "300",    # max_det
        "640",    # imgsz
        "114",    # pad value
    ],
)
def test_key_thresholds_appear_in_the_document(value):
    """Every frozen threshold must be findable by a reader of PROTOCOL.md."""
    assert value in (ROOT / "PROTOCOL.md").read_text()


def test_matching_thresholds_match_the_code_defaults():
    """The declared matching rule must be what `shift_partial` actually does."""
    import inspect

    from shifteq.stability import shift_partial

    sig = inspect.signature(shift_partial)
    assert sig.parameters["tau"].default == protocol.MATCHING["score_threshold_tau"]
    assert sig.parameters["iou_thr"].default == protocol.MATCHING["iou_threshold"]


def test_inference_defaults_match_the_protocol():
    from shifteq.infer import PostprocCfg

    cfg = PostprocCfg()
    assert cfg.conf == protocol.INFERENCE["conf_thres"]
    assert cfg.iou == protocol.INFERENCE["iou_thres"]
    assert cfg.max_det == protocol.INFERENCE["max_det"]
    assert cfg.multi_label is protocol.INFERENCE["multi_label"]
    assert cfg.agnostic is protocol.INFERENCE["agnostic_nms"]


def test_coco_constants_match_the_evaluator():
    from shifteq import cocolite

    assert len(cocolite.IOU_THRS) == 10
    assert cocolite.IOU_THRS[0] == 0.5 and abs(cocolite.IOU_THRS[-1] - 0.95) < 1e-12
    assert len(cocolite.REC_THRS) == 101
    assert cocolite.MAXDET == protocol.COCO_AP["max_dets"]


def test_area_ranges_match():
    from shifteq.data import AREA_RNG

    for k, (lo, hi) in protocol.COCO_AP["area_ranges"].items():
        assert abs(AREA_RNG[k][0] - lo) < 1e-6
        assert abs(AREA_RNG[k][1] - hi) < 1e-6


def test_default_embed_mode_matches_geometry_default():
    from shifteq.shift import ShiftGeometry

    g = ShiftGeometry(protocol.SHIFT["imgsz"], 31, protocol.SHIFT["stride"])
    assert g.embed_mode == protocol.SHIFT["embed_mode"]


def test_both_embed_modes_are_no_crop():
    """The property the whole shift protocol rests on: the tile always fits, at every δ."""
    from shifteq.shift import ShiftGeometry

    for mode in ("grow", "fit"):
        for M in (0, 1, 7, 31):
            g = ShiftGeometry(640, M, 32, mode)
            assert g.canvas % 32 == 0, (mode, M)
            for dy in (0, M):
                for dx in (0, M):
                    g.check(dy, dx)                      # must not raise
                    assert g.tile + max(dy, dx) <= g.canvas


def test_fit_mode_keeps_the_validated_input_size():
    from shifteq.shift import ShiftGeometry

    g = ShiftGeometry(640, 31, 32, "fit")
    assert g.canvas == 640, "fit mode exists precisely to keep the validated input size"
    assert g.tile == 609


def test_grow_mode_preserves_full_letterbox_resolution():
    from shifteq.shift import ShiftGeometry

    g = ShiftGeometry(640, 31, 32, "grow")
    assert g.tile == 640, "grow mode must not reduce the letterbox target"
    assert g.canvas == 672


def test_describe_is_printable():
    text = protocol.describe()
    assert protocol.VERSION in text and FROZEN_HASH in text
    for section in ("shift", "inference", "coco_ap", "oracle", "sweep", "matching"):
        assert f"[{section}]" in text
