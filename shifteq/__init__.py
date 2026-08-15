"""Shift equivariance of NMS-free real-time detectors.

Two metric families, deliberately kept apart:

  oracle.py  dAP = AP_best - AP_worst, the Manfredi & Wang envelope. Per-image
             adversarial shift selection via greedy Algorithm 1. Comparable to prior
             work only at max_shift = 1.

  sweep.py   AP(delta) over a uniform 32-px shift grid. Every image gets the same
             delta, so each point is an ordinary COCO AP and the curve shows the
             network's stride response.

Collapsing one into the other -- quoting max-min over the sweep as "dAP", or the oracle
envelope as "the range over shifts" -- is the main way this study can go wrong on paper.
"""

__version__ = "0.1.0"

from . import (  # noqa: F401
    cocolite, data, detstore, infer, modelzoo, oracle, pipeline, shift,
    stabindex, stability, sweep, tta, utils,
)

__all__ = [
    "cocolite", "data", "detstore", "infer", "modelzoo", "oracle", "pipeline",
    "shift", "stabindex", "stability", "sweep", "tta", "utils",
]
