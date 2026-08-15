"""Fork-based worker pool for the CPU side of the sweep.

The sweep alternates a GPU-bound step (forward passes for one delta) with two CPU-bound
steps (COCO matching over 4 area ranges, and the flicker/jitter partial). Run serially,
the GPU idles through the CPU half and the CPU idles through the GPU half; on a 24-core
box that wastes most of the machine.

So: submit delta_k's CPU work to a pool and immediately start delta_{k+1}'s forward pass,
collecting results one step behind.

Ground truth goes to the workers once, through the pool initializer, never per task.
Detections travel as raw arrays (or are re-read from disk in the worker) rather than as
objects. `forkserver` is preferred over `fork`: by the time the pool is created torch has
already started worker threads and possibly a CUDA context, and forking a multithreaded
process is a documented deadlock risk. forkserver pays one pickle of the ground truth per
worker -- a few MB, once -- to avoid it.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from .cocolite import accumulate, build_matches, canonical_order
from .data import CocoGT
from .detstore import ShiftDets
from .stability import GtIndex, ShiftPartial, shift_partial
from .utils import LOGGER

_G: dict[str, Any] = {}


def _mp_context():
    """forkserver first: by the time a pool is created torch has started worker threads
    and possibly a CUDA context, and forking a multithreaded process risks inheriting a
    held lock. The initializer payload is a few MB, so paying one pickle per worker to
    avoid that is free."""
    want = os.environ.get("SHIFTEQ_MP_CONTEXT")
    order = [want] if want else ["forkserver", "fork"]
    for name in order:
        try:
            return mp.get_context(name)
        except ValueError:
            continue
    return None


def _init_worker(gt: CocoGT, idx: GtIndex, class_map_inv: np.ndarray,
                 npig: dict[str, np.ndarray], tau: float, iou_thr: float) -> None:
    _G.update(gt=gt, idx=idx, cmi=class_map_inv, npig=npig, tau=tau, iou_thr=iou_thr)
    # The matching code is elementwise numpy with no BLAS calls, so there is nothing to
    # oversubscribe -- but torch, if it gets imported here, defaults to all cores.
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass


def _task_area(payload: tuple, area: str) -> tuple[str, dict]:
    dets = _unpack(payload)
    m = build_matches(_G["gt"], dets, _G["cmi"], area_key=area)
    o = canonical_order(m)
    r = accumulate(m.cat[o], m.match[o], m.ign[o], _G["npig"][area], _G["gt"].num_cats)
    return area, r


def _task_stability(payload: tuple) -> ShiftPartial:
    dets = _unpack(payload)
    return shift_partial(_G["gt"], _G["idx"], dets, _G["cmi"], _G["tau"], _G["iou_thr"])


def _pack(d: ShiftDets) -> tuple:
    """Ship the raw arrays; ShiftDets rebuilds its offsets on the far side."""
    return (d.counts, d.cls, d.score, d.xyxy, d.shift)


def _unpack(p: tuple) -> ShiftDets:
    counts, cls, score, xyxy, shift = p
    return ShiftDets(counts=counts, cls=cls, score=score, xyxy=xyxy, shift=tuple(shift))


@dataclass
class PendingShift:
    shift: tuple[int, int]
    n_dets: int
    t_infer: float
    area_futures: list
    stab_future: Any | None


class SweepPool:
    def __init__(self, gt: CocoGT, idx: GtIndex, class_map_inv: np.ndarray,
                 npig: dict[str, np.ndarray], tau: float = 0.25, iou_thr: float = 0.5,
                 workers: int | None = None):
        n = workers if workers is not None else max(1, min(16, (os.cpu_count() or 4) - 2))
        self.workers = n
        self._ex = None
        if n > 1:
            ctx = _mp_context()
            if ctx is not None:
                self._ex = ProcessPoolExecutor(
                    max_workers=n, mp_context=ctx, initializer=_init_worker,
                    initargs=(gt, idx, class_map_inv, npig, tau, iou_thr),
                )
                LOGGER.info(f"sweep pool: {n} {ctx.get_start_method()} workers")
        if self._ex is None:
            _init_worker(gt, idx, class_map_inv, npig, tau, iou_thr)
            LOGGER.info("sweep pool: running inline (workers=1 or no usable start method)")

    def submit(self, dets: ShiftDets, areas: tuple[str, ...], want_stability: bool,
               t_infer: float) -> PendingShift:
        p = _pack(dets)
        if self._ex is None:
            return PendingShift(
                dets.shift, dets.n_dets, t_infer,
                [_Done(_task_area(p, a)) for a in areas],
                _Done(_task_stability(p)) if want_stability else None,
            )
        return PendingShift(
            dets.shift, dets.n_dets, t_infer,
            [self._ex.submit(_task_area, p, a) for a in areas],
            self._ex.submit(_task_stability, p) if want_stability else None,
        )

    def close(self):
        if self._ex is not None:
            self._ex.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class _Done:
    """Future-shaped wrapper for the inline path."""

    def __init__(self, value):
        self._v = value

    def result(self):
        return self._v


def match_shifts_parallel(gt: CocoGT, idx: GtIndex, class_map_inv: np.ndarray,
                          npig: dict[str, np.ndarray], store, shifts: list[tuple[int, int]],
                          areas: tuple[str, ...], workers: int | None = None):
    """Build MatchArrays for every (shift, area) in parallel. Used by the oracle setup,
    where the 16 tasks at M=1 are fully independent and each takes minutes."""
    n = workers if workers is not None else max(1, min(16, (os.cpu_count() or 4) - 2))
    tasks = [(s, a) for s in shifts for a in areas]
    out: dict[str, list] = {a: [None] * len(shifts) for a in areas}
    si = {s: i for i, s in enumerate(shifts)}

    if n <= 1:
        for s, a in tasks:
            out[a][si[s]] = build_matches(gt, store.load(*s), class_map_inv, area_key=a)
        return out

    ctx = _mp_context()
    if ctx is None:
        for s, a in tasks:
            out[a][si[s]] = build_matches(gt, store.load(*s), class_map_inv, area_key=a)
        return out

    LOGGER.info(f"matching {len(tasks)} (shift, area) tasks on {n} workers")
    with ProcessPoolExecutor(max_workers=n, mp_context=ctx, initializer=_init_worker,
                             initargs=(gt, idx, class_map_inv, npig, 0.25, 0.5)) as ex:
        futs = {ex.submit(_task_match_from_store, str(store.dir), s, a): (s, a) for s, a in tasks}
        for f, (s, a) in futs.items():
            out[a][si[s]] = f.result()
            LOGGER.info(f"  matched delta={s} area={a}")
    return out


def _task_match_from_store(store_dir: str, shift: tuple[int, int], area: str):
    """Load detections inside the worker so ~30 MB per shift is read from disk rather
    than pushed through a pipe from the parent."""
    from pathlib import Path

    from .detstore import DetStore

    st = DetStore(Path(store_dir).parent, Path(store_dir).name)
    return build_matches(_G["gt"], st.load(*shift), _G["cmi"], area_key=area)
