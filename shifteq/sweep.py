"""The uniform-shift curve AP(delta) over the 32-px grid.

Every image gets the *same* shift, so AP(delta) is an ordinary COCO number and the curve
is directly interpretable: its period exposes the network's stride response, its range is
the honest "how much does a translation cost you in practice" figure, and its shape can be
compared across heads without any oracle machinery.

This is a different object from `oracle.dAP`. Reporting max(AP) - min(AP) over this grid
as "dAP" would be wrong, and reporting the oracle envelope as "the range over shifts"
would be wrong in the other direction (it is much wider, because it optimises per image).
Keep them in separate tables.

Results stream to a JSONL as each shift finishes, so a run that dies at shift 200 of 256
resumes from 200.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .cocolite import npig_per_cat
from .data import AREA_KEYS, CocoGT
from .detstore import DetStore
from .infer import ShiftRunner
from .pipeline import PendingShift, SweepPool
from .stability import StabilityAccumulator
from .utils import LOGGER, shift_grid


def _rows(path: Path) -> dict[tuple[int, int], dict]:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[(r["dy"], r["dx"])] = r
    return out


def run_sweep(
    runner: ShiftRunner,
    gt: CocoGT,
    class_map_inv: np.ndarray,
    out_jsonl: Path,
    max_shift: int = 31,
    step: int = 2,
    stability: StabilityAccumulator | None = None,
    stab_index=None,
    store: DetStore | None = None,
    resume: bool = True,
    areas: tuple[str, ...] = AREA_KEYS,
    workers: int | None = None,
) -> list[dict]:
    """GPU forward passes for delta_{k+1} overlap the CPU evaluation of delta_k."""
    grid = shift_grid(max_shift, step)
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = _rows(out_jsonl) if resume else {}
    npig = {a: npig_per_cat(gt, a) for a in areas}

    LOGGER.info(
        f"sweep: {len(grid)} shifts (max={max_shift}, step={step}) x {gt.num_images} images "
        f"= {len(grid) * gt.num_images:,} forwards; {len(done)} already done"
    )
    if done and stability is not None and store is None:
        LOGGER.warning(
            f"{len(done)} shifts have cached AP rows but the flicker/jitter accumulators are "
            "online and cannot be resumed -- inference will re-run for every shift. Pass a "
            "DetStore (--persist-dets) if you expect to resume a sweep with stability enabled."
        )

    todo = [s for s in grid if s not in done or stability is not None]
    idx = stability.idx if stability is not None else None
    t_start = time.perf_counter()

    def finish(p: PendingShift, n: int) -> None:
        if p.shift not in done:
            row = {"dy": p.shift[0], "dx": p.shift[1], "n_dets": p.n_dets,
                   "t_infer": round(p.t_infer, 2)}
            for f in p.area_futures:
                a, r = f.result()
                if a == "all":
                    row["AP"], row["AP50"] = r["AP"], r["AP50"]
                else:
                    row[f"AP_{a}"] = r["AP"]
            row["t_total"] = round(time.perf_counter() - t_start, 2)
            with open(out_jsonl, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            done[p.shift] = row
            LOGGER.info(
                f"  [{n}/{len(todo)}] delta={p.shift} AP={row['AP']:.4f} "
                f"AP50={row['AP50']:.4f} infer={p.t_infer:.1f}s"
            )
        elif p.area_futures:
            for f in p.area_futures:
                f.result()
        if p.stab_future is not None:
            part = p.stab_future.result()
            stability.merge(part)
            if stab_index is not None:
                stab_index.merge(part)

    with SweepPool(gt, idx, class_map_inv, npig,
                   tau=stability.tau if stability else 0.25,
                   iou_thr=stability.iou_thr if stability else 0.5,
                   workers=workers) as pool:
        pending: PendingShift | None = None
        for n, (dy, dx) in enumerate(todo, 1):
            t0 = time.perf_counter()
            if store is not None and store.has(dy, dx):
                dets = store.load(dy, dx)
            else:
                dets = runner.run_shift(dy, dx)
                if store is not None:
                    store.save(dy, dx, dets.counts, dets.cls, dets.score, dets.xyxy)
            t_inf = time.perf_counter() - t0

            need_ap = (dy, dx) not in done
            nxt = pool.submit(dets, areas if need_ap else (), stability is not None, t_inf)
            if pending is not None:
                finish(pending, n - 1)
            pending = nxt
        if pending is not None:
            finish(pending, len(todo))

    return [done[k] for k in sorted(done)]


def curve_stats(rows: list[dict], key: str = "AP") -> dict:
    """Summarise AP(delta): range, std, and the stride-periodicity check.

    `period_32_gap` compares delta=(0,0) with the corners of the sweep. If the network's
    shift response were purely a stride artefact, AP at a full-stride offset would return
    to the delta=0 value. It is worth reporting how far it does not.
    """
    v = np.array([r[key] for r in rows], dtype=np.float64)
    by = {(r["dy"], r["dx"]): r[key] for r in rows}
    out = {
        f"{key}_at_0": by.get((0, 0), float("nan")),
        f"{key}_min": float(v.min()),
        f"{key}_max": float(v.max()),
        f"{key}_range": float(v.max() - v.min()),
        f"{key}_mean": float(v.mean()),
        f"{key}_std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "n_shifts": len(v),
    }
    mx = max(max(k) for k in by)
    for corner in [(0, mx), (mx, 0), (mx, mx)]:
        if corner in by:
            out[f"{key}_at_{corner[0]}_{corner[1]}"] = by[corner]
    argmin = min(by, key=by.get)
    argmax = max(by, key=by.get)
    out[f"{key}_argmin"] = list(argmin)
    out[f"{key}_argmax"] = list(argmax)
    return out


def subsample_fidelity(rows_full: list[dict], step: int, key: str = "AP") -> dict:
    """How much the step-k grid distorts the curve, measured against a step-1 run.

    The proposal promises exactly this check. We take the dense rows, subsample them on
    the coarse lattice, and report the error in the statistics we intend to publish --
    not the error in the curve pointwise, which nobody quotes.
    """
    dense = {(r["dy"], r["dx"]): r[key] for r in rows_full}
    coarse = {k: v for k, v in dense.items() if k[0] % step == 0 and k[1] % step == 0}
    if not coarse:
        return {}
    dv = np.array(list(dense.values()))
    cv = np.array(list(coarse.values()))
    return {
        "step": step,
        "n_dense": len(dv),
        "n_coarse": len(cv),
        f"{key}_range_dense": float(dv.max() - dv.min()),
        f"{key}_range_coarse": float(cv.max() - cv.min()),
        f"{key}_range_abs_err": float(abs((dv.max() - dv.min()) - (cv.max() - cv.min()))),
        f"{key}_min_abs_err": float(abs(dv.min() - cv.min())),
        f"{key}_max_abs_err": float(abs(dv.max() - cv.max())),
        f"{key}_std_abs_err": float(abs(dv.std(ddof=1) - cv.std(ddof=1))),
    }


def load_rows(path: Path) -> list[dict]:
    d = _rows(Path(path))
    return [d[k] for k in sorted(d)]


def to_grid(rows: list[dict], key: str = "AP") -> tuple[np.ndarray, list[int], list[int]]:
    """Rows -> a 2-D array for heatmap plotting."""
    dys = sorted({r["dy"] for r in rows})
    dxs = sorted({r["dx"] for r in rows})
    g = np.full((len(dys), len(dxs)), np.nan)
    yi = {v: i for i, v in enumerate(dys)}
    xi = {v: i for i, v in enumerate(dxs)}
    for r in rows:
        g[yi[r["dy"]], xi[r["dx"]]] = r[key]
    return g, dys, dxs
