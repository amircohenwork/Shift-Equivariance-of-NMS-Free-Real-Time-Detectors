"""On-disk detection cache, keyed by (run, shift).

Inference is run **once** per (model, head, shift) and everything downstream -- the
oracle, the uniform sweep, flicker, jitter -- reads from here. That separation is what
makes the two metric families share GPU work: the oracle's max_shift=1 grid is a subset
of the 32-px sweep grid, so its four shifts are already paid for.

Layout:
    <root>/<run_id>/manifest.json
    <root>/<run_id>/dets_dy{dy:03d}_dx{dx:03d}.npz

Each npz holds detections for all images at one shift, concatenated in image order and
sorted by descending score within an image:
    counts int32 [N]      detections per image, aligned to manifest["image_ids"]
    cls    int16 [D]      contiguous class index (0..nc-1), NOT the COCO category id
    score  float32 [D]
    xyxy   float32 [D,4]  ORIGINAL-image coordinates, already de-shifted

Storing de-shifted boxes means the cache is directly comparable across shifts: any
difference in these numbers is detector shift-variance, with the geometry divided out.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .utils import LOGGER, read_json, write_json


@dataclass
class RunManifest:
    run_id: str
    model: str
    head: str                      # "e2e" | "nms"
    weights_sha256: str
    weight_fingerprint: str
    imgsz: int
    stride: int
    preproc: str
    max_shift: int
    step: int
    canvas: int
    pad_value: int
    conf: float
    iou: float
    max_det: int
    multi_label: bool
    device: str
    dtype: str
    fused: bool                    # must be False; recorded so a reviewer can check
    fusion_evidence: dict
    image_ids: list[int] = field(default_factory=list)
    shifts_done: list[list[int]] = field(default_factory=list)
    ultralytics_version: str = ""
    torch_version: str = ""
    notes: str = ""


class DetStore:
    def __init__(self, root: str | Path, run_id: str):
        self.dir = Path(root) / run_id
        self.run_id = run_id
        self._manifest: RunManifest | None = None

    # -- manifest ----------------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    def write_manifest(self, m: RunManifest) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._manifest = m
        write_json(self.manifest_path, asdict(m))

    @property
    def manifest(self) -> RunManifest:
        if self._manifest is None:
            self._manifest = RunManifest(**read_json(self.manifest_path))
        return self._manifest

    # -- detections --------------------------------------------------------------
    def path_for(self, dy: int, dx: int) -> Path:
        return self.dir / f"dets_dy{dy:03d}_dx{dx:03d}.npz"

    def has(self, dy: int, dx: int) -> bool:
        return self.path_for(dy, dx).exists()

    def save(self, dy: int, dx: int, counts: np.ndarray, cls: np.ndarray,
             score: np.ndarray, xyxy: np.ndarray) -> None:
        assert counts.sum() == len(cls) == len(score) == len(xyxy), "ragged detection arrays"
        self.dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.path_for(dy, dx),
            counts=counts.astype(np.int32),
            cls=cls.astype(np.int16),
            score=score.astype(np.float32),
            xyxy=xyxy.astype(np.float32),
        )

    def load(self, dy: int, dx: int) -> "ShiftDets":
        with np.load(self.path_for(dy, dx)) as z:
            return ShiftDets(
                counts=z["counts"], cls=z["cls"], score=z["score"], xyxy=z["xyxy"], shift=(dy, dx)
            )

    def available_shifts(self) -> list[tuple[int, int]]:
        out = []
        for p in sorted(self.dir.glob("dets_dy*_dx*.npz")):
            stem = p.stem
            dy = int(stem.split("_dy")[1].split("_")[0])
            dx = int(stem.split("_dx")[1])
            out.append((dy, dx))
        return out

    def size_on_disk(self) -> int:
        return sum(p.stat().st_size for p in self.dir.glob("*.npz"))

    def fingerprint(self) -> dict:
        """Content hash of the detections this store holds, for the consuming results file.

        A results file that records only a `run_id` cannot be traced back to the bytes it
        was computed from. This project has already lost time to that: a killed job once
        overlapped a new one and mixed two NMS kernels into one cache, and later, when two
        committed results disagreed on a quantity that should have been identical, nothing
        in either file could say whether the code or the input had differed. (It was the
        code -- but only a recomputation could establish that.)

        Hashes the shift arrays themselves rather than the file bytes, so it is stable
        against npz recompression and metadata.
        """
        h = hashlib.sha256()
        shifts = self.available_shifts()
        for dy, dx in shifts:
            d = self.load(dy, dx)
            h.update(f"{dy},{dx}".encode())
            for a in (d.counts, d.cls, d.score, d.xyxy):
                h.update(np.ascontiguousarray(a).tobytes())
        return {"run_id": self.run_id, "n_shifts": len(shifts),
                "shifts": [list(s) for s in shifts],
                "sha256": h.hexdigest()[:32],
                "bytes_on_disk": self.size_on_disk()}


@dataclass
class ShiftDets:
    counts: np.ndarray   # int32 [N]
    cls: np.ndarray      # int16 [D]
    score: np.ndarray    # float32 [D]
    xyxy: np.ndarray     # float32 [D,4]
    shift: tuple[int, int]

    def __post_init__(self):
        self.offsets = np.zeros(len(self.counts) + 1, dtype=np.int64)
        np.cumsum(self.counts, out=self.offsets[1:])

    def image_slice(self, i: int) -> slice:
        return slice(int(self.offsets[i]), int(self.offsets[i + 1]))

    def per_image(self, i: int):
        s = self.image_slice(i)
        return self.cls[s], self.score[s], self.xyxy[s]

    @property
    def n_images(self) -> int:
        return len(self.counts)

    @property
    def n_dets(self) -> int:
        return int(self.counts.sum())


def to_coco_json(
    dets: ShiftDets,
    image_ids: list[int],
    class_map: np.ndarray,
    round_coords: bool = True,
) -> list[dict]:
    """COCO-format prediction list, byte-compatible with Ultralytics' predictions.json.

    `round_coords` reproduces Ultralytics' `round(x, 3)` / `round(score, 5)`; keep it on
    when the goal is to reproduce a published number, off when comparing two of our own
    runs (the rounding quantises exactly the sub-pixel differences we are studying).
    """
    out = []
    for i, iid in enumerate(image_ids):
        s = dets.image_slice(i)
        b = dets.xyxy[s].astype(np.float64).copy()
        b[:, 2:] -= b[:, :2]                     # xyxy -> xywh (top-left + size)
        for (x, y, w, h), sc, c in zip(b, dets.score[s], dets.cls[s]):
            box = [round(float(v), 3) for v in (x, y, w, h)] if round_coords else [float(v) for v in (x, y, w, h)]
            out.append(
                {
                    "image_id": int(iid),
                    "category_id": int(class_map[int(c)]),
                    "bbox": box,
                    "score": round(float(sc), 5) if round_coords else float(sc),
                }
            )
    return out


def coco80_to_coco91() -> np.ndarray:
    """Contiguous 0..79 model class index -> COCO category id, as Ultralytics maps it."""
    from ultralytics.data import converter

    return np.asarray(converter.coco80_to_coco91_class(), dtype=np.int64)


def summarise(store: DetStore) -> None:
    m = store.manifest
    shifts = store.available_shifts()
    LOGGER.info(
        f"{store.run_id}: {m.model}/{m.head}  {len(shifts)} shifts  "
        f"{store.size_on_disk() / 1e6:.1f} MB  fused={m.fused}"
    )
