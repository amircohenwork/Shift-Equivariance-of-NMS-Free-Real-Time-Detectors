"""Batched inference over the shift grid.

One `run_shift` call = one full pass of the image set at one delta, returning detections
already mapped back to original-image coordinates. Nothing here decides what to do with
them; `sweep.py` streams them into metrics, `scripts/03_build_cache.py` persists them for
the oracle.

Post-processing is delegated to `ultralytics.utils.nms.non_max_suppression` with exactly
the arguments `DetectionValidator.postprocess` uses (conf=0.001, iou=0.7, nc=0,
multi_label=True, max_det=300). Re-implementing NMS here would silently break the
Phase-0 gate, and "our NMS differs slightly from theirs" is precisely the kind of
confound that makes a head-vs-head comparison meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import PreprocCache
from .detstore import ShiftDets
from .modelzoo import ModelSpec, head_is, resolve_device
from .shift import ShiftGeometry, deshift_boxes, embed
from .utils import LOGGER


@dataclass
class PostprocCfg:
    conf: float = 0.001
    iou: float = 0.7
    max_det: int = 300
    multi_label: bool = True
    agnostic: bool = False


class ShiftRunner:
    def __init__(
        self,
        net: torch.nn.Module,
        spec: ModelSpec,
        cache: PreprocCache,
        geom: ShiftGeometry,
        pp: PostprocCfg,
        device: str | torch.device = "cuda",
        batch: int = 32,
        pad_value: float = 114.0,
        amp: bool = False,
        pad_mode: str = "constant",
    ):
        self.pad_mode = pad_mode
        self.net = net
        self.spec = spec
        self.cache = cache
        self.geom = geom
        self.pp = pp
        self.device = resolve_device(device)
        self.batch = batch
        self.pad_value = float(pad_value)
        self.amp = amp
        self._buf: torch.Tensor | None = None
        self._pinned: torch.Tensor | None = None
        self.is_rtdetr = spec.preproc == "stretch"
        # Cheap sanity trap: a mismatch here means the canvas is not stride-aligned and
        # the anchor grid would silently change shape between shifts.
        assert self.geom.canvas % spec.stride == 0, (self.geom.canvas, spec.stride)

        # `ultralytics.utils.nms` picks its NMS kernel with
        #     use_torchvision = ... and "torchvision" in sys.modules
        # and otherwise falls back to a pure-PyTorch implementation. Ultralytics does the
        # import inside AutoBackend.warmup, which we bypass -- so without this line the
        # one-to-many path silently runs a *different* NMS from the validator (and 22x
        # slower: 350 s vs 16 s per shift over COCO val on an A10). Both halves of that
        # matter; the fidelity half matters more.
        if not self.is_rtdetr:
            import torchvision  # noqa: F401  (presence in sys.modules is the switch)

    # ------------------------------------------------------------------ #
    def _batches(self):
        n = len(self.cache.image_ids)
        for b0 in range(0, n, self.batch):
            b1 = min(b0 + self.batch, n)
            # np.array(..., copy=True): the cache is a read-only memmap and torch refuses
            # to wrap non-writable buffers without a warning.
            chunk = np.array(self.cache.array[b0:b1], copy=True)
            t = torch.from_numpy(chunk)
            if self.device.type == "cuda":
                if self._pinned is None or self._pinned.shape[0] < t.shape[0]:
                    self._pinned = torch.empty_like(t, device="cpu").pin_memory()
                p = self._pinned[: t.shape[0]]
                p.copy_(t)
                t = p.to(self.device, non_blocking=True)
            else:
                t = t.to(self.device)
            yield b0, b1, t

    @torch.inference_mode()
    def run_shift(self, dy: int, dx: int) -> ShiftDets:
        self.geom.check(dy, dx)
        head = head_is(self.net)
        counts = np.zeros(len(self.cache.image_ids), dtype=np.int32)
        all_cls, all_score, all_box = [], [], []

        for b0, b1, imgs in self._batches():
            self._buf = embed(imgs, dy, dx, self.geom, self.pad_value, out=self._buf,
                              pad_mode=self.pad_mode)
            with torch.autocast(self.device.type, enabled=self.amp):
                raw = self.net(self._buf)
            outs = self._postprocess(raw, head)

            for k, det in enumerate(outs):
                i = b0 + k
                if det.numel() == 0:
                    continue
                d = det.float().cpu().numpy()
                box = deshift_boxes(
                    d[:, :4], dy, dx, self.cache.meta[i], self.geom.imgsz,
                    preproc=self.spec.preproc, clip=not self.is_rtdetr,
                )
                counts[i] = len(box)
                all_box.append(box)
                all_score.append(d[:, 4].astype(np.float32))
                all_cls.append(d[:, 5].astype(np.int16))

        cat = lambda xs, dt, cols=None: (  # noqa: E731
            np.concatenate(xs) if xs else np.zeros((0, cols) if cols else (0,), dtype=dt)
        )
        return ShiftDets(
            counts=counts,
            cls=cat(all_cls, np.int16),
            score=cat(all_score, np.float32),
            xyxy=cat(all_box, np.float32, 4),
            shift=(dy, dx),
        )

    # ------------------------------------------------------------------ #
    def _postprocess(self, raw, head: str) -> list[torch.Tensor]:
        if self.is_rtdetr:
            return self._postprocess_rtdetr(raw)
        from ultralytics.utils import nms

        return nms.non_max_suppression(
            raw,
            self.pp.conf,
            self.pp.iou,
            nc=0,
            multi_label=self.pp.multi_label,
            agnostic=self.pp.agnostic,
            max_det=self.pp.max_det,
            end2end=(head == "e2e"),
            rotated=False,
        )

    def _postprocess_rtdetr(self, raw) -> list[torch.Tensor]:
        """RT-DETR: top-k is inside the decoder; output is normalised cxcywh + score + cls.

        The decoder normalises against the *input* tensor, so the de-normalisation factor
        is `canvas`, not `imgsz`. Using imgsz here would shrink every box by
        imgsz/canvas = 640/672 and quietly wreck the whole RT-DETR arm.
        """
        from ultralytics.utils import ops

        preds = raw[0] if isinstance(raw, (list, tuple)) else raw
        bboxes, scores, labels = preds.split((4, 1, 1), dim=-1)
        bboxes = ops.xywh2xyxy(bboxes) * self.geom.canvas
        scores, labels = scores.squeeze(-1), labels.squeeze(-1)
        out = []
        for bb, sc, lb in zip(bboxes, scores, labels):
            m = (sc > self.pp.conf).nonzero().squeeze(1)
            order = sc[m].argsort(descending=True)[: self.pp.max_det]
            m = m[order]
            out.append(torch.cat([bb[m], sc[m, None], lb[m, None]], dim=-1))
        return out


def infer_geometry(spec: ModelSpec, max_shift: int) -> ShiftGeometry:
    return ShiftGeometry(imgsz=spec.imgsz, max_shift=max_shift, stride=spec.stride)


def log_plan(spec: ModelSpec, geom: ShiftGeometry, n_images: int, n_shifts: int, batch: int):
    LOGGER.info(
        f"plan: {spec.name} imgsz={geom.imgsz} canvas={geom.canvas} "
        f"images={n_images} shifts={n_shifts} forwards={n_images * n_shifts:,} batch={batch}"
    )
