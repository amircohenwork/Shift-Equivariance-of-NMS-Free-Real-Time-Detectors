"""The shift harness: canvas embedding, and the exact inverse map for predictions.

Protocol
--------
Manfredi & Wang (ECCV'20 W, arXiv:2008.05787, Sec. 4 / Fig. 2) do not crop to shift.
They embed each image into a *larger* black canvas at offset delta, following Azulay &
Weiss. That matters for detection specifically: cropping even a few pixels can remove
part of an object and change the ground truth, which would confound "the detector moved
its answer" with "the input genuinely lost information".

We keep that property and adapt it to a YOLO pipeline:

    original image --[Ultralytics val letterbox]--> imgsz x imgsz
                   --[paste at (dy, dx)]---------> canvas x canvas       (canvas >= imgsz + max_shift)

`canvas` is rounded up to a multiple of the network stride so every shift sees an input
of identical size and an identical anchor grid; only the *content* moves. If the canvas
changed size with delta, the number of anchor cells would change too and we would be
measuring two things at once.

Setting `max_shift = 0` collapses canvas to imgsz and the harness becomes the plain
Ultralytics val path. That degenerate case is the Phase-0 identity gate: it must
reproduce the released numbers exactly, because it *is* the released pipeline.

Sign convention
---------------
A shift of (dy, dx) moves image content DOWN by dy and RIGHT by dx. An ideally
equivariant detector would therefore emit boxes translated by (+dx, +dy), and
de-shifting subtracts them back. Shifts are non-negative by construction: the canvas
grows to the bottom-right, so delta=(0,0) puts the image at the top-left corner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .data import ImageMeta


def canvas_size(imgsz: int, max_shift: int, stride: int = 32) -> int:
    """Smallest stride-aligned canvas that fits the image at every shift in the grid."""
    return int(math.ceil((imgsz + max_shift) / stride) * stride)


@dataclass
class ShiftGeometry:
    """Two ways to buy a shift budget without ever cropping.

    `grow`  letterbox to `imgsz`, embed in a canvas of align(imgsz + max_shift). Keeps the
            full letterbox resolution and stays byte-identical to the Ultralytics val
            dataloader, but shows the network a size it was not validated at (measured:
            +0.029 AP e2e / +0.068 nms at delta=0 on the full val set; gate B6's
            500-image probe said +0.225 for e2e, which was mostly sampling noise).

    `fit`   letterbox to `imgsz - max_shift`, embed in a canvas of exactly `imgsz`. The
            network input is the validated size; the shift budget comes out of the image
            scale instead. Closer to Manfredi & Wang in spirit, and the variant to use if a
            reviewer objects that `grow` changes the input distribution.

    Both are strictly no-crop: the whole letterboxed tile is present at every delta, so an
    object at the frame edge keeps its entire neighbourhood. Only the tile's absolute
    position in the canvas changes.
    """

    imgsz: int
    max_shift: int
    stride: int
    embed_mode: str = "grow"

    def __post_init__(self):
        if self.embed_mode not in ("grow", "fit"):
            raise ValueError(f"embed_mode must be 'grow' or 'fit', got {self.embed_mode!r}")
        if self.embed_mode == "fit" and self.max_shift >= self.imgsz:
            raise ValueError("max_shift must be smaller than imgsz in 'fit' mode")

    @property
    def canvas(self) -> int:
        if self.embed_mode == "fit":
            return self.imgsz
        return canvas_size(self.imgsz, self.max_shift, self.stride)

    @property
    def tile(self) -> int:
        """Side of the letterboxed image that gets pasted into the canvas."""
        return self.imgsz - self.max_shift if self.embed_mode == "fit" else self.imgsz

    @property
    def is_identity(self) -> bool:
        return self.max_shift == 0 and self.canvas == self.imgsz == self.tile

    def check(self, dy: int, dx: int) -> None:
        if not (0 <= dy <= self.max_shift and 0 <= dx <= self.max_shift):
            raise ValueError(f"shift ({dy},{dx}) outside [0,{self.max_shift}]^2")
        if self.tile + max(dy, dx) > self.canvas:
            raise ValueError("canvas too small for requested shift")


def embed(
    imgs_u8: torch.Tensor,
    dy: int,
    dx: int,
    geom: ShiftGeometry,
    pad_value: float = 114.0,
    out: torch.Tensor | None = None,
    pad_mode: str = "constant",
) -> torch.Tensor:
    """uint8 [B,3,imgsz,imgsz] -> float32 [B,3,canvas,canvas] in [0,1], shifted by (dy,dx).

    `pad_mode` is the second half of the padding ablation: the letterbox border is set
    when the cache is built, the *canvas* border here. Constant padding gives the network
    a clean absolute-position cue (Kayhan & van Gemert); replicate/reflect remove that cue
    while keeping the geometry identical.

    `out` lets the caller hand in a persistent buffer; at 256 shifts x hundreds of
    batches, re-allocating the canvas every step is pure allocator churn.
    """
    b, c, h, w = imgs_u8.shape
    assert h == w == geom.tile, (h, w, geom.tile)
    geom.check(dy, dx)
    n = geom.canvas

    if pad_mode != "constant":
        # F.pad wants (left, right, top, bottom) and only handles float for these modes.
        import torch.nn.functional as F

        x = imgs_u8.to(torch.float32)
        y = F.pad(x, (dx, n - w - dx, dy, n - h - dy), mode=pad_mode)
        return y.div_(255.0)

    if out is None or out.shape != (b, c, n, n):
        out = torch.empty((b, c, n, n), dtype=torch.float32, device=imgs_u8.device)
    out.fill_(pad_value)
    out[:, :, dy : dy + h, dx : dx + w] = imgs_u8.to(torch.float32)
    return out.div_(255.0)


def embed_numpy(img_u8: np.ndarray, dy: int, dx: int, geom: ShiftGeometry, pad_value: int = 114) -> np.ndarray:
    """CPU twin of `embed`, for tests that must not depend on a GPU being present."""
    c, h, w = img_u8.shape
    n = geom.canvas
    out = np.full((c, n, n), pad_value, dtype=np.uint8)
    out[:, dy : dy + h, dx : dx + w] = img_u8
    return out


# --------------------------------------------------------------------------------------
# inverse map: canvas pixels -> original image pixels
# --------------------------------------------------------------------------------------


def deshift_boxes(
    boxes: np.ndarray,
    dy: int,
    dx: int,
    meta: ImageMeta,
    imgsz: int,
    preproc: str = "letterbox",
    clip: bool = True,
) -> np.ndarray:
    """[D,4] xyxy in canvas coords -> [D,4] xyxy in original-image coords.

    Step 1 (canvas -> letterbox) is the shift inverse and is exact in integer pixels.
    Step 2 (letterbox -> original) reproduces `ultralytics.utils.ops.scale_boxes` for
    the letterbox family, and `RTDETRValidator.pred_to_json`'s independent per-axis
    rescale for the stretch family. Reproducing them rather than reinventing them is
    what makes the max_shift=0 identity gate meaningful.
    """
    b = np.array(boxes, dtype=np.float64, copy=True).reshape(-1, 4)
    b[:, [0, 2]] -= dx
    b[:, [1, 3]] -= dy

    if preproc == "stretch":
        b[:, [0, 2]] *= meta.ori_w / imgsz
        b[:, [1, 3]] *= meta.ori_h / imgsz
        return b.astype(np.float32)

    b[:, [0, 2]] -= meta.pad_x
    b[:, [1, 3]] -= meta.pad_y
    # Ultralytics uses the single scalar ratio_pad[0][0] (the y gain) for both axes.
    b /= meta.gain_y
    if clip:
        np.clip(b[:, [0, 2]], 0, meta.ori_w, out=b[:, [0, 2]])
        np.clip(b[:, [1, 3]], 0, meta.ori_h, out=b[:, [1, 3]])
    return b.astype(np.float32)


def shift_boxes_forward(
    boxes: np.ndarray, dy: int, dx: int, meta: ImageMeta, imgsz: int, preproc: str = "letterbox"
) -> np.ndarray:
    """Original-image xyxy -> canvas xyxy. Exact inverse of `deshift_boxes(clip=False)`;
    exists so the round-trip can be asserted analytically instead of hoped for."""
    b = np.array(boxes, dtype=np.float64, copy=True).reshape(-1, 4)
    if preproc == "stretch":
        b[:, [0, 2]] *= imgsz / meta.ori_w
        b[:, [1, 3]] *= imgsz / meta.ori_h
    else:
        b *= meta.gain_y
        b[:, [0, 2]] += meta.pad_x
        b[:, [1, 3]] += meta.pad_y
    b[:, [0, 2]] += dx
    b[:, [1, 3]] += dy
    return b.astype(np.float32)


def deshift_boxes_torch(
    boxes: torch.Tensor,
    dy: int,
    dx: int,
    gain: torch.Tensor,
    pad: torch.Tensor,
    ori_hw: torch.Tensor,
    preproc: str = "letterbox",
    clip: bool = True,
) -> torch.Tensor:
    """Batched device-side de-shift. `gain` [B], `pad` [B,2] (x,y), `ori_hw` [B,2] (h,w).

    For the stretch family `gain` carries the per-axis factors instead; see `infer.py`.
    """
    b = boxes.clone().to(torch.float32)
    b[..., 0::2] -= dx
    b[..., 1::2] -= dy
    if preproc == "stretch":
        b[..., 0::2] *= (ori_hw[:, 1] / gain[:, 0]).view(-1, 1, 1)
        b[..., 1::2] *= (ori_hw[:, 0] / gain[:, 1]).view(-1, 1, 1)
        return b
    b[..., 0::2] -= pad[:, 0].view(-1, 1, 1)
    b[..., 1::2] -= pad[:, 1].view(-1, 1, 1)
    b /= gain[:, 0].view(-1, 1, 1)
    if clip:
        b[..., 0::2] = b[..., 0::2].clamp(min=0).minimum(ori_hw[:, 1].view(-1, 1, 1))
        b[..., 1::2] = b[..., 1::2].clamp(min=0).minimum(ori_hw[:, 0].view(-1, 1, 1))
    return b
