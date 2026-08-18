#!/usr/bin/env python3
"""Reconnaissance before committing GPU hours to the cross-checks.

For each candidate model: can we fetch it, does it load unfused, which aggregation
heads does the checkpoint actually serve, and how long does one forward pass over
the canvas take? Everything downstream is budgeted from this.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

import torch

from shifteq.modelzoo import (MODELS, available_heads, load_detector, resolve_weights,
                              set_head)

CANDIDATES = ["yolov8n", "yolov10n", "yolo11n", "rtdetr-l"]

for name in CANDIDATES:
    spec = MODELS[name]
    print(f"\n=== {name} ===")
    try:
        w = resolve_weights(spec)
        print(f"  weights: {w} ({w.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        print(f"  FETCH FAILED: {type(e).__name__}: {e}")
        continue
    try:
        net, spec = load_detector(spec, device="cuda")
    except Exception as e:
        print(f"  LOAD FAILED: {type(e).__name__}: {e}")
        continue
    have = available_heads(net)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"  declared heads={spec.heads} available={have} params={n_par / 1e6:.2f}M "
          f"preproc={spec.preproc}")

    for head in have:
        try:
            set_head(net, head)
        except Exception as e:
            print(f"  head={head}: set_head FAILED {type(e).__name__}: {e}")
            continue
        canvas = 704 if spec.preproc == "letterbox" else 704
        x = torch.zeros(32, 3, canvas, canvas, device="cuda")
        with torch.inference_mode():
            for _ in range(2):
                net(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                out = net(x)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 3
        shape = out[0].shape if isinstance(out, (list, tuple)) else out.shape
        per_img = dt / 32 * 1000
        print(f"  head={head}: {dt * 1000:.0f} ms/batch32 = {per_img:.2f} ms/img "
              f"-> {per_img * 5000 / 1000:.1f} s/shift over 5000 imgs | out {tuple(shape)}")
        del x
        torch.cuda.empty_cache()
    del net
    torch.cuda.empty_cache()
