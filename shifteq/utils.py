"""Small shared helpers: determinism, logging, hashing, JSON IO."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("shifteq")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def set_deterministic(seed: int = 0, warn_only: bool = True) -> None:
    """Pin every source of run-to-run variation we can reach.

    cudnn.benchmark is the important one: autotuning picks a different conv algorithm
    depending on machine state, which changes low-order bits of the logits. In a study
    whose entire subject is sub-pixel-scale output instability, that noise floor would
    be indistinguishable from the signal.
    """
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # TF32 silently drops mantissa bits on Ampere+; off by default for reproducibility.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except Exception as e:  # older torch / unsupported op
        LOGGER.warning(f"use_deterministic_algorithms unavailable: {e}")


def sha256_of_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_of_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def state_dict_fingerprint(module) -> str:
    """Order-independent-of-nothing fingerprint of a module's parameters.

    Used to prove that the e2e and NMS runs really did share backbone/neck weights, and
    that nothing (fusing, half(), inplace surgery) mutated them between runs.
    """
    import torch

    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        t = v.detach().to(torch.float64).cpu().contiguous()
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


@contextmanager
def timer(name: str):
    t0 = time.perf_counter()
    yield
    LOGGER.info(f"[timer] {name}: {time.perf_counter() - t0:.2f}s")


def shift_grid(max_shift: int, step: int = 1) -> list[tuple[int, int]]:
    """Row-major (dy, dx) grid over [0, max_shift]^2 with the given step.

    Convention throughout this repo: a shift is stored and passed as (dy, dx),
    matching numpy/torch spatial axis order. Only the CLI parses "x,y" strings.
    """
    vals = list(range(0, max_shift + 1, step))
    return [(dy, dx) for dy in vals for dx in vals]


def parse_shift(s: str) -> tuple[int, int]:
    dy, dx = (int(v) for v in s.replace("(", "").replace(")", "").split(","))
    return dy, dx
