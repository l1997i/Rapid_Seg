"""Safe loader for the preprocessed SemanticKITTI `.pth` frames.

The frames from the `larryshaw0079/SemanticKITTI-SP` HuggingFace dataset store a
dict of *inline numpy arrays* (keys: coord, remission, semantic_label, ...).
`torch.load(weights_only=False)` would unpickle arbitrary objects (code-exec
risk for third-party files). Instead we use a `RestrictedUnpickler` that
whitelists ONLY the globals required to rebuild numpy arrays, and refuses any
`persistent_id`. Any other global raises -- so no attacker-controlled code path
can execute during load.
"""
from __future__ import annotations

import io
import pickle
import zipfile
from typing import Any, Dict

import numpy as np

# Only these globals are needed to reconstruct plain numpy arrays / scalars.
_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("_codecs", "encode"),
    # newer numpy (>=1.25) namespace variants
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) in _ALLOWED:
            import importlib
            return getattr(importlib.import_module(module), name)
        raise pickle.UnpicklingError(f"blocked global during safe load: {module}.{name}")

    def persistent_load(self, pid):  # torch storages would use this; reject.
        raise pickle.UnpicklingError("unexpected persistent_id in numpy-only .pth")


def load_kitti_pth(path: str) -> Dict[str, Any]:
    """Load a `.pth` frame's numpy dict safely (no arbitrary code execution)."""
    with zipfile.ZipFile(path) as z:
        pkl_name = next(n for n in z.namelist() if n.endswith("data.pkl"))
        raw = z.read(pkl_name)
    return RestrictedUnpickler(io.BytesIO(raw)).load()


def is_valid_frame(path: str) -> bool:
    """True if the frame fully loads and has the fields RAPiD-Seg needs."""
    try:
        d = load_kitti_pth(path)
        return all(k in d for k in ("coord", "remission", "semantic_label")) and \
            isinstance(d["coord"], np.ndarray) and d["coord"].shape[0] > 0
    except Exception:
        return False
