"""Restore NumPy<2 aliases that the pinned wandb release still imports.

This env ships numpy 2.x while wandb 0.13 references np.float_/np.int_/... at
import time.  `site` imports sitecustomize during interpreter startup, before
any user module, so defining the aliases here fixes `import wandb` everywhere
without touching installed packages.  Activated by putting this directory on
PYTHONPATH.
"""
import numpy as _np

_ALIASES = {
    "float_": _np.float64,
    "int_": _np.int64,
    "complex_": _np.complex128,
    "unicode_": _np.str_,
    "str_": _np.str_,
    "bool8": _np.bool_,
    "object_": object,
    "long": _np.int64,
}

for _name, _target in _ALIASES.items():
    if not hasattr(_np, _name):
        setattr(_np, _name, _target)
