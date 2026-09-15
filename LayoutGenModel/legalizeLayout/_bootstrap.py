"""把 LayoutGenModel/diffusion 挂上 sys.path, 并修掉 numpy 2.0 的别名问题。

为什么需要这一层
----------------
1. ``diffusion/`` 里全是扁平 import (``import guidance`` / ``import utils`` /
   ``import common``), 所以必须把 ``LayoutGenModel``、``LayoutGenModel/diffusion``
   和仓库根同时挂上, 顺序与 run_*.sh 里的 ``PYTHONPATH=".:diffusion:.."`` 一致。

2. 本环境的 numpy 是 2.x, 而 ``utils.py`` 顶层 ``import wandb``, 该 wandb 版本在
   导入期访问 ``np.float_`` / ``np.complex_`` 等 numpy 2.0 已删除的别名, 直接
   ImportError。这里补回别名, 是让既有代码原样跑起来的最小改动 —— 不去动
   ``utils.py``, 也不去改环境。别名只在 numpy 缺失时才补, 不覆盖任何现存属性。
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent                   # legalizeLayout/


def _find_layoutgen_dir(start: Path) -> Path:
    """向上找 ``LayoutGenModel/``, 按**内容**认, 不按相对层数。

    为什么不能写死 ``_HERE.parent``: 本工具被移动过一次
    (``flow_tap/legalizeLayout`` -> ``LayoutGenModel/legalizeLayout``)。移动后
    ``_HERE.parent`` 变成 LayoutGenModel 自身, 于是 ``_LAYOUTGEN_DIR`` 被拼成
    ``LayoutGenModel/LayoutGenModel`` —— ``sys.path`` 挂上一串不存在的路径 (静默,
    不报错), 模型 checkpoint 的相对路径也全部解析失败, 报出来是"文件找不到"而
    看不出根因是这个。按内容认目录后, 放在 LayoutGenModel 内部或与它同级都能工作。
    """
    for base in (start, *start.parents):
        if (base / "diffusion" / "utils.py").is_file():      # base 就是 LayoutGenModel
            return base
        nested = base / "LayoutGenModel"
        if (nested / "diffusion" / "utils.py").is_file():    # base 是仓库根
            return nested
    raise RuntimeError(
        f"从 {start} 逐级向上都找不到含 diffusion/utils.py 的 LayoutGenModel/, "
        "本工具必须放在 LayoutGenModel/ 内部或与它同级。"
    )


_LAYOUTGEN_DIR = _find_layoutgen_dir(_HERE)               # LayoutGenModel/
_REPO_ROOT = _LAYOUTGEN_DIR.parent                        # flow_tap/
_DIFFUSION_DIR = _LAYOUTGEN_DIR / "diffusion"             # LayoutGenModel/diffusion/ —— 仓库模块都在这


def _install_numpy_aliases() -> None:
    import numpy as np

    aliases = {
        "float_": np.float64,
        "complex_": np.complex128,
        "unicode_": np.str_,
        "bool8": np.bool_,
        "object0": np.object_,
        "int0": np.intp,
        "uint0": np.uintp,
        "str0": np.str_,
        "bytes0": np.bytes_,
        "void0": np.void,
        "NaN": np.nan,
        "Inf": np.inf,
        "infty": np.inf,
        "NINF": -np.inf,
        "PINF": np.inf,
        "NZERO": -0.0,
        "PZERO": 0.0,
        "round_": np.round,
        "product": np.prod,
        "cumproduct": np.cumprod,
        "alltrue": np.all,
        "sometrue": np.any,
        "msort": np.sort,
        "row_stack": np.vstack,
    }
    if hasattr(np, "trapezoid"):
        aliases["trapz"] = np.trapezoid
    if hasattr(np, "isin"):
        aliases["in1d"] = np.isin
    for name, value in aliases.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def setup() -> None:
    """幂等: 重复调用无副作用。"""
    if getattr(setup, "_done", False):
        return
    _install_numpy_aliases()
    # 顺序与 run_*.sh 的 PYTHONPATH=".:diffusion:.." 一致: legalizeLayout 在最前
    # (它自己的模块名优先), 然后是 diffusion (仓库的扁平 import), 再是上层。
    for path in (_HERE, _DIFFUSION_DIR, _LAYOUTGEN_DIR, _REPO_ROOT, _REPO_ROOT / "thermalmodel"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    setup._done = True


def repo_root() -> Path:
    return _REPO_ROOT


def resolve_path(value, default=None) -> Path:
    """把配置里的相对路径解析到仓库根下。"""
    if value in (None, "", "none", "None"):
        if default is None:
            return None
        return (repo_root() / str(default)).resolve()
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    candidate = (repo_root() / path).resolve()
    if candidate.exists():
        return candidate
    return (Path.cwd() / path).resolve()
