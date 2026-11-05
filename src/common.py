"""各模块共用的小工具：读配置、固定随机种子、选设备、创建实验输出目录。"""

from __future__ import annotations

import json
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """读取 yaml 配置。支持 `_base_: base.yaml` 继承，以及 "a.b=1" 形式的覆盖。"""
    import yaml

    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("_base_", None)
    if base:
        cfg = _deep_merge(load_config(path.parent / base), cfg)
    for key, value in (overrides or {}).items():
        node = cfg
        *parents, last = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = value
    return cfg


def _deep_merge(base: dict, new: dict) -> dict:
    out = dict(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(prefer: str = "auto") -> str:
    """auto：有 CUDA 用 CUDA，Mac 上用 MPS，否则 CPU。"""
    if prefer != "auto":
        return prefer
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def make_run_dir(cfg: Dict[str, Any], name: str, root: str | Path = "outputs") -> Path:
    """创建 outputs/<时间>_<名称>/，并保存本次配置和代码版本，保证实验可复现。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / root / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    meta = {"config": cfg, "git_commit": git_commit(), "created": stamp}
    (run_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir
