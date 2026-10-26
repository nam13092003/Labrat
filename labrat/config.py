"""Nạp cấu hình: config.yaml (chung) + competitions/<slug>/project.yaml (riêng) + .env."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("LABRAT_ROOT", Path(__file__).resolve().parent.parent))
TEMPLATES = ROOT / "templates"


def load_env(path: Path | None = None) -> None:
    """Đọc .env đơn giản (KEY=VALUE). Biến môi trường đã có thì giữ nguyên."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.strip().strip('"').strip("'")
        if val:
            os.environ.setdefault(key.strip(), val)


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def competitions_root(cfg: dict) -> Path:
    return ROOT / cfg.get("paths", {}).get("competitions_dir", "competitions")


def comp_dir(cfg: dict, slug: str) -> Path:
    return competitions_root(cfg) / slug


def load_config(slug: str | None = None) -> dict:
    load_env()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
    if slug:
        p = comp_dir(cfg, slug) / "project.yaml"
        if p.exists():
            cfg = deep_merge(cfg, yaml.safe_load(p.read_text(encoding="utf-8")) or {})
        cfg["slug"] = slug
    user = os.environ.get("KAGGLE_USERNAME")
    if user and cfg.get("kaggle", {}).get("username") in (None, "", "your-kaggle-username"):
        cfg.setdefault("kaggle", {})["username"] = user
    return cfg


