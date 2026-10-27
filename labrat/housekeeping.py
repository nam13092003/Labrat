"""Giữ ổ đĩa VPS không bị đầy.

- prune_large(): xoá file quá lớn trong output kernel đã tải về (checkpoint, cache đặc trưng...).
- clean(): dọn những thứ Labrat tự tạo và an toàn để xoá:
    * file > download_max_file_mb trong competitions/*/experiments/*/out/   (kết quả đã nằm trong registry.json)
    * thư mục bundle competitions/*/experiments/*/_bundle_*                 (tạo lại được mỗi lần push)
    * thư mục tạm .labrat_tmp/* không còn hoạt động                         (file nén dataset bị bỏ lại)
    * file tải dở *.part đã cũ trong dropbox/ và competitions/
  KHÔNG đụng: dữ liệu (data/, dropbox/), registry/NOTES/báo cáo, file ngoài thư mục Labrat.
- disk(): dung lượng còn trống của ổ chứa Labrat.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from .config import ROOT, competitions_root, load_config

STALE_TMP_SECONDS = 2 * 3600     # thư mục tạm không có file nào thay đổi trong 2 giờ → bỏ lại
STALE_PART_SECONDS = 3 * 3600


def download_max_mb(cfg: dict) -> float:
    return float(cfg.get("outputs", {}).get("download_max_file_mb", 50))


def disk(path: Path | None = None) -> dict:
    u = shutil.disk_usage(path or ROOT)
    return {"free_gb": round(u.free / 1e9, 1), "total_gb": round(u.total / 1e9, 1),
            "used_pct": round(100 * u.used / u.total, 1)}


def _size(p: Path) -> int:
    if p.is_file() or p.is_symlink():
        return p.lstat().st_size
    return sum(f.lstat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())


def prune_large(folder: Path, max_mb: float, dry_run: bool = False) -> list[dict]:
    """Xoá file thường > max_mb trong folder (không theo symlink). Trả về danh sách đã xoá."""
    out = []
    if max_mb <= 0 or not Path(folder).exists():
        return out
    limit = max_mb * 1024 * 1024
    for f in sorted(Path(folder).rglob("*")):
        if f.is_file() and not f.is_symlink() and f.stat().st_size > limit:
            out.append({"path": str(f.relative_to(folder)), "mb": round(f.stat().st_size / 1048576, 1)})
            if not dry_run:
                f.unlink()
    return out


def _newest_mtime(p: Path) -> float:
    times = [p.stat().st_mtime] + [f.stat().st_mtime for f in p.rglob("*") if f.exists()]
    return max(times)


def clean(cfg: dict | None = None, dry_run: bool = False) -> dict:
    cfg = cfg or load_config()
    from .kaggle_ops import TMP_ROOT
    now = time.time()
    freed, items = 0, []

    def drop(p: Path, why: str):
        nonlocal freed
        size = _size(p)
        freed += size
        items.append({"path": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
                      "mb": round(size / 1048576, 1), "why": why})
        if not dry_run:
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() and not p.is_symlink() else p.unlink(missing_ok=True)

    croot = competitions_root(cfg)
    limit = download_max_mb(cfg) * 1024 * 1024
    for out_dir in croot.glob("*/experiments/*/out"):
        for f in out_dir.rglob("*"):
            if f.is_file() and not f.is_symlink() and f.stat().st_size > limit:
                drop(f, "output kernel lớn (kết quả đã có trong registry)")
    for b in croot.glob("*/experiments/*/_bundle_*"):
        drop(b, "bundle cũ (tạo lại khi push)")
    if TMP_ROOT.exists():
        for t in TMP_ROOT.iterdir():
            if now - _newest_mtime(t) > STALE_TMP_SECONDS:
                drop(t, "file tạm kaggle CLI bị bỏ lại")
    for base in (croot, ROOT / "dropbox"):
        if base.exists():
            for f in base.rglob("*.part"):
                if f.is_file() and now - f.stat().st_mtime > STALE_PART_SECONDS:
                    drop(f, "file tải dở")
    items.sort(key=lambda x: -x["mb"])
    return {"dry_run": dry_run, "freed_gb": round(freed / 1e9, 2), "n_items": len(items),
            "items": items[:30], "disk": disk()}


def big_dirs(n: int = 8) -> list[dict]:
    """Thư mục con lớn nhất trong Labrat và /tmp (chỉ để báo cáo, không xoá)."""
    rows = []
    for base in (ROOT, Path("/tmp")):
        if not base.exists():
            continue
        for p in base.iterdir():
            try:
                if p.is_dir() and not p.is_symlink() and p.name not in (".venv", ".git"):
                    rows.append({"path": str(p), "gb": round(_size(p) / 1e9, 2)})
            except (PermissionError, OSError):
                pass
    for p in competitions_root(load_config()).glob("*/*"):
        if p.is_dir():
            rows.append({"path": str(p), "gb": round(_size(p) / 1e9, 2)})
    return sorted(rows, key=lambda r: -r["gb"])[:n]
