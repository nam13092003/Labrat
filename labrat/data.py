"""Nhận dữ liệu từ người dùng và đưa lên Kaggle cho MỌI tài khoản.

  labrat data-add <slug> <url|đường dẫn>     tải về / sao chép vào competitions/<slug>/data/raw/, tự giải nén
  labrat dataset  <slug> <thư mục> --name X  tải thư mục lên thành dataset riêng tư <username>/X
                                          trên TẤT CẢ tài khoản, ghi X vào project.yaml (own_datasets)

Dataset riêng tư của tài khoản A không đọc được từ kernel của tài khoản B, nên mỗi tài khoản
giữ một bản sao. Bundler tự đổi "X" thành "<username của tài khoản chạy>/X".

Giới hạn: Telegram Bot API chỉ cho tải file ≤ 20 MB. Dữ liệu lớn hơn: gửi link
(Google Drive, Hugging Face, Dropbox, URL trực tiếp) rồi agent dùng data-add.
"""
from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml

from . import kaggle_ops
from .accounts import load_accounts
from .bundle import slugify
from .config import comp_dir

ARCHIVES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2")


def _download(url: str, dest_dir: Path) -> Path:
    host = urlparse(url).netloc
    if "drive.google.com" in host or "docs.google.com" in host:
        try:
            import gdown
        except ImportError:
            raise SystemExit("Link Google Drive cần gói gdown: pip install gdown")
        out = gdown.download(url, output=str(dest_dir) + "/", fuzzy=True, quiet=True)
        if not out:
            raise SystemExit("gdown không tải được (file chưa chia sẻ công khai?)")
        return Path(out)
    name = Path(urlparse(url).path).name or "download.bin"
    path = dest_dir / name
    with requests.get(url, stream=True, timeout=600, headers={"User-Agent": "labrat/0.2"}) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return path


def _extract(path: Path) -> Path | None:
    name = path.name.lower()
    if not name.endswith(ARCHIVES):
        return None
    target = path.parent / slugify(name.split(".")[0])
    target.mkdir(exist_ok=True)
    if name.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            z.extractall(target)
    else:
        with tarfile.open(path) as t:
            t.extractall(target, filter="data")
    return target


def data_add(cfg: dict, src: str, link: bool = False) -> dict:
    """link=True: tạo symlink thay vì sao chép (dữ liệu lớn nằm sẵn trong dropbox)."""
    raw = comp_dir(cfg, cfg["slug"]) / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    p = Path(src).expanduser()
    if p.exists():
        dest = raw / p.name
        if link:
            if dest.is_symlink():
                dest.unlink()
            if not dest.exists():
                dest.symlink_to(p.resolve(), target_is_directory=p.is_dir())
        elif p.is_dir():
            shutil.copytree(p, dest, dirs_exist_ok=True)
        else:
            shutil.copy(p, dest)
    else:
        from . import fetcher
        dest = raw / fetcher.guess_name(src)
        res = fetcher.download(src, dest, max_gb=float(cfg.get("downloads", {}).get("max_gb", 50)))
        return {"saved": str(dest), "extracted_to": str(dest) if res["extracted"] else None,
                "n_files": res["n_files"], "total_mb": res["total_mb"], "sample": res["sample"]}
    extracted = _extract(dest) if dest.is_file() and not dest.is_symlink() else None
    root = extracted or dest
    files = [f for f in (root.rglob("*") if root.is_dir() else [root]) if f.is_file()]
    return {"saved": str(dest), "extracted_to": str(extracted) if extracted else None,
            "n_files": len(files), "total_mb": round(sum(f.stat().st_size for f in files) / 1e6, 2),
            "sample": [str(f.relative_to(raw)) for f in files[:20]]}


def upload_dataset(cfg: dict, folder: Path, name: str, message: str = "update") -> dict:
    folder = Path(folder)
    if not folder.is_dir():
        raise SystemExit(f"{folder} không phải thư mục")
    ds = slugify(name)[:50]
    if len(ds) < 5:
        ds = f"{ds}-data"
    title = name[:50] if len(name) >= 6 else f"{name} data"
    results = {}
    for acc in load_accounts(cfg):
        try:
            out = kaggle_ops.dataset_upsert(acc, folder, ds, title, message)
            results[acc.name] = {"ref": f"{acc.username}/{ds}", "ok": True, "cli": out[-200:]}
        except kaggle_ops.KaggleError as e:
            results[acc.name] = {"ref": f"{acc.username}/{ds}", "ok": False, "error": str(e)[-400:]}
    proj = comp_dir(cfg, cfg["slug"]) / "project.yaml"
    p = yaml.safe_load(proj.read_text(encoding="utf-8")) or {}
    if ds not in (p.get("own_datasets") or []):
        p.setdefault("own_datasets", []).append(ds)
        proj.write_text(yaml.safe_dump(p, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {"dataset": ds, "accounts": results,
            "note": "Chờ `kaggle datasets status` = ready trên từng tài khoản trước khi push kernel dùng nó"}


def dataset_ready(cfg: dict, name: str) -> dict:
    return {a.name: kaggle_ops.dataset_status(a, f"{a.username}/{name}") for a in load_accounts(cfg)}
