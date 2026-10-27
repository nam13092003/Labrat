"""Bọc Kaggle CLI chính thức (pip install kaggle>=1.8), theo từng tài khoản.

Chỉ dùng các lệnh: kernels push/status/output, datasets create/version/status,
competitions files/download, kernels list (kiểm tra đăng nhập).
KHÔNG có lệnh nộp bài — công cụ này không nộp bài.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .accounts import Account, env_for
from .config import ROOT

# Thư mục tạm riêng cho mọi lệnh kaggle (CLI nén dataset vào TMPDIR trước khi upload — 4 GB dữ liệu là
# 4 GB file tạm). Xoá ngay sau lệnh; thư mục bỏ lại do tiến trình bị giết được housekeeping dọn sau.
TMP_ROOT = ROOT / ".labrat_tmp"


class KaggleError(RuntimeError):
    pass


def _bin() -> str:
    # cron/systemd không kích hoạt venv → tìm cả cạnh trình thông dịch hiện tại
    b = shutil.which("kaggle") or shutil.which("kaggle", path=str(Path(sys.executable).parent))
    if not b:
        raise KaggleError("Không tìm thấy lệnh `kaggle`. Cài bằng: pip install -r requirements.txt")
    return b


def run(args: list[str], acc: Account | None = None, timeout: int = 900) -> str:
    env = env_for(acc) if acc else os.environ.copy()
    tmp = TMP_ROOT / f"{os.getpid()}-{time.time_ns()}"
    tmp.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = env["TMP"] = env["TEMP"] = str(tmp)
    try:
        proc = subprocess.run([_bin(), *args], capture_output=True, text=True, timeout=timeout, env=env)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    who = f"[{acc.name}/{acc.username}] " if acc else ""
    if proc.returncode != 0:
        raise KaggleError(f"{who}kaggle {' '.join(args)} lỗi ({proc.returncode}):\n{out}")
    if re.search(r"\b(401|403)\b.*(Client Error|Unauthorized|Forbidden)", out):
        raise KaggleError(who + out)
    return out


def whoami(acc: Account) -> str:
    """Kiểm tra đăng nhập: liệt kê kernel của chính tài khoản."""
    return run(["kernels", "list", "--mine", "--page-size", "1"], acc, timeout=120)


# ---------------- kernels ----------------
def push_kernel(acc: Account, folder: Path, accelerator: str | None = None, timeout_s: int | None = None) -> str:
    args = ["kernels", "push", "-p", str(folder)]
    if accelerator:
        args += ["--accelerator", accelerator]
    if timeout_s:
        args += ["-t", str(int(timeout_s))]
    try:
        return run(args, acc)
    except KaggleError as e:
        # CLI cũ chưa có --accelerator: chuyển sang trường machine_shape trong metadata
        if accelerator and ("unrecognized arguments" in str(e) or "--accelerator" in str(e)):
            meta_p = Path(folder) / "kernel-metadata.json"
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            meta["machine_shape"] = accelerator
            meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            args = [a for a in args if a not in ("--accelerator", accelerator)]
            return run(args, acc)
        raise


_STATUS_ORDER = ["cancel", "error", "complete", "running", "queued"]


def kernel_status(acc: Account, ref: str) -> str:
    """Trả về một trong: queued, running, complete, error, cancelled, unknown."""
    out = run(["kernels", "status", ref], acc, timeout=120).lower()
    for key in _STATUS_ORDER:
        if key in out:
            return "cancelled" if key == "cancel" else key
    return "unknown"


def kernel_output(acc: Account, ref: str, dest: Path, pattern: str | None = None) -> dict:
    """Tải output kernel. pattern = regex tên file (chỉ tải file khớp; cần kaggle CLI có --file-pattern).
    CLI cũ không hỗ trợ → tải hết (người gọi tự cắt file lớn sau đó)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    args = ["kernels", "output", ref, "-p", str(dest), "-o"]
    filtered = False
    if pattern:
        try:
            run(args + ["--file-pattern", pattern], acc, timeout=1800)
            filtered = True
        except KaggleError as e:
            if "unrecognized arguments" not in str(e) and "--file-pattern" not in str(e):
                raise
    if not filtered:
        run(args, acc, timeout=3600)
    return {"filtered": filtered, "files": sorted(p for p in dest.rglob("*") if p.is_file())}


# ---------------- datasets ----------------
def dataset_upsert(acc: Account, folder: Path, slug: str, title: str, message: str = "update") -> str:
    """Tạo dataset riêng tư <username>/<slug> nếu chưa có, hoặc tạo version mới."""
    folder = Path(folder)
    ref = f"{acc.username}/{slug}"
    (folder / "dataset-metadata.json").write_text(json.dumps(
        {"title": title, "id": ref, "licenses": [{"name": "CC0-1.0"}]}, indent=2), encoding="utf-8")
    try:
        return run(["datasets", "version", "-p", str(folder), "-m", message, "-r", "zip"], acc, timeout=7200)
    except KaggleError as e:
        if "not found" in str(e).lower() or "404" in str(e) or "does not exist" in str(e).lower():
            return run(["datasets", "create", "-p", str(folder), "-r", "zip"], acc, timeout=7200)
        raise


def dataset_status(acc: Account, ref: str) -> str:
    return run(["datasets", "status", ref], acc, timeout=120).strip().lower()


# ---------------- competitions ----------------
def competition_files(acc: Account, slug: str) -> str:
    return run(["competitions", "files", slug], acc, timeout=300)


def competition_download(acc: Account, slug: str, dest: Path) -> str:
    Path(dest).mkdir(parents=True, exist_ok=True)
    return run(["competitions", "download", "-c", slug, "-p", str(dest)], acc, timeout=7200)
