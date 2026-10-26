"""Tải dữ liệu từ đường dẫn người dùng gửi (qua Telegram hoặc `kra get`).

Hỗ trợ:
  Google Drive  file (drive.google.com/file/d/…, open?id=…, uc?id=…) và thư mục (drive/folders/…, ≤ 50 file/lần)
  Google Sheets docs.google.com/spreadsheets/d/<id> → tải CSV (sheet đầu, hoặc gid trong link)
  Hugging Face  huggingface.co/datasets/<org>/<tên> (cả repo) · …/resolve|blob/<rev>/<file> (một file)
  Kaggle        kaggle.com/datasets/<owner>/<slug> (dùng tài khoản Kaggle đầu tiên)
  Dropbox       link chia sẻ (tự đổi dl=1)
  GitHub        github.com/<owner>/<repo> (zip nhánh mặc định) · …/blob/… → raw
  Link trực tiếp http(s) tới file (csv, zip, parquet, …)
Không coi là "dữ liệu" (để agent xử lý): link cuộc thi Kaggle, arXiv, file PDF (→ nghiên cứu paper).

Tải vào <dropbox>/<tên>/, tự giải nén zip/tar, kiểm tra dung lượng đĩa, báo tiến độ qua callback.
"""
from __future__ import annotations

import os
import re
import shutil
import tarfile
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlencode, urlparse, urlunparse

import requests

from .bundle import slugify

URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
ARCHIVES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz")
UA = {"User-Agent": "Mozilla/5.0 (kaggle-research-agent)"}
MIN_FREE_GB = 2.0


class FetchError(RuntimeError):
    pass


# ---------------------------------------------------------------- nhận diện link
def classify(url: str) -> tuple[str, dict]:
    u = urlparse(url)
    host, path = u.netloc.lower().removeprefix("www."), u.path
    q = parse_qs(u.query)
    if host.endswith("kaggle.com"):
        m = re.match(r"/datasets/([^/]+)/([^/?#]+)", path)
        if m:
            return "kaggle_dataset", {"ref": f"{m.group(1)}/{m.group(2)}", "name": m.group(2)}
        if path.startswith(("/competitions/", "/c/")):
            return "competition", {}
        return "unknown", {}
    if "arxiv.org" in host or path.lower().endswith(".pdf"):
        return "paper", {}
    if host == "docs.google.com" and "/spreadsheets/d/" in path:
        sid = path.split("/spreadsheets/d/")[1].split("/")[0]
        gid = (q.get("gid") or [None])[0] or (re.search(r"gid=(\d+)", u.fragment or "") or [None, None])[1]
        return "gsheet", {"id": sid, "gid": gid}
    if host in ("drive.google.com", "docs.google.com"):
        if "/folders/" in path:
            return "gdrive_folder", {"id": path.split("/folders/")[1].split("/")[0]}
        m = re.search(r"/d/([\w-]{10,})", path)
        fid = m.group(1) if m else (q.get("id") or [None])[0]
        if fid:
            return "gdrive_file", {"id": fid}
    if host == "huggingface.co":
        parts = [p for p in path.split("/") if p]
        if parts and parts[0] == "datasets" and len(parts) >= 3:
            repo = f"{parts[1]}/{parts[2]}"
            if len(parts) >= 6 and parts[3] in ("resolve", "blob"):
                return "direct", {"url": f"https://huggingface.co/datasets/{repo}/resolve/{parts[4]}/"
                                         + "/".join(parts[5:]), "hf_token": True}
            sub = "/".join(parts[5:]) if len(parts) >= 5 and parts[3] == "tree" else None
            return "hf_dataset", {"repo": repo, "revision": parts[4] if sub is not None else None,
                                  "subdir": sub, "name": parts[2]}
        if len(parts) >= 5 and parts[2] in ("resolve", "blob"):
            return "direct", {"url": f"https://huggingface.co/{parts[0]}/{parts[1]}/resolve/{parts[3]}/"
                                     + "/".join(parts[4:]), "hf_token": True}
    if host in ("dropbox.com", "dl.dropboxusercontent.com"):
        q["dl"] = ["1"]
        return "direct", {"url": urlunparse(u._replace(query=urlencode({k: v[0] for k, v in q.items()})))}
    if host == "github.com":
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 5 and parts[2] == "blob":
            return "direct", {"url": f"https://raw.githubusercontent.com/{parts[0]}/{parts[1]}/{parts[3]}/"
                                     + "/".join(parts[4:])}
        if len(parts) == 2 or (len(parts) >= 2 and parts[2] == "tree"):
            return "direct", {"url": f"https://github.com/{parts[0]}/{parts[1]}/archive/HEAD.zip",
                              "filename": f"{parts[1]}.zip"}
    return "direct", {"url": url}


def data_urls(text: str) -> tuple[list[str], list[str]]:
    """Tách link trong tin nhắn: (link dữ liệu cần tải, link để agent tự xử lý)."""
    data, other = [], []
    for url in URL_RE.findall(text or ""):
        url = url.rstrip(".,;")
        kind, _ = classify(url)
        (other if kind in ("competition", "paper", "unknown") else data).append(url)
    return data, other


def guess_name(url: str) -> str:
    kind, info = classify(url)
    if info.get("name"):
        return slugify(info["name"])[:40]
    p = Path(unquote(urlparse(info.get("url", url)).path))
    stem = p.name
    for ext in ARCHIVES + (".csv", ".parquet", ".json", ".jsonl", ".txt", ".xlsx", ".npz", ".zip"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    base = slugify(stem)[:40]
    if kind.startswith("gdrive") or kind == "gsheet" or len(base) < 3:
        base = f"data-{datetime.now():%Y%m%d-%H%M}"
    return base


# ---------------------------------------------------------------- tiện ích
def _dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def _check_space(dest: Path, need: int | None, max_bytes: int):
    free = shutil.disk_usage(dest).free
    if need and need > max_bytes:
        raise FetchError(f"Dữ liệu {need/1e9:.1f} GB vượt giới hạn {max_bytes/1e9:.0f} GB "
                         "(downloads.max_gb trong config.yaml).")
    if need and need * 2.2 > free - MIN_FREE_GB * 1e9:  # x2.2: file nén + bản giải nén
        raise FetchError(f"Không đủ ổ đĩa: cần ~{need*2.2/1e9:.1f} GB, còn trống {free/1e9:.1f} GB.")
    if free < MIN_FREE_GB * 1e9:
        raise FetchError(f"Ổ đĩa sắp đầy (còn {free/1e9:.1f} GB).")


def _filename_from(resp: requests.Response, fallback: str) -> str:
    cd = resp.headers.get("content-disposition", "")
    m = re.search(r"filename\*=UTF-8''([^;]+)", cd) or re.search(r'filename="?([^";]+)"?', cd)
    name = unquote(m.group(1)) if m else fallback
    return Path(name).name or "download.bin"


def _http(url: str, dest: Path, max_bytes: int, filename: str | None = None, token: str | None = None,
          cancel: threading.Event | None = None) -> Path:
    headers = dict(UA)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with requests.get(url, stream=True, timeout=(30, 300), headers=headers, allow_redirects=True) as r:
        if r.status_code in (401, 403):
            raise FetchError(f"Bị từ chối ({r.status_code}) — link riêng tư hoặc cần đăng nhập "
                             "(Hugging Face gated: đặt HF_TOKEN trong .env).")
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        total = int(r.headers.get("content-length") or 0) or None
        _check_space(dest, total, max_bytes)
        name = filename or _filename_from(r, Path(unquote(urlparse(r.url).path)).name or "download.bin")
        out = dest / name
        part = Path(str(out) + ".part")
        try:
            _stream(r, part, ctype, max_bytes, cancel)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        os.replace(part, out)
        return out


def _stream(r: requests.Response, part: Path, ctype: str, max_bytes: int, cancel) -> None:
    first, written = True, 0
    with open(part, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            if first and "text/html" in ctype and chunk.lstrip()[:15].lower().startswith((b"<!doctype", b"<html")):
                raise FetchError("Link trả về một trang web chứ không phải file. Hãy gửi link tải trực tiếp "
                                 "(hoặc bật chia sẻ công khai).")
            first = False
            f.write(chunk)
            written += len(chunk)
            if written > max_bytes:
                raise FetchError(f"Vượt giới hạn {max_bytes/1e9:.0f} GB, đã dừng tải.")
            if cancel and cancel.is_set():
                raise FetchError("Đã huỷ.")


def extract_archives(root: Path) -> list[str]:
    done = []
    for p in sorted(root.rglob("*")):
        n = p.name.lower()
        if not p.is_file() or not n.endswith(ARCHIVES):
            continue
        target = p.parent  # giải nén ngay cạnh file nén
        try:
            if n.endswith(".zip"):
                with zipfile.ZipFile(p) as z:
                    z.extractall(target)  # zipfile tự loại đường dẫn tuyệt đối/".."
            else:
                with tarfile.open(p) as t:
                    t.extractall(target, filter="data")
            p.unlink()  # bỏ file nén sau khi giải nén thành công để tiết kiệm đĩa
            done.append(p.name)
        except (zipfile.BadZipFile, tarfile.TarError) as e:
            done.append(f"{p.name} (lỗi giải nén: {e})")
    return done


# ---------------------------------------------------------------- tải
def download(url: str, dest: Path, *, max_gb: float = 50, progress=None,
             cancel: threading.Event | None = None) -> dict:
    """Tải `url` vào thư mục `dest`. progress(bytes_so_far, total_or_None, note) gọi định kỳ."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    max_bytes = int(max_gb * 1e9)
    kind, info = classify(url)
    if kind in ("competition", "paper", "unknown"):
        raise FetchError(f"Link loại '{kind}' không phải dữ liệu để tải; agent sẽ xử lý.")
    _check_space(dest, None, max_bytes)
    before = _dir_bytes(dest)

    stop = threading.Event()

    def monitor():
        while not stop.wait(15):
            if progress:
                progress(_dir_bytes(dest) - before, None, kind)
    th = threading.Thread(target=monitor, daemon=True)
    th.start()
    t0 = time.time()
    try:
        if kind == "direct":
            token = os.environ.get("HF_TOKEN") if info.get("hf_token") else None
            _http(info["url"], dest, max_bytes, info.get("filename"), token, cancel)
        elif kind == "gsheet":
            u = f"https://docs.google.com/spreadsheets/d/{info['id']}/export?format=csv"
            if info.get("gid"):
                u += f"&gid={info['gid']}"
            _http(u, dest, max_bytes, f"sheet-{info['id'][:8]}{'-' + info['gid'] if info.get('gid') else ''}.csv",
                  cancel=cancel)
        elif kind == "gdrive_file":
            import gdown
            out = gdown.download(id=info["id"], output=str(dest) + os.sep, quiet=True, resume=True,
                                 cancel=cancel)
            if not out:
                raise FetchError("Google Drive không cho tải: file cần được chia sẻ "
                                 "'Bất kỳ ai có đường liên kết'.")
        elif kind == "gdrive_folder":
            import gdown
            outs = gdown.download_folder(id=info["id"], output=str(dest), quiet=True, resume=True)
            if not outs:
                raise FetchError("Không tải được thư mục Google Drive (chưa chia sẻ công khai, hoặc > 50 file — "
                                 "hãy nén thành một file zip rồi gửi link file).")
        elif kind == "hf_dataset":
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=info["repo"], repo_type="dataset", revision=info.get("revision"),
                              local_dir=str(dest), token=os.environ.get("HF_TOKEN") or None,
                              allow_patterns=[f"{info['subdir']}/*"] if info.get("subdir") else None)
            shutil.rmtree(dest / ".cache", ignore_errors=True)
        elif kind == "kaggle_dataset":
            from . import kaggle_ops
            from .accounts import load_accounts
            from .config import load_config
            acc = load_accounts(load_config())[0]
            kaggle_ops.run(["datasets", "download", "-d", info["ref"], "-p", str(dest), "--unzip"], acc,
                           timeout=7200)
    except FetchError:
        raise
    except Exception as e:
        raise FetchError(f"{type(e).__name__}: {str(e)[:400]}") from e
    finally:
        stop.set()
    extracted = extract_archives(dest)
    files = [f for f in dest.rglob("*") if f.is_file() and not f.name.endswith(".part")]
    for leftover in dest.rglob("*.part"):
        leftover.unlink()
    return {"url": url, "kind": kind, "dest": str(dest), "n_files": len(files),
            "total_mb": round(sum(f.stat().st_size for f in files) / 1e6, 2), "extracted": extracted,
            "seconds": round(time.time() - t0, 1),
            "sample": [str(f.relative_to(dest)) for f in sorted(files)[:15]]}
