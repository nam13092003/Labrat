"""Tải mô hình / dữ liệu NGAY TRÊN KAGGLE, không đi qua VPS.

  labrat kaggle-fetch <slug> <tên> --hf <repo_id> [--repo-type model|dataset] [--include "*.safetensors"]
  labrat kaggle-fetch <slug> <tên> --url <link-tải-trực-tiếp> [--url ...]
  labrat kaggle-fetch <slug> <tên> --cmd "<lệnh shell tải về thư mục hiện tại>"
  labrat kaggle-fetch-status <slug> [<tên>]

Cách làm: trên MỖI tài khoản Kaggle, đẩy một kernel CPU (có internet, không tốn quota GPU) tải về
/kaggle/working/<tên>/. Output của kernel đó được Kaggle giữ lại và gắn vào kernel thí nghiệm như một nguồn
dữ liệu (kernel_sources) — thí nghiệm khai báo `fetched: [<tên>]` trong job.yaml, file xuất hiện ở
/kaggle/input/<tiêu-đề-kernel-tải>/<tên>/ (dùng find_input). Mỗi tài khoản một bản vì output kernel riêng tư
chỉ tài khoản sở hữu đọc được.

Giới hạn: output kernel tối đa ~20 GB; phiên CPU tối đa 12h. Chỉ nguồn công khai — KHÔNG đưa token (HF, ...)
vào kernel. Mô hình có sẵn trên Kaggle Models → dùng `model_sources` trong job.yaml, khỏi tải.
"""
from __future__ import annotations

import json
import shutil

import yaml

from . import kaggle_ops
from .accounts import load_accounts
from .bundle import slugify
from .config import ROOT, comp_dir
from .registry import now_iso


def title_for(cfg: dict, name: str) -> str:
    prefix = cfg.get("kaggle", {}).get("title_prefix", "labrat")
    return slugify(f"{prefix}-fetch-{cfg['slug'][:14]}-{name}")[:50].strip("-")


def _script(name: str, hf: str | None, repo_type: str, include: list[str], urls: list[str], cmd: str | None) -> str:
    spec = json.dumps({"name": name, "hf": hf, "repo_type": repo_type, "include": include, "urls": urls, "cmd": cmd},
                      ensure_ascii=False)
    return f'''# Labrat kaggle-fetch: tải về /kaggle/working/<tên>/ (output được giữ làm nguồn cho kernel khác)
import json, os, subprocess, sys, time, urllib.parse
S = json.loads({spec!r})
OUT = os.path.join("/kaggle/working", S["name"])
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
if S["hf"]:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=S["hf"], repo_type=S["repo_type"], local_dir=OUT,
                      allow_patterns=S["include"] or None)
for u in S["urls"]:
    fn = os.path.basename(urllib.parse.urlparse(u).path) or "download"
    subprocess.run(["wget", "-q", "-O", os.path.join(OUT, fn), u], check=True)
if S["cmd"]:
    subprocess.run(S["cmd"], shell=True, cwd=OUT, check=True)
for root, dirs, files in os.walk(OUT):  # bỏ cache của huggingface_hub
    if ".cache" in dirs:
        subprocess.run(["rm", "-rf", os.path.join(root, ".cache")])
files = []
for root, _, fs in os.walk(OUT):
    for f in fs:
        p = os.path.join(root, f)
        files.append({{"path": os.path.relpath(p, OUT), "mb": round(os.path.getsize(p) / 1e6, 1)}})
total = sum(f["mb"] for f in files)
json.dump({{"name": S["name"], "files": files, "total_mb": round(total, 1), "seconds": round(time.time() - t0)}},
          open(os.path.join(OUT, "_manifest.json"), "w"), indent=1)
print(f"[labrat] tải xong {{len(files)}} file, {{total:.0f}} MB trong {{time.time() - t0:.0f}}s")
'''


def _project_file(cfg: dict):
    return comp_dir(cfg, cfg["slug"]) / "project.yaml"


def _save(cfg: dict, name: str, entry: dict) -> None:
    f = _project_file(cfg)
    p = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    p.setdefault("kaggle_fetched", {})[name] = entry
    f.write_text(yaml.safe_dump(p, allow_unicode=True, sort_keys=False), encoding="utf-8")


def create(cfg: dict, name: str, *, hf: str | None = None, repo_type: str = "model", include: list[str] | None = None,
           urls: list[str] | None = None, cmd: str | None = None, accounts: list[str] | None = None) -> dict:
    name = slugify(name)
    if not (hf or urls or cmd):
        raise SystemExit("Cần --hf, --url hoặc --cmd.")
    title = title_for(cfg, name)
    accs = [a for a in load_accounts(cfg) if not accounts or a.name in accounts]
    pushed = {}
    for acc in accs:
        folder = ROOT / ".labrat_tmp" / f"kfetch-{name}-{acc.name}"
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        (folder / "main.py").write_text(_script(name, hf, repo_type, include or [], urls or [], cmd), encoding="utf-8")
        (folder / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{acc.username}/{title}", "title": title, "code_file": "main.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []},
            indent=2), encoding="utf-8")
        try:
            kaggle_ops.push_kernel(acc, folder, accelerator=None, timeout_s=12 * 3600)
            pushed[acc.name] = {"ref": f"{acc.username}/{title}", "status": "queued"}
        except kaggle_ops.KaggleError as e:
            pushed[acc.name] = {"ref": f"{acc.username}/{title}", "status": "push_error", "error": str(e)[-300:]}
        finally:
            shutil.rmtree(folder, ignore_errors=True)
    entry = {"title": title, "source": {"hf": hf, "repo_type": repo_type, "include": include, "urls": urls,
                                        "cmd": cmd}, "pushed_at": now_iso(), "accounts": pushed}
    _save(cfg, name, entry)
    return {"name": name, "title": title, "accounts": pushed,
            "use": f"job.yaml: fetched: [{name}]  → file ở /kaggle/input/{title}/{name}/ (find_input)",
            "next": f"Chờ `labrat kaggle-fetch-status {cfg['slug']} {name}` báo complete trên mọi tài khoản."}


def status(cfg: dict, name: str | None = None) -> dict:
    fetched = (cfg.get("kaggle_fetched") or {})
    names = [name] if name else list(fetched)
    accs = {a.name: a for a in load_accounts(cfg)}
    out = {}
    for n in names:
        e = fetched.get(n)
        if not e:
            raise SystemExit(f"Chưa có bản tải '{n}'. Có: {list(fetched)}")
        for an, info in e.get("accounts", {}).items():
            if info.get("status") in ("complete", "push_error") or an not in accs:
                continue
            try:
                info["status"] = kaggle_ops.kernel_status(accs[an], info["ref"])
            except kaggle_ops.KaggleError as ex:
                info["status"] = "unknown"
                info["error"] = str(ex)[-200:]
        _save(cfg, n, e)
        out[n] = {"title": e["title"], "accounts": {a: i["status"] for a, i in e["accounts"].items()},
                  "ready": all(i["status"] == "complete" for i in e["accounts"].values())}
    return out


def kernel_sources_for(cfg: dict, job: dict, username: str) -> list[str]:
    """kernel_sources cho các bản tải mà job.yaml khai báo `fetched: [...]`, đúng tài khoản đang chạy."""
    out = []
    fetched = cfg.get("kaggle_fetched") or {}
    for n in job.get("fetched") or []:
        e = fetched.get(slugify(n))
        if not e:
            raise SystemExit(f"job.yaml dùng fetched '{n}' nhưng chưa tải: labrat kaggle-fetch {cfg['slug']} {n} ...")
        if not any(i.get("ref", "").startswith(username + "/") and i.get("status") != "push_error"
                   for i in e.get("accounts", {}).values()):
            raise SystemExit(f"Bản tải '{n}' chưa có trên tài khoản {username} — chạy lại kaggle-fetch.")
        out.append(f"{username}/{e['title']}")
    return out
