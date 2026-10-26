"""Thư mục thả dữ liệu: người dùng chép dữ liệu vào, agent tự nhận.

Cấu trúc (mặc định <project>/dropbox, đổi bằng paths.dropbox trong config.yaml):

    dropbox/
      churn-2026/              ← MỖI THƯ MỤC CON = một bộ dữ liệu / một dự án
        train.csv
        test.csv
        task.md                ← (khuyên dùng) mô tả bài toán: mục tiêu, metric, ngân sách GPU...
      playground-series-s6e1/  ← trùng tên dự án đã có → coi là dữ liệu bổ sung cho dự án đó
      paper.pdf                ← file lẻ ở gốc cũng được nhận (PDF → agent nghiên cứu paper)

Quy tắc nhận:
- Bỏ qua file ẩn, file đang chép dở (.part .tmp .crdownload .filepart), thư mục `_archive`.
- Chỉ nhận khi đã "ổn định": không đổi kích thước giữa hai lần quét và file mới nhất đã cũ hơn 3 phút
  (để không xử lý khi đang chép). Lệnh /scan trên Telegram bỏ qua yêu cầu quét hai lần.
- Mỗi lần nội dung thay đổi (thêm/sửa file) sẽ được báo lại như một bản cập nhật.
Dữ liệu gốc KHÔNG bị di chuyển hay xoá; agent sao chép/đọc từ đây.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .config import ROOT, competitions_root, load_config

STATE = ROOT / ".dropbox_state.json"
SKIP_SUFFIX = (".part", ".tmp", ".crdownload", ".filepart", ".partial", ".swp")
TASK_FILES = ("task.md", "task.txt", "mo-ta.md", "mo-ta.txt", "README.md", "README.txt", "readme.md")
SETTLE_SECONDS = 180


def _skip(name: str) -> bool:
    n = name.lower()
    return name.startswith((".", "_", "~$")) or n.endswith(SKIP_SUFFIX) or n.startswith("readme")


def dropbox_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    p = Path(cfg.get("paths", {}).get("dropbox", "dropbox")).expanduser()
    d = p if p.is_absolute() else ROOT / p
    d.mkdir(parents=True, exist_ok=True)
    return d


def _files(item: Path) -> list[Path]:
    if item.is_file():
        return [item]
    out = []
    for f in item.rglob("*"):
        rel = f.relative_to(item)
        if any(part.startswith(".") or part.startswith("~$") for part in rel.parts):
            continue
        if f.is_file() and not f.name.lower().endswith(SKIP_SUFFIX) and f.name != "dataset-metadata.json":
            out.append(f)
    return out


def _signature(files: list[Path]) -> dict:
    sizes = [f.stat().st_size for f in files]
    mtimes = [f.stat().st_mtime for f in files]
    return {"n_files": len(files), "bytes": sum(sizes), "newest": max(mtimes) if mtimes else 0}


def _task_text(item: Path) -> str:
    if item.is_dir():
        for name in TASK_FILES:
            p = item / name
            if p.exists():
                return p.read_text(encoding="utf-8", errors="replace")[:2000]
    return ""


def scan(force: bool = False) -> list[dict]:
    """Quét dropbox. Trả về các mục MỚI hoặc VỪA THAY ĐỔI đã ổn định (mỗi phiên bản chỉ báo một lần)."""
    cfg = load_config()
    root = dropbox_dir(cfg)
    projects = {p.parent.name for p in competitions_root(cfg).glob("*/project.yaml")}
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    now = time.time()
    ready, seen = [], set()
    for item in sorted(root.iterdir()):
        name = item.name
        if _skip(name):
            continue
        files = _files(item)
        if not files:
            continue
        seen.add(name)
        sig = _signature(files)
        st = state.get(name, {})
        stable = now - sig["newest"] > SETTLE_SECONDS and (force or st.get("last_sig") == sig)
        st["last_sig"] = sig
        if stable and st.get("announced_sig") != sig:
            kind = "update" if st.get("announced_sig") else "new"
            st["announced_sig"] = sig
            st["announced_at"] = now
            exts = sorted({f.suffix.lower() or "(không đuôi)" for f in files})
            ready.append({
                "name": name, "path": str(item), "kind": kind, "is_file": item.is_file(),
                "n_files": sig["n_files"], "total_mb": round(sig["bytes"] / 1e6, 2), "extensions": exts[:12],
                "sample": [str(f.relative_to(item)) if item.is_dir() else f.name for f in files[:15]],
                "matches_project": name if name in projects else None,
                "task": _task_text(item),
            })
        state[name] = st
    for gone in set(state) - seen:  # mục đã bị xoá khỏi dropbox
        state.pop(gone)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    return ready


def announce_now(path: Path, kind: str = "new") -> dict:
    """Đăng ký ngay một mục vừa tải xong (bỏ qua chờ ổn định) để cron không báo lặp."""
    path = Path(path)
    cfg = load_config()
    projects = {p.parent.name for p in competitions_root(cfg).glob("*/project.yaml")}
    files = _files(path)
    sig = _signature(files)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    st = state.get(path.name, {})
    if st.get("announced_sig"):
        kind = "update"
    st.update(last_sig=sig, announced_sig=sig, announced_at=time.time())
    state[path.name] = st
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    return {"name": path.name, "path": str(path), "kind": kind, "is_file": path.is_file(),
            "n_files": sig["n_files"], "total_mb": round(sig["bytes"] / 1e6, 2),
            "extensions": sorted({f.suffix.lower() or "(không đuôi)" for f in files})[:12],
            "sample": [str(f.relative_to(path)) for f in files[:15]] if path.is_dir() else [path.name],
            "matches_project": path.name if path.name in projects else None, "task": _task_text(path)}


def listing() -> list[dict]:
    cfg = load_config()
    root = dropbox_dir(cfg)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    rows = []
    for item in sorted(root.iterdir()):
        if _skip(item.name):
            continue
        files = _files(item)
        sig = _signature(files)
        st = state.get(item.name, {})
        status = ("đã giao agent" if st.get("announced_sig") == sig else
                  "đang chép / chờ ổn định" if time.time() - sig["newest"] <= SETTLE_SECONDS else "chờ lần quét tới")
        rows.append({"name": item.name, "n_files": sig["n_files"], "total_mb": round(sig["bytes"] / 1e6, 2),
                     "status": status})
    return rows


def wake_message(it: dict) -> str:
    head = ("Dữ liệu CẬP NHẬT" if it["kind"] == "update" else "Dữ liệu MỚI") + \
        f" trong dropbox: {it['path']} ({it['n_files']} file, {it['total_mb']} MB, {', '.join(it['extensions'])})."
    if it["matches_project"]:
        head += f" Trùng tên dự án '{it['matches_project']}' → dữ liệu bổ sung cho dự án đó."
    else:
        head += " Chưa có dự án tương ứng."
    head += "\nVí dụ file: " + ", ".join(it["sample"][:8])
    if it["task"]:
        head += f"\nMô tả của người dùng (task.md):\n{it['task']}"
    else:
        head += "\nKhông có task.md: nếu chưa rõ mục tiêu/metric/ngân sách, hãy hỏi người dùng trước khi tiêu GPU."
    head += "\nLàm theo Quy trình C trong AGENT.md (PDF → Quy trình B)."
    return head
