"""Cầu nối Telegram ↔ agent (long polling — không cần mở cổng, không cần domain).

  python -m kra.bridge          (chạy như service systemd: kra-bridge)

- Chỉ nhận tin từ TELEGRAM_CHAT_ID (mọi chat khác bị bỏ qua và ghi log).
- Lệnh nhanh xử lý tại chỗ, không tốn LLM: /help /status /accounts /new /stop <slug> /resume <slug>
- Tin khác → hàng đợi → harness (một lượt một lúc) → trả lời về Telegram.
- File gửi kèm (≤ 20 MB, giới hạn của Bot API) lưu vào inbox/<ngày>/ và báo cho agent đường dẫn.
- Cron (scripts/tick_all.py) thả "wake" vào .wake_queue/; bridge gom lại và gọi agent.
"""
from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import requests
import yaml

from . import agent, budget, dropbox, experiments, fetcher
from .config import ROOT, comp_dir, competitions_root, load_config, load_env

WAKE_DIR = ROOT / ".wake_queue"
INBOX = ROOT / "inbox"
OFFSET_FILE = ROOT / ".tg_offset"
MAX_FILE = 20 * 1024 * 1024

HELP = """Lệnh nhanh:
/status — tình hình mọi dự án
/accounts — quota & kernel từng tài khoản Kaggle
/new — bắt đầu phiên agent mới (quên ngữ cảnh cũ; lượt sau sẽ cache miss một lần)
/usage — token & tỉ lệ cache hit 24h qua
/stop <slug> — tạm dừng dự án (không đẩy thí nghiệm mới)
/resume <slug> — tiếp tục dự án
/dropbox — xem thư mục thả dữ liệu
/scan — quét thư mục thả dữ liệu ngay (không chờ cron)
/get <link> [tên] — tải dữ liệu từ link (gửi link trong tin nhắn thường cũng được)
/cancel — huỷ lượt tải đang chạy

Gửi link dữ liệu (Google Drive/Sheets, Hugging Face, Kaggle dataset, Dropbox, GitHub, link trực tiếp)
kèm mô tả bài toán → tự tải về dropbox rồi giao cho agent. Đặt tên thư mục: thêm "tên: <ten>".

Mọi tin khác được chuyển cho agent, ví dụ:
• "Nghiên cứu cuộc thi https://www.kaggle.com/competitions/..., ngân sách 15h GPU"
• "Đọc paper 2406.12345 và đề xuất hướng thực nghiệm cho dự án X"
• gửi file dữ liệu (≤ 20 MB) hoặc link Google Drive / Hugging Face kèm mô tả bài toán
• "báo cáo dự án X" """


class Telegram:
    def __init__(self):
        load_env()
        self.token = os.environ["TELEGRAM_BOT_TOKEN"]
        self.chat = str(os.environ["TELEGRAM_CHAT_ID"])
        self.base = f"https://api.telegram.org/bot{self.token}"

    def call(self, method, timeout=70, **params):
        r = requests.post(f"{self.base}/{method}", data=params, timeout=timeout)
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(f"{method}: {data}")
        return data["result"]

    def send(self, text: str) -> int | None:
        text = text.strip() or "(trống)"
        mid = None
        for i in range(0, len(text), 4000):
            try:
                mid = self.call("sendMessage", chat_id=self.chat, text=text[i:i + 4000],
                                disable_web_page_preview="true")["message_id"]
            except Exception as e:
                print("send lỗi:", e)
        return mid

    def edit(self, mid: int | None, text: str):
        if mid is None:
            self.send(text)
            return
        try:
            self.call("editMessageText", chat_id=self.chat, message_id=mid, text=text[:4000],
                      disable_web_page_preview="true")
        except Exception as e:  # "message is not modified" và lỗi tạm thời: bỏ qua
            if "not modified" not in str(e):
                print("edit lỗi:", e)

    def typing(self):
        try:
            self.call("sendChatAction", timeout=10, chat_id=self.chat, action="typing")
        except Exception:
            pass

    def download(self, file_id: str, name: str) -> Path:
        info = self.call("getFile", file_id=file_id)
        d = INBOX / datetime.now().strftime("%Y%m%d")
        d.mkdir(parents=True, exist_ok=True)
        path = d / Path(name).name
        if path.exists():
            path = d / f"{path.stem}-{int(time.time())}{path.suffix}"
        r = requests.get(f"https://api.telegram.org/file/bot{self.token}/{info['file_path']}", timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
        return path


# ---------------------------------------------------------------- lệnh nhanh
def _projects():
    root = competitions_root(load_config())
    return sorted(p.parent.name for p in root.glob("*/project.yaml") if not p.parent.name.startswith("_"))


def quick_status() -> str:
    lines = []
    for slug in _projects():
        cfg = load_config(slug)
        reg = experiments.registry_for(cfg)
        best = reg.best(cfg.get("metric_direction", "maximize"))
        b = budget.summary(cfg, reg)
        act = ", ".join(f"{e['id']}({e['status']})" for e in reg.active()) or "không"
        lines.append(f"• {slug}{' [TẠM DỪNG]' if cfg.get('paused') else ''}: best "
                     f"{best['cv']:.5f} ({best['id']})" if best else f"• {slug}: chưa có kết quả")
        lines[-1] += f"\n  đang chạy: {act} · GPU {b['project_used']:.1f}/{b['project_cap']:.0f}h · {b['phase']}"
    return "\n".join(lines) or "Chưa có dự án nào."


def quick_accounts() -> str:
    rows = budget.accounts_status(load_config())
    return "\n".join(f"• {r['name']} ({r['username']}): {r['week_used']:.1f}/{r['weekly_gpu_hours']:.0f}h tuần · "
                     f"{'GPU: ' + r['gpu_busy'] if r['gpu_busy'] else 'rảnh'}" for r in rows)


def set_paused(slug: str, paused: bool) -> str:
    cfg = load_config()
    proj = comp_dir(cfg, slug) / "project.yaml"
    if not proj.exists():
        return f"Không có dự án {slug}. Có: {', '.join(_projects())}"
    p = yaml.safe_load(proj.read_text(encoding="utf-8")) or {}
    if paused:
        p["paused"] = True
    else:
        p.pop("paused", None)
    proj.write_text(yaml.safe_dump(p, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return (f"⏸ Đã tạm dừng {slug}: không đẩy thí nghiệm mới; kernel đang chạy vẫn được thu kết quả."
            if paused else f"▶️ Đã tiếp tục {slug}.")


def handle_quick(text: str) -> str | None:
    parts = text.strip().split()
    cmd = parts[0].split("@")[0].lower() if parts else ""
    arg = parts[1] if len(parts) > 1 else None
    if cmd in ("/start", "/help"):
        return HELP
    if cmd == "/status":
        return quick_status()
    if cmd == "/accounts":
        return quick_accounts()
    if cmd == "/usage":
        u = agent.usage_summary(24)
        if not u.get("turns_with_usage"):
            return (f"{u.get('turns', 0)} lượt agent trong 24h. Harness hiện tại không trả số liệu token "
                    "(chỉ Claude Code có) — xem tỉ lệ cache hit trên dashboard của provider.")
        rate = f"{100*u['cache_hit_rate']:.0f}%" if u["cache_hit_rate"] is not None else "—"
        return (f"24h: {u['turns']} lượt · cache hit {rate}\n"
                f"input: {u['cache_read']:,} cache · {u['input_uncached']:,} không cache · "
                f"{u['cache_write']:,} ghi cache\noutput: {u['output']:,}"
                + (f"\nchi phí ước tính (theo harness): ${u['cost_usd']}" if u["cost_usd"] else ""))
    if cmd == "/new":
        agent.reset_session()
        return "🆕 Lượt tiếp theo sẽ mở phiên agent mới."
    if cmd == "/dropbox":
        rows = dropbox.listing()
        body = "\n".join(f"• {r['name']}: {r['n_files']} file, {r['total_mb']} MB — {r['status']}" for r in rows)
        return f"📂 {dropbox.dropbox_dir()}\n" + (body or "(trống)")
    if cmd in ("/stop", "/resume"):
        if not arg:
            return f"Cần tên dự án: {cmd} <slug>. Có: {', '.join(_projects())}"
        return set_paused(arg, cmd == "/stop")
    return None


# ---------------------------------------------------------------- tải dữ liệu từ link
NAME_RE = re.compile(r"(?:tên|ten|name)\s*[:=]\s*([\w.\-]+)", re.I)


def _fmt_bytes(n: float) -> str:
    return f"{n/1e9:.2f} GB" if n >= 1e9 else f"{n/1e6:.1f} MB"


class Downloader(threading.Thread):
    """Tải tuần tự từng yêu cầu (không chặn agent), báo tiến độ bằng cách sửa một tin nhắn."""

    def __init__(self, tg: "Telegram", agent_q: "queue.Queue[dict]"):
        super().__init__(daemon=True)
        self.tg, self.agent_q = tg, agent_q
        self.q: "queue.Queue[dict]" = queue.Queue()
        self.cancel = threading.Event()
        self.current = None

    def submit(self, urls: list[str], text: str, name: str | None):
        self.q.put({"urls": urls, "text": text, "name": name})
        if self.current or self.q.qsize() > 1:
            self.tg.send(f"📥 Đã xếp hàng tải ({self.q.qsize()} yêu cầu đang chờ).")

    def run(self):
        while True:
            job = self.q.get()
            self.current = job
            self.cancel.clear()
            try:
                self._do(job)
            except Exception:
                self.tg.send("⚠️ Lỗi khi tải:\n" + traceback.format_exc()[-1200:])
            finally:
                self.current = None

    def _do(self, job):
        cfg = load_config()
        max_gb = float(cfg.get("downloads", {}).get("max_gb", 50))
        root = dropbox.dropbox_dir(cfg)
        name = fetcher.slugify(job["name"])[:50] if job["name"] else fetcher.guess_name(job["urls"][0])
        dest = root / name
        if dest.exists() and any(dest.iterdir()) and not job["name"]:
            dest = root / f"{name}-{datetime.now():%m%d-%H%M}"
        mid = self.tg.send(f"⬇️ Bắt đầu tải {len(job['urls'])} link → dropbox/{dest.name}/")
        results, errors = [], []
        for i, url in enumerate(job["urls"], 1):
            t0, label = time.time(), f"[{i}/{len(job['urls'])}]"

            def progress(done, total, kind, label=label, t0=t0):
                speed = done / max(time.time() - t0, 1)
                pct = f" ({100*done/total:.0f}%)" if total else ""
                self.tg.edit(mid, f"⬇️ {label} {kind}: {_fmt_bytes(done)}{pct} · {_fmt_bytes(speed)}/s\n"
                                  f"→ dropbox/{dest.name}/   (/cancel để huỷ)")
            try:
                results.append(fetcher.download(url, dest, max_gb=max_gb, progress=progress, cancel=self.cancel))
            except fetcher.FetchError as e:
                errors.append(f"• {url[:80]}\n  {e}")
        if not results:
            self.tg.edit(mid, "❌ Không tải được:\n" + "\n".join(errors))
            if not any(dest.iterdir()):
                dest.rmdir()
            return
        task = job["text"].strip()
        if task and not (dest / "task.md").exists():
            (dest / "task.md").write_text(f"# Mô tả từ tin nhắn Telegram ({datetime.now():%Y-%m-%d %H:%M})\n\n"
                                          f"{task}\n\nNguồn:\n" + "\n".join(f"- {u}" for u in job["urls"]) + "\n",
                                          encoding="utf-8")
        item = dropbox.announce_now(dest)
        summary = (f"✅ Đã tải xong → dropbox/{dest.name}/: {item['n_files']} file, {item['total_mb']} MB"
                   + (f"\nĐã giải nén: {', '.join(sum((r['extracted'] for r in results), []))}"
                      if any(r["extracted"] for r in results) else ""))
        if errors:
            summary += "\n⚠️ Một số link lỗi:\n" + "\n".join(errors)
        self.tg.edit(mid, summary + "\nĐã giao cho agent.")
        prompt = dropbox.wake_message(item)
        if job["text"].strip():
            prompt += f"\n\nTin nhắn gốc của người dùng:\n{job['text'].strip()}"
        self.agent_q.put({"source": "download", "prompt": prompt})


# ---------------------------------------------------------------- vòng chính
def worker(tg: Telegram, q: "queue.Queue[dict]", busy: threading.Event):
    while True:
        try:
            item = q.get(timeout=5)
        except queue.Empty:
            item = None
        if item is None:  # không có tin người dùng → xem hàng đợi wake từ cron
            wakes = sorted(WAKE_DIR.glob("*.json")) if WAKE_DIR.exists() else []
            if not wakes:
                continue
            msgs = []
            for w in wakes:
                try:
                    msgs.append(json.loads(w.read_text())["message"])
                except Exception:
                    pass
                w.unlink(missing_ok=True)
            item = {"source": "cron", "prompt": "Sự kiện tự động:\n" + "\n".join(f"- {m}" for m in msgs)}
        busy.set()
        stop_typing = threading.Event()

        def keep_typing():
            while not stop_typing.wait(5):
                tg.typing()
        threading.Thread(target=keep_typing, daemon=True).start()
        try:
            res = agent.run_agent(item["prompt"], source=item["source"])
            prefix = "🤖 " if item["source"] == "cron" else ""
            if res["reply"]:
                tg.send(prefix + res["reply"])
        except Exception:
            tg.send("⚠️ Lỗi khi gọi agent:\n" + traceback.format_exc()[-1500:])
        finally:
            stop_typing.set()
            busy.clear()


def main():
    tg = Telegram()
    q: "queue.Queue[dict]" = queue.Queue()
    busy = threading.Event()
    threading.Thread(target=worker, args=(tg, q, busy), daemon=True).start()
    dl = Downloader(tg, q)
    dl.start()
    offset = int(OFFSET_FILE.read_text()) if OFFSET_FILE.exists() else None
    h = agent.load_harness()
    print(f"bridge chạy: harness={h['name']} chat={tg.chat}")
    tg.send(f"🟢 Agent đã khởi động (harness: {h['name']}{' · ' + h['model'] if h['model'] else ''}). /help")
    while True:
        try:
            params = {"timeout": 50, "allowed_updates": json.dumps(["message"])}
            if offset is not None:
                params["offset"] = offset
            updates = tg.call("getUpdates", **params)
        except Exception as e:
            print("getUpdates lỗi:", e)
            time.sleep(10)
            continue
        for u in updates:
            offset = u["update_id"] + 1
            OFFSET_FILE.write_text(str(offset))
            m = u.get("message") or {}
            if str(m.get("chat", {}).get("id")) != tg.chat:
                print("bỏ qua chat lạ:", m.get("chat", {}).get("id"))
                continue
            text = m.get("text") or m.get("caption") or ""
            files = []
            doc = m.get("document")
            photo = (m.get("photo") or [None])[-1]
            for f, name in ((doc, (doc or {}).get("file_name", "file")),
                            (photo, f"photo-{int(time.time())}.jpg")):
                if not f:
                    continue
                if f.get("file_size", 0) > MAX_FILE:
                    tg.send(f"⚠️ {name} lớn hơn 20 MB (giới hạn Telegram Bot). Hãy gửi link "
                            "(Google Drive công khai, Hugging Face, Dropbox, URL trực tiếp).")
                    continue
                try:
                    files.append(tg.download(f["file_id"], name))
                except Exception as e:
                    tg.send(f"⚠️ Không tải được {name}: {e}")
            low = text.strip().split(" ")[0].split("@")[0].lower() if text.strip() else ""
            if low == "/cancel" and not files:
                if dl.current:
                    dl.cancel.set()
                    tg.send("⏹ Đang huỷ lượt tải (Hugging Face / Kaggle / thư mục Drive chỉ dừng sau khi xong file hiện tại).")
                else:
                    tg.send("Không có lượt tải nào đang chạy.")
                continue
            data_links, other_links = fetcher.data_urls(text)
            if (low == "/get" or (data_links and not low.startswith("/"))) and not files:
                if not data_links:
                    tg.send("Cách dùng: /get <link> [tên-thư-mục]  (hoặc gửi link kèm mô tả bài toán)")
                    continue
                rest = fetcher.URL_RE.sub(" ", text)
                if low == "/get":
                    rest = rest.split(None, 1)[1] if len(rest.split(None, 1)) > 1 else ""
                m = NAME_RE.search(rest)
                name = m.group(1) if m else None
                if m:
                    rest = NAME_RE.sub(" ", rest)
                elif low == "/get" and rest.strip() and len(rest.split()) == 1:
                    name, rest = rest.strip(), ""
                desc = re.sub(r"\s+", " ", rest).strip()
                if other_links:
                    desc += "\nLink khác trong tin nhắn: " + " ".join(other_links)
                dl.submit(data_links, desc, name)
                continue
            if low == "/scan" and not files:
                items = dropbox.scan(force=True)
                if not items:
                    tg.send("📂 Không có dữ liệu mới đã chép xong (file sửa trong 3 phút gần nhất được coi là "
                            "đang chép). Xem /dropbox.")
                for it in items:
                    tg.send(f"📂 {it['name']}: {it['n_files']} file, {it['total_mb']} MB — đã giao cho agent.")
                    q.put({"source": "dropbox", "prompt": dropbox.wake_message(it)})
                continue
            if text.startswith("/") and not files:
                reply = handle_quick(text)
                if reply is not None:
                    tg.send(reply)
                    continue
            if not text and not files:
                continue
            prompt = text
            if files:
                prompt += "\n\nNgười dùng gửi file (đã lưu trên máy):\n" + "\n".join(f"- {p}" for p in files)
            if busy.is_set() or not q.empty():
                tg.send("📥 Đã nhận, sẽ xử lý sau khi xong việc hiện tại.")
            q.put({"source": "telegram", "prompt": prompt})


if __name__ == "__main__":
    main()
