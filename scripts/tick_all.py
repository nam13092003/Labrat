"""Chạy định kỳ bằng cron trên VPS: poll mọi cuộc thi, gửi thông báo, đánh thức agent khi cần.

Đánh thức agent khi:
  - có thí nghiệm vừa kết thúc (thành công / lỗi / smoke xong), hoặc
  - còn tài khoản GPU rảnh, chưa dừng, và đã quá LABRAT_IDLE_HOURS (mặc định 3) kể từ lần đánh thức trước
    (phòng khi agent quên đẩy thí nghiệm tiếp theo).
Mặc định "đánh thức" = thả vào .wake_queue/ để bridge Telegram gọi agent và gửi trả lời.
Có thể thay bằng lệnh riêng qua biến LABRAT_WAKE_CMD (thông điệp ở biến LABRAT_WAKE_MSG).
Ngoài ra, PHIÊN LÀM VIỆC ĐỊNH KỲ (config `autowork.minutes`, mặc định 60): mỗi dự án đang hoạt động (chưa tạm dừng,
chưa hết ngân sách) được đánh thức tối thiểu mỗi N phút để tự tiếp tục nghiên cứu — kể cả khi mọi GPU đang bận
(dùng thời gian cho phân tích lỗi, đọc tài liệu, chuẩn bị thí nghiệm kế tiếp). Không có gì đáng báo → agent trả lời
bắt đầu bằng "[im lặng]" và bridge không gửi Telegram.
Cũng quét thư mục dropbox: dữ liệu mới/cập nhật đã chép xong → báo Telegram + giao agent.
Dự án `paused: true` vẫn được thu kết quả nhưng không bị đánh thức để đẩy thí nghiệm mới.
Bỏ qua dự án chưa có thí nghiệm nào và thư mục bắt đầu bằng "_".

Bridge Telegram tự chạy script này mỗi `tick.interval_minutes` (mặc định 15) — KHÔNG cần cron. Cron vẫn dùng được
song song (khoá .tick.lock đảm bảo không chạy chồng; chạy lặp cũng không báo trùng vì poll chỉ báo kernel mới xong).
Cron ví dụ (mỗi 15 phút):
  */15 * * * * /home/labrat/Labrat/.venv/bin/python /home/labrat/Labrat/scripts/tick_all.py >> /home/labrat/labrat-tick.log 2>&1
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labrat import budget, dropbox, experiments, housekeeping, telegram  # noqa: E402
from labrat.cli import _notify_events  # noqa: E402
from labrat.config import competitions_root, load_config  # noqa: E402

IDLE_H = float(os.environ.get("LABRAT_IDLE_HOURS", "3"))
QUIET = "[im lặng]"
STATE = ROOT / ".tick_state.json"


def enqueue(key: str, msg: str) -> None:
    cmd = os.environ.get("LABRAT_WAKE_CMD")
    if not cmd:  # mặc định: thả vào hàng đợi để bridge Telegram gọi agent
        wq = ROOT / ".wake_queue"
        wq.mkdir(exist_ok=True)
        (wq / f"{int(time.time() * 1000)}-{key}.json").write_text(json.dumps({"slug": key, "message": msg},
                                                                           ensure_ascii=False))
        print(f"  wake → hàng đợi: {msg[:160]}")
        return
    r = subprocess.run(cmd, shell=True, env=dict(os.environ, LABRAT_WAKE_MSG=msg, LABRAT_SLUG=key),
                       capture_output=True, text=True, timeout=300)
    print(f"  wake rc={r.returncode} {(r.stdout + r.stderr)[-300:]}")


def wake(slug: str, reason: str, state: dict) -> None:
    state[slug] = time.time()
    enqueue(slug, f"tick {slug}: {reason}. Đọc `python -m labrat tick {slug}` / registry rồi làm bước tiếp theo "
                  f"theo AGENT.md.")


def pending_wakes() -> set[str]:
    wq = ROOT / ".wake_queue"
    out = set()
    for f in (wq.glob("*.json") if wq.exists() else []):
        try:
            out.add(json.loads(f.read_text())["slug"])
        except Exception:
            pass
    return out


def autowork_due(base: dict, cfg: dict, slug: str, state: dict, chk: dict | None, pending: set[str],
                 now: float | None = None) -> bool:
    """Đến giờ phiên làm việc định kỳ của dự án chưa?"""
    minutes = float((base.get("autowork") or {}).get("minutes", 60) or 0)
    if minutes <= 0 or cfg.get("paused") or slug in pending:
        return False
    if chk and chk.get("stop"):
        return False
    now = now or time.time()
    last = max(state.get(slug, 0), state.get("auto:" + slug, 0))
    return now - last >= minutes * 60


def autowork_message(slug: str, chk: dict | None) -> str:
    gpu = ""
    if chk:
        free = chk["budget"]["free_gpu_slots"]
        gpu = (f" Hiện có {free} tài khoản GPU rảnh." if chk.get("can_push_gpu_now") else
               " Hiện chưa đẩy được GPU (" + ("; ".join(chk.get("reasons") or []) or "mọi tài khoản bận") + ").")
    return (f"PHIÊN LÀM VIỆC ĐỊNH KỲ cho dự án {slug}.{gpu} Tự tiếp tục nghiên cứu theo AGENT.md: thu & phân tích kết "
            f"quả mới, cập nhật plan.md, chuẩn bị (smoke) và đẩy thí nghiệm tiếp nếu có tài khoản rảnh; nếu GPU đều bận "
            f"thì dùng lượt này cho phân tích lỗi, EDA, đọc tài liệu, chuẩn bị thí nghiệm kế tiếp. Trả lời 1–3 dòng về "
            f"việc đã làm; nếu không có gì đáng báo, trả lời bắt đầu bằng \"{QUIET}\" (sẽ không gửi Telegram).")


def scan_dropbox() -> None:
    try:
        items = dropbox.scan()
    except Exception as e:
        print(f"  dropbox lỗi: {e}")
        return
    for it in items:
        label = "cập nhật" if it["kind"] == "update" else "mới"
        telegram.send_message(f"📂 Dữ liệu {label}: <b>{telegram.esc(it['name'])}</b> "
                              f"({it['n_files']} file, {it['total_mb']} MB) — đã giao cho agent.", load_config())
        enqueue("dropbox-" + it["name"], dropbox.wake_message(it))


def check_disk(cfg: dict, state: dict) -> None:
    """Ổ sắp đầy → tự dọn phần an toàn; vẫn thiếu → cảnh báo Telegram (tối đa 6 giờ/lần)."""
    dcfg = cfg.get("disk", {})
    d = housekeeping.disk()
    if d["free_gb"] < float(dcfg.get("auto_clean_free_gb", 5)):
        r = housekeeping.clean(cfg)
        print(f"  disk: còn {d['free_gb']} GB → tự dọn {r['freed_gb']} GB")
        if r["freed_gb"] >= 0.5:
            telegram.send_message(f"🧹 Ổ đĩa còn {d['free_gb']} GB → đã tự dọn {r['freed_gb']} GB "
                                  f"(output lớn/file tạm). Còn {r['disk']['free_gb']} GB.", cfg)
        d = r["disk"]
    if d["free_gb"] < float(dcfg.get("warn_free_gb", 5)) and time.time() - state.get("_disk_warn", 0) > 6 * 3600:
        state["_disk_warn"] = time.time()
        rows = "\n".join(f"• {x['gb']} GB {telegram.esc(x['path'])}" for x in housekeeping.big_dirs(6))
        telegram.send_message(f"⚠️ <b>Ổ đĩa VPS sắp đầy</b>: còn {d['free_gb']} GB / {d['total_gb']} GB. "
                              f"Bot sẽ ngừng hoạt động khi đầy.\nLớn nhất:\n{rows}", cfg)


def main():
    base = load_config()
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] tick")
    check_disk(base, state)
    scan_dropbox()
    pending = pending_wakes()
    for proj in sorted(competitions_root(base).glob("*/project.yaml")):
        slug = proj.parent.name
        if slug.startswith("_"):
            continue
        cfg = load_config(slug)
        reg = experiments.registry_for(cfg)
        if not reg.experiments:  # dự án mới / vừa reset: chỉ có phiên làm việc định kỳ
            if autowork_due(base, cfg, slug, state, None, pending):
                state["auto:" + slug] = time.time()
                enqueue(slug, autowork_message(slug, None))
            continue
        try:  # dự án tạm dừng vẫn được thu kết quả kernel đang chạy
            events = experiments.poll(cfg)
        except Exception as e:  # mạng lỗi, token hết hạn...
            print(f"  {slug}: poll lỗi: {e}")
            continue
        _notify_events(cfg, events)
        reg = experiments.registry_for(cfg)
        finished = [ev for ev in events if ev["status"] not in ("queued", "running", "unknown")]
        chk = budget.stop_check(cfg, reg)
        print(f"  {slug}: active={[e['id'] for e in reg.active()]} finished={[ev['exp'] for ev in finished]} "
              f"stop={chk['stop']} pause={chk['pause_until_reset']} paused={bool(cfg.get('paused'))}")
        if finished:
            wake(slug, "có thí nghiệm vừa kết thúc: " + ", ".join(f"{ev['exp']}={ev['status']}" for ev in finished)
                 + (" (dự án đang TẠM DỪNG: chỉ ghi nhận kết quả, không đẩy thí nghiệm mới)"
                    if cfg.get("paused") else ""), state)
        elif (not cfg.get("paused") and chk["can_push_gpu_now"]
              and time.time() - state.get(slug, 0) > IDLE_H * 3600):
            free = chk["budget"]["free_gpu_slots"]
            wake(slug, f"còn {free} tài khoản GPU rảnh và đã {IDLE_H:g} giờ không có sự kiện", state)
        elif autowork_due(base, cfg, slug, state, chk, pending):
            state["auto:" + slug] = time.time()
            enqueue(slug, autowork_message(slug, chk))
    STATE.write_text(json.dumps(state))


def run_locked() -> None:
    import fcntl
    lock = open(ROOT / ".tick.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"[{datetime.now():%Y-%m-%d %H:%M}] tick khác đang chạy — bỏ qua")
        return
    try:
        main()
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    run_locked()
