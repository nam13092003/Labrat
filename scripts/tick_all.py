"""Chạy định kỳ bằng cron trên VPS: poll mọi cuộc thi, gửi thông báo, đánh thức agent khi cần.

Đánh thức agent khi:
  - có thí nghiệm vừa kết thúc (thành công / lỗi / smoke xong), hoặc
  - còn tài khoản GPU rảnh, chưa dừng, và đã quá KRA_IDLE_HOURS (mặc định 3) kể từ lần đánh thức trước
    (phòng khi agent quên đẩy thí nghiệm tiếp theo).
Mặc định "đánh thức" = thả vào .wake_queue/ để bridge Telegram gọi agent và gửi trả lời.
Có thể thay bằng lệnh riêng qua biến KRA_WAKE_CMD (thông điệp ở biến KRA_WAKE_MSG).
Cũng quét thư mục dropbox: dữ liệu mới/cập nhật đã chép xong → báo Telegram + giao agent.
Dự án `paused: true` vẫn được thu kết quả nhưng không bị đánh thức để đẩy thí nghiệm mới.
Bỏ qua dự án chưa có thí nghiệm nào và thư mục bắt đầu bằng "_".

Cron ví dụ (mỗi 15 phút):
  */15 * * * * /home/kra/kaggle-research-agent/.venv/bin/python /home/kra/kaggle-research-agent/scripts/tick_all.py >> /home/kra/kra-tick.log 2>&1
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

from kra import budget, dropbox, experiments, telegram  # noqa: E402
from kra.cli import _notify_events  # noqa: E402
from kra.config import competitions_root, load_config  # noqa: E402

IDLE_H = float(os.environ.get("KRA_IDLE_HOURS", "3"))
STATE = ROOT / ".tick_state.json"


def enqueue(key: str, msg: str) -> None:
    cmd = os.environ.get("KRA_WAKE_CMD")
    if not cmd:  # mặc định: thả vào hàng đợi để bridge Telegram gọi agent
        wq = ROOT / ".wake_queue"
        wq.mkdir(exist_ok=True)
        (wq / f"{int(time.time() * 1000)}-{key}.json").write_text(json.dumps({"slug": key, "message": msg},
                                                                           ensure_ascii=False))
        print(f"  wake → hàng đợi: {msg[:160]}")
        return
    r = subprocess.run(cmd, shell=True, env=dict(os.environ, KRA_WAKE_MSG=msg, KRA_SLUG=key),
                       capture_output=True, text=True, timeout=300)
    print(f"  wake rc={r.returncode} {(r.stdout + r.stderr)[-300:]}")


def wake(slug: str, reason: str, state: dict) -> None:
    state[slug] = time.time()
    enqueue(slug, f"tick {slug}: {reason}. Đọc `python -m kra tick {slug}` / registry rồi làm bước tiếp theo "
                  f"theo AGENT.md.")


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


def main():
    base = load_config()
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] tick")
    scan_dropbox()
    for reg_path in sorted(competitions_root(base).glob("*/registry.json")):
        slug = reg_path.parent.name
        if slug.startswith("_"):
            continue
        cfg = load_config(slug)
        try:  # dự án tạm dừng vẫn được thu kết quả kernel đang chạy
            events = experiments.poll(cfg)
        except Exception as e:  # mạng lỗi, token hết hạn...
            print(f"  {slug}: poll lỗi: {e}")
            continue
        _notify_events(cfg, events)
        reg = experiments.registry_for(cfg)
        if not reg.experiments:
            continue
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
    STATE.write_text(json.dumps(state))


if __name__ == "__main__":
    main()
