"""Gọi harness (Claude Code / Codex / OpenCode / DeepSeek Harness / lệnh tùy chỉnh) ở chế độ headless.

Cấu hình trong harness.yaml (trình cài đặt sinh ra). Mỗi adapter khai báo:
  cmd        lệnh khi tiếp tục phiên trước     — {prompt}, {model}, {outfile}, {workdir}
  first_cmd  lệnh khi bắt đầu phiên mới (sau /new hoặc lần đầu)
  model_args chèn thêm khi có model, ví dụ ["-m", "{model}"]
  output     text | claude_json | outfile
  stateless  true → harness không nhớ phiên; bridge gửi kèm lịch sử gần đây
  env        biến môi trường thêm
Cờ dòng lệnh của các harness thay đổi theo phiên bản: nếu lệnh lỗi, sửa harness.yaml rồi
chạy `python -m kra.agent --test`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import yaml

from .config import ROOT, load_env

HARNESS_FILE = ROOT / "harness.yaml"
STATE_FILE = ROOT / ".agent_state.json"
CHAT_LOG = ROOT / ".chat_log.jsonl"
LOCK_FILE = ROOT / ".agent.lock"

# Mặc định cho từng harness. Trình cài đặt chép vào harness.yaml để người dùng chỉnh được.
DEFAULT_ADAPTERS = {
    "claude": {
        "first_cmd": ["claude", "-p", "{prompt}", "--permission-mode", "acceptEdits",
                      "--allowedTools", "Bash,Read,Edit,Write,Glob,Grep,WebFetch,WebSearch",
                      "--append-system-prompt-file", "AGENT.md", "--output-format", "json"],
        "cmd": ["claude", "-p", "{prompt}", "--continue", "--permission-mode", "acceptEdits",
                "--allowedTools", "Bash,Read,Edit,Write,Glob,Grep,WebFetch,WebSearch",
                "--append-system-prompt-file", "AGENT.md", "--output-format", "json"],
        "model_args": ["--model", "{model}"],
        "output": "claude_json",
        "stateless": False,
        # Giữ system prompt ỔN ĐỊNH giữa các lượt để provider (DeepSeek, Anthropic...) cache được tiền tố:
        #  - git status snapshot thay đổi mỗi khi file trong project đổi → bỏ
        #  - attribution block ở ĐẦU system prompt (vân tay prompt; bản cũ có token theo request) → bỏ
        #  - auto memory có thể chèn nội dung khác nhau giữa các lượt → tắt (đã có registry/NOTES.md)
        "env": {"CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS": "1", "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
                "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"},
    },
    "codex": {
        # AGENTS.md (bản sao AGENT.md) được Codex tự nạp. Sandbox workspace-write + mở mạng cho Kaggle CLI.
        "first_cmd": ["codex", "-c", "sandbox_mode=\"workspace-write\"",
                      "-c", "sandbox_workspace_write.network_access=true",
                      "-c", "approval_policy=\"never\"",
                      "exec", "--output-last-message", "{outfile}", "{prompt}"],
        "cmd": ["codex", "-c", "sandbox_mode=\"workspace-write\"",
                "-c", "sandbox_workspace_write.network_access=true",
                "-c", "approval_policy=\"never\"",
                "exec", "resume", "--last", "{prompt}"],
        "model_args": ["-m", "{model}"],
        "model_args_position": 1,   # chèn ngay sau "codex"
        "output": "outfile_or_text",
        "stateless": False,
    },
    "opencode": {
        "first_cmd": ["opencode", "run", "--auto", "{prompt}"],
        "cmd": ["opencode", "run", "--continue", "--auto", "{prompt}"],
        "model_args": ["-m", "{model}"],
        "model_args_position": 2,
        "output": "text",
        "stateless": False,
    },
    "dsh": {
        # DeepSeek Harness (developer preview): chế độ một lượt, không có cờ tiếp tục phiên.
        "first_cmd": ["npx", "-y", "@deepseek-ai/dsh", "--profile", "headless", "{prompt}"],
        "cmd": ["npx", "-y", "@deepseek-ai/dsh", "--profile", "headless", "{prompt}"],
        "output": "text",
        "stateless": True,
        "env": {"DSH_CWD": "{workdir}"},
    },
    "custom": {
        "first_cmd": ["echo", "Chưa cấu hình harness: {prompt}"],
        "cmd": ["echo", "Chưa cấu hình harness: {prompt}"],
        "output": "text",
        "stateless": True,
    },
}

# Bố cục prompt tối ưu cache tiền tố (DeepSeek cache tự động, khớp từ ĐẦU request):
#   [STATIC: không bao giờ đổi] → [lịch sử: chỉ nối thêm trong một "epoch"] → [ĐỘNG: giờ, nguồn, tin mới]
# Không đặt thời gian / số liệu thay đổi ở phần đầu.
STATIC_PREFIX = """Bạn là Kaggle Research Agent, làm việc trong thư mục {workdir} theo AGENT.md.
Câu trả lời cuối cùng của bạn được gửi NGUYÊN VĂN tới người dùng qua Telegram: viết tiếng Việt,
ngắn gọn, có số liệu; văn bản thường (không dùng bảng Markdown).
"""
STATELESS_NOTE = "Harness này không nhớ phiên: trước khi làm gì, đọc AGENT.md trong thư mục làm việc.\n"
DYNAMIC_HEAD = "\n[{now} · nguồn: {source}]\n"


def load_harness() -> dict:
    load_env()
    h = yaml.safe_load(HARNESS_FILE.read_text(encoding="utf-8")) if HARNESS_FILE.exists() else {}
    h = h or {}
    name = h.get("harness", "custom")
    adapter = dict(DEFAULT_ADAPTERS.get(name, DEFAULT_ADAPTERS["custom"]))
    adapter.update((h.get("adapters") or {}).get(name, {}))
    return {"name": name, "model": h.get("model") or "", "workdir": str(ROOT / h.get("workdir", ".")),
            "timeout_minutes": float(h.get("timeout_minutes", 45)),
            "history_turns": int(h.get("history_turns", 8)), "adapter": adapter}


def _state() -> dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}


def _save_state(s: dict) -> None:
    STATE_FILE.write_text(json.dumps(s))


def reset_session() -> None:
    s = _state()
    s["has_session"] = False
    _save_state(s)


def _render_turn(r: dict) -> str:
    # định dạng CỐ ĐỊNH (cùng độ cắt) để mỗi lượt cũ luôn ra đúng một chuỗi → tiền tố giữ nguyên
    return f"- [{r['ts']}] {r['source']}: {r['prompt'][:400]}\n  → bạn trả lời: {r['reply'][:600]}\n"


def _history(st: dict, keep: int) -> str:
    """Lịch sử cho harness không nhớ phiên, theo "epoch" chỉ-nối-thêm.

    Trong một epoch, mọi lượt cũ giữ nguyên vị trí và nội dung, nên request sau có tiền tố trùng request trước
    (cache hit). Khi quá dài (> 2×keep lượt), mở epoch mới giữ lại keep/2 lượt gần nhất (miss một lần)."""
    if not CHAT_LOG.exists() or keep <= 0:
        return ""
    lines = CHAT_LOG.read_text(encoding="utf-8").splitlines()
    start = int(st.get("epoch_start", 0))
    if start > len(lines):
        start = 0
    if len(lines) - start > 2 * keep:
        start = max(0, len(lines) - max(1, keep // 2))
        st["epoch_start"] = start
    out = []
    for l in lines[start:]:
        try:
            out.append(_render_turn(json.loads(l)))
        except Exception:
            pass
    return "Lịch sử hội thoại:\n" + "".join(out) if out else ""


def build_cmd(h: dict, prompt: str, first: bool, outfile: str) -> list[str]:
    a = h["adapter"]
    tmpl = list(a["first_cmd"] if first else a["cmd"])
    subs = {"{prompt}": prompt, "{model}": h["model"], "{outfile}": outfile, "{workdir}": h["workdir"]}
    cmd = []
    for t in tmpl:
        for k, v in subs.items():
            t = t.replace(k, v)
        cmd.append(t)
    if h["model"] and a.get("model_args"):
        extra = [x.replace("{model}", h["model"]) for x in a["model_args"]]
        pos = int(a.get("model_args_position", len(cmd)))
        cmd[pos:pos] = extra
    return cmd


def _usage_from(data: dict) -> dict | None:
    u = data.get("usage") or {}
    if not u:
        return None
    # Định dạng Anthropic (Claude Code; endpoint Anthropic của DeepSeek) hoặc OpenAI/DeepSeek gốc
    hit = u.get("cache_read_input_tokens", u.get("prompt_cache_hit_tokens", 0)) or 0
    write = u.get("cache_creation_input_tokens", 0) or 0
    miss = u.get("input_tokens", u.get("prompt_cache_miss_tokens", 0)) or 0
    return {"input_uncached": miss, "cache_read": hit, "cache_write": write,
            "output": u.get("output_tokens", u.get("completion_tokens", 0)) or 0,
            "cost_usd": data.get("total_cost_usd")}


def _parse(h: dict, proc: subprocess.CompletedProcess, outfile: str) -> tuple[str, dict | None]:
    mode = h["adapter"].get("output", "text")
    if mode == "claude_json":
        try:
            data = json.loads(proc.stdout.strip().splitlines()[-1])
            return str(data.get("result", "")).strip(), _usage_from(data)
        except Exception:
            return proc.stdout.strip()[-4000:], None
    if mode.startswith("outfile"):
        p = Path(outfile)
        if p.exists() and p.read_text(encoding="utf-8").strip():
            return p.read_text(encoding="utf-8").strip(), None
    return (proc.stdout or "").strip()[-4000:], None


def usage_summary(hours: float = 24) -> dict:
    """Thống kê token & tỉ lệ cache hit của các lượt gần đây (hiện có với Claude Code)."""
    if not CHAT_LOG.exists():
        return {"turns": 0}
    cutoff = datetime.now().timestamp() - hours * 3600
    tot = {"turns": 0, "turns_with_usage": 0, "input_uncached": 0, "cache_read": 0, "cache_write": 0,
           "output": 0, "cost_usd": 0.0}
    for l in CHAT_LOG.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(l)
        except Exception:
            continue
        if datetime.fromisoformat(r["ts"]).timestamp() < cutoff:
            continue
        tot["turns"] += 1
        u = r.get("usage")
        if u:
            tot["turns_with_usage"] += 1
            for k in ("input_uncached", "cache_read", "cache_write", "output"):
                tot[k] += u.get(k) or 0
            tot["cost_usd"] += u.get("cost_usd") or 0
    inp = tot["input_uncached"] + tot["cache_read"] + tot["cache_write"]
    tot["cache_hit_rate"] = round(tot["cache_read"] / inp, 3) if inp else None
    tot["cost_usd"] = round(tot["cost_usd"], 4)
    return tot


class _Lock:
    def __enter__(self):
        self.f = open(LOCK_FILE, "w")
        try:
            import fcntl
            fcntl.flock(self.f, fcntl.LOCK_EX)
        except ImportError:  # Windows: bỏ qua khoá
            pass
        return self

    def __exit__(self, *exc):
        self.f.close()


def run_agent(prompt: str, source: str = "telegram") -> dict:
    h = load_harness()
    a = h["adapter"]
    with _Lock():
        st = _state()
        first = a.get("stateless") or not st.get("has_session")
        full = STATIC_PREFIX.format(workdir=h["workdir"])
        if a.get("stateless"):
            full += STATELESS_NOTE + _history(st, h["history_turns"])
        full += DYNAMIC_HEAD.format(now=datetime.now().strftime("%Y-%m-%d %H:%M"), source=source) + prompt
        env = os.environ.copy()
        env.update({k: str(v).replace("{workdir}", h["workdir"]) for k, v in (a.get("env") or {}).items()})
        t0 = time.time()
        with tempfile.NamedTemporaryFile(prefix="kra_reply_", suffix=".txt", delete=False) as tf:
            outfile = tf.name

        def go(is_first):
            return subprocess.run(build_cmd(h, full, is_first, outfile), cwd=h["workdir"], env=env,
                                  capture_output=True, text=True, timeout=h["timeout_minutes"] * 60)
        try:
            proc = go(first)
            if proc.returncode != 0 and not first:  # chưa có phiên để tiếp tục → mở phiên mới
                proc = go(True)
                first = True
            reply, usage = _parse(h, proc, outfile)
            ok = proc.returncode == 0
            if not ok and not reply:
                reply = f"⚠️ Harness {h['name']} lỗi (mã {proc.returncode}):\n{(proc.stderr or '')[-1500:]}"
        except subprocess.TimeoutExpired:
            usage = None
            ok, reply = False, f"⚠️ Harness chạy quá {h['timeout_minutes']:g} phút, đã dừng lượt này."
        except FileNotFoundError as e:
            usage = None
            ok, reply = False, f"⚠️ Không tìm thấy lệnh harness: {e}. Kiểm tra harness.yaml / PATH."
        finally:
            Path(outfile).unlink(missing_ok=True)
        if ok:
            st["has_session"] = True
            _save_state(st)
        with open(CHAT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), "source": source,
                                "prompt": prompt, "reply": reply, "ok": ok, "usage": usage,
                                "seconds": round(time.time() - t0, 1)}, ensure_ascii=False) + "\n")
        _save_state(st)  # lưu epoch_start
    return {"ok": ok, "reply": reply, "harness": h["name"], "new_session": first, "usage": usage}


def main():
    ap = argparse.ArgumentParser(description="Gọi harness đã cấu hình")
    ap.add_argument("prompt", nargs="?")
    ap.add_argument("--test", action="store_true", help="kiểm tra harness trả lời được")
    ap.add_argument("--new", action="store_true", help="bắt đầu phiên mới")
    ap.add_argument("--show-cmd", action="store_true")
    a = ap.parse_args()
    if a.new:
        reset_session()
    if a.show_cmd:
        h = load_harness()
        print(json.dumps({"first": build_cmd(h, "<prompt>", True, "<outfile>"),
                          "continue": build_cmd(h, "<prompt>", False, "<outfile>")}, indent=2, ensure_ascii=False))
        return
    prompt = a.prompt or ("Kiểm tra cài đặt: chạy `python -m kra accounts` rồi trả lời một dòng: "
                          "harness hoạt động, có bao nhiêu tài khoản Kaggle.") if a.test else a.prompt
    if not prompt:
        ap.error("cần prompt hoặc --test")
    print(json.dumps(run_agent(prompt, source="cli-test" if a.test else "cli"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
