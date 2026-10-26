#!/usr/bin/env python3
"""Trình cài đặt Kaggle Research Agent (chạy trên VPS Linux, bằng user thường — không cần root).

    python3 install.py              cài mới hoặc cấu hình lại (giữ giá trị cũ làm mặc định)
    python3 install.py --only kaggle|telegram|harness|services

Các bước: venv + thư viện → chọn harness + API provider → tài khoản Kaggle → Telegram
→ sinh harness.yaml, .env, file dịch vụ systemd, cron → kiểm tra.
"""
import getpass
import json
import os
import shlex
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VPY = VENV / "bin" / "python"
ENV_FILE = ROOT / ".env"
NPM_PREFIX = Path.home() / ".local"

HARNESSES = {
    "1": ("claude", "Claude Code", "@anthropic-ai/claude-code", "claude"),
    "2": ("codex", "OpenAI Codex CLI", "@openai/codex", "codex"),
    "3": ("opencode", "OpenCode", "opencode-ai", "opencode"),
    "4": ("dsh", "DeepSeek Harness (developer preview, chạy qua npx)", None, "npx"),
    "5": ("custom", "Lệnh tùy chỉnh", None, None),
}


# ---------------------------------------------------------------- tiện ích
def say(msg=""):
    print(msg, flush=True)


def head(msg):
    say(f"\n\033[1m== {msg} ==\033[0m")


def ask(prompt, default=None, secret=False, choices=None):
    while True:
        show = f" [{'đã có' if secret and default else default}]" if default else ""
        q = f"{prompt}{show}: "
        v = getpass.getpass(q) if secret else input(q)
        v = v.strip() or (default or "")
        if choices and v not in choices:
            say(f"  Chọn một trong: {', '.join(choices)}")
            continue
        if v or default == "":
            return v
        say("  Không được để trống.")


def yes(prompt, default=True):
    v = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    return default if not v else v.startswith("y")


def read_env():
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def write_env(env: dict):
    groups = [("Kaggle", "KAGGLE_"), ("Telegram", "TELEGRAM_"), ("Harness / API provider", "")]
    used, lines = set(), ["# Sinh bởi install.py — chứa bí mật, KHÔNG chia sẻ / commit."]
    for title, pre in groups:
        keys = [k for k in env if k not in used and (k.startswith(pre) if pre else True)]
        if not keys:
            continue
        lines.append(f"\n# {title}")
        for k in keys:
            lines.append(f"{k}={env[k]}")
            used.add(k)
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(ENV_FILE, 0o600)


def run(cmd, check=True, **kw):
    say(f"  $ {cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))}")
    return subprocess.run(cmd, check=check, shell=isinstance(cmd, str), **kw)


# ---------------------------------------------------------------- 0. venv
def ensure_venv():
    if Path(sys.prefix).resolve() == VENV.resolve():
        return
    head("Môi trường Python")
    if not VPY.exists():
        run([sys.executable, "-m", "venv", str(VENV)])
    run([str(VPY), "-m", "pip", "install", "-q", "--upgrade", "pip"])
    run([str(VPY), "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")])
    os.execv(str(VPY), [str(VPY), __file__, *sys.argv[1:]])


# ---------------------------------------------------------------- 1. harness + provider
def ensure_node():
    if shutil.which("node") and shutil.which("npm"):
        return True
    say("  Cần Node.js ≥ 18 (cho Claude Code / Codex / OpenCode / DeepSeek Harness).")
    if yes("  Cài Node.js 22 qua NodeSource (cần sudo)?"):
        run("curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && sudo apt-get install -y nodejs")
    return bool(shutil.which("node"))


def npm_install(pkg):
    env = dict(os.environ, PATH=f"{NPM_PREFIX}/bin:{os.environ['PATH']}")
    run(["npm", "install", "-g", "--prefix", str(NPM_PREFIX), pkg], env=env)


def configure_harness(env: dict) -> dict:
    head("Chọn harness")
    for k, (_, label, _, _) in HARNESSES.items():
        say(f"  {k}. {label}")
    cur = env.get("KRA_HARNESS")
    cur_key = next((k for k, v in HARNESSES.items() if v[0] == cur), "1")
    name, label, pkg, binary = HARNESSES[ask("Harness", cur_key, choices=list(HARNESSES))]
    env["KRA_HARNESS"] = name
    h = {"harness": name, "workdir": ".", "timeout_minutes": 45, "history_turns": 8, "model": ""}

    if name != "custom" and ensure_node():
        found = shutil.which(binary) or (NPM_PREFIX / "bin" / binary).exists()
        if pkg and (not found or yes(f"  Cập nhật {pkg} lên bản mới nhất?", False)):
            npm_install(pkg)

    head(f"API provider cho {label}")
    if name == "claude":
        say("  1. Anthropic API key\n  2. DeepSeek (endpoint tương thích Anthropic)\n"
            "  3. Gateway tương thích Anthropic khác (OpenRouter, LiteLLM, ...)\n"
            "  4. Đăng nhập tài khoản Claude (tự chạy `claude` một lần để đăng nhập)")
        c = ask("Provider", env.get("KRA_PROVIDER_CHOICE", "1"), choices=["1", "2", "3", "4"])
        env["KRA_PROVIDER_CHOICE"] = c
        anth = ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_MODEL",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
                "CLAUDE_CODE_SUBAGENT_MODEL")
        keep = {"1": ("ANTHROPIC_API_KEY",), "2": anth[1:], "3": anth[1:4], "4": ()}[c]
        for k in anth:
            if k not in keep:
                env.pop(k, None)
        if c == "1":
            env["ANTHROPIC_API_KEY"] = ask("ANTHROPIC_API_KEY", env.get("ANTHROPIC_API_KEY"), secret=True)
            h["model"] = ask("Model (Enter = mặc định của Claude Code)", "")
        elif c == "2":
            env["ANTHROPIC_BASE_URL"] = "https://api.deepseek.com/anthropic"
            env["ANTHROPIC_AUTH_TOKEN"] = ask("DeepSeek API key", env.get("ANTHROPIC_AUTH_TOKEN"), secret=True)
            big = ask("Model chính", env.get("ANTHROPIC_MODEL", "deepseek-v4-pro"))
            small = ask("Model nhỏ/nhanh", env.get("ANTHROPIC_DEFAULT_HAIKU_MODEL", "deepseek-flash"))
            env.update(ANTHROPIC_MODEL=big, ANTHROPIC_DEFAULT_OPUS_MODEL=big, ANTHROPIC_DEFAULT_SONNET_MODEL=big,
                       ANTHROPIC_DEFAULT_HAIKU_MODEL=small, CLAUDE_CODE_SUBAGENT_MODEL=small)
        elif c == "3":
            env["ANTHROPIC_BASE_URL"] = ask("Base URL (tương thích Anthropic Messages API)",
                                            env.get("ANTHROPIC_BASE_URL"))
            env["ANTHROPIC_AUTH_TOKEN"] = ask("Token", env.get("ANTHROPIC_AUTH_TOKEN"), secret=True)
            env["ANTHROPIC_MODEL"] = ask("Tên model", env.get("ANTHROPIC_MODEL"))
        else:
            say("  → Sau khi cài xong, chạy `claude` một lần trong terminal để đăng nhập.")

    elif name == "codex":
        say("  1. OpenAI API key\n  2. Đăng nhập ChatGPT (tự chạy `codex login --device-auth`)\n"
            "  3. Provider tương thích OpenAI khác (DeepSeek, OpenRouter, ...)")
        c = ask("Provider", env.get("KRA_PROVIDER_CHOICE", "1"), choices=["1", "2", "3"])
        env["KRA_PROVIDER_CHOICE"] = c
        if c == "1":
            env["OPENAI_API_KEY"] = ask("OPENAI_API_KEY", env.get("OPENAI_API_KEY"), secret=True)
            h["model"] = ask("Model (Enter = mặc định)", "")
            codex = shutil.which("codex") or str(NPM_PREFIX / "bin" / "codex")
            try:  # lưu key vào ~/.codex để exec dùng được
                subprocess.run([codex, "login", "--with-api-key"], input=env["OPENAI_API_KEY"], text=True)
            except FileNotFoundError:
                say("  Chưa thấy lệnh codex; sau khi cài, chạy: printenv OPENAI_API_KEY | codex login --with-api-key")
        elif c == "2":
            say("  → Sau khi cài xong, chạy `codex login --device-auth` trong terminal.")
        else:
            pid = ask("Mã provider (chữ thường, vd deepseek)", env.get("KRA_CODEX_PROVIDER", "deepseek"))
            base = ask("Base URL", env.get("KRA_CODEX_BASE_URL", "https://api.deepseek.com/v1"))
            key_env = f"{pid.upper()}_API_KEY"
            env[key_env] = ask(key_env, env.get(key_env), secret=True)
            wire = ask("wire_api (responses | chat — tùy provider hỗ trợ)", env.get("KRA_CODEX_WIRE", "responses"))
            h["model"] = ask("Tên model", env.get("KRA_CODEX_MODEL", "deepseek-v4-pro"))
            env.update(KRA_CODEX_PROVIDER=pid, KRA_CODEX_BASE_URL=base, KRA_CODEX_WIRE=wire,
                       KRA_CODEX_MODEL=h["model"])
            cdir = Path.home() / ".codex"
            cdir.mkdir(exist_ok=True)
            cfgp = cdir / "config.toml"
            block = (f'\n[model_providers.{pid}]\nname = "{pid}"\nbase_url = "{base}"\n'
                     f'env_key = "{key_env}"\nwire_api = "{wire}"\n')
            text = cfgp.read_text() if cfgp.exists() else ""
            if f"[model_providers.{pid}]" not in text:
                text = f'model_provider = "{pid}"\nmodel = "{h["model"]}"\n' + text + block
                cfgp.write_text(text)
                say(f"  Đã ghi {cfgp}")
            else:
                say(f"  {cfgp} đã có provider {pid} — giữ nguyên.")

    elif name == "opencode":
        say("  OpenCode dùng key từ biến môi trường của provider (hoặc `opencode auth login`).")
        prov = ask("Provider (anthropic | openai | deepseek | openrouter | google | ...)",
                   env.get("KRA_OPENCODE_PROVIDER", "deepseek"))
        key_env = {"google": "GOOGLE_GENERATIVE_AI_API_KEY"}.get(prov, f"{prov.upper()}_API_KEY")
        key_env = ask("Tên biến chứa API key", key_env)
        env[key_env] = ask(key_env, env.get(key_env), secret=True)
        h["model"] = ask("Model dạng provider/model (xem `opencode models`)", env.get("KRA_OPENCODE_MODEL",
                                                                                     f"{prov}/"))
        env.update(KRA_OPENCODE_PROVIDER=prov, KRA_OPENCODE_MODEL=h["model"])

    elif name == "dsh":
        env["DEEPSEEK_API_KEY"] = ask("DEEPSEEK_API_KEY", env.get("DEEPSEEK_API_KEY"), secret=True)
        b = ask("DEEPSEEK_BASE_URL (Enter = mặc định)", env.get("DEEPSEEK_BASE_URL", ""))
        if b:
            env["DEEPSEEK_BASE_URL"] = b
        say("  Lưu ý: DeepSeek Harness đang ở bản developer preview; chế độ headless không nhớ phiên,\n"
            "  bridge sẽ gửi kèm lịch sử hội thoại gần đây ở mỗi lượt.")

    else:
        say("  Nhập lệnh gọi harness một lượt, dùng {prompt} làm chỗ chèn tin nhắn.\n"
            "  Ví dụ: aider --yes --message {prompt}")
        cmdline = ask("Lệnh")  # lưu trong harness.yaml, không lưu .env (có khoảng trắng)
        parts = shlex.split(cmdline)
        remembers = yes("  Harness này có tự nhớ phiên giữa các lần gọi không?", False)
        h["adapters"] = {"custom": {"first_cmd": parts, "cmd": parts, "output": "text",
                                    "stateless": not remembers}}
    h["timeout_minutes"] = float(ask("Thời gian tối đa một lượt agent (phút)", str(h["timeout_minutes"])))
    return h


def write_harness(h: dict):
    import yaml
    sys.path.insert(0, str(ROOT))
    from kra.agent import DEFAULT_ADAPTERS
    adapters = h.pop("adapters", None) or {h["harness"]: DEFAULT_ADAPTERS[h["harness"]]}
    h["adapters"] = adapters
    (ROOT / "harness.yaml").write_text(
        "# Sinh bởi install.py. Sửa cmd/first_cmd nếu phiên bản harness đổi cờ dòng lệnh,\n"
        "# rồi kiểm tra: .venv/bin/python -m kra.agent --test\n"
        + yaml.safe_dump(h, allow_unicode=True, sort_keys=False), encoding="utf-8")
    # Codex / OpenCode tự nạp AGENTS.md ở thư mục gốc
    if h["harness"] in ("codex", "opencode", "custom", "dsh"):
        link = ROOT / "AGENTS.md"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to("AGENT.md")
    # Codex yêu cầu thư mục là git repo; harness khác không cần (git status đổi liên tục, không có ích)
    if h["harness"] == "codex" and not (ROOT / ".git").exists() and shutil.which("git"):
        run(["git", "init", "-q", str(ROOT)], check=False)
    say("  Đã ghi harness.yaml")


# ---------------------------------------------------------------- 2. Kaggle
def configure_kaggle(env: dict):
    head("Tài khoản Kaggle")
    say("  Mỗi tài khoản: kaggle.com → Settings → API → Generate New Token (dạng KGAT_...).\n"
        "  Mỗi tài khoản chạy 1 kernel GPU cùng lúc và có quota ~30h/tuần riêng.\n"
        "  Nhớ: với mỗi cuộc thi, MỌI tài khoản phải bấm chấp nhận luật trên web.")
    old = [n for n in env.get("KAGGLE_ACCOUNTS", "").split(",") if n]
    n = int(ask("Số tài khoản", str(len(old) or 1), choices=[str(i) for i in range(1, 9)]))
    names = []
    for i in range(n):
        default = old[i] if i < len(old) else f"acc{i + 1}"
        name = ask(f"\n  Tên gọi tài khoản #{i + 1}", default).lower().replace(" ", "")
        p = f"KAGGLE_{name.upper()}_"
        env[p + "USERNAME"] = ask("    Username Kaggle", env.get(p + "USERNAME"))
        env[p + "TOKEN"] = ask("    API token", env.get(p + "TOKEN"), secret=True)
        names.append(name)
    for o in old:  # xoá tài khoản bị bỏ
        if o not in names:
            for suffix in ("USERNAME", "TOKEN", "KEY", "WEEKLY_HOURS"):
                env.pop(f"KAGGLE_{o.upper()}_{suffix}", None)
    env["KAGGLE_ACCOUNTS"] = ",".join(names)


# ---------------------------------------------------------------- 3. Telegram
def configure_telegram(env: dict):
    head("Telegram")
    say("  Tạo bot: nhắn @BotFather → /newbot → lấy token.")
    env["TELEGRAM_BOT_TOKEN"] = ask("Bot token", env.get("TELEGRAM_BOT_TOKEN"), secret=True)
    if env.get("TELEGRAM_CHAT_ID") and yes(f"  Giữ chat id {env['TELEGRAM_CHAT_ID']}?", True):
        return
    input("  Mở Telegram, nhắn bất kỳ tin nào cho bot của bạn, rồi nhấn Enter...")
    try:
        with urllib.request.urlopen(f"https://api.telegram.org/bot{env['TELEGRAM_BOT_TOKEN']}/getUpdates",
                                    timeout=30) as r:
            data = json.load(r)
        chats = {str(u["message"]["chat"]["id"]): u["message"]["chat"].get("first_name") or
                 u["message"]["chat"].get("title") for u in data.get("result", []) if "message" in u}
    except Exception as e:
        say(f"  Không gọi được Telegram: {e}")
        chats = {}
    if chats:
        for cid, nm in chats.items():
            say(f"    {cid}  ({nm})")
        env["TELEGRAM_CHAT_ID"] = ask("Chat id của bạn", next(iter(chats)))
    else:
        env["TELEGRAM_CHAT_ID"] = ask("Không tự tìm được — nhập chat id", env.get("TELEGRAM_CHAT_ID"))
    say("  Bridge chỉ nhận lệnh từ đúng chat id này.")


# ---------------------------------------------------------------- 4. dịch vụ
def write_services():
    head("Dịch vụ nền (systemd + cron)")
    user = getpass.getuser()
    deploy = ROOT / "deploy"
    deploy.mkdir(exist_ok=True)
    path = f"{VENV}/bin:{NPM_PREFIX}/bin:/usr/local/bin:/usr/bin:/bin"
    unit = f"""[Unit]
Description=Kaggle Research Agent — Telegram bridge
After=network-online.target
Wants=network-online.target

[Service]
User={user}
WorkingDirectory={ROOT}
EnvironmentFile={ENV_FILE}
Environment=PATH={path}
Environment=HOME={Path.home()}
ExecStart={VPY} -m kra.bridge
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    (deploy / "kra-bridge.service").write_text(unit)
    cron = (f"*/15 * * * * cd {ROOT} && set -a && . ./.env && set +a && PATH={path} "
            f"{VPY} scripts/tick_all.py >> {Path.home()}/kra-tick.log 2>&1")
    (deploy / "crontab.txt").write_text(cron + "\n")
    say(f"  Đã ghi {deploy}/kra-bridge.service và crontab.txt")

    if yes("  Cài service kra-bridge vào systemd ngay (cần sudo)?"):
        run(f"sudo cp {deploy}/kra-bridge.service /etc/systemd/system/ && sudo systemctl daemon-reload "
            f"&& sudo systemctl enable kra-bridge && sudo systemctl restart kra-bridge", check=False)
    if yes("  Thêm lịch cron mỗi 15 phút?"):
        cur = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
        if "scripts/tick_all.py" in cur:
            cur = "\n".join(l for l in cur.splitlines() if "scripts/tick_all.py" not in l) + "\n"
        subprocess.run(["crontab", "-"], input=cur.rstrip("\n") + ("\n" if cur.strip() else "") + cron + "\n",
                       text=True, check=False)
        say("  Đã cập nhật crontab.")


# ---------------------------------------------------------------- 5. kiểm tra
def verify(env):
    head("Kiểm tra")
    e = dict(os.environ, **env, PATH=f"{VENV}/bin:{NPM_PREFIX}/bin:{os.environ['PATH']}")
    run([str(VPY), "-m", "kra", "accounts", "--check"], check=False, env=e, cwd=ROOT)
    if yes("  Gửi tin thử qua Telegram?"):
        run([str(VPY), "-c", "from kra import telegram; print(telegram.send_message('✅ Cài đặt kra xong'))"],
            check=False, env=e, cwd=ROOT)
    if yes("  Chạy thử một lượt harness (tốn một ít token)?"):
        run([str(VPY), "-m", "kra.agent", "--test", "--new"], check=False, env=e, cwd=ROOT)


def main():
    ensure_venv()
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    env = read_env()
    say("Kaggle Research Agent — cài đặt. Bí mật được lưu trong .env (chmod 600).")
    if only in (None, "harness"):
        h = configure_harness(env)
        write_env(env)
        write_harness(h)
    if only in (None, "kaggle"):
        configure_kaggle(env)
        write_env(env)
    if only in (None, "telegram"):
        configure_telegram(env)
        write_env(env)
    if only in (None, "services"):
        write_services()
    verify(env)
    head("Xong")
    drop = ROOT / "dropbox"
    drop.mkdir(exist_ok=True)
    say(f"  Thư mục thả dữ liệu: {drop}\n"
        f"    Từ Windows: scp -r .\\<thu-muc-du-lieu> {getpass.getuser()}@<IP>:{drop}/\n"
        "    hoặc WinSCP (SFTP) kéo thả. Mỗi bộ dữ liệu một thư mục con, kèm task.md. Nhắn /scan để xử lý ngay.")
    say("  Log bridge: journalctl -u kra-bridge -f      Log cron: tail -f ~/kra-tick.log\n"
        "  Đổi harness/provider: python3 install.py --only harness && sudo systemctl restart kra-bridge\n"
        "  Nhắn /help cho bot để bắt đầu.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        say("\nĐã huỷ.")
