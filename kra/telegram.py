"""Gửi tin nhắn / file qua Telegram Bot API. Thiếu token thì in ra màn hình thay vì lỗi."""
from __future__ import annotations

import os
from pathlib import Path

import requests

LIMIT = 4000


def _creds():
    return os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")


def enabled(cfg: dict | None = None) -> bool:
    tok, chat = _creds()
    return bool(tok and chat) and (cfg or {}).get("telegram", {}).get("enabled", True)


def send_message(text: str, cfg: dict | None = None, parse_mode: str = "HTML") -> bool:
    if not enabled(cfg):
        print("[telegram tắt]\n" + text)
        return False
    tok, chat = _creds()
    ok = True
    for i in range(0, len(text), LIMIT):
        r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", timeout=60, data={
            "chat_id": chat, "text": text[i:i + LIMIT], "parse_mode": parse_mode,
            "disable_web_page_preview": "true"})
        if not r.ok and parse_mode:  # lỗi định dạng HTML → gửi lại dạng thô
            r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage", timeout=60,
                              data={"chat_id": chat, "text": text[i:i + LIMIT]})
        ok &= r.ok
    return ok


def send_document(path: Path, caption: str = "", cfg: dict | None = None) -> bool:
    if not enabled(cfg):
        print(f"[telegram tắt] file: {path}  {caption}")
        return False
    tok, chat = _creds()
    with open(path, "rb") as f:
        r = requests.post(f"https://api.telegram.org/bot{tok}/sendDocument", timeout=300,
                          data={"chat_id": chat, "caption": caption[:1000], "parse_mode": "HTML"},
                          files={"document": (Path(path).name, f, "text/html")})
    return r.ok


def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
