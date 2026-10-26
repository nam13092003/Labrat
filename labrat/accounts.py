"""Nhiều tài khoản Kaggle — mỗi tài khoản có quota GPU tuần riêng và chạy được 1 kernel GPU cùng lúc.

Khai báo trong .env:
    KAGGLE_ACCOUNTS=acc1,acc2,acc3
    KAGGLE_ACC1_USERNAME=ten_kaggle_1
    KAGGLE_ACC1_TOKEN=KGAT_...          # token mới (khuyên dùng)
    # KAGGLE_ACC1_KEY=...               # hoặc key kiểu cũ (kaggle.json)
    # KAGGLE_ACC1_WEEKLY_HOURS=30       # tùy chọn, mặc định budget.weekly_gpu_hours
    ...

Không khai báo KAGGLE_ACCOUNTS → dùng một tài khoản "default" từ config.yaml (kaggle.username)
và KAGGLE_API_TOKEN / ~/.kaggle như trước.

Mỗi lệnh kaggle chạy với KAGGLE_CONFIG_DIR riêng của tài khoản để không lẫn thông tin đăng nhập.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .config import ROOT

CRED_ROOT = ROOT / ".kaggle_accounts"
_STRIP = ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_CONFIG_DIR")


@dataclass
class Account:
    name: str
    username: str
    token: str | None = None
    key: str | None = None
    weekly_gpu_hours: float = 30.0
    legacy_default: bool = False  # dùng thông tin đăng nhập sẵn có của máy

    def public(self) -> dict:
        return {"name": self.name, "username": self.username, "weekly_gpu_hours": self.weekly_gpu_hours,
                "auth": "token" if self.token else ("key" if self.key else "máy (~/.kaggle)")}


def load_accounts(cfg: dict) -> list[Account]:
    weekly = float(cfg.get("budget", {}).get("weekly_gpu_hours", 30))
    names = [n.strip() for n in os.environ.get("KAGGLE_ACCOUNTS", "").split(",") if n.strip()]
    if not names:
        user = cfg.get("kaggle", {}).get("username") or os.environ.get("KAGGLE_USERNAME")
        if not user or user == "your-kaggle-username":
            raise SystemExit("Chưa khai báo tài khoản Kaggle: đặt KAGGLE_ACCOUNTS trong .env "
                             "(hoặc kaggle.username trong config.yaml).")
        return [Account("default", user, token=os.environ.get("KAGGLE_API_TOKEN"),
                        weekly_gpu_hours=weekly, legacy_default=not os.environ.get("KAGGLE_API_TOKEN"))]
    accs = []
    for n in names:
        p = f"KAGGLE_{n.upper()}_"
        user = os.environ.get(p + "USERNAME")
        token, key = os.environ.get(p + "TOKEN"), os.environ.get(p + "KEY")
        if not user or not (token or key):
            raise SystemExit(f"Tài khoản {n}: thiếu {p}USERNAME hoặc {p}TOKEN/{p}KEY trong .env")
        accs.append(Account(n, user, token=token, key=key,
                            weekly_gpu_hours=float(os.environ.get(p + "WEEKLY_HOURS", weekly))))
    return accs


def get_account(cfg: dict, name: str | None) -> Account:
    accs = load_accounts(cfg)
    if name is None:
        return accs[0]
    for a in accs:
        if a.name == name or a.username == name:
            return a
    raise SystemExit(f"Không có tài khoản {name}. Có: {[a.name for a in accs]}")


def env_for(acc: Account) -> dict:
    """Môi trường cho subprocess `kaggle` của một tài khoản."""
    env = os.environ.copy()
    if acc.legacy_default:
        return env
    for k in _STRIP:
        env.pop(k, None)
    d = CRED_ROOT / acc.name
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(CRED_ROOT, 0o700)
    os.chmod(d, 0o700)
    if acc.token:
        f = d / "access_token"
        f.write_text(acc.token)
        env["KAGGLE_API_TOKEN"] = acc.token
    else:
        f = d / "kaggle.json"
        f.write_text(json.dumps({"username": acc.username, "key": acc.key}))
        env["KAGGLE_USERNAME"], env["KAGGLE_KEY"] = acc.username, acc.key
    os.chmod(f, 0o600)
    env["KAGGLE_CONFIG_DIR"] = str(d)
    return env


def run_account(run: dict, accs: list[Account]) -> str:
    """Tên tài khoản của một lần chạy (bản ghi cũ không có trường account → tài khoản đầu)."""
    if run.get("account"):
        return run["account"]
    ref = run.get("kernel_ref", "")
    for a in accs:
        if ref.startswith(a.username + "/"):
            return a.name
    return accs[0].name if accs else "default"
