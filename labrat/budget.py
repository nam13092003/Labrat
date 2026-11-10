"""Ngân sách GPU (theo từng tài khoản Kaggle) và điều kiện dừng.

- Quota tuần: RIÊNG cho từng tài khoản (mặc định 30h), reset 00:00 UTC thứ Bảy.
  Kaggle không có API trả quota còn lại → ta tự cộng dồn thời gian phiên GPU, theo USERNAME Kaggle, trên:
    * mọi dự án của bot này, và
    * mọi bot Labrat khác trên cùng máy (thư mục anh em có competitions/ — bot dùng chung tài khoản), và
    * hiệu chỉnh bằng số liệu thật đọc trên kaggle.com/me (`labrat quota <tk> --used H`, lưu ở ~/.labrat_quota.json,
      dùng chung mọi bot): từ thời điểm hiệu chỉnh, giờ đã dùng = số đọc trên web + các lần chạy đẩy sau đó.
- Trần dự án: TỔNG giờ GPU một dự án được tiêu trên mọi tài khoản.
- Mỗi tài khoản chỉ chạy 1 kernel GPU cùng lúc → N tài khoản = N thí nghiệm song song.
- Kernel đang chạy được tính là max(thời gian đã chạy, ước tính lúc đẩy) để không cam kết quá ngân sách.

Quota tính theo thời gian phiên, KHÔNG theo số GPU: một phiên T4 x2 chạy 1h tốn 1h quota.
"""
from __future__ import annotations

import json
import os
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .accounts import load_accounts, run_account
from .config import ROOT, competitions_root
from .registry import ACTIVE, Registry, better, parse_iso


def kaggle_week_start(now: datetime | None = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    days_since_sat = (now.weekday() - 5) % 7  # Mon=0 … Sat=5
    return (now - timedelta(days=days_since_sat)).replace(hour=0, minute=0, second=0, microsecond=0)


def run_gpu_hours(run: dict, overhead: float, now: datetime | None = None) -> float:
    if run.get("kind") != "gpu":
        return 0.0
    if run.get("gpu_hours") is not None:
        return float(run["gpu_hours"])
    now = now or datetime.now(timezone.utc)
    start = parse_iso(run.get("pushed_at"))
    end = parse_iso(run.get("finished_at"))
    if run.get("wall_seconds"):
        return run["wall_seconds"] / 3600 + overhead
    if start and end:
        return max(0.0, (end - start).total_seconds() / 3600)
    if start:  # đang chạy: giữ chỗ theo ước tính
        elapsed = (now - start).total_seconds() / 3600
        return max(elapsed, float(run.get("est_hours") or 0))
    return 0.0


def quota_file() -> Path:
    return Path(os.environ.get("LABRAT_QUOTA_FILE", Path.home() / ".labrat_quota.json"))


def load_calibration() -> dict:
    try:
        return json.loads(quota_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def set_calibration(username: str, used_hours: float, source: str = "kaggle.com/me") -> dict:
    cal = load_calibration()
    cal[username] = {"used": round(float(used_hours), 2), "at": datetime.now(timezone.utc).isoformat(),
                     "source": source}
    f = quota_file()
    f.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    try:
        f.chmod(0o600)
    except OSError:
        pass
    return cal[username]


def sibling_roots(cfg: dict) -> list[Path]:
    """Thư mục competitions/ của các bot Labrat khác trên cùng máy (dùng chung tài khoản Kaggle)."""
    b = cfg.get("budget", {}) or {}
    if b.get("quota_scan_roots") is not None:
        return [Path(os.path.expanduser(str(x))) for x in b["quota_scan_roots"]]
    if os.environ.get("LABRAT_NO_SIBLINGS"):
        return []
    out, me = [], ROOT.resolve()
    try:
        for d in sorted(ROOT.parent.iterdir()):
            if d.is_dir() and d.resolve() != me and (d / "competitions").is_dir() and (d / "labrat").is_dir():
                out.append(d / "competitions")
    except OSError:
        pass
    return out


def run_username(run: dict) -> str | None:
    ref = run.get("kernel_ref") or ""
    return ref.split("/")[0] if "/" in ref else None


def all_projects_runs(cfg: dict) -> list[tuple[str, dict, dict]]:
    """[(slug, experiment, run)] của mọi dự án."""
    out = []
    root = competitions_root(cfg)
    if root.exists():
        for reg_path in root.glob("*/registry.json"):
            for e in Registry(reg_path).experiments:
                for r in e.get("runs", []):
                    out.append((reg_path.parent.name, e, r))
    return out


def other_bots_runs(cfg: dict) -> list[tuple[str, dict, dict]]:
    out = []
    for root in sibling_roots(cfg):
        for reg_path in root.glob("*/registry.json"):
            try:
                exps = Registry(reg_path).experiments
            except Exception:
                continue
            for e in exps:
                for r in e.get("runs", []):
                    out.append((f"{root.parent.name}:{reg_path.parent.name}", e, r))
    return out


def accounts_status(cfg: dict) -> list[dict]:
    b = cfg.get("budget", {})
    overhead = float(b.get("startup_overhead_hours", 0.1))
    reserve = float(b.get("reserve_hours", 1.0))
    now = datetime.now(timezone.utc)
    week_start = kaggle_week_start(now)
    accs = load_accounts(cfg)
    by_user = {a.username: a.name for a in accs}
    cal = load_calibration()
    rows = {a.name: {**a.public(), "week_used": 0.0, "gpu_busy": None, "cpu_busy": 0, "other_bots_h": 0.0}
            for a in accs}
    cal_at = {}
    for a in accs:
        c = cal.get(a.username)
        t = parse_iso(c.get("at")) if c else None
        if t and t >= week_start:
            cal_at[a.name] = t
            rows[a.name]["week_used"] = float(c["used"])
            rows[a.name]["calibrated"] = {"used": c["used"], "at": c["at"]}
    sources = [(s, e, r, False) for s, e, r in all_projects_runs(cfg)] + \
              [(s, e, r, True) for s, e, r in other_bots_runs(cfg)]
    for slug, e, r, other in sources:
        user = run_username(r)
        name = by_user.get(user) if user else (None if other else run_account(r, accs))
        if name not in rows:
            continue
        t = parse_iso(r.get("pushed_at"))
        if t and t >= week_start and (name not in cal_at or t > cal_at[name]):
            h = run_gpu_hours(r, overhead, now)
            rows[name]["week_used"] += h
            if other:
                rows[name]["other_bots_h"] += h
        if not r.get("finished_at") and e.get("status") in ACTIVE:
            if r.get("kind") == "gpu":
                rows[name]["gpu_busy"] = f"{slug}/{e['id']}"
            elif not other:
                rows[name]["cpu_busy"] += 1
    for row in rows.values():
        row["week_used"] = round(row["week_used"], 2)
        row["other_bots_h"] = round(row["other_bots_h"], 2)
        row["week_left"] = round(max(0.0, row["weekly_gpu_hours"] - row["week_used"] - reserve), 2)
    return list(rows.values())


def pick_account(cfg: dict, est_hours: float, gpu: bool = True) -> dict | None:
    rows = accounts_status(cfg)
    if gpu:
        free = [r for r in rows if not r["gpu_busy"] and r["week_left"] >= est_hours]
        return max(free, key=lambda r: r["week_left"]) if free else None
    return min(rows, key=lambda r: (r["cpu_busy"], r["gpu_busy"] is not None))


def summary(cfg: dict, reg: Registry) -> dict:
    b = cfg.get("budget", {})
    overhead = float(b.get("startup_overhead_hours", 0.1))
    accs = accounts_status(cfg)
    project_used = sum(run_gpu_hours(r, overhead) for r in reg.all_runs())
    project_cap = float(b.get("project_gpu_hours", 20))
    project_left = max(0.0, project_cap - project_used)
    week_left_total = sum(a["week_left"] for a in accs)

    phases = b.get("phases", {"explore": 0.2, "exploit": 0.6, "final": 0.2})
    frac = project_used / project_cap if project_cap else 1.0
    if frac < phases.get("explore", 0.2):
        phase = "explore"
    elif frac < phases.get("explore", 0.2) + phases.get("exploit", 0.6):
        phase = "exploit"
    else:
        phase = "final"
    return {
        "project_used": round(project_used, 2),
        "project_cap": project_cap,
        "project_left": round(project_left, 2),
        "week_used": round(sum(a["week_used"] for a in accs), 2),
        "week_limit": round(sum(a["weekly_gpu_hours"] for a in accs), 1),
        "week_left": round(week_left_total, 2),
        "week_resets_at": (kaggle_week_start() + timedelta(days=7)).isoformat(),
        "remaining": round(min(project_left, week_left_total), 2),
        "free_gpu_slots": sum(1 for a in accs if not a["gpu_busy"] and a["week_left"] > 0),
        "accounts": accs,
        "phase": phase,
        "phase_fraction": round(frac, 3),
    }


def typical_run_hours(reg: Registry, default: float = 1.0) -> float:
    hrs = [r["gpu_hours"] for r in reg.all_runs() if r.get("kind") == "gpu" and r.get("gpu_hours")]
    return round(statistics.median(hrs), 2) if hrs else default


def stop_check(cfg: dict, reg: Registry, next_run_hours: float | None = None) -> dict:
    s = cfg.get("stop", {})
    direction = cfg.get("metric_direction", "maximize")
    min_delta = float(s.get("min_delta", 0.0))
    patience = int(s.get("patience", 4))
    target = s.get("target_score")

    bud = summary(cfg, reg)
    est = next_run_hours or typical_run_hours(reg)
    stop_reasons, pause_reasons = [], []

    if bud["project_left"] < est:
        stop_reasons.append(f"hết ngân sách GPU dự án (còn {bud['project_left']}h < ước tính {est}h)")
    elif not any(a["week_left"] >= est for a in bud["accounts"]):
        pause_reasons.append(f"mọi tài khoản hết quota tuần cho một lần chạy ~{est}h; "
                             f"chờ tới {bud['week_resets_at']}")

    done = reg.completed()
    best_val, since = None, 0
    for e in done:
        if better(e["cv"], best_val, direction, min_delta):
            best_val, since = e["cv"], 0
        else:
            since += 1
    if patience and since >= patience:
        stop_reasons.append(f"{since} thí nghiệm liên tiếp không cải thiện > {min_delta}")
    if target is not None and best_val is not None:
        if (best_val >= target) if direction == "maximize" else (best_val <= target):
            stop_reasons.append(f"đạt mục tiêu CV {target} (best {best_val})")

    return {
        "stop": bool(stop_reasons),
        "pause_until_reset": not stop_reasons and bool(pause_reasons),
        "reasons": stop_reasons + pause_reasons,
        "best_cv": best_val,
        "since_improvement": since,
        "estimated_next_run_hours": est,
        "can_push_gpu_now": not stop_reasons and pick_account(cfg, est) is not None,
        "budget": bud,
    }
