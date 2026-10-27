"""Tạo thí nghiệm, đẩy lên Kaggle, thu kết quả."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import budget, housekeeping, kaggle_ops
from .bundle import build_bundle, slugify
from .accounts import get_account, load_accounts, run_account
from .config import TEMPLATES, comp_dir
from .registry import ACTIVE, Registry, now_iso, parse_iso


def registry_for(cfg: dict) -> Registry:
    return Registry(comp_dir(cfg, cfg["slug"]) / "registry.json")


def exp_path(cfg: dict, exp: dict) -> Path:
    return comp_dir(cfg, cfg["slug"]) / "experiments" / exp["folder"]


def list_templates() -> list[str]:
    return sorted(p.name for p in (TEMPLATES / "experiments").iterdir() if p.is_dir())


def new_experiment(cfg: dict, *, name: str, hypothesis: str, template: str | None = None,
                   parent: str | None = None) -> dict:
    reg = registry_for(cfg)
    exp_id = reg.next_id()
    folder = f"{exp_id}-{slugify(name)[:30]}"
    dest = comp_dir(cfg, cfg["slug"]) / "experiments" / folder
    if parent:  # kế thừa code của thí nghiệm cha
        src_dir = exp_path(cfg, reg.get(parent))
    else:
        template = template or "gbdt_folds"
        src_dir = TEMPLATES / "experiments" / template
        if not src_dir.exists():
            raise SystemExit(f"Không có template {template}. Có: {', '.join(list_templates())}")
    shutil.copytree(src_dir, dest, ignore=shutil.ignore_patterns("out", "_bundle*", "__pycache__", "NOTES.md"))
    (dest / "NOTES.md").write_text(
        f"# {exp_id} — {name}\n\n**Giả thuyết:** {hypothesis}\n\n"
        f"**Kế thừa từ:** {parent or template}\n\n## Thay đổi so với cha\n\n- \n\n"
        "## Kết quả\n\n(điền sau khi chạy)\n\n## Phân tích & bài học\n\n- \n", encoding="utf-8")
    rec = reg.add({"id": exp_id, "folder": folder, "name": name, "hypothesis": hypothesis,
                   "parent": parent, "template": template if not parent else None})
    reg.save()
    return rec


# ---------------------------------------------------------------- push
def push(cfg: dict, exp_id: str, *, smoke: bool, force: bool = False, skip_smoke: bool = False,
         est_hours: float | None = None, account: str | None = None) -> dict:
    """Đẩy thí nghiệm lên Kaggle.

    GPU: tự chọn tài khoản đang rảnh (không có kernel GPU) còn nhiều quota tuần nhất,
    trừ khi chỉ định --account. Mỗi tài khoản tối đa 1 kernel GPU cùng lúc.
    """
    reg = registry_for(cfg)
    exp = reg.get(exp_id)
    est = est_hours or budget.typical_run_hours(reg)

    if exp.get("status") in ACTIVE and not force:
        raise SystemExit(f"{exp_id} đang chạy ({exp['status']}). Dùng `labrat poll` trước.")
    if not smoke:
        if exp.get("status") not in ("smoke_ok", "complete", "error") and not (skip_smoke or force):
            raise SystemExit(f"{exp_id} chưa qua smoke test (trạng thái {exp.get('status')}). "
                             "Chạy `labrat push --smoke` trước, hoặc thêm --skip-smoke.")
        chk = budget.stop_check(cfg, reg, est)
        if (chk["stop"] or chk["pause_until_reset"]) and not force:
            raise SystemExit("Không đẩy: " + "; ".join(chk["reasons"]))

    accs = {a["name"]: a for a in budget.accounts_status(cfg)}
    if account:
        acc = get_account(cfg, account)
        st = accs[acc.name]
        if not smoke and st["gpu_busy"] and not force:
            raise SystemExit(f"Tài khoản {acc.name} đang chạy GPU: {st['gpu_busy']}")
        if not smoke and st["week_left"] < est and not force:
            raise SystemExit(f"Tài khoản {acc.name} chỉ còn {st['week_left']}h quota tuần (< {est}h)")
    else:
        pick = budget.pick_account(cfg, est, gpu=not smoke)
        if pick is None:
            busy = []
            for a in accs.values():
                state = a["gpu_busy"] or "còn %sh tuần" % a["week_left"]
                busy.append(f"{a['name']}: {state}")
            raise SystemExit("Không còn tài khoản rảnh đủ quota cho ~%sh: %s" % (est, "; ".join(busy)))
        acc = get_account(cfg, pick["name"])

    folder, ref = build_bundle(exp_path(cfg, exp), cfg=cfg, exp=exp, username=acc.username, smoke=smoke)
    k = cfg.get("kaggle", {})
    accel = None if smoke else k.get("accelerator", "NvidiaTeslaT4")
    timeout_s = 3600 if smoke else float(k.get("max_runtime_hours", 9)) * 3600
    out = kaggle_ops.push_kernel(acc, folder, accelerator=accel, timeout_s=timeout_s)

    run = {"kind": "smoke" if smoke else "gpu", "account": acc.name, "kernel_ref": ref,
           "pushed_at": now_iso(), "status": "queued", "accelerator": accel,
           "est_hours": None if smoke else est, "push_output": out[-500:]}
    exp.setdefault("runs", []).append(run)
    exp["status"] = "smoke_running" if smoke else "running"
    reg.save()
    return {"exp": exp_id, "account": acc.name, "kernel_ref": ref, "kind": run["kind"],
            "accelerator": accel, "est_hours": run["est_hours"],
            "url": f"https://www.kaggle.com/code/{ref}", "cli": out[-300:]}


# ---------------------------------------------------------------- poll
# Mặc định chỉ tải về VPS: metrics/log/văn bản/ảnh nhỏ và dự đoán OOF. Thêm bằng job.yaml `fetch_outputs`
# (regex) hoặc config `outputs.fetch_pattern`. File > outputs.download_max_file_mb luôn bị xoá sau khi tải.
DEFAULT_FETCH = (r"(^|/)(metrics[^/]*\.json|[^/]*\.(log|txt|md|csv|tsv|yaml|yml|png|svg|html)"
                 r"|oof[^/]*\.(npz|npy|parquet|csv))$")


def fetch_pattern(cfg: dict, exp: dict) -> str:
    pats = [cfg.get("outputs", {}).get("fetch_pattern") or DEFAULT_FETCH]
    try:
        job = yaml.safe_load((exp_path(cfg, exp) / "job.yaml").read_text(encoding="utf-8")) or {}
        if job.get("fetch_outputs"):
            pats.append(str(job["fetch_outputs"]))
    except Exception:
        pass
    return "|".join(f"(?:{p})" for p in pats)


def _apply_metrics(cfg: dict, exp: dict, run: dict, m: dict) -> None:
    overhead = float(cfg.get("budget", {}).get("startup_overhead_hours", 0.05))
    run["wall_seconds"] = m.get("wall_seconds")
    run["n_gpu"] = m.get("n_gpu")
    run["gpu_util"] = m.get("gpu_util")
    if run["kind"] == "gpu":
        run["gpu_hours"] = round((m.get("wall_seconds") or 0) / 3600 + overhead, 3)
    if run["kind"] == "smoke":
        exp["status"] = "smoke_ok" if m.get("ok") else "smoke_failed"
    else:
        exp["status"] = "complete" if m.get("ok") and m.get("cv") is not None else "error"
        exp["cv"], exp["cv_std"] = m.get("cv"), m.get("cv_std")
        exp["finished_at"] = run["finished_at"]
        exp["metrics"] = {k: m.get(k) for k in ("cv", "cv_std", "selected_job", "final", "n_gpu",
                                                "gpu_util", "wall_seconds", "deadline_hit", "mode")}
        exp["metrics"]["jobs"] = [{k: j.get(k) for k in ("name", "gpu", "returncode", "seconds", "metrics",
                                                          "skipped", "killed")} for j in m.get("jobs", [])]
    failed = [j for j in m.get("jobs", []) if j.get("returncode") not in (0, None)]
    exp["last_errors"] = [{"job": j["name"], "log_tail": (j.get("log_tail") or "")[-3000:]} for j in failed]


def poll(cfg: dict) -> list[dict]:
    reg = registry_for(cfg)
    accs = load_accounts(cfg)
    events = []
    for exp in reg.active():
        run = exp["runs"][-1]
        acc = get_account(cfg, run_account(run, accs))
        try:
            status = kaggle_ops.kernel_status(acc, run["kernel_ref"])
        except kaggle_ops.KaggleError as e:
            events.append({"exp": exp["id"], "kind": run["kind"], "status": "unknown",
                           "account": acc.name, "error": str(e)[-300:]})
            continue
        run["status"] = status
        if status in ("queued", "running", "unknown"):
            started = parse_iso(run["pushed_at"])
            hrs = (datetime.now(timezone.utc) - started).total_seconds() / 3600
            events.append({"exp": exp["id"], "kind": run["kind"], "status": status, "account": acc.name,
                           "elapsed_h": round(hrs, 2)})
            continue
        run["finished_at"] = now_iso()
        dest = exp_path(cfg, exp) / "out" / run["kind"]
        if dest.exists():
            shutil.rmtree(dest)
        try:
            got = kaggle_ops.kernel_output(acc, run["kernel_ref"], dest, pattern=fetch_pattern(cfg, exp))
            run["output_filtered"] = got["filtered"]
            pr = housekeeping.prune_large(dest, housekeeping.download_max_mb(cfg))
            if pr:
                run["output_pruned"] = pr
        except kaggle_ops.KaggleError as e:
            run["output_error"] = str(e)[-500:]
        mfile = next(iter(dest.rglob("metrics.json")), None) if dest.exists() else None
        if mfile:
            _apply_metrics(cfg, exp, run, json.loads(mfile.read_text(encoding="utf-8")))
        else:
            # không có metrics.json: kernel chết trước khi runner ghi kết quả
            if run["kind"] == "gpu":
                start = parse_iso(run["pushed_at"])
                run["gpu_hours"] = round((datetime.now(timezone.utc) - start).total_seconds() / 3600, 3)
            exp["status"] = "smoke_failed" if run["kind"] == "smoke" else "error"
            logs = sorted(dest.rglob("*.log")) if dest.exists() else []
            exp["last_errors"] = [{"job": "kernel", "log_tail": logs[-1].read_text(errors="replace")[-3000:]
                                   if logs else f"Kernel kết thúc với trạng thái {status}, không có output"}]
        events.append({"exp": exp["id"], "kind": run["kind"], "status": exp["status"], "account": acc.name,
                       "cv": exp.get("cv"), "gpu_hours": run.get("gpu_hours"), "n_gpu": run.get("n_gpu"),
                       "gpu_util": run.get("gpu_util")})
    reg.save()
    return events
