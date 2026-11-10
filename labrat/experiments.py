"""Tạo thí nghiệm, đẩy lên Kaggle, thu kết quả."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import budget, housekeeping, kaggle_ops, policy
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
                   parent: str | None = None, evidence: str | None = None) -> dict:
    reg = registry_for(cfg)
    policy.check_idea(cfg, name, hypothesis)
    policy.check_evidence(cfg, evidence, len(reg.experiments))
    policy.check_prior_work(cfg, comp_dir(cfg, cfg["slug"]), len(reg.experiments))
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
        f"**Căn cứ (dữ liệu / phân tích lỗi / paper):** {evidence or '(baseline)'}\n\n"
        "**Cơ chế — vì sao kỳ vọng cải thiện:** \n\n**Dự đoán (nếu đúng thì CV / lát cắt nào đổi ra sao):** \n\n"
        f"**Kế thừa từ:** {parent or template}\n\n## Thay đổi so với cha\n\n- \n\n"
        "## Kết quả\n\n(điền sau khi chạy)\n\n## Phân tích & bài học\n\n- \n", encoding="utf-8")
    rec = reg.add({"id": exp_id, "folder": folder, "name": name, "hypothesis": hypothesis,
                   "evidence": evidence, "parent": parent, "template": template if not parent else None})
    reg.save()
    return rec


def record_result(cfg: dict, *, name: str, hypothesis: str, cv: float, cv_std: float | None = None,
                  method: str = "llm", cost_usd: float = 0.0, notes: str = "", artifacts: list[str] | None = None,
                  evidence: str | None = None) -> dict:
    """Ghi kết quả một thí nghiệm KHÔNG chạy trên kernel (vd zero/few-shot qua API) vào sổ thí nghiệm."""
    reg = registry_for(cfg)
    policy.check_idea(cfg, name, hypothesis, method)
    exp_id = reg.next_id()
    folder = f"{exp_id}-{slugify(name)[:30]}"
    d = comp_dir(cfg, cfg["slug"]) / "experiments" / folder
    d.mkdir(parents=True, exist_ok=True)
    (d / "NOTES.md").write_text(f"# {exp_id} — {name}\n\n**Giả thuyết:** {hypothesis}\n\n**Phương pháp:** {method}"
                                f" (không dùng GPU Kaggle)\n\n**Kết quả:** CV = {cv}"
                                f"{f' ± {cv_std}' if cv_std is not None else ''} · chi phí API ${cost_usd}\n\n"
                                f"{notes}\n\nFile: {', '.join(artifacts or [])}\n", encoding="utf-8")
    ts = now_iso()
    rec = reg.add({"id": exp_id, "folder": folder, "name": name, "hypothesis": hypothesis, "template": None,
                   "evidence": evidence, "parent": None, "status": "complete", "cv": float(cv), "cv_std": cv_std, "finished_at": ts,
                   "method": method, "api_cost_usd": round(float(cost_usd), 4),
                   "metrics": {"cv": float(cv), "cv_std": cv_std, "method": method, "artifacts": artifacts or []},
                   "runs": [{"kind": "api", "pushed_at": ts, "finished_at": ts, "status": "complete",
                             "cost_usd": round(float(cost_usd), 4)}]})
    reg.save()
    return rec


# ---------------------------------------------------------------- push
def code_hash(cfg: dict, exp: dict) -> str:
    """Băm code + job.yaml của thí nghiệm (bỏ out/, bundle, NOTES) — để biết cấu hình có bị đổi sau lần chạy."""
    import hashlib
    d = exp_path(cfg, exp)
    h = hashlib.sha256()
    for p in sorted(d.rglob("*")):
        rel = p.relative_to(d)
        if (not p.is_file() or rel.parts[0] in ("out", "NOTES.md") or rel.parts[0].startswith("_bundle")
                or "__pycache__" in rel.parts):
            continue
        h.update(rel.as_posix().encode()); h.update(p.read_bytes())
    return h.hexdigest()[:16]


def run_label(run: dict) -> str:
    """Tên thư mục output của một lần chạy: smoke | quick | gpu | seed<N>."""
    if run.get("kind") == "smoke":
        return "smoke"
    if run.get("fidelity") == "quick":
        return "quick"
    if run.get("fidelity") == "report":
        return f"seed{run.get('seed')}"
    return "gpu"


def _choose_account(cfg: dict, est: float, *, smoke: bool, account: str | None, force: bool):
    accs = {a["name"]: a for a in budget.accounts_status(cfg)}
    if account:
        acc = get_account(cfg, account)
        st = accs[acc.name]
        if not smoke and st["gpu_busy"] and not force:
            raise SystemExit(f"Tài khoản {acc.name} đang chạy GPU: {st['gpu_busy']}")
        if not smoke and st["week_left"] < est and not force:
            raise SystemExit(f"Tài khoản {acc.name} chỉ còn {st['week_left']}h quota tuần (< {est}h)")
        return acc
    pick = budget.pick_account(cfg, est, gpu=not smoke)
    if pick is None:
        busy = [f"{a['name']}: {a['gpu_busy'] or 'còn %sh tuần' % a['week_left']}" for a in accs.values()]
        raise SystemExit("Không còn tài khoản rảnh đủ quota cho ~%sh: %s" % (est, "; ".join(busy)))
    return get_account(cfg, pick["name"])


def parent_source(cfg: dict, reg: Registry, exp: dict) -> tuple[str, str] | None:
    """job.yaml `parent_outputs: true` → (tài khoản, kernel_ref) của lần chạy đầy đủ của thí nghiệm cha.

    Output kernel riêng tư chỉ tài khoản sở hữu đọc được → thí nghiệm con PHẢI chạy trên đúng tài khoản đó.
    """
    jf = exp_path(cfg, exp) / "job.yaml"
    job = yaml.safe_load(jf.read_text(encoding="utf-8")) if jf.exists() else {}
    if not (job or {}).get("parent_outputs"):
        return None
    if not exp.get("parent"):
        raise SystemExit(f"{exp['id']}: parent_outputs cần thí nghiệm cha (new-exp --parent).")
    main = _main_run(reg.get(exp["parent"]))
    if not main:
        raise SystemExit(f"{exp['parent']} chưa có lần chạy đầy đủ thành công để lấy checkpoint.")
    return main["account"], main["kernel_ref"]


def _launch(cfg: dict, exp: dict, acc, *, smoke: bool, est: float, fidelity: str = "full",
            seed: int | None = None, extra_sources: list[str] | None = None) -> dict:
    k = cfg.get("kaggle", {})
    quick = fidelity == "quick"
    seed = seed if seed is not None else int(cfg.get("research", {}).get("base_seed", 42))
    folder, ref = build_bundle(exp_path(cfg, exp), cfg=cfg, exp=exp, username=acc.username, smoke=smoke,
                               quick=quick, seed=seed, extra_kernel_sources=extra_sources)
    accel = None if smoke else k.get("accelerator", "NvidiaTeslaT4")
    max_h = float(k.get("quick_max_hours", 1.5)) if quick else float(k.get("max_runtime_hours", 9))
    timeout_s = 3600 if smoke else max_h * 3600
    out = kaggle_ops.push_kernel(acc, folder, accelerator=accel, timeout_s=timeout_s)
    run = {"kind": "smoke" if smoke else "gpu", "account": acc.name, "kernel_ref": ref,
           "pushed_at": now_iso(), "status": "queued", "accelerator": accel,
           "est_hours": None if smoke else est, "push_output": out[-500:], "seed": seed,
           "code_hash": code_hash(cfg, exp)}
    if not smoke:
        run["fidelity"] = fidelity
    exp.setdefault("runs", []).append(run)
    return run


def notes_sections(cfg: dict, exp: dict) -> dict:
    """Đọc các mục người viết trong NOTES.md: căn cứ, cơ chế, dự đoán, thay đổi so với cha."""
    import re
    f = exp_path(cfg, exp) / "NOTES.md"
    t = f.read_text(encoding="utf-8") if f.exists() else ""
    out = {}
    for key, label in (("evidence", "Căn cứ"), ("mechanism", "Cơ chế"), ("prediction", "Dự đoán")):
        m = re.search(r"\*\*" + label + r"[^*]*:\*\*[ \t]*(.*?)(?=\n\*\*|\n## |\Z)", t, re.S)
        out[key] = (m.group(1).strip() if m else "")
    m = re.search(r"## Thay đổi so với cha[ \t]*\n(.*?)(?=\n## |\Z)", t, re.S)
    ch = m.group(1).strip() if m else ""
    out["changes"] = "" if ch in ("", "-") else ch
    return out


def _is_first(reg: Registry, exp: dict) -> bool:
    return bool(reg.experiments) and reg.experiments[0]["id"] == exp["id"]


def check_ready(cfg: dict, reg: Registry, exp: dict) -> dict:
    """Trước khi chạy GPU: NOTES.md phải nói rõ thí nghiệm sẽ làm gì (để báo người dùng)."""
    secs = notes_sections(cfg, exp)
    r = cfg.get("research", {}) or {}
    if r.get("require_prediction", True) and policy.settings(cfg)["enforce"] and not _is_first(reg, exp) \
            and len(secs["prediction"]) < 10:
        raise SystemExit(f"Điền NOTES.md của {exp['id']} trước khi chạy GPU: mục **Dự đoán** (nếu giả thuyết đúng thì CV "
                         "/ lát cắt nào đổi khoảng bao nhiêu), **Cơ chế** (vì sao), và **Thay đổi so với cha**. "
                         "Nội dung này được gửi cho người dùng trước khi kernel chạy.")
    return secs


def announce(cfg: dict, reg: Registry, exp: dict, *, level: str, account: str | None, est: float | None,
             seeds: list[int] | None = None) -> bool:
    """Báo người dùng qua Telegram TRƯỚC khi đẩy kernel GPU."""
    from . import telegram
    if not (cfg.get("telegram", {}) or {}).get("announce_runs", True) or not telegram.enabled(cfg):
        return False
    E = telegram.esc
    secs = notes_sections(cfg, exp)
    clip = lambda x, n=350: (x[:n] + "…") if len(x) > n else x
    lvl = {"full": "chạy đầy đủ", "quick": "bản rẻ (sàng lọc)",
           "report": f"chốt kết quả, seed {', '.join(map(str, seeds or []))}"}.get(level, level)
    lines = [f"🚀 <b>{E(cfg['slug'])} · {E(exp['id'])}</b> {E(exp.get('name', ''))} — {E(lvl)}",
             f"<b>Giả thuyết:</b> {E(clip(exp.get('hypothesis', '')))}"]
    ev = exp.get("evidence") or secs["evidence"]
    if ev and ev != "(baseline)":
        lines.append(f"<b>Căn cứ:</b> {E(clip(ev))}")
    if exp.get("parent") or secs["changes"]:
        lines.append(f"<b>Thay đổi so với {E(exp.get('parent') or 'nền')}:</b> {E(clip(secs['changes'] or '(chưa ghi)'))}")
    if secs["mechanism"]:
        lines.append(f"<b>Cơ chế:</b> {E(clip(secs['mechanism']))}")
    if secs["prediction"]:
        lines.append(f"<b>Dự đoán:</b> {E(clip(secs['prediction']))}")
    try:
        sm = budget.summary(cfg, reg)
        bud = f" · dự án đã dùng {sm['project_used']}/{sm['project_cap']}h"
    except Exception:
        bud = ""
    lines.append(f"⏱ {('tài khoản ' + E(account) + ' · ') if account else ''}~{est or '?'}h GPU{('/seed' if level == 'report' else '')}{bud}")
    try:
        return telegram.send_message("\n".join(lines), cfg)
    except Exception:
        return False


def push(cfg: dict, exp_id: str, *, smoke: bool, quick: bool = False, force: bool = False,
         skip_smoke: bool = False, est_hours: float | None = None, account: str | None = None) -> dict:
    """Đẩy thí nghiệm lên Kaggle.

    smoke: CPU, dữ liệu nhỏ (không tốn quota). quick: GPU bản rẻ (job.yaml quick_args / quick_max_jobs,
    tối đa kaggle.quick_max_hours) để sàng lọc ý tưởng. Mặc định: chạy đầy đủ.
    GPU: tự chọn tài khoản đang rảnh (không có kernel GPU) còn nhiều quota tuần nhất,
    trừ khi chỉ định --account. Mỗi tài khoản tối đa 1 kernel GPU cùng lúc.
    """
    reg = registry_for(cfg)
    exp = reg.get(exp_id)
    k = cfg.get("kaggle", {})
    est = est_hours or budget.typical_run_hours(reg)
    if quick:
        est = est_hours or min(est, float(k.get("quick_max_hours", 1.5)))
    job_file = exp_path(cfg, exp) / "job.yaml"
    if job_file.exists():
        policy.check_job(cfg, yaml.safe_load(job_file.read_text(encoding="utf-8")) or {})

    if exp.get("status") in ACTIVE and not force:
        raise SystemExit(f"{exp_id} đang chạy ({exp['status']}). Dùng `labrat poll` trước.")
    if not smoke:
        if exp.get("status") not in ("smoke_ok", "quick_ok", "complete", "error") and not (skip_smoke or force):
            raise SystemExit(f"{exp_id} chưa qua smoke test (trạng thái {exp.get('status')}). "
                             "Chạy `labrat push --smoke` trước, hoặc thêm --skip-smoke.")
        chk = budget.stop_check(cfg, reg, est)
        if (chk["stop"] or chk["pause_until_reset"]) and not force:
            raise SystemExit("Không đẩy: " + "; ".join(chk["reasons"]))

    if not smoke:
        check_ready(cfg, reg, exp)
    ps = parent_source(cfg, reg, exp)
    if ps:
        if account and get_account(cfg, account).name != ps[0]:
            raise SystemExit(f"{exp_id} dùng output của {exp['parent']} (tài khoản {ps[0]}) → phải chạy trên {ps[0]}.")
        account = ps[0]
    acc = _choose_account(cfg, est, smoke=smoke, account=account, force=force)
    announced = False if smoke else announce(cfg, reg, exp, level="quick" if quick else "full", account=acc.name,
                                              est=round(est, 2))
    run = _launch(cfg, exp, acc, smoke=smoke, est=est, fidelity="quick" if quick else "full",
                  extra_sources=[ps[1]] if ps else None)
    run["announced"] = announced
    exp["status"] = "smoke_running" if smoke else "running"
    reg.save()
    return {"exp": exp_id, "account": acc.name, "kernel_ref": run["kernel_ref"],
            "kind": run["kind"] + ("/quick" if quick else ""), "accelerator": run["accelerator"],
            "est_hours": run["est_hours"], "url": f"https://www.kaggle.com/code/{run['kernel_ref']}",
            "cli": run["push_output"][-300:]}


def _main_run(exp: dict) -> dict | None:
    """Lần chạy đầy đủ cuối cùng đã thành công (cho ra exp['cv'])."""
    for r in reversed(exp.get("runs", [])):
        if r.get("kind") == "gpu" and r.get("fidelity", "full") == "full":
            return r if r.get("ok", True) and r.get("finished_at") else None
    return None


def report_seeds(cfg: dict, exp_id: str, n: int | None = None, *, est_hours: float | None = None,
                 force: bool = False) -> dict:
    """Giai đoạn CHỐT KẾT QUẢ: chạy lại CẤU HÌNH ĐÃ CHỐT với n seed để báo cáo trung bình ± độ lệch.

    Không phải dò seed: không đổi cấu hình, không chọn seed — `cv` của thí nghiệm giữ nguyên là lần chạy chính
    (seed gốc), mọi seed đều được báo cáo. Seed chỉ đổi khởi tạo mô hình / thứ tự dữ liệu (LABRAT_SEED),
    KHÔNG đổi cách chia fold. Mỗi seed một kernel trên tài khoản rảnh; gọi lại để đẩy nốt seed còn thiếu.
    """
    r = cfg.get("research", {}) or {}
    n = int(n or r.get("report_seeds", 3))
    max_n = int(r.get("max_report_seeds", 5))
    if n < 2 or n > max_n:
        raise SystemExit(f"Số seed báo cáo phải trong 2..{max_n} (research.max_report_seeds).")
    reg = registry_for(cfg)
    exp = reg.get(exp_id)
    if exp.get("status") not in ("complete", "reporting"):
        raise SystemExit(f"{exp_id} chưa có lần chạy đầy đủ thành công (trạng thái {exp.get('status')}).")
    main = _main_run(exp)
    if not main:
        raise SystemExit(f"{exp_id}: không tìm thấy lần chạy đầy đủ thành công.")
    if main.get("code_hash") and main["code_hash"] != code_hash(cfg, exp) and not force:
        raise SystemExit(f"Code/job.yaml của {exp_id} đã đổi sau lần chạy chính → không còn là cấu hình đã chốt. "
                         "Tạo thí nghiệm mới (--parent) và chạy đầy đủ trước.")
    reported = [e for e in reg.experiments if e.get("report") or e.get("status") == "reporting"]
    max_exp = int(r.get("max_report_experiments", 6))
    if exp not in reported and len(reported) >= max_exp and not force:
        raise SystemExit(f"Đã có {len(reported)} thí nghiệm chạy nhiều seed (tối đa {max_exp}). Chỉ dành cho "
                         "phương pháp cuối, baseline mạnh nhất và các ablation chính.")
    base = int(main.get("seed") or r.get("base_seed", 42))
    seeds = [base + i for i in range(n)]
    have = {x.get("seed") for x in exp["runs"] if x.get("fidelity") == "report"
            and (x.get("ok") or not x.get("finished_at"))}
    todo = [s for s in seeds[1:] if s not in have]
    est = est_hours or main.get("gpu_hours") or budget.typical_run_hours(reg)
    pushed, waiting = [], []
    ps = parent_source(cfg, reg, exp)
    if todo:
        announce(cfg, reg, exp, level="report", account=None, est=round(est, 2), seeds=seeds)
    for s in todo:
        chk = budget.stop_check(cfg, reg, est)
        if (chk["stop"] or chk["pause_until_reset"]) and not force:
            waiting.append(s); continue
        try:
            acc = _choose_account(cfg, est, smoke=False, account=ps[0] if ps else None, force=False)
        except SystemExit:
            waiting.append(s); continue
        run = _launch(cfg, exp, acc, smoke=False, est=est, fidelity="report", seed=s,
                      extra_sources=[ps[1]] if ps else None)
        exp["status"] = "reporting"
        reg.save()
        pushed.append({"seed": s, "account": acc.name, "kernel_ref": run["kernel_ref"]})
    _aggregate_report(exp)
    reg.save()
    return {"exp": exp_id, "seeds": seeds, "pushed": pushed, "waiting_for_free_account": waiting,
            "report": exp.get("report"),
            "next": "Chờ `labrat wait`; gọi lại lệnh này để đẩy các seed còn chờ." if waiting else "Chờ `labrat wait`."}


def _aggregate_report(exp: dict) -> None:
    import statistics
    main = _main_run(exp)
    if not main or exp.get("cv") is None:
        return
    rows = [{"seed": main.get("seed", 42), "cv": exp["cv"]}]
    rows += [{"seed": x.get("seed"), "cv": x["cv"]} for x in exp.get("runs", [])
             if x.get("fidelity") == "report" and x.get("ok") and x.get("cv") is not None]
    if len(rows) < 2:
        return
    cvs = [x["cv"] for x in rows]
    exp["report"] = {"n": len(rows), "seeds": [x["seed"] for x in rows], "cvs": cvs,
                     "mean": statistics.mean(cvs), "std": statistics.stdev(cvs),
                     "min": min(cvs), "max": max(cvs)}


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
    overhead = float(cfg.get("budget", {}).get("startup_overhead_hours", 0.1))
    run["wall_seconds"] = m.get("wall_seconds")
    run["n_gpu"] = m.get("n_gpu")
    run["gpu_util"] = m.get("gpu_util")
    if run["kind"] == "gpu":
        run["gpu_hours"] = round((m.get("wall_seconds") or 0) / 3600 + overhead, 3)
    ok = bool(m.get("ok")) and (run["kind"] == "smoke" or m.get("cv") is not None)
    run["ok"] = ok
    fidelity = run.get("fidelity", "full")
    if run["kind"] == "smoke":
        exp["status"] = "smoke_ok" if m.get("ok") else "smoke_failed"
    elif fidelity == "report":
        # seed báo cáo: KHÔNG đổi cv / trạng thái chính của thí nghiệm
        run["cv"], run["cv_std"] = m.get("cv"), m.get("cv_std")
    elif fidelity == "quick":
        exp["status"] = "quick_ok" if ok else "error"
        exp["quick_cv"], exp["quick_cv_std"] = m.get("cv"), m.get("cv_std")
        run["cv"], run["cv_std"] = m.get("cv"), m.get("cv_std")
    else:
        exp["status"] = "complete" if ok else "error"
        exp["cv"], exp["cv_std"] = m.get("cv"), m.get("cv_std")
        exp["finished_at"] = run["finished_at"]
        exp["metrics"] = {k: m.get(k) for k in ("cv", "cv_std", "selected_job", "final", "n_gpu",
                                                "gpu_util", "wall_seconds", "deadline_hit", "mode")}
        exp["metrics"]["jobs"] = [{k: j.get(k) for k in ("name", "gpu", "returncode", "seconds", "metrics",
                                                          "skipped", "killed")} for j in m.get("jobs", [])]
        exp.pop("report", None)  # cấu hình chạy lại → kết quả nhiều seed cũ không còn đúng
    failed = [j for j in m.get("jobs", []) if j.get("returncode") not in (0, None)]
    errs = [{"job": j["name"], "log_tail": (j.get("log_tail") or "")[-3000:]} for j in failed]
    if fidelity == "report":
        if errs:
            run["errors"] = errs
    else:
        exp["last_errors"] = errs


def _pending_runs(exp: dict) -> list[dict]:
    if exp.get("status") == "reporting":
        return [r for r in exp.get("runs", []) if r.get("fidelity") == "report" and not r.get("finished_at")]
    return exp["runs"][-1:] if exp.get("runs") else []


def poll(cfg: dict) -> list[dict]:
    reg = registry_for(cfg)
    accs = load_accounts(cfg)
    events = []
    for exp in reg.active():
        for run in _pending_runs(exp):
            ev = _poll_run(cfg, exp, run, accs)
            if ev:
                events.append(ev)
        if exp.get("status") == "reporting" and not _pending_runs(exp):
            exp["status"] = "complete"
            _aggregate_report(exp)
            rep = exp.get("report")
            events.append({"exp": exp["id"], "kind": "report", "status": "report_done", "cv": exp.get("cv"),
                           "report": rep})
    reg.save()
    return events


def _poll_run(cfg: dict, exp: dict, run: dict, accs) -> dict | None:
    acc = get_account(cfg, run_account(run, accs))
    label = run_label(run)
    base = {"exp": exp["id"], "kind": run["kind"], "fidelity": run.get("fidelity"), "seed": run.get("seed"),
            "account": acc.name}
    try:
        status = kaggle_ops.kernel_status(acc, run["kernel_ref"])
    except kaggle_ops.KaggleError as e:
        return {**base, "status": "unknown", "error": str(e)[-300:]}
    run["status"] = status
    if status in ("queued", "running", "unknown"):
        started = parse_iso(run["pushed_at"])
        hrs = (datetime.now(timezone.utc) - started).total_seconds() / 3600
        return {**base, "status": status, "elapsed_h": round(hrs, 2)}
    run["finished_at"] = now_iso()
    dest = exp_path(cfg, exp) / "out" / label
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
        run["ok"] = False
        logs = sorted(dest.rglob("*.log")) if dest.exists() else []
        err = [{"job": "kernel", "log_tail": logs[-1].read_text(errors="replace")[-3000:]
                if logs else f"Kernel kết thúc với trạng thái {status}, không có output"}]
        if run.get("fidelity") == "report":
            run["errors"] = err
        else:
            exp["status"] = "smoke_failed" if run["kind"] == "smoke" else "error"
            exp["last_errors"] = err
    st = exp["status"] if run.get("fidelity") != "report" else ("seed_ok" if run.get("ok") else "seed_error")
    return {**base, "status": st, "cv": run.get("cv", exp.get("cv")) if run.get("fidelity") else exp.get("cv"),
            "gpu_hours": run.get("gpu_hours"), "n_gpu": run.get("n_gpu"), "gpu_util": run.get("gpu_util")}
