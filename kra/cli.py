"""Giao diện dòng lệnh. Mọi lệnh in JSON (dễ cho agent đọc).

Dự án
  kra init <slug> [--kind competition|custom] [--metric AUC] [--direction maximize|minimize]
                  [--title ...] [--gpu-hours 20]
  kra fetch <slug> [--download]                      thông tin/dữ liệu cuộc thi (kind=competition)
Tài khoản Kaggle
  kra accounts [--check]                             trạng thái quota & kernel từng tài khoản
Nghiên cứu
  kra research <slug> "<query>" [--limit 10]         arXiv + Semantic Scholar
  kra lit-add <slug> --title ... --url ... [--relevance 1-5] [--notes] [--tags] [--source kaggle]
  kra paper <slug> <arxiv-id|url|pdf>                tải & trích văn bản 1 paper, tạo analysis.md
  kra paper-report <slug> <pid> [--send]             thẻ HTML từ analysis.md
Dữ liệu
  kra get <url> [--name X]                           tải dữ liệu từ link vào dropbox/X/
  kra dropbox [--scan [--force]]                     thư mục thả dữ liệu: liệt kê / quét mục mới
  kra data-add <slug> <url|path> [--link]            nhận dữ liệu người dùng vào data/raw/
  kra dataset <slug> <folder> --name X [--message]   tải lên MỌI tài khoản (dataset riêng tư)
  kra dataset-status <slug> <name>
Thực nghiệm
  kra new-exp <slug> --name ... --hypothesis ... [--template gbdt_folds|torch_ddp] [--parent exp-003]
  kra push <slug> <exp> [--smoke] [--account acc2] [--skip-smoke] [--force] [--est-hours 1.5]
  kra poll <slug> | wait <slug> | tick <slug> | status <slug> | budget <slug> | stop-check <slug>
Báo cáo
  kra report <slug> [--send]
  kra notify <slug> "<text>"
  kra templates
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import yaml

from . import budget, data, dropbox, experiments, kaggle_ops, papers, report, research, telegram
from .accounts import load_accounts
from .config import comp_dir, load_config


def out(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


PROBLEM_CARD = """# Thẻ bài toán — {slug}

> Agent điền từ trang cuộc thi (Overview, Data, Evaluation, Rules) hoặc từ mô tả của người dùng
> (dự án custom: dữ liệu người dùng gửi, paper cần tái hiện...).

## Tóm tắt bài toán
- Mục tiêu:
- Loại bài: (tabular / CV / NLP / time series / ...)
- Đơn vị dự đoán:

## Đánh giá
- Metric:
- Chiều tốt: (cao hơn / thấp hơn tốt hơn)
- Chi tiết tính điểm:

## Dữ liệu
- Nguồn: (cuộc thi Kaggle / người dùng gửi / dataset công khai)
- Các file & kích thước:
- Số dòng / số mẫu:
- Đặc điểm (missing, mất cân bằng, phân phối train/test, rò rỉ thời gian...):

## Luật / ràng buộc
- Dữ liệu ngoài:
- Mô hình pretrained:
- Deadline:

## Chiến lược validation
- Cách chia fold và lý do:
- Rủi ro lệch CV–test:
"""


def cmd_init(a):
    cfg = load_config()
    d = comp_dir(cfg, a.slug)
    (d / "experiments").mkdir(parents=True, exist_ok=True)
    proj = d / "project.yaml"
    if not proj.exists():
        p = {"kind": a.kind, "title": a.title or a.slug, "metric": a.metric or "",
             "metric_direction": a.direction, "budget": {"project_gpu_hours": a.gpu_hours},
             "stop": {"patience": 4, "min_delta": 0.0005, "target_score": None},
             "datasets": [], "own_datasets": []}
        if a.kind == "custom":
            p["no_competition_source"] = True
        else:
            p["competition"] = a.slug
        proj.write_text(yaml.safe_dump(p, allow_unicode=True, sort_keys=False), encoding="utf-8")
    if not (d / "problem_card.md").exists():
        (d / "problem_card.md").write_text(PROBLEM_CARD.format(slug=a.slug), encoding="utf-8")
    for name, content in (("literature.json", {"items": [], "queries": []}),
                          ("registry.json", {"experiments": []}),
                          ("report_notes.json", {"summary": "", "insights": [], "next_directions": []})):
        if not (d / name).exists():
            (d / name).write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
    hint = ("Mọi tài khoản Kaggle phải bấm chấp nhận luật trên trang cuộc thi."
            if a.kind == "competition" else "Dự án custom: đưa dữ liệu lên bằng `kra data-add` + `kra dataset`.")
    out({"created": str(d), "kind": a.kind, "note": hint})


def cmd_fetch(a):
    cfg = load_config(a.slug)
    acc = load_accounts(cfg)[0]
    d = comp_dir(cfg, a.slug)
    files = kaggle_ops.competition_files(acc, a.slug)
    (d / "data_files.txt").write_text(files, encoding="utf-8")
    res = {"files": files.splitlines()[:50]}
    # kiểm tra mọi tài khoản đã tham gia cuộc thi (cần để kernel đọc được dữ liệu)
    joined = {}
    for ac in load_accounts(cfg):
        try:
            kaggle_ops.competition_files(ac, a.slug)
            joined[ac.name] = True
        except kaggle_ops.KaggleError as e:
            joined[ac.name] = f"CHƯA: {str(e)[-150:]}"
    res["accounts_joined"] = joined
    if a.download:
        res["download"] = kaggle_ops.competition_download(acc, a.slug, d / "data")
    out(res)


def cmd_accounts(a):
    cfg = load_config()
    rows = budget.accounts_status(cfg)
    if a.check:
        for acc, row in zip(load_accounts(cfg), rows):
            try:
                kaggle_ops.whoami(acc)
                row["login"] = "ok"
            except kaggle_ops.KaggleError as e:
                row["login"] = f"LỖI: {str(e)[-200:]}"
    out({"accounts": rows, "week_resets_at": budget.summary(cfg, experiments.Registry(
        comp_dir(cfg, "_none") / "registry.json"))["week_resets_at"]})


def cmd_research(a):
    cfg = load_config(a.slug)
    out(research.search(comp_dir(cfg, a.slug) / "literature.json", a.query, a.limit))


def cmd_lit_add(a):
    cfg = load_config(a.slug)
    item = {"source": a.source, "title": a.title, "url": a.url, "year": a.year or "",
            "notes": a.notes or "", "relevance": a.relevance, "abstract": "",
            "method_tags": [t.strip() for t in (a.tags or "").split(",") if t.strip()]}
    out({"added": research.add_items(comp_dir(cfg, a.slug) / "literature.json", [item])})


def cmd_paper(a):
    cfg = load_config(a.slug)
    out(papers.fetch_paper(cfg, a.src, a.pid))


def cmd_paper_report(a):
    cfg = load_config(a.slug)
    path = papers.render_analysis(cfg, a.pid)
    sent = telegram.send_document(path, f"Phân tích paper · {a.pid}", cfg) if a.send else False
    out({"report": str(path), "sent": sent})


def cmd_data_add(a):
    cfg = load_config(a.slug)
    out(data.data_add(cfg, a.src, link=a.link))


def cmd_get(a):
    from . import fetcher
    cfg = load_config()
    root = dropbox.dropbox_dir(cfg)
    dest = root / (fetcher.slugify(a.name) if a.name else fetcher.guess_name(a.url))
    try:
        res = fetcher.download(a.url, dest, max_gb=float(cfg.get("downloads", {}).get("max_gb", 50)))
    except fetcher.FetchError as e:
        out({"error": str(e)})
        sys.exit(2)
    dropbox.announce_now(dest)  # đã xử lý → cron không giao lặp lại
    out(res)


def cmd_dropbox(a):
    if a.scan:
        out({"ready": dropbox.scan(force=a.force)})
    else:
        out({"dropbox": str(dropbox.dropbox_dir()), "items": dropbox.listing()})


def cmd_dataset(a):
    cfg = load_config(a.slug)
    out(data.upload_dataset(cfg, a.folder, a.name, a.message))


def cmd_dataset_status(a):
    cfg = load_config(a.slug)
    out(data.dataset_ready(cfg, a.name))


def cmd_new_exp(a):
    cfg = load_config(a.slug)
    rec = experiments.new_experiment(cfg, name=a.name, hypothesis=a.hypothesis, template=a.template,
                                     parent=a.parent)
    out({"exp": rec["id"], "folder": str(experiments.exp_path(cfg, rec)),
         "next": "Sửa src/ và job.yaml, rồi `kra push --smoke`"})


def cmd_push(a):
    cfg = load_config(a.slug)
    out(experiments.push(cfg, a.exp, smoke=a.smoke, force=a.force, skip_smoke=a.skip_smoke,
                         est_hours=a.est_hours, account=a.account))


def _notify_events(cfg, events):
    for ev in events:
        if ev["status"] in ("queued", "running", "unknown"):
            continue
        e = experiments.registry_for(cfg).get(ev["exp"])
        icon = {"complete": "✅", "smoke_ok": "🧪", "error": "❌", "smoke_failed": "❌"}.get(ev["status"], "ℹ️")
        util = ", ".join(f"GPU{g['gpu']} {g['util_mean']:.0f}%" for g in (ev.get("gpu_util") or []))
        msg = (f"{icon} <b>{telegram.esc(cfg['slug'])} · {telegram.esc(ev['exp'])}</b> "
               f"{telegram.esc(e.get('name',''))}: {telegram.esc(ev['status'])} "
               f"[{telegram.esc(ev.get('account',''))}]")
        if ev.get("cv") is not None:
            msg += f"\nCV = <b>{ev['cv']:.5f}</b>"
        if ev.get("gpu_hours"):
            msg += f" · {ev['gpu_hours']:.2f}h GPU"
        if util:
            msg += f"\n{telegram.esc(util)}"
        if ev["kind"] == "gpu" and ev.get("n_gpu") is not None and ev["n_gpu"] < 2:
            msg += f"\n⚠️ chỉ nhận {ev['n_gpu']} GPU"
        if e.get("last_errors"):
            msg += "\n<pre>" + telegram.esc(e["last_errors"][0]["log_tail"][-600:]) + "</pre>"
        telegram.send_message(msg, cfg)


def cmd_poll(a):
    cfg = load_config(a.slug)
    events = experiments.poll(cfg)
    if not a.quiet:
        _notify_events(cfg, events)
    out(events)


def cmd_wait(a):
    cfg = load_config(a.slug)
    interval = int(cfg.get("poll_interval_minutes", 10)) * 60
    t_end = time.time() + a.max_minutes * 60
    while True:
        events = experiments.poll(cfg)
        _notify_events(cfg, events)
        active = experiments.registry_for(cfg).active()
        finished = [ev for ev in events if ev["status"] not in ("queued", "running", "unknown")]
        # với nhiều tài khoản: trả về ngay khi có một thí nghiệm xong để agent đẩy tiếp
        if not active or (a.any and finished):
            out({"done": not active, "events": events, "still_running": [e["id"] for e in active]})
            return
        if time.time() > t_end:
            out({"done": False, "events": events, "still_running": [e["id"] for e in active]})
            return
        time.sleep(interval)


def cmd_tick(a):
    cfg = load_config(a.slug)
    events = experiments.poll(cfg)
    _notify_events(cfg, events)
    reg = experiments.registry_for(cfg)
    chk = budget.stop_check(cfg, reg)
    out({"events": events, "active": [e["id"] for e in reg.active()], "stop_check": chk,
         "paused": bool(cfg.get("paused")),
         "free_gpu_slots": chk["budget"]["free_gpu_slots"],
         "ready_for_next": chk["can_push_gpu_now"] and not cfg.get("paused")})


def cmd_status(a):
    cfg = load_config(a.slug)
    reg = experiments.registry_for(cfg)
    out([{"id": e["id"], "name": e.get("name"), "status": e.get("status"), "cv": e.get("cv"),
          "cv_std": e.get("cv_std"), "parent": e.get("parent"),
          "account": (e.get("runs") or [{}])[-1].get("account")} for e in reg.experiments])


def cmd_budget(a):
    cfg = load_config(a.slug)
    out(budget.summary(cfg, experiments.registry_for(cfg)))


def cmd_stop_check(a):
    cfg = load_config(a.slug)
    out(budget.stop_check(cfg, experiments.registry_for(cfg)))


def cmd_report(a):
    cfg = load_config(a.slug)
    path = report.build(cfg)
    sent = False
    if a.send:
        telegram.send_message(report.short_text(cfg), cfg)
        sent = telegram.send_document(path, "Thẻ báo cáo — mở bằng trình duyệt", cfg)
    out({"report": str(path), "sent": sent})


def cmd_notify(a):
    cfg = load_config(a.slug)
    out({"sent": telegram.send_message(a.text, cfg)})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kra", description="Kaggle Research Agent")
    sp = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, *slug_args):
        s = sp.add_parser(name)
        for x in slug_args:
            s.add_argument(x)
        s.set_defaults(f=fn)
        return s

    s = add("init", cmd_init, "slug")
    s.add_argument("--kind", default="competition", choices=["competition", "custom"])
    s.add_argument("--direction", default="maximize", choices=["maximize", "minimize"])
    s.add_argument("--metric"); s.add_argument("--title"); s.add_argument("--gpu-hours", type=float, default=20)
    s = add("fetch", cmd_fetch, "slug"); s.add_argument("--download", action="store_true")
    s = add("accounts", cmd_accounts); s.add_argument("--check", action="store_true")
    s = add("research", cmd_research, "slug", "query"); s.add_argument("--limit", type=int, default=10)
    s = add("lit-add", cmd_lit_add, "slug")
    s.add_argument("--title", required=True); s.add_argument("--url", required=True)
    s.add_argument("--year"); s.add_argument("--notes"); s.add_argument("--relevance", type=int)
    s.add_argument("--source", default="kaggle"); s.add_argument("--tags")
    s = add("paper", cmd_paper, "slug", "src"); s.add_argument("--pid")
    s = add("paper-report", cmd_paper_report, "slug", "pid"); s.add_argument("--send", action="store_true")
    s = add("data-add", cmd_data_add, "slug", "src")
    s.add_argument("--link", action="store_true", help="symlink thay vì sao chép (dữ liệu lớn trong dropbox)")
    s = add("get", cmd_get, "url"); s.add_argument("--name")
    s = add("dropbox", cmd_dropbox); s.add_argument("--scan", action="store_true")
    s.add_argument("--force", action="store_true", help="không chờ lần quét thứ hai")
    s = add("dataset", cmd_dataset, "slug", "folder")
    s.add_argument("--name", required=True); s.add_argument("--message", default="update")
    add("dataset-status", cmd_dataset_status, "slug", "name")
    s = add("new-exp", cmd_new_exp, "slug")
    s.add_argument("--name", required=True); s.add_argument("--hypothesis", required=True)
    s.add_argument("--template"); s.add_argument("--parent")
    s = add("push", cmd_push, "slug", "exp")
    s.add_argument("--smoke", action="store_true"); s.add_argument("--skip-smoke", action="store_true")
    s.add_argument("--force", action="store_true"); s.add_argument("--est-hours", type=float)
    s.add_argument("--account")
    s = add("poll", cmd_poll, "slug"); s.add_argument("--quiet", action="store_true")
    s = add("wait", cmd_wait, "slug"); s.add_argument("--max-minutes", type=int, default=600)
    s.add_argument("--any", action="store_true", help="trả về ngay khi MỘT thí nghiệm xong")
    for name, fn in (("tick", cmd_tick), ("status", cmd_status), ("budget", cmd_budget),
                     ("stop-check", cmd_stop_check)):
        add(name, fn, "slug")
    s = add("report", cmd_report, "slug"); s.add_argument("--send", action="store_true")
    add("notify", cmd_notify, "slug", "text")
    add("templates", lambda a: out(experiments.list_templates()))
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        a.f(a)
    except kaggle_ops.KaggleError as e:
        out({"error": str(e)})
        sys.exit(2)


if __name__ == "__main__":
    main()
