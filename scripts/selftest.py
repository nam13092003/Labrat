"""Kiểm tra offline (không gọi Kaggle/Telegram/LLM thật — các lệnh ngoài được giả lập):
  1. tên kernel, nạp nhiều tài khoản, thông tin đăng nhập tách theo tài khoản
  2. bundle gbdt_folds → chạy main.py cục bộ như trong kernel (mode parallel, 2 slot) + smoke
  3. đẩy song song: 3 tài khoản → 3 kernel GPU, cái thứ 4 bị chặn; tài khoản hết quota bị bỏ qua
  4. poll giả lập → registry, budget, stop-check
  5. thẻ báo cáo HTML (có phần tài khoản)
  6. paper: PDF cục bộ → text.txt + analysis.md → thẻ HTML
  7. data-add: file zip → giải nén; thư mục thả dữ liệu (dropbox)
  8. bridge: lệnh nhanh; agent: dựng lệnh cho mọi harness, chạy harness giả
  9. (nếu có torch) smoke template torch_ddp

Chạy:  python scripts/selftest.py        (dọn sạch sau khi chạy; --keep để giữ lại báo cáo mẫu)
"""
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# môi trường giả: 3 tài khoản Kaggle, không Telegram
FAKE = {"KAGGLE_ACCOUNTS": "a1,a2,a3"}
for i in (1, 2, 3):
    FAKE[f"KAGGLE_A{i}_USERNAME"] = f"user{i}"
    FAKE[f"KAGGLE_A{i}_TOKEN"] = f"KGAT_fake{i}"
os.environ.update(FAKE)
for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
    os.environ.pop(k, None)
os.environ["TELEGRAM_BOT_TOKEN"] = ""  # chặn load_env điền từ .env thật

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from labrat import accounts, agent, bridge, budget, data, experiments, kaggle_ops, papers, report  # noqa: E402
from labrat.bundle import build_bundle, kernel_title  # noqa: E402
from labrat.cli import main as cli  # noqa: E402
from labrat.config import comp_dir, load_config  # noqa: E402
from labrat.registry import now_iso  # noqa: E402

SLUG = "_selftest"
KEEP = "--keep" in sys.argv
BACKUP = {}


def check(cond, msg):
    print(("  ✓ " if cond else "  ✗ ") + msg)
    if not cond:
        raise SystemExit(1)


def run_bundle_locally(bundle_dir: Path, data_dir: Path) -> dict:
    work = Path(tempfile.mkdtemp(prefix="labrat_work_"))
    env = dict(os.environ, LABRAT_WORKDIR=str(work), LABRAT_INPUT_DIR=str(data_dir))
    r = subprocess.run([sys.executable, str(bundle_dir / "main.py")], env=env, capture_output=True,
                       text=True, timeout=900)
    print("\n".join("    " + l for l in r.stdout.strip().splitlines()[-4:]))
    if r.returncode != 0:
        print(r.stderr[-3000:])
    m = json.loads((work / "metrics.json").read_text())
    shutil.rmtree(work)
    return m


def backup(path: Path):
    BACKUP[path] = path.read_bytes() if path.exists() else None


def restore():
    for path, content in BACKUP.items():
        if content is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(content)


def make_data() -> Path:
    d = Path(tempfile.mkdtemp(prefix="labrat_data_"))
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, 6))
    y = ((X[:, 0] + 0.5 * X[:, 1] ** 2 - X[:, 2] + rng.normal(scale=0.7, size=n)) > 0.4).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["cat"] = rng.choice(list("abc"), n)
    df["id"], df["target"] = np.arange(n), y
    df.to_csv(d / "train.csv", index=False)
    return d


def main():
    shutil.rmtree(ROOT / "competitions" / SLUG, ignore_errors=True)
    for f in (agent.HARNESS_FILE, agent.STATE_FILE, agent.CHAT_LOG):
        backup(f)
    creds_existed = accounts.CRED_ROOT.exists()
    data_dir = make_data()
    try:
        _tests(data_dir)
    finally:
        restore()
        shutil.rmtree(data_dir, ignore_errors=True)
        if not creds_existed:
            shutil.rmtree(accounts.CRED_ROOT, ignore_errors=True)
        else:
            for i in (1, 2, 3):
                shutil.rmtree(accounts.CRED_ROOT / f"a{i}", ignore_errors=True)
        if not KEEP:
            shutil.rmtree(ROOT / "competitions" / SLUG, ignore_errors=True)
    print("\nTất cả kiểm tra đều qua.")


def _tests(data_dir: Path):
    print("1) tên kernel + tài khoản")
    t = kernel_title("labrat", "playground-series-s6e1", "exp-012", "Target encoding + DART, 2 seeds", False)
    check(5 <= len(t) <= 50 and t == t.lower() and " " not in t, f"title hợp lệ: {t}")
    cfg0 = load_config()
    accs = accounts.load_accounts(cfg0)
    check([a.username for a in accs] == ["user1", "user2", "user3"], "nạp 3 tài khoản từ .env")
    env = accounts.env_for(accs[1])
    tok = Path(env["KAGGLE_CONFIG_DIR"]) / "access_token"
    check(env["KAGGLE_API_TOKEN"] == "KGAT_fake2" and tok.read_text() == "KGAT_fake2"
          and stat.S_IMODE(tok.stat().st_mode) == 0o600, "thông tin đăng nhập tách riêng, chmod 600")

    print("2) init + bundle + chạy cục bộ")
    cli(["init", SLUG, "--metric", "AUC", "--title", "Selftest Tabular", "--gpu-hours", "20"])
    cfg = load_config(SLUG)
    exp = experiments.new_experiment(cfg, name="baseline", hypothesis="mốc so sánh", template="gbdt_folds")
    edir = experiments.exp_path(cfg, exp)
    job = yaml.safe_load((edir / "job.yaml").read_text())
    job["cpu_slots"] = 2  # không có GPU: vẫn chạy 2 slot song song để thử hàng đợi
    (edir / "job.yaml").write_text(yaml.safe_dump(job, allow_unicode=True))
    bdir, ref = build_bundle(edir, cfg=cfg, exp=exp, username="user2", smoke=False)
    meta = json.loads((bdir / "kernel-metadata.json").read_text())
    check(meta["enable_gpu"] and meta["competition_sources"] == [SLUG] and ref.startswith("user2/"),
          "metadata GPU, nguồn dữ liệu cuộc thi, đúng username")
    m = run_bundle_locally(bdir, data_dir)
    check(m["ok"] and len([j for j in m["jobs"] if not j.get("post")]) == 5, "runner: 5 fold thành công")
    check(m["cv"] is not None and 0.5 < m["cv"] <= 1.0, f"CV tổng = {m['cv']:.4f} ± {m['cv_std']:.4f}")
    sdir, sref = build_bundle(edir, cfg=cfg, exp=exp, username="user1", smoke=True)
    ms = run_bundle_locally(sdir, data_dir)
    check(sref.endswith("-cpu") and ms["ok"] and ms["smoke"], "smoke: kernel CPU riêng, 1 fold")

    print("3) đẩy song song trên nhiều tài khoản (giả lập kaggle)")
    pushed = []
    kaggle_ops.push_kernel = lambda acc, folder, accelerator=None, timeout_s=None: (
        pushed.append((acc.name, accelerator, json.loads((Path(folder) / "kernel-metadata.json").read_text())))
        or "Kernel version 1 successfully pushed")
    reg = experiments.registry_for(cfg)
    ids = [exp["id"]]
    for i in range(3):
        ids.append(experiments.new_experiment(cfg, name=f"variant-{i}", hypothesis=f"biến thể {i}",
                                              template="gbdt_folds")["id"])
    reg = experiments.registry_for(cfg)
    for e in reg.experiments:
        e["status"] = "smoke_ok"
    reg.save()
    outs = [experiments.push(cfg, x, smoke=False, est_hours=1.5) for x in ids[:3]]
    check(sorted(o["account"] for o in outs) == ["a1", "a2", "a3"], "3 thí nghiệm → 3 tài khoản khác nhau")
    check(all(p[1] == "NvidiaTeslaT4" and p[2]["id"].startswith(f"user{p[0][1]}/") for p in pushed),
          "accelerator T4 x2, kernel thuộc đúng tài khoản")
    try:
        experiments.push(cfg, ids[3], smoke=False, est_hours=1.5)
        check(False, "lần đẩy thứ 4 phải bị chặn")
    except SystemExit as e:
        check("Không còn tài khoản rảnh" in str(e), "lần đẩy thứ 4 bị chặn: mọi tài khoản đang bận")
    b = budget.summary(cfg, experiments.registry_for(cfg))
    check(abs(b["project_used"] - 4.5) < 0.01, f"kernel đang chạy được giữ chỗ theo ước tính: {b['project_used']}h")

    print("4) poll giả lập → registry")
    finished = {outs[0]["kernel_ref"]}

    def fake_status(acc, ref):
        return "complete" if ref in finished else "running"

    def fake_output(acc, ref, dest):
        dest.mkdir(parents=True, exist_ok=True)
        mm = dict(m, wall_seconds=3600 * 1.2, n_gpu=2,
                  gpu_util=[{"gpu": 0, "util_mean": 87.0, "util_max": 100, "mem_max_mb": 9200},
                            {"gpu": 1, "util_mean": 81.0, "util_max": 99, "mem_max_mb": 8800}])
        (dest / "metrics.json").write_text(json.dumps(mm))
        return [dest / "metrics.json"]
    kaggle_ops.kernel_status, kaggle_ops.kernel_output = fake_status, fake_output
    events = experiments.poll(cfg)
    done = [e for e in events if e["status"] == "complete"]
    check(len(done) == 1 and done[0]["account"] == outs[0]["account"] and done[0]["cv"],
          f"1 xong ({done[0]['account']}, cv={done[0]['cv']:.4f}), 2 đang chạy")
    o4 = experiments.push(cfg, ids[3], smoke=False, est_hours=1.5)
    check(o4["account"] == outs[0]["account"], f"tài khoản vừa rảnh ({o4['account']}) nhận thí nghiệm tiếp theo")

    # tài khoản a2 gần hết quota tuần → không được chọn
    reg = experiments.registry_for(cfg)
    hog = reg.add({"id": reg.next_id(), "folder": "hog", "name": "dài", "hypothesis": "chạy dài"})
    hog["runs"] = [{"kind": "gpu", "account": "a2", "pushed_at": now_iso(), "finished_at": now_iso(),
                    "gpu_hours": 28.5}]
    hog.update(status="error")
    reg.save()
    rows = {r["name"]: r for r in budget.accounts_status(cfg)}
    check(rows["a2"]["week_left"] == 0.0, "tài khoản dùng 28.5h → hết quota (trừ dự phòng 1h)")

    print("5) budget + stop-check + báo cáo")
    finished.update(o["kernel_ref"] for o in outs[1:] + [o4])
    experiments.poll(cfg)
    reg = experiments.registry_for(cfg)
    chk = budget.stop_check(cfg, reg)
    check(chk["stop"] and "ngân sách" in chk["reasons"][0],
          f"dừng khi vượt trần dự án 20h (đã dùng {chk['budget']['project_used']}h)")
    ws = budget.kaggle_week_start(datetime(2026, 10, 26, 5, tzinfo=timezone.utc))
    check(ws.weekday() == 5 and ws.day == 24, f"tuần quota bắt đầu {ws.date()}")
    reg.experiments.remove(next(e for e in reg.experiments if e["folder"] == "hog"))
    t0 = datetime.now(timezone.utc) - timedelta(hours=10)
    for i, e in enumerate(reg.completed()):
        e["cv"] = m["cv"] + [0.0, 0.004, -0.001, 0.006][i % 4] + 0.0002 * i
        e["finished_at"] = (t0 + timedelta(hours=i)).isoformat(timespec="seconds")
    reg.save()
    (comp_dir(cfg, SLUG) / "report_notes.json").write_text(json.dumps({
        "summary": "Phân loại nhị phân tabular. Baseline XGBoost 5-fold, sau đó 3 biến thể chạy song song trên "
                   "3 tài khoản Kaggle (6 GPU T4).",
        "insights": ["Biến thể 2 tốt nhất (+0.006 CV).", "Cả 2 GPU mỗi phiên đạt ~85% sử dụng."],
        "next_directions": [{"title": "Stacking XGBoost + CatBoost", "why": "Sai số khác nhau ở nhóm cat hiếm.",
                             "cost": "~3h GPU", "expected_gain": "+0.002"}]}, ensure_ascii=False))
    path = report.build(cfg)
    html = path.read_text()
    check("<svg" in html and "Tài khoản Kaggle" in html and "a3" in html, f"báo cáo: {path.relative_to(ROOT)}")

    print("6) nghiên cứu paper (PDF cục bộ)")
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        pdf = Path(tempfile.mkdtemp()) / "Sparse Mixture Tabular.pdf"
        c = canvas.Canvas(str(pdf), pagesize=A4)
        for page in range(3):
            y = 800
            for line in range(45):
                c.drawString(40, y, f"Section {page}.{line}: we propose a sparse mixture of experts for tabular data.")
                y -= 16
            c.showPage()
        c.save()
        res = papers.fetch_paper(cfg, str(pdf))
        check(res["pages"] == 3 and res["chars"] > 2000 and Path(res["analysis"]).exists(),
              f"paper {res['pid']}: {res['pages']} trang, {res['chars']} ký tự, có analysis.md")
        h = papers.render_analysis(cfg, res["pid"]).read_text()
        check("<table>" in h and "Đề xuất thực nghiệm" in h, "thẻ HTML phân tích paper (bảng đề xuất)")
        lit = json.loads((comp_dir(cfg, SLUG) / "literature.json").read_text())
        check(any(i.get("status") == "deep_read" for i in lit["items"]), "ghi vào literature.json")
        shutil.rmtree(pdf.parent)
    except ImportError:
        print("  - bỏ qua (không có reportlab để tạo PDF thử)")

    print("7) nhận dữ liệu người dùng")
    z = Path(tempfile.mkdtemp()) / "churn-data.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(data_dir / "train.csv", "train.csv")
    res = data.data_add(cfg, str(z))
    check(res["extracted_to"] and res["n_files"] == 1 and res["total_mb"] > 0, f"giải nén: {res['sample']}")
    shutil.rmtree(z.parent)
    uploads = []
    kaggle_ops.dataset_upsert = lambda acc, folder, slug, title, message="": uploads.append(acc.name) or "ok"
    up = data.upload_dataset(cfg, Path(res["extracted_to"]), "churn data")
    cfg = load_config(SLUG)
    _, ref2 = build_bundle(edir, cfg=cfg, exp=exp, username="user3", smoke=False)
    meta2 = json.loads((edir / "_bundle_gpu" / "kernel-metadata.json").read_text())
    check(uploads == ["a1", "a2", "a3"] and "user3/churn-data" in meta2["dataset_sources"],
          "dataset tải lên mọi tài khoản; kernel của user3 gắn user3/churn-data")

    print("7b) thư mục thả dữ liệu")
    from labrat import dropbox
    import time as _t
    drop = dropbox.dropbox_dir()
    item = drop / "zz-selftest-drop"
    item.mkdir(exist_ok=True)
    shutil.copy(data_dir / "train.csv", item / "train.csv")
    (item / "task.md").write_text("Metric: AUC, nhãn: target")
    (item / "dang-chep.csv.part").write_text("x")
    old = _t.time() - 600
    for f in item.iterdir():
        os.utime(f, (old, old))
    try:
        check(not [x for x in dropbox.scan() if x["name"] == item.name], "lần quét đầu: chỉ ghi nhận, chưa giao")
        got = [x for x in dropbox.scan() if x["name"] == item.name]
        check(got and got[0]["n_files"] == 2 and "AUC" in got[0]["task"], "lần quét hai: giao agent (bỏ .part, đọc task.md)")
        check(not [x for x in dropbox.scan() if x["name"] == item.name], "không giao lặp lại")
        (item / "them.csv").write_text("a\n1")
        check(not [x for x in dropbox.scan(force=True) if x["name"] == item.name], "file vừa chép (<3 phút) → chờ")
        os.utime(item / "them.csv", (old, old))
        upd = [x for x in dropbox.scan(force=True) if x["name"] == item.name]
        check(upd and upd[0]["kind"] == "update", "thêm file → báo cập nhật")
        r = data.data_add(cfg, str(item), link=True)
        check(Path(r["saved"]).is_symlink(), "data-add --link: symlink, không sao chép")
    finally:
        shutil.rmtree(item, ignore_errors=True)
        dropbox.scan()

    print("7c) gửi link qua Telegram → tự tải (server HTTP cục bộ, bot giả)")
    import functools
    import http.server
    import io
    import queue as _q
    import socketserver
    import threading
    from labrat import fetcher
    srv_dir = Path(tempfile.mkdtemp())
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.write(data_dir / "train.csv", "train.csv")
    (srv_dir / "zz-churn.zip").write_bytes(buf.getvalue())

    class _H(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    srv = socketserver.TCPServer(("127.0.0.1", 0), functools.partial(_H, directory=str(srv_dir)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1"

    class FakeTG:
        def __init__(self):
            self.msgs = []

        def send(self, t):
            self.msgs.append(t)
            return len(self.msgs)

        def edit(self, mid, t):
            self.msgs.append(t)
    tg, aq = FakeTG(), _q.Queue()
    dl = bridge.Downloader(tg, aq)
    text = f"Dữ liệu churn khách hàng, metric AUC, nhãn target http://127.0.0.1:{srv.server_address[1]}/zz-churn.zip"
    links, _ = fetcher.data_urls(text)
    try:
        dl._do({"urls": links, "text": fetcher.URL_RE.sub(" ", text), "name": None})
        dest = dropbox.dropbox_dir() / "zz-churn"
        task = aq.get_nowait()
        check((dest / "train.csv").exists() and (dest / "task.md").exists(), "tải + giải nén vào dropbox/zz-churn/, ghi task.md")
        check(task["source"] == "download" and "AUC" in task["prompt"], "giao agent kèm mô tả từ tin nhắn")
        check(any("Đã tải xong" in m for m in tg.msgs), "báo tiến độ / hoàn tất qua Telegram")
        check(not [x for x in dropbox.scan(force=True) if x["name"] == "zz-churn"], "cron không giao lặp lại")
        bad = f"http://127.0.0.1:{srv.server_address[1]}/khong-co.zip"
        dl._do({"urls": [bad], "text": "", "name": "zz-bad"})
        check(any("Không tải được" in m for m in tg.msgs) and not (dropbox.dropbox_dir() / "zz-bad").exists(),
              "link lỗi → báo lỗi, không để lại thư mục rỗng")
    finally:
        srv.shutdown()
        shutil.rmtree(srv_dir, ignore_errors=True)
        shutil.rmtree(dropbox.dropbox_dir() / "zz-churn", ignore_errors=True)
        dropbox.scan()

    print("7d) đọc trang web cuộc thi ngoài Kaggle")
    from labrat import pages
    site = Path(tempfile.mkdtemp()) / "2027"
    site.mkdir()
    nav = '<nav><a href="description.html">Mô tả</a> <a href="data.html">Data</a> <a href="rules.html">Rules</a></nav>'
    (site / "description.html").write_text(
        f"<html><head><title>XACLE 2027 Challenge</title></head><body>{nav}<main><h1>Task description</h1>"
        "<p>Predict the audio-text alignment score for each clip. " + "Context sentence. " * 40 + "</p>"
        "<h2>Evaluation</h2><p>Submissions are ranked by Spearman correlation (SRCC).</p>"
        "<ul><li>Baseline: 0.334</li><li>Team limit: 5</li></ul>"
        '<a href="https://other-site.org/x.html">bên ngoài</a></main></body></html>', encoding="utf-8")
    (site / "data.html").write_text(
        f"<html><head><title>Data</title></head><body>{nav}<main><h1>Dataset</h1><p>" + "Info. " * 80 +
        '</p><a href="files/train.zip">train.zip</a> <a href="https://drive.google.com/file/d/1AbCdEfGhIjKlMn/view">'
        "dev set</a><table><tr><th>split</th><th>clips</th></tr><tr><td>train</td><td>7500</td></tr></table>"
        "</main></body></html>", encoding="utf-8")
    (site / "rules.html").write_text(f"<html><head><title>Rules</title></head><body>{nav}<main><h1>Rules</h1><p>"
                                     + "External data allowed if public. " * 20 + "</p></main></body></html>",
                                     encoding="utf-8")
    srv2 = socketserver.TCPServer(("127.0.0.1", 0), functools.partial(_H, directory=str(site.parent)))
    threading.Thread(target=srv2.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv2.server_address[1]}/2027"
    try:
        msg = f"nghiên cứu cuộc thi {base}/description.html với ngân sách 15h GPU"
        dl_links, other = fetcher.data_urls(msg)
        check(not dl_links and other, "link trang web → chuyển agent, KHÔNG đưa bộ tải file")
        res = pages.crawl(f"{base}/description.html", slug=SLUG, follow=1)
        titles = sorted(p_["title"] for p_ in res["pages"])
        check(titles == ["Data", "Rules", "XACLE 2027 Challenge"], f"đọc 3 trang cùng site: {titles}")
        txt = Path(next(p_["file"] for p_ in res["pages"] if p_["title"].startswith("XACLE"))).read_text()
        check("## Evaluation" in txt and "Spearman" in txt and "- Baseline: 0.334" in txt,
              "giữ cấu trúc: tiêu đề, danh sách, metric")
        dtxt = Path(next(p_["file"] for p_ in res["pages"] if p_["title"] == "Data")).read_text()
        check("train | 7500" in dtxt, "bảng → dòng 'a | b'")
        check(any(u.endswith("files/train.zip") for u in res["data_links"])
              and any("drive.google.com" in u for u in res["data_links"])
              and not any("other-site" in u for u in res["data_links"]),
              f"gom link dữ liệu: {len(res['data_links'])} link (zip + Drive), bỏ link ngoài site")
        tg2, aq2 = FakeTG(), _q.Queue()
        bridge.Downloader(tg2, aq2)._do({"urls": [f"{base}/description.html"], "text": "/get thử", "name": "zz-web"})
        fwd = aq2.get_nowait()
        check("labrat page" in fwd["prompt"] and any("trang web" in m_ for m_ in tg2.msgs)
              and not (dropbox.dropbox_dir() / "zz-web").exists(),
              "/get vào trang web → chuyển agent đọc, không báo lỗi cụt")
    finally:
        srv2.shutdown()
        shutil.rmtree(site.parent, ignore_errors=True)

    print("8) bridge + agent")
    check("/status" in bridge.handle_quick("/help"), "/help")
    check("Đã tạm dừng" in bridge.handle_quick(f"/stop {SLUG}") and load_config(SLUG).get("paused"), "/stop")
    check("tiếp tục" in bridge.handle_quick(f"/resume {SLUG}") and not load_config(SLUG).get("paused"), "/resume")
    check(bridge.handle_quick("xin chào") is None, "tin thường → chuyển cho agent")
    for name in ("claude", "codex", "opencode", "dsh"):
        agent.HARNESS_FILE.write_text(yaml.safe_dump({"harness": name, "model": "m-x"}))
        h = agent.load_harness()
        first, cont = agent.build_cmd(h, "P", True, "/tmp/o"), agent.build_cmd(h, "P", False, "/tmp/o")
        check("P" in first and "m-x" in " ".join(first) or name == "dsh", f"{name}: {' '.join(cont)[:90]}")
    agent.HARNESS_FILE.write_text(yaml.safe_dump({"harness": "custom", "adapters": {"custom": {
        "first_cmd": [sys.executable, "-c", "import sys; print('NHẬN:', sys.argv[1][-30:])", "{prompt}"],
        "cmd": [sys.executable, "-c", "import sys; print('NHẬN:', sys.argv[1][-30:])", "{prompt}"],
        "output": "text", "stateless": True}}}, allow_unicode=True))
    agent.reset_session()
    r = agent.run_agent("trạng thái dự án selftest", source="selftest")
    check(r["ok"] and "trạng thái dự án selftest" in r["reply"], f"harness giả trả lời: {r['reply']}")
    r2 = agent.run_agent("lượt hai", source="selftest")
    check("lượt hai" in r2["reply"], "lượt thứ hai (harness không nhớ phiên → gửi kèm lịch sử)")

    print("8b) tối ưu cache tiền tố (DeepSeek/Anthropic)")
    cap = Path(tempfile.mkdtemp()) / "prompts.jsonl"
    rec = ("import sys,json; open(%r,'a').write(json.dumps(sys.argv[1])+'\\n'); "
           "print(json.dumps({'result':'ok','usage':{'input_tokens':120,'cache_read_input_tokens':4800,"
           "'cache_creation_input_tokens':80,'output_tokens':50},'total_cost_usd':0.001}))") % str(cap)
    agent.HARNESS_FILE.write_text(yaml.safe_dump({"harness": "custom", "history_turns": 4, "adapters": {"custom": {
        "first_cmd": [sys.executable, "-c", rec, "{prompt}"], "cmd": [sys.executable, "-c", rec, "{prompt}"],
        "output": "claude_json", "stateless": True}}}, allow_unicode=True))
    agent.CHAT_LOG.unlink(missing_ok=True)
    st = agent._state(); st.pop("epoch_start", None); agent._save_state(st)
    for i in range(6):
        agent.run_agent(f"tin nhắn số {i}", source="selftest")
    ps = [json.loads(l) for l in cap.read_text().splitlines()]
    dyn = [p_[:p_.index("\n[")] for p_ in ps]  # phần trước khối động (giờ/nguồn)
    check(all("[" not in d.split("\n")[0] for d in dyn) and ps[0].startswith("Bạn là Labrat"),
          "giờ / nguồn nằm ở CUỐI prompt, không ở đầu")
    grow = [ps[i + 1].startswith(dyn[i]) for i in range(len(ps) - 1)]
    check(sum(grow) >= 4, f"lượt sau giữ nguyên tiền tố của lượt trước ({sum(grow)}/{len(grow)} cặp; "
                          "1 lần đổi epoch là chủ đích)")
    u = agent.usage_summary(1)
    check(u["turns_with_usage"] == 6 and abs(u["cache_hit_rate"] - 4800 / 5000) < 1e-3,
          f"thống kê cache: hit {100*u['cache_hit_rate']:.0f}% trên {u['turns']} lượt")
    check("cache hit 96%" in bridge.handle_quick("/usage"), "/usage trên Telegram")
    claude_env = agent.DEFAULT_ADAPTERS["claude"]["env"]
    check(claude_env["CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS"] == "1" and claude_env["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0",
          "Claude Code: bỏ git status + attribution block khỏi system prompt")
    shutil.rmtree(cap.parent)

    print("9) template torch_ddp (smoke, nếu có torch)")
    try:
        import torch  # noqa: F401
        e2 = experiments.new_experiment(cfg, name="mlp", hypothesis="kiểm tra DDP", template="torch_ddp")
        d2, _ = build_bundle(experiments.exp_path(cfg, e2), cfg=cfg, exp=e2, username="user1", smoke=True)
        m2 = run_bundle_locally(d2, data_dir)
        check(m2["ok"] and m2["cv"] is not None, f"torch smoke cv={m2['cv']}")
    except ImportError:
        print("  - bỏ qua (không có torch)")


if __name__ == "__main__":
    main()
