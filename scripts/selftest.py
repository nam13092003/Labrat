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
import time
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
os.environ["LABRAT_NO_SIBLINGS"] = "1"  # không cộng giờ của bot thật trên cùng máy
os.environ["LABRAT_QUOTA_FILE"] = str(Path(tempfile.mkdtemp()) / "quota.json")

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
    try:
        experiments.new_experiment(cfg, name="variant-x", hypothesis="biến thể", template="gbdt_folds",
                                   evidence="phân tích lỗi baseline: lát cắt X sai nhiều hơn 20%")
        check(False, "thiếu prior_work.md phải bị chặn")
    except SystemExit as e:
        check("prior_work.md" in str(e), "thí nghiệm thứ 2 khi chưa có prior_work.md → bị chặn")
    (comp_dir(cfg, SLUG) / "prior_work.md").write_text(
        "# Lời giải mạnh nhất đã công bố\n\n## Đội top kỳ trước\n- Phương pháp: GBDT với đặc trưng tương tác, CV 0.91 "
        "trên tập dev chính thức.\n- Cơ chế: tương tác giữa các cột phân loại hiếm.\n- Còn bỏ ngỏ: lệch train/test "
        "ở cột thời gian.\n- Nguồn: technical report kỳ trước.\n", encoding="utf-8")
    for i in range(3):
        ids.append(experiments.new_experiment(cfg, name=f"variant-{i}", hypothesis=f"biến thể {i}",
                                              template="gbdt_folds",
                                              evidence=f"phân tích lỗi baseline: lát cắt {i} sai nhiều hơn 20%")["id"])
    reg = experiments.registry_for(cfg)
    for e in reg.experiments:
        e["status"] = "smoke_ok"
    reg.save()
    try:
        experiments.push(cfg, ids[1], smoke=False, est_hours=1.5)
        check(False, "push GPU khi NOTES chưa có Dự đoán phải bị chặn")
    except SystemExit as e:
        check("Dự đoán" in str(e) and not pushed, "push GPU bị chặn khi NOTES.md chưa ghi Dự đoán (trước khi tạo kernel)")

    def fill_notes(eid, k):
        nf = experiments.exp_path(cfg, experiments.registry_for(cfg).get(eid)) / "NOTES.md"
        t = nf.read_text(encoding="utf-8")
        t = t.replace("**Cơ chế — vì sao kỳ vọng cải thiện:** ", f"**Cơ chế — vì sao kỳ vọng cải thiện:** cơ chế {k}")
        t = t.replace("**Dự đoán (nếu đúng thì CV / lát cắt nào đổi ra sao):** ",
                      f"**Dự đoán (nếu đúng thì CV / lát cắt nào đổi ra sao):** CV tăng ~0.00{k}, nhóm cat hiếm")
        t = t.replace("## Thay đổi so với cha\n\n- ", f"## Thay đổi so với cha\n\n- đổi learning rate {k}")
        nf.write_text(t, encoding="utf-8")
    for k, x in enumerate(ids[1:], 1):
        fill_notes(x, k)
    secs = experiments.notes_sections(cfg, experiments.registry_for(cfg).get(ids[1]))
    check(secs["prediction"].startswith("CV tăng") and secs["mechanism"] == "cơ chế 1"
          and secs["changes"] == "- đổi learning rate 1" and "lát cắt 0" in secs["evidence"],
          "đọc được Căn cứ / Cơ chế / Dự đoán / Thay đổi từ NOTES.md")
    from labrat import telegram as _tg
    sent_msgs = []
    _en, _sm = _tg.enabled, _tg.send_message
    _tg.enabled = lambda cfg=None: True
    _tg.send_message = lambda text, cfg=None, parse_mode="HTML": sent_msgs.append((text, len(pushed))) or True
    outs = [experiments.push(cfg, x, smoke=False, est_hours=1.5) for x in ids[:3]]
    _tg.enabled, _tg.send_message = _en, _sm
    check(len(sent_msgs) == 3 and [n for _, n in sent_msgs] == [0, 1, 2],
          "báo Telegram TRƯỚC mỗi lần đẩy GPU (trước khi kernel được tạo)")
    msg = sent_msgs[1][0]
    check("Giả thuyết" in msg and "Căn cứ" in msg and "Dự đoán" in msg and "CV tăng ~0.001" in msg
          and "đổi learning rate 1" in msg and "1.5h GPU" in msg, "nội dung báo trước: giả thuyết, căn cứ, thay đổi, dự đoán, giờ GPU")
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

    print("3b) chính sách nghiên cứu: chặn dò seed / bỏ phiếu, bắt buộc căn cứ")
    from labrat import policy
    n_before = len(experiments.registry_for(cfg).experiments)
    for nm, hy in [("best-seed", "chọn seed tốt nhất"), ("lgbm 5 seeds", "trung bình 5 seeds giảm dao động"),
                   ("vote3", "bỏ phiếu đa số 3 mô hình"), ("blend", "blend lgbm + catboost"),
                   ("ens", "ensemble OOF các mô hình tốt nhất"), ("x", "chạy nhiều seed rồi lấy trung bình")]:
        try:
            experiments.new_experiment(cfg, name=nm, hypothesis=hy, evidence="quan sát đủ dài để qua kiểm tra căn cứ")
            check(False, f"phải chặn: {nm} / {hy}")
        except SystemExit as e:
            check("từ chối" in str(e), f"chặn '{hy}'")
    try:
        experiments.new_experiment(cfg, name="focal-loss", hypothesis="focal loss cho lớp hiếm")
        check(False, "thiếu --evidence phải bị chặn")
    except SystemExit as e:
        check("--evidence" in str(e), "thí nghiệm thứ 2+ thiếu --evidence → bị chặn")
    for ok_hy in ["đổi loss cho khớp metric", "augmentation theo tính chất bất biến của dữ liệu", "seed cố định 42, đổi kiến trúc",
                  "majority class chiếm 80%", "test-time augmentation ensemble (TTA)", "mixup blending ảnh",
                  "EMA trọng số (model averaging)", "snapshot ensemble trong 1 lần huấn luyện", "bagging_fraction 0.8",
                  "stacking thêm 2 tầng transformer", "trung bình các fold"]:
        policy.check_idea(cfg, "ok", ok_hy)
    check(True, "kỹ thuật cải thiện MỘT mô hình (TTA, EMA, mixup, snapshot, bagging GBDT, majority class) không bị chặn")
    for bad_hy in ["voting classifier", "kết hợp nhiều mô hình", "ensemble of 3 backbones", "average of five models",
                   "stacking các mô hình bằng meta-learner", "5 seeds rồi lấy trung bình"]:
        try:
            policy.check_idea(cfg, "x", bad_hy); check(False, f"phải chặn: {bad_hy}")
        except SystemExit:
            pass
    check(True, "vẫn chặn ensemble nhiều mô hình / nhiều seed")
    check(len(experiments.registry_for(cfg).experiments) == n_before, "thí nghiệm bị chặn không vào registry")
    seed_job = {"mode": "parallel", "script": "src/train.py",
                "jobs": [{"name": "s1", "args": "--fold 0 --seed 1"}, {"name": "s2", "args": "--fold 0 --seed 2"}]}
    fold_job = {"mode": "parallel", "script": "src/train.py",
                "jobs": [{"name": f"f{k}", "args": f"--fold {k}"} for k in range(5)]}
    ab_job = {"mode": "parallel", "jobs": [{"name": "a", "args": "--loss mse"}, {"name": "b", "args": "--loss rank"}]}
    env_job = {"mode": "parallel", "jobs": [{"name": "a", "env": {"SEED": "1"}}, {"name": "b", "env": {"SEED": "2"}}]}
    for bad, label in [(seed_job, "2 job chỉ khác --seed"), (env_job, "2 job chỉ khác biến SEED"),
                       ({"args": "--n-seeds 5"}, "tham số --n-seeds")]:
        try:
            policy.check_job(cfg, bad)
            check(False, f"phải chặn job.yaml: {label}")
        except SystemExit:
            check(True, f"push chặn job.yaml: {label}")
    policy.check_job(cfg, fold_job); policy.check_job(cfg, ab_job)
    check(True, "job.yaml theo fold / A/B hai giả thuyết vẫn được")
    vdir = experiments.exp_path(cfg, experiments.registry_for(cfg).get(ids[3]))
    orig_job = (vdir / "job.yaml").read_text()
    (vdir / "job.yaml").write_text(yaml.safe_dump(seed_job))
    try:
        experiments.push(cfg, ids[3], smoke=True)
        check(False, "labrat push phải từ chối job dò seed")
    except SystemExit as e:
        check("seed" in str(e), "labrat push từ chối job.yaml dò seed trước khi tạo kernel")
    (vdir / "job.yaml").write_text(orig_job)
    cfg_off = dict(cfg, research={"enforce": False})
    policy.check_idea(cfg_off, "vote", "bỏ phiếu"); policy.check_job(cfg_off, seed_job)
    check(True, "research.enforce: false tắt được (chỉ khi người dùng yêu cầu)")

    print("4) poll giả lập → registry")
    finished = {outs[0]["kernel_ref"]}

    def fake_status(acc, ref):
        return "complete" if ref in finished else "running"

    seen_patterns = []

    def fake_output(acc, ref, dest, pattern=None):
        seen_patterns.append(pattern)
        dest.mkdir(parents=True, exist_ok=True)
        mm = dict(m, wall_seconds=3600 * 1.2, n_gpu=2,
                  gpu_util=[{"gpu": 0, "util_mean": 87.0, "util_max": 100, "mem_max_mb": 9200},
                            {"gpu": 1, "util_mean": 81.0, "util_max": 99, "mem_max_mb": 8800}])
        (dest / "metrics.json").write_text(json.dumps(mm))
        # CLI cũ không lọc được → giả lập có checkpoint 60 MB lọt về
        (dest / "outputs").mkdir(exist_ok=True)
        with open(dest / "outputs" / "checkpoint.pth", "wb") as f:
            f.truncate(60 * 1024 * 1024)
        return {"filtered": False, "files": sorted(dest.rglob("*"))}
    kaggle_ops.kernel_status, kaggle_ops.kernel_output = fake_status, fake_output
    events = experiments.poll(cfg)
    done = [e for e in events if e["status"] == "complete"]
    check(len(done) == 1 and done[0]["account"] == outs[0]["account"] and done[0]["cv"],
          f"1 xong ({done[0]['account']}, cv={done[0]['cv']:.4f}), 2 đang chạy")
    import re as _re
    pat = _re.compile(seen_patterns[0])
    check(all(pat.search(x) for x in ("metrics.json", "outputs/metrics_fold0.json", "logs/fold0.log",
                                      "outputs/oof_all.npz", "outputs/sub.csv"))
          and not any(pat.search(x) for x in ("outputs/checkpoint-30.pth", "outputs/tok_train_n7500.npz",
                                              "outputs/m2d_clap.zip")),
          "chỉ xin tải metrics/log/csv/oof — không tải checkpoint, cache .npz, zip")
    e0 = experiments.registry_for(cfg).get(outs[0]["exp"])
    big = experiments.exp_path(cfg, e0) / "out" / "gpu" / "outputs" / "checkpoint.pth"
    check(not big.exists() and e0["runs"][-1].get("output_pruned"),
          "file > 50 MB lọt về (CLI cũ) bị xoá ngay sau khi tải")
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
        "next_directions": [{"title": "Mã hoá mục tiêu cho cột phân loại hiếm", "why": "Sai số tập trung ở nhóm cat hiếm.",
                             "cost": "~3h GPU", "expected_gain": "+0.002"}]}, ensure_ascii=False))
    path = report.build(cfg)
    html = path.read_text()
    check("<svg" in html and "Tài khoản Kaggle" in html and "a3" in html, f"báo cáo: {path.relative_to(ROOT)}")

    print("5b) bản rẻ (--quick) + chốt kết quả nhiều seed (report-seeds)")
    import base64 as _b64, re as _re2
    cfg2 = dict(cfg, budget=dict(cfg.get("budget", {}), project_gpu_hours=1000), stop={"patience": 0})

    def spec_of(folder):
        txt = (Path(folder) / "main.py").read_text()
        return json.loads(_b64.b64decode(_re2.search(r"b64decode\('([^']+)'\)", txt).group(1)))["spec"]

    def fake_output2(acc, ref, dest, pattern=None):
        dest.mkdir(parents=True, exist_ok=True)
        bump = {"quick": -0.01, "seed43": 0.002, "seed44": -0.001}.get(dest.name, 0.0)
        (dest / "metrics.json").write_text(json.dumps(dict(m, cv=m["cv"] + bump, wall_seconds=1800, n_gpu=2)))
        return {"filtered": True, "files": []}
    kaggle_ops.kernel_output = fake_output2
    kaggle_ops.kernel_status = lambda acc, ref: "complete"
    qd, _ = build_bundle(experiments.exp_path(cfg, experiments.registry_for(cfg).get(ids[2])), cfg=cfg,
                         exp=experiments.registry_for(cfg).get(ids[2]), username="user1", smoke=False, quick=True)
    sp = spec_of(qd)
    check(sp["quick"] and sp["max_runtime_hours"] == 1.5 and sp["seed"] == 42, "bundle --quick: trần 1.5h, seed 42")
    sp_job = yaml.safe_load((experiments.exp_path(cfg, experiments.registry_for(cfg).get(ids[2])) / "job.yaml").read_text())
    mq = run_bundle_locally(qd, data_dir)
    check(mq["ok"] and mq["quick"] and len([j for j in mq["jobs"] if not j.get("post")]) == sp_job["quick_max_jobs"],
          f"runner --quick: chỉ chạy {sp_job['quick_max_jobs']} fold, cv={mq['cv']:.4f}")
    cv_before = experiments.registry_for(cfg).get(ids[2])["cv"]
    oq = experiments.push(cfg2, ids[2], smoke=False, quick=True)
    check(oq["kind"] == "gpu/quick" and oq["est_hours"] <= 1.5, f"push --quick (ước tính {oq['est_hours']}h)")
    experiments.poll(cfg2)
    eq = experiments.registry_for(cfg).get(ids[2])
    check(eq["status"] == "quick_ok" and abs(eq["quick_cv"] - (m["cv"] - 0.01)) < 1e-9 and eq["cv"] == cv_before,
          "kết quả bản rẻ vào quick_cv, không đè cv đầy đủ; output ở out/quick")
    check((experiments.exp_path(cfg, eq) / "out" / "quick" / "metrics.json").exists(), "output bản rẻ tách riêng")

    tgt = ids[1]
    et = experiments.registry_for(cfg).get(tgt)
    cv_t = et["cv"]
    trainf = experiments.exp_path(cfg, et) / "src" / "train.py"
    orig_train = trainf.read_text()
    trainf.write_text(orig_train + "\n# đổi cấu hình\n")
    try:
        experiments.report_seeds(cfg2, tgt, 3)
        check(False, "code đổi sau lần chạy chính → phải chặn report-seeds")
    except SystemExit as e:
        check("đã đổi" in str(e), "report-seeds chặn khi code đã đổi sau lần chạy chính (không còn là cấu hình chốt)")
    trainf.write_text(orig_train)
    for bad_n in (1, 9):
        try:
            experiments.report_seeds(cfg2, tgt, bad_n)
            check(False, "n ngoài khoảng phải bị chặn")
        except SystemExit:
            pass
    check(True, "số seed ngoài 2..5 bị chặn")
    n_push = len(pushed)
    rs = experiments.report_seeds(cfg2, tgt, 3)
    check(sorted(x["seed"] for x in rs["pushed"]) == [43, 44] and len({x["account"] for x in rs["pushed"]}) == 2,
          f"report-seeds: seed gốc 42 dùng lại lần chạy chính, đẩy seed 43/44 lên 2 tài khoản {[x['account'] for x in rs['pushed']]}")
    check(len(pushed) == n_push + 2 and experiments.registry_for(cfg).get(tgt)["status"] == "reporting",
          "trạng thái reporting trong lúc chạy")
    busy = [a["name"] for a in budget.accounts_status(cfg2) if a["gpu_busy"]]
    check(len(busy) == 2, f"tài khoản chạy seed được tính là bận GPU: {busy}")
    sd, _ = build_bundle(experiments.exp_path(cfg, et), cfg=cfg, exp=et, username="user1", smoke=False, seed=43)
    check(spec_of(sd)["seed"] == 43, "bundle truyền seed → runner xuất LABRAT_SEED")
    evs = experiments.poll(cfg2)
    er = experiments.registry_for(cfg).get(tgt)
    rep = er.get("report") or {}
    check(er["status"] == "complete" and rep.get("n") == 3 and rep["seeds"] == [42, 43, 44] and er["cv"] == cv_t,
          f"đủ seed → complete; báo cáo {rep.get('mean', 0):.4f} ± {rep.get('std', 0):.4f}, cv chính giữ nguyên")
    check(any(ev["status"] == "report_done" for ev in evs), "sự kiện report_done (báo Telegram một lần)")
    check((experiments.exp_path(cfg, er) / "out" / "seed43" / "metrics.json").exists(), "output mỗi seed tách riêng")
    rs2 = experiments.report_seeds(cfg2, tgt, 3)
    check(not rs2["pushed"], "gọi lại khi đã đủ seed → không đẩy thêm")
    check("seed:" in report.build(cfg).read_text(), "thẻ báo cáo hiện mean ± std nhiều seed")

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

    print("7a) giữ ổ đĩa gọn")
    from labrat import housekeeping
    # runner trong kernel: file lớn bị chuyển khỏi output, file giữ lại theo keep_outputs
    work = Path(tempfile.mkdtemp(prefix="labrat_work_")); scr = Path(tempfile.mkdtemp(prefix="labrat_scr_"))
    (work / "outputs").mkdir()
    for name, mb in (("outputs/pretrained.pth", 3), ("outputs/best_model.pt", 3), ("outputs/oof.npz", 0.1)):
        with open(work / name, "wb") as f:
            f.truncate(int(mb * 1024 * 1024))
    ns = {}
    exec(compile((ROOT / "templates" / "runner.py").read_text(), "runner", "exec"), ns)
    pr = ns["_labrat_prune_outputs"](work, scr, 1, ["outputs/best_model.pt"])
    check([x["path"] for x in pr] == ["outputs/pretrained.pth"] and (work / "outputs/best_model.pt").exists()
          and (work / "outputs/oof.npz").exists() and (scr / "pruned/outputs/pretrained.pth").exists(),
          "runner: file lớn chuyển sang scratch, giữ best_model.pt (keep_outputs) và oof nhỏ")
    shutil.rmtree(work); shutil.rmtree(scr)
    # mọi lệnh kaggle có TMPDIR riêng và được xoá
    tmp_seen = {}
    real_sub_run = kaggle_ops.subprocess.run

    def spy(cmd, **kw):
        tmp_seen["dir"] = kw["env"]["TMPDIR"]
        Path(kw["env"]["TMPDIR"], "xacle-data.tar").write_bytes(b"x" * 1000)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")
    kaggle_ops.subprocess.run = spy
    kaggle_ops._bin = lambda: "kaggle"
    try:
        kaggle_ops.run(["datasets", "version"], accounts.load_accounts(cfg)[0])
    finally:
        kaggle_ops.subprocess.run = real_sub_run
    check(tmp_seen["dir"].startswith(str(kaggle_ops.TMP_ROOT)) and not Path(tmp_seen["dir"]).exists(),
          "lệnh kaggle dùng thư mục tạm riêng, file nén tạm bị xoá sau lệnh")
    # clean: bundle + output lớn + tmp bị bỏ lại; không đụng data/
    stale = kaggle_ops.TMP_ROOT / "dead-1"; stale.mkdir(parents=True, exist_ok=True)
    (stale / "left.tar").write_bytes(b"0" * 2048)
    old = time.time() - 5 * 3600
    os.utime(stale / "left.tar", (old, old)); os.utime(stale, (old, old))
    keep_data = comp_dir(cfg, SLUG) / "data" / "big.bin"; keep_data.parent.mkdir(parents=True, exist_ok=True)
    with open(keep_data, "wb") as f:
        f.truncate(80 * 1024 * 1024)
    ob = experiments.exp_path(cfg, e0) / "out" / "gpu" / "outputs" / "cache.npz"
    with open(ob, "wb") as f:
        f.truncate(70 * 1024 * 1024)
    dry = housekeeping.clean(cfg, dry_run=True)
    check(ob.exists() and dry["freed_gb"] >= 0.07, f"clean --dry-run: báo {dry['freed_gb']} GB, chưa xoá gì")
    r = housekeeping.clean(cfg)
    check(not ob.exists() and not stale.exists() and keep_data.exists()
          and not list(comp_dir(cfg, SLUG).glob("experiments/*/_bundle_*")),
          f"clean: xoá output lớn, bundle, tmp bỏ lại ({r['n_items']} mục, {r['freed_gb']} GB); giữ nguyên data/")
    keep_data.unlink()
    st = bridge.quick_status()
    check("💾 Ổ đĩa" in st and "GB" in bridge.handle_quick("/disk") and "🧹" in bridge.handle_quick("/clean"),
          "/status có dung lượng ổ; /disk và /clean trên Telegram")

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

    print("8c) gọi model khác qua OpenRouter (giả lập API)")
    from labrat import llm
    tmpd = Path(tempfile.mkdtemp())
    llm.LEDGER, llm.CACHE = tmpd / "usage.jsonl", tmpd / "models.json"
    os.environ["OPENROUTER_API_KEY"] = "sk-or-fake"
    fake_models = [
        {"id": "cheap/text-mini", "context_length": 128000, "pricing": {"prompt": "0.0000001", "completion": "0.0000004"},
         "architecture": {"input_modalities": ["text"]}, "top_provider": {"max_completion_tokens": 8000}},
        {"id": "vision/flash", "context_length": 1000000, "pricing": {"prompt": "0.0000003", "completion": "0.0000025"},
         "architecture": {"input_modalities": ["text", "image", "file"]}, "top_provider": {"max_completion_tokens": 64000}},
        {"id": "big/expensive", "context_length": 200000, "pricing": {"prompt": "0.00006", "completion": "0.00024"},
         "architecture": {"input_modalities": ["text", "image"]}, "top_provider": {}},
    ]
    sent = []
    llm._get = lambda path, timeout=60: ({"data": fake_models} if path == "/models" else
                                         {"data": {"limit": 10, "limit_remaining": 7.5, "usage": 2.5}})

    def fake_post(path, body, timeout=600):
        sent.append(body)
        if "LỖI" in json.dumps(body, ensure_ascii=False):
            return {"choices": [{"error": {"code": 502, "message": "provider down"}}]}
        u = {"prompt_tokens": 1200, "completion_tokens": 300}
        if body.get("response_format"):
            return {"model": body["model"], "choices": [{"message": {"content": '{"label": "pos"}'},
                                                           "finish_reason": "stop"}], "usage": u}
        if body["model"] != "cheap/text-mini":
            u["cost"] = 0.0021
        return {"model": body["model"], "choices": [{"message": {"content": "Có overfit từ epoch 7."},
                                                       "finish_reason": "stop"}], "usage": u}
    llm._post = fake_post
    try:
        # key chỉ có trong .env (chưa nạp vào biến môi trường) → llm vẫn phải đọc được
        from labrat import config as _cfgmod
        fake_root = tmpd / "root"; fake_root.mkdir()
        (fake_root / ".env").write_text("OPENROUTER_API_KEY=sk-or-from-dotenv\n")
        saved_root, saved_key = _cfgmod.ROOT, os.environ.pop("OPENROUTER_API_KEY")
        _cfgmod.ROOT = fake_root
        try:
            check(llm._key() == "sk-or-from-dotenv", "key chỉ nằm trong .env vẫn được đọc (llm models / balance)")
        finally:
            _cfgmod.ROOT = saved_root
            os.environ["OPENROUTER_API_KEY"] = saved_key
        ms = llm.models(modality="image")
        check([x["id"] for x in ms] == ["vision/flash", "big/expensive"] and ms[0]["usd_per_M_in"] == 0.3,
              "models: lọc theo ảnh, rẻ nhất trước, giá USD/1M token")
        check([x["id"] for x in llm.models(max_price=1)] == ["cheap/text-mini", "vision/flash"], "models: lọc theo giá")
        img = tmpd / "loss.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 100)
        try:
            llm.ask("xem ảnh", "cheap/text-mini", files=[str(img)])
            check(False, "model chỉ đọc chữ phải từ chối ảnh")
        except llm.LLMError as e:
            check("không nhận ảnh" in str(e), "model không có thị giác → từ chối ảnh, gợi ý lọc --modality image")
        r = llm.ask("Loss có overfit không?", "vision/flash", files=[str(img)], system="Bạn là chuyên gia ML",
                    out=str(tmpd / "ans.md"), purpose="exp-007 overfit")
        b = sent[-1]
        parts = b["messages"][1]["content"]
        check(b["messages"][0]["role"] == "system" and parts[0]["type"] == "text"
              and parts[1]["image_url"]["url"].startswith("data:image/png;base64,"),
              "gửi ảnh dạng data URL + system prompt")
        check(r["cost_usd"] == 0.0021 and (tmpd / "ans.md").read_text().startswith("Có overfit"),
              "ghi chi phí thật (usage.cost) và lưu câu trả lời ra --out")
        notes = tmpd / "log.txt"
        notes.write_text("epoch 1 loss 0.9\n" * 50)
        r2 = llm.ask("tóm tắt log", "cheap/text-mini", files=[str(notes)])
        check("FILE: log.txt" in sent[-1]["messages"][0]["content"][0]["text"]
              and abs(r2["cost_usd"] - (1200 * 1e-7 + 300 * 4e-7)) < 1e-9,
              "file văn bản được chèn vào prompt; thiếu usage.cost → tự tính theo bảng giá")
        try:
            from reportlab.pdfgen import canvas as _cv
            pdf = tmpd / "paper.pdf"
            c = _cv.Canvas(str(pdf)); c.drawString(40, 800, "Table 2: results"); c.showPage(); c.save()
            llm.ask("đọc bảng", "big/expensive", files=[str(pdf)], max_tokens=500)
            fb = sent[-1]
            check(fb["messages"][0]["content"][1]["type"] == "file"
                  and fb["messages"][0]["content"][1]["file"]["file_data"].startswith("data:application/pdf;base64,")
                  and fb.get("plugins") == [{"id": "file-parser", "pdf": {"engine": "cloudflare-ai"}}],
                  "PDF: gửi part 'file'; model không đọc PDF trực tiếp → dùng parser cloudflare-ai (miễn phí)")
        except ImportError:
            print("  - bỏ qua kiểm tra PDF (không có reportlab)")
        try:
            llm.ask("phân tích sâu", "big/expensive", max_tokens=8000)
            check(False, "phải chặn khi ước tính vượt trần mỗi lần")
        except llm.LLMError as e:
            check("trần mỗi lần" in str(e), "chặn trước khi gọi nếu ước tính vượt trần $0.5/lần")
        with open(llm.LEDGER, "a") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "model": "x", "cost": 2.999}) + "\n")
        try:
            llm.ask("thêm một câu", "vision/flash")
            check(False, "phải chặn khi vượt trần ngày")
        except llm.LLMError as e:
            check("trần ngày" in str(e), "chặn khi tổng chi tiêu trong ngày vượt trần $3")
        n_before = len(sent)
        llm.ask("thêm một câu", "vision/flash", allow_over_cap=True)
        check(len(sent) == n_before + 1, "--allow-over-cap (khi người dùng đồng ý) cho phép gọi")
        try:
            llm.ask("LỖI", "vision/flash", allow_over_cap=True)
            check(False, "lỗi provider phải báo")
        except llm.LLMError as e:
            check("provider down" in str(e), "lỗi từ provider được báo rõ")
        sp = llm.spend()
        check(sp["today_usd"] > 3 and "vision/flash" in sp["by_model_month"], f"spend: hôm nay ${sp['today_usd']}")
        check("OpenRouter: hôm nay $" in bridge.handle_quick("/usage"), "/usage hiện chi tiêu OpenRouter")
        check(llm.balance()["limit_remaining"] == 7.5, "balance: tín dụng còn lại của key")

        print("8d) LLM làm phương pháp thực nghiệm: batch zero/few-shot, tăng cường, dịch")
        llm.CACHE_DIR = tmpd / "llmcache"
        vdata = tmpd / "val.jsonl"
        vdata.write_text("".join(json.dumps({"id": f"r{i}", "text": f"câu số {i}", "label": "pos"}) + "\n"
                                for i in range(30)))
        shots = tmpd / "shots.jsonl"
        shots.write_text(json.dumps({"text": "tuyệt vời", "output": '{"label": "pos"}'}) + "\n"
                         + json.dumps({"text": "tệ quá", "output": '{"label": "neg"}'}) + "\n")
        tpl = "Phân loại cảm xúc: {text}"
        n0 = len(sent)
        d = llm.batch(str(vdata), tpl, "cheap/text-mini", str(tmpd / "zs.jsonl"), dry_run=True)
        check(d["rows_todo"] == 30 and d["estimate_upper_usd"] > 0 and "câu số 0" in d["example_prompt"]
              and len(sent) == n0, f"--dry-run: ước tính ${d['estimate_upper_usd']} cho 30 dòng, không gọi API")
        r = llm.batch(str(vdata), tpl, "cheap/text-mini", str(tmpd / "zs.jsonl"), examples=str(shots),
                      system="Trả lời JSON", json_mode=True, limit=10, purpose="zero-shot thử")
        b = sent[-1]
        roles = [m["role"] for m in b["messages"]]
        check(r["processed_now"] == 10 and roles == ["system", "user", "assistant", "user", "assistant", "user"]
              and b["messages"][-1]["content"].startswith("Phân loại cảm xúc: câu số"),
              "few-shot: ví dụ thành cặp user/assistant trước câu hỏi; chạy 10 dòng (--limit)")
        check(b["provider"]["data_collection"] == "deny" and b["temperature"] == 0.0 and b["seed"] == 42,
              "mặc định: nhà cung cấp không lưu dữ liệu, temperature 0, seed 42")
        outs_ = [json.loads(l) for l in (tmpd / "zs.jsonl").read_text().splitlines()]
        check(all(o["parsed"] == {"label": "pos"} for o in outs_) and Path(str(tmpd / "zs.jsonl") + ".meta.json").exists(),
              "--json: kết quả được parse; lưu .meta.json để tái lập")
        n1 = len(sent)
        r = llm.batch(str(vdata), tpl, "cheap/text-mini", str(tmpd / "zs.jsonl"), examples=str(shots),
                      system="Trả lời JSON", json_mode=True, limit=10)
        check(r["rows_todo"] == 0 and len(sent) == n1, "chạy lại: dòng đã xong được bỏ qua (tiếp tục được)")
        (tmpd / "zs.jsonl").unlink()
        r = llm.batch(str(vdata), tpl, "cheap/text-mini", str(tmpd / "zs.jsonl"), examples=str(shots),
                      system="Trả lời JSON", json_mode=True, limit=10)
        check(r["from_cache"] == 10 and r["cost_usd"] == 0 and len(sent) == n1,
              "xoá kết quả rồi chạy lại: lấy từ cache, không trả tiền lần hai")
        r = llm.batch(str(vdata), tpl, "vision/flash", str(tmpd / "aug.jsonl"), budget_usd=0.01, purpose="tăng cường")
        check(r["stopped"] and r["processed_now"] <= 5 and not r["complete"],
              f"vượt ngân sách: chạy thử {r['processed_now']} dòng, dự kiến ${r['projected_total_usd']}, dừng chờ duyệt")
        try:
            llm.batch(str(vdata), tpl, "vision/flash", str(tmpd / "aug.jsonl"), budget_usd=5)
            check(False, "ngân sách lớn phải cần --approved")
        except llm.LLMError as e:
            check("--approved" in str(e), "ngân sách > $2 mà chưa duyệt → từ chối")
        try:
            llm.batch(str(vdata), tpl, "vision/flash", str(tmpd / "aug.jsonl"), approved=True)
            check(False, "--approved phải kèm --budget-usd")
        except llm.LLMError as e:
            check("--budget-usd" in str(e), "--approved bắt buộc kèm --budget-usd")
        r = llm.batch(str(vdata), tpl, "vision/flash", str(tmpd / "aug.jsonl"), budget_usd=1, approved=True)
        check(r["complete"] and r["rows_with_output"] == 30, "đã duyệt: chạy tiếp từ chỗ dừng, đủ 30 dòng")
        try:
            llm.batch(str(vdata), "Dịch: {noi_dung}", "cheap/text-mini", str(tmpd / "tr.jsonl"), limit=2)
            check(False, "template sai cột phải báo")
        except llm.LLMError as e:
            check("noi_dung" in str(e) and "text" in str(e), "template dùng cột không có → báo rõ các cột hiện có")
        cfgr = load_config(SLUG)
        gpu_before = budget.summary(cfgr, experiments.registry_for(cfgr))["project_used"]
        rec = experiments.record_result(cfgr, name="zero-shot cheap-mini", hypothesis="LLM zero-shot làm baseline",
                                        cv=0.71, method="zero-shot", cost_usd=0.003,
                                        artifacts=[str(tmpd / "zs.jsonl")])
        regr = experiments.registry_for(cfgr)
        check(any(e["id"] == rec["id"] for e in regr.completed())
              and budget.summary(cfgr, regr)["project_used"] == gpu_before,
              "labrat record: kết quả zero-shot vào sổ thí nghiệm, không tính vào giờ GPU")
        check("zero-shot cheap-mini" in report.build(cfgr).read_text(), "kết quả LLM hiện trong thẻ báo cáo")
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)

    print("9) template torch_ddp (smoke, nếu có torch)")
    try:
        import torch  # noqa: F401
        e2 = experiments.new_experiment(cfg, name="mlp", hypothesis="kiểm tra DDP", template="torch_ddp",
                                         evidence="EDA: quan hệ phi tuyến giữa đặc trưng, thử mạng nơ-ron")
        d2, _ = build_bundle(experiments.exp_path(cfg, e2), cfg=cfg, exp=e2, username="user1", smoke=True)
        m2 = run_bundle_locally(d2, data_dir)
        check(m2["ok"] and m2["cv"] is not None, f"torch smoke cv={m2['cv']}")
    except ImportError:
        print("  - bỏ qua (không có torch)")

    print("10) giờ GPU (hiệu chỉnh, bot dùng chung), phiên định kỳ, tải trên Kaggle, checkpoint của cha")
    cfg = load_config(SLUG)
    now_utc = datetime.now(timezone.utc)
    budget.set_calibration("user1", 20.0)
    a1 = next(r for r in budget.accounts_status(cfg) if r["name"] == "a1")
    reserve = float(cfg["budget"].get("reserve_hours", 1.0))
    check(abs(a1["week_used"] - 20.0) < 1e-6 and abs(a1["week_left"] - (a1["weekly_gpu_hours"] - 20 - reserve)) < 1e-6,
          f"hiệu chỉnh theo kaggle.com: a1 = 20h đã dùng (các lần chạy trước mốc hiệu chỉnh không cộng thêm)")
    calf = Path(os.environ["LABRAT_QUOTA_FILE"])
    cal = json.loads(calf.read_text())
    cal["user1"]["at"] = (budget.kaggle_week_start() - timedelta(days=1)).isoformat()
    calf.write_text(json.dumps(cal))
    a1b = next(r for r in budget.accounts_status(cfg) if r["name"] == "a1")
    check(not a1b.get("calibrated"), "hiệu chỉnh từ tuần trước bị bỏ qua (quota đã reset)")
    check("✅" in bridge.handle_quick("/quota a1 7.5") and json.loads(calf.read_text())["user1"]["used"] == 7.5,
          "/quota a1 7.5 trên Telegram ghi hiệu chỉnh")
    check("Không có tài khoản" in bridge.handle_quick("/quota zz 3"), "/quota tài khoản sai → báo lỗi, không crash")
    calf.unlink()

    other = Path(tempfile.mkdtemp()) / "Labrat9" / "competitions"
    (other / "p").mkdir(parents=True)
    t_now = now_utc.isoformat()
    (other / "p" / "registry.json").write_text(json.dumps({"experiments": [
        {"id": "exp-001", "status": "complete", "runs": [
            {"kind": "gpu", "account": "x", "kernel_ref": "user2/k1", "pushed_at": t_now, "finished_at": t_now,
             "gpu_hours": 3.0}]},
        {"id": "exp-002", "status": "running", "runs": [
            {"kind": "gpu", "account": "y", "kernel_ref": "user3/k2", "pushed_at": t_now, "est_hours": 2.0}]},
        {"id": "exp-003", "status": "complete", "runs": [
            {"kind": "gpu", "account": "z", "kernel_ref": "stranger/k3", "pushed_at": t_now, "finished_at": t_now,
             "gpu_hours": 9.0}]}]}))
    cfgS = dict(cfg, budget=dict(cfg["budget"], quota_scan_roots=[str(other)]))
    rows = {r["name"]: r for r in budget.accounts_status(cfgS)}
    base_rows = {r["name"]: r for r in budget.accounts_status(cfg)}
    check(abs(rows["a2"]["other_bots_h"] - 3.0) < 1e-6
          and abs(rows["a2"]["week_used"] - base_rows["a2"]["week_used"] - 3.0) < 1e-6,
          "giờ của bot khác dùng chung tài khoản (theo username) được cộng vào")
    check(rows["a3"]["gpu_busy"] and "Labrat9" in rows["a3"]["gpu_busy"],
          f"kernel GPU đang chạy của bot khác → tài khoản bận: {rows['a3']['gpu_busy']}")
    check(sum(r["other_bots_h"] for r in rows.values()) < 6, "tài khoản lạ (username khác) không bị tính")

    import importlib.util
    spec = importlib.util.spec_from_file_location("tick_all", ROOT / "scripts" / "tick_all.py")
    tick = importlib.util.module_from_spec(spec); spec.loader.exec_module(tick)
    base = {"autowork": {"minutes": 60}}
    nowt = time.time()
    check(tick.autowork_due(base, cfg, "p", {}, None, set(), nowt), "phiên định kỳ: lần đầu → đánh thức")
    check(not tick.autowork_due(base, cfg, "p", {"p": nowt - 600}, None, set(), nowt), "vừa có sự kiện 10 phút trước → chưa")
    check(tick.autowork_due(base, cfg, "p", {"auto:p": nowt - 3700}, None, set(), nowt), "đã > 60 phút → đánh thức")
    check(not tick.autowork_due(base, dict(cfg, paused=True), "p", {}, None, set(), nowt), "dự án tạm dừng → không")
    check(not tick.autowork_due(base, cfg, "p", {}, None, {"p"}, nowt), "đã có lượt chờ trong hàng đợi → không chồng thêm")
    check(not tick.autowork_due(base, cfg, "p", {}, {"stop": True}, set(), nowt), "hết ngân sách / dừng → không")
    check(not tick.autowork_due({"autowork": {"minutes": 0}}, cfg, "p", {}, None, set(), nowt), "minutes: 0 → tắt")
    check("[im lặng]" in tick.autowork_message("p", None), "lời nhắc định kỳ cho phép trả lời im lặng")

    from labrat import kfetch
    compile(kfetch._script("deb", "microsoft/deberta-v3-base", "model", ["*.json"], ["https://x/y.bin"], "echo hi"),
            "fetch.py", "exec")
    n_push = len(pushed)
    kf = kfetch.create(cfg, "deberta", hf="microsoft/deberta-v3-base", include=["*.safetensors"])
    metas = [m for _, _, m in pushed[n_push:]]
    check(len(metas) == 3 and all(not m["enable_gpu"] and m["enable_internet"] for m in metas)
          and {m["id"].split("/")[0] for m in metas} == {"user1", "user2", "user3"},
          "kaggle-fetch: một kernel CPU có internet trên MỖI tài khoản (không tốn quota GPU)")
    check(not list((ROOT / ".labrat_tmp").glob("kfetch-*")), "không để lại thư mục tạm trên VPS")
    cfg = load_config(SLUG)
    st = kfetch.status(cfg, "deberta")
    check(st["deberta"]["ready"], "kaggle-fetch-status: complete trên mọi tài khoản → ready")
    cfg = load_config(SLUG)
    ex = experiments.registry_for(cfg).get(ids[2])
    jf = experiments.exp_path(cfg, ex) / "job.yaml"
    orig = jf.read_text()
    jd = yaml.safe_load(orig); jd["fetched"] = ["deberta"]; jf.write_text(yaml.safe_dump(jd))
    bd, _ = build_bundle(experiments.exp_path(cfg, ex), cfg=cfg, exp=ex, username="user2", smoke=False)
    ks = json.loads((bd / "kernel-metadata.json").read_text())["kernel_sources"]
    check(ks == [f"user2/{kf['title']}"], f"job.yaml fetched → kernel_sources đúng tài khoản: {ks}")
    jd["fetched"] = ["chua-tai"]; jf.write_text(yaml.safe_dump(jd))
    try:
        build_bundle(experiments.exp_path(cfg, ex), cfg=cfg, exp=ex, username="user2", smoke=False)
        check(False, "fetched chưa tải phải báo lỗi")
    except SystemExit as e:
        check("kaggle-fetch" in str(e), "fetched chưa tải → báo lỗi kèm lệnh cần chạy")
    jf.write_text(orig)

    par = experiments.registry_for(cfg).get(ids[1])
    pmain = experiments._main_run(par)
    child = experiments.new_experiment(cfg, name="finetune-tiep", hypothesis="huấn luyện tiếp từ checkpoint của cha",
                                       parent=ids[1], evidence="learning curve exp cha chưa bão hoà ở epoch cuối")
    cjf = experiments.exp_path(cfg, child) / "job.yaml"
    cj = yaml.safe_load(cjf.read_text()); cj["parent_outputs"] = True; cjf.write_text(yaml.safe_dump(cj))
    fill_notes(child["id"], 7)
    rg = experiments.registry_for(cfg); rg.get(child["id"])["status"] = "smoke_ok"; rg.save()
    other_acc = next(a for a in ("a1", "a2", "a3") if a != pmain["account"])
    try:
        experiments.push(cfg2, child["id"], smoke=False, account=other_acc, est_hours=1.0)
        check(False, "chạy con trên tài khoản khác cha phải bị chặn")
    except SystemExit as e:
        check(pmain["account"] in str(e), "parent_outputs: tài khoản khác cha → bị chặn")
    oc = experiments.push(cfg2, child["id"], smoke=False, est_hours=1.0, force=True)
    meta = pushed[-1][2]
    check(oc["account"] == pmain["account"] and pmain["kernel_ref"] in meta["kernel_sources"],
          f"parent_outputs: tự chạy trên tài khoản của cha ({oc['account']}), gắn output kernel cha")


if __name__ == "__main__":
    main()
