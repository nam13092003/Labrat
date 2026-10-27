# ============================================================================
# LABRAT RUNNER — chạy BÊN TRONG kernel Kaggle.
# Được bundler ghép sau phần payload (LABRAT_FILES, LABRAT_SPEC). Chỉ dùng thư viện chuẩn.
#
# Mục tiêu: tận dụng cả 2 GPU T4 trong một phiên (quota tính theo thời gian phiên).
#   mode: parallel  → hàng đợi job, mỗi GPU một "slot", 2 job chạy đồng thời
#                     (CUDA_VISIBLE_DEVICES=0 và =1). Hợp cho: fold CV, seed, biến thể
#                     siêu tham số, GBDT.
#   mode: ddp       → 1 job huấn luyện phân tán bằng torchrun --nproc_per_node=<số GPU>.
#                     Hợp cho: một mô hình DL lớn cần cả 2 GPU.
#
# Hợp đồng với script huấn luyện (biến môi trường):
#   LABRAT_METRICS_PATH  nơi script ghi JSON kết quả của job, ví dụ {"cv": 0.81, ...}
#   LABRAT_OUTPUT_DIR    thư mục output dùng chung (oof, checkpoint, ...)
#   LABRAT_DEADLINE      epoch giây — quá mốc này phải lưu checkpoint và thoát
#   LABRAT_SMOKE         "1" khi chạy thử trên CPU với dữ liệu nhỏ
#   LABRAT_N_GPUS, LABRAT_JOB_NAME, LABRAT_GPU_SLOT
#   LABRAT_SCRATCH    thư mục tạm KHÔNG được lưu làm output (mô hình pretrained, cache đặc trưng...)
#
# Cuối phiên, file > output_max_mb trong /kaggle/working bị chuyển sang LABRAT_SCRATCH (trừ khi khớp
# job.yaml `keep_outputs`) để Kaggle không lưu chúng làm output và VPS không phải tải về.
# ============================================================================
import json as _json
import os as _os
import queue as _queue
import shlex as _shlex
import signal as _signal
import subprocess as _sp
import sys as _sys
import threading as _th
import time as _time
from pathlib import Path as _Path


def _labrat_workdir():
    w = _os.environ.get("LABRAT_WORKDIR")
    if w:
        return _Path(w)
    return _Path("/kaggle/working") if _Path("/kaggle/working").exists() else _Path.cwd() / "_local_run"


def _labrat_gpus():
    try:
        out = _sp.run(["nvidia-smi", "--query-gpu=index,name,memory.total",
                       "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({"index": int(parts[0]), "name": parts[1], "memory_mb": int(float(parts[2]))})
        return gpus
    except Exception:
        return []


class _GpuSampler(_th.Thread):
    """Lấy mẫu % sử dụng từng GPU mỗi 30s để kiểm chứng cả 2 GPU đều làm việc."""

    def __init__(self, n, every=30):
        super().__init__(daemon=True)
        self.n, self.every, self.samples, self._stop_evt = n, every, [], _th.Event()

    def run(self):
        while self.n and not self._stop_evt.wait(self.every):
            try:
                out = _sp.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True,
                              timeout=20).stdout
                row = [[float(x) for x in l.split(",")] for l in out.strip().splitlines()]
                self.samples.append(row)
            except Exception:
                pass

    def stop(self):
        self._stop_evt.set()

    def summary(self):
        if not self.samples:
            return []
        res = []
        for i in range(self.n):
            util = [s[i][0] for s in self.samples if len(s) > i]
            mem = [s[i][1] for s in self.samples if len(s) > i]
            if util:
                res.append({"gpu": i, "util_mean": round(sum(util) / len(util), 1),
                            "util_max": max(util), "mem_max_mb": max(mem)})
        return res


def _labrat_prune_outputs(work, scratch, max_mb, keep):
    """Chuyển file > max_mb trong thư mục làm việc sang scratch (không phải output) trừ khi khớp keep."""
    import fnmatch as _fn
    import shutil as _sh
    pruned = []
    if max_mb <= 0:
        return pruned
    limit = max_mb * 1024 * 1024
    for p in sorted(work.rglob("*")):
        try:
            if not p.is_file() or p.is_symlink() or p.stat().st_size <= limit:
                continue
            rel = p.relative_to(work).as_posix()
            if any(_fn.fnmatch(rel, k) or _fn.fnmatch(p.name, k) for k in keep):
                continue
            mb = round(p.stat().st_size / 1048576, 1)
            dst = scratch / "pruned" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            _sh.move(str(p), str(dst))
            pruned.append({"path": rel, "mb": mb})
        except Exception as e:  # không để việc dọn làm hỏng kết quả
            print(f"[labrat] không chuyển được {p}: {e}")
    return pruned


def _labrat_tail(path, n=80):
    try:
        return "\n".join(_Path(path).read_text(errors="replace").splitlines()[-n:])
    except Exception:
        return ""


def _labrat_run(name, cmd, env, cwd, log_path, hard_deadline):
    t0 = _time.time()
    with open(log_path, "w") as log:
        log.write(f"$ {cmd}\n")
        log.flush()
        p = _sp.Popen(cmd, shell=True, cwd=cwd, env=env, stdout=log, stderr=_sp.STDOUT,
                      start_new_session=True)
        killed = False
        while p.poll() is None:
            if _time.time() > hard_deadline and not killed:
                log.write("\n[labrat] quá hạn cứng — gửi SIGTERM\n")
                log.flush()
                _os.killpg(p.pid, _signal.SIGTERM)
                killed = True
                try:
                    p.wait(timeout=120)
                except _sp.TimeoutExpired:
                    _os.killpg(p.pid, _signal.SIGKILL)
            _time.sleep(2)
    return {"name": name, "cmd": cmd, "returncode": p.returncode, "killed": killed,
            "seconds": round(_time.time() - t0, 1)}


def _labrat_cmd(script, args, launcher):
    return f"{launcher} {_shlex.quote(script)} {args or ''}".strip()


def labrat_main(files, spec):
    t0 = _time.time()
    work = _labrat_workdir()
    src, logs, outs = work / "src", work / "logs", work / "outputs"
    for d in (src, logs, outs):
        d.mkdir(parents=True, exist_ok=True)

    import base64 as _b64
    for rel, b64 in files.items():
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(_b64.b64decode(b64))

    job = spec["job"]
    smoke = bool(spec.get("smoke"))
    direction = spec.get("metric_direction", job.get("metric_direction", "maximize"))
    gpus = [] if smoke else _labrat_gpus()
    n_gpu = len(gpus)
    max_h = float(spec.get("max_runtime_hours", 9))
    margin = float(spec.get("safety_margin_minutes", 20)) * 60
    soft_deadline = t0 + max_h * 3600 - margin          # script nên tự dừng trước mốc này
    hard_deadline = t0 + max_h * 3600 - margin / 3      # runner sẽ SIGTERM sau mốc này

    print(f"[labrat] exp={spec.get('exp_id')} smoke={smoke} gpus={n_gpu} {[g['name'] for g in gpus]}")
    if not smoke and n_gpu < int(job.get("expect_gpus", 2)):
        print(f"[labrat] CẢNH BÁO: chỉ thấy {n_gpu} GPU, mong đợi {job.get('expect_gpus', 2)}")

    base_env = _os.environ.copy()
    scratch = _Path(_os.environ.get("LABRAT_SCRATCH_DIR", "/tmp/labrat_scratch"))
    scratch.mkdir(parents=True, exist_ok=True)
    base_env.update({
        "LABRAT_SCRATCH": str(scratch),
        "LABRAT_SMOKE": "1" if smoke else "0",
        "LABRAT_DEADLINE": str(int(soft_deadline)),
        "LABRAT_OUTPUT_DIR": str(outs),
        "LABRAT_N_GPUS": str(n_gpu),
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    base_env.update({k: str(v) for k, v in (job.get("env") or {}).items()})

    pip = job.get("pip") or []
    if pip:
        r = _sp.run([_sys.executable, "-m", "pip", "install", "-q", *pip], capture_output=True, text=True)
        print("[labrat] pip:", "ok" if r.returncode == 0 else r.stderr[-2000:])

    sampler = _GpuSampler(n_gpu)
    sampler.start()
    results = []
    mode = job.get("mode", "parallel")

    if mode == "ddp":
        script, args = job["script"], job.get("args", "")
        if smoke and job.get("smoke_args") is not None:
            args = job["smoke_args"]
        if n_gpu >= 2:
            launcher = f"torchrun --standalone --nproc_per_node={n_gpu}"
        else:
            launcher = _sys.executable
        env = dict(base_env, LABRAT_JOB_NAME="main", LABRAT_GPU_SLOT="all",
                   LABRAT_METRICS_PATH=str(outs / "metrics_main.json"))
        results.append(_labrat_run("main", _labrat_cmd(script, args, launcher), env, src,
                                logs / "main.log", hard_deadline))
    else:
        jobs = list(job.get("jobs") or [])
        if smoke:
            jobs = jobs[: int(job.get("smoke_max_jobs", 1))]
            for j in jobs:
                if j.get("smoke_args") is not None:
                    j["args"] = j["smoke_args"]
        slots = n_gpu if n_gpu else int(job.get("cpu_slots", 1))
        q = _queue.Queue()
        for j in jobs:
            q.put(j)
        lock = _th.Lock()

        def worker(slot):
            while True:
                try:
                    j = q.get_nowait()
                except _queue.Empty:
                    return
                if _time.time() > soft_deadline:
                    with lock:
                        results.append({"name": j["name"], "skipped": "deadline", "returncode": None})
                    continue
                env = dict(base_env, LABRAT_JOB_NAME=j["name"], LABRAT_GPU_SLOT=str(slot),
                           LABRAT_METRICS_PATH=str(outs / f"metrics_{j['name']}.json"))
                env.update({k: str(v) for k, v in (j.get("env") or {}).items()})
                env["CUDA_VISIBLE_DEVICES"] = str(slot) if n_gpu else ""
                res = _labrat_run(j["name"], _labrat_cmd(j.get("script", job.get("script")), j.get("args", ""),
                                                   _sys.executable), env, src,
                               logs / f"{j['name']}.log", hard_deadline)
                res["gpu"] = slot if n_gpu else None
                with lock:
                    results.append(res)
                    print(f"[labrat] xong {j['name']} (gpu {res['gpu']}) rc={res['returncode']} {res['seconds']}s")

        threads = [_th.Thread(target=worker, args=(i,)) for i in range(slots)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    post = job.get("post")
    if post:
        env = dict(base_env, LABRAT_JOB_NAME="post", LABRAT_METRICS_PATH=str(outs / "metrics_final.json"))
        env["CUDA_VISIBLE_DEVICES"] = ""
        r = _labrat_run("post", _labrat_cmd(post["script"], post.get("args", ""), _sys.executable), env, src,
                     logs / "post.log", t0 + max_h * 3600)
        r["post"] = True
        results.append(r)

    sampler.stop()

    # ---- gom kết quả ----
    def _load(p):
        try:
            return _json.loads(_Path(p).read_text())
        except Exception:
            return None

    for r in results:
        r["metrics"] = _load(outs / f"metrics_{r['name']}.json") if not r.get("post") else None
        r["log_tail"] = _labrat_tail(logs / f"{r['name']}.log", 40 if r.get("returncode") == 0 else 120)

    final = _load(outs / "metrics_final.json")
    train_results = [r for r in results if not r.get("post")]
    cv, cv_std, selected = None, None, None
    if final and final.get("cv") is not None:
        cv, cv_std = final.get("cv"), final.get("cv_std")
    else:
        scored = [r for r in train_results if r.get("metrics") and r["metrics"].get("cv") is not None]
        if len(scored) == 1:
            cv, cv_std = scored[0]["metrics"]["cv"], scored[0]["metrics"].get("cv_std")
            selected = scored[0]["name"]
        elif scored:  # nhiều biến thể độc lập → lấy biến thể tốt nhất
            pick = max if direction == "maximize" else min
            b = pick(scored, key=lambda r: r["metrics"]["cv"])
            cv, cv_std, selected = b["metrics"]["cv"], b["metrics"].get("cv_std"), b["name"]

    ok = all(r.get("returncode") == 0 for r in results if not r.get("skipped")) and bool(results)
    pruned = _labrat_prune_outputs(work, scratch, float(spec.get("output_max_mb", 200)),
                                   list(job.get("keep_outputs") or []))
    summary = {
        "exp_id": spec.get("exp_id"),
        "smoke": smoke,
        "ok": ok,
        "mode": mode,
        "n_gpu": n_gpu,
        "gpus": gpus,
        "gpu_util": sampler.summary(),
        "wall_seconds": round(_time.time() - t0, 1),
        "deadline_hit": _time.time() > soft_deadline,
        "cv": cv,
        "cv_std": cv_std,
        "selected_job": selected,
        "final": final,
        "jobs": results,
        "outputs_pruned": pruned,
    }
    (work / "metrics.json").write_text(_json.dumps(summary, indent=2, ensure_ascii=False))
    if pruned:
        print(f"[labrat] đã chuyển {len(pruned)} file lớn ra khỏi output "
              f"({sum(x['mb'] for x in pruned):.0f} MB): {[x['path'] for x in pruned][:5]}")
    print(f"[labrat] HOÀN TẤT ok={ok} cv={cv} wall={summary['wall_seconds']}s")
    print(_json.dumps({k: summary[k] for k in ("ok", "cv", "cv_std", "n_gpu", "gpu_util", "wall_seconds")}))
    # luôn thoát mã 0 để Kaggle giữ output; trạng thái thật nằm ở metrics.json["ok"]
