"""Chính sách nghiên cứu: chặn "cày điểm" bằng seed và bỏ phiếu, buộc thí nghiệm có căn cứ.

Người dùng muốn agent NGHIÊN CỨU (tính chất dữ liệu, cơ chế của mô hình, cách đặt lại bài toán, ý tưởng mới)
thay vì tốn tài nguyên vào:
  - dò / chọn seed tốt nhất, chạy nhiều seed rồi lấy trung bình
  - chạy nhiều mô hình rồi bỏ phiếu / trung bình / blend / stack để tăng điểm

Kiểm tra ở hai chỗ:
  - new-exp / record: tên + giả thuyết không được là các kiểu trên; thí nghiệm (trừ baseline đầu tiên) phải có
    --evidence: quan sát từ dữ liệu / phân tích lỗi / paper làm căn cứ cho giả thuyết
  - push: job.yaml không được có nhiều job chỉ khác nhau seed, hay tham số chạy nhiều seed trong một job

Tắt / nới (CHỈ khi người dùng yêu cầu): config.yaml → research.enforce: false, hoặc sửa research.forbid.
"""
from __future__ import annotations

import re

DEFAULT_FORBID = {
    "dò/chọn/trung bình seed": (
        r"(best|tốt\s*nhất|optimal|tối\s*ưu|lucky)[\s_-]*seed|seed[\s_-]*(search|sweep|select\w*|tuning|lottery|hunt)"
        r"|(dò|chọn|tìm|quét|thử)\s+(nhiều\s+|các\s+)?seed|multi[\s_-]*seeds?|nhiều\s+seed"
        r"|seeds?[\s_-]*(avg|averag\w*|ensembl\w*|bagging|mean)|trung\s*bình\s*(\d+\s*|các\s*|nhiều\s*)?seeds?"
        r"|\b([2-9]|\d{2,})\s*seeds?\b.{0,30}(trung\s*bình|avg|averag|mean|ensembl|vote|bỏ\s*phiếu)"),
    "ensemble nhiều mô hình": (
        r"\bvoting\s*(classifier|regressor|ensemble)|\b(hard|soft|majority)[\s_-]*vot\w*|bỏ\s*phiếu|đa\s*số\s*phiếu"
        r"|\bvote\s+(giữa|của|between|across)"
        r"|ensembl\w*(\s+\S+){0,4}?\s+(models?|mô\s*hình|backbones?|kiến\s*trúc|architectures?|oofs?)\b"
        r"|(models?|mô\s*hình|backbones?)\s+ensembl\w*"
        r"|\bblend\w*\s+(of\s+|các\s+|nhiều\s+)?(models?|mô\s*hình|predictions?|dự\s*đoán|oofs?)\b|\bblend\w*\s+\S+\s*\+"
        r"|\bstack\w*\s+(of\s+|các\s+|nhiều\s+)?(models?|mô\s*hình|predictions?|dự\s*đoán|oofs?)\b"
        r"|meta[\s-]*(model|learner|mô\s*hình)"
        r"|kết\s*hợp\s*(dự\s*đoán\s*(của\s*)?)?(nhiều|các|hai|ba|\d+)\s*(mô\s*hình|models?)"
        r"|trung\s*bình\s*(dự\s*đoán\s*(của\s*)?)?(nhiều|các|hai|ba|\d+)\s*(mô\s*hình|models?)"
        r"|average\s+(of\s+)?(the\s+)?(\w+\s+)?(models|predictions\s+of\s+\w+\s+models)"
        r"|\b(lgbm|lightgbm|xgb\w*|catboost|cnn|lstm|bert|vit)\s*\+\s*(lgbm|lightgbm|xgb\w*|catboost|cnn|lstm|bert|vit)\b"),
}

# Kỹ thuật cải thiện MỘT mô hình — được phép dù có chữ "ensemble", "blend", "majority"... (bỏ khỏi văn bản trước khi xét)
ALLOW = (r"test[\s-]*time[\s-]*augment\w*(\s+ensembl\w*)?|\btta\b|\bema\b|exponential\s+moving\s+average|\bswa\b"
         r"|stochastic\s+weight\s+averag\w*|snapshot\s+ensembl\w*|checkpoint\s+averag\w*|model\s+averaging\s*\(ema\)"
         r"|mix[\s-]*up\s*(blend\w*)?|cut[\s-]*mix|bagging_(fraction|freq)|feature_fraction|bagging\s+(feature|fraction)"
         r"|majority\s+class|lớp\s+đa\s+số|minority\s+class|stack\w*\s+(thêm\s+)?\d*\s*(tầng|lớp|layers?|blocks?)"
         r"|multi[\s-]*sample\s+dropout|k[\s-]*fold|fold\s+averag\w*|trung\s*bình\s*(các\s*)?fold")

GUIDANCE = ("Labrat ưu tiên NGHIÊN CỨU thay vì cày điểm: hãy tìm cải thiện từ tính chất dữ liệu (phân phối, nhiễu nhãn, "
            "lệch train/test, cấu trúc nhóm/thời gian), phân tích lỗi của thí nghiệm trước, cơ chế/thiên kiến quy nạp "
            "của mô hình, cách đặt lại bài toán (loss khớp metric, multi-task, biến đổi nhãn), hoặc ý tưởng từ paper. "
            "Các kỹ thuật cải thiện MỘT mô hình (TTA, EMA/SWA, snapshot, mixup, k-fold) vẫn được dùng. "
            "Xem AGENT.md mục 'Mục tiêu và thứ tự ưu tiên' và docs/agent/nghien-cuu.md.")


def settings(cfg: dict) -> dict:
    r = cfg.get("research", {}) or {}
    return {"enforce": r.get("enforce", True), "require_evidence": r.get("require_evidence", True),
            "forbid": r.get("forbid") or DEFAULT_FORBID}


def check_idea(cfg: dict, *texts: str) -> None:
    s = settings(cfg)
    if not s["enforce"]:
        return
    blob = re.sub(ALLOW, " ", " ".join(t for t in texts if t), flags=re.IGNORECASE)
    for label, pat in s["forbid"].items():
        m = re.search(pat, blob, re.IGNORECASE)
        if m:
            raise SystemExit(f"Thí nghiệm bị từ chối ({label}: '{m.group(0)}'). {GUIDANCE}")


def check_evidence(cfg: dict, evidence: str | None, n_existing: int) -> None:
    s = settings(cfg)
    if not s["enforce"] or not s["require_evidence"] or n_existing == 0:
        return  # thí nghiệm đầu tiên (baseline) không cần căn cứ
    if not evidence or len(evidence.strip()) < 20:
        raise SystemExit("Thiếu --evidence (≥ 20 ký tự): căn cứ cho giả thuyết. Hai dạng đều được: "
                         "luồng nền mạnh — 'prior_work: đội top kỳ trước dùng X, +0.01'; luồng nghiên cứu — quan sát có "
                         "số liệu, vd 'phân tích lỗi exp-004: 70% lỗi ở nhóm <X> (15% dữ liệu), metric nhóm này 0.21 so "
                         "với 0.58 toàn tập'.")


def check_prior_work(cfg: dict, project_dir, n_existing: int) -> None:
    """Từ thí nghiệm thứ 2: phải có prior_work.md — lời giải mạnh nhất đã công bố (top các kỳ trước, SOTA)."""
    s = settings(cfg)
    if not s["enforce"] or not (cfg.get("research", {}) or {}).get("require_prior_work", True) or n_existing == 0:
        return
    from pathlib import Path
    f = Path(project_dir) / "prior_work.md"
    if not f.exists() or len(f.read_text(encoding="utf-8").strip()) < 200:
        raise SystemExit(f"Thiếu {f.name} (hoặc quá sơ sài). Trước khi thử ý tưởng riêng, ghi các lời giải mạnh "
                         "nhất đã công bố cho bài này hoặc bài tương tự (đội top các kỳ trước, paper SOTA): "
                         "phương pháp, điểm đã báo cáo, CƠ CHẾ khiến nó hiệu quả, nguồn. Thí nghiệm nền nên tái hiện "
                         "lời giải mạnh nhất; đóng góp của bạn là cải thiện có cơ sở trên nền đó.")


_SEED_ARG = re.compile(r"(--?seed[\s=]+\S+|\bseed\s*=\s*\S+)", re.IGNORECASE)
_MULTI_SEED = re.compile(r"--?(n[-_]?seeds|num[-_]?seeds|seeds)\b", re.IGNORECASE)


def check_job(cfg: dict, job: dict) -> None:
    """Chặn job.yaml chạy nhiều seed (nhiều job chỉ khác seed, hoặc tham số --seeds/--n-seeds)."""
    if not settings(cfg)["enforce"]:
        return
    all_args = [str(job.get("args") or "")] + [str(j.get("args") or "") for j in job.get("jobs") or []]
    for a in all_args:
        if _MULTI_SEED.search(a):
            raise SystemExit(f"job.yaml chạy nhiều seed ('{_MULTI_SEED.search(a).group(0)}'). {GUIDANCE}")
    jobs = job.get("jobs") or []
    if len(jobs) >= 2:
        def norm(j):
            env = {k: v for k, v in (j.get("env") or {}).items() if "seed" not in k.lower()}
            return (j.get("script", job.get("script")), _SEED_ARG.sub("", str(j.get("args") or "")).split(), sorted(env.items()))
        raw = {(j.get("script"), str(j.get("args") or ""), str(sorted((j.get("env") or {}).items()))) for j in jobs}
        if len({repr(norm(j)) for j in jobs}) == 1 and len(raw) > 1:
            raise SystemExit("job.yaml có nhiều job chỉ khác nhau seed — đó là chạy nhiều seed, không phải thí nghiệm. "
                             "Dùng 2 GPU cho 2 giả thuyết khác nhau (A/B), ablation, hoặc các fold CV. " + GUIDANCE)
