"""Đóng gói một thí nghiệm thành kernel Kaggle.

`kaggle kernels push` chỉ tải lên MỘT file code. Bundler nhúng toàn bộ thư mục
src/ (base64) cùng runner vào một main.py duy nhất, rồi sinh kernel-metadata.json.
File lớn (dữ liệu, trọng số) phải đưa lên dưới dạng dataset, không nhúng.
"""
from __future__ import annotations

import base64
import json
import re
import shutil
from pathlib import Path

import yaml

from .config import TEMPLATES

MAX_EMBED_BYTES = 2_000_000  # nhúng quá 2MB thì nên chuyển sang dataset
SKIP_SUFFIX = {".pyc", ".ipynb_checkpoints"}


def slugify(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return re.sub(r"-{2,}", "-", s)


def kernel_title(prefix: str, comp_slug: str, exp_id: str, name: str, smoke: bool) -> str:
    """Kaggle: slug = tiêu đề viết thường, nối bằng '-'; tiêu đề dài 5–50 ký tự."""
    suffix = "-cpu" if smoke else ""
    head = slugify(f"{prefix}-{comp_slug[:14]}-{exp_id}")
    room = 50 - len(head) - len(suffix) - 1
    tail = slugify(name)[:max(room, 0)].strip("-")
    title = f"{head}-{tail}" if tail else head
    return (title + suffix)[:50].strip("-")


def collect_files(src: Path) -> dict[str, str]:
    files, total = {}, 0
    for p in sorted(src.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p.suffix in SKIP_SUFFIX:
            continue
        data = p.read_bytes()
        total += len(data)
        files[p.relative_to(src).as_posix()] = base64.b64encode(data).decode()
    if total > MAX_EMBED_BYTES:
        raise ValueError(f"src/ nặng {total/1e6:.1f}MB > {MAX_EMBED_BYTES/1e6:.0f}MB. "
                         "Hãy đưa file lớn lên dataset (kra dataset) và gắn vào project.yaml.")
    return files


def build_bundle(exp_dir: Path, *, cfg: dict, exp: dict, username: str, smoke: bool) -> tuple[Path, str]:
    exp_dir = Path(exp_dir)
    job = yaml.safe_load((exp_dir / "job.yaml").read_text(encoding="utf-8")) or {}
    files = collect_files(exp_dir / "src")
    k = cfg.get("kaggle", {})

    spec = {
        "exp_id": exp["id"],
        "smoke": smoke,
        "job": job,
        "metric_direction": cfg.get("metric_direction", job.get("metric_direction", "maximize")),
        "max_runtime_hours": 1.0 if smoke else float(k.get("max_runtime_hours", 9)),
        "safety_margin_minutes": 5 if smoke else float(k.get("safety_margin_minutes", 20)),
    }
    payload = base64.b64encode(json.dumps({"files": files, "spec": spec}).encode()).decode()
    runner = (TEMPLATES / "runner.py").read_text(encoding="utf-8")
    main_py = (
        "# AUTO-GENERATED bởi kra — đừng sửa tay. Nguồn: experiments/" + exp_dir.name + "\n"
        "import base64 as _kb64, json as _kjson\n"
        f"_KRA = _kjson.loads(_kb64.b64decode('{payload}').decode())\n"
        "KRA_FILES, KRA_SPEC = _KRA['files'], _KRA['spec']\n\n"
        + runner
        + "\n\nkra_main(KRA_FILES, KRA_SPEC)\n"
    )

    out = exp_dir / ("_bundle_cpu" if smoke else "_bundle_gpu")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "main.py").write_text(main_py, encoding="utf-8")

    title = kernel_title(k.get("title_prefix", "kra"), cfg["slug"], exp["id"], exp.get("name", ""), smoke)
    ref = f"{username}/{title}"
    # dataset công khai/người khác: "owner/slug"; dataset riêng (đã tải lên MỌI tài khoản): chỉ "slug"
    own = list(cfg.get("own_datasets") or []) + list(job.get("own_datasets") or [])
    datasets = (list(cfg.get("datasets") or []) + list(job.get("dataset_sources") or [])
                + [f"{username}/{d}" for d in own])
    meta = {
        "id": ref,
        "title": title,
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": not smoke,
        "enable_internet": bool(k.get("enable_internet", True)),
        "dataset_sources": sorted(set(datasets)),
        "competition_sources": [] if cfg.get("no_competition_source") else [cfg["slug"]],
        "kernel_sources": list(job.get("kernel_sources") or []),
        "model_sources": list(job.get("model_sources") or []),
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out, ref
