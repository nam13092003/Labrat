"""Sổ thí nghiệm: competitions/<slug>/registry.json.

Mỗi thí nghiệm:
  id, name, hypothesis, parent, template, status, created_at,
  cv, cv_std, metrics (dict đầy đủ từ runner), notes,
  runs: [{kind: smoke|gpu, kernel_ref, version, pushed_at, finished_at,
          status, wall_seconds, gpu_hours, n_gpu, gpu_util}]

Trạng thái thí nghiệm:
  draft → smoke_running → smoke_ok | smoke_failed → running → complete | error
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ACTIVE = {"smoke_running", "running"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s)


class Registry:
    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {"experiments": []}

    # ---- lưu / đọc ----
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    @property
    def experiments(self) -> list[dict]:
        return self.data["experiments"]

    def get(self, exp_id: str) -> dict:
        for e in self.experiments:
            if e["id"] == exp_id or e.get("folder") == exp_id:
                return e
        raise KeyError(f"Không có thí nghiệm {exp_id}")

    def next_id(self) -> str:
        return f"exp-{len(self.experiments) + 1:03d}"

    def add(self, rec: dict) -> dict:
        rec.setdefault("created_at", now_iso())
        rec.setdefault("status", "draft")
        rec.setdefault("runs", [])
        self.experiments.append(rec)
        return rec

    def update(self, exp_id: str, **kw) -> dict:
        e = self.get(exp_id)
        e.update(kw)
        return e

    # ---- truy vấn ----
    def active(self) -> list[dict]:
        return [e for e in self.experiments if e.get("status") in ACTIVE]

    def completed(self) -> list[dict]:
        done = [e for e in self.experiments if e.get("status") == "complete" and e.get("cv") is not None]
        return sorted(done, key=lambda e: e.get("finished_at") or e.get("created_at"))

    def best(self, direction: str = "maximize") -> dict | None:
        done = self.completed()
        if not done:
            return None
        key = (lambda e: e["cv"]) if direction == "maximize" else (lambda e: -e["cv"])
        return max(done, key=key)

    def all_runs(self) -> list[dict]:
        return [r for e in self.experiments for r in e.get("runs", [])]


def better(a: float, b: float | None, direction: str, min_delta: float = 0.0) -> bool:
    """a có tốt hơn b (ít nhất min_delta) không."""
    if b is None:
        return True
    return (a - b > min_delta) if direction == "maximize" else (b - a > min_delta)
