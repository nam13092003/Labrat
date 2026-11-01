"""Gọi model bất kỳ trên OpenRouter khi có lợi cho tác vụ (agent dùng qua `labrat llm …`).

Ví dụ khi nào nên dùng (xem AGENT.md):
  - đọc ảnh / biểu đồ / hình trong paper khi model của harness không có thị giác
  - PDF scan (OCR), tài liệu rất dài vượt context của harness
  - xin ý kiến thứ hai từ model mạnh khi bế tắc (review kế hoạch, tìm bug)
  - xử lý hàng loạt rẻ (gắn nhãn, tóm tắt) bằng model nhỏ

  labrat llm models [--search q] [--modality image|file|audio] [--max-price USD_per_M_input] [--refresh]
  labrat llm ask "<prompt>" --model <id> [--file f]... [--system s] [--max-tokens n] [--json]
                 [--out file] [--purpose "..."] [--prompt-file p]
  labrat llm spend        chi tiêu hôm nay / tháng này / theo model
  labrat llm balance      tín dụng còn lại của key (OpenRouter)

An toàn chi phí: trần mỗi lần gọi / mỗi ngày / mỗi tháng (config.yaml → llm). Ước tính trước khi gọi,
từ chối nếu vượt; chi phí thật (usage.cost) ghi vào .llm_usage.jsonl.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from .config import ROOT, load_config

API = "https://openrouter.ai/api/v1"
LEDGER = ROOT / ".llm_usage.jsonl"
CACHE = ROOT / ".cache" / "openrouter_models.json"
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
TEXT_EXT = {".txt", ".md", ".py", ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml", ".log", ".html", ".tex",
            ".ipynb", ".sh", ".cfg", ".toml", ".xml"}
MAX_IMAGE_MB = 15
MAX_PDF_MB = 30


class LLMError(RuntimeError):
    pass


# ---------------------------------------------------------------- cấu hình
def settings(cfg: dict | None = None) -> dict:
    c = (cfg or load_config()).get("llm", {}) or {}
    return {
        "enabled": c.get("enabled", True),
        "per_call_usd": float(c.get("per_call_usd", 0.5)),
        "daily_usd": float(c.get("daily_usd", 3.0)),
        "monthly_usd": float(c.get("monthly_usd", 30.0)),
        "max_tokens": int(c.get("max_tokens", 4000)),
        "max_text_chars": int(c.get("max_text_chars", 400_000)),
        "pdf_engine": c.get("pdf_engine", "cloudflare-ai"),
        "deny_models": list(c.get("deny_models") or []),
    }


def _key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not k:
        raise LLMError("Chưa có OPENROUTER_API_KEY trong .env (chạy: python3 install.py --only llm).")
    return k


def _headers() -> dict:
    return {"Authorization": f"Bearer {_key()}", "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/nam13092003/Labrat", "X-Title": "Labrat"}


def _get(path: str, timeout: int = 60) -> dict:
    r = requests.get(API + path, headers=_headers(), timeout=timeout)
    if r.status_code >= 400:
        raise LLMError(f"OpenRouter {path} lỗi {r.status_code}: {r.text[:300]}")
    return r.json()


def _post(path: str, body: dict, timeout: int = 600) -> dict:
    last = None
    for attempt in range(4):
        try:
            r = requests.post(API + path, headers=_headers(), data=json.dumps(body), timeout=timeout)
        except requests.RequestException as e:
            last = f"mạng: {e}"
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            last = f"{r.status_code}: {r.text[:200]}"
            time.sleep(min(60, 5 * 2 ** attempt))
            continue
        if r.status_code >= 400:
            try:
                msg = r.json().get("error", {}).get("message") or r.text
            except ValueError:
                msg = r.text
            raise LLMError(f"OpenRouter lỗi {r.status_code}: {str(msg)[:500]}")
        return r.json()
    raise LLMError(f"OpenRouter không phản hồi sau 4 lần thử ({last})")


# ---------------------------------------------------------------- danh sách model
def _all_models(refresh: bool = False) -> list[dict]:
    if not refresh and CACHE.exists() and time.time() - CACHE.stat().st_mtime < 24 * 3600:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    data = _get("/models").get("data", [])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(data), encoding="utf-8")
    return data


def _price(m: dict, field: str) -> float:
    try:
        return max(0.0, float((m.get("pricing") or {}).get(field) or 0))
    except (TypeError, ValueError):
        return 0.0


def get_model(model_id: str) -> dict | None:
    for refresh in (False, True):
        for m in _all_models(refresh):
            if m.get("id") == model_id or m.get("canonical_slug") == model_id:
                return m
    return None


def models(search: str | None = None, modality: str | None = None, max_price: float | None = None,
           limit: int = 40, refresh: bool = False) -> list[dict]:
    rows = []
    for m in _all_models(refresh):
        mods = (m.get("architecture") or {}).get("input_modalities") or ["text"]
        text = f"{m.get('id')} {m.get('name')} {m.get('description', '')}".lower()
        if search and not all(w in text for w in search.lower().split()):
            continue
        if modality and modality not in mods:
            continue
        pin, pout = _price(m, "prompt") * 1e6, _price(m, "completion") * 1e6
        if max_price is not None and pin > max_price:
            continue
        rows.append({"id": m["id"], "context": m.get("context_length"),
                     "usd_per_M_in": round(pin, 3), "usd_per_M_out": round(pout, 3),
                     "input": mods, "max_out": (m.get("top_provider") or {}).get("max_completion_tokens")})
    rows.sort(key=lambda r: (r["usd_per_M_in"] + r["usd_per_M_out"], r["id"]))
    return rows[:limit]


# ---------------------------------------------------------------- chi tiêu
def _ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    out = []
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def spend() -> dict:
    now = datetime.now(timezone.utc)
    day, month = now.strftime("%Y-%m-%d"), now.strftime("%Y-%m")
    rows = _ledger()
    by_model: dict[str, float] = {}
    today = this_month = 0.0
    for r in rows:
        c = float(r.get("cost") or 0)
        if r.get("ts", "").startswith(day):
            today += c
        if r.get("ts", "").startswith(month):
            this_month += c
            by_model[r.get("model", "?")] = by_model.get(r.get("model", "?"), 0) + c
    s = settings()
    return {"today_usd": round(today, 4), "month_usd": round(this_month, 4), "calls": len(rows),
            "daily_cap": s["daily_usd"], "monthly_cap": s["monthly_usd"], "per_call_cap": s["per_call_usd"],
            "by_model_month": {k: round(v, 4) for k, v in sorted(by_model.items(), key=lambda x: -x[1])}}


def balance() -> dict:
    d = _get("/key").get("data", {})
    return {k: d.get(k) for k in ("label", "limit", "limit_remaining", "usage", "usage_daily", "is_free_tier")}


# ---------------------------------------------------------------- dựng tin nhắn
def _data_url(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def build_content(prompt: str, files: list[str], s: dict, model: dict | None) -> tuple[list, dict, list]:
    """Trả về (content parts, ước lượng {chars, images, pdf_pages}, plugins)."""
    parts, plugins, texts = [], [], []
    est = {"chars": len(prompt), "images": 0, "pdf_pages": 0}
    mods = ((model or {}).get("architecture") or {}).get("input_modalities") or ["text"]
    for f in files:
        p = Path(f).expanduser()
        if not p.is_file():
            raise LLMError(f"Không thấy file {p}")
        ext, mb = p.suffix.lower(), p.stat().st_size / 1048576
        if ext in IMAGE_EXT:
            if "image" not in mods:
                raise LLMError(f"Model không nhận ảnh. Chọn model có 'image' (labrat llm models --modality image).")
            if mb > MAX_IMAGE_MB:
                raise LLMError(f"Ảnh {p.name} {mb:.1f} MB > {MAX_IMAGE_MB} MB — thu nhỏ trước.")
            mime = mimetypes.guess_type(p.name)[0] or "image/png"
            parts.append({"type": "image_url", "image_url": {"url": _data_url(p, mime)}})
            est["images"] += 1
        elif ext == ".pdf":
            if mb > MAX_PDF_MB:
                raise LLMError(f"PDF {p.name} {mb:.1f} MB > {MAX_PDF_MB} MB — tách nhỏ hoặc trích văn bản trước.")
            parts.append({"type": "file", "file": {"filename": p.name,
                                                   "file_data": _data_url(p, "application/pdf")}})
            try:
                from pypdf import PdfReader
                est["pdf_pages"] += len(PdfReader(str(p)).pages)
            except Exception:
                est["pdf_pages"] += max(1, int(mb * 10))
            if "file" not in mods and not plugins:  # model không đọc PDF trực tiếp → parser của OpenRouter
                plugins.append({"id": "file-parser", "pdf": {"engine": s["pdf_engine"]}})
        elif ext in TEXT_EXT or mb < 2:
            t = p.read_text(encoding="utf-8", errors="replace")
            if len(t) > s["max_text_chars"]:
                t = t[: s["max_text_chars"]] + f"\n…[cắt bớt: file dài {len(t)} ký tự]"
            texts.append(f"\n\n===== FILE: {p.name} =====\n{t}")
            est["chars"] += len(t)
        else:
            raise LLMError(f"Không gửi được {p.name} ({ext}, {mb:.1f} MB): chỉ hỗ trợ ảnh, PDF, file văn bản.")
    parts.insert(0, {"type": "text", "text": prompt + "".join(texts)})
    return parts, est, plugins


def estimate_cost(model: dict | None, est: dict, max_tokens: int) -> float:
    if not model:
        return 0.0
    tokens_in = est["chars"] / 3.2 + est["images"] * 1500 + est["pdf_pages"] * 900
    return (tokens_in * _price(model, "prompt") + max_tokens * _price(model, "completion")
            + est["images"] * _price(model, "image") + _price(model, "request"))


# ---------------------------------------------------------------- gọi
def ask(prompt: str, model_id: str, files: list[str] | None = None, system: str | None = None,
        max_tokens: int | None = None, json_mode: bool = False, temperature: float | None = None,
        out: str | None = None, purpose: str = "", allow_over_cap: bool = False) -> dict:
    s = settings()
    if not s["enabled"]:
        raise LLMError("llm.enabled = false trong config.yaml")
    if any(model_id.startswith(d) for d in s["deny_models"]):
        raise LLMError(f"Model {model_id} bị chặn (llm.deny_models).")
    model = get_model(model_id)
    if model is None and model_id != "openrouter/auto":
        raise LLMError(f"Không có model '{model_id}'. Tìm bằng: labrat llm models --search <từ khoá>")
    max_tokens = int(max_tokens or s["max_tokens"])
    cap_out = (model or {}).get("top_provider", {}).get("max_completion_tokens")
    if cap_out:
        max_tokens = min(max_tokens, int(cap_out))
    parts, est, plugins = build_content(prompt, files or [], s, model)

    cost_est = estimate_cost(model, est, max_tokens)
    sp = spend()
    problems = []
    if cost_est > s["per_call_usd"]:
        problems.append(f"ước tính ${cost_est:.3f} > trần mỗi lần ${s['per_call_usd']}")
    if sp["today_usd"] + cost_est > s["daily_usd"]:
        problems.append(f"hôm nay đã tiêu ${sp['today_usd']:.3f}, trần ngày ${s['daily_usd']}")
    if sp["month_usd"] + cost_est > s["monthly_usd"]:
        problems.append(f"tháng này đã tiêu ${sp['month_usd']:.3f}, trần tháng ${s['monthly_usd']}")
    if problems and not allow_over_cap:
        raise LLMError("Không gọi: " + "; ".join(problems)
                       + ". Chọn model rẻ hơn / bớt dữ liệu, hoặc hỏi người dùng nâng trần (config.yaml → llm).")

    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": parts}]
    body = {"model": model_id, "messages": messages, "max_tokens": max_tokens}
    if plugins:
        body["plugins"] = plugins
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if temperature is not None:
        body["temperature"] = temperature

    t0 = time.time()
    resp = _post("/chat/completions", body)
    ch = (resp.get("choices") or [{}])[0]
    if ch.get("error"):
        raise LLMError(f"Model lỗi: {ch['error'].get('message', ch['error'])}")
    content = (ch.get("message") or {}).get("content") or ""
    if isinstance(content, list):  # một số provider trả về danh sách part
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    u = resp.get("usage") or {}
    pt, ct = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
    cost = u.get("cost")
    if cost is None and model:
        cost = pt * _price(model, "prompt") + ct * _price(model, "completion")
    cost = round(float(cost or 0), 6)
    used_model = resp.get("model") or model_id

    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": used_model,
                            "prompt_tokens": pt, "completion_tokens": ct, "cost": cost,
                            "estimate": round(cost_est, 6), "purpose": purpose[:200],
                            "files": [Path(x).name for x in files or []],
                            "seconds": round(time.time() - t0, 1)}, ensure_ascii=False) + "\n")
    res = {"model": used_model, "cost_usd": cost, "prompt_tokens": pt, "completion_tokens": ct,
           "finish_reason": ch.get("finish_reason"), "spend_today_usd": round(sp["today_usd"] + cost, 4)}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(content, encoding="utf-8")
        res["out"] = str(out)
        res["preview"] = content[:500]
    else:
        res["text"] = content if len(content) <= 6000 else content[:6000] + "\n…[dùng --out để lấy đầy đủ]"
    if ch.get("finish_reason") == "length":
        res["warning"] = "Bị cắt do max_tokens — tăng --max-tokens nếu cần."
    return res
