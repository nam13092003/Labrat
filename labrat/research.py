"""Tìm paper liên quan (arXiv + Semantic Scholar) và lưu vào literature.json.

Lời giải top của các cuộc thi Kaggle cũ (writeup, discussion) thường giá trị hơn paper —
agent tìm bằng web search của harness rồi ghi vào bằng `labrat lit-add`.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from .registry import now_iso

UA = {"User-Agent": "Labrat/0.1"}


def search_arxiv(query: str, limit: int = 10) -> list[dict]:
    q = urllib.parse.urlencode({"search_query": f"all:{query}", "start": 0, "max_results": limit,
                                "sortBy": "relevance"})
    r = requests.get(f"https://export.arxiv.org/api/query?{q}", headers=UA, timeout=60)
    r.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    items = []
    for e in ET.fromstring(r.text).findall("a:entry", ns):
        pdf = next((l.get("href") for l in e.findall("a:link", ns) if l.get("title") == "pdf"), None)
        items.append({
            "source": "arxiv",
            "title": re.sub(r"\s+", " ", e.findtext("a:title", "", ns)).strip(),
            "year": (e.findtext("a:published", "", ns) or "")[:4],
            "url": e.findtext("a:id", "", ns),
            "pdf": pdf,
            "authors": [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)][:6],
            "abstract": re.sub(r"\s+", " ", e.findtext("a:summary", "", ns)).strip()[:1200],
        })
    return items


def search_semantic_scholar(query: str, limit: int = 10) -> list[dict]:
    headers = dict(UA)
    if os.environ.get("S2_API_KEY"):
        headers["x-api-key"] = os.environ["S2_API_KEY"]
    params = {"query": query, "limit": limit,
              "fields": "title,year,abstract,citationCount,url,externalIds,openAccessPdf,venue,authors"}
    for attempt in range(4):
        r = requests.get("https://api.semanticscholar.org/graph/v1/paper/search", params=params,
                         headers=headers, timeout=60)
        if r.status_code == 429:
            time.sleep(3 * (attempt + 1))
            continue
        r.raise_for_status()
        break
    else:
        return []
    items = []
    for p in r.json().get("data", []):
        items.append({
            "source": "semantic_scholar",
            "title": p.get("title"),
            "year": str(p.get("year") or ""),
            "url": p.get("url"),
            "pdf": (p.get("openAccessPdf") or {}).get("url"),
            "venue": p.get("venue"),
            "citations": p.get("citationCount"),
            "arxiv": (p.get("externalIds") or {}).get("ArXiv"),
            "authors": [a.get("name") for a in (p.get("authors") or [])][:6],
            "abstract": (p.get("abstract") or "")[:1200],
        })
    return items


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"items": [], "queries": []}


def add_items(path: Path, items: list[dict], query: str | None = None) -> int:
    lit = load(path)
    seen = {_norm(i["title"]) for i in lit["items"]}
    added = 0
    for it in items:
        k = _norm(it.get("title"))
        if not k or k in seen:
            continue
        seen.add(k)
        it.update({"query": query, "added_at": now_iso(), "status": it.get("status", "unread"),
                   "relevance": it.get("relevance"), "notes": it.get("notes", ""),
                   "method_tags": it.get("method_tags", [])})
        lit["items"].append(it)
        added += 1
    if query:
        lit["queries"].append({"q": query, "at": now_iso(), "added": added})
    path.write_text(json.dumps(lit, indent=2, ensure_ascii=False), encoding="utf-8")
    return added


def search(path: Path, query: str, limit: int = 10, sources=("arxiv", "s2")) -> dict:
    found, errors = [], []
    if "arxiv" in sources:
        try:
            found += search_arxiv(query, limit)
        except Exception as e:
            errors.append(f"arxiv: {e}")
    if "s2" in sources:
        try:
            found += search_semantic_scholar(query, limit)
        except Exception as e:
            errors.append(f"semantic_scholar: {e}")
    added = add_items(path, found, query)
    return {"query": query, "found": len(found), "added": added, "errors": errors,
            "titles": [f"[{i['source']}] {i['title']} ({i['year']})" for i in found]}
