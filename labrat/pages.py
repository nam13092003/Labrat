"""Đọc trang web (trang cuộc thi ngoài Kaggle, trang Kaggle, blog writeup...) thành văn bản cho agent.

  labrat page <url> [--slug <dự án>] [--follow N]

- Tải HTML, bỏ script/style/menu, giữ cấu trúc: tiêu đề → "#", danh sách → "-", bảng → "a | b".
- --follow N: đi tiếp tối đa N trang con CÙNG SITE (cùng thư mục gốc), ví dụ description → data → rules → timeline.
- Gom các link trông như dữ liệu (csv/zip/Drive/HF/Kaggle dataset...) vào links.json để agent tải bằng `labrat get`.
- Trang dựng bằng JavaScript (Kaggle...) gần như trống khi tải thường → tự dùng trình duyệt headless
  (Playwright) nếu đã cài: pip install playwright && playwright install --with-deps chromium

Lưu vào competitions/<slug>/pages/ (có --slug) hoặc webpages/<site>/.
"""
from __future__ import annotations

import json
import re
from collections import deque
from datetime import datetime
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import requests

from .bundle import slugify
from .config import ROOT, comp_dir
from .fetcher import DATA_EXT, classify

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"}
MIN_TEXT = 400          # ít hơn → coi là trang JS, thử trình duyệt headless
SKIP_TAGS = ("script", "style", "noscript", "svg", "nav", "footer", "header", "form", "iframe")


def _render_js(url: str) -> str | None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    try:
        with sync_playwright() as p:
            b = p.chromium.launch()
            page = b.new_page(user_agent=UA["User-Agent"])
            page.goto(url, wait_until="networkidle", timeout=60_000)
            page.wait_for_timeout(1500)
            html = page.content()
            b.close()
            return html
    except Exception as e:
        print(f"[page] trình duyệt headless lỗi: {e}")
        return None


def html_to_text(html: str, base: str) -> tuple[str, str, list[dict]]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.get_text(strip=True) if soup.title else "") or base
    links = []
    for a in soup.find_all("a", href=True):
        href = urldefrag(urljoin(base, a["href"]))[0]
        if href.startswith(("http://", "https://")):
            links.append({"url": href, "text": a.get_text(" ", strip=True)[:120]})
    for t in soup(SKIP_TAGS):
        t.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    out = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "tr", "dt", "dd", "blockquote"]):
        if el.find_parent(["li", "tr", "pre"]) is not None and el.name not in ("tr",):
            continue  # đã lấy qua phần tử cha
        txt = el.get_text(" ", strip=True)
        if not txt:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            out.append("\n" + "#" * int(el.name[1]) + " " + txt)
        elif el.name == "li":
            out.append("- " + txt)
        elif el.name == "tr":
            out.append(" | ".join(c.get_text(" ", strip=True) for c in el.find_all(["th", "td"])))
        elif el.name == "pre":
            out.append("```\n" + el.get_text() + "\n```")
        else:
            out.append(txt)
    text = "\n".join(out).strip()
    if len(text) < MIN_TEXT:  # cấu trúc lạ: lấy toàn bộ chữ
        text = re.sub(r"\n{3,}", "\n\n", main.get_text("\n", strip=True))
    return title, text, links


def _is_data_link(url: str) -> bool:
    kind, _ = classify(url)
    return kind in ("gdrive_file", "gdrive_folder", "gsheet", "hf_dataset", "kaggle_dataset") or (
        kind == "direct" and urlparse(url).path.lower().endswith(DATA_EXT))


def fetch_page(url: str) -> dict:
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    ctype = r.headers.get("content-type", "")
    if "html" not in ctype:
        return {"url": url, "title": url, "text": "", "links": [], "not_html": ctype}
    title, text, links = html_to_text(r.text, r.url)
    rendered = False
    if len(text) < MIN_TEXT:
        html = _render_js(url)
        if html:
            title, text, links = html_to_text(html, url)
            rendered = True
    return {"url": r.url, "title": title, "text": text, "links": links, "rendered_js": rendered}


def crawl(url: str, slug: str | None = None, follow: int = 0, max_pages: int = 12) -> dict:
    from .config import load_config
    cfg = load_config()
    u = urlparse(url)
    if slug:
        out_dir = comp_dir(cfg, slug) / "pages"
    else:
        out_dir = ROOT / "webpages" / slugify(u.netloc + "-" + u.path.rsplit("/", 1)[0])[:60]
    out_dir.mkdir(parents=True, exist_ok=True)
    scope = u.scheme + "://" + u.netloc + (u.path.rsplit("/", 1)[0] if "." in u.path.rsplit("/", 1)[-1] else u.path)
    seen, queue, saved, data_links, warnings = set(), deque([(url, 0)]), [], {}, []
    while queue and len(saved) < max_pages:
        cur, depth = queue.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        try:
            pg = fetch_page(cur)
        except Exception as e:
            warnings.append(f"{cur}: {e}")
            continue
        if pg.get("not_html"):
            data_links[cur] = f"(file {pg['not_html']})"
            continue
        name = slugify(urlparse(pg["url"]).path.strip("/").replace("/", "-") or "index")[:60] or "index"
        path = out_dir / f"{name}.md"
        path.write_text(f"# {pg['title']}\n\nNguồn: {pg['url']}\nLấy lúc: {datetime.now():%Y-%m-%d %H:%M}"
                        f"{' (render JavaScript)' if pg.get('rendered_js') else ''}\n\n{pg['text']}\n",
                        encoding="utf-8")
        saved.append({"url": pg["url"], "title": pg["title"], "file": str(path), "chars": len(pg["text"])})
        if len(pg["text"]) < MIN_TEXT and not pg.get("rendered_js"):
            warnings.append(f"{cur}: rất ít chữ — trang dựng bằng JavaScript? Cài Playwright để đọc "
                            "(pip install playwright && playwright install --with-deps chromium).")
        for ln in pg["links"]:
            href = ln["url"]
            if _is_data_link(href):
                data_links.setdefault(href, ln["text"])
            elif depth < follow and href.startswith(scope) and href not in seen \
                    and not urlparse(href).path.lower().endswith(DATA_EXT + (".pdf",)):
                queue.append((href, depth + 1))
    (out_dir / "links.json").write_text(json.dumps(
        [{"url": k, "text": v} for k, v in data_links.items()], indent=2, ensure_ascii=False), encoding="utf-8")
    return {"saved_to": str(out_dir), "pages": saved, "data_links": list(data_links)[:30],
            "warnings": warnings,
            "next": "Đọc các file .md; tải dữ liệu bằng `labrat get <url> --name <slug>`"}
