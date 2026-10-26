"""Nghiên cứu sâu MỘT paper: tải về, trích văn bản, tạo khung phân tích, xuất thẻ HTML.

  labrat paper <slug> 2406.12345            (mã arXiv)
  labrat paper <slug> https://arxiv.org/abs/2406.12345
  labrat paper <slug> https://.../paper.pdf
  labrat paper <slug> /đường/dẫn/paper.pdf  (file người dùng gửi qua Telegram)

Kết quả: competitions/<slug>/papers/<pid>/{paper.pdf, text.txt, meta.json, analysis.md}
Agent đọc text.txt, điền analysis.md (gồm bảng đề xuất thực nghiệm), rồi
`labrat paper-report <slug> <pid> --send` để gửi thẻ HTML.
"""
from __future__ import annotations

import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from . import research
from .bundle import slugify
from .config import comp_dir
from .registry import now_iso

UA = {"User-Agent": "Labrat/0.2"}
ARXIV_ID = re.compile(r"(?:arxiv\.org/(?:abs|pdf|html)/)?(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?",
                      re.I)

ANALYSIS_TEMPLATE = """# Phân tích paper: {title}

- **Nguồn:** {url}
- **Tác giả / năm:** {authors} ({year})
- **Văn bản trích xuất:** `text.txt` ({pages} trang)
- **Liên quan tới dự án:** {slug}

## 1. Tóm tắt (3–5 câu)

## 2. Đóng góp chính
-

## 3. Phương pháp
> Đủ chi tiết để cài đặt lại: kiến trúc, hàm mất mát, thuật toán, giả mã nếu cần.

## 4. Thiết lập thực nghiệm gốc
| Mục | Giá trị trong paper |
|---|---|
| Dữ liệu | |
| Metric | |
| Baseline so sánh | |
| Siêu tham số chính | |
| Phần cứng / thời gian | |

## 5. Kết quả gốc và tuyên bố cần kiểm chứng
-

## 6. Khả năng tái hiện trên Kaggle (2×T4, 15 GB/GPU, fp16, ≤ 9h/phiên)
- Bộ nhớ ước tính:
- Thời gian ước tính:
- Cần thu nhỏ gì (model/batch/dữ liệu):
- Code chính thức / pretrained có sẵn:

## 7. Đề xuất thực nghiệm
| # | Mục tiêu | Thay đổi so với baseline | Cách dùng 2 GPU | Ước tính GPU | Tiêu chí thành công |
|---|---|---|---|---|---|
| P1 | Tái hiện kết quả cốt lõi | | | | |
| P2 | Ablation thành phần quan trọng nhất | | | | |
| P3 | Áp dụng vào dữ liệu của dự án | | | | |

## 8. Rủi ro và câu hỏi mở
-
"""


def _get(url: str, **kw) -> requests.Response:
    r = requests.get(url, headers=UA, timeout=kw.pop("timeout", 120), **kw)
    r.raise_for_status()
    return r


def _arxiv_meta(aid: str) -> dict:
    r = _get(f"https://export.arxiv.org/api/query?id_list={aid}", timeout=60)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    e = ET.fromstring(r.text).find("a:entry", ns)
    if e is None:
        return {}
    return {
        "title": re.sub(r"\s+", " ", e.findtext("a:title", "", ns)).strip(),
        "year": (e.findtext("a:published", "", ns) or "")[:4],
        "authors": [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)],
        "abstract": re.sub(r"\s+", " ", e.findtext("a:summary", "", ns)).strip(),
        "url": f"https://arxiv.org/abs/{aid}",
    }


def extract_pdf_text(pdf: Path) -> tuple[str, int]:
    from pypdf import PdfReader
    reader = PdfReader(str(pdf))
    parts = []
    for i, page in enumerate(reader.pages):
        try:
            parts.append(f"\n\n===== Trang {i + 1} =====\n" + (page.extract_text() or ""))
        except Exception as e:  # trang lỗi không làm hỏng cả file
            parts.append(f"\n\n===== Trang {i + 1} (lỗi trích xuất: {e}) =====\n")
    return "".join(parts), len(reader.pages)


def fetch_paper(cfg: dict, src: str, pid: str | None = None) -> dict:
    slug = cfg["slug"]
    meta: dict = {"source": src}
    local = Path(src).expanduser()
    m = None if local.exists() else ARXIV_ID.search(src)
    if local.exists():
        meta.update(title=local.stem, url=str(local))
        pid = pid or slugify(local.stem)[:40]
    elif m and ("arxiv" in src.lower() or re.fullmatch(r"\s*" + ARXIV_ID.pattern + r"\s*", src, re.I)):
        aid = m.group(1) + (m.group(2) or "")
        meta.update(_arxiv_meta(aid), arxiv=aid)
        pid = pid or "arxiv-" + aid.replace("/", "-")
        src_pdf = f"https://arxiv.org/pdf/{aid}"
    else:
        meta.update(url=src, title=src.rsplit("/", 1)[-1])
        pid = pid or slugify(src.rsplit("/", 1)[-1].replace(".pdf", ""))[:40] or "paper"
        src_pdf = src

    pdir = comp_dir(cfg, slug) / "papers" / pid
    pdir.mkdir(parents=True, exist_ok=True)
    pdf = pdir / "paper.pdf"
    if local.exists():
        if local.suffix.lower() == ".pdf":
            shutil.copy(local, pdf)
        else:  # file văn bản/markdown người dùng gửi
            shutil.copy(local, pdir / local.name)
            pdf = None
    else:
        r = _get(src_pdf, timeout=300)
        if b"%PDF" in r.content[:1024]:
            pdf.write_bytes(r.content)
        else:  # trang HTML: lưu lại để agent đọc
            (pdir / "page.html").write_text(r.text, encoding="utf-8")
            pdf = None

    pages, text = 0, ""
    if pdf and pdf.exists():
        text, pages = extract_pdf_text(pdf)
    elif (pdir / "page.html").exists():
        text = re.sub(r"<[^>]+>", " ", (pdir / "page.html").read_text(encoding="utf-8"))
        text = re.sub(r"\s+", " ", text)
    elif local.exists():
        text = (pdir / local.name).read_text(encoding="utf-8", errors="replace")
    (pdir / "text.txt").write_text(text, encoding="utf-8")

    meta.update(pid=pid, pages=pages, chars=len(text), fetched_at=now_iso())
    (pdir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    analysis = pdir / "analysis.md"
    if not analysis.exists():
        analysis.write_text(ANALYSIS_TEMPLATE.format(
            title=meta.get("title", pid), url=meta.get("url", src), year=meta.get("year", "?"),
            authors=", ".join((meta.get("authors") or [])[:5]) or "?", pages=pages or "?", slug=slug),
            encoding="utf-8")

    research.add_items(comp_dir(cfg, slug) / "literature.json", [{
        "source": "paper", "title": meta.get("title", pid), "year": meta.get("year", ""),
        "url": meta.get("url", src), "abstract": (meta.get("abstract") or "")[:1200],
        "status": "deep_read", "relevance": 5, "notes": f"papers/{pid}/analysis.md"}])
    warn = None
    if len(text) < 2000:
        warn = "Trích xuất được rất ít văn bản (PDF scan?). Hãy đọc trực tiếp paper.pdf hoặc nhờ người dùng gửi bản khác."
    return {"pid": pid, "dir": str(pdir), "title": meta.get("title"), "pages": pages, "chars": len(text),
            "text": str(pdir / "text.txt"), "analysis": str(analysis), "warning": warn}


def render_analysis(cfg: dict, pid: str) -> Path:
    import html as _html

    import markdown

    from .report import CSS
    pdir = comp_dir(cfg, cfg["slug"]) / "papers" / pid
    md = (pdir / "analysis.md").read_text(encoding="utf-8")
    meta = json.loads((pdir / "meta.json").read_text(encoding="utf-8"))
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    extra = """
.md h1{font-size:22px;margin:0 0 12px}.md h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;
color:var(--muted);margin:26px 0 10px}.md table{margin:8px 0}.md blockquote{margin:6px 0;padding:4px 12px;
border-left:3px solid var(--line);color:var(--muted)}.md code{background:var(--soft);padding:1px 5px;border-radius:4px}
.md p{margin:6px 0}.md table td:first-child{white-space:nowrap}"""
    doc = (f'<!doctype html><html lang="vi"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>{_html.escape(meta.get("title", pid))[:80]}</title><style>{CSS}{extra}</style></head>'
           f'<body><div class="wrap"><div class="card"><span class="pill">Phân tích paper</span>'
           f'<div class="md" style="margin-top:12px">{body}</div></div>'
           f'<footer>Labrat · {_html.escape(cfg["slug"])} · {_html.escape(pid)}</footer>'
           f'</div></body></html>')
    out = pdir / "analysis.html"
    out.write_text(doc, encoding="utf-8")
    return out
