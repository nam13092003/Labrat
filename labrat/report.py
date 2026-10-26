"""Sinh thẻ báo cáo HTML tự chứa (không phụ thuộc CDN) để gửi qua Telegram.

Nguồn dữ liệu:
  registry.json      — kết quả thí nghiệm (máy ghi)
  literature.json    — paper/lời giải đã thu thập
  report_notes.json  — phần nhận định do agent viết:
      {"summary": "...", "insights": ["..."],
       "next_directions": [{"title": "...", "why": "...", "cost": "~3h GPU", "expected_gain": "+0.002"}],
       "stop_reason": "..."}
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

from . import budget
from .config import comp_dir
from .registry import Registry, better

E = html.escape


def _fmt(v, nd=5):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return E(str(v))


def _chart(done: list[dict], direction: str) -> str:
    if not done:
        return '<p class="muted">Chưa có thí nghiệm GPU hoàn tất.</p>'
    W, H, L, R, T, B = 640, 220, 56, 16, 14, 34
    ys = [e["cv"] for e in done]
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.15 or abs(hi) * 0.01 or 0.01
    lo, hi = lo - pad, hi + pad
    n = len(done)

    def X(i):
        return L + (W - L - R) * (i / (n - 1) if n > 1 else 0.5)

    def Y(v):
        return T + (H - T - B) * (1 - (v - lo) / (hi - lo))

    grid, labels = [], []
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = Y(v)
        grid.append(f'<line x1="{L}" x2="{W-R}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        labels.append(f'<text x="{L-8}" y="{y+4:.1f}" class="tick" text-anchor="end">{v:.4f}</text>')

    best, step_pts = None, []
    for i, e in enumerate(done):
        if better(e["cv"], best, direction):
            best = e["cv"]
        step_pts.append((X(i), Y(best)))
    path = f"M{step_pts[0][0]:.1f},{step_pts[0][1]:.1f}"
    for (x0, y0), (x1, y1) in zip(step_pts, step_pts[1:]):
        path += f" H{x1:.1f} V{y1:.1f}"

    dots, xl = [], []
    best_val = best
    for i, e in enumerate(done):
        cls = "dot best" if e["cv"] == best_val else "dot"
        dots.append(f'<circle cx="{X(i):.1f}" cy="{Y(e["cv"]):.1f}" r="4.5" class="{cls}">'
                    f'<title>{E(e["id"])} {E(e.get("name",""))}: {e["cv"]:.5f}</title></circle>')
        if n <= 14 or i % max(1, n // 10) == 0 or i == n - 1:
            xl.append(f'<text x="{X(i):.1f}" y="{H-12}" class="tick" text-anchor="middle">'
                      f'{E(e["id"].replace("exp-", "#"))}</text>')
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Điểm CV theo thí nghiệm">'
            + "".join(grid) + "".join(labels)
            + f'<path d="{path}" class="bestline"/>' + "".join(dots) + "".join(xl) + "</svg>"
            '<div class="legend"><span><i class="sw dotc"></i>CV từng thí nghiệm</span>'
            '<span><i class="sw linec"></i>Tốt nhất đến hiện tại</span></div>')


def _gpu_bars(util: list[dict] | None) -> str:
    if not util:
        return '<p class="muted">Chưa có số liệu sử dụng GPU.</p>'
    rows = []
    for g in util:
        u = g.get("util_mean", 0)
        rows.append(f'<div class="gpurow"><span>GPU {g["gpu"]}</span>'
                    f'<div class="bar"><div style="width:{min(u,100):.0f}%"></div></div>'
                    f'<span class="num">{u:.0f}% · {g.get("mem_max_mb",0)/1024:.1f} GB</span></div>')
    return "".join(rows)


def _accounts(accs: list[dict]) -> str:
    rows = []
    for a in accs:
        lim = a["weekly_gpu_hours"] or 1
        pct = min(100, 100 * a["week_used"] / lim)
        state = f"đang chạy {E(a['gpu_busy'])}" if a.get("gpu_busy") else "rảnh"
        rows.append(f'<div class="gpurow"><span>{E(a["name"])}</span>'
                    f'<div class="bar"><div style="width:{pct:.0f}%"></div></div>'
                    f'<span class="num">{a["week_used"]:.1f}/{lim:.0f}h · {state}</span></div>')
    return "".join(rows) or '<p class="muted">Không có tài khoản.</p>'


CSS = """
:root{--bg:#f6f5f1;--card:#fff;--ink:#1c1d1f;--muted:#6b6e74;--line:#e4e2dc;--acc:#2f6f5e;--acc2:#c9833a;
--good:#2f6f5e;--bad:#b4473a;--soft:#eef3f1}
@media (prefers-color-scheme:dark){:root{--bg:#141516;--card:#1d1f21;--ink:#ecebe7;--muted:#9a9ca1;--line:#2e3134;
--acc:#6fb8a1;--acc2:#e0a35f;--good:#6fb8a1;--bad:#e0786a;--soft:#22302c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
.wrap{max-width:860px;margin:0 auto;padding:24px 16px 48px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;margin-bottom:14px}
h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}h2{font-size:13px;text-transform:uppercase;
letter-spacing:.08em;color:var(--muted);margin:0 0 12px;font-weight:600}
.muted{color:var(--muted)}.pill{display:inline-block;padding:2px 10px;border-radius:99px;font-size:12px;
font-weight:600;background:var(--soft);color:var(--acc)}.pill.stop{background:rgba(180,71,58,.12);color:var(--bad)}
.kpis{display:flex;flex-wrap:wrap;gap:16px}.kpi{flex:1 1 150px}
.kpi .v{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.kpi .l{font-size:12px;color:var(--muted)}
.bar{flex:1;height:8px;background:var(--line);border-radius:99px;overflow:hidden}
.bar>div{height:100%;background:var(--acc);border-radius:99px}
.budget{display:flex;align-items:center;gap:10px;margin-top:6px}
.gpurow{display:flex;align-items:center;gap:10px;margin:6px 0}.gpurow span:first-child{width:52px;font-size:13px}
.num{font-variant-numeric:tabular-nums;font-size:13px;color:var(--muted);min-width:110px;text-align:right}
.gpurow span:first-child{width:64px;font-size:13px;overflow:hidden;text-overflow:ellipsis}
svg{width:100%;height:auto;display:block}.grid{stroke:var(--line);stroke-width:1}
.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.bestline{fill:none;stroke:var(--acc);stroke-width:2}.dot{fill:var(--card);stroke:var(--acc2);stroke-width:2}
.dot.best{fill:var(--acc);stroke:var(--acc)}
.legend{display:flex;gap:16px;font-size:12px;color:var(--muted);margin-top:6px}
.sw{display:inline-block;width:12px;height:3px;margin-right:6px;vertical-align:middle}
.dotc{background:var(--acc2)}.linec{background:var(--acc)}
.tablewrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--muted);font-weight:600;border-bottom:1px solid var(--line);padding:6px 8px;white-space:nowrap}
td{border-bottom:1px solid var(--line);padding:7px 8px;vertical-align:top}td.n{font-variant-numeric:tabular-nums;white-space:nowrap}
tr.best td{background:var(--soft)}.st{font-size:12px;color:var(--muted)}.st.error,.st.smoke_failed{color:var(--bad)}
ul{margin:0;padding-left:20px}li{margin:4px 0}
.dir{border-left:3px solid var(--acc);padding:6px 0 6px 12px;margin:10px 0}
.dir b{display:block}.dir .meta{font-size:12px;color:var(--muted)}
a{color:var(--acc)}footer{font-size:12px;color:var(--muted);text-align:center;margin-top:18px}
"""


def build(cfg: dict) -> Path:
    slug = cfg["slug"]
    cdir = comp_dir(cfg, slug)
    reg = Registry(cdir / "registry.json")
    direction = cfg.get("metric_direction", "maximize")
    done = reg.completed()
    best = reg.best(direction)
    chk = budget.stop_check(cfg, reg)
    bud = chk["budget"]
    notes = json.loads((cdir / "report_notes.json").read_text(encoding="utf-8")) \
        if (cdir / "report_notes.json").exists() else {}
    lit = json.loads((cdir / "literature.json").read_text(encoding="utf-8")) \
        if (cdir / "literature.json").exists() else {"items": []}

    stopped = chk["stop"] or bool(notes.get("stop_reason"))
    status = "Đã dừng" if stopped else ("Tạm dừng chờ quota" if chk["pause_until_reset"] else "Đang chạy")
    reason = notes.get("stop_reason") or "; ".join(chk["reasons"])
    title = cfg.get("title") or slug
    metric = cfg.get("metric") or "CV"
    used_pct = 100 * bud["project_used"] / bud["project_cap"] if bud["project_cap"] else 0
    last_util = next((r.get("gpu_util") for e in reversed(reg.experiments) for r in reversed(e.get("runs", []))
                      if r.get("kind") == "gpu" and r.get("gpu_util")), None)

    # bảng thí nghiệm
    rows = []
    for e in reg.experiments:
        gh = sum(r.get("gpu_hours") or 0 for r in e.get("runs", []) if r.get("kind") == "gpu")
        acc = next((r.get("account") for r in reversed(e.get("runs", [])) if r.get("kind") == "gpu"), "") or "—"
        cls = "best" if best and e["id"] == best["id"] else ""
        rows.append(f'<tr class="{cls}"><td class="n">{E(e["id"])}</td><td><b>{E(e.get("name",""))}</b>'
                    f'<div class="muted">{E(e.get("hypothesis",""))}</div></td>'
                    f'<td class="n">{_fmt(e.get("cv"))}</td><td class="n">{_fmt(e.get("cv_std"), 4)}</td>'
                    f'<td class="n">{gh:.2f}h</td><td class="n">{E(acc)}</td><td><span class="st {E(e.get("status",""))}">'
                    f'{E(e.get("status",""))}</span></td></tr>')

    insights = "".join(f"<li>{E(x)}</li>" for x in notes.get("insights", [])) or \
        '<li class="muted">Agent chưa ghi nhận định.</li>'
    dirs = "".join(
        f'<div class="dir"><b>{i+1}. {E(d.get("title",""))}</b>{E(d.get("why",""))}'
        f'<div class="meta">Chi phí: {E(d.get("cost","?"))} · Kỳ vọng: {E(d.get("expected_gain","?"))}</div></div>'
        for i, d in enumerate(notes.get("next_directions", []))) or '<p class="muted">Chưa có đề xuất.</p>'
    top_lit = sorted(lit["items"], key=lambda i: (i.get("relevance") or 0, i.get("citations") or 0),
                     reverse=True)[:6]
    lit_html = "".join(
        f'<li><a href="{E(i.get("url") or "#")}">{E(i.get("title",""))}</a> '
        f'<span class="muted">({E(i.get("year",""))}{", " + E(i["notes"]) if i.get("notes") else ""})</span></li>'
        for i in top_lit) or '<li class="muted">Chưa có tài liệu.</li>'

    now = datetime.now(timezone.utc).astimezone().strftime("%d/%m/%Y %H:%M")
    doc = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{E(title)} · báo cáo</title>
<style>{CSS}</style></head><body><div class="wrap">
<div class="card"><span class="pill {'stop' if stopped else ''}">{status}</span>
<h1 style="margin-top:10px">{E(title)}</h1>
<div class="muted">Kaggle · {E(slug)} · metric {E(metric)} ({'cao hơn tốt hơn' if direction=='maximize' else 'thấp hơn tốt hơn'})</div>
{f'<p style="margin:12px 0 0">{E(reason)}</p>' if reason else ''}</div>

<div class="card kpis">
<div class="kpi"><div class="l">CV tốt nhất</div><div class="v">{_fmt(best['cv'] if best else None)}</div>
<div class="l">{E(best['id'] + ' · ' + best.get('name','')) if best else ''}</div></div>
<div class="kpi"><div class="l">Thí nghiệm GPU xong</div><div class="v">{len(done)}</div>
<div class="l">tổng {len(reg.experiments)} thí nghiệm</div></div>
<div class="kpi"><div class="l">GPU đã dùng (dự án)</div><div class="v">{bud['project_used']:.1f}h</div>
<div class="l">trên {bud['project_cap']:.0f}h · giai đoạn {E(bud['phase'])}</div></div>
<div class="kpi"><div class="l">Quota tuần ({len(bud['accounts'])} tài khoản)</div><div class="v">{bud['week_used']:.1f}h</div>
<div class="l">trên {bud['week_limit']:.0f}h</div></div>
<div style="flex-basis:100%" class="budget"><div class="bar"><div style="width:{min(used_pct,100):.0f}%"></div></div>
<span class="num">{used_pct:.0f}% ngân sách dự án</span></div></div>

<div class="card"><h2>Tóm tắt</h2><p style="margin:0">{E(notes.get('summary') or 'Chưa có tóm tắt.')}</p></div>
<div class="card"><h2>Tiến triển CV</h2>{_chart(sorted(done, key=lambda e: e['id']), direction)}</div>
<div class="card"><h2>Nhận định</h2><ul>{insights}</ul></div>
<div class="card"><h2>Hướng tiếp theo</h2>{dirs}</div>
<div class="card"><h2>Thí nghiệm</h2><div class="tablewrap"><table>
<tr><th>#</th><th>Thí nghiệm</th><th>CV</th><th>±</th><th>GPU</th><th>TK</th><th>Trạng thái</th></tr>
{''.join(rows) or '<tr><td colspan=7 class="muted">Chưa có.</td></tr>'}</table></div></div>
<div class="card"><h2>Tài khoản Kaggle (quota tuần)</h2>{_accounts(bud['accounts'])}</div>
<div class="card"><h2>Sử dụng 2 GPU (lần chạy gần nhất)</h2>{_gpu_bars(last_util)}</div>
<div class="card"><h2>Tài liệu chính ({len(lit['items'])} mục)</h2><ul>{lit_html}</ul></div>
<footer>Labrat · tạo lúc {now}</footer></div></body></html>"""

    out_dir = cdir / "reports"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"report-{datetime.now().strftime('%Y%m%d-%H%M')}.html"
    path.write_text(doc, encoding="utf-8")
    (out_dir / "latest.html").write_text(doc, encoding="utf-8")
    return path


def short_text(cfg: dict) -> str:
    """Tin nhắn Telegram ngắn đi kèm thẻ HTML."""
    reg = Registry(comp_dir(cfg, cfg["slug"]) / "registry.json")
    direction = cfg.get("metric_direction", "maximize")
    best = reg.best(direction)
    chk = budget.stop_check(cfg, reg)
    b = chk["budget"]
    lines = [f"<b>{E(cfg.get('title') or cfg['slug'])}</b>",
             f"CV tốt nhất: <b>{_fmt(best['cv'] if best else None)}</b>"
             + (f" ({E(best['id'])})" if best else ""),
             f"GPU: {b['project_used']:.1f}/{b['project_cap']:.0f}h dự án · {b['week_used']:.1f}/{b['week_limit']:.0f}h tuần "
             f"({len(b['accounts'])} tài khoản, {b['free_gpu_slots']} đang rảnh)",
             f"Thí nghiệm xong: {len(reg.completed())}/{len(reg.experiments)}"]
    if chk["reasons"]:
        lines.append("⏹ " + E("; ".join(chk["reasons"])))
    return "\n".join(lines)
