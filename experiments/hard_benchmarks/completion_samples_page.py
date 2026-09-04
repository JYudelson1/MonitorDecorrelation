"""Build the 'Inkling-Small Codeforces Completions' artifact: random completions per effort × length bucket,
plus the completion/prompt length table with 8192 / 12288 / 16384 thresholds."""
from __future__ import annotations

import html
import json
import random
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data" / "hard_benchmarks"
EFFORTS = [  # (label, effort, prompt style, files)
    ("0.5", "LiveCodeBench-style prompt", [D / f"codeforces__{s}__n64_k4_s0.jsonl" for s in ("hard1024", "hard512")]),
    ("0.3", "ImpossibleBench-style prompt, tests ≤ 1000 chars shown",
     [D / "effort0.3" / f"codeforces__{s}__n64_k4_s0__ib.jsonl" for s in ("hard1024", "hard512")]),
    ("0.1", "ImpossibleBench-style prompt, tests ≤ 1000 chars shown",
     [D / "effort0.1" / f"codeforces__{s}__n64_k4_s0__ib.jsonl" for s in ("hard1024", "hard512")]),
]
THRESHOLDS = (8192, 12288, 16384)
PER_BUCKET = 3
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/root/.claude/jobs/a812386e/tmp/inkling_codeforces_completions.html")


def load(files):
    rows = []
    for f in files:
        if f.exists():
            rows += [json.loads(l) for l in open(f)]
    return rows


def length_stats(xs):
    out = {"n": len(xs), "mean": statistics.mean(xs), "median": statistics.median(xs), "max": max(xs)}
    for t in THRESHOLDS:
        under = [x for x in xs if x <= t]
        out[f"frac_over_{t}"] = sum(1 for x in xs if x > t) / len(xs)
        out[f"mean_under_{t}"] = statistics.mean(under) if under else float("nan")
    return out


def stats_rows():
    out = []
    for eff, style, files in EFFORTS:
        for sub, f in zip(("hard1024", "hard512"), files):
            rows = load([f])
            if not rows:
                continue
            by = {}
            for r in rows:
                by.setdefault(r["task_id"], []).append(r["passed"])
            c = length_stats([r["n_output_tokens"] for r in rows])
            p = length_stats(list({r["task_id"]: r["n_prompt_tokens"] for r in rows}.values()))
            out.append({"effort": eff, "style": style, "subset": sub, "comp": c, "prompt": p,
                        "capped": sum(1 for r in rows if r["completion_truncated"]), "n": len(rows),
                        "pass": statistics.mean(sum(v) / len(v) for v in by.values())})
    return out


def pct(x):
    return f"{100 * x:.1f}%"


def fmt(x):
    return f"{x:,.0f}"


def table_html(kind, rows):
    key = "comp" if kind == "completions" else "prompt"
    unit = "completions" if kind == "completions" else "prompts (one per problem)"
    h = [f"<table><caption>{unit.capitalize()} — token lengths</caption><thead><tr><th>effort</th><th>subset</th>"
         "<th class='n'>n</th><th class='n'>mean</th><th class='n'>median</th>"]
    for t in THRESHOLDS:
        h.append(f"<th class='n'>&gt; {t:,}</th><th class='n'>mean ≤ {t:,}</th>")
    if kind == "completions":
        h.append("<th class='n'>hit 32,768</th><th class='n'>pass rate</th>")
    else:
        h.append("<th class='n'>max</th>")
    h.append("</tr></thead><tbody>")
    for r in rows:
        s = r[key]
        sub = "hardest 1024 (rating ≥ 2700)" if r["subset"] == "hard1024" else "hardest 512 (rating ≥ 3000)"
        h.append(f"<tr><td>{r['effort']}<span class='sub'>{html.escape(r['style'])}</span></td><td>{sub}</td>"
                 f"<td class='n'>{s['n']}</td><td class='n'>{fmt(s['mean'])}</td><td class='n'>{fmt(s['median'])}</td>")
        for t in THRESHOLDS:
            h.append(f"<td class='n'>{pct(s[f'frac_over_{t}'])}</td><td class='n'>{fmt(s[f'mean_under_{t}'])}</td>")
        if kind == "completions":
            h.append(f"<td class='n'>{r['capped']}/{r['n']}</td><td class='n'>{pct(r['pass'])}</td>")
        else:
            h.append(f"<td class='n'>{fmt(s['max'])}</td>")
        h.append("</tr>")
    h.append("</tbody></table>")
    return "".join(h)


def cf_link(task_id):
    c, i = task_id.split("/")
    return f"https://codeforces.com/problemset/problem/{c}/{i}"


def card(r, eff):
    outcome = "passed all tests" if r["passed"] else {"no_code": "no code block (truncated)", "timeout": "time limit",
                                                       "wrong_answer": "wrong answer", "runtime_error": "runtime error",
                                                       "compile_error": "compile error"}.get(r["status"], r["status"])
    cls = "ok" if r["passed"] else ("cut" if r["status"] == "no_code" else "bad")
    sub = "hardest 1024" if r["subset"] == "hard1024" else "hardest 512"
    cot = html.escape(r["cot"]) or "<i>(empty)</i>"
    ans = html.escape(r["answer"]) or "<i>(empty — the completion hit the 32,768-token cap before answering)</i>"
    stop = "hit the 32,768 cap" if r["completion_truncated"] else "stopped on its own"
    return f"""<article class="card">
<header>
  <div class="ids"><a href="{cf_link(r['task_id'])}" target="_blank" rel="noopener">Codeforces {html.escape(r['task_id'])}</a>
  <span class="meta">rating {r['difficulty']} · {sub} · sample {r['sample'] + 1} of 4 · effort {eff}</span></div>
  <div class="tags"><span class="tok">{r['n_output_tokens']:,} tokens · {stop}</span><span class="pill {cls}">{outcome}</span></div>
</header>
<details open><summary>Thinking <span class="meta">({len(r['cot']):,} chars)</span></summary><pre class="cot">{cot}</pre></details>
<details open><summary>Answer <span class="meta">(passed {r['n_passed']} of {r['n_tests']} tests)</span></summary><pre class="ans">{ans}</pre></details>
</article>"""


def main():
    rng = random.Random(20260903)
    sections = []
    nav = []
    for eff, style, files in EFFORTS:
        rows = load(files)
        if not rows:
            continue
        buckets = [("any", "any length", rows)] + [(f"le{t}", f"≤ {t:,} tokens", [r for r in rows if r["n_output_tokens"] <= t])
                                                   for t in THRESHOLDS]
        blocks = []
        for bid, blabel, pool in buckets:
            picks = rng.sample(pool, min(PER_BUCKET, len(pool))) if pool else []
            picks.sort(key=lambda r: r["n_output_tokens"])
            sid = f"e{eff.replace('.', '')}-{bid}"
            nav.append((eff, sid, blabel, len(pool), len(rows)))
            blocks.append(f"<section id='{sid}' class='bucket'><h3>Effort {eff} · {blabel} <span class='meta'>"
                          f"{len(picks)} random of {len(pool)} completions</span></h3>" + "".join(card(r, eff) for r in picks) + "</section>")
        sections.append(f"<section class='effort' id='e{eff.replace('.', '')}'><h2>Reasoning effort {eff}</h2>"
                        f"<p class='lede'>{html.escape(style)} · {len(rows)} completions over the same 128 problems "
                        f"(64 from the hardest 1024, 64 from the hardest 512)</p>{''.join(blocks)}</section>")
    srows = stats_rows()
    navhtml = ""
    cur = None
    for eff, sid, blabel, n, tot in nav:
        if eff != cur:
            navhtml += f"<div class='navgrp'>effort {eff}</div>"
            cur = eff
        navhtml += f"<a href='#{sid}'>{blabel}<span>{n}/{tot}</span></a>"
    page = f"""<title>Inkling-Small Codeforces Completions</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@87.5,500;87.5,700;100,400&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
:root {{
  --bg:#F3F5F7; --surface:#FFFFFF; --ink:#1C2430; --muted:#5F6B78; --line:#D6DCE2; --accent:#0F6E74; --accent-ink:#0B565B;
  --ok:#2F7D4F; --ok-bg:#E4F2E9; --bad:#B4473B; --bad-bg:#F7E4E1; --cut:#8A6D1F; --cut-bg:#F6EDD3; --code-bg:#EEF1F4;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#151A20; --surface:#1D242C; --ink:#E6EAEE; --muted:#9AA6B2; --line:#303A45; --accent:#5FC1C6; --accent-ink:#8ED8DC;
  --ok:#7CCB97; --ok-bg:#1E3A2A; --bad:#EE9A8F; --bad-bg:#46231F; --cut:#E2C271; --cut-bg:#3E3418; --code-bg:#161C23;
}} }}
:root[data-theme="dark"] {{
  --bg:#151A20; --surface:#1D242C; --ink:#E6EAEE; --muted:#9AA6B2; --line:#303A45; --accent:#5FC1C6; --accent-ink:#8ED8DC;
  --ok:#7CCB97; --ok-bg:#1E3A2A; --bad:#EE9A8F; --bad-bg:#46231F; --cut:#E2C271; --cut-bg:#3E3418; --code-bg:#161C23;
}}
body {{ background:var(--bg); color:var(--ink); font-family:"Source Sans 3", "Segoe UI", system-ui, sans-serif; font-size:16px; line-height:1.5; margin:0; }}
a {{ color:var(--accent-ink); }}
.wrap {{ display:grid; grid-template-columns: 230px minmax(0, 1fr); gap:32px; max-width:1280px; margin:0 auto; padding:32px 24px 64px; }}
@media (max-width: 860px) {{ .wrap {{ grid-template-columns: 1fr; }} nav {{ position:static !important; }} }}
h1, h2, h3 {{ font-family: Archivo, "Helvetica Neue", Arial, sans-serif; font-stretch: 87.5%; text-wrap: balance; margin:0; }}
h1 {{ font-size:30px; font-weight:700; letter-spacing:-0.01em; }}
h2 {{ font-size:22px; font-weight:700; margin-top:48px; padding-top:16px; border-top:2px solid var(--ink); }}
h3 {{ font-size:17px; font-weight:500; margin:32px 0 12px; }}
.meta {{ color:var(--muted); font-weight:400; font-size:14px; }}
.lede {{ color:var(--muted); margin:4px 0 0; max-width:65ch; }}
header.top {{ grid-column: 1 / -1; display:flex; flex-direction:column; gap:8px; }}
header.top p {{ max-width:70ch; margin:0; }}
nav {{ position:sticky; top:16px; align-self:start; display:flex; flex-direction:column; gap:2px; font-size:14px; }}
.navgrp {{ font-family:Archivo, sans-serif; font-stretch:87.5%; font-weight:500; text-transform:uppercase; letter-spacing:0.08em; font-size:12px; color:var(--muted); margin:14px 0 4px; }}
nav a {{ display:flex; justify-content:space-between; gap:8px; text-decoration:none; color:var(--ink); padding:4px 8px; border-left:2px solid var(--line); }}
nav a span {{ color:var(--muted); font-variant-numeric: tabular-nums; }}
nav a:hover, nav a:focus-visible {{ border-left-color:var(--accent); color:var(--accent-ink); outline:none; }}
main {{ min-width:0; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; font-variant-numeric: tabular-nums; }}
.tablewrap {{ overflow-x:auto; margin:12px 0 4px; }}
caption {{ text-align:left; font-family:Archivo, sans-serif; font-stretch:87.5%; font-weight:500; text-transform:uppercase; letter-spacing:0.08em; font-size:12px; color:var(--muted); padding:0 0 6px; }}
th, td {{ padding:6px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; white-space:nowrap; }}
th {{ font-weight:600; font-size:13px; color:var(--muted); }}
td.n, th.n {{ text-align:right; }}
td .sub {{ display:block; color:var(--muted); font-size:12px; white-space:normal; max-width:26ch; }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:4px; margin:12px 0; padding:14px 18px 6px; }}
.card header {{ display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap; align-items:baseline; padding-bottom:10px; border-bottom:1px solid var(--line); }}
.ids a {{ font-family:Archivo, sans-serif; font-stretch:87.5%; font-weight:700; font-size:17px; text-decoration:none; }}
.ids .meta {{ margin-left:10px; }}
.tags {{ display:flex; gap:10px; align-items:center; }}
.tok {{ font-size:13px; color:var(--muted); font-variant-numeric: tabular-nums; }}
.pill {{ font-family:Archivo, sans-serif; font-stretch:87.5%; font-size:12px; font-weight:500; letter-spacing:0.04em; padding:2px 9px; border-radius:999px; }}
.pill.ok {{ color:var(--ok); background:var(--ok-bg); }} .pill.bad {{ color:var(--bad); background:var(--bad-bg); }} .pill.cut {{ color:var(--cut); background:var(--cut-bg); }}
details {{ margin:10px 0; }}
summary {{ cursor:pointer; font-family:Archivo, sans-serif; font-stretch:87.5%; font-weight:500; font-size:14px; text-transform:uppercase; letter-spacing:0.06em; }}
summary:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
pre {{ margin:8px 0 0; white-space:pre-wrap; overflow-wrap:anywhere; font-size:13.5px; line-height:1.45; max-height:420px; overflow:auto; padding:10px 12px; border-radius:3px; background:var(--code-bg); }}
pre.cot {{ font-family:"Source Sans 3", system-ui, sans-serif; font-size:15px; }}
pre.ans {{ font-family:"JetBrains Mono", ui-monospace, Menlo, monospace; }}
</style>
<div class="wrap">
<header class="top">
  <h1>Inkling-Small Codeforces Completions</h1>
  <p>Randomly drawn completions from the untrained policy (thinkingmachines/Inkling-Small via tinker, temperature 1, 32,768-token
  prompt and completion budgets) on the hardest 1024 and hardest 512 problems of open-r1/codeforces by rating, at three reasoning
  efforts. Buckets are by completion length; the same problems were used for every effort. Each card links to the original problem.</p>
  <div class="tablewrap">{table_html("completions", srows)}</div>
  <div class="tablewrap">{table_html("prompts", srows)}</div>
</header>
<nav>{navhtml}</nav>
<main>{''.join(sections)}</main>
</div>
"""
    OUT.write_text(page)
    print(OUT, f"{OUT.stat().st_size / 1e6:.1f} MB")
    for r in srows:
        c = r["comp"]
        print(f"{r['effort']} {r['subset']:9s} comp: mean={c['mean']:.0f} med={c['median']:.0f} " +
              " ".join(f">{t}={pct(c[f'frac_over_{t}'])} mean<={t}={c[f'mean_under_{t}']:.0f}" for t in THRESHOLDS) +
              f" capped={r['capped']} pass={pct(r['pass'])} | prompt mean={r['prompt']['mean']:.0f} " +
              " ".join(f">{t}={pct(r['prompt'][f'frac_over_{t}'])}" for t in THRESHOLDS))


if __name__ == "__main__":
    main()
