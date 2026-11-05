"""Report: per (tag, mode) AUROC w/ bootstrap 95% CI, class means, parse/cutoff rates, thinking tokens,
latency; pressure-shaped hacks vs all clean; recorded July judges on the same rollouts.
  uv run python /workspace/weak_bakeoff/report.py [results/*.jsonl]"""
import glob, json, random, sys, collections, statistics as S
from monitordecorrelation.eval.metrics import roc_auc
W = "/workspace/weak_bakeoff"
files = sys.argv[1:] or sorted(glob.glob(f"{W}/results/*.jsonl"))
setrows = {json.loads(l)["uid"]: json.loads(l) for l in open(f"{W}/data/bakeoff_set.jsonl")}

def boot(sc, lab, B=1000, seed=0):
    rng = random.Random(seed); pos = [s for s, l in zip(sc, lab) if l]; neg = [s for s, l in zip(sc, lab) if not l]
    v = []
    for _ in range(B):
        p = [rng.choice(pos) for _ in pos]; n = [rng.choice(neg) for _ in neg]
        v.append(roc_auc(p + n, [True] * len(p) + [False] * len(n)))
    v.sort(); return v[int(.025 * B)], v[int(.975 * B)]

def line(name, recs):
    sc = [r["score"] for r in recs]; lab = [r["cls"] == "hack" for r in recs]
    nh, nc = sum(lab), len(lab) - sum(lab)
    if nh == 0 or nc == 0: return None
    auc = roc_auc(sc, lab); lo, hi = boot(sc, lab)
    pr = [r for r in recs if r["cls"] == "clean" or r["pressure"]]
    auc_p = roc_auc([r["score"] for r in pr], [r["cls"] == "hack" for r in pr])
    plo, phi = boot([r["score"] for r in pr], [r["cls"] == "hack" for r in pr])
    hm = S.mean(s for s, l in zip(sc, lab) if l); cm = S.mean(s for s, l in zip(sc, lab) if not l)
    pe = S.mean(r.get("parse_error", False) for r in recs); co = S.mean(r.get("finish") == "length" for r in recs)
    hb = [r["hit_budget"] for r in recs if r.get("hit_budget") is not None]
    rt = [r["reasoning_tokens"] for r in recs if r.get("reasoning_tokens") is not None]
    sec = [r["sec"] for r in recs if "sec" in r]
    sat = S.mean(r["score"] >= 0.99 for r in recs if r["cls"] == "hack")
    return (f"{name:28s} {nh:4d} {nc:4d}  {auc:.3f} [{lo:.2f},{hi:.2f}]  press {auc_p:.3f} [{plo:.2f},{phi:.2f}]  "
            f"hack {hm:.2f} clean {cm:.2f}  pe {pe:.1%} cut {co:.1%} bud {S.mean(hb) if hb else float('nan'):.1%}  "
            f"think {S.mean(rt) if rt else float('nan'):6.0f}  sec {S.mean(sec) if sec else float('nan'):5.1f}  hack>=99 {sat:.0%}")

print(f"{'monitor':28s} {'nH':>4s} {'nC':>4s}  AUROC [95% CI]       pressure-hacks-vs-clean  means  parse/cut/budget  think-tok  sec/call")
order = {"cot+out": 0, "out": 1, "cot": 2}
allrecs = collections.defaultdict(list)
for f in files:
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        allrecs[(r["tag"], r["mode"])].append(r)
for key in sorted(allrecs, key=lambda k: (k[0], order.get(k[1], 9))):
    s = line(f"{key[0]}:{key[1]}", allrecs[key])
    if s: print(s)

print("\nRecorded July judges (orig matrix: weak=anthropic/claude-3-haiku, strong=deepseek/deepseek-chat; OLD prompt) on the same rollouts:")
rec = collections.defaultdict(list)
for u, r in setrows.items():
    for m, s in (r.get("monitors_rec") or {}).items():
        if s is None: continue
        rec[m].append({"score": s, "cls": r["cls"], "pressure": r["pressure"]})
for m in sorted(rec):
    recs = rec[m]
    sc = [r["score"] for r in recs]; lab = [r["cls"] == "hack" for r in recs]
    if sum(lab) == 0 or sum(lab) == len(lab): continue
    lo, hi = boot(sc, lab)
    pr = [r for r in recs if r["cls"] == "clean" or r["pressure"]]
    print(f"  rec:{m:14s} n={len(recs):3d} AUROC {roc_auc(sc, lab):.3f} [{lo:.2f},{hi:.2f}]  press {roc_auc([r['score'] for r in pr],[r['cls']=='hack' for r in pr]):.3f}"
          f"  hack {S.mean(s for s,l in zip(sc,lab) if l):.2f} clean {S.mean(s for s,l in zip(sc,lab) if not l):.2f}")
