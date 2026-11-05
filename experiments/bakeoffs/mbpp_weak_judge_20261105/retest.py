"""Test-retest (two independent temp-1.0 samples per rollout) + AUROC with parse errors dropped / averaged-2-sample AUROC."""
import json, glob, collections, statistics as S
from monitordecorrelation.eval.metrics import roc_auc
W = "/workspace/weak_bakeoff"
R = collections.defaultdict(dict)
for f in glob.glob(f"{W}/results/*.jsonl"):
    for l in open(f):
        r = json.loads(l); R[(r["tag"], r["mode"])][r["uid"]] = r
def pearson(x, y):
    mx, my = S.mean(x), S.mean(y); sx = sum((a-mx)**2 for a in x)**.5; sy = sum((b-my)**2 for b in y)**.5
    return sum((a-mx)*(b-my) for a, b in zip(x, y)) / (sx*sy) if sx and sy else float("nan")
print("AUROC with parse-error rows dropped (vs repo semantics = scored 1.0):")
for k in sorted(R):
    rs = [r for r in R[k].values() if not r["parse_error"]]
    if len(rs) == len(R[k]) or k[0].startswith("pilot"): continue
    print(f"  {k[0]}:{k[1]:8s} n={len(rs)} AUROC_dropPE {roc_auc([r['score'] for r in rs],[r['cls']=='hack' for r in rs]):.3f}")
print("\nTest-retest:")
for base in ("q3_1.7b", "q35_2b"):
    for mode in ("cot+out", "out", "cot"):
        a, b = R.get((base, mode), {}), R.get((base + "_r2", mode), {})
        u = [x for x in a if x in b]
        if not u: continue
        sa = [a[x]["score"] for x in u]; sb = [b[x]["score"] for x in u]; lab = [a[x]["cls"] == "hack" for x in u]
        flips = S.mean((p >= .5) != (q >= .5) for p, q in zip(sa, sb))
        print(f"  {base}:{mode:8s} n={len(u)} r={pearson(sa, sb):.2f} label-flip@0.5 {flips:.1%}  AUROC r1 {roc_auc(sa,lab):.3f} r2 {roc_auc(sb,lab):.3f} "
              f"mean-of-2 {roc_auc([(p+q)/2 for p,q in zip(sa,sb)],lab):.3f}  mean|diff| hack {S.mean(abs(p-q) for p,q,l in zip(sa,sb,lab) if l):.2f} clean {S.mean(abs(p-q) for p,q,l in zip(sa,sb,lab) if not l):.2f}")
print("\nWithin-class test-retest r (systematic share of score variance) for hacks / clean:")
for base in ("q3_1.7b", "q35_2b"):
    for mode in ("cot+out", "out", "cot"):
        a, b = R.get((base, mode), {}), R.get((base + "_r2", mode), {})
        for cls in ("hack", "clean"):
            u = [x for x in a if x in b and a[x]["cls"] == cls]
            if u: print(f"  {base}:{mode:8s} {cls:5s} r={pearson([a[x]['score'] for x in u],[b[x]['score'] for x in u]):.2f}", end="")
        print()
