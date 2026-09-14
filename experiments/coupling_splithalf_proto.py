"""PROTOTYPE (not wired in): split-half / cross-fit IV coupling for the CAUSAL rows (trained-against target →
held-out), pooled over seeds. Methodology + the pinned decision: docs/DEGRADATION_METRICS.md §"Split-half".
Run: `uv run python experiments/coupling_splithalf_proto.py [batch_dir]` (default: the sep3 batch).

Metric: mu_hack (mean monitor score on strict-hack rollouts). Collapse steps (unparsed>0.5) dropped.
Point estimate: median over random task splits, Δ pairs pooled across the target's runs.
CI: task bootstrap (resample tasks with replacement, independently per run), 90%."""
import json,glob,collections,numpy as np,sys
from pathlib import Path
from monitordecorrelation.eval.coupling import train_target
B=sys.argv[1] if len(sys.argv)>1 else 'data/runs/mbpp_matrix_sep3_20260903'; runs=sorted(glob.glob(B.rstrip('/')+'/*/'))
SH={'probe_ood':'p_ood','cot_only_weak':'cotW','cot_only_strong':'cotS','cot+out_weak':'c+oW','cot+out_strong':'c+oS','out_weak':'outW','out_strong':'outS','probe_iid':'p_iid'}
ALIAS={'cot_weak':'cot+out_weak'}
rng=np.random.default_rng(0)
CANON=['probe_ood','probe_iid','cot_only_weak','cot_only_strong','cot+out_weak','cot+out_strong','out_weak','out_strong']
def load(d):
    recs=[json.loads(l) for l in open(Path(d)/'eval_rollouts_slim.jsonl') if l.strip()]
    mons=list(CANON); assert set(recs[0]['monitors'])==set(CANON), recs[0]['monitors']; by=collections.defaultdict(list)
    for r in recs: by[r['step']].append(r)
    steps=[s for s in sorted(by) if np.mean([bool(r.get('unparsed')) for r in by[s]])<=0.5]
    IDX={s:collections.defaultdict(list) for s in steps}
    for s in steps:
        for r in by[s]: IDX[s][r['task_id']].append((np.array([r['monitors'][m]['score'] for m in mons],float),bool(r['behavior_present'])))
    return mons,steps,IDX,sorted({r['task_id'] for r in recs})
def series(IDX,steps,n,task_list):
    out=np.full((n,len(steps)),np.nan)
    for t,s in enumerate(steps):
        S=[sc for tk in task_list for sc,l in IDX[s][tk] if l]
        if len(S)>=3: out[:,t]=np.nanmean(np.array(S),0)
    return out
def run_diffs(run,task_list):
    mons,steps,IDX,_=run; n=len(mons); tk=list(task_list); rng.shuffle(tk); h=len(tk)//2
    SA,SB=series(IDX,steps,n,tk[:h]),series(IDX,steps,n,tk[h:]); ok=~(np.isnan(SA).any(0)|np.isnan(SB).any(0))
    if ok.sum()<3: return None
    return np.diff(SA[:,ok],axis=1),np.diff(SB[:,ok],axis=1)
def pooled(runs_loaded,task_lists):
    ds=[x for x in (run_diffs(r,t) for r,t in zip(runs_loaded,task_lists)) if x]
    if not ds: return None
    dA=np.hstack([a for a,_ in ds]); dB=np.hstack([b for _,b in ds]); n=dA.shape[0]
    C=np.array([[np.cov(dA[i],dB[j])[0,1] for j in range(n)] for i in range(n)])
    return C,dA.shape[1],len(ds)
by_t=collections.defaultdict(list)
for d in runs: by_t[ALIAS.get(train_target(d),train_target(d))].append(load(d))
MONS=None
rows={}
for tgt,RL in sorted(by_t.items()):
    MONS=RL[0][0]; ti=MONS.index(tgt); n=len(MONS)
    P=[p for p in (pooled(RL,[r[3] for r in RL]) for _ in range(300)) if p]
    Cs=np.array([p[0] for p in P]); npairs=int(np.median([p[1] for p in P])); nr=int(np.median([p[2] for p in P]))
    C=np.median(Cs,0); selfcov=np.diag(C); biv=C[ti]/C[ti,ti]
    BS=[]
    for b in range(400):
        p=pooled(RL,[list(rng.choice(r[3],len(r[3]),replace=True)) for r in RL])
        if p: BS.append(p[0])
    BS=np.array(BS); sc_lo,sc_hi=np.nanpercentile(BS[:,ti,ti],[5,95]); bl=BS[:,ti,:]/BS[:,ti,ti][:,None]; blo,bhi=np.nanpercentile(bl,[5,95],axis=0)
    # self-consistency r for every monitor (diagnostic), pooled
    selfr=np.array([np.median([p[0][i,i] for p in P]) for i in range(n)])
    rows[tgt]=dict(biv=biv,blo=blo,bhi=bhi,npairs=npairs,nr=nr,selfcov=selfcov,sc_lo=sc_lo,sc_hi=sc_hi,Cdiag=np.diag(C))
print('# (B) SPLIT-HALF IV, causal rows, μ_hack, 3 seeds pooled per target, collapse steps dropped\n')
print('## driver gate per row: target\'s own split-half signal cov (×10⁻³), 90% task-bootstrap CI\n')
print('| target | cov | CI | runs used | Δ-pairs |\n|---|---|---|---|---|')
for t,r in rows.items(): print(f"| {SH[t]} | {r['selfcov'][MONS.index(t)]*1e3:+.2f} | [{r['sc_lo']*1e3:+.2f}, {r['sc_hi']*1e3:+.2f}] | {r['nr']} | {r['npairs']} |")
print('\n## β_IV(target → held-out), 90% CI, **bold** = CI excludes 0\n')
print('| target → held-out | '+' | '.join(SH[m] for m in MONS)+' |'); print('|---|'+'---|'*len(MONS))
for t,r in rows.items():
    cells=[]
    for j,m in enumerate(MONS):
        if m==t: cells.append('1'); continue
        s=f"{r['biv'][j]:+.2f} [{r['blo'][j]:+.2f},{r['bhi'][j]:+.2f}]"
        cells.append(f'**{s}**' if (r['blo'][j]>0 or r['bhi'][j]<0) else s)
    print(f'| **{SH[t]}** | '+' | '.join(cells)+' |')
print('\n## held-out signal (split-half cov ×10⁻³ of each monitor\'s own Δμ_hack, pooled over that target\'s runs): is the responder moving at all?\n')
print('| target | '+' | '.join(SH[m] for m in MONS)+' |'); print('|---|'+'---|'*len(MONS))
for t,r in rows.items(): print(f'| {SH[t]} | '+' | '.join(f"{v*1e3:+.2f}" for v in r['Cdiag'])+' |')
