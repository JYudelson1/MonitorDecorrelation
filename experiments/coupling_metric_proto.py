"""Prototype / reference implementation for the degradation-COUPLING metric (NOT wired into the pipeline).

beta_{A->B} = "for each unit monitor A degrades, how much does held-out monitor B degrade", measured as
a within-run slope in a variance-stabilizing transform of AUROC. See docs/DEGRADATION_METRICS.md for the
methodology (transforms, the errors-in-variables attenuation + Deming fix, the shared-eval-noise null,
and the strict-vs-loose / NaN ladder). Run: `uv run python experiments/coupling_metric_proto.py`.

Demonstrates on synthetic data: (1) exact recovery of a known beta when noiseless, (2) attenuation of the
plain slope under finite-eval noise + partial recovery by Deming/TLS, (3) the leverage guard (a run where
A barely degrades yields a high-variance, untrustworthy beta -> flag via leverage_A, don't hard-filter).
"""
from statistics import NormalDist
N = NormalDist()

# ---- transforms (clip to keep them finite; eps ~ AUROC granularity) ----
def clip(a, eps=1e-3): return np.clip(a, eps, 1-eps)
def dprime(a):  return np.sqrt(2)*np.array([N.inv_cdf(x) for x in clip(a)])   # SDT separation, symmetric @0.5
def logit(a):   a=clip(a); return np.log(a/(1-a))                             # symmetric @0.5
def reldec(a):  return -np.log10(1-clip(a))                                   # "reliability decades" (Joey)

TRANSFORMS = {"dprime": dprime, "logit": logit, "reldec": reldec}

# ---- the estimator ----
def coupling(auc_A, auc_B, T, baseline=0):
    """auc_A/auc_B: arrays of shape (n_seeds, n_evalsteps). Returns beta (through-origin slope of
    B-degradation on A-degradation, pooled over seeds+steps), Pearson r, A's degradation 'leverage'."""
    TA, TB = T(auc_A.ravel()).reshape(auc_A.shape), T(auc_B.ravel()).reshape(auc_B.shape)
    dA = (TA[:, [baseline]] - TA).ravel()   # amount A has degraded from baseline (>0 = degraded)
    dB = (TB[:, [baseline]] - TB).ravel()
    Saa, Sbb, Sab = np.sum(dA*dA), np.sum(dB*dB), np.sum(dA*dB)
    beta_ols = Sab/Saa                                    # regress dB on dA (attenuates under x-noise)
    # Deming / TLS through origin with error-variance ratio lam = var(noise_dB)/var(noise_dA):
    lam = 1.0
    beta_tls = (Sbb - lam*Saa + np.sqrt((Sbb-lam*Saa)**2 + 4*lam*Sab**2)) / (2*Sab)
    r = Sab/np.sqrt(Saa*Sbb)
    leverage = dA.reshape(auc_A.shape).max(axis=1).mean()  # how much A actually degraded (power)
    return dict(beta_ols=beta_ols, beta_tls=beta_tls, r=r, leverage_A=leverage)

# ---- synthetic run: A degrades over training; B degrades as beta_true * A (in transform space) ----
def simulate(beta_true, n_seeds=3, n_steps=15, eval_n=64, transform=dprime, noise=True, seed=0):
    rng = np.random.default_rng(seed)
    aucA = np.zeros((n_seeds, n_steps)); aucB = np.zeros((n_seeds, n_steps))
    for s in range(n_seeds):
        # A's TRUE separation shrinks ~linearly from d'(0.98) toward d'(0.6) over training
        tA0, tAT = transform(np.array([0.98]))[0], transform(np.array([0.60]))[0]
        trueTA = np.linspace(tA0, tAT, n_steps)
        trueTB = transform(np.array([0.95]))[0] - beta_true*(tA0 - trueTA)   # B starts 0.95, couples at beta
        # back to AUROC
        aA = np.array([N.cdf(x/np.sqrt(2)) for x in trueTA])
        aB = np.array([N.cdf(x/np.sqrt(2)) for x in trueTB])
        if noise:  # finite-eval AUROC sampling noise (approx DeLong SE, balanced classes)
            def se(a,n): a=clip(a); return np.sqrt(a*(1-a)/(n/2))  # crude but right order
            aA = clip(aA + rng.normal(0, se(aA, eval_n)))
            aB = clip(aB + rng.normal(0, se(aB, eval_n)))
        aucA[s], aucB[s] = aA, aB
    return aucA, aucB

print("Recovering a KNOWN coupling (beta_true set in d' space), transform=dprime\n")
print(f"{'beta_true':>9} | {'beta_ols(noiseless)':>19} | {'beta_ols(eval=64)':>17} | {'beta_tls(eval=64)':>17} | {'r':>5} | {'levA':>5}")
for bt in [0.0, 0.5, 1.0, 1.5]:
    a0,b0 = simulate(bt, noise=False, seed=1)
    c0 = coupling(a0,b0,dprime)
    a1,b1 = simulate(bt, noise=True, seed=1)
    c1 = coupling(a1,b1,dprime)
    print(f"{bt:9.2f} | {c0['beta_ols']:19.3f} | {c1['beta_ols']:17.3f} | {c1['beta_tls']:17.3f} | {c1['r']:5.2f} | {c1['leverage_A']:5.2f}")

print("\nSame data, three transforms (beta_true=1.0, eval=64):")
a1,b1 = simulate(1.0, noise=True, seed=2)
for name,T in TRANSFORMS.items():
    c = coupling(a1,b1,T)
    print(f"  {name:8}: beta_ols={c['beta_ols']:.3f}  beta_tls={c['beta_tls']:.3f}  r={c['r']:.2f}")

print("\nLeverage guard: if A barely degrades, beta is high-variance (unreliable) — flag via leverage_A:")
def simulate_flat(seed):  # A stays ~0.97, B stays ~0.95 (no real degradation)
    rng=np.random.default_rng(seed)
    aA=clip(0.97+rng.normal(0,0.03,(3,15))); aB=clip(0.95+rng.normal(0,0.03,(3,15))); return aA,aB
betas=[coupling(*simulate_flat(s),dprime)['beta_ols'] for s in range(8)]
print(f"  flat-A runs: beta_ols across 8 seeds = {np.mean(betas):.2f} ± {np.std(betas):.2f}  (leverage_A~{coupling(*simulate_flat(0),dprime)['leverage_A']:.2f} → LOW power, distrust)")
