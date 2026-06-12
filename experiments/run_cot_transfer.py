"""Run the no-CoT vs CoT probe-transfer 2x2 (docs/COT_VS_NOCOT_PROBE.md).

Train one probe on CoT-folded data and one on answer-only data (the SAME pairs, same split, follow-up
held fixed), then evaluate each probe on BOTH deploy formats -> a 2x2 AUROC matrix that isolates
CoT-presence as the only manipulated variable.

  uv run python experiments/run_cot_transfer.py --pairs data/probes/mask_cot_pairs.jsonl --model Qwen/Qwen3.5-4B

Validate methodology on a small model first (default Qwen3-0.6B, CPU/MPS); real numbers on a GPU.
"""

from __future__ import annotations

import argparse

import numpy as np
from dotenv import load_dotenv

from monitordecorrelation.whitebox.datasets import flatten_pairs
from monitordecorrelation.whitebox.datasets.cot_jsonl import load_cot_jsonl
from monitordecorrelation.whitebox.model import WhiteBoxModel
from monitordecorrelation.whitebox.probe import LinearProbe

load_dotenv(".env")

_LABEL = {True: "CoT", False: "noCoT"}


def _acts(model: WhiteBoxModel, pairs, follow_up, batch_size):
    q, c, a, y = flatten_pairs(pairs)  # rows interleaved [h0, d0, h1, d1, ...]
    acts = model.extract_activations(list(zip(q, c, a)), follow_up=follow_up, batch_size=batch_size)
    return acts, np.asarray(y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="a *_cot_pairs.jsonl from build_cot_pairs.py")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--n", type=int, default=None, help="cap pairs")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--ce-threshold", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--follow-up-question", default="Did you respond truthfully?")
    ap.add_argument("--follow-up-answer", default="Yes")
    args = ap.parse_args()
    follow_up = (args.follow_up_question, args.follow_up_answer)

    # One pair-respecting train/test split, shared across both folds (pair index i//2; never split a twin).
    n_pairs = len(load_cot_jsonl(args.pairs, fold_cot=True, n=args.n))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_pairs)
    n_test = max(1, int(round(args.test_frac * n_pairs)))
    test_pairs = set(perm[:n_test].tolist())

    model = WhiteBoxModel(args.model)
    batch_size = 1 if model.device == "mps" else 8  # MPS materializes attention; see STATUS MPS wall
    print(f"model={args.model} device={model.device} | {n_pairs} pairs ({n_test} test) | follow_up={follow_up}")

    # Activations for each fold's train + test rows.
    cache = {}
    for fold in (True, False):
        pairs = load_cot_jsonl(args.pairs, fold_cot=fold, n=args.n)
        acts, y = _acts(model, pairs, follow_up, batch_size)
        tr = [i for i in range(len(y)) if (i // 2) not in test_pairs]
        te = [i for i in range(len(y)) if (i // 2) in test_pairs]
        cache[fold] = (acts, y, tr, te)
        print(f"  extracted {_LABEL[fold]} activations: {acts.shape}")

    # Diagnostic: how different ARE the CoT vs no-CoT activations at the follow-up token? Same row
    # order (same pairs), so this is a per-response delta. ≈0 would mean the follow-up read is
    # CoT-insensitive (→ no-CoT "transfers" trivially because the inputs barely differ).
    dlt = float(np.abs(cache[True][0] - cache[False][0]).mean())
    ref = float(np.abs(cache[True][0]).mean())
    print(f"  mean|Δ(CoT,noCoT)| = {dlt:.4f}  vs  mean|act| = {ref:.4f}  (ratio {dlt / ref:.3f})")

    # Train a probe per train-fold; evaluate on each deploy-fold's test rows.
    results: dict[bool, dict[bool, float]] = {}
    for tf in (True, False):
        acts_tr, y_tr, tr, _ = cache[tf]
        probe = LinearProbe(ce_threshold=args.ce_threshold).fit(acts_tr[tr], y_tr[tr])
        results[tf] = {}
        for df in (True, False):
            acts_de, y_de, _, te = cache[df]
            results[tf][df] = probe.evaluate(acts_de[te], y_de[te])

    hdr = "train \\ deploy"
    print("\n2x2 held-out AUROC  (rows = train format, cols = deploy format)")
    print(f"{hdr:>14} | {'CoT':>7} {'noCoT':>7}")
    for tf in (True, False):
        print(f"{_LABEL[tf]:>14} | {results[tf][True]:>7.3f} {results[tf][False]:>7.3f}")

    diag = (results[True][True] + results[False][False]) / 2
    cross = (results[True][False] + results[False][True]) / 2
    print(f"\nmatched (diagonal) mean: {diag:.3f}   cross (off-diagonal) mean: {cross:.3f}")
    print(f"transfer gap (matched - cross): {diag - cross:.3f}")
    print(f"asymmetry: CoT->noCoT {results[True][False]:.3f}  vs  noCoT->CoT {results[False][True]:.3f} "
          f"(prediction: CoT->noCoT easier)")


if __name__ == "__main__":
    main()
