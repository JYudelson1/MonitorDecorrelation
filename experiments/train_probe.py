"""Train a white-box linear probe on contrastive deception/sycophancy data.

A probe is generic over its training data: ``--datasets`` is a comma-list of registered adapters, so
the Atlas "diverse deception" probe is just ``--datasets doluschat,truthfulqa,mask,...``. We extract
follow-up-token activations from a local base model, fit a per-layer LR (dropping weak layers), report
held-out AUROC, and save the probe (its meta records the exact dataset list, so it's self-describing).

Run (ood deception probe, tiny model, CPU OK):
  uv run python experiments/train_probe.py --datasets doluschat --out data/probes/ood
Real run: add --model Qwen/Qwen3-8B (only change).
"""

from __future__ import annotations

import argparse

import numpy as np
from dotenv import load_dotenv

from monitordecorrelation.whitebox.datasets import flatten_pairs, load_contrastive
from monitordecorrelation.whitebox.model import WhiteBoxModel
from monitordecorrelation.whitebox.probe import LinearProbe

load_dotenv()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--datasets", default="doluschat", help="comma-list of registered adapters")
    ap.add_argument("--n-per", type=int, default=1000, help="max contrastive PAIRS per source")
    ap.add_argument("--out", default="data/probes/probe")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--ce-threshold", type=float, default=0.6)
    ap.add_argument("--threshold", type=float, default=0.5, help="deploy decision threshold")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = [s.strip() for s in args.datasets.split(",") if s.strip()]
    print(f"datasets={names}  model={args.model}")
    pairs = load_contrastive(names, n_per=args.n_per, seed=args.seed)
    questions, cots, answers, labels = flatten_pairs(pairs)
    triples = list(zip(questions, cots, answers))
    y = np.asarray(labels)
    print(f"{len(pairs)} pairs -> {len(triples)} rows ({int(y.sum())} deceptive / {int((1-y).sum())} honest)")

    model = WhiteBoxModel(args.model)
    print(f"device={model.device} dtype={model.dtype} n_layers={model.n_layers} d_model={model.d_model}")
    print("extracting activations (follow-up token) ...")
    acts = model.extract_activations(triples, batch_size=args.batch_size)

    # Stratified, pair-respecting split: split by PAIR index so an honest/deceptive twin never
    # straddles train/test (would leak). Rows are interleaved [h0,d0,h1,d1,...] from flatten_pairs.
    rng = np.random.default_rng(args.seed)
    n_pairs = len(pairs)
    perm = rng.permutation(n_pairs)
    n_test = max(1, int(round(args.test_frac * n_pairs)))
    test_pairs = set(perm[:n_test].tolist())
    train_rows = [i for i in range(len(triples)) if (i // 2) not in test_pairs]
    test_rows = [i for i in range(len(triples)) if (i // 2) in test_pairs]

    probe = LinearProbe(threshold=args.threshold, ce_threshold=args.ce_threshold)
    probe.fit(acts[train_rows], y[train_rows])
    probe.meta["datasets"] = names
    probe.meta["model_name"] = args.model
    auroc = probe.evaluate(acts[test_rows], y[test_rows])

    print(f"kept layers ({len(probe.kept_layers)}/{model.n_layers}): {probe.kept_layers}")
    print("per-layer train CE: " + ", ".join(f"{l}:{probe.layer_ce[l]:.2f}" for l in sorted(probe.layer_ce)))
    print(f"HELD-OUT AUROC: {auroc:.3f}  (n_test_rows={len(test_rows)})")

    probe.save(args.out)
    print(f"saved probe -> {args.out}")


if __name__ == "__main__":
    main()
