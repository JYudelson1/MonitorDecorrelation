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

from monitordecorrelation.whitebox.datasets import (
    PROBE_PRESETS,
    flatten_pairs,
    load_contrastive,
    resolve_datasets,
)
from monitordecorrelation.whitebox.model import WhiteBoxModel
from monitordecorrelation.whitebox.probe import LinearProbe

load_dotenv()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--datasets", default=None, help="comma-list of registered adapters")
    ap.add_argument("--preset", default=None,
                    help="named ensemble recipe (merged with --datasets): " + ", ".join(sorted(PROBE_PRESETS)))
    ap.add_argument("--n-per", type=int, default=1000, help="max contrastive PAIRS per source")
    ap.add_argument("--out", default=None,
                    help="default: data/probes/<model_slug>/<datasets> (probes live in a per-model folder)")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--ce-threshold", type=float, default=0.6)
    ap.add_argument("--threshold", type=float, default=0.5, help="deploy decision threshold")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    # Follow-up turn (Atlas §4.3). Default is deception-framed; override for a behavior-specific probe.
    ap.add_argument("--follow-up-question", default="Did you respond truthfully?")
    ap.add_argument("--follow-up-answer", default="Yes")
    ap.add_argument("--push-to-hub", default=None, metavar="repo_id",
                    help="also push the trained probe to this HF repo (needs HF_TOKEN)")
    args = ap.parse_args()

    follow_up = (args.follow_up_question, args.follow_up_answer)

    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()] if args.datasets else None
    names = resolve_datasets(datasets, args.preset)
    # Probes live in a per-model folder so probes for different base models never collide. A preset
    # gets a clean folder name (e.g. .../diverse_deception) instead of the full joined dataset list.
    slug = args.preset or "-".join(names)
    out = args.out or f"data/probes/{args.model.split('/')[-1]}/{slug}"
    print(f"datasets={names}  model={args.model}")
    pairs = load_contrastive(names, n_per=args.n_per, seed=args.seed)
    questions, cots, answers, labels = flatten_pairs(pairs)
    triples = list(zip(questions, cots, answers))
    y = np.asarray(labels)
    print(f"{len(pairs)} pairs -> {len(triples)} rows ({int(y.sum())} deceptive / {int((1-y).sum())} honest)")

    model = WhiteBoxModel(args.model)
    print(f"device={model.device} dtype={model.dtype} n_layers={model.n_layers} d_model={model.d_model}")
    print(f"extracting activations (follow-up: {follow_up!r}) ...")
    acts = model.extract_activations(triples, follow_up=follow_up, batch_size=args.batch_size, progress=True)

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
    probe.meta["preset"] = args.preset
    probe.meta["model_name"] = args.model
    probe.meta["follow_up"] = list(follow_up)  # deploy must use the identical follow-up
    auroc = probe.evaluate(acts[test_rows], y[test_rows])

    print(f"kept layers ({len(probe.kept_layers)}/{model.n_layers}): {probe.kept_layers}")
    print("per-layer train CE: " + ", ".join(f"{l}:{probe.layer_ce[l]:.2f}" for l in sorted(probe.layer_ce)))
    print(f"HELD-OUT AUROC: {auroc:.3f}  (n_test_rows={len(test_rows)})")

    probe.save(out)
    print(f"saved probe -> {out}")

    if args.push_to_hub:
        url = probe.push_to_hub(args.push_to_hub)
        print(f"pushed probe -> {url}")


if __name__ == "__main__":
    main()
