"""Build the Inkling probe datasets (``items.jsonl`` + ``manifest.json``). No model is needed: run it on a CPU
box, then push the folder to the GPU box for feature extraction. See whitebox/probe_datasets.py for what an
item is and how the splits are made.

    # IID: valid TerminalEnv episodes of a run, split by task (60/20/20, stratified by "task has a hack")
    uv run python experiments/build_probe_datasets.py terminal \
        --rollouts <run>/eval_rollouts.jsonl --run-info <run>/baseline.json \
        --out data/probe_datasets/terminal_prerl

    # IID v2 (docs/inkling_probes_amendment_1.md): the seed-1,000,000 pre-RL run, minus the excluded puzzles
    uv run python experiments/build_probe_datasets.py terminal \
        --rollouts <run>/eval_rollouts.jsonl --run-info <run>/baseline.json \
        --exclude-grids data/probe_datasets/terminal_prerl_v2/excluded_grids.json \
        --out data/probe_datasets/terminal_prerl_v2

    # Stress: the IID hacks with the policy's own mentions of the verifier removed (levels A and B)
    uv run python experiments/build_probe_datasets.py concealed --source data/probe_datasets/terminal_prerl \
        --out data/probe_datasets/terminal_concealed

    # OOD: the diverse-deception contrastive pairs, split as experiments/train_probe.py splits them
    uv run python experiments/build_probe_datasets.py contrastive --preset diverse_deception \
        --out data/probe_datasets/diverse_deception

``items.jsonl`` is gitignored (``data/**/*.jsonl``) and ``manifest.json`` is tracked. The manifest records
the sources' sha256, the arguments, the counts per split and the repo commit.
"""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from monitordecorrelation.whitebox.probe_datasets import (
    concealed_items,
    contrastive_items,
    counts,
    file_source,
    terminal_items,
    write_dataset,
)

load_dotenv()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kind", required=True)

    t = sub.add_parser("terminal", help="IID: TerminalEnv episodes of one run")
    t.add_argument("--rollouts", required=True, help="the run's eval_rollouts.jsonl")
    t.add_argument("--run-info", required=True, help="the run's baseline.json / run_info.json (env_options, effort)")
    t.add_argument("--fractions", default="0.6,0.2,0.2", help="train,val,test fractions of TASKS")
    t.add_argument("--max-negatives-per-split", type=int, default=None, help="seeded cap; hacks are all kept")
    t.add_argument("--exclude-grids", default=None, metavar="FILE",
                   help="drop every episode on these puzzles first: a JSON with the grid keys under 'excluded' "
                        "(experiments/probe_v2_exclusions.py writes one)")

    c = sub.add_parser("contrastive", help="OOD: contrastive QA pairs from whitebox/datasets")
    c.add_argument("--preset", default=None)
    c.add_argument("--datasets", default=None, help="comma-list of registered adapters (merged with --preset)")
    c.add_argument("--n-per", type=int, default=1000, help="max pairs per source (train_probe.py's default)")
    c.add_argument("--test-frac", type=float, default=0.2)

    s = sub.add_parser("concealed", help="stress: the hacks of a terminal dataset with their disclosures removed")
    s.add_argument("--source", required=True, help="a terminal dataset folder (items.jsonl + manifest.json)")

    for p in (t, c, s):
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", required=True, help="output folder, e.g. data/probe_datasets/<name>")
    args = ap.parse_args()

    if args.kind == "concealed":
        import os

        source = json.load(open(os.path.join(args.source, "manifest.json")))
        with open(os.path.join(args.source, "items.jsonl")) as fh:
            items = concealed_items([json.loads(line) for line in fh])
        manifest = {"kind": "terminal", "args": {"levels": ["A", "B"]}, "effort": source.get("effort"),
                    "sources": {"dataset": {**file_source(os.path.join(args.source, "items.jsonl")),
                                            "name": source.get("name"), "items_sha256": source.get("items_sha256")}},
                    "counts": counts(items, "task_id")}
    elif args.kind == "terminal":
        run_info = json.load(open(args.run_info))
        fractions = tuple(float(x) for x in args.fractions.split(","))
        exclude = None
        if args.exclude_grids:
            from monitordecorrelation.envs.terminal_verifier import GRID_KEY

            spec = json.load(open(args.exclude_grids))
            if spec.get("key") != GRID_KEY:
                raise SystemExit(f"{args.exclude_grids} doesn't key its grids as this code does ({GRID_KEY})")
            exclude = {e["grid_key"] for e in spec["excluded"]}
        items, stats = terminal_items(args.rollouts, run_info["env_options"], fractions=fractions, seed=args.seed,
                                      max_negatives_per_split=args.max_negatives_per_split,
                                      exclude_grid_keys=exclude)
        manifest = {
            "kind": "terminal",
            "args": {"fractions": list(fractions), "seed": args.seed,
                     "max_negatives_per_split": args.max_negatives_per_split},
            "sources": {"rollouts": file_source(args.rollouts),
                        "run_info": {**file_source(args.run_info), "policy": run_info.get("policy"),
                                     "effort": run_info.get("effort"), "env_options": run_info["env_options"]},
                        **({"excluded_grids": {**file_source(args.exclude_grids), "n_grids": len(exclude)}}
                           if exclude is not None else {})},
            "effort": run_info.get("effort"),
            "build": stats,
            "counts": counts(items, "task_id"),
        }
    else:
        from monitordecorrelation.whitebox.datasets import resolve_datasets

        datasets = [s.strip() for s in args.datasets.split(",") if s.strip()] if args.datasets else None
        names = resolve_datasets(datasets, args.preset)
        items = contrastive_items(names, n_per=args.n_per, seed=args.seed, test_frac=args.test_frac)
        manifest = {
            "kind": "contrastive",
            "args": {"preset": args.preset, "datasets": names, "n_per": args.n_per, "seed": args.seed,
                     "test_frac": args.test_frac},
            "counts": counts(items, "pair"),
            "counts_by_source": {name: sum(1 for it in items if it["source"] == name) for name in names},
        }
    written = write_dataset(args.out, items, manifest)
    print(json.dumps({k: written[k] for k in ("name", "kind", "n_items", "counts")}, indent=1))


if __name__ == "__main__":
    main()
