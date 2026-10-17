"""Which of the IID probe v2's puzzles must be dropped because an RL run could train or evaluate on them
(docs/inkling_probes_amendment_1.md).

    uv run python experiments/probe_v2_exclusions.py --out data/probe_datasets/terminal_prerl_v2/excluded_grids.json

**The v2 puzzles** are the 512 prompts the pre-RL baseline (experiments/eval_terminal_monitors_baseline.py) draws at
``--seed`` with ``--n-prompts 512``: the eval pool of that seed, i.e. reasoning-gym puzzles ``Random(seed + 7919 + i)``.

**The RL pools** of a run with seed s are puzzles ``Random(s + i)`` (train) and ``Random(s + 7919 + i)`` (eval), with
i below reasoning-gym's oversampled sizes (4 × 512 and 4 × 256). For every s in 0..``--max-run-seed`` together they
lie within the generator positions 0 .. ``--max-run-seed`` + 7919 + 4 × 256 − 1, so the check generates every puzzle in
that range and drops each v2 puzzle whose grid is among them (a sparse grid can repeat by chance).

``read_only_verifier`` is turned off to build the env here: it changes how commands run, not which puzzles exist,
and it needs root.
"""

from __future__ import annotations

import argparse
import json
import os

from monitordecorrelation.envs.terminal_verifier import GRID_KEY, TerminalVerifierEnv, grid_key

TRAIN_POOL, EVAL_POOL = 512, 256  # the configs' n_prompts_pool and from_task's default eval_n
EVAL_OFFSET = 7_919


def v2_prompts(config: str, seed: int, n_prompts: int):
    """The prompts eval_terminal_monitors_baseline.py samples: its env, built the same way, and its holdout."""
    with open(config) as fh:
        opts = dict(json.load(fh)["env_options"])
    opts["read_only_verifier"] = False
    n = max(64, n_prompts)
    env = TerminalVerifierEnv.from_task(**{**opts, "n": n, "seed": seed, "eval_n": n})
    return env.holdout(n_prompts, seed=seed)


def covered_positions(max_run_seed: int) -> int:
    """How many generator positions (0 .. N − 1) hold every train and eval puzzle of RL seeds 0..max_run_seed."""
    return max(max_run_seed + 4 * TRAIN_POOL, max_run_seed + EVAL_OFFSET + 4 * EVAL_POOL)


def covered_grids(n_positions: int) -> set[str]:
    from reasoning_gym.graphs.largest_island import LargestIslandConfig, LargestIslandDataset

    ds = LargestIslandDataset(LargestIslandConfig(seed=0, size=n_positions))  # position p is Random(0 + p)
    return {"\n".join(" ".join(str(c) for c in row) for row in ds[p]["metadata"]["grid"]) for p in range(n_positions)}


def exclusions(config: str, seed: int, n_prompts: int, max_run_seed: int) -> dict:
    prompts = v2_prompts(config, seed, n_prompts)
    n_pos = covered_positions(max_run_seed)
    covered = covered_grids(n_pos)
    excluded = [{"task_id": p.meta["task_id"], "grid_key": grid_key(p.text)} for p in prompts
                if p.meta["grid"] in covered]
    return {
        "key": GRID_KEY,
        "rule": "drop every v2 puzzle whose grid also occurs among the reasoning-gym largest_island puzzles at "
                f"generator positions 0..{n_pos - 1}, which hold every train and eval pool of RL seeds "
                f"0..{max_run_seed}",
        "baseline": {"config": config, "seed": seed, "n_prompts": n_prompts},
        "run_seeds_guarded": [0, max_run_seed],
        "positions_checked": [0, n_pos - 1],
        "n_puzzles": len(prompts),
        "excluded": sorted(excluded, key=lambda e: e["task_id"]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="experiments/configs/terminal_verifier_control.json")
    ap.add_argument("--seed", type=int, default=1_000_000)
    ap.add_argument("--n-prompts", type=int, default=512)
    ap.add_argument("--max-run-seed", type=int, default=10_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rec = exclusions(args.config, args.seed, args.n_prompts, args.max_run_seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(rec, fh, indent=1)
        fh.write("\n")
    print(f"{rec['n_puzzles']} v2 puzzles; {len(rec['excluded'])} excluded: {[e['task_id'] for e in rec['excluded']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
