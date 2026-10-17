"""The IID probe v2's puzzles are clear of every RL pool (docs/inkling_probes_amendment_1.md).

- The amendment is the one the v2 probes are fitted under: its sha256 is pinned here.
- The committed exclusion list is exactly what the amendment's rule gives, so the v2 puzzles left after it share no
  grid with any training or evaluation pool of RL seeds 0..10,000.
- The same check flags every puzzle of the v1 dataset, which is seed 0's evaluation pool: it would have caught v1.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AMENDMENT = REPO / "docs" / "inkling_probes_amendment_1.md"
AMENDMENT_SHA256 = "84a8d0a4a6cfb96e151478cdfa207b3b58c796d2be36535f43303cd0197a6e95"
EXCLUDED = REPO / "data" / "probe_datasets" / "terminal_prerl_v2" / "excluded_grids.json"
CONFIG = "experiments/configs/terminal_verifier_control.json"

pytest.importorskip("reasoning_gym")
_spec = importlib.util.spec_from_file_location("_probe_v2_exclusions", REPO / "experiments" / "probe_v2_exclusions.py")
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)


def test_amendment_file_has_the_recorded_hash():
    assert hashlib.sha256(AMENDMENT.read_bytes()).hexdigest() == AMENDMENT_SHA256


def test_committed_exclusions_are_what_the_rule_gives(monkeypatch):
    monkeypatch.chdir(REPO)
    committed = json.loads(EXCLUDED.read_text())
    b = committed["baseline"]
    now = ex.exclusions(b["config"], b["seed"], b["n_prompts"], committed["run_seeds_guarded"][1])
    assert now == committed
    assert b == {"config": CONFIG, "seed": 1_000_000, "n_prompts": 512}
    assert committed["run_seeds_guarded"] == [0, 10_000] and committed["n_puzzles"] == 512


def test_v2_puzzles_left_share_no_grid_with_any_rl_pool(monkeypatch):
    monkeypatch.chdir(REPO)
    committed = json.loads(EXCLUDED.read_text())
    dropped = {e["task_id"] for e in committed["excluded"]}
    prompts = ex.v2_prompts(CONFIG, 1_000_000, 512)
    covered = ex.covered_grids(ex.covered_positions(10_000))
    kept = [p for p in prompts if p.meta["task_id"] not in dropped]
    assert len(kept) == 512 - len(dropped) and not any(p.meta["grid"] in covered for p in kept)


def test_the_check_flags_every_v1_puzzle(monkeypatch):
    monkeypatch.chdir(REPO)
    v1 = ex.v2_prompts(CONFIG, 0, 512)  # the v1 pre-RL run's prompts: seed 0's evaluation pool
    covered = ex.covered_grids(ex.covered_positions(10_000))
    assert {p.meta["task_id"].rsplit("_", 1)[0] for p in v1} == {"eval_7919"}
    assert all(p.meta["grid"] in covered for p in v1)
