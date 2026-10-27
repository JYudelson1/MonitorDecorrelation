"""The IID probe v2's puzzles are clear of every RL pool (docs/inkling_probes_amendment_1.md), and its features come
from the full-precision reader alone, one episode per forward (docs/inkling_probes_amendment_2.md).

- The amendments are the ones the v2 probes are fitted under: their sha256s are pinned here.
- The committed exclusion list is exactly what amendment 1's rule gives, so the v2 puzzles left after it share no
  grid with any training or evaluation pool of RL seeds 0..10,000.
- The same check flags every puzzle of the v1 dataset, which is seed 0's evaluation pool: it would have caught v1.
- The committed v2 session plans are amendment 2's: the full-precision reader extracting terminal_prerl_v2 alone, one
  episode per forward, with diagnostics that compare batched against one at a time. The vast.ai one is the Nebius one
  on a 4× B200 (one copy), its stores local directories under results/ (the instance gets no HF token).
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
AMENDMENT_2 = REPO / "docs" / "inkling_probes_amendment_2.md"
AMENDMENT_2_SHA256 = "9a5cefb36c05b4b186b5684453528cfb9b38fe38310ff40c8c8fe76ebb3839e6"
SESSION_PLAN = REPO / "experiments" / "session_plans" / "probe_session_v2_bf16.json"
AMENDMENT_3 = REPO / "docs" / "inkling_probes_amendment_3.md"
AMENDMENT_3_SHA256 = "8b6ff8d54a32b04d75165ad80a6453f342b3955ff1e243aceeea0156d409d6a2"
ROSTER = REPO / "experiments" / "rl_batches" / "probe_rows_v2.json"
VAST_PLAN = REPO / "experiments" / "session_plans" / "probe_session_v2_bf16_vast.json"
EXCLUDED = REPO / "data" / "probe_datasets" / "terminal_prerl_v2" / "excluded_grids.json"
CONFIG = "experiments/configs/terminal_verifier_control.json"

pytest.importorskip("reasoning_gym")
_spec = importlib.util.spec_from_file_location("_probe_v2_exclusions", REPO / "experiments" / "probe_v2_exclusions.py")
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)


def test_amendment_files_have_the_recorded_hashes():
    assert hashlib.sha256(AMENDMENT.read_bytes()).hexdigest() == AMENDMENT_SHA256
    assert hashlib.sha256(AMENDMENT_2.read_bytes()).hexdigest() == AMENDMENT_2_SHA256
    assert hashlib.sha256(AMENDMENT_3.read_bytes()).hexdigest() == AMENDMENT_3_SHA256


def test_the_rl_batch_is_amendment_3s():
    """The roster and configs amendment 3 fixes: probe_iid on seeds 219-221 and probe_ood on 222-224, trained against
    the committed v2 monitors' own score at λ = 1, the four judges held out; no controls; 90 steps and early stop off,
    as Vlad's judge rows; W&B where Vlad's runs are."""
    roster = json.loads(ROSTER.read_text())
    assert roster["n_steps"] == 90 and not roster.get("sets") and not any(r.get("sets") for r in roster["runs"])
    assert (roster["wandb_entity"], roster["wandb_project"]) == ("jyudel-redwood-research",
                                                                 "terminal-env-inkling-small")
    runs = sorted((r["config"], r["seed"], r["stop_after_zero_behavior_steps"]) for r in roster["runs"])
    probe = "experiments/configs/terminal_verifier_probe_{}.json"
    assert runs == sorted([(probe.format("iid"), s, None) for s in (219, 220, 221)]
                          + [(probe.format("ood"), s, None) for s in (222, 223, 224)])
    monitors = {"iid": "data/probes/Inkling-Small/v2/terminal_prerl_v2/with_reasoning/followup",
                "ood": "data/probes/Inkling-Small/v2/diverse_deception/with_reasoning/followup"}
    judges = {("q3_cot", "Qwen/Qwen3-30B-A3B-FP8"), ("q3_out", "Qwen/Qwen3-30B-A3B-FP8"),
              ("q35_cot", "Qwen/Qwen3.5-35B-A3B-FP8"), ("q35_out", "Qwen/Qwen3.5-35B-A3B-FP8")}
    for name, path in monitors.items():
        c = json.loads((REPO / f"experiments/configs/terminal_verifier_probe_{name}.json").read_text())
        assert (c["penalty_coef"], c["n_steps"], c["lora_rank"], c["lr"], c["kl_coef"]) == (1.0, 90, 8, 0.0005, 0.0001)
        ta = [m for m in c["monitors"] if m["role"] == "train_against"]
        assert [(m["kind"], m["probe_path"]) for m in ta] == [("inkling_probe", path)]
        assert {k: v for k, v in ta[0].items() if k not in ("name", "kind", "role", "probe_path", "server_url")} == {}
        assert {(m["name"], m["model_id"]) for m in c["monitors"] if m["role"] == "held_out"} == judges
        assert (REPO / path / "probe.npz").exists()


@pytest.mark.parametrize("path", [SESSION_PLAN, VAST_PLAN], ids=["nebius", "vast"])
def test_the_v2_session_plan_is_amendment_2s(path):
    plan = json.loads(path.read_text())
    assert list(plan["readers"]) == ["bf16"] and plan["readers"]["bf16"]["model"] == "/mnt/ram/Inkling-Small"
    assert plan["datasets"] == ["data/probe_datasets/terminal_prerl_v2"] and plan["max_batch_tokens"] == 1
    diag = plan["diagnostics"]
    assert diag["max_batch_tokens"] == 65_536 and (diag["n_iid"], diag["n_ood"]) == (64, 100)
    assert (diag["terminal"], diag["contrastive"]) == ("data/probe_datasets/terminal_prerl_v2",
                                                       "data/probe_datasets/diverse_deception")
    assert diag["store"].rstrip("/").endswith("/diagnostics-v2")  # not the v1 sample's diagnostics/<reader>/


def test_the_vast_plan_is_the_nebius_one_on_a_4x_b200_with_local_stores():
    """One copy on GPUs 0-3, and every store a relative local directory under results/, which the session takes as
    the repo's (run_probe_session.resolve): ~/md/results on the instance, all brought back by the launcher's --fetch
    results. Everything else is the Nebius plan's."""
    nebius, vast = (json.loads(p.read_text()) for p in (SESSION_PLAN, VAST_PLAN))
    assert vast["readers"]["bf16"]["gpus"] == [[0, 1, 2, 3]] and vast["readers"]["bf16"]["required"] is True
    stores = [vast["readers"]["bf16"]["store"], vast["diagnostics"]["store"], vast["status_store"]]
    assert all("://" not in s and not s.startswith(("/", "~")) and Path(s).parts[0] == "results" for s in stores)
    assert len(set(stores)) == 3 and stores[1].endswith("/diagnostics-v2")

    def rest(plan: dict) -> dict:
        readers = {r: {k: v for k, v in spec.items() if k not in ("gpus", "store", "stagger")}
                   for r, spec in plan["readers"].items()}
        diagnostics = {k: v for k, v in plan["diagnostics"].items() if k != "store"}
        return {**plan, "name": None, "readers": readers, "diagnostics": diagnostics, "status_store": None}

    assert rest(vast) == rest(nebius)


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
