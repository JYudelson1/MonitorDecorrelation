"""experiments/run_probe_session.py end to end on CPU: tiny 4-bit and BF16 Inkling checkpoints, a tiny TerminalEnv
dataset and a tiny contrastive one, with directory stores standing in for the HF buckets.

Checks:
- Both readers extract both datasets (the 4-bit one split over two workers) and run their diagnostics.
- A deadline that leaves no time for a single shard ends with 76, not a relaunch; a second run finishes.
- A broken optional reader doesn't fail the session and isn't retried; the required reader still finishes.
- A worker stopped by the deadline or SIGTERM with uploads still in flight exits 75 (resumable), not 1 (failed).
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time

import pytest
from inkling_tiny import write_tiny_checkpoints
from test_extract_probe_features import RUN_ENV, TURNS

from monitordecorrelation.whitebox.probe_datasets import counts, terminal_items, write_dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "experiments", "run_probe_session.py")


@pytest.fixture
def setup(tmp_path):
    bf16_dir, nvfp4_dir, _, _ = write_tiny_checkpoints(tmp_path, vocab_size=201024)
    rollouts = tmp_path / "eval_rollouts.jsonl"
    rows = [{"task_id": f"t{t}", "question": f"Solve {t}.", "behavior_present": t == 0 and k == 0,
             "invalid_reason": None,
             "env_meta": {"verifier_mode": "submit_corrupted", "submission_ends": "first",
                          "last_turn_submit_notice": True, "turns": TURNS}}
            for t in range(5) for k in range(2)]
    rollouts.write_text("".join(json.dumps(r) + "\n" for r in rows))
    items, _ = terminal_items(str(rollouts), RUN_ENV, seed=0)
    terminal = tmp_path / "terminal_tiny"
    write_dataset(str(terminal), items, {"kind": "terminal", "effort": 0.5, "counts": counts(items, "task_id")})
    toy = [{"id": f"{i}/{'deceptive' if lab else 'honest'}", "pair": i, "label": lab,
            "split": "test" if i >= 4 else "train", "source": "toy", "question": f"Is {i} even?", "cot": "",
            "answer": ("Yes" if i % 2 == 0 else "No") if not lab else ("No" if i % 2 == 0 else "Yes"), "meta": {}}
           for i in range(6) for lab in (0, 1)]
    contrastive = tmp_path / "toy_contrastive"
    write_dataset(str(contrastive), toy, {"kind": "contrastive", "counts": counts(toy, "pair")})
    from monitordecorrelation.whitebox.probe_datasets import concealed_items

    for it in items:  # the toy hacks used the verifier, so the concealment set has items
        it.setdefault("diagnostics", {})["used_verifier"] = bool(it["label"])
    concealed = tmp_path / "toy_concealed"  # also kind "terminal": must not replace the real one in diagnostics
    hidden = concealed_items(items)
    write_dataset(str(concealed), hidden, {"kind": "terminal", "effort": 0.5, "counts": counts(hidden, "task_id")})
    stores = tmp_path / "stores"
    plan = {"name": "cpu-rehearsal", "out": str(tmp_path / "session"), "shard_size": 4,
            "readers": {"nvfp4": {"model": str(nvfp4_dir), "gpus": [[0], [1]], "device": "cpu",
                                  "store": str(stores / "nvfp4"), "required": True,
                                  "full_activations": {"terminal_tiny": 2}},
                        "bf16": {"model": str(bf16_dir), "gpus": [[0]], "store": str(stores / "bf16"),
                                 "required": False, "stagger": True}},
            "datasets": [str(terminal), str(contrastive), str(concealed)],
            "diagnostics": {"store": str(stores / "diagnostics"), "n_iid": 4, "n_ood": 4},
            "status_store": str(stores / "status")}
    return tmp_path, plan, stores


def run(tmp_path, plan, env=None):
    path = tmp_path / f"plan-{time.time_ns()}.json"
    path.write_text(json.dumps(plan))
    return subprocess.run([sys.executable, SCRIPT, "--plan", str(path)], capture_output=True, text=True, cwd=REPO,
                          timeout=1500, env={**os.environ, **(env or {})})


def shards(store) -> set[str]:
    names = {str(p.relative_to(store)) for p in store.rglob("*") if p.is_file()}
    return {n[:-6] for n in names if n.endswith(".jsonl") and "/runs/" not in f"/{n}"
            and n[:-6] + ".safetensors" in names}


def test_both_readers_extract_everything_and_run_diagnostics(setup):
    tmp_path, plan, stores = setup
    cp = run(tmp_path, plan)
    assert cp.returncode == 0, cp.stdout[-4000:] + cp.stderr[-2000:]
    want_terminal = {f"terminal_tiny/{v}/shard_{k:05d}" for v in ("with_reasoning", "without_reasoning")
                     for k in range(3)}
    want_toy = {f"toy_contrastive/without_reasoning/shard_{k:05d}" for k in range(3)}
    want_hidden = {f"toy_concealed/{v}/shard_00000" for v in ("with_reasoning", "without_reasoning")}
    for reader in ("nvfp4", "bf16"):
        assert shards(stores / reader) == want_terminal | want_toy | want_hidden, reader
        assert (stores / "diagnostics" / reader / "done.json").exists()
        assert (stores / "diagnostics" / reader / "diag.safetensors").exists()
    runs = list((stores / "nvfp4" / "terminal_tiny" / "runs").glob("*.json"))
    assert sorted(json.loads(r.read_text())["part"] for r in runs) == [0, 1]  # two workers shared the dataset
    status = json.loads((stores / "status" / "status.json").read_text())
    assert {r: v["state"] for r, v in status["readers"].items()} == {"nvfp4": "done", "bf16": "done"}
    code = status["attempts"][0]["code"]  # the code version, here and in every output of the session
    assert set(code) == {"commit", "dirty", "diff_sha256", "source"}

    def same_code(other: dict) -> bool:  # each process takes its own: the working tree may change in between
        return (other["commit"], other["source"]) == (code["commit"], code["source"])

    assert same_code(json.loads((stores / "nvfp4" / "terminal_tiny" / "index.json").read_text())["code"])
    assert all(same_code(json.loads(r.read_text())["code"]) for r in runs)
    assert "[nvfp4.1" in cp.stdout and "[bf16.0" in cp.stdout  # worker output is merged, with prefixes
    diag = json.loads((stores / "diagnostics" / "nvfp4" / "done.json").read_text())
    assert same_code(diag["code"])
    assert diag["n_inputs"] == 3 * 2 + 4 and diag["jitter_rel_max"] < 0.05  # 2 + 1 TerminalEnv items x 2 views
    labels = {d["label"] for d in diag["items"] if d["view"] == "with_reasoning"}
    assert labels == {0, 1}  # from the real TerminalEnv set, not the concealment set (hacks only)


def test_no_time_for_a_shard_ends_with_76_and_a_second_run_finishes(setup):
    tmp_path, plan, stores = setup
    plan["readers"].pop("bf16")
    plan.pop("diagnostics")
    first = run(tmp_path, plan, env={"NEBIUS_JOB_DEADLINE": str(time.time() + 60)})  # < the 180 s margin
    assert first.returncode == 76, first.stdout[-4000:]
    assert not shards(stores / "nvfp4")
    second = run(tmp_path, plan)
    assert second.returncode == 0, second.stdout[-4000:]
    assert len(shards(stores / "nvfp4")) == 11  # 6 TerminalEnv + 3 contrastive + 2 concealment


def test_a_broken_optional_reader_does_not_fail_the_session(setup):
    tmp_path, plan, stores = setup
    plan["readers"]["bf16"]["model"] = str(tmp_path / "no-such-checkpoint")
    plan.pop("diagnostics")
    first = run(tmp_path, plan)
    assert first.returncode == 0, first.stdout[-4000:]
    status = json.loads((stores / "status" / "status.json").read_text())
    assert status["readers"]["bf16"]["state"] == "failed" and status["readers"]["nvfp4"]["state"] == "done"
    second = run(tmp_path, plan)
    assert second.returncode == 0 and "failed in an earlier attempt" in second.stdout and "not retried" in second.stdout


class _Prepared:
    """One shard to do, as ``prepare`` would report it."""

    def mine(self):
        return [("without_reasoning", 0)]

    def done(self, view, k):
        return False


@pytest.mark.parametrize("summary, sigterm, expected", [
    # the deadline stopped it while an upload was still running: resumable (this was 1, a "failed" reader)
    ({"stopped_early": "deadline: 200s left", "unsaved": ["uploads still running when time ran out"],
      "out_of_time": True, "shards_extracted": 2}, False, 75),
    # SIGTERM during the last shard: the loop ended, but the 60 s notice cut its upload off
    ({"stopped_early": None, "unsaved": ["uploads still running when time ran out"], "out_of_time": True,
      "shards_extracted": 3}, True, 75),
    # an upload that failed during a stop is redone on resume as well
    ({"stopped_early": "SIGTERM: the VM is being stopped", "unsaved": ["without_reasoning/shard_00001 failed"],
      "shards_extracted": 1}, True, 75),
    # stopped at its deadline before finishing a single shard: no time for anything, not relaunched
    ({"stopped_early": "deadline: 60s left", "unsaved": [], "shards_extracted": 0}, False, 76),
    # the store refuses uploads (quota, permission): a relaunch could save nothing either
    ({"stopped_early": "SIGTERM: the VM is being stopped", "unsaved": ["the store refuses uploads: 403"],
      "store_broken": "HTTPError: 403 Forbidden", "shards_extracted": 1}, True, 1),
    # uploads failed with nothing stopping the run
    ({"stopped_early": None, "unsaved": ["without_reasoning/shard_00000 failed"], "shards_extracted": 1}, False, 1),
])
def test_a_stop_stays_resumable_with_uploads_in_flight(tmp_path, monkeypatch, summary, sigterm, expected):
    """The worker's exit code after its extraction, with the model and the extraction faked."""
    spec = importlib.util.spec_from_file_location("run_probe_session", SCRIPT)
    session = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(session)

    def extract(model, p, *, stop, deadline, log):
        if sigterm:
            signal.raise_signal(signal.SIGTERM)  # Nebius's notice, caught by the worker's StopFlag
        return {"out_of_time": False, "store_broken": None, **summary}

    monkeypatch.setattr(session, "prepare", lambda job: _Prepared())
    monkeypatch.setattr(session, "load_reader", lambda spec: object())
    monkeypatch.setattr(session, "extract", extract)
    plan = {"out": str(tmp_path), "readers": {"nvfp4": {"model": "m", "gpus": [[0]]}}, "datasets": ["data/x"]}
    before = signal.getsignal(signal.SIGTERM)
    try:
        assert session.worker(plan, "nvfp4", 0) == expected
    finally:
        signal.signal(signal.SIGTERM, before)
