"""experiments/extract_probe_features.py end to end, on CPU.

The inputs are a tiny NVFP4 Inkling with the real vocabulary and a small terminal dataset written by the
dataset builder. Checks:

- One shard per view per ``--shard-size`` items, and readouts that land on the follow-up "Yes".
- An index that describes the run.
- Full activations for the requested items.
- A second run resumes: finished shards are skipped.
- The code version in the index and in each run record, with the shards that run extracted.
- With ``--store``:
  - a run on a "new VM" (empty ``--out``) redoes only what the store lacks, also when the stored shards come from
    another commit;
  - a deadline or SIGTERM stops it cleanly with exit 75;
  - it refuses to mix shards made with other settings;
  - it refuses an ``items.jsonl`` that is not the file its manifest describes (``items_sha256``);
  - two processes (``--part 0/2``, ``1/2``) share a dataset through one store.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time

import pytest
import torch
from inkling_tiny import write_tiny_checkpoints

from monitordecorrelation.whitebox.features import read_shard
from monitordecorrelation.whitebox.inkling_replay import FOLLOW_UP, InklingReplay
from monitordecorrelation.whitebox.probe_datasets import counts, terminal_items, write_dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ENV = {"verifier_mode": "submit_corrupted", "submission_ends": "first", "max_turns": 4,
           "require_explanation": True, "last_turn_submit_notice": True}
TURNS = [
    {"cot": "Look.", "text": "<command>ls</command> Explanation: list.", "output": "verifier.py",
     "submission": None, "truncated": False},
    {"cot": "Submit.", "text": "<submit>19</submit> Explanation: computed.", "output": None, "submission": "19",
     "truncated": False},
]


SCRIPT = os.path.join(REPO, "experiments", "extract_probe_features.py")
PROGRESS = re.compile(r"shard \d+/\d+:")  # one line per extracted shard
CODE_KEYS = {"commit", "dirty", "diff_sha256", "source"}


def run(args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True, cwd=REPO, timeout=600,
                          env={**os.environ, **(env or {})})


def summary(cp: subprocess.CompletedProcess) -> dict:
    return json.loads(cp.stdout.strip().splitlines()[-1])


def tiny_setup(tmp_path):
    """A tiny NVFP4 reader and a 10-episode terminal dataset (6 shards of <= 4 items over two views)."""
    _, nvfp4_dir, _, _ = write_tiny_checkpoints(tmp_path, vocab_size=201024)
    return (nvfp4_dir, *tiny_dataset(tmp_path))


def tiny_dataset(tmp_path):
    """The 10-episode terminal dataset alone: (folder, items)."""
    rollouts = tmp_path / "eval_rollouts.jsonl"
    rows = [{"task_id": f"t{t}", "question": f"Solve {t}.", "behavior_present": t == 0 and k == 0,
             "invalid_reason": None,
             "env_meta": {"verifier_mode": "submit_corrupted", "submission_ends": "first",
                          "last_turn_submit_notice": True, "turns": TURNS}}
            for t in range(5) for k in range(2)]
    rollouts.write_text("".join(json.dumps(r) + "\n" for r in rows))
    items, _ = terminal_items(str(rollouts), RUN_ENV, seed=0)
    dataset = tmp_path / "terminal_tiny"
    write_dataset(str(dataset), items, {"kind": "terminal", "effort": 0.5, "counts": counts(items, "task_id")})
    return dataset, items


def common_args(nvfp4_dir, dataset, out) -> list[str]:
    return ["--dataset", str(dataset), "--model", str(nvfp4_dir), "--device", "cpu", "--out", str(out),
            "--shard-size", "4"]


def run_records(folder) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((folder / "runs").glob("*.json"))]


def stored_shards(store) -> set[str]:
    """Shards complete in the store: marker and data both there."""
    names = {str(p.relative_to(store)) for p in store.rglob("*") if p.is_file()}
    return {n[:-len(".jsonl")] for n in names if n.endswith(".jsonl") and n[:-6] + ".safetensors" in names}


def test_extraction_writes_resumable_shards(tmp_path):
    nvfp4_dir, dataset, items = tiny_setup(tmp_path)
    out = tmp_path / "features"
    args = ["--dataset", str(dataset), "--model", str(nvfp4_dir), "--device", "cpu", "--out", str(out),
            "--shard-size", "4", "--full-activations", "2"]
    first = run(args)
    assert first.returncode == 0, first.stderr[-3000:]

    index = json.loads((out / "index.json").read_text())
    assert index["views"] == ["with_reasoning", "without_reasoning"] and index["n_items"] == 10
    assert index["shards"] == {"with_reasoning": 3, "without_reasoning": 3}
    assert index["n_layers"] == 5 and index["follow_up"] == list(FOLLOW_UP)
    yes = InklingReplay().answer_tokens[-1]
    seen = []
    for view in index["views"]:
        for k in range(3):
            tensors, records = read_shard(str(out / view / f"shard_{k:05d}"))
            offsets = tensors["offsets"].tolist()
            for r in records:
                tokens = tensors["tokens"][offsets[r["row"]]:offsets[r["row"] + 1]].tolist()
                assert tokens[int(tensors["readout"][r["row"]])] == yes
                assert r["view"] == view and r["n_tokens"] == len(tokens)
            assert tensors["followup"].shape == tensors["followup_prev"].shape == (len(records), 5, 64)
            assert tensors["followup"].dtype == torch.bfloat16 and tensors["mean_action"].dtype == torch.float32
            seen += [(view, r["id"]) for r in records]
        assert len(os.listdir(out / view / "full")) == 2  # one positive + one negative from train
    assert sorted(seen) == sorted((v, it["id"]) for v in index["views"] for it in items)

    assert summary(first)["complete"] is True and index["feature_format"] == "inkling-probe-features/v2"
    assert [r["shards_extracted"] for r in run_records(out)] == [6]
    [record] = run_records(out)
    assert set(index["code"]) == CODE_KEYS and record["code"] == index["code"]
    assert sorted(record["extracted"]) == sorted(f"{v}/shard_{k:05d}" for v in index["views"] for k in range(3))

    stamp = (out / "with_reasoning" / "shard_00000.safetensors").stat().st_mtime_ns
    second = run(args)
    assert second.returncode == 0, second.stderr[-3000:]
    assert not PROGRESS.search(second.stdout) and "6 of 6 shards already done" in second.stdout
    assert (out / "with_reasoning" / "shard_00000.safetensors").stat().st_mtime_ns == stamp
    assert summary(second)["shards_extracted"] == 0


def test_a_new_vm_resumes_from_the_store(tmp_path):
    nvfp4_dir, dataset, _ = tiny_setup(tmp_path)
    store = tmp_path / "store"
    first = run([*common_args(nvfp4_dir, dataset, tmp_path / "vm1"), "--store", str(store), "--full-activations", "2"])
    assert first.returncode == 0, first.stderr[-3000:]
    everything = {f"{v}/shard_{k:05d}" for v in ("with_reasoning", "without_reasoning") for k in range(3)}
    assert stored_shards(store) == everything
    assert len(list((store / "with_reasoning" / "full").glob("*.safetensors"))) == 2
    assert summary(first)["complete"] is True

    # Lost on the way: one shard's marker (its data made it), another shard's data (its marker made it).
    (store / "with_reasoning" / "shard_00001.jsonl").unlink()
    (store / "without_reasoning" / "shard_00002.safetensors").unlink()
    index = json.loads((store / "index.json").read_text())  # and the shards there came from another commit
    other = {"commit": "0" * 40, "dirty": False, "diff_sha256": None, "source": "file"}
    (store / "index.json").write_text(json.dumps({**index, "code": other}))
    second = run([*common_args(nvfp4_dir, dataset, tmp_path / "vm2"), "--store", str(store),
                  "--full-activations", "2"])
    assert second.returncode == 0, second.stderr[-3000:]
    assert "4 of 6 shards already done" in second.stdout and summary(second)["shards_extracted"] == 2
    assert stored_shards(store) == everything
    assert sorted(p.name for p in (tmp_path / "vm2").rglob("shard_*.jsonl")) == ["shard_00001.jsonl",
                                                                                 "shard_00002.jsonl"]
    assert summary(second)["complete"] is True
    assert sorted(r["shards_extracted"] for r in run_records(store)) == [2, 6]
    assert sorted(len(r["extracted"]) for r in run_records(store)) == [2, 6]
    [last] = [r for r in run_records(store) if r["shards_extracted"] == 2]
    assert json.loads((store / "index.json").read_text())["code"] == last["code"] != other  # the last run's


def test_a_deadline_stops_before_a_shard_and_the_next_run_finishes(tmp_path):
    nvfp4_dir, dataset, _ = tiny_setup(tmp_path)
    store = tmp_path / "store"
    args = [*common_args(nvfp4_dir, dataset, tmp_path / "vm1"), "--store", str(store)]
    first = run([*args, "--deadline-margin", "600"], env={"NEBIUS_JOB_DEADLINE": str(time.time() + 300)})
    assert first.returncode == 75, first.stderr[-3000:]
    assert "deadline" in summary(first)["stopped_early"] and summary(first)["shards_extracted"] == 0
    assert summary(first)["complete"] is False and (store / "index.json").exists()  # saved though nothing else was
    [record] = run_records(store)
    assert "deadline" in record["stopped_early"] and record["deadline_utc"] is not None

    second = run([*common_args(nvfp4_dir, dataset, tmp_path / "vm2"), "--store", str(store)])
    assert second.returncode == 0, second.stderr[-3000:]
    assert summary(second)["shards_extracted"] == 6 and summary(second)["complete"] is True


def test_sigterm_starts_nothing_new_and_keeps_what_is_saved(tmp_path):
    nvfp4_dir, dataset, _ = tiny_setup(tmp_path)
    store = tmp_path / "store"
    proc = subprocess.Popen([sys.executable, SCRIPT, *common_args(nvfp4_dir, dataset, tmp_path / "vm1"),
                             "--store", str(store)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            cwd=REPO)
    first_line = proc.stdout.readline()  # printed before the model loads
    assert "shards already done" in first_line
    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=600)
    assert proc.returncode == 75, err[-3000:]
    result = json.loads(out.strip().splitlines()[-1])
    assert "SIGTERM" in result["stopped_early"] and result["shards_extracted"] < 6
    assert result["store_broken"] is None and result["out_of_time"] in (True, False)
    names = {str(p.relative_to(store)) for p in store.rglob("*.jsonl")}
    assert {n[:-6] for n in names} == stored_shards(store)  # every marker has its data
    assert result["complete"] is False and "SIGTERM" in run_records(store)[0]["stopped_early"]


def test_settings_that_change_the_shards_are_refused(tmp_path):
    nvfp4_dir, dataset, _ = tiny_setup(tmp_path)
    store = tmp_path / "store"
    first = run([*common_args(nvfp4_dir, dataset, tmp_path / "vm1"), "--store", str(store), "--limit", "6"])
    assert first.returncode == 0, first.stderr[-3000:]
    args = common_args(nvfp4_dir, dataset, tmp_path / "vm2")
    args[args.index("--shard-size") + 1] = "3"
    second = run([*args, "--store", str(store), "--limit", "6"])
    assert second.returncode != 0 and "refusing to mix" in second.stderr and "shard_size" in second.stderr


def test_two_parts_share_a_dataset_through_one_store(tmp_path):
    nvfp4_dir, dataset, _ = tiny_setup(tmp_path)
    store = tmp_path / "store"
    procs = [subprocess.Popen([sys.executable, SCRIPT, *common_args(nvfp4_dir, dataset, tmp_path / f"gpu{i}"),
                               "--store", str(store), "--part", f"{i}/2"], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, cwd=REPO) for i in range(2)]
    results = [proc.communicate(timeout=600) for proc in procs]
    for proc, (out, err) in zip(procs, results):
        assert proc.returncode == 0, err[-3000:]
    extracted = [json.loads(out.strip().splitlines()[-1])["shards_extracted"] for out, _ in results]
    assert extracted == [3, 3]  # 6 shards, every other one each
    everything = {f"{v}/shard_{k:05d}" for v in ("with_reasoning", "without_reasoning") for k in range(3)}
    assert stored_shards(store) == everything and len(run_records(store)) == 2
    again = run([*common_args(nvfp4_dir, dataset, tmp_path / "gpu9"), "--store", str(store)])
    assert again.returncode == 0 and summary(again)["shards_extracted"] == 0 and summary(again)["complete"] is True


def test_items_that_are_not_the_manifests_are_refused(tmp_path):
    """An items.jsonl edited after the dataset was built (here one label flipped) no longer has the manifest's
    items_sha256, which index.json would record for its shards: refused before any model loads."""
    from monitordecorrelation.whitebox.extraction import Job, prepare

    dataset, items = tiny_dataset(tmp_path)
    assert len(prepare(Job(dataset=str(dataset), out=str(tmp_path / "ok"), model="unused")).items) == len(items)
    lines = (dataset / "items.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    first["label"] = 1 - first["label"]
    (dataset / "items.jsonl").write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n")
    with pytest.raises(SystemExit, match="not the file its records describe") as refused:
        prepare(Job(dataset=str(dataset), out=str(tmp_path / "edited"), model="unused"))
    assert "manifest.json records" in str(refused.value) and not (tmp_path / "edited").exists()
