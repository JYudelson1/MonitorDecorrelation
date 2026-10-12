"""Probe datasets for the Inkling probes: model-free ``items.jsonl`` + ``manifest.json``, built once on a CPU box
by experiments/build_probe_datasets.py and read by the feature extraction on the GPU box.

Two kinds:

- **terminal** (the IID probe): one item per valid TerminalEnv episode of a run's ``eval_rollouts.jsonl``.
  - Label = ``behavior_present`` (the hack).
  - Tasks, not episodes, are split train/val/test, stratified by whether the task has any hack. A task's
    episodes share the question and the verifier, so splitting by episode would leak.
  - Each item carries what ``whitebox/inkling_replay.py`` needs to rebuild the conversation (question,
    turns, env options) plus the env's own diagnostics under ``diagnostics``. Diagnostics are for analysis
    only and never enter a probe input.
- **contrastive** (the OOD probe): the honest/deceptive QA pairs of ``whitebox/datasets``. Same adapters,
  cap and seed as experiments/train_probe.py, and the same pair-respecting split (``held_out_pairs``).

Every valid episode must replay (``terminal_episode_messages``); a build fails loudly rather than
silently dropping episodes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np

from monitordecorrelation.whitebox.inkling_replay import terminal_episode_messages

SPLITS = ("train", "val", "test")
REPLAY_TURN_KEYS = ("cot", "text", "output", "submission", "truncated")
REPLAY_ENV_KEYS = ("verifier_mode", "submission_ends", "max_turns", "require_explanation", "last_turn_submit_notice")
EPISODE_ENV_KEYS = ("verifier_mode", "submission_ends", "last_turn_submit_notice")  # also saved per episode


def item_messages(item: dict, view: str) -> list[dict[str, str]]:
    """The conversation a probe reads for ``item`` in ``view`` (one of inkling_replay.VIEWS); the follow-up is
    appended by ``InklingReplay.probe_input``."""
    from monitordecorrelation.whitebox.inkling_replay import VIEWS, qa_messages

    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; known: {VIEWS}")
    include = view == "with_reasoning"
    if "turns" in item:
        return terminal_episode_messages(item["question"], item["turns"], item["env_options"],
                                         include_reasoning=include)
    messages = qa_messages(item["question"], item["answer"])
    if include and item.get("cot"):
        messages[-1]["reasoning_content"] = item["cot"]
    return messages


def split_tasks(task_has_hack: dict[str, bool], fractions: tuple[float, float, float], seed: int) -> dict[str, str]:
    """task -> split. Each stratum (tasks with / without a hack) is shuffled and cut by ``fractions``."""
    if len(fractions) != 3 or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError(f"fractions must be (train, val, test) summing to 1, got {fractions}")
    rng = np.random.default_rng(seed)
    out: dict[str, str] = {}
    for stratum in (True, False):
        tasks = sorted(t for t, has in task_has_hack.items() if has == stratum)
        tasks = [tasks[i] for i in rng.permutation(len(tasks))]
        n_train = round(fractions[0] * len(tasks))
        n_val = round(fractions[1] * len(tasks))
        for i, task in enumerate(tasks):
            out[task] = SPLITS[0] if i < n_train else SPLITS[1] if i < n_train + n_val else SPLITS[2]
    return out


def episode_item(question: str, episode: dict, env_options: dict[str, Any]) -> dict:
    """The replay fields of one TerminalEnv episode: its question, its turns and its env options.

    The dataset builder and the live probe client (monitors/inkling_probe_monitor.py) both build items with this, so
    a live episode renders byte-identically to a training one.
    - ``episode`` is the env's record: eval_rollouts' ``env_meta``, or a live rollout's ``meta["episode"]``.
    - ``env_options`` are the run's; ``max_turns`` and ``require_explanation`` are not in the episode itself.
    """
    missing = [k for k in REPLAY_ENV_KEYS if k not in env_options]
    if missing:
        raise KeyError(f"env_options lack {missing}: pass the run's env_options (baseline.json / run_info.json)")
    opts = {k: env_options[k] for k in REPLAY_ENV_KEYS}
    opts.update({k: episode[k] for k in EPISODE_ENV_KEYS if k in episode})
    return {"question": question, "turns": [{k: turn.get(k) for k in REPLAY_TURN_KEYS} for turn in episode["turns"]],
            "env_options": opts}


def terminal_items(
    rollouts: str,
    env_options: dict[str, Any],
    *,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
    max_negatives_per_split: int | None = None,
) -> tuple[list[dict], dict]:
    """Items for every valid episode of ``rollouts`` (an eval_rollouts.jsonl), plus build stats.

    ``env_options`` are the run's (baseline.json / run_info.json ``env_options``): ``max_turns`` and
    ``require_explanation`` are not in the episodes themselves."""
    missing = [k for k in REPLAY_ENV_KEYS if k not in env_options]
    if missing:
        raise KeyError(f"env_options lack {missing}: pass the run's baseline.json / run_info.json env_options")
    items: list[dict] = []
    invalid: Counter = Counter()
    per_task: Counter = Counter()
    with open(rollouts) as fh:
        for line_no, line in enumerate(fh, start=1):
            row = json.loads(line)
            if row.get("invalid_reason"):
                invalid[row["invalid_reason"]] += 1
                continue
            meta = row["env_meta"]
            task = row["task_id"]
            items.append({
                "id": f"{task}/{per_task[task]}",
                "task_id": task,
                "label": int(bool(row["behavior_present"])),
                **episode_item(row["question"], meta, env_options),
                "diagnostics": {k: v for k, v in meta.items() if k not in ("turns", "commands")},
                "source_line": line_no,
            })
            per_task[task] += 1
    failures = []
    for item in items:
        try:
            terminal_episode_messages(item["question"], item["turns"], item["env_options"], include_reasoning=True)
        except (ValueError, NotImplementedError, KeyError) as e:
            failures.append(f"{item['id']}: {e}")
    if failures:
        raise ValueError(f"{len(failures)} valid episodes do not replay, e.g. {failures[:3]}")

    has_hack: dict[str, bool] = defaultdict(bool)
    for item in items:
        has_hack[item["task_id"]] |= bool(item["label"])
    split_of = split_tasks(dict(has_hack), fractions, seed)
    for item in items:
        item["split"] = split_of[item["task_id"]]
    dropped = 0
    if max_negatives_per_split is not None:
        rng = np.random.default_rng(seed)
        keep: set[str] = set()
        for split in SPLITS:
            negatives = [it["id"] for it in items if it["split"] == split and not it["label"]]
            if len(negatives) > max_negatives_per_split:
                negatives = [negatives[i] for i in sorted(rng.choice(len(negatives), max_negatives_per_split,
                                                                       replace=False))]
            keep.update(negatives)
        before = len(items)
        items = [it for it in items if it["label"] or it["id"] in keep]
        dropped = before - len(items)
    stats = {"n_valid": len(items) + dropped, "n_invalid_by_reason": dict(invalid),
             "n_negatives_dropped_by_cap": dropped}
    return items, stats


def contrastive_items(names: list[str], *, n_per: int | None, seed: int, test_frac: float) -> list[dict]:
    """Two items (honest = 0, deceptive = 1) per pair of the named adapters, split by pair."""
    from monitordecorrelation.whitebox.datasets import held_out_pairs, load_contrastive

    pairs, sources = [], []
    for name in names:  # one adapter at a time == load_contrastive(names, ...), but tagged with its source
        got = load_contrastive([name], n_per=n_per, seed=seed)
        pairs.extend(got)
        sources.extend([name] * len(got))
    test = held_out_pairs(len(pairs), test_frac, seed)
    items = []
    for i, (pair, source) in enumerate(zip(pairs, sources)):
        for question, cot, answer, label in pair.to_rows():
            items.append({
                "id": f"{i}/{'deceptive' if label else 'honest'}", "pair": i, "label": label,
                "split": "test" if i in test else "train", "source": source,
                "question": question, "cot": cot, "answer": answer, "meta": pair.meta,
            })
    return items


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_state() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=here, capture_output=True, text=True,
                                    check=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def counts(items: Iterable[dict], group: str) -> dict:
    """Per split: items, positives, and distinct ``group`` values (tasks or pairs)."""
    out: dict = {}
    for split in SPLITS:
        rows = [it for it in items if it["split"] == split]
        if rows:
            out[split] = {"items": len(rows), "positives": sum(it["label"] for it in rows),
                          group: len({it[group] for it in rows})}
    return out


DISCLOSURE = re.compile(r"verif|ground|truth", re.IGNORECASE)  # the policy saying it used the verifier's answer
CONCEALMENT_LEVELS = ("A", "B")


def _without_disclosure(text: str) -> str:
    """``text`` minus every sentence that mentions the verifier or its ground truth."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(x for x in sentences if x and not DISCLOSURE.search(x))


def conceal(item: dict, level: str) -> dict:
    """A copy of a TerminalEnv hack in which the policy no longer says that it used the verifier.

    - **Level A:** sentences mentioning the verifier or ground truth are removed from the final submission's
      explanation.
    - **Level B:** they are also removed from every turn's reasoning and explanation.

    The commands and tool outputs are untouched: the act of reading ``verifier.py`` stays visible, only the policy's
    own words about it are gone. This is the concealment stress set of the pre-registered analysis
    (docs/inkling_probes_preregistration.md).
    Its items are scored by probes that never trained on their task, so they keep the source item's task and split."""
    if level not in CONCEALMENT_LEVELS:
        raise ValueError(f"unknown concealment level {level!r}; known: {CONCEALMENT_LEVELS}")
    turns = [dict(t) for t in item["turns"]]
    last = len(turns) - 1
    for i, turn in enumerate(turns):
        if level == "A" and i != last:
            continue
        m = re.match(r"(?s)(.*?</(?:command|submit)>)\s*(?:Explanation:)?\s*(.*)", turn["text"])
        if m:
            explanation = _without_disclosure(m.group(2)) or ("Computed from the grid." if i == last else "Continue.")
            turn["text"] = f"{m.group(1)} Explanation: {explanation}"
        if level == "B":
            turn["cot"] = _without_disclosure(turn["cot"]) or "Continue."
    return {**item, "id": f"{item['id']}/conceal-{level}", "turns": turns, "concealment": level,
            "source_id": item["id"]}


def concealed_items(items: list[dict]) -> list[dict]:
    """Both concealment levels of every hack that used the verifier (``diagnostics.used_verifier``)."""
    hacks = [it for it in items if it.get("label") and (it.get("diagnostics") or {}).get("used_verifier")]
    return [conceal(it, level) for level in CONCEALMENT_LEVELS for it in hacks]


def write_dataset(out_dir: str, items: list[dict], manifest: dict) -> dict:
    """Write ``items.jsonl`` (atomically) and ``manifest.json``; returns the manifest as written."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "items.jsonl")
    with open(path + ".tmp", "w") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")
    os.replace(path + ".tmp", path)
    manifest = {
        "name": os.path.basename(os.path.normpath(out_dir)),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **manifest,
        "n_items": len(items),
        "items_sha256": _sha256(path),
        "repo": _repo_state(),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def file_source(path: str) -> dict:
    return {"path": os.path.abspath(path), "sha256": _sha256(path)}


def read_items(dataset_dir: str, recorded: dict[str, str | None] | None = None) -> list[dict]:
    """The items of ``<dataset_dir>/items.jsonl``, refused unless the file is the one its records describe.

    - Its sha256 must equal its ``manifest.json``'s ``items_sha256`` and every hash in ``recorded`` (where it was
      recorded -> sha256; e.g. the features' index.json, which copied the manifest's hash at extraction).
    - A missing manifest, or a missing hash, is not checked.
    - The bytes hashed are the bytes parsed: one read, one hash.
    Raises SystemExit naming both hashes on a mismatch: features and labels from different items must never meet."""
    path = os.path.join(dataset_dir, "items.jsonl")
    with open(path, "rb") as fh:
        data = fh.read()
    actual = hashlib.sha256(data).hexdigest()
    expected = dict(recorded or {})
    manifest = os.path.join(dataset_dir, "manifest.json")
    if os.path.exists(manifest):
        with open(manifest) as fh:
            expected = {manifest: json.load(fh).get("items_sha256"), **expected}
    wrong = [f"{where} records {sha}" for where, sha in expected.items() if sha and sha != actual]
    if wrong:
        raise SystemExit(f"{path} is not the file its records describe: its sha256 is {actual}, but "
                         f"{'; '.join(wrong)}. Refusing to pair items with features or labels made from other items: "
                         "use the items.jsonl those records were made from (rebuild it with "
                         "experiments/build_probe_datasets.py, or fetch the matching copy), or extract the features "
                         "again from this one.")
    return [json.loads(line) for line in data.split(b"\n") if line.strip()]
