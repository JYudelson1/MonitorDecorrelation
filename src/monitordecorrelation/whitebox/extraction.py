"""Probe-feature extraction for one dataset with a reader that is already loaded.

``experiments/extract_probe_features.py`` is the command line for one dataset; ``experiments/run_probe_session.py``
loads a reader once and runs several datasets and parts with it.

- **Shards:** ``<out>/<view>/shard_NNNNN.{safetensors,jsonl}``, in the format of whitebox/features.py.
- **Saved as it goes (``store``):** each finished shard also goes to the store (an HF bucket prefix or a directory)
  from a background thread (whitebox/store.py). Data goes first; the shard's ``.jsonl`` goes last and marks it
  complete.
- **Resumable:** shards that are complete locally or in the store are skipped, so a run on a new VM continues where
  the last one stopped.
  - ``index.json`` fixes the settings that decide what each shard holds: dataset, reader, views, shard size and so
    on. A run with other settings refuses to start rather than mix shards.
  - Never change the shard size between runs.
- **Parts:** ``part``/``parts`` split the shards between processes that share the store. A process takes the
  shards whose position in the (view, shard) list is ``part`` modulo ``parts``. Each process writes its own run
  record under ``runs/``.
- **Stops before it is stopped:**
  - It starts a shard only if the shard and the upload backlog fit before ``deadline`` (epoch seconds; the
    launcher's ``NEBIUS_JOB_DEADLINE``).
  - After SIGTERM (Nebius's 60 s notice before stopping a spot VM) it starts nothing new.
  - Either way the run reports ``stopped_early``; the caller exits 75 (``EXIT_UNFINISHED``).
"""

from __future__ import annotations

import json
import os
import signal
import socket
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

EXIT_UNFINISHED = 75  # stopped early (deadline or preemption notice); finished shards are saved: run again to resume
EXIT_NO_PROGRESS = 76  # stopped at the deadline before finishing a single shard: a rerun with the same --hours won't help
# Settings that fix which items land in which shard and what a shard holds: a resumed run must match them.
RESUME_KEYS = ("dataset", "model_name", "effort", "follow_up", "views", "splits", "limit", "n_items", "shard_size",
               "full_activations", "feature_format", "versions")
MODEL_KEYS = ("n_layers", "d_model", "reader", "max_batch_tokens")
RECORD_KEYS = ("id", "label", "split", "task_id", "pair", "source")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class StopFlag:
    """Set by SIGTERM: Nebius sends it 60 s before stopping a spot VM (then SIGKILL)."""

    def __init__(self) -> None:
        self.reason: str | None = None
        self.at: float | None = None

    def install(self) -> "StopFlag":
        signal.signal(signal.SIGTERM, self._handler)
        return self

    def _handler(self, signum, frame) -> None:
        self.reason, self.at = "SIGTERM: the VM is being stopped (spot preemption notice)", time.time()


def deadline_reason(time_left: float, shard_seconds: float, backlog_seconds: float, margin: float) -> str | None:
    """Why not to start the next shard: it and the upload backlog would not finish ``margin`` s before the deadline."""
    need = 1.25 * shard_seconds + backlog_seconds + margin
    if time_left >= need:
        return None
    return (f"deadline: {time_left:.0f}s left, but the next shard (~{shard_seconds:.0f}s) and the upload backlog "
            f"(~{backlog_seconds:.0f}s) need ~{need:.0f}s including the {margin:.0f}s margin")


def check_resume(now_settings: dict, previous: dict, where: str, keys) -> None:
    """Refuse to add shards to ``where`` if they would differ from the ones already there."""
    diff = [f"{k}: {previous.get(k)!r} there, {now_settings.get(k)!r} now" for k in keys  # missing = different:
            if previous.get(k) != now_settings.get(k)]  # an older format lacks keys, and must not pass
    if diff:
        raise SystemExit(f"{where} holds shards made with other settings; refusing to mix them ({'; '.join(diff)})")


def write_json(path: str, data: dict) -> None:
    with open(path + ".tmp", "w") as fh:
        json.dump(data, fh, indent=1)
    os.replace(path + ".tmp", path)


def pick_full(items: list[dict], n: int) -> set[str]:
    """``n`` train items for whole-sequence activations: half positives, half negatives, in dataset order."""
    train = [it for it in items if it["split"] == "train"]
    pos = [it["id"] for it in train if it["label"]][: n // 2]
    neg = [it["id"] for it in train if not it["label"]][: n - len(pos)]
    return set(pos + neg)


@dataclass
class Job:
    """One dataset for one reader (or one part of it)."""
    dataset: str  # a folder with items.jsonl + manifest.json
    out: str  # local folder (the VM's RAM disk)
    model: str  # the reader's checkpoint path or hub id, for the record and the resume check
    store: str | None = None
    views: list[str] | None = None  # default: by dataset kind
    splits: str = "train,val,test"
    limit: int | None = None
    shard_size: int = 256
    max_batch_tokens: int = 65536
    full_activations: int = 0
    part: int = 0
    parts: int = 1
    deadline_margin: float = 180.0
    rate_guess: float = 3000.0  # tokens/s until the first shard is timed
    upload_rate_guess: float = 30e6  # bytes/s until the first upload is timed


@dataclass
class Prepared:
    job: Job
    items: list[dict]
    views: list[str]
    effort: float
    full_ids: set[str]
    n_shards: int  # per view
    settings: dict
    store: object = None
    stored: dict = field(default_factory=dict)
    previous: list = field(default_factory=list)  # (index dict, where)

    def mine(self) -> list[tuple[str, int]]:
        """This part's (view, shard) pairs, in order."""
        all_shards = [(v, k) for v in self.views for k in range(self.n_shards)]
        return [s for g, s in enumerate(all_shards) if g % self.job.parts == self.job.part]

    def local(self, view: str, k: int) -> bool:
        return os.path.exists(os.path.join(self.job.out, f"{view}/shard_{k:05d}.jsonl"))

    def saved(self, view: str, k: int) -> bool:
        """In the store (marker and data; the marker goes up last), or locally when there is no store."""
        if self.store is None:
            return self.local(view, k)
        rel = f"{view}/shard_{k:05d}"
        return rel + ".jsonl" in self.stored and rel + ".safetensors" in self.stored

    def done(self, view: str, k: int) -> bool:
        """Needs no extraction: saved, or finished here and waiting for its upload."""
        return self.local(view, k) or self.saved(view, k)


def prepare(job: Job) -> Prepared:
    """Read the dataset and check what earlier runs left (locally and in the store) before any model loads, so a
    settings mismatch fails in seconds, not after a 5-minute load."""
    from monitordecorrelation.whitebox.features import FEATURE_FORMAT
    from monitordecorrelation.whitebox.inkling_replay import FOLLOW_UP, PROBE_EFFORT, VIEWS
    from monitordecorrelation.whitebox.store import open_store

    if not 0 <= job.part < job.parts:
        raise SystemExit(f"part {job.part} of {job.parts}: want 0 <= part < parts")
    manifest = json.load(open(os.path.join(job.dataset, "manifest.json")))
    splits = set(job.splits.split(","))
    with open(os.path.join(job.dataset, "items.jsonl")) as fh:
        items = [it for it in map(json.loads, fh) if it["split"] in splits]
    items = items[: job.limit] if job.limit else items
    views = list(job.views) if job.views else (list(VIEWS) if manifest["kind"] == "terminal" else ["without_reasoning"])
    effort = float(PROBE_EFFORT if manifest.get("effort") is None else manifest["effort"])
    settings = json.loads(json.dumps({
        "dataset": {k: manifest.get(k) for k in ("name", "kind", "items_sha256", "n_items")},
        "model": job.model, "model_name": os.path.basename(job.model.rstrip("/")), "effort": effort,
        "follow_up": list(FOLLOW_UP), "views": views, "splits": sorted(splits), "limit": job.limit,
        "n_items": len(items), "shard_size": job.shard_size, "full_activations": job.full_activations,
        "feature_format": FEATURE_FORMAT, "versions": library_versions(),
    }))
    p = Prepared(job, items, views, effort, pick_full(items, job.full_activations) if job.full_activations else set(),
                 (len(items) + job.shard_size - 1) // job.shard_size, settings)
    os.makedirs(job.out, exist_ok=True)
    index_path = os.path.join(job.out, "index.json")
    if os.path.exists(index_path):
        p.previous.append((json.load(open(index_path)), index_path))
    if job.store:
        p.store = open_store(job.store)
        p.stored = p.store.files()
        if "index.json" in p.stored:
            p.store.get("index.json", index_path + ".stored")
            p.previous.append((json.load(open(index_path + ".stored")), f"{p.store.url}/index.json"))
    for prev, where in p.previous:
        check_resume(settings, prev, where, RESUME_KEYS)
    return p


def complete(p: Prepared) -> bool:
    """Every shard of every part is saved: in the store if there is one, else locally."""
    if p.store is not None:
        p.stored = p.store.files()
    return all(p.saved(v, k) for v in p.views for k in range(p.n_shards))


def library_versions() -> dict:
    """Versions that can change the numbers; a relaunch from the same lockfile has the same ones."""
    import torch
    import transformers

    return {"torch": torch.__version__, "transformers": transformers.__version__}


def extract(model, p: Prepared, *, stop: StopFlag | None = None, deadline: float = float("inf"), log=print) -> dict:
    """Extract this part's missing shards with ``model``. Returns the run summary; ``stopped_early`` says why it
    stopped before finishing (deadline or SIGTERM), and ``unsaved`` lists what didn't reach the store."""
    import torch
    from safetensors.torch import save_file

    from monitordecorrelation.whitebox.features import extract_features, special_ids, write_shard
    from monitordecorrelation.whitebox.inkling_replay import FOLLOW_UP, InklingReplay
    from monitordecorrelation.whitebox.probe_datasets import item_messages
    from monitordecorrelation.whitebox.store import Uploader

    job, stop = p.job, stop or StopFlag()
    cfg = getattr(model.config, "text_config", None) or model.config
    index = {**p.settings, "n_layers": cfg.num_hidden_layers + 1, "d_model": cfg.hidden_size,
             "shards": {v: p.n_shards for v in p.views}, "reader": reader_info(model, job.model),
             "max_batch_tokens": job.max_batch_tokens}
    for prev, where in p.previous:
        check_resume(index, prev, where, MODEL_KEYS)
    index_path = os.path.join(job.out, "index.json")
    write_json(index_path, index)
    started = now()
    run = {"started_utc": started, "host": socket.gethostname(), "part": job.part, "parts": job.parts,
           "deadline_utc": None if deadline == float("inf") else
           datetime.fromtimestamp(deadline, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "shards_extracted": 0, "tokens": 0, "seconds": None, "finished_utc": None, "stopped_early": None}
    run_rel = f"runs/{started.replace(':', '')}-{socket.gethostname()}-part{job.part}of{job.parts}.json"
    run_path = os.path.join(job.out, run_rel)
    os.makedirs(os.path.dirname(run_path), exist_ok=True)
    write_json(run_path, run)
    uploader = Uploader(p.store, log=log) if p.store is not None else None
    if uploader:
        uploader.submit("index.json", [(index_path, "index.json"), (run_path, run_rel)])

    if uploader:  # finished here earlier but never saved (an upload failed): send again, data before marker
        for view, k in p.mine():
            rel = f"{view}/shard_{k:05d}"
            if p.local(view, k) and not p.saved(view, k):
                prefix = os.path.join(job.out, rel)
                with open(prefix + ".jsonl") as fh:
                    names = [json.loads(line)["id"].replace("/", "__") + ".safetensors" for line in fh]
                fulls = [(os.path.join(job.out, view, "full", n), f"{view}/full/{n}") for n in names
                         if os.path.exists(os.path.join(job.out, view, "full", n))]
                uploader.submit(rel, [(prefix + ".safetensors", rel + ".safetensors"), *fulls],
                                marker=(prefix + ".jsonl", rel + ".jsonl"))
                log(f"{rel}: finished earlier but not in the store; uploading again")
    replay = InklingReplay(effort=p.effort, follow_up=FOLLOW_UP)
    sp = special_ids(replay)
    answers = answer_ids(replay)
    todo = [(v, k) for v, k in p.mine() if not p.done(v, k)]
    log(f"part {job.part}/{job.parts} of {os.path.basename(job.dataset)}: {len(todo)} of {len(p.mine())} shards to do")
    t0, extract_seconds, stopped = time.time(), 0.0, None
    for view in p.views:
        os.makedirs(os.path.join(job.out, view, "full"), exist_ok=True)
    for view, k in todo:
        rel = f"{view}/shard_{k:05d}"
        prefix = os.path.join(job.out, rel)
        chunk = p.items[k * job.shard_size: (k + 1) * job.shard_size]
        inputs = [replay.probe_input(item_messages(it, view)) for it in chunk]
        n_tok = sum(len(x.tokens) for x in inputs)
        rate = run["tokens"] / extract_seconds if extract_seconds > 0 else job.rate_guess
        backlog = uploader.pending_bytes() / (uploader.rate() or job.upload_rate_guess) if uploader else 0.0
        if uploader and uploader.broken:  # nothing more can be saved: a failure, not something to resume
            log(f"stopping before {rel}: the store refuses uploads ({uploader.broken})")
            break
        stopped = stop.reason or deadline_reason(deadline - time.time(), n_tok / rate, backlog, job.deadline_margin)
        if stopped:
            log(f"stopping before {rel}: {stopped}")
            break
        full = [i for i, it in enumerate(chunk) if it["id"] in p.full_ids]
        t = time.time()
        features = extract_features(model, inputs, sp, max_batch_tokens=job.max_batch_tokens, full=full,
                                    answers=answers)
        full_files = []  # written before the shard, whose .jsonl marks everything of it complete
        for i in full:
            name = chunk[i]["id"].replace("/", "__")
            path = os.path.join(job.out, view, "full", f"{name}.safetensors")
            save_file({"hidden": features["full"][i], "tokens": torch.tensor(inputs[i].tokens, dtype=torch.int32)},
                      path + ".tmp")
            os.replace(path + ".tmp", path)
            full_files.append((path, f"{view}/full/{name}.safetensors"))
        records = [{**{key: it[key] for key in RECORD_KEYS if key in it}, "view": view,
                    "n_tokens": len(x.tokens)} for it, x in zip(chunk, inputs)]
        if view != "with_reasoning":  # no reasoning in this view: the mean over it is all zeros (counts say 0)
            features.pop("mean_reasoning", None)
        write_shard(prefix, features, inputs, records)
        if uploader:
            uploader.submit(rel, [(prefix + ".safetensors", rel + ".safetensors"), *full_files],
                            marker=(prefix + ".jsonl", rel + ".jsonl"))
        dt = time.time() - t
        extract_seconds += dt
        run["tokens"] += n_tok
        run["shards_extracted"] += 1
        log(f"{view} shard {k + 1}/{p.n_shards}: {len(chunk)} items, {n_tok} tokens, {dt:.1f}s "
            f"({n_tok / max(dt, 1e-9):.0f} tok/s)")

    run.update(seconds=round(time.time() - t0, 1), finished_utc=now(), stopped_early=stopped)
    write_json(run_path, run)
    unsaved = []
    if uploader:
        uploader.submit("run record", [(run_path, run_rel)])
        timeout = None if deadline == float("inf") else max(5.0, deadline - time.time() - 15)
        if stop.at is not None:  # SIGKILL comes 60 s after SIGTERM
            timeout = max(1.0, min(timeout or float("inf"), stop.at + 55 - time.time()))
        if not uploader.close(timeout):
            unsaved.append("uploads still running when time ran out")
        unsaved += [f"{name} failed" for name in uploader.failed]
        if unsaved:
            log(f"NOT all saved to {p.store.url}: {'; '.join(unsaved)}. The files are still in {job.out}.")
    if uploader and uploader.broken:
        unsaved.append(f"the store refuses uploads: {uploader.broken}")
    return {"dataset": os.path.basename(job.dataset), "part": job.part, "parts": job.parts, "n_items": len(p.items),
            "views": p.views, "shards": index["shards"], "shards_extracted": run["shards_extracted"],
            "tokens_this_run": run["tokens"], "seconds_this_run": run["seconds"],
            "complete": stopped is None and not unsaved and complete(p), "stopped_early": stopped, "unsaved": unsaved}


def answer_ids(replay) -> tuple[int, int] | None:
    """Token ids of "Yes" and "No" as the follow-up answer, if each is a single token (else no self-report)."""
    yes = list(replay.answer_tokens)
    no = list(replay.renderer.tokenizer.encode_ordinary("No"))
    return (yes[0], no[0]) if len(yes) == 1 and len(no) == 1 else None


def reader_info(model, model_path: str) -> dict:
    """Which reader made the features: loader, dtype, and the checkpoint's revision when the HF download recorded
    it (``.cache/huggingface/download/*.metadata`` next to the weights)."""
    from monitordecorrelation.whitebox.inkling_nvfp4 import is_nvfp4_checkpoint

    revision = None
    meta_dir = os.path.join(model_path, ".cache", "huggingface", "download")
    if os.path.isdir(meta_dir):
        for name in sorted(os.listdir(meta_dir)):
            if name.endswith(".metadata"):
                with open(os.path.join(meta_dir, name)) as fh:
                    revision = fh.readline().strip() or None  # first line: the commit hash
                break
    return {"path": model_path, "loader": "nvfp4" if os.path.isdir(model_path) and is_nvfp4_checkpoint(model_path)
            else "transformers", "dtype": str(next(model.parameters()).dtype), "revision": revision}


def job_dict(job: Job) -> dict:
    return asdict(job)
