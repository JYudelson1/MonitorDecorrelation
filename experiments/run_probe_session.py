"""The GPU session: every reader extracts every dataset on one VM, saving as it goes, resumable after a preemption.

    uv run python experiments/run_probe_session.py --plan experiments/session_plans/probe_session.json

**The plan (JSON):**

    {"name": "...", "out": "/mnt/ram/session",
     "readers": {"nvfp4": {"model": "/mnt/ram/Inkling-Small-NVFP4", "gpus": [[0], [1]], "device": "cuda:0",
                           "store": "hf://buckets/.../features/nvfp4", "required": true,
                           "full_activations": {"terminal_prerl": 10}},
                 "bf16":  {"model": "/mnt/ram/Inkling-Small", "gpus": [[2, 3, 4], [5, 6, 7]], "required": false,
                           "store": "hf://buckets/.../features/bf16", "stagger": true}},
     "datasets": ["data/probe_datasets/terminal_prerl", "data/probe_datasets/diverse_deception"],
     "shard_size": 256, "max_batch_tokens": 65536,
     "diagnostics": {"store": "hf://buckets/.../diagnostics", "n_iid": 32, "n_ood": 100},
     "status_store": "hf://buckets/.../sessions/<name>"}

**Workers.**
- Each GPU group of a reader gets one worker: this script with ``--worker``, pinned with CUDA_VISIBLE_DEVICES.
- A worker loads its reader once. It then extracts its part of every dataset, where the part is its index among
  the reader's GPU groups (whitebox/extraction.py).
- Then the first worker of each reader runs the diagnostics (whitebox/diagnostics.py).
- With ``"stagger": true``, a reader's next worker starts only once the previous one has loaded (or after 20
  minutes), so two full-precision copies don't load into RAM at once.
- **Parts across VMs:** ``"parts_total": n`` and ``"part_offset": i`` let several VMs share a reader. This VM's workers
  take parts i, i+1, … of n. The diagnostics run only where the offset is 0.

**Logs and exit code.** Worker output is merged into this process's log, each line prefixed with the worker. Exit
code:
- 1 if a worker of a required reader failed (a relaunch wouldn't fix it);
- else 75 if any worker stopped early with progress or on SIGTERM, so the launcher relaunches and everything resumes.
  A stop stays resumable when uploads were still running or failed then: shards missing from the store are extracted
  again on resume. A worker fails (1) if its store refuses uploads, or if uploads failed with nothing stopping it;
- else 76 if a worker stopped at its deadline before finishing a single shard: ``--hours`` leaves too little time
  after the setup, so a relaunch would only repeat that;
- else 0.

**Optional readers that fail.**
- A failed optional reader (e.g. full precision that won't load) is recorded in ``status.json`` in the status store.
- It is not retried on a relaunch unless ``--retry-failed`` is given.

**Code version.** ``status.json`` records each attempt's code version (``code``, provenance.code_version: on a VM, the
CODE_VERSION.json that scripts/nebius_job.py pushed), as do the extraction's index and run records and the
diagnostics' ``done.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

from monitordecorrelation.provenance import code_version  # noqa: E402
from monitordecorrelation.whitebox.extraction import (  # noqa: E402
    EXIT_NO_PROGRESS, EXIT_UNFINISHED, Job, StopFlag, extract, now, prepare, stop_verdict)

STAGGER_TIMEOUT = 1200.0


def log(msg: str) -> None:
    print(f"[session {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dataset_name(path: str) -> str:
    return os.path.basename(path.rstrip("/"))


def jobs_for(plan: dict, reader: str, copy: int) -> list[Job]:
    spec = plan["readers"][reader]
    jobs = []
    for ds in plan["datasets"]:
        name = dataset_name(ds)
        jobs.append(Job(dataset=os.path.join(REPO, ds) if not os.path.isabs(ds) else ds,
                        out=os.path.join(plan["out"], reader, name, f"part{copy}"), model=spec["model"],
                        store=f"{spec['store'].rstrip('/')}/{name}" if spec.get("store") else None,
                        shard_size=plan.get("shard_size", 256), max_batch_tokens=plan.get("max_batch_tokens", 65536),
                        full_activations=(spec.get("full_activations") or {}).get(name, 0),
                        part=spec.get("part_offset", 0) + copy, parts=spec.get("parts_total", len(spec["gpus"])),
                        limit=plan.get("limit")))
    return jobs


# ------------------------------------------------------------------------------------------------ worker


def load_reader(spec: dict):
    from monitordecorrelation.whitebox.inkling_nvfp4 import is_nvfp4_checkpoint, load_inkling_nvfp4

    if is_nvfp4_checkpoint(spec["model"]):
        return load_inkling_nvfp4(spec["model"], device=spec.get("device", "cuda:0")).eval()
    from monitordecorrelation.whitebox.model import load_inkling_text

    return load_inkling_text(spec["model"])  # across the visible GPUs; no tokenizer needed


def run_diagnostics(model, plan: dict, reader: str, wlog) -> None:
    """The reader's diagnostics on the fixed sample, saved under ``<diagnostics store>/<reader>/`` unless done."""
    from safetensors.torch import save_file

    from monitordecorrelation.whitebox import diagnostics
    from monitordecorrelation.whitebox.features import special_ids
    from monitordecorrelation.whitebox.inkling_replay import FOLLOW_UP, PROBE_EFFORT, InklingReplay
    from monitordecorrelation.whitebox.probe_datasets import item_messages, read_items
    from monitordecorrelation.whitebox.store import open_store

    cfg = plan["diagnostics"]
    store = open_store(f"{cfg['store'].rstrip('/')}/{reader}")
    if "done.json" in store.files():
        wlog("diagnostics already done")
        return
    by_kind: dict = {}  # the sample's sources: cfg["terminal"] / cfg["contrastive"], else the first dataset of each
    for kind in ("terminal", "contrastive"):  # kind in the plan (the concealment set is "terminal" too, and later)
        candidates = [cfg[kind]] if cfg.get(kind) else plan["datasets"]
        for ds in candidates:
            path = os.path.join(REPO, ds) if not os.path.isabs(ds) else ds
            if json.load(open(os.path.join(path, "manifest.json")))["kind"] == kind:
                by_kind[kind] = read_items(path)  # refused unless it is the file its manifest describes
                break
    sample = diagnostics.pick_sample(by_kind.get("terminal", []), by_kind.get("contrastive", []),
                                     cfg.get("n_iid", 64), cfg.get("n_ood", 100))
    replay = InklingReplay(effort=PROBE_EFFORT, follow_up=FOLLOW_UP)
    inputs = [replay.probe_input(item_messages(it, view)) for it, view in sample]
    wlog(f"diagnostics on {len(inputs)} inputs ({sum(len(p.tokens) for p in inputs)} tokens)")
    tensors, summary = diagnostics.run(model, inputs, special_ids(replay),
                                       max_batch_tokens=plan.get("max_batch_tokens", 65536))
    summary.update(reader=reader, model=plan["readers"][reader]["model"], finished_utc=now(), code=code_version(),
                   items=[{"id": it["id"], "view": view, "label": it["label"], "split": it["split"]}
                          for it, view in sample])
    out = os.path.join(plan["out"], "diagnostics", reader)
    os.makedirs(out, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, os.path.join(out, "diag.safetensors"))
    with open(os.path.join(out, "done.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    store.put([(os.path.join(out, "diag.safetensors"), "diag.safetensors")])
    store.put([(os.path.join(out, "done.json"), "done.json")])  # the marker, after the data
    wlog(f"diagnostics saved: {json.dumps({k: summary[k] for k in ('tokens_per_s_batched', 'jitter_rel_mean')})}")


def worker(plan: dict, reader: str, copy: int) -> int:
    stop = StopFlag().install()
    deadline = float(os.environ.get("NEBIUS_JOB_DEADLINE") or "inf")
    name = f"{reader}.{copy}"

    def wlog(msg: str) -> None:
        print(f"[{name} {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    spec = plan["readers"][reader]
    prepared = [prepare(job) for job in jobs_for(plan, reader, copy)]  # settings checked before the slow load
    want_diag = copy == 0 and spec.get("part_offset", 0) == 0 and plan.get("diagnostics")
    todo = sum(not p.done(v, k) for p in prepared for v, k in p.mine())
    if not todo and not want_diag:
        wlog("nothing to do")
        return 0
    t = time.time()
    model = load_reader(spec)
    wlog(f"loaded {spec['model']} in {time.time() - t:.0f}s")
    loaded = os.path.join(plan["out"], f"{name}.loaded")
    open(loaded, "w").write(now())
    extracted = 0
    for p in prepared:
        summary = extract(model, p, stop=stop, deadline=deadline, log=wlog)
        wlog(json.dumps(summary))
        extracted += summary["shards_extracted"]
        verdict = stop_verdict(summary, sigterm=stop.reason is not None)
        if verdict == "failed":  # the store refuses uploads, or they failed with nothing stopping the run
            return 1
        if verdict == "stopped":  # resumable, even with uploads in flight: unsaved shards are redone on resume
            return EXIT_UNFINISHED if stop.reason or extracted else EXIT_NO_PROGRESS
    if want_diag:
        if stop.reason or time.time() > deadline - 600:
            wlog("not enough time left for the diagnostics")
            return EXIT_UNFINISHED
        run_diagnostics(model, plan, reader, wlog)
    return 0


# ------------------------------------------------------------------------------------------------ driver


def pump(proc: subprocess.Popen, prefix: str) -> None:
    for line in proc.stdout:
        print(line if line.startswith("[") else f"[{prefix}] {line}", end="", flush=True)


def load_status(plan: dict) -> tuple:
    from monitordecorrelation.whitebox.store import open_store

    if not plan.get("status_store"):
        return None, {}
    store = open_store(plan["status_store"])
    if "status.json" in store.files():
        local = os.path.join(plan["out"], "status.previous.json")
        store.get("status.json", local)
        return store, json.load(open(local))
    return store, {}


def save_status(store, plan: dict, status: dict) -> None:
    path = os.path.join(plan["out"], "status.json")
    with open(path, "w") as fh:
        json.dump(status, fh, indent=1)
    if store is not None:
        store.put([(path, "status.json")])


def driver(plan: dict, plan_path: str, retry_failed: bool) -> int:
    os.makedirs(plan["out"], exist_ok=True)
    stopping = {"at": None}
    signal.signal(signal.SIGTERM, lambda s, f: stopping.update(at=time.time()))  # workers stop themselves
    store, status = load_status(plan)
    status.setdefault("attempts", []).append({"started_utc": now(), "host": os.uname().nodename,
                                              "code": code_version()})
    readers = status.setdefault("readers", {})
    procs: dict = {}
    order = []
    for reader, spec in plan["readers"].items():
        if readers.get(reader, {}).get("state") == "failed" and not spec.get("required") and not retry_failed:
            log(f"{reader}: failed in an earlier attempt ({readers[reader].get('why')}); not retried")
            continue
        order += [(reader, c, spec) for c in range(len(spec["gpus"]))]

    def start(reader: str, copy: int, spec: dict) -> None:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, spec["gpus"][copy])), "PYTHONUNBUFFERED": "1",
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
               "OMP_NUM_THREADS": str(plan.get("omp_threads", 16))}
        proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--plan", plan_path, "--worker",
                                 "--reader", reader, "--copy", str(copy)], env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, cwd=REPO)
        threading.Thread(target=pump, args=(proc, f"{reader}.{copy}"), daemon=True).start()
        procs[(reader, copy)] = proc
        log(f"started {reader}.{copy} on GPUs {spec['gpus'][copy]} (pid {proc.pid})")

    pending, started_at = list(order), {}
    while pending or any(p.poll() is None for p in procs.values()):
        for item in list(pending):
            reader, copy, spec = item
            prev = procs.get((reader, copy - 1))
            if copy > 0 and spec.get("stagger") and prev is not None and prev.poll() is None:
                loaded = os.path.exists(os.path.join(plan["out"], f"{reader}.{copy - 1}.loaded"))
                if not loaded and time.time() - started_at[(reader, copy - 1)] < STAGGER_TIMEOUT:
                    continue  # let the previous copy finish loading first
            if stopping["at"] is None:
                start(reader, copy, spec)
                started_at[(reader, copy)] = time.time()
            pending.remove(item)
        time.sleep(2)

    codes = {k: p.returncode for k, p in procs.items()}
    log(f"workers ended: {codes}")
    verdicts = set()
    for reader, spec in plan["readers"].items():
        mine = {c: code for (r, c), code in codes.items() if r == reader}
        if not mine:
            continue
        failed = {c: code for c, code in mine.items() if code not in (0, EXIT_UNFINISHED, EXIT_NO_PROGRESS)}
        if failed:
            readers[reader] = {"state": "failed", "codes": mine, "why": f"exit codes {mine}"}
            verdicts.add(1 if spec.get("required") else 0)
        elif EXIT_UNFINISHED in mine.values() or stopping["at"] is not None:
            readers[reader] = {"state": "unfinished", "codes": mine}
            verdicts.add(EXIT_UNFINISHED)
        elif EXIT_NO_PROGRESS in mine.values():
            readers[reader] = {"state": "no time", "codes": mine}
            verdicts.add(EXIT_NO_PROGRESS)
        else:
            readers[reader] = {"state": "done", "codes": mine}
    result = next((code for code in (1, EXIT_UNFINISHED, EXIT_NO_PROGRESS) if code in verdicts), 0)
    status["attempts"][-1].update(finished_utc=now(), result=result)
    save_status(store, plan, status)
    log(f"session result {result}: {json.dumps(readers)}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, help="the session plan (JSON)")
    ap.add_argument("--worker", action="store_true", help="internal: run one reader copy")
    ap.add_argument("--reader")
    ap.add_argument("--copy", type=int, default=0)
    ap.add_argument("--retry-failed", action="store_true", help="retry optional readers that failed before")
    args = ap.parse_args()
    plan = json.load(open(args.plan))
    if args.worker:
        return worker(plan, args.reader, args.copy)
    return driver(plan, args.plan, args.retry_failed)


if __name__ == "__main__":
    sys.exit(main())
