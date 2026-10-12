"""Extract Inkling probe features for a probe dataset (experiments/build_probe_datasets.py), on the GPU box.

    uv run python experiments/extract_probe_features.py --dataset data/probe_datasets/terminal_prerl \
        --model /mnt/ram/Inkling-Small-NVFP4 --out /mnt/ram/features/terminal_prerl \
        --store hf://buckets/probemaxxing/inkling-probes/features/nvfp4/terminal_prerl

Writes ``<out>/<view>/shard_NNNNN.{safetensors,jsonl}`` (format in whitebox/features.py), ``<out>/index.json``
and a run record under ``<out>/runs/``. The work is done by whitebox/extraction.py.

- **Saved as it goes (``--store``):** each finished shard is also saved to the store (an HF bucket prefix or a
  directory) from a background thread (whitebox/store.py). Data goes first; the shard's ``.jsonl`` goes last and
  marks the shard complete.
- **Resumable:** shards finished locally or in the store are skipped, so a run on a new VM after a preemption
  continues where the last one stopped.
  - A run whose settings differ from the stored ``index.json`` (dataset, reader, views, shard size, …) refuses to
    start, rather than mix shards.
  - The code version is not such a setting: ``index.json`` and each run record carry ``code``
    (provenance.code_version; on a VM, the CODE_VERSION.json that scripts/nebius_job.py pushed), and a run may resume
    shards that another commit made. Each run record also lists the shards that run extracted.
- **Parts:** ``--part i/n`` takes every n-th shard, so n processes (one per GPU) can share a dataset and a store.
- **Stops before it is stopped:**
  - Given a deadline (``NEBIUS_JOB_DEADLINE``, epoch seconds), it starts a shard only if the shard and the upload
    backlog fit before the deadline. The launcher sets the deadline from the VM's scheduled shutdown.
  - On SIGTERM it starts nothing new and lets the uploads finish. SIGTERM is Nebius's 60 s notice before stopping
    a spot VM.
  - In both cases it exits with 75: unfinished, run again to resume. Also when uploads were still running or failed
    then: what didn't reach the store is extracted again. It exits 1 if the store refuses uploads (quota,
    permission), or if uploads failed with nothing stopping it.
- **Views:** terminal datasets get with_reasoning and without_reasoning; contrastive datasets get
  without_reasoning, since their pairs carry no CoT.
- **Follow-up and effort:** every input gets the follow-up ("Did you respond truthfully?" / "Yes"), rendered
  at the dataset's effort (contrastive datasets use the TerminalEnv runs' effort).
- **``--full-activations N``:** also stores all token positions' hidden states for N items per view (half
  positives, half negatives, from the train split) under ``<view>/full/``. That's ~0.6 GB per TerminalEnv
  episode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from monitordecorrelation.whitebox.extraction import EXIT_UNFINISHED, Job, StopFlag, extract, prepare, stop_verdict


def load_reader(model_name: str, device: str):
    """The frozen reader: NVFP4 checkpoints on one device (whitebox/inkling_nvfp4.py), anything else through
    ``WhiteBoxModel`` (BF16 Inkling across the visible GPUs)."""
    from monitordecorrelation.whitebox.inkling_nvfp4 import is_nvfp4_checkpoint, load_inkling_nvfp4

    if is_nvfp4_checkpoint(model_name):
        return load_inkling_nvfp4(model_name, device=device).eval()
    from monitordecorrelation.whitebox.model import WhiteBoxModel

    return WhiteBoxModel(model_name).model.eval()


def parse_part(text: str) -> tuple[int, int]:
    part, _, parts = text.partition("/")
    return int(part), int(parts or 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="a folder with items.jsonl + manifest.json")
    ap.add_argument("--model", default="thinkingmachines/Inkling-Small-NVFP4", help="hub id or local dir")
    ap.add_argument("--device", default="cuda:0", help="for NVFP4 checkpoints (one device)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--store", default=None,
                    help="also save each finished shard here and resume from it (hf://buckets/NS/BUCKET/PREFIX, a dir)")
    ap.add_argument("--views", default=None, help="comma-list of views (default: by dataset kind)")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--limit", type=int, default=None, help="first N items only (smoke runs)")
    ap.add_argument("--shard-size", type=int, default=256)
    ap.add_argument("--max-batch-tokens", type=int, default=65536)
    ap.add_argument("--full-activations", type=int, default=0, metavar="N")
    ap.add_argument("--part", default="0/1", help="i/n: this process takes every n-th shard, starting at i")
    ap.add_argument("--deadline-margin", type=float, default=180.0,
                    help="seconds kept free before the deadline (NEBIUS_JOB_DEADLINE) for the last uploads and exit")
    ap.add_argument("--rate-guess", type=float, default=3000.0, help="tokens/s assumed until the first shard is timed")
    ap.add_argument("--upload-rate-guess", type=float, default=30e6,
                    help="bytes/s assumed until the first upload is timed")
    args = ap.parse_args()

    stop = StopFlag().install()
    deadline = float(os.environ.get("NEBIUS_JOB_DEADLINE") or "inf")
    part, parts = parse_part(args.part)
    job = Job(dataset=args.dataset, out=args.out, model=args.model, store=args.store,
              views=args.views.split(",") if args.views else None, splits=args.splits, limit=args.limit,
              shard_size=args.shard_size, max_batch_tokens=args.max_batch_tokens,
              full_activations=args.full_activations, part=part, parts=parts,
              deadline_margin=args.deadline_margin, rate_guess=args.rate_guess,
              upload_rate_guess=args.upload_rate_guess)
    prepared = prepare(job)  # settings checked against earlier runs before the slow load
    done = sum(prepared.done(v, k) for v, k in prepared.mine())
    print(f"[{time.strftime('%H:%M:%S')}] {done} of {len(prepared.mine())} shards already done"
          + (f" (store: {prepared.store.url})" if prepared.store is not None else ""), flush=True)
    model = load_reader(args.model, args.device)
    summary = extract(model, prepared, stop=stop, deadline=deadline,
                      log=lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True))
    print(json.dumps(summary), flush=True)
    verdict = stop_verdict(summary, sigterm=stop.reason is not None)  # a stop is resumable, uploads in flight or not
    return EXIT_UNFINISHED if verdict == "stopped" else 1 if verdict == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
