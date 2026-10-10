#!/usr/bin/env python3
"""Download a reader's features from their store to hub, optionally keeping only some tensors of each shard.

    uv run python scripts/fetch_features.py hf://buckets/probemaximizer/inkling-probes-bulk/features/bf16 \
        ../scratch/features/bf16 --keep followup,counts,answer_logp

**Why slim:** hub's disk has room for one reader's full features, not two. Comparing the readers needs only the
follow-up readout (``compare_readers.py``), which is about a third of each shard.

**How:**
- Only complete units are fetched: a marker ``<dataset>/<view>/shard_N.jsonl`` together with its data file.
- One shard at a time per worker: it downloads the data next to its destination, writes the slim copy, deletes the
  download, and writes the marker last. So an interrupted fetch leaves no marker without its data, and a rerun
  skips the shards it already has.
- Small files (``index.json``, ``runs/*.json``) are copied as they are.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from monitordecorrelation.whitebox.store import Store, open_store

SMALL_FILE_BYTES = 10 * 2**20


def slim(src: str, dest: str, keep: list[str]) -> list[str]:
    """Copy the ``keep`` tensors of the safetensors file ``src`` (with its metadata) to ``dest``; returns the
    requested tensors it did not have."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(src, framework="pt") as f:
        have = set(f.keys())
        tensors = {k: f.get_tensor(k) for k in keep if k in have}
        meta = f.metadata()
    save_file(tensors, dest, metadata=meta)
    return [k for k in keep if k not in have]


def units(files: dict[str, int], datasets: list[str] | None) -> tuple[list[str], list[str]]:
    """(the markers of complete shards, the small other files) in the store's listing."""
    markers, small = [], []
    for rel, size in sorted(files.items()):
        if datasets and rel.split("/", 1)[0] not in datasets:
            continue
        name = os.path.basename(rel)
        if name.startswith("shard_") and name.endswith(".jsonl"):
            if rel[: -len(".jsonl")] + ".safetensors" in files:
                markers.append(rel)
        elif not name.startswith("shard_") and size <= SMALL_FILE_BYTES:
            small.append(rel)
    return markers, small


def fetch(store: Store, dest: str, keep: list[str] | None = None, datasets: list[str] | None = None,
          workers: int = 4, log=print) -> dict:
    markers, small = units(store.files(), datasets)
    for rel in small:
        local = os.path.join(dest, rel)
        store.get(rel, local + ".tmp")
        os.replace(local + ".tmp", local)
    todo = [m for m in markers if not (os.path.exists(os.path.join(dest, m))
                                       and os.path.exists(os.path.join(dest, m[: -len(".jsonl")] + ".safetensors")))]
    log(f"{len(markers)} complete shards in {store.url}; {len(markers) - len(todo)} already here; fetching {len(todo)}"
        + (f", keeping {','.join(keep)}" if keep else ""))
    missing: set[str] = set()
    t0, done = time.time(), [0]

    def one(marker: str) -> None:
        data = marker[: -len(".jsonl")] + ".safetensors"
        local = os.path.join(dest, data)
        if keep:
            store.get(data, local + ".full")
            missing.update(slim(local + ".full", local + ".tmp", keep))
            os.remove(local + ".full")
        else:
            store.get(data, local + ".tmp")
        os.replace(local + ".tmp", local)
        store.get(marker, os.path.join(dest, marker) + ".tmp")
        os.replace(os.path.join(dest, marker) + ".tmp", os.path.join(dest, marker))  # the marker last
        done[0] += 1
        if done[0] % 10 == 0 or done[0] == len(todo):
            log(f"  {done[0]}/{len(todo)} shards ({time.time() - t0:.0f}s)")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, todo))
    if missing:
        log(f"note: some shards had no {sorted(missing)}")
    return {"shards": len(markers), "fetched": len(todo), "missing_tensors": sorted(missing)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store", help="hf://buckets/NAMESPACE/BUCKET/PREFIX or a local folder")
    ap.add_argument("dest")
    ap.add_argument("--keep", help="comma-list of tensors to keep (default: the whole shard)")
    ap.add_argument("--datasets", help="comma-list (default: every dataset in the store)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    fetch(open_store(args.store), args.dest, args.keep.split(",") if args.keep else None,
          args.datasets.split(",") if args.datasets else None, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
