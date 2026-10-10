"""How fast this machine saves to a store (an HF bucket): the speed at which a GPU session's shards reach the bucket.

    uv run python scripts/bucket_speed_test.py --store hf://buckets/probemaxxing/inkling-probes/tmp/speedtest \
        --out speedtest/speedtest.json

- **What it uploads:** files of random bytes. Like bf16 features, they don't compress and are never deduplicated.
- **How:** through whitebox/store.py's ``Uploader``, the path the extraction uses: a data file, then a small marker.
- **Phases:**
  1. one uploader: one extraction process;
  2. ``--parallel`` uploaders, in separate processes (default 4 and 8): the session runs four reader copies.
  Each uploader stops after ``--seconds`` or ``--files`` files, whichever comes first.
- **Then:** it downloads two files back (the resume path) and deletes what each phase uploaded.
- **Output:** a JSON summary in ``--out`` and on stdout.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import socket
import sys
import tempfile
import time
from datetime import datetime, timezone

import numpy as np

from monitordecorrelation.whitebox.store import Uploader, open_store


def make_files(folder: str, n: int, size: int, seed: int) -> list[str]:
    """``n`` files of ``size`` unique random bytes (no two chunks alike, so nothing deduplicates)."""
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        path = os.path.join(folder, f"f{i:03d}.bin")
        with open(path, "wb") as fh:
            fh.write(rng.bytes(size))
        paths.append(path)
    return paths


def upload(url: str, prefix: str, paths: list[str], marker: str, seconds: float) -> dict:
    """Upload ``paths`` one after another, each as a unit (data, then the marker), until ``seconds`` have passed."""
    errors: list[str] = []
    uploader = Uploader(open_store(url), retries=3, backoff=2.0,
                        log=lambda msg: errors.append(msg) if "failed" in msg else None)
    t0, sent, done = time.time(), 0, 0
    for i, path in enumerate(paths):
        if time.time() - t0 >= seconds:
            break
        uploader.submit(f"{prefix}/f{i:03d}", [(path, f"{prefix}/f{i:03d}.bin")],
                        marker=(marker, f"{prefix}/f{i:03d}.done"))
        uploader.flush()
        if uploader.failed:
            break
        sent += os.path.getsize(path)
        done += 1
    uploader.close()
    elapsed = time.time() - t0
    return {"files": done, "bytes": sent, "seconds": round(elapsed, 2),
            "mb_per_s": round(sent / elapsed / 1e6, 1) if elapsed > 0 else None, "errors": errors}


def _worker(args: tuple) -> dict:
    return upload(*args)


def run_phase(name: str, url: str, run_id: str, workers: int, ram: str, file_mb: float, files: int,
              seconds: float, marker: str, seed: int) -> dict:
    size = int(file_mb * 1e6)
    jobs = []
    for w in range(workers):
        paths = make_files(os.path.join(ram, name, f"w{w}"), files, size, seed + 1000 * w)
        jobs.append((url, f"{run_id}/{name}/w{w}", paths, marker, seconds))
    t0 = time.time()
    if workers == 1:
        results = [_worker(jobs[0])]
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            results = pool.map(_worker, jobs)
    wall = time.time() - t0
    total = sum(r["bytes"] for r in results)
    phase = {"phase": name, "uploaders": workers, "file_mb": file_mb, "files": sum(r["files"] for r in results),
             "bytes": total, "wall_seconds": round(wall, 2),
             "mb_per_s": round(total / wall / 1e6, 1) if wall > 0 else None, "per_uploader": results}
    print(f"{name}: {workers} uploader(s), {phase['files']} files, {total / 1e9:.2f} GB in {wall:.0f}s = "
          f"{phase['mb_per_s']} MB/s in total", flush=True)
    shutil.rmtree(os.path.join(ram, name), ignore_errors=True)
    return phase


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, help="hf://buckets/NS/BUCKET/PREFIX (or a local dir, for tests)")
    ap.add_argument("--out", required=True, help="where the JSON summary goes")
    ap.add_argument("--file-mb", type=float, default=500.0, help="file size; a feature shard is ~500 MB")
    ap.add_argument("--files", type=int, default=6, help="at most this many files per uploader per phase")
    ap.add_argument("--seconds", type=float, default=90.0, help="each uploader stops starting files after this")
    ap.add_argument("--parallel", default="4,8", help="comma-list of parallel uploader counts after the single one")
    ap.add_argument("--ram", default=os.environ.get("NEBIUS_JOB_RAM") or tempfile.gettempdir(),
                    help="where the test files are written (the VM's RAM disk)")
    args = ap.parse_args()

    run_id = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{socket.gethostname()}"
    ram = os.path.join(args.ram, f"speedtest-{run_id}")
    os.makedirs(ram, exist_ok=True)
    marker = os.path.join(ram, "marker.jsonl")
    with open(marker, "w") as fh:
        fh.write('{"row": 0}\n')
    store = open_store(args.store)
    summary = {"store": store.url, "run_id": run_id, "host": socket.gethostname(), "cpus": os.cpu_count(),
               "started_utc": f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}", "phases": [], "download": None,
               "leftovers": None}
    try:
        counts = [1] + [int(n) for n in args.parallel.split(",") if n.strip()]
        for k, workers in enumerate(counts):
            name = "single" if workers == 1 else f"parallel{workers}"
            summary["phases"].append(run_phase(name, args.store, run_id, workers, ram, args.file_mb, args.files,
                                               args.seconds, marker, seed=k))
            stored = [rel for rel in store.files() if rel.startswith(f"{run_id}/{name}/")]
            if name == "single":  # the resume path: read two shards back
                wanted = sorted(rel for rel in stored if rel.endswith(".bin"))[:2]
                t0, got = time.time(), 0
                for rel in wanted:
                    local = os.path.join(ram, "download", os.path.basename(rel))
                    store.get(rel, local)
                    got += os.path.getsize(local)
                dt = time.time() - t0
                summary["download"] = {"files": len(wanted), "bytes": got, "seconds": round(dt, 2),
                                       "mb_per_s": round(got / dt / 1e6, 1) if dt > 0 and got else None}
                print(f"download: {got / 1e9:.2f} GB in {dt:.1f}s = {summary['download']['mb_per_s']} MB/s",
                      flush=True)
            store.delete(stored)
    finally:
        try:  # whatever happened, leave nothing behind in the store
            left = [rel for rel in store.files() if rel.startswith(f"{run_id}/")]
            store.delete(left)
            summary["leftovers"] = len([rel for rel in store.files() if rel.startswith(f"{run_id}/")])
        except Exception as e:  # noqa: BLE001
            summary["leftovers"] = f"cleanup failed: {e}"
        shutil.rmtree(ram, ignore_errors=True)
        summary["finished_utc"] = f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}"
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=1)
        print(json.dumps({k: summary[k] for k in ("store", "cpus", "download", "leftovers")}
                         | {"mb_per_s": {p["phase"]: p["mb_per_s"] for p in summary["phases"]}}), flush=True)
    failed = any(r["errors"] for p in summary["phases"] for r in p["per_uploader"])
    return 1 if failed or summary["leftovers"] else 0


if __name__ == "__main__":
    sys.exit(main())
