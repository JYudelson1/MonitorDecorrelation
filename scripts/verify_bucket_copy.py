#!/usr/bin/env python3
"""Check that a local folder is in a Hugging Face bucket prefix byte for byte, before the local copy is deleted.

    uv run python scripts/verify_bucket_copy.py ../scratch/features/nvfp4 \
        hf://buckets/probemaxxing/inkling-probes/features/nvfp4

**How:** ``HfApi.list_bucket_tree`` lists every file under the prefix with its size and Xet content hash;
``hf_xet.hash_files`` computes the same hash for the local files (``--batch`` at a time) without uploading them.
Nothing is downloaded: 53 GB of 4-bit features took 32 s on hub.

**Exit code:**
- 0 if every local file is in the bucket with the same size and hash;
- 1 if a local file is missing from the bucket or differs, if the folder has no files, or, with ``--exact``, if the
  bucket has files the folder doesn't (a slim local copy, e.g. scripts/fetch_features.py ``--keep``, never matches);
- 2 on a bad argument.

It prints progress, every missing or differing file and a summary line ending in ``mismatches N``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

HF_PREFIX = "hf://buckets/"


def parse_bucket(url: str) -> tuple[str, str]:
    """``hf://buckets/NAMESPACE/BUCKET[/PREFIX]`` -> ("NAMESPACE/BUCKET", "PREFIX")."""
    namespace, _, rest = url[len(HF_PREFIX):].partition("/") if url.startswith(HF_PREFIX) else ("", "", "")
    bucket, _, prefix = rest.partition("/")
    if not namespace or not bucket:
        raise ValueError(f"{url!r}: want hf://buckets/NAMESPACE/BUCKET[/PREFIX]")
    return f"{namespace}/{bucket}", prefix.strip("/")


def remote_files(api, bucket_id: str, prefix: str) -> dict[str, tuple[int, str | None]]:
    """{path below the prefix: (size, Xet hash)} for every file under ``prefix``."""
    lead = f"{prefix}/" if prefix else ""
    return {f.path[len(lead):]: (int(f.size), getattr(f, "xet_hash", None))
            for f in api.list_bucket_tree(bucket_id, prefix=lead or None, recursive=True)
            if getattr(f, "size", None) is not None and f.path.startswith(lead)}


def local_files(root: str) -> list[str]:
    return sorted(os.path.relpath(os.path.join(d, n), root).replace(os.sep, "/")
                  for d, _, names in os.walk(root) for n in names)


def verify(local: str, url: str, api=None, hash_files=None, batch: int = 16, log=print) -> dict:
    """Compare ``local`` with the bucket prefix ``url``; returns the counts and the problem files."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    if hash_files is None:
        from hf_xet import hash_files
    bucket_id, prefix = parse_bucket(url)
    remote = remote_files(api, bucket_id, prefix)
    files = local_files(local)
    missing, mismatched, size, t0 = [], [], 0, time.time()
    for k in range(0, len(files), batch):
        chunk = files[k:k + batch]
        for rel, info in zip(chunk, hash_files([os.path.join(local, rel) for rel in chunk]), strict=True):
            size += info.file_size
            want = remote.get(rel)
            if want is None:
                missing.append(rel)
            elif want != (info.file_size, info.hash):
                mismatched.append({"path": rel, "local": (info.file_size, info.hash), "bucket": want})
        log(f"{k + len(chunk)}/{len(files)} files, {size / 1e9:.1f} GB, {time.time() - t0:.0f}s, "
            f"{len(missing) + len(mismatched)} problems")
    return {"local_files": len(files), "bucket_files": len(remote), "bytes": size, "missing_from_bucket": missing,
            "mismatched": mismatched, "only_in_bucket": sorted(set(remote) - set(files))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("local", help="the local folder")
    ap.add_argument("bucket", help="hf://buckets/NAMESPACE/BUCKET/PREFIX")
    ap.add_argument("--exact", action="store_true", help="also fail if the bucket has files the folder doesn't")
    ap.add_argument("--batch", type=int, default=16, help="files hashed per hf_xet call")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.local):
        ap.error(f"{args.local} is not a folder")
    try:
        parse_bucket(args.bucket)
    except ValueError as e:
        ap.error(str(e))
    r = verify(args.local, args.bucket, batch=args.batch)
    for rel in r["missing_from_bucket"]:
        print(f"MISSING from the bucket: {rel}")
    for m in r["mismatched"]:
        print(f"MISMATCH {m['path']}: local (size, hash) {m['local']}, bucket {m['bucket']}")
    if args.exact:
        for rel in r["only_in_bucket"]:
            print(f"ONLY IN THE BUCKET: {rel}")
    print(f"local files {r['local_files']}, bucket files {r['bucket_files']}, only in the bucket "
          f"{len(r['only_in_bucket'])}, missing from the bucket {len(r['missing_from_bucket'])}, mismatches "
          f"{len(r['mismatched'])}")
    if not r["local_files"]:
        print(f"no files in {args.local}: nothing verified")
    bad = r["missing_from_bucket"] or r["mismatched"] or not r["local_files"] or (args.exact and r["only_in_bucket"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
