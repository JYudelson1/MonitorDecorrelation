"""Compile the C++ special judges (AetherCode testlib checkers, ICPC-Eval spj) once, cached by sha1."""
from __future__ import annotations

import glob
import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

CACHE = "/root/hf_cache/checkers"
TESTLIB = os.path.join(CACHE, "testlib.h")


def checker_path(src: str) -> str:
    return os.path.join(CACHE, hashlib.sha1(src.encode()).hexdigest())


def compile_checker(src: str) -> tuple[str, str | None]:
    """-> (binary path, error). Cached: an existing binary is reused."""
    out = checker_path(src)
    if os.path.exists(out):
        return out, None
    cpp = out + ".cpp"
    with open(cpp, "w") as fh:
        fh.write(src.replace("\r\n", "\n"))
    p = subprocess.run(["g++", "-O2", "-std=gnu++17", "-w", "-I", CACHE, cpp, "-o", out],
                       capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        with open(out + ".err", "w") as fh:
            fh.write(p.stderr)
        return out, p.stderr[-800:]
    return out, None


def _snap(repo):
    return glob.glob(f"/root/hf_cache/hub/datasets--{repo.replace('/', '--')}/snapshots/*")[0]


def main():
    import pyarrow.parquet as pq
    srcs = {}
    fs = sorted(glob.glob(_snap("m-a-p/AetherCode") + "/v1_2024/*.parquet"))
    for r in pq.ParquetDataset(fs).read(columns=["id", "checker"]).to_pylist():
        if r["checker"] and r["checker"].strip():
            srcs.setdefault(r["checker"], []).append(f"aether:{r['id']}")
    fs = sorted(glob.glob(_snap("RUC-AIBOX/ICPC-Eval") + "/data/*.parquet"))
    for r in pq.ParquetDataset(fs).read(columns=["id", "spj_code"]).to_pylist():
        if r["spj_code"] and r["spj_code"].strip():
            srcs.setdefault(r["spj_code"], []).append(f"icpc:{r['id']}")
    print(f"{len(srcs)} unique checker sources", flush=True)
    with ThreadPoolExecutor(64) as ex:
        results = list(ex.map(lambda kv: (kv[1], compile_checker(kv[0])), srcs.items()))
    bad = [(ids, err) for ids, (path, err) in results if err]
    print(f"compiled {len(results) - len(bad)} ok, {len(bad)} failed")
    for ids, err in bad:
        print("FAILED", ids, err.strip().splitlines()[-1][:200])


if __name__ == "__main__":
    main()
