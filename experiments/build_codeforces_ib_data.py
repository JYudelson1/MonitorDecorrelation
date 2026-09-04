"""Build the local Codeforces-IB dataset (statement + visible tests + hidden tests) from open-r1/codeforces.

    uv run python experiments/build_codeforces_ib_data.py --n-hardest 1024          # -> data/codeforces_ib/hardest1024.jsonl.gz
    uv run python experiments/build_codeforces_ib_data.py --n-hardest 512
    uv run python experiments/build_codeforces_ib_data.py --min-rating 2400 --out data/codeforces_ib/r2400.jsonl.gz

Selection is by Codeforces rating only (hardest-N, fixed tie-break shuffle) — never by any model's
performance. Prompt-length policy (see ``envs/codeforces_ib.py::select_tests``): a test is visible only if
input+output <= --max-test-chars; at most --max-visible are shown; the rendered prompt must fit
--max-prompt-chars; problems with fewer than --min-visible visible or --min-hidden hidden tests are dropped.
Downloads one ~64 MB parquet per contest into HF_HOME (hundreds of contests for the hardest 1024).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/root/hf_cache")

from monitordecorrelation.envs.codeforces_ib import DATA_DIR, build_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-hardest", type=int, default=1024)
    ap.add_argument("--min-rating", type=int, default=None, help="use every rated problem >= this instead of --n-hardest")
    ap.add_argument("--out", default=None, help="default: data/codeforces_ib/hardest<N>.jsonl.gz (or r<rating>)")
    ap.add_argument("--max-test-chars", type=int, default=1000)
    ap.add_argument("--max-visible", type=int, default=12)
    ap.add_argument("--min-visible", type=int, default=4)
    ap.add_argument("--min-hidden", type=int, default=1)
    ap.add_argument("--max-hidden", type=int, default=16)
    ap.add_argument("--max-hidden-test-chars", type=int, default=100_000)
    ap.add_argument("--max-prompt-chars", type=int, default=12_000)
    ap.add_argument("--reserve-hidden", type=int, default=2,
                    help="hold back small tests as hidden until at least this many hidden tests exist")
    ap.add_argument("--download-workers", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out) if a.out else DATA_DIR / (f"r{a.min_rating}.jsonl.gz" if a.min_rating else f"hardest{a.n_hardest}.jsonl.gz")
    stats = build_dataset(
        out, n_hardest=a.n_hardest, min_rating=a.min_rating, download_workers=a.download_workers,
        max_test_chars=a.max_test_chars, max_visible=a.max_visible, min_visible=a.min_visible,
        min_hidden=a.min_hidden, max_hidden=a.max_hidden, max_hidden_test_chars=a.max_hidden_test_chars,
        max_prompt_chars=a.max_prompt_chars, reserve_hidden=a.reserve_hidden,
    )
    print(json.dumps({"out": str(out), **stats}, indent=2))
    with open(str(out) + ".build.json", "w") as fh:
        json.dump({"out": str(out), "args": vars(a), **stats}, fh, indent=2)


if __name__ == "__main__":
    main()
