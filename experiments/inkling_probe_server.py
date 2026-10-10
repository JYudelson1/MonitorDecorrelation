"""Serve Inkling probes to RL runs from one GPU (whitebox/probe_server.py).

    uv run python experiments/inkling_probe_server.py --model /mnt/ram/Inkling-Small-NVFP4 --max-batch-tokens 1 \
        --probes data/probes/Inkling-Small/terminal_prerl/with_reasoning/followup,data/probes/Inkling-Small/diverse_deception/with_reasoning/followup \
        --port 8178

Then, on the machine running the RL loop: ``ssh -N -L 8178:127.0.0.1:8178 md@<vm>``, and set the runs' probe server
URL to ``http://127.0.0.1:8178`` (monitors/inkling_probe_monitor.py).

The server refuses to start if a probe's golden episodes render differently or score differently than when the probe
was fitted (``--golden-tolerance``, in logits).
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the reader checkpoint (local dir): NVFP4 or BF16 Inkling-Small")
    ap.add_argument("--probes", required=True, help="comma-list of probe directories (probe.npz + meta.json)")
    ap.add_argument("--device", default="cuda:0", help="for NVFP4 (one device); BF16 uses the visible GPUs")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8178)
    ap.add_argument("--max-batch-tokens", type=int, default=65536,
                    help="padded tokens per forward: 1 scores one episode per forward, the pre-registered batching "
                         "rule's fix (it fired on 2026-10-10, see md_files/PROBE_RESULTS.md); 65536 batches like the "
                         "extraction")
    ap.add_argument("--golden-tolerance", type=float, default=None,
                    help="max |Δlogit| on the golden episodes (default: each probe's own, else 1.0)")
    args = ap.parse_args()

    from monitordecorrelation.whitebox.extraction import reader_info
    from monitordecorrelation.whitebox.inkling_nvfp4 import is_nvfp4_checkpoint, load_inkling_nvfp4
    from monitordecorrelation.whitebox.model import load_inkling_text
    from monitordecorrelation.whitebox.probe_server import ProbeScorer, Server

    dirs = [d for d in args.probes.split(",") if d]
    t = time.time()
    model = (load_inkling_nvfp4(args.model, device=args.device) if is_nvfp4_checkpoint(args.model)
             else load_inkling_text(args.model)).eval()
    print(f"[probe-server] loaded {args.model} in {time.time() - t:.0f}s", flush=True)
    scorer = ProbeScorer(model, dirs, max_batch_tokens=args.max_batch_tokens, reader_info=reader_info(model, args.model))
    tolerances = [json.load(open(f"{d}/meta.json")).get("golden_tolerance", 1.0) for d in dirs]
    report = scorer.golden_check(dirs, args.golden_tolerance if args.golden_tolerance is not None else max(tolerances))
    print(f"[probe-server] golden check passed: {json.dumps(report)}", flush=True)
    server = Server(scorer, args.host, args.port, log=lambda m: print(f"[probe-server] {m}", flush=True))
    server.start()
    print(json.dumps(scorer.meta(), indent=1), flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
