"""Serve Inkling probes to RL runs on the reader ``--model`` (whitebox/probe_server.py).

**4-bit reader** (NVFP4; the v1 monitors, data/probes/Inkling-Small/): our loader, on the one GPU ``--device`` names.

    uv run python experiments/inkling_probe_server.py --model /mnt/ram/Inkling-Small-NVFP4 \
        --probes data/probes/Inkling-Small/terminal_prerl/with_reasoning/followup,data/probes/Inkling-Small/diverse_deception/with_reasoning/followup \
        --port 8178

**Full-precision reader** (BF16; the v2 monitors, data/probes/Inkling-Small/v2/, fitted and served on it):
- **GPUs:** a checkpoint that isn't NVFP4 loads with transformers across every GPU the process sees
  (``device_map="auto"``; ``--device`` doesn't apply). Choose them with ``CUDA_VISIBLE_DEVICES``. The log's "loaded
  ... on" line names the devices it took: on too few GPUs transformers puts the rest on the CPU (``cpu`` or
  ``meta``), which works but is far too slow for RL runs.
- **One copy needs 3–4 B300s,** so an 8-GPU VM holds two copies: two servers, each on four GPUs and its own port.
- **Each RL run is pointed at one of them** (its monitors' server URL). The copies serve the same probes on the same
  reader, so a run gets the same scores from either.

    P=data/probes/Inkling-Small/v2
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python experiments/inkling_probe_server.py --model /mnt/ram/Inkling-Small \
        --probes $P/terminal_prerl_v2/with_reasoning/followup,$P/diverse_deception/with_reasoning/followup --port 8178
    CUDA_VISIBLE_DEVICES=4,5,6,7 uv run python experiments/inkling_probe_server.py --model /mnt/ram/Inkling-Small \
        --probes $P/terminal_prerl_v2/with_reasoning/followup,$P/diverse_deception/with_reasoning/followup --port 8179

Then, on the machine running the RL loop: ``ssh -N -L 8178:127.0.0.1:8178 md@<vm>`` (add ``-L 8179:127.0.0.1:8179``
for a second copy), and set each run's probe server URL to ``http://127.0.0.1:8178`` or ``http://127.0.0.1:8179``
(monitors/inkling_probe_monitor.py).

**The server refuses to start:**
- if a probe is for another reader than ``--model``: its meta.json ``reader_info`` names another loader or revision
  (e.g. the v1 monitors, served on the 4-bit reader, on the full-precision one);
- if a probe's golden episodes render differently or score differently than when the probe was deployed, in logits:
  one episode by more than ``--golden-tolerance``, or all of them by more than ``--golden-mean-tolerance`` on average
  (a systematic shift; the default is calibrated in whitebox/probe_server.py).

**Memory:** an episode whose dense attention doesn't fit is retried with its attention in query chunks; one that
doesn't fit even so is answered 413 (whitebox/probe_server.py), which the client doesn't retry.
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def main() -> int:
    from monitordecorrelation.whitebox.probe_server import GOLDEN_MEAN_TOLERANCE

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the reader checkpoint (local dir): NVFP4 or BF16 Inkling-Small")
    ap.add_argument("--probes", required=True, help="comma-list of probe directories (probe.npz + meta.json), all for "
                                                    "this reader")
    ap.add_argument("--device", default="cuda:0",
                    help="for NVFP4 (one device). BF16 ignores it: it spreads over every visible GPU "
                         "(CUDA_VISIBLE_DEVICES; one copy needs 3-4 B300s)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8178, help="e.g. 8179 for a second copy on the same VM")
    ap.add_argument("--max-batch-tokens", type=int, default=1,
                    help="padded tokens per forward. The default 1 scores one episode per forward: the pre-registered "
                         "batching rule's fix, since the rule fired on 2026-10-10 (docs/INKLING_PROBES.md, section 7). "
                         "65536 batches like the extraction: ~2x faster, but a score then depends on its batch")
    ap.add_argument("--golden-tolerance", type=float, default=None,
                    help="max |Δlogit| on the golden episodes (default: each probe's own, else 1.0)")
    ap.add_argument("--golden-mean-tolerance", type=float, default=GOLDEN_MEAN_TOLERANCE,
                    help="max |mean Δlogit| over each probe's golden episodes: a systematic shift (default "
                         f"{GOLDEN_MEAN_TOLERANCE}, probe_server.GOLDEN_MEAN_TOLERANCE; inf turns the check off)")
    args = ap.parse_args()

    from monitordecorrelation.whitebox.extraction import reader_info
    from monitordecorrelation.whitebox.inkling_nvfp4 import is_nvfp4_checkpoint, load_inkling_nvfp4
    from monitordecorrelation.whitebox.model import load_inkling_text
    from monitordecorrelation.whitebox.probe_server import ProbeScorer, Server

    dirs = [d for d in args.probes.split(",") if d]
    t = time.time()
    model = (load_inkling_nvfp4(args.model, device=args.device) if is_nvfp4_checkpoint(args.model)
             else load_inkling_text(args.model)).eval()
    devices = sorted({str(p.device) for p in model.parameters()})  # BF16: its GPUs; cpu or meta = offloaded, slow
    print(f"[probe-server] loaded {args.model} in {time.time() - t:.0f}s on {', '.join(devices)}", flush=True)
    info = reader_info(model, args.model)
    if not info.get("revision"):  # reader_info reads it from the metadata `hf download --local-dir` leaves
        print(f"[probe-server] WARNING: can't tell which revision {args.model} is; monitors can't check it. Download "
              "it with `hf download REPO --revision REV --local-dir DIR`.", flush=True)
    scorer = ProbeScorer(model, dirs, max_batch_tokens=args.max_batch_tokens, reader_info=info)
    report = {}
    for d in dirs:  # each probe at its own max tolerance; the mean tolerance is the server's
        tol = args.golden_tolerance
        if tol is None:
            tol = json.load(open(f"{d}/meta.json")).get("golden_tolerance", 1.0)
        report.update(scorer.golden_check([d], tol, mean_tolerance=args.golden_mean_tolerance))
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
