"""GPU check of the Inkling probe server with the deployed probes, on the serving reader.

    uv run python experiments/check_probe_server.py --model /mnt/ram/Inkling-Small-NVFP4 \
        --probes-root data/probes/Inkling-Small --out results/check_probe_server.json

1. **Golden check** of every deployed probe, scoring one episode per forward (``--max-batch-tokens 1``, the batching
   rule's fix) and batched as in the extraction (65,536 tokens):
   - max |Δlogit| within the probe's own tolerance;
   - |mean Δlogit| within ``--golden-mean-tolerance`` (a systematic shift; whitebox/probe_server.py
     ``GOLDEN_MEAN_TOLERANCE``).
   Each probe's max, mean and sd of Δlogit go into the report, also when it fails.
2. **End to end:** the HTTP server and ``InklingProbeMonitor`` score the golden episodes as RL rollouts would arrive;
   the scores must equal the scorer's.
3. **Throughput** of both batchings, in tokens per second.

Exits 1 if a check fails.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time


def main(argv: list[str] | None = None) -> int:
    from monitordecorrelation.whitebox.probe_server import GOLDEN_MEAN_TOLERANCE

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--probes-root", default="data/probes/Inkling-Small")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0", help="for NVFP4 (one device); BF16 uses the visible GPUs")
    ap.add_argument("--golden-mean-tolerance", type=float, default=GOLDEN_MEAN_TOLERANCE,
                    help="max |mean Δlogit| over each probe's golden episodes: a systematic shift (default "
                         f"{GOLDEN_MEAN_TOLERANCE}, probe_server.GOLDEN_MEAN_TOLERANCE; inf turns the check off)")
    args = ap.parse_args(argv)

    import numpy as np

    from monitordecorrelation.monitors.inkling_probe_monitor import InklingProbeMonitor
    from monitordecorrelation.types import Prompt, Rollout
    from monitordecorrelation.whitebox.extraction import reader_info
    from monitordecorrelation.whitebox.inkling_nvfp4 import is_nvfp4_checkpoint, load_inkling_nvfp4
    from monitordecorrelation.whitebox.model import load_inkling_text
    from monitordecorrelation.whitebox.probe_datasets import EPISODE_ENV_KEYS
    from monitordecorrelation.whitebox.probe_server import ProbeScorer, Server, probe_sha

    log = lambda m: print(f"[check {time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731
    dirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(args.probes_root, "*", "*", "*", "probe.npz")))
    t = time.time()
    model = (load_inkling_nvfp4(args.model, device=args.device) if is_nvfp4_checkpoint(args.model)
             else load_inkling_text(args.model)).eval()
    info = reader_info(model, args.model)
    log(f"loaded {args.model} in {time.time() - t:.0f}s; {len(dirs)} probes")
    report: dict = {"reader": info, "probes": {}, "throughput": {}, "ok": True}

    for mbt in (1, 65536):
        scorer = ProbeScorer(model, dirs, max_batch_tokens=mbt, reader_info=info)
        for d in dirs:
            name = os.path.relpath(d, args.probes_root)
            tol = json.load(open(os.path.join(d, "meta.json")))["golden_tolerance"]
            mean_tol = args.golden_mean_tolerance
            try:
                r = scorer.golden_check([d], tol, mean_tolerance=mean_tol)[probe_sha(d)]
                res = {**r, "ok": r["max_dlogit"] <= tol and abs(r["mean_dlogit"]) <= mean_tol}
            except RuntimeError as e:  # a GoldenCheckError carries the probe's numbers
                res = {**getattr(e, "report", {}), "tolerance": tol, "mean_tolerance": mean_tol, "ok": False,
                       "error": str(e)}
            report["probes"].setdefault(name, {})[f"golden_max_batch_tokens_{mbt}"] = res
            report["ok"] &= res["ok"]
            log(f"golden {name} (max_batch_tokens {mbt}): {res}")
        episodes = [json.loads(line)["episode"] for line in open(os.path.join(dirs[0], "golden.jsonl"))]
        shas = [probe_sha(d) for d in dirs]
        t = time.time()
        out = scorer.score(episodes, shas)
        n_tok = sum(sum(o["n_tokens"].values()) for o in out)
        report["throughput"][str(mbt)] = {"episodes": len(episodes), "tokens": n_tok,
                                          "tokens_per_s": n_tok / (time.time() - t)}
        log(f"throughput (max_batch_tokens {mbt}): {report['throughput'][str(mbt)]}")

    # end to end through HTTP: one IID and one OOD probe, as an RL run's monitors would ask
    scorer = ProbeScorer(model, dirs, max_batch_tokens=1, reader_info=info)
    server = Server(scorer, port=0, log=log)
    server.start()
    url = f"http://127.0.0.1:{server.port}"
    e2e = {}
    for d in [d for d in dirs if d.endswith("followup")][:2]:
        rows = [json.loads(line) for line in open(os.path.join(d, "golden.jsonl"))]
        env_options = rows[0]["episode"]["env_options"]
        rollouts = [Rollout(prompt=Prompt(text=r["episode"]["question"]), cot="", output="",
                            meta={"episode": {"turns": r["episode"]["turns"],
                                              **{k: r["episode"]["env_options"][k] for k in EPISODE_ENV_KEYS}}})
                    for r in rows]
        mon = InklingProbeMonitor(os.path.relpath(d, args.probes_root), d, url, env_options=env_options,
                                  run_effort=0.5, log=log)
        t = time.time()
        got = mon.score_batch(rollouts)
        want = scorer.score([r["episode"] for r in rows], [mon.sha])
        diff = float(np.max(np.abs([g.meta["logit"] - w["scores"][mon.sha]["logit"] for g, w in zip(got, want)])))
        e2e[mon.name] = {"n": len(got), "seconds": time.time() - t, "max_dlogit_vs_scorer": diff, "ok": diff < 1e-3}
        report["ok"] &= e2e[mon.name]["ok"]
        log(f"end to end {mon.name}: {e2e[mon.name]}")
    server.stop()
    report["end_to_end"] = e2e
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    log(f"ok = {report['ok']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
