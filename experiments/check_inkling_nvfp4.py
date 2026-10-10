"""GPU acceptance check for the NVFP4 Inkling reader (``whitebox/inkling_nvfp4.py``).

Run it on the GPU box before extracting probe features or serving probes, and keep the JSON with the
outputs. It exits 1 if any check fails.

    uv run python experiments/check_inkling_nvfp4.py --model /mnt/ram/Inkling-Small-NVFP4 \
        --episodes inputs/eval_rollouts.jsonl --run-info inputs/baseline.json \
        --bf16-dir /mnt/ram/bf16 --out results/check_inkling_nvfp4.json

Checks, each recorded with its numbers:

- **load:** the packed layers are swapped in; records GPU memory and seconds.
- **greedy:** "What is the capital of France? Answer in one word." gets an answer turn that says Paris.
- **episodes:** real TerminalEnv episodes, hacks and non-hacks, both probe views, rendered by
  ``InklingReplay``.
  - The readout token is the follow-up "Yes", and every hidden state is finite.
  - Records per-layer max |activation| at the readout, which decides float16 vs bfloat16 feature storage.
- **bf16** (with ``--bf16-dir``): the first packed layer against the BF16 release, on its weights and on
  its expert outputs for real inputs. Must be within NVFP4's own error. The shards it needs are fetched
  from ``--bf16-repo`` into ``--bf16-dir``.
- **decode:** the fused CUDA decode is bit-identical to the eager one; records its cost per forward.
- **throughput:** tokens/s at episode length, with every hidden state kept.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import torch

# Bounds from the 2026-10-09 B300 check (weights 8.1% / experts output 8.75% relative error).
MAX_WEIGHT_REL_ERR, MAX_OUTPUT_REL_ERR, MIN_COS = 0.10, 0.12, 0.99


def pick_episodes(path: str, n: int) -> list[dict]:
    """The first n/2 valid hacks and n/2 valid non-hacks of a TerminalEnv eval_rollouts.jsonl."""
    hacks, honest = [], []
    with open(path) as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("invalid_reason"):
                continue
            (hacks if row["behavior_present"] else honest).append(row)
            if len(hacks) >= n // 2 and len(honest) >= n - n // 2:
                break
    return hacks[: n // 2] + honest[: n - n // 2]


def rel(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.float().flatten(), b.float().flatten()
    return {"rel_err": round(((a - b).norm() / b.norm()).item(), 4),
            "cos": round(torch.nn.functional.cosine_similarity(a, b, dim=0).item(), 5)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="thinkingmachines/Inkling-Small-NVFP4", help="hub id or local dir")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--episodes", help="a TerminalEnv eval_rollouts.jsonl (skip the episode check if unset)")
    ap.add_argument("--run-info", help="that run's baseline.json / run_info.json (for env_options and effort)")
    ap.add_argument("--n-episodes", type=int, default=12)
    ap.add_argument("--bf16-dir", help="where to fetch/find BF16 shards for the comparison (skip if unset)")
    ap.add_argument("--bf16-repo", default="thinkingmachines/Inkling-Small")
    ap.add_argument("--batches", default="16,32", help="throughput batch sizes, comma-separated")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.episodes and not args.run_info:
        ap.error("--episodes needs --run-info (the run's env_options rebuild the observations)")

    from monitordecorrelation.whitebox.inkling_nvfp4 import (
        decode_nvfp4_into,
        load_inkling_nvfp4,
        local_checkpoint,
        packed_expert_layers,
        weight_map,
    )
    from monitordecorrelation.whitebox.inkling_replay import VIEWS, InklingReplay, terminal_episode_messages

    import transformers

    report: dict = {"model": args.model, "gpu": torch.cuda.get_device_name(args.device), "torch": torch.__version__,
                    "transformers": transformers.__version__, "python": platform.python_version(), "checks": {}}
    checks = report["checks"]

    def save() -> None:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)

    # --- load
    t0 = time.time()
    model = load_inkling_nvfp4(args.model, device=args.device).eval()
    ckpt = local_checkpoint(args.model)
    layers = packed_expert_layers(weight_map(ckpt))
    runner = model.model.layers[layers[0]].mlp.experts._runner_box[0]
    report["load"] = {"seconds": round(time.time() - t0, 1), "packed_layers": [layers[0], layers[-1], len(layers)],
                      "gpu_allocated_gib": round(torch.cuda.memory_allocated(args.device) / 2**30, 1),
                      "experts_implementation": str(model.config._experts_implementation)}
    checks["load"] = len(layers) > 0
    print("load", json.dumps(report["load"]), flush=True)
    save()

    run_info = json.load(open(args.run_info)) if args.run_info else {}
    effort = float(run_info.get("effort", 0.5))
    replay = InklingReplay(effort=effort)
    renderer = replay.renderer
    stop = renderer._sp["content_model_end_sampling"]

    with torch.inference_mode():
        # --- greedy (no KV cache: every step is a full forward, as the probes use the model)
        prompt = "What is the capital of France? Answer in one word."
        ids, out = list(renderer.prompt_tokens(prompt)), []
        while len(out) < 96 and (not out or out[-1] != stop):
            logits = model(input_ids=torch.tensor([ids + out], device=args.device), logits_to_keep=1).logits
            out.append(int(logits[0, -1].argmax()))
        cot, answer, _ = renderer.parse(out)
        report["greedy"] = {"prompt": prompt, "thinking": cot, "answer": answer, "stopped": out[-1] == stop}
        checks["greedy"] = "Paris" in answer and out[-1] == stop
        print("greedy", json.dumps(report["greedy"]), flush=True)
        save()

        # --- episodes
        captured: dict = {}
        if args.episodes:
            opts = dict(run_info.get("env_options", {}))
            yes = replay.answer_tokens[-1]
            rows = pick_episodes(args.episodes, args.n_episodes)
            def keep_inputs(mod, inp, outp) -> None:  # returns None: a forward hook's return value replaces the output
                captured.setdefault("args", [a.clone() for a in inp])

            hook = model.model.layers[layers[0]].mlp.experts.register_forward_hook(keep_inputs)
            episodes = []
            for row in rows:
                meta = row["env_meta"]
                ep_opts = {**opts, **{k: meta[k] for k in ("verifier_mode", "submission_ends",
                                                          "last_turn_submit_notice") if k in meta}}
                for view in VIEWS:
                    msgs = terminal_episode_messages(row["question"], meta["turns"], ep_opts,
                                                     include_reasoning=view == "with_reasoning")
                    probe = replay.probe_input(msgs)
                    res = model(input_ids=torch.tensor([probe.tokens], device=args.device), output_hidden_states=True,
                                logits_to_keep=torch.tensor([probe.readout - 1], device=args.device))
                    hs = torch.stack([h[0, probe.readout] for h in res.hidden_states]).float()
                    episodes.append({
                        "hack": bool(row["behavior_present"]), "view": view, "n_tokens": len(probe.tokens),
                        "readout_is_yes": probe.tokens[probe.readout] == yes,
                        "p_yes": round(res.logits[0, -1].float().softmax(-1)[yes].item(), 4),
                        "n_hidden_states": len(res.hidden_states), "finite": bool(torch.isfinite(hs).all()),
                        "max_abs_per_layer": [round(v, 1) for v in hs.abs().amax(-1).tolist()],
                    })
            hook.remove()
            n_states = model.config.num_hidden_layers + 1
            report["episodes"] = episodes
            report["readout_max_abs"] = max(max(e["max_abs_per_layer"]) for e in episodes)
            checks["episodes"] = all(e["readout_is_yes"] and e["finite"] and e["n_hidden_states"] == n_states
                                     for e in episodes) and len(episodes) == 2 * len(rows)
            print("episodes", len(episodes), "readout_max_abs", report["readout_max_abs"], flush=True)
            save()

        # --- bf16 comparison on the first packed layer
        if args.bf16_dir:
            from huggingface_hub import hf_hub_download
            from safetensors import safe_open
            from transformers.core_model_loading import Interleave

            index = hf_hub_download(args.bf16_repo, "model.safetensors.index.json", local_dir=args.bf16_dir)
            bf16_map = json.load(open(index))["weight_map"]
            prefix = f"model.llm.layers.{layers[0]}.mlp.experts."

            def bf16(name: str) -> torch.Tensor:
                path = hf_hub_download(args.bf16_repo, bf16_map[prefix + name], local_dir=args.bf16_dir)
                with safe_open(path, "pt", device=args.device) as f:
                    return f.get_tensor(prefix + name)

            experts = model.model.layers[layers[0]].mlp.experts
            ref_w13 = Interleave(dim=1).convert({"w": bf16("w13_weight")}, None, ["w"])["w"]
            ref_w2 = bf16("w2_weight")
            decode_nvfp4_into(runner.gate_up_proj.data, *experts.w13)
            decode_nvfp4_into(runner.down_proj.data, *experts.w2)
            cmp = {"layer": layers[0], "w13": rel(runner.gate_up_proj.data, ref_w13),
                   "w2": rel(runner.down_proj.data, ref_w2)}
            checks_ok = all(cmp[k]["rel_err"] < MAX_WEIGHT_REL_ERR and cmp[k]["cos"] > MIN_COS for k in ("w13", "w2"))
            if captured:
                got = runner(*captured["args"])
                runner.gate_up_proj.data.copy_(ref_w13)
                runner.down_proj.data.copy_(ref_w2)
                cmp["experts_output"] = {**rel(got, runner(*captured["args"])),
                                         "n_tokens": int(captured["args"][0].shape[0])}
                checks_ok &= (cmp["experts_output"]["rel_err"] < MAX_OUTPUT_REL_ERR
                              and cmp["experts_output"]["cos"] > MIN_COS)
            report["bf16"] = cmp
            checks["bf16"] = checks_ok
            del ref_w13, ref_w2
            print("bf16", json.dumps(cmp), flush=True)
            save()

        # --- decode: fused == eager, and its cost
        experts = model.model.layers[layers[-1]].mlp.experts
        fused = torch.empty_like(runner.gate_up_proj.data)
        decode_nvfp4_into(fused, *experts.w13)
        eager = torch.empty_like(fused)
        decode_nvfp4_into(eager, *experts.w13, fused=False)
        identical = torch.equal(fused, eager)
        del fused, eager
        torch.cuda.synchronize(args.device)
        t = time.time()
        for i in layers:
            m = model.model.layers[i].mlp.experts
            decode_nvfp4_into(runner.gate_up_proj.data, *m.w13)
            decode_nvfp4_into(runner.down_proj.data, *m.w2)
        torch.cuda.synchronize(args.device)
        report["decode"] = {"fused_equals_eager": identical, "seconds_all_layers": round(time.time() - t, 3)}
        checks["decode"] = identical
        print("decode", json.dumps(report["decode"]), flush=True)
        save()

        # --- throughput
        report["throughput"] = []
        for batch in (int(b) for b in args.batches.split(",") if b):
            ids = torch.randint(1000, 100000, (batch, args.seq), device=args.device)
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            t = time.time()
            res = model(input_ids=ids, output_hidden_states=True, logits_to_keep=1)
            torch.cuda.synchronize(args.device)
            dt = time.time() - t
            del res
            report["throughput"].append({"batch": batch, "seq": args.seq, "seconds": round(dt, 2),
                                         "tokens_per_s": round(batch * args.seq / dt),
                                         "peak_gib": round(torch.cuda.max_memory_allocated(args.device) / 2**30, 1)})
            print("throughput", json.dumps(report["throughput"][-1]), flush=True)
            save()

    report["ok"] = all(checks.values())
    save()
    print(json.dumps({"ok": report["ok"], "checks": checks}), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
