"""Capability evals: every checkpoint in a JSON file vs the base model it was trained from.

    uv run python experiments/eval_checkpoints.py --checkpoints checkpoints.json --out data/capeval/run1.json

``--checkpoints`` is ``{"name": "tinker://…/weights/…" | "tinker://…/sampler_weights/…", …}``. Each
checkpoint's base model is read from tinker (``weights_info``); every distinct base model is evaluated
once. Training-state paths (``/weights/``) are converted once to sampler weights (cached in
``<out>.state.json``).

Benchmarks (``--benchmarks``, default all; see ``monitordecorrelation.capeval``):
  knowledge & reasoning  mmlu_pro (50/category = 700 q) · gpqa_diamond (198 q × 4) · math500 (500 × 2)
  agentic coding         lcb (LiveCodeBench v6 easy+medium, 95 × 2) · exercism (Python track, 140)
                         · canitedit (100 problems × {descriptive, lazy})
  free-form              coherence (MT-Bench 80 two-turn × 3, LLM judge: coherence / consistency /
                         rl_fried, + repetition)

Everything — all models × benchmarks × items × samples — runs concurrently (one asyncio loop; a global
cap on in-flight tinker requests with agent turns prioritized, a cap on sandbox containers / command
executions, a cap on judge calls). Every finished sample is appended to ``<out>.samples.jsonl`` at once;
``--resume`` skips what is already there (errored samples are retried, failed judgements re-judged).
At the end the per-model scores, the checkpoint-minus-base deltas (paired, cluster bootstrap 95% CI),
the confounder diagnostics (thinking forced / answers truncated / context / turn limit / format
errors / command timeouts / extraction or judge failures) and the token usage (millions; per base model
with its checkpoints, + the judge; input cache-miss / cache-hit / output, as reported by tinker and
OpenRouter) are printed and written to ``--out`` (each sample's usage is also in ``<out>.samples.jsonl``).

    uv run python experiments/eval_checkpoints.py --show data/capeval/run1.json   # reprint a finished run

Needs: TINKER_API_KEY (+ OPENROUTER_API_KEY for ``coherence``) in .env; docker for the agentic ones.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from monitordecorrelation.capeval.agentic import CanItEdit, Exercism, LiveCodeBench
from monitordecorrelation.capeval.bench import Benchmark, Item, RunCtx
from monitordecorrelation.capeval.coherence import DEFAULT_JUDGE, Judge, MTBenchCoherence
from monitordecorrelation.capeval.policy import USAGE, Budget, ModelSpec, Policy, PrioritySemaphore, SamplingConfig
from monitordecorrelation.capeval.qa import GPQADiamond, Math500, MMLUPro
from monitordecorrelation.capeval.report import aggregate_usage, build_report, iter_sample_lines, render_report

ALL_BENCHMARKS = ("mmlu_pro", "gpqa_diamond", "math500", "lcb", "exercism", "canitedit", "coherence")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", help="JSON {name: tinker path} (required unless --show)")
    p.add_argument("--out", help="detailed results JSON (samples go to <out>.samples.jsonl); required unless --show")
    p.add_argument("--show", metavar="OUT_JSON",
                   help="print the report of a finished run from its --out file and exit (no API calls)")
    p.add_argument("--benchmarks", default=",".join(ALL_BENCHMARKS),
                   help=f"comma-separated subset of {','.join(ALL_BENCHMARKS)}")
    p.add_argument("--resume", action="store_true", help="continue from <out>.samples.jsonl")
    p.add_argument("--limit", type=int, default=None, help="max items per benchmark (smoke tests)")
    p.add_argument("--no-base", action="store_true", help="skip the base models (deltas need them)")
    g = p.add_argument_group("benchmark sizes")
    g.add_argument("--mmlu-pro-per-category", type=int, default=50)
    g.add_argument("--mmlu-pro-samples", type=int, default=1)
    g.add_argument("--gpqa-samples", type=int, default=4)
    g.add_argument("--math500-samples", type=int, default=2)
    g.add_argument("--lcb-samples", type=int, default=2)
    g.add_argument("--lcb-difficulties", default="easy,medium")
    g.add_argument("--exercism-samples", type=int, default=1)
    g.add_argument("--canitedit-samples", type=int, default=1)
    g.add_argument("--coherence-samples", type=int, default=3)
    g = p.add_argument_group("generation (identical for every model)")
    g.add_argument("--temperature", type=float, default=0.6)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--top-k", type=int, default=20)
    g.add_argument("--thinking-effort", type=float, default=0.5,
                   help="reasoning effort for TML (Inkling) models; the terminal-env runs trained at 0.5")
    g.add_argument("--think-budget", type=int, default=36864,
                   help="single-turn QA: thinking budget (+ answer budget = 38912, Qwen's recommended output "
                        "length for hard reasoning benchmarks)")
    g.add_argument("--answer-budget", type=int, default=2048, help="single-turn QA: forced-answer budget")
    g.add_argument("--chat-think-budget", type=int, default=16384, help="coherence: per turn")
    g.add_argument("--chat-answer-budget", type=int, default=4096)
    g.add_argument("--agent-think-budget", type=int, default=16384, help="agentic: per turn")
    g.add_argument("--agent-answer-budget", type=int, default=4096)
    g = p.add_argument_group("agentic scaffold")
    g.add_argument("--max-turns", type=int, default=30)
    g.add_argument("--cmd-timeout", type=float, default=60.0)
    g.add_argument("--obs-chars", type=int, default=10000)
    g.add_argument("--lcb-test-timeout", type=float, default=10.0)
    g = p.add_argument_group("judge")
    g.add_argument("--judge-model", default=DEFAULT_JUDGE,
                   help="OpenRouter model of the coherence judge. Changing it on --resume re-judges the saved "
                        "conversations (no re-sampling).")
    g.add_argument("--judge-concurrency", type=int, default=32)
    g = p.add_argument_group("parallelism / infra")
    g.add_argument("--max-inflight", type=int, default=768,
                   help="concurrent tinker sampling requests (measured: 256 -> ~39k tok/s, 768 -> ~89k tok/s)")
    g.add_argument("--max-containers", type=int, default=400,
                   help="concurrent agent sandboxes (idle most of the time: they wait on the model); clamped "
                        "to what the machine's RAM can hold in the worst case (see capeval/sandbox.py)")
    g.add_argument("--max-exec", type=int, default=max(2, (os.cpu_count() or 4) - 3),
                   help="concurrent sandbox command executions / gradings (each ≤ 1 CPU); also capped "
                        "box-wide, across every capeval run on the machine, at cores − 2")
    g.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/capeval"))
    g.add_argument("--sampler-ttl-days", type=float, default=30.0,
                   help="TTL of the sampler checkpoints created from /weights/ paths")
    a = p.parse_args()
    if a.show is None and (a.checkpoints is None or a.out is None):
        p.error("--checkpoints and --out are required (unless --show)")
    a.benchmarks = [b.strip() for b in a.benchmarks.split(",") if b.strip()]
    bad = set(a.benchmarks) - set(ALL_BENCHMARKS)
    if bad:
        p.error(f"unknown benchmarks: {sorted(bad)}")
    return a


def log(msg: str) -> None:
    from tqdm import tqdm

    tqdm.write(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------
def load_checkpoints(path: str) -> dict[str, str]:
    raw = json.load(open(path))
    if isinstance(raw, list):
        raw = {d["name"]: d["path"] for d in raw}
    if not isinstance(raw, dict) or not raw:
        raise SystemExit(f"{path}: expected a non-empty JSON object {{name: tinker path}}")
    out = {}
    for name, p in raw.items():
        p = str(p).strip()
        if not p.startswith("tinker://"):
            raise SystemExit(f"{path}: {name!r} is not a tinker:// path: {p!r}")
        if name.startswith("base:"):
            raise SystemExit(f"{path}: checkpoint names may not start with 'base:' ({name!r})")
        out[str(name)] = p
    return out


def resolve_models(sc, ckpts: dict[str, str], state: dict, ttl_days: float) -> list[ModelSpec]:
    """Base model of every checkpoint (tinker weights_info) + a sampler-weights path for each."""
    rc = sc.create_rest_client()
    with ThreadPoolExecutor(8) as ex:
        infos = dict(zip(ckpts, ex.map(lambda p: rc.get_weights_info_by_tinker_path(p).result(),
                                        ckpts.values())))
    specs: list[ModelSpec] = []
    for name, path in ckpts.items():
        info = infos[name]
        specs.append(ModelSpec(key=name, base_model=info.base_model, is_base=False, tinker_path=path,
                               weights_info=info.model_dump(mode="json")))
    bases = list(dict.fromkeys(s.base_model for s in specs))
    cache = state.setdefault("sampler_paths", {})

    def convert(spec: ModelSpec) -> str:
        if "/sampler_weights/" in spec.tinker_path:
            return spec.tinker_path
        if spec.tinker_path in cache:
            return cache[spec.tinker_path]
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", spec.key)[:60]
        log(f"converting {spec.key} to sampler weights (training state -> sampler checkpoint)…")
        tc = sc.create_training_client_from_state(spec.tinker_path)
        path = tc.save_weights_for_sampler(name=f"capeval-{safe}-{int(time.time())}",
                                           ttl_seconds=int(ttl_days * 86400)).result().path
        cache[spec.tinker_path] = path
        return path

    with ThreadPoolExecutor(8) as ex:
        for spec, sp in zip(specs, ex.map(convert, specs)):
            spec.sampler_path = sp
    return [ModelSpec(key=f"base:{b}", base_model=b, is_base=True) for b in bases] + specs


def make_policies(sc, specs: list[ModelSpec], args, sem: PrioritySemaphore, state: dict) -> dict[str, Policy]:
    from transformers import AutoTokenizer

    from monitordecorrelation.rl.renderers import is_tml_policy

    ctx_len = {m.model_name: m.max_context_length for m in sc.get_server_capabilities().supported_models}
    toks = {}
    pol = {}
    sampling = SamplingConfig(args.temperature, args.top_p, args.top_k)
    for s in specs:
        if s.base_model not in ctx_len:
            raise SystemExit(f"{s.base_model}: not in tinker's supported models (no context length)")
        if s.is_base:
            client = sc.create_sampling_client(base_model=s.base_model)
        else:
            try:
                client = sc.create_sampling_client(model_path=s.sampler_path)
            except Exception as e:  # noqa: BLE001 — e.g. an expired cached conversion
                if s.sampler_path == s.tinker_path:
                    raise
                log(f"{s.key}: cached sampler path unusable ({type(e).__name__}); converting again")
                state["sampler_paths"].pop(s.tinker_path, None)
                [s2] = [x for x in resolve_models(sc, {s.key: s.tinker_path}, state, args.sampler_ttl_days)
                        if not x.is_base]
                s.sampler_path = s2.sampler_path
                client = sc.create_sampling_client(model_path=s.sampler_path)
        tok = None
        if not is_tml_policy(s.base_model):
            if s.base_model not in toks:
                toks[s.base_model] = AutoTokenizer.from_pretrained(s.base_model)
            tok = toks[s.base_model]
        pol[s.key] = Policy(key=s.key, base_model=s.base_model, sampling_client=client, sampling=sampling,
                            semaphore=sem, context_length=ctx_len[s.base_model],
                            thinking_effort=args.thinking_effort, hf_tokenizer=tok)
    return pol


# ---------------------------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------------------------
def make_benchmarks(args) -> list[Benchmark]:
    qa = Budget(args.think_budget, args.answer_budget)
    agent = dict(budget=Budget(args.agent_think_budget, args.agent_answer_budget), max_turns=args.max_turns,
                 cmd_timeout=args.cmd_timeout, obs_chars=args.obs_chars)
    make = {
        "mmlu_pro": lambda: MMLUPro(per_category=args.mmlu_pro_per_category, n_samples=args.mmlu_pro_samples,
                                    budget=qa),
        "gpqa_diamond": lambda: GPQADiamond(n_samples=args.gpqa_samples, budget=qa),
        "math500": lambda: Math500(n_samples=args.math500_samples, budget=qa),
        "lcb": lambda: LiveCodeBench(difficulties=tuple(args.lcb_difficulties.split(",")),
                                     test_timeout=args.lcb_test_timeout, n_samples=args.lcb_samples, **agent),
        "exercism": lambda: Exercism(n_samples=args.exercism_samples, **agent),
        "canitedit": lambda: CanItEdit(n_samples=args.canitedit_samples, **agent),
        "coherence": lambda: MTBenchCoherence(n_samples=args.coherence_samples,
                                              budget=Budget(args.chat_think_budget, args.chat_answer_budget),
                                              judge_model=args.judge_model),
    }
    return [make[b]() for b in ALL_BENCHMARKS if b in args.benchmarks]


async def load_items(bench: Benchmark, ctx: RunCtx, args) -> list[Item]:
    items = await asyncio.to_thread(bench.load, args.cache_dir)
    validate = getattr(bench, "validate", None)
    if validate is not None:
        # Keyed on what the grader depends on (dataset version, grader version, sandbox image) — not on
        # scaffold settings like max_turns, which do not change whether a reference solution passes.
        gkey = {k: v for k, v in bench.settings().items() if k in ("grader_version", "commit", "kinds", "file")}
        vhash = hashlib.sha256(json.dumps(gkey, sort_keys=True).encode()).hexdigest()[:12]
        vpath = os.path.join(args.cache_dir, f"validation_{bench.name}_{vhash}_"
                                             f"{ctx.sandbox.image.split(':')[1]}.json")
        if os.path.exists(vpath):
            dropped = json.load(open(vpath))
            bench.dropped.update(dropped)
            items = [it for it in items if it.id not in dropped]
        else:
            log(f"{bench.name}: validating the grader on {len(items)} reference solutions (cached after)…")
            items = await validate(ctx, items)
            with open(vpath, "w") as f:
                json.dump(bench.dropped, f, indent=1)
        if bench.dropped:
            log(f"{bench.name}: {len(bench.dropped)} items dropped by validation (listed in the output)")
    if args.limit is not None and len(items) > args.limit:
        keep = set(random.Random(0).sample(range(len(items)), args.limit))
        items = [it for i, it in enumerate(items) if i in keep]
    return items


# ---------------------------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------------------------
def record_key(r: dict) -> tuple:
    return (r["model"], r["bench"], r["item"], r["sample"])


def load_cache(path: str) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a torn last line from a killed run
                    out[record_key(r)] = r
    return out


async def run_jobs(jobs: list[tuple], benches: dict[str, Benchmark], policies: dict[str, Policy],
                   ctx: RunCtx, writer, results: dict[tuple, dict], rejudge: list[dict]) -> None:
    from tqdm import tqdm

    per_bench: dict[str, int] = {}
    for _, b, _, _ in jobs:
        per_bench[b] = per_bench.get(b, 0) + 1
    for r in rejudge:
        per_bench[r["bench"]] = per_bench.get(r["bench"], 0) + 1
    total = sum(per_bench.values())
    overall = tqdm(total=total, desc="all", position=0, dynamic_ncols=True, smoothing=0.02)
    bars = {b: tqdm(total=n, desc=f"{b:<12}", position=i + 1, dynamic_ncols=True, leave=True, smoothing=0.02)
            for i, (b, n) in enumerate(sorted(per_bench.items(), key=lambda kv: list(benches).index(kv[0])))}
    n_err = 0

    def done(rec: dict) -> None:
        nonlocal n_err
        writer.write(json.dumps(rec, default=str) + "\n")
        writer.flush()
        results[record_key(rec)] = rec
        if rec.get("error"):
            n_err += 1
            overall.set_postfix_str(f"errors={n_err}")
            log(f"ERROR {rec['bench']}/{rec['model']}/{rec['item']}#{rec['sample']}: {rec['error'][:300]}")
        bars[rec["bench"]].update(1)
        overall.update(1)

    async def one(job: tuple) -> None:
        model, bname, item, sample = job
        bench = benches[bname]
        t0 = time.time()
        rec = {"model": model, "bench": bname, "item": item.id, "sample": sample, "cluster": item.cluster,
               "group": item.group, "fingerprint": bench.fingerprint()}
        usage: dict = {}
        token = USAGE.set(usage)  # this task's API requests add their token usage here (policy.add_usage)
        try:
            rec.update(await bench.run(ctx, policies[model], item, sample))
            rec["error"] = None
        except Exception as e:  # noqa: BLE001 — recorded, excluded from scores, retried on --resume
            rec.update(scores={}, diag={}, transcript={}, error=f"{type(e).__name__}: {e}"[:3000],
                       traceback=traceback.format_exc()[-6000:])
        finally:
            USAGE.reset(token)
        rec["usage"] = usage  # what producing THIS line cost (an errored attempt's tokens included)
        rec["elapsed_s"] = round(time.time() - t0, 2)
        rec["finished"] = dt.datetime.now().isoformat(timespec="seconds")
        done(rec)

    async def redo_judge(rec: dict) -> None:
        usage: dict = {}
        token = USAGE.set(usage)
        try:
            await benches[rec["bench"]].judge_record(ctx, rec)
        except Exception as e:  # noqa: BLE001
            rec["diag"]["judge_fail"] = True
            rec["transcript"]["judge_error"] = f"{type(e).__name__}: {e}"[:2000]
        finally:
            USAGE.reset(token)
        # The line this re-judgement writes carries only ITS tokens (the generation's are on the earlier
        # line), so summing every line of the samples file counts each request once.
        rec["usage"] = usage
        rec["rejudged"] = True
        rec["fingerprint"] = benches[rec["bench"]].fingerprint()  # a legacy record is now current
        done(rec)

    try:
        await asyncio.gather(*[one(j) for j in jobs], *[redo_judge(r) for r in rejudge])
    finally:
        for b in bars.values():
            b.close()
        overall.close()


def interleave(lists: list[list]) -> list:
    out = []
    for i in range(max((len(x) for x in lists), default=0)):
        out.extend(x[i] for x in lists if i < len(x))
    return out


# ---------------------------------------------------------------------------------------------
async def amain(args) -> None:
    import tinker

    from monitordecorrelation.capeval.sandbox import SandboxManager

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    stem = args.out[:-5] if args.out.endswith(".json") else args.out
    samples_path, state_path = stem + ".samples.jsonl", stem + ".state.json"
    if not args.resume and os.path.exists(samples_path):
        raise SystemExit(f"{samples_path} exists: pass --resume to continue it, or choose another --out")
    os.makedirs(args.cache_dir, exist_ok=True)
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}

    ckpts = load_checkpoints(args.checkpoints)
    sc = tinker.ServiceClient()
    log(f"resolving {len(ckpts)} checkpoints…")
    specs = await asyncio.to_thread(resolve_models, sc, ckpts, state, args.sampler_ttl_days)
    if args.no_base:
        specs = [s for s in specs if not s.is_base]
    with open(state_path, "w") as f:
        json.dump(state, f, indent=1)
    for s in specs:
        log(f"model {s.key}: base={s.base_model}" + ("" if s.is_base else f" sampler={s.sampler_path}"))

    benches = make_benchmarks(args)
    sandbox = None
    if any(b.family == "agentic" for b in benches):
        sandbox = SandboxManager(n_exec=args.max_exec, max_containers=args.max_containers)
        await asyncio.to_thread(sandbox.prepare)  # image, reap dead runs' containers, size the caps
        for note in sandbox.notes:
            log(f"sandbox: {note}")
        lim = sandbox.limits
        log(f"sandbox: up to {lim['max_containers']} containers / {lim['max_exec']} concurrent commands in this "
            f"run (box-wide caps: {lim['box']['containers']} / {lim['box']['exec']}); each ≤ "
            f"{lim['per_container']['memory_mb']} MB RAM, 1 CPU, no disk")
    judge = Judge(args.judge_model, max_concurrency=args.judge_concurrency) \
        if any(b.family == "judge" for b in benches) else None
    ctx = RunCtx(sandbox=sandbox, judge=judge, cache_dir=args.cache_dir)
    try:
        items = {}
        for b in benches:
            items[b.name] = await load_items(b, ctx, args)
            log(f"benchmark {b.name}: {len(items[b.name])} items × {b.n_samples} samples")

        sem = PrioritySemaphore(args.max_inflight)
        policies = await asyncio.to_thread(make_policies, sc, specs, args, sem, state)
        with open(state_path, "w") as f:
            json.dump(state, f, indent=1)

        cache = load_cache(samples_path) if args.resume else {}
        results: dict[tuple, dict] = {}
        rejudge: list[dict] = []
        per_bench_jobs = []
        n_reused = n_stale = 0
        for b in benches:
            fps = {b.fingerprint()} | b.legacy_fingerprints()
            jobs = []
            for it in items[b.name]:
                for k in range(b.n_samples):
                    for s in specs:
                        r = cache.get((s.key, b.name, it.id, k))
                        if r is not None and r.get("fingerprint") in fps and not r.get("error"):
                            # A failed judgement, or one by another judge: re-judge the saved sample.
                            if b.needs_rejudge(r):
                                rejudge.append(r)
                            else:
                                results[record_key(r)] = r
                            n_reused += 1
                            continue
                        n_stale += r is not None and r.get("fingerprint") not in fps
                        jobs.append((s.key, b.name, it, k))
            per_bench_jobs.append(jobs)
        jobs = interleave(per_bench_jobs)
        if args.resume:
            log(f"resume: reusing {n_reused} samples ({len(rejudge)} to re-judge: judge failed, or another judge "
                f"scored them), {n_stale} stale "
                f"(settings changed), {len(jobs)} to run")
        meta = {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "argv": sys.argv, "args": vars(args),
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "out_file": os.path.abspath(args.out),
            "sandbox_limits": sandbox.limits if sandbox is not None else None,
            "samples_file": os.path.abspath(samples_path),
            "benchmarks": {b.name: {"family": b.family, "metrics": list(b.metrics),
                                    "lower_is_better": sorted(b.lower_is_better), "n_items": len(items[b.name]),
                                    "n_samples": b.n_samples, "settings": b.settings(),
                                    "fingerprint": b.fingerprint(), "dropped": getattr(b, "dropped", {})}
                           for b in benches},
        }
        t0 = time.time()
        with open(samples_path, "a") as writer:
            await run_jobs(jobs, {b.name: b for b in benches}, policies, ctx, writer, results, rejudge)
        meta["wall_clock_s"] = round(time.time() - t0, 1)
    finally:
        if sandbox is not None:
            sandbox.cleanup()

    report = build_report(specs=specs, benches=benches, results=results, meta=meta, deltas=not args.no_base,
                          usage=aggregate_usage(iter_sample_lines(samples_path)))
    blob = json.dumps(report, indent=1, default=str)
    with open(args.out, "w") as f:
        f.write(blob)
    # Rendered from the JSON exactly as written, so `--show <out>` prints the same text later.
    print(render_report(json.loads(blob)))


def main() -> None:
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
    args = parse_args()
    if args.show is not None:
        with open(args.show) as f:
            print(render_report(json.load(f)))
        return
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\ninterrupted — finished samples are saved; rerun with --resume to continue", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
