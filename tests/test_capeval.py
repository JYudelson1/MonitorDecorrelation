"""Offline tests for the capability-eval harness (``monitordecorrelation.capeval``): answer extraction,
the agent scaffold's parsing, the judge-reply parser, the priority semaphore, and the paired statistics.
The sandbox graders and the tinker sampling path are exercised live (see STATUS.md)."""

from __future__ import annotations

import asyncio

import numpy as np

from monitordecorrelation.capeval.agentic import extract_commands, format_observation
from monitordecorrelation.capeval.coherence import parse_judgement, repetition
from monitordecorrelation.capeval.policy import PrioritySemaphore
from monitordecorrelation.capeval.qa import extract_letter
from monitordecorrelation.capeval.sandbox import ExecResult
from monitordecorrelation.capeval import stats


def test_extract_letter_takes_the_last_final_answer():
    assert extract_letter("The answer is (B).", "ABCD") == "B"
    assert extract_letter("maybe the answer is A ... The answer is (C)", "ABCD") == "C"
    assert extract_letter("**Answer: D**", "ABCD") == "D"
    assert extract_letter("Answer: $\\boxed{C}$", "ABCD") == "C"
    assert extract_letter("The answer is **(J)**", "ABCDEFGHIJ") == "J"
    assert extract_letter("I think A is right", "ABCD") is None
    assert extract_letter("The answer is (H)", "ABCD") is None  # not a valid option here


def test_extract_commands_needs_bash_fences():
    assert extract_commands("ok\n```bash\nls -la\n```\n") == ["ls -la"]
    assert extract_commands("```bash\nsubmit\n```") == ["submit"]
    assert extract_commands("```python\nprint(1)\n```") == []
    assert len(extract_commands("```bash\na\n```\n```sh\nb\n```")) == 2
    assert extract_commands("<command>ls</command>") == []
    assert extract_commands("```bash\nsubmit\n```\n```bash\nls\n```")[0] == "submit"


def test_format_observation_elides_the_middle_and_flags_timeouts():
    res = ExecResult(output="x" * 30000, returncode=0, timed_out=False, n_bytes=30000)
    obs = format_observation(res, 60, 1000)
    assert "characters elided" in obs and len(obs) < 1500
    res = ExecResult(output="partial", returncode=124, timed_out=True, n_bytes=7)
    assert "did not finish within 60 seconds" in format_observation(res, 60, 1000)


def test_parse_judgement_validates_ranges():
    ok = parse_judgement('blah {"coherence": 8, "consistency": 9, "rl_fried": 2, "fried_signs": [], '
                         '"rationale": "fine"} trailing')
    assert ok == {"coherence": 8.0, "consistency": 9.0, "rl_fried": 2.0, "fried_signs": [], "rationale": "fine"}
    assert parse_judgement('{"coherence": 11, "consistency": 9, "rl_fried": 2}') is None
    assert parse_judgement('{"coherence": true, "consistency": 9, "rl_fried": 2}') is None
    assert parse_judgement("no json here") is None


def test_repetition():
    assert repetition("a b c d e f g h") == 0.0
    assert repetition("a b c d " * 10) > 0.8


def test_rejudging_a_saved_record_shows_the_judge_exactly_what_fresh_judging_did():
    from monitordecorrelation.capeval.bench import clip_middle
    from monitordecorrelation.capeval.coherence import render_conversation

    rng = np.random.default_rng(0)
    for n in (0, 10, 3999, 4000, 4001, 4500, 4501, 20000):
        thinking = "".join(rng.choice(list("abc \n"), n))
        fresh = [{"user": "q", "thinking": thinking, "reply": "r" * 12000}]
        stored = [{"user": "q", "thinking": thinking, "thinking_chars": n, "reply": "r" * 12000}]
        # Records written before 2026-11-05 kept only a 3000 + 1500-char clip of the thinking.
        legacy = [{**stored[0], "thinking": clip_middle(thinking, 3000, 1500)}]
        assert render_conversation(stored) == render_conversation(legacy) == render_conversation(fresh), n


def _coh_bench(judge_model):
    from monitordecorrelation.capeval.coherence import MTBenchCoherence
    from monitordecorrelation.capeval.policy import Budget

    return MTBenchCoherence(n_samples=3, budget=Budget(16384, 4096), judge_model=judge_model)


def test_coherence_identity_excludes_the_judge_and_old_records_are_recognized():
    import hashlib
    import json

    a, b = _coh_bench("google/gemini-4-argon"), _coh_bench("x-ai/grok-4.7")
    assert a.fingerprint() == b.fingerprint()  # the judge does not change what a sample is
    assert a.settings()["judge_model"] == "google/gemini-4-argon"  # but it is recorded
    # The fingerprint the previous code gave these samples (judge hashed in; Sonnet 5.5, prompt v1).
    old = hashlib.sha256(json.dumps({"name": "coherence", "n_samples": 3, "think_budget": 16384,
                                     "answer_budget": 4096, "judge_model": "anthropic/claude-sonnet-5.5",
                                     "judge_prompt_version": 1}, sort_keys=True).encode()).hexdigest()[:16]
    assert old in a.legacy_fingerprints() and old != a.fingerprint()
    legacy = {"diag": {"judge_fail": False}, "transcript": {"judge": {"verdict": {}}}}  # no judge name
    assert a.needs_rejudge(legacy)  # scored by Sonnet -> re-judge
    assert not _coh_bench("anthropic/claude-sonnet-5.5").needs_rejudge(legacy)
    mine = {"diag": {"judge_fail": False}, "transcript": {"judge": {"model": "google/gemini-4-argon",
                                                                    "prompt_version": 1}}}
    assert not a.needs_rejudge(mine) and b.needs_rejudge(mine)
    assert a.needs_rejudge({**mine, "diag": {"judge_fail": True}})


def test_rejudging_drops_the_previous_judges_scores():
    from monitordecorrelation.capeval.bench import RunCtx

    class FailingJudge:
        model = "google/gemini-4-argon"

        async def judge(self, conversation):
            return None, {"fail_reason": "content_filter"}

    rec = {"scores": {"repetition": 0.1, "coherence": 9.0, "consistency": 9.0, "rl_fried": 1.0},
           "diag": {"judge_fail": False},
           "transcript": {"conversation": [{"user": "q", "thinking": "t", "thinking_chars": 1, "reply": "r"}],
                          "judge": {"verdict": {"coherence": 9.0}}}}
    ctx = RunCtx(sandbox=None, judge=FailingJudge(), cache_dir="")
    asyncio.run(_coh_bench("google/gemini-4-argon").judge_record(ctx, rec))
    assert rec["scores"] == {"repetition": 0.1}
    assert rec["diag"]["judge_fail"] and rec["transcript"]["judge"]["model"] == "google/gemini-4-argon"


# ---------------------------------------------------------------------------------------------
# Transcripts are complete: nothing clipped, and output the parser cannot represent is kept raw
# ---------------------------------------------------------------------------------------------
def _scripted_inkling(completions: list[list[int]]):
    """A real Inkling ``Policy`` (TML rendering + parsing) whose sampler returns ``completions`` in order."""
    from types import SimpleNamespace

    from monitordecorrelation.capeval.policy import Policy

    pol = Policy(key="base:thinkingmachines/Inkling-Small", base_model="thinkingmachines/Inkling-Small",
                 sampling_client=None, sampling=None, semaphore=None, context_length=65536, thinking_effort=0.5)
    queue = list(completions)

    async def sample(tokens, max_tokens, seed, priority):
        return SimpleNamespace(tokens=queue.pop(0), stop_reason="stop")

    pol._sample = sample
    return pol


def _tml_turn(tok, thinking: str, text: str | None = None, call: str | None = None) -> list[int]:
    """Inkling's token framing for a turn: thinking, then a text message and/or a native bash tool call."""
    import json

    sp, o = tok.encode_special, (lambda s: list(tok.encode_ordinary(s)))
    ids = [sp("message_model"), sp("content_thinking"), *o(thinking), sp("end_message")]
    if text is not None:
        ids += [sp("message_model"), sp("content_text"), *o(text), sp("end_message")]
    if call is not None:
        ids += [sp("message_model"), *o("bash"), sp("content_invoke_tool_json"),
                *o(json.dumps({"name": "bash", "args": {"command": call}})), sp("end_message")]
    return ids + [sp("content_model_end_sampling")]


def test_agent_transcript_is_complete_and_keeps_a_native_tool_call():
    from contextlib import asynccontextmanager

    from monitordecorrelation.capeval.agentic import FORMAT_ERROR, agent_diag, run_agent
    from monitordecorrelation.capeval.bench import RunCtx
    from monitordecorrelation.capeval.policy import Budget

    long_thought = " ".join(f"step{i}" for i in range(4000))  # ~28k chars: the old store kept 1.2k
    tok = _scripted_inkling([])._tml.tokenizer
    pol = _scripted_inkling([
        _tml_turn(tok, "look first", call="cat /work/main.py"),          # native call: no text at all
        _tml_turn(tok, long_thought, text="```bash\nwrite\n```"),
        _tml_turn(tok, "done", text="```bash\nsubmit\n```"),
    ])
    out = "y" * 30000

    class Box:
        files: dict = {}

        async def exec(self, cmd, timeout):
            self.files["solution.py"] = "x = 1\n"
            return ExecResult(output=out, returncode=0, timed_out=False, n_bytes=len(out))

        async def read(self, name):
            return self.files.get(name)

    class Sandbox:
        @asynccontextmanager
        async def container(self, files):
            yield Box()

    ep = asyncio.run(run_agent(RunCtx(sandbox=Sandbox()), pol, seed_parts=("bx", "i", 0), task="t",
                               graded_note="", files={}, graded_files=["solution.py"],
                               budget=Budget(16384, 4096), max_turns=30, cmd_timeout=60, obs_chars=8000))
    t0, t1, t2 = ep["turns"]
    assert t0["format_error"] and t0["answer"] == "" and t0["cot"] == "look first"
    assert '"command": "cat /work/main.py"' in t0["raw"] and "<|content_invoke_tool_json|>" in t0["raw"]
    assert t0["observation"] == FORMAT_ERROR.format(cut="")
    assert t1["cot"] == long_thought and "raw" not in t1
    assert t1["observation"] == format_observation(ExecResult(out, 0, False, len(out)), 60, 8000)
    assert len(t1["observation"]) > 8000  # what the model saw, unclipped (the old store kept 3k)
    assert t2.get("submit") and ep["end_reason"] == "submitted"
    d = agent_diag(ep)
    assert (d["format_errors"], d["n_unparsed"], d["n_gen"]) == (1, 1, 3)
    rate = stats.diagnostics([{"diag": d}, {"diag": {**d, "format_errors": 0}}, {"diag": {**d, "format_errors": 3}}])
    assert rate["format_error_ep_rate"] == 2 / 3


def test_qa_transcript_keeps_the_whole_thinking_and_raw():
    from monitordecorrelation.capeval.bench import Item, RunCtx
    from monitordecorrelation.capeval.policy import Budget
    from monitordecorrelation.capeval.qa import _SingleTurn

    class QA(_SingleTurn):
        name = "qx"

        def prompt(self, item):
            return "q"

        def grade(self, item, answer):
            return 0.0, True, None

    long_thought = " ".join(f"step{i}" for i in range(4000))
    tok = _scripted_inkling([])._tml.tokenizer
    pol = _scripted_inkling([_tml_turn(tok, long_thought, text="Let me compute.", call="python -c 1"),
                             _tml_turn(tok, "short", text="The answer is (A).")])
    item = Item(id="i", cluster="i", group="g", data={"gold": "A"})
    a = asyncio.run(QA(n_samples=1, budget=Budget(36864, 2048)).run(RunCtx(), pol, item, 0))
    assert a["transcript"]["cot"] == long_thought and a["transcript"]["answer"] == "Let me compute."
    assert "python -c 1" in a["transcript"]["raw"] and a["diag"]["n_unparsed"] == 1
    b = asyncio.run(QA(n_samples=1, budget=Budget(36864, 2048)).run(RunCtx(), pol, item, 1))
    assert "raw" not in b["transcript"] and b["diag"]["n_unparsed"] == 0


def test_coherence_record_keeps_the_whole_thinking_and_raw():
    from monitordecorrelation.capeval.bench import Item, RunCtx

    class Judge:
        model = "google/gemini-4-argon"

        async def judge(self, conversation):
            return {"coherence": 7.0, "consistency": 8.0, "rl_fried": 2.0}, {}

    long_thought = " ".join(f"step{i}" for i in range(4000))
    tok = _scripted_inkling([])._tml.tokenizer
    pol = _scripted_inkling([_tml_turn(tok, long_thought, text="Hi."), _tml_turn(tok, "t2", call="ls")])
    item = Item(id="i", cluster="i", group="g", data={"turns": ["q1", "q2"]})
    rec = asyncio.run(_coh_bench("google/gemini-4-argon").run(RunCtx(judge=Judge()), pol, item, 0))
    c1, c2 = rec["transcript"]["conversation"]
    assert c1["thinking"] == long_thought and c1["thinking_chars"] == len(long_thought) and "raw" not in c1
    assert c2["reply"] == "" and '"command": "ls"' in c2["raw"] and rec["diag"]["n_unparsed"] == 1
    assert rec["scores"]["coherence"] == 7.0


def test_priority_semaphore_serves_low_priority_first():
    async def main():
        sem = PrioritySemaphore(1)
        order = []
        await sem.acquire(1)  # hold the only permit

        async def waiter(tag, prio):
            async with sem.slot(prio):
                order.append(tag)

        tasks = [asyncio.create_task(waiter("single-a", 1)), asyncio.create_task(waiter("single-b", 1))]
        await asyncio.sleep(0)
        tasks.append(asyncio.create_task(waiter("agent", 0)))
        await asyncio.sleep(0)
        sem.release()
        await asyncio.gather(*tasks)
        return order

    assert asyncio.run(main()) == ["agent", "single-a", "single-b"]


def test_priority_semaphore_survives_cancelled_waiters():
    async def main():
        sem = PrioritySemaphore(1)
        await sem.acquire(0)
        t = asyncio.create_task(sem.acquire(0))
        await asyncio.sleep(0)
        t.cancel()
        await asyncio.sleep(0)
        sem.release()  # the cancelled waiter must not swallow the permit
        await asyncio.wait_for(sem.acquire(0), 1)

    asyncio.run(main())


def _recs(model, scores, clusters=None):
    clusters = clusters or [f"c{i}" for i in range(len(scores))]
    return [{"model": model, "item": f"i{i}", "sample": 0, "cluster": clusters[i], "scores": {"pass": s},
             "error": None} for i, s in enumerate(scores)]


def test_paired_delta_is_paired_and_detects_a_real_shift():
    rng = np.random.default_rng(0)
    base = rng.random(400)
    ck = np.clip(base - 0.1 + 0.01 * rng.standard_normal(400), 0, 1)
    d = stats.paired_delta(_recs("ck", ck), _recs("base", base), "pass")
    assert abs(d["mean"] - (ck - base).mean()) < 1e-12
    assert d["significant"] and d["ci_hi"] < 0
    # the paired CI is far tighter than the spread of either model's items
    s = stats.score_summary(_recs("base", base), "pass")
    assert (d["ci_hi"] - d["ci_lo"]) < 0.2 * (s["ci_hi"] - s["ci_lo"])


def test_paired_delta_drops_unpaired_and_errored_items():
    a = _recs("ck", [1, 1, 0, 1])
    b = _recs("base", [0, 1, 0])
    a[3]["error"] = "boom"
    d = stats.paired_delta(a, b, "pass")
    assert d["n_items"] == 3 and abs(d["mean"] - 1 / 3) < 1e-12


def test_cluster_bootstrap_counts_clusters_not_items():
    s = stats.score_summary(_recs("m", [1, 1, 0, 0], clusters=["a", "a", "b", "b"]), "pass")
    assert s["n_items"] == 4 and s["n_clusters"] == 2 and s["mean"] == 0.5


# ---------------------------------------------------------------------------------------------
# Token usage + report round trip
# ---------------------------------------------------------------------------------------------
def test_usage_is_per_sample_task_and_reaches_worker_threads():
    from monitordecorrelation.capeval.policy import USAGE, add_usage

    async def sample(i):
        usage = {}
        token = USAGE.set(usage)
        try:
            add_usage("tinker:M", input_tokens=100, cache_hit=40, output=7)
            await asyncio.sleep(0)  # interleave with the other sample
            # a judge call runs in a worker thread: it must still see THIS sample's dict
            await asyncio.to_thread(add_usage, "openrouter:J", input_tokens=10 * i, cache_hit=0, output=1,
                                    cost_usd=0.5)
        finally:
            USAGE.reset(token)
        return usage

    async def main():
        return await asyncio.gather(sample(1), sample(2))

    u1, u2 = asyncio.run(main())
    assert u1["tinker:M"] == {"requests": 1, "input_miss": 60, "input_hit": 40, "output": 7, "cost_usd": 0}
    assert u1["openrouter:J"]["input_miss"] == 10 and u2["openrouter:J"]["input_miss"] == 20
    add_usage("tinker:M", input_tokens=5, cache_hit=0, output=1)  # outside a sample: a no-op, no error


def test_aggregate_usage_sums_every_line_by_account_model_and_benchmark():
    from monitordecorrelation.capeval.report import aggregate_usage

    u = lambda i, h, o: {"requests": 1, "input_miss": i, "input_hit": h, "output": o, "cost_usd": 0}  # noqa: E731
    lines = [
        {"model": "base:Q", "bench": "a", "usage": {"tinker:Q": u(10, 0, 5)}},
        {"model": "ck", "bench": "a", "usage": {"tinker:Q": u(1, 9, 2), "openrouter:J": u(3, 0, 1)}},
        {"model": "ck", "bench": "b"},  # written before usage was tracked
    ]
    agg = aggregate_usage(lines)
    assert agg["by_account"]["tinker:Q"] == {"requests": 2, "input_miss": 11, "input_hit": 9, "output": 7,
                                             "cost_usd": 0}
    assert agg["by_evaluated_model"]["ck"]["openrouter:J"]["input_miss"] == 3
    assert agg["n_lines_without_usage"] == 1 and agg["n_sample_lines"] == 3


def test_report_renders_identically_from_the_saved_file(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    from monitordecorrelation.capeval.bench import Benchmark
    from monitordecorrelation.capeval.policy import ModelSpec
    from monitordecorrelation.capeval.report import aggregate_usage, build_report, render_report

    class B(Benchmark):
        name, family, metrics, lower_is_better = "bx", "judge", ("coherence", "rl_fried"), frozenset({"rl_fried"})
        dropped = {}

    specs = [ModelSpec(key="base:Q", base_model="Q", is_base=True),
             ModelSpec(key="ck", base_model="Q", is_base=False, tinker_path="tinker://x/weights/y")]
    rng = np.random.default_rng(1)
    results, lines = {}, []
    for m in ("base:Q", "ck"):
        for i in range(20):
            r = {"model": m, "bench": "bx", "item": f"i{i}", "sample": 0, "cluster": f"i{i}", "group": "g",
                 "error": None, "scores": {"coherence": float(rng.integers(1, 11)), "rl_fried": 2.0},
                 "diag": {"n_gen": 2, "n_forced": int(i == 0), "n_truncated": 0, "gen_tokens": 100},
                 "usage": {"tinker:Q": {"requests": 2, "input_miss": 50, "input_hit": 25, "output": 100,
                                        "cost_usd": 0}}}
            results[(m, "bx", r["item"], 0)] = r
            lines.append(r)
    meta = {"benchmarks": {"bx": {"metrics": ["coherence", "rl_fried"], "lower_is_better": ["rl_fried"]}},
            "wall_clock_s": 60.0, "out_file": "o.json", "samples_file": "o.samples.jsonl"}
    rep = build_report(specs=specs, benches=[B()], results=results, usage=aggregate_usage(lines), meta=meta)
    text = render_report(rep)
    assert "Token usage, millions" in text and "Q (base + 1 checkpoint)" in text
    assert "0.00" in text
    # Checkpoint − base table: one row per checkpoint, one column per benchmark:metric.
    delta_tab = text.split("=== Checkpoint − base")[1].split("===", 2)[1].split("\n===")[0].strip().splitlines()
    assert delta_tab[0].split()[:4] == ["checkpoint", "base", "bx:coherence", "bx:rl_fried↓"]
    assert delta_tab[-1].split()[:2] == ["ck", "Q"]
    path = tmp_path / "rep.json"
    path.write_text(json.dumps(rep, default=str))
    assert render_report(json.loads(path.read_text())) == text
    script = Path(__file__).resolve().parents[1] / "experiments" / "eval_checkpoints.py"
    shown = subprocess.run([sys.executable, str(script), "--show", str(path)], capture_output=True, text=True,
                           check=True).stdout
    assert shown == text + "\n"


# ---------------------------------------------------------------------------------------------
# Sandbox capacity: box-wide budget, cross-process semaphore, orphan detection
# ---------------------------------------------------------------------------------------------
def _fake_box(monkeypatch, total_mb, avail_mb, cpus):
    from monitordecorrelation.capeval import sandbox as sb

    monkeypatch.setattr(sb, "_meminfo_mb", lambda key: {"MemTotal": total_mb, "MemAvailable": avail_mb}[key])
    monkeypatch.setattr(sb, "_cpu_count", lambda: cpus)
    return sb


def test_box_budget_keeps_the_worst_case_inside_ram(monkeypatch):
    sb = _fake_box(monkeypatch, total_mb=100_000, avail_mb=100_000, cpus=16)
    box = sb.box_limits()
    assert box["exec"] == 14
    worst = sb.HOST_RESERVE_MB + box["exec"] * sb.MEM_LIMIT_MB + box["containers"] * sb.IDLE_FOOTPRINT_MB
    assert worst <= 100_000 < worst + sb.IDLE_FOOTPRINT_MB  # as many containers as fit, not one more
    containers, exec_n, notes = sb.plan_limits(10_000, 64)
    assert (containers, exec_n) == (box["containers"], 14) and len(notes) == 2


def test_plan_limits_shrinks_to_free_memory_and_other_runs(monkeypatch):
    sb = _fake_box(monkeypatch, total_mb=100_000, avail_mb=40_000, cpus=16)
    containers, exec_n, notes = sb.plan_limits(400, 8, other_live_containers=50)
    room = (40_000 - sb.HOST_RESERVE_MB - 8 * sb.MEM_LIMIT_MB - 50 * sb.IDLE_FOOTPRINT_MB) // sb.IDLE_FOOTPRINT_MB
    assert containers == room < 400 and exec_n == 8 and any("free now" in n for n in notes)
    sb = _fake_box(monkeypatch, total_mb=12_000, avail_mb=12_000, cpus=4)
    import pytest

    with pytest.raises(RuntimeError, match="not enough RAM"):
        sb.plan_limits(10, 2)


def test_box_semaphore_is_shared_across_instances(monkeypatch, tmp_path):
    monkeypatch.setenv("MD_GLOBAL_SEM_DIR", str(tmp_path))
    from monitordecorrelation.capeval.sandbox import BoxSemaphore

    async def main():
        a, b = BoxSemaphore("t", 2), BoxSemaphore("t", 2)  # as if in two processes
        fa1, fb1 = await a.acquire(), await b.acquire()
        third = asyncio.create_task(a.acquire())
        await asyncio.sleep(0.2)
        assert not third.done()  # both permits held, across "processes"
        BoxSemaphore.release(fb1)
        fd = await asyncio.wait_for(third, 5)
        BoxSemaphore.release(fd)
        BoxSemaphore.release(fa1)

    asyncio.run(main())


def test_owner_liveness():
    import os
    import socket

    from monitordecorrelation.capeval import sandbox as sb

    assert sb._owner_alive(sb.OWNER)
    me_host, me_pid, start = sb.OWNER.rsplit(":", 2)
    assert not sb._owner_alive(f"{me_host}:{me_pid}:{int(start) + 1}")  # pid reused by another process
    assert not sb._owner_alive(f"{socket.gethostname()}:999999999:1")
    assert sb._owner_alive(f"some-other-host:{os.getpid()}:1")  # not ours to judge
    assert not sb._owner_alive("")
