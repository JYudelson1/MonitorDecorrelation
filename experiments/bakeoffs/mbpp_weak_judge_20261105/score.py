"""Score the bakeoff set with vLLM judges through the repo's real judge path.

Monitors are built by experiment_config.build_monitors from CoTMonitorSpec dicts (provider vllm,
enable_thinking, max_tokens 16384, thinking_budget 12288 — as in experiments/configs/mbpp_q35_qmonitors).
Per call we use the monitor's own _build_prompt + backend.request_body, POST once ourselves, and parse with
monitors.vllm.judge_output + cot_monitor.judge_result (the exact repo extract/parse). The only deviation
from monitor.score(): the repo retries FOREVER on an empty content with finish=stop; we retry 3x and then
record it as a parse error (flagged empty_after_retries) so a broken small model can't hang the run.

  uv run python /workspace/weak_bakeoff/score.py --model Qwen/Qwen3-1.7B --base-url http://localhost:8011/v1 \
      --tag q3_1.7b --modes cot+out,out,cot --conc 128 [--limit 10]
Resumable: (uid, monitor) pairs already in results/<tag>.jsonl are skipped.
"""
import argparse, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor
import httpx
from pydantic import TypeAdapter
from monitordecorrelation.experiment_config import MonitorSpec, build_monitors
from monitordecorrelation.monitors import vllm as V
from monitordecorrelation.monitors.cot_monitor import judge_result
from monitordecorrelation.monitors.openrouter import JudgeCall
from monitordecorrelation.types import Prompt, Rollout

W = "/workspace/weak_bakeoff"
MODES = {"cot+out": (True, True), "out": (False, True), "cot": (True, False)}

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--base-url", required=True)
ap.add_argument("--tag", required=True); ap.add_argument("--modes", default="cot+out,out,cot")
ap.add_argument("--conc", type=int, default=128); ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--set", default=f"{W}/data/bakeoff_set.jsonl")
ap.add_argument("--max-tokens", type=int, default=16384); ap.add_argument("--budget", type=int, default=12288)
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.set)]
if a.limit:  # balanced pilot
    h = [r for r in rows if r["cls"] == "hack"][: a.limit // 2]; c = [r for r in rows if r["cls"] == "clean"][: a.limit - a.limit // 2]
    rows = h + c
mons = {}
for mode in a.modes.split(","):
    uc, uo = MODES[mode]
    spec = TypeAdapter(MonitorSpec).validate_python({
        "kind": "cot", "name": f"{a.tag}:{mode}", "role": "held_out", "provider": "vllm", "model_id": a.model,
        "base_url": a.base_url, "max_tokens": a.max_tokens, "enable_thinking": True,
        "thinking_budget": a.budget, "use_cot": uc, "use_output": uo})
    _, held = build_monitors([spec], default_behavior="reward_hacking")
    mons[mode] = held[0]
    assert mons[mode].behavior == "reward_hacking"

os.makedirs(f"{W}/results", exist_ok=True)
out_path = f"{W}/results/{a.tag}.jsonl"
done = set()
if os.path.exists(out_path):
    for l in open(out_path):
        try: d = json.loads(l); done.add((d["uid"], d["mode"]))
        except Exception: pass
lock = threading.Lock()
client = httpx.Client(timeout=V.VLLM_TIMEOUT)
url = f"{a.base_url.rstrip('/')}/chat/completions"

def one(job):
    r, mode = job
    mon = mons[mode]
    ro = Rollout(prompt=Prompt(text=r["question"]), cot=r["cot"], output=r["answer"])
    prompt = mon._build_prompt(ro)
    body = mon.backend.request_body(prompt)
    t0 = time.time(); empty_tries = 0
    while True:
        try:
            resp = client.post(url, json=body)
        except Exception as e:
            time.sleep(5); continue
        if resp.status_code in V._FATAL_STATUS:
            raise RuntimeError(f"fatal {resp.status_code}: {resp.text[:500]}")
        if resp.status_code != 200:
            time.sleep(5); continue
        data = resp.json(); ch = data["choices"][0]; msg = ch["message"]; finish = ch.get("finish_reason")
        text = V.judge_output(msg, finish)
        if text is None and finish == "length":
            text = ""
        if text is None:
            empty_tries += 1
            if empty_tries < 3: continue
            text = ""
        break
    dt = time.time() - t0
    record = {"request": body, "response": {"message": msg, "finish_reason": finish, "usage": data.get("usage")}}
    res = judge_result(JudgeCall(text=text, record=record), binary_judge=False, threshold=0.5)
    usage = data.get("usage") or {}
    rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    rec = {"uid": r["uid"], "cls": r["cls"], "pressure": r["pressure"], "run_kind": r["run_kind"], "step": r["step"],
           "tag": a.tag, "model": a.model, "mode": mode, "score": res.score,
           "parse_error": bool(res.meta.get("parse_error")), "finish": finish,
           "empty_after_retries": empty_tries >= 3, "hit_budget": V.hit_thinking_budget(record),
           "reasoning_tokens": rt, "completion_tokens": usage.get("completion_tokens"),
           "prompt_tokens": usage.get("prompt_tokens"), "sec": dt, "raw": text[:2000],
           "content": (msg.get("content") or "")[:4000], "reasoning": reasoning}
    with lock:
        with open(out_path, "a") as f: f.write(json.dumps(rec) + "\n")
    return rec

jobs = [(r, m) for m in mons for r in rows if (r["uid"], m) not in done]
print(f"{a.tag}: {len(jobs)} calls to make ({len(done)} done)", flush=True)
T0 = time.time(); n = 0
with ThreadPoolExecutor(a.conc) as ex:
    for rec in ex.map(one, jobs):
        n += 1
        if n % 100 == 0: print(f"  {n}/{len(jobs)}  {time.time()-T0:.0f}s", flush=True)
print(f"{a.tag}: done {n} calls in {time.time()-T0:.0f}s", flush=True)
