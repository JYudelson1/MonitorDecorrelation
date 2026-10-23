"""Send synthetic TerminalEnv episodes of given lengths to a running Inkling probe server, one at a time, and report
what each got: its score, an error status (a 503: e.g. the GPU out of memory), or no answer in time.

For the RL batch's rehearsal (scripts/rl_batch.py). A valid RL episode can be much longer than any golden one (up to 4
turns of 3,072 sampled tokens, plus the commands' output; Vlad's 72,145 valid eval episodes reached 7,541 tokens in
the with-reasoning view, none above 8k), and the reader's dense attention bias grows with the square of the length
(whitebox/features.py: ~340 B × T² per episode, ~49 GB at 12k tokens, ~87 GB at 16k). On a 4× B200 copy of the
full-precision reader, ~55 GB per GPU is left beside the weights, so an episode past ~12k tokens may not fit. The
server answers such a failure with 503, which the RL loop's client retries forever: this check says where that starts.

    .venv/bin/python experiments/check_probe_long_episodes.py --url http://127.0.0.1:8178 \\
        --probe data/probes/Inkling-Small/v2/terminal_prerl_v2/with_reasoning/followup --tokens 8000,12000,16000 \\
        --out results/long_episodes.json

- **The episodes:** the probe's first golden episode, its first command's output padded with rows of 0/1 cells (as the
  env prints grids) until the probe's view renders to the length asked, as the server renders it (``InklingReplay``).
- **No retries:** each request is sent once and asked again only while the server says it is still running (202), up
  to ``--timeout``; a 503 is the answer.
- After each, ``/v1/meta`` says whether the server still answers.

Exit status 0 if every request was answered (scores or an error status) and the server still answered after the
last; 1 otherwise. The report: per length, the tokens rendered, the HTTP status, the seconds and the score or error.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

FILLER = "0 1 1 0 1 0 0 1 1 0\n"


def post(url: str, body: dict, timeout: float) -> tuple[int, dict | str]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:2000]


def alive(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/v1/meta", timeout=30) as resp:
            return resp.status == 200
    except (OSError, ValueError):
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8178")
    ap.add_argument("--probe", required=True, help="a probe folder the server serves (its sha is sent)")
    ap.add_argument("--tokens", default="8000,12000,16000", help="comma-list of lengths, in the probe's view")
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds per episode before it counts as unanswered")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from monitordecorrelation.whitebox.inkling_replay import FOLLOW_UP, InklingReplay
    from monitordecorrelation.whitebox.probe_datasets import item_messages
    from monitordecorrelation.whitebox.probe_server import PROTOCOL, item_sha, probe_sha

    meta = json.loads(open(f"{args.probe}/meta.json").read())
    with open(f"{args.probe}/golden.jsonl") as fh:
        golden = json.loads(fh.readline())["episode"]
    replay = InklingReplay(effort=float(meta.get("effort", 0.5)), follow_up=FOLLOW_UP)
    view, sha = meta["view"], probe_sha(args.probe)
    base = {k: golden[k] for k in ("question", "turns", "env_options")}
    # a replay turn (probe_datasets.REPLAY_TURN_KEYS) ran a command if it has an output; else pad the first thinking
    turn = next((i for i, t in enumerate(base["turns"]) if t.get("output") is not None), None)
    field, turn = ("output", turn) if turn is not None else ("cot", 0)

    def build(rows: int) -> dict:
        item = json.loads(json.dumps(base))
        item["turns"][turn][field] = (item["turns"][turn].get(field) or "") + FILLER * rows
        return item

    def length(item: dict) -> int:
        return len(replay.probe_input(item_messages(item, view)).tokens)

    report = {"probe": args.probe, "sha": sha, "view": view, "url": args.url, "results": []}
    ok = True
    for target in [int(t) for t in args.tokens.split(",") if t]:
        lo, hi = 0, 1
        while length(build(hi)) < target:
            lo, hi = hi, hi * 2
        while hi - lo > 1:  # the fewest filler rows that reach the target
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if length(build(mid)) < target else (lo, mid)
        item = build(hi)
        item["key"] = item_sha(item)[:16]
        n = length({k: item[k] for k in ("question", "turns", "env_options")})
        body = {"protocol": PROTOCOL, "probes": [sha], "episodes": [item], "wait_s": 60}
        t0, row = time.time(), {"target_tokens": target, "tokens": n}
        while True:
            try:
                status, answer = post(f"{args.url}/v1/score", body, 120)
            except OSError as e:
                status, answer = None, f"{type(e).__name__}: {e}"
            if status == 202 and time.time() - t0 < args.timeout:
                continue  # still running: the same body joins the same job
            break
        row.update(status=status, seconds=round(time.time() - t0, 1))
        if status == 200:
            res = answer["results"][0]
            row.update(score=res["scores"][sha]["score"], logit=res["scores"][sha]["logit"],
                       n_tokens=res.get("n_tokens"), forward_s=(answer.get("timing") or {}).get("forward_s"))
        else:
            row["error"] = answer if isinstance(answer, str) else json.dumps(answer)[:2000]
            ok &= status is not None and status != 202
        row["server_alive_after"] = alive(args.url)
        ok &= row["server_alive_after"]
        report["results"].append(row)
        print(json.dumps(row)[:600], flush=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)
    report["ok"] = ok
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
