"""The Inkling probe server: scores TerminalEnv episodes with deployed probes on one GPU, for RL runs, over HTTP.

``experiments/inkling_probe_server.py`` is the command line; monitors/inkling_probe_monitor.py is the client.

**Scoring is exactly the training extraction:**
1. ``InklingReplay`` at the probes' effort, with their follow-up;
2. ``item_messages`` in each probe's view;
3. ``extract_features`` with the same ``max_batch_tokens``;
4. the probe's readout and layer;
5. ``SelectedProbe``.

The score is σ(logit − t₁%), so 0.5 is the probe's 1%-FPR operating point.

**Protocol** (JSON over HTTP, localhost only; Vlad's machine reaches it through an SSH tunnel):
- ``GET /v1/meta`` returns {protocol, reader, effort, follow_up, probes: {sha: {name, dataset, view, readout, layer,
  threshold_1}}}.
- ``POST /v1/score`` takes {protocol, probes: [sha], episodes: [{key, question, turns, env_options}], wait_s}.
  - It returns 200 {results: [{key, n_tokens, conversation_sha256, scores: {sha: {logit, score}}, router_topk}]},
    or 202 {status, position} if not done within ``wait_s``. The client then repeats the same request.
  - A request is identified by the hash of its body, so a repeat joins the same job instead of starting a new one.
  - Errors: 409 for an unknown probe or protocol, 422 for an episode that can't be replayed.
- One GPU worker thread runs the jobs in order; the HTTP threads only queue them and wait.

**Golden check:** each probe directory may hold ``golden.jsonl``: episodes with their stored token hashes and logits.
At startup the server re-scores them, and refuses to start if a rendering differs, a logit moves more than
``golden_tolerance``, or the logits moved on average by more than a tight tolerance (a systematic shift;
``GOLDEN_MEAN_TOLERANCE`` by default). That catches drift in transformers, the kernels, the replay or the reader before
it can look like monitor degradation.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

PROTOCOL = 1

GOLDEN_MEAN_TOLERANCE = 0.3
"""Default bound on |mean Δlogit| over a probe's golden episodes (Δ = served − golden logit), in logits.

**Why a second check:** the max |Δ| check (each probe's ``golden_tolerance``, 2.5–3.4) is there for a wrong reader,
layer or rendering, which throws single episodes far off. A systematic shift passes it: every logit moving ~0.3 (a
different library version or kernel) stays far inside 2.5. Yet a +0.3 shift raises the false-alarm rate at t₁% from 1%
to 1.2–1.7% (the IID probes' honest dev episodes). The mean of the golden Δs exposes such a shift.

**Calibration** (4-bit reader, follow-up probes; the mean probes' golden jitter is several times smaller):
- **No change, only batching.** The golden logits come from batched features, while the server scores one episode per
  forward (or batched with other neighbours). Per-episode Δ then has mean −0.01 to −0.06 and sd 0.3–0.6 logits.
  So the mean over a probe's 32 golden episodes has a standard error of 0.05–0.11.
  - A correct stack exceeds 0.3 with probability ~1.2% per probe at the worst mean (−0.06) and sd (0.6).
  - With each follow-up probe's own mean and sd (the 4-bit diagnostics sample, 228 inputs), the chance that any of
    the four does is ~1%.
- **A real shift.** The 4-bit and full-precision readers differ by +0.16 / −0.36 logits on average.
  - A 0.36 shift is caught with probability 0.5–0.99 per probe over that noise range (more often when it moves
    several probes at once).
  - A shift of ≥ 0.55 is caught with ≥ 95% probability per probe.
  - A 0.16 shift mostly passes: 32 golden logits with batching noise can't resolve it.
- **Tighter costs false refusals.** 0.25 would catch a 0.3 shift more often (per follow-up probe 0.54–0.89 instead of
  0.31–0.69), but a correct stack would fail it ~4× as often (~4% that one of the four follow-up probes does).
- **Later:** once the golden logits are re-derived one episode per forward (the batching rule's later fix), the null
  mean is ~0 and its spread small, so this can be much tighter (measure that spread first).

The servers' ``--golden-mean-tolerance`` overrides it (``inf`` turns the check off). It lives here, not in meta.json:
meta.json is part of the probe's identity (``probe_sha``)."""


class GoldenCheckError(RuntimeError):
    """A golden episode renders or scores differently. ``report`` has the probe's numbers, when they were computed."""

    def __init__(self, message: str, report: dict | None = None):
        super().__init__(message)
        self.report = report or {}


def probe_sha(probe_dir: str) -> str:
    """The probe's identity: sha256 over its probe.npz and meta.json."""
    h = hashlib.sha256()
    for name in ("probe.npz", "meta.json"):
        with open(os.path.join(probe_dir, name), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def conversation_sha(tokens) -> str:
    return hashlib.sha256(np.asarray(tokens, dtype=np.int64).tobytes()).hexdigest()


@dataclass
class Served:
    sha: str
    name: str
    probe: object  # SelectedProbe
    view: str
    readout: str


@dataclass
class Job:
    request: dict
    done: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None
    error: tuple[int, str] | None = None


class ProbeScorer:
    """Loads nothing itself: give it the reader (an Inkling causal LM) and the probe directories."""

    def __init__(self, model, probe_dirs: list[str], *, max_batch_tokens: int = 65536, reader_info: dict | None = None):
        from monitordecorrelation.whitebox.features import special_ids
        from monitordecorrelation.whitebox.inkling_replay import FOLLOW_UP, InklingReplay
        from monitordecorrelation.whitebox.probe_fit import SelectedProbe

        self.model, self.max_batch_tokens, self.reader_info = model, max_batch_tokens, reader_info or {}
        self.probes: dict[str, Served] = {}
        efforts = set()
        for d in probe_dirs:
            p = SelectedProbe.load(d)
            sha = probe_sha(d)
            self.probes[sha] = Served(sha, p.meta.get("name") or os.path.basename(os.path.normpath(d)), p,
                                      p.meta["view"], p.meta["readout"])
            efforts.add(float(p.meta.get("effort", 0.5)))
        if len(efforts) > 1:
            raise ValueError(f"probes rendered at different efforts {sorted(efforts)}: serve them separately")
        self.effort = efforts.pop() if efforts else 0.5
        self.replay = InklingReplay(effort=self.effort, follow_up=FOLLOW_UP)
        self.sp = special_ids(self.replay)

    def meta(self) -> dict:
        return {"protocol": PROTOCOL, "reader": self.reader_info, "effort": self.effort,
                "follow_up": list(self.replay.follow_up),
                "probes": {s.sha: {"name": s.name, "dataset": s.probe.meta.get("dataset"), "view": s.view,
                                   "readout": s.readout, "layer": s.probe.layer, "threshold_1": s.probe.threshold_1}
                           for s in self.probes.values()}}

    def score(self, episodes: list[dict], shas: list[str]) -> list[dict]:
        """Score ``episodes`` (replay items) with the probes ``shas``: one feature pass per view needed."""
        from monitordecorrelation.whitebox.features import extract_features
        from monitordecorrelation.whitebox.probe_datasets import item_messages
        from monitordecorrelation.whitebox.probe_fit import ViewFeatures, feature_matrix

        unknown = [s for s in shas if s not in self.probes]
        if unknown:
            raise LookupError(f"not served here: {unknown}")
        out = [{"key": ep.get("key"), "scores": {}} for ep in episodes]
        for view in sorted({self.probes[s].view for s in shas}):
            inputs = [self.replay.probe_input(item_messages(ep, view)) for ep in episodes]
            feats = extract_features(self.model, inputs, self.sp, max_batch_tokens=self.max_batch_tokens)
            vf = ViewFeatures([str(i) for i in range(len(episodes))], feats, [])
            for i, p in enumerate(inputs):
                out[i].setdefault("n_tokens", {})[view] = len(p.tokens)
                out[i].setdefault("conversation_sha256", {})[view] = conversation_sha(p.tokens)
                if "router_topk" in feats:
                    out[i].setdefault("router_topk", {})[view] = feats["router_topk"][i].sort(-1).values.tolist()
            for s in shas:
                served = self.probes[s]
                if served.view != view:
                    continue
                x, keep = feature_matrix(vf, served.readout, view)
                layer = x[:, served.probe.layer, :].float().numpy().astype(np.float64)
                logit = served.probe.logit(layer)
                score = served.probe.score(layer)
                for i in range(len(episodes)):
                    out[i]["scores"][s] = {"logit": float(logit[i]), "score": float(score[i]) if keep[i] else None}
        return out

    def golden_check(self, probe_dirs: list[str], tolerance: float, *,
                     mean_tolerance: float = GOLDEN_MEAN_TOLERANCE) -> dict:
        """Re-score each probe's golden episodes; raises ``GoldenCheckError`` if a rendering or the logits have moved.

        With Δ = served − golden logit over a probe's golden episodes, it requires:
        - every rendering identical (sha256);
        - max |Δ| ≤ ``tolerance``: no single episode far off;
        - |mean Δ| ≤ ``mean_tolerance``: no systematic shift (``GOLDEN_MEAN_TOLERANCE``);
        - every Δ finite.

        Returns {sha: {n, max_dlogit, mean_dlogit, sd_dlogit, tolerance, mean_tolerance}}. A failure's error carries
        the same numbers for its probe as ``report``."""
        report = {}
        for d in probe_dirs:
            path = os.path.join(d, "golden.jsonl")
            if not os.path.exists(path):
                continue
            sha = probe_sha(d)
            rows = [json.loads(line) for line in open(path)]
            got = self.score([r["episode"] for r in rows], [sha])
            view = self.probes[sha].view
            for r, g in zip(rows, got):
                if g["conversation_sha256"][view] != r["conversation_sha256"]:
                    raise GoldenCheckError(f"golden check: {d}: episode {r['episode'].get('key')} renders differently")
            dl = np.array([g["scores"][sha]["logit"] - r["logit"] for r, g in zip(rows, got)], dtype=np.float64)
            stats = {"n": len(rows), "max_dlogit": float(np.abs(dl).max()) if dl.size else 0.0,
                     "mean_dlogit": float(dl.mean()) if dl.size else 0.0,
                     "sd_dlogit": float(dl.std(ddof=1)) if dl.size > 1 else 0.0,
                     "tolerance": tolerance, "mean_tolerance": mean_tolerance}
            problems = []
            if not np.isfinite(dl).all():
                problems.append("a logit is not finite")
            if stats["max_dlogit"] > tolerance:
                problems.append(f"a logit moved by {stats['max_dlogit']:.4f} > {tolerance}")
            if abs(stats["mean_dlogit"]) > mean_tolerance:
                problems.append(f"the logits shifted by {stats['mean_dlogit']:+.4f} on average, beyond "
                                f"±{mean_tolerance} (sd {stats['sd_dlogit']:.4f})")
            if problems:
                raise GoldenCheckError(f"golden check: {d}: {'; '.join(problems)} over {len(rows)} episodes", stats)
            report[sha] = stats
        return report


class Server:
    """HTTP front for a ``ProbeScorer``: one GPU worker thread; requests are idempotent by body hash."""

    def __init__(self, scorer: ProbeScorer, host: str = "127.0.0.1", port: int = 8178, log=print):
        from http.server import ThreadingHTTPServer

        self.scorer, self.log = scorer, log
        self.jobs: dict[str, Job] = {}
        self.queue: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.httpd = ThreadingHTTPServer((host, port), self._handler())
        self.port = self.httpd.server_address[1]
        self.worker = threading.Thread(target=self._work, name="gpu-worker", daemon=True)

    def start(self) -> None:
        self.worker.start()
        threading.Thread(target=self.httpd.serve_forever, name="http", daemon=True).start()
        self.log(f"probe server on 127.0.0.1:{self.port}: {len(self.scorer.probes)} probes")

    def stop(self) -> None:
        self.httpd.shutdown()
        self.queue.put(None)

    def _work(self) -> None:
        while True:
            key = self.queue.get()
            if key is None:
                return
            job = self.jobs[key]
            t = time.time()
            try:
                results = self.scorer.score(job.request["episodes"], job.request["probes"])
                job.result = {"results": results, "timing": {"forward_s": round(time.time() - t, 3)}}
            except LookupError as e:
                job.error = (409, str(e))
            except (ValueError, KeyError, NotImplementedError, RuntimeError) as e:
                job.error = (422, f"{type(e).__name__}: {e}")
            except Exception as e:  # noqa: BLE001 — a bad batch must not kill the worker
                job.error = (500, f"{type(e).__name__}: {e}")
            job.done.set()

    def submit(self, request: dict) -> tuple[int, dict]:
        if request.get("protocol") != PROTOCOL:
            return 409, {"error": f"protocol {request.get('protocol')!r}; this server speaks {PROTOCOL}"}
        unknown = [s for s in request.get("probes", []) if s not in self.scorer.probes]
        if unknown:
            return 409, {"error": f"probes not served here: {unknown}"}
        body = {k: request[k] for k in ("probes", "episodes")}
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        with self.lock:
            job = self.jobs.get(key)
            if job is None:
                job = self.jobs[key] = Job(body)
                self.queue.put(key)
        if not job.done.wait(float(request.get("wait_s", 20))):
            return 202, {"status": "queued", "position": self.queue.qsize()}
        if job.error:
            return job.error[0], {"error": job.error[1]}
        return 200, job.result

    def _handler(self):
        server = self

        from http.server import BaseHTTPRequestHandler

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code: int, payload: dict) -> None:
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802
                if self.path == "/v1/meta":
                    self._send(200, server.scorer.meta())
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):  # noqa: N802
                if self.path != "/v1/score":
                    self._send(404, {"error": "not found"})
                    return
                try:
                    request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                except (ValueError, json.JSONDecodeError):
                    self._send(400, {"error": "bad JSON"})
                    return
                self._send(*server.submit(request))

            def log_message(self, fmt, *args):  # quiet: the RL runs poll often
                pass

        return Handler
