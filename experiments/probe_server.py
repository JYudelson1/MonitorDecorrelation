"""Shared frozen-model probe-activation server.

The white-box probes read activations by running the policy's generated text through a FROZEN copy of
the base model (original-model probing). Each RL run process otherwise loads its own ~16 GB copy, so
run-parallelism is capped by GPU memory (≈4 on an 80 GB card). This server loads the model ONCE and
serves ``extract_activations`` to all runs over localhost, so J runs share a single copy — parallelism
is then bounded by tinker/API limits, not memory.

    uv run python experiments/probe_server.py --model Qwen/Qwen3-8B --port 8177
    # point runs at it, either:
    export PROBE_SERVER_URL=http://127.0.0.1:8177        # env (queue_runs.sh -P does this)
    # …or per config:  {"probe_server_url": "http://127.0.0.1:8177", ...}

Endpoints: GET /health, GET /meta → {model_name, n_layers, d_model},
POST /extract_activations {items:[[q,cot,ans]…], follow_up:[q,a]|null, batch_size, preserve_thinking}
→ numpy .npy bytes [n, n_layers, d_model]. GPU work is serialised by a lock (one card); connections are
threaded so many runs can queue concurrently.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

import numpy as np

from monitordecorrelation.whitebox.model import WhiteBoxModel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8177)
    args = ap.parse_args()

    print(f"[probe_server] loading {args.model} …", flush=True)
    wb = WhiteBoxModel(args.model)
    lock = threading.Lock()  # the GPU can't run concurrent forward passes — serialise them
    meta = json.dumps({"model_name": wb.model_name, "n_layers": wb.n_layers, "d_model": wb.d_model}).encode()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send(200, b'{"ok":true}')
            elif self.path == "/meta":
                self._send(200, meta)
            else:
                self._send(404, b'{"error":"not found"}')

        def do_POST(self):  # noqa: N802
            if self.path != "/extract_activations":
                self._send(404, b'{"error":"not found"}')
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                items = [tuple(x) for x in req["items"]]
                fu = tuple(req["follow_up"]) if req.get("follow_up") else None
                with lock:
                    acts = wb.extract_activations(
                        items, follow_up=fu, batch_size=int(req.get("batch_size", 8)),
                        preserve_thinking=bool(req.get("preserve_thinking", False)))
                buf = BytesIO()
                np.save(buf, acts)
                self._send(200, buf.getvalue(), "application/octet-stream")
            except Exception as e:  # noqa: BLE001 — return the error rather than dropping the connection
                self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode())

        def log_message(self, *a):  # quiet (one line/request would spam a long run)
            pass

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[probe_server] {wb.model_name} on http://{args.host}:{args.port} "
          f"(n_layers={wb.n_layers}, d_model={wb.d_model}) — ready", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
