"""Probe-server resilience: a CUDA OOM halves the batch instead of failing the request, and the remote
client retries a 5xx / unreachable server (but never a 4xx) before the error reaches the run."""

from __future__ import annotations

import importlib.util
import io
import urllib.error
from pathlib import Path

import numpy as np
import pytest
import torch

from monitordecorrelation.whitebox import model as wbm

_REPO = Path(__file__).resolve().parents[1]


def _server():
    spec = importlib.util.spec_from_file_location("_probe_server", _REPO / "experiments" / "probe_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeWB:
    """Raises CUDA OOM for any batch larger than ``fits``; records the batch sizes it was asked for."""

    def __init__(self, fits: int):
        self.fits, self.asked = fits, []

    def extract_activations(self, items, *, follow_up, batch_size, preserve_thinking):
        self.asked.append(batch_size)
        if batch_size > self.fits:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return np.zeros((len(items), 2, 3), dtype=np.float32)


def test_oom_halves_the_batch_until_it_fits(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    wb = _FakeWB(fits=2)
    logs = []
    out = _server().extract_with_oom_backoff(wb, [("q", "c", "a")] * 5, follow_up=None, batch_size=8,
                                             preserve_thinking=False, log=lambda *a, **k: logs.append(a[0]))
    assert out.shape == (5, 2, 3)
    assert wb.asked == [8, 4, 2]
    assert len(logs) == 2 and "batch_size 2" in logs[-1]


def test_oom_at_batch_size_one_still_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        _server().extract_with_oom_backoff(_FakeWB(fits=0), [("q", "c", "a")], follow_up=None, batch_size=4,
                                           preserve_thinking=False, log=lambda *a, **k: None)


def _npy(arr) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_client_retries_5xx_and_connection_errors(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 500, "boom", {}, io.BytesIO(b'{"error":"OOM"}'))
        if len(calls) == 2:
            raise urllib.error.URLError("connection refused")
        return _Resp(_npy(np.ones((1, 2, 3), dtype=np.float32)))

    monkeypatch.setattr(__import__("urllib.request").request, "urlopen", fake_urlopen)
    out = wbm._post_with_retry("http://x/extract_activations", b"{}", sleep=lambda s: None)
    assert out.shape == (1, 2, 3) and len(calls) == 3


def test_client_never_retries_4xx(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, io.BytesIO(b"bad"))

    monkeypatch.setattr(__import__("urllib.request").request, "urlopen", fake_urlopen)
    with pytest.raises(urllib.error.HTTPError):
        wbm._post_with_retry("http://x/extract_activations", b"{}", sleep=lambda s: None)
    assert len(calls) == 1


def test_client_gives_up_after_the_retry_window(monkeypatch):
    def fake_urlopen(req, timeout):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(__import__("urllib.request").request, "urlopen", fake_urlopen)
    with pytest.raises(urllib.error.URLError):
        wbm._post_with_retry("http://x/extract_activations", b"{}", retry_s=0.0, sleep=lambda s: None)
