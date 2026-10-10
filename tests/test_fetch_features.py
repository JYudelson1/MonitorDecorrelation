"""scripts/fetch_features.py: complete shards only, the slim copy readable by LayerSource, the marker last, reruns
skip what is already here."""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from monitordecorrelation.whitebox.probe_fit import LayerSource
from monitordecorrelation.whitebox.store import LocalStore

spec = importlib.util.spec_from_file_location(
    "fetch_features", os.path.join(os.path.dirname(__file__), "..", "scripts", "fetch_features.py"))
ff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ff)


def shard(root, view, k, n=3, complete=True):
    d = os.path.join(root, "ds", view)
    os.makedirs(d, exist_ok=True)
    g = torch.Generator().manual_seed(k)
    save_file({"followup": torch.randn(n, 4, 8, generator=g).bfloat16(), "mean_action": torch.randn(n, 4, 8),
               "counts": torch.ones(n, 2, dtype=torch.int32), "answer_logp": torch.zeros(n, 2)},
              os.path.join(d, f"shard_{k:05d}.safetensors"), metadata={"feature_format": "test"})
    if complete:
        with open(os.path.join(d, f"shard_{k:05d}.jsonl"), "w") as fh:
            fh.writelines(json.dumps({"id": f"e{k}-{i}"}) + "\n" for i in range(n))


def test_fetch_slims_complete_shards_and_skips_on_rerun(tmp_path):
    src, dest = tmp_path / "store", tmp_path / "dest"
    shard(src, "with_reasoning", 0)
    shard(src, "with_reasoning", 1)
    shard(src, "with_reasoning", 2, complete=False)  # data without its marker: still uploading
    (src / "ds" / "index.json").write_text("{}")
    out = ff.fetch(LocalStore(str(src)), str(dest), keep=["followup", "counts", "answer_logp", "router_topk"],
                   log=lambda m: None)
    assert out == {"shards": 2, "fetched": 2, "missing_tensors": ["router_topk"]}
    got = dest / "ds" / "with_reasoning"
    assert sorted(os.listdir(got)) == [f"shard_0000{k}.{e}" for k in (0, 1) for e in ("jsonl", "safetensors")]
    assert (dest / "ds" / "index.json").exists()
    with safe_open(str(got / "shard_00000.safetensors"), framework="pt") as f:
        assert set(f.keys()) == {"followup", "counts", "answer_logp"} and f.metadata() == {"feature_format": "test"}
    a = LayerSource(str(dest / "ds"), "with_reasoning", "followup")
    b = LayerSource(str(src / "ds"), "with_reasoning", "followup")
    assert a.ids == ["e0-0", "e0-1", "e0-2", "e1-0", "e1-1", "e1-2"]
    np.testing.assert_array_equal(a.layer(2), b.layer(2))
    assert ff.fetch(LocalStore(str(src)), str(dest), keep=["followup"], log=lambda m: None)["fetched"] == 0
