"""Offline tests for ``init_from_tinker_ckpt``: start a fresh run at a tinker checkpoint's weights with
a fresh optimizer — config validation, the backend's load semantics (fake tinker), the runner wiring,
and queue_runs.sh's -I / -R plumbing (a `uv` shim records the worker's argv)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from monitordecorrelation.experiment_config import ExperimentConfig, apply_overrides

_REPO = Path(__file__).resolve().parents[1]
_CKPT = "tinker://abc:train:0/weights/some_run-final"
_CFG = _REPO / "experiments/configs/mbpp_inkling_qmonitors/row_control.json"


def _cfg(**kw) -> ExperimentConfig:
    return ExperimentConfig.model_validate({"run_name": "t", "subset": "nlp", **kw})


# ---- config ----------------------------------------------------------------------------------------

def test_config_accepts_a_training_state_path_and_records_it():
    cfg = _cfg(init_from_tinker_ckpt=_CKPT)
    assert cfg.model_dump()["init_from_tinker_ckpt"] == _CKPT  # → config.json
    assert _cfg().init_from_tinker_ckpt is None


@pytest.mark.parametrize("bad", ["tinker://abc:train:0/sampler_weights/x", "abc/weights/x", "/tmp/weights/x"])
def test_config_rejects_anything_but_a_training_state_path(bad):
    with pytest.raises(ValidationError, match="training-state path"):
        _cfg(init_from_tinker_ckpt=bad)


def test_config_rejects_it_on_the_transformers_backend():
    with pytest.raises(ValidationError, match="transformers"):
        _cfg(init_from_tinker_ckpt=_CKPT, backend="transformers")


def test_set_override_is_validated():
    assert apply_overrides(_cfg(), [f"init_from_tinker_ckpt={_CKPT}"]).init_from_tinker_ckpt == _CKPT
    with pytest.raises(ValidationError):
        apply_overrides(_cfg(), ["init_from_tinker_ckpt=tinker://x/sampler_weights/y"])


# ---- backend (fake tinker) -------------------------------------------------------------------------

class _FakeTC:
    def __init__(self, calls):
        self.calls = calls

    def load_state(self, path):
        self.calls.append(("load_state", path))
        return SimpleNamespace(result=lambda: None)

    def load_state_with_optimizer(self, path):
        self.calls.append(("load_state_with_optimizer", path))
        return SimpleNamespace(result=lambda: None)


def _fake_tinker(monkeypatch, info):
    import monitordecorrelation.backends.tinker_backend as tb

    calls = []

    class SC:
        def create_rest_client(self):
            return SimpleNamespace(get_weights_info_by_tinker_path=lambda p: SimpleNamespace(
                result=lambda: (calls.append(("weights_info", p)), info)[1]))

        def create_lora_training_client(self, base_model, rank, seed):
            calls.append(("create", base_model, rank))
            return _FakeTC(calls)

        def create_sampling_client(self, base_model):
            calls.append(("ref_sampler", base_model))

    monkeypatch.setattr(tb.tinker, "ServiceClient", SC)
    monkeypatch.setattr(tb, "make_renderer", lambda *a, **k: SimpleNamespace())
    return tb.TinkerBackend, calls


def _info(base_model="Qwen/Qwen3-8B", lora_rank=8, **kw):
    from tinker.types import WeightsInfoResponse

    return WeightsInfoResponse(base_model=base_model, is_lora=True, lora_rank=lora_rank,
                               **{"train_unembed": True, "train_mlp": True, "train_attn": True, **kw})


def test_backend_loads_weights_only_and_keeps_kl_on_the_base(monkeypatch):
    TB, calls = _fake_tinker(monkeypatch, _info())
    b = TB("Qwen/Qwen3-8B", lora_rank=8, kl_coef=1e-4, init_from=_CKPT)
    assert calls == [("weights_info", _CKPT), ("create", "Qwen/Qwen3-8B", 8), ("load_state", _CKPT),
                     ("ref_sampler", "Qwen/Qwen3-8B")]
    assert b.init_info["base_model"] == "Qwen/Qwen3-8B" and b.init_info["lora_rank"] == 8


def test_backend_resume_supersedes_init(monkeypatch):
    TB, calls = _fake_tinker(monkeypatch, _info())
    TB("Qwen/Qwen3-8B", lora_rank=8, init_from=_CKPT, resume_from="tinker://r/weights/run-40")
    loads = [c for c in calls if c[0].startswith("load_state")]
    assert loads == [("load_state_with_optimizer", "tinker://r/weights/run-40")]


@pytest.mark.parametrize("info, match", [(_info(base_model="Qwen/Qwen3-32B"), "base model"),
                                         (_info(lora_rank=16), "LoRA rank"),
                                         (_info(train_mlp=False), "train_mlp=False")])
def test_backend_refuses_a_mismatched_checkpoint_before_creating_a_client(monkeypatch, info, match):
    TB, calls = _fake_tinker(monkeypatch, info)
    with pytest.raises(ValueError, match=match):
        TB("Qwen/Qwen3-8B", lora_rank=8, init_from=_CKPT)
    assert [c[0] for c in calls] == ["weights_info"]


# ---- runner wiring ---------------------------------------------------------------------------------

class _Stop(Exception):
    pass


def test_run_experiment_hands_the_checkpoint_to_the_backend(monkeypatch, tmp_path):
    import importlib.util

    import monitordecorrelation.backends.tinker_backend as tb

    spec = importlib.util.spec_from_file_location("_run_experiment", _REPO / "experiments/run_experiment.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    got = {}

    def spy(*a, **k):
        got.update(k)
        raise _Stop

    monkeypatch.setattr(tb, "TinkerBackend", spy)
    monkeypatch.setattr(run, "make_env", lambda cfg: SimpleNamespace())
    monkeypatch.setattr(run, "validate_token_budgets", lambda cfg, env: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--config", str(_CFG),
                                      "--set", f"init_from_tinker_ckpt={_CKPT}", "run_name=x"])
    with pytest.raises(_Stop):
        run.main()
    assert got["init_from"] == _CKPT and got["resume_from"] is None


# ---- queue_runs.sh worker --------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_queue_runs_worker_passes_the_checkpoint(tmp_path):
    shim = tmp_path / "bin"
    shim.mkdir()
    argv_file = tmp_path / "argv"
    (shim / "uv").write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "{argv_file}"\n')
    (shim / "uv").chmod(0o755)
    runname = f"_pytest_init_ckpt_{uuid.uuid4().hex[:8]}"
    run_dir = _REPO / "data/runs" / runname
    try:
        env = {**os.environ, "PATH": f"{shim}:{os.environ['PATH']}", "QR_INIT_CKPT": _CKPT,
               "QR_EXTRA_SETS": "n_steps=2"}
        rec = f"3\t{_CFG}\t{runname}\tdata/runs/{runname}/run.log"
        subprocess.run(["bash", str(_REPO / "scripts/queue_runs.sh"), "__worker", rec], env=env, check=True,
                       capture_output=True)
        args = argv_file.read_text().split("\n")[:-1]
        sets = args[args.index("--set") + 1:]
        assert sets == ["seed=3", f"run_name={runname}", f"init_from_tinker_ckpt={_CKPT}", "n_steps=2"]
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
