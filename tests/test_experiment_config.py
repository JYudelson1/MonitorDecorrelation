"""Offline tests for the config/env-factory/monitor-build seam (no network, no model loads)."""

from __future__ import annotations

import pytest

from monitordecorrelation.experiment_config import ExperimentConfig, build_monitors


@pytest.fixture(autouse=True)
def _dummy_openrouter_key(monkeypatch):
    # CoTMonitor.__init__ requires a key to exist; these tests never call the API.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-used")


def _cfg(**kw) -> ExperimentConfig:
    base = dict(run_name="t", monitors=[{"kind": "cot", "name": "m", "role": "train_against",
                                         "model_id": "anthropic/claude-3.5-haiku"}])
    base.update(kw)
    return ExperimentConfig.model_validate(base)


def test_env_literal_accepts_both_envs():
    assert _cfg(env="sycophancy").env == "sycophancy"
    assert _cfg(env="mbpp_honeypot").env == "mbpp_honeypot"
    with pytest.raises(Exception):
        _cfg(env="not_an_env")


def test_make_env_dispatch(monkeypatch):
    import monitordecorrelation.envs.factory as factory

    calls = {}

    class _Fake:
        behavior_name = "fake"

        @classmethod
        def from_dataset(cls, **kw):
            calls.update(kw)
            return cls()

    monkeypatch.setattr("monitordecorrelation.envs.mbpp_honeypot.MbppHoneypotEnv", _Fake)
    env = factory.make_env(_cfg(env="mbpp_honeypot", n_prompts_pool=7, seed=3))
    assert isinstance(env, _Fake)
    assert calls == {"n": 7, "seed": 3}  # subset is (correctly) not passed to MBPP


def test_cot_monitor_inherits_env_behavior():
    """A CoT spec with no explicit behavior takes the env's behavior_name (reward_hacking for MBPP)."""
    cfg = _cfg(env="mbpp_honeypot")
    train_against, _ = build_monitors(cfg.monitors, default_behavior="reward_hacking")
    assert train_against[0].behavior == "reward_hacking"


def test_explicit_behavior_overrides_default():
    cfg = _cfg(monitors=[{"kind": "cot", "name": "m", "role": "held_out",
                          "model_id": "x", "behavior": "deception"}])
    _, held_out = build_monitors(cfg.monitors, default_behavior="reward_hacking")
    assert held_out[0].behavior == "deception"


def test_default_behavior_falls_back_to_sycophancy():
    """Back-compat: old configs with no behavior + no env-derived default stay sycophancy."""
    cfg = _cfg()
    train_against, _ = build_monitors(cfg.monitors)
    assert train_against[0].behavior == "sycophancy"


# ---- --set overrides (experiments/run_experiment.py) -------------------------------------------


def _runner():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "_run_experiment", Path(__file__).resolve().parents[1] / "experiments" / "run_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_set_null_clears_the_thinking_budget():
    """`--set think_budget=null` must yield None (no budget → one sampling call per turn), not the
    string 'null' — which would only blow up once sampling started."""
    run = _runner()
    cfg = run.apply_overrides(_cfg(think_budget=1536), ["think_budget=null"])
    assert cfg.think_budget is None
    assert run.apply_overrides(_cfg(), ["think_budget=None"]).think_budget is None


def test_set_overrides_are_validated_not_just_assigned():
    import pytest
    from pydantic import ValidationError

    run = _runner()
    assert run.apply_overrides(_cfg(), ["n_steps=3", "lr=2e-4"]).n_steps == 3
    with pytest.raises(SystemExit):           # typo'd field
        run.apply_overrides(_cfg(), ["n_stpes=3"])
    with pytest.raises(ValidationError):      # out-of-range value (thinking_effort must be < 1)
        run.apply_overrides(_cfg(), ["thinking_effort=1.5"])
    with pytest.raises(ValidationError):      # wrong type
        run.apply_overrides(_cfg(), ["n_steps=lots"])
