"""Offline tests for the config/env-factory/monitor-build seam (no network, no model loads)."""

from __future__ import annotations

import pytest

from monitordecorrelation.experiment_config import ExperimentConfig, build_monitors


@pytest.fixture(autouse=True)
def _dummy_openrouter_key(monkeypatch):
    # CoTMonitor.__init__ requires a key to exist; these tests never call the API.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-used")


def _cfg(**kw) -> ExperimentConfig:
    base = dict(
        run_name="t",
        monitors=[
            {
                "kind": "cot",
                "name": "m",
                "role": "train_against",
                "model_id": "anthropic/claude-3.5-haiku",
            }
        ],
    )
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

    monkeypatch.setattr(
        "monitordecorrelation.envs.mbpp_honeypot.MbppHoneypotEnv", _Fake
    )
    env = factory.make_env(_cfg(env="mbpp_honeypot", n_prompts_pool=7, seed=3))
    assert isinstance(env, _Fake)
    assert calls == {"n": 7, "seed": 3}  # subset is (correctly) not passed to MBPP


def test_cot_monitor_inherits_env_behavior():
    """A CoT spec with no explicit behavior takes the env's behavior_name (reward_hacking for MBPP)."""
    cfg = _cfg(env="mbpp_honeypot")
    train_against, _ = build_monitors(cfg.monitors, default_behavior="reward_hacking")
    assert train_against[0].behavior == "reward_hacking"


def test_explicit_behavior_overrides_default():
    cfg = _cfg(
        monitors=[
            {
                "kind": "cot",
                "name": "m",
                "role": "held_out",
                "model_id": "x",
                "behavior": "deception",
            }
        ]
    )
    _, held_out = build_monitors(cfg.monitors, default_behavior="reward_hacking")
    assert held_out[0].behavior == "deception"


def test_default_behavior_falls_back_to_sycophancy():
    """Back-compat: old configs with no behavior + no env-derived default stay sycophancy."""
    cfg = _cfg()
    train_against, _ = build_monitors(cfg.monitors)
    assert train_against[0].behavior == "sycophancy"


def test_binary_judge_defaults_false_and_passes_through():
    cfg = _cfg()
    train_against, _ = build_monitors(cfg.monitors)
    assert train_against[0].binary_judge is False

    cfg = _cfg(
        monitors=[
            {
                "kind": "cot",
                "name": "m",
                "role": "train_against",
                "model_id": "x",
                "binary_judge": True,
            }
        ]
    )
    train_against, _ = build_monitors(cfg.monitors)
    assert train_against[0].binary_judge is True


# ---- --set overrides (experiments/run_experiment.py) -------------------------------------------


def _runner():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "_run_experiment",
        Path(__file__).resolve().parents[1] / "experiments" / "run_experiment.py",
    )
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


class _EnvWithDefault:
    default_think_budget = 1536


class _EnvWithoutDefault:
    pass


def test_think_budget_resolution_null_means_no_budget_auto_means_env_default():
    """The bug this guards: `--set think_budget=null` used to be silently replaced by the env's
    default_think_budget (1536) in the training loop, so 'no budget' was unreachable on the terminal
    env. Now: absent key → "auto" → env default; explicit null → None → no budget; int → int."""
    from monitordecorrelation.experiment_config import resolve_think_budget

    run = _runner()
    assert _cfg().think_budget == "auto"  # the key absent from a config == env default
    assert _cfg(think_budget=None).think_budget is None  # "think_budget": null in a config
    assert resolve_think_budget("auto", _EnvWithDefault()) == 1536
    assert resolve_think_budget("auto", _EnvWithoutDefault()) is None
    assert resolve_think_budget(None, _EnvWithDefault()) is None
    assert resolve_think_budget(1024, _EnvWithDefault()) == 1024
    # end to end through the CLI override path
    cfg = run.apply_overrides(_cfg(think_budget=1536), ["think_budget=null"])
    assert resolve_think_budget(cfg.think_budget, _EnvWithDefault()) is None
    cfg = run.apply_overrides(_cfg(think_budget=None), ["think_budget=auto"])
    assert resolve_think_budget(cfg.think_budget, _EnvWithDefault()) == 1536


def test_set_overrides_are_validated_not_just_assigned():
    import pytest
    from pydantic import ValidationError

    run = _runner()
    assert run.apply_overrides(_cfg(), ["n_steps=3", "lr=2e-4"]).n_steps == 3
    with pytest.raises(SystemExit):  # typo'd field
        run.apply_overrides(_cfg(), ["n_stpes=3"])
    with pytest.raises(
        ValidationError
    ):  # out-of-range value (thinking_effort must be < 1)
        run.apply_overrides(_cfg(), ["thinking_effort=1.5"])
    with pytest.raises(ValidationError):  # wrong type
        run.apply_overrides(_cfg(), ["n_steps=lots"])


def _gemini_cfg() -> ExperimentConfig:
    """A terminal-matrix-shaped config: gemini-2.5 judges (reasoning off) + gemini-3.5 (effort)."""
    return ExperimentConfig.model_validate(
        dict(
            run_name="t",
            monitors=[
                {"kind": "cot", "name": "g25_out", "role": "held_out",
                 "model_id": "google/gemini-2.5-flash-lite"},
                {"kind": "cot", "name": "g35_out", "role": "train_against",
                 "model_id": "google/gemini-3.5-flash-lite", "reasoning_effort": "low"},
                {"kind": "cot", "name": "g35_cot", "role": "held_out",
                 "model_id": "google/gemini-3.5-flash-lite", "reasoning_effort": "low"},
            ],
        )
    )


def test_gemini_25_may_not_be_given_judge_side_reasoning():
    """gemini-2.5 answers the SCORE line with reasoning off; turning it on would silently change what
    a held-out judge measures mid-matrix, so it's rejected at LOAD rather than ignored."""
    from pydantic import ValidationError

    for knob in ({"reasoning_effort": "low"}, {"reasoning_max_tokens": 256}):
        with pytest.raises(ValidationError, match="must run with reasoning OFF"):
            ExperimentConfig.model_validate(
                dict(run_name="t", monitors=[
                    {"kind": "cot", "name": "g25_out", "role": "train_against",
                     "model_id": "google/gemini-2.5-flash-lite", **knob}])
            )
    # …and the same rule guards a hand-built monitor, not just a config.
    from monitordecorrelation.monitors.cot_monitor import CoTMonitor

    with pytest.raises(ValueError, match="must run with reasoning OFF"):
        CoTMonitor("g25_out", "google/gemini-2.5-flash-lite", reasoning_effort="low")


def test_reasoning_effort_and_budget_are_mutually_exclusive():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="mutually exclusive"):
        ExperimentConfig.model_validate(
            dict(run_name="t", monitors=[
                {"kind": "cot", "name": "g35_out", "role": "train_against",
                 "model_id": "google/gemini-3.5-flash-lite",
                 "reasoning_effort": "low", "reasoning_max_tokens": 256}])
        )


def test_set_overrides_a_single_monitor_field():
    """``--set monitors.<name>.<field>`` makes per-judge settings a launch flag, not a forked config."""
    run = _runner()
    cfg = run.apply_overrides(_gemini_cfg(), ["monitors.g35_out.reasoning_effort=medium"])
    assert [m.reasoning_effort for m in cfg.monitors] == [None, "medium", "low"]


def test_set_overrides_every_monitor_of_a_model_family():
    run = _runner()
    cfg = run.apply_overrides(
        _gemini_cfg(), ["monitors.model:gemini-3.5.reasoning_effort=medium"]
    )
    assert [m.reasoning_effort for m in cfg.monitors] == [None, "medium", "medium"]
    assert cfg.monitors[0].model_id == "google/gemini-2.5-flash-lite"  # untouched


def test_monitor_overrides_are_validated_like_any_other():
    from pydantic import ValidationError

    run = _runner()
    with pytest.raises(SystemExit, match="matched no monitor"):
        run.apply_overrides(_gemini_cfg(), ["monitors.g35_nope.reasoning_effort=low"])
    with pytest.raises(SystemExit, match="has no field"):
        run.apply_overrides(_gemini_cfg(), ["monitors.g35_out.reasoning_effrt=low"])
    with pytest.raises(ValidationError):  # not one of low/medium/high
        run.apply_overrides(_gemini_cfg(), ["monitors.g35_out.reasoning_effort=lowish"])
    # a wildcard that would switch reasoning on for the gemini-2.5 judges is refused, by name
    with pytest.raises(ValidationError, match="g25_out"):
        run.apply_overrides(_gemini_cfg(), ["monitors.*.reasoning_effort=medium"])
