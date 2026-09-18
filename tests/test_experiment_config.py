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


def test_malformed_or_conflicting_set_items_fail_loudly():
    """Nothing passed to --set may be silently dropped: no '=', an empty key, a key given twice, or one
    monitor field set by two selectors (one value would silently lose) are all SystemExit."""
    from monitordecorrelation.experiment_config import apply_overrides

    for bad, msg in [
        (["n_steps"], "not of the form key=value"),
        (["=3"], "not of the form key=value"),
        (["n_steps=3", "n_steps=4"], "more than once"),
        (["monitors.g35_out.threshold=0.6", "monitors.g35_out.threshold=0.7"], "only one can win"),
        (["monitors.*.threshold=0.6", "monitors.g35_out.threshold=0.7"], "only one can win"),
        (["monitors.threshold=0.6"], "not a monitor override"),
        (["monitors..threshold=0.6"], "not a monitor override"),
    ]:
        with pytest.raises(SystemExit, match=msg):
            apply_overrides(_gemini_cfg(), bad)
    # an empty value is null, not dropped: a required string field then fails validation
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        apply_overrides(_gemini_cfg(), ["run_name="])
    # distinct fields on the same monitor, or the same field on different monitors, are fine
    cfg = apply_overrides(_gemini_cfg(), ["monitors.g35_out.threshold=0.6",
                                          "monitors.g35_cot.threshold=0.7",
                                          "monitors.g35_out.reasoning_effort=high"])
    assert [m.threshold for m in cfg.monitors] == [0.5, 0.6, 0.7]


def test_set_refuses_fields_the_script_never_reads():
    """An eval that reads only part of the config must refuse a --set it would silently ignore."""
    from monitordecorrelation.experiment_config import apply_overrides

    kw = dict(allowed_fields={"max_tokens", "monitors"}, allowed_monitor_fields={"threshold"},
              not_allowed_hint="Use --model.")
    assert apply_overrides(_gemini_cfg(), ["max_tokens=99", "monitors.*.threshold=0.9"], **kw).max_tokens == 99
    with pytest.raises(SystemExit, match=r"never reads \['policy'\].*Use --model"):
        apply_overrides(_gemini_cfg(), ["policy=foo"], **kw)
    with pytest.raises(SystemExit, match="never reads the monitor field 'role'"):
        apply_overrides(_gemini_cfg(), ["monitors.g35_out.role=held_out"], **kw)
    with pytest.raises(SystemExit, match="unknown config field"):  # typos still read as typos
        apply_overrides(_gemini_cfg(), ["max_tokns=99"], **kw)


# ---- env_options.<key> overrides + propagation into the terminal-verifier env ---------------------

_TV_CONFIG = "experiments/configs/terminal_verifier_control.json"
_REPO = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_set_env_options_key_merges_into_env_options():
    from monitordecorrelation.experiment_config import apply_overrides, load_config

    cfg = load_config(_REPO / _TV_CONFIG)
    assert cfg.env_options["verifier_mode"] == "corrupted"
    out = apply_overrides(cfg, ["env_options.verifier_mode=verifier_bug", "env_options.max_turns=6"])
    assert out.env_options == {**cfg.env_options, "verifier_mode": "verifier_bug", "max_turns": 6}
    # a new key is added; the config's other options survive
    assert apply_overrides(cfg, ["env_options.min_ones=4"]).env_options["min_ones"] == 4
    with pytest.raises(SystemExit, match="set more than once"):
        apply_overrides(cfg, ["env_options.verifier_mode=possible", "env_options.verifier_mode=corrupted"])
    with pytest.raises(SystemExit, match="both given"):
        apply_overrides(cfg, ["env_options={}", "env_options.verifier_mode=possible"])
    with pytest.raises(SystemExit, match="names no option"):
        apply_overrides(cfg, ["env_options.=x"])
    # a script that never reads env_options refuses the dotted form too
    with pytest.raises(SystemExit, match=r"never reads \['env_options'\]"):
        apply_overrides(cfg, ["env_options.verifier_mode=possible"], allowed_fields={"max_tokens"})


def test_every_terminal_verifier_config_uses_the_corrupted_verifier():
    import json

    paths = sorted((_REPO / "experiments" / "configs").glob("*.json"))
    tv = [p for p in paths if json.loads(p.read_text()).get("env") == "terminal_verifier"]
    assert len(tv) == 5
    for p in tv:
        opts = json.loads(p.read_text())["env_options"]
        assert opts["verifier_mode"] == "corrupted" and "impossible" not in opts, p.name


class _Stop(Exception):
    pass


def _load_script(name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"_{name}", _REPO / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("sets, want", [([], "corrupted"),
                                        (["env_options.verifier_mode=possible"], "possible"),
                                        (["env_options.verifier_mode=corrupted_negative"], "corrupted_negative"),
                                        (["env_options.verifier_mode=verifier_bug"], "verifier_bug")])
def test_run_experiment_propagates_verifier_mode_to_the_env(monkeypatch, sets, want):
    """Drive run_experiment.main() from argv up to the env it builds (then stop, before any tinker call)."""
    import sys

    import monitordecorrelation.backends.tinker_backend as tb
    from monitordecorrelation.envs.factory import make_env as real_make_env

    run = _load_script("run_experiment")
    built = []

    def spy(cfg):
        built.append(real_make_env(cfg))
        raise _Stop

    monkeypatch.setattr(tb, "TinkerBackend", lambda *a, **k: object())
    monkeypatch.setattr(run, "make_env", spy)
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--config", str(_REPO / _TV_CONFIG),
                                      "--set", "n_prompts_pool=16", *sets])
    with pytest.raises(_Stop):
        run.main()
    (env,) = built
    assert {it.verifier_mode for it in env.items + env.eval_items} == {want}
    assert env.holdout(1)[0].meta["verifier_mode"] == want


@pytest.mark.parametrize("sets, want", [([], "corrupted"),
                                        (["env_options.verifier_mode=possible"], "possible"),
                                        (["env_options.verifier_mode=corrupted_negative"], "corrupted_negative"),
                                        (["env_options.verifier_mode=verifier_bug"], "verifier_bug")])
def test_eval_terminal_monitors_baseline_propagates_verifier_mode_to_the_env(monkeypatch, sets, want):
    """Drive eval_terminal_monitors_baseline.main() from argv up to the env it builds (then stop at
    the first tinker call)."""
    import sys

    ev = _load_script("eval_terminal_monitors_baseline")
    built = []

    class SpyEnv(ev.TerminalVerifierEnv):
        @classmethod
        def from_task(cls, **kw):
            env = super().from_task(**kw)
            built.append(env)
            return env

    def no_tinker():
        raise _Stop

    monkeypatch.setattr(ev, "TerminalVerifierEnv", SpyEnv)
    monkeypatch.setattr(ev.tinker, "ServiceClient", no_tinker)
    monkeypatch.setattr(sys, "argv", ["eval_terminal_monitors_baseline.py", "--config", str(_REPO / _TV_CONFIG),
                                      "--n-prompts", "4", "--set", *sets])
    with pytest.raises(_Stop):
        ev.main()
    (env,) = built
    assert {it.verifier_mode for it in env.items + env.eval_items} == {want}
    # the config's other options reach the env too
    assert env.max_turns == 4 and env.output_view == "transcript" and env.command_timeout == 30.0
