"""Offline tests for the config/env-factory/monitor-build seam (no network, no model loads)."""

from __future__ import annotations

import pytest

from monitordecorrelation.experiment_config import (
    ENVS_WITH_SUBSET,
    ExperimentConfig,
    build_monitors,
    validate_token_budgets,
)


@pytest.fixture(autouse=True)
def _dummy_openrouter_key(monkeypatch):
    # CoTMonitor.__init__ requires a key to exist; these tests never call the API.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-used")


def _cfg(**kw) -> ExperimentConfig:
    """A minimal VALID config. The relevance rules (see ``_check_nothing_is_ignored``) make a few keys
    conditional, so fill exactly the ones this config's env/monitors call for: a `subset` for the envs
    that have slices, and a λ only when something is trained against."""
    base = dict(
        run_name="t",
        monitors=[
            {
                "kind": "cot",
                "name": "m",
                "role": "train_against",
                "model_id": "google/gemini-2.5-flash-lite",
            }
        ],
    )
    base.update(kw)
    if base.get("env", "sycophancy") in ENVS_WITH_SUBSET:
        base.setdefault("subset", "nlp")
    if any(m["role"] == "train_against" for m in base["monitors"]):
        base.setdefault("penalty_coef", 1.0)
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
                "model_id": "google/gemini-2.5-flash-lite",
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
                "model_id": "google/gemini-2.5-flash-lite",
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
    """A terminal-matrix-shaped config: a gemini-2.5 judge (model-default reasoning) + gemini-3.5
    judges (explicit effort — they mandate reasoning)."""
    return ExperimentConfig.model_validate(
        dict(
            run_name="t",
            subset="nlp",
            penalty_coef=1.0,
            monitors=[
                {"kind": "cot", "name": "g25_out", "role": "held_out",
                 "model_id": "google/gemini-2.5-flash-lite"},
                {"kind": "cot", "name": "g35_out", "role": "train_against",
                 "model_id": "google/gemini-3.5-flash-lite", "reasoning": {"effort": "low"}},
                {"kind": "cot", "name": "g35_cot", "role": "held_out",
                 "model_id": "google/gemini-3.5-flash-lite"},
            ],
        )
    )


def _one_judge(model_id: str, **kw) -> ExperimentConfig:
    return ExperimentConfig.model_validate(dict(run_name="t", subset="nlp", monitors=[
        {"kind": "cot", "name": "j", "role": "held_out", "model_id": model_id, **kw}]))


def test_gemini_25_reasoning_is_off_or_a_budget_and_defaults_to_the_largest_budget():
    """gemini-2.5-flash-lite: {"enabled": false} or {"max_tokens": N ≥ 512}; absent = {"max_tokens": 2048}.
    The spec keeps what the config said (None = the model default); the MONITOR sends the resolved one."""
    from pydantic import ValidationError

    g25 = "google/gemini-2.5-flash-lite"
    for given, sent in [(None, {"max_tokens": 2048}), ({"enabled": False}, {"enabled": False}),
                        ({"max_tokens": 512}, {"max_tokens": 512}),
                        ({"max_tokens": 2047}, {"max_tokens": 2047})]:
        cfg = _one_judge(g25) if given is None else _one_judge(g25, reasoning=given)
        assert cfg.monitors[0].reasoning == given
        _, (mon,) = build_monitors(cfg.monitors)
        assert mon.reasoning == sent and mon._request_body("p")["reasoning"] == sent
    for bad, msg in [({"max_tokens": 511}, "clamps"), ({"max_tokens": 1}, "clamps"),
                     ({"max_tokens": 4096}, "completion cap"), ({"max_tokens": "512"}, "must be an int"),
                     ({"effort": "low"}, "unsupported reasoning"), ({"enabled": True}, "unsupported"),
                     ({"enabled": False, "max_tokens": 512}, "unsupported"), ("off", "valid dictionary")]:
        with pytest.raises(ValidationError, match=msg):
            _one_judge(g25, reasoning=bad)


def test_gemini_35_reasoning_is_an_effort_or_a_budget_and_defaults_to_effort_low():
    """gemini-3.5-flash-lite mandates reasoning: an effort or a budget, sent as given; absent =
    {"effort": "low"}. Turning it off (a 400 per call) or anything malformed fails at LOAD.
    The spec keeps what the config said (None = the model default); the MONITOR sends the resolved one."""
    from pydantic import ValidationError

    g35 = "google/gemini-3.5-flash-lite"
    for given, sent in [(None, {"effort": "low"}), ({"effort": "low"}, {"effort": "low"}),
                        ({"effort": "medium"}, {"effort": "medium"}), ({"effort": "high"}, {"effort": "high"}),
                        ({"max_tokens": 256}, {"max_tokens": 256})]:
        cfg = _one_judge(g35) if given is None else _one_judge(g35, reasoning=given)
        assert cfg.monitors[0].reasoning == given
        _, (mon,) = build_monitors(cfg.monitors)
        assert mon.reasoning == sent and mon._request_body("p")["reasoning"] == sent
    for bad in ({"enabled": False}, {"effort": "lowish"}, {"effort": "low", "max_tokens": 256},
                {"max_tokens": 0}):
        with pytest.raises(ValidationError, match="monitor 'j'"):
            _one_judge(g35, reasoning=bad)


def test_any_other_judge_model_is_refused_whatever_its_reasoning():
    """Reasoning support is specialized to the two geminis; for any other OpenRouter model even the
    default is refused (what its reasoning object does is unestablished) — in a config AND by hand."""
    from pydantic import ValidationError

    from monitordecorrelation.monitors.agent_cot_monitor import AgentCoTMonitor
    from monitordecorrelation.monitors.cot_monitor import CoTMonitor

    for model in ("anthropic/claude-3-haiku", "deepseek/deepseek-chat", "google/gemini-2.5-flash",
                  "google/gemini-2.5-flash-lite-preview-09-2025", "google/gemini-3.5-flash",
                  "openai/gpt-5.4-mini"):
        for kw in ({}, {"reasoning": {"enabled": False}}, {"reasoning": {"effort": "low"}}):
            with pytest.raises(ValidationError, match="specialized to google/gemini-2.5-flash-lite and "
                                                      "google/gemini-3.5-flash-lite"):
                _one_judge(model, **kw)
            for cls in (CoTMonitor, AgentCoTMonitor):
                with pytest.raises(ValueError, match="Implement support for this model"):
                    cls("j", model, **kw)


def test_legacy_reasoning_keys_are_refused_with_the_translation():
    """The old keys are not silently reinterpreted: 'both null' used to mean reasoning OFF, which is no
    longer gemini-2.5's default — so a legacy config must be rewritten, and the error says how."""
    from pydantic import ValidationError

    from monitordecorrelation.experiment_config import load_monitor_specs

    for legacy in ({"reasoning_effort": "low"}, {"reasoning_max_tokens": 256},
                   {"reasoning_effort": None, "reasoning_max_tokens": None}):
        with pytest.raises(ValidationError, match=r"replaced by `reasoning`.*\{\"enabled\": false\}"):
            _one_judge("google/gemini-3.5-flash-lite", **legacy)
    import json
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:  # old run config.json
        json.dump({"monitors": [{"kind": "cot", "name": "g25", "role": "held_out",
                                 "model_id": "google/gemini-2.5-flash-lite",
                                 "reasoning_effort": None, "reasoning_max_tokens": None}]}, f)
    with pytest.raises(ValidationError, match="replaced by `reasoning`"):
        load_monitor_specs(f.name)


def test_set_overrides_a_single_monitor_field():
    """``--set monitors.<name>.<field>`` makes per-judge settings a launch flag, not a forked config;
    a JSON value is parsed, and the override reaches the monitor's request body."""
    run = _runner()
    cfg = run.apply_overrides(_gemini_cfg(), ['monitors.g35_out.reasoning={"effort":"medium"}'])
    assert [m.reasoning for m in cfg.monitors] == [None, {"effort": "medium"}, None]
    ta, held = build_monitors(cfg.monitors)
    assert [m._request_body("p")["reasoning"] for m in ta + held] == \
        [{"effort": "medium"}, {"max_tokens": 2048}, {"effort": "low"}]


def test_set_overrides_every_monitor_of_a_model_family():
    run = _runner()
    cfg = run.apply_overrides(
        _gemini_cfg(), ['monitors.model:gemini-3.5.reasoning={"effort":"medium"}',
                        'monitors.model:gemini-2.5.reasoning={"enabled":false}']
    )
    assert [m.reasoning for m in cfg.monitors] == \
        [{"enabled": False}, {"effort": "medium"}, {"effort": "medium"}]
    assert cfg.monitors[0].model_id == "google/gemini-2.5-flash-lite"  # untouched
    cfg = run.apply_overrides(_gemini_cfg(), ['monitors.g25_out.reasoning={"max_tokens":1024}'])
    _, held = build_monitors(cfg.monitors)
    assert held[0]._request_body("p")["reasoning"] == {"max_tokens": 1024}
    # …and replaced WHOLE, never merged: {"enabled": false} over a budget drops the budget
    cfg = run.apply_overrides(cfg, ['monitors.g25_out.reasoning={"enabled":false}'])
    assert cfg.monitors[0].reasoning == {"enabled": False}
    # null puts a judge back on its model's default
    cfg = run.apply_overrides(cfg, ["monitors.g25_out.reasoning=null"])
    assert cfg.monitors[0].reasoning is None
    assert build_monitors(cfg.monitors)[1][0].reasoning == {"max_tokens": 2048}


def test_monitor_overrides_are_validated_like_any_other():
    from pydantic import ValidationError

    run = _runner()
    with pytest.raises(SystemExit, match="matched no monitor"):
        run.apply_overrides(_gemini_cfg(), ['monitors.g35_nope.reasoning={"effort":"low"}'])
    with pytest.raises(SystemExit, match="has no field"):
        run.apply_overrides(_gemini_cfg(), ['monitors.g35_out.reasoning_effort=low'])
    with pytest.raises(SystemExit, match="does not parse"):  # malformed JSON is never a raw string
        run.apply_overrides(_gemini_cfg(), ['monitors.g35_out.reasoning={"effort":low}'])
    with pytest.raises(ValidationError):  # not one of low/medium/high
        run.apply_overrides(_gemini_cfg(), ['monitors.g35_out.reasoning={"effort":"lowish"}'])
    # a wildcard that would give the gemini-2.5 judge an effort is refused, by name
    with pytest.raises(ValidationError, match="g25_out"):
        run.apply_overrides(_gemini_cfg(), ['monitors.*.reasoning={"effort":"medium"}'])
    # …and one that would switch the gemini-3.5 judges' mandatory reasoning off, too
    with pytest.raises(ValidationError, match="g35_out"):
        run.apply_overrides(_gemini_cfg(), ['monitors.*.reasoning={"enabled":false}'])
    # switching a judge's model re-resolves its (None = model-default) reasoning for the NEW model,
    # never carrying the old model's default over
    cfg = run.apply_overrides(_gemini_cfg(), ["monitors.g25_out.model_id=google/gemini-3.5-flash-lite"])
    assert cfg.monitors[0].reasoning is None
    assert build_monitors(cfg.monitors)[1][0]._request_body("p")["reasoning"] == {"effort": "low"}
    # …while an explicit setting valid on gemini-2.5 but not on gemini-3.5 is still refused
    with pytest.raises(ValidationError, match="g25_out"):
        run.apply_overrides(_gemini_cfg(), ['monitors.g25_out.reasoning={"enabled":false}',
                                            "monitors.g25_out.model_id=google/gemini-3.5-flash-lite"])


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
                                          'monitors.g35_out.reasoning={"effort":"high"}'])
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
                                        (["env_options.verifier_mode=verifier_bug"], "verifier_bug"),
                                        (["env_options.verifier_mode=submit_corrupted",
                                          "env_options.submission_ends=correct"], "submit_corrupted")])
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


# ---- nothing a config sets may be silently ignored ----------------------------------------------
# One test per relevance rule. The shape is always the same: the key is REQUIRED where it takes
# effect and REJECTED where it does not, so a run can never quietly use something other than what the
# config says. `_cfg` fills the conditional keys, so each test perturbs exactly one of them.


def _err(**kw) -> str:
    """The validation message for a config that must not load."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as e:
        _cfg(**kw)
    return str(e.value)


def test_subset_is_required_by_sliced_envs_and_refused_by_the_others():
    assert _cfg(env="sycophancy", subset="political").subset == "political"
    assert _cfg(env="mbpp_honeypot").subset is None
    assert "subset" in _err(env="sycophancy", subset=None)
    assert "would be ignored" in _err(env="mbpp_honeypot", subset="nlp")


def test_terminal_verifier_rejects_an_unknown_task_instead_of_coercing_it():
    """The factory used to replace any subset it didn't know with 'largest_island', so a typo'd task
    trained on a different env than the config named."""
    import monitordecorrelation.envs.factory as factory

    with pytest.raises(ValueError, match="unknown task 'not_a_task'"):
        factory.make_env(_cfg(env="terminal_verifier", subset="not_a_task", n_prompts_pool=2))


def test_thinking_effort_is_required_by_tml_policies_and_refused_by_the_others():
    assert _cfg(policy="thinkingmachines/Inkling-Small", thinking_effort=0.5).thinking_effort == 0.5
    assert _cfg(policy="Qwen/Qwen3-8B").thinking_effort is None
    assert "thinking_effort" in _err(policy="thinkingmachines/Inkling-Small")
    assert "would be ignored" in _err(policy="Qwen/Qwen3-8B", thinking_effort=0.5)


def test_exactly_one_penalty_knob_and_only_when_a_monitor_is_trained_against():
    held_out = [{"kind": "cot", "name": "m", "role": "held_out", "model_id": "google/gemini-2.5-flash-lite"}]
    ramp = {"start_penalty": 0.0, "end_penalty": 1.0}
    assert _cfg(penalty_coef=None, penalty_schedule=ramp).penalty_schedule == ramp
    assert "exactly one" in _err(penalty_coef=0.5, penalty_schedule=ramp)  # the ramp used to just win
    assert "exactly one" in _err(penalty_coef=None)
    # a control run applies no penalty at all, so neither knob may be set
    assert _cfg(monitors=held_out, penalty_coef=None).penalty_coef is None
    assert "no train_against monitor" in _err(monitors=held_out, penalty_coef=0.5)
    assert "no train_against monitor" in _err(monitors=held_out, penalty_schedule=ramp)


def test_kl_discount_factor_tracks_kl_coef():
    assert _cfg(kl_coef=1e-4, kl_discount_factor=0.0).kl_discount_factor == 0.0
    assert "kl_discount_factor" in _err(kl_coef=1e-4)
    assert "would be ignored" in _err(kl_discount_factor=0.5)  # kl_coef defaults to 0 = no KL penalty


def test_probe_server_url_needs_a_probe_to_serve():
    probe = [{"kind": "probe", "name": "p", "role": "held_out", "probe_path": "data/probes/x"}]
    assert _cfg(monitors=probe, penalty_coef=None, probe_server_url="http://x").probe_server_url
    assert "no monitor is a probe" in _err(probe_server_url="http://x")


def test_transformers_backend_refuses_the_knobs_it_does_not_implement():
    assert "kl_coef" in _err(backend="transformers", kl_coef=1e-4, kl_discount_factor=0.0)
    assert "only the tinker backend" in _err(
        backend="transformers", policy="thinkingmachines/Inkling-Small", thinking_effort=0.5
    )


class _SingleTurnEnv:
    multi_turn = False


class _MultiTurnEnv:
    multi_turn = True
    max_turns = 4
    default_think_budget = 1536


def test_token_budget_keys_must_match_how_a_turn_is_sampled():
    """max_tokens sizes a whole call; think_budget + answer_tokens size the two calls of a budgeted
    turn. Whichever pair is not in force would be read by nobody, so it is refused."""
    # single-turn env: one call of max_tokens, no budgeting at all
    assert validate_token_budgets(_cfg(max_tokens=1024), _SingleTurnEnv()) is None
    with pytest.raises(ValueError, match="max_tokens"):
        validate_token_budgets(_cfg(), _SingleTurnEnv())
    with pytest.raises(ValueError, match="would be ignored"):
        validate_token_budgets(_cfg(max_tokens=1024, answer_tokens=512), _SingleTurnEnv())
    with pytest.raises(ValueError, match="single-turn"):
        validate_token_budgets(_cfg(max_tokens=1024, think_budget=256), _SingleTurnEnv())

    # multi-turn with a budget (from the env, or explicit): the answer call needs its own size
    assert validate_token_budgets(_cfg(answer_tokens=512), _MultiTurnEnv()) == 1536
    assert validate_token_budgets(_cfg(think_budget=256, answer_tokens=512), _MultiTurnEnv()) == 256
    with pytest.raises(ValueError, match="answer_tokens"):
        validate_token_budgets(_cfg(), _MultiTurnEnv())
    with pytest.raises(ValueError, match="max_tokens=3072 would be ignored"):
        validate_token_budgets(_cfg(max_tokens=3072, answer_tokens=512), _MultiTurnEnv())

    # multi-turn, budget explicitly off: back to one call of max_tokens
    assert validate_token_budgets(_cfg(think_budget=None, max_tokens=3072), _MultiTurnEnv()) is None
    with pytest.raises(ValueError, match="answer_tokens=512 would be ignored"):
        validate_token_budgets(_cfg(think_budget=None, max_tokens=3072, answer_tokens=512), _MultiTurnEnv())


def test_run_episodes_enforces_the_same_split_as_the_config():
    """The config layer is not the only guard: the episode driver itself refuses the argument it
    would not read, so a hand-written call cannot pass one either."""
    from monitordecorrelation.rl.episodes import run_episodes

    with pytest.raises(ValueError, match="max_tokens"):
        run_episodes(None, None, _MultiTurnEnv(), [], max_tokens=999, think_budget=100, answer_tokens=20)
    with pytest.raises(ValueError, match="answer_tokens"):
        run_episodes(None, None, _MultiTurnEnv(), [], max_tokens=999, answer_tokens=20)
    with pytest.raises(ValueError, match="max_tokens must be given"):
        run_episodes(None, None, _MultiTurnEnv(), [])


def test_every_repo_config_satisfies_the_relevance_rules():
    """The configs shipped in experiments/configs/ are the worked examples of these rules."""
    from monitordecorrelation.experiment_config import load_config

    import json

    from pydantic import ValidationError

    from monitordecorrelation.monitors.judge_reasoning import SUPPORTED_JUDGES

    paths = sorted((_REPO / "experiments" / "configs").rglob("*.json"))
    assert len(paths) > 10
    n_loaded = 0
    for p in paths:
        raw = json.loads(p.read_text())
        if not any(m.get("role") == "train_against" for m in raw.get("monitors", [])):
            # a control says it carries no λ explicitly, rather than by omission
            assert "penalty_coef" in raw and raw["penalty_coef"] is None, p
        judges = {m["model_id"] for m in raw.get("monitors", [])
                  if m.get("kind", "cot") == "cot" and m.get("provider", "openrouter") == "openrouter"}
        if judges - set(SUPPORTED_JUDGES):
            # The older configs judge with models (claude / deepseek) whose reasoning behaviour
            # monitors.judge_reasoning does not establish: they must be refused loudly, not run.
            with pytest.raises(ValidationError, match="specialized to"):
                load_config(p)
            continue
        cfg = load_config(p)  # schema + relevance rules
        env = _MultiTurnEnv() if cfg.env == "terminal_verifier" else _SingleTurnEnv()
        validate_token_budgets(cfg, env)
        n_loaded += 1
    assert n_loaded > 10


def test_every_rl_field_reaches_the_training_loop(monkeypatch, tmp_path):
    """The other half of "nothing is ignored": a key the schema accepts must actually ARRIVE.

    ``save_every: 6`` was accepted, written into the run folder, and then dropped on the way to
    ``RunConfig`` — every run silently checkpointed at the default 500 (i.e. step 0 only), which is
    only discoverable by noticing the missing files afterwards. So assert the whole hand-off.
    """
    import sys

    import monitordecorrelation.backends.tinker_backend as tb

    run = _load_script("run_experiment")
    seen = {}

    def spy(run_config, env, backend, **kw):
        seen["cfg"] = run_config
        seen["kw"] = kw
        raise _Stop

    monkeypatch.setattr(tb, "TinkerBackend", lambda *a, **k: object())
    monkeypatch.setattr(run, "run_grpo", spy)
    monkeypatch.chdir(tmp_path)  # the runner writes data/runs/<run_name>/config.json
    overrides = {"n_steps": 9, "batch_size": 3, "group_size": 5, "eval_every": 2, "eval_size": 7,
                 "eval_samples_per_prompt": 4, "save_every": 6, "lora_rank": 8, "seed": 11,
                 "penalty_coef": 0.25, "kl_coef": 0.001, "kl_discount_factor": 0.5, "max_tokens": 321,
                 "lr": 0.0007, "n_prompts_pool": 16}
    # a config that trains against a monitor, so penalty_coef is one of the fields in play
    cfg_path = _REPO / "experiments" / "configs" / "terminal_verifier_gemini25_out.json"
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--config", str(cfg_path),
                                      "--set", *[f"{k}={v}" for k, v in overrides.items()]])
    with pytest.raises(_Stop):
        run.main()

    rc = seen["cfg"]
    for field in ("n_steps", "batch_size", "group_size", "eval_every", "eval_size",
                  "eval_samples_per_prompt", "save_every", "lora_rank", "seed", "penalty_coef",
                  "kl_coef"):
        assert getattr(rc, field) == overrides[field], f"{field} never reached RunConfig"
    assert rc.learning_rate == overrides["lr"]
    assert seen["kw"]["max_tokens"] == overrides["max_tokens"]  # this config runs with no think_budget
    assert seen["kw"]["think_budget"] is None and seen["kw"]["answer_tokens"] is None


def _gemini_configs() -> list[str]:
    """Every shipped config with a gemini judge (relative to experiments/configs)."""
    import json

    root = _REPO / "experiments" / "configs"
    return sorted(str(p.relative_to(root)) for p in root.rglob("*.json")
                  if any("gemini" in str(m.get("model_id")) for m in json.loads(p.read_text()).get("monitors") or []))


def test_gemini_configs_are_found():
    assert len(_gemini_configs()) >= 22  # guards the parametrization below against a silent empty glob


@pytest.mark.parametrize("cfg_name", _gemini_configs())
def test_judge_reasoning_reaches_every_judge_call(monkeypatch, tmp_path, cfg_name):
    """Judge reasoning, from the model default and from ``--set``, must arrive in the request body of
    every judge the training loop is handed — through the REAL runner (run_experiment.main), not just
    the helpers it calls — and be recorded as sent (run_info's monitor records, config.json). Run over
    EVERY shipped gemini config: none sets ``reasoning``, so each judge runs on its model's default."""
    import json
    import sys

    import monitordecorrelation.backends.tinker_backend as tb
    import monitordecorrelation.envs.factory as factory
    from monitordecorrelation.rl.train import _monitor_info

    class _Env:
        behavior_name = "reward_hacking"
        multi_turn = False
        default_think_budget = None

    class _TermEnv(_Env):
        multi_turn = True
        max_turns = 4
        default_think_budget = 1536

    cfg_path = _REPO / "experiments" / "configs" / cfg_name
    multi_turn = json.loads(cfg_path.read_text())["env"] == "terminal_verifier"  # AgentCoTMonitor judges
    run = _load_script("run_experiment")
    monkeypatch.setattr(tb, "TinkerBackend", lambda *a, **k: object())
    if not multi_turn:  # keep the MBPP dataset out of it; the terminal env builds offline
        monkeypatch.setattr(run, "make_env", lambda cfg: _Env())
    # probes (the MBPP rows hold two out) would load the 8B base model; they carry no judge reasoning
    import monitordecorrelation.monitors.probe_monitor as pm
    import monitordecorrelation.whitebox.model as wbm
    import monitordecorrelation.whitebox.probe as wbp

    monkeypatch.setattr(wbp.LinearProbe, "load", classmethod(lambda cls, path: type("P", (), {"meta": {}})()))
    monkeypatch.setattr(wbm, "WhiteBoxModel", lambda *a, **k: object())
    monkeypatch.setattr(pm, "ProbeMonitor", lambda name, *a, **k: type("PM", (), {"name": name})())
    import monitordecorrelation.monitors.vllm as vllm_mod

    monkeypatch.setattr(vllm_mod, "check_server", lambda *a, **k: None)  # no live vLLM server in tests
    monkeypatch.chdir(tmp_path)
    file_cfg = {m["name"]: m for m in json.loads(cfg_path.read_text())["monitors"]}

    def launch(*sets):
        seen = {}

        def spy(run_config, env, backend, **kw):
            seen.update(kw)
            raise _Stop

        monkeypatch.setattr(run, "run_grpo", spy)
        monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--config", str(cfg_path),
                                          "--set", "run_name=r", *sets])
        with pytest.raises(_Stop):
            run.main()
        judges = [m for m in seen["train_against"] + seen["held_out"] if hasattr(m, "model_id")]
        assert judges and all(
            type(j).__name__ == ("AgentCoTMonitor" if multi_turn else "CoTMonitor") for j in judges)
        # vLLM judges take no `reasoning` (their thinking is enable_thinking / thinking_budget)
        assert all(j.reasoning is None and "reasoning" not in j._request_body("p")
                   for j in judges if j.backend.provider == "vllm")
        judges = [j for j in judges if j.backend.provider == "openrouter"]
        sent = {j.name: j._request_body("p")["reasoning"] for j in judges}
        roles = {m.name: "train_against" for m in seen["train_against"]}
        # recorded exactly as sent, in run_info (via rl/train.py's monitor records) …
        assert {j.name: _monitor_info(j, roles.get(j.name, "held_out"))["reasoning"]
                for j in judges} == sent
        # … and the effective config.json carries the override, not the file's value
        written = {m["name"]: m["reasoning"]
                   for m in json.loads((tmp_path / "data/runs/r/config.json").read_text())["monitors"]
                   if m["kind"] == "cot" and m["provider"] == "openrouter"}
        assert written == {m["name"]: m["reasoning"] for m in seen["run_info"]["config"]["monitors"]
                           if m["kind"] == "cot" and m["provider"] == "openrouter"}
        assert set(written) == set(sent)
        return sent, written

    g25 = [n for n, m in file_cfg.items() if m.get("model_id") == "google/gemini-2.5-flash-lite"]
    g35 = [n for n, m in file_cfg.items() if m.get("model_id") == "google/gemini-3.5-flash-lite"]
    assert g25 and g35 and set(g25) | set(g35) == {n for n, m in file_cfg.items()
                                                   if m["kind"] == "cot" and m.get("provider") != "vllm"}
    assert not any("reasoning" in m for m in file_cfg.values())  # every judge on its model's default

    sent, written = launch()  # the file as shipped
    assert all(sent[n] == {"max_tokens": 2048} and written[n] is None for n in g25)
    assert all(sent[n] == {"effort": "low"} and written[n] is None for n in g35)

    sent, written = launch('monitors.model:gemini-2.5.reasoning={"max_tokens":1024}',
                           'monitors.model:gemini-3.5.reasoning={"effort":"medium"}')
    assert all(sent[n] == written[n] == {"max_tokens": 1024} for n in g25)
    assert all(sent[n] == written[n] == {"effort": "medium"} for n in g35)

    sent, written = launch(f'monitors.{g25[0]}.reasoning={{"enabled":false}}')
    assert sent[g25[0]] == written[g25[0]] == {"enabled": False}
    assert all(sent[n] == {"max_tokens": 2048} for n in g25[1:])  # the other judges keep the default

    sent, written = launch(f"monitors.{g25[0]}.reasoning=null")  # null → the model default
    assert written[g25[0]] is None and sent[g25[0]] == {"max_tokens": 2048}
