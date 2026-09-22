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
                "model_id": "anthropic/claude-3.5-haiku",
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
            subset="nlp",
            penalty_coef=1.0,
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
    held_out = [{"kind": "cot", "name": "m", "role": "held_out", "model_id": "x"}]
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

    paths = sorted((_REPO / "experiments" / "configs").rglob("*.json"))
    assert len(paths) > 10
    for p in paths:
        cfg = load_config(p)  # schema + relevance rules
        env = _MultiTurnEnv() if cfg.env == "terminal_verifier" else _SingleTurnEnv()
        validate_token_budgets(cfg, env)


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
