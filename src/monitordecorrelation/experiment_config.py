"""Schema-validated experiment config for the generic runner (``experiments/run_experiment.py``).

One config describes a whole run: policy, env, RL hyperparams, and a list of **monitors** of mixed
kinds (CoT judges + white-box probes), each tagged ``train_against`` or ``held_out``. So
"train against a probe", "train against weak CoT while holding out strong CoT + two probes", etc. are
all just different ``monitors`` lists — no per-experiment scripts.

Validation is via pydantic with ``extra="forbid"`` and a discriminated union on ``kind``, so a typo'd
key or a malformed monitor is **rejected at load** rather than silently misbehaving. Loads JSON or YAML.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # reject unknown keys


class CoTMonitorSpec(_Strict):
    kind: Literal["cot"] = "cot"
    name: str
    role: Literal["train_against", "held_out"]
    model_id: str  # OpenRouter model id (the judge)
    behavior: str | None = (
        None  # None -> the env's behavior_name (sycophancy / reward_hacking / …)
    )
    threshold: float = 0.5
    use_cot: bool = (
        True  # False = output-only monitor (judge the answer, ignore the CoT)
    )
    use_output: bool = True  # False = ignore the output (judge the CoT only)
    reasoning: dict | None = Field(
        None,
        description="judge-side reasoning: the OpenRouter `reasoning` object every call of this judge "
        "sends, validated per judge model by `monitors.judge_reasoning.resolve_reasoning` (at LOAD). "
        'gemini-2.5-flash-lite: {"enabled": false} (off) or {"max_tokens": N} (on, thinking budget N, '
        "512 <= N < 2048); absent = the default, the smallest budget {\"max_tokens\": 512}. "
        'gemini-3.5-flash-lite: REQUIRED (it mandates reasoning) — {"effort": "low"|"medium"|"high"} '
        '(preferred) or {"max_tokens": N}. NB on gemini-3.5-flash-lite \'low\' behaves like no '
        "reasoning at all — measured on blatant reward-hacking terminal transcripts, 24/40 calls came "
        "back with `reasoning_tokens == 0` and 17/40 scored a reflexive 0, vs 0/40 at 'medium'. Any "
        "other judge model is REJECTED: reasoning support is specialized to those two.",
    )
    binary_judge: bool = (
        False  # True = YES/NO verdict instead of 0–100 score; omitted → False
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_reasoning_keys(cls, data):
        """``reasoning_effort`` / ``reasoning_max_tokens`` were replaced by ``reasoning``. Say so, with the
        translation, instead of pydantic's bare "extra inputs are not permitted" — and never guess:
        under the old keys "both null" meant reasoning OFF, which is no longer gemini-2.5's default."""
        legacy = sorted({"reasoning_effort", "reasoning_max_tokens"} & set(data or {})) \
            if isinstance(data, dict) else []
        if legacy:
            raise ValueError(
                f"monitor {data.get('name')!r}: {legacy} were replaced by `reasoning` (an OpenRouter "
                'reasoning object). Translate: reasoning_effort=E → {"effort": E}; '
                'reasoning_max_tokens=N → {"max_tokens": N}; both null (the old reasoning-OFF '
                'default) → {"enabled": false}. See monitors/judge_reasoning.py for what each judge '
                "model accepts."
            )
        return data

    @model_validator(mode="after")
    def _check_reasoning(self) -> "CoTMonitorSpec":
        """Reject a reasoning setting the judge model would not honour as written — or any judge model
        whose reasoning behaviour is not established — at LOAD, not at the first judge call. Same
        resolver ``CoTMonitor.__init__`` runs for hand-built monitors, so the two cannot drift. The
        field keeps what the config said (``None`` stays ``None`` = the model's default): resolving
        here would pin one model's default onto the spec, where a later ``--set`` of ``model_id``
        would silently carry it to another model."""
        from monitordecorrelation.monitors.judge_reasoning import resolve_reasoning

        resolve_reasoning(self.model_id, self.reasoning, monitor=self.name)
        return self


class ProbeMonitorSpec(_Strict):
    kind: Literal["probe"] = "probe"
    name: str
    role: Literal["train_against", "held_out"]
    probe_path: str  # dir saved by train_probe.py
    probe_model: str | None = (
        None  # default = probe meta's model_name (original-model: frozen base)
    )
    threshold: float | None = None
    batch_size: int = 8


MonitorSpec = Annotated[
    Union[CoTMonitorSpec, ProbeMonitorSpec], Field(discriminator="kind")
]

# The envs that slice their dataset, i.e. the ones ``subset`` means something to. Anywhere else a
# subset would be dropped on the floor by ``factory.make_env``, so the config rejects it.
ENVS_WITH_SUBSET = frozenset({"sycophancy", "impossiblebench", "terminal_verifier"})


class ExperimentConfig(_Strict):
    run_name: str
    experiment: str = "experiment"
    description: str = ""  # free-text note (self-documenting configs)
    policy: str = "Qwen/Qwen3-8B"
    backend: Literal["tinker", "transformers"] = "tinker"
    env: Literal[
        "sycophancy", "mbpp_honeypot", "impossiblebench", "terminal_verifier"
    ] = "sycophancy"
    subset: str | None = Field(
        None,
        description="which slice of the env to run. REQUIRED by the envs that have slices "
        "(sycophancy: political/nlp/… | impossiblebench: impossible/oneoff/conflicting | "
        "terminal_verifier: the reasoning-gym task, largest_island) and REJECTED by the ones that "
        "do not (mbpp_honeypot), where it would be silently ignored. The value itself is checked by "
        "the env (factory.make_env).",
    )
    env_options: dict = Field(
        default_factory=dict,
        description="env-specific constructor kwargs (impossiblebench: e.g. {'reward_mode': 'fraction', "
        "'exec_workers': 32}; terminal_verifier: {'verifier_mode': 'corrupted', 'offset_range': [4, 9], "
        "'max_turns': 4, 'output_view': 'transcript', 'command_timeout': 30}, plus 'submission_ends': "
        "'first'|'correct' — required by, and only by, the submit_* verifier_modes); validated by the env "
        "constructor",
    )
    n_steps: int = 40
    batch_size: int = Field(8, description="distinct prompts per step")
    group_size: int = Field(8, description="rollouts per prompt (GRPO group)")
    eval_every: int = Field(
        10, description="run the held-out eval phase every N steps (+ 0 + final)"
    )
    eval_size: int = Field(
        32, description="number of distinct held-out eval PROMPTS (fixed set)"
    )
    eval_samples_per_prompt: int = Field(
        1,
        description="rollouts sampled per eval prompt (>1 → higher-fidelity AUROC/d′ per eval, "
        "at eval_size×this monitor-scoring cost)",
    )
    save_every: int = Field(500, description="save the policy every N steps")
    probe_server_url: str | None = Field(
        None,
        description="if set, probes read activations from a shared probe_server.py instead of "
        "each run loading the base model locally (env PROBE_SERVER_URL is the fallback)",
    )
    max_tokens: int | None = Field(
        None,
        description="tokens per sampling call, for the runs where a call IS the whole turn: every "
        "single-turn env, and a multi-turn env with NO thinking budget. REQUIRED there, and REJECTED "
        "when a `think_budget` is in force (the budget sizes the thinking call and `answer_tokens` "
        "the answer, so `max_tokens` would be silently ignored). Checked by "
        "`validate_token_budgets` once the env — hence the resolved budget — is known.",
    )
    think_budget: int | None | Literal["auto"] = Field(
        "auto",
        description="multi-turn envs, thinking policies (Qwen3, Inkling): cap each turn's thinking at "
        "N tokens — when hit, the renderer's budget-forcing suffix closes it and the answer is sampled "
        "with `answer_tokens`. 'auto' (the default, i.e. the key is absent) = the env's "
        "`default_think_budget` (None for envs without one). null/None = NO budget: a turn is one call "
        "of max_tokens (a long think then eats the turn) — explicit, never overridden by the env default. "
        "Resolved once by `resolve_think_budget`; the training loop only ever sees int | None. "
        "MULTI-TURN ENVS ONLY: a single-turn env samples one call per rollout and never budgets it, so "
        "anything but 'auto' there is rejected by `validate_token_budgets` rather than ignored.",
    )
    answer_tokens: int | None = Field(
        None,
        description="tokens for the post-<think> answer that a `think_budget` forces. REQUIRED "
        "exactly when a budget is in force, and REJECTED otherwise (no budget → no forced answer → "
        "the value would be silently ignored). The mirror image of `max_tokens`; both are checked by "
        "`validate_token_budgets`.",
    )
    thinking_effort: float | None = Field(
        None,
        ge=0.0,
        lt=1.0,
        description="reasoning-effort conditioning for TML-rendered policies (Inkling): 0.1=minimal, "
        "0.7=medium, 0.9=high. Lower effort = far shorter CoT, so it trades monitor-visible reasoning "
        "against tokens per rollout. REQUIRED for a `thinkingmachines/*` policy and REJECTED for any "
        "other (HF-chat-template policies — Qwen3 & co. — have no such knob, so the value would be "
        "silently ignored); see `renderers.is_tml_policy`.",
    )
    penalty_coef: float | None = Field(
        None,
        description="constant weight λ on the train-against monitor penalty in the reward. Set "
        "EXACTLY ONE of `penalty_coef` / `penalty_schedule`, and only when the run has a "
        "train_against monitor — a control run applies no penalty at all, so either key would be "
        "silently ignored there.",
    )
    penalty_schedule: dict | None = Field(
        None,
        description="{'start_penalty': x, 'end_penalty': y}: λ ramps linearly x→y across n_steps "
        "(hack-then-hide curriculum — low λ to establish hacking, then rising λ to drive obfuscation). "
        "The alternative to a constant `penalty_coef`: set exactly one of the two, and only on a run "
        "that trains against a monitor.",
    )

    @field_validator("penalty_schedule")
    @classmethod
    def _check_schedule(cls, v):
        if v is not None and set(v) != {"start_penalty", "end_penalty"}:
            raise ValueError(
                "penalty_schedule must be {'start_penalty': float, 'end_penalty': float}"
            )
        return v

    kl_coef: float = Field(
        0.0,
        description="per-token KL-to-base penalty (tinker-cookbook "
        "incorporate_kl_penalty); >0 anchors the policy → prevents the "
        "reward-over-optimization collapse. 0 = off (default).",
    )
    kl_discount_factor: float | None = Field(
        None,
        description="discount for future KL in the penalty (0.0 = off). REQUIRED when `kl_coef` > 0 "
        "and REJECTED when it is 0 — with no KL penalty there is nothing to discount, so the value "
        "would be silently ignored.",
    )
    lora_rank: int = 16
    lr: float | None = Field(
        None, description="None = TM-estimated LoRA LR for the policy"
    )
    seed: int = 0
    wandb_project: str | None = Field(
        None,
        description="W&B project to log this run to (config file or --set wandb_project=…). "
        "None = the LoggingConfig default, 'monitor-decorrelation'.",
    )
    n_prompts_pool: int = Field(
        512, description="how many env prompts to draw the run's prompts from"
    )
    monitors: list[MonitorSpec] = Field(
        default_factory=list,
        description="mixed CoT judges / probes, each train_against or held_out. EMPTY is legal and "
        "means a pure-task-reward run: no monitor in the gradient and none measured (the plain-RL "
        "baseline for a new env).",
    )

    # ---- no key may be silently ignored -------------------------------------------------------
    # House rule (see the field descriptions): a key that only applies to SOME runs defaults to
    # None, is REQUIRED where it applies, and is REJECTED where it does not — so a setting that
    # cannot take effect fails the config instead of quietly doing nothing. The checks that need
    # only the config live here; the ones that need the constructed env (``max_tokens`` /
    # ``answer_tokens`` / ``think_budget``) live in ``validate_token_budgets``, which the runner
    # calls once the env exists.
    @model_validator(mode="after")
    def _check_nothing_is_ignored(self) -> "ExperimentConfig":
        from monitordecorrelation.rl.renderers import is_tml_policy

        errs: list[str] = []

        if self.env in ENVS_WITH_SUBSET and self.subset is None:
            errs.append(
                f"env {self.env!r} is sliced by `subset`, which is not set — set it (sycophancy: "
                "political/nlp/… | impossiblebench: impossible/oneoff/conflicting | "
                "terminal_verifier: largest_island)"
            )
        if self.env not in ENVS_WITH_SUBSET and self.subset is not None:
            errs.append(
                f"env {self.env!r} has no slices, so subset={self.subset!r} would be ignored — drop the key"
            )

        tml = is_tml_policy(self.policy)
        if tml and self.thinking_effort is None:
            errs.append(
                f"policy {self.policy!r} is TML-rendered, where reasoning effort is a real knob with no "
                "safe default — set `thinking_effort` (0.1=minimal, 0.5=medium-low, 0.9=high)"
            )
        if not tml and self.thinking_effort is not None:
            errs.append(
                f"policy {self.policy!r} uses an HF chat template, which has no reasoning-effort knob, so "
                f"thinking_effort={self.thinking_effort} would be ignored — drop the key"
            )

        trains_against = [m.name for m in self.monitors if m.role == "train_against"]
        if trains_against:
            if (self.penalty_coef is None) == (self.penalty_schedule is None):
                errs.append(
                    "set exactly one of `penalty_coef` (constant λ) and `penalty_schedule` (λ ramp): "
                    f"got penalty_coef={self.penalty_coef!r} and penalty_schedule={self.penalty_schedule!r}, "
                    f"with {len(trains_against)} train_against monitor(s) ({', '.join(trains_against)})"
                )
        elif self.penalty_coef is not None or self.penalty_schedule is not None:
            errs.append(
                "no train_against monitor, so no monitor penalty enters the reward and "
                f"penalty_coef={self.penalty_coef!r} / penalty_schedule={self.penalty_schedule!r} would be "
                "ignored — drop them (a control run's reward is the task reward alone)"
            )

        if self.kl_coef > 0 and self.kl_discount_factor is None:
            errs.append(
                f"kl_coef={self.kl_coef} turns the KL-to-base penalty on, so `kl_discount_factor` applies "
                "and must be set explicitly (0.0 = no discount)"
            )
        if self.kl_coef == 0 and self.kl_discount_factor is not None:
            errs.append(
                f"kl_coef is 0 (no KL penalty), so kl_discount_factor={self.kl_discount_factor} would be "
                "ignored — drop the key"
            )

        if self.probe_server_url is not None and not any(m.kind == "probe" for m in self.monitors):
            errs.append(
                f"probe_server_url={self.probe_server_url!r} is set but no monitor is a probe, so nothing "
                "would read from it — drop the key"
            )

        if self.backend == "transformers":
            # The local backend implements plain GRPO only: no KL-to-base, and no TML rendering.
            if self.kl_coef:
                errs.append(
                    "the transformers backend has no KL-to-base penalty (it is a tinker-only path), so "
                    f"kl_coef={self.kl_coef} would be ignored — set kl_coef=0 or use the tinker backend"
                )
            if tml:
                errs.append(
                    f"policy {self.policy!r} is TML-rendered and only the tinker backend can render it — "
                    "the transformers backend would fall back to an HF chat template it has no tokenizer for"
                )

        if errs:
            raise ValueError("; ".join(errs))
        return self


def validate_token_budgets(cfg: "ExperimentConfig", env) -> int | None:
    """Resolve ``think_budget`` against ``env`` and reject every token-budget key this run would ignore.

    The three keys divide by how a turn is sampled, which only the env knows:

    * single-turn env — one call per rollout, no budgeting: ``max_tokens`` sizes it; ``think_budget``
      (anything but the ``"auto"`` default) and ``answer_tokens`` are meaningless and rejected.
    * multi-turn env, budget in force — the thinking call is ``think_budget`` long and the forced
      answer ``answer_tokens`` long; ``max_tokens`` is never read, so it is rejected.
    * multi-turn env, ``think_budget: null`` — a turn is one call of ``max_tokens``; nothing forces an
      answer, so ``answer_tokens`` is rejected.

    Returns the resolved budget (``int | None``), which is what the training loop takes.
    """
    multi_turn = bool(getattr(env, "multi_turn", False))
    env_default = getattr(env, "default_think_budget", None)
    if not multi_turn:
        if env_default is not None:  # an env bug, not a config one — a single turn is never budgeted
            raise ValueError(
                f"{type(env).__name__} is single-turn but declares default_think_budget={env_default}, "
                "which nothing would apply"
            )
        if cfg.think_budget != "auto":
            raise ValueError(
                f"env {cfg.env!r} is single-turn: a rollout is ONE sampling call of max_tokens and no "
                f"thinking budget is ever applied, so think_budget={cfg.think_budget!r} would be ignored — "
                "drop the key (its default, \"auto\", resolves to no budget here)"
            )
    budget = resolve_think_budget(cfg.think_budget, env)
    if budget is None:
        if cfg.max_tokens is None:
            raise ValueError(
                "no thinking budget is in force, so each sampling call is sized by `max_tokens` — set it"
            )
        if cfg.answer_tokens is not None:
            raise ValueError(
                f"no thinking budget is in force (think_budget={cfg.think_budget!r} → None), so no answer is "
                f"ever forced and answer_tokens={cfg.answer_tokens} would be ignored — drop the key"
            )
    else:
        if cfg.answer_tokens is None:
            raise ValueError(
                f"think_budget={budget} caps each turn's thinking, after which the answer is sampled "
                "separately — set `answer_tokens` to size it"
            )
        if cfg.max_tokens is not None:
            raise ValueError(
                f"think_budget={budget} sizes the thinking call and answer_tokens the answer, so "
                f"max_tokens={cfg.max_tokens} would be ignored — drop the key"
            )
    return budget


def load_config(path: str | Path) -> ExperimentConfig:
    """Load + validate a JSON or YAML experiment config. Raises pydantic ValidationError if invalid."""
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return ExperimentConfig.model_validate(data)


def load_monitor_specs(path: str | Path) -> list[MonitorSpec]:
    """Just the ``monitors`` list of a config file, validated, WITHOUT the whole-run rules.

    For the post-hoc scripts that only re-score saved rollouts (``rescore_eval_rollouts``,
    ``eval_monitors_on_rollouts``): they read nothing but the judge battery, and the config they are
    pointed at is usually a finished run's ``config.json`` — which may predate the current relevance
    rules (or describe an env/policy combination those rules would now reject). Re-validating the
    whole run there would block re-scoring over a key that cannot affect a judge call. Every monitor
    spec is still fully validated; ``load_config`` remains the strict path for anything that RUNS."""
    from pydantic import TypeAdapter

    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return TypeAdapter(list[MonitorSpec]).validate_python(data.get("monitors", []))


def _coerce(v: str):
    if v.lower() in ("null", "none", ""):
        return None  # e.g. --set think_budget=null → NO thinking budget (one call per turn, no env default)
    if v[:1] in ("{", "["):
        # A JSON object / list, e.g. --set 'monitors.g25_cot.reasoning={"enabled":false}'. Malformed
        # JSON is an error, never a fallback to the raw string.
        try:
            return json.loads(v)
        except json.JSONDecodeError as e:
            raise SystemExit(f"--set: value {v!r} starts like JSON but does not parse: {e}") from e
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def _monitor_matches(mon: dict, selector: str) -> bool:
    """Does ``selector`` pick this monitor? ``*`` = all, ``model:<substr>`` = by model id, else by name."""
    if selector == "*":
        return True
    if selector.startswith("model:"):
        return selector[len("model:"):] in (mon.get("model_id") or "")
    return mon.get("name") == selector


def apply_overrides(
    cfg: ExperimentConfig,
    sets: list[str],
    *,
    allowed_fields: set[str] | None = None,
    allowed_monitor_fields: set[str] | None = None,
    not_allowed_hint: str = "",
) -> ExperimentConfig:
    """Apply ``--set key=value`` overrides and RE-VALIDATE. ``model_copy(update=…)`` skips validation,
    so a typo'd key or an out-of-range value would sail through and fail deep inside the run (or, worse,
    train something subtly different); round-tripping through the schema keeps ``--set`` as strict as
    the config file itself.

    A key of the form ``monitors.<selector>.<field>`` overrides a field on the matching monitor(s)
    instead of a top-level field — so per-run judge settings (notably ``reasoning``) are a launch
    flag, not a forked config file. The selector is a monitor ``name``, ``model:<substring of
    model_id>``, or ``*`` for every monitor. A value starting with ``{`` or ``[`` is parsed as JSON
    (quote it for the shell)::

        --set 'monitors.g35_out.reasoning={"effort":"medium"}'           # one judge, by name
        --set 'monitors.model:gemini-3.5.reasoning={"effort":"medium"}'  # every gemini-3.5 judge
        --set 'monitors.model:gemini-2.5.reasoning={"enabled":false}'    # gemini-2.5 judges: off
        --set 'monitors.model:gemini-2.5.reasoning={"max_tokens":1024}'  # … or a bigger budget
        --set monitors.*.threshold=0.6                                   # all of them

    ``reasoning`` is replaced WHOLE, never merged key by key (``{"enabled": false}`` + a budget would
    be contradictory).

    A key of the form ``env_options.<key>`` sets one entry of the ``env_options`` dict, keeping the
    rest (``--set env_options=…`` would replace the whole dict)::

        --set env_options.verifier_mode=possible

    Nothing is ever silently dropped — each of these is a SystemExit naming the offending item: an
    item without ``=``, an unknown field, a selector that matches no monitor (a silently-ignored
    override is how you end up analysing a run that trained against something else), and the same
    field set twice (on one monitor, possibly via two different selectors), where one value would
    silently lose. The result is re-validated like any other override, so e.g. pointing
    ``{"effort": …}`` at a gemini-2.5 judge (or ``{"enabled": false}`` at a gemini-3.5 one) fails
    loudly right here.

    ``allowed_fields`` / ``allowed_monitor_fields`` are for scripts that read only part of the config
    (e.g. an eval that never trains): a valid field the script would never look at is refused too,
    with ``not_allowed_hint`` appended to the error. None = every schema field is allowed.
    """
    if not sets:
        return cfg
    overrides: dict = {}
    monitor_overrides: list[tuple[str, str, str, object]] = []  # (the --set item, selector, field, value)
    env_option_overrides: dict = {}
    for kv in sets:
        key, eq, value = kv.partition("=")
        if not eq or not key:
            raise SystemExit(f"--set: {kv!r} is not of the form key=value")
        if key.startswith("monitors."):
            selector, _, field = key[len("monitors."):].rpartition(".")
            if not selector or not field:
                raise SystemExit(
                    f"--set: {key!r} is not a monitor override; expected "
                    "monitors.<name|model:substr|*>.<field>=<value>"
                )
            monitor_overrides.append((kv, selector, field, _coerce(value)))
        elif key.startswith("env_options."):
            opt = key[len("env_options."):]
            if not opt:
                raise SystemExit(f"--set: {kv!r} names no option; expected env_options.<key>=<value>")
            if opt in env_option_overrides:
                raise SystemExit(f"--set: {key!r} is set more than once")
            env_option_overrides[opt] = _coerce(value)
        else:
            if key in overrides:
                raise SystemExit(f"--set: {key!r} is set more than once")
            overrides[key] = _coerce(value)
    unknown = set(overrides) - set(type(cfg).model_fields)
    if unknown:
        raise SystemExit(f"--set: unknown config field(s) {sorted(unknown)}")
    if env_option_overrides and "env_options" in overrides:
        raise SystemExit("--set: env_options=… and env_options.<key>=… both given; pass one form")
    touched = set(overrides) | ({"env_options"} if env_option_overrides else set())
    if allowed_fields is not None and (ignored := touched - allowed_fields):
        raise SystemExit(
            f"--set: this script never reads {sorted(ignored)}, so the override would be silently "
            f"ignored. Overridable here: {sorted(allowed_fields)}. {not_allowed_hint}".rstrip()
        )
    data = cfg.model_dump()
    set_by: dict[tuple[str, str], str] = {}  # (monitor name, field) → the --set item that set it
    for kv, selector, field, value in monitor_overrides:
        matched = [m for m in data["monitors"] if _monitor_matches(m, selector)]
        if not matched:
            names = ", ".join(f"{m['name']} ({m.get('model_id') or m.get('probe_path')})"
                              for m in data["monitors"]) or "(none)"
            raise SystemExit(
                f"--set: monitor selector {selector!r} matched no monitor. Configured: {names}"
            )
        for mon in matched:
            spec = ProbeMonitorSpec if mon.get("kind") == "probe" else CoTMonitorSpec
            if field not in spec.model_fields:
                raise SystemExit(
                    f"--set: monitor {mon['name']!r} ({mon.get('kind', 'cot')}) has no field "
                    f"{field!r}; known: {sorted(spec.model_fields)}"
                )
            if allowed_monitor_fields is not None and field not in allowed_monitor_fields:
                raise SystemExit(
                    f"--set: this script never reads the monitor field {field!r}, so {kv!r} would be "
                    f"silently ignored. Overridable here: {sorted(allowed_monitor_fields)}. "
                    f"{not_allowed_hint}".rstrip()
                )
            if (mon["name"], field) in set_by:
                raise SystemExit(
                    f"--set: {kv!r} and {set_by[mon['name'], field]!r} both set {field!r} on monitor "
                    f"{mon['name']!r}; only one can win — pass one"
                )
            set_by[mon["name"], field] = kv
            mon[field] = value
    data["env_options"] = {**data["env_options"], **env_option_overrides}
    return type(cfg).model_validate({**data, **overrides})


def resolve_think_budget(think_budget: int | None | Literal["auto"], env) -> int | None:
    """The ONE place the config's ``think_budget`` becomes the ``int | None`` the sampling code takes.

    ``"auto"`` (the field default) → the env's ``default_think_budget`` (None if the env declares
    none, e.g. any single-turn env); an int → that int; ``None`` (``"think_budget": null`` in the
    config, ``--set think_budget=null``, or ``None`` in code) → **no budget**. The env default is a
    fallback for configs that don't mention the key, never an override of an explicit null — that
    was the bug where ``--set think_budget=null`` still ran with the env's 1536."""
    if think_budget == "auto":
        return getattr(env, "default_think_budget", None)
    return think_budget


def build_monitors(
    specs: list[MonitorSpec],
    default_behavior: str = "sycophancy",
    probe_server_url: str | None = None,
    *,
    multi_turn: bool = False,
) -> tuple[list, list]:
    """Construct (train_against, held_out) monitor objects. WhiteBoxModels are shared across probes
    that use the same ``probe_model`` so we never load the same model twice. ``probe_server_url`` (set)
    → probes proxy activation reads to a shared ``probe_server.py`` instead of each loading the base
    model locally (see experiments/probe_server.py). A CoT monitor with no
    explicit ``behavior`` inherits ``default_behavior`` (the env's ``behavior_name``) so a config's
    monitors target the right rubric (sycophancy vs reward_hacking) without repeating it per monitor.

    ``multi_turn`` (the env's ``multi_turn`` flag — the terminal env) makes every ``kind: cot`` spec an
    ``AgentCoTMonitor`` (chat-transcript judge over the per-turn episode record) instead of a
    ``CoTMonitor`` (the single-turn two-section judge). Same spec fields either way; a CoT-only
    (``use_output: false``) spec renders the reasoning-only transcript (the task, then each turn's
    thinking; commands, explanations and tool output withheld — and the prompt says so)."""
    from monitordecorrelation.monitors.agent_cot_monitor import AgentCoTMonitor
    from monitordecorrelation.monitors.cot_monitor import CoTMonitor
    from monitordecorrelation.monitors.probe_monitor import ProbeMonitor
    from monitordecorrelation.whitebox.model import WhiteBoxModel
    from monitordecorrelation.whitebox.probe import LinearProbe

    judge_cls = AgentCoTMonitor if multi_turn else CoTMonitor
    wb_cache: dict[str, WhiteBoxModel] = {}
    by_role: dict[str, list] = {"train_against": [], "held_out": []}
    for s in specs:
        if s.kind == "cot":
            mon = judge_cls(
                s.name,
                s.model_id,
                behavior=s.behavior or default_behavior,
                threshold=s.threshold,
                use_cot=s.use_cot,
                use_output=s.use_output,
                reasoning=s.reasoning,
                binary_judge=s.binary_judge,
            )
        else:  # probe
            probe = LinearProbe.load(s.probe_path)
            model_name = s.probe_model or probe.meta.get(
                "model_name", "Qwen/Qwen3-0.6B"
            )
            if model_name not in wb_cache:
                wb_cache[model_name] = WhiteBoxModel(
                    model_name, server_url=probe_server_url
                )
            mon = ProbeMonitor(
                s.name,
                wb_cache[model_name],
                probe,
                threshold=s.threshold,
                batch_size=s.batch_size,
            )
        by_role[s.role].append(mon)
    return by_role["train_against"], by_role["held_out"]
