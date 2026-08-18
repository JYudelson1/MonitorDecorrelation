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

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # reject unknown keys


class CoTMonitorSpec(_Strict):
    kind: Literal["cot"] = "cot"
    name: str
    role: Literal["train_against", "held_out"]
    model_id: str  # OpenRouter model id (the judge)
    behavior: str | None = None  # None -> the env's behavior_name (sycophancy / reward_hacking / …)
    threshold: float = 0.5
    use_cot: bool = True  # False = output-only monitor (judge the answer, ignore the CoT)


class ProbeMonitorSpec(_Strict):
    kind: Literal["probe"] = "probe"
    name: str
    role: Literal["train_against", "held_out"]
    probe_path: str  # dir saved by train_probe.py
    probe_model: str | None = None  # default = probe meta's model_name (original-model: frozen base)
    threshold: float | None = None
    batch_size: int = 8


MonitorSpec = Annotated[Union[CoTMonitorSpec, ProbeMonitorSpec], Field(discriminator="kind")]


class ExperimentConfig(_Strict):
    run_name: str
    experiment: str = "experiment"
    description: str = ""  # free-text note (self-documenting configs)
    policy: str = "Qwen/Qwen3-8B"
    backend: Literal["tinker", "transformers", "nemo"] = "tinker"
    env: Literal["sycophancy", "mbpp_honeypot", "impossiblebench"] = "sycophancy"
    subset: str = "nlp"  # sycophancy: political/nlp/… | impossiblebench: impossible/oneoff/conflicting
    env_options: dict = Field(
        default_factory=dict,
        description="env-specific constructor kwargs (impossiblebench only, e.g. "
        "{'reward_mode': 'fraction', 'exec_workers': 32}); validated by the env constructor")
    n_steps: int = 40
    batch_size: int = Field(8, description="distinct prompts per step")
    group_size: int = Field(8, description="rollouts per prompt (GRPO group)")
    eval_every: int = Field(10, description="run the held-out eval phase every N steps (+ 0 + final)")
    eval_size: int = Field(32, description="number of distinct held-out eval PROMPTS (fixed set)")
    eval_samples_per_prompt: int = Field(
        1, description="rollouts sampled per eval prompt (>1 → higher-fidelity AUROC/d′ per eval, "
        "at eval_size×this monitor-scoring cost)")
    probe_server_url: str | None = Field(
        None, description="if set, probes read activations from a shared probe_server.py instead of "
        "each run loading the base model locally (env PROBE_SERVER_URL is the fallback)")
    max_tokens: int = 1024
    max_prompt_tokens: int = Field(
        4096, description="nemo backend only: prompt-token allowance. NeMo-RL sizes ONE sequence "
        "budget for prompt+completion, so it needs this on top of max_tokens; a prompt longer than "
        "this is truncated by the dataloader. Ignored by the tinker backend, which caps only the "
        "completion.")
    n_gpus: int | None = Field(
        None, description="nemo backend only: GPUs to train on (one node). None = every GPU visible "
        "on this machine. Sets cluster.gpus_per_node and the vLLM tensor-parallel size.")
    nemo_options: dict = Field(
        default_factory=dict,
        description="nemo backend only: last-word overrides of the derived MD_* knobs that "
        "parametrize experiments/configs/nemo/grpo.yaml, e.g. {'train_micro_batch_size': 2, "
        "'moe': true}. Keys may be bare or MD_-prefixed. See backends/nemo/params.py.")
    thinking_effort: float = Field(
        0.9, ge=0.0, lt=1.0,
        description="reasoning-effort conditioning for TML-rendered policies (Inkling): 0.1=minimal, "
        "0.7=medium, 0.9=high. Lower effort = far shorter CoT, so it trades monitor-visible reasoning "
        "against tokens per rollout. Ignored by HF-chat-template policies (Qwen3 & co.).")
    penalty_coef: float = 1.0  # weight on the train-against monitor penalty in the reward
    penalty_schedule: dict | None = Field(
        None, description="None → constant penalty_coef. Else {'start_penalty': x, 'end_penalty': y}: λ "
        "ramps linearly x→y across n_steps (hack-then-hide curriculum — low λ to establish hacking, then "
        "rising λ to drive obfuscation). Overrides penalty_coef.")

    @field_validator("penalty_schedule")
    @classmethod
    def _check_schedule(cls, v):
        if v is not None and set(v) != {"start_penalty", "end_penalty"}:
            raise ValueError("penalty_schedule must be {'start_penalty': float, 'end_penalty': float}")
        return v
    kl_coef: float = Field(0.0, description="per-token KL-to-base penalty (tinker-cookbook "
                           "incorporate_kl_penalty); >0 anchors the policy → prevents the "
                           "reward-over-optimization collapse. 0 = off (default).")
    kl_discount_factor: float = Field(0.0, description="discount for future KL in the penalty; 0 = off")
    lora_rank: int = 16
    lr: float | None = Field(None, description="None = TM-estimated LoRA LR for the policy")
    seed: int = 0
    n_prompts_pool: int = Field(512, description="how many env prompts to draw the run's prompts from")
    monitors: list[MonitorSpec] = Field(
        default_factory=list,
        description="mixed CoT judges / probes, each train_against or held_out. EMPTY is legal and "
        "means a pure-task-reward run: no monitor in the gradient and none measured (the plain-RL "
        "baseline for a new env).")


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


def build_monitors(specs: list[MonitorSpec], default_behavior: str = "sycophancy",
                   probe_server_url: str | None = None) -> tuple[list, list]:
    """Construct (train_against, held_out) monitor objects. WhiteBoxModels are shared across probes
    that use the same ``probe_model`` so we never load the same model twice. ``probe_server_url`` (set)
    → probes proxy activation reads to a shared ``probe_server.py`` instead of each loading the base
    model locally (see experiments/probe_server.py). A CoT monitor with no
    explicit ``behavior`` inherits ``default_behavior`` (the env's ``behavior_name``) so a config's
    monitors target the right rubric (sycophancy vs reward_hacking) without repeating it per monitor."""
    from monitordecorrelation.monitors.cot_monitor import CoTMonitor
    from monitordecorrelation.monitors.probe_monitor import ProbeMonitor
    from monitordecorrelation.whitebox.model import WhiteBoxModel
    from monitordecorrelation.whitebox.probe import LinearProbe

    wb_cache: dict[str, WhiteBoxModel] = {}
    by_role: dict[str, list] = {"train_against": [], "held_out": []}
    for s in specs:
        if s.kind == "cot":
            mon = CoTMonitor(s.name, s.model_id, behavior=s.behavior or default_behavior,
                             threshold=s.threshold, use_cot=s.use_cot)
        else:  # probe
            probe = LinearProbe.load(s.probe_path)
            model_name = s.probe_model or probe.meta.get("model_name", "Qwen/Qwen3-0.6B")
            if model_name not in wb_cache:
                wb_cache[model_name] = WhiteBoxModel(model_name, server_url=probe_server_url)
            mon = ProbeMonitor(
                s.name, wb_cache[model_name], probe, threshold=s.threshold, batch_size=s.batch_size
            )
        by_role[s.role].append(mon)
    return by_role["train_against"], by_role["held_out"]
