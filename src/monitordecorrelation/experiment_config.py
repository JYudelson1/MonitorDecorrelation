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

from pydantic import BaseModel, ConfigDict, Field


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
    backend: Literal["tinker", "transformers"] = "tinker"
    env: Literal["sycophancy", "mbpp_honeypot"] = "sycophancy"
    subset: str = "nlp"
    n_steps: int = 40
    batch_size: int = Field(8, description="distinct prompts per step")
    group_size: int = Field(8, description="rollouts per prompt (GRPO group)")
    eval_every: int = Field(10, description="run the held-out eval phase every N steps (+ 0 + final)")
    eval_size: int = Field(32, description="size of the fixed held-out eval set")
    max_tokens: int = 1024
    penalty_coef: float = 1.0  # weight on the train-against monitor penalty in the reward
    kl_coef: float = Field(0.0, description="per-token KL-to-base penalty (tinker-cookbook "
                           "incorporate_kl_penalty); >0 anchors the policy → prevents the "
                           "reward-over-optimization collapse. 0 = off (default).")
    kl_discount_factor: float = Field(0.0, description="discount for future KL in the penalty; 0 = off")
    lora_rank: int = 16
    lr: float | None = Field(None, description="None = TM-estimated LoRA LR for the policy")
    seed: int = 0
    n_prompts_pool: int = Field(512, description="how many env prompts to draw the run's prompts from")
    monitors: list[MonitorSpec] = Field(min_length=1)


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


def build_monitors(specs: list[MonitorSpec], default_behavior: str = "sycophancy") -> tuple[list, list]:
    """Construct (train_against, held_out) monitor objects. WhiteBoxModels are shared across probes
    that use the same ``probe_model`` so we never load the same model twice. A CoT monitor with no
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
                wb_cache[model_name] = WhiteBoxModel(model_name)
            mon = ProbeMonitor(
                s.name, wb_cache[model_name], probe, threshold=s.threshold, batch_size=s.batch_size
            )
        by_role[s.role].append(mon)
    return by_role["train_against"], by_role["held_out"]
