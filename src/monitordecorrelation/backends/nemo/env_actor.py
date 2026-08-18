"""This project's ``Env`` + ``Monitor``s, wearing NeMo-RL's ``EnvironmentInterface``.

NeMo-RL owns the RL loop; the *reward* is ours, and this actor is where it is computed:

    reward = task_reward - penalty_coef * mean(train_against monitor scores)

exactly as ``rl/train.py`` computes it for the tinker backend. Held-out monitors never touch this
number — they are scored only by the ``role="val"`` instance, on the held-out eval set, purely as a
measurement. ``behavior_present`` (the oracle) is likewise measured and never fed back.

Two instances exist per run:

  ``role="train"``  scores the train-against monitors on every training batch and produces the
                    ``metrics.jsonl`` / ``rollouts.jsonl`` rows.
  ``role="val"``    scores EVERY monitor on the fixed held-out set and produces the
                    ``eval_metrics.jsonl`` / ``eval_rollouts.jsonl`` rows — the degradation curves.

The rows are not written here. ``step()`` buffers them and ``pop_records()`` hands them to the
driver, which is the only place that knows the authoritative training step number (NeMo-RL passes it
to the logger, not to the environment). See ``driver.py``.

Runs inside NeMo-RL's virtualenv, so it may import ray/torch/transformers but NOT tinker.
"""

from __future__ import annotations

from statistics import mean
from typing import Any

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

from monitordecorrelation.envs.factory import make_env
from monitordecorrelation.eval.rows import eval_row, train_row
from monitordecorrelation.experiment_config import ExperimentConfig, build_monitors
from monitordecorrelation.rl.text import split_cot_answer
from monitordecorrelation.rl.train import score_env, score_monitors
from monitordecorrelation.types import Prompt, Rollout


@ray.remote(max_concurrency=1000)  # pragma: no cover — needs a ray cluster
class MonitorDecorrelationEnv(EnvironmentInterface):
    """cfg keys: ``experiment_config`` (the ExperimentConfig as a dict), ``role`` ("train"|"val"),
    and optionally ``monitor_workers`` (API-judge concurrency; default 16, as in ``rl/train.py``)."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = ExperimentConfig.model_validate(cfg["experiment_config"])
        self.role = cfg["role"]
        assert self.role in ("train", "val"), self.role
        self.env = make_env(self.cfg)
        train_against, held_out = build_monitors(
            self.cfg.monitors,
            default_behavior=self.env.behavior_name,
            probe_server_url=self.cfg.probe_server_url,
        )
        # The train-against monitors are the ONLY ones in the reward, in either role. The val role
        # additionally *measures* the held-out ones.
        self.train_against = train_against
        self.scored = [*train_against, *held_out] if self.role == "val" else list(train_against)
        self.monitor_workers = int(cfg.get("monitor_workers", 16))  # API-judge concurrency
        self._tokenizer = None  # lazily loaded; only used for the token-count metrics
        self._calls = 0
        self._records: list[dict[str, Any]] = []

    # -- helpers --------------------------------------------------------------------------------
    @property
    def tokenizer(self):
        """The policy tokenizer, for the ``tokens/…`` metrics only.

        NeMo-RL hands ``step()`` role+content, not the sampled token ids, so completion lengths are
        recovered by re-encoding the assistant text. That round-trips exactly for well-formed text
        and is what ``tokens/truncated_rate`` is read off — which matters a lot at a 16k completion
        budget, where a rising output length is only interpretable next to the truncation rate.
        """
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.policy)
        return self._tokenizer

    def _penalty_coef(self) -> float:
        """λ for this step: constant ``penalty_coef``, or the linear ``penalty_schedule`` ramp.

        The schedule needs the step index; NeMo-RL does not pass one to the environment, so the
        train role counts its own ``step()`` calls — which is exact because ``max_rollout_turns`` is
        1 and this repo's envs are single-step, so there is precisely one call per training step.
        """
        if not self.cfg.penalty_schedule:
            return self.cfg.penalty_coef
        frac = self._calls / max(1, self.cfg.n_steps - 1)
        s, e = self.cfg.penalty_schedule["start_penalty"], self.cfg.penalty_schedule["end_penalty"]
        return s + (e - s) * min(1.0, frac)

    def _to_rollouts(self, message_log_batch, metadata) -> list[Rollout]:
        rollouts = []
        for conversation, meta in zip(message_log_batch, metadata):
            answer_text = "".join(
                str(m["content"]) for m in conversation if m["role"] == "assistant"
            )
            cot, output = split_cot_answer(answer_text)
            prompt = Prompt(text=meta["prompt_text"], meta=dict(meta["prompt_meta"]))
            n_out = len(self.tokenizer.encode(answer_text, add_special_tokens=False))
            rollouts.append(
                Rollout(
                    prompt=prompt, cot=cot, output=output,
                    meta={"n_prompt_tokens": int(meta["n_prompt_tokens"]),
                          "n_output_tokens": n_out,
                          "full_text": answer_text},
                )
            )
        return rollouts

    # -- EnvironmentInterface -------------------------------------------------------------------
    def step(
        self, message_log_batch: list[LLMMessageLogType], metadata: list[dict[str, Any]]
    ) -> EnvironmentReturn:
        rollouts = self._to_rollouts(message_log_batch, metadata)
        env_results = score_env(self.env, rollouts)
        mon = score_monitors(self.scored, rollouts, self.monitor_workers)

        # A partially-failed train-against score (NaN sentinel) must not produce a NaN reward →
        # coalesce to 0 (that rollout just isn't penalised this step). Without this, one API hiccup
        # would NaN the advantages and corrupt the update.
        def _pen(score: float) -> float:
            return 0.0 if score != score else score

        penalties = [
            mean(_pen(mon[m.name][i].score) for m in self.train_against) if self.train_against else 0.0
            for i in range(len(rollouts))
        ]
        pen_coef = self._penalty_coef()
        rewards = [er.task_reward - pen_coef * p for er, p in zip(env_results, penalties)]

        self._records.append(self._make_record(rollouts, env_results, mon, penalties, pen_coef, rewards))
        self._calls += 1

        return EnvironmentReturn(
            observations=[{"role": "environment", "content": ""} for _ in rollouts],
            metadata=metadata,
            next_stop_strings=[None] * len(rollouts),
            rewards=torch.tensor(rewards, dtype=torch.float32).cpu(),
            terminateds=torch.ones(len(rollouts), dtype=torch.float32).cpu(),
            answers=None,
        )

    def global_post_process_and_metrics(self, batch: BatchedDataDict) -> tuple[BatchedDataDict, dict]:
        """Required by the interface. NeMo-RL r0.7.0's GRPO loop never calls it, so all of this
        project's metrics are routed through ``pop_records`` instead — keeping one code path rather
        than two that could disagree."""
        return batch, {}

    # -- driver-facing --------------------------------------------------------------------------
    def pop_records(self) -> list[dict[str, Any]]:
        """Hand over (and clear) the per-call metric rows + rollout dumps buffered since the last
        call. The driver stamps them with the authoritative step number and writes the jsonl."""
        records, self._records = self._records, []
        return records

    def monitor_names(self) -> list[str]:
        return [m.name for m in self.scored]

    def behavior_name(self) -> str:
        return self.env.behavior_name

    def _make_record(self, rollouts, env_results, mon, penalties, pen_coef, rewards) -> dict[str, Any]:
        """One buffered record: the shared-schema metric row (step filled in by the driver) plus the
        per-rollout dump, in exactly the format ``rl/train.py`` writes."""
        names = [m.name for m in self.scored]
        if self.role == "val":
            row = eval_row(
                step=-1, rollouts=rollouts, env_results=env_results, monitor_results=mon,
                monitor_names=names, env=self.env, max_tokens=self.cfg.max_tokens,
            )
        else:
            row = train_row(
                step=-1, rollouts=rollouts, env_results=env_results, monitor_results=mon,
                train_against_names=[m.name for m in self.train_against], env=self.env,
                max_tokens=self.cfg.max_tokens, penalties=penalties, pen_coef=pen_coef,
                rewards=rewards,
            )
        dumps = []
        for i, (r, er) in enumerate(zip(rollouts, env_results)):
            scores = {n: {"score": mon[n][i].score, "label": mon[n][i].label} for n in names}
            common = {
                "question": r.prompt.text, "cot": r.cot, "answer": r.output, "monitors": scores,
                "task_id": (r.prompt.meta or {}).get("task_id"),
            }
            if self.role == "val":
                dumps.append({
                    **common,
                    "behavior_present": bool(er.behavior_present),  # = strict oracle
                    "loose_rh": bool(er.meta["loose_rh"]) if "loose_rh" in er.meta else None,
                    "hardcoding": bool(er.meta["hardcoding"]) if "hardcoding" in er.meta else None,
                    "unparsed": er.meta.get("unparsed"),
                    "env_meta": er.meta,
                })
            else:
                dumps.append({
                    **common,
                    "env": {"task_reward": er.task_reward, "behavior_present": er.behavior_present,
                            "choice": er.meta.get("choice"), "unparsed": er.meta.get("unparsed"),
                            "meta": er.meta},
                    "reward": rewards[i],
                })
        # A train-against monitor that scored NOTHING means the reward has no penalty signal — the
        # run has silently become a no-penalty control. Flagged here, acted on in the driver.
        dead = [m.name for m in self.train_against
                if mon[m.name] and all(x.score != x.score for x in mon[m.name])]
        return {"row": row, "rollouts": dumps, "dead_train_against": dead,
                "n_rollouts": len(rollouts)}
