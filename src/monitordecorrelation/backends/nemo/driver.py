"""NeMo-RL entry point for this project — the thing that actually runs GRPO on the ``nemo`` backend.

Runs inside NeMo-RL's virtualenv (see ``launcher.py``, which is what starts it). It is deliberately
the same shape as NeMo-RL's own ``examples/run_grpo.py`` / ``examples/run_grpo_sliding_puzzle.py``:
load the config, build the datasets and the environment, ``setup()``, ``grpo_train()``. Everything
project-specific is in three places:

1. the datasets are this repo's ``Env`` prompt pool + its train-disjoint holdout (``dataset.py``);
2. the environment is this repo's ``Env`` + ``Monitor``s, computing the monitor-penalised reward
   (``env_actor.py``);
3. ``logger.log_metrics`` is wrapped so that every training step and every validation writes the
   project's own ``metrics.jsonl`` / ``eval_metrics.jsonl`` / ``rollouts.jsonl`` /
   ``eval_rollouts.jsonl`` — the exact artifacts ``eval/degradation.py`` and ``eval/plots.py`` read.
   That hook is also where the authoritative step number lives: NeMo-RL gives it to the logger, not
   to the environment.

Usage (normally via ``experiments/run_experiment.py --set backend=nemo``):

    python -m monitordecorrelation.backends.nemo.driver \\
        --experiment-config experiments/configs/impossiblebench_nemo.json \\
        --nemo-config experiments/configs/nemo/grpo.yaml \\
        [hydra.style.overrides=...]
"""

from __future__ import annotations

import argparse
import json
import os
import pprint
import sys
from pathlib import Path
from typing import Any

import ray
from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES, init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)

from monitordecorrelation.backends.nemo.dataset import PromptDataset
from monitordecorrelation.backends.nemo.env_actor import MonitorDecorrelationEnv
from monitordecorrelation.envs.factory import make_env
from monitordecorrelation.experiment_config import load_config as load_experiment_config

TASK_NAME = "monitordecorrelation"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-config", required=True, help="this repo's ExperimentConfig")
    ap.add_argument("--nemo-config", required=True, help="the NeMo-RL config (MD_* env-parametrized)")
    ap.add_argument("--run-dir", default=None, help="default: data/runs/<run_name>")
    return ap.parse_known_args()


class RunWriter:
    """Writes the project's run artifacts, driven by NeMo-RL's ``logger.log_metrics`` calls.

    ``prefix="train"`` -> one training step finished; ``prefix="validation"`` -> one held-out eval
    finished. In both cases the corresponding environment actor has exactly one buffered record
    (one ``step()`` call per batch, ``max_rollout_turns=1``), which we stamp with the step number
    NeMo-RL just reported and append to the jsonl.
    """

    def __init__(self, run_dir: Path, train_env, val_env, resuming: bool = False) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.envs = {"train": train_env, "validation": val_env}
        # Truncate on a fresh run (as rl/train.py does), append when NeMo-RL resumed from a
        # checkpoint — otherwise a resume would silently drop the steps already on disk.
        mode = "a" if resuming else "w"
        self.files = {
            "train": (run_dir / "metrics.jsonl").open(mode),
            "validation": (run_dir / "eval_metrics.jsonl").open(mode),
        }
        self.rollout_files = {
            "train": (run_dir / "rollouts.jsonl").open(mode),
            "validation": (run_dir / "eval_rollouts.jsonl").open(mode),
        }

    def flush_role(self, role: str, step: int) -> dict[str, float]:
        """Drain one role's buffered records, write them, and return the metric row so the caller
        can also push it into NeMo-RL's own loggers (W&B / tensorboard)."""
        merged: dict[str, float] = {}
        for record in ray.get(self.envs[role].pop_records.remote()):
            if record["dead_train_against"]:
                # Unlike a held-out monitor (a measurement), a dead train-against monitor means the
                # reward has no penalty term — the run has silently become a no-penalty control.
                # Aborting beats burning GPU-hours on a mislabelled run.
                raise RuntimeError(
                    f"train-against monitor(s) {record['dead_train_against']} failed to score ANY of "
                    f"{record['n_rollouts']} rollouts at step {step} — aborting (no training signal). "
                    f"Check the monitor / API / model id."
                )
            row = {**record["row"], "step": step}
            self.files[role].write(json.dumps(row) + "\n")
            for dump in record["rollouts"]:
                self.rollout_files[role].write(
                    json.dumps({"step": step, **dump}, default=str) + "\n"
                )
            merged.update({k: v for k, v in row.items() if k != "step"})
        self.files[role].flush()
        self.rollout_files[role].flush()
        return merged

    def close(self) -> None:
        for f in (*self.files.values(), *self.rollout_files.values()):
            f.close()


def attach_run_writer(logger, writer: RunWriter) -> None:
    """Wrap ``logger.log_metrics`` so the project's rows are written (and also logged to W&B) at the
    exact moment NeMo-RL reports a step. Train steps are reported 1-based (``total_steps + 1``);
    we store them 0-based so a run lines up with the tinker backend's ``metrics.jsonl``."""
    original = logger.log_metrics

    def log_metrics(metrics, step, prefix="", *args, **kwargs):
        role = "train" if prefix == "train" else "validation" if prefix == "validation" else None
        if role is not None:
            our_step = max(0, step - 1) if role == "train" else step
            extra = writer.flush_role(role, our_step)
            metrics = {**metrics, **extra}
        return original(metrics, step, prefix, *args, **kwargs)

    logger.log_metrics = log_metrics


def make_env_actor(experiment_cfg, role: str):
    """One environment actor. It runs in the driver's own interpreter (``PY_EXECUTABLES.SYSTEM``)
    because it needs this repo importable, and takes no GPU — grading is subprocess/API work."""
    return MonitorDecorrelationEnv.options(
        num_gpus=0,
        runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM, "env_vars": dict(os.environ)},
    ).remote({"experiment_config": experiment_cfg.model_dump(), "role": role})


def main() -> None:
    args, overrides = parse_args()
    register_omegaconf_resolvers()

    experiment_cfg = load_experiment_config(args.experiment_config)
    run_dir = Path(args.run_dir or f"data/runs/{experiment_cfg.run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.nemo_config)
    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)
    resolved = OmegaConf.to_container(config, resolve=True)
    # The record of what NeMo-RL was actually told, next to the experiment config it came from.
    (run_dir / "nemo_config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(resolved)))
    master_config = MasterConfig(**resolved)
    print("Final NeMo-RL config:")
    pprint.pprint(resolved)

    init_ray()
    set_seed(master_config.grpo["seed"])

    tokenizer = get_tokenizer(master_config.policy["tokenizer"])
    master_config.policy["generation"] = configure_generation_config(
        master_config.policy["generation"], tokenizer
    )

    # ---- data: the env's prompt pool, materialised ------------------------------------------
    # holdout() first, exactly as rl/train.py does, so the eval set is carved out before the
    # training prompts are drawn (envs whose holdout removes items from the pool depend on it).
    env = make_env(experiment_cfg)
    eval_prompts = env.holdout(experiment_cfg.eval_size, seed=experiment_cfg.seed)
    n_train = master_config.grpo["max_num_steps"] * master_config.grpo["num_prompts_per_step"]
    train_prompts = [env.sample_prompt() for _ in range(n_train)]
    per_prompt = max(1, experiment_cfg.eval_samples_per_prompt)
    val_prompts = [p for p in eval_prompts for _ in range(per_prompt)]
    print(f"  ✓ {len(train_prompts)} training prompts, {len(val_prompts)} eval rollouts "
          f"({len(eval_prompts)} held-out prompts x {per_prompt})", flush=True)

    dataset = PromptDataset(train_prompts, tokenizer, TASK_NAME, experiment_cfg.max_prompt_tokens)
    val_dataset = PromptDataset(val_prompts, tokenizer, TASK_NAME, experiment_cfg.max_prompt_tokens)

    # ---- environments -----------------------------------------------------------------------
    train_env = make_env_actor(experiment_cfg, "train")
    val_env = make_env_actor(experiment_cfg, "val")
    task_to_env = {TASK_NAME: train_env}
    val_task_to_env = {TASK_NAME: val_env}

    (
        policy, policy_generation, _nemo_gym, _cluster, dataloader, val_dataloader, loss_fn,
        logger, checkpointer, grpo_state, master_config, _teachers, _aliases,
    ) = setup(master_config, tokenizer, dataset, val_dataset)

    writer = RunWriter(run_dir, train_env, val_env,
                       resuming=grpo_state.get("current_step", 0) > 0)
    attach_run_writer(logger, writer)
    _write_run_info(run_dir, experiment_cfg, resolved, ray.get(val_env.monitor_names.remote()),
                    ray.get(val_env.behavior_name.remote()))

    try:
        grpo_train(
            policy, policy_generation, dataloader, val_dataloader, tokenizer, loss_fn,
            task_to_env, val_task_to_env, logger, checkpointer, grpo_state, master_config,
        )
    finally:
        writer.close()
    print("\nnemo backend finished OK")


def _write_run_info(run_dir: Path, cfg, nemo_config: dict[str, Any], monitor_names, behavior) -> None:
    """``run_info.json`` in the schema ``eval/degradation.py`` expects — which detector this run
    trained against is the row label of the degradation matrix, so it has to be durable."""
    import datetime

    train_against = [m.name for m in cfg.monitors if m.role == "train_against"]
    info = {
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_name": cfg.run_name,
        "policy": cfg.policy,
        "backend": "nemo",
        "experiment": cfg.experiment,
        "subset": cfg.subset,
        "config": cfg.model_dump(),
        "nemo_config": nemo_config,
        "env": {"name": cfg.env, "behavior_name": behavior},
        "train_against": [{"name": n, "role": "train_against"} for n in train_against],
        "held_out": [{"name": n, "role": "held_out"} for n in monitor_names if n not in train_against],
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
