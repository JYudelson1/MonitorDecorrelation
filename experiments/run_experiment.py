"""Generic, config-driven experiment runner — the single entry point for all degradation runs.

A run is fully described by a validated JSON/YAML config (see ``experiment_config.ExperimentConfig``
and ``experiments/configs/``): policy, env, RL hyperparams, and a list of monitors (CoT judges and/or
white-box probes), each ``train_against`` or ``held_out``. This replaces the per-experiment scripts
(train_against_cot / train_against_probe) — those are now just different configs.

Usage:
  uv run python experiments/run_experiment.py --config experiments/configs/train_against_cot.json
  uv run python experiments/run_experiment.py --config experiments/configs/full_matrix.yaml

  # one-off overrides (handy for quick tests; everything else comes from the config):
  uv run python experiments/run_experiment.py --config <cfg> --set run_name=quick n_steps=2

The config is schema-validated (pydantic, extra keys forbidden) — a malformed config fails fast.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from monitordecorrelation.config import LoggingConfig, RunConfig
from monitordecorrelation.envs.factory import make_env
from monitordecorrelation.experiment_config import build_monitors, load_config
from monitordecorrelation.hyperparams import get_lr
from monitordecorrelation.rl.train import run_grpo

load_dotenv()


def _wandb_logged_in() -> bool:
    """True iff a wandb credential is configured *locally* — the ``WANDB_API_KEY`` env var, or a
    ``~/.netrc`` entry for the wandb host (which is exactly what ``wandb login`` writes). This is a pure
    local lookup: no network call, no key verification, no interactive prompt — so it's safe to call at
    the start of a detached/headless batch without any risk of hanging."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    host = (
        os.environ.get("WANDB_HOST")
        or os.environ.get("WANDB_BASE_URL")
        or "api.wandb.ai"
    )
    host = host.split("://", 1)[-1].split("/", 1)[
        0
    ]  # tolerate a full URL → bare hostname for netrc
    import netrc

    try:
        nf = os.environ.get("WANDB_NETRC")  # wandb honours this override; mirror it
        rc = netrc.netrc(nf) if nf else netrc.netrc()
        auth = rc.authenticators(host)
        return bool(
            auth and auth[2]
        )  # auth = (login, account, password); password = the api key
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False


def _resolve_wandb_mode() -> str:
    """Sync to wandb iff this machine is logged in — otherwise stay offline. An explicit ``WANDB_MODE``
    env var (online/offline/disabled) always wins, as a power-user override / escape hatch."""
    return os.environ.get("WANDB_MODE") or (
        "online" if _wandb_logged_in() else "offline"
    )


def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config", required=True, help="path to a JSON or YAML experiment config"
    )
    ap.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="key=value",
        help="override top-level config fields (e.g. --set run_name=quick n_steps=2)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.set:
        overrides = {k: _coerce(v) for k, v in (kv.split("=", 1) for kv in args.set)}
        cfg = cfg.model_copy(update=overrides)

    lr = cfg.lr if cfg.lr is not None else get_lr(cfg.policy)

    # Backend
    if cfg.backend == "tinker":
        from monitordecorrelation.backends.tinker_backend import TinkerBackend

        backend = TinkerBackend(
            cfg.policy,
            lora_rank=cfg.lora_rank,
            learning_rate=lr,
            seed=cfg.seed,
            kl_coef=cfg.kl_coef,
            kl_discount_factor=cfg.kl_discount_factor,
            thinking_effort=cfg.thinking_effort,
        )
    else:
        from monitordecorrelation.backends.transformers_backend import (
            TransformersBackend,
        )

        backend = TransformersBackend(
            cfg.policy, lora_rank=cfg.lora_rank, learning_rate=lr
        )

    env = make_env(cfg)
    probe_server_url = cfg.probe_server_url or os.environ.get("PROBE_SERVER_URL")
    train_against, held_out = build_monitors(
        cfg.monitors,
        default_behavior=env.behavior_name,
        probe_server_url=probe_server_url,
    )
    if probe_server_url:
        print(f"  probes → shared server {probe_server_url}")

    # Write the EFFECTIVE config (after --set overrides are applied) into the run folder, so a run is
    # trivially + exactly reproducible — copying the raw source file would drop the overrides:
    #   uv run python experiments/run_experiment.py --config data/runs/<run>/config.<ext>
    run_dir = Path("data/runs") / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    effective = cfg.model_dump()
    if Path(args.config).suffix.lower() in (".yaml", ".yml"):
        import yaml

        (run_dir / "config.yaml").write_text(yaml.safe_dump(effective, sort_keys=False))
    else:
        (run_dir / "config.json").write_text(json.dumps(effective, indent=2))

    # W&B grouping: every run of one sweep (the matrix's monitor rows × seeds) shares a group so they
    # cluster in the UI; tags make them filterable by env / model / seed / which monitor was trained
    # against (or "control"). Group is per (experiment, model) so multiple models stay separate.
    short = cfg.policy.split("/")[-1]
    ta_names = [m.name for m in cfg.monitors if m.role == "train_against"]
    _wandb_group = f"{cfg.experiment}/{short}"
    _wandb_tags = [
        cfg.experiment,
        cfg.env,
        short,
        f"seed{cfg.seed}",
        *(ta_names or ["control"]),
    ]

    run_config = RunConfig(
        env=cfg.env,
        backend=cfg.backend,
        base_model=cfg.policy,
        rl_algo="grpo",
        batch_size=cfg.batch_size,
        group_size=cfg.group_size,
        n_steps=cfg.n_steps,
        eval_every=cfg.eval_every,
        eval_size=cfg.eval_size,
        eval_samples_per_prompt=cfg.eval_samples_per_prompt,
        penalty_coef=cfg.penalty_coef,
        penalty_schedule=cfg.penalty_schedule,
        kl_coef=cfg.kl_coef,
        kl_discount_factor=cfg.kl_discount_factor,
        lora_rank=cfg.lora_rank,
        learning_rate=lr,
        seed=cfg.seed,
        logging=LoggingConfig(
            run_name=cfg.run_name,
            wandb_mode=_resolve_wandb_mode(),
            wandb_group=_wandb_group,
            wandb_tags=_wandb_tags,
            log_fraction=1.0,
        ),
    )

    def names(ms):
        return ", ".join(getattr(m, "name", "?") for m in ms) or "(none)"

    print(
        f"[{cfg.experiment}] run_name={cfg.run_name} policy={cfg.policy} backend={cfg.backend} lr={lr:.2e}"
    )
    print(
        f"  wandb: {run_config.logging.wandb_mode}"
        + (
            " (syncing — logged in)"
            if run_config.logging.wandb_mode == "online"
            else " (local only)"
        )
    )
    subset_note = (
        f" subset={cfg.subset}" if cfg.env in ("sycophancy", "impossiblebench") else ""
    )
    print(
        f"  env={cfg.env} behavior={env.behavior_name} | {cfg.batch_size}x{cfg.group_size} "
        f"rollouts/step x {cfg.n_steps} steps{subset_note}"
    )
    print(f"  train-against: {names(train_against)}  |  held-out: {names(held_out)}")

    run_grpo(
        run_config,
        env,
        backend,
        train_against=train_against,
        held_out=held_out,
        max_tokens=cfg.max_tokens,
        run_info={
            "experiment": cfg.experiment,
            "subset": cfg.subset,
            "lr": lr,
            "config": cfg.model_dump(),
        },
    )
    print(f"\n{cfg.experiment} finished OK")


if __name__ == "__main__":
    main()
