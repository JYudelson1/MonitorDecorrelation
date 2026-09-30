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

  # per-monitor overrides — monitors.<name|model:substr|*>.<field>, e.g. the judge-side reasoning of
  # every gemini-3.5 judge, or turning the gemini-2.5 judges' reasoning off / giving them a bigger
  # budget. `reasoning` is an OpenRouter reasoning object (JSON — quote it), validated per judge model
  # (monitors/judge_reasoning.py), and replaced whole:
  uv run python experiments/run_experiment.py --config <cfg> \
      --set 'monitors.model:gemini-3.5.reasoning={"effort":"medium"}' run_name=<...>_eff-medium
  uv run python experiments/run_experiment.py --config <cfg> \
      --set 'monitors.model:gemini-2.5.reasoning={"enabled":false}' run_name=<...>_g25-off

The config is schema-validated (pydantic, extra keys forbidden) — a malformed config fails fast.

Run directories and resuming
  A run writes to data/runs/<run_name>/. Launching a run whose directory already exists FAILS (nothing
  in it is touched) unless --resume is given. To reuse a name whose old run you no longer want, delete
  its directory first:
      rm -rf data/runs/<run_name>
  (its tinker checkpoints live on tinker and are not deleted by this; manage them with
  `uv run tinker checkpoint list` / `uv run tinker checkpoint delete <tinker://…>`.)

  --resume continues a run from its LATEST saved training state (weights + optimizer): the final one
  of a finished run, else the last save_every checkpoint of a crashed / killed one (those expire
  after 4 weeks). Same config, with n_steps greater than the step resumed from:
      # a crashed run, to its original length (the config it saved):
      uv run python experiments/run_experiment.py --config data/runs/<run_name>/config.json --resume
      # a finished 90-step run → 180 steps:
      uv run python experiments/run_experiment.py --config data/runs/<run_name>/config.json \
          --set n_steps=180 --resume
  It appends to the same directory (first dropping whatever was logged after the saved state),
  equivalent to having run n_steps in one go (details: rl/train.py, "Resuming"). Every other config
  field must equal the saved state's, and --resume on a name with no saved state fails.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from monitordecorrelation.config import LoggingConfig, RunConfig
from monitordecorrelation.envs.factory import make_env
from monitordecorrelation.experiment_config import (
    apply_overrides,
    build_monitors,
    load_config,
    validate_token_budgets,
)
from monitordecorrelation.hyperparams import get_lr
from monitordecorrelation.rl.train import check_resumable, load_resume_state, run_grpo

load_dotenv()


def _wandb_logged_in() -> bool:
    """True iff a wandb credential is configured *locally* — the ``WANDB_API_KEY`` env var, or a
    ``~/.netrc`` entry for the wandb host (which is exactly what ``wandb login`` writes). This is a pure
    local lookup: no network call, no key verification, no interactive prompt — so it's safe to call at
    the start of a detached/headless batch without any risk of hanging."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    host = os.environ.get("WANDB_HOST") or os.environ.get("WANDB_BASE_URL") or "api.wandb.ai"
    host = host.split("://", 1)[-1].split("/", 1)[0]  # tolerate a full URL → bare hostname for netrc
    import netrc

    try:
        nf = os.environ.get("WANDB_NETRC")  # wandb honours this override; mirror it
        rc = netrc.netrc(nf) if nf else netrc.netrc()
        auth = rc.authenticators(host)
        return bool(auth and auth[2])  # auth = (login, account, password); password = the api key
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False


def _resolve_wandb_mode() -> str:
    """Sync to wandb iff this machine is logged in — otherwise stay offline. An explicit ``WANDB_MODE``
    env var (online/offline/disabled) always wins, as a power-user override / escape hatch."""
    return os.environ.get("WANDB_MODE") or ("online" if _wandb_logged_in() else "offline")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="path to a JSON or YAML experiment config")
    ap.add_argument("--set", nargs="*", default=[], metavar="key=value",
                    help="override top-level config fields (e.g. --set run_name=quick n_steps=2), or a "
                         "per-monitor field via monitors.<name|model:substr|*>.<field> (e.g. --set "
                         "'monitors.model:gemini-3.5.reasoning={\"effort\":\"medium\"}'), or one env option via "
                         "env_options.<key> (e.g. --set env_options.verifier_mode=possible)")
    ap.add_argument("--resume", action="store_true",
                    help="continue the run in data/runs/<run_name>/ from its latest saved training state "
                         "(weights + optimizer: the final one, or the last save_every one of an unfinished "
                         "run) up to this config's n_steps (must exceed that state's step); every other "
                         "config field must match the run's. Without it, an existing run dir is an error.")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)

    # The run directory decides fresh-vs-resume, before anything costs time or touches the disk.
    # `run.log` alone does not count as a run: scripts/queue_runs.sh creates it just before launching.
    run_dir = Path("data/runs") / cfg.run_name
    existing = run_dir.is_dir() and any(p.name != "run.log" for p in run_dir.iterdir())
    resume_state = None
    if not args.resume and existing:
        raise SystemExit(
            f"{run_dir} already exists — refusing to overwrite a previous run named {cfg.run_name!r}.\n"
            f"  • to CONTINUE that run from its latest saved state, add --resume (finished run: and a "
            f"larger n_steps);\n"
            f"  • to start over under this name, delete the old run first:  rm -rf {run_dir}\n"
            f"  • or pick another name:  --set run_name=<new name>"
        )
    if args.resume:
        if not existing:
            raise SystemExit(f"--resume: there is no run to resume in {run_dir} (nothing is started from "
                             "scratch under --resume; drop the flag to start a new run)")
        if cfg.backend != "tinker":
            raise SystemExit(f"--resume needs the tinker backend (backend={cfg.backend!r} does not save "
                             "optimizer state)")
        resume_state = load_resume_state(run_dir)
        # Everything but n_steps must be what the run ran with (both sides JSON-normalised, which is
        # how the snapshot stored it).
        now = json.loads(json.dumps(cfg.model_dump()))
        then = resume_state["config"]
        diff = sorted(k for k in now.keys() | then.keys() if k != "n_steps" and now.get(k) != then.get(k))
        if diff:
            raise SystemExit(
                f"--resume: the config differs from the run's in {', '.join(diff)}:\n"
                + "\n".join(f"  {k}: run had {then.get(k)!r}, now {now.get(k)!r}" for k in diff)
            )
        try:
            check_resumable(resume_state, n_steps=cfg.n_steps, penalty_schedule=cfg.penalty_schedule,
                            stop_after_zero_behavior_steps=cfg.stop_after_zero_behavior_steps)
        except ValueError as e:
            raise SystemExit(f"--resume: {e}") from e
        print(f"  resuming from the step-{resume_state['steps_done']} state "
              f"({'final' if resume_state['eval_done'] else 'save_every'}): {resume_state['state_checkpoint']}")

    # LR: the config's explicit value, else TM's LoRA-LR heuristic. That heuristic is only calibrated
    # for some families (it refuses Inkling outright), so translate its exception into instructions
    # rather than a bare traceback 30 seconds before the run would have started.
    if cfg.lr is not None:
        lr = cfg.lr
    else:
        try:
            lr = get_lr(cfg.policy)
        except Exception as e:
            raise SystemExit(
                f"No learning rate: tinker-cookbook's LoRA-LR heuristic does not cover "
                f"{cfg.policy!r} ({type(e).__name__}: {e}). Set \"lr\" explicitly in the config "
                f"(or --set lr=2e-4)."
            ) from e

    # Env FIRST: it decides how a turn is sampled, so the token-budget keys can only be checked
    # once it exists — and a config error should cost nothing, i.e. land before the backend opens a
    # tinker session.
    try:
        env = make_env(cfg)  # each env validates its own subset / env_options values
        think_budget = validate_token_budgets(cfg, env)  # int | None from here on
    except ValueError as e:
        raise SystemExit(f"config {args.config}: {e}") from e

    # Backend
    if cfg.backend == "tinker":
        from monitordecorrelation.backends.tinker_backend import TinkerBackend
        backend = TinkerBackend(cfg.policy, lora_rank=cfg.lora_rank, learning_rate=lr, seed=cfg.seed,
                                kl_coef=cfg.kl_coef,
                                # None exactly when kl_coef is 0, i.e. when the discount is unused.
                                kl_discount_factor=cfg.kl_discount_factor or 0.0,
                                thinking_effort=cfg.thinking_effort,
                                resume_from=resume_state["state_checkpoint"] if resume_state else None)
    else:
        from monitordecorrelation.backends.transformers_backend import TransformersBackend
        # kl_coef / thinking_effort are rejected by ExperimentConfig for this backend (it implements
        # neither), so everything the config sets here is actually used.
        backend = TransformersBackend(cfg.policy, lora_rank=cfg.lora_rank, learning_rate=lr,
                                      seed=cfg.seed)
    probe_server_url = cfg.probe_server_url or os.environ.get("PROBE_SERVER_URL")
    # Multi-turn (agentic) envs get AgentCoTMonitor judges — a chat transcript of the episode — in
    # place of the single-turn CoTMonitor; every other env is unchanged.
    train_against, held_out = build_monitors(cfg.monitors, default_behavior=env.behavior_name,
                                             probe_server_url=probe_server_url,
                                             multi_turn=bool(getattr(env, "multi_turn", False)))
    if probe_server_url:
        print(f"  probes → shared server {probe_server_url}")

    # Write the EFFECTIVE config (after --set overrides are applied) into the run folder, so a run is
    # trivially + exactly reproducible — copying the raw source file would drop the overrides:
    #   uv run python experiments/run_experiment.py --config data/runs/<run>/config.<ext>
    # On --resume this replaces the run's config with the one that now describes the directory
    # (the larger n_steps), in the format the run already used.
    run_dir.mkdir(parents=True, exist_ok=True)
    effective = cfg.model_dump()
    as_yaml = (Path(args.config).suffix.lower() in (".yaml", ".yml") if not args.resume
               else (run_dir / "config.yaml").exists())
    if as_yaml:
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
    # Project: from the config file or --set wandb_project=…; else the LoggingConfig default.
    _wandb_project = cfg.wandb_project or LoggingConfig.wandb_project
    _wandb_tags = [cfg.experiment, cfg.env, short, f"seed{cfg.seed}", *(ta_names or ["control"])]

    run_config = RunConfig(
        env=cfg.env, backend=cfg.backend, base_model=cfg.policy, rl_algo="grpo",
        batch_size=cfg.batch_size, group_size=cfg.group_size, n_steps=cfg.n_steps,
        eval_every=cfg.eval_every, eval_size=cfg.eval_size,
        eval_samples_per_prompt=cfg.eval_samples_per_prompt,
        penalty_coef=cfg.penalty_coef, penalty_schedule=cfg.penalty_schedule, kl_coef=cfg.kl_coef,
        kl_discount_factor=cfg.kl_discount_factor, lora_rank=cfg.lora_rank, learning_rate=lr,
        seed=cfg.seed, save_every=cfg.save_every,
        stop_after_zero_behavior_steps=cfg.stop_after_zero_behavior_steps,
        logging=LoggingConfig(run_name=cfg.run_name, wandb_mode=_resolve_wandb_mode(),
                              wandb_project=_wandb_project,
                              wandb_group=_wandb_group, wandb_tags=_wandb_tags, log_fraction=1.0),
    )

    def names(ms):
        return ", ".join(getattr(m, "name", "?") for m in ms) or "(none)"

    print(f"[{cfg.experiment}] run_name={cfg.run_name} policy={cfg.policy} backend={cfg.backend} lr={lr:.2e}")
    print(f"  wandb: {run_config.logging.wandb_mode} project={run_config.logging.wandb_project}"
          + (" (syncing — logged in)" if run_config.logging.wandb_mode == "online" else " (local only)"))
    subset_note = f" subset={cfg.subset}" if cfg.subset is not None else ""
    print(f"  env={cfg.env} behavior={env.behavior_name} | {cfg.batch_size}x{cfg.group_size} "
          f"rollouts/step x {cfg.n_steps} steps{subset_note}")
    print(f"  train-against: {names(train_against)}  |  held-out: {names(held_out)}")
    if getattr(env, "multi_turn", False):
        src = "env default" if cfg.think_budget == "auto" else "config"
        print(f"  multi-turn: max_turns={getattr(env, 'max_turns', '?')} think_budget={think_budget} ({src}) "
              + (f"answer_tokens={cfg.answer_tokens}" if think_budget
                 else f"— NO thinking budget: max_tokens/turn={cfg.max_tokens}"))

    run_grpo(
        run_config, env, backend, train_against=train_against, held_out=held_out,
        max_tokens=cfg.max_tokens, think_budget=think_budget, answer_tokens=cfg.answer_tokens,
        run_info={"experiment": cfg.experiment, "subset": cfg.subset, "lr": lr,
                  "think_budget": think_budget,  # the RESOLVED value (config.json may say "auto")
                  "config": cfg.model_dump()},
        resume=resume_state,
    )
    print(f"\n{cfg.experiment} finished OK")


if __name__ == "__main__":
    main()
