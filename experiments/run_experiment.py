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

  # per-monitor overrides — monitors.<name|model:substr|*>.<field>, e.g. the judge-side reasoning
  # effort of every gemini-3.5 judge (the gemini-2.5 judges must stay reasoning-off and will refuse):
  uv run python experiments/run_experiment.py --config <cfg> \
      --set monitors.model:gemini-3.5.reasoning_effort=medium run_name=<...>_eff-medium

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
from monitordecorrelation.experiment_config import (
    CoTMonitorSpec,
    ProbeMonitorSpec,
    build_monitors,
    load_config,
    resolve_think_budget,
)
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


def _coerce(v: str):
    if v.lower() in ("null", "none", ""):
        return None  # e.g. --set think_budget=null → NO thinking budget (one call per turn, no env default)
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


def apply_overrides(cfg, sets: list[str]):
    """Apply ``--set key=value`` overrides and RE-VALIDATE. ``model_copy(update=…)`` skips validation,
    so a typo'd key or an out-of-range value would sail through and fail deep inside the run (or, worse,
    train something subtly different); round-tripping through the schema keeps ``--set`` as strict as
    the config file itself.

    A key of the form ``monitors.<selector>.<field>`` overrides a field on the matching monitor(s)
    instead of a top-level field — so per-run judge settings (notably ``reasoning_effort`` for the
    gemini-3.x judges) are a launch flag, not a forked config file. The selector is a monitor
    ``name``, ``model:<substring of model_id>``, or ``*`` for every monitor::

        --set monitors.g35_out.reasoning_effort=medium          # one judge, by name
        --set monitors.model:gemini-3.5.reasoning_effort=medium # every gemini-3.5 judge
        --set monitors.*.threshold=0.6                          # all of them

    A selector that matches nothing is an error (a silently-ignored override is how you end up
    analysing a run that trained against something else). The result is re-validated like any other
    override, so e.g. pointing ``reasoning_effort`` at a gemini-2.5 judge fails loudly right here.
    """
    if not sets:
        return cfg
    overrides: dict = {}
    monitor_overrides: list[tuple[str, str, object]] = []
    for kv in sets:
        key, value = kv.split("=", 1)
        if key.startswith("monitors."):
            selector, _, field = key[len("monitors."):].rpartition(".")
            if not selector or not field:
                raise SystemExit(
                    f"--set: {key!r} is not a monitor override; expected "
                    "monitors.<name|model:substr|*>.<field>=<value>"
                )
            monitor_overrides.append((selector, field, _coerce(value)))
        else:
            overrides[key] = _coerce(value)
    unknown = set(overrides) - set(type(cfg).model_fields)
    if unknown:
        raise SystemExit(f"--set: unknown config field(s) {sorted(unknown)}")
    data = cfg.model_dump()
    for selector, field, value in monitor_overrides:
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
            mon[field] = value
    return type(cfg).model_validate({**data, **overrides})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="path to a JSON or YAML experiment config")
    ap.add_argument("--set", nargs="*", default=[], metavar="key=value",
                    help="override top-level config fields (e.g. --set run_name=quick n_steps=2), or a "
                         "per-monitor field via monitors.<name|model:substr|*>.<field> (e.g. --set "
                         "monitors.model:gemini-3.5.reasoning_effort=medium)")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)

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

    # Backend
    if cfg.backend == "tinker":
        from monitordecorrelation.backends.tinker_backend import TinkerBackend
        backend = TinkerBackend(cfg.policy, lora_rank=cfg.lora_rank, learning_rate=lr, seed=cfg.seed,
                                kl_coef=cfg.kl_coef, kl_discount_factor=cfg.kl_discount_factor,
                                thinking_effort=cfg.thinking_effort)
    else:
        from monitordecorrelation.backends.transformers_backend import TransformersBackend
        backend = TransformersBackend(cfg.policy, lora_rank=cfg.lora_rank, learning_rate=lr)

    env = make_env(cfg)
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
        seed=cfg.seed,
        logging=LoggingConfig(run_name=cfg.run_name, wandb_mode=_resolve_wandb_mode(),
                              wandb_project=_wandb_project,
                              wandb_group=_wandb_group, wandb_tags=_wandb_tags, log_fraction=1.0),
    )

    def names(ms):
        return ", ".join(getattr(m, "name", "?") for m in ms) or "(none)"

    print(f"[{cfg.experiment}] run_name={cfg.run_name} policy={cfg.policy} backend={cfg.backend} lr={lr:.2e}")
    print(f"  wandb: {run_config.logging.wandb_mode} project={run_config.logging.wandb_project}"
          + (" (syncing — logged in)" if run_config.logging.wandb_mode == "online" else " (local only)"))
    subset_note = f" subset={cfg.subset}" if cfg.env in ("sycophancy", "impossiblebench") else ""
    print(f"  env={cfg.env} behavior={env.behavior_name} | {cfg.batch_size}x{cfg.group_size} "
          f"rollouts/step x {cfg.n_steps} steps{subset_note}")
    print(f"  train-against: {names(train_against)}  |  held-out: {names(held_out)}")
    think_budget = resolve_think_budget(cfg.think_budget, env)  # int | None from here on
    if getattr(env, "multi_turn", False):
        src = "env default" if cfg.think_budget == "auto" else "config"
        print(f"  multi-turn: max_turns={getattr(env, 'max_turns', '?')} think_budget={think_budget} ({src}) "
              f"answer_tokens={cfg.answer_tokens}"
              + ("" if think_budget else f" — NO thinking budget: max_tokens/turn={cfg.max_tokens}"))

    run_grpo(
        run_config, env, backend, train_against=train_against, held_out=held_out,
        max_tokens=cfg.max_tokens, think_budget=think_budget, answer_tokens=cfg.answer_tokens,
        run_info={"experiment": cfg.experiment, "subset": cfg.subset, "lr": lr,
                  "think_budget": think_budget,  # the RESOLVED value (config.json may say "auto")
                  "config": cfg.model_dump()},
    )
    print(f"\n{cfg.experiment} finished OK")


if __name__ == "__main__":
    main()
