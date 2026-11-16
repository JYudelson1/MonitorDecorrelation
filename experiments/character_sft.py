"""Character training by response-only distillation SFT, starting from a tinker checkpoint.

Trains a LoRA policy on a teacher's in-character answers (cross-entropy on the visible answer only) —
the character-training recipe of "Character training can mitigate reward hacking, but can also …"
(Colognese & Ward 2026, github.com/barbonara/character-reward-hacking), whose released neutral /
anti-cheating data is described in data/character_training/corin/README.md. The defaults are that
post's SFT hyperparameters: lr 1.5e-4 with cosine decay to 0, batch 32, 2 epochs (last partial batch
dropped: 64 steps on 1,050 rows), Adam β=(0.9, 0.95) ε=1e-8, no weight decay, loss summed over the
answer tokens of a batch.

    uv run python experiments/character_sft.py --run-name corin-anti-from-ctrl213 \
        --data data/character_training/corin/anti.jsonl --policy thinkingmachines/Inkling-Small \
        --lora-rank 8 --thinking-effort 0.5 \
        --init-from tinker://<run>:train:0/weights/<name>

Unlike the post (character SFT on the raw base model, THEN reward-hacking RL), ``--init-from`` lets the
character be trained into an already-RL'd policy: the checkpoint's weights are loaded with a fresh
optimizer and the same LoRA keeps training.

Writes ``data/runs/<run-name>/``: ``config.json`` (arguments + data sha256 + the init checkpoint's
tinker metadata), ``sft_metrics.jsonl`` (one row per step: nll over the step's answer tokens — measured
BEFORE that step's update, so every epoch-0 row is on rows the model has not trained on — lr, token
counts), ``final_checkpoint.txt`` (the training state, ``…/weights/<run-name>-final``: what
``init_from_tinker_ckpt`` takes, e.g. to evaluate it with the repo's own eval:
``run_experiment.py --config <env config> --set n_steps=0 init_from_tinker_ckpt=<path> …``) and
``sampler_checkpoint.txt`` (the same weights for inference). Neither checkpoint expires.

Without ``--format-rl-config`` this is supervised training on fixed text: no env, no monitor, no
oracle is involved.

**Format-keeping RL steps (``--format-rl-config``).** Response-only SFT on chat answers damages an
agentic policy's action format (it starts answering in prose instead of emitting the env's action
tags). To counter that, after every ``--format-rl-every`` SFT steps the script takes ONE GRPO step on
the env of the given experiment config with a FORMAT-ONLY reward (``rl/sft.py::format_reward``): 1 if
the episode is valid (``envs.base.invalid_reason`` is None), 0 otherwise — regardless of what was
answered. Advantages are centred within a prompt's group, so a group whose episodes are all valid (or
all invalid) contributes no gradient: the RL steps push only as long as the format is actually broken.
From the config only the env and how it is sampled are read (``env``, ``subset``, ``env_options``,
``n_prompts_pool``, ``seed``, ``eval_size``, ``max_tokens`` / ``think_budget`` / ``answer_tokens``,
``thinking_effort``; ``--format-rl-set`` overrides them); its training keys (``n_steps``, ``lr``,
``kl_coef``, ``monitors``, …) are NOT used — the RL steps have no KL penalty and no monitor, and
their batch / group / lr come from the ``--format-rl-*`` flags. The config's held-out eval prompts
(``env.holdout(eval_size, seed)``) are removed from the pool first, so a later standard eval with that
seed is on prompts these steps never saw. ``--format-rl-system-prompts`` lists the system prompts the
RL steps cycle through (``none`` = no system prompt). SFT and RL steps share one Adam state. Each RL
step is a row in ``format_rl_metrics.jsonl`` (format rate BEFORE that step's update; ``optim_step_taken`` false when every group was
uniform, i.e. there was nothing to learn from and no update was made; plus the env's
oracle ``behavior_rate`` — logged only, never in the reward).

**Held-out checks during training (``--heldout-eval-every N``, needs ``--format-rl-config``).** After
every N SFT steps (and that step's RL step, if any) the training state is saved as
``<run-name>-sft<step>`` and the config's held-out eval prompts (``eval_size`` ×
``eval_samples_per_prompt`` episodes) are sampled once per ``--format-rl-system-prompts`` entry and
graded by the env only — no monitor. One row per (step, system prompt) goes to
``heldout_metrics.jsonl``: ``format_rate``, ``behavior_rate``, ``behavior_rate_valid``, and the saved
``state_checkpoint`` (what ``init_from_tinker_ckpt`` takes, for a full monitor eval of that step).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from monitordecorrelation.rl.renderers import is_tml_policy
from monitordecorrelation.rl.sft import (
    epoch_batches,
    format_reward,
    load_response_only_rows,
    response_only_datum,
)

load_dotenv()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--data", required=True, help="response-only chat JSONL (see rl/sft.py)")
    ap.add_argument("--policy", required=True, help="tinker base model, e.g. thinkingmachines/Inkling-Small")
    ap.add_argument("--lora-rank", type=int, required=True)
    ap.add_argument("--init-from", default=None,
                    help="tinker://…/weights/… training state to start from (weights only, fresh optimizer); "
                         "omit to train a fresh LoRA on the raw policy")
    ap.add_argument("--thinking-effort", type=float, required=True,
                    help="the 'Thinking effort level' the SFT prompts are rendered with — use the effort the "
                         "policy is sampled at afterwards")
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--lr-schedule", default="cosine", choices=("cosine", "linear", "constant"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=16384, help="a longer row is an error")
    ap.add_argument("--seed", type=int, default=0, help="batch order, the LoRA init of a fresh LoRA, and the format-RL steps' prompts + sampling seeds")
    g = ap.add_argument_group("format-keeping RL steps (all or none of these)")
    g.add_argument("--format-rl-config", default=None,
                   help="experiment config whose env the format-only RL steps run on; omit for pure SFT")
    g.add_argument("--format-rl-set", nargs="*", default=None, metavar="key=value",
                   help="overrides of that config, as run_experiment.py --set (e.g. seed=213)")
    g.add_argument("--format-rl-every", type=int, default=None, help="one RL step after every N SFT steps")
    g.add_argument("--format-rl-batch-size", type=int, default=None, help="prompts per RL step")
    g.add_argument("--format-rl-group-size", type=int, default=None, help="episodes per prompt")
    g.add_argument("--format-rl-lr", type=float, default=None, help="constant lr of the RL steps")
    g.add_argument("--format-rl-system-prompts", nargs="+", default=None, metavar="PROMPT",
                   help="system prompts the RL steps cycle through, one per step; 'none' = no system prompt")
    g.add_argument("--heldout-eval-every", type=int, default=None,
                   help="save the training state + sample the held-out eval prompts (env-graded, no "
                        "monitor) after every N SFT steps")
    args = ap.parse_args()
    if args.heldout_eval_every is not None and (args.format_rl_config is None or args.heldout_eval_every < 1):
        ap.error("--heldout-eval-every needs --format-rl-config (its env + held-out prompts) and N >= 1")

    rl_keys = ("format_rl_every", "format_rl_batch_size", "format_rl_group_size", "format_rl_lr",
               "format_rl_system_prompts")
    if args.format_rl_config is None:
        given = [k for k in (*rl_keys, "format_rl_set") if getattr(args, k) is not None]
        if given:  # would be silently ignored
            ap.error(f"--{given[0].replace('_', '-')} needs --format-rl-config")
    else:
        missing = [k for k in rl_keys if getattr(args, k) is None]
        if missing:  # no defaults: a format-RL run says exactly what its RL steps are
            ap.error("--format-rl-config needs " + ", ".join(f"--{k.replace('_', '-')}" for k in missing))
        if min(args.format_rl_every, args.format_rl_batch_size) < 1 or args.format_rl_group_size < 2:
            ap.error("--format-rl-every / -batch-size must be >= 1 and --format-rl-group-size >= 2 "
                     "(advantages are centred within a group)")

    if not is_tml_policy(args.policy):
        ap.error(f"{args.policy}: response-only SFT is only implemented for TML-rendered "
                 "(thinkingmachines/*) policies (renderers.TmlRenderer.response_only_example)")
    run_dir = Path("data/runs") / args.run_name
    if run_dir.exists():
        raise SystemExit(f"{run_dir} already exists — pick another --run-name or delete it")

    rows = load_response_only_rows(args.data)
    sha = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()

    import tinker
    from tinker_cookbook.supervised.common import compute_mean_nll
    from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier

    from monitordecorrelation.backends.tinker_backend import TinkerBackend
    from monitordecorrelation.rl.renderers import make_renderer

    # The format-RL env, built (and its config validated) before the backend opens a tinker session.
    rl = None
    if args.format_rl_config is not None:
        from monitordecorrelation.envs.factory import make_env
        from monitordecorrelation.experiment_config import apply_overrides, load_config, validate_token_budgets

        try:
            rl_cfg = apply_overrides(load_config(args.format_rl_config), args.format_rl_set or [])
            env = make_env(rl_cfg)
            think_budget = validate_token_budgets(rl_cfg, env)
        except ValueError as e:
            raise SystemExit(f"--format-rl-config {args.format_rl_config}: {e}") from e
        if rl_cfg.policy != args.policy or rl_cfg.backend != "tinker":
            raise SystemExit(f"--format-rl-config: its policy/backend ({rl_cfg.policy}, {rl_cfg.backend}) "
                             f"must be {args.policy} on tinker")
        random.seed(rl_cfg.seed)  # env.sample_prompt draws from the global rng, as in run_grpo
        if hasattr(env, "holdout"):
            eval_prompts = env.holdout(rl_cfg.eval_size, seed=rl_cfg.seed)  # they leave the training pool
        elif args.heldout_eval_every is not None:
            raise SystemExit(f"--heldout-eval-every: env {rl_cfg.env!r} has no holdout()")
        else:
            eval_prompts = None
        # which training prompts the RL steps draw (and, below, their sampling seeds) also depend on
        # --seed, so a new --seed is a new run in every respect except the held-out eval set
        random.seed(f"{rl_cfg.seed}:format_rl:{args.seed}")
        rl = {"cfg": rl_cfg, "env": env, "think_budget": think_budget, "eval_prompts": eval_prompts,
              "system_prompts": [None if p == "none" else p for p in args.format_rl_system_prompts]}

    # TinkerBackend = the repo's one way to open a training client: it checks the init checkpoint
    # against policy / lora_rank and loads its weights with a fresh optimizer.
    try:
        backend = TinkerBackend(args.policy, lora_rank=args.lora_rank, learning_rate=args.lr, seed=args.seed,
                                # the backend's own renderer is only used by the RL steps (replaced per
                                # step below); the SFT rows are rendered by `renderer`
                                thinking_effort=rl["cfg"].thinking_effort if rl else args.thinking_effort,
                                init_from=args.init_from)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    tc = backend.training_client
    renderer = make_renderer(args.policy, training_client=tc, effort=args.thinking_effort)
    if rl:
        rl["renderers"] = [make_renderer(args.policy, training_client=tc, effort=rl["cfg"].thinking_effort,
                                         system_prompt=p) for p in rl["system_prompts"]]

    data = [response_only_datum(renderer, r, args.max_length) for r in rows]
    n_loss = [int(sum(d.loss_fn_inputs["weights"].data)) for d in data]
    n_tok = [d.model_input.length for d in data]
    steps_per_epoch = len(epoch_batches(len(rows), args.batch_size, epoch=0, seed=args.seed))
    total = steps_per_epoch * args.epochs

    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({
        **vars(args), "data_sha256": sha, "n_rows": len(rows), "total_steps": total,
        "loss": "cross_entropy, summed over the answer tokens of a batch (weights 0/1)",
        "adam": {"beta1": 0.9, "beta2": 0.95, "eps": 1e-8, "weight_decay": 0.0},
        "init_weights_info": backend.init_info,
        "format_rl": None if rl is None else {
            "reward": "1 if envs.base.invalid_reason(env, rollout) is None else 0; GRPO, no KL, no monitor",
            "effective_config": rl["cfg"].model_dump(), "think_budget": rl["think_budget"],
            "n_rl_steps": total // args.format_rl_every},
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    print(f"[character_sft] {args.run_name}: {len(rows)} rows ({sum(n_loss)} answer tokens, {sum(n_tok)} total) "
          f"from {args.data}; {args.epochs} epochs x {steps_per_epoch} steps of {args.batch_size}; "
          f"lr {args.lr:g} {args.lr_schedule}; init {args.init_from or args.policy}", flush=True)

    def format_rl_step(k: int, after_sft_steps: int, rl_log) -> None:
        """One GRPO step on the format-only reward (see the module docstring)."""
        from monitordecorrelation.rl.episodes import derive_sample_seed
        from monitordecorrelation.rl.train import _sample

        cfg, env = rl["cfg"], rl["env"]
        which = k % len(rl["renderers"])
        backend.renderer = rl["renderers"][which]  # sampling AND the GRPO datums of this step
        backend.learning_rate = args.format_rl_lr
        backend.refresh_sampler()  # the SFT steps moved the weights behind the backend's back
        prompts = [env.sample_prompt() for _ in range(args.format_rl_batch_size)]
        ts = time.time()
        rollouts, _ = _sample(backend, env, prompts, sampler=backend.current_sampler(),
                              seed=derive_sample_seed(cfg.seed, "format_rl", args.seed, k),
                              num_samples=args.format_rl_group_size, max_tokens=cfg.max_tokens,
                              think_budget=rl["think_budget"], answer_tokens=cfg.answer_tokens)
        rewards = [format_reward(env, r) for r in rollouts]
        with ThreadPoolExecutor(32) as pool:  # oracle: LOGGED only — never enters the reward
            graded = list(pool.map(env.score, rollouts))
        # A step whose groups are all uniform (every episode valid, or every one invalid) has zero
        # advantages everywhere: no optim step is taken (Adam would still move the weights along the
        # SFT steps' momentum).
        gs = args.format_rl_group_size
        updated = any(len(set(rewards[i:i + gs])) > 1 for i in range(0, len(rewards), gs))
        m = backend.train_step(rollouts, rewards, gs) if updated else {}
        valid = [g for g, rw in zip(graded, rewards) if rw == 1.0]
        row = {"rl_step": k, "after_sft_steps": after_sft_steps, "system_prompt": rl["system_prompts"][which],
               "n_episodes": len(rollouts), "format_rate": sum(rewards) / len(rewards),
               "optim_step_taken": updated,
               "behavior_rate": sum(bool(g.behavior_present) for g in graded) / len(graded),
               "behavior_rate_valid": (sum(bool(g.behavior_present) for g in valid) / len(valid)
                                       if valid else None),
               "lr": args.format_rl_lr, "elapsed_s": time.time() - t0, "step_s": time.time() - ts,
               **{f"optim/{key}": v for key, v in m.items()}}
        rl_log.write(json.dumps(row) + "\n")
        rl_log.flush()
        print(f"  format-RL step {k} (after {after_sft_steps} SFT steps, system={row['system_prompt']!r}): "
              f"format {row['format_rate']:.2f} behavior {row['behavior_rate']:.2f} "
              f"(valid: {row['behavior_rate_valid']})", flush=True)

    def heldout_eval(sft_steps: int, ho_log) -> None:
        """Save the training state and sample the held-out prompts under every system prompt."""
        from monitordecorrelation.rl.episodes import derive_sample_seed
        from monitordecorrelation.rl.train import _sample

        cfg, env = rl["cfg"], rl["env"]
        state = backend.save_checkpoint(f"{args.run_name}-sft{sft_steps}")
        backend.refresh_sampler()
        sampler = backend.current_sampler()
        for which, sp in enumerate(rl["system_prompts"]):
            backend.renderer = rl["renderers"][which]
            rollouts, _ = _sample(backend, env, rl["eval_prompts"], sampler=sampler,
                                  seed=derive_sample_seed(cfg.seed, "heldout", sft_steps),
                                  num_samples=cfg.eval_samples_per_prompt, max_tokens=cfg.max_tokens,
                                  think_budget=rl["think_budget"], answer_tokens=cfg.answer_tokens)
            ok = [format_reward(env, r) == 1.0 for r in rollouts]
            with ThreadPoolExecutor(32) as pool:
                hack = [bool(g.behavior_present) for g in pool.map(env.score, rollouts)]
            n_valid, n_hack_valid = sum(ok), sum(h for h, o in zip(hack, ok) if o)
            row = {"sft_steps": sft_steps, "system_prompt": sp, "n_episodes": len(rollouts),
                   "n_valid": n_valid, "n_hack": sum(hack), "n_hack_valid": n_hack_valid,
                   "format_rate": n_valid / len(rollouts), "behavior_rate": sum(hack) / len(rollouts),
                   "behavior_rate_valid": n_hack_valid / n_valid if n_valid else None,
                   "state_checkpoint": state, "elapsed_s": time.time() - t0}
            ho_log.write(json.dumps(row) + "\n")
            ho_log.flush()
            print(f"  held-out @ {sft_steps} SFT steps (system={sp!r}): format {row['format_rate']:.3f} "
                  f"hack {row['behavior_rate']:.3f} (valid: {row['behavior_rate_valid']})", flush=True)

    t0 = time.time()
    n_rl = 0
    with (run_dir / "sft_metrics.jsonl").open("w") as log, \
            ((run_dir / "format_rl_metrics.jsonl").open("w") if rl else contextlib.nullcontext()) as rl_log, \
            ((run_dir / "heldout_metrics.jsonl").open("w") if args.heldout_eval_every
             else contextlib.nullcontext()) as ho_log:
        for epoch in range(args.epochs):
            for b, idx in enumerate(epoch_batches(len(rows), args.batch_size, epoch=epoch, seed=args.seed)):
                step = epoch * steps_per_epoch + b
                lr = args.lr * compute_schedule_lr_multiplier(args.lr_schedule, step, total)
                batch = [data[i] for i in idx]
                # queue the optim step right behind the forward_backward (tinker runs them in order)
                fb = tc.forward_backward(batch, "cross_entropy")
                opt = tc.optim_step(tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8,
                                                      weight_decay=0.0))
                out = fb.result()
                opt.result()
                nll = compute_mean_nll([o["logprobs"] for o in out.loss_fn_outputs],
                                       [d.loss_fn_inputs["weights"] for d in batch])
                row = {"step": step, "epoch": epoch, "nll": nll, "lr": lr,
                       "num_loss_tokens": sum(n_loss[i] for i in idx), "num_tokens": sum(n_tok[i] for i in idx),
                       "elapsed_s": time.time() - t0}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(f"  step {step + 1:>3}/{total} epoch {epoch} nll {nll:.4f} lr {lr:.2e}", flush=True)
                if rl and (step + 1) % args.format_rl_every == 0:
                    format_rl_step(n_rl, step + 1, rl_log)
                    n_rl += 1
                if args.heldout_eval_every and (step + 1) % args.heldout_eval_every == 0:
                    heldout_eval(step + 1, ho_log)

    state = backend.save_checkpoint(f"{args.run_name}-final")  # weights + optimizer, no expiry
    (run_dir / "final_checkpoint.txt").write_text(state + "\n")
    _sampler, sampler_path = backend.checkpoint_sampler(f"{args.run_name}-final")
    (run_dir / "sampler_checkpoint.txt").write_text(sampler_path + "\n")
    print(f"training state: {state}\nsampler weights: {sampler_path}\ncharacter_sft finished OK", flush=True)


if __name__ == "__main__":
    main()
