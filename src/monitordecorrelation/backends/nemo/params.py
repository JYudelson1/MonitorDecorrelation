"""Derive the ``MD_*`` knobs that parametrize ``experiments/configs/nemo/grpo.yaml``.

One place, pure functions, no heavy imports — so the whole mapping from "what the experiment config
says" to "what NeMo-RL is told" is unit-testable without a GPU, without Ray and without NeMo-RL
installed (see ``tests/test_nemo_params.py``).

The *values* here are the measured 2xB200 Qwen3.5 recipes (parallel layout, micro-batch size, vLLM
memory fraction, MoE backend) generalised to an arbitrary GPU count and sequence length; the
*semantics* (advantage estimator, KL placement, optimizer, LoRA alpha) live in the YAML and mirror
``backends/tinker_backend.py``. See the header of that YAML for the parity argument.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from monitordecorrelation.experiment_config import ExperimentConfig

# Tokens per packed training micro-batch. The 2xB200 recipes were measured at
# max_total_sequence_length 4096 with train_micro_batch_size 4 (dense) and 8 (MoE); we hold that
# token budget fixed and let the micro-batch size fall out of the actual sequence length, so a
# 16k-completion run does not silently multiply the activation footprint by 5.
MICRO_BATCH_TOKEN_BUDGET = {"dense": 4096 * 4, "moe": 4096 * 8}

# --- vLLM memory fraction under colocated generation -------------------------------------------
# ``gpu_memory_utilization`` is a fraction of TOTAL GPU memory, taken up front: vLLM initialises
# before the policy worker, so it sees an empty GPU and claims ``gmu * total``. Everything the
# trainer holds then has to live in what is left. The binding moment is vLLM's ``wake_up()`` after a
# training step, when it re-maps that whole reservation while the trainer still holds its weights —
# which is why this is tested over TWO RL steps, never one.
#
# So the budget is:
#     gmu * total  +  policy_weights/tp  +  vllm_residue  +  overhead   <=   total
#
# The weight term is exact (the Hub's safetensors metadata, see ``policy_weight_gib``) and divides by
# the tensor-parallel size. Measured at refit ("GPU Memory after refit complete", i.e. the trainer's
# residual while vLLM is awake), against 2 * params / tp:
#     Qwen3.5-4B-Base       tp1   predicted  8.7 GiB   measured  8.49
#     Qwen3.5-9B-Base       tp1   predicted 18.0 GiB   measured 17.54
#     Qwen3.5-9B-Base       tp2   predicted  9.0 GiB   measured  8.79
#     Qwen3.5-35B-A3B-Base  tp2   predicted 33.5 GiB   measured 32.75
# A consistent 0.975 ratio on four points, so the un-discounted value is used and is slightly
# conservative. Note the count is of the WHOLE checkpoint including the vision tower that
# megatron-bridge builds but this text-only task never runs — which is what the trainer actually
# holds, and why a name-parsed "4B" (4.66B real) would be wrong.
#
# The second term is vLLM's own non-discardable state. Sleeping does NOT return everything: the
# weights are backed up to CPU and the KV pool is discarded, but CUDA graphs, the MoE workspaces and
# the Gated-DeltaNet state blocks (sized by max_num_seqs, not by the rollout count) stay resident and
# must be paid for a SECOND time when wake_up() re-maps the whole reservation. It scales with the
# model. Measured as (memory outside the reservation at wake_up) - (trainer reserved):
#     Qwen3.5-4B-Base       8.7 GiB weights   ->  0.6 GiB
#     Qwen3.5-35B-A3B-Base 67.0 GiB weights   -> 10.2 GiB
# i.e. ~0.16 x the checkpoint, which is what VLLM_RESIDUE_FRACTION encodes.
VLLM_RESIDUE_FRACTION = 0.16

# Flat margin on top, for CUDA context, NCCL buffers, Megatron's LoRA grad buffers and allocator
# fragmentation. See MEASURED_FIT below for what this buys.
TRAINER_OVERHEAD_GIB = 6.0

# MEASURED_FIT (2026-08-18, 2xB200 179.06 GiB/GPU, ImpossibleBench, seq 20480, TWO RL steps so the
# step-2 wake_up() is actually exercised — a one-step run passes settings that die on step 2):
#   model                 GPUs  gmu   result  peak/GPU  free   KV cache
#   Qwen3.5-4B-Base       1     0.90  fits    165.7     13.4   138.9 GiB / 1.14M tokens
#   Qwen3.5-4B-Base       2     0.90  fits    170.2      8.9   143.8 GiB / 2.36M tokens
#   Qwen3.5-9B-Base       2     0.85  fits    169.6      9.5   129.4 GiB / 2.12M tokens  (was 84.8 at 0.6)
#   Qwen3.5-35B-A3B-Base  2     0.78  OOM     170.3       --   cumem_allocator.cpp wake_up, step 2
#   Qwen3.5-35B-A3B-Base  2     0.72  fits    165.5     13.5    82.2 GiB / 2.15M tokens  (was 51.9 at 0.55)
# These are the values the BUDGET produces; what is actually emitted is 5% lower still, see
# GPU_MEMORY_SAFETY_FACTOR below.
# The 35B boundary is therefore between 0.72 and 0.78 and the rule lands on 0.72, with 13.5 GiB of
# measured headroom at the peak. That headroom is the point of TRAINER_OVERHEAD_GIB: these arms run
# 2x4 = 8 rollouts where a real run is 8x8 = 64, and while the trainer's residual is weights-only
# (it does not grow with the rollout count) vLLM's scheduler state does, so the margin is deliberate
# rather than tuned away.

# Never hand vLLM the entire GPU even when the arithmetic allows it: this is also the backstop if
# the detected total or the weight count is wrong.
MAX_GPU_MEMORY_UTILIZATION = 0.90
MIN_GPU_MEMORY_UTILIZATION = 0.30

# Applied to the capped fraction, so every derived value is 5% below the largest one measured to
# fit. The MEASURED_FIT table above is what the budget alone produces; running that close to the
# ceiling leaves the allocator no room to breathe, and the cost of being wrong is a hard crash at
# step-2 wake_up() rather than a slowdown. Concretely: 0.90 -> 0.855, 0.85 -> 0.807, 0.72 -> 0.684.
# The fallback constants are not scaled — they already sit well below the derived budget.
GPU_MEMORY_SAFETY_FACTOR = 0.95

# Used only when the policy's size cannot be determined (no network and an unparseable name). These
# are the hand-tuned values this backend shipped with, measured on the 2xB200 Qwen3.5 recipes.
FALLBACK_GPU_MEMORY_UTILIZATION = {"dense": 0.6, "moe": 0.55}

SEQUENCE_LENGTH_ROUND = 64  # NeMo-RL's sequence_packing.sequence_length_round

# NOTE (measured 2026-08-18, do not redo): NeMo-RL hands ``policy.make_sequence_length_divisible_by``
# to the sequence packer as ``sequence_length_pad_multiple`` (lm_policy.Policy.__init__), so it also
# quantises every training micro-batch's tensor length. Leaving it at the tensor-parallel size (1 for
# a dense policy) means each micro-batch has an arbitrary length and Dynamo re-traces on it — 16 new
# recompiles on step 1, 7 on step 2, and Megatron's fused ops guard on the exact packed length
# ("tensor 'qkv' size mismatch at index 1. expected 18157, actual 17557").
# That is real, but it is WARMUP, not a steady-state cost: torch's automatic_dynamic_shapes marks the
# dim dynamic after the second distinct value, and an 8-step 4B/2-GPU run converges to 0 new
# recompiles by step 3 and stays at ~4.8s / ~15.1k tokens/s/gpu of policy_training.
# Rounding to a coarse multiple was tried (1024 and 2048) and is NOT worth it:
#   converged policy_training (steps 3,4,5,7)   pad=1: 4.85s / 15150 tok/s/gpu
#                                            pad=2048: 5.10s / 14520 tok/s/gpu   (~5% slower — the
#                                                      padding overhead lands directly on the step)
# and it does not remove the occasional recompile either: a step that produced a genuinely new length
# still cost 45.3s (pad=1) vs 32.8s (pad=2048), because coarse buckets only relocate which bucket is
# novel. So this stays at the natural value.


def detect_n_gpus() -> int:
    """Number of GPUs visible on this machine (the default for ``cfg.n_gpus``).

    Uses ``nvidia-smi`` rather than torch so this stays importable in a CPU-only checkout; honours
    ``CUDA_VISIBLE_DEVICES`` the way the training job will. Returns 1 when nothing can be detected,
    which keeps the config renderable on a laptop.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [d for d in visible.split(",") if d.strip() != ""]
        return max(1, len(devices))
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return 1
    try:
        out = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=30, check=True)
    except (subprocess.SubprocessError, OSError):
        return 1
    return max(1, sum(1 for line in out.stdout.splitlines() if line.startswith("GPU ")))


def is_moe(policy: str) -> bool:
    """True for a mixture-of-experts policy, by the ``-<total>B-A<active>B-`` naming convention
    (Qwen3.5-35B-A3B-Base, Qwen3-30B-A3B, …). MoE changes the expert-parallel layout, the vLLM MoE
    backend and the micro-batch budget; nothing else. Override with
    ``nemo_options: {"moe": true/false}`` for a model that does not follow the convention."""
    return bool(re.search(r"\d+b-a\d+(\.\d+)?b", policy.lower()))


def parallel_layout(policy: str, n_gpus: int, moe: bool) -> dict[str, Any]:
    """(tensor, expert, expert-tensor) parallel sizes + sequence-parallel, for one node of GPUs.

    Dense: TP1, so every GPU is a data-parallel rank — a dense model under LoRA holds its weights
    comfortably, and DP halves the gradient-accumulation chain instead of splitting each microbatch.

    Measured on Qwen3.5-9B-Base, 2xB200, seq 20480, mbs 1, gbs 8, converged steps only (steps 1-2 are
    torch.compile warmup and must be discarded — see the SEQUENCE_LENGTH_ROUND note). Training-phase
    time per step and per-GPU throughput:
        TP1 x DP2 (this)   6.18s / 6.22s   12,097 / 12,033 tok/s/GPU   peak 129.4 / 128.0 GB
        TP2 x DP1          8.28s            9,047 tok/s/GPU            peak 118.9 GB
    TP1 is 25% faster (34% more throughput), reproduced across two runs 0.8% apart, and the TP2 arm
    ran with the warmer inductor cache, so the gap is if anything understated. TP2 does buy ~10 GB of
    headroom (sequence-parallel activation sharding + the vocab column-parallel split) — that is the
    trade, and it only becomes worth taking when a model is close to the memory ceiling, which is
    exactly why the MoE branch below keeps TP2.

    MoE on >= 2 GPUs: TP2 x ETP1 x EP2 -> DP1. TP1 was measured three times on the 35B-A3B recipe
    and loses badly — dropping TP also drops the sequence-parallel activation sharding and the
    vocab column-parallel split, roughly doubling the per-rank footprint, which that close to the
    183 GB ceiling turns into allocator thrash and leaves the trainer too fat for vLLM to wake up.
    """
    if moe and n_gpus >= 2:
        # ETP1 means each expert lives whole on one rank, so expert-parallel spans the world:
        # EP = world / ETP. On 2 GPUs that is the recipe's TP2 x ETP1 x EP2 -> DP1.
        etp = 1
        return {"tp": 2, "ep": n_gpus // etp, "etp": etp, "sequence_parallel": True}
    return {"tp": 1, "ep": 1, "etp": 1, "sequence_parallel": False}


def detect_gpu_memory_gib() -> float | None:
    """Total memory of one GPU, GiB, or None when it cannot be determined.

    None is meaningful: it makes ``gpu_memory_utilization`` fall back to the hand-tuned constants
    rather than size a budget against a guessed GPU. Uses ``nvidia-smi`` for the same reason as
    ``detect_n_gpus`` — no torch import, works in a CPU-only checkout."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    sizes = [float(line) for line in out.stdout.split() if line.strip().replace(".", "").isdigit()]
    return min(sizes) / 1024 if sizes else None  # MiB -> GiB; min = the GPU that will bind first


@lru_cache(maxsize=None)
def policy_weight_gib(policy: str) -> float | None:
    """bf16 weight footprint of the whole checkpoint, GiB — exact, or None if it can't be found.

    Read from the Hub's **safetensors metadata**, which is a metadata-only request (no weights) and
    is architecture-agnostic: it counts every tensor actually in the checkpoint, including MoE
    experts and the vision tower that megatron-bridge builds for the Qwen3.5 family. That is exactly
    what the trainer holds, and it is why this does not parse the size out of the model name — the
    name says "4B" where the checkpoint is 4.66B, a 16% error straight into the memory budget.

    There is precedent for a launch-time network call (``hyperparams.get_lr`` does one). On failure
    this returns None and the caller falls back, so an offline machine still renders a config."""
    try:
        from huggingface_hub import get_safetensors_metadata
    except ImportError:
        return None
    # The metadata read draws a per-shard progress bar, which would interleave with the run banner
    # on every single launch. Silence it just for this call, restoring whatever the caller had.
    from huggingface_hub.utils import (
        are_progress_bars_disabled,
        disable_progress_bars,
        enable_progress_bars,
    )

    was_disabled = are_progress_bars_disabled()
    disable_progress_bars()
    try:
        counts = get_safetensors_metadata(policy).parameter_count
    except Exception:  # noqa: BLE001 — offline, gated repo, no safetensors: all mean "unknown"
        return None
    finally:
        if not was_disabled:
            enable_progress_bars()
    total = sum(counts.values())
    return 2 * total / 2**30 if total else None  # bf16


def gpu_memory_utilization(
    weight_gib: float | None,
    tp: int,
    moe: bool,
    total_gib: float | None = None,
    overhead_gib: float = TRAINER_OVERHEAD_GIB,
) -> float:
    """The largest vLLM memory fraction that still leaves room for the trainer.

    Pure, so the budget is unit-testable without a GPU or a network. ``weight_gib``/``total_gib``
    of None mean "unknown" and select the conservative fallback constants.

    More KV cache is strictly better for generation throughput, so this maximises rather than tunes:
    the only question the number answers is whether the trainer still fits beside it."""
    if not weight_gib or not total_gib:
        return FALLBACK_GPU_MEMORY_UTILIZATION["moe" if moe else "dense"]
    # Everything that is resident when vLLM re-maps its reservation: the trainer's weights (sharded
    # by TP), vLLM's non-discardable state, and a flat margin.
    resident_gib = (
        weight_gib / max(1, tp) + VLLM_RESIDUE_FRACTION * weight_gib + overhead_gib
    )
    fraction = (total_gib - resident_gib) / total_gib
    # The safety factor is applied AFTER the cap, so it is a real 5% off whatever we would otherwise
    # have emitted — applying it first would let the cap swallow it for any model whose raw budget
    # already exceeds MAX (the 4B's raw fraction is 0.910).
    capped = min(MAX_GPU_MEMORY_UTILIZATION, fraction)
    return round(
        max(MIN_GPU_MEMORY_UTILIZATION, capped * GPU_MEMORY_SAFETY_FACTOR), 3
    )


def vllm_tensor_parallel_size(n_gpus: int, moe: bool) -> int:
    """vLLM's tensor-parallel size for colocated generation. 1 for dense, ``n_gpus`` for MoE.

    NeMo-RL derives the number of vLLM replicas as ``world_size // tensor_parallel_size``
    (``vllm_generation.py``: ``dp_size = cluster.world_size() // model_parallel_size``). So TP1 on an
    N-GPU node is not "use one GPU" — it is N independent engines, each holding a full copy of the
    weights, with the step's rollouts sharded across them. For a dense policy that is strictly better
    than one TP-sharded engine: generation is memory-bandwidth-bound and embarrassingly parallel over
    requests, so replicating trades a few GiB of duplicated weights for the removal of an all-reduce
    on every decoded token. Measured faster on this repo's dense runs, which is why it is the default.

    MoE stays sharded. A 35B-A3B replica is ~67 GiB of weights inside a ~122 GiB vLLM budget, and the
    whole MoE generation path here (the triton backend, the max_num_seqs cap, the Gated-DeltaNet state
    blocks and the refit) was measured at TP = n_gpus; replicating it is untested and much tighter."""
    return n_gpus if moe else 1


def round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def sequence_length(max_tokens: int, max_prompt_tokens: int, tp: int) -> int:
    """Total sequence budget = prompt allowance + completion budget, rounded up so it satisfies both
    ``sequence_packing.sequence_length_round`` and ``make_sequence_length_divisible_by`` (= TP)."""
    return round_up(max_prompt_tokens + max_tokens, SEQUENCE_LENGTH_ROUND * max(1, tp))


def micro_batch_size(seq_len: int, moe: bool) -> int:
    return max(1, MICRO_BATCH_TOKEN_BUDGET["moe" if moe else "dense"] // seq_len)


def nemo_env_vars(
    cfg: "ExperimentConfig",
    *,
    lr: float,
    n_gpus: int | None = None,
    run_dir: str = "",
    wandb_enabled: bool = False,
    wandb_group: str | None = None,
    wandb_tags: list[str] | None = None,
) -> dict[str, str]:
    """The full ``MD_*`` environment for one run, ready to hand to the NeMo-RL driver.

    ``cfg.nemo_options`` is the escape hatch and is applied LAST, so anything derived here can be
    overridden per-run (keys may be given bare — ``{"tp": 2}`` — or fully qualified — ``{"MD_TP": 2}``).
    """
    opts = dict(cfg.nemo_options or {})
    n_gpus = int(opts.pop("n_gpus", None) or n_gpus or cfg.n_gpus or detect_n_gpus())
    moe = bool(opts.pop("moe", is_moe(cfg.policy)))
    layout = parallel_layout(cfg.policy, n_gpus, moe)
    seq_len = sequence_length(cfg.max_tokens, cfg.max_prompt_tokens, layout["tp"])
    # vLLM gets everything the trainer does not need (see the budget note at the top of this file).
    # Both inputs are overridable so an offline machine, or a policy whose checkpoint the Hub cannot
    # be asked about, can still be sized by hand.
    weight_gib = opts.pop("policy_weight_gib", None) or policy_weight_gib(cfg.policy)
    total_gib = opts.pop("gpu_total_gib", None) or detect_gpu_memory_gib()
    gmu = gpu_memory_utilization(weight_gib, layout["tp"], moe, total_gib)
    # MoE needs Triton (vLLM's default SM100 FlashInfer-TRTLLM MoE backend swizzles expert weights
    # into a layout the refit weight_loader cannot index) and a decode-concurrency setting; a dense
    # policy takes no vllm_kwargs at all.
    # NOTE on max_num_seqs, left at 1024 deliberately: the 2xB200 recipe this came from describes it
    # as a *cap* on vLLM's then-default of 1024, but the pinned vLLM (0.20) defaults to 128
    # (config/scheduler.py DEFAULT_MAX_NUM_SEQS), so as written it RAISES decode concurrency 8x.
    # That matters because each concurrent decode needs a Gated-DeltaNet state block, and those
    # blocks are part of the non-discardable residue VLLM_RESIDUE_FRACTION above pays for — so a
    # lower value would likely buy back KV cache. Not changed here; flagged for a separate decision.
    vllm_kwargs: dict[str, Any] = (
        {"moe_backend": "triton", "max_num_seqs": 1024} if moe else {}
    )

    env = {
        "MD_MODEL_NAME": cfg.policy,
        "MD_RUN_NAME": cfg.run_name,
        # --- RL schedule: one nemo "step" == one rl/train.py step -------------------------------
        "MD_NUM_PROMPTS_PER_STEP": cfg.batch_size,
        "MD_GROUP_SIZE": cfg.group_size,
        "MD_TRAIN_GLOBAL_BATCH_SIZE": cfg.batch_size * cfg.group_size,
        "MD_N_STEPS": cfg.n_steps,
        "MD_EVAL_EVERY": cfg.eval_every,
        # the held-out eval set, each prompt repeated eval_samples_per_prompt times
        "MD_VAL_SAMPLES": cfg.eval_size * max(1, cfg.eval_samples_per_prompt),
        "MD_SEED": cfg.seed,
        # --- optimization ----------------------------------------------------------------------
        "MD_LR": float(lr),
        "MD_LORA_RANK": cfg.lora_rank,
        "MD_LORA_ALPHA": 32,  # tinker fixes LoRA alpha at 32 regardless of rank
        "MD_KL_COEF": float(cfg.kl_coef),
        # --- sequence budget -------------------------------------------------------------------
        "MD_MAX_TOKENS": cfg.max_tokens,
        "MD_MAX_TOTAL_SEQUENCE_LENGTH": seq_len,
        "MD_TRAIN_MICRO_BATCH_SIZE": micro_batch_size(seq_len, moe),
        # --- topology --------------------------------------------------------------------------
        "MD_GPUS_PER_NODE": n_gpus,
        "MD_TP": layout["tp"],
        "MD_EP": layout["ep"],
        "MD_ETP": layout["etp"],
        "MD_SEQUENCE_PARALLEL": layout["sequence_parallel"],
        "MD_VLLM_TP": vllm_tensor_parallel_size(n_gpus, moe),
        "MD_GPU_MEMORY_UTILIZATION": gmu,
        "MD_VLLM_KWARGS": vllm_kwargs,
        # --- bookkeeping -----------------------------------------------------------------------
        "MD_CHECKPOINT_DIR": f"{run_dir}/nemo_checkpoints" if run_dir else "results/grpo",
        "MD_SAVE_PERIOD": cfg.n_steps,  # only the final step, like the tinker backend's one save
        "MD_LOG_DIR": f"{run_dir}/nemo_logs" if run_dir else "logs",
        "MD_WANDB_ENABLED": wandb_enabled,
        "MD_WANDB_PROJECT": "monitor-decorrelation",
        "MD_WANDB_GROUP": wandb_group,
        "MD_WANDB_TAGS": list(wandb_tags or []),
    }
    for key, value in opts.items():
        env[key if key.startswith("MD_") else f"MD_{key.upper()}"] = value
    return {k: _as_env_value(v) for k, v in env.items()}


# OmegaConf's value grammar accepts an unquoted token wherever it is unambiguous. A quoted string is
# only legal as a *value*, never as a dict KEY — `{"a": 1}` is a parse error where `{a: 1}` is fine —
# so JSON is not a valid encoding for the dict-valued knobs and we render literals ourselves.
_BARE_STRING = re.compile(r"^[A-Za-z_][A-Za-z0-9_./-]*$")


def _as_env_value(value: Any) -> str:
    """Render a Python value as the string the config's ``${oc.decode:${oc.env:…}}`` parses back.

    Top-level strings are passed through untouched (a model name, a run name), so the handful of
    plain ``${oc.env:…}``-without-decode keys read naturally in the config too.
    """
    return value if isinstance(value, str) else _literal(value)


def _literal(value: Any) -> str:
    """One value in OmegaConf's literal grammar (also valid YAML flow style)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    if isinstance(value, str):
        # Quote anything that could be read as a number, a bool, or punctuation.
        return value if _BARE_STRING.match(value) else json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        # Keys stay bare — quoting them is a GrammarParseError in OmegaConf.
        return "{" + ", ".join(f"{k}: {_literal(v)}" for k, v in value.items()) + "}"
    raise TypeError(f"cannot render {value!r} ({type(value).__name__}) as a NeMo-RL config value")
