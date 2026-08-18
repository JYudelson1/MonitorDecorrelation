"""The nemo backend's config derivation — the part that must be right before a GPU is touched.

Covers (a) the parity contract with the tinker backend (batch/step/seed/LoRA/KL propagation) and
(b) the parallelism + memory table lifted from the measured 2xB200 Qwen3.5 recipes.
"""

from __future__ import annotations

import pytest

from monitordecorrelation.backends.nemo.params import (
    gpu_memory_utilization,
    is_moe,
    micro_batch_size,
    nemo_env_vars,
    parallel_layout,
    sequence_length,
)
from monitordecorrelation.experiment_config import ExperimentConfig


# The vLLM memory budget needs the policy's weight size (a Hub metadata call) and the GPU's total
# memory (nvidia-smi). Both are pinned here so the suite is deterministic, offline and machine
# independent; the budget arithmetic itself is tested directly against `gpu_memory_utilization`.
PINNED_BUDGET = {"policy_weight_gib": 8.7, "gpu_total_gib": 179.06}  # Qwen3.5-4B-Base on a B200


def cfg(**kw) -> ExperimentConfig:
    base = dict(run_name="t", backend="nemo", env="impossiblebench", policy="Qwen/Qwen3.5-4B-Base")
    merged = {**base, **kw}
    merged["nemo_options"] = {**PINNED_BUDGET, **(merged.get("nemo_options") or {})}
    return ExperimentConfig(**merged)


def test_is_moe_reads_the_active_param_suffix():
    assert is_moe("Qwen/Qwen3.5-35B-A3B-Base")
    assert is_moe("Qwen/Qwen3-30B-A3B")
    assert not is_moe("Qwen/Qwen3.5-4B-Base")
    assert not is_moe("Qwen/Qwen3-8B")


def test_dense_layout_is_pure_data_parallel():
    for n in (1, 2, 8):
        assert parallel_layout("Qwen/Qwen3.5-4B-Base", n, moe=False) == {
            "tp": 1, "ep": 1, "etp": 1, "sequence_parallel": False
        }


def test_moe_layout_matches_the_measured_2gpu_recipe():
    # TP2 x ETP1 x EP2 -> DP1, with sequence_parallel following TP > 1.
    assert parallel_layout("Qwen/Qwen3.5-35B-A3B-Base", 2, moe=True) == {
        "tp": 2, "ep": 2, "etp": 1, "sequence_parallel": True
    }
    # A single GPU has no expert-parallel layout to choose.
    assert parallel_layout("Qwen/Qwen3.5-35B-A3B-Base", 1, moe=True)["tp"] == 1


def test_sequence_length_is_prompt_plus_completion_and_divisible():
    assert sequence_length(max_tokens=16384, max_prompt_tokens=4096, tp=1) == 20480
    assert sequence_length(max_tokens=16384, max_prompt_tokens=4096, tp=2) == 20480
    # rounded up to a multiple of 64 * tp, never down (truncating the budget would silently cut
    # completions short)
    got = sequence_length(max_tokens=1000, max_prompt_tokens=100, tp=2)
    assert got >= 1100 and got % 128 == 0


def test_micro_batch_holds_the_token_budget_not_the_sequence_count():
    # the recipes were measured at 4096: dense mbs 4, MoE mbs 8
    assert micro_batch_size(4096, moe=False) == 4
    assert micro_batch_size(4096, moe=True) == 8
    # a 5x longer sequence shrinks the micro-batch instead of 5x-ing the activation footprint
    assert micro_batch_size(20480, moe=False) == 1
    assert micro_batch_size(20480, moe=True) == 1
    assert micro_batch_size(10 ** 9, moe=False) == 1  # never zero


def test_experiment_config_fields_reach_nemo():
    c = cfg(batch_size=8, group_size=16, n_steps=25, eval_every=5, eval_size=32,
            eval_samples_per_prompt=2, seed=7, lora_rank=32, kl_coef=1e-4, max_tokens=16384)
    env = nemo_env_vars(c, lr=2e-4, n_gpus=1)
    assert env["MD_NUM_PROMPTS_PER_STEP"] == "8"
    assert env["MD_GROUP_SIZE"] == "16"
    # force_on_policy_ratio is only legal while this equality holds — it is what makes the PPO
    # ratio 1.0 and matches tinker's single optim step per rollout batch.
    assert env["MD_TRAIN_GLOBAL_BATCH_SIZE"] == "128"
    assert env["MD_N_STEPS"] == "25"
    assert env["MD_EVAL_EVERY"] == "5"
    assert env["MD_VAL_SAMPLES"] == "64"  # eval_size * eval_samples_per_prompt
    assert env["MD_SEED"] == "7"
    assert env["MD_LORA_RANK"] == "32"
    assert env["MD_LORA_ALPHA"] == "32"  # fixed, as on tinker
    assert env["MD_KL_COEF"] == "0.0001"
    assert env["MD_LR"] == "0.0002"
    assert env["MD_MAX_TOKENS"] == "16384"


def test_gpu_count_drives_topology():
    c = cfg(max_tokens=16384)
    one = nemo_env_vars(c, lr=1e-4, n_gpus=1)
    two = nemo_env_vars(c, lr=1e-4, n_gpus=2)
    assert one["MD_GPUS_PER_NODE"] == "1" and one["MD_VLLM_TP"] == "1"
    assert two["MD_GPUS_PER_NODE"] == "2" and two["MD_VLLM_TP"] == "2"


def test_moe_gets_the_triton_backend():
    moe = nemo_env_vars(cfg(policy="Qwen/Qwen3.5-35B-A3B-Base"), lr=1e-4, n_gpus=2)
    dense = nemo_env_vars(cfg(), lr=1e-4, n_gpus=2)
    assert moe["MD_VLLM_KWARGS"] == "{moe_backend: triton, max_num_seqs: 1024}"
    assert dense["MD_VLLM_KWARGS"] == "{}"


def test_vllm_memory_fraction_is_what_the_trainer_does_not_need():
    """gmu * total + weights/tp + overhead <= total, i.e. vLLM gets the rest of the GPU."""
    total = 179.06
    # A big policy leaves less for KV than a small one, on the same GPU.
    big = gpu_memory_utilization(67.0, tp=1, moe=False, total_gib=total)
    small = gpu_memory_utilization(8.7, tp=1, moe=False, total_gib=total)
    assert big < small
    # Sharding the weights across tensor-parallel ranks gives the fraction straight back.
    assert gpu_memory_utilization(67.0, tp=2, moe=True, total_gib=total) > big
    # The arithmetic itself, on a case where no clamp binds: the trainer's weights (sharded by TP),
    # plus vLLM's non-discardable residue, plus the flat margin.
    assert big == round((total - (67.0 + 0.16 * 67.0 + 6.0)) / total, 3)
    # Clamped at both ends: a tiny model does not get the whole GPU, a model that cannot fit at all
    # still yields a usable fraction rather than a negative one.
    assert gpu_memory_utilization(0.5, tp=1, moe=False, total_gib=total) == 0.90
    assert gpu_memory_utilization(400.0, tp=1, moe=False, total_gib=total) == 0.30


def test_unknown_policy_size_falls_back_to_the_measured_constants():
    """No network and an undetectable GPU must still render a config, conservatively."""
    assert gpu_memory_utilization(None, tp=1, moe=False, total_gib=179.06) == 0.6
    assert gpu_memory_utilization(None, tp=2, moe=True, total_gib=179.06) == 0.55
    assert gpu_memory_utilization(8.7, tp=1, moe=False, total_gib=None) == 0.6


def test_nemo_options_have_the_last_word():
    c = cfg(nemo_options={"train_micro_batch_size": 3, "MD_GPU_MEMORY_UTILIZATION": 0.42, "moe": True})
    env = nemo_env_vars(c, lr=1e-4, n_gpus=2)
    assert env["MD_TRAIN_MICRO_BATCH_SIZE"] == "3"
    assert env["MD_GPU_MEMORY_UTILIZATION"] == "0.42"
    assert env["MD_TP"] == "2"  # forced MoE => the MoE layout, despite the dense model name


def test_values_are_rendered_in_omegaconfs_literal_grammar_not_json():
    """``${oc.decode:…}`` parses OmegaConf's grammar, which is NOT JSON: a quoted dict key is a
    GrammarParseError. Values are therefore rendered by hand — see ``_literal``."""
    env = nemo_env_vars(cfg(max_tokens=16384), lr=1e-4, n_gpus=2, wandb_tags=["a", "b"])
    assert env["MD_WANDB_TAGS"] == "[a, b]"
    assert env["MD_WANDB_ENABLED"] == "false"
    assert env["MD_SEQUENCE_PARALLEL"] == "false"
    assert env["MD_WANDB_GROUP"] == "null"
    # A top-level string is passed through raw: those keys are read by a bare ${oc.env:...}.
    assert env["MD_MODEL_NAME"] == "Qwen/Qwen3.5-4B-Base"
    # Inside a container it is quoted only when a bare token would be ambiguous.
    moe = nemo_env_vars(cfg(nemo_options={"vllm_kwargs": {"moe_backend": "triton", "x": "1.5"}}),
                        lr=1e-4, n_gpus=1)
    assert moe["MD_VLLM_KWARGS"] == '{moe_backend: triton, x: "1.5"}'


@pytest.mark.parametrize("policy,n_gpus", [("Qwen/Qwen3.5-4B-Base", 1),
                                           ("Qwen/Qwen3.5-4B-Base", 2),
                                           ("Qwen/Qwen3.5-35B-A3B-Base", 2)])
def test_every_tested_topology_renders_a_complete_environment(policy, n_gpus):
    env = nemo_env_vars(cfg(policy=policy, max_tokens=16384), lr=1e-4, n_gpus=n_gpus)
    assert all(isinstance(v, str) for v in env.values())
    assert env["MD_MAX_TOTAL_SEQUENCE_LENGTH"] == "20480"


# --- the config file itself -------------------------------------------------------------------
# These render experiments/configs/nemo/grpo.yaml through OmegaConf exactly as NeMo-RL's loader
# does. They exist because the env-var values are *not* JSON: OmegaConf's grammar rejects a quoted
# dict KEY, so `{"moe_backend": "triton"}` is a GrammarParseError where `{moe_backend: triton}`
# parses. A unit test on the derivation alone would not have caught that.

omegaconf = pytest.importorskip("omegaconf")

NEMO_CONFIG = "experiments/configs/nemo/grpo.yaml"


def resolve(env: dict[str, str]) -> dict:
    """The config as NeMo-RL would see it, under the given MD_* environment."""
    import os

    from omegaconf import OmegaConf

    if not OmegaConf.has_resolver("mul"):  # NeMo-RL's register_omegaconf_resolvers(), inlined
        OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
        OmegaConf.register_new_resolver("div", lambda a, b: a / b)
        OmegaConf.register_new_resolver("max", lambda a, b: max(a, b))
    old = dict(os.environ)
    os.environ.update(env)
    try:
        # `defaults:` inheritance is NeMo-RL's own loader; merge the parent by hand so this test
        # needs only omegaconf, not an installed nemo_rl.
        cfg = OmegaConf.load(NEMO_CONFIG)
        parent_path = str(cfg.pop("defaults"))
        base = OmegaConf.load(f"experiments/configs/nemo/{parent_path}")
        return OmegaConf.to_container(OmegaConf.merge(base, cfg), resolve=True)
    finally:
        os.environ.clear()
        os.environ.update(old)


@pytest.mark.parametrize("policy,n_gpus,tp,ep,sp,vllm_kwargs", [
    ("Qwen/Qwen3.5-4B-Base", 1, 1, 1, False, {}),
    ("Qwen/Qwen3.5-4B-Base", 2, 1, 1, False, {}),
    ("Qwen/Qwen3.5-35B-A3B-Base", 2, 2, 2, True,
     {"moe_backend": "triton", "max_num_seqs": 1024}),
])
def test_config_resolves_for_each_tested_topology(policy, n_gpus, tp, ep, sp, vllm_kwargs):
    gmu = gpu_memory_utilization(PINNED_BUDGET["policy_weight_gib"], tp=tp, moe=bool(vllm_kwargs),
                                 total_gib=PINNED_BUDGET["gpu_total_gib"])
    c = resolve(nemo_env_vars(cfg(policy=policy, max_tokens=16384), lr=4.9e-4, n_gpus=n_gpus,
                              wandb_tags=["ib", "seed0"]))
    mega, gen = c["policy"]["megatron_cfg"], c["policy"]["generation"]
    assert c["policy"]["model_name"] == policy
    assert c["cluster"]["gpus_per_node"] == n_gpus
    assert (mega["tensor_model_parallel_size"], mega["expert_model_parallel_size"]) == (tp, ep)
    assert mega["sequence_parallel"] is sp
    assert gen["vllm_cfg"]["tensor_parallel_size"] == n_gpus
    assert gen["vllm_kwargs"] == vllm_kwargs
    assert gen["vllm_cfg"]["gpu_memory_utilization"] == gmu
    assert gen["max_new_tokens"] == 16384
    assert c["policy"]["max_total_sequence_length"] == 20480 == gen["vllm_cfg"]["max_model_len"]
    # make_sequence_length_divisible_by IS also the packer's per-sequence pad multiple, and a
    # sequence that pads past its bin capacity is a hard error in packing/algorithms.py.
    pad = c["policy"]["make_sequence_length_divisible_by"]
    assert pad == tp
    assert c["policy"]["max_total_sequence_length"] % pad == 0
    assert c["logger"]["wandb"]["tags"] == ["ib", "seed0"]
    assert mega["peft"]["target_modules"] == ["*language_model*linear_qkv",
                                              "*language_model*linear_proj",
                                              "*language_model*linear_fc1",
                                              "*language_model*linear_fc2"]


def test_config_matches_the_tinker_backends_grpo_semantics():
    """The parity contract, asserted on the file rather than trusted to a comment."""
    c = resolve(nemo_env_vars(cfg(batch_size=8, group_size=8, n_steps=25, seed=3, lora_rank=32,
                                  kl_coef=1e-4), lr=2e-4, n_gpus=1))
    g, loss, pol = c["grpo"], c["loss_fn"], c["policy"]
    # cookbook compute_advantages: mean-centred within the group, nothing else
    assert g["normalize_rewards"] is False and g["use_leave_one_out_baseline"] is False
    # one optimizer step per rollout batch => PPO ratio is 1 by construction
    assert loss["force_on_policy_ratio"] is True
    assert pol["train_global_batch_size"] == g["num_prompts_per_step"] * g["num_generations_per_prompt"]
    # incorporate_kl_penalty puts KL on the reward, not in the loss
    assert loss["use_kl_in_reward"] is True and loss["reference_policy_kl_penalty"] == 1e-4
    # tinker AdamParams defaults, as the cookbook's train_step passes them
    opt = pol["megatron_cfg"]["optimizer"]
    assert (opt["adam_beta1"], opt["adam_beta2"], opt["adam_eps"]) == (0.9, 0.95, 1e-8)
    assert opt["weight_decay"] == 0.0 and opt["clip_grad"] == 0.0 and opt["lr"] == 2e-4
    sched = pol["megatron_cfg"]["scheduler"]
    assert sched["lr_decay_style"] == "constant" and sched["lr_warmup_iters"] == 0
    assert pol["megatron_cfg"]["peft"]["dim"] == 32 and pol["megatron_cfg"]["peft"]["alpha"] == 32
    assert pol["generation"]["temperature"] == 1.0  # rl/train.py samples at 1.0
    assert g["max_rollout_turns"] == 1 and g["overlong_filtering"] is False
    assert g["seed"] == 3 and g["max_num_steps"] == 25
    assert g["val_at_start"] is True and g["val_at_end"] is True
