"""Shared helper for the Inkling tests: a tiny random Inkling saved in the hub layout, and its NVFP4 twin.

The hub layout means ``model.llm.*`` names, ``w13_weight`` and an ``inkling_mm_model`` config. In the twin,
one MoE layer's routed experts are packed the way ``Inkling-Small-NVFP4`` packs them: U8 E2M1 codes, FP8
block scales, a per-expert FP32 scale and the TML names.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from monitordecorrelation.whitebox.inkling_nvfp4 import BLOCK, E2M1_VALUES

PACKED_LAYER = 3  # like the release: the dense layers and the first MoE layer (2) stay BF16
E2M1 = torch.tensor(E2M1_VALUES)


def tiny_config(vocab_size: int = 512) -> dict:
    """Inkling-Small's config.json, shrunk: same keys and the same layer structure (2 dense, then MoE)."""
    text = {
        "torch_dtype": "bfloat16", "hidden_size": 64, "num_hidden_layers": 4, "vocab_size": vocab_size,
        "unpadded_vocab_size": vocab_size - 12, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 16, "d_rel": 4, "rel_extent": 64, "q_bias": False, "o_bias": False,
        "log_scaling_n_floor": 128000, "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-06, "use_embed_norm": True,
        "local_layer_ids": [0, 1, 2], "dense_mlp_idx": 2, "use_sconv": True, "sconv_kernel_size": 4,
        "logits_mup_width_multiplier": 16.0, "final_logit_softcapping": None, "swa_head_dim": 16,
        "swa_num_attention_heads": 4, "swa_num_key_value_heads": 2, "sliding_window_size": 8,
        "n_routed_experts": 8, "num_experts_per_tok": 2, "n_shared_experts": 2, "shared_expert_sink": True,
        "dense_intermediate_size": 128, "intermediate_size": 32, "route_scale": 8.0, "use_gate_bias": True,
        "gate_activation": "sigmoid", "norm_after_topk": True, "use_global_scale": True,
    }
    return {
        "architectures": ["InklingForConditionalGeneration"], "model_type": "inkling_mm_model", "eos_token_id": 1,
        "text_config": text,
        "vision_config": {"vision_encoder_type": "hmlp", "decoder_dmodel": 64, "patch_size": 40,
                          "temporal_patch_size": 2, "n_channels": 3, "n_layers": 4, "use_vision_norm": True},
        "audio_config": {"decoder_dmodel": 64, "n_mel_bins": 80, "mel_vocab_size": 16, "bias": False,
                         "dmel_min_value": -7.0, "dmel_max_value": 2.0, "use_audio_norm": True, "audio_mode": "dmel"},
    }


def quantize_nvfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference quantizer, following ModelOpt's recipe.

    - Per-expert ``scale2`` = amax / (6 * 448).
    - Per-16-value FP8 ``scale`` = block amax / (6 * scale2).
    - Code = the nearest E2M1 value of w / (scale * scale2).

    Returns (packed, scale, scale2, codes). ``packed`` holds two codes per byte, the even index in the low
    nibble."""
    w = w.float()
    n_exp, rows, cols = w.shape
    scale2 = w.abs().amax(dim=(1, 2)) / (6.0 * 448.0)
    blocks = w.view(n_exp, rows, cols // BLOCK, BLOCK)
    scale = (blocks.abs().amax(-1) / (6.0 * scale2.view(-1, 1, 1))).to(torch.float8_e4m3fn)
    x = (blocks / (scale.float() * scale2.view(-1, 1, 1)).unsqueeze(-1)).view(n_exp, rows, cols)
    magnitude = (x.abs().unsqueeze(-1) - E2M1[:8]).abs().argmin(-1)
    codes = magnitude + 8 * ((x < 0) & (magnitude > 0)).long()
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    return packed, scale, scale2, codes


def dequantize_reference(codes, scale, scale2) -> torch.Tensor:
    n_exp, rows, cols = codes.shape
    w = E2M1[codes].view(n_exp, rows, cols // BLOCK, BLOCK)
    return (w * (scale.float() * scale2.view(-1, 1, 1)).unsqueeze(-1)).view(n_exp, rows, cols)


def write_tiny_checkpoints(root: Path, vocab_size: int = 512) -> tuple[Path, Path, dict, dict]:
    """(bf16_dir, nvfp4_dir, codes, scales). ``codes``/``scales`` are the packed layer's, per tensor name."""
    from transformers import AutoConfig, InklingForCausalLM

    from monitordecorrelation.whitebox.model import _patch_inkling_moe_intermediate, _register_inkling_text_conversions

    config_json = json.dumps(tiny_config(vocab_size))
    bf16_dir, nvfp4_dir = root / "bf16", root / "nvfp4"
    bf16_dir.mkdir()
    (bf16_dir / "config.json").write_text(config_json)
    config = _patch_inkling_moe_intermediate(AutoConfig.from_pretrained(bf16_dir), str(bf16_dir))
    _register_inkling_text_conversions()
    torch.manual_seed(0)
    model = InklingForCausalLM(config.get_text_config()).to(torch.bfloat16)
    model.save_pretrained(bf16_dir)  # save_original_format: transformers reverts its conversions -> hub layout
    (bf16_dir / "config.json").write_text(config_json)

    tensors = load_file(str(bf16_dir / "model.safetensors"))
    prefix = f"model.llm.layers.{PACKED_LAYER}.mlp.experts."
    assert prefix + "w13_weight" in tensors, sorted(tensors)[:20]  # the hub names the loader reads
    packed_part, codes, scales = {}, {}, {}
    for name in ("w13_weight", "w2_weight"):
        w = tensors.pop(prefix + name)
        packed, scale, scale2, codes[name] = quantize_nvfp4(w)
        scales[name] = (scale, scale2)
        packed_part.update({
            prefix + name: packed, prefix + name + ".scale": scale, prefix + name + ".scale2": scale2,
            prefix + name + ".input_amax": torch.ones(1, dtype=torch.bfloat16),
            prefix + name + ".original_shape": torch.tensor(list(w.shape)),
        })
    nvfp4_dir.mkdir()
    shards = {"model-00001-of-00002.safetensors": tensors, "model-00002-of-00002.safetensors": packed_part}
    for shard, part in shards.items():
        save_file(part, str(nvfp4_dir / shard), metadata={"format": "pt"})
    index = {"metadata": {}, "weight_map": {k: shard for shard, part in shards.items() for k in part}}
    (nvfp4_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    (nvfp4_dir / "config.json").write_text(config_json)
    quant = {"quantization": {"quant_algo": "NVFP4", "group_size": BLOCK}}
    (nvfp4_dir / "hf_quant_config.json").write_text(json.dumps(quant))
    return bf16_dir, nvfp4_dir, codes, scales
