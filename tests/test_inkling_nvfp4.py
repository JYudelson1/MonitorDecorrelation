"""Inkling NVFP4 loader (whitebox/inkling_nvfp4.py), offline on CPU.

The test builds a tiny random Inkling and saves it in the hub layout (``model.llm.*``, ``w13_weight``),
with an ``inkling_mm_model`` config like the real one. It then packs one MoE layer's routed experts the
way ``Inkling-Small-NVFP4`` does: U8 E2M1 codes, FP8 block scales, a per-expert FP32 scale and the TML
names. The NVFP4 load must compute exactly what the same model computes with those experts decoded to
BF16. If a transformers bump moves the loader seam, these tests fail.
"""

from __future__ import annotations

import pytest
import torch

from monitordecorrelation.whitebox.inkling_nvfp4 import (
    decode_nvfp4_into,
    is_nvfp4_checkpoint,
    load_inkling_nvfp4,
    packed_expert_layers,
    weight_map,
)

from inkling_tiny import PACKED_LAYER, dequantize_reference, quantize_nvfp4, tiny_config, write_tiny_checkpoints


def test_decode_matches_the_reference_dequantization():
    torch.manual_seed(0)
    w = torch.randn(3, 8, 64) * 0.05
    packed, scale, scale2, codes = quantize_nvfp4(w)
    out = torch.empty(3, 8, 64, dtype=torch.bfloat16)
    decode_nvfp4_into(out, packed, scale, scale2)
    assert torch.equal(out, dequantize_reference(codes, scale, scale2).to(torch.bfloat16))
    rel_err = ((out.float() - w).norm() / w.norm()).item()
    assert rel_err < 0.15  # NVFP4's own error (~0.08 on the real weights); a wrong nibble order gives ~1.4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused decode is the CUDA path")
def test_fused_decode_is_bit_identical_to_eager_on_cuda():
    torch.manual_seed(0)
    packed, scale, scale2, _ = (t.cuda() for t in quantize_nvfp4(torch.randn(4, 32, 128) * 0.05))
    fused, eager = (torch.empty(4, 32, 128, dtype=torch.bfloat16, device="cuda") for _ in range(2))
    decode_nvfp4_into(fused, packed, scale, scale2)
    decode_nvfp4_into(eager, packed, scale, scale2, fused=False)
    assert torch.equal(fused, eager)


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    """(bf16_dir, nvfp4_dir, codes, scales): the tiny model in the hub layout, and the same checkpoint with
    layer 3's routed experts NVFP4-packed (two shards + index, like the release)."""
    return write_tiny_checkpoints(tmp_path_factory.mktemp("inkling"))


def test_checkpoint_detection(checkpoints):
    bf16_dir, nvfp4_dir, _, _ = checkpoints
    assert is_nvfp4_checkpoint(str(nvfp4_dir)) and not is_nvfp4_checkpoint(str(bf16_dir))
    assert packed_expert_layers(weight_map(str(nvfp4_dir))) == [PACKED_LAYER]
    assert packed_expert_layers(weight_map(str(bf16_dir))) == []


def test_stock_transformers_still_cannot_load_it(checkpoints):
    """The reason the module exists. If this starts passing a load, transformers learned the format and the
    shim may be retired."""
    from transformers import AutoConfig, InklingForCausalLM

    from monitordecorrelation.whitebox.model import _patch_inkling_moe_intermediate

    _, nvfp4_dir, _, _ = checkpoints
    config = _patch_inkling_moe_intermediate(AutoConfig.from_pretrained(nvfp4_dir), str(nvfp4_dir))
    with pytest.raises(RuntimeError, match="mismatched"):
        InklingForCausalLM.from_pretrained(nvfp4_dir, config=config.get_text_config(), dtype=torch.bfloat16)


def test_nvfp4_load_computes_what_the_decoded_bf16_model_computes(checkpoints):
    from transformers import AutoConfig, InklingForCausalLM
    from transformers.core_model_loading import Interleave

    from monitordecorrelation.whitebox.inkling_nvfp4 import PackedNVFP4Experts
    from monitordecorrelation.whitebox.model import _patch_inkling_moe_intermediate

    bf16_dir, nvfp4_dir, codes, scales = checkpoints
    nvfp4 = load_inkling_nvfp4(str(nvfp4_dir), device="cpu").eval()
    assert isinstance(nvfp4.model.layers[PACKED_LAYER].mlp.experts, PackedNVFP4Experts)
    assert not isinstance(nvfp4.model.layers[2].mlp.experts, PackedNVFP4Experts)

    config = _patch_inkling_moe_intermediate(AutoConfig.from_pretrained(bf16_dir), str(bf16_dir))
    reference = InklingForCausalLM.from_pretrained(bf16_dir, config=config.get_text_config(),
                                                   dtype=torch.bfloat16).eval()
    original = InklingForCausalLM.from_pretrained(bf16_dir, config=config.get_text_config(),
                                                  dtype=torch.bfloat16).eval()
    experts = reference.model.layers[PACKED_LAYER].mlp.experts
    w13 = dequantize_reference(codes["w13_weight"], *scales["w13_weight"]).to(torch.bfloat16)
    experts.gate_up_proj.data.copy_(Interleave(dim=1).convert({"w": w13}, None, ["w"])["w"])
    experts.down_proj.data.copy_(dequantize_reference(codes["w2_weight"], *scales["w2_weight"]).to(torch.bfloat16))

    torch.manual_seed(1)
    ids = torch.randint(0, 500, (2, 24))  # longer than the sliding window
    with torch.no_grad():
        got = nvfp4(input_ids=ids, output_hidden_states=True)
        want = reference(input_ids=ids, output_hidden_states=True)
        unquantized = original(input_ids=ids).logits
    assert len(got.hidden_states) == tiny_config()["text_config"]["num_hidden_layers"] + 1
    for g, w in zip(got.hidden_states, want.hidden_states):
        assert torch.equal(g, w)
    assert torch.equal(got.logits, want.logits)
    assert not torch.equal(got.logits, unquantized)  # the packed layer matters, so a wrong decode would show
