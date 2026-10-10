"""Inkling-Small-NVFP4 on ONE GPU in transformers: the routed experts stay packed in memory, and each layer's
experts are decoded to BF16 just before that layer runs (weights NVFP4, activations BF16, i.e. W4A16).

Why this exists: transformers 5.19 cannot load ``thinkingmachines/Inkling-Small-NVFP4``. This was tested on a
B300 on 2026-10-09, including the HF launch blog's own ``from_pretrained`` call. Three reasons:

- The checkpoint's ``config.json`` has no ``quantization_config``.
- transformers' NVFP4 quantizer refuses pre-quantized checkpoints.
- The packed experts are shape mismatches for the BF16 ``InklingExperts`` parameters, which transformers
  would replace with ~470 GiB of random weights.

Everything else in the checkpoint is BF16 and loads normally: attention, shared experts, the dense layers,
the embeddings and unembedding, and the first MoE layer's experts. So: let ``from_pretrained`` load those,
swap each quantized layer's experts for ``PackedNVFP4Experts`` before transformers would allocate them, then
read the packed tensors straight from the safetensors files.

The format of a quantized expert tensor ``w`` uses custom TML names, not ModelOpt's
``weight_scale``/``weight_scale_2``:

- ``w`` U8 [E, R, C/2]: two FP4 (E2M1) codes per byte, low nibble first.
- ``w.scale`` F8_E4M3 [E, R, C/16]: one scale per 16 values along the last dim.
- ``w.scale2`` F32 [E]: one scale per expert.
- value = code × scale × scale2.
- ``w.input_amax`` (the activation scale for W4A4 serving) and ``w.original_shape`` are not needed.

Against the BF16 release this decode gives 8.1% relative weight error (cosine 0.997), which is NVFP4's own
error. The alternative decodes are at 141% error.

Cost for Inkling-Small: 167 GiB on the GPU (vs ~530 GB in BF16) plus one 12 GiB decode buffer, so a single
B300 rather than 4+ GPUs; a B200 is too small. The fused decode adds ~0.14 s per forward pass on a B300.
"""

from __future__ import annotations

import json
import os
import re

import torch

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
BLOCK = 16  # values per FP8 block scale, along the last dim
QUANT_CONFIG = "hf_quant_config.json"
_PACKED_EXPERTS = re.compile(r"model\.llm\.layers\.(\d+)\.mlp\.experts\.w13_weight\.scale")


def local_checkpoint(model_name: str) -> str:
    """``model_name`` itself if it is a directory, else its hub snapshot (downloaded if not cached)."""
    if os.path.isdir(model_name):
        return model_name
    from huggingface_hub import snapshot_download

    return snapshot_download(model_name)


def is_nvfp4_checkpoint(model_name: str) -> bool:
    """True if the checkpoint ships an ``hf_quant_config.json`` declaring NVFP4 (a ModelOpt export)."""
    if os.path.isdir(model_name):
        path = os.path.join(model_name, QUANT_CONFIG)
        if not os.path.exists(path):
            return False
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            path = hf_hub_download(model_name, QUANT_CONFIG)
        except EntryNotFoundError:
            return False
    with open(path) as fh:
        return json.load(fh).get("quantization", {}).get("quant_algo") == "NVFP4"


def weight_map(ckpt: str) -> dict[str, str]:
    """Tensor name -> shard file, for sharded and single-file checkpoints alike."""
    index = os.path.join(ckpt, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as fh:
            return json.load(fh)["weight_map"]
    from safetensors import safe_open

    with safe_open(os.path.join(ckpt, "model.safetensors"), "pt") as f:
        return {key: "model.safetensors" for key in f.keys()}


def packed_expert_layers(wmap: dict[str, str]) -> list[int]:
    """Indices of the decoder layers whose routed experts are NVFP4-packed in the checkpoint."""
    return sorted(int(m.group(1)) for key in wmap if (m := _PACKED_EXPERTS.fullmatch(key)))


_LUTS: dict[torch.device, torch.Tensor] = {}
_fused_decode = None


def _lut(device: torch.device) -> torch.Tensor:
    if device not in _LUTS:
        _LUTS[device] = torch.tensor(E2M1_VALUES, device=device)
    return _LUTS[device]


def _decode(out, packed, scale, scale2, lut) -> None:
    n_exp, rows, half = packed.shape
    codes = torch.stack(((packed & 0xF).long(), (packed >> 4).long()), dim=-1)  # low nibble first
    w = lut[codes].view(n_exp, rows, 2 * half // BLOCK, BLOCK)
    w = w * (scale.float() * scale2.float().view(n_exp, 1, 1)).unsqueeze(-1)
    out.copy_(w.view(n_exp, rows, 2 * half))


def decode_nvfp4_into(out: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor, scale2: torch.Tensor, *,
                      fused: bool | None = None) -> None:
    """``out[e]`` <- the weights of expert ``e``, decoded from ``packed`` / ``scale`` / ``scale2``.

    ``fused`` (the default on CUDA) runs one torch.compile'd kernel. Otherwise the decode runs eagerly in
    chunks of experts, which bounds the float32 temporaries. The two are bit-identical (checked on a B300 by
    experiments/check_inkling_nvfp4.py)."""
    global _fused_decode
    lut = _lut(packed.device)
    if (packed.is_cuda if fused is None else fused):
        if _fused_decode is None:
            _fused_decode = torch.compile(_decode, dynamic=False, fullgraph=True)
        _fused_decode(out, packed, scale, scale2, lut)
        return
    for s in range(0, packed.shape[0], 32):
        _decode(out[s:s + 32], packed[s:s + 32], scale[s:s + 32], scale2[s:s + 32], lut)


class PackedNVFP4Experts(torch.nn.Module):
    """Stands in for a quantized layer's ``InklingExperts``. It holds the packed weights, and on each forward
    decodes them into the shared BF16 ``runner``: an ``InklingExperts`` built from the model's config, so the
    expert math and the experts implementation (eager / grouped_mm) are transformers' own."""

    def __init__(self, runner_box: list) -> None:
        super().__init__()
        self._runner_box = runner_box  # a list: the runner is shared, so it must not register as a submodule
        self.w13: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None  # rows in gate_up_proj order
        self.w2: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def forward(self, hidden_states, top_k_index, top_k_weights):
        runner = self._runner_box[0]
        decode_nvfp4_into(runner.gate_up_proj.data, *self.w13)
        decode_nvfp4_into(runner.down_proj.data, *self.w2)
        return runner(hidden_states, top_k_index, top_k_weights)


def load_inkling_nvfp4(model_name: str, *, device: str = "cuda:0"):
    """Text-only ``InklingForCausalLM`` from an Inkling NVFP4 checkpoint, entirely on ``device``.

    Uses the same text-only conversions and expert-width patch as ``WhiteBoxModel``'s BF16 path. Hooks
    ``PreTrainedModel._finalize_model_loading`` (transformers-internal), so it is tied to the locked
    transformers. tests/test_inkling_nvfp4.py round-trips a tiny model and fails if that seam moves."""
    from safetensors import safe_open
    from transformers import AutoConfig, InklingForCausalLM
    from transformers.core_model_loading import Interleave
    from transformers.modeling_utils import PreTrainedModel
    from transformers.models.inkling.modeling_inkling import InklingExperts

    from monitordecorrelation.whitebox.model import _patch_inkling_moe_intermediate, _register_inkling_text_conversions

    ckpt = local_checkpoint(model_name)
    wmap = weight_map(ckpt)
    layers = packed_expert_layers(wmap)
    if not layers:
        raise ValueError(f"{model_name!r} has no NVFP4-packed routed experts")
    config = _patch_inkling_moe_intermediate(AutoConfig.from_pretrained(ckpt), ckpt)
    _register_inkling_text_conversions()

    runner_box: list = []
    expected = {f"model.layers.{i}.mlp.experts.{name}" for i in layers for name in ("gate_up_proj", "down_proj")}
    original = PreTrainedModel._finalize_model_loading

    def _finalize(model, load_config, loading_info):
        # The packed experts are exactly the mismatches; anything else means the checkpoint or the loader
        # changed, and finalizing would silently fill it with random weights.
        mismatched = {key for key, _, _ in loading_info.mismatched_keys}
        if mismatched != expected or loading_info.missing_keys:
            raise RuntimeError(
                f"unexpected NVFP4 load state: mismatched-vs-expected {sorted(mismatched ^ expected)[:6]}, "
                f"missing {sorted(loading_info.missing_keys)[:6]}"
            )
        loading_info.mismatched_keys.clear()
        for i in layers:
            model.model.layers[i].mlp.experts = PackedNVFP4Experts(runner_box)
        return original(model, load_config, loading_info)

    PreTrainedModel._finalize_model_loading = staticmethod(_finalize)
    try:
        model = InklingForCausalLM.from_pretrained(
            ckpt, config=config.get_text_config(), device_map={"": device}, dtype=torch.bfloat16
        )
    finally:
        PreTrainedModel._finalize_model_loading = staticmethod(original)

    with torch.device("meta"):
        runner = InklingExperts(model.config)
    for name in ("gate_up_proj", "down_proj"):
        shape = getattr(runner, name).shape
        buffer = torch.empty(shape, dtype=torch.bfloat16, device=device)
        setattr(runner, name, torch.nn.Parameter(buffer, requires_grad=False))
    runner_box.append(runner)

    interleave = Interleave(dim=1)  # transformers' w13 -> gate_up_proj row order; rows of codes and scales alike

    def tensor(key: str) -> torch.Tensor:
        with safe_open(os.path.join(ckpt, wmap[key]), "pt", device=str(device)) as f:
            return f.get_tensor(key)

    for i in layers:
        prefix = f"model.llm.layers.{i}.mlp.experts."
        experts = model.model.layers[i].mlp.experts
        w13 = [interleave.convert({"w": tensor(prefix + "w13_weight" + s)}, None, ["w"])["w"] for s in ("", ".scale")]
        experts.w13 = (w13[0], w13[1], tensor(prefix + "w13_weight.scale2"))
        experts.w2 = tuple(tensor(prefix + "w2_weight" + s) for s in ("", ".scale", ".scale2"))
        for (packed, scale, scale2), target in ((experts.w13, runner.gate_up_proj), (experts.w2, runner.down_proj)):
            n_exp, rows, cols = target.shape
            if (packed.shape, scale.shape, scale2.shape) != ((n_exp, rows, cols // 2), (n_exp, rows, cols // BLOCK),
                                                             (n_exp,)):
                raise RuntimeError(f"layer {i}: packed shapes {packed.shape}/{scale.shape}/{scale2.shape} "
                                   f"do not decode to {tuple(target.shape)}")
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return model
