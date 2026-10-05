"""``WhiteBoxModel``: a local HF model used only to *read activations* (never to generate).

The Atlas follow-up technique: raw-generation activations don't generalize, so we append a fixed
follow-up turn (``User: Did you respond truthfully?`` / ``Assistant: Yes``) to the rollout and read
the residual stream at the **final token** of that follow-up, per layer. The *same* follow-up is
applied at train and deploy time.

Model-agnostic: default is a tiny ``Qwen/Qwen3-0.6B`` for local/CPU validation; ``Qwen/Qwen3-8B`` for
the real run is just a ``model_name`` arg. Inkling checkpoints are multimodal, but we load
**text-only** via ``InklingForCausalLM`` (language tower + ``lm_head``; vision/audio never
allocated) and put CoT in ``reasoning_content`` rather than Qwen ``<think>`` tags. Device
auto-selects cuda -> mps -> cpu for ordinary CausalLMs; Inkling uses ``device_map="auto"``.
Dtype is bf16 on accelerators, fp32 on CPU.
"""

from __future__ import annotations

import threading
from typing import Sequence, Union

import numpy as np

from monitordecorrelation.types import Rollout

# Default follow-up turn (Atlas §4.3), deception-framed. Adjustable per probe/dataset: pass a
# different ``follow_up`` to ``extract_activations`` (e.g. a sycophancy-specific question). The probe
# stores the follow-up it was trained with in its meta, so deploy uses the *identical* one.
FOLLOWUP_QUESTION = "Did you respond truthfully?"
FOLLOWUP_ANSWER = "Yes"
DEFAULT_FOLLOWUP = (FOLLOWUP_QUESTION, FOLLOWUP_ANSWER)

# Matches ``DEFAULT_THINKING_EFFORT`` (0.9) in rl/renderers — Inkling HF chat template string form.
DEFAULT_REASONING_EFFORT = "high"

# An item can be a Rollout or a raw (question, cot, answer) triple.
Item = Union[Rollout, "tuple[str, str, str]"]


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _as_triple(item: Item) -> tuple[str, str, str]:
    if isinstance(item, Rollout):
        return item.prompt.text, item.cot, item.output
    return item  # type: ignore[return-value]


def fold_assistant(cot: str, answer: str) -> str:
    """The assistant message the probe reads: CoT wrapped in think tags + the answer. Shared so
    train-time (dataset adapters) and deploy-time (live rollouts) produce byte-identical text.

    Qwen / HF-chat only — Inkling puts CoT in ``reasoning_content`` (see ``_build_messages``)."""
    return f"<think>{cot}</think>\n{answer}" if cot else answer


def _is_tml_model_name(model_name: str) -> bool:
    """Same predicate as ``rl.renderers.is_tml_policy`` (Inkling / thinkingmachines/*), kept local so
    whitebox does not import the RL renderer stack (tinker)."""
    return model_name.split(":")[0].startswith("thinkingmachines/")


def _is_multimodal_config(config) -> bool:
    """True for Inkling-style multimodal LMs (``inkling_mm_model``, ``*ForConditionalGeneration``)."""
    mt = (getattr(config, "model_type", None) or "").lower()
    if "inkling" in mt or mt.endswith("_mm_model"):
        return True
    arches = getattr(config, "architectures", None) or []
    return any(isinstance(a, str) and a.endswith("ForConditionalGeneration") for a in arches)


def _patch_inkling_moe_intermediate(config, model_name: str):
    """Inkling-Small checkpoints store MoE expert width as ``text_config.intermediate_size`` and dense
    MLP width as ``dense_intermediate_size``. Transformers remaps the latter onto ``intermediate_size``
    but leaves ``moe_intermediate_size`` at the *full* Inkling default (3072), so expert ``gate_up_proj``
    is built as ``[256, 6144, 4096]`` while the ckpt is ``[256, 4096, 4096]`` (2×2048). Patch before
    ``from_pretrained`` so weights match. No-op when the raw config already sets ``moe_intermediate_size``.
    """
    mt = (getattr(config, "model_type", None) or "").lower()
    if "inkling" not in mt and not _is_tml_model_name(model_name):
        return config
    from transformers.configuration_utils import PreTrainedConfig

    raw, _ = PreTrainedConfig.get_config_dict(model_name)
    text_raw = raw.get("text_config", raw)
    if "moe_intermediate_size" in text_raw:
        return config
    if "intermediate_size" not in text_raw or "dense_intermediate_size" not in text_raw:
        return config
    tc = config.get_text_config() if hasattr(config, "get_text_config") else config
    tc.moe_intermediate_size = int(text_raw["intermediate_size"])
    return config


def _register_inkling_text_conversions() -> None:
    """Hub Inkling checkpoints use ModelOpt names (``model.llm.*``, ``w13_weight``, …). The stock
    conversion map is registered under ``inkling_mm_model`` for ``InklingForConditionalGeneration``.
    Text-only ``InklingForCausalLM`` (``inkling_text``) does not see that map, so without this every
    weight is ``missing`` → tqdm ``Loading weights: 0it`` then a hang reallocating the MoE. Mirror the
    MM map with ``model.language_model.*`` targets rewritten to ``model.*``.
    """
    from transformers.conversion_mapping import (
        WeightConverter,
        WeightRenaming,
        get_checkpoint_conversion_mapping,
        register_checkpoint_conversion_mapping,
    )

    mm = get_checkpoint_conversion_mapping("inkling_mm_model")
    if not mm:
        return

    def _rewrite(tp):
        if isinstance(tp, str):
            return tp.replace("model.language_model.", "model.")
        if isinstance(tp, list):
            return [_rewrite(t) for t in tp]
        return tp

    adapted = []
    for transform in mm:
        src = transform.source_patterns
        tgt = _rewrite(transform.target_patterns)
        if isinstance(transform, WeightConverter):
            adapted.append(
                WeightConverter(
                    source_patterns=src,
                    target_patterns=tgt,
                    operations=list(transform.operations),
                    force_cpu=transform.force_cpu,
                )
            )
        else:
            adapted.append(WeightRenaming(source_patterns=src, target_patterns=tgt))
    # Class-name lookup wins in get_model_conversion_mapping; register both for safety.
    register_checkpoint_conversion_mapping("InklingForCausalLM", adapted, overwrite=True)
    register_checkpoint_conversion_mapping("inkling_text", adapted, overwrite=True)


def _install_inkling_finalize_guard(*, max_missing_gib: float = 2.0):
    """Wrap ``PreTrainedModel._finalize_model_loading`` so we print missing/unexpected key stats
    *before* HF materializes missing meta tensors (the silent hang after ``Loading weights: 100%``
    when VRAM is already full). Raises if missing params would allocate more than ``max_missing_gib``.

    Returns a restore callback.
    """
    from transformers.modeling_utils import PreTrainedModel

    orig = PreTrainedModel._finalize_model_loading

    @staticmethod
    def _guarded(model, load_config, loading_info):
        missing = sorted(loading_info.missing_and_mismatched())
        unexpected = sorted(getattr(loading_info, "unexpected_keys", set()) or [])
        mismatched = sorted(getattr(loading_info, "mismatched_keys", set()) or [])
        # Estimate bytes for missing *parameters* still on meta (what finalize is about to allocate).
        rows: list[tuple[int, str, tuple]] = []
        for key in missing:
            try:
                p = model.get_parameter_or_buffer(key)
            except Exception:
                continue
            rows.append((int(p.numel()) * max(int(p.element_size()), 1), key, tuple(p.shape)))
        rows.sort(reverse=True)
        missing_bytes = sum(b for b, _, _ in rows)
        print(
            f"[WhiteBoxModel] post-load: missing/mismatched={len(missing)} "
            f"unexpected={len(unexpected)} mismatched={len(mismatched)} "
            f"missing≈{missing_bytes / 1024**3:.2f} GiB"
        )
        for b, key, shape in rows[:15]:
            print(f"  missing {b / 1024**3:7.2f} GiB  {key}  {shape}")
        if len(rows) > 15:
            print(f"  … +{len(rows) - 15} more missing keys")
        if unexpected:
            print(f"  unexpected (first 10): {unexpected[:10]}")
        if missing_bytes > max_missing_gib * 1024**3:
            raise RuntimeError(
                f"Refusing to finalize load: {missing_bytes / 1024**3:.1f} GiB of weights are still "
                f"missing and would be randomly re-initialized on GPU (this is the hang after "
                f"'Loading weights: 100%' with VRAM already full). Fix the Inkling weight conversion "
                f"/ NVFP4 load path instead of allocating them. Top missing keys printed above."
            )
        print("[WhiteBoxModel] finalize: materializing any remaining missing keys…")
        return orig(model, load_config, loading_info)

    PreTrainedModel._finalize_model_loading = _guarded

    def _restore() -> None:
        PreTrainedModel._finalize_model_loading = orig

    return _restore


class WhiteBoxModel:
    remote: bool = False  # class default so stubs / __init__-bypassing callers behave as local
    _multimodal: bool = False  # class default for __new__ stubs (CausalLM / Qwen path)
    # Serializes local activation reads. One model is shared by every probe on it
    # (experiment_config.build_monitors), and the RL loop can score it from the training thread and a
    # background eval at once — neither the forward pass nor the tokenizer (``padding_side`` is set per
    # call) is safe to run concurrently. ``__init__`` gives each local instance its own; this class
    # default only serves __init__-bypassing stubs. (A remote model needs none: probe_server.py
    # serializes the GPU work itself.)
    _lock = threading.Lock()

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", device: str | None = None,
                 server_url: str | None = None,
                 reasoning_effort: str = DEFAULT_REASONING_EFFORT) -> None:
        """Local mode (default): load the HF model for activation reads. **Remote mode** (``server_url``
        set): load NOTHING locally — proxy ``extract_activations`` to a shared ``probe_server.py`` that
        holds one copy of the model for all runs. Same ``.extract_activations`` interface either way, so
        ``ProbeMonitor`` is unchanged. Remote mode removes the per-run 16 GB model copy → far higher
        run-parallelism (bounded then by tinker/API limits, not local GPU memory).

        ``reasoning_effort`` only affects multimodal (Inkling) chat templating; ignored for CausalLM."""
        self.remote = server_url is not None
        if self.remote:
            import json
            import urllib.request

            self.server_url = server_url.rstrip("/")
            with urllib.request.urlopen(f"{self.server_url}/meta", timeout=120) as r:
                meta = json.loads(r.read())
            self.model_name = meta["model_name"]
            self._n_layers, self._d_model = meta["n_layers"], meta["d_model"]
            return

        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.dtype = torch.bfloat16 if (device or _pick_device()) in ("cuda", "mps") else torch.float32

        try:
            config = AutoConfig.from_pretrained(model_name)
        except ValueError as e:
            raise ValueError(
                f"Transformers does not recognize the architecture for {model_name!r}. "
                f"Inkling needs transformers>=5.14.0 (this env may still be on an older pin from "
                f"tinker-cookbook). Original error: {e}"
            ) from e

        self._multimodal = _is_multimodal_config(config) or _is_tml_model_name(model_name)
        if self._multimodal:
            # Text-only load: Inkling hubs are multimodal, but probes only need the language residual
            # stream. ``InklingForCausalLM`` allocates text + lm_head only. Hub tensors are named
            # ``model.llm.*`` (ModelOpt); register the MM→text conversion map first or every weight
            # is missing and load hangs at ``Loading weights: 0it``.
            from transformers import InklingForCausalLM

            config = _patch_inkling_moe_intermediate(config, model_name)
            text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
            _register_inkling_text_conversions()

            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            try:
                # Do NOT pass key_mapping here — that would replace/stack badly with the registered
                # WeightConverter pipeline (MoE w13→gate_up, attn renames, …). Hidden states at forward.
                # Do NOT force dtype=bfloat16 on *-NVFP4: let the checkpoint/quantizer decide; forcing
                # BF16 can materialize dense weights (~3× VRAM) if a quant path is active.
                load_kw: dict = {
                    "config": text_config,
                    "device_map": "auto",
                    "output_loading_info": True,
                }
                if "nvfp4" not in model_name.lower() and "fp4" not in model_name.lower():
                    load_kw["dtype"] = self.dtype
                print(f"[WhiteBoxModel] from_pretrained({model_name!r}) text-only InklingForCausalLM…")
                restore = _install_inkling_finalize_guard()
                try:
                    self.model, loading_info = InklingForCausalLM.from_pretrained(model_name, **load_kw)
                finally:
                    restore()
                # loading_info is a LoadStateDictInfo (or dict, depending on transformers version)
                if isinstance(loading_info, dict):
                    mk = loading_info.get("missing_keys") or []
                    uk = loading_info.get("unexpected_keys") or []
                    print(f"[WhiteBoxModel] load done: missing={len(mk)} unexpected={len(uk)}")
                else:
                    print(
                        f"[WhiteBoxModel] load done: missing={len(getattr(loading_info, 'missing_keys', []) or [])} "
                        f"unexpected={len(getattr(loading_info, 'unexpected_keys', []) or [])}"
                    )
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to load Inkling text tower from {model_name!r}. "
                    f"Inkling-Small BF16 is ~266B params (12B active); prefer "
                    f"``thinkingmachines/Inkling-Small-NVFP4`` on ≥2× H200, free other GPU "
                    f"processes, or set ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``. "
                    f"Original error: {e}"
                ) from e
            self.model.eval()
            self.device = str(next(self.model.parameters()).device)
            print(f"[WhiteBoxModel] ready device={self.device} n_layers={self.n_layers} d_model={self.d_model}")
        else:
            self.processor = None
            self.device = device or _pick_device()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=self.dtype, output_hidden_states=True
            )
            self.model.to(self.device)
            self.model.eval()

        self._lock = threading.Lock()

    def _extract_remote(self, items, follow_up, batch_size, preserve_thinking) -> np.ndarray:
        """POST the (question, cot, answer) triples to the shared server; it renders + reads activations
        and returns the [n, n_layers, d_model] array (numpy .npy over the wire, localhost)."""
        import json
        import urllib.request
        from io import BytesIO

        triples = [list(_as_triple(it)) for it in items]
        if not triples:
            return np.empty((0, self.n_layers, self.d_model), dtype=np.float32)
        payload = json.dumps({"items": triples, "follow_up": list(follow_up) if follow_up else None,
                              "batch_size": batch_size, "preserve_thinking": preserve_thinking}).encode()
        req = urllib.request.Request(f"{self.server_url}/extract_activations", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:  # generous: big batch on a shared GPU
            return np.load(BytesIO(r.read()))

    def _build_messages(
        self, question: str, cot: str, answer: str, follow_up: tuple[str, str] | None
    ) -> list[dict]:
        """The rollout turn (+ optional follow-up turn).

        ``follow_up=None`` → **within-generation**: the rollout is the FINAL turn, so we read its own
        last answer token with the CoT still in context. Required for CoT probes on *thinking* models:
        the chat template strips ``<think>`` from non-final turns, so the follow-up variant (a later
        turn) is structurally no-CoT. ``follow_up=(q, a)`` → the Atlas follow-up technique (no-CoT on
        thinking models, fine for non-reasoning models).

        Inkling / multimodal: CoT goes in ``reasoning_content`` (HF chat template →
        ``<|content_thinking|>``); Qwen / HF-chat: CoT is folded into content via ``fold_assistant``.
        """
        if self._multimodal:
            assistant: dict = {"role": "assistant", "content": answer}
            if cot:
                assistant["reasoning_content"] = cot
            msgs = [
                {"role": "user", "content": question},
                assistant,
            ]
            if follow_up is None:
                return msgs
            fu_q, fu_a = follow_up
            return msgs + [
                {"role": "user", "content": fu_q},
                {"role": "assistant", "content": fu_a},
            ]

        assistant_text = fold_assistant(cot, answer)
        msgs = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_text},
        ]
        if follow_up is None:
            return msgs
        fu_q, fu_a = follow_up
        return msgs + [
            {"role": "user", "content": fu_q},
            {"role": "assistant", "content": fu_a},
        ]

    def _thinking_preserving_template(self) -> str | None:
        """A copy of the model's chat template patched to KEEP `<think>` on assistant turns that have
        it. Qwen's default strips reasoning from turns BEFORE the last user query, which deletes the
        rollout's CoT once a follow-up turn is appended — so the follow-up technique is otherwise no-CoT
        on thinking models. Returns None if the template isn't the known Qwen structure (graceful: no
        preservation). Cached. No-op for Inkling (its template already keeps ``reasoning_content``)."""
        if self._multimodal:
            return None
        cache = getattr(self, "_preserve_tmpl", "UNSET")
        if cache != "UNSET":
            return cache
        result: str | None = None
        tmpl = getattr(self.tokenizer, "chat_template", None)
        if tmpl and "ns.last_query_index" in tmpl:
            lines = tmpl.split("\n")
            idx = next((i for i, l in enumerate(lines) if "loop.index0 > ns.last_query_index" in l), -1)
            block = lines[idx : idx + 9] if idx >= 0 else []
            keep = next((l for l in block if "<think>" in l), None)  # the keep-reasoning render line
            plain = next((l for l in block if "<think>" not in l and "im_start" in l), None)  # content
            if keep and plain:
                new = ["        {%- if reasoning_content %}", keep,
                       "        {%- else %}", plain, "        {%- endif %}"]
                result = "\n".join(lines[:idx] + new + lines[idx + 9 :])
        self._preserve_tmpl = result
        return result

    def _render(self, item: Item, follow_up: tuple[str, str] | None,
                preserve_thinking: bool = False) -> str:
        """Render one item's conversation to a string via the chat template.

        ``add_generation_prompt=False`` because the response is already present — we read activations
        over a complete conversation. ``preserve_thinking=True`` uses a patched template that keeps the
        rollout's `<think>` even when a follow-up turn follows it (else Qwen strips it). Inkling: uses
        the processor template with ``reasoning_effort``; ``preserve_thinking`` is a no-op."""
        q, cot, ans = _as_triple(item)
        messages = self._build_messages(q, cot, ans, follow_up)
        kw = dict(tokenize=False, add_generation_prompt=False)

        if self._multimodal:
            proc = self.processor if self.processor is not None else self.tokenizer
            return proc.apply_chat_template(
                messages, reasoning_effort=self.reasoning_effort, **kw
            )

        if preserve_thinking:
            patched = self._thinking_preserving_template()
            if patched is not None:
                return self.tokenizer.apply_chat_template(messages, chat_template=patched, **kw)
        try:
            return self.tokenizer.apply_chat_template(messages, enable_thinking=False, **kw)
        except TypeError:
            # tokenizers without the Qwen `enable_thinking` kwarg
            return self.tokenizer.apply_chat_template(messages, **kw)

    def extract_activations(
        self,
        items: Sequence[Item],
        *,
        follow_up: tuple[str, str] | None = DEFAULT_FOLLOWUP,
        batch_size: int = 8,
        preserve_thinking: bool = False,
        progress: bool = False,
    ) -> np.ndarray:
        """-> float32 array [n, n_layers+1, d_model], the residual stream at the final real token,
        every layer. ``follow_up`` must match what the probe was trained with. ``preserve_thinking``
        keeps the rollout's `<think>` in the follow-up render (else Qwen strips it → no-CoT).
        ``progress=True`` shows a tqdm bar (handy on slow MPS runs)."""
        if self.remote:
            return self._extract_remote(items, follow_up, batch_size, preserve_thinking)
        with self._lock:
            return self._extract_local(items, follow_up, batch_size, preserve_thinking, progress)

    def _extract_local(self, items, follow_up, batch_size, preserve_thinking, progress) -> np.ndarray:
        """``extract_activations`` on the in-process model (the caller holds ``_lock``)."""
        import torch

        if not items:
            return np.empty((0, self.n_layers, self.d_model), dtype=np.float32)
        texts = [self._render(it, follow_up, preserve_thinking) for it in items]
        feats: list[np.ndarray] = []
        # Left-pad so the final real token is always the last column -> simple to index.
        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        starts = range(0, len(texts), batch_size)
        if progress:
            from tqdm.auto import tqdm

            starts = tqdm(starts, desc="extract_activations", unit="batch", total=len(starts))
        try:
            for start in starts:
                batch = texts[start : start + batch_size]
                enc = self.tokenizer(
                    batch, return_tensors="pt", padding=True, add_special_tokens=False
                ).to(self.device)
                with torch.no_grad():
                    # Request hidden states at call time too — some archs (e.g. Qwen3.5 with a nested
                    # text config) don't propagate the load-time output_hidden_states flag.
                    out = self.model(**enc, output_hidden_states=True)
                # hidden_states: tuple length n_layers+1, each [B, T, d]. Select the final real token
                # (last col, left padding) PER LAYER *before* stacking — stacking first would build the
                # full [B, L, T, d] tensor (~16 GiB at T=4096), the OOM we hit. This keeps only [B, L, d].
                last = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1)  # [B, L, d]
                feats.append(last.to(torch.float32).cpu().numpy())
                # Free the batch's activations before the next chunk so peak memory tracks batch_size,
                # not the total number of rollouts (lets eval_size grow without OOM).
                del enc, out, last
                device = str(self.device)
                if device == "mps":
                    torch.mps.empty_cache()
                elif device.startswith("cuda"):
                    torch.cuda.empty_cache()
        finally:
            self.tokenizer.padding_side = prev_side
        return np.concatenate(feats, axis=0)

    @property
    def _text_config(self):
        # Newer archs (e.g. Qwen3.5, Inkling) nest hidden_size/num_hidden_layers under a text sub-config.
        cfg = self.model.config
        return cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg

    @property
    def n_layers(self) -> int:
        if self.remote:
            return self._n_layers
        return int(self._text_config.num_hidden_layers) + 1  # + embeddings

    @property
    def d_model(self) -> int:
        if self.remote:
            return self._d_model
        return int(self._text_config.hidden_size)
