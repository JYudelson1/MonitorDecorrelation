"""``WhiteBoxModel``: a local HF model used only to *read activations* (never to generate).

The Atlas follow-up technique: raw-generation activations don't generalize, so we append a fixed
follow-up turn (``User: Did you respond truthfully?`` / ``Assistant: Yes``) to the rollout and read
the residual stream at the **final token** of that follow-up, per layer. The *same* follow-up is
applied at train and deploy time.

Model-agnostic: default is a tiny ``Qwen/Qwen3-0.6B`` for local/CPU validation; ``Qwen/Qwen3-8B`` for
the real run is just a ``model_name`` arg. Device auto-selects cuda -> mps -> cpu; dtype is bf16 on
accelerators, fp32 on CPU.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

from monitordecorrelation.types import Rollout

# Default follow-up turn (Atlas §4.3), deception-framed. Adjustable per probe/dataset: pass a
# different ``follow_up`` to ``extract_activations`` (e.g. a sycophancy-specific question). The probe
# stores the follow-up it was trained with in its meta, so deploy uses the *identical* one.
FOLLOWUP_QUESTION = "Did you respond truthfully?"
FOLLOWUP_ANSWER = "Yes"
DEFAULT_FOLLOWUP = (FOLLOWUP_QUESTION, FOLLOWUP_ANSWER)

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
    train-time (dataset adapters) and deploy-time (live rollouts) produce byte-identical text."""
    return f"<think>{cot}</think>\n{answer}" if cot else answer


class WhiteBoxModel:
    remote: bool = False  # class default so stubs / __init__-bypassing callers behave as local

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", device: str | None = None,
                 server_url: str | None = None) -> None:
        """Local mode (default): load the HF model for activation reads. **Remote mode** (``server_url``
        set): load NOTHING locally — proxy ``extract_activations`` to a shared ``probe_server.py`` that
        holds one copy of the model for all runs. Same ``.extract_activations`` interface either way, so
        ``ProbeMonitor`` is unchanged. Remote mode removes the per-run 16 GB model copy → far higher
        run-parallelism (bounded then by tinker/API limits, not local GPU memory)."""
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
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device or _pick_device()
        self.dtype = torch.bfloat16 if self.device in ("cuda", "mps") else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.dtype, output_hidden_states=True
        )
        self.model.to(self.device)
        self.model.eval()

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
        """The rollout turn (+ optional follow-up turn). The CoT is folded into the assistant message.

        ``follow_up=None`` → **within-generation**: the rollout is the FINAL turn, so we read its own
        last answer token with the CoT still in context. Required for CoT probes on *thinking* models:
        the chat template strips ``<think>`` from non-final turns, so the follow-up variant (a later
        turn) is structurally no-CoT. ``follow_up=(q, a)`` → the Atlas follow-up technique (no-CoT on
        thinking models, fine for non-reasoning models)."""
        assistant = fold_assistant(cot, answer)
        msgs = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant},
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
        preservation). Cached."""
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
        rollout's `<think>` even when a follow-up turn follows it (else Qwen strips it)."""
        q, cot, ans = _as_triple(item)
        messages = self._build_messages(q, cot, ans, follow_up)
        kw = dict(tokenize=False, add_generation_prompt=False)
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
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
        finally:
            self.tokenizer.padding_side = prev_side
        return np.concatenate(feats, axis=0)

    @property
    def _text_config(self):
        # Newer archs (e.g. Qwen3.5) nest hidden_size/num_hidden_layers under a text sub-config.
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
