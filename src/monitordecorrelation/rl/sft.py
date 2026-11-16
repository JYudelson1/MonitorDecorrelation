"""Response-only supervised fine-tuning (SFT) data: chat rows -> tinker datums + epoch batches.

Used by ``experiments/character_sft.py`` (character training by distillation: a teacher's in-character
answers, trained on as the policy's visible answer). The loop itself lives in that script; this module
is the part worth testing offline.

Rows are ``{"messages": [system?, user, assistant], "enable_thinking": false}`` — the format of the
released Corin character data (github.com/barbonara/character-reward-hacking, ``data/character_training``).
Only the assistant's text (+ its end-of-turn tokens) carries loss. How the prompt and the no-thinking
turn are framed is model-family specific, so the renderer builds the tokens
(``renderer.response_only_example``); the next-token shift is tinker-cookbook's
``datum_from_model_input_weights``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import tinker


def load_response_only_rows(path: str | Path) -> list[dict]:
    """Rows of a response-only chat JSONL as ``{"system": str | None, "user": str, "assistant": str}``,
    in file order. A row of any other shape is an error (nothing is skipped or reinterpreted): the
    roles must be [system,] user, assistant; ``enable_thinking`` must be present and false (a row that
    asks for thinking would need a reasoning target, which this path does not build); no other key."""
    rows = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        where = f"{path}:{n}"
        if not isinstance(row, dict) or set(row) != {"messages", "enable_thinking"}:
            raise ValueError(f"{where}: expected exactly the keys 'messages' and 'enable_thinking', got "
                             f"{sorted(row) if isinstance(row, dict) else type(row).__name__}")
        if row["enable_thinking"] is not False:
            raise ValueError(f"{where}: enable_thinking={row['enable_thinking']!r} — only response-only "
                             "rows (enable_thinking: false) are supported")
        msgs = row["messages"]
        roles = [m.get("role") for m in msgs]
        if roles not in (["system", "user", "assistant"], ["user", "assistant"]):
            raise ValueError(f"{where}: roles {roles} — expected [system,] user, assistant")
        if any(not isinstance(m.get("content"), str) or not m["content"] for m in msgs):
            raise ValueError(f"{where}: every message needs non-empty string content")
        rows.append({"system": msgs[0]["content"] if len(msgs) == 3 else None,
                     "user": msgs[-2]["content"], "assistant": msgs[-1]["content"]})
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def response_only_datum(renderer, row: dict, max_length: int) -> tinker.Datum:
    """One cross-entropy datum for ``row``: weight 0 on the prompt, 1 on every target token. A row
    longer than ``max_length`` tokens is an error, not a truncation (a cut answer would train the
    policy to stop mid-sentence without its end-of-turn tokens)."""
    import torch
    from tinker_cookbook.supervised.common import datum_from_model_input_weights

    if not hasattr(renderer, "response_only_example"):
        raise NotImplementedError(
            f"renderer {getattr(renderer, 'name', type(renderer).__name__)!r} has no response-only SFT "
            "framing (only TML / Inkling policies do so far)")
    prompt, target = renderer.response_only_example(row["user"], row["assistant"], system=row["system"])
    if len(prompt) + len(target) > max_length:
        raise ValueError(f"row is {len(prompt) + len(target)} tokens > max_length={max_length}")
    weights = torch.cat([torch.zeros(len(prompt)), torch.ones(len(target))])
    return datum_from_model_input_weights(tinker.ModelInput.from_ints(prompt + target), weights,
                                          max_length, reduction="none")


def epoch_batches(n_rows: int, batch_size: int, *, epoch: int, seed: int) -> list[list[int]]:
    """Row indices of one epoch's batches: a fresh shuffle per (seed, epoch), full batches only — the
    last partial batch is dropped, as in the Corin SFT runs (1,050 rows, batch 32 -> 32 steps/epoch)."""
    if n_rows < batch_size:
        raise ValueError(f"{n_rows} rows cannot fill one batch of {batch_size}")
    order = list(range(n_rows))
    random.Random(f"{seed}:{epoch}").shuffle(order)
    return [order[i:i + batch_size] for i in range(0, n_rows - batch_size + 1, batch_size)]


def format_reward(env, rollout) -> float:
    """The FORMAT-ONLY reward of the format-keeping RL steps ``character_sft.py`` can interleave with
    SFT: 1.0 for a VALID rollout (``envs.base.invalid_reason`` is None: not truncated, every turn
    parseable, and — terminal env — an answer was submitted), else 0.0. It never looks at whether the
    answer is right, at the oracle (``behavior_present``), or at any monitor: a hack and an honest
    answer in the right format both get 1."""
    from monitordecorrelation.envs.base import invalid_reason

    return 1.0 if invalid_reason(env, rollout) is None else 0.0
