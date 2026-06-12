"""CoT-pair builder pure logic: base rates, nudge decision, pairing. Offline (no tinker/network).

Run: uv run python tests/test_cot_pairs.py
"""

from __future__ import annotations

from monitordecorrelation.types import Prompt, Rollout
from monitordecorrelation.whitebox.cot_pairs import (
    LabeledRollout,
    base_rates,
    decide_nudges,
    pair_to_dict,
    pick_pair,
)


def _lr(behavior: bool, finished: bool = True, parsed: bool = True, evasive: bool = False, cot: str = "reasoning") -> LabeledRollout:
    return LabeledRollout(
        rollout=Rollout(prompt=Prompt(text="q"), cot=cot, output="ans"),
        behavior=behavior,
        finished=finished,
        parsed=parsed,
        evasive=evasive,
    )


def test_base_rates():
    labeled = [_lr(True), _lr(True), _lr(False), _lr(True, finished=False),
               _lr(False, parsed=False), _lr(False, evasive=True)]
    r = base_rates(labeled)
    assert r["n_total"] == 6 and r["n_usable"] == 3  # unfinished / unparsed / EVASIVE all dropped
    assert r["present"] == 2 and r["absent"] == 1
    assert abs(r["present_rate"] - 2 / 3) < 1e-9
    print("base_rates OK")


def test_decide_nudges():
    # mostly deceptive -> need the honest nudge, not the deceptive one
    assert decide_nudges({"present_rate": 0.95}, 0.2) == {"need_deceptive": False, "need_honest": True}
    # balanced -> no nudge (use natural rollouts for both)
    assert decide_nudges({"present_rate": 0.5}, 0.2) == {"need_deceptive": False, "need_honest": False}
    # mostly honest -> need the deceptive nudge
    assert decide_nudges({"present_rate": 0.05}, 0.2)["need_deceptive"] is True
    print("decide_nudges OK")


def test_pick_pair_and_dict():
    dec, hon = pick_pair([_lr(True, cot="lie"), _lr(False, cot="truth")])
    assert dec.behavior is True and hon.behavior is False
    d = pair_to_dict("Q", dec, hon)
    assert d == {
        "prompt": "Q",
        "honest_cot": "truth",
        "honest_answer": "ans",
        "deceptive_cot": "lie",
        "deceptive_answer": "ans",
        "via": "natural",  # neither side nudged
    }
    # a nudged side flips via -> "nudged"
    dn = _lr(True, cot="lie")
    dn.nudged = True
    assert pair_to_dict("Q", dn, hon)["via"] == "nudged"
    # natural-first preference: an earlier usable honest beats a later one
    g = [_lr(True), _lr(False, cot="first"), _lr(False, cot="second")]
    assert pick_pair(g)[1].rollout.cot == "first"
    # missing a class / only-unusable members -> None
    assert pick_pair([_lr(True), _lr(True)]) is None
    assert pick_pair([_lr(True), _lr(False, parsed=False)]) is None
    # an evasive (hedge) honest is excluded -> no clean honest -> None
    assert pick_pair([_lr(True), _lr(False, evasive=True)]) is None
    print("pick_pair + pair_to_dict OK")


def test_cot_jsonl_fold_toggle():
    """Same jsonl -> CoT (fold <think>) vs no-CoT (answer only) training sets."""
    import json
    import tempfile
    from pathlib import Path

    from monitordecorrelation.whitebox.datasets.cot_jsonl import load_cot_jsonl

    row = {
        "prompt": "Q",
        "honest_cot": "think H",
        "honest_answer": "A",
        "deceptive_cot": "think D",
        "deceptive_answer": "B",
        "via": "natural",
    }
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "pairs.jsonl"
        p.write_text(json.dumps(row) + "\n")

        cot = load_cot_jsonl(p, fold_cot=True)[0]
        assert "<think>think H</think>" in cot.honest and cot.honest.endswith("A")
        assert "<think>think D</think>" in cot.deceptive
        assert cot.meta["fold_cot"] is True and cot.meta["via"] == "natural"

        nocot = load_cot_jsonl(p, fold_cot=False)[0]
        assert nocot.honest == "A" and nocot.deceptive == "B"
        assert "<think>" not in nocot.honest and nocot.meta["fold_cot"] is False
    print("cot_jsonl fold toggle OK")


if __name__ == "__main__":
    test_base_rates()
    test_decide_nudges()
    test_pick_pair_and_dict()
    test_cot_jsonl_fold_toggle()
    print("ALL PASS")
