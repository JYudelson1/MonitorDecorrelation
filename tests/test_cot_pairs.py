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


def _lr(behavior: bool, finished: bool = True, parsed: bool = True, cot: str = "reasoning") -> LabeledRollout:
    return LabeledRollout(
        rollout=Rollout(prompt=Prompt(text="q"), cot=cot, output="ans"),
        behavior=behavior,
        finished=finished,
        parsed=parsed,
    )


def test_base_rates():
    labeled = [_lr(True), _lr(True), _lr(False), _lr(True, finished=False), _lr(False, parsed=False)]
    r = base_rates(labeled)
    assert r["n_total"] == 5 and r["n_usable"] == 3  # two dropped (unfinished / unparsed)
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
    print("pick_pair + pair_to_dict OK")


if __name__ == "__main__":
    test_base_rates()
    test_decide_nudges()
    test_pick_pair_and_dict()
    print("ALL PASS")
