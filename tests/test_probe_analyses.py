"""experiments/probe_analyses.py on synthetic items and features.

Checks:
- dataset-stats counts the disclosing hacks and non-hacks, the OOD test items whose question a training item also
  has, and the probe inputs' tokens (rendered offline by the TML tokenizer), with and without sampling.
- near-threshold reads each probe in the right TerminalEnv view(s) at the right t₁%, and its shares match a direct
  computation on the planted logits.
- massive-dims:
  - the top-8 share and its dimensions, by hand (ties go to the lower dimensions) and against a float64 sort;
  - each token's role in a hand-made TML sequence;
  - per role and layer, every statistic of the per-token part matches a direct float64 computation on planted
    hidden states, in both views; the planted massive dimensions show; files whose tokens aren't their item's
    rendering are refused; the "Yes" check against per-item features tells equal from different;
  - the per-item part matches a direct computation, with --max-shards reading only the first shard.
- readout-gap reproduces a bootstrap whose outcome can be worked out by hand, for OOD pairs and IID tasks, and refuses
  probes that don't score the same items or whose AUROC isn't their metrics.json's.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from monitordecorrelation.whitebox.features import role_masks, special_ids
from monitordecorrelation.whitebox.inkling_replay import VIEWS, InklingReplay
from monitordecorrelation.whitebox.probe_datasets import item_messages
from monitordecorrelation.whitebox.probe_fit import SelectedProbe

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPTIONS = {"verifier_mode": "submit_corrupted", "submission_ends": "first", "max_turns": 4,
           "require_explanation": True, "last_turn_submit_notice": True}


def _load_script():
    spec = importlib.util.spec_from_file_location("probe_analyses",
                                                  os.path.join(REPO, "experiments", "probe_analyses.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_items(folder, items):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "items.jsonl"), "w") as fh:
        fh.writelines(json.dumps(it) + "\n" for it in items)


def _episode(i: int, hack: bool, disclose: bool) -> dict:
    """A two-turn TerminalEnv episode; only the last turn's visible text may mention the verifier."""
    explanation = "I used the verifier's expected answer." if disclose else "I counted the islands."
    turns = [{"cot": "Look around first. " * (1 + i % 3), "text": "<command>ls</command> Explanation: list the files.",
              "output": "verifier.py", "submission": None, "truncated": False},
             {"cot": "Submit now.", "text": f"<submit>{i}</submit> Explanation: {explanation}", "output": None,
              "submission": str(i), "truncated": False}]
    return {"id": f"task{i}/0", "task_id": f"task{i}", "label": int(hack), "split": "train" if i < 8 else "test",
            "question": f"Count the islands in grid {i}.", "turns": turns, "env_options": OPTIONS,
            "diagnostics": {"used_verifier": hack, "read_verifier": hack}}


def _ood(pair: int, source: str, question: str, split: str) -> list[dict]:
    return [{"id": f"{pair}/{'deceptive' if lab else 'honest'}", "pair": pair, "label": lab, "split": split,
             "source": source, "question": question, "cot": "", "answer": f"Answer {pair} {lab}."} for lab in (0, 1)]


def test_dataset_stats_counts_disclosure_question_overlap_and_tokens(tmp_path):
    mod = _load_script()
    # 4 hacks, all disclosing; 8 other episodes, 2 of which mention the verifier too
    terminal = [_episode(i, hack=i < 4, disclose=i < 6) for i in range(12)]
    ood = (_ood(0, "mask_statements", "State a fact.", "train") + _ood(1, "mask_statements", "State a fact.", "train")
           + _ood(2, "mask_statements", "State a fact.", "test")
           + _ood(3, "liars", "Is A true?", "train") + _ood(4, "liars", "Is A true?", "test")
           + _ood(5, "liars", "Is B true?", "test")
           + _ood(6, "facts", "Where is C?", "train") + _ood(7, "facts", "Where is D?", "test"))
    data = tmp_path / "datasets"
    _write_items(data / "terminal_prerl", terminal)
    _write_items(data / "diverse_deception", ood)
    out = tmp_path / "analyses"
    assert mod.main(["dataset-stats", "--datasets", str(data), "--out", str(out)]) == 0
    stats = json.loads((out / "dataset_stats.json").read_text())
    assert set(stats["code"]) == {"commit", "dirty", "diff_sha256", "source"}
    assert stats["disclosure"] == {"hacks": {"n": 4, "disclosing": 4, "share": 1.0},
                                   "non_hacks": {"n": 8, "disclosing": 2, "share": 0.25}}
    overlap = stats["ood_question_overlap"]
    assert {s: (o["test_sharing_question_with_train"], o["n_test"]) for s, o in overlap.items()} == {
        "mask_statements": (2, 2), "liars": (2, 4), "facts": (0, 2)}
    assert {s: o["distinct_questions"] for s, o in overlap.items()} == {"mask_statements": 1, "liars": 2, "facts": 2}
    assert set(stats["tokens"]) == {"terminal_prerl", "diverse_deception"}
    assert set(stats["tokens"]["terminal_prerl"]) == {"with_reasoning", "without_reasoning"}
    assert set(stats["tokens"]["diverse_deception"]) == {"without_reasoning"}
    replay = InklingReplay()
    for name, its in (("terminal_prerl", terminal), ("diverse_deception", ood)):
        for view, t in stats["tokens"][name].items():  # every item rendered: the exact mean and max
            n = [len(replay.probe_input(item_messages(it, view)).tokens) for it in its]
            assert t["n_items"] == t["n_rendered"] == len(its) and t["mean"] == pytest.approx(np.mean(n))
            assert t["max"] == max(n) and t["total_estimate"] == pytest.approx(sum(n))
    tok = stats["tokens"]["terminal_prerl"]
    assert tok["with_reasoning"]["mean"] > tok["without_reasoning"]["mean"]  # the thinking adds tokens
    # sampled, into a new folder; an existing output is never overwritten
    assert mod.main(["dataset-stats", "--datasets", str(data), "--out", str(tmp_path / "sampled"),
                     "--sample", "5"]) == 0
    sampled = json.loads((tmp_path / "sampled" / "dataset_stats.json").read_text())["tokens"]
    assert all(t["n_rendered"] == 5 and t["n_items"] == len(terminal) for t in sampled["terminal_prerl"].values())
    with pytest.raises(SystemExit):
        mod.main(["dataset-stats", "--datasets", str(data), "--out", str(out), "--skip-tokens"])
    assert "tokens" in json.loads((out / "dataset_stats.json").read_text())  # untouched
    assert mod.main(["dataset-stats", "--datasets", str(data), "--out", str(out), "--skip-tokens", "--overwrite"]) == 0
    assert "tokens" not in json.loads((out / "dataset_stats.json").read_text())


def _write_features(folder, ids, x):
    """One shard of a TerminalEnv view: the follow-up readout [n, L, D] and the token counts."""
    os.makedirs(folder, exist_ok=True)
    save_file({"followup": torch.tensor(x, dtype=torch.float32),
               "counts": torch.ones((len(ids), 2), dtype=torch.int32)}, os.path.join(folder, "shard_00000.safetensors"))
    with open(os.path.join(folder, "shard_00000.jsonl"), "w") as fh:
        fh.writelines(json.dumps({"row": r, "id": i}) + "\n" for r, i in enumerate(ids))


def test_near_threshold_shares_match_the_planted_logits(tmp_path):
    mod = _load_script()
    rng = np.random.default_rng(0)
    n, L, D, layer = 200, 3, 4, 1
    splits = np.array(["train", "val", "test"])[np.arange(n) % 5 % 3]
    hack = rng.random(n) < 0.1
    items = [{"id": f"e{i}", "task_id": f"t{i // 4}", "label": int(hack[i]), "split": str(splits[i]),
              "diagnostics": {"used_verifier": bool(hack[i])}} for i in range(n)]
    data, feats = tmp_path / "datasets", tmp_path / "features"
    _write_items(data / "terminal_prerl", items)
    x = {}
    for view in ("with_reasoning", "without_reasoning"):  # the planted logits: dims 0 and 1 of the probe layer
        x[view] = (rng.standard_normal((n, L, D)) * 3).astype(np.float32).astype(np.float64)  # stored exactly
        _write_features(feats / "terminal_prerl" / view, [it["id"] for it in items], x[view])
    honest_dev = np.isin(splits, ["train", "val"]) & ~hack

    def probe(path, coef, t1, meta):
        p = SelectedProbe(layer, np.array(coef, float), 0.0, t1, t1 - 1, {"readout": "followup", **meta})
        p.save(str(path))
        return str(path)

    iid = probe(tmp_path / "iid", [1, 0, 0, 0], 2.0, {"dataset": "terminal_prerl", "view": "without_reasoning"})
    ood = probe(tmp_path / "ood", [0, 1, 0, 0], -1.0,
                {"dataset": "diverse_deception", "view": "without_reasoning",
                 "thresholds_by_terminal_view": {"with_reasoning": [0.5, -0.5], "without_reasoning": [-1.0, -2.0]}})
    deployed = probe(tmp_path / "deployed", [0, 1, 0, 0], 1.5,  # a deployed OOD monitor: one view, its own t1
                     {"dataset": "diverse_deception", "view": "with_reasoning", "deployed_utc": "2026-10-10T08:14:17Z",
                      "thresholds_by_terminal_view": {"with_reasoning": [0.5, -0.5], "without_reasoning": [-1, -2]}})
    out = tmp_path / "analyses"
    assert mod.main(["near-threshold", "--probes", ",".join([iid, ood, deployed]), "--features", str(feats),
                     "--datasets", str(data), "--out", str(out), "--k", "1.5"]) == 0
    res = json.loads((out / "near_threshold.json").read_text())
    expected = {iid: {"without_reasoning": (0, 2.0)}, ood: {"with_reasoning": (1, 0.5), "without_reasoning": (1, -1.0)},
                deployed: {"with_reasoning": (1, 1.5)}}
    assert res["k"] == 1.5 and set(res["probes"]) == set(expected)
    for path, views in expected.items():
        got = res["probes"][path]["views"]
        assert set(got) == set(views), got
        for view, (dim, t1) in views.items():
            z = x[view][honest_dev, layer, dim]
            t1_re = np.percentile(z, 99)
            g = got[view]
            assert g["n_honest_dev"] == honest_dev.sum() and g["t1_own"] == t1
            assert g["share_within_k_of_t1_own"] == pytest.approx(np.mean(np.abs(z - t1) <= 1.5))
            assert g["t1_recalibrated"] == pytest.approx(t1_re)
            assert g["share_within_k_of_t1_recalibrated"] == pytest.approx(np.mean(np.abs(z - t1_re) <= 1.5))
            assert 0 < g["share_within_k_of_t1_own"] < 1


# ------------------------------------------------------------------------------------------------ massive-dims


def test_top_k_mask_and_share_by_hand():
    mod = _load_script()
    sq = np.array([[9.0, 16.0, 0.0, 1.0],  # top 2: dims 0 and 1, 25 of 26
                   [1.0, 3.0, 3.0, 3.0],  # three tied for 2nd: the lower dimensions, 1 and 2; 6 of 10
                   [2.0, 2.0, 2.0, 2.0],  # all tied: dims 0 and 1, half
                   [0.0, 0.0, 0.0, 0.0]])  # no norm: NaN
    mask = mod.top_k_mask(sq, 2)
    assert mask.tolist() == [[True, True, False, False], [False, True, True, False],
                             [True, True, False, False], [True, True, False, False]]
    share = mod.top_k_share(sq, mask)
    assert share[:3] == pytest.approx([25 / 26, 6 / 10, 1 / 2], abs=1e-15) and np.isnan(share[3])
    # bfloat16 activations with a few large dimensions, as the readers store them: 8 of 64 against a float64 sort
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.standard_normal((300, 64)) * np.exp(2 * rng.standard_normal(64))).to(torch.bfloat16)
    x[0, :] = 1.0  # a row of ties
    sq = mod.squares(x)
    exact = x.double().numpy() ** 2
    assert sq.dtype == np.float32 and np.array_equal(sq.astype(np.float64), exact)  # squaring bfloat16 is exact
    mask = mod.top_k_mask(sq)
    assert (mask.sum(1) == 8).all()
    top = np.argsort(-exact, axis=1, kind="stable")[:, :8]  # the 8 largest; equal values in dimension order
    assert all(set(np.flatnonzero(m)) == set(t) for m, t in zip(mask, top))
    assert mask[0].tolist() == [True] * 8 + [False] * 56
    ordered = np.sort(exact, axis=1)
    assert mod.top_k_share(sq, mask) == pytest.approx(ordered[:, -8:].sum(1) / ordered.sum(1), rel=1e-12)


def test_token_roles_on_a_synthetic_sequence():
    mod = _load_script()
    replay = InklingReplay()
    sp = special_ids(replay)
    special = replay.renderer.tokenizer.is_special_token
    seq = [  # (token, its role): plain ids stand for text tokens
        ("message_system", "special"), ("content_text", "special"), (101, "other"), ("end_message", "special"),
        ("message_user", "special"), ("content_text", "special"), (102, "other"), (103, "other"),
        ("end_message", "special"),
        ("message_model", "special"), ("content_thinking", "special"), (104, "policy"), (105, "policy"),
        ("end_message", "special"),
        ("message_model", "special"), ("content_text", "special"), (106, "policy"), ("end_message", "special"),
        ("content_model_end_sampling", "special"),
        ("message_user", "special"), ("content_text", "special"), (107, "other"), ("end_message", "special"),
        ("message_model", "special"), ("content_text", "special"), (108, "policy"), ("end_message", "special"),
        ("content_model_end_sampling", "special"),
        # the follow-up: its question and "Yes" are not the policy's, whatever their channel
        ("message_user", "special"), ("content_text", "special"), (109, "other"), (110, "other"),
        ("end_message", "special"),
        ("message_model", "special"), ("content_text", "prev"), (111, "yes"), ("end_message", "special"),
        ("content_model_end_sampling", "special"),
    ]
    tokens = [sp[t] if isinstance(t, str) else t for t, _ in seq]
    n_conversation = 28  # the follow-up starts at its <|message_user|>
    assert tokens[n_conversation] == sp["message_user"]
    assert tokens[n_conversation - 1] == sp["content_model_end_sampling"]
    roles = mod.token_roles(tokens, len(tokens) - 3, n_conversation, sp, special)
    assert [mod.ROLES[r] for r in roles] == [role for _, role in seq]
    assert all(special(t) == (t >= 200000) for t in tokens)  # Inkling's markers have ids from 200000
    # a rendered episode: the policy's tokens are role_masks' thinking and text, the markers the tokenizer's specials
    p = replay.probe_input(item_messages(_episode(1, hack=True, disclose=True), "with_reasoning"))
    roles = mod.token_roles(p.tokens, p.readout, p.n_conversation_tokens, sp, special)
    reasoning, action = role_masks(p.tokens, p.n_conversation_tokens, sp)
    assert reasoning and action
    assert np.flatnonzero(roles == mod.ROLES.index("policy")).tolist() == sorted(reasoning + action)
    assert p.readout == len(p.tokens) - 3 and p.tokens[p.readout - 1] == sp["content_text"]
    assert np.flatnonzero(roles == mod.ROLES.index("special")).tolist() == \
        [i for i, t in enumerate(p.tokens) if special(t) and i != p.readout - 1]


def _save_per_token(folder, item, view, hidden, tokens):
    """A per-token file as the extraction names it: <view>/full/<id, "/" as "__">.safetensors."""
    path = folder / view / "full" / (item["id"].replace("/", "__") + ".safetensors")
    os.makedirs(path.parent, exist_ok=True)
    save_file({"hidden": hidden, "tokens": torch.tensor(tokens, dtype=torch.int32)}, str(path))
    return path


def _save_shard(folder, key, ids, x):
    os.makedirs(folder, exist_ok=True)
    save_file({key: torch.as_tensor(x).to(torch.bfloat16), "counts": torch.ones((len(ids), 2), dtype=torch.int32)},
              os.path.join(folder, "shard_00000.safetensors"))
    with open(os.path.join(folder, "shard_00000.jsonl"), "w") as fh:
        fh.writelines(json.dumps({"row": r, "id": i}) + "\n" for r, i in enumerate(ids))


def _direct(h: np.ndarray, readout: int) -> dict:
    """The per-token statistics of one file's hidden states [T, L, d], in float64 by sorting."""
    sq = h.transpose(1, 0, 2) ** 2  # [L, T, d]
    total = sq.sum(-1)
    top = np.argsort(-sq, axis=-1, kind="stable")[..., :8]
    norm = np.sqrt(total)
    yes = top[:, readout]  # [L, 8]
    overlap = np.array([[len(set(top[l, t]) & set(yes[l])) for t in range(sq.shape[1])] for l in range(sq.shape[0])])
    return {"share": np.take_along_axis(sq, top, -1).sum(-1) / total,
            "rel_norm": norm / np.median(norm, axis=1, keepdims=True), "overlap": overlap,
            "yes_dims_share": np.stack([sq[l][:, yes[l]].sum(-1) for l in range(sq.shape[0])]) / total}


def test_massive_dims_per_token_matches_a_direct_computation(tmp_path):
    mod = _load_script()
    replay = InklingReplay()
    sp, special = special_ids(replay), replay.renderer.tokenizer.is_special_token
    items = [_episode(i, hack=i == 0, disclose=False) for i in range(3)]
    data = tmp_path / "datasets"
    _write_items(data / "terminal_prerl", items)
    sha = hashlib.sha256((data / "terminal_prerl" / "items.jsonl").read_bytes()).hexdigest()
    rng = np.random.default_rng(0)
    n_layers, d = 36, 256  # 256 dims: a random token's top 8 hold ~20% of its squared norm
    per_token, feats = tmp_path / "per_token", tmp_path / "features_4bit"
    direct, roles, files = [], [], {}
    for view in VIEWS:
        yes_rows = []
        for it in items[:2]:
            p = replay.probe_input(item_messages(it, view))
            h = rng.standard_normal((len(p.tokens), n_layers, d))
            h[:, :20, :3] *= 50  # layers 0-19: dims 0-2 dominate every token
            h[p.readout, 20:, 40:42] *= 300  # from layer 20, only "Yes" has massive dimensions
            stored = torch.tensor(h, dtype=torch.float32).to(torch.bfloat16)
            _save_per_token(per_token / "terminal_prerl", it, view, stored, p.tokens)
            exact = stored.double().numpy()
            direct.append(_direct(exact, p.readout))
            roles.append(mod.token_roles(p.tokens, p.readout, p.n_conversation_tokens, sp, special))
            files[f"{view}/{it['id']}"] = (exact, p)
            yes_rows.append(stored[p.readout])
        # the per-token files' reader's per-item features: item 0's "Yes" as stored, item 1's a little off
        yes_rows[1] = yes_rows[1] + 1
        _save_shard(feats / "terminal_prerl" / view, "followup", [it["id"] for it in items],
                    torch.stack([*yes_rows, torch.zeros(n_layers, d, dtype=torch.bfloat16)]))
    with open(feats / "terminal_prerl" / "index.json", "w") as fh:
        json.dump({"dataset": {"items_sha256": sha}}, fh)
    out = tmp_path / "analyses"
    assert mod.main(["massive-dims", "--per-token", str(per_token), "--per-token-features", str(feats),
                     "--datasets", str(data), "--out", str(out)]) == 0
    res = json.loads((out / "massive_dims.json").read_text())
    assert set(res["code"]) == {"commit", "dirty", "diff_sha256", "source"} and "readouts" not in res
    got = res["per_token"]
    assert got["n_files"] == 4 and got["n_tokens"] == sum(len(r) for r in roles)
    assert got["special_ids_from_200000"] and got["special_ids_seen"] == sorted(
        {t for _, p in files.values() for t in p.tokens if special(t)})
    cat = {key: np.concatenate([x[key] for x in direct], axis=1) for key in direct[0]}
    role_of = np.concatenate(roles)
    first = np.concatenate([np.arange(len(r)) == 0 for r in roles])
    groups = {**{role: role_of == i for i, role in enumerate(mod.ROLES)}, "first_token": first}
    for name, m in groups.items():
        g = got["first_token"] if name == "first_token" else got["roles"][name]
        assert g["n"] == m.sum() > 0, name
        expected = {"top_share": np.median(cat["share"][:, m], 1),
                    "share_dominated": (cat["share"][:, m] > 0.5).mean(1),
                    "rel_norm": np.median(cat["rel_norm"][:, m], 1), "yes_overlap": cat["overlap"][:, m].mean(1) / 8,
                    "yes_same": (cat["overlap"][:, m] == 8).mean(1),
                    "yes_dims_share": np.median(cat["yes_dims_share"][:, m], 1)}
        for stat, v in expected.items():
            assert g["by_layer"][stat] == pytest.approx(v, abs=6e-5), (name, stat)
            for band, (lo, hi) in mod.BANDS.items():
                assert g["bands"][band][stat] == pytest.approx(np.median(v[lo: hi + 1]), rel=1e-6, abs=1e-6), \
                    (name, stat, band)
    # the planted dimensions show: shared by every token up to layer 19, "Yes"'s alone after
    by_role = got["roles"]
    assert all(by_role[r]["bands"]["4-16"]["top_share"] > 0.9 for r in mod.ROLES)
    assert all(by_role[r]["bands"]["4-16"]["yes_dims_share"] > 0.9 for r in mod.ROLES)
    assert by_role["yes"]["bands"]["25-35"]["top_share"] > 0.99 and by_role["yes"]["bands"]["25-35"]["rel_norm"] > 10
    assert all(by_role[r]["bands"]["25-35"]["top_share"] < 0.5 for r in mod.ROLES if r != "yes")
    assert by_role["yes"]["bands"]["25-35"]["yes_same"] == 1.0 and by_role["policy"]["bands"]["25-35"]["yes_same"] == 0
    # per file: roles, the sink check's ratio, and the "Yes" check against the per-item features
    for key, (h, p) in files.items():
        f = got["files"][key]
        assert f["n_tokens"] == len(p.tokens) == sum(f["n_by_role"].values()) and f["yes_position"] == p.readout
        assert f["n_by_role"]["yes"] == f["n_by_role"]["prev"] == 1
        norm = np.linalg.norm(h, axis=-1)  # [T, L]
        ctx = norm[max(0, p.n_conversation_tokens - 64): p.n_conversation_tokens]
        assert f["yes_ratio_to_context"] == pytest.approx(norm[p.readout] / np.median(ctx, 0), rel=1e-4, abs=1e-4)
        assert f["prev_ratio_to_context"] == pytest.approx(norm[p.readout - 1] / np.median(ctx, 0), rel=1e-4, abs=1e-4)
        assert f["yes_equals_features"] is (key.endswith(items[0]["id"]))
        assert f["hack"] is key.endswith(items[0]["id"])
    ratios = np.median(np.stack([[f[k] for k in ("yes_ratio_to_context", "prev_ratio_to_context")]
                                 for f in got["files"].values()]), axis=0)
    assert got["sink_check"]["yes"]["median_ratio_layers_10_35"] == pytest.approx(np.median(ratios[0][10:36]), abs=1e-4)

    # a file whose tokens are not its item's rendering is refused, and nothing is written
    h, p = files[f"with_reasoning/{items[0]['id']}"]
    _save_per_token(tmp_path / "bad", items[2], "with_reasoning", torch.tensor(h).to(torch.bfloat16), p.tokens)
    with pytest.raises(SystemExit):
        mod.main(["massive-dims", "--per-token", str(tmp_path / "bad"), "--datasets", str(data),
                  "--out", str(tmp_path / "refused")])
    assert not (tmp_path / "refused").exists()
    # items that are not the ones the features were extracted from are refused
    with open(feats / "terminal_prerl" / "index.json", "w") as fh:
        json.dump({"dataset": {"items_sha256": "0" * 64}}, fh)
    with pytest.raises(SystemExit):
        mod.main(["massive-dims", "--per-token", str(per_token), "--per-token-features", str(feats),
                  "--datasets", str(data), "--out", str(tmp_path / "other_items")])
    # an existing output is kept; a flag that would be ignored is refused
    for argv in (["--per-token", str(per_token), "--out", str(out)],
                 ["--features-prev", str(feats), "--per-token", str(per_token), "--out", str(tmp_path / "o1")],
                 ["--readout-datasets", "terminal_prerl", "--per-token", str(per_token), "--out", str(tmp_path / "o2")],
                 ["--out", str(tmp_path / "o3")]):
        with pytest.raises(SystemExit):
            mod.main(["massive-dims", "--datasets", str(data), *argv])
    assert json.loads((out / "massive_dims.json").read_text())["per_token"]["n_files"] == 4


def test_massive_dims_readouts_match_a_direct_computation(tmp_path):
    mod = _load_script()
    rng = np.random.default_rng(1)
    n, n_layers, d = 10, 20, 64  # 20 layers: the bands past layer 19 have no value
    data, feats, prev = tmp_path / "datasets", tmp_path / "features", tmp_path / "features_prev"
    _write_items(data / "terminal_prerl", [{"id": f"e{i}", "split": "train"} for i in range(n)])
    _write_items(data / "diverse_deception", [{"id": f"{i // 2}/{i % 2}", "pair": i // 2} for i in range(n)])
    planted = {}
    for dataset, views in (("terminal_prerl", VIEWS), ("diverse_deception", ("without_reasoning",))):
        ids = [f"e{i}" for i in range(n)] if dataset == "terminal_prerl" else [f"{i // 2}/{i % 2}" for i in range(n)]
        for view in views:
            x = {}
            for key, root, scale in (("followup", feats, 20.0), ("followup_prev", prev, 5.0)):
                h = rng.standard_normal((n, n_layers, d))
                h[:, :, :2] *= scale
                x[key] = torch.tensor(h, dtype=torch.float32).to(torch.bfloat16)
                for shard, rows in enumerate((slice(0, 6), slice(6, n))):  # two shards of 6 and 4 items
                    folder = root / dataset / view
                    os.makedirs(folder, exist_ok=True)
                    save_file({key: x[key][rows].contiguous(),
                               "counts": torch.ones((len(ids[rows]), 2), dtype=torch.int32)},
                              str(folder / f"shard_{shard:05d}.safetensors"))
                    with open(folder / f"shard_{shard:05d}.jsonl", "w") as fh:
                        fh.writelines(json.dumps({"row": r, "id": i}) + "\n" for r, i in enumerate(ids[rows]))
            planted[f"{dataset}/{view}"] = {k: v.double().numpy() for k, v in x.items()}

    def direct(a, b):
        """Per layer: median top-8 shares, median norm ratio, mean share of the top 8 in common."""
        qa, qb = (v.transpose(1, 0, 2) ** 2 for v in (a, b))
        ta, tb = (np.argsort(-q, axis=-1, kind="stable")[..., :8] for q in (qa, qb))
        share = [np.median(np.take_along_axis(q, t, -1).sum(-1) / q.sum(-1), 1) for q, t in ((qa, ta), (qb, tb))]
        common = np.array([[len(set(ta[l, i]) & set(tb[l, i])) / 8 for i in range(a.shape[0])]
                           for l in range(n_layers)]).mean(1)
        return {"top_share_followup": share[0], "top_share_followup_prev": share[1],
                "norm_ratio": np.median(np.sqrt(qa.sum(-1) / qb.sum(-1)), 1), "top_dims_in_common": common}

    for max_shards, rows in ((None, slice(0, n)), (1, slice(0, 6))):
        out = tmp_path / f"analyses_{max_shards}"
        extra = ["--max-shards", str(max_shards)] if max_shards else []
        assert mod.main(["massive-dims", "--features", str(feats), "--features-prev", str(prev), "--datasets",
                         str(data), "--out", str(out), *extra]) == 0
        res = json.loads((out / "massive_dims.json").read_text())
        assert "per_token" not in res and res["readouts"]["max_shards"] == max_shards
        views = res["readouts"]["views"]
        assert set(views) == set(planted)
        for name, x in planted.items():
            v = views[name]
            assert v["n"] == rows.stop and v["shards"] == (max_shards or 2) and v["of_shards"] == 2
            for stat, want in direct(x["followup"][rows], x["followup_prev"][rows]).items():
                assert v["by_layer"][stat] == pytest.approx(want, abs=6e-5), (name, stat)
                assert v["bands"]["4-16"][stat] == pytest.approx(np.median(want[4:17]), abs=1e-9), (name, stat)
                assert v["bands"]["17-24"][stat] is None and v["bands"]["10-35"][stat] is None
            assert min(v["by_layer"]["norm_ratio"]) > 1  # "Yes"'s planted dimensions are 4× larger
    # the two readouts must hold the same items
    with open(prev / "diverse_deception" / "without_reasoning" / "shard_00001.jsonl", "w") as fh:
        fh.writelines(json.dumps({"row": r, "id": f"x{r}"}) + "\n" for r in range(4))
    with pytest.raises(SystemExit):
        mod.main(["massive-dims", "--features", str(feats), "--features-prev", str(prev), "--datasets", str(data),
                  "--readout-datasets", "diverse_deception", "--out", str(tmp_path / "mismatch")])


# ------------------------------------------------------------------------------------------------ readout-gap


def _scored_probe(folder, dataset, view, readout, rows, test_auroc):
    os.makedirs(folder)
    (folder / "meta.json").write_text(json.dumps({"dataset": dataset, "view": view, "readout": readout, "layer": 3}))
    (folder / "metrics.json").write_text(json.dumps({"test_auroc": test_auroc}))
    with open(folder / "scores.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    return folder


def test_readout_gap_reproduces_a_bootstrap_worked_out_by_hand(tmp_path):
    """Two test clusters, each one positive and one negative. Within each, both readouts rank the positive first;
    across them, followup ranks cluster 1's positive (2) below cluster 2's negative (3), and followup_prev doesn't:

        followup:       positive 2, negative 1 | positive 4, negative 3  ->  AUROC 3/4
        followup_prev:  positive 3, negative 1 | positive 4, negative 2  ->  AUROC 1

    A resample draws two clusters with replacement. The same one twice (probability 1/2) gives both readouts AUROC 1,
    a difference of 0; one of each gives 1/4. So the difference is 1/4, its 95% CI [0, 1/4], the share of resamples
    above 0 about 1/2, and the sd 1/4·√(s(1 − s)) for that share s. Spearman over the four items: 1 − 6·2/(4·15) = 0.8.
    OOD clusters are pairs, IID ones tasks (with ``used_verifier`` as the label); the IID probe's val rows have no test
    logit and don't count."""
    mod = _load_script()
    logits = {"followup": (2.0, 1.0, 4.0, 3.0), "followup_prev": (3.0, 1.0, 4.0, 2.0)}  # pos 1, neg 1, pos 2, neg 2
    data = tmp_path / "datasets"
    ood = [{"id": f"{pair}/{kind}", "pair": pair, "label": int(kind == "deceptive"), "split": "test"}
           for pair in (1, 2) for kind in ("deceptive", "honest")]
    iid = [{"id": f"t{task}/{j}", "task_id": f"t{task}", "label": 0, "split": "test",
            "diagnostics": {"used_verifier": j == 0}} for task in (1, 2) for j in (0, 1)]
    iid.append({"id": "t3/0", "task_id": "t3", "label": 0, "split": "val", "diagnostics": {"used_verifier": True}})
    _write_items(data / "diverse_deception", ood)
    _write_items(data / "terminal_prerl", iid)
    folders = {"ood": tmp_path / "probes" / "diverse_deception" / "without_reasoning",
               "iid": tmp_path / "probes" / "terminal_prerl" / "with_reasoning"}
    for readout, z in logits.items():
        auc = 0.75 if readout == "followup" else 1.0
        _scored_probe(folders["ood"] / readout, "diverse_deception", "without_reasoning", readout,
                      [{"id": it["id"], "split": "test", "label": it["label"], "test_logit": s}
                       for it, s in zip(ood, z)], auc)
        _scored_probe(folders["iid"] / readout, "terminal_prerl", "with_reasoning", readout,
                      [{"id": it["id"], "split": it["split"], "used_verifier": int(it["diagnostics"]["used_verifier"]),
                        "task_id": it["task_id"], "test_logit": s} for it, s in zip(iid, (*z, None))], auc)
    out = tmp_path / "analyses"
    assert mod.main(["readout-gap", "--probes", f"{folders['ood']},{folders['iid']}", "--datasets", str(data),
                     "--out", str(out)]) == 0
    res = json.loads((out / "readout_gap.json").read_text())
    assert res["readouts"] == ["followup", "followup_prev"] and res["difference"] == "followup_prev - followup"
    assert set(res["code"]) == {"commit", "dirty", "diff_sha256", "source"}
    for kind, folder in folders.items():
        g = res["probes"][str(folder)]
        assert (g["cluster"], g["n"], g["n_positive"], g["n_clusters"]) == \
            ("pair" if kind == "ood" else "task_id", 4, 2, 2)
        assert g["test_auroc"] == {"followup": 0.75, "followup_prev": 1.0} and g["difference"] == 0.25
        assert g["layers"] == {"followup": 3, "followup_prev": 3}
        b = g["bootstrap"]
        assert (b["n"], b["seed"], b["n_nan"]) == (2000, 0, 0) and b["ci"] == [0.0, 0.25]
        s = b["share_above_zero"]
        assert 0.45 < s < 0.55 and b["sd"] == pytest.approx(0.25 * np.sqrt(s * (1 - s)), rel=1e-12)
        assert g["spearman"] == pytest.approx(0.8)
    # the bootstrap is the same for both: the resampled clusters depend on the seed only
    assert res["probes"][str(folders["ood"])]["bootstrap"] == res["probes"][str(folders["iid"])]["bootstrap"]
    # --readouts B,A flips the sign; --n-boot and --seed are passed on
    assert mod.main(["readout-gap", "--probes", str(folders["ood"]), "--readouts", "followup_prev,followup",
                     "--n-boot", "300", "--seed", "1", "--datasets", str(data),
                     "--out", str(tmp_path / "flipped")]) == 0
    flipped = json.loads((tmp_path / "flipped" / "readout_gap.json").read_text())["probes"][str(folders["ood"])]
    assert flipped["difference"] == -0.25 and flipped["bootstrap"]["ci"] == [-0.25, 0.0]
    assert (flipped["bootstrap"]["n"], flipped["bootstrap"]["seed"]) == (300, 1)

    # refused: an AUROC that is not the one in metrics.json; probes that score different items; one readout
    (folders["ood"] / "followup" / "metrics.json").write_text(json.dumps({"test_auroc": 0.7}))
    with pytest.raises(SystemExit):
        mod.main(["readout-gap", "--probes", str(folders["ood"]), "--datasets", str(data),
                  "--out", str(tmp_path / "wrong_auroc")])
    (folders["ood"] / "followup" / "metrics.json").write_text(json.dumps({"test_auroc": 0.75}))
    with open(folders["ood"] / "followup_prev" / "scores.jsonl") as fh:
        rows = [json.loads(line) for line in fh][:3]
    with open(folders["ood"] / "followup_prev" / "scores.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    for argv in (["--probes", str(folders["ood"])], ["--probes", str(folders["iid"]), "--readouts", "followup"]):
        with pytest.raises(SystemExit):
            mod.main(["readout-gap", *argv, "--datasets", str(data), "--out", str(tmp_path / "refused")])
    assert not (tmp_path / "refused").exists()
