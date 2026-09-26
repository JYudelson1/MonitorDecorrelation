"""Judge reasoning must reach the judges the eval scripts build for themselves (outside build_monitors).

``experiments/run_experiment.py`` is covered in test_experiment_config.py; these are the scripts that
construct ``AgentCoTMonitor`` / ``CoTMonitor`` from a config's monitor specs by hand, where a field can
be silently dropped on the way (no network, no model loads).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from monitordecorrelation.types import Prompt, Rollout

_REPO = Path(__file__).resolve().parents[1]
G25, G35 = "google/gemini-2.5-flash-lite", "google/gemini-3.5-flash-lite"


@pytest.fixture(autouse=True)
def _dummy_openrouter_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-used")


@pytest.fixture(autouse=True)
def _no_vllm_server_check(monkeypatch):
    import monitordecorrelation.monitors.vllm as vllm_mod

    monkeypatch.setattr(vllm_mod, "check_server", lambda *a, **k: None)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_{name}", _REPO / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Stop(Exception):
    pass


@pytest.mark.parametrize("cfg_name", sorted(p.name for p in (_REPO / "experiments" / "configs").glob("terminal_verifier_*.json")))
def test_eval_terminal_monitors_baseline_sends_the_configured_reasoning(monkeypatch, tmp_path, cfg_name):
    """Config file + ``--set`` → the AgentCoTMonitors this script scores with (and its run_info), for
    every terminal config: none sets ``reasoning``, so each judge runs on its model's default."""
    mod = _load_script("eval_terminal_monitors_baseline")
    seen = {}

    class _Scorer:
        def __init__(self, judges, *a, **k):
            seen["judges"] = judges

    def _stop(*a, **k):
        raise _Stop

    class _Sampling:
        def create_sampling_client(self, **k):
            return object()

    monkeypatch.setattr(mod.tinker, "ServiceClient", _Sampling)
    monkeypatch.setattr(mod, "make_renderer", lambda *a, **k: None)
    monkeypatch.setattr(mod, "MonitorScorer", _Scorer)
    monkeypatch.setattr(mod, "run_episodes", _stop)
    cfg = _REPO / "experiments" / "configs" / cfg_name
    import json

    cfg_monitors = json.loads(cfg.read_text())["monitors"]

    def sent(*sets):
        monkeypatch.setattr(sys, "argv", ["eval_terminal_monitors_baseline.py", "--config", str(cfg),
                                          "--out", str(tmp_path / "b.json"), "--set", *sets])
        with pytest.raises(_Stop):
            mod.main()
        # vLLM judges (the control config's q3_* / q35_*) take no `reasoning` — their thinking
        # settings must reach the request as configured instead
        for j in seen["judges"]:
            if j.backend.provider == "vllm":
                spec = next(m for m in cfg_monitors if m["name"] == j.name)
                body = j._request_body("p")
                assert "reasoning" not in body and body["max_tokens"] == spec["max_tokens"]
                assert body["chat_template_kwargs"] == {"enable_thinking": spec["enable_thinking"]}
                assert body.get("thinking_token_budget") == spec.get("thinking_budget")
        return {j.name: j._request_body("p")["reasoning"] for j in seen["judges"]
                if j.backend.provider == "openrouter"}

    assert sent() == {"g25_cot": {"max_tokens": 2048}, "g25_out": {"max_tokens": 2048},
                      "g35_cot": {"effort": "low"}, "g35_out": {"effort": "low"}}
    assert sent('monitors.model:gemini-2.5.reasoning={"enabled":false}',
                'monitors.g35_out.reasoning={"effort":"medium"}') == \
        {"g25_cot": {"enabled": False}, "g25_out": {"enabled": False},
         "g35_cot": {"effort": "low"}, "g35_out": {"effort": "medium"}}


def test_eval_monitors_on_rollouts_config_path_keeps_each_judges_reasoning(monkeypatch, capsys):
    mod = _load_script("eval_monitors_on_rollouts")
    r = Rollout(prompt=Prompt(text="q"), cot="c", output="a")
    monkeypatch.setattr(mod, "load_saved_rollouts", lambda path: [(r, True), (r, False)])
    cfg = _REPO / "experiments" / "configs" / "mbpp_matrix_sep18" / "row_control.json"
    monkeypatch.setattr(sys, "argv", ["eval_monitors_on_rollouts.py", "--rollouts", "x.jsonl",
                                      "--config", str(cfg), "--dry-run"])
    mod.main()
    out = capsys.readouterr().out
    for name, reasoning in [("cot_only_weak", "{'max_tokens': 2048}"), ("out_strong", "{'effort': 'low'}")]:
        line = next(ln for ln in out.splitlines() if ln.strip().startswith(f"- {name} "))
        assert f"reasoning={reasoning}" in line


def test_eval_monitors_on_rollouts_refuses_an_unsupported_judge_up_front(monkeypatch):
    """No longer swallowed into a per-model FAILED row: the script stops before any call."""
    mod = _load_script("eval_monitors_on_rollouts")
    r = Rollout(prompt=Prompt(text="q"), cot="c", output="a")
    monkeypatch.setattr(mod, "load_saved_rollouts", lambda path: [(r, True), (r, False)])
    monkeypatch.setattr(sys, "argv", ["eval_monitors_on_rollouts.py", "--rollouts", "x.jsonl",
                                      "--models", f"{G25},anthropic/claude-3-haiku"])
    with pytest.raises(ValueError, match="Implement support for this model"):
        mod.main()


def test_rescore_eval_rollouts_builds_judges_with_the_configs_reasoning():
    mod = _load_script("rescore_eval_rollouts")
    judges = mod.build_judges(str(_REPO / "experiments" / "configs" / "mbpp_inkling_sep19" / "control.json"))
    by_model = {(j.model_id, j._request_body("p")["reasoning"]["max_tokens" if j.model_id == G25 else "effort"])
                for j in judges}
    assert by_model == {(G25, 2048), (G35, "low")}
