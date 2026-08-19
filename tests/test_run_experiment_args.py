"""``--set key=value`` overrides must fail loudly.

``model_copy(update=...)`` bypasses pydantic validation, so an unknown field or an ill-typed value
used to be accepted silently and then ignored for the whole (hours-long, paid) run."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "run_experiment", Path(__file__).resolve().parents[1] / "experiments" / "run_experiment.py")
run_experiment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_experiment)

from monitordecorrelation.experiment_config import ExperimentConfig  # noqa: E402

apply = run_experiment._apply_overrides


def cfg() -> ExperimentConfig:
    return ExperimentConfig(run_name="t", env="impossiblebench", n_steps=40)


def test_valid_overrides_are_applied_and_typed():
    c = apply(cfg(), ["n_steps=2", "batch_size=2", "seed=3", "lr=2e-4", "thinking_effort=0.1",
                      'env_options={"reward_mode": "fraction"}'])
    assert (c.n_steps, c.batch_size, c.seed) == (2, 2, 3)
    assert c.lr == 2e-4 and c.thinking_effort == 0.1
    assert c.env_options == {"reward_mode": "fraction"}


def test_str_fields_are_not_number_coerced():
    assert apply(cfg(), ["run_name=123"]).run_name == "123"


def test_unknown_field_exits_with_a_did_you_mean():
    with pytest.raises(SystemExit) as e:
        apply(cfg(), ["field_that_doesnt_exist=1"])
    assert "unknown config field" in str(e.value)
    with pytest.raises(SystemExit, match="n_steps"):
        apply(cfg(), ["n_step=2"])


def test_ill_typed_value_is_rejected():
    with pytest.raises(SystemExit, match="valid integer"):
        apply(cfg(), ["n_steps=abc"])


def test_out_of_range_value_is_rejected_by_the_field_validator():
    with pytest.raises(SystemExit, match="less than 1"):
        apply(cfg(), ["thinking_effort=1.5"])
    with pytest.raises(SystemExit, match="penalty_schedule"):
        apply(cfg(), ['penalty_schedule={"start": 0.0}'])


def test_missing_equals_sign_is_rejected():
    with pytest.raises(SystemExit, match="expects key=value"):
        apply(cfg(), ["n_steps"])


def test_monitors_survive_the_revalidation_round_trip():
    c = ExperimentConfig(run_name="t", monitors=[
        {"kind": "cot", "name": "m", "role": "held_out", "model_id": "x/y"}])
    assert [m.name for m in apply(c, ["n_steps=2"]).monitors] == ["m"]
