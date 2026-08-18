"""The environment the NeMo-RL child process is started with.

Two things here are easy to get wrong and expensive to notice later:
(1) ``.env`` must reach NeMo-RL (that is where ``WANDB_API_KEY``, and later the monitor API keys,
    live) without overriding an explicit shell export; and
(2) this project's virtualenv must NOT reach NeMo-RL — Ray copies the driver's environment into
    every worker, and NeMo-RL's workers shell out to ``python3``/``uv``, so a leaked ``VIRTUAL_ENV``
    or ``PATH`` entry silently points them at the wrong interpreter.
"""

from __future__ import annotations

import os
from pathlib import Path

from monitordecorrelation.backends.nemo import launcher


def test_dotenv_is_read_and_blank_values_dropped(tmp_path: Path):
    (tmp_path / ".env").write_text(
        "WANDB_API_KEY=abc123\n"
        "OPENROUTER_API_KEY = sk-or-xyz\n"
        "# a comment\n"
        "EMPTY=\n"
    )
    values = launcher.dotenv_values(tmp_path / ".env")
    assert values["WANDB_API_KEY"] == "abc123"
    assert values["OPENROUTER_API_KEY"] == "sk-or-xyz"
    assert values.get("EMPTY") in (None, "")


def test_dotenv_values_on_a_missing_file_is_empty(tmp_path: Path):
    assert launcher.dotenv_values(tmp_path / ".env") == {}


def test_scrub_venv_repoints_every_interpreter_signal(tmp_path: Path):
    target = tmp_path / "nemo" / ".venv"
    env = {
        "PATH": f"/proj/.venv/bin{os.pathsep}/usr/local/bin{os.pathsep}/usr/bin",
        "VIRTUAL_ENV": "/proj/.venv",
        "UV_RUN_RECURSION_DEPTH": "1",
        "UV_PROJECT_ENVIRONMENT": "/proj/.venv",
        "PYTHONHOME": "/proj/.venv",
    }
    out = launcher.scrub_venv(dict(env), target)
    assert out["VIRTUAL_ENV"] == str(target)
    assert out["UV_PROJECT_ENVIRONMENT"] == str(target)
    assert "PYTHONHOME" not in out and "UV_RUN_RECURSION_DEPTH" not in out
    path = out["PATH"].split(os.pathsep)
    assert path[0] == str(target / "bin")       # NeMo-RL's interpreter wins a bare `python3`
    assert "/proj/.venv/bin" not in path        # ...and ours is gone entirely
    assert "/usr/bin" in path                   # system entries survive


def test_scrub_venv_also_drops_the_running_interpreters_bin(tmp_path: Path):
    """``uv run`` may have put us in a venv without exporting VIRTUAL_ENV; sys.prefix still says so."""
    import sys

    own_bin = str(Path(sys.prefix) / "bin")
    out = launcher.scrub_venv({"PATH": f"{own_bin}{os.pathsep}/usr/bin"}, tmp_path / ".venv")
    assert own_bin not in out["PATH"].split(os.pathsep)
