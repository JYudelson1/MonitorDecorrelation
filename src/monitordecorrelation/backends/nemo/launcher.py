"""Start a NeMo-RL run from this project's venv.

``experiments/run_experiment.py --set backend=nemo`` ends up here. The job of this module is
everything that has to happen *outside* NeMo-RL's virtualenv:

* derive the ``MD_*`` knobs that parametrize ``experiments/configs/nemo/grpo.yaml`` (``params.py``);
* resolve the learning rate — ``hyperparams.get_lr`` is tinker-cookbook's LoRA-LR heuristic and is
  only installed here, so the number is computed on this side and passed down;
* propagate ``.env`` (WANDB_API_KEY today; monitor API keys as soon as a config lists monitors) into
  the child process, and decide the W&B mode the same way the tinker path does;
* exec ``backends/nemo/driver.py`` with NeMo-RL's own interpreter.

Two escape hatches, both environment variables: ``MD_NEMO_RL_ROOT`` picks a different NeMo-RL
checkout, ``MD_NEMO_CONFIG`` a different NeMo-RL config file.

The two virtualenvs are necessarily separate: NeMo-RL pins Python 3.13 + torch 2.11 + Megatron +
vLLM, this project pins Python >= 3.11 + the tinker SDK. They meet over ``PYTHONPATH`` (this repo's
``src/``, which NeMo-RL's venv can import because everything the driver touches needs only numpy /
pydantic / datasets / httpx / torch, all of which NeMo-RL already has).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from monitordecorrelation.backends.nemo.params import detect_n_gpus, nemo_env_vars

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_NEMO_RL_ROOT = REPO_ROOT / "third_party" / "nemo-rl"
DEFAULT_NEMO_CONFIG = REPO_ROOT / "experiments" / "configs" / "nemo" / "grpo.yaml"


def nemo_rl_root() -> Path:
    """The NeMo-RL checkout to run. Defaults to the vendored submodule (``git clone --recursive``
    or ``git submodule update --init --recursive``); ``MD_NEMO_RL_ROOT`` points elsewhere."""
    root = Path(os.environ.get("MD_NEMO_RL_ROOT", DEFAULT_NEMO_RL_ROOT))
    if not (root / "nemo_rl").is_dir():
        raise FileNotFoundError(
            f"NeMo-RL not found at {root}. It is a git submodule of this repo — run\n"
            f"    git submodule update --init --recursive\n"
            f"(or clone this repo with --recursive), or set MD_NEMO_RL_ROOT to another checkout."
        )
    return root


def nemo_python(root: Path) -> list[str]:
    """The command prefix that runs a Python module in NeMo-RL's environment.

    A venv at ``<nemo-rl>/.venv`` (what ``uv sync`` in that directory builds) is used directly. This
    project deliberately does NOT reuse ``UV_PROJECT_ENVIRONMENT`` if one is exported: on a NeMo-RL
    container that variable points at the image's own venv, and letting the two projects share it
    makes each ``uv run`` silently uninstall the other's dependencies.
    """
    venv_python = root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return [str(venv_python)]
    if shutil.which("uv") is None:
        raise FileNotFoundError(
            f"no NeMo-RL venv at {venv_python} and `uv` is not on PATH; build it with\n"
            f"    UV_PROJECT_ENVIRONMENT={root}/.venv uv sync --locked --directory {root}"
        )
    # Let uv build/refresh it in place. UV_PROJECT_ENVIRONMENT is pinned so this never touches
    # whatever environment the surrounding shell is using for this project.
    return ["uv", "run", "--directory", str(root), "--locked", "python"]


# Variables that describe *this* project's virtualenv. They must not reach NeMo-RL: Ray copies the
# driver's environment into every worker's runtime_env, and NeMo-RL's workers shell out (Megatron's
# dataset-helper `make`, uv, python3-config), so a stale VIRTUAL_ENV/PATH makes those subprocesses
# resolve `python3` to this project's interpreter inside NeMo-RL's own venv-per-worker.
_VENV_VARS = ("VIRTUAL_ENV", "PYTHONHOME", "UV", "UV_RUN_RECURSION_DEPTH", "PYTHONEXECUTABLE")


def scrub_venv(env: dict[str, str], target_venv: Path) -> dict[str, str]:
    """Repoint every "which virtualenv am I in" signal at NeMo-RL's venv.

    Dropping the variables is not enough: ``uv run`` also prepends its venv's ``bin`` to ``PATH``,
    and that entry is what a bare ``python3`` in a Makefile picks up. So the launching venv's bin
    directories are removed from ``PATH`` and NeMo-RL's is prepended.
    """
    stale_bins = {str(Path(env[v]) / "bin") for v in _VENV_VARS if env.get(v)}
    stale_bins.add(str(Path(sys.prefix) / "bin"))
    for var in _VENV_VARS:
        env.pop(var, None)
    path = [d for d in env.get("PATH", "").split(os.pathsep) if d and d not in stale_bins]
    env["PATH"] = os.pathsep.join([str(target_venv / "bin"), *path])
    env["VIRTUAL_ENV"] = str(target_venv)
    # `uv run --directory <nemo-rl>` inside a worker must build/refresh NeMo-RL's env, never ours.
    env["UV_PROJECT_ENVIRONMENT"] = str(target_venv)
    return env


def dotenv_values(path: Path) -> dict[str, str]:
    """``.env`` as a plain dict, without importing dotenv into the child. Values are passed to the
    NeMo-RL process so W&B and (later) monitor API keys work there too."""
    if not path.exists():
        return {}
    from dotenv import dotenv_values as _dotenv_values

    return {k: v for k, v in _dotenv_values(path).items() if v is not None}


def launch(
    cfg,
    *,
    lr: float,
    wandb_mode: str,
    wandb_group: str | None = None,
    wandb_tags: list[str] | None = None,
    run_dir: Path | None = None,
    extra_overrides: list[str] | None = None,
) -> int:
    """Run one experiment on NeMo-RL. Returns the child's exit code (0 = success)."""
    root = nemo_rl_root()
    run_dir = run_dir or (REPO_ROOT / "data" / "runs" / cfg.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    n_gpus = cfg.n_gpus or detect_n_gpus()

    env = dict(os.environ)
    for key, value in dotenv_values(REPO_ROOT / ".env").items():
        env.setdefault(key, value)  # an explicit shell export wins over .env, as load_dotenv() does
    env["WANDB_MODE"] = wandb_mode
    env.update(
        nemo_env_vars(
            cfg,
            lr=lr,
            n_gpus=n_gpus,
            run_dir=str(run_dir),
            wandb_enabled=wandb_mode != "disabled",
            wandb_group=wandb_group,
            wandb_tags=wandb_tags,
        )
    )
    # This repo, importable from NeMo-RL's interpreter.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    scrub_venv(env, root / ".venv")  # keep the two projects' environments apart

    experiment_config_path = run_dir / "config.json"  # the EFFECTIVE config, written by the runner
    cmd = [
        *nemo_python(root),
        "-m", "monitordecorrelation.backends.nemo.driver",
        "--experiment-config", str(experiment_config_path),
        "--nemo-config", os.environ.get("MD_NEMO_CONFIG", str(DEFAULT_NEMO_CONFIG)),
        "--run-dir", str(run_dir),
        *(extra_overrides or []),
    ]
    print(f"[nemo] {n_gpus} GPU(s) | {root}")
    print(f"[nemo] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, env=env, cwd=str(REPO_ROOT)).returncode

