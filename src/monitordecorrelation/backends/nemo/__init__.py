"""NeMo-RL backend (``backend: "nemo"``).

Unlike ``tinker``/``transformers`` this backend does not implement the ``RLBackend`` sample/
train_step protocol: NeMo-RL owns its own GRPO loop, its Megatron policy workers and its vLLM
generation, and it lives in a separate virtualenv (Python 3.13 + torch 2.11 + Megatron + vLLM) that
cannot be merged with this project's. So the seam is one level up: we render NeMo-RL's config from
the experiment config, hand NeMo-RL our environment (as a Ray actor that computes
``task_reward - penalty_coef * monitor_score``) and our prompts (as its dataset), and let it train.

Module map:
  ``params``     — pure derivation of the ``MD_*`` knobs from an ``ExperimentConfig`` (no heavy deps)
  ``launcher``   — runs in THIS project's venv: resolves .env, then subprocesses the driver
  ``driver``     — runs in NeMo-RL's venv: builds datasets + the env actor, calls setup()/grpo_train()
  ``env_actor``  — the NeMo-RL ``EnvironmentInterface`` wrapping our ``Env`` + ``Monitor``s
  ``dataset``    — our ``Prompt``s as NeMo-RL ``DatumSpec``s

Only ``params`` and ``launcher`` are importable from this project's venv; ``driver``/``env_actor``/
``dataset`` import ``nemo_rl``/``ray``/``torch`` and are imported only inside NeMo-RL's venv.
"""

from monitordecorrelation.backends.nemo.params import nemo_env_vars

__all__ = ["nemo_env_vars"]
