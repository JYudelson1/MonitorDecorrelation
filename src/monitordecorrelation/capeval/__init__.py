"""Capability evals for tinker checkpoints vs the base model they were trained from.

Entry point: ``experiments/eval_checkpoints.py``. Seven benchmarks in three families:

- knowledge & reasoning (single-turn, ``qa.py``): MMLU-Pro (stratified subset), GPQA-Diamond, MATH-500;
- agentic coding (multi-turn bash agent in a docker sandbox, ``agentic.py``): LiveCodeBench (v6 slice,
  easy+medium), Exercism-Python (hidden tests), CanItEdit (instructed code edits, hidden tests);
- free-form coherence (``coherence.py``): MT-Bench's 80 two-turn prompts, graded by an LLM judge for
  coherence, self-consistency, and how "RL-fried" the model looks.

Everything model-family specific (Qwen3.5's HF chat template vs Inkling's TML rendering, thinking-budget
forcing, context limits) is in ``policy.py``; the sandbox in ``sandbox.py``; statistics (cluster
bootstrap, paired deltas) in ``stats.py``.
"""
