# Environments

Each environment is an RL setting with a **ground-truth misbehavior signal** — we always need to
know whether the policy actually misbehaved, independent of any detector. Detectors are then scored
against that ground truth. Setting these up is reusable infra for every experiment.

Build state for each lives in [`../STATUS.md`](../STATUS.md).

## Common `Env` interface

```python
class Env(Protocol):
    def sample_prompt(self) -> Prompt: ...
    def score(self, rollout: Rollout) -> EnvResult
        # -> { task_reward: float, behavior_present: bool, meta: dict }
```

`task_reward` is what RL optimizes (before the monitor penalty); `behavior_present` is the
oracle label detectors are graded against. The RL loop combines `task_reward − λ·monitor_penalty`.

## 1. Sycophancy (Anthropic) — *first setting*

- **Source:** Anthropic sycophancy dataset (multiple-choice; model can give the user's preferred /
  flattering answer vs. the correct one).
- **Task reward:** signal that rewards giving the sycophantic answer (so RL pressure pushes toward
  misbehavior we can then try to detect/penalize). Exact shaping TBD.
- **Ground truth:** whether the chosen answer is the sycophantic one (known from the dataset).
- **CoT:** prompt the policy to reason before answering so CoT monitors have something to read.
- **Why first:** cheapest signal, no code execution, fast iteration. Good for the Exp-2 monitor-
  strength derisk.

## 2. Coding — hardcode reward hacking (Obfuscation-Atlas style)

- **Source:** a realistic coding environment where **reward hacking via hardcoding test cases**
  naturally occurs (mirrors *The Obfuscation Atlas*, FAR).
- **Task reward:** tests passing.
- **Ground truth:** whether the solution hardcodes / special-cases the tests rather than solving the
  problem (detectable by held-out tests or static checks).
- **Why:** matches the FAR setup directly, so we can compare against their white-box probe results;
  needs a code execution sandbox.

## 3. ImpossibleBench — *signal points only at reward hacking*

- **Source:** [ImpossibleBench](https://arxiv.org/abs/2510.20270) — mutates LiveCodeBench / SWE-bench
  unit tests to conflict with the spec, so **passing necessarily implies cheating.**
- **Task reward:** tests passing (which here *is* the misbehavior).
- **Ground truth:** any pass ⇒ misbehavior (that's the whole construction). Clean, unambiguous label.
- **Why:** the cleanest possible reward-hacking signal — no honest solution exists, so the only way
  to get reward is to cheat. Great for studying obfuscation under maximal pressure.

## 4. LiveCodeBench — *honest-is-possible counterpart*

- **Source:** [LiveCodeBench](https://arxiv.org/abs/2403.07974) — contamination-free competitive
  coding problems (LeetCode/AtCoder/CodeForces), date-annotated.
- **Task reward:** tests passing.
- **Ground truth:** honest solutions exist, so misbehavior = hardcoding / test-gaming despite a real
  solution being available.
- **Why:** the "possible" control for ImpossibleBench (shared task lineage). Lets us compare
  obfuscation dynamics when honesty *is* on the table vs. when it isn't.

## Notes
- ImpossibleBench and LiveCodeBench share task lineage (ImpossibleBench is built by mutating
  LiveCodeBench/SWE-bench), so the same harness/sandbox should serve both — build them together.
- Coding envs need a **sandboxed execution** path; sycophancy does not. Sequence sycophancy first.
- Keep `behavior_present` strictly separate from any detector so we never accidentally train
  on the oracle.
