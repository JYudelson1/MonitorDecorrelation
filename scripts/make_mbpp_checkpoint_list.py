# Build data/misc/mbpp_q3_8b_checkpoints/mbpp_q3_8b_checkpoints.py from run dirs:
#   uv run python scripts/make_mbpp_checkpoint_list.py data/runs data/misc/mbpp_q3_8b_checkpoints/mbpp_q3_8b_checkpoints.py
# (final_checkpoints.tsv and checkpoints_by_run/ were derived from that file.)
import glob, json, os, re, sys

runs_dir, out = sys.argv[1], sys.argv[2]
CONDS = ["control", "cot_q17_weak", "out_q17_weak", "cot_q35_strong", "out_q35_strong",
         "probe_iid", "probe_ood", "probe_iid_cot", "probe_ood_cot"]
SEEDS = [0, 1, 2, 3]


def run_dir(cond, seed):
    name = f"mbpp_q3_8b_qmonitors_{cond}_s{seed}" + ("" if cond == "control" else "_pen05")
    return os.path.join(runs_dir, name)


sampler, state, final_state, final_sampler = {}, {}, {}, {}
for cond in CONDS:
    for d_ in (sampler, state, final_state, final_sampler):
        d_[cond] = {}
    for seed in SEEDS:
        d = run_dir(cond, seed)
        ev = {}
        for l in open(f"{d}/eval_checkpoints.jsonl"):
            if l.strip():
                x = json.loads(l)
                ev[x["step"]] = x["path"]  # a resumed run may re-log a step: keep the last
        steps = sorted(ev)
        assert steps == list(range(0, 181, 3)), (d, steps[:3], steps[-3:], len(steps))
        sampler[cond][seed] = [ev[s] for s in steps]
        st = {int(re.search(r"checkpoint_(\d+)\.txt$", p).group(1)): open(p).read().strip()
              for p in glob.glob(f"{d}/checkpoint_*.txt")}
        for s in range(0, 172, 9):  # seed 0 was run to 90 then resumed to 180: its step-90 state is leg 1's "-final"
            if s not in st and os.path.exists(f"{d}/resume_states/step_{s:06d}.json"):
                st[s] = json.load(open(f"{d}/resume_states/step_{s:06d}.json"))["state_checkpoint"]
        assert sorted(st) == list(range(0, 172, 9)), (d, sorted(st))
        state[cond][seed] = [st[s] for s in sorted(st)]
        final_state[cond][seed] = open(f"{d}/final_checkpoint.txt").read().strip()
        final_sampler[cond][seed] = ev[180]
        assert final_state[cond][seed].endswith(f"{os.path.basename(d)}-final")


def dump(name, obj, doc):
    lines = [f"# {doc}", f"{name} = {{"]
    for cond, by_seed in obj.items():
        lines.append(f"    {cond!r}: {{")
        for seed, v in by_seed.items():
            if isinstance(v, list):
                lines.append(f"        {seed}: [  # {os.path.basename(run_dir(cond, seed))}")
                lines += [f"            {p!r}," for p in v]
                lines.append("        ],")
            else:
                lines.append(f"        {seed}: {v!r},")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines)


header = '''"""Tinker checkpoints of the MBPP-Honeypot Qwen3-8B monitor-decorrelation matrix (penalty 0.5, 180 steps).

Every dict is indexed  DICT[condition][seed]  (seed in 0..3). Conditions:
  control          no monitor in the reward (the run is named control_s<seed>; all others are <cond>_s<seed>_pen05)
  cot_q17_weak     trained against Qwen3-1.7B judge reading CoT + answer
  out_q17_weak     trained against Qwen3-1.7B judge reading the answer only
  cot_q35_strong   trained against Qwen3.5-35B-A3B judge reading CoT + answer
  out_q35_strong   trained against Qwen3.5-35B-A3B judge reading the answer only
  probe_iid        trained against the MBPP-trained linear probe (answer only)
  probe_ood        trained against the diverse-deception-trained linear probe (answer only)
  probe_iid_cot    probe_iid weights, reading CoT + answer
  probe_ood_cot    probe_ood weights, reading CoT + answer

SAMPLER_CHECKPOINTS   sampler weights at every eval, steps 0, 3, ..., 180 (61 per run; list index i = step 3*i).
                      Usable directly with tinker sampling clients. These do not expire.
STATE_CHECKPOINTS     full training state (weights + optimizer) every 9 steps, 0..171 (20 per run; index i = step 9*i).
                      For resuming / branching training. 4-week TTL: these EXPIRE ~Dec 4-12 2026 unless extended
                      (exception: seed 0's step-90 entry is the "-final" of its first 90-step leg, which does not expire;
                      seed-0 runs were trained to 90 steps, then resumed to 180, so their paths span two tinker run ids).
FINAL_SAMPLER_CHECKPOINTS   step-180 sampler weights (= SAMPLER_CHECKPOINTS[c][s][-1]). Do not expire.
FINAL_STATE_CHECKPOINTS     step-180 training state ("-final"). Do not expire.

The step is also in every path (…-eval-<step> / …-<step>).
"""

CONDITIONS = %r
SEEDS = %r
''' % (CONDS, SEEDS)

body = "\n\n".join([
    dump("FINAL_SAMPLER_CHECKPOINTS", final_sampler, "step-180 sampler weights, one str per run"),
    dump("FINAL_STATE_CHECKPOINTS", final_state, "step-180 training state, one str per run"),
    dump("SAMPLER_CHECKPOINTS", sampler, "sampler weights every 3 steps (index i = step 3*i)"),
    dump("STATE_CHECKPOINTS", state, "training state every 9 steps (index i = step 9*i); EXPIRES ~Dec 2026"),
])
open(out, "w").write(header + "\n" + body + "\n")
ns = {}
exec(open(out).read(), ns)
assert all(len(ns["SAMPLER_CHECKPOINTS"][c][s]) == 61 and len(ns["STATE_CHECKPOINTS"][c][s]) == 20
           for c in CONDS for s in SEEDS)
print(f"wrote {out}: {len(CONDS)} conditions x {len(SEEDS)} seeds")
