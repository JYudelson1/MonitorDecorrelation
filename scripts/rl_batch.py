#!/usr/bin/env python3
"""The RL batch driver: a roster of TerminalEnv runs (the v2 probe rows) on three machines, from this machine.

    python3 scripts/rl_batch.py plan  --roster experiments/rl_batches/probe_rows_v2.json      # checks and costs only
    python3 scripts/rl_batch.py start --roster ROSTER --out DIR [--rehearsal]   # machines, servers, tunnels, runs, end
    python3 scripts/rl_batch.py start ... --probe-provider nebius --judges-provider nebius --runs-from OLD/runs
    python3 scripts/rl_batch.py status|sync|extend|down|watchdog --out DIR
    python3 scripts/rl_batch.py restart-server probe|judges --out DIR [--new-instance]
    python3 scripts/rl_batch.py restart-run|stop-run RUN --out DIR
    python3 scripts/rl_batch.py resume --out DIR       # watch a batch again after this driver died
    python3 scripts/rl_batch.py tunnel-test --out DIR  # the plumbing test: a dummy server through the tunnel, ≤ $1

**The roster** (``experiments/rl_batches/*.json``): a batch name, the W&B entity and project, ``n_steps`` for every run
(90, or 360 to see the trained-against monitor fall), ``sets`` for every run, and the runs: each a ``run_name``, a
``config``, a ``seed`` (0 to 10,000: the range the v2 probe's exclusion covers) and ``stop_after_zero_behavior_steps``
(null or a count, always stated: Vlad's judge rows ran without the early stop, his controls with 16), plus its own
``sets``. Nothing in a run's config is overridden but by what the roster states; the run gets ``--set seed run_name
wandb_project n_steps stop_after_zero_behavior_steps`` and the sets. The probe's score in the reward is the monitor's
business (monitors/inkling_probe_monitor.py): a roster row can change it with a set, nothing here assumes it.

**Machines** (scripts/nebius_job.py ``up``; names start with ``dennis-``, ``NEBIUS_JOB_NAME_PREFIX``):
- **rl:** a Nebius CPU VM on demand (cpu-d3 32vcpu-128gb, Ubuntu 24.04) in one of ``--rl-regions`` (default eu-west2,
  uk-south1, uk-south2, eu-west1: not us-north1, whose 3 public IPv4 the GPU VMs need, nor the regions without CPU
  quota), set up for root jobs (``--root-job``: the terminal env's read-only verifier runs the policy's commands as
  ``nobody``). It runs every run, one process each.
- **probe:** the full-precision reader and ``experiments/inkling_probe_server.py`` serving every inkling_probe monitor
  of the roster on 127.0.0.1:8178, one episode per forward. ``--probe-provider vast`` (the default): a vast.ai 4×
  B200, one copy. ``nebius``: a Nebius spot VM, 8× B200 by default (B200 alone reproduces the probes' golden logits;
  its platform is gpu-b200-sxm-a in me-west1, gpu-b200-sxm in us-central1; bid $4.5/GPU-h), two copies of the reader
  (GPUs 0-3 at :8179, 4-7 at :8180) behind scripts/load_balancer.py at :8178 (``--probe-copies``, ``--probe-gpu-ids``).
  Its probe folders are pinned: their sha256 (probe.npz + meta.json, the server's own probe identity) is taken from
  the committed files, checked on the machine before the server starts, on the RL machine before the runs start, and
  in ``/v1/meta`` of each copy and the balancer; each copy runs its own golden check as it starts.
- **judges:** Vlad's four vLLM engines (scripts/setup_vllm.sh: Qwen3-30B-A3B-FP8 on its first GPU at :8001, three
  Qwen3.5-35B-A3B-FP8 at :8003-8005 behind the balancer at :8002), vLLM pinned, every server on 127.0.0.1 and restarted
  on its own when it dies (``VLLM_RESTART``). ``--judges-provider vast`` (the default): a vast.ai 4× H200. ``nebius``:
  a Nebius spot VM, 8× H200 by default (bid $2.5/GPU-h), the same layout on GPUs 0-3 (setup_vllm.sh's ``VLLM_GPUS``,
  ``--judges-gpu-ids``), 4-7 idle.
A Nebius server's platform (a comma list: each priced where it is offered), preset, bid (or ``--*-on-demand``) and
regions are its role's settings (``--probe-*``, ``--judges-*``); the launcher puts it in the cheapest region with
capacity, under the bid. No key ever goes to a vast.ai instance, nor to a server's VM. The runs' keys (tinker, W&B) go
to the RL machine only, over SSH into its RAM disk, read by the job as it starts and deleted at once.

**Tunnels:** the runs reach the servers at 127.0.0.1 (8178 the probe, 8001/8002 the judges, as the configs say)
through SSH tunnels from the RL machine (scripts/tunnel.sh: keepalives, reconnect forever). Their key is made for the
batch on the RL machine, in its RAM; only its public key reaches the servers' machines, appended to the login user's
authorized_keys (root on vast.ai, md on Nebius) with ``restrict,port-forwarding,permitopen=…``: it opens nothing but
the server's ports.

**The run:** on the RL machine, ``vm-supervise`` (this file, as root) starts every run as soon as the servers answer,
and restarts a run that fails with ``--resume`` from its last saved state (``save_every``), up to ``--max-attempts``.
A run counts as done once its final checkpoint and its final resume state are written. It refuses to start without
the W&B key and the roster's entity (W&B would go offline quietly), and stops a run whose W&B URL names another entity
or project. Control files from this driver restart or stop one run, or all.

**Resuming runs of an earlier batch** (``start --runs-from DIR``): before the supervisor starts, each roster run's folder
DIR/<run_name>/ goes to the RL machine's data/runs/<run_name>/, and the supervisor resumes it from its newest
resume_states/step_*.json (logged for each run; a finished one is done). A run without a save there is refused unless
``--allow-fresh`` (it starts fresh then), a save of another run (run_name, seed) always; the estimate, and so the
machines' hours, count only the steps left.

**The driver** (``start``, then its watch, ``resume`` after it died):
- **Health:** every few minutes, each server on its own machine (the probe's shas on every copy's port and the
  balancer's, each judge engine) and through the tunnels on the RL machine (60 s each: an eval burst slows them). A
  tunnel down two rounds in a row is restarted (its key authorized again); a server that is down for
  ``--server-dead-minutes`` is restarted on its machine (``nebius_job.py serve``), at most ``--max-server-restarts``
  times. A vast.ai instance that went away is replaced, at most once (``--no-replace`` turns that off). A Nebius
  server VM that is preempted, stopped or deleted (its state, by the Nebius CLI) is ended (its disk and public IP
  freed, the ledger settled) and a new VM made, after every preemption: after a backoff (``--replace-backoff-minutes``
  × 2^(k-1), at most 30 min), up to ``--max-replacements`` per role, for the hours left of the batch's cap; its ``up``
  runs on its own while the watch goes on (batch.json records it, so a later driver or the watchdog takes it over),
  then its tunnel moves there. The runs wait meanwhile: their clients retry. One stopped at its own ``--hours`` is the
  batch's end, not replaced. At the start, a Nebius server whose ``up`` failed gets a new VM the same way.
- **Sync:** the run folders come back into ``DIR/runs/`` every ``--sync-minutes`` (rsync), and at the end.
- **Leases:** a vast.ai instance starts for ``--lease-hours`` and is extended while the batch runs, so that it always
  has at least half a lease left. A Nebius VM (the RL machine, a Nebius server) starts with the batch's cap as its
  ``--hours`` (``--rl-hours``, else from the estimate: what the ledger reserves) and its shutdown is moved to a lease
  from now, never past that cap. Every 15 minutes each kept Nebius spot server's price is sampled into its job.json,
  so the ledger settles it from its prices, not its bid. If this driver dies, nothing extends them: they stop within a
  lease. ``watchdog`` (a one-shot: the hourly hub scheduled job, on the batch output/gpu_jobs/CURRENT_BATCH names,
  which ``start`` writes and the verified end removes; ``--current-batch``) does what the watch does once: extend,
  sync, health (a preempted Nebius server's new VM started, or taken up once made), or end everything if the batch is
  over, its RL machine is gone or its start died; it acts only if no driver holds the batch's lock.
- **The end:** once every run is done or failed (or the RL machine is gone): a last sync, then every machine ended and
  verified gone (``nebius_job.py down``, with the ledger): a new VM still being made too (its ``up`` stopped first),
  and a replaced one whose end was not verified then. Ctrl-C ends everything too; SIGTERM or SIGHUP (this machine
  going away) leave the machines to their leases and say how to ``resume`` or ``down``.

**Rehearsal** (``start --rehearsal``): the same batch, each run named ``rehearsal-…`` with ``n_steps`` 2, an eval and
a save at every step, logged to W&B in ``--rehearsal-entity`` (default the roster's) and ``--rehearsal-project``
(default <project>-rehearsal), plus the checklist, whose results go to ``DIR/rehearsal.json``: check_probe_server.py
on the serving GPUs (each copy's) before the server starts; a run killed while it runs, after a save, and resumed from
it; synthetic 8k/12k/16k-token episodes sent to the server (experiments/check_probe_long_episodes.py); the per-step
wall time and the probe queue's depth with every run at once; the server, one judge engine and a tunnel killed, with
the runs going on; on Nebius, a server's VM stopped as a preemption would (a new one must serve, the runs waiting); the
W&B entity of every run; ``nobody`` unable to read the keys on the RL machine; and, if the repo has
experiments/rescore_terminal_judges.py and ``--compare-runs`` names run folders, its compare mode against the judges.

Stdlib only, like scripts/nebius_job.py, whose machinery it uses (``Remote``, ``HubGpu``, the keys' transport).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "nebius_job.py"


def _load_launcher():
    """scripts/nebius_job.py as a module (stdlib only), for its SSH, hub-gpu and key machinery."""
    spec = importlib.util.spec_from_file_location("nebius_job", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("nebius_job", module)  # its dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return sys.modules["nebius_job"]


nj = _load_launcher()

# ------------------------------------------------------------------------------------------------- constants

PROBE_PORT = 8178  # the probe server (or the balancer before its copies), on its machine and on the RL machine
GPUS_PER_COPY = 4  # one copy of the BF16 reader: 4 B200s, 3-4 B300s (experiments/inkling_probe_server.py)
JUDGE_PORTS = (8001, 8002)  # q3, and the q35 balancer: as the configs' base_urls say
Q35_BACKENDS = (8003, 8004, 8005)
JUDGE_MODELS = {8001: "Qwen/Qwen3-30B-A3B-FP8", 8002: "Qwen/Qwen3.5-35B-A3B-FP8", 8003: "Qwen/Qwen3.5-35B-A3B-FP8",
                8004: "Qwen/Qwen3.5-35B-A3B-FP8", 8005: "Qwen/Qwen3.5-35B-A3B-FP8"}
JUDGE_URLS = {"http://localhost:8001/v1": 8001, "http://localhost:8002/v1": 8002}
# The judges' Hugging Face revisions: main on 2026-10-23, unchanged since before Vlad's batch (2026-09-30).
JUDGE_REVISIONS = {"Qwen/Qwen3-30B-A3B-FP8": "d206ba732169f29bb77fbf80fc2c4b81d4d30782",
                   "Qwen/Qwen3.5-35B-A3B-FP8": "9d1823d2dee688a6b25e77009dc727688c44936e",
                   "AngelSlim/Qwen3-a3B_eagle3": "266a50ea8c9dbedb729d7c18a98dd79f5b39b5c2"}
VLLM_VERSION = "0.30.0"  # Vlad's: "the latest version" during his batch (2026-09-30 to 10-02); 0.31.0 came out 10-05
READER = "thinkingmachines/Inkling-Small@8cc5877b44d343f88b92086aa1fb72897950f06a"  # the v2 probes' reader, BF16
READER_DIR = "/mnt/ram/Inkling-Small"
PROBE_URL = f"http://127.0.0.1:{PROBE_PORT}"
V2_MANIFEST = "data/probe_datasets/terminal_prerl_v2/manifest.json"  # the env the v2 probes' episodes were made in
SEED_MAX = 10_000  # docs/inkling_probes_amendment_1.md: the v2 exclusion covers the RL seeds 0 to 10,000
SECRETS = ("tinker_api_key", "wandb_api_key")  # to the RL machine only, as TINKER_API_KEY / WANDB_API_KEY
REMOTE_BATCH = "rl_batch"  # in the RL machine's copy of the repo: plan.json, state/
TUNNEL_KEY = "/mnt/ram/rl-batch/tunnel_key"  # made on the RL machine, in its RAM (Nebius: a tmpfs)
TUNNEL_DIR = "rl_batch_tunnels"  # in the RL machine's home: <name>.conf, .pid, .ssh.pid, .state, .log
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")
SET_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_.:*-]*")
ROSTER_KEYS = {"name", "description", "wandb_entity", "wandb_project", "n_steps", "sets", "runs"}
ROW_KEYS = {"run_name", "config", "seed", "stop_after_zero_behavior_steps", "sets", "note"}
OWN_SETS = {"seed", "run_name", "wandb_project", "n_steps", "stop_after_zero_behavior_steps"}  # set by the roster
# Measured on 2026-10-23 (docs/GPU_RUNBOOK.md, "v2 probe rows"): the v2 extraction on 4× B200, one episode per forward,
# ~0.35-0.48 s per with-reasoning episode (3.5-4.6k tokens/s); Vlad's eval episodes average 1.4k such tokens.
PROBE_SECONDS_PER_EPISODE = 0.36
CONTROL_STEP_MINUTES = 1.3  # Vlad's controls (W&B): a step of sampling, no train-against monitor, judges at evals
JUDGE_CALLS_PER_HOUR = 22_500  # what Vlad's q3 GPU, and his three q35 engines, each served with 12 rows at once


def utc(t: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if t is None else t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Driver:
    """Where the driver's log lines go: stdout and, once the batch has a folder, DIR/driver.log."""
    log_file: Path | None = None


def log(msg: str) -> None:
    line = f"[rl_batch {datetime.now(UTC):%H:%M:%S}] {msg}"
    with contextlib.suppress(OSError, RuntimeError):
        print(line, flush=True)
    if Driver.log_file is not None:
        with contextlib.suppress(OSError):
            with Driver.log_file.open("a") as fh:
                fh.write(line + "\n")


# --------------------------------------------------------------------------------------------------- roster


@dataclass
class Row:
    run_name: str
    config: str  # repo-relative
    seed: int
    stop_after_zero_behavior_steps: int | None
    sets: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class Roster:
    name: str
    wandb_entity: str
    wandb_project: str
    n_steps: int
    sets: list[str]
    rows: list[Row]
    path: str = ""
    description: str = ""


def _check_sets(sets, where: str) -> list[str]:
    if not isinstance(sets, list) or not all(isinstance(s, str) for s in sets):
        raise SystemExit(f"{where}: sets must be a list of 'key=value' strings")
    for s in sets:
        key, eq, _ = s.partition("=")
        if not eq or not SET_KEY.fullmatch(key):
            raise SystemExit(f"{where}: {s!r} is not key=value")
        if key in OWN_SETS:
            raise SystemExit(f"{where}: {key} is the roster's own field (state it there, not in sets)")
    return list(sets)


def load_roster(path: str | Path) -> Roster:
    """A roster file, every key checked: an unknown key, a missing one or a value out of range is a SystemExit
    (nothing the batch runs with may be guessed)."""
    path = Path(path)
    data = json.loads(path.read_text())
    where = str(path)
    if not isinstance(data, dict) or set(data) - ROSTER_KEYS or not {"name", "wandb_entity", "wandb_project",
                                                                       "n_steps", "runs"} <= set(data):
        raise SystemExit(f"{where}: a roster has the keys {sorted(ROSTER_KEYS)} (all but description and sets "
                         f"required), got {sorted(data) if isinstance(data, dict) else type(data).__name__}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}", str(data["name"])):
        raise SystemExit(f"{where}: name: lowercase letters, digits and -")
    n_steps = data["n_steps"]
    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or not 1 <= n_steps <= 10_000:
        raise SystemExit(f"{where}: n_steps must be a whole number of steps, got {n_steps!r}")
    for k in ("wandb_entity", "wandb_project"):
        if not isinstance(data[k], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", data[k]):
            raise SystemExit(f"{where}: {k} must be a W&B name, got {data[k]!r}")
    sets = _check_sets(data.get("sets", []), f"{where}: sets")
    rows, names = [], set()
    if not isinstance(data["runs"], list) or not data["runs"]:
        raise SystemExit(f"{where}: runs must be a non-empty list")
    for i, r in enumerate(data["runs"]):
        rw = f"{where}: runs[{i}]"
        if not isinstance(r, dict) or set(r) - ROW_KEYS or not {"run_name", "config", "seed",
                                                                 "stop_after_zero_behavior_steps"} <= set(r):
            raise SystemExit(f"{rw}: a run has run_name, config, seed and stop_after_zero_behavior_steps (stated, "
                             f"null for none), optionally sets and note; got {sorted(r) if isinstance(r, dict) else r!r}")
        if not RUN_NAME.fullmatch(str(r["run_name"])) or r["run_name"] in names:
            raise SystemExit(f"{rw}: run_name {r['run_name']!r} is not a valid, unique run name")
        names.add(r["run_name"])
        seed = r["seed"]
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= SEED_MAX:
            raise SystemExit(f"{rw}: seed must be 0 to {SEED_MAX:,} (the RL seeds the v2 probe's exclusion covers, "
                             f"docs/inkling_probes_amendment_1.md), got {seed!r}")
        stop = r["stop_after_zero_behavior_steps"]
        if stop is not None and (not isinstance(stop, int) or isinstance(stop, bool) or stop < 1):
            raise SystemExit(f"{rw}: stop_after_zero_behavior_steps must be null or a positive count, got {stop!r}")
        if not (REPO / str(r["config"])).is_file():
            raise SystemExit(f"{rw}: config {r['config']} not found in {REPO}")
        rows.append(Row(str(r["run_name"]), str(r["config"]), seed, stop, _check_sets(r.get("sets", []), f"{rw}.sets"),
                        str(r.get("note", ""))))
    return Roster(str(data["name"]), data["wandb_entity"], data["wandb_project"], n_steps, sets, rows,
                  str(path), str(data.get("description", "")))


def rehearsal_roster(roster: Roster, project: str | None, entity: str | None) -> Roster:
    """The rehearsal of ``roster``: every run renamed rehearsal-<name>, 2 steps with an eval and a save at each (so the
    kill-and-resume drill resumes from a save), in its own W&B project (``project``, default <project>-rehearsal) and
    entity (``entity``, default the roster's), everything else the batch's."""
    rows = [dataclasses.replace(r, run_name=f"rehearsal-{r.run_name}") for r in roster.rows]
    return dataclasses.replace(roster, name=f"{roster.name}-rehearsal"[:41], n_steps=2, sets=[*roster.sets,
                               "eval_every=1", "save_every=1"], rows=rows,
                               wandb_project=project or f"{roster.wandb_project}-rehearsal",
                               wandb_entity=entity or roster.wandb_entity)


# ----------------------------------------------------------------------------------------------------- plan


def sha256_files(*paths: Path) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest()


def probe_sha(probe_dir: Path) -> str:
    """whitebox/probe_server.probe_sha, in the stdlib: sha256 over probe.npz then meta.json."""
    return sha256_files(probe_dir / "probe.npz", probe_dir / "meta.json")


def git_clean(paths: list[str]) -> str | None:
    """Why ``paths`` (repo-relative) are not exactly as committed (None: they are): untracked, or changed."""
    for p in paths:
        cp = subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", p], capture_output=True,
                            stdin=subprocess.DEVNULL)
        if cp.returncode != 0:
            return f"{p} is not committed"
    cp = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", *paths], capture_output=True,
                        text=True, stdin=subprocess.DEVNULL)
    if cp.returncode != 0 or cp.stdout.strip():
        return f"uncommitted changes: {cp.stdout.strip()[:300] or cp.stderr.strip()[:300]}"
    return None


@dataclass
class RunPlan:
    row: Row
    argv: list[str]  # run_experiment.py's, from the repo on the RL machine
    probes: list[str]  # inkling_probe monitors' probe_path
    judges: list[int]  # the judge ports its monitors use


@dataclass
class Plan:
    roster: Roster
    runs: list[RunPlan]
    probes: dict  # probe_path -> pinned sha256 (sorted)
    judges: bool
    rehearsal: bool = False


def run_argv(roster: Roster, row: Row, extra: list[str] = ()) -> list[str]:
    """``run_experiment.py`` for ``row``: its config, then every override the roster states (seed, run name, project,
    n_steps, early stop), the roster's sets, the row's, and ``extra``. The same each time, so ``--resume`` gets the
    config the run started with."""
    stop = "null" if row.stop_after_zero_behavior_steps is None else str(row.stop_after_zero_behavior_steps)
    return [".venv/bin/python", "experiments/run_experiment.py", "--config", row.config, "--set",
            f"seed={row.seed}", f"run_name={row.run_name}", f"wandb_project={roster.wandb_project}",
            f"n_steps={roster.n_steps}", f"stop_after_zero_behavior_steps={stop}", *roster.sets, *row.sets, *extra]


def config_monitors(cfg: dict) -> tuple[list[dict], list[dict]]:
    """(inkling probes, vLLM judges) of a config's monitors."""
    mons = cfg.get("monitors") or []
    return ([m for m in mons if m.get("kind") == "inkling_probe"],
            [m for m in mons if m.get("kind", "cot") == "cot" and m.get("provider") == "vllm"])


def build_plan(roster: Roster, rehearsal: bool = False, manifest: str = V2_MANIFEST) -> Plan:
    """Every run's command, the probe folders to serve (pinned: committed, their shas), whether judges are needed;
    SystemExit on anything the machines can't serve: a probe not at 127.0.0.1:8178, a judge not at localhost:8001/8002,
    an env not the v2 probes' (``manifest``: the env and effort the v2 episodes were made with), a non-TML policy."""
    want_env = None
    if (REPO / manifest).is_file():
        src = (json.loads((REPO / manifest).read_text()).get("sources") or {}).get("run_info") or {}
        want_env = (src.get("env_options"), src.get("effort"))
    runs, probes, judges = [], {}, False
    for row in roster.rows:
        cfg = json.loads((REPO / row.config).read_text())
        where = f"{row.run_name} ({row.config})"
        if cfg.get("env") != "terminal_verifier" or not str(cfg.get("policy", "")).startswith("thinkingmachines/"):
            raise SystemExit(f"{where}: the batch runs TerminalEnv runs of a TML policy (Inkling)")
        inkling, vllm = config_monitors(cfg)
        for m in inkling:
            if m.get("server_url", "").rstrip("/") != PROBE_URL:
                raise SystemExit(f"{where}: monitor {m.get('name')} reads {m.get('server_url')}; the batch serves "
                                 f"probes at {PROBE_URL}")
            probes[m["probe_path"]] = None
        ports = []
        for m in vllm:
            port = JUDGE_URLS.get(str(m.get("base_url", "")).rstrip("/"))
            if port is None or JUDGE_MODELS[port] != m.get("model_id"):
                raise SystemExit(f"{where}: judge {m.get('name')} ({m.get('model_id')} at {m.get('base_url')}) is not "
                                 "one the batch serves (Qwen3-30B-A3B-FP8 at :8001, Qwen3.5-35B-A3B-FP8 at :8002)")
            ports.append(port)
        judges |= bool(ports)
        if inkling and want_env is not None and want_env[0]:
            keys = ("verifier_mode", "submission_ends", "max_turns", "require_explanation", "last_turn_submit_notice")
            env = cfg.get("env_options") or {}
            off = {k: (env.get(k), want_env[0].get(k)) for k in keys if env.get(k) != want_env[0].get(k)}
            if off or cfg.get("thinking_effort") != want_env[1]:
                raise SystemExit(f"{where}: its episodes would not be rendered as the v2 probes' were ({manifest}): "
                                 f"{off or ''} effort {cfg.get('thinking_effort')} vs {want_env[1]}")
        runs.append(RunPlan(row, run_argv(roster, row), [m["probe_path"] for m in inkling], sorted(set(ports))))
    for p in sorted(probes):
        files = [f"{p}/probe.npz", f"{p}/meta.json", f"{p}/golden.jsonl"]
        if not all((REPO / f).is_file() for f in files):
            raise SystemExit(f"probe folder {p} is not deployed here (probe.npz, meta.json, golden.jsonl)")
        if problem := git_clean(files):
            raise SystemExit(f"probe folder {p}: {problem}: the server and the runs must read the committed probe "
                             "(its sha256 is its identity; deploy_probes.py rewrites meta.json)")
        meta = json.loads((REPO / p / "meta.json").read_text())
        info = meta.get("reader_info") or {}
        if info.get("revision") != READER.split("@")[1]:
            raise SystemExit(f"probe folder {p} is for the reader at revision {info.get('revision')}, not the batch's "
                             f"{READER}")
        probes[p] = probe_sha(REPO / p)
    return Plan(roster, runs, dict(sorted(probes.items())), judges, rehearsal)


def estimate(plan: Plan, copies: int = 1, done: dict | None = None) -> dict:
    """How long and how loaded, from the roster and the measurements above: probe-server hours (one episode per forward,
    every valid train episode of a probe row and every valid eval episode, ~97%; ``copies`` of the reader share
    them), judge calls per hour against what Vlad's engines served, and the batch's hours: the probe server's, or the
    runs' own pace if that is slower. ``done``: the steps a run has done already (``--runs-from``), not counted."""
    cfg = {r.row.config: json.loads((REPO / r.row.config).read_text()) for r in plan.runs}
    n = plan.roster.n_steps
    probe_s, judge_calls, slowest = 0.0, {8001: 0, 8002: 0}, 0.0
    for r in plan.runs:
        c = cfg[r.row.config]
        start = int((done or {}).get(r.row.run_name, 0))
        if start >= n:  # finished already
            continue
        every = 1 if plan.rehearsal else int(c.get("eval_every", 3))
        evals = sum(1 for step in range(0, n, every) if step >= start) + 1  # every eval_every steps, and the final one
        train_eps = (n - start) * c.get("batch_size", 8) * c.get("group_size", 16) * 0.97
        eval_eps = evals * c.get("eval_size", 64) * c.get("eval_samples_per_prompt", 4) * 0.97
        inkling, vllm = config_monitors(c)
        for m in inkling:  # a train-against monitor scores every train episode too
            probe_s += (eval_eps + (train_eps if m.get("role") == "train_against" else 0)) * PROBE_SECONDS_PER_EPISODE
        for m in vllm:
            judge_calls[JUDGE_URLS[m["base_url"].rstrip("/")]] += eval_eps + (
                train_eps if m.get("role") == "train_against" else 0)
        slowest = max(slowest, (n - start) * CONTROL_STEP_MINUTES / 60)
    probe_h = probe_s / 3600 / max(1, copies)
    hours = max(probe_h, slowest)
    per_h = {p: (c / hours if hours else 0.0) for p, c in judge_calls.items()}
    return {"n_steps": n, "probe_server_hours": round(probe_h, 1), "batch_hours": round(hours, 1),
            "judge_calls": judge_calls, "judge_calls_per_hour": {p: round(v) for p, v in per_h.items()},
            "judge_load": {p: round(v / JUDGE_CALLS_PER_HOUR, 2) for p, v in per_h.items()}}


# ------------------------------------------------------------------------------------------ settings + state


@dataclass
class Settings:
    """The machines and the watch, from the command line (``add_settings``); saved in batch.json."""
    prefix: str = "dennis-"
    owner: str = "dennis"
    experiment: str = "probes-v2-rl"
    rl_preset: str = "32vcpu-128gb"
    rl_hours: float = 0.0  # 0: from the estimate (the batch's hours × 1.5 + 2, at least 6)
    probe_gpu: str = "B200"
    probe_gpus: int = 4
    probe_disk_gib: int = 650
    probe_max_price: float | None = 60.0  # USD/h for the instance: 4× B200 was $30-43 on 2026-10-23
    judge_gpus_names: str = "H200,H200 NVL"  # the first with an offer
    judge_gpus: int = 4
    judge_disk_gib: int = 250
    judge_max_price: float | None = 30.0  # 4× H200 NVL $16-20, 8× H200 $40 on 2026-10-23
    # Where the servers run: vast.ai (the instances above), or a Nebius VM each (spot by default, under its bid per
    # GPU-hour; the cheapest of its platforms and regions with capacity), probe_disk_gib / judge_disk_gib of disk. The
    # probe server on B200, the only GPU its golden logits match on (2026-10-28: B300 off by a mean |Δlogit| of
    # 0.25-0.35), whose platform is gpu-b200-sxm-a in me-west1 and gpu-b200-sxm in us-central1 (8× $30-34/h spot).
    probe_provider: str = "vast"
    judges_provider: str = "vast"
    probe_platform: str = "gpu-b200-sxm-a,gpu-b200-sxm"  # a comma list: each where it is offered
    probe_preset: str = "8gpu-160vcpu-1792gb"
    judges_platform: str = "gpu-h200-sxm"
    judges_preset: str = "8gpu-128vcpu-1600gb"
    probe_on_demand: bool = False
    judges_on_demand: bool = False
    probe_max_gpu_price: float = 4.5  # USD per GPU-hour, NEBIUS_MAX_GPU_PRICE's kind: Nebius never charges more
    judges_max_gpu_price: float = 2.5
    probe_regions: str | None = None  # a comma list (None: every region of the regions file)
    judges_regions: str | None = None
    # The RL machine's regions, cpu-d3 $1.06/h in each (2026-10-28): not us-north1, whose 3 public IPs the GPU VMs
    # need, nor us-central1 and eu-north1 (no non-GPU vCPU quota there).
    rl_regions: str | None = "eu-west2,uk-south1,uk-south2,eu-west1"
    probe_copies: int = 0  # 0: one on vast.ai; on Nebius one per 4 GPUs (2 on an 8-GPU VM), behind a balancer
    probe_gpu_ids: str | None = None  # the probe server's GPUs, split between its copies (None: all of its machine's)
    judges_gpu_ids: str | None = None  # setup_vllm.sh's 2 or 4 GPUs (None: a 2- or 4-GPU machine's, else GPUs 0-3)
    # Nebius judges: a folder whose flashinfer/ and vllm/ go to the VM's ~/.cache/ before its servers start. A first
    # start on B300 JIT-compiles FlashInfer's MoE kernels for sm_103a, ~15 min (2026-10-28); a cache from an earlier
    # VM of the same GPU and vLLM skips that, and one that doesn't match is only compiled again.
    judges_cache: str | None = None
    max_replacements: int = 8  # Nebius server VMs: new ones per role (a vast.ai instance is replaced once)
    replace_backoff_minutes: float = 1.0  # before the k-th new VM of a role: this × 2^(k-1), at most 30 min
    lease_hours: float = 4.0
    sync_minutes: float = 10.0
    health_minutes: float = 5.0
    server_dead_minutes: float = 10.0
    max_server_restarts: int = 3
    replace: bool = True
    max_attempts: int = 8
    vm_dead_minutes: float = 30.0
    stall_minutes: float = 45.0
    rsync: str = "rsync"
    budget_file: str | None = None
    hub_gpu: str = "hub-gpu"


SERVER_ROLES = ("probe", "judges")
JUDGES_CACHE_DIRS = ("flashinfer", "vllm")  # of --judges-cache: each to the judges VM's ~/.cache/<name>
PROVIDERS = ("vast", "nebius")


def provider_of(s: Settings, role: str) -> str:
    """Where ``role`` runs: the RL machine on Nebius, each server on vast.ai or Nebius (``--probe-provider``,
    ``--judges-provider``)."""
    if role == "rl":
        return "nebius"
    return s.probe_provider if role.startswith("probe") else s.judges_provider


def nebius_machine(s: Settings, role: str) -> dict:
    """A Nebius server VM: its platform, preset, pricing (spot with a bid per GPU-hour, or on demand), regions, disk."""
    if role.startswith("probe"):
        return {"platform": s.probe_platform, "preset": s.probe_preset, "on_demand": s.probe_on_demand,
                "bid": s.probe_max_gpu_price, "regions": s.probe_regions, "disk_gib": s.probe_disk_gib}
    return {"platform": s.judges_platform, "preset": s.judges_preset, "on_demand": s.judges_on_demand,
            "bid": s.judges_max_gpu_price, "regions": s.judges_regions, "disk_gib": s.judge_disk_gib}


def role_gpus(s: Settings, role: str) -> int:
    """The GPUs of ``role``'s machine: its vast.ai count, or its Nebius preset's."""
    if provider_of(s, role) == "nebius":
        return nj.gpu_count(nebius_machine(s, role)["preset"])
    return s.probe_gpus if role.startswith("probe") else s.judge_gpus


def probe_copies(s: Settings) -> int:
    """Reader copies on the probe server's machine: ``--probe-copies``, else one on vast.ai and one per
    ``GPUS_PER_COPY`` GPUs on Nebius (2 on an 8-GPU VM)."""
    if s.probe_copies:
        return s.probe_copies
    return max(1, role_gpus(s, "probe") // GPUS_PER_COPY) if provider_of(s, "probe") == "nebius" else 1


def gpu_ids(value: str | None) -> list[int] | None:
    """A comma list of GPU indices (None: none given)."""
    if not value:
        return None
    try:
        return [int(g) for g in value.split(",") if g.strip()]
    except ValueError:
        raise SystemExit(f"GPU indices, a comma list like 0,1,2,3: got {value!r}") from None


def probe_gpus_of(s: Settings) -> list[int] | None:
    """The probe server's GPUs, split between its copies: ``probe_gpu_ids``, else every GPU of its machine with several
    copies (None with one: the server takes what it sees)."""
    return gpu_ids(s.probe_gpu_ids) or (list(range(role_gpus(s, "probe"))) if probe_copies(s) > 1 else None)


def judges_gpus_of(s: Settings) -> list[int] | None:
    """setup_vllm.sh's GPUs: ``judges_gpu_ids``, else GPUs 0-3 of a machine with more than 4 (None: its 2 or 4)."""
    return gpu_ids(s.judges_gpu_ids) or ([0, 1, 2, 3] if role_gpus(s, "judges") > 4 else None)


def check_settings(s: Settings) -> None:
    """SystemExit on settings no machine can serve: an unknown provider, a Nebius server without GPUs or without a bid,
    copies that don't split the probe server's GPUs evenly, judges on other than 2 or 4 GPUs (setup_vllm.sh's
    layouts), GPU indices the machine doesn't have, a judges cache that nothing would use."""
    for role in SERVER_ROLES:
        p = provider_of(s, role)
        if p not in PROVIDERS:
            raise SystemExit(f"--{role}-provider: one of {', '.join(PROVIDERS)}, got {p!r}")
        if p == "nebius":
            vm = nebius_machine(s, role)
            if any(nj.is_cpu_platform(x) for x in nj.platforms_of(vm["platform"])) or not re.match(r"\d+gpu-",
                                                                                                    vm["preset"]):
                raise SystemExit(f"--{role}-platform/--{role}-preset: a GPU VM, got {vm['platform']}/{vm['preset']}")
            if not vm["on_demand"] and not (vm["bid"] or 0) > 0:
                raise SystemExit(f"--{role}-max-gpu-price: a spot bid in USD per GPU-hour, got {vm['bid']!r}")
    for role, ids in (("probe", probe_gpus_of(s)), ("judges", judges_gpus_of(s))):
        if ids is not None and (len(set(ids)) != len(ids) or not all(0 <= g < role_gpus(s, role) for g in ids)):
            raise SystemExit(f"--{role}-gpu-ids {ids}: distinct GPUs of its machine's {role_gpus(s, role)}")
    copies, gpus = probe_copies(s), len(probe_gpus_of(s) or range(role_gpus(s, "probe")))
    if copies < 1 or gpus % copies:
        raise SystemExit(f"--probe-copies {copies}: the probe server's {gpus} GPUs must split evenly between copies")
    if len(judges_gpus_of(s) or range(role_gpus(s, "judges"))) not in (2, 4):
        raise SystemExit(f"the judges on {judges_gpus_of(s) or role_gpus(s, 'judges')} GPU(s): setup_vllm.sh serves on "
                         "2 or 4 (--judges-gpu-ids; GPUs 0-3 of a machine with more by default)")
    if s.judges_cache and (provider_of(s, "judges") != "nebius" or not any(
            (Path(s.judges_cache).expanduser() / sub).is_dir() for sub in JUDGES_CACHE_DIRS)):
        raise SystemExit(f"--judges-cache {s.judges_cache}: for Nebius judges, a folder with "
                         f"{' or '.join(d + '/' for d in JUDGES_CACHE_DIRS)}")


class Batch:
    """DIR/batch.json: what the batch is (roster, plan, settings) and where it stands (machines, the job, events), saved
    after every change and read again by resume, status, sync, extend, down and watchdog."""

    def __init__(self, out: Path) -> None:
        self.out = Path(out)
        self.path = self.out / "batch.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"events": [], "machines": {}}

    def save(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=str) + "\n")
        tmp.replace(self.path)

    def event(self, what: str, **info) -> None:
        self.data.setdefault("events", []).append({"at": utc(), "what": what, **info})
        self.save()
        log(what + (f" ({json.dumps(info, default=str)[:300]})" if info else ""))

    @property
    def machines(self) -> dict:
        return self.data.setdefault("machines", {})

    @property
    def settings(self) -> Settings:
        return Settings(**self.data.get("settings", {}))

    @property
    def phase(self) -> str:
        return self.data.get("phase", "new")

    def set_phase(self, phase: str) -> None:
        self.data["phase"] = phase
        self.event(f"phase: {phase}")


@contextlib.contextmanager
def batch_lock(out: Path, wait: bool = True):
    """DIR/driver.lock, held by the driver that watches the batch (``start``, ``resume``) and by ``watchdog`` /
    ``down`` while they act, so two never act at once. Yields False if ``wait`` is False and another holds it."""
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "driver.lock", "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        fh.write(f"{os.getpid()} {utc()}\n")
        fh.flush()
        try:
            yield True
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# ------------------------------------------------------------------------------------- launcher + commands


def launcher_env() -> dict:
    """The launcher's environment: this one (``use_prefix`` has set NEBIUS_JOB_NAME_PREFIX)."""
    return dict(os.environ)


def use_prefix(s: Settings) -> None:
    """Every machine's name starts with ``s.prefix``: NEBIUS_JOB_NAME_PREFIX for the launchers, unless it is set."""
    os.environ.setdefault("NEBIUS_JOB_NAME_PREFIX", s.prefix)


def launch(args: list[str], logfile: Path, wait: bool = True):
    """scripts/nebius_job.py ``args``, its output appended to ``logfile``; returns its exit code (``wait``) or the
    Popen."""
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a") as fh:
        fh.write(f"\n=== {utc()} nebius_job.py {shlex.join(args)}\n")
        fh.flush()
        proc = subprocess.Popen([sys.executable, str(LAUNCHER), *args], stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=launcher_env(), start_new_session=True)
    if not wait:
        return proc
    try:
        return proc.wait()
    except KeyboardInterrupt:  # the launcher has its own handlers: pass it on, and let it clean up
        proc.send_signal(signal.SIGINT)
        proc.wait()
        raise


PROBE_CHECK = ('check() { have=$(cat "$1/probe.npz" "$1/meta.json" | sha256sum | cut -d" " -f1); '
               'if [ "$have" = "$2" ]; then echo "probe $1: $2, the pinned one"; else '
               'echo "probe $1: sha256 $have, NOT the pinned $2: refusing to serve it"; exit 1; fi; }')


def probe_server_command(probes: dict, rehearsal: bool, copies: int = 1, gpus: list[int] | None = None) -> str:
    """The probe server's step: refuse a probe folder that isn't the pinned one, then (rehearsal) the GPU check on the
    serving GPUs, then the server: every probe, one episode per forward. PyTorch's expandable segments are the
    server's own default (experiments/inkling_probe_server.py), so they aren't set here. ``gpus``: the GPUs it may use
    (CUDA_VISIBLE_DEVICES; None: every GPU it sees).

    With ``copies`` > 1 the ``gpus`` are split between them: copy k on its own GPUs and port (``probe_ports``),
    scripts/load_balancer.py on 127.0.0.1:8178 before them. The folders are checked once (the copies serve the same
    ones); each copy runs its own golden check as it starts (the server refuses to start otherwise), the rehearsal's GPU
    check runs on each copy's GPUs at once. The step ends, and stops the rest, when any of its processes does: the
    health checks then find it down, and it is restarted whole."""
    lines = [PROBE_CHECK, *(f"check {shlex.quote(p)} {sha}" for p, sha in probes.items()), "mkdir -p results"]

    def serve(port: int) -> str:
        return (f".venv/bin/python experiments/inkling_probe_server.py --model {READER_DIR} --port {port} "
                f"--max-batch-tokens 1 --probes {shlex.quote(','.join(probes))}")

    def gpu_check(suffix: str) -> str:
        return (f".venv/bin/python experiments/check_probe_server.py --model {READER_DIR} --probes-root "
                f"data/probes/Inkling-Small/v2 --out results/check_probe_server_v2{suffix}.json "
                f"> results/check_probe_server_v2{suffix}.log 2>&1")

    if copies == 1:
        env = f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpus))} " if gpus else ""
        if rehearsal:
            lines.append(env + gpu_check("") + '; echo "check_probe_server.py exited $?"')
        lines.append(f"{env}exec " + serve(PROBE_PORT))
        return "\n".join(lines)
    gpus = gpus or list(range(GPUS_PER_COPY * copies))
    per = len(gpus) // copies
    devices = [",".join(str(g) for g in gpus[k * per:(k + 1) * per]) for k in range(copies)]
    ports = probe_ports(copies)
    if rehearsal:
        lines += [f'( CUDA_VISIBLE_DEVICES={d} {gpu_check(f"-{k}")}; echo "check_probe_server.py on GPUs {d} exited '
                  '$?" ) &' for k, d in enumerate(devices)]
        lines.append("wait")
    lines.append("pids=()")
    lines += [f"CUDA_VISIBLE_DEVICES={d} {serve(ports[k])} > >(sed -u 's/^/[copy {k} :{ports[k]}] /') 2>&1 & "
              "pids+=($!)" for k, d in enumerate(devices)]
    lines.append(f".venv/bin/python scripts/load_balancer.py --listen-port {PROBE_PORT} --backend-ports "
                 f"{' '.join(map(str, ports[:-1]))} --listen-host 127.0.0.1 "
                 f"> >(sed -u 's/^/[balancer :{PROBE_PORT}] /') 2>&1 & pids+=($!)")
    lines += ['wait -n "${pids[@]}"; rc=$?',
              'echo "a probe server process exited ($rc): stopping the others"',
              'kill "${pids[@]}" 2>/dev/null; wait "${pids[@]}" 2>/dev/null; exit $(( rc ? rc : 1 ))']
    return "\n".join(lines)


def probe_ports(copies: int) -> list[int]:
    """The probe server's ports on its machine: 8178 alone, or each copy's (8179, 8180, …) then the balancer's."""
    return [PROBE_PORT] if copies == 1 else [*(PROBE_PORT + 1 + k for k in range(copies)), PROBE_PORT]


def probe_health(copies: int = 1) -> str:
    """The probe server's ``--health``: every port of ``probe_ports`` answers /v1/meta."""
    return " && ".join(f"curl -sf -m 20 http://127.0.0.1:{p}/v1/meta > /dev/null" for p in probe_ports(copies))


PROBE_HEALTH = probe_health()
JUDGES_COMMAND = (f"VLLM_VERSION={VLLM_VERSION} VLLM_HOST=127.0.0.1 VLLM_RESTART=1 exec bash scripts/setup_vllm.sh")
JUDGES_HEALTH = " && ".join(f"curl -sf -m 10 http://127.0.0.1:{p}/v1/models | grep -qF {JUDGE_MODELS[p]}"
                            for p in (8001, *Q35_BACKENDS, 8002))


def judges_command(gpus: list[int] | None = None) -> str:
    """The judges' step: Vlad's layout (setup_vllm.sh: Qwen3-30B on the first GPU, the three Qwen3.5-35B on the next
    three), on ``gpus`` (setup_vllm.sh's VLLM_GPUS; None: the machine's 2 or 4)."""
    return f"VLLM_GPUS={','.join(map(str, gpus))} {JUDGES_COMMAND}" if gpus else JUDGES_COMMAND


def server_command(role: str, s: Settings, plan: Plan, rehearsal: bool | None = None) -> str:
    """``role``'s server step on its machine (``rehearsal``: the plan's unless given)."""
    if role.startswith("probe"):
        return probe_server_command(plan.probes, plan.rehearsal if rehearsal is None else rehearsal, probe_copies(s),
                                    probe_gpus_of(s))
    return judges_command(judges_gpus_of(s))


def health_command(s: Settings, role: str) -> str:
    """``role``'s ``--health`` on its machine."""
    return probe_health(probe_copies(s)) if role.startswith("probe") else JUDGES_HEALTH


def vast_up_args(role: str, s: Settings, out: Path, hours: float, gpu: str, plan: Plan, batch_name: str) -> list[str]:
    """``up --provider vast`` for the probe server or the judges: the instance, kept, with its server started and
    healthy. ``gpu``: a name, or a comma list in order of preference."""
    probe = role.startswith("probe")
    what = "Inkling-Small probe server (v2 probes)" if probe else "vLLM judges (Qwen3-30B, Qwen3.5-35B)"
    args = ["up", "--provider", "vast", "--gpu", gpu, "--gpus", str(s.probe_gpus if probe else s.judge_gpus),
            "--disk-gib", str(s.probe_disk_gib if probe else s.judge_disk_gib), "--hours", f"{hours:g}",
            "--name", f"{s.prefix}{'probe-srv' if role == 'probe' else role}", "--out", str(out),
            "--purpose", f"MonitorDecorrelation RL batch {batch_name}: {what} for the runs on the Nebius VM",
            "--tag", f"owner={s.owner}", "--tag", f"experiment={s.experiment}",
            "--tag", f"step={'probe-server' if probe else 'judges'}",
            "--health", health_command(s, role),
            "--health-timeout", str(5400 if plan.rehearsal and probe else 3600), "--wait-hours", "0.5",
            "--rsync", s.rsync, "--hub-gpu", s.hub_gpu]
    price = s.probe_max_price if probe else s.judge_max_price
    if price is not None:
        args += ["--max-price", f"{price:g}"]
    if probe:
        args += ["--hf", READER]
    if s.budget_file:
        args += ["--budget-file", s.budget_file]
    return [*args, "--", server_command(role, s, plan)]


def nebius_up_args(role: str, s: Settings, out: Path, hours: float, plan: Plan) -> list[str]:
    """``up`` for the probe server or the judges on Nebius: the VM in the cheapest of its regions with capacity (spot
    under its bid, or on demand), kept, with its server started and healthy; ``hours`` its shutdown cap (the
    ledger's reservation)."""
    probe, vm = role.startswith("probe"), nebius_machine(s, role)
    args = ["up", "--platform", vm["platform"], "--preset", vm["preset"], "--disk-gib", str(vm["disk_gib"]),
            "--hours", f"{hours:g}", "--name", f"{s.prefix}{'probe-srv' if role == 'probe' else role}", "--out",
            str(out), "--health", health_command(s, role), "--health-timeout",
            str(5400 if plan.rehearsal and probe else 3600), "--wait-hours", "0.5"]
    args += ["--on-demand"] if vm["on_demand"] else ["--max-gpu-price", f"{vm['bid']:g}"]
    if vm["regions"]:
        args += ["--regions", vm["regions"]]
    if probe:
        args += ["--hf", READER]
    elif s.judges_cache:
        for sub in JUDGES_CACHE_DIRS:
            if (Path(s.judges_cache).expanduser() / sub).is_dir():
                args += ["--push", f"{Path(s.judges_cache).expanduser().resolve() / sub}:~/.cache/{sub}"]
    if s.budget_file:
        args += ["--budget-file", s.budget_file]
    return [*args, "--", server_command(role, s, plan)]


def rl_up_args(s: Settings, out: Path, hours: float) -> list[str]:
    """``up`` for the RL machine: the pre-RL rollouts' CPU VM (docs/INFRA.md), set up for root jobs, kept, in one of
    ``rl_regions``."""
    args = ["up", "--platform", "cpu-d3", "--preset", s.rl_preset, "--on-demand", "--image-family",
            "ubuntu24.04-driverless", "--root-job", "--hours", f"{hours:g}", "--name", f"{s.prefix}rl", "--out",
            str(out), "--wait-hours", "0.5"]
    if s.rl_regions:
        args += ["--regions", s.rl_regions]
    return args + (["--budget-file", s.budget_file] if s.budget_file else [])


def up_args(role: str, s: Settings, out: Path, hours: float, plan: Plan, gpu: str | None = None,
            lease: float | None = None) -> list[str]:
    """``role``'s ``up``: the RL machine for ``hours``; a vast.ai server for a ``lease`` on ``gpu``; a Nebius one for
    ``hours``."""
    if role == "rl":
        return rl_up_args(s, out, hours)
    if provider_of(s, role) == "nebius":
        return nebius_up_args(role, s, out, hours, plan)
    return vast_up_args(role, s, out, lease or s.lease_hours, gpu or s.probe_gpu, plan, plan.roster.name)


def judge_gpu(hub, s: Settings) -> tuple[str, dict | None]:
    """The first of ``s.judge_gpus_names`` with an offer for the judges' GPUs (CUDA 13), and that offer."""
    names = [n.strip() for n in s.judge_gpus_names.split(",") if n.strip()]
    for name in names:
        offers = nj.vast_candidates(hub.offers(gpu=name, gpus=s.judge_gpus, disk_gb=nj.disk_gb(s.judge_disk_gib),
                                               max_price=s.judge_max_price), nj.VAST_MIN_CUDA, s.judge_max_price)
        if offers:
            return name, offers[0]
    return names[0] if names else "H200", None


# ------------------------------------------------------------------------------------------------- machines


def machine_record(out: Path) -> dict:
    """A launcher's job.json (empty if it wrote none)."""
    try:
        return json.loads((Path(out) / "job.json").read_text())
    except (OSError, ValueError):
        return {}


def project_of(region: str | None, *files: str | None) -> str | None:
    """The Nebius project of ``region``, from the first of the regions ``files`` that names it (None: none does)."""
    for f in files:
        if region and f:
            with contextlib.suppress(OSError, ValueError, KeyError, SystemExit):
                for r in nj.load_regions(f)[1]:
                    if r.name == region:
                        return r.project
    return None


class Ops:
    """The batch's machines as this driver reaches them: the Nebius VMs over SSH (their ``up``'s address and
    known_hosts, the Nebius key) and the Nebius CLI (their state), the vast.ai instances through hub-gpu (their list
    rows, hub-gpu's key, which never leaves this machine). Remotes are cached and found again after a failure;
    ``procs`` holds the ``up`` of each new server VM this driver started."""

    def __init__(self, batch: Batch) -> None:
        self.batch, self.s = batch, batch.settings
        self.hub = nj.HubGpu(self.s.hub_gpu)
        self._remotes: dict = {}
        self.procs: dict = {}

    def nebius(self, role: str) -> nj.Remote:
        m = self.batch.machines[role]
        if role not in self._remotes:
            key = self.batch.data.get("ssh_key") or os.environ.get("NEBIUS_SSH_KEY")
            self._remotes[role] = nj.Remote(m["host"], Path(m["out"]) / "known_hosts",
                                            str(Path(key).expanduser()) if key else None)
        return self._remotes[role]

    def rl(self) -> nj.Remote:
        return self.nebius("rl")

    def server(self, role: str, fresh: bool = False) -> nj.Remote:
        """``role``'s server machine over SSH: its Nebius VM, or its vast.ai instance (VmStopped if that went away)."""
        if self.batch.machines[role].get("provider") != "nebius":
            return self.vast(role, fresh)
        if fresh:
            self.forget(role)
        return self.nebius(role)

    def vm(self, role: str) -> nj.Instance | None:
        """``role``'s Nebius VM as its project lists it (None: not listed: deleted); InfraError if that is unknown."""
        m = self.batch.machines[role]
        if not m.get("project"):
            raise nj.InfraError(f"the project of {role}'s VM {m.get('name')} is unknown (NEBIUS_REGIONS_FILE)")
        rows = nj.Nebius(os.environ.get("NEBIUS_BIN", "nebius"), m["project"]).instances()
        return next((i for i in rows if i.id == m.get("instance") or i.name == m.get("name")), None)

    def row(self, role: str) -> dict | None:
        """The ``hub-gpu list`` row of ``role``'s instance (None: not listed)."""
        m = self.batch.machines.get(role) or {}
        want = int(m["instance"]) if str(m.get("instance", "")).isdigit() else m.get("name")
        return next((r for r in self.hub.instances() if want in (r.get("instance"), r.get("name"))), None)

    def vast(self, role: str, fresh: bool = False) -> nj.Remote:
        if fresh or role not in self._remotes:
            row = self.row(role)
            if row is None or nj.vast_gone(row):
                raise nj.VmStopped(f"{role}: {nj.vast_gone(row)}")
            folder = self.batch.out / "ssh" / role
            folder.mkdir(parents=True, exist_ok=True)
            self._remotes[role] = nj.vast_remote(self.hub, row, folder)
        return self._remotes[role]

    def forget(self, role: str) -> None:
        self._remotes.pop(role, None)


# What ssh itself writes to stderr on a connection, not the command: vast.ai's login banner (every connection,
# 2026-10-27) and the first connection's known-hosts warning. Left in, it followed the command's JSON and every
# curl_json on a vast.ai instance read "not JSON": the watch would have taken healthy servers for dead.
SSH_NOISE = re.compile(r"^(Welcome to vast\.ai\. |Have fun!\s*$|Warning: Permanently added )")


def run_text(remote: nj.Remote, command: str, timeout: float = 120, stdin_text: str | None = None) -> tuple[int, str]:
    """``command`` on ``remote``: (exit code, stdout and stderr, without ssh's own lines: ``SSH_NOISE``). An SSH
    timeout is (255, ...)."""
    try:
        if stdin_text is None:
            cp = remote.run(command, timeout=timeout, check=False)
        else:
            with tempfile_with(stdin_text) as fh:
                cp = remote.run(command, timeout=timeout, check=False, stdin=fh)
    except subprocess.TimeoutExpired:
        return nj.SSH_DOWN, "ssh timed out"
    err = "".join(line for line in cp.stderr.decode(errors="replace").splitlines(keepends=True)
                  if not SSH_NOISE.match(line))
    return cp.returncode, cp.stdout.decode(errors="replace") + err


def json_in(text: str):
    """The first JSON object or array in ``text`` that decodes, whatever follows it: a report ``cat`` on a vast.ai
    instance, where ssh's banner may surround it (run_text drops the lines it knows: ``SSH_NOISE``)."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            with contextlib.suppress(ValueError):
                return decoder.raw_decode(text, i)[0]
    raise ValueError("no JSON value in the text")


@contextlib.contextmanager
def tempfile_with(text: str):
    import tempfile

    with tempfile.TemporaryFile() as fh:
        fh.write(text.encode())
        fh.seek(0)
        yield fh


# -------------------------------------------------------------------------------------------------- tunnels


def make_tunnel_key(rl: nj.Remote, batch_name: str) -> str:
    """The batch's tunnel key, made on the RL machine in its RAM disk (once); returns its public key."""
    d = os.path.dirname(TUNNEL_KEY)
    rc, out = run_text(rl, f"install -d -m 700 {d} && {{ [ -s {TUNNEL_KEY} ] || ssh-keygen -q -t ed25519 -N '' -C "
                           f"{shlex.quote('rl-batch ' + batch_name)} -f {TUNNEL_KEY}; }} && cat {TUNNEL_KEY}.pub", 60)
    pub = out.strip().splitlines()[-1] if out.strip() else ""
    if rc != 0 or not pub.startswith("ssh-ed25519 "):
        raise nj.InfraError(f"could not make the tunnel key on the RL machine ({rc}): {out[-300:]}")
    return pub


def authorized_line(pubkey: str, ports: list[int]) -> str:
    """The tunnel key's authorized_keys line on a server's machine: forwarding to its server's ports and nothing else
    (no shell, no command, no agent, no pty)."""
    opens = ",".join(f'permitopen="127.0.0.1:{p}"' for p in sorted(ports))
    return f'restrict,port-forwarding,{opens},command="/bin/false" {pubkey}'


# The line on stdin, appended once. vast.ai's authorized_keys ends without a newline (2026-10-23): its last line is
# ended first, or the key would be glued to it (and match no line).
AUTHORIZE = ('umask 077 && mkdir -p "$HOME/.ssh" && f="$HOME/.ssh/authorized_keys" && touch "$f" && line=$(cat) && '
             '[ -n "$line" ] && { grep -qxF -- "$line" "$f" || { if [ -s "$f" ] && [ -n "$(tail -c 1 "$f")" ]; then '
             'echo >> "$f"; fi; printf "%s\\n" "$line" >> "$f"; }; } && grep -cxF -- "$line" "$f"')


def authorize(remote: nj.Remote, line: str) -> None:
    """Append ``line`` to the login user's authorized_keys on a server's machine (root on vast.ai), once (on stdin)."""
    rc, out = run_text(remote, AUTHORIZE, 60, stdin_text=line + "\n")
    if rc != 0:
        raise nj.InfraError(f"could not authorize the tunnel key on {remote.host} ({rc}): {out[-300:]}")


def tunnel_conf(target: str, port: int | None, forwards: list[str], known_hosts: str) -> str:
    """scripts/tunnel.sh's NAME.conf (shell assignments, every value quoted); ``port`` None: 22 (a Nebius VM)."""
    return "".join(f"{k}={shlex.quote(v)}\n" for k, v in (("TARGET", target), ("PORT", str(port or 22)),
                                                         ("KEY", TUNNEL_KEY), ("KNOWN_HOSTS", known_hosts),
                                                         ("FORWARDS", " ".join(forwards))))


def tunnel_forwards(role: str, local_port: int | None = None) -> list[str]:
    """The -L specs of ``role``'s tunnel: the probe's port (copy 0 at 8178), or the judges' two."""
    if role.startswith("probe"):
        return [f"{local_port or PROBE_PORT}:127.0.0.1:{PROBE_PORT}"]
    return [f"{p}:127.0.0.1:{p}" for p in JUDGE_PORTS]


TUNNEL_ALIVE = ('f="$HOME/{d}/{n}.pid"; [ -s "$f" ] && pid=$(cat "$f") && [ -r "/proc/$pid/cmdline" ] && '
                'tr "\\0" " " < "/proc/$pid/cmdline" | grep -qF tunnel.sh')


def start_tunnel(rl: nj.Remote, name: str, target: str, port: int, forwards: list[str], instance: str) -> None:
    """Write ``name``'s conf on the RL machine (a known_hosts of the instance's own: its address may be reused) and start
    its loop unless it runs; a loop that runs is pointed at the new target by killing its live ssh."""
    conf = tunnel_conf(target, port, forwards, f"{name}.{instance}.known_hosts")  # in the tunnel's folder
    rc, out = run_text(rl, f'mkdir -p "$HOME/{TUNNEL_DIR}" && cat > "$HOME/{TUNNEL_DIR}/{name}.conf"', 60,
                       stdin_text=conf)
    if rc != 0:
        raise nj.InfraError(f"could not write the {name} tunnel's conf ({rc}): {out[-300:]}")
    alive = TUNNEL_ALIVE.format(d=TUNNEL_DIR, n=name)
    inner = f'exec bash "$HOME/{nj.REMOTE_REPO}/scripts/tunnel.sh" "$HOME/{TUNNEL_DIR}" {shlex.quote(name)}'
    start = f"setsid nohup bash -lc {shlex.quote(inner)} > /dev/null 2>&1 < /dev/null &"
    kick = f'[ -s "$HOME/{TUNNEL_DIR}/{name}.ssh.pid" ] && kill "$(cat "$HOME/{TUNNEL_DIR}/{name}.ssh.pid")"'
    rc, out = run_text(rl, f"if {alive}; then {kick}; echo retargeted; else {start} echo started; fi", 60)
    if rc != 0:
        raise nj.InfraError(f"could not start the {name} tunnel ({rc}): {out[-300:]}")


def tunnel_state(rl: nj.Remote, name: str) -> dict:
    """``name``'s loop alive?, its ssh's pid, its state line and the last lines of its log."""
    alive = TUNNEL_ALIVE.format(d=TUNNEL_DIR, n=name)
    rc, out = run_text(rl, f'if {alive}; then echo loop=1; else echo loop=0; fi; '
                           f'echo "ssh=$(cat "$HOME/{TUNNEL_DIR}/{name}.ssh.pid" 2>/dev/null)"; '
                           f'echo "state=$(cat "$HOME/{TUNNEL_DIR}/{name}.state" 2>/dev/null)"; '
                           f'tail -n 3 "$HOME/{TUNNEL_DIR}/{name}.log" 2>/dev/null | sed "s/^/log=/"', 60)
    if rc == nj.SSH_DOWN:
        return {"reachable": False}
    info = {"reachable": True, "loop": "loop=1" in out, "log": []}
    for line in out.splitlines():
        if line.startswith("ssh="):
            info["ssh_pid"] = line[4:].strip() or None
        elif line.startswith("state="):
            info["state"] = line[6:].strip()
        elif line.startswith("log="):
            info["log"].append(line[4:])
    return info


def kick_tunnel(rl: nj.Remote, name: str) -> None:
    """Kill ``name``'s live ssh: its loop connects again (a forced reconnect)."""
    run_text(rl, f'[ -s "$HOME/{TUNNEL_DIR}/{name}.ssh.pid" ] && kill "$(cat "$HOME/{TUNNEL_DIR}/{name}.ssh.pid")"', 60)


def stop_tunnel(rl: nj.Remote, name: str) -> None:
    alive = TUNNEL_ALIVE.format(d=TUNNEL_DIR, n=name)
    run_text(rl, f'if {alive}; then kill "$(cat "$HOME/{TUNNEL_DIR}/{name}.pid")"; fi', 60)


# --------------------------------------------------------------------------------------------------- health


def curl_json(remote: nj.Remote, url: str, timeout: int = 20) -> tuple[bool, object]:
    """GET ``url`` on ``remote`` (curl there): (answered, the JSON or what went wrong)."""
    rc, out = run_text(remote, f"curl -sf -m {timeout} {shlex.quote(url)}", timeout + 60)
    if rc == nj.SSH_DOWN:
        return False, "ssh failed"
    if rc != 0:
        return False, f"curl exit {rc}"
    try:
        return True, json.loads(out)
    except ValueError:
        return False, f"not JSON: {out[:120]!r}"


def probe_ok(meta: object, pinned: dict, port: int = PROBE_PORT) -> str | None:
    """Why a probe server's ``/v1/meta`` answer is not the batch's (None: it is): every pinned sha served."""
    if not isinstance(meta, dict):
        return f"no answer on :{port} ({meta})"
    missing = sorted(set(pinned.values()) - set(meta.get("probes") or {}))
    return f"the server at :{port} doesn't serve {missing}" if missing else None


def judges_ok(remote: nj.Remote, ports=(8001, *Q35_BACKENDS, 8002), timeout: int = 10) -> str | None:
    """Why the judges on ``remote`` are not all up (None: each port serves its model within ``timeout`` seconds)."""
    for p in ports:
        ok, answer = curl_json(remote, f"http://127.0.0.1:{p}/v1/models", timeout)
        if not ok:
            return f":{p}: {answer}"
        served = {m.get("id") for m in (answer.get("data") or [])} if isinstance(answer, dict) else set()
        if JUDGE_MODELS[p] not in served:
            return f":{p} serves {sorted(served)}, not {JUDGE_MODELS[p]}"
    return None


def server_health(ops: Ops, role: str, pinned: dict) -> str | None:
    """Why ``role``'s server is not healthy on its own machine (None: it is): every probe port (each copy's and the
    balancer's) serving every pinned sha, or each judge engine its model."""
    remote = ops.server(role)
    if role.startswith("probe"):
        for port in probe_ports(probe_copies(ops.s)):
            ok, meta = curl_json(remote, f"http://127.0.0.1:{port}/v1/meta")
            if why := probe_ok(meta, pinned, port):
                return why
        return None
    return judges_ok(remote)


# Through the tunnels: during an eval burst (256 episodes × 4 judges per run at once) /v1/models took over the 10-20 s
# the checks allowed, and healthy judges read "down through the tunnel (:8001: curl exit 28)" (2026-10-27). So a longer
# timeout, and a tunnel repaired only when two rounds in a row find it down (health_round).
TUNNEL_CHECK_SECONDS = 60


def through_tunnels(ops: Ops, plan_probes: dict, roles: list[str]) -> dict:
    """What the runs see on the RL machine: per role, None (healthy through its tunnel) or what is wrong."""
    rl, out = ops.rl(), {}
    for role in roles:
        if role.startswith("probe"):
            port = ops.batch.machines[role].get("local_port", PROBE_PORT)
            ok, meta = curl_json(rl, f"http://127.0.0.1:{port}/v1/meta", TUNNEL_CHECK_SECONDS)
            out[role] = probe_ok(meta, plan_probes, port)
        else:
            out[role] = judges_ok(rl, JUDGE_PORTS, TUNNEL_CHECK_SECONDS)
    return out


# ------------------------------------------------------------------------------------- the job on the RL VM

JOB_STEP = "rl-supervisor"  # the detached step on the RL machine (its k-th start: rl-supervisor-k)
STATE = f"{REMOTE_BATCH}/state"  # in the RL machine's repo copy: status.json, control/, supervisor.log


def secrets_list() -> list:
    return [nj.parse_secret(n) for n in SECRETS]


def start_job(ops: Ops, plan_path: Path, k: int, check_nobody: bool = False) -> dict:
    """Push the plan, send the keys, and start the supervisor as root on the RL machine (its k-th start). The keys are
    read by the job's root shell as it starts, into its environment, and deleted from the RAM disk at once.
    ``check_nobody`` (rehearsal): first check that ``nobody`` cannot read them. Returns what was checked."""
    rl, checks = ops.rl(), {}
    keys = secrets_list()
    for s in keys:
        if problem := nj.secret_problem(s.path):
            raise SystemExit(f"--secret {s.name}: {s.path} is {problem}; the owner puts it there with `hub-secret set "
                             f"{s.name}`")
    rc, out = run_text(rl, f"mkdir -p {nj.REMOTE_REPO}/{STATE}/control", 60)
    if rc != 0:
        raise nj.InfraError(f"could not make the state folder on the RL machine: {out[-300:]}")
    nj.push_path(rl, str(plan_path), f"{REMOTE_BATCH}/plan.json")
    nj.send_secrets(rl, keys)
    if check_nobody:
        rc, out = run_text(rl, f"sudo -n -u nobody cat {nj.REMOTE_SECRETS}/* > /dev/null 2>&1; echo $?", 60)
        checks["nobody_reads_keys"] = out.strip().splitlines()[-1] if out.strip() else "?"
    remove = shlex.join(["rm", "-f", "--", *(nj.remote_secret(s.name) for s in keys)])
    command = (f"{remove}\nexec .venv/bin/python scripts/rl_batch.py vm-supervise --plan {REMOTE_BATCH}/plan.json "
               f"--state {STATE}")
    env = [("WANDB_ENTITY", ops.batch.data["roster"]["wandb_entity"]), ("WANDB_MODE", "online")]
    step = JOB_STEP if k == 1 else f"{JOB_STEP}-{k}"
    nj.start(rl, step, nj.job_script(command, env, secrets=keys, root=True), cwd=nj.REMOTE_REPO)
    ops.batch.data.setdefault("job", {}).update(step=step, starts=k, started=utc())
    ops.batch.event(f"the supervisor started on the RL machine as root ({step}); keys sent: "
                    f"{', '.join(s.name for s in keys)}")
    return checks


def job_exit(ops: Ops) -> int | None:
    """The supervisor step's exit code (None: still running, or unreachable)."""
    step = (ops.batch.data.get("job") or {}).get("step")
    if not step:
        return None
    rc, out = run_text(ops.rl(), f"cat {nj.REMOTE_LOGS}/{step}.exit 2>/dev/null; true", 60)
    out = out.strip()
    return int(out) if rc == 0 and out.lstrip("-").isdigit() else None


def vm_status(ops: Ops) -> dict | None:
    """The supervisor's status.json on the RL machine (None: unreachable or not written yet)."""
    rc, out = run_text(ops.rl(), f"cat {nj.REMOTE_REPO}/{STATE}/status.json", 60)
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except ValueError:
        return None


def control(ops: Ops, what: str) -> None:
    """A control file for the supervisor: restart-<run>, stop-<run>, stop-all."""
    rc, out = run_text(ops.rl(), f"touch {nj.REMOTE_REPO}/{STATE}/control/{shlex.quote(what)}", 60)
    if rc != 0:
        raise nj.InfraError(f"could not write the control file {what}: {out[-300:]}")


def sync(ops: Ops, runs: list[str]) -> dict:
    """rsync every run folder and the supervisor's state from the RL machine into DIR/runs/ and DIR/vm/; per path,
    "ok", "missing" or the error. Resumable and incremental (nebius_job.fetch_rsync); files still being written wait."""
    rl, done = ops.rl(), {}
    for path, dest in [*((f"data/runs/{r}", ops.batch.out / "runs") for r in runs),
                       (STATE, ops.batch.out / "vm")]:
        try:
            done[path] = nj.fetch_rsync(rl, path, dest, ops.s.rsync, timeout=1800, exclude=nj.IN_PROGRESS)
        except Exception as e:  # noqa: BLE001 — the next round tries again
            done[path] = f"error: {e}"
    ops.batch.data["last_sync"] = {"at": utc(), "result": done}
    ops.batch.save()
    return done


# --------------------------------------------------------------------------------------------------- leases


def lease_hours_for(row: dict, lease: float, now: float | None = None) -> float | None:
    """The hours-from-start an instance (its list row) should be extended to (None: no need): its end at least half
    a lease away, else moved to a lease from now, in quarter hours, at most the 72 h of a timed run."""
    now = time.time() if now is None else now
    started, deadline = row.get("started"), row.get("deadline")
    if not started or not deadline or row.get("ended"):
        return None
    if float(deadline) - now >= lease * 1800:
        return None
    want = math.ceil(((now + lease * 3600) - float(started)) / 900) / 4
    have = (float(deadline) - float(started)) / 3600
    want = min(want, 72.0)
    return want if want > have + 1e-6 else None


def extend_leases(ops: Ops) -> None:
    """Every vast.ai instance of the batch kept at least half a lease from its end (``nebius_job.py extend``), and
    every Nebius VM's shutdown (the RL machine's, a Nebius server's) moved to a lease from now, never past its
    ``--hours`` (the ledger's reservation)."""
    s, b = ops.s, ops.batch
    for role, m in b.machines.items():
        if m.get("provider") != "vast" or m.get("ended"):
            continue
        try:
            row = ops.row(role)
        except nj.InfraError as e:
            log(f"lease of {role}: could not list the instances ({e})")
            continue
        want = lease_hours_for(row or {}, s.lease_hours)
        if want is None:
            continue
        code = launch(["extend", "--provider", "vast", "--name", m["name"], "--hours", f"{want:g}", "--hub-gpu",
                       s.hub_gpu, *(["--budget-file", s.budget_file] if s.budget_file else [])],
                      b.out / "launch" / f"{role}-extend.log")
        if code == 0:
            b.event(f"lease: {role} ({m['name']}) extended to {want:g} h after its start")
        else:
            b.event(f"!!! lease: extending {role} ({m['name']}) to {want:g} h was refused (exit {code}; see "
                    f"launch/{role}-extend.log); it ends at {utc(float(row['deadline']))} unless extended")
    for role, m in list(b.machines.items()):
        if m.get("provider") != "nebius" or not m.get("host") or m.get("ended"):
            continue
        created = nj._seconds(m["created"]) if m.get("created") else None
        cap = created + float(m.get("hours") or 0) * 3600 if created else None
        until = time.time() + s.lease_hours * 3600
        if cap:
            until = min(until, cap)
        minutes = max(5, int((until - time.time()) / 60))
        rc, out = run_text(ops.rl() if role == "rl" else ops.server(role),
                           f"sudo -n shutdown -h +{minutes} > /dev/null 2>&1; echo $?", 60)
        if rc == 0:
            m["shutdown_at"] = utc(time.time() + minutes * 60)
            b.save()


def sample_prices(batch: Batch) -> None:
    """Each kept Nebius spot server VM's price now (the cost calculator, as its launcher samples it while it runs),
    appended to its job.json's price_samples: ``down`` settles its ledger entry from those, conservatively, instead of
    at its bid for all the time after its ``up`` (whose sampler stopped with it)."""
    for role, m in batch.machines.items():
        if role == "rl" or m.get("provider") != "nebius" or not m.get("spot") or not m.get("host") or m.get("ended"):
            continue
        path = Path(m["out"]) / "job.json"
        try:
            rec = json.loads(path.read_text())
            price = nj.Nebius(os.environ.get("NEBIUS_BIN", "nebius"), m["project"]).estimate_hourly(
                m["platform"], m["preset"], True)
        except (OSError, ValueError, KeyError, TypeError, nj.InfraError) as e:
            log(f"price of {role}'s VM: not sampled ({type(e).__name__}: {e})")
            continue
        rec.setdefault("price_samples", []).append([nj.utc(), price])
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1) + "\n")
        tmp.replace(path)


# ---------------------------------------------------------------------------- on the RL machine: the supervisor


@dataclass
class RunState:
    name: str
    argv: list[str]
    state: str = "pending"  # pending | waiting | running | backoff | done | failed | stopped
    attempts: int = 0  # failed attempts (a run restarted on purpose doesn't count)
    restarts: int = 0  # restarted on purpose (control restart-<run>)
    pid: int | None = None
    adopted: bool = False  # a run a previous supervisor started, still running
    started: float | None = None
    last_exit: int | None = None
    next_try: float = 0.0
    note: str = ""
    wandb: str | None = None
    modes: list = field(default_factory=list)  # how each attempt started: fresh / resume / aside


def run_done(run_dir: Path) -> bool:
    """A run is done once its final checkpoint is written and its latest resume state is the finished run's
    (rl/train.py writes both last; a run stopped early by stop_after_zero_behavior_steps is done too)."""
    if not (run_dir / "final_checkpoint.txt").is_file():
        return False
    states = sorted((run_dir / "resume_states").glob("step_*.json"))
    try:
        return bool(states) and bool(json.loads(states[-1].read_text()).get("eval_done"))
    except (OSError, ValueError):
        return False


def run_start_mode(run_dir: Path) -> str:
    """How a run starts again: ``fresh`` (no folder, or only its run.log), ``resume`` (a saved state:
    run_experiment.py --resume), ``aside`` (a folder but no saved state, e.g. it failed before step 0's save: moved
    aside, then fresh)."""
    if not run_dir.is_dir() or all(p.name == "run.log" for p in run_dir.iterdir()):
        return "fresh"
    if any((run_dir / "resume_states").glob("step_*.json")):
        return "resume"
    return "aside"


WANDB_URL = re.compile(r"https://wandb\.ai/([^/\s]+)/([^/\s]+)/runs/([A-Za-z0-9]+)")
# The clients' retry warnings (monitors/inkling_probe_monitor.py every 3 minutes; monitors/openrouter.post_chat, which
# the vLLM judges use, after 6 attempts): what a waiting run waits for.
RETRY_WARNING = re.compile(r"no answer from the probe server for \d+s|: attempt \d+ failed \(.*\); retrying")


def wandb_check(log_text: str, entity: str, project: str) -> tuple[str | None, str | None]:
    """(the run's W&B URL, why it is wrong or None) from the tail of its run.log."""
    urls = WANDB_URL.findall(log_text)
    if not urls:
        offline = re.search(r"wandb: (?:W&B syncing is set to .offline|.*offline mode)|wandb: offline", log_text)
        return None, ("W&B is offline (no key, or WANDB_MODE)" if offline else None)
    e, p, rid = urls[-1]
    url = f"https://wandb.ai/{e}/{p}/runs/{rid}"
    return url, (None if (e, p) == (entity, project) else f"W&B logs to {e}/{p}, not {entity}/{project}")


def tcp_connections(port: int) -> int:
    """Established TCP connections to ``port`` on this machine (/proc/net/tcp*): with the probe server's port, how many
    requests the runs have waiting on it (each holds a connection while it long-polls)."""
    n, hexport = 0, f"{port:04X}"
    for f in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(f).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) > 3 and parts[2].rsplit(":", 1)[-1] == hexport and parts[3] == "01":
                n += 1
    return n


def pid_ended(pid: int | None) -> bool:
    """Has process ``pid`` ended? Gone, or a zombie its parent hasn't reaped (/proc/<pid>/stat state Z)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (OSError, TypeError):
        return True
    return stat.rsplit(")", 1)[-1].split()[0] in ("Z", "X")


def http_json(url: str, timeout: float = 15.0) -> object:
    import urllib.request

    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


class Supervisor:
    """``vm-supervise``: every run of the plan, one process each, as root in the repo on the RL machine.
    - **Before anything:** the keys (TINKER_API_KEY, WANDB_API_KEY), the W&B entity and online mode, root, and the
      pinned probes' shas in this copy of the repo; any of them wrong is exit 2, loudly.
    - **Starting a run:** only once every server answers through the tunnels (the probe server serving the pinned
      shas, each judge its model): a monitor checks its server as it is built and fails the run otherwise. Fresh, or
      ``--resume`` from its last saved state, or (a folder without one) the folder moved aside and fresh.
    - **A run that exits:** done if its final state is written; else, after a backoff (``retry_s``, doubling, at most
      15 minutes), started again, up to ``max_attempts`` failed attempts.
    - **A run whose W&B URL names another entity or project, or that went offline,** is stopped and failed.
    - **Control files** in STATE/control: ``restart-<run>`` (stop it and start it again at once, ``--resume``),
      ``stop-<run>``, ``stop-all``.
    - **Status:** STATE/status.json every few seconds: each run's state, attempts, step times, W&B URL; the probe
      queue's depth (established connections to the probe port) over time.
    - **Its own end:** once every run is done, failed or stopped (exit 0 only if all are done). SIGTERM stops the
      supervisor only: the runs, each in a session of its own, go on, and a new supervisor takes them over."""

    def __init__(self, plan: dict, state: Path, repo: Path, *, popen=subprocess.Popen, clock=time.time,
                 sleep=time.sleep, poll_s: float = 5.0, health=None) -> None:
        self.plan, self.state, self.repo = plan, Path(state), Path(repo)
        self.popen, self.clock, self.sleep, self.poll_s = popen, clock, sleep, poll_s
        self.health_fn = health or self._health
        self.max_attempts = int(plan.get("max_attempts", 8))
        self.retry_s = float(plan.get("retry_s", 60))
        self.runs = [RunState(r["name"], list(r["argv"])) for r in plan["runs"]]
        self.procs: dict = {}
        self.queue: list = []  # [time, connections] samples of the probe port
        self.down: str | None = None
        self.stopping = False
        self._said_down = 0.0
        self.log_path = self.state / "supervisor.log"

    # -------------------------------------------------------------------------------------------- helpers

    def say(self, msg: str) -> None:
        line = f"[supervisor {datetime.now(UTC):%H:%M:%S}] {msg}"
        print(line, flush=True)
        with contextlib.suppress(OSError):
            with self.log_path.open("a") as fh:
                fh.write(line + "\n")

    def run_dir(self, rs: RunState) -> Path:
        return self.repo / "data" / "runs" / rs.name

    def check_env(self) -> str | None:
        """Why the runs can't start here (None: they can)."""
        missing = [k for k in ("TINKER_API_KEY", "WANDB_API_KEY") if not os.environ.get(k, "").strip()]
        if missing:
            return f"{', '.join(missing)} not set: the runs would fail (tinker) or log to W&B nowhere (offline)"
        if os.environ.get("WANDB_ENTITY") != self.plan["entity"]:
            return f"WANDB_ENTITY is {os.environ.get('WANDB_ENTITY')!r}, not the roster's {self.plan['entity']!r}"
        if os.environ.get("WANDB_MODE") != "online":
            return f"WANDB_MODE is {os.environ.get('WANDB_MODE')!r}, not online: a run without W&B would go unnoticed"
        if self.plan.get("need_root", True) and os.geteuid() != 0:
            return "not root: the terminal env's read-only verifier runs the policy's commands as nobody"
        for path, sha in (self.plan.get("probes") or {}).items():
            try:
                have = probe_sha(self.repo / path)
            except OSError as e:
                return f"probe {path}: {e}"
            if have != sha:
                return f"probe {path} here is {have}, not the pinned {sha}: the monitor would ask for a probe the server " \
                       "doesn't serve (409, fatal)"
        return None

    def _health(self) -> str | None:
        """What is down among the servers, through the tunnels (None: all up)."""
        for h in self.plan.get("health") or []:
            try:
                answer = http_json(h["url"])
            except Exception as e:  # noqa: BLE001
                return f"{h['name']} ({h['url']}): {type(e).__name__}: {e}"
            if "shas" in h:
                missing = sorted(set(h["shas"]) - set((answer or {}).get("probes") or {}))
                if missing:
                    return f"{h['name']} doesn't serve the pinned probes {missing}"
            if "model" in h and h["model"] not in {m.get("id") for m in (answer or {}).get("data") or []}:
                return f"{h['name']} doesn't serve {h['model']}"
        return None

    # ------------------------------------------------------------------------------------------- the runs

    def adopt(self) -> None:
        """Runs a previous supervisor started and that still run (STATE/pids/<run>.pid): watched, not started twice."""
        for rs in self.runs:
            f = self.state / "pids" / f"{rs.name}.pid"
            try:
                pid = int(f.read_text().strip())
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            except (OSError, ValueError):
                continue
            if any(a.endswith(b"experiments/run_experiment.py") for a in cmd) and f"run_name={rs.name}".encode() in cmd \
                    and not pid_ended(pid):
                rs.pid, rs.adopted, rs.state, rs.started = pid, True, "running", self.clock()
                self.say(f"{rs.name}: running (pid {pid}), started by an earlier supervisor: watching it")
        for rs in self.runs:
            if rs.state == "pending" and run_done(self.run_dir(rs)):
                rs.state = "done"

    def launch(self, rs: RunState) -> None:
        d = self.run_dir(rs)
        mode = run_start_mode(d)
        if mode == "aside":
            k = 1
            while (d.parent / f"{rs.name}.attempt-{k}").exists():
                k += 1
            d.rename(d.parent / f"{rs.name}.attempt-{k}")
            self.say(f"{rs.name}: its folder has no saved state; moved aside as {rs.name}.attempt-{k}, starting fresh")
        d.mkdir(parents=True, exist_ok=True)
        argv = [*rs.argv, *(["--resume"] if mode == "resume" else [])]
        with open(d / "run.log", "ab") as fh:
            fh.write(f"\n=== {utc()} supervisor: {mode} start: {shlex.join(argv)}\n".encode())
            fh.flush()
            proc = self.popen(argv, cwd=str(self.repo), stdout=fh, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL, start_new_session=True)
        self.procs[rs.name] = proc
        rs.pid, rs.adopted, rs.state, rs.started, rs.note, rs.wandb = proc.pid, False, "running", self.clock(), "", None
        rs.modes.append(mode)
        (self.state / "pids").mkdir(parents=True, exist_ok=True)
        (self.state / "pids" / f"{rs.name}.pid").write_text(f"{proc.pid}\n")
        self.say(f"{rs.name}: started ({mode}, pid {proc.pid})")

    def exited(self, rs: RunState) -> int | None:
        """The run's exit code if it ended (an adopted run's code can't be known: 0 if done, else 1)."""
        if rs.adopted:
            if not pid_ended(rs.pid):
                return None
            return 0 if run_done(self.run_dir(rs)) else 1
        proc = self.procs.get(rs.name)
        return None if proc is None else proc.poll()

    def signal_run(self, rs: RunState, sig: int = signal.SIGTERM) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError, TypeError):
            os.killpg(rs.pid, sig)  # its session's group: the run and the env's subprocesses

    def controls(self) -> None:
        folder = self.state / "control"
        for f in sorted(folder.glob("*")) if folder.is_dir() else []:
            what = f.name
            f.unlink(missing_ok=True)
            if what == "stop-all":
                self.say("control: stop-all")
                self.stopping = True
                for rs in self.runs:
                    if rs.state == "running":
                        self.signal_run(rs)
                    if rs.state not in ("done", "failed"):
                        rs.state, rs.note = "stopped", "stop-all"
                continue
            verb, _, name = what.partition("-")
            rs = next((r for r in self.runs if r.name == name), None)
            if rs is None or verb not in ("restart", "stop"):
                self.say(f"control: {what}: no such run or command")
                continue
            self.say(f"control: {verb} {name}")
            if rs.state == "running":
                self.signal_run(rs)
                self.wait_gone(rs)
            if verb == "stop":
                rs.state, rs.note = "stopped", "stop-" + name
            else:
                rs.restarts += 1
                rs.state, rs.next_try, rs.note = "pending", 0.0, "restarted on purpose"
                if rs.attempts >= self.max_attempts:
                    rs.attempts = self.max_attempts - 1  # a failed run restarted by hand gets one more go

    def wait_gone(self, rs: RunState, grace: float = 120.0) -> None:
        t0 = self.clock()
        while self.exited(rs) is None and self.clock() - t0 < grace:
            self.sleep(1)
        if self.exited(rs) is None:
            self.signal_run(rs, signal.SIGKILL)
            self.sleep(2)
        self.exited(rs)  # reap it

    def check_wandb(self, rs: RunState) -> None:
        if rs.wandb is not None:
            return
        try:
            with open(self.run_dir(rs) / "run.log", "rb") as fh:
                fh.seek(max(0, os.fstat(fh.fileno()).st_size - 400_000))
                text = fh.read().decode(errors="replace")
        except OSError:
            return
        url, problem = wandb_check(text, self.plan["entity"], self.plan["project"])
        if problem:
            self.say(f"!!! {rs.name}: {problem}: stopping it")
            self.signal_run(rs)
            self.wait_gone(rs)
            rs.state, rs.note, rs.attempts = "failed", problem, self.max_attempts
        elif url:
            rs.wandb = url
            self.say(f"{rs.name}: W&B {url}")

    def round(self) -> bool:
        """One pass: controls, ended runs, starts, W&B, status. True once every run is final."""
        now = self.clock()
        self.controls()
        for rs in self.runs:
            if rs.state != "running":
                continue
            code = self.exited(rs)
            self.check_wandb(rs)  # also once it ended: a quick run still gets its URL checked
            if rs.state != "running" or code is None:
                continue
            rs.last_exit, rs.pid = code, None
            (self.state / "pids" / f"{rs.name}.pid").unlink(missing_ok=True)
            if run_done(self.run_dir(rs)):
                rs.state = "done"
                (self.run_dir(rs) / "QUEUE_DONE").write_text(utc() + "\n")  # scripts/queue_runs.sh's sentinel
                self.say(f"{rs.name}: done (exit {code})")
            elif rs.note == "restarted on purpose" or self.stopping:
                pass
            else:
                rs.attempts += 1
                if rs.attempts >= self.max_attempts:
                    rs.state = "failed"
                    self.say(f"!!! {rs.name}: exited {code}, attempt {rs.attempts} of {self.max_attempts}: failed")
                else:
                    wait = min(900.0, self.retry_s * 2 ** (rs.attempts - 1))
                    rs.state, rs.next_try = "backoff", now + wait
                    self.say(f"{rs.name}: exited {code} (attempt {rs.attempts} of {self.max_attempts}); "
                             f"starting it again in {wait:.0f}s")
        todo = [rs for rs in self.runs if rs.state in ("pending", "waiting", "backoff") and rs.next_try <= now]
        if todo and not self.stopping:
            self.down = self.health_fn()
            if self.down:
                for rs in todo:
                    rs.state = "waiting"
                if now - self._said_down > 300:
                    self.say(f"waiting for the servers before starting {', '.join(r.name for r in todo)}: {self.down}")
                    self._said_down = now
            else:
                for rs in todo:
                    self.launch(rs)
        self.queue.append([round(now), tcp_connections(PROBE_PORT)])
        self.queue = self.queue[-2880:]
        self.write_status()
        return all(rs.state in ("done", "failed", "stopped") for rs in self.runs)

    def last_warning(self, rs: RunState) -> str | None:
        """The run's latest retry warning in its run.log: the probe client's (no answer from the probe server for Ns)
        or a judge call's (attempt N failed; retrying). A run that waits shows why."""
        try:
            with open(self.run_dir(rs) / "run.log", "rb") as fh:
                fh.seek(max(0, os.fstat(fh.fileno()).st_size - 100_000))
                lines = fh.read().decode(errors="replace").splitlines()
        except OSError:
            return None
        hit = [line for line in lines if RETRY_WARNING.search(line)]
        return hit[-1][-300:] if hit else None

    def step_times(self, rs: RunState) -> dict:
        rows = []
        with contextlib.suppress(OSError):
            for line in (self.run_dir(rs) / "metrics.jsonl").read_text().splitlines():
                with contextlib.suppress(ValueError):
                    r = json.loads(line)
                    rows.append((r.get("step"), r.get("time/total_s"), r.get("time/score_s")))
        evals = 0
        with contextlib.suppress(OSError):
            evals = sum(1 for _ in (self.run_dir(rs) / "eval_metrics.jsonl").open())
        totals = sorted(t for _, t, _ in rows if isinstance(t, (int, float)))
        return {"steps": len(rows), "evals": evals,
                "step_s_median": round(totals[len(totals) // 2], 1) if totals else None,
                "last_step_s": round(rows[-1][1], 1) if rows and isinstance(rows[-1][1], (int, float)) else None,
                "last_probe_wait_s": round(rows[-1][2], 1) if rows and isinstance(rows[-1][2], (int, float)) else None}

    def write_status(self, phase: str | None = None) -> None:
        final = all(rs.state in ("done", "failed", "stopped") for rs in self.runs)
        status = {"at": utc(), "phase": phase or ("done" if final and all(rs.state == "done" for rs in self.runs)
                                                  else "failed" if final else "running"),
                  "supervisor_pid": os.getpid(),
                  "servers_down": self.down, "probe_connections": self.queue[-1][1] if self.queue else None,
                  "probe_queue": self.queue[-240:],
                  "runs": {rs.name: {**{k: v for k, v in dataclasses.asdict(rs).items() if k != "argv"},
                                     **self.step_times(rs),
                                     "last_warning": self.last_warning(rs) if rs.state == "running" else None}
                           for rs in self.runs}}
        tmp = self.state / "status.json.tmp"
        tmp.write_text(json.dumps(status, indent=1) + "\n")
        tmp.replace(self.state / "status.json")

    def run(self) -> int:
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / "control").mkdir(exist_ok=True)
        problem = self.check_env()
        if problem:
            self.say(f"!!! refusing to start: {problem}")
            self.write_status("refused")
            return 2
        self.say(f"{len(self.runs)} runs: {', '.join(rs.name for rs in self.runs)}; W&B {self.plan['entity']}/"
                 f"{self.plan['project']}; at most {self.max_attempts} failed attempts each")
        self.adopt()
        while not self.round():
            self.sleep(self.poll_s)
        self.write_status()
        done = sum(rs.state == "done" for rs in self.runs)
        self.say(f"all runs ended: {done} done, {len(self.runs) - done} failed or stopped")
        return 0 if done == len(self.runs) else 1


def cmd_vm_supervise(args) -> int:
    plan = json.loads(Path(args.plan).read_text())
    sup = Supervisor(plan, Path(args.state), REPO)

    def on_term(signum, frame):  # the supervisor only: the runs go on, and the next supervisor adopts them
        sup.say(f"{signal.Signals(signum).name}: the supervisor stops; the runs go on")
        with contextlib.suppress(Exception):
            sup.write_status("supervisor-stopped")
        os._exit(143)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGHUP, on_term)
    return sup.run()


# ---------------------------------------------------------------------------------------------- the driver


class Detach(Exception):
    """SIGTERM or SIGHUP to the driver: stop watching, leave the machines to their leases."""


def roles_of(plan: Plan) -> list[str]:
    return [*(["probe"] if plan.probes else []), *(["judges"] if plan.judges else [])]


def vm_plan(batch: Batch, plan: Plan) -> dict:
    """What the supervisor reads on the RL machine (rl_batch/plan.json)."""
    s, roster = batch.settings, plan.roster
    health = []
    if plan.probes:
        health.append({"name": "the probe server", "url": f"{PROBE_URL}/v1/meta",
                       "shas": sorted(set(plan.probes.values()))})
    if plan.judges:
        health += [{"name": f"the judge at :{p}", "url": f"http://127.0.0.1:{p}/v1/models", "model": JUDGE_MODELS[p]}
                   for p in JUDGE_PORTS]
    return {"batch": roster.name, "entity": roster.wandb_entity, "project": roster.wandb_project,
            "runs": [{"name": r.row.run_name, "argv": r.argv} for r in plan.runs], "probes": plan.probes,
            "health": health, "max_attempts": s.max_attempts, "retry_s": 60}


def plan_from(batch: Batch) -> Plan:
    """The batch's plan as saved (resume, watchdog, …): the roster it started with, its pinned probes."""
    r = batch.data["roster"]
    roster = Roster(**{**r, "rows": [Row(**row) for row in r["rows"]]})
    runs = [RunPlan(Row(**x["row"]), x["argv"], x["probes"], x["judges"]) for x in batch.data["plan"]["runs"]]
    return Plan(roster, runs, batch.data["plan"]["probes"], batch.data["plan"]["judges"],
                batch.data["plan"].get("rehearsal", False))


def region_names(value: str | None) -> list[str]:
    return [r.strip() for r in (value or "").split(",") if r.strip()]


def nebius_offers(s: Settings, roles: list[str], hours: float) -> dict:
    """Each Nebius server's cheapest region now (read-only: every region's price and capacity, the launcher's
    ``survey``) under its bid, and room in the ledger for their worst cases: bid × GPUs × ``hours`` for spot, the
    on-demand price × ``hours`` otherwise. SystemExit if a server has no region, or the ledger no room."""
    path = os.environ["NEBIUS_REGIONS_FILE"]
    tenant, regions = nj.load_regions(path)
    found = {}
    for role in roles:
        vm = nebius_machine(s, role)
        here = [r for r in regions if not vm["regions"] or r.name in region_names(vm["regions"])]
        gpus = nj.gpu_count(vm["preset"])
        cap = None if vm["on_demand"] else vm["bid"] * gpus
        offers = nj.survey(os.environ.get("NEBIUS_BIN", "nebius"), here, tenant, vm["platform"], vm["preset"],
                           not vm["on_demand"])
        order = nj.candidates(offers, cap)
        if not order:
            raise SystemExit(f"{role}: no region offers {gpus}× {vm['platform']} ({vm['preset']}) now"
                             + (f" under its bid (${vm['bid']:g}/GPU-h, ${cap:g}/h)" if cap else "") + ": "
                             + "; ".join(f"{o.region}: {o.skip_reason(cap)}" for o in offers))
        found[role] = {"region": order[0].region, "price": order[0].price, "spot": not vm["on_demand"],
                       "worst_usd": round((cap or order[0].price) * hours, 2),
                       "regions": [[o.region, o.platform, o.price] for o in offers if o.price is not None]}
    if s.budget_file and Path(s.budget_file).expanduser().exists():
        budget = nj.Budget(s.budget_file)
        need = sum(v["worst_usd"] for v in found.values())
        if budget.limit is not None and budget.committed() + need > budget.limit + 1e-9:
            raise SystemExit(f"the ledger {budget.path}: ${budget.committed():.2f} of its ${budget.limit:g} spent or "
                             f"reserved, and the Nebius servers could cost up to ${need:.2f} over {hours:g} h (bid × "
                             "GPUs × hours, as each reserves): raise its limit_usd, or give a smaller --rl-hours")
    return found


def preflight(plan: Plan, s: Settings, est: dict, hours: float = 6.0) -> dict:
    """Everything that can refuse before anything is started: the name prefix, the keys, rsync, the Nebius settings
    (every region named in the regions file), a committed tree; for vast.ai servers hub-gpu's GPUs and room for their
    leases (the leases shortened to fit, at least 2 h), and the offers; for Nebius servers a region under the bid and
    room in the ledger for ``hours``. Returns what it found (prices, the judges' GPU, the leases, Nebius regions)."""
    prefix = os.environ.get("NEBIUS_JOB_NAME_PREFIX", "").strip()
    if prefix and not s.prefix.startswith(prefix):
        raise SystemExit(f"--prefix {s.prefix!r} doesn't start with NEBIUS_JOB_NAME_PREFIX={prefix!r}")
    for k in SECRETS:
        if problem := nj.secret_problem(nj.parse_secret(k).path):
            raise SystemExit(f"the key {k} is {problem}: the owner puts it there with `hub-secret set {k}`")
    if shutil.which(s.rsync) is None:
        raise SystemExit(f"rsync ({s.rsync}) not found: --rsync or RSYNC_BIN (on hub: ~/.local/bin/rsync)")
    for k in ("NEBIUS_REGIONS_FILE", "NEBIUS_SSH_KEY"):
        if not os.environ.get(k):
            raise SystemExit(f"{k} is not set: run with bin/gpu-job's settings (docs/GPU_RUNBOOK.md, v2 probe rows)")
    check_settings(s)
    roles = roles_of(plan)
    nebius_roles = [r for r in roles if provider_of(s, r) == "nebius"]
    known = {r.name for r in nj.load_regions(os.environ["NEBIUS_REGIONS_FILE"])[1]}
    for flag, value in (("--rl-regions", s.rl_regions), *((f"--{r}-regions", nebius_machine(s, r)["regions"])
                                                           for r in nebius_roles)):
        if unknown := sorted(set(region_names(value)) - known):
            raise SystemExit(f"{flag}: {unknown} are not in {os.environ['NEBIUS_REGIONS_FILE']}")
    code = nj.code_version(REPO)
    if problem := nj.code_problem(code):
        raise SystemExit(f"{REPO}: {problem}: every machine gets the committed code, and the probes are pinned to it")
    found = {"code": code}
    if nebius_roles:
        found["nebius"] = nebius_offers(s, nebius_roles, hours)
    roles = [r for r in roles if provider_of(s, r) == "vast"]
    if not roles:
        return found
    hub = nj.HubGpu(s.hub_gpu)
    lim = hub.limits()
    mine = lim.get("limits") or {}
    gpus = (s.probe_gpus if "probe" in roles else 0) + (s.judge_gpus if "judges" in roles else 0)
    free = int(mine.get("gpus") or 0) - int(lim.get("gpus_in_use") or 0)
    if gpus > free:
        raise SystemExit(f"hub-gpu limits: the batch needs {gpus} GPUs at once, {free} are free of the project's "
                         f"{mine.get('gpus')} ({lim.get('gpus_in_use')} in use): wait for the others to end, or ask "
                         "the owner for more")
    prices = {}
    if "probe" in roles:
        offers = nj.vast_candidates(hub.offers(gpu=s.probe_gpu, gpus=s.probe_gpus, disk_gb=nj.disk_gb(s.probe_disk_gib),
                                               max_price=s.probe_max_price), nj.VAST_MIN_CUDA, s.probe_max_price)
        if not offers:
            raise SystemExit(f"no vast.ai offer of {s.probe_gpus}× {s.probe_gpu} now (CUDA ≥ 13, --probe-max-price)")
        prices["probe"] = float(offers[0]["usd_per_hour"])
    if "judges" in roles:
        name, offer = judge_gpu(hub, s)
        if offer is None:
            raise SystemExit(f"no vast.ai offer of {s.judge_gpus}× any of {s.judge_gpus_names} now")
        found["judge_gpu"], prices["judges"] = name, float(offer["usd_per_hour"])
    hourly, room = sum(prices.values()), float(lim.get("room_usd") or 0)
    lease = s.lease_hours
    if hourly * lease > room:
        lease = math.floor(room / hourly * 4) / 4
        if lease < 2:
            raise SystemExit(f"hub-gpu limits: ${room:.2f} left in any 24 hours, ${hourly:.2f}/h for the servers: not "
                             "even 2 h of a lease (the owner raises usd_per_day in Project options)")
        log(f"lease shortened to {lease:g} h to fit the ${room:.2f} hub-gpu leaves in any 24 hours")
    total = hourly * (est["batch_hours"] + 1)
    found.update(prices=prices, hourly=round(hourly, 2), lease_hours=lease, room_usd=room,
                 vast_total_usd=round(total, 2), usd_per_day=mine.get("usd_per_day"))
    if total > float(mine.get("usd_per_day") or 0):
        log(f"WARNING: the batch's servers cost about ${total:,.0f} over ~{est['batch_hours'] + 1:.0f} h, above "
            f"hub-gpu's ${mine.get('usd_per_day')} a day: each lease extension needs room in the 24 hours before it, "
            "so the batch stops when the room runs out unless the owner raises the daily limit")
    return found


def machine_from_record(role: str, provider: str, out: Path) -> dict:
    rec = machine_record(out)
    m = {"role": role, "provider": provider, "out": str(out), "name": rec.get("name"),
         "instance": rec.get("instance_id"), "host": rec.get("host"), "kept": rec.get("kept"),
         "created": (rec.get("times") or {}).get("create_start"), "hours": rec.get("hours"),
         "code": (rec.get("code") or {}).get("commit"), "errors": rec.get("errors")}
    if provider == "vast":
        m["port"] = ((rec.get("vast") or {}).get("ssh") or {}).get("port")
        m["ends"] = (rec.get("vast") or {}).get("ends_utc")
    else:  # where it is (its project: the Nebius CLI finds its state there) and what it is
        flags = rec.get("flags") or {}
        m.update(region=rec.get("region"), spot=rec.get("spot"), platform=rec.get("platform") or flags.get("platform"),
                 preset=flags.get("preset"), teardown_verified=rec.get("teardown_verified"),
                 project=project_of(rec.get("region"), os.environ.get("NEBIUS_REGIONS_FILE"),
                                    flags.get("regions_file")))
    return m


def backoff_seconds(s: Settings, k: int) -> float:
    """The wait before a role's k-th new Nebius VM: ``replace_backoff_minutes`` × 2^(k-1), at most 30 minutes."""
    return min(1800.0, s.replace_backoff_minutes * 60 * 2 ** (k - 1))


def end_failed_up(batch: Batch, role: str, m: dict) -> None:
    """What a failed Nebius ``up`` may have left (it deletes its VM and verifies, unless it died first): ended by name
    (``nebius_job.py down``, verified; a VM that never was is nothing to end)."""
    if not m.get("name") or m.get("teardown_verified") is True:
        return
    s = batch.settings
    code = launch(["down", "--name", m["name"], *(["--budget-file", s.budget_file] if s.budget_file else [])],
                  batch.out / "launch" / f"{role}-down-{m['name']}.log")
    if code != 0:
        batch.data.setdefault("retired", []).append({**m, "retired": utc(), "why": "a failed up"})
    batch.event(f"{role}: whatever {m['name']} left " + (
        "is gone, verified" if code == 0 else f"is NOT verified gone (exit {code}): ended again at the end"))


def machines_up(batch: Batch, plan: Plan, s: Settings, found: dict, rl_hours: float) -> None:
    """The ``up``s at once (the RL machine and a Nebius server for ``rl_hours``, a vast.ai server for its lease), each
    logging to DIR/launch/<role>.log; their records into batch.json. A Nebius server whose ``up`` failed (preempted
    while it booted, no capacity) gets a new VM after a backoff, up to ``max_replacements``; any other that failed:
    InfraError (the launcher has ended what it made; the driver ends the others)."""
    roles = ["rl", *roles_of(plan)]
    procs, codes, later = {}, {}, {}

    def up(role: str, k: int = 0) -> None:
        name = role if k == 0 else f"{role}-r{k}"
        out = batch.out / "launch" / name
        args = up_args(role, s, out, rl_hours, plan, gpu=s.probe_gpu if role == "probe" else found.get("judge_gpu"),
                       lease=found.get("lease_hours"))
        batch.machines[role] = {"role": role, "provider": provider_of(s, role), "out": str(out), "starting": utc(),
                                **({"replacements": k} if k else {})}
        procs[role] = launch(args, batch.out / "launch" / f"{name}.log", wait=False)
        batch.machines[role]["pid"] = getattr(procs[role], "pid", None)  # if the start dies, end_all stops it
        batch.save()
        log(f"{role}: nebius_job.py up started (log: launch/{name}.log)")

    for role in roles:
        up(role)
    while len(codes) < len(roles):
        for role in roles:
            if role in codes or (role in later and time.time() < later[role][1]):
                continue
            if role in later:
                up(role, later.pop(role)[0])
                continue
            proc = procs[role]
            if proc.poll() is None:
                continue
            m = batch.machines[role] = {**batch.machines[role], **machine_from_record(
                role, batch.machines[role]["provider"], Path(batch.machines[role]["out"]))}
            code, log_name = proc.returncode, Path(m["out"]).name
            batch.event(f"{role}: up exited {code}" + (f", {m['name']} kept" if code == 0 else
                                                       "; see launch/" + log_name + ".log"))
            k = int(m.get("replacements", 0)) + 1
            if (code != 0 or not m.get("kept")) and role != "rl" and m["provider"] == "nebius" \
                    and s.replace and k <= s.max_replacements:
                end_failed_up(batch, role, m)
                later[role] = (k, time.time() + backoff_seconds(s, k))
                batch.event(f"{role}: a new VM in {backoff_seconds(s, k) / 60:g} min (new VM {k} of at most "
                            f"{s.max_replacements})")
                continue
            codes[role] = code
        time.sleep(5)
    bad = [r for r, c in codes.items() if c != 0 or not batch.machines[r].get("kept")]
    if bad:
        raise nj.InfraError(f"up failed for {', '.join(bad)}")
    commits = {r: batch.machines[r].get("code") for r in roles}
    if len(set(commits.values())) != 1:
        raise nj.InfraError(f"the machines got different code: {commits}")


def wire(batch: Batch, ops: Ops, plan: Plan, roles: list[str] | None = None, timeout: float = 600) -> None:
    """The tunnels: the batch's key on the RL machine, its public key on each server's machine (restricted to that
    server's ports), a loop each; then wait until every server answers through them."""
    rl = ops.rl()
    pub = batch.data.get("tunnel_pub") or make_tunnel_key(rl, plan.roster.name)
    batch.data["tunnel_pub"] = pub
    for role in roles or roles_of(plan):
        remote = ops.server(role, fresh=True)
        ports = [PROBE_PORT] if role.startswith("probe") else list(JUDGE_PORTS)
        authorize(remote, authorized_line(pub, ports))
        m = batch.machines[role]
        start_tunnel(rl, role, remote.host, remote.port, tunnel_forwards(role), str(m.get("instance")))
        batch.event(f"tunnel {role}: RL machine 127.0.0.1:{','.join(map(str, ports))} → {remote.host} -p "
                    f"{remote.port or 22}")
    t0 = time.time()
    while True:
        bad = {r: why for r, why in through_tunnels(ops, plan.probes, roles or roles_of(plan)).items() if why}
        if not bad:
            batch.event("every server answers through its tunnel on the RL machine")
            return
        if time.time() - t0 > timeout:
            raise nj.InfraError(f"through the tunnels: {bad}")
        time.sleep(15)


def restart_server(batch: Batch, ops: Ops, plan: Plan, role: str, new_instance: bool = False) -> bool:
    """``role``'s server started again on its machine (``nebius_job.py serve``: what runs as its step is stopped
    first; the probe folders checked against their pins first), or on a new machine: a vast.ai instance (``up``, on the
    first of ``judge_gpus_names`` with an offer, each asked while it waits), its tunnel moved there and the old one
    ended; a Nebius VM through ``begin_replacement``, waited for here. True if healthy after."""
    s, m = batch.settings, batch.machines[role]
    if m.get("provider") == "nebius" and (new_instance or not m.get("host")):
        begin_replacement(batch, role, "by hand", by_hand=True)
        while (done := advance_replacement(batch, ops, plan, role)) is None:
            time.sleep(15)
        return done
    k = int(m.get("restarts", 0)) + 1
    probe = role.startswith("probe")
    command = server_command(role, s, plan, rehearsal=False)
    if new_instance:
        out = batch.out / "launch" / f"{role}-r{k}"
        gpu = s.probe_gpu if probe else s.judge_gpus_names
        code = launch(vast_up_args(role, s, out, s.lease_hours, gpu, dataclasses.replace(plan, rehearsal=False),
                                   plan.roster.name), batch.out / "launch" / f"{role}-r{k}.log")
        old = dict(m)
        if code != 0:
            batch.event(f"!!! {role}: a new instance failed (exit {code}; launch/{role}-r{k}.log)")
            return False
        batch.machines[role] = {**machine_from_record(role, "vast", out), "restarts": k,
                                "replacements": int(m.get("replacements", 0)) + 1}
        batch.save()
        ops.forget(role)
        wire(batch, ops, plan, [role])
        launch(["down", "--provider", "vast", "--name", old["name"], "--hub-gpu", s.hub_gpu,
                *(["--budget-file", s.budget_file] if s.budget_file else [])],
               batch.out / "launch" / f"{role}-down-old.log")
        batch.event(f"{role}: moved to a new instance {batch.machines[role]['name']}; the old one ({old['name']}) ended")
        return True
    out = batch.out / "launch" / f"{role}-serve-{k}"
    if m.get("provider") == "nebius":
        key = batch.data.get("ssh_key") or os.environ.get("NEBIUS_SSH_KEY")
        where = ["serve", "--host", m["host"], *(["--ssh-key", key] if key else []), "--free-gpus"]
    else:
        where = ["serve", "--provider", "vast", "--name", m["name"]]
    code = launch([*where, "--out", str(out), *(["--hub-gpu", s.hub_gpu] if m.get("provider") != "nebius" else []),
                   "--health", health_command(s, role), "--health-timeout", "3600", "--", command],
                  batch.out / "launch" / f"{role}-serve-{k}.log")
    m["restarts"] = k
    batch.event(f"{role}: server restarted on {m['name']} (exit {code}; launch/{role}-serve-{k}.log)")
    return code == 0


# --------------------------------------------------------------------------- Nebius server VMs: new ones

# A Nebius server VM that is preempted (stopped: on-preemption stop), stopped at its shutdown timer or deleted is ended
# (``retire``: a stopped VM still holds its disk and its region's public IP) and a new one made, every time, after a
# backoff, up to max_replacements per role. The new VM's ``up`` runs on its own (start_new_session) while the watch goes
# on: batch.json records it (its --out, its pid), so a driver started later, or the watchdog, takes it over.


def batch_deadline(batch: Batch) -> float | None:
    """When the batch's machines stop at the latest: the RL machine's shutdown cap (its creation + ``--hours``)."""
    m = batch.machines.get("rl") or {}
    if m.get("created") and m.get("hours"):
        return nj._seconds(m["created"]) + float(m["hours"]) * 3600
    return None


def at_its_cap(m: dict, now: float | None = None) -> bool:
    """Whether a Nebius VM that stopped did so at its own ``--hours`` shutdown (not a preemption): the batch's end."""
    now = time.time() if now is None else now
    if not (m.get("created") and m.get("hours")):
        return False
    return now > nj._seconds(m["created"]) + float(m["hours"]) * 3600 - 300


def retire(batch: Batch, role: str, why: str) -> None:
    """``role``'s VM ended (``nebius_job.py down``, verified, its ledger entry settled) and kept in batch.json's
    ``retired`` list; ``end_all`` ends it again if that failed."""
    m = batch.machines[role]
    if not m.get("name"):
        return
    if not m.get("ended"):
        s = batch.settings
        code = launch(["down", "--name", m["name"], *(["--budget-file", s.budget_file] if s.budget_file else [])],
                      batch.out / "launch" / f"{role}-down-{m['name']}.log")
        if code == 0:
            m["ended"] = utc()
        batch.event(f"{role}: {m['name']} " + ("ended, verified" if code == 0 else
                                               f"NOT verified ended (exit {code}): ended again at the end"))
    batch.data.setdefault("retired", []).append({**m, "retired": utc(), "why": why})


def begin_replacement(batch: Batch, role: str, why: str, by_hand: bool = False) -> None:
    """``role``'s Nebius VM gone (``why``), or to be moved by hand: it ends (``retire``), and a new VM is due after
    ``backoff_seconds`` (at once ``by_hand``), the k-th of at most ``max_replacements`` (by hand: one more, whatever
    the count). ``advance_replacement`` makes it."""
    s, m = batch.settings, batch.machines[role]
    k = int(m.get("replacements", 0)) + 1
    retire(batch, role, why)
    slot = {"role": role, "provider": "nebius", "replacements": k - 1}
    batch.data.setdefault("bad_since", {}).pop(f"{role}-server", None)
    if not by_hand and (not s.replace or k > s.max_replacements):
        batch.machines[role] = {**slot, "abandoned": why}
        batch.event(f"!!! {role}: no new VM ({'--no-replace' if not s.replace else f'{k - 1} new ones already'}); the "
                    f"runs wait for it (restart-server {role} --new-instance --out {batch.out} makes one)")
        return
    wait = 0.0 if by_hand else backoff_seconds(s, k)
    batch.machines[role] = {**slot, "replacing": {"k": k, "why": why, "not_before": time.time() + wait}}
    batch.event(f"{role}: a new VM in {wait / 60:g} min (new VM {k}" + ("" if by_hand else
                                                                     f" of at most {s.max_replacements}") + f"): {why}")


def launcher_running(pid, out) -> bool:
    """Is ``pid`` a live nebius_job.py with ``out`` in its command line (an ``up`` an earlier driver started)?"""
    try:
        cmd = Path(f"/proc/{int(pid)}/cmdline").read_bytes().split(b"\0")
    except (OSError, TypeError, ValueError):
        return False
    return not pid_ended(int(pid)) and any(a.endswith(b"nebius_job.py") for a in cmd) and str(out).encode() in cmd


def up_exited(ops: Ops, role: str, r: dict) -> int | None:
    """The exit code of the ``up`` making ``role``'s new VM (None: it still runs). One started by an earlier driver is
    read from its job.json once it is gone: 0 if it kept the VM, else 2."""
    proc = ops.procs.get(role)
    if proc is not None:
        return proc.poll()
    if launcher_running(r.get("pid"), r.get("out")):
        return None
    return 0 if machine_record(Path(r["out"])).get("kept") else 2


def advance_replacement(batch: Batch, ops: Ops, plan: Plan, role: str) -> bool | None:
    """One step of ``role``'s new Nebius VM (``begin_replacement``): once its backoff is over, its ``up`` started for
    the hours left of the batch's cap (none if under half an hour is left); once that has exited, the VM taken (its
    tunnel moved there) or, if it failed, what it left ended and the next one due. True once the new VM serves, False
    if given up, None meanwhile."""
    s, m = batch.settings, batch.machines[role]
    r, now = m["replacing"], time.time()
    k, name = int(r["k"]), f"{role}-r{r['k']}"
    if "out" not in r:
        if now < float(r["not_before"]):
            return None
        deadline = batch_deadline(batch)
        left = (deadline - now) / 3600 if deadline else float(batch.data.get("hours") or s.lease_hours)
        if left < 0.5:
            batch.machines[role] = {**m, "replacing": None, "abandoned": f"{r['why']}; the batch's cap is near"}
            batch.event(f"!!! {role}: no new VM: the batch's cap is {left * 60:.0f} min away")
            return False
        hours = max(1.0, math.ceil(left * 4) / 4)
        out = batch.out / "launch" / name
        ops.procs[role] = proc = launch(nebius_up_args(role, s, out, hours, dataclasses.replace(plan, rehearsal=False)),
                                        batch.out / "launch" / f"{name}.log", wait=False)
        r.update(out=str(out), pid=getattr(proc, "pid", None), started=utc(), hours=hours)
        batch.event(f"{role}: new VM {k}: nebius_job.py up started for {hours:g} h (log: launch/{name}.log)")
        return None
    code = up_exited(ops, role, r)
    if code is None:
        return None
    ops.procs.pop(role, None)
    rec = machine_from_record(role, "nebius", Path(r["out"]))
    if code == 0 and rec.get("kept"):
        batch.machines[role] = {**rec, "replacements": k}
        ops.forget(role)
        batch.event(f"{role}: moved to a new VM {rec['name']} in {rec.get('region')} (new VM {k}, "
                    f"{(now - nj._seconds(r['started'])) / 60:.0f} min after its up started)")
        if (want := (batch.data.get("code") or {}).get("commit")) and rec.get("code") and rec["code"] != want:
            batch.event(f"!!! {role}: the new VM got commit {rec['code'][:12]}, not the batch's {want[:12]} (a commit "
                        "since the start; its probe folders were checked against their pins all the same)")
        try:
            wire(batch, ops, plan, [role])
        except Exception as e:  # noqa: BLE001 — the health rounds repair a tunnel
            batch.event(f"!!! tunnel {role}: no answer through it yet ({type(e).__name__}: {e}); the health rounds "
                        "repair it")
        return True
    batch.event(f"!!! {role}: new VM {k}: up exited {code} (launch/{name}.log)")
    end_failed_up(batch, role, rec)
    batch.machines[role] = {"role": role, "provider": "nebius", "replacements": k}
    if not s.replace or k + 1 > s.max_replacements:
        batch.machines[role]["abandoned"] = f"{r['why']}; {k} new VMs failed or went"
        batch.event(f"!!! {role}: no new VM after {k}; the runs wait for it (restart-server {role} --new-instance "
                    f"--out {batch.out} makes one)")
        return False
    batch.machines[role]["replacing"] = {"k": k + 1, "why": r["why"], "not_before": now + backoff_seconds(s, k + 1)}
    batch.event(f"{role}: a new VM in {backoff_seconds(s, k + 1) / 60:g} min (new VM {k + 1} of at most "
                f"{s.max_replacements})")
    return None


def replacements_round(batch: Batch, ops: Ops, plan: Plan) -> None:
    """``advance_replacement`` for every role waiting for a new Nebius VM."""
    for role in roles_of(plan):
        if (batch.machines.get(role) or {}).get("replacing"):
            advance_replacement(batch, ops, plan, role)


def nebius_gone(ops: Ops, role: str) -> str | None:
    """Why ``role``'s Nebius VM can't serve (None: it can): its state (STOPPED after a preemption or its shutdown
    timer, …) or not listed (deleted). InfraError if the Nebius CLI can't tell."""
    inst = ops.vm(role)
    if inst is None:
        return "not listed in its project (deleted)"
    return None if inst.state in nj.ALIVE_STATES else inst.state


def stop_pending_up(batch: Batch, ops: Ops | None, role: str, grace: float = 900) -> None:
    """An ``up`` still running as the batch ends (a new Nebius VM's, or one of a start that died): SIGTERM to it (its
    session: the launcher deletes what it made, verified), waited for; the machine its job.json names is then ended
    with the rest (``end_all``)."""
    m = batch.machines[role]
    r = m.get("replacing") or ({"out": m["out"], "pid": m.get("pid")} if m.get("out") and not m.get("name") else {})
    if not r.get("out"):
        return
    proc = ops.procs.get(role) if ops is not None else None

    def running() -> bool:
        return proc.poll() is None if proc is not None else launcher_running(r.get("pid"), r["out"])

    if running():
        log(f"{role}: stopping the up of its new VM (pid {r.get('pid')}): it deletes what it made")
        with contextlib.suppress(ProcessLookupError, PermissionError, TypeError, ValueError):
            os.killpg(int(r["pid"]), signal.SIGTERM)
        t0 = time.time()
        while running() and time.time() - t0 < grace:
            time.sleep(5)
    if name := machine_record(Path(r["out"])).get("name"):
        m["name"] = name


def end_all(batch: Batch, ops: Ops | None, plan: Plan | None, why: str) -> bool:
    """The end: the supervisor told to stop (if it still runs), a last sync, the VM's logs, then every machine ended and
    verified gone (``nebius_job.py down``): a new VM still being made too (its ``up`` stopped first), and a replaced
    one whose end could not be verified then. True if every machine is verified gone; then CURRENT_BATCH no longer
    names the batch."""
    batch.event(f"ending the batch: {why}")
    batch.set_phase("ending")
    s = batch.settings
    if ops is not None and batch.machines.get("rl", {}).get("host") and not batch.machines["rl"].get("ended"):
        with contextlib.suppress(Exception):
            if job_exit(ops) is None and (batch.data.get("job") or {}).get("step"):
                control(ops, "stop-all")
                t0 = time.time()
                while job_exit(ops) is None and time.time() - t0 < 180:
                    time.sleep(10)
        if plan is not None:
            with contextlib.suppress(Exception):
                log(f"last sync: {sync(ops, [r.row.run_name for r in plan.runs])}")
        with contextlib.suppress(Exception):
            nj.fetch_rsync(ops.rl(), f"~/{nj.REMOTE_LOGS}", batch.out / "vm", s.rsync, timeout=600)
            nj.fetch_rsync(ops.rl(), f"~/{TUNNEL_DIR}", batch.out / "vm", s.rsync, timeout=600)
    ok = True
    for role in list(batch.machines):
        stop_pending_up(batch, ops, role)
    retired = [(m.get("role", "retired"), m) for m in batch.data.get("retired", []) if m.get("name")]
    for role, m in [*batch.machines.items(), *retired]:
        if m.get("ended") or not m.get("name"):
            continue
        args = (["down", "--provider", "vast", "--name", m["name"], "--hub-gpu", s.hub_gpu] if m["provider"] == "vast"
                else ["down", "--name", m["name"]])
        code = launch([*args, *(["--budget-file", s.budget_file] if s.budget_file else [])],
                      batch.out / "launch" / f"{role}-down.log")
        if code == 0:
            m["ended"] = utc()
            if m["provider"] == "vast" and ops is not None:
                with contextlib.suppress(Exception):
                    row = ops.row(role)
                    m["cost_usd"] = nj.vast_cost(row) if row else None
        else:
            ok = False
        batch.event(f"{role}: {m['name']} " + ("ended, verified" if code == 0 else
                                               f"NOT verified ended (exit {code}): see launch/{role}-down.log"))
    batch.set_phase("ended" if ok else "end-failed")
    if ok:
        release_pointer(batch)
    else:
        log("!!! a machine may still bill: python3 scripts/rl_batch.py down --out " + str(batch.out))
    return ok


def check_pointer(pointer: Path, out: Path) -> None:
    """SystemExit if CURRENT_BATCH (``pointer``) names another batch than ``out`` that has not ended: the watchdog's
    scheduled job watches the one it names."""
    with contextlib.suppress(OSError):
        other = Path((pointer.read_text().splitlines() or [""])[0].strip())
        if str(other) not in ("", ".") and other != out.resolve() and (other / "batch.json").exists() and \
                (phase := Batch(other).phase) not in ("ended", "new"):
            raise SystemExit(f"{pointer} names {other}, a batch that has not ended ({phase}): end it (rl_batch.py down "
                             f"--out {other}), or give this one --current-batch FILE")


def claim_pointer(batch: Batch, pointer: Path) -> None:
    """CURRENT_BATCH (``pointer``, beside the batch folders: output/gpu_jobs/), which the watchdog's scheduled job
    reads: this batch's folder, absolute, from its start until every machine of it is verified gone."""
    check_pointer(pointer, batch.out)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(f"{batch.out.resolve()}\n")
    batch.data["pointer"] = str(pointer.resolve())
    batch.event(f"{pointer} names this batch (the watchdog's)")


def release_pointer(batch: Batch) -> None:
    """CURRENT_BATCH removed, if it still names this batch."""
    pointer = Path(batch.data["pointer"]) if batch.data.get("pointer") else None
    with contextlib.suppress(OSError):
        if pointer and (pointer.read_text().splitlines() or [""])[0].strip() == str(batch.out.resolve()):
            pointer.unlink()
            log(f"{pointer} removed: the batch has ended")


def requests(batch: Batch) -> list[str]:
    """Requests a manual command left for the driver (DIR/requests/<what>), taken."""
    folder = batch.out / "requests"
    out = []
    for f in sorted(folder.glob("*")) if folder.is_dir() else []:
        out.append(f.name)
        f.unlink(missing_ok=True)
    return out


def watch(batch: Batch, ops: Ops, plan: Plan, once: bool = False) -> str:
    """The batch until it is over: every minute the supervisor's status; every ``sync_minutes`` a sync; every
    ``health_minutes`` the servers and tunnels (and their repair); every 10 minutes the leases. Returns why it ended:
    "done", "failed", "rl-gone" or, with ``once``, "watching". An error in a round (an SSH hiccup, a refused call) is
    logged and the next round goes on: only those three end the batch."""
    last = {"sync": time.time() if not once else 0, "health": 0.0, "lease": 0.0, "say": 0.0}
    errors = 0
    while True:
        try:
            why = watch_round(batch, ops, plan, last)
            errors = 0
        except (Detach, KeyboardInterrupt):
            raise
        except Exception as e:  # noqa: BLE001 — the watch outlives a bad round
            import traceback

            errors += 1
            batch.event(f"!!! watch round failed ({errors} in a row): {type(e).__name__}: {e}",
                        traceback=traceback.format_exc()[-1500:])
            why = None
        if why is not None:
            return why
        if once:
            return "watching"
        time.sleep(60)


def watch_round(batch: Batch, ops: Ops, plan: Plan, last: dict) -> str | None:
    """One round of ``watch``: why the batch is over, or None."""
    s = batch.settings
    roles = roles_of(plan)
    runs = [r.row.run_name for r in plan.runs]
    bad_since: dict = batch.data.setdefault("bad_since", {})
    now = time.time()
    for what in requests(batch):
        log(f"request: {what}")
        if what.startswith("restart-server-"):
            restart_server(batch, ops, plan, what[len("restart-server-"):], new_instance=False)
        elif what.startswith("replace-server-"):
            role = what[len("replace-server-"):]
            if (batch.machines.get(role) or {}).get("provider") == "nebius":  # made while the watch goes on
                begin_replacement(batch, role, "by hand", by_hand=True)
            else:
                restart_server(batch, ops, plan, role, new_instance=True)
        elif what == "sync":
            last["sync"] = 0
        elif what == "extend":
            last["lease"] = 0
    status = vm_status(ops)
    exit_code = job_exit(ops)
    if status is None and exit_code is None:
        bad_since.setdefault("rl", now)
        if now - bad_since["rl"] > s.vm_dead_minutes * 60:
            batch.event(f"!!! the RL machine has been unreachable for {s.vm_dead_minutes:g} min")
            return "rl-gone"
    else:
        bad_since.pop("rl", None)
        batch.data["vm_status"] = status
    if status and status.get("phase") == "running":
        stalled(batch, status, now)
    if exit_code is not None:
        batch.data.setdefault("job", {})["exit"] = exit_code
        phase = (status or {}).get("phase")
        if phase in ("done", "failed", "refused") or exit_code == 0:
            batch.event(f"the supervisor exited {exit_code} ({phase})")
            return "done" if exit_code == 0 else "failed"
        starts = int((batch.data.get("job") or {}).get("starts", 1))
        if starts >= 4:
            batch.event(f"!!! the supervisor exited {exit_code} {starts} times")
            return "failed"
        batch.event(f"the supervisor exited {exit_code} with runs unfinished: starting it again (it takes over the "
                    "runs still running)")
        start_job(ops, batch.out / "plan.json", starts + 1)
    if now - last["sync"] >= s.sync_minutes * 60:
        last["sync"] = now
        done = sync(ops, runs)
        log("sync: " + ", ".join(f"{p.rsplit('/', 1)[-1]}={v if len(str(v)) < 40 else 'error'}"
                                 for p, v in done.items()))
    replacements_round(batch, ops, plan)
    if now - last["health"] >= s.health_minutes * 60:
        last["health"] = now
        health_round(batch, ops, plan, roles, bad_since)
    if now - last["lease"] >= 600:
        last["lease"] = now
        extend_leases(ops)
    if now - last.setdefault("price", 0.0) >= nj.PRICE_RECHECK_SECONDS:
        last["price"] = now
        sample_prices(batch)
    if now - last["say"] >= 900 and status:
        last["say"] = now
        log(status_line(status))
    batch.save()
    return None


def stalled(batch: Batch, status: dict, now: float) -> bool:
    """Whether no run has finished a step or an eval for ``stall_minutes``, said loudly once per such stretch with each
    running run's last retry warning. Health checks can't see every stall: a probe server whose GPU worker is wedged,
    or an episode that fails on every attempt (an out of memory: 503, retried forever), still answers /v1/meta."""
    runs = status.get("runs") or {}
    progress = sum(int(r.get("steps") or 0) + int(r.get("evals") or 0) for r in runs.values())
    st = batch.data.setdefault("stall", {"progress": progress, "since": now, "warned": 0.0})
    if progress != st["progress"]:
        st.update(progress=progress, since=now, warned=0.0)
        return False
    limit = batch.settings.stall_minutes * 60
    if now - st["since"] < limit:
        return False
    if now - st["warned"] >= limit:
        st["warned"] = now
        batch.event(f"!!! no run has finished a step or an eval for {(now - st['since']) / 60:.0f} min: a stall the "
                    "health checks don't see? Each run's last retry warning follows",
                    waiting={n: r.get("last_warning") for n, r in runs.items() if r.get("state") == "running"})
    return True


def health_round(batch: Batch, ops: Ops, plan: Plan, roles: list[str], bad_since: dict) -> None:
    """Each server on its machine and through its tunnel; repairs. Down through the tunnel two rounds in a row but up
    on its machine: the tunnel (its loop started again, the key authorized again, its ssh killed for a fresh one).
    Down on its machine for ``server_dead_minutes``: the server restarted there, at most ``max_server_restarts`` times.
    Its machine gone: a vast.ai instance replaced (``replace``), once; a Nebius VM preempted, stopped or deleted (its
    state, by the Nebius CLI) ended and a new one made, every time (``begin_replacement``: a backoff, at most
    ``max_replacements``), the runs waiting meanwhile; one stopped at its own ``--hours`` is the batch's end."""
    s = batch.settings
    live = [r for r in roles if not (batch.machines[r].get("replacing") or batch.machines[r].get("abandoned"))]
    seen = through_tunnels(ops, plan.probes, live) if live else {}
    strikes = batch.data.setdefault("tunnel_strikes", {})
    for role in live:
        m = batch.machines[role]
        if m.get("provider") == "nebius":
            try:
                gone = nebius_gone(ops, role)
            except nj.InfraError as e:
                log(f"health {role}: could not read its VM's state ({e})")
                gone = None
            if gone and at_its_cap(m):
                if not m.get("capped"):
                    m["capped"] = utc()
                    batch.event(f"!!! {role}: its VM {m.get('name')} is {gone} at its --hours cap: not replaced")
                continue
            if gone:
                batch.event(f"!!! {role}: its VM {m.get('name')} is {gone}")
                ops.forget(role)
                begin_replacement(batch, role, f"{m.get('name')} {gone}")
                continue
        else:
            try:
                row = ops.row(role)
            except nj.InfraError as e:
                log(f"health {role}: could not list the instances ({e})")
                continue
            gone = nj.vast_gone(row)
            if gone:
                batch.event(f"!!! {role}: its instance {m.get('name')} is {gone}")
                if s.replace and int(m.get("replacements", 0)) < 1:
                    restart_server(batch, ops, plan, role, new_instance=True)
                continue
        try:
            here = server_health(ops, role, plan.probes)
        except (nj.InfraError, nj.VmStopped) as e:
            here = f"no SSH ({e})"
            ops.forget(role)
        key = f"{role}-server"
        if here is None:
            bad_since.pop(key, None)
            if not seen.get(role):
                strikes.pop(role, None)
                continue
            strikes[role] = int(strikes.get(role, 0)) + 1
            if strikes[role] < 2:
                log(f"tunnel {role}: down through the tunnel ({seen[role]}) but up on its machine: repaired if it is "
                    "still so next round")
                continue
            strikes.pop(role)
            batch.event(f"tunnel {role}: down through the tunnel ({seen[role]}) but up on its machine, two rounds in a "
                        "row: repairing")
            with contextlib.suppress(Exception):
                authorize(ops.server(role), authorized_line(batch.data["tunnel_pub"],
                                                            [PROBE_PORT] if role.startswith("probe")
                                                            else list(JUDGE_PORTS)))
                st = tunnel_state(ops.rl(), role)
                remote = ops.server(role)
                if not st.get("loop"):
                    start_tunnel(ops.rl(), role, remote.host, remote.port, tunnel_forwards(role),
                                 str(m.get("instance")))
                else:
                    kick_tunnel(ops.rl(), role)
            continue
        first = bad_since.setdefault(key, time.time())
        log(f"health {role}: {here} (for {(time.time() - first) / 60:.0f} min)")
        if time.time() - first >= s.server_dead_minutes * 60:
            if int(m.get("restarts", 0)) >= s.max_server_restarts:
                batch.event(f"!!! {role}: still down after {m.get('restarts')} restarts; the runs wait for it "
                            f"(restart-server {role} --new-instance moves it)")
                bad_since[key] = time.time()
                continue
            if restart_server(batch, ops, plan, role):
                bad_since.pop(key, None)
            else:
                bad_since[key] = time.time()


def status_line(status: dict) -> str:
    runs = status.get("runs") or {}
    parts = [f"{n}: {r['state']} {r.get('steps', 0)} steps" + (f" ({r['step_s_median']:.0f}s/step)"
                                                              if r.get("step_s_median") else "")
             for n, r in runs.items()]
    return (f"runs ({status.get('phase')}, probe queue {status.get('probe_connections')}): " + "; ".join(parts)
            + (f"; servers down: {status['servers_down']}" if status.get("servers_down") else ""))


# ------------------------------------------------------------------------------------------------ rehearsal


def wait_for(what: str, test, timeout: float, every: float = 15.0):
    """Poll ``test()`` until it returns something truthy (returned) or ``timeout`` passes (None)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        with contextlib.suppress(Exception):
            got = test()
            if got:
                return got
        time.sleep(every)
    log(f"rehearsal: gave up waiting for {what} after {timeout:.0f}s")
    return None


def kill_and_resume(ops: Ops, plan: Plan, put) -> None:
    """The rehearsal's drill: the first run killed (``restart-<run>``) while it runs, once it has a save (the
    rehearsal saves at every step); it must start again with ``--resume`` from that save."""
    victim = next((r.row.run_name for r in plan.runs), None)
    if victim is None:
        return

    def running_with_a_save():
        if ((vm_status(ops) or {}).get("runs") or {}).get(victim, {}).get("state") != "running":
            return None
        rc, out = run_text(ops.rl(), f"ls {nj.REMOTE_REPO}/data/runs/{victim}/resume_states/", 60)
        saves = sorted(w for w in out.split() if w.startswith("step_"))
        return saves[-1] if rc == 0 and saves else None

    newest = wait_for(f"{victim} running with a save", running_with_a_save, 1800, 10)
    if newest:
        control(ops, f"restart-{victim}")
    resumed = bool(newest) and wait_for(f"{victim} resumed", lambda: ((vm_status(ops) or {}).get("runs") or {}).get(
        victim, {}).get("modes", [])[-1:] == ["resume"], 600)
    put("a run killed while it ran, after a save, and resumed from it (--resume)", bool(newest and resumed), run=victim,
        resumed_from=newest)


def preemption_drill(batch: Batch, ops: Ops, plan: Plan, put, timeout: float = 3 * 3600) -> None:
    """The rehearsal's drill on Nebius: a server's VM (the probe's, else the judges') stopped as a spot preemption
    stops it; the health rounds must find it gone, end it and make a new VM, its tunnel moved there, the runs waiting
    meanwhile, not failing."""
    role = next((r for r in roles_of(plan) if batch.machines[r].get("provider") == "nebius"), None)
    if role is None:
        return
    old = dict(batch.machines[role])
    nj.Nebius(os.environ.get("NEBIUS_BIN", "nebius"), old["project"]).stop_instance(old["instance"])
    batch.event(f"rehearsal: {role}'s VM {old['name']} stopped, as a preemption stops it; the runs must wait")
    t0, bad_since = time.time(), batch.data.setdefault("bad_since", {})

    def replaced():
        health_round(batch, ops, plan, [role], bad_since)
        replacements_round(batch, ops, plan)
        m = batch.machines[role]
        if m.get("abandoned"):
            return "abandoned"
        if m.get("name") and m["name"] != old["name"] and not m.get("replacing"):
            return m["name"] if through_tunnels(ops, plan.probes, [role])[role] is None else None
        return None

    new = wait_for(f"a new VM for {role}", replaced, timeout, 60)
    st = vm_status(ops) or {}
    put(f"{role}'s Nebius VM stopped (a simulated preemption), a new one made and serving, the runs waiting",
        bool(new) and new != "abandoned" and not any(r["attempts"] for r in (st.get("runs") or {}).values()),
        old=old["name"], new=new, minutes=round((time.time() - t0) / 60), runs=st.get("runs"))


def rehearse(batch: Batch, ops: Ops, plan: Plan, checks: dict, compare_runs: list[str]) -> dict:
    """The rehearsal's checklist, run while its runs run (results: DIR/rehearsal.json; every item says ok or not).
    The faults go in one at a time: a run killed after a save while it runs (``--resume``), the probe server stopped
    and started again, one judge engine killed (setup_vllm.sh starts it again), the tunnels' ssh killed, a Nebius
    server's VM stopped (a simulated preemption: a new VM). Then the runs must finish with no failed attempt but that
    one restart."""
    rec = {"started": utc(), "items": {}}
    items = rec["items"]

    def put(name: str, ok: bool, **info) -> None:
        items[name] = {"ok": bool(ok), **info}
        (batch.out / "rehearsal.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
        batch.event(f"rehearsal: {name}: {'ok' if ok else 'NOT ok'}", **{k: v for k, v in info.items()
                                                                          if len(str(v)) < 300})

    code = str(checks.get("nobody_reads_keys"))
    put("nobody cannot read the keys on the RL machine", code not in ("0", "?"), exit_code=code)
    runs = [r.row.run_name for r in plan.runs]
    st = wait_for("every run started", lambda: (s := vm_status(ops)) and all(
        (s["runs"][n]["state"] == "running" and s["runs"][n].get("wandb")) or s["runs"][n]["state"] == "done"
        for n in runs) and s, 1800)
    entity, project = plan.roster.wandb_entity, plan.roster.wandb_project
    urls = {n: ((st or {}).get("runs") or {}).get(n, {}).get("wandb") for n in runs}
    put("every run logs to W&B in the roster's entity and project",
        all(u and f"wandb.ai/{entity}/{project}/" in u for u in urls.values()), urls=urls)
    pid = (st or {}).get("supervisor_pid")
    rc, out = run_text(ops.rl(), f"sudo -n -u nobody cat /proc/{int(pid or 0)}/environ > /dev/null 2>&1; echo $?", 60)
    code = out.strip().splitlines()[-1] if out.strip() else "?"
    put("nobody cannot read the supervisor's environment (where the keys are once their files are gone)",
        bool(pid) and code not in ("0", "?"), exit_code=code, pid=pid)
    kill_and_resume(ops, plan, put)  # first: the 2-step runs ended before it when it came last (2026-10-27)
    if "probe" in roles_of(plan):
        iid = next(iter(plan.probes))
        rc, out = run_text(ops.server("probe"), f"cd {nj.REMOTE_REPO} && .venv/bin/python "
                                                 "experiments/check_probe_long_episodes.py --url "
                                                 f"http://127.0.0.1:{PROBE_PORT} --probe {shlex.quote(iid)} "
                                                 "--tokens 8000,12000,16000 --out results/long_episodes.json", 3600)
        rc2, report = run_text(ops.server("probe"), f"cat {nj.REMOTE_REPO}/results/long_episodes.json", 60)
        try:
            long = json_in(report)
        except ValueError:
            long = {"error": out[-1500:]}
        put("synthetic 8k/12k/16k-token episodes on the probe server (16k: out of memory expected)", rc == 0,
            results=long)
        copies = probe_copies(batch.settings)
        gpus = {}
        for sfx in [""] if copies == 1 else [f"-{k}" for k in range(copies)]:
            rc, gpu_report = run_text(ops.server("probe"),
                                      f"cat {nj.REMOTE_REPO}/results/check_probe_server_v2{sfx}.json", 60)
            rc2, gpu_log = run_text(ops.server("probe"),
                                    f"tail -n 40 {nj.REMOTE_REPO}/results/check_probe_server_v2{sfx}.log", 60)
            try:
                gpu = json_in(gpu_report)
            except ValueError:
                gpu = {"ok": False, "error": gpu_report[-500:]}
            gpus[sfx] = ({k: gpu.get(k) for k in ("ok", "throughput", "end_to_end")}, gpu_log[-2000:])
        put("check_probe_server.py on the serving GPUs, before the server"
            + (f" (each of the {copies} copies' GPUs)" if copies > 1 else ""),
            all(g["ok"] is True for g, _ in gpus.values()),
            report=gpus[""][0] if copies == 1 else {f"copy{s}": g for s, (g, _) in gpus.items()},
            log_tail=gpus[""][1] if copies == 1 else {f"copy{s}": t for s, (_, t) in gpus.items()})
        nj.stop_step(ops.server("probe"), nj.SERVE_STEP, free_gpus=True)
        batch.event("rehearsal: the probe server stopped; the runs must wait for it, not fail")
        time.sleep(120)
        ok = restart_server(batch, ops, plan, "probe")
        st = vm_status(ops) or {}
        put("the probe server killed and started again, the runs waiting", ok and not any(
            r["attempts"] for r in (st.get("runs") or {}).values()), runs=st.get("runs"))
    if "judges" in roles_of(plan):
        remote = ops.server("judges")
        rc, out = run_text(remote, "pid=$(ss -Hltnp 'sport = :8004' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2); "
                                   'pg=$(ps -o pgid= -p "$pid" | tr -d " "); echo "engine pid $pid group $pg"; '
                                   'kill -TERM -- "-$pg"', 60)
        batch.event(f"rehearsal: the judge engine on :8004 killed ({out.strip()})")
        back = wait_for("the judge engine on :8004 back", lambda: judges_ok(remote, (8004,)) is None, 1800, 30)
        put("one judge engine killed and started again by setup_vllm.sh", bool(back), kill=out.strip())
    for role in roles_of(plan):
        kick_tunnel(ops.rl(), role)
        back = wait_for(f"the {role} tunnel back", lambda: through_tunnels(ops, plan.probes, [role])[role] is None,
                        300, 10)
        put(f"the {role} tunnel's ssh killed and reconnected", bool(back), state=tunnel_state(ops.rl(), role))
    preemption_drill(batch, ops, plan, put)
    final = wait_for("every rehearsal run to end", lambda: (s := vm_status(ops)) and s.get("phase") in (
        "done", "failed") and s, 7200, 30) or vm_status(ops) or {}
    rs = final.get("runs") or {}
    put("every run finished, no failed attempt (one restart on purpose)", final.get("phase") == "done" and not any(
        r["attempts"] for r in rs.values()), runs={n: {k: r.get(k) for k in ("state", "attempts", "restarts", "modes",
                                                                             "steps", "step_s_median")}
                                                   for n, r in rs.items()})
    put("per-step wall time and the probe queue's depth, every run at once", True,
        step_s={n: r.get("step_s_median") for n, r in rs.items()},
        probe_queue_max=max((q[1] for q in final.get("probe_queue") or []), default=None),
        probe_queue=final.get("probe_queue"))
    if compare_runs:
        rec["compare"] = judge_compare(batch, ops, compare_runs)
        put("judge consistency (rescore_terminal_judges.py compare) against Vlad's live scores",
            rec["compare"].get("exit") == 0, **{k: v for k, v in rec["compare"].items() if k != "report"})
    rec["ended"] = utc()
    (batch.out / "rehearsal.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    return rec


def judge_compare(batch: Batch, ops: Ops, run_dirs: list[str], passes: int = 1, workers: int | None = None) -> dict:
    """experiments/rescore_terminal_judges.py compare on the RL machine, through the tunnels to the batch's judges: the
    given run folders (Vlad's, with eval_rollouts.jsonl) pushed there first, 4 + 4 episodes per (run, eval step). Its
    report comes back into DIR/judge_compare/. With ``passes`` > 1, the same sample is scored again into
    DIR/judge_compare/pass<i>/ (a pass reuses only its own folder's verdicts): the judges' retest spread."""
    if not (REPO / "experiments" / "rescore_terminal_judges.py").is_file():
        return {"exit": None, "skipped": "experiments/rescore_terminal_judges.py is not in this repo yet"}
    rl, names = ops.rl(), []
    for d in run_dirs:
        name = Path(d).name
        names.append(f"{REMOTE_BATCH}/compare/{name}")
        nj.push_path(rl, d, f"{REMOTE_BATCH}/compare/{name}")
    rc, tails = 0, []
    for i in range(1, passes + 1):
        sub = f"{REMOTE_BATCH}/judge_compare" + (f"/pass{i}" if passes > 1 else "")
        rc, out = run_text(rl, f"cd {nj.REMOTE_REPO} && .venv/bin/python experiments/rescore_terminal_judges.py compare "
                               f"{' '.join(map(shlex.quote, names))} --out-dir {sub} --pos 4 --neg 4"
                               + (f" --workers {workers}" if workers else ""), 6 * 3600)
        tails.append(out[-3000:])
        if rc != 0:
            break
    with contextlib.suppress(Exception):
        nj.fetch_rsync(rl, f"{REMOTE_BATCH}/judge_compare", batch.out, batch.settings.rsync, timeout=1800)
    reports = [str(batch.out / "judge_compare" / (f"pass{i}" if passes > 1 else "") / "compare_report.json")
               for i in range(1, passes + 1)]
    return {"exit": rc, "tail": tails[-1], "report": reports[0] if passes == 1 else reports}


def cmd_judge_compare(args) -> int:
    """The judge compare against a running batch's judges (amendment 3: while the batch runs), beside its driver: it
    only pushes the run folders to the RL machine and runs the compare there, through the batch's tunnels."""
    batch = batch_from(args)
    if not batch.machines.get("rl", {}).get("host"):
        raise SystemExit(f"{batch.out}: the batch has no RL machine yet")
    rec = judge_compare(batch, Ops(batch), args.runs, args.passes, args.workers)
    log(f"judge-compare: exit {rec.get('exit')}; reports {rec.get('report')}")
    print(rec.get("tail", ""))
    return 0 if rec.get("exit") == 0 else 1


# ------------------------------------------------------------------------------------------------- commands


def settings_from(args) -> Settings:
    s = Settings()
    for f in dataclasses.fields(Settings):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(s, f.name, v)
    s.rsync = s.rsync if shutil.which(s.rsync) else (os.environ.get("RSYNC_BIN") or next(
        (p for p in (shutil.which("rsync"), str(Path("~/.local/bin/rsync").expanduser())) if p and Path(p).exists()),
        s.rsync))
    s.budget_file = s.budget_file or os.environ.get("NEBIUS_BUDGET_FILE")
    check_settings(s)
    return s


def cap_hours(s: Settings, est: dict) -> float:
    """The batch's cap: every Nebius VM's ``--hours`` (the RL machine's, a Nebius server's; what the ledger reserves),
    ``--rl-hours``, else the estimate × 1.5 + 2 h, 6 to 100."""
    return s.rl_hours or min(100.0, max(6.0, math.ceil(est["batch_hours"] * 1.5 + 2)))


def runs_from(plan: Plan, folder: str | Path, allow_fresh: bool = False) -> dict:
    """``start --runs-from DIR``: each roster run's folder, DIR/<run_name>/, and what it resumes from: its newest
    resume_states/step_*.json (its steps done; a finished run's final one). A run without one (no folder, or a folder
    without a save) is refused unless ``allow_fresh`` (it starts fresh then); a save naming another run (its config's
    run_name or seed) is refused. SystemExit with every problem, before anything is started."""
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise SystemExit(f"--runs-from {folder}: not a folder")
    found, problems = {}, []
    for r in plan.runs:
        name, d = r.row.run_name, folder / r.row.run_name
        saves = sorted((d / "resume_states").glob("step_*.json")) if d.is_dir() else []
        if not saves:
            why = "no folder" if not d.is_dir() else "no resume_states/step_*.json in its folder"
            if allow_fresh:
                found[name] = {"fresh": True, "why": why}
            else:
                problems.append(f"{name}: {why}")
            continue
        try:
            state = json.loads(saves[-1].read_text())
            steps = int(state["steps_done"])
        except (OSError, ValueError, KeyError, TypeError) as e:
            problems.append(f"{name}: {saves[-1]} unreadable ({type(e).__name__}: {e})")
            continue
        cfg = state.get("config") or {}
        if off := [f"its {k} is {cfg[k]!r}" for k, want in (("run_name", name), ("seed", r.row.seed))
                   if k in cfg and cfg[k] != want]:
            problems.append(f"{name}: {saves[-1]} is another run's ({', '.join(off)})")
            continue
        found[name] = {"from": str(d), "resume_state": saves[-1].name, "steps_done": steps,
                       "eval_done": bool(state.get("eval_done")), "done": run_done(d)}
    if problems:
        raise SystemExit(f"--runs-from {folder}: " + "; ".join(problems)
                         + ("" if allow_fresh else " (--allow-fresh starts a run without a save fresh)"))
    for name, v in found.items():
        log(f"--runs-from: {name} " + (f"starts fresh ({v['why']})" if v.get("fresh") else
                                       f"is done ({v['resume_state']})" if v["done"] else
                                       f"resumes from {v['resume_state']} ({v['steps_done']} steps done)"))
    return found


def push_runs(batch: Batch, ops: Ops, found: dict) -> None:
    """Each ``runs_from`` folder onto the RL machine as data/runs/<run_name>/, before the supervisor starts (it resumes
    a run with a save from its newest, and takes a finished one as done); each checked there, InfraError if not."""
    rl = ops.rl()
    for name, v in found.items():
        if v.get("fresh"):
            continue
        nj.push_path(rl, v["from"], f"data/runs/{name}")
        rc, out = run_text(rl, f"ls {nj.REMOTE_REPO}/data/runs/{shlex.quote(name)}/resume_states/", 60)
        if rc != 0 or v["resume_state"] not in out.split():
            raise nj.InfraError(f"--runs-from: {name}'s {v['resume_state']} is not on the RL machine after its push "
                                f"({rc}: {out[-300:]})")
        batch.event(f"--runs-from: {name} on the RL machine; it " + (
            "is done" if v["done"] else f"resumes from {v['resume_state']} ({v['steps_done']} steps done)"))


def print_plan(plan: Plan, est: dict, found: dict | None, s: Settings) -> None:
    roster = plan.roster
    print(f"== roster {roster.name} ({roster.path}): {len(plan.runs)} runs, n_steps {roster.n_steps}, W&B "
          f"{roster.wandb_entity}/{roster.wandb_project}" + (" (REHEARSAL)" if plan.rehearsal else ""))
    for r in plan.runs:
        print(f"  {r.row.run_name:30} seed {r.row.seed:5}  early stop {r.row.stop_after_zero_behavior_steps}  "
              f"{r.row.config}")
        print(f"    {shlex.join(r.argv)}")
    print("== pinned probes (sha256 of probe.npz + meta.json, as committed)")
    for p, sha in plan.probes.items():
        print(f"  {p}: {sha}")
    print(f"== estimate: the probe server busy {est['probe_server_hours']} h; the batch ~{est['batch_hours']} h after "
          f"its ~1 h of setup; judge calls {est['judge_calls']} (~{est['judge_calls_per_hour']} per hour: "
          f"{est['judge_load']} of what Vlad's q3 GPU and his three q35 engines each served)")
    if found and found.get("prices"):
        print(f"== vast.ai now: {found['prices']} $/h ({found.get('judge_gpu')} for the judges); ~${found['hourly']}/h "
              f"together, ~${found['vast_total_usd']:,} for the batch; hub-gpu: ${found['room_usd']:.2f} left in any 24 "
              f"hours of ${found.get('usd_per_day')}; leases of {found['lease_hours']:g} h, extended while it runs")
    for role, v in ((found or {}).get("nebius") or {}).items():
        vm = nebius_machine(s, role)
        print(f"== {role} on Nebius: {vm['platform']} ({vm['preset']}), "
              + (f"spot under a bid of ${vm['bid']:g}/GPU-h" if v["spot"] else "on demand")
              + f"; the cheapest now {v['region']}, ${v['price']:.2f}/h (priced: "
              + ", ".join(f"{r} {p} ${price:.2f}" for r, p, price in v["regions"])
              + f"); the ledger reserves up to ${v['worst_usd']:,.2f}"
              + (f"; {probe_copies(s)} reader copies" if role.startswith("probe") else ""))
    print(f"== the RL machine: Nebius cpu-d3/{s.rl_preset} on demand (~$1.06/h for 32vcpu-128gb)"
          + (f", in one of {s.rl_regions}" if s.rl_regions else ""))


def batch_from(args) -> Batch:
    out = Path(args.out).expanduser()
    if not (out / "batch.json").exists():
        raise SystemExit(f"no batch in {out} (batch.json)")
    batch = Batch(out)
    Driver.log_file = out / "driver.log"
    use_prefix(batch.settings)
    return batch


def install_driver_signals() -> None:
    def detach(signum, frame):
        raise Detach(signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, detach)
    if signal.getsignal(signal.SIGHUP) != signal.SIG_IGN:
        signal.signal(signal.SIGHUP, detach)


def rehearsal_only(args) -> None:
    """The rehearsal's own options are refused without --rehearsal: given alone, they would be dropped silently."""
    given = [f for f, v in (("--rehearsal-project", args.rehearsal_project), ("--rehearsal-entity",
             args.rehearsal_entity), ("--compare-runs", getattr(args, "compare_runs", None))) if v]
    if given and not args.rehearsal:
        raise SystemExit(f"{', '.join(given)}: the rehearsal's options; give --rehearsal too (or leave them out)")


def cmd_plan(args) -> int:
    roster = load_roster(args.roster)
    rehearsal_only(args)
    if args.rehearsal:
        roster = rehearsal_roster(roster, args.rehearsal_project, args.rehearsal_entity)
    if args.n_steps:
        roster = dataclasses.replace(roster, n_steps=args.n_steps)
    plan = build_plan(roster, args.rehearsal)
    s = settings_from(args)
    est = estimate(plan, probe_copies(s))
    found = None
    if not args.offline:
        try:
            found = preflight(plan, s, est, cap_hours(s, est))
        except SystemExit as e:
            print(f"!! a start would refuse now: {e}")
    print_plan(plan, est, found, s)
    return 0


def cmd_start(args) -> int:
    roster = load_roster(args.roster)
    rehearsal_only(args)
    if args.n_steps:
        roster = dataclasses.replace(roster, n_steps=args.n_steps)
    if args.rehearsal:
        roster = rehearsal_roster(roster, args.rehearsal_project, args.rehearsal_entity)
    plan = build_plan(roster, args.rehearsal)
    s = settings_from(args)
    use_prefix(s)
    out = Path(nj.expand(args.out, roster.name)).expanduser()
    batch = Batch(out)
    if batch.path.exists():
        raise SystemExit(f"{out} holds a batch already ({batch.phase}): resume it, or give another --out")
    pointer = Path(args.current_batch).expanduser() if args.current_batch else out.parent / "CURRENT_BATCH"
    check_pointer(pointer, out)
    if args.allow_fresh and not args.runs_from:
        raise SystemExit("--allow-fresh: a run of --runs-from without a save starts fresh; give --runs-from DIR")
    pushes = runs_from(plan, args.runs_from, args.allow_fresh) if args.runs_from else {}
    out.mkdir(parents=True, exist_ok=True)
    Driver.log_file = out / "driver.log"
    est = estimate(plan, probe_copies(s), {n: v["steps_done"] for n, v in pushes.items() if "steps_done" in v})
    rl_hours = cap_hours(s, est)
    found = preflight(plan, s, est, rl_hours)
    print_plan(plan, est, found, s)
    batch.data.update(roster=dataclasses.asdict(roster), settings=dataclasses.asdict(s), estimate=est,
                      found={k: v for k, v in found.items() if k != "code"}, code=found["code"],
                      ssh_key=os.environ.get("NEBIUS_SSH_KEY"), rehearsal=bool(args.rehearsal),
                      plan={"runs": [dataclasses.asdict(r) for r in plan.runs], "probes": plan.probes,
                            "judges": plan.judges, "rehearsal": plan.rehearsal},
                      compare_runs=args.compare_runs or [], hours=rl_hours, runs_from=pushes)
    with batch_lock(out):  # from its first save: a "starting" batch no driver holds is a start that died (watchdog)
        batch.set_phase("starting")
        claim_pointer(batch, pointer)
        install_driver_signals()
        ops = None
        try:
            machines_up(batch, plan, s, found, rl_hours)
            ops = Ops(batch)
            push_runs(batch, ops, pushes)
            wire(batch, ops, plan)
            (out / "plan.json").write_text(json.dumps(vm_plan(batch, plan), indent=1) + "\n")
            checks = start_job(ops, out / "plan.json", 1, check_nobody=bool(args.rehearsal))
            batch.set_phase("running")
            extend_leases(ops)
            if args.rehearsal:
                rehearse(batch, ops, plan, checks, args.compare_runs or [])
            why = watch(batch, ops, plan)
        except Detach as e:
            if batch.phase == "starting":
                batch.event(f"driver stopped by {e} during the start: the watchdog ends what it started (or now: "
                            f"python3 scripts/rl_batch.py down --out {out})")
                return 75
            batch.event(f"driver stopped by {e}: the machines are left to their leases (vast.ai instances end within "
                        f"{s.lease_hours:g} h, the Nebius VMs power off then); resume: python3 scripts/rl_batch.py "
                        f"resume --out {out}; end now: python3 scripts/rl_batch.py down --out {out}")
            return 75
        except KeyboardInterrupt:
            end_all(batch, ops or (Ops(batch) if batch.machines.get("rl", {}).get("host") else None), plan,
                    "interrupted (Ctrl-C)")
            return 130
        except BaseException as e:  # noqa: BLE001 — SystemExit too: whatever stopped the start, nothing is left to bill
            import traceback

            traceback.print_exc()
            end_all(batch, ops or (Ops(batch) if batch.machines.get("rl", {}).get("host") else None), plan,
                    f"fatal: {type(e).__name__}: {e}")
            return 2
        ok = end_all(batch, ops, plan, f"the batch is over ({why})")
        return 0 if ok and why == "done" else (2 if not ok else 1)


def cmd_resume(args) -> int:
    """Watch a batch again (its driver died): the machines must still be up; a tunnel down is repaired by the watch, and
    a supervisor that exited with runs unfinished is started again (the keys sent again)."""
    batch = batch_from(args)
    if batch.phase in ("ended", "ending"):
        raise SystemExit(f"{batch.out}: the batch is {batch.phase}")
    plan = plan_from(batch)
    with batch_lock(batch.out, wait=False) as got:
        if not got:
            raise SystemExit("another driver holds the batch (driver.lock)")
        install_driver_signals()
        ops = Ops(batch)
        batch.event("resumed by a new driver")
        try:
            health_round(batch, ops, plan, roles_of(plan), batch.data.setdefault("bad_since", {}))
            if job_exit(ops) is not None and (vm_status(ops) or {}).get("phase") == "running":
                start_job(ops, batch.out / "plan.json", int(batch.data["job"].get("starts", 1)) + 1)
            why = watch(batch, ops, plan)
        except Detach as e:
            batch.event(f"driver stopped by {e}: the machines are left to their leases")
            return 75
        except KeyboardInterrupt:
            end_all(batch, ops, plan, "interrupted (Ctrl-C)")
            return 130
        except BaseException as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            end_all(batch, ops, plan, f"fatal: {type(e).__name__}: {e}")
            return 2
        ok = end_all(batch, ops, plan, f"the batch is over ({why})")
        return 0 if ok and why == "done" else (2 if not ok else 1)


def cmd_status(args) -> int:
    batch = batch_from(args)
    print(f"batch {batch.out}: {batch.phase}; roster {batch.data['roster']['name']}, n_steps "
          f"{batch.data['roster']['n_steps']}" + (" (rehearsal)" if batch.data.get("rehearsal") else ""))
    ops = Ops(batch)
    for role, m in batch.machines.items():
        line = f"  {role:7} {m.get('name') or '-':26} {m.get('provider'):6}"
        if m.get("ended"):
            line += f" ended {m['ended']}" + (f", ~${m['cost_usd']:.2f}" if m.get("cost_usd") is not None else "")
        elif m["provider"] == "vast":
            with contextlib.suppress(Exception):
                row = ops.row(role) or {}
                line += (f" {row.get('state')}, ${float(row.get('usd_per_hour') or 0):.2f}/h, ends "
                         f"{utc(float(row['deadline'])) if row.get('deadline') else '?'}, ~${nj.vast_cost(row) or 0:.2f}"
                         " so far")
        elif (r := m.get("replacing")) and r.get("out"):
            line += f" a new VM {r['k']}: its up started {r['started']} (launch/{Path(r['out']).name}.log)"
        elif r:
            line += f" a new VM {r['k']}: due {utc(float(r['not_before']))} ({r['why']})"
        elif m.get("abandoned"):
            line += f" NO VM: {m['abandoned']}"
        else:
            line += (f" {m.get('host')}" + (f" in {m['region']}" if m.get("region") else "")
                     + f", shutdown {m.get('shutdown_at', '?')}"
                     + (f", new VM {m['replacements']}" if m.get("replacements") else ""))
        print(line)
    for m in batch.data.get("retired") or []:
        print(f"  {m.get('role') or '-':7} {m.get('name') or '-':26} {m.get('provider') or '-':6} retired "
              f"{m.get('retired')} ({m.get('why')})" + ("" if m.get("ended") else "; NOT verified ended"))
    status = None
    with contextlib.suppress(Exception):
        status = vm_status(ops) if batch.machines.get("rl", {}).get("host") and not batch.machines["rl"].get(
            "ended") else batch.data.get("vm_status")
    status = status or batch.data.get("vm_status")
    if status:
        print(f"  runs ({status.get('at')}): " + status_line(status))
        for name, r in (status.get("runs") or {}).items():
            print(f"    {name:30} {r.get('state'):8} steps {r.get('steps', 0):3} evals {r.get('evals', 0):3} "
                  f"attempts {r.get('attempts', 0)} restarts {r.get('restarts', 0)}  {r.get('wandb') or ''}")
            if r.get("last_warning"):
                print(f"      last retry warning: {r['last_warning']}")
    print(f"  last sync: {(batch.data.get('last_sync') or {}).get('at')}; events: {len(batch.data.get('events', []))} "
          f"(last: {(batch.data.get('events') or [{}])[-1].get('what')})")
    return 0


def ask_or_do(args, what: str, action) -> int:
    """A manual command: done at once if no driver watches the batch, else left as a request it takes up within a
    minute (DIR/requests/<what>)."""
    batch = batch_from(args)
    with batch_lock(batch.out, wait=False) as got:
        if got:
            return action(batch)
    (batch.out / "requests").mkdir(exist_ok=True)
    (batch.out / "requests" / what).write_text(utc() + "\n")
    log(f"a driver watches {batch.out}: asked it to {what} (requests/{what})")
    return 0


def cmd_restart_server(args) -> int:
    what = f"{'replace' if args.new_instance else 'restart'}-server-{args.role}"

    def act(batch):
        return 0 if restart_server(batch, Ops(batch), plan_from(batch), args.role, args.new_instance) else 2

    return ask_or_do(args, what, act)


def cmd_run_control(args) -> int:
    batch = batch_from(args)
    control(Ops(batch), f"{args.cmd.split('-')[0]}-{args.run}")
    batch.event(f"control: {args.cmd} {args.run}")
    return 0


def cmd_sync(args) -> int:
    return ask_or_do(args, "sync", lambda b: (log(f"sync: {sync(Ops(b), [r['row']['run_name'] for r in b.data['plan']['runs']])}"), 0)[1])


def cmd_extend(args) -> int:
    return ask_or_do(args, "extend", lambda b: (extend_leases(Ops(b)), 0)[1])


def cmd_down(args) -> int:
    batch = batch_from(args)
    with batch_lock(batch.out, wait=False) as got:
        if not got and not args.force:
            raise SystemExit("a driver watches this batch: Ctrl-C it (it ends everything), or down --force")
        plan = plan_from(batch)
        ops = Ops(batch) if batch.machines.get("rl", {}).get("host") else None
        return 0 if end_all(batch, ops, plan, "down") else 2


def cmd_watchdog(args) -> int:
    """One check, for cron or a hub scheduled job (it reads output/gpu_jobs/CURRENT_BATCH, which ``start`` writes):
    nothing if a driver watches the batch; else one round of the watch (the leases: the vast.ai instances' and every
    Nebius VM's shutdown timer; a sync; the servers' health, a preempted Nebius VM's new one started or taken up), or
    the end of it all if its runs are over, its RL machine has been gone for ``vm_dead_minutes``, or its start died."""
    batch = batch_from(args)
    if batch.phase in ("ended", "new"):
        return 0
    with batch_lock(batch.out, wait=False) as got:
        if not got:
            log("watchdog: a driver watches the batch; nothing to do")
            return 0
        plan, ops = plan_from(batch), Ops(batch)
        if batch.phase == "starting":  # its start holds the lock until it is running or ended: the start died
            return 0 if end_all(batch, ops if batch.machines.get("rl", {}).get("host") else None, plan,
                                "watchdog: its start died (no driver holds it)") else 2
        why = watch(batch, ops, plan, once=True)
        if why in ("done", "failed", "rl-gone"):
            return 0 if end_all(batch, ops, plan, f"watchdog: the batch is over ({why})") else 2
        log("watchdog: the batch runs without a driver; leases extended, run folders synced")
        return 0


# ------------------------------------------------------------------------------------------- the plumbing test


def tunnel_test(batch: Batch, ops: Ops, out: Path) -> dict:
    """The tunnel through a dummy server: reachable from the RL machine; the key restricted (no command, no other
    port); a killed ssh and a dropped network both reconnected. Every check timed; results to DIR/tunnel_test.json."""
    res: dict = {}
    rl, vast = ops.rl(), ops.vast("dummy")

    def through() -> str:
        rc, text = run_text(rl, f"curl -sf -m 10 http://127.0.0.1:{PROBE_PORT}/ok.txt", 40)
        return text.strip() if rc == 0 else ""

    def save() -> None:
        (out / "tunnel_test.json").write_text(json.dumps(res, indent=1, default=str) + "\n")

    pub = make_tunnel_key(rl, "tunnel-test")
    authorize(vast, authorized_line(pub, [PROBE_PORT]))
    t0 = time.time()
    start_tunnel(rl, "dummy", vast.host, vast.port, tunnel_forwards("probe"), str(batch.machines["dummy"]["instance"]))
    got = wait_for("the dummy server through the tunnel", through, 120, 3)
    res["through_tunnel"] = {"ok": got == "tunnel-ok", "answer": got, "seconds": round(time.time() - t0, 1),
                             "target": f"{vast.host} -p {vast.port}"}
    save()
    opts = (f"-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/{TUNNEL_DIR}/probe.kh "
            f"-o IdentitiesOnly=yes -o IdentityFile={TUNNEL_KEY} -o ConnectTimeout=20 -p {vast.port} {vast.host}")
    rc, text = run_text(rl, f"ssh {opts} echo shell-ran 2>&1; echo rc=$?", 60)
    res["key_runs_no_command"] = {"ok": "shell-ran" not in text, "said": text.strip()[-300:]}
    rc, text = run_text(rl, f"ssh -N -o ExitOnForwardFailure=yes -L 9999:127.0.0.1:22 {opts} < /dev/null 2>&1 & "
                            "pid=$!; sleep 4; timeout 8 bash -c 'exec 3<>/dev/tcp/127.0.0.1/9999 && head -c 4 <&3' 2>&1; "
                            'echo; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; true', 90)
    res["key_opens_no_other_port"] = {"ok": "SSH-" not in text, "said": text.strip()[-300:]}
    save()
    t0 = time.time()
    kick_tunnel(rl, "dummy")
    time.sleep(1)
    got = wait_for("the tunnel back after its ssh was killed", through, 120, 2)
    res["reconnect_after_kill"] = {"ok": got == "tunnel-ok", "seconds": round(time.time() - t0, 1),
                                   "state": tunnel_state(rl, "dummy")}
    save()
    rc, ip = run_text(rl, f"getent ahostsv4 {shlex.quote(vast.host.split('@')[-1])} | awk 'NR==1{{print $1}}'", 30)
    ip = ip.strip().splitlines()[0] if ip.strip() else ""
    if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", ip):
        t0 = time.time()
        run_text(rl, f"sudo -n ip route add blackhole {ip}/32", 30)
        down = wait_for("the tunnel's ssh to notice the dead network", lambda: "down since" in tunnel_state(
            rl, "dummy").get("state", ""), 180, 5)
        noticed = round(time.time() - t0, 1)
        time.sleep(20)
        run_text(rl, f"sudo -n ip route del blackhole {ip}/32", 30)
        t1 = time.time()
        got = wait_for("the tunnel back after the network returned", through, 180, 3)
        res["reconnect_after_network_drop"] = {"ok": bool(down) and got == "tunnel-ok", "blackholed": ip,
                                               "ssh_noticed_after_s": noticed,
                                               "back_after_network_s": round(time.time() - t1, 1),
                                               "state": tunnel_state(rl, "dummy")}
    else:
        res["reconnect_after_network_drop"] = {"ok": False, "error": f"could not resolve {vast.host} ({ip!r})"}
    save()
    stop_tunnel(rl, "dummy")
    with contextlib.suppress(Exception):
        nj.fetch_rsync(rl, f"~/{TUNNEL_DIR}", out / "vm", batch.settings.rsync, timeout=300)
    return res


def cmd_tunnel_test(args) -> int:
    """The plumbing test: the cheapest on-demand Nebius CPU VM and the cheapest vast.ai GPU (CUDA ≥ 13, under
    ``--vast-max-price``) running a dummy HTTP server on 127.0.0.1:8178; the batch's own tunnel code between them
    (``tunnel_test``); then both ended and verified gone. No model, no vLLM, no tinker, no key on the GPU."""
    s = settings_from(args)
    use_prefix(s)
    out = Path(nj.expand(args.out, "tunnel-test")).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    Driver.log_file = out / "driver.log"
    batch = Batch(out)
    batch.data.update(settings=dataclasses.asdict(s), ssh_key=os.environ.get("NEBIUS_SSH_KEY"), phase="starting")
    batch.save()
    rl_args = ["up", "--platform", "cpu-d3", "--preset", args.nebius_preset, "--on-demand", "--image-family",
               "ubuntu24.04-driverless", "--hours", "1", "--name", f"{s.prefix}tuntest", "--out",
               str(out / "launch" / "rl"), "--wait-hours", "0", *(["--budget-file", s.budget_file] if s.budget_file
                                                                   else [])]
    dummy = ("mkdir -p results/www && echo tunnel-ok > results/www/ok.txt && exec python3 -m http.server "
             f"{PROBE_PORT} --bind 127.0.0.1 --directory results/www")
    vast_args = ["up", "--provider", "vast", "--hours", "1", "--max-price", f"{args.vast_max_price:g}", "--disk-gib",
                 "40", "--name", f"{s.prefix}tun-gpu", "--out", str(out / "launch" / "dummy"), "--purpose",
                 "MonitorDecorrelation: plumbing test of the RL batch's SSH tunnels (a dummy HTTP server; no key, no "
                 "model)", "--tag", f"owner={s.owner}", "--tag", f"experiment={s.experiment}", "--tag",
                 "step=tunnel-test", "--health", f"curl -sf http://127.0.0.1:{PROBE_PORT}/ok.txt", "--health-timeout",
                 "900", "--wait-hours", "0", "--rsync", s.rsync, "--hub-gpu", s.hub_gpu,
                 *(["--budget-file", s.budget_file] if s.budget_file else []), "--", dummy]
    res = {"started": utc()}
    ops = None
    try:
        procs = {"rl": launch(rl_args, out / "launch" / "rl.log", wait=False),
                 "dummy": launch(vast_args, out / "launch" / "dummy.log", wait=False)}
        for role, proc in procs.items():
            code = proc.wait()
            batch.machines[role] = machine_from_record(role, "nebius" if role == "rl" else "vast",
                                                       out / "launch" / role)
            batch.event(f"{role}: up exited {code}")
        if not all(m.get("kept") for m in batch.machines.values()):
            raise nj.InfraError("a machine did not come up; see launch/*.log")
        ops = Ops(batch)
        res.update(tunnel_test(batch, ops, out))
    except KeyboardInterrupt:
        res["error"] = "interrupted"
    except Exception as e:  # noqa: BLE001 — whatever happened, both machines end
        import traceback

        traceback.print_exc()
        res["error"] = f"{type(e).__name__}: {e}"
    finally:
        res["ended_all"] = end_all(batch, ops, None, "tunnel test over")
        with contextlib.suppress(Exception):
            hub = nj.HubGpu(s.hub_gpu)
            row = next((r for r in hub.instances() if r.get("name") == batch.machines.get("dummy", {}).get("name")),
                       None)
            res["vast_cost_usd"] = nj.vast_cost(row) if row else None
            res["vast_row"] = {k: (row or {}).get(k) for k in ("instance", "gpu", "usd_per_hour", "started", "ended",
                                                                  "state", "tags")}
        rl_rec = machine_record(out / "launch" / "rl")
        if rl_rec.get("times", {}).get("create_start"):
            hours = (time.time() - nj._seconds(rl_rec["times"]["create_start"])) / 3600
            price = (rl_rec.get("offers") or [{}])[0].get("price") or 0.07
            res["nebius_cost_usd_estimate"] = round(hours * float(price), 3)
        res["ended"] = utc()
        (out / "tunnel_test.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
        log(f"tunnel test: {json.dumps({k: (v.get('ok') if isinstance(v, dict) and 'ok' in v else v) for k, v in res.items()}, default=str)}")
    checks = [v for v in res.values() if isinstance(v, dict) and "ok" in v]
    return 0 if res.get("ended_all") and checks and all(v["ok"] for v in checks) and not res.get("error") else 1


# ---------------------------------------------------------------------------------------------------- CLI


def add_settings(p: argparse.ArgumentParser) -> None:
    d = Settings()
    p.add_argument("--prefix", help=f"machine names start with it (default {d.prefix}; NEBIUS_JOB_NAME_PREFIX)")
    p.add_argument("--owner", help=f"the owner= tag (default {d.owner})")
    p.add_argument("--experiment", help=f"the experiment= tag (default {d.experiment})")
    p.add_argument("--rl-preset", help=f"the RL machine's cpu-d3 preset (default {d.rl_preset})")
    p.add_argument("--rl-hours", type=float, help="its shutdown cap (default: the estimate × 1.5 + 2 h, 6 to 100)")
    p.add_argument("--probe-gpu", help=f"the probe server's GPU (default {d.probe_gpu})")
    p.add_argument("--probe-gpus", type=int, help=f"its count (default {d.probe_gpus}: one copy of the BF16 reader)")
    p.add_argument("--probe-disk-gib", type=int, help=f"its disk (default {d.probe_disk_gib}: 496 GiB of weights)")
    p.add_argument("--probe-max-price", type=float, help=f"USD/h cap for its instance (default {d.probe_max_price:g})")
    p.add_argument("--judge-gpu", dest="judge_gpus_names",
                   help=f"the judges' GPU, or a comma list in order of preference (default {d.judge_gpus_names!r})")
    p.add_argument("--judge-gpus", type=int, help=f"their count (default {d.judge_gpus}: setup_vllm.sh's 4-GPU layout)")
    p.add_argument("--judge-disk-gib", type=int, help=f"their disk (default {d.judge_disk_gib})")
    p.add_argument("--judge-max-price", type=float, help=f"USD/h cap for their instance (default {d.judge_max_price:g})")
    for role in SERVER_ROLES:
        dflt = nebius_machine(d, role)
        p.add_argument(f"--{role}-provider", choices=PROVIDERS,
                       help=f"where the {role} server runs: vast (default), or nebius (a VM of its own, spot by "
                            "default; its shutdown timer is its lease; a new VM after every preemption)")
        p.add_argument(f"--{role}-platform", help=f"Nebius: its platform, or a comma list (each where it is offered; "
                                                  f"default {dflt['platform']})")
        p.add_argument(f"--{role}-preset", help=f"Nebius: its preset (default {dflt['preset']})")
        p.add_argument(f"--{role}-on-demand", action="store_const", const=True, default=None,
                       help="Nebius: on demand instead of spot (no bid)")
        p.add_argument(f"--{role}-max-gpu-price", type=float, metavar="USD",
                       help=f"Nebius: the spot bid per GPU-hour, NEBIUS_MAX_GPU_PRICE's kind (default {dflt['bid']:g})")
        p.add_argument(f"--{role}-regions", help="Nebius: a comma list of its regions (default: every region of the "
                                                 "regions file)")
        p.add_argument(f"--{role}-gpu-ids", dest=f"{role}_gpu_ids",
                       help="the GPUs it uses on its machine, e.g. 4,5,6,7 (default: the probe server all, split "
                            "between its copies; the judges a 2- or 4-GPU machine's, else 0-3)")
    p.add_argument("--judges-cache", metavar="DIR",
                   help="Nebius judges: DIR/flashinfer and DIR/vllm go to the VM's ~/.cache/ before the servers start "
                        "(compiled kernels of an earlier VM: a B300's first start compiles for ~15 min)")
    p.add_argument("--rl-regions", help=f"the RL machine's Nebius regions (default {d.rl_regions})")
    p.add_argument("--probe-copies", type=int,
                   help="reader copies on the probe server's machine, behind a balancer at :8178 (default: one on "
                        f"vast.ai, one per {GPUS_PER_COPY} GPUs on Nebius)")
    p.add_argument("--max-replacements", type=int,
                   help=f"Nebius servers: new VMs per role, after preemptions (default {d.max_replacements})")
    p.add_argument("--replace-backoff-minutes", type=float,
                   help=f"before a role's k-th new VM: this × 2^(k-1), at most 30 (default "
                        f"{d.replace_backoff_minutes:g})")
    p.add_argument("--lease-hours", type=float, help=f"each vast.ai lease (default {d.lease_hours:g})")
    p.add_argument("--sync-minutes", type=float, help=f"between run-folder syncs (default {d.sync_minutes:g})")
    p.add_argument("--health-minutes", type=float, help=f"between health checks (default {d.health_minutes:g})")
    p.add_argument("--server-dead-minutes", type=float,
                   help=f"a server down this long is restarted (default {d.server_dead_minutes:g})")
    p.add_argument("--max-server-restarts", type=int, help=f"per server (default {d.max_server_restarts})")
    p.add_argument("--no-replace", dest="replace", action="store_const", const=False, default=None,
                   help="never replace a machine that went away (default: a vast.ai instance once, a Nebius VM every "
                        "time, up to --max-replacements)")
    p.add_argument("--max-attempts", type=int, help=f"failed attempts per run (default {d.max_attempts})")
    p.add_argument("--vm-dead-minutes", type=float,
                   help=f"the RL machine unreachable this long ends the batch (default {d.vm_dead_minutes:g})")
    p.add_argument("--stall-minutes", type=float,
                   help=f"no run finishing a step or an eval this long is reported (default {d.stall_minutes:g})")
    p.add_argument("--rsync", default=os.environ.get("RSYNC_BIN"), help="this machine's rsync (env RSYNC_BIN)")
    p.add_argument("--budget-file", help="the spending ledger (default $NEBIUS_BUDGET_FILE)")
    p.add_argument("--hub-gpu", default=os.environ.get("HUB_GPU_BIN"), help="the hub-gpu CLI (env HUB_GPU_BIN)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, func, hlp in (("plan", cmd_plan, "the batch's runs, pinned probes, estimate and prices; starts nothing"),
                            ("start", cmd_start, "machines, servers, tunnels, runs, the watch, the end")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--roster", required=True, help="a roster file (experiments/rl_batches/*.json)")
        p.add_argument("--rehearsal", action="store_true", help="2 steps, an eval at each, and the checklist")
        p.add_argument("--rehearsal-project", help="the rehearsal's W&B project (default <project>-rehearsal)")
        p.add_argument("--rehearsal-entity", help="the rehearsal's W&B entity (default the roster's)")
        p.add_argument("--n-steps", type=int, help="every run's n_steps instead of the roster's (e.g. 360)")
        add_settings(p)
        p.set_defaults(func=func)
    sub.choices["plan"].add_argument("--offline", action="store_true", help="no hub-gpu calls (no prices)")
    sub.choices["start"].add_argument("--out", required=True, help="the batch's folder ({name}, {stamp} expand)")
    sub.choices["start"].add_argument("--compare-runs", nargs="*", default=[],
                                      help="rehearsal: run folders for rescore_terminal_judges.py compare")
    sub.choices["start"].add_argument("--runs-from", metavar="DIR",
                                      help="each run's folder DIR/<run_name>/ to the RL machine before the supervisor "
                                           "starts: it resumes from its newest resume_states/step_*.json (refused "
                                           "without one)")
    sub.choices["start"].add_argument("--allow-fresh", action="store_true",
                                      help="with --runs-from: a run without a save there starts fresh instead")
    sub.choices["start"].add_argument("--current-batch", metavar="FILE",
                                      help="the file naming the batch for the watchdog's scheduled job (default: "
                                           "CURRENT_BATCH beside the batch's folder, output/gpu_jobs/)")
    for name, func, hlp in (("resume", cmd_resume, "watch a batch again after its driver died"),
                            ("status", cmd_status, "machines, runs, last sync"),
                            ("sync", cmd_sync, "the run folders now"),
                            ("extend", cmd_extend, "the leases now"),
                            ("watchdog", cmd_watchdog, "one check, for cron: extend, or end a batch that is over")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--out", required=True)
        p.set_defaults(func=func)
    p = sub.add_parser("down", help="end the batch: stop the runs, sync, end every machine and verify")
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true", help="even if a driver holds the batch")
    p.set_defaults(func=cmd_down)
    p = sub.add_parser("restart-server", help="a server started again on its instance (or --new-instance)")
    p.add_argument("role", choices=("probe", "judges"))
    p.add_argument("--out", required=True)
    p.add_argument("--new-instance", action="store_true", help="on a fresh instance, the tunnel moved there")
    p.set_defaults(func=cmd_restart_server)
    for name in ("restart-run", "stop-run"):
        p = sub.add_parser(name, help=f"{name.split('-')[0]} one run (the supervisor's control file)")
        p.add_argument("run")
        p.add_argument("--out", required=True)
        p.set_defaults(func=cmd_run_control)
    p = sub.add_parser("judge-compare", help="rescore_terminal_judges.py compare against a running batch's judges")
    p.add_argument("--out", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="run folders (eval_rollouts.jsonl + run_info.json)")
    p.add_argument("--passes", type=int, default=2, help="the same sample scored this many times (default 2)")
    p.add_argument("--workers", type=int, help="judge calls in flight (the compare's --workers; default its own)")
    p.set_defaults(func=cmd_judge_compare)
    p = sub.add_parser("vm-supervise", help="(on the RL machine, as root) every run of the plan")
    p.add_argument("--plan", required=True)
    p.add_argument("--state", required=True)
    p.set_defaults(func=cmd_vm_supervise)
    p = sub.add_parser("tunnel-test", help="the plumbing test: a dummy server through the tunnel")
    p.add_argument("--out", required=True)
    p.add_argument("--nebius-preset", default="2vcpu-8gb", help="the cheapest cpu-d3 preset (default 2vcpu-8gb)")
    p.add_argument("--vast-max-price", type=float, default=0.2, help="USD/h cap for the GPU (default 0.2)")
    add_settings(p)
    p.set_defaults(func=cmd_tunnel_test)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
