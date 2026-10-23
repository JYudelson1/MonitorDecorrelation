#!/usr/bin/env python3
"""Run a job on a fresh Nebius VM (a GPU one by default, or CPU-only), a vast.ai instance or any SSH box, and always
clean up.

This automates what we first did by hand on 2026-10-09 (the Inkling-Small-NVFP4 load test on one
B300), so a GPU job is a single command:

    python3 scripts/nebius_job.py run --hours 2 --out runs/{stamp}-{name} \\
        --hf thinkingmachines/Inkling-Small-NVFP4 --push ../episodes.jsonl:data/episodes.jsonl \\
        --fetch results -- uv run python experiments/some_job.py --out results
    python3 scripts/nebius_job.py up --hours 4 --out runs/{stamp}-{name}     # keep the VM, print ssh
    python3 scripts/nebius_job.py exec --host md@1.2.3.4 --out runs/x -- CMD  # any SSH box, no Nebius
    python3 scripts/nebius_job.py down --name md-job-3fa9c1                   # delete + verify, any region
    python3 scripts/nebius_job.py list                                        # every region
    python3 scripts/nebius_job.py prices                                      # every region's price now
    python3 scripts/nebius_job.py run --provider vast --gpu B200 --gpus 4 --hours 5 --out runs/{stamp}-{name} -- CMD
    # run/up accept --dry-run: price check, then the create argv, cloud-init and remote scripts; create nothing

Where the VM goes: ``--regions-file`` (env NEBIUS_REGIONS_FILE) lists the candidates, e.g.
``{"tenant": ID, "regions": {"us-north1": {"project", "subnet", "image_project"}, ...}}``.
- Before creating anything, run/up check every region's current price for the VM, using Nebius's cost
  calculator. For spot, that is the region's spot price. They also read the tenant's capacity advice.
- The VM goes to the cheapest region that offers the platform, has capacity and quota, and is under
  ``--max-price`` (USD/hour for the whole VM; no cap by default). If creation fails there, they clean up and
  try the next.
- ``--regions a,b`` narrows the list. ``--project-id`` / ``--subnet-id`` / ``--image-project`` instead pin
  one project.
- Spot prices move every 15 minutes, and on 2026-10-09 the same B300 cost $0.99/h in us-north1 and $7.69/h in
  uk-south1. So ``job.json`` records every region's quote, the chosen region, the price every 15 minutes
  while the VM exists, and an estimated cost. The CLI cannot show what a VM actually cost; the console's
  billing pages can.

The job command is everything after ``--``. It runs in the pushed repo on the VM. If it is a single
argument it is treated as a shell string (pipes etc. work); otherwise the arguments are shell-quoted.
The job gets PYTHONUNBUFFERED=1, NEBIUS_JOB_RAM=/mnt/ram (where ``--hf`` puts weights), each
``--env K=V`` and each ``--secret`` (below). ``{stamp}`` and ``{name}`` in ``--out`` are expanded.

What ``run`` does (``exec`` is the SSH part only, so a non-Nebius box works too):
1. Pick the region (above). Create the VM there, wait for RUNNING with a public IP, then wait for SSH.
2. Bootstrap: RAM disk, lingering, uv.
3. Start the ``--hf`` downloads, detached.
4. Push the repo, with its ``CODE_VERSION.json`` (below), and the ``--push`` inputs.
5. ``uv sync --locked``. If torch can't see the GPU, fall back to scripts/setup_box.sh. (No GPU check on a CPU VM or
   with ``exec --no-gpu``; ``--root-job`` adds its own steps, below.)
6. Wait for the downloads, then run the job, detached and polled.
   - With ``--upload``, the HF token reaches the VM before the job, so the job can save its own progress as it goes
     (``extract_probe_features.py --store``). The token stays until the upload step ends.
   - The ``--secret`` keys reach the VM right before the job and are deleted when the job step ends.
   - The job gets ``NEBIUS_JOB_DEADLINE``, the VM's scheduled shutdown in epoch seconds, so it can stop cleanly
     before ``--hours`` cuts it off.
   - When the job ends, the VM schedules its own power-off 15 minutes later, unless ``--hours`` comes sooner. If
     the launcher has died (e.g. its machine restarted), the VM stops itself instead of idling until ``--hours``.
7. Upload (``--upload``), then fetch the ``--fetch`` paths and all remote logs.
8. Delete the VM and verify that neither it nor its boot disk is left.
Step 8 runs in a ``finally``, so errors, preemption and Ctrl-C all still delete; only ``--keep``
opts out. ``job.json`` in ``--out`` records the VM, flags, UTC timestamps, exit code and whether
the VM was preempted. Each step's log is mirrored into ``--out`` while it streams, so it survives
a preemption.
- SIGTERM and SIGHUP (the launcher's machine shutting down, its terminal closing) are handled like Ctrl-C. Under
  ``nohup`` SIGHUP stays ignored, so the launcher outlives its terminal.
- None of them interrupts step 8: one that arrives then is held until the VM is deleted, and the launcher stops
  there (no further attempt). SIGTERM or SIGHUP during step 7's fetches skips the rest of them, to delete sooner.

CPU VMs, keys and root jobs (2026-10-17, for the pre-RL terminal rollouts: they sample from tinker and need root):
- ``--platform cpu-d3 --preset 32vcpu-128gb --on-demand --image-family ubuntu24.04-driverless`` is a CPU-only VM
  ($1.06/h on demand; 16vcpu-64gb $0.53/h). CPU VMs run on-demand only. Capacity advice covers GPU presets only, so
  a CPU preset it doesn't list is still priced in every region, its capacity and quota unknown. NEBIUS_MAX_GPU_PRICE
  (a spot bid, which a wrapper may set by default) is ignored for ``--on-demand``; ``--max-gpu-price`` with it is
  refused. The setup is ``uv sync --locked`` and the Python version, without the torch/CUDA check (``exec
  --no-gpu`` does the same on any box).
- ``--secret NAME[=ENV]`` (repeatable): the key in ~/.secrets/NAME on this machine (``hub-secret set NAME``) is the
  job's $ENV (default: NAME upper-cased). Like the HF token it goes over SSH stdin into an owner-only file on the RAM
  disk (/mnt/ram/secrets/NAME), right before the job. The job step reads it when it starts and deletes it when it
  ends; the launcher deletes it too if it stops first. A missing or empty key is refused before anything is created.
  No key reaches a command line, a log, ``job.json`` or the dry run: only names do.
- ``--root-job`` runs the job as root, as the RL runs did: the terminal env's ``read_only_verifier`` runs the
  policy's commands as ``nobody`` and refuses to start otherwise. The setup installs uv's CPython
  ``--sandbox-python`` (default 3.11.16) into /opt/uv-python, where ``nobody`` can run it, syncs the venv on exactly
  that version and makes $HOME (so the repo) private (``chmod 750``). It then fails unless, as root, the env's
  sandbox check passes, the venv runs that version and ``nobody`` cannot read the repo. The job runs through
  ``sudo -n -E -H``, with its environment (``--env``, the keys, NEBIUS_JOB_DEADLINE, …), root's HOME and
  UV_NO_SYNC=1. Its command should call ``.venv/bin/python …`` (or ``uv run --no-sync …``): as root, a syncing
  ``uv run`` would rewrite the venv. ``job.json`` records ``root_job`` and ``sandbox_python``. The pre-RL rollouts:

    python3 scripts/nebius_job.py run --hours 3 --platform cpu-d3 --preset 32vcpu-128gb --on-demand \\
        --image-family ubuntu24.04-driverless --secret tinker_api_key --root-job --out runs/{stamp}-{name} \\
        --upload data/runs/pre-rl-probe-v2=hf://buckets/probemaxxing/inkling-probes/inputs/pre-rl-probe-v2/{stamp} \\
        --fetch data/runs/pre-rl-probe-v2 -- .venv/bin/python experiments/eval_terminal_monitors_baseline.py \\
        --config experiments/configs/terminal_verifier_control.json --n-prompts 512 --samples 64 \\
        --episode-workers N --run-name pre-rl-probe-v2 --set 'monitors=[]' seed=1000000

vast.ai (``run --provider vast``, 2026-10-23, when Nebius had no B200/B300 capacity), through hub's ``hub-gpu`` CLI:
billed to the account the owner bills this project to, within the limits the owner sets (``hub-gpu limits``). Nebius's
flags (``--platform``, ``--keep``, ``--max-gpu-price``, …) are refused with it, its own (``--gpu``, ``--gpus``,
``--offer``, …) without it.
- The offer: the cheapest of ``hub-gpu offers`` for ``--gpu`` × ``--gpus`` with ``--disk-gib`` of disk whose driver runs
  CUDA ``--min-cuda`` (default 13.0: the locked torch is a CUDA 13.0 build, which also rules out GPUs older than
  Turing), under ``--max-price`` (USD/h for the whole instance); or ``--offer ID``. If the start is refused (the offer
  was taken meanwhile), the cheapest of a fresh listing is started instead, once. While none qualifies, the offers are
  checked again every 15 minutes for ``--wait-hours``.
- Before the start: its worst case, price × ``--hours``, must fit in what ``hub-gpu limits`` leaves of the project's 24
  hours, and under ``--max-cost``; ``--hours`` must be within its timed runs' limit (72 h), the GPUs within its count.
- The start is timed (``hub-gpu start --hours``): vast.ai destroys the instance at its end, with what is on it. Its name
  is ``--name`` plus a random suffix (lowercase letters, digits and -, at most 24 characters, new among the project's
  instances), its purpose line ``--purpose`` (default: the job), its tags ``--tag`` plus launcher= and commit=.
- Then as on Nebius, over SSH as root with hub-gpu's key (~/.ssh/hub-gpu): bootstrap, downloads, push, setup, job. SSH
  is waited for while the instance pulls its image (listed as loading; at most 40 minutes, and not past the job's
  deadline). The bootstrap mounts a tmpfs at /mnt/ram if the container may, else /mnt/ram is a plain directory on the
  instance's disk (said so in bootstrap.log); it installs rsync and a C compiler if missing, and uv with only uv's own
  Python.
- The job's ``NEBIUS_JOB_DEADLINE`` is the instance's end less ``--reserve-minutes`` (default 30, at most a quarter of
  ``--hours``), kept for the fetch; the launcher stops waiting for the job halfway into that reserve. There is no
  power-off after the job: the launcher ends the instance, and vast.ai destroys it at its end anyway.
- ``--fetch`` paths come back with rsync (uncompressed, resumable) every 10 minutes while the job runs, so what it saved
  is here before the instance's end (12 MB/s through vast.ai's SSH proxy on 2026-10-23), and once more with the logs
  after it. Then ``hub-gpu end``, verified in ``hub-gpu list``. Errors, Ctrl-C, SIGTERM and SIGHUP end it too, held as
  on Nebius; there is no ``--keep``. An instance that disappears, ends or stops running mid-job is reported (exit 3);
  nothing relaunches it.
- No secret reaches the instance (someone else's machine): ``--secret``, ``--hf-token-file`` and ``--upload`` (an HF
  bucket needs a token) are refused, so are an ``--env`` that looks like a key (a name with TOKEN, KEY, SECRET, … and
  a long value) and a ``--push`` from a key folder or with a .env file. ``--hf`` downloads are unauthenticated, as on
  Nebius. Results come back only through ``--fetch``.
- The ledger (``--budget-file``) records the instance as a vast.ai entry, reserved at its worst case and settled at its
  price × the time from its start to its verified end, but not counted against ``limit_usd``: vast.ai is billed to
  another account, under hub's limits for this project, which are checked instead.

Relaunching (``run --attempts N``, default 3): after a spot preemption, or a job that exits 75 (it stopped early,
e.g. at its deadline, and can resume), a fresh VM runs the job again, in whichever region is cheapest then. That
continues up to N VMs in all.
- Each VM is checked against the budget like the first.
- The job must resume from what it saved; ``extract_probe_features.py --store`` does.
- Attempt k logs to ``--out``/retry-k/.
- When no region qualifies (price above the bid or cap, no capacity), run/up re-check every 15 minutes for
  ``--wait-hours`` (default 2).

Leftovers: run/up first ``reap`` the ledger. Reap finds reservations whose launcher died mid-run, deletes their VM
once it has stopped, and settles them from that launcher's sampled prices. ``reap`` does only that.
- A VM whose job.json reap can't find (the ledger records ``--out`` as an absolute path; older entries may be
  relative) is treated as still running its job: only its ``--hours`` cap makes it overdue.
- A kept VM (``up``, ``run --keep``) stays reserved until ``down`` deletes it (with the ledger, ``--budget-file`` or
  NEBIUS_BUDGET_FILE) or reap finds it gone; either settles it from the prices its launcher sampled.

Code version: the pushed repo has no .git, so each push also writes ``CODE_VERSION.json`` into it: the launching
machine's ``code_version`` of what it pushed (commit, dirty, diff_sha256; src/monitordecorrelation/provenance.py).
Code on the VM stamps its outputs from that file. ``job.json`` records it as ``code``.
- run/up/exec refuse a working tree with uncommitted changes (tracked changes, or untracked files that would travel)
  before anything is created, and again at each push (the tree may change during a run), unless ``--allow-dirty``.
- With ``--allow-dirty``, a push with changes saves them in that attempt's ``--out``: ``code.diff``, the patch from
  the commit to what was pushed (its sha256 is ``diff_sha256``; ``git apply`` it on a checkout of the commit to
  rebuild the code), and ``code_untracked.txt``, the untracked files in it.

Lessons from the manual session, built in here:
- Weights go to a tmpfs RAM disk at /mnt/ram, for two reasons.
  - The VM's network SSD writes about 47 MB/s. An NVFP4 checkpoint would have taken over half an
    hour; it took 1 min 46 s to RAM.
  - /dev/shm is not safe either: systemd-logind (RemoveIPC=yes, the default) deletes a user's
    /dev/shm files when their last session ends, which silently removed a finished 171 GB download.
  Lingering is enabled too, so the user never fully logs out.
- Long remote steps run detached and are polled: ``setsid nohup bash -lc '…' > log 2>&1 < /dev/null &``,
  with any ``cd`` inside the inner shell. ``cd x && nohup … &`` backgrounds a subshell that keeps
  the SSH channel open, so ssh hangs until the step finishes.
- Nothing uses ``pkill -f``: a pattern that also appears in the invoking command line kills the
  invoking shell.
- Spot VMs need a pricing model (``--follows-spot-price``), and the managed boot disk needs a name.
  ``--recovery-policy FAIL`` makes the in-VM auto-shutdown (``--hours``, the spending cap) stop the
  VM; under RECOVER a guest shutdown restarts it and billing continues.
- Each run gets its own known_hosts in ``--out``, because public IPs are reused across VMs.
- No secrets go to the VM but the HF token (``--upload``) and the ``--secret`` keys, each over SSH stdin into an
  owner-only file on the RAM disk, deleted after use. Only public HF repos are downloaded (no HF_TOKEN), and the repo
  is pushed as ``git ls-files`` plus untracked-but-not-ignored files, so .env*, .venv and .git never travel.

Exit status: 0 on success; the job's own exit code if it failed; 2 for an infrastructure failure;
3 if the VM stopped mid-run (spot preemption or the --hours cap; a vast.ai instance that went away); 75 if the job
stopped early to resume later and no attempt was left; 130 on Ctrl-C, SIGTERM or SIGHUP (also when it arrived while
the VM was deleted).
Stdlib only, so plain ``python3`` (3.11+) runs it.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.util
import json
import math
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAM_DIR = "/mnt/ram"
REMOTE_REPO = "md"  # relative to the remote $HOME, where every ssh command starts
REMOTE_LOGS = "nebius_job"  # ditto: <step>.log / <step>.exit for the detached steps
SEP = "---nebius-job-sep---"
EXIT_INFRA, EXIT_STOPPED, EXIT_INTERRUPTED = 2, 3, 130
EXIT_UNFINISHED = 75  # the job's own code for "stopped early, saved its progress, run me again to resume"
SCHEDULED_SHUTDOWN = "/run/systemd/shutdown/scheduled"  # systemd's pending `shutdown -h +N`: USEC=<epoch µs>
POWEROFF_AFTER_JOB_MINUTES = 15
PRICE_RECHECK_SECONDS = 900  # spot prices change at most every 15 minutes
MIN_RELAUNCH_HOURS = 0.5  # a relaunch needs ~15 min of setup before it does anything
ALIVE_STATES = {"RUNNING", "STARTING", "CREATING", "UPDATING"}
SSH_DOWN = 255  # ssh's own failure code (the remote command's codes pass through)


def _load_provenance():
    """src/monitordecorrelation/provenance.py, by its path: the package's __init__ needs the project's dependencies,
    that file only the stdlib."""
    spec = importlib.util.spec_from_file_location(
        "nebius_job_provenance", REPO / "src" / "monitordecorrelation" / "provenance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


provenance = _load_provenance()


class InfraError(RuntimeError):
    """Provisioning, SSH, setup or teardown failed."""


class VmStopped(RuntimeError):
    """The VM left RUNNING under us: spot preemption or the --hours auto-shutdown."""


class LauncherSignal(KeyboardInterrupt):
    """SIGTERM or SIGHUP to the launcher (e.g. its machine shutting down): handled like Ctrl-C, so the VM is still
    deleted."""


class HeldSignals:
    """Signals that arrived while a VM was being deleted (``signals_held``): the first one (None: none), and whether
    SIGTERM/SIGHUP may currently cut the best-effort collection short."""

    def __init__(self) -> None:
        self.signum: int | None = None
        self.cut_collection = False

    def hold(self, signum: int) -> None:
        if self.signum is None:
            self.signum = signum
        if self.cut_collection and signum != signal.SIGINT:  # the machine or terminal is going away: delete first
            self.cut_collection = False
            raise LauncherSignal(f"signal {signum}")
        log(f"{signal.Signals(signum).name} held off while the VM is deleted; the launcher stops once it is gone")


_held: HeldSignals | None = None  # set while a VM is being deleted: signals are held instead of raised


def _launcher_signal(signum, frame):
    """The launcher's handler for Ctrl-C, SIGTERM and SIGHUP: raise KeyboardInterrupt (``LauncherSignal`` for SIGTERM
    and SIGHUP), so the VM is still deleted; while it is being deleted (``_held``, set by ``signals_held``), hold the
    signal instead. Holding starts with one assignment, not a handler swap, so no signal slips in between; blocking
    signals in the main thread could not ensure that while the price sampler's thread runs."""
    held = _held
    if held is not None:
        held.hold(signum)
        return
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    raise LauncherSignal(f"signal {signum}")


def install_launcher_signals() -> dict:
    """``_launcher_signal`` for Ctrl-C, SIGTERM and SIGHUP. A Ctrl-C or SIGHUP this process inherited as ignored stays
    ignored: a launcher started with ``nohup`` must outlive its terminal. Returns the previous handlers, for
    ``restore_signals``."""
    old = {}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        if sig != signal.SIGTERM and signal.getsignal(sig) == signal.SIG_IGN:
            if sig == signal.SIGHUP:
                log("SIGHUP is ignored (nohup): the launcher keeps going if its terminal closes")
            continue
        old[sig] = signal.signal(sig, _launcher_signal)
    return old


def restore_signals(old: dict) -> None:
    for sig, handler in old.items():
        signal.signal(sig, handler)


def utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    try:
        print(f"[nebius_job {datetime.now(UTC):%H:%M:%S}] {msg}", flush=True)
    except (OSError, RuntimeError):  # a hung-up terminal (EIO), or a line logged by a signal handler mid-print
        pass  # (reentrant call): a log line must never stop a teardown


# ---------------------------------------------------------------------------------------- pure parts


def render_cloud_init(user: str, pubkey: str, hours: float) -> str:
    """cloud-config for a fresh VM: a passwordless-sudo user with ``pubkey``, a RAM disk at /mnt/ram,
    lingering, and a guest shutdown ``hours`` after boot (the spending cap)."""
    if not re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}", user):
        raise ValueError(f"bad user name {user!r}")
    minutes = max(1, round(hours * 60))
    return "\n".join([
        "#cloud-config",
        "users:",
        f"  - name: {user}",
        "    sudo: ALL=(ALL) NOPASSWD:ALL",
        "    shell: /bin/bash",
        "    ssh_authorized_keys:",
        f"      - {json.dumps(pubkey.strip())}",  # a JSON string is a valid YAML scalar, whatever the comment
        "runcmd:",
        f"  - mkdir -p {RAM_DIR}",
        f"  - mount -t tmpfs -o size=90%,mode=1777 tmpfs {RAM_DIR}",
        f"  - loginctl enable-linger {user}",
        f"  - shutdown -h +{minutes}",
        "",
    ])


@dataclass
class VmSpec:
    name: str
    project_id: str
    subnet_id: str
    platform: str = "gpu-b300-sxm"
    preset: str = "1gpu-24vcpu-346gb"
    spot: bool = True
    disk_gib: int = 200
    image_family: str = "ubuntu24.04-cuda13.0"
    image_project: str = "project-e03public-images"
    hours: float = 2.0
    user: str = "md"
    pricing_policy_id: str | None = None  # spot with a max bid (Nebius pricing policy) instead of following the price


def create_argv(nebius: str, spec: VmSpec, user_data: str) -> list[str]:
    """The ``nebius compute instance create`` call that worked on 2026-10-09 (blocking; prints the
    instance as JSON once it is RUNNING)."""
    argv = [nebius, "compute", "instance", "create",
            "--parent-id", spec.project_id, "--name", spec.name,
            "--resources-platform", spec.platform, "--resources-preset", spec.preset]
    if spec.spot and spec.pricing_policy_id:
        argv += ["--spot-pricing-policy-id", spec.pricing_policy_id, "--preemptible-on-preemption", "stop"]
    elif spec.spot:
        argv += ["--follows-spot-price", "--preemptible-on-preemption", "stop"]
    else:
        argv += ["--on-demand"]
    nic = [{"name": "eth0", "ip_address": {}, "public_ip_address": {}, "subnet_id": spec.subnet_id}]
    return argv + [
        "--recovery-policy", "FAIL",
        "--boot-disk-attach-mode", "READ_WRITE",
        "--boot-disk-managed-disk-name", f"{spec.name}-boot",
        "--boot-disk-managed-disk-type", "network_ssd",
        "--boot-disk-managed-disk-size-gibibytes", str(spec.disk_gib),
        "--boot-disk-managed-disk-source-image-family-image-family", spec.image_family,
        "--boot-disk-managed-disk-source-image-family-parent-id", spec.image_project,
        "--network-interfaces", json.dumps(nic),
        "--cloud-init-user-data", user_data,
        "--format", "json",
    ]


@dataclass
class Instance:
    id: str
    name: str
    state: str
    ip: str | None
    platform: str = ""
    preemptible: bool = False
    created_at: str = ""


def parse_instance(d: dict) -> Instance:
    meta, spec, status = d.get("metadata") or {}, d.get("spec") or {}, d.get("status") or {}
    ip = None
    for nic in status.get("network_interfaces") or []:
        address = (nic.get("public_ip_address") or {}).get("address") or ""
        if address:
            ip = address.split("/")[0]  # "89.124.37.250/32"
            break
    return Instance(id=meta.get("id", ""), name=meta.get("name", ""), state=status.get("state", "UNKNOWN"),
                    ip=ip, platform=(spec.get("resources") or {}).get("platform", ""),
                    preemptible="preemptible" in spec, created_at=meta.get("created_at", ""))


@dataclass
class Disk:
    id: str
    name: str
    state: str
    size_gib: str = ""


def parse_disk(d: dict) -> Disk:
    meta, spec, status = d.get("metadata") or {}, d.get("spec") or {}, d.get("status") or {}
    return Disk(id=meta.get("id", ""), name=meta.get("name", ""), state=status.get("state", "UNKNOWN"),
                size_gib=str(spec.get("size_gibibytes", "")))


def parse_items(text: str) -> list[dict]:
    return (json.loads(text) if text.strip() else {}).get("items") or []


def parse_hf(spec: str) -> tuple[str, str | None, str]:
    """``ORG/REPO[@REVISION][=NAME]`` -> (repo id, revision, folder name under /mnt/ram)."""
    repo_rev, _, name = spec.partition("=")
    repo_id, _, rev = repo_rev.partition("@")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id):
        raise ValueError(f"--hf wants ORG/REPO[@REVISION][=NAME], got {spec!r}")
    name = name or repo_id.split("/")[1]
    if not re.fullmatch(r"[\w.-]+", name):
        raise ValueError(f"bad --hf folder name {name!r}")
    return repo_id, rev or None, name


def parse_push(spec: str) -> tuple[str, str]:
    local, sep, remote = spec.rpartition(":")
    if not sep or not local or not remote:
        raise ValueError(f"--push wants LOCAL:REMOTE, got {spec!r}")
    return local, remote


def remote_path(path: str) -> str:
    """A remote path as seen from $HOME (where ssh starts): absolute stays, ``~/x`` is home-relative,
    anything else is relative to the pushed repo."""
    if path.startswith("/"):
        return path
    return path[2:] if path.startswith("~/") else f"{REMOTE_REPO}/{path}"


def fetch_dest(path: str, out: Path) -> Path:
    """Where a fetched path lands: repo-relative paths keep their layout under ``out``; absolute and
    ``~/`` paths land at its top."""
    p = path.rstrip("/")
    return out if p.startswith(("/", "~/")) else out / os.path.dirname(p)


def shippable(path: str) -> bool:
    """Never push secrets, environments or VCS internals, even if someone tracked them."""
    parts = Path(path).parts
    return not any(p in (".git", ".venv", "__pycache__") or p.startswith(".env") for p in parts) \
        and not path.endswith(".pyc")


def repo_files(repo: Path = REPO) -> list[str]:
    """What travels: tracked files plus untracked-but-not-ignored ones, as they are on disk now."""
    def ls(*extra: str) -> list[str]:
        out = subprocess.run(["git", "-C", str(repo), "ls-files", "-z", *extra], capture_output=True,
                             check=True, timeout=120, stdin=subprocess.DEVNULL).stdout
        return [p for p in out.decode().split("\0") if p]

    files = sorted(set(ls()) | set(ls("--others", "--exclude-standard")))
    return [f for f in files if shippable(f) and (repo / f).is_file()]


def code_version(repo: Path) -> dict:
    """``provenance.code_version`` of the repo being pushed: {commit, dirty, diff_sha256, source}."""
    return provenance.code_version(repo)


def save_code_changes(out: Path) -> dict:
    """``provenance.save_changes``: the stamp, and the uncommitted changes' patch and untracked list in ``out``."""
    return provenance.save_changes(out, REPO)


def code_problem(code: dict) -> str | None:
    """Why ``code`` may not be pushed without --allow-dirty (None: it may)."""
    if code["dirty"] is False:
        return None
    if code["dirty"]:
        return "it has uncommitted changes (tracked changes, or untracked files that would travel)"
    return "its commit can't be told (not a git checkout, or git failed)"


def describe_code(code: dict) -> str:
    if not code.get("commit"):
        return "code version unknown"
    changes = f" + uncommitted changes (patch sha256 {(code['diff_sha256'] or '?')[:12]})" if code["dirty"] else ""
    return f"commit {code['commit'][:12]}{changes}"


def check_code(allow_dirty: bool) -> dict:
    """The launch check, before anything is created: the repo's code version, or SystemExit if it has uncommitted
    changes (or no knowable commit) and ``allow_dirty`` is not set."""
    code = code_version(REPO)
    problem = code_problem(code)
    if problem and not allow_dirty:
        raise SystemExit(f"refusing to push {REPO}: {problem}. Commit the changes, or pass --allow-dirty to push them "
                         f"anyway; their patch is then saved in --out ({provenance.PATCH_FILE}).")
    log(f"code: {describe_code(code)}" + (f" ({problem}; --allow-dirty)" if problem else ""))
    return code


def detached(script: str, step: str, cwd: str | None = None) -> str:
    """A remote shell line that starts ``script`` fully detached from the SSH session and returns at
    once. Output goes to ~/nebius_job/<step>.log; the exit code lands in ~/nebius_job/<step>.exit (written
    via a rename, so a reader never sees half of it). Only the ``setsid`` command is backgrounded (``;``
    separators, ``cd`` inside), so no subshell keeps the SSH channel open."""
    logs = f'"$HOME/{REMOTE_LOGS}"'
    cd = f"cd {shlex.quote(cwd)} || exit 1\n" if cwd else ""  # its own line: `cd x && script` guards line 1 only
    inner = (f"( {cd}{script}\n); rc=$?; echo $rc > {logs}/{step}.exit.tmp && "
             f"mv {logs}/{step}.exit.tmp {logs}/{step}.exit")
    return (f"mkdir -p {REMOTE_LOGS}; rm -f {REMOTE_LOGS}/{step}.exit; "
            f"setsid nohup bash -lc {shlex.quote(inner)} > {REMOTE_LOGS}/{step}.log 2>&1 < /dev/null &")


def poll_command(step: str, offset: int) -> str:
    """Exit file FIRST, then the log from ``offset``: once the exit code shows, the log read after it
    is complete."""
    return (f"cat {REMOTE_LOGS}/{step}.exit 2>/dev/null; echo {SEP}; "
            f"tail -c +{offset + 1} {REMOTE_LOGS}/{step}.log 2>/dev/null | head -c 4194304; exit 0")


def parse_poll(raw: bytes) -> tuple[int | None, bytes]:
    head, sep, tail = raw.partition((SEP + "\n").encode())
    if not sep:
        raise ValueError("poll output lacks the separator")
    head = head.strip()
    return (int(head) if head else None), tail


def display_lines(text: str) -> list[str]:
    """Lines as a terminal would show them: a progress bar's ``\\r`` updates collapse to the last."""
    return [line.rstrip("\r").rsplit("\r", 1)[-1] for line in text.split("\n")]


def bootstrap_script() -> str:
    return f"""set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
if command -v cloud-init >/dev/null; then cloud-init status --wait >/dev/null 2>&1 || true; fi
if ! mountpoint -q {RAM_DIR}; then
  if sudo -n true 2>/dev/null; then
    sudo mkdir -p {RAM_DIR} && sudo mount -t tmpfs -o size=90%,mode=1777 tmpfs {RAM_DIR}
  else
    echo "WARNING: no passwordless sudo, so {RAM_DIR} is NOT a RAM disk" >&2; mkdir -p {RAM_DIR}
  fi
fi
sudo -n loginctl enable-linger "$USER" 2>/dev/null || true
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
command -v uv >/dev/null || {{ echo "uv install failed" >&2; exit 1; }}
uv --version
echo "ram disk: $(df -h {RAM_DIR} | tail -1)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "no nvidia-smi"
sudo -n shutdown --show 2>&1 | head -1 || true
"""


def download_script(hf: list[tuple[str, str | None, str]]) -> str:
    """Every ``--hf`` repo downloads at once (one process each), each timed; the step fails if any download does."""
    lines = ["set -uo pipefail", 'export PATH="$HOME/.local/bin:$PATH"',
             "export HF_XET_HIGH_PERFORMANCE=1 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HF_HUB_DISABLE_PROGRESS_BARS=1",
             "pids=()"]
    for repo_id, rev, name in hf:
        dest = shlex.quote(f"{RAM_DIR}/{name}")
        rev_arg = f" --revision {shlex.quote(rev)}" if rev else ""
        get = f'uvx --from "huggingface_hub[hf_xet]" hf download {shlex.quote(repo_id)}{rev_arg} --local-dir {dest}'
        lines.append(f"( t0=$(date +%s); {get} > /dev/null && echo \"{name}: $(du -sh {dest} | cut -f1) in "
                     f"$(( $(date +%s) - t0 ))s\" ) & pids+=($!)")
    lines += ['rc=0; for pid in "${pids[@]}"; do wait "$pid" || rc=1; done',
              f"df -h {RAM_DIR} | tail -1", "exit $rc"]
    return "\n".join(lines)


def is_cpu_platform(platform: str) -> bool:
    """Nebius's CPU-only platforms (``cpu-d3``, ``cpu-e2``): no GPU, and no capacity advice for their presets."""
    return platform.startswith("cpu-")


SECRETS_DIR = "~/.secrets"  # this machine's keys, a file each: hub's store (`hub-secret set NAME` writes NAME there)
SECRET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")  # hub-secret's names
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
LAUNCHER_ENV = {"PATH", "HOME", "PYTHONUNBUFFERED", "NEBIUS_JOB_RAM", "NEBIUS_JOB_DEADLINE", "HF_XET_HIGH_PERFORMANCE",
                "HF_TOKEN_PATH", "UV_NO_SYNC"}  # set by the job script itself


@dataclass(frozen=True)
class Secret:
    """One ``--secret NAME[=ENV]``: the key in ``SECRETS_DIR``/NAME becomes the job's $ENV. Holds where the key is,
    never the key."""
    name: str
    env: str
    path: Path


def parse_secret(spec: str) -> Secret:
    """``NAME[=ENV]`` -> Secret; ENV defaults to NAME upper-cased. A bad ENV is not echoed: it may be a key typed by
    mistake (``--secret NAME=VALUE``, as for ``--env``)."""
    name, sep, env = spec.partition("=")
    if not SECRET_NAME.fullmatch(name):
        raise ValueError(f"--secret wants NAME[=ENV], NAME a file in {SECRETS_DIR} (letters, digits and . _ -, as "
                         f"hub-secret names them), got the name {name!r}")
    env = env if sep else name.upper()
    if not ENV_NAME.fullmatch(env):
        raise ValueError(f"--secret {name}: " + (
            "after = comes the name of the job's environment variable (letters, digits and _), not a key: the key "
            "is read from the file" if sep else
            f"{env!r} is not an environment variable name; name one: --secret {name}=ENV"))
    return Secret(name, env, Path(SECRETS_DIR).expanduser() / name)


def job_secrets(specs: list[str], taken: set[str]) -> list[Secret]:
    """The ``--secret`` flags: each NAME and each ENV once, and no ENV that ``--env`` (``taken``) or the launcher
    already sets."""
    out: list[Secret] = []
    for spec in specs:
        s = parse_secret(spec)
        if any(o.name == s.name for o in out):
            raise ValueError(f"--secret {s.name} is given twice")
        if s.env in taken or s.env in LAUNCHER_ENV or any(o.env == s.env for o in out):
            raise ValueError(f"--secret {s.name}: ${s.env} is already set (by --env, another --secret or the launcher)")
        out.append(s)
    return out


def secret_problem(path: Path) -> str | None:
    """Why the key file ``path`` can't be sent (None: it can). It is read to tell an empty one, and never shown."""
    try:
        if not path.is_file():
            return "missing" if not path.exists() else "not a file"
        return None if path.read_bytes().strip() else "empty"
    except OSError as e:
        return f"unreadable ({e.strerror})"


REMOTE_SECRETS = f"{RAM_DIR}/secrets"  # tmpfs, 0700: each key's file, from right before the job to its end


def remote_secret(name: str) -> str:
    return f"{REMOTE_SECRETS}/{name}"


def secret_reads(secrets: list[Secret]) -> str:
    """Shell lines that read each key from its file on the VM into its variable, exported for the job. A missing file
    fails the job before its command runs."""
    return "".join(f'{s.env}="$(cat {shlex.quote(remote_secret(s.name))})" || exit 1\nexport {s.env}\n'
                   for s in secrets)


UV_PYTHON_DIR = "/opt/uv-python"  # --root-job: uv's Python (UV_PYTHON_INSTALL_DIR), where every user can run it
DEFAULT_SANDBOX_PYTHON = "3.11.16"  # the RL runs' CPython
# How a root job runs: as root, with the calling step's environment (-E; sudo still drops a few names, e.g. PYTHONPATH
# and LD_*), root's HOME (-H: what root writes stays out of the login user's home), and the step's PATH (sudo would
# put its secure_path in its place). sudo keeps the working directory.
ROOT_SUDO = 'sudo -n -E -H env "PATH=$PATH"'
# --root-job's setup check, run the way the job runs (ROOT_SUDO, the venv's python, in the repo); argv: the version and
# a file of the repo. The env's own sandbox check (root, and an interpreter its sandbox user can run), the exact
# Python, and that the sandbox user cannot read the repo.
ROOT_CHECK = """\
import os, subprocess, sys
from monitordecorrelation.envs.terminal_verifier import SANDBOX_USER, _sandbox_ids
version, repo_file, here = sys.argv[1], sys.argv[2], sys.version.split()[0]
uid, gid, python = _sandbox_ids()  # raises unless root, and unless the sandbox user can run this interpreter
print(f"root job: euid {os.geteuid()}, venv Python {here}; the env runs commands as {SANDBOX_USER} ({uid}:{gid}) "
      f"with {python}")
if here != version:
    sys.exit(f"root job: the venv runs Python {here}, not --sandbox-python {version}")
peek = subprocess.run([python, "-c", "import sys; open(sys.argv[1]).close()", repo_file], user=uid, group=gid,
                      extra_groups=[], cwd="/", capture_output=True)
if peek.returncode == 0:
    sys.exit(f"root job: {SANDBOX_USER} can read {repo_file}: the repo must be private to its owner")
print(f"root job: {SANDBOX_USER} cannot read the repo")
"""


def root_command(body: str) -> str:
    """A shell line that runs ``body`` as root (``ROOT_SUDO``) in the current directory."""
    return f"{ROOT_SUDO} bash -c {shlex.quote(body)}"


# A machine without a GPU (a plain OS image) may lack a C compiler, and a locked package builds from source (reasoning-gym's
# pycosat, 2026-10-23). The GPU images have one, as Vlad's container does.
BUILD_TOOLS = ("command -v cc >/dev/null || { sudo -n apt-get update -qq && sudo -n env DEBIAN_FRONTEND=noninteractive "
               "apt-get install -y -qq build-essential >/dev/null; }")


def setup_script(extra_gpu: bool, gpu: bool = True, sandbox_python: str | None = None) -> str:
    """The setup step, in the pushed repo: ``uv sync --locked``, then
    - with a GPU: check that torch sees it (else retarget torch with scripts/setup_box.sh), and show it;
    - without one (``cpu-*`` platforms, ``exec --no-gpu``): show the Python version;
    - with ``sandbox_python`` (``--root-job``): the sync runs on uv's CPython of exactly that version, installed into
      ``UV_PYTHON_DIR``, where every user can run it (the env's sandbox user runs the policy's commands with it).
      Then $HOME, and so the repo, is made private, and the step fails unless ``ROOT_CHECK`` passes run as the job
      will be (``ROOT_SUDO``)."""
    extra = " --extra gpu" if extra_gpu else ""
    lines = ["set -euo pipefail", 'export PATH="$HOME/.local/bin:$PATH"']
    if not gpu:
        lines += [BUILD_TOOLS, 'echo "cc: $(cc --version | head -1)"']
    if sandbox_python is None:
        lines.append(f"uv sync --locked{extra}")
    else:
        v = sandbox_python
        lines += ["umask 022  # uv's Python must be readable and runnable by every user",
                  f'sudo -n install -d -m 755 -o "$(id -un)" {UV_PYTHON_DIR}',
                  f"export UV_PYTHON_INSTALL_DIR={UV_PYTHON_DIR}",
                  f"uv python install {v}",
                  f'case "$(readlink -f .venv/bin/python 2>/dev/null)" in {UV_PYTHON_DIR}/*) ;; *) rm -rf .venv ;; '
                  "esac  # a venv on another interpreter is rebuilt",
                  f"uv sync --locked{extra} --managed-python --python {v}"]
    if gpu:
        lines += ["if ! uv run --no-sync python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'"
                  "; then",
                  '  echo "torch cannot see the GPU with the locked wheel: retargeting via scripts/setup_box.sh"',
                  f"  bash scripts/setup_box.sh{extra}",
                  "fi",
                  "uv run --no-sync python -c 'import torch as t; print(\"torch\", t.__version__, t.version.cuda, "
                  "t.cuda.get_device_name(0))'"]
    elif sandbox_python is None:
        lines.append("uv run --no-sync python -c 'import sys; print(\"python\", sys.version.split()[0], "
                     "\"(no GPU: no torch/CUDA check)\")'")
    if sandbox_python is not None:
        lines += ['chmod 750 "$HOME"  # the repo private: the policy\'s commands run as another user',
                  'echo "root job: $(. /etc/os-release && echo "$PRETTY_NAME"), $(getconf GNU_LIBC_VERSION), '
                  '/bin/sh -> $(readlink -f /bin/sh)"',
                  f'{ROOT_SUDO} .venv/bin/python -c {shlex.quote(ROOT_CHECK)} {v} "$PWD/pyproject.toml"']
    return "\n".join(lines) + "\n"


def job_script(command: str, env: list[tuple[str, str]], poweroff_minutes: int | None = None,
               deadline_fallback: int | None = None, *, secrets: list[Secret] = (), root: bool = False) -> str:
    """The job step.

    - It exports ``NEBIUS_JOB_DEADLINE``, the VM's scheduled shutdown (the ``--hours`` cap) in epoch seconds, and
      saves it for the upload step. A job can then stop cleanly before the cap; extract_probe_features.py does.
      If systemd's record of the shutdown can't be read, ``deadline_fallback`` is used instead: the launcher's
      estimate, VM creation + ``--hours``, a little before the real shutdown.
    - With ``poweroff_minutes``, the VM powers itself off that long after the job ends, unless a sooner shutdown is
      already scheduled. So if the launcher has died, the VM stops on its own instead of running until ``--hours``.
    - ``secrets`` (``--secret``): each key is read from its file on the RAM disk as the job starts (``secret_reads``:
      the script holds paths, never a key), and the files are deleted when the step ends, however it ends (EXIT trap).
    - ``root`` (``--root-job``): the command runs as root (``ROOT_SUDO``) with the step's environment, the ``--env``
      values again (sudo drops a few names even with -E), the keys (read by root) and UV_NO_SYNC=1, so a ``uv run``
      in it cannot re-sync the venv as root. The deadline, the power-off and the exit code stay this step's.
    """
    exports = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in env)
    head = (f'export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1 NEBIUS_JOB_RAM={RAM_DIR} '
            f'HF_XET_HIGH_PERFORMANCE=1 HF_TOKEN_PATH="{REMOTE_HF_TOKEN}"\n{exports}'
            + (f"trap {shlex.quote(shlex.join(['rm', '-f', '--', *(remote_secret(s.name) for s in secrets)]))} EXIT\n"
               if secrets else "")
            + f'usec=$(sudo -n cat {SCHEDULED_SHUTDOWN} 2>/dev/null | sed -n "s/^USEC=//p")\n'
            + (f'[ -n "$usec" ] || usec={int(deadline_fallback)}000000\n' if deadline_fallback else "")
            + 'if [ -n "$usec" ]; then\n'
            '  export NEBIUS_JOB_DEADLINE=$((usec / 1000000))\n'
            f'  echo "$NEBIUS_JOB_DEADLINE" > "$HOME/{REMOTE_LOGS}/deadline"\n'
            'fi\n')
    body = secret_reads(secrets) + command
    if root:
        body = root_command(f"export UV_NO_SYNC=1\n{exports}{body}")
    if poweroff_minutes is None:
        return head + body
    m = int(poweroff_minutes)
    return (f"{head}(\n{body}\n)\nrc=$?\n"
            f"want=$(( $(date +%s) + {m} * 60 ))\n"
            f'cur=$(sudo -n cat {SCHEDULED_SHUTDOWN} 2>/dev/null | sed -n "s/^USEC=//p")\n'
            'if [ -z "$cur" ] || [ $((cur / 1000000)) -gt "$want" ]; then\n'
            f'  sudo -n shutdown -h +{m} >/dev/null 2>&1 && '
            f'echo "nebius_job: the VM powers itself off in {m} min unless it is deleted first"\n'
            'fi\n'
            'exit $rc')


def job_command(cmd: list[str]) -> str:
    return cmd[0] if len(cmd) == 1 else shlex.join(cmd)


def expand_out(template: str, name: str, stamp: str | None = None) -> Path:
    return Path(expand(template, name, stamp)).expanduser()


def expand(template: str, name: str, stamp: str | None = None) -> str:
    stamp = stamp or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return template.replace("{stamp}", stamp).replace("{name}", name)


_BUCKET_DEST = re.compile(r"hf://buckets/[\w.-]+/[\w.-]+(/[^\s]*)?")


def parse_upload(spec: str) -> tuple[str, str]:
    """``REMOTE_PATH=hf://buckets/NAMESPACE/BUCKET[/PREFIX]`` -> (remote path, destination). A directory is
    synced into the prefix; a file is copied to it."""
    src, sep, dest = spec.partition("=")
    if not sep or not src or not _BUCKET_DEST.fullmatch(dest.rstrip("/")):
        raise ValueError(f"--upload wants REMOTE_PATH=hf://buckets/NAMESPACE/BUCKET[/PREFIX], got {spec!r}")
    return src, dest.rstrip("/")


REMOTE_HF_TOKEN = f"{RAM_DIR}/.hf_token"  # tmpfs; huggingface_hub reads it via HF_TOKEN_PATH. Job start → upload end


def upload_script(uploads: list[tuple[str, str]]) -> str:
    """Sync each path to its bucket destination with the repo venv's ``hf`` CLI, then delete the token
    (``trap``: also when an upload fails). It first restores the ``--hours`` shutdown that the job's end replaced
    with a sooner power-off: the launcher is evidently alive."""
    deadline = f'"$HOME/{REMOTE_LOGS}/deadline"'
    lines = ["set -uo pipefail", f"trap 'rm -f \"{REMOTE_HF_TOKEN}\"' EXIT",
             f'export PATH="$HOME/.local/bin:$PATH" HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 '
             f'HF_TOKEN_PATH="{REMOTE_HF_TOKEN}"',
             # the launcher is alive: back to the --hours shutdown, which the job's end moved closer
             f'if [ -s {deadline} ]; then m=$(( ($(cat {deadline}) - $(date +%s)) / 60 )); '
             f'[ "$m" -gt 0 ] && sudo -n shutdown -h "+$m" >/dev/null 2>&1; fi',
             'HF="$HOME/md/.venv/bin/hf"', "rc=0"]
    for src, dest in uploads:
        # the step runs in the repo: relative paths are the repo's, ~/ is the remote home, absolute stays
        path = f'"$HOME"/{shlex.quote(src[2:])}' if src.startswith("~/") else shlex.quote(src)
        q_dest = shlex.quote(dest)
        lines.append(f'if [ -d {path} ]; then "$HF" buckets sync {path} {q_dest} || rc=1; '
                     f'elif [ -e {path} ]; then "$HF" buckets cp {path} {q_dest} || rc=1; '
                     f"else printf 'upload: %s does not exist\\n' {path}; rc=1; fi")
    # the job is over: back to a power-off soon, in case the launcher dies before deleting the VM
    lines.append(f"sudo -n shutdown -h +{POWEROFF_AFTER_JOB_MINUTES} >/dev/null 2>&1 || true")
    return "\n".join([*lines, "exit $rc"])


# ------------------------------------------------------------------------------------- Nebius calls


class Nebius:
    """The handful of ``nebius`` CLI calls we need. Each has a timeout and no stdin, so an expired
    federation login fails loudly instead of waiting for a browser."""

    def __init__(self, binary: str, project_id: str) -> None:
        self.binary, self.project_id = binary, project_id

    def _run(self, args: list[str], timeout: float = 900) -> str:
        cp = subprocess.run([self.binary, *args], capture_output=True, text=True, timeout=timeout,
                            stdin=subprocess.DEVNULL, start_new_session=True, check=False)
        if cp.returncode != 0:
            raise InfraError(f"nebius {' '.join(args[:3])} failed ({cp.returncode}): {cp.stderr.strip()[-2000:]}")
        return cp.stdout

    def create(self, argv: list[str]) -> Instance:
        return parse_instance(json.loads(self._run(argv[1:], timeout=1800)))

    def get(self, instance_id: str) -> Instance:
        return parse_instance(json.loads(self._run(["compute", "instance", "get", "--id", instance_id,
                                                    "--format", "json"], timeout=120)))

    def instances(self) -> list[Instance]:
        return [parse_instance(d) for d in parse_items(self._run(
            ["compute", "instance", "list", "--parent-id", self.project_id, "--format", "json"], timeout=120))]

    def disks(self) -> list[Disk]:
        return [parse_disk(d) for d in parse_items(self._run(
            ["compute", "disk", "list", "--parent-id", self.project_id, "--format", "json"], timeout=120))]

    def delete_instance(self, instance_id: str) -> None:
        self._run(["compute", "instance", "delete", "--id", instance_id], timeout=1800)

    def delete_disk(self, disk_id: str) -> None:
        self._run(["compute", "disk", "delete", "--id", disk_id], timeout=900)

    def estimate_hourly(self, platform: str, preset: str, spot: bool) -> float:
        """The cost calculator's USD/hour for one such VM in this project's region. For spot, that is the
        current spot price, which moves every 15 minutes."""
        pricing = (["--resource-spec-compute-instance-spec-follows-spot-price",
                    "--resource-spec-compute-instance-spec-preemptible-on-preemption", "stop"] if spot
                   else ["--resource-spec-compute-instance-spec-on-demand"])
        out = self._run(["billing", "v1alpha1", "calculator", "estimate",
                         "--resource-spec-compute-instance-spec-parent-id", self.project_id,
                         "--resource-spec-compute-instance-spec-resources-platform", platform,
                         "--resource-spec-compute-instance-spec-resources-preset", preset,
                         *pricing, "--format", "json"], timeout=120)
        return float(json.loads(out)["hourly_cost"]["general"]["total"]["cost"])

    def pricing_policies(self) -> list[dict]:
        return parse_items(self._run(["billing", "pricing-policy", "list", "--parent-id", self.project_id,
                                      "--format", "json"], timeout=120))

    def create_pricing_policy(self, name: str, platform: str, max_gpu_price: float) -> str:
        out = self._run(["billing", "pricing-policy", "create", "--parent-id", self.project_id, "--name", name,
                         "--compute-instance-spec-v1-platform", platform,
                         "--pricing-max-price-v1-max-price", f"{max_gpu_price:.3f}", "--format", "json"], timeout=600)
        return json.loads(out)["metadata"]["id"]

    def capacity(self, tenant: str) -> list[dict]:
        """Capacity advice for every region, platform and preset of the tenant (not a reservation)."""
        return parse_items(self._run(["capacity", "resource-advice", "list", "--parent-id", tenant,
                                      "--format", "json"], timeout=120))


# ------------------------------------------------------------------------------------ regions + prices


@dataclass
class Region:
    """Where a VM can go: one project in one region, its subnet, and that region's public-image project."""
    name: str
    project: str
    subnet: str
    image_project: str


def load_regions(path: str) -> tuple[str | None, list[Region]]:
    """A regions file: ``{"tenant": ID, "regions": {NAME: {"project", "subnet", "image_project"}}}``. The
    tenant is only needed for capacity advice."""
    data = json.loads(Path(path).expanduser().read_text())
    regions = [Region(name, r["project"], r["subnet"], r["image_project"]) for name, r in data["regions"].items()]
    if not regions:
        raise SystemExit(f"{path}: no regions")
    return data.get("tenant"), regions


@dataclass
class Offer:
    """One region's price and capacity for the requested VM, when it was checked."""
    region: str
    price: float | None = None  # USD/hour for the whole VM: the spot price, or on-demand
    level: str | None = None  # capacity advice: HIGH / MEDIUM / LOW / LIMIT_REACHED (None: unknown)
    available: int | None = None  # quota left for this preset (None: unknown)
    note: str = ""  # why there is no price, or that the region does not offer the platform

    def skip_reason(self, max_price: float | None) -> str | None:
        if self.note:
            return self.note
        if self.price is None:
            return "no price"
        if max_price is not None and self.price > max_price:
            return f"above the ${max_price:g}/h cap"
        if self.level == "LIMIT_REACHED":
            return "no capacity (advice: LIMIT_REACHED)"
        if self.available == 0:
            return "quota used up"
        return None


def capacity_of(items: list[dict], region: str, platform: str, preset: str, spot: bool
                ) -> tuple[bool, str | None, int | None]:
    """(offered, level, available) for one region/platform/preset from ``capacity resource-advice list``."""
    for item in items:
        spec = item.get("spec") or {}
        ci = spec.get("compute_instance") or {}
        if (spec.get("region"), ci.get("platform"), (ci.get("preset") or {}).get("name")) == (region, platform, preset):
            status = (item.get("status") or {}).get("preemptible" if spot else "on_demand") or {}
            level = (status.get("availability_level") or "").removeprefix("AVAILABILITY_LEVEL_") or None
            available = status.get("available")
            return True, level, int(available) if available is not None else None
    return False, None, None


def survey(nebius: str, regions: list[Region], tenant: str | None, platform: str, preset: str,
           spot: bool) -> list[Offer]:
    """Price and capacity of the VM in every region. Read-only: the cost calculator and capacity advice.

    The advice covers GPU presets only: a CPU preset it doesn't list is still priced, its capacity and quota left
    unknown (a region without the platform fails the calculator or the creation, and is skipped)."""
    capacity = None
    if tenant:
        try:
            capacity = Nebius(nebius, regions[0].project).capacity(tenant)
        except InfraError as e:
            log(f"capacity advice unavailable, choosing by price alone: {e}")
    offers = []
    for region in regions:
        offer = Offer(region.name)
        if capacity is not None:
            offered, offer.level, offer.available = capacity_of(capacity, region.name, platform, preset, spot)
            if not offered and not is_cpu_platform(platform):
                offer.note = f"{platform}/{preset} not offered here"
                offers.append(offer)
                continue
        try:
            offer.price = Nebius(nebius, region.project).estimate_hourly(platform, preset, spot)
        except (InfraError, ValueError, KeyError, TypeError) as e:
            offer.note = f"no price ({str(e)[:120]})"
        offers.append(offer)
    return offers


def candidates(offers: list[Offer], max_price: float | None) -> list[Offer]:
    """The regions to try, cheapest first. Ties go to more capacity."""
    rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    ok = [o for o in offers if o.skip_reason(max_price) is None]
    return sorted(ok, key=lambda o: (o.price, rank.get(o.level or "", 3)))


def offer_table(offers: list[Offer], max_price: float | None, chosen: str | None) -> str:
    lines = [f"  {'region':12} {'$/hour':>8}  {'capacity':13} {'quota':>5}  note"]
    for o in offers:
        price = f"{o.price:.2f}" if o.price is not None else "-"
        note = "<- cheapest, tried first" if o.region == chosen else (o.skip_reason(max_price) or "")
        lines.append(f"  {o.region:12} {price:>8}  {o.level or '?':13} "
                     f"{o.available if o.available is not None else '?':>5}  {note}")
    return "\n".join(lines)


def gpu_count(preset: str) -> int:
    m = re.match(r"(\d+)gpu", preset)
    return int(m.group(1)) if m else 1


def machine_label(spec: VmSpec) -> str:
    """The VM in the ledger and logs: GPUs × platform, or a CPU VM's platform/preset (it has no GPU to count)."""
    if is_cpu_platform(spec.platform):
        return f"{spec.platform}/{spec.preset}"
    return f"{gpu_count(spec.preset)}× {spec.platform}"


def policy_name(platform: str, max_gpu_price: float) -> str:
    return f"md-cap-{platform}-{max_gpu_price:.3f}".replace(".", "p")


def ensure_policy(nb: "Nebius", platform: str, max_gpu_price: float) -> str:
    """The id of an active pricing policy in ``nb``'s project bidding ``max_gpu_price`` USD per GPU-hour for
    ``platform``; created if missing. Nebius never charges a VM on it more than the bid. Above the bid it does not
    schedule (and preempts), so a price spike cannot run up the bill."""
    for p in nb.pricing_policies():
        spec, status = p.get("spec") or {}, p.get("status") or {}
        plat = ((spec.get("compute_instance_spec") or {}).get("v1") or {}).get("platform")
        bid = ((spec.get("pricing") or {}).get("max_price_v1") or {}).get("max_price")
        if plat == platform and bid is not None and abs(float(bid) - max_gpu_price) < 1e-9 \
                and status.get("state", "STATE_ACTIVE") == "STATE_ACTIVE":
            return p["metadata"]["id"]
    log(f"creating pricing policy {policy_name(platform, max_gpu_price)} (bid ${max_gpu_price:g}/GPU-h)")
    return nb.create_pricing_policy(policy_name(platform, max_gpu_price), platform, max_gpu_price)


class BudgetExceeded(SystemExit):
    """A VM's worst case does not fit in what is left of the budget."""


class Budget:
    """A spending ledger for the VMs this launcher creates: ``{"limit_usd": L, "entries": [...]}``.

    - **Before a VM is created,** its worst case is checked and reserved. With a bid, that is bid × GPUs × --hours;
      without one, it is the on-demand price, which spot never exceeds.
    - **Afterwards,** the reservation is settled at the estimated cost from the sampled prices.
    - **A kept VM** (``up``, ``run --keep``) stays reserved at its worst case until ``down`` deletes it (with the
      ledger) or ``reap`` finds it gone.
    - **Reservations** also record the launcher's pid and host, and the VM's region, so ``reap`` can settle them if
      the launcher dies.
    - **vast.ai instances** (``record``) are entries with ``"provider": "vast"``, reserved at their worst case and
      settled like the rest, but neither checked against ``limit_usd`` nor counted in ``committed``: vast.ai is billed
      to another account, within the limits hub sets for this project (``hub-gpu limits``), which the launcher checks
      instead. ``reap`` leaves them alone.
    - The file is re-read before every change, so launchers running side by side don't overwrite each other."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser()
        self._load()

    def _load(self) -> None:
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"limit_usd": None, "entries": []}

    @contextlib.contextmanager
    def locked(self):
        """Exclusive across launchers: re-read, change, save."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                self._load()
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @property
    def limit(self) -> float | None:
        return self.data.get("limit_usd")

    def committed(self) -> float:
        return round(sum(e["usd"] for e in self.data["entries"] if e.get("provider") != "vast"), 2)

    def vast_total(self) -> float:
        return round(sum(e["usd"] for e in self.data["entries"] if e.get("provider") == "vast"), 2)

    def _check(self, worst: float, what: str) -> None:
        if self.limit is not None and self.committed() + worst > self.limit + 1e-9:
            raise BudgetExceeded(f"budget: {what} could cost up to ${worst:.2f}, but ${self.committed():.2f} of the "
                                 f"${self.limit:g} budget is already spent or reserved ({self.path})")

    def check(self, worst: float, what: str) -> None:
        with self.locked():
            self._check(worst, what)

    def reserve(self, job: str, worst: float, what: str, **info) -> None:
        """Check and reserve in one locked step, so two launchers can't both take the last dollars."""
        with self.locked():
            self._check(worst, what)
            self.data["entries"].append({"when": utc(), "job": job, "what": what, "kind": "reserved",
                                         "usd": round(worst, 2), **info})
            self.save()

    def record(self, job: str, worst: float, what: str, **info) -> None:
        """A vast.ai instance's entry, reserved at its worst case for the record: not checked, not counted."""
        with self.locked():
            self.data["entries"].append({"when": utc(), "job": job, "what": what, "kind": "reserved",
                                         "usd": round(worst, 2), "provider": "vast", **info})
            self.save()

    def annotate(self, job: str, **info) -> None:
        with self.locked():
            for e in self.data["entries"]:
                if e.get("job") == job and e.get("kind") == "reserved":
                    e.update(info)
            self.save()

    def settle(self, job: str, usd: float, note: str) -> None:
        with self.locked():
            for e in self.data["entries"]:
                if e.get("job") == job and e.get("kind") == "reserved":
                    e.update(kind="spent", usd=round(usd, 2), note=note, settled=utc())
            self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1) + "\n")
        tmp.replace(self.path)

    def summary(self) -> str:
        lim = "no limit" if self.limit is None else f"limit ${self.limit:g}"
        left = "" if self.limit is None else f", ${self.limit - self.committed():.2f} left"
        vast = any(e.get("provider") == "vast" for e in self.data["entries"])
        return (f"budget: ${self.committed():.2f} spent or reserved ({lim}{left})"
                + (f"; vast.ai, not counted (another account, hub-gpu limits): ${self.vast_total():.2f}" if vast
                   else ""))


def _seconds(stamp: str) -> float:
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp()


def estimated_cost(samples: list[list], end: str, *, conservative: bool = False,
                   cap_hourly: float | None = None) -> float | None:
    """USD from [[utc, $/hour], …] price samples up to ``end``.

    - **Point estimate:** each price holds until the next sample.
    - **``conservative``:** each interval is charged at the higher of its two bounding samples. With no closing
      sample, the last interval is charged at ``cap_hourly`` (the bid × GPUs) when known. This is what the budget
      settles at: a spot price that rose between samples, e.g. just before a bid-triggered preemption, is not
      under-counted.
    - **``cap_hourly``:** the bid × GPUs. Nebius never charges more than this, so every interval is capped at it."""
    if not samples:
        return None
    total = 0.0
    bounds = [*samples[1:], [end, None]]
    for (t0, price), (t1, nxt) in zip(samples, bounds):
        rate = price
        if conservative:
            rate = max(price, nxt) if nxt is not None else (cap_hourly if cap_hourly is not None else price)
        if cap_hourly is not None:
            rate = min(rate, cap_hourly)
        total += rate * max(0.0, _seconds(t1) - _seconds(t0)) / 3600
    return round(total, 2)


class PriceSampler:
    """Records the VM's price every ``interval`` seconds while it exists, so ``job.json`` can carry a cost
    estimate. The CLI cannot show what a VM actually cost, and spot prices move every 15 minutes."""

    def __init__(self, sample, samples: list, interval: float = 900.0, on_sample=None) -> None:
        self._sample, self.samples, self.interval, self._on_sample = sample, samples, interval, on_sample
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self, at: str, price: float) -> None:
        self.samples.append([at, price])
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample_now()

    def sample_now(self) -> None:
        try:
            self.samples.append([utc(), self._sample()])
            if self._on_sample is not None:
                self._on_sample()
        except Exception as e:  # noqa: BLE001 — a missed sample only coarsens the estimate
            log(f"price sample failed: {e}")

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)


def wait_running(nb: Nebius, inst: Instance, timeout: float = 900) -> Instance:
    deadline = time.time() + timeout
    while not (inst.state == "RUNNING" and inst.ip):
        if inst.state not in ALIVE_STATES:
            raise InfraError(f"instance {inst.id} is {inst.state} before it ever ran")
        if time.time() > deadline:
            raise InfraError(f"instance {inst.id} not RUNNING with a public IP after {timeout:.0f}s")
        time.sleep(10)
        inst = nb.get(inst.id)
    return inst


def teardown(nb: Nebius, name: str, instance_id: str | None = None, timeout: float = 300) -> None:
    """Delete the VM (by id, and by name in case creation died before we learned the id) and its
    managed boot disk, then verify that neither is left. Raises InfraError if anything remains.

    Every cloud call is retried until ``timeout``: one failed listing must not leave a VM running. If the listing
    never works, the known ``instance_id`` is still deleted (and the verification then raises)."""
    boot = f"{name}-boot"
    deadline = time.time() + timeout
    ids = {instance_id} if instance_id else set()
    while True:
        try:
            ids |= {i.id for i in nb.instances() if i.name == name}
            break
        except InfraError as e:
            if time.time() > deadline:
                if not ids:
                    raise
                log(f"  listing failed ({e}); deleting {sorted(ids)} by id")
                break
            log(f"  listing failed, retrying: {e}")
            time.sleep(15)
    for iid in sorted(ids):
        log(f"deleting instance {iid}")
        while True:  # e.g. a preempted VM still STOPPING refuses deletion for a while
            try:
                nb.delete_instance(iid)
                break
            except InfraError as e:
                if time.time() > deadline:
                    raise
                log(f"  delete failed, retrying: {e}")
                time.sleep(15)
    while True:
        try:
            left_i = [i for i in nb.instances() if i.name == name or i.id in ids]
            left_d = [d for d in nb.disks() if d.name == boot]
        except InfraError as e:
            if time.time() > deadline:
                raise
            log(f"  listing failed, retrying: {e}")
            time.sleep(15)
            continue
        if not left_i and not left_d:
            log(f"verified: no instance {name!r} and no disk {boot!r} left")
            return
        if time.time() > deadline:
            raise InfraError(f"teardown incomplete; delete by hand: instances={[(i.id, i.state) for i in left_i]} "
                             f"disks={[(d.id, d.state) for d in left_d]}")
        for d in left_d:  # the managed boot disk normally goes with the instance; mop up an orphan
            if not left_i and d.state.upper() != "DELETING":
                log(f"deleting orphaned boot disk {d.id}")
                with contextlib.suppress(InfraError):
                    nb.delete_disk(d.id)
        time.sleep(10)


# ---------------------------------------------------------------------------------------- SSH side


@dataclass
class Remote:
    host: str  # user@ip
    known_hosts: Path
    key: str | None = None
    port: int | None = None  # a vast.ai instance's SSH port (None: 22)

    def ssh(self) -> list[str]:
        """ssh and its options, without the host."""
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"UserKnownHostsFile={self.known_hosts}", "-o", "ConnectTimeout=15",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]
        if self.port is not None:
            argv += ["-p", str(self.port)]
        if self.key:
            argv += ["-i", self.key, "-o", "IdentitiesOnly=yes"]
        return argv

    def argv(self, command: str) -> list[str]:
        return self.ssh() + [self.host, command]

    def rsh(self) -> str:
        """rsync's ``-e``: the same ssh."""
        return shlex.join(self.ssh())

    def run(self, command: str, *, timeout: float = 600, stdin=None, check: bool = True) -> subprocess.CompletedProcess:
        """``command`` runs under ``bash -c`` whatever the remote login shell is."""
        cp = subprocess.run(self.argv(f"bash -c {shlex.quote(command)}"), capture_output=True, timeout=timeout,
                            stdin=stdin if stdin is not None else subprocess.DEVNULL, check=False)
        if check and cp.returncode != 0:
            raise InfraError(f"ssh {self.host} failed ({cp.returncode}): "
                             f"{cp.stderr.decode(errors='replace').strip()[-2000:]}")
        return cp

    def pipe_from(self, local_argv: list[str], command: str, timeout: float = 3600) -> None:
        """``local_argv | ssh host command`` (e.g. tar up a directory and unpack it remotely)."""
        with subprocess.Popen(local_argv, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL) as src:
            cp = subprocess.run(self.argv(f"bash -c {shlex.quote(command)}"), stdin=src.stdout,
                                capture_output=True, timeout=timeout, check=False)
            src.stdout.close()
            src_rc = src.wait()
        if src_rc != 0 or cp.returncode != 0:
            raise InfraError(f"push to {self.host} failed (local {src_rc}, remote {cp.returncode}): "
                             f"{cp.stderr.decode(errors='replace').strip()[-2000:]}")

    def pipe_to(self, command: str, local_argv: list[str], timeout: float = 3600) -> None:
        """``ssh host command | local_argv`` (e.g. tar up remote outputs and unpack them here)."""
        with tempfile.TemporaryFile() as errf, \
                subprocess.Popen(self.argv(f"bash -c {shlex.quote(command)}"), stdout=subprocess.PIPE,
                                 stderr=errf, stdin=subprocess.DEVNULL) as src:
            cp = subprocess.run(local_argv, stdin=src.stdout, capture_output=True, timeout=timeout, check=False)
            src.stdout.close()
            src_rc = src.wait()
            errf.seek(0)
            err = errf.read().decode(errors="replace")
        if src_rc != 0 or cp.returncode != 0:
            raise InfraError(f"fetch from {self.host} failed (remote {src_rc}, local {cp.returncode}): "
                             f"{(err + cp.stderr.decode(errors='replace')).strip()[-2000:]}")


@dataclass
class JobSpec:
    out: Path
    command: str | None = None  # shell text run in the pushed repo; None = no job (``up``)
    pushes: list[tuple[str, str]] = field(default_factory=list)
    hf: list[tuple[str, str | None, str]] = field(default_factory=list)
    fetches: list[str] = field(default_factory=list)
    uploads: list[tuple[str, str]] = field(default_factory=list)  # (remote path, hf://buckets/... destination)
    hf_token_file: Path | None = None  # on the launching machine; on the VM from the job's start until the upload ends
    env: list[tuple[str, str]] = field(default_factory=list)
    extra_gpu: bool = False
    poll_interval: float = 15.0
    poweroff_minutes: int | None = None  # the VM powers itself off this long after the job (run, not kept)
    allow_dirty: bool = False  # push uncommitted changes (their patch is saved in ``out``)
    secrets: list[Secret] = field(default_factory=list)  # --secret: on the VM from right before the job to its end
    gpu: bool = True  # the setup checks that torch sees the GPU (not on cpu-* platforms, nor with exec --no-gpu)
    root_job: bool = False  # --root-job: the job runs as root, on uv's CPython ``sandbox_python`` (see setup_script)
    sandbox_python: str = DEFAULT_SANDBOX_PYTHON
    provider: str = "nebius"  # "vast": a vast.ai instance (vast_bootstrap_script, fetch_rsync)
    ends: float | None = None  # vast.ai: when the instance is destroyed (epoch s), whatever runs on it
    reserve: float = 0.0  # vast.ai: seconds before ``ends`` kept for the fetch: the job's deadline is ends - reserve
    rsync: str = "rsync"  # vast.ai: this machine's rsync, for the fetch


def job_setup_script(job: JobSpec) -> str:
    return setup_script(job.extra_gpu, gpu=job.gpu, sandbox_python=job.sandbox_python if job.root_job else None)


def job_fields(job: JobSpec) -> dict:
    """What job.json records of the job beyond its flags: whether it runs as root, on which Python, and the key names
    (never a key)."""
    return {"root_job": job.root_job, "sandbox_python": job.sandbox_python if job.root_job else None,
            "secrets": {s.name: s.env for s in job.secrets}}


@dataclass
class Record:
    """job.json: what ran where, and how it ended."""
    mode: str
    name: str | None = None
    instance_id: str | None = None
    ip: str | None = None
    host: str | None = None
    spot: bool | None = None
    hours: float | None = None
    flags: dict = field(default_factory=dict)
    command: str | None = None
    times: dict = field(default_factory=dict)
    exit_code: int | None = None
    preempted: bool = False
    stopped_state: str | None = None
    errors: list = field(default_factory=list)
    fetched: dict = field(default_factory=dict)
    uploaded: dict = field(default_factory=dict)  # destination -> "ok" / "failed"
    kept: bool = False
    teardown_verified: bool | None = None
    region: str | None = None
    max_price: float | None = None
    offers: list = field(default_factory=list)  # every region's price/capacity at launch
    price_samples: list = field(default_factory=list)  # [utc, $/hour] while the VM existed
    estimated_cost_usd: float | None = None  # from price_samples; the console's billing has the real figure
    attempt: int = 1  # run --attempts: which VM this was
    creates: list = field(default_factory=list)  # every creation tried: region, price, start, minutes, error
    retries: list = field(default_factory=list)  # on attempt 1: the --out dirs of the later attempts
    code: dict | None = None  # the code version pushed, as written to the VM's CODE_VERSION.json
    root_job: bool = False  # --root-job: the job ran as root
    sandbox_python: str | None = None  # --root-job: the exact CPython of the venv (and of the env's sandbox)
    secrets: dict = field(default_factory=dict)  # --secret: key name -> the job's variable (names only, never a key)
    provider: str = "nebius"
    vast: dict = field(default_factory=dict)  # vast.ai: limits, offer, start answer, last hub-gpu list row, …

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / "job.json.tmp"
        tmp.write_text(json.dumps(asdict(self), indent=1) + "\n")
        tmp.replace(out / "job.json")


def wait_ssh(remote: Remote, check_vm=None, timeout: float = 600) -> None:
    deadline, tries = time.time() + timeout, 0
    while True:
        try:
            if remote.run("true", timeout=40, check=False).returncode == 0:
                return
        except subprocess.TimeoutExpired:
            pass
        tries += 1
        if check_vm:  # every try: tonight's preemptions all came mid-boot, and a stopped VM is billed until deleted
            check_vm()
        if time.time() > deadline:
            raise InfraError(f"no SSH to {remote.host} after {timeout:.0f}s")
        time.sleep(10)


def start(remote: Remote, step: str, script: str, cwd: str | None = None) -> None:
    remote.run(detached(script, step, cwd), timeout=120)


def poll(remote: Remote, step: str, out: Path, *, interval: float = 15.0, timeout: float | None = None,
         check_vm=None, max_failures: int = 12, during=None, every: float = 600.0) -> int:
    """Wait for a detached step to finish. Streams new log lines to stdout and appends them to
    ``out/<step>.log`` (so the log survives a preemption); returns the step's exit code. ``during`` (vast.ai's fetch
    so far) is called every ``every`` seconds while the step runs."""
    offset, failures, pending, shown = 0, 0, "", ""
    t0 = last = time.time()
    local_log = out / f"{step}.log"
    local_log.write_bytes(b"")
    while True:
        try:
            cp = remote.run(poll_command(step, offset), timeout=120, check=False)
            ok = cp.returncode != SSH_DOWN
        except subprocess.TimeoutExpired:
            ok = False
        if not ok:
            failures += 1
            log(f"{step}: ssh poll failed ({failures}/{max_failures})")
            if check_vm:
                check_vm()  # raises VmStopped if the VM is gone
            if failures >= max_failures:
                raise InfraError(f"lost SSH to {remote.host} while waiting for {step}")
            time.sleep(min(60.0, interval * failures))
            continue
        failures = 0
        code, new = parse_poll(cp.stdout)
        if new:
            offset += len(new)
            with local_log.open("ab") as f:
                f.write(new)
            lines = display_lines(pending + new.decode(errors="replace"))
            pending = lines.pop()
            for line in lines:
                print(f"  {step}| {line}", flush=True)
            if "\r" in new.decode(errors="replace") and pending.strip() and pending != shown:
                print(f"  {step}| {pending.strip()}", flush=True)  # live progress-bar state
                shown = pending
        if code is not None and len(new) >= 4194304:
            continue  # more log than one read returns: fetch the rest before finishing
        if code is not None:
            if pending.strip() and pending != shown:
                print(f"  {step}| {pending}", flush=True)
            log(f"{step} exited {code} after {time.time() - t0:.0f}s")
            return code
        if timeout and time.time() - t0 > timeout:
            raise InfraError(f"{step} still running after {timeout:.0f}s")
        if during is not None and time.time() - last >= every:
            during()
            last = time.time()
        time.sleep(interval)


def push_repo(remote: Remote, files: list[str]) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".lst", delete=False) as f:
        f.write("\0".join(files))
        listing = f.name
    try:
        remote.pipe_from(["tar", "-C", str(REPO), "--null", "-T", listing, "-czf", "-"],
                         f"mkdir -p {REMOTE_REPO} && tar -C {REMOTE_REPO} -xzf -")
    finally:
        os.unlink(listing)


def stamp_push(job: JobSpec, rec: Record) -> dict:
    """The code version of what is about to be pushed, taken now: the tree may have changed since launch, or since the
    last attempt. Uncommitted changes without --allow-dirty are refused (InfraError); with it, their patch and
    untracked list are saved in ``job.out``. Recorded in job.json as ``code``."""
    code = code_version(REPO)
    problem = code_problem(code)
    if problem and not job.allow_dirty:
        raise InfraError(f"not pushing {REPO}: {problem}, and --allow-dirty was not given")
    if code["dirty"] and code["source"] == "git":
        try:
            code = save_code_changes(job.out)  # stamps again, from the same bytes as the saved patch
        except Exception as e:  # noqa: BLE001 — pushing changes nobody could rebuild defeats --allow-dirty
            raise InfraError(f"could not save the uncommitted changes in {job.out}: {e}") from e
        if code["dirty"]:
            log(f"pushing uncommitted changes (--allow-dirty): their patch is {job.out / provenance.PATCH_FILE}")
    elif problem:
        log(f"pushing anyway (--allow-dirty): {problem}; with no git checkout, no patch can be saved")
    if not code["dirty"]:  # a patch left by an earlier push into the same --out is not this one's
        for name in (provenance.PATCH_FILE, provenance.UNTRACKED_FILE):
            (job.out / name).unlink(missing_ok=True)
    rec.code = code
    rec.save(job.out)
    return code


def push_code_version(remote: Remote, code: dict) -> None:
    """CODE_VERSION.json at the root of the pushed repo, which has no .git: provenance.code_version reads it there."""
    target = f"{REMOTE_REPO}/{provenance.CODE_VERSION_FILE}"
    with tempfile.TemporaryFile() as fh:
        fh.write((json.dumps(code, indent=1) + "\n").encode())
        fh.seek(0)
        remote.run(f"mkdir -p {REMOTE_REPO} && cat > {target}.tmp && mv {target}.tmp {target}", stdin=fh, timeout=60)


def push_path(remote: Remote, local: str, dest: str) -> None:
    src, target = Path(local).expanduser(), shlex.quote(remote_path(dest))
    if src.is_dir():
        remote.pipe_from(["tar", "-C", str(src), "-czf", "-", "."], f"mkdir -p {target} && tar -C {target} -xzf -")
    elif src.is_file():
        with src.open("rb") as fh:
            remote.run(f'mkdir -p "$(dirname {target})" && cat > {target}', stdin=fh, timeout=3600)
    else:
        raise InfraError(f"--push source {local} does not exist")


def fetch_path(remote: Remote, path: str, dest: Path) -> str:
    """Copy the remote file or directory ``path`` to ``dest/<basename>``. Returns "ok" or "missing"."""
    full = remote_path(path.rstrip("/"))
    base, name = os.path.split(full)
    rc = remote.run(f"test -e {shlex.quote(full)}", check=False, timeout=60).returncode
    if rc == SSH_DOWN:
        raise InfraError(f"ssh {remote.host} unreachable")
    if rc != 0:
        return "missing"
    dest.mkdir(parents=True, exist_ok=True)
    remote.pipe_to(f"tar -C {shlex.quote(base or '.')} -czf - {shlex.quote(name)}",
                   ["tar", "-C", str(dest), "-xzf", "-"])
    return "ok"


RSYNC_VANISHED = 24  # rsync: some source files vanished during the copy (a job still writing); the rest came


def fetch_rsync(remote: Remote, path: str, dest: Path, rsync: str = "rsync", timeout: float = 3600,
                exclude: tuple[str, ...] = ()) -> str:
    """``fetch_path`` with rsync over the remote's ssh (vast.ai): uncompressed (tar's gzip would be the bottleneck for
    tens of GB of tensors) and resumable: ``--partial``, and a broken copy is tried twice more, going on where it
    stopped. Only what changed travels, so it can run again as the job goes on (``exclude``: patterns left out). Tar
    over ssh if the remote has no rsync. Returns "ok" or "missing"; InfraError if it fails or isn't done within
    ``timeout`` seconds."""
    full = remote_path(path.rstrip("/"))
    rc = remote.run(f"test -e {shlex.quote(full)} || exit 3; command -v rsync >/dev/null || exit 4", check=False,
                    timeout=60).returncode
    if rc == SSH_DOWN:
        raise InfraError(f"ssh {remote.host} unreachable")
    if rc == 3:
        return "missing"
    if rc == 4:
        log(f"fetch {path}: no rsync on the remote, so tar over ssh")
        return fetch_path(remote, path, dest)
    if rc != 0:
        raise InfraError(f"fetch {path}: the remote check failed ({rc})")
    dest.mkdir(parents=True, exist_ok=True)
    argv = [rsync, "-a", "--partial", "--timeout=300", "--stats", *(f"--exclude={p}" for p in exclude), "-e",
            remote.rsh(), f"{remote.host}:{full}", f"{dest}/"]
    t0, tries = time.time(), 0
    while True:
        tries += 1
        try:
            cp = subprocess.run(argv, capture_output=True, stdin=subprocess.DEVNULL, check=False,
                                timeout=max(30.0, t0 + timeout - time.time()))
        except subprocess.TimeoutExpired:
            raise InfraError(f"rsync of {path} not done after {timeout:.0f}s") from None
        if cp.returncode in (0, RSYNC_VANISHED):
            size = re.search(rb"Total transferred file size: ([\d,.]+)", cp.stdout)
            n = int(re.sub(rb"[,.]", b"", size.group(1))) if size else 0
            dt = time.time() - t0
            log(f"rsync of {path}: {n / 1e9:.2f} GB in {dt:.0f}s ({n / 1e6 / max(dt, 1e-9):.0f} MB/s)"
                + (" (some files vanished during the copy)" if cp.returncode else ""))
            return "ok"
        err = cp.stderr.decode(errors="replace").strip()[-1500:]
        if tries >= 3 or time.time() - t0 > timeout - 60:
            raise InfraError(f"rsync of {path} failed ({cp.returncode}): {err}")
        log(f"rsync of {path} failed ({cp.returncode}), trying again: {err[-300:]}")
        time.sleep(10)


VAST_FETCH_EVERY = 600.0  # seconds between the rsyncs of the --fetch paths while a vast.ai job runs
IN_PROGRESS = ("*.part", "*.tmp")  # files still being written (whitebox/store.py, extraction's write_json)


def fetch_so_far(remote: Remote, job: JobSpec) -> None:
    """vast.ai, while the job runs: rsync the ``--fetch`` paths as they are, so what the job saved is here before the
    instance's end. Through vast.ai's SSH proxy rsync ran at 12 MB/s (2026-10-23): the v2 session's ~51 GB would take
    70 minutes after the job, but arrive as it goes, at the 5-9 MB/s it extracts. Files still being written wait for
    the next round; the final fetch takes everything. Best effort: a failed round only waits for the next."""
    for path in job.fetches:
        left = 3600.0 if job.ends is None else job.ends - job.reserve - time.time()
        if left < 60:
            return
        try:
            fetch_rsync(remote, path, fetch_dest(path, job.out), job.rsync, timeout=left, exclude=IN_PROGRESS)
        except Exception as e:  # noqa: BLE001 — the final fetch tries again
            log(f"fetch so far, {path}: {e}")


def vm_deadline(rec: Record) -> float | None:
    """When the in-VM ``--hours`` shutdown hits (creation + hours; the real one is a little later)."""
    if rec.hours and rec.times.get("create_start"):
        return _seconds(rec.times["create_start"]) + rec.hours * 3600
    return None


def remote_steps(remote: Remote, job: JobSpec, rec: Record, check_vm=None) -> int | None:
    """Everything over SSH up to the job's end. Returns the job's exit code (None: no job)."""
    wait_ssh(remote, check_vm)
    rec.times["ssh_ready"] = utc()
    vast = job.provider == "vast"
    log(f"ssh ready: {remote.host}; bootstrapping ("
        + ("a RAM disk if the container allows one, rsync, uv)" if vast else "RAM disk, lingering, uv)"))
    cp = remote.run(vast_bootstrap_script() if vast else bootstrap_script(), timeout=1200, check=False)
    (job.out / "bootstrap.log").write_bytes(cp.stdout + cp.stderr)
    for line in (cp.stdout + cp.stderr).decode(errors="replace").strip().splitlines():
        print(f"  bootstrap| {line}", flush=True)
    if cp.returncode != 0:
        raise InfraError(f"bootstrap failed ({cp.returncode}); see bootstrap.log")
    if job.hf:
        start(remote, "download", download_script(job.hf))
        log(f"downloading {', '.join(r for r, _, _ in job.hf)} to {RAM_DIR} (detached)")
    files = repo_files()
    code = stamp_push(job, rec)
    log(f"pushing the repo: {len(files)} files, {describe_code(code)}")
    push_repo(remote, files)
    push_code_version(remote, code)
    for local, dest in job.pushes:
        log(f"pushing {local} -> {remote_path(dest)}")
        push_path(remote, local, dest)
    start(remote, "setup", job_setup_script(job), cwd=REMOTE_REPO)
    if (code := poll(remote, "setup", job.out, interval=10, timeout=3600, check_vm=check_vm)) != 0:
        checks = ["uv sync", *(["CUDA check"] if job.gpu else []), *(["root-job check"] if job.root_job else [])]
        raise InfraError(f"setup ({' / '.join(checks)}) failed with {code}; see setup.log")
    rec.times["setup_done"] = utc()
    if job.hf:
        if (code := poll(remote, "download", job.out, interval=10, timeout=7200, check_vm=check_vm)) != 0:
            raise InfraError(f"download failed with {code}; see download.log")
        rec.times["downloads_done"] = utc()
    rec.save(job.out)
    if job.command is None:
        return None
    if job.uploads:  # before the job, so it can save its progress as it goes; removed when the upload step ends
        send_token(remote, job.hf_token_file)
        log(f"HF token on the VM (mode 600) for the job and the upload: {REMOTE_HF_TOKEN}")
    if job.secrets:  # read by the job step as it starts, deleted when it ends
        send_secrets(remote, job.secrets)
        log(f"keys on the VM (owner-only, in {REMOTE_SECRETS}) for the job: "
            + ", ".join(f"{s.name} as ${s.env}" for s in job.secrets))
    log(f"job{' (as root)' if job.root_job else ''}: {job.command}")
    fallback = None  # if the VM's own record of its shutdown can't be read: creation + --hours, slightly early
    if job.ends is not None:  # vast.ai (no shutdown record in a container): the instance's end, less the reserve
        fallback = int(job.ends - job.reserve)
    elif rec.hours and rec.times.get("create_start"):
        fallback = int(_seconds(rec.times["create_start"]) + rec.hours * 3600)
    start(remote, "job", job_script(job.command, job.env, job.poweroff_minutes, fallback, secrets=job.secrets,
                                    root=job.root_job), cwd=REMOTE_REPO)
    rec.times["job_start"] = utc()
    rec.save(job.out)
    cap = vm_deadline(rec)  # past the cap + 10 min the VM can only be gone or wedged: stop waiting
    timeout = None if cap is None else max(60.0, cap + 600 - time.time())
    if job.ends is not None:  # vast.ai destroys the instance at its end: stop waiting halfway into the reserve,
        timeout = max(60.0, job.ends - job.reserve / 2 - time.time())  # so what the job made can still be fetched
    during = (lambda: fetch_so_far(remote, job)) if vast and job.fetches else None  # here before the instance's end
    code = poll(remote, "job", job.out, interval=job.poll_interval, check_vm=check_vm, timeout=timeout, during=during,
                every=VAST_FETCH_EVERY)
    rec.times["job_end"], rec.exit_code = utc(), code
    rec.save(job.out)
    if job.uploads:
        if cap is not None and cap - time.time() < 300:
            log("skipping the final upload: the --hours cap is less than 5 min away (the job saves its own progress)")
            rec.errors.append("final upload skipped: too close to the --hours cap")
        else:
            upload(remote, job, rec, check_vm)
    return code


def send_token(remote: Remote, token_file: Path) -> None:
    """The HF token goes to the VM over SSH stdin into an owner-only file: never on a command line or in a log."""
    with token_file.open("rb") as fh:
        remote.run(f'umask 077 && mkdir -p "$(dirname "{REMOTE_HF_TOKEN}")" && cat > "{REMOTE_HF_TOKEN}"', stdin=fh,
                   timeout=60)


def send_secrets(remote: Remote, secrets: list[Secret]) -> None:
    """Each key goes to the VM like the HF token: over SSH stdin into a new owner-only file (umask 077), in a 0700
    folder of the login user's own on the RAM disk (not a link, not someone else's). Never on a command line or in a
    log; an error shows the remote shell's message, never the key."""
    folder = shlex.quote(REMOTE_SECRETS)
    for s in secrets:
        target = shlex.quote(remote_secret(s.name))
        with s.path.open("rb") as fh:
            remote.run(f"umask 077 && mkdir -p {folder} && [ ! -L {folder} ] && [ -O {folder} ] && chmod 700 {folder} "
                       f"&& rm -f -- {target} && cat > {target}", stdin=fh, timeout=60)


def remove_secrets(remote: Remote, secrets: list[Secret]) -> None:
    """Best effort: the key files off the VM, in case the job step's own EXIT trap didn't run (the launcher stopped
    before or during the job; a kept VM keeps its RAM disk). Errors are swallowed; a signal still propagates, so
    SIGTERM/SIGHUP can cut the collection short as before."""
    if not secrets:
        return
    files = shlex.join([remote_secret(s.name) for s in secrets])
    with contextlib.suppress(Exception):
        remote.run(f"rm -f -- {files}; rmdir {shlex.quote(REMOTE_SECRETS)} 2>/dev/null; exit 0", timeout=60,
                   check=False)


def upload(remote: Remote, job: JobSpec, rec: Record, check_vm=None) -> None:
    """Results straight from the VM to the HF bucket. Partial results of a failed job are uploaded too."""
    log(f"uploading {', '.join(dest for _, dest in job.uploads)}")
    send_token(remote, job.hf_token_file)
    start(remote, "upload", upload_script(job.uploads), cwd=REMOTE_REPO)
    code = poll(remote, "upload", job.out, interval=10, timeout=7200, check_vm=check_vm)
    remote.run(f'rm -f "{REMOTE_HF_TOKEN}"', timeout=60, check=False)  # belt and braces; the step's trap did it
    status = "ok" if code == 0 else f"failed ({code}); see upload.log"
    for _, dest in job.uploads:
        rec.uploaded[dest] = status
    rec.times["upload_done"] = utc()
    rec.save(job.out)


def collect(remote: Remote, job: JobSpec, rec: Record) -> None:
    """Best effort: the --fetch paths, then every remote step log (into ``remote_logs/``; the
    streamed copies are already in ``--out``). Never raises: a failed fetch must not skip the teardown."""
    for path in [*job.fetches, f"~/{REMOTE_LOGS}"]:
        dest = job.out / "remote_logs" if path == f"~/{REMOTE_LOGS}" else fetch_dest(path, job.out)
        try:
            if job.provider == "vast":  # before the instance's end, which destroys what is on it
                left = 3600.0 if job.ends is None else max(60.0, job.ends - time.time() - 60)
                rec.fetched[path] = fetch_rsync(remote, path, dest, job.rsync, timeout=left)
            else:
                rec.fetched[path] = fetch_path(remote, path, dest)
        except Exception as e:  # noqa: BLE001
            rec.fetched[path] = f"error: {e}"
        log(f"fetch {path}: {rec.fetched[path]}")


@contextlib.contextmanager
def signals_held():
    """Ctrl-C, SIGTERM and SIGHUP must not abort deleting the VM. Inside the block each is held off and recorded in
    the ``HeldSignals`` it yields, so the caller can stop once the VM is gone. ``cmd_vm`` has installed
    ``_launcher_signal`` already (a caller that hasn't gets it here). A signal this process ignores (SIGHUP under
    nohup) stays ignored."""
    global _held
    held, old = HeldSignals(), {}
    _held = held
    try:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            if signal.getsignal(sig) not in (signal.SIG_IGN, _launcher_signal):
                old[sig] = signal.signal(sig, _launcher_signal)
        yield held
    finally:
        _held = None
        restore_signals(old)


# ------------------------------------------------------------------------------------ vast.ai (hub-gpu)


HUB_GPU_KEY = "~/.ssh/hub-gpu"  # hub-gpu's key for this project's instances; it attaches the .pub to each it starts
VAST_NAME = re.compile(r"[a-z0-9-]{1,24}")  # hub-gpu's names: lowercase letters, digits and -, at most 24
VAST_MAX_TAGS = 8  # hub-gpu's, of which the launcher takes two (launcher=, commit=)
VAST_MIN_CUDA = 13.0  # the locked torch is a CUDA 13.0 build: a driver with an older CUDA can't run it
VAST_RESERVE_MINUTES = 30.0  # before the instance's end, for the fetch (at most a quarter of --hours)
VAST_STARTS = 2  # a refused start (the offer was taken meanwhile) is tried once more, on a fresh listing's cheapest
VAST_LOADING_MINUTES = 40.0  # an instance may stay "loading" (pulling its image) this long before it is given up
VAST_DEAD = {"exited", "stopped", "offline", "error", "failed", "destroyed", "deleted", "ended", "inactive"}
# GPUs before Turing (compute capability < 7.5: Maxwell, Pascal, Volta), by hub-gpu's names. Their drivers may run CUDA
# 13.0, but CUDA 13 builds such as the locked torch have no kernels for them (2026-10-23: a GTX 1080 was the cheapest
# CUDA 13.0 offer). The offers carry no compute capability.
PRE_TURING = re.compile(r"GTX (9|10)\d\d|GTX TITAN|TITAN [XV]|GT \d|TESLA [KMPV]|QUADRO ([KMP]\d|G[PV]100)|[PV]100",
                        re.IGNORECASE)
SECRET_WORDS = {"TOKEN", "KEY", "SECRET", "PASSWORD", "PASSWD", "CREDENTIALS", "APIKEY", "AUTH"}
SECRET_DIRS = ("~/.secrets", "~/.ssh", "~/.nebius", "~/.cache/huggingface", "~/.config/gcloud", "~/.aws")


class HubGpu:
    """The ``hub-gpu`` CLI: hub's vast.ai service for this project, within the limits the owner sets for it
    (``hub-gpu --help``). Each call asks for the service's own JSON (``--json``), with a timeout and no stdin. The
    vast.ai key stays with hub: it never reaches this machine or an instance."""

    def __init__(self, binary: str = "hub-gpu") -> None:
        self.binary = binary

    def _run(self, args: list[str], timeout: float = 300) -> dict:
        try:
            cp = subprocess.run([self.binary, "--json", *args], capture_output=True, text=True, timeout=timeout,
                                stdin=subprocess.DEVNULL, start_new_session=True, check=False)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise InfraError(f"hub-gpu {args[0]}: {e}") from None
        if cp.returncode != 0:
            raise InfraError(f"hub-gpu {args[0]} failed ({cp.returncode}): {(cp.stderr or cp.stdout).strip()[-2000:]}")
        try:
            return json.loads(cp.stdout) if cp.stdout.strip() else {}
        except ValueError:
            raise InfraError(f"hub-gpu {args[0]} did not answer JSON: {cp.stdout.strip()[:300]!r}") from None

    def limits(self) -> dict:
        return self._run(["limits"], timeout=120)

    def offers(self, *, gpu: str | None, gpus: int, disk_gb: int, max_price: float | None) -> list[dict]:
        return self._run(offers_args(gpu, gpus, disk_gb, max_price), timeout=180).get("offers") or []

    def start(self, offer: int, **kw) -> dict:
        return self._run(start_args(offer, **kw), timeout=600)

    def instances(self) -> list[dict]:
        return self._run(["list"], timeout=120).get("instances") or []

    def end(self, instance) -> dict:
        return self._run(["end", str(instance)], timeout=300)

    def attach_key(self, instance) -> None:
        """``hub-gpu ssh N true``: attaches this project's key to the instance again, which its start may not have
        done (``ssh_key_attached``), then runs ``true``. Its outcome doesn't matter."""
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run([self.binary, "ssh", str(instance), "true"], capture_output=True, timeout=90,
                           stdin=subprocess.DEVNULL, start_new_session=True, check=False)


def offers_args(gpu: str | None, gpus: int, disk_gb: int, max_price: float | None) -> list[str]:
    """``hub-gpu offers`` for ``gpus`` × ``gpu`` (None: any) with ``disk_gb`` of disk: as many as it lists (50)."""
    args = ["offers", "--gpus", str(gpus), "--disk", str(disk_gb), "--limit", "50"]
    if gpu:
        args += ["--gpu", gpu]
    return args + (["--max-price", f"{max_price:g}"] if max_price is not None else [])


def start_args(offer: int, *, name: str, purpose: str, tags: list[str], hours: float, disk_gb: int,
               image: str | None = None) -> list[str]:
    """``hub-gpu start``: timed, so vast.ai destroys the instance ``hours`` after its start whatever happens here."""
    args = ["start", str(offer), "--name", name, "--purpose", purpose]
    for tag in tags:
        args += ["--tag", tag]
    return args + ["--hours", f"{hours:g}", "--disk", str(disk_gb)] + (["--image", image] if image else [])


def disk_gb(gib: float) -> int:
    """``--disk-gib`` in the GB that hub-gpu takes, rounded up."""
    return math.ceil(gib * 2**30 / 1e9 - 1e-9)


def offer_skip(offer: dict, min_cuda: float, max_price: float | None) -> str | None:
    """Why a vast.ai offer doesn't qualify (None: it does). Its ``cuda`` is the newest CUDA its driver runs; for CUDA 13
    the GPU must also be Turing or newer (``PRE_TURING``)."""
    cuda = offer.get("cuda")
    if cuda is None or float(cuda) < min_cuda - 1e-9:
        return f"its driver's CUDA {cuda if cuda is not None else '?'} < {min_cuda:g}"
    if min_cuda >= 13 - 1e-9 and PRE_TURING.match(str(offer.get("gpu") or "").strip()):
        return "older than Turing: no CUDA 13 kernels"
    if max_price is not None and float(offer["usd_per_hour"]) > max_price + 1e-9:
        return f"above the ${max_price:g}/h cap"
    return None


def vast_candidates(offers: list[dict], min_cuda: float, max_price: float | None) -> list[dict]:
    """The qualifying offers, cheapest first; ties go to the more reliable machine."""
    ok = [o for o in offers if offer_skip(o, min_cuda, max_price) is None]
    return sorted(ok, key=lambda o: (float(o["usd_per_hour"]), -float(o.get("reliability") or 0)))


def vast_offer_table(offers: list[dict], min_cuda: float, max_price: float | None, chosen: int | None,
                     why: str = "cheapest") -> str:
    """The offers by price, down to a few past the chosen one."""
    rows = sorted(offers, key=lambda o: float(o["usd_per_hour"]))
    at = next((i for i, o in enumerate(rows) if o["offer"] == chosen), -1)
    shown = rows[:max(at + 5, 8)][:16]
    lines = [f"  {'offer':>10}  {'GPUs':<22} {'$/hour':>7} {'CUDA':>5} {'RAM':>6} {'reliab.':>7}  {'where':<22} note"]
    for o in shown:
        note = f"<- {why}, tried first" if o["offer"] == chosen else (offer_skip(o, min_cuda, max_price) or "")
        gpus, cuda = f"{o.get('gpus')}× {o.get('gpu')}"[:22], o.get("cuda") if o.get("cuda") is not None else "?"
        lines.append(f"  {o['offer']:>10}  {gpus:<22} {float(o['usd_per_hour']):>7.3f} {cuda:>5} "
                     f"{float(o.get('ram_gb') or 0):>5.0f}G {float(o.get('reliability') or 0):>7.3f}  "
                     f"{str(o.get('where') or '?')[:22]:<22} {note}")
    if len(rows) > len(shown):
        lines.append(f"  … {len(rows) - len(shown)} more")
    return "\n".join(lines)


def limits_problem(lim: dict, gpus: int, hours: float, worst: float | None = None,
                   max_cost: float | None = None) -> str | None:
    """Why ``hub-gpu limits`` (``lim``), or ``--max-cost``, rule out ``gpus`` GPUs for ``hours`` at a worst case of
    ``worst`` USD (None: they don't)."""
    mine = lim.get("limits") or {}
    if not mine:
        return f"project {lim.get('project')} may not rent GPUs (the owner switches it on in Project options)"
    if hours > float(mine.get("hours") or 0) + 1e-9:
        return f"--hours {hours:g} is above the {float(mine.get('hours') or 0):g} h a timed run of this project has"
    in_use = int(lim.get("gpus_in_use") or 0)
    if in_use + gpus > int(mine.get("gpus") or 0):
        return (f"{gpus} more GPU(s) with {in_use} in use is above the {mine.get('gpus')} this project may use at "
                "once")
    room = float(lim.get("room_usd") or 0)
    if worst is not None and worst > room + 1e-9:
        return f"its worst case, ${worst:.2f}, is above the ${room:.2f} this project has left in any 24 hours"
    if worst is not None and max_cost is not None and worst > max_cost + 1e-9:
        return f"its worst case, ${worst:.2f}, is above --max-cost ${max_cost:g}"
    return None


def vast_name(prefix: str) -> str:
    """``prefix``-<6 hex>: a hub-gpu name (lowercase letters, digits and -, at most 24), new among this project's
    instances with near certainty (``vast_attempt`` checks it against ``hub-gpu list``)."""
    if not re.fullmatch(r"[a-z][a-z0-9-]*", prefix) or len(prefix) > 17:
        raise SystemExit(f"--name with --provider vast: lowercase letters, digits and dashes, starting with a letter, "
                         f"at most 17 characters (a -XXXXXX suffix follows; hub-gpu names have at most 24), got "
                         f"{prefix!r}")
    return f"{prefix}-{secrets.token_hex(3)}"


def vast_purpose(given: str | None, command: str | None) -> str:
    """The instance's purpose line (``--purpose``, else the job command), which the owner sees."""
    if given is not None:
        if "\n" in given or "\r" in given or not given.strip():
            raise SystemExit("--purpose is one line, not empty: what the instance is for")
        return given.strip()
    line = " ".join(f"MonitorDecorrelation job (scripts/nebius_job.py): {command}".split())
    return line if len(line) <= 160 else line[:159] + "…"


def vast_tags(given: list[str], code: dict) -> list[str]:
    """``--tag``s, plus the launcher's and the commit pushed."""
    commit = (code.get("commit") or "unknown")[:12] + ("-dirty" if code.get("dirty") else "")
    tags = [*given, "launcher=nebius-job", f"commit={commit}"]
    if len(tags) > VAST_MAX_TAGS:
        raise SystemExit(f"--tag: at most {VAST_MAX_TAGS - 2} (hub-gpu takes {VAST_MAX_TAGS}, and the launcher adds "
                         "launcher=nebius-job and commit=…)")
    return tags


def vast_ssh(row: dict) -> tuple[str, int | None] | None:
    """(user@host, port) of an instance, from its ``hub-gpu list`` row: its direct address if it has one, else
    vast.ai's proxy, as ``hub-gpu ssh`` picks. None while it has neither."""
    ssh = row.get("ssh") or {}
    address = ssh.get("direct") or ssh.get("proxy")
    if not address:
        return None
    words, port, target = shlex.split(address), None, None  # "ssh -p PORT root@HOST"
    rest = iter(words[1:] if words and words[0] == "ssh" else words)
    for word in rest:
        if word == "-p":
            port = int(next(rest))
        elif word in ("-o", "-i", "-l", "-J"):
            next(rest, None)
        elif not word.startswith("-"):
            target = word
    if target is None:
        raise InfraError(f"can't read the SSH address {address!r}")
    return target, port


def vast_gone(row: dict | None) -> str | None:
    """Why an instance (its ``hub-gpu list`` row; None: not listed) can't be running a job (None: it can)."""
    if row is None:
        return "gone from hub-gpu list"
    if row.get("ended"):
        return f"ended at {datetime.fromtimestamp(float(row['ended']), UTC):%H:%M:%S} UTC"
    if row.get("paused"):
        return "paused"
    state = str(row.get("state") or "")
    return state if state.lower() in VAST_DEAD else None


def vast_row(hub: HubGpu, instance) -> dict | None:
    return next((r for r in hub.instances() if r.get("instance") == instance), None)


def vast_wait_running(hub: HubGpu, instance, timeout: float = 1800) -> dict:
    """Wait until the instance runs and has an SSH address (pulling its image can take minutes); returns its list row.
    An address listed for 2 minutes is tried whatever the state says. InfraError if the instance goes away first (not
    listed 2 minutes after its start counts), or after ``timeout``."""
    t0, last, addressed = time.time(), None, None
    while True:
        try:
            row = vast_row(hub, instance)
        except InfraError as e:
            log(f"could not list the instances: {e}")
        else:
            if (gone := vast_gone(row)) and (row is not None or time.time() - t0 > 120):
                raise InfraError(f"instance {instance} is {gone} before it ever ran")
            row = row or {}
            state = str(row.get("state") or "?")
            if state != last:
                log(f"instance {instance}: {state}")
                last = state
            if vast_ssh(row):
                addressed = addressed or time.time()
                if state.lower() == "running" or time.time() - addressed > 120:
                    return row
        if time.time() > t0 + timeout:
            raise InfraError(f"instance {instance} not running with an SSH address after {timeout:.0f}s")
        time.sleep(10)


def vast_state(hub: HubGpu, instance) -> str | None:
    """The instance's state in ``hub-gpu list`` (None: the listing failed); VmStopped if it can't run a job."""
    try:
        row = vast_row(hub, instance)
    except InfraError as e:
        log(f"could not read the instance's state: {e}")
        return None
    if why := vast_gone(row):
        raise VmStopped(why)
    return str(row.get("state") or "?")


def vast_wait_ssh(hub: HubGpu, remote: Remote, instance, attached: bool, until: float | None = None,
                  timeout: float = 900) -> None:
    """``wait_ssh`` for a vast.ai instance. hub-gpu may list it as running at once, then as loading while its image is
    pulled (2026-10-23: a GTX 1650 host was still loading after 15 minutes). So ``timeout`` counts from when it is
    listed as running; while it is loading it may take ``VAST_LOADING_MINUTES``; and never past ``until`` (the job's
    deadline). VmStopped if it goes away. Its key is attached again (``hub-gpu ssh``) if its start says it did not
    attach, and after every 6 failed tries."""
    if not attached:
        log("the start says the SSH key did not attach: attaching it again (hub-gpu ssh)")
        hub.attach_key(instance)
    t0, ran, tries, said = time.time(), None, 0, 0.0
    while True:
        try:
            if remote.run("true", timeout=40, check=False).returncode == 0:
                log(f"ssh works after {time.time() - t0:.0f}s")
                return
        except subprocess.TimeoutExpired:
            pass
        tries += 1
        state, now = vast_state(hub, instance), time.time()
        if state is not None and state.lower() == "loading":
            ran = None
            if now - t0 > VAST_LOADING_MINUTES * 60:
                raise InfraError(f"instance {instance} still loading (pulling its image) after "
                                 f"{VAST_LOADING_MINUTES:g} min")
        elif ran is None:
            ran = now
        if ran is not None and now - ran > timeout:
            raise InfraError(f"no SSH to {remote.host} (port {remote.port}) {timeout:.0f}s after it was {state}")
        if until is not None and now > until:
            raise InfraError(f"no SSH to {remote.host} before the job's deadline (the instance is {state})")
        if now - said > 120:
            log(f"waiting for ssh: the instance is {state} ({now - t0:.0f}s)")
            said = now
        if tries % 6 == 0:
            hub.attach_key(instance)
        time.sleep(10)


def vast_teardown(hub: HubGpu, name: str, instance=None, timeout: float = 300) -> dict | None:
    """End the instance (by number, and by name in case its start failed before its number was known), then verify
    that ``hub-gpu list`` shows it ended, or not at all. Every call is retried until ``timeout``, an end that the list
    doesn't show yet asked again each minute; InfraError if the instance is still there then. Returns its last list
    row (None: never listed)."""
    deadline, ids, asked = time.time() + timeout, {instance} - {None}, {}
    while True:
        try:
            rows = hub.instances()
        except InfraError as e:
            if time.time() > deadline:
                raise InfraError(f"could not verify the end; end it by hand (hub-gpu end {name}): {e}") from None
            log(f"  listing failed, retrying: {e}")
            time.sleep(15)
            continue
        mine = [r for r in rows if r.get("instance") in ids or r.get("name") == name]
        ids |= {r.get("instance") for r in mine}
        left = [r for r in mine if not r.get("ended")]
        if not left:
            log(f"verified: instance {name!r} " + (f"ended ({', '.join(str(r.get('state')) for r in mine)})" if mine
                                                     else "not listed"))
            return mine[-1] if mine else None
        if time.time() > deadline:
            raise InfraError(f"end incomplete; end it by hand (hub-gpu end {name}): "
                             f"{[(r.get('instance'), r.get('state')) for r in left]}")
        for r in left:
            if r.get("instance") in asked and time.time() - asked[r.get("instance")] < 60:
                continue
            log(f"ending instance {r.get('instance')} ({r.get('name')})")
            asked[r.get("instance")] = time.time()
            try:
                hub.end(r.get("instance"))
            except InfraError as e:
                log(f"  end failed, retrying: {e}")
                asked.pop(r.get("instance"))
        time.sleep(5)


def vast_bootstrap_script() -> str:
    """A vast.ai instance's bootstrap. It is a docker container: the login user is root, and there is no systemd.
    - No tmux at ssh logins (vast.ai's images start one unless ~/.no_auto_tmux exists; the steps run in login shells).
    - A tmpfs at /mnt/ram if the container may mount one (80% of its memory limit), else a plain directory on the
      instance's disk, said so in the log: the ``--hf`` downloads and the job's /mnt/ram files then use that disk.
    - rsync (``--fetch``) and a C compiler (a locked package builds from source) if missing; uv, and only uv's own
      Python (an image's may lack the headers that build needs).
    - What the job gets: the disks, memory, /dev/shm, the GPUs and the newest CUDA of their driver."""
    return f"""set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
S=""; [ "$(id -u)" = 0 ] || S="sudo -n"
touch "$HOME/.no_auto_tmux"  # vast.ai's images start tmux at each ssh login unless this file exists
mkdir -p {RAM_DIR} 2>/dev/null || $S mkdir -p {RAM_DIR}
mem=$(( $(awk '/^MemTotal:/ {{print $2}}' /proc/meminfo) * 1024 ))
lim=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)
case "$lim" in ''|max) lim=$mem ;; esac
[ "$lim" -le "$mem" ] 2>/dev/null || lim=$mem
size=$(( lim / 100 * 80 ))
if mountpoint -q {RAM_DIR}; then
  echo "ram disk: {RAM_DIR} is a mount already"
elif err=$($S mount -t tmpfs -o size=$size,mode=1777 tmpfs {RAM_DIR} 2>&1); then
  echo "ram disk: a tmpfs at {RAM_DIR}, $(( size / 1000000000 )) GB (80% of the container's memory limit)"
else
  echo "NO RAM DISK: the container may not mount a tmpfs (${{err:-mount failed}}), so {RAM_DIR} is a plain directory \\
on the instance's disk: the --hf downloads and the job's files there use that disk"
fi
need=""
command -v rsync >/dev/null || need="$need rsync"
command -v cc >/dev/null || need="$need build-essential"
command -v curl >/dev/null || need="$need curl ca-certificates"
if [ -n "$need" ]; then
  echo "installing:$need"
  {{ $S apt-get update -qq && $S env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $need; }} >/dev/null 2>&1 \\
    || echo "WARNING: could not install$need"
fi
command -v rsync >/dev/null || echo "WARNING: no rsync here, so --fetch falls back to tar over ssh"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
command -v uv >/dev/null || {{ echo "uv install failed" >&2; exit 1; }}
mkdir -p "$HOME/.config/uv"
grep -qs '^python-preference' "$HOME/.config/uv/uv.toml" \\
  || echo 'python-preference = "only-managed"' >> "$HOME/.config/uv/uv.toml"
uv --version
echo "ram disk: $(df -h {RAM_DIR} | tail -1)"
echo "home disk: $(df -h "$HOME" | tail -1)"
echo "memory: $(( mem / 1000000000 )) GB, the container's limit $(( lim / 1000000000 )) GB; \\
/dev/shm: $(df -h /dev/shm 2>/dev/null | tail -1)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "no nvidia-smi"
nvidia-smi 2>/dev/null | grep -o "CUDA Version: [0-9.]*" || true
"""


def secret_env(name: str, value: str) -> bool:
    """An ``--env`` that looks like a key: a word of its name is TOKEN, KEY, SECRET, … and its value is long (a
    setting such as HF_HUB_DISABLE_IMPLICIT_TOKEN=1 is not)."""
    return any(w in SECRET_WORDS for w in name.upper().split("_")) and len(value) >= 16


def secret_push(local: str) -> str | None:
    """Why the ``--push`` source ``local`` may hold a key (None: it doesn't): it is in one of this machine's key
    folders (``SECRET_DIRS``), or it is or holds a .env or .netrc file."""
    src = Path(local).expanduser().resolve()
    for folder in SECRET_DIRS:
        f = Path(folder).expanduser().resolve()
        if src == f or f in src.parents:
            return f"it is in {folder}"
    for p in [src, *(src.rglob("*") if src.is_dir() else [])]:
        if p.name.startswith(".env") or p.name == ".netrc":
            return f"{p} may hold keys"
    return None


# --------------------------------------------------------------------------------------- commands


def _require(value: str | None, flag: str, env: str) -> str:
    if not value:
        raise SystemExit(f"{flag} (or ${env}) is required")
    return value


def vm_spec(args) -> VmSpec:
    """The VM, minus where it goes (``in_region`` fills that in)."""
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,40}", args.name):
        raise SystemExit(f"--name must be lowercase letters, digits and dashes, got {args.name!r}")
    if is_cpu_platform(args.platform) and not args.on_demand:
        raise SystemExit(f"{args.platform} is a CPU platform: CPU VMs run --on-demand here (the spot bid, "
                         "--max-gpu-price, is per GPU-hour)")
    return VmSpec(name=f"{args.name}-{secrets.token_hex(3)}", project_id="", subnet_id="",
                  platform=args.platform, preset=args.preset, spot=not args.on_demand, disk_gib=args.disk_gib,
                  image_family=args.image_family, image_project="", hours=args.hours, user=args.user)


def spot_bid(args) -> float | None:
    """The spot bid per GPU-hour: ``--max-gpu-price``, else NEBIUS_MAX_GPU_PRICE. An on-demand VM has a fixed price:
    a bid given on the command line is refused there, and the environment's (a wrapper's default) ignored."""
    if args.max_gpu_price is not None:
        if args.on_demand:
            raise SystemExit("--max-gpu-price is a spot bid; it does not apply to --on-demand")
        return args.max_gpu_price
    env = os.environ.get("NEBIUS_MAX_GPU_PRICE", "").strip()
    if not env:
        return None
    if args.on_demand:
        log(f"--on-demand: ignoring NEBIUS_MAX_GPU_PRICE={env}, a spot bid (an on-demand VM has a fixed price)")
        return None
    try:
        return float(env)
    except ValueError:
        raise SystemExit(f"NEBIUS_MAX_GPU_PRICE={env!r} is not a number (USD per GPU-hour)") from None


def in_region(spec: VmSpec, region: Region) -> VmSpec:
    return replace(spec, project_id=region.project, subnet_id=region.subnet, image_project=region.image_project)


def resolve_regions(args, need_subnet: bool = True) -> tuple[str | None, list[Region]]:
    """The candidate regions: every region of ``--regions-file`` (or those named by ``--regions``), or the
    single project of ``--project-id`` with ``--subnet-id`` / ``--image-project``."""
    if args.regions_file and args.project_id:
        raise SystemExit("--regions-file and --project-id exclude each other (narrow the file with --regions)")
    if args.regions_file:
        tenant, regions = load_regions(args.regions_file)
        if args.regions:
            wanted = [r.strip() for r in args.regions.split(",") if r.strip()]
            unknown = sorted(set(wanted) - {r.name for r in regions})
            if unknown:
                raise SystemExit(f"--regions: {unknown} are not in {args.regions_file}")
            regions = [r for r in regions if r.name in wanted]
        return tenant, regions
    if args.regions:
        raise SystemExit("--regions needs --regions-file (or $NEBIUS_REGIONS_FILE)")
    project = _require(args.project_id, "--regions-file or --project-id", "NEBIUS_REGIONS_FILE / $NEBIUS_PROJECT_ID")
    subnet = _require(args.subnet_id, "--subnet-id", "NEBIUS_SUBNET_ID") if need_subnet else ""
    return None, [Region("default", project, subnet, getattr(args, "image_project", "") or "")]


def job_spec(args, cmd: list[str], name: str) -> JobSpec:
    env = []
    for item in args.env:
        k, sep, v = item.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
            raise SystemExit(f"--env wants KEY=VALUE, got {item!r}")
        env.append((k, v))
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    try:
        hf = [parse_hf(s) for s in args.hf]
        pushes = [parse_push(s) for s in args.push]
        uploads = [(src, expand(dest, name, stamp)) for src, dest in map(parse_upload, args.upload)]
    except ValueError as e:
        raise SystemExit(str(e)) from None
    token_file = None
    if uploads:
        token_file = Path(args.hf_token_file).expanduser()
        if not (token_file.is_file() and token_file.stat().st_size > 0) and not getattr(args, "dry_run", False):
            raise SystemExit(f"--upload needs a Hugging Face token in {token_file} (--hf-token-file)")
    try:
        keys = job_secrets(args.secret, {k for k, _ in env})
    except ValueError as e:
        raise SystemExit(str(e)) from None
    for s in keys:  # before anything is priced or created; a dry run shows the problem instead
        problem = secret_problem(s.path)
        if problem and not getattr(args, "dry_run", False):
            raise SystemExit(f"--secret {s.name}: {s.path} is {problem}; put the key there with `hub-secret set "
                             f"{s.name}`")
    if keys and getattr(args, "cmd", None) == "up":
        log(f"up runs no job, so --secret {', '.join(s.name for s in keys)} is not sent: give it to exec")
    gpu = not getattr(args, "no_gpu", False) and not is_cpu_platform(getattr(args, "platform", "") or "")
    if args.extra_gpu and not gpu:
        raise SystemExit("--extra-gpu needs a GPU (not a cpu-* platform, nor exec --no-gpu)")
    if args.sandbox_python is not None and not args.root_job:
        raise SystemExit("--sandbox-python is the Python of a --root-job; it does not apply without --root-job")
    sandbox_python = args.sandbox_python or DEFAULT_SANDBOX_PYTHON
    if not re.fullmatch(r"\d+\.\d+\.\d+", sandbox_python):
        raise SystemExit(f"--sandbox-python wants an exact CPython version like {DEFAULT_SANDBOX_PYTHON}, "
                         f"got {sandbox_python!r}")
    return JobSpec(out=expand_out(args.out, name, stamp), command=job_command(cmd) if cmd else None,
                   pushes=pushes, hf=hf, fetches=list(args.fetch), uploads=uploads, hf_token_file=token_file,
                   env=env, extra_gpu=args.extra_gpu, poll_interval=args.poll_interval, allow_dirty=args.allow_dirty,
                   secrets=keys, gpu=gpu, root_job=args.root_job, sandbox_python=sandbox_python)


def dry_run(nebius: str, spec: VmSpec, job: JobSpec, pubkey: str, keep: bool) -> int:
    """Print what ``run``/``up`` would do in the chosen region. Nothing is created and nothing goes over SSH;
    the only cloud calls (before this) are the read-only price and capacity checks."""
    user_data = render_cloud_init(spec.user, pubkey, spec.hours)
    argv = create_argv(nebius, spec, "<CLOUD_INIT>")
    print("== create (dry run: nothing is created) ==")
    print(shlex.join(argv))
    print(f"== cloud-init (--cloud-init-user-data) ==\n{user_data}")
    print_remote_plan(job, f"{spec.user}@<ip>", bootstrap_script())
    print("== fetch ==")
    for path in job.fetches:
        print(f"{remote_path(path)} -> {fetch_dest(path, job.out) / os.path.basename(path.rstrip('/'))}")
    print(f"~/{REMOTE_LOGS}/ -> {job.out / 'remote_logs'}; job.json -> {job.out / 'job.json'}")
    end = "kept (prints the ssh command)" if keep else "deleted, then verified gone (instance and boot disk)"
    print(f"== teardown ==\n{spec.name}: {end}; in-VM auto-shutdown after {spec.hours:g} h")
    return 0


def print_remote_plan(job: JobSpec, target: str, bootstrap: str, deadline: int | None = None) -> None:
    """The dry run's SSH part, from the push to the job's uploads: what goes to ``target`` and what runs there.
    ``deadline``: the job's NEBIUS_JOB_DEADLINE if the machine has no shutdown record (vast.ai)."""
    files = repo_files()
    size = sum((REPO / f).stat().st_size for f in files if (REPO / f).is_file())
    print(f"== push ==\n{len(files)} repo files, {size / 1e6:.1f} MB uncompressed: "
          f"{REPO} -> {target}:{REMOTE_REPO}/")
    for local, dest in job.pushes:
        print(f"{local} -> {remote_path(dest)}")
    code = code_version(REPO)
    print(f"== code version (now; taken again at each push) -> {REMOTE_REPO}/{provenance.CODE_VERSION_FILE} and "
          f"job.json ==\n{json.dumps(code)}")
    if code["dirty"] and code["source"] == "git":
        print(f"uncommitted changes (--allow-dirty): their patch -> {job.out / provenance.PATCH_FILE}, the untracked "
              f"files in it -> {job.out / provenance.UNTRACKED_FILE}")
    print(f"== bootstrap (over ssh) ==\n{bootstrap}")
    if job.hf:
        print(f"== downloads (detached, started right after bootstrap) ==\n{download_script(job.hf)}\n")
    print(f"== setup (detached, in ~/{REMOTE_REPO}) ==\n{job_setup_script(job)}")
    if job.secrets and job.command is None:
        print("== keys: not sent (up runs no job; give --secret to exec) ==")
    elif job.secrets:
        print("== keys (names only: a key is never shown) ==")
        for s in job.secrets:
            problem = secret_problem(s.path)
            print(f"{s.path} -> {remote_secret(s.name)} on the VM, the job's ${s.env}"
                  + (f"  !! {problem}: a real run refuses it (hub-secret set {s.name})" if problem else ""))
        print("each over ssh stdin into an owner-only file (umask 077) right before the job; the job step reads it as "
              "it starts and deletes it when it ends (EXIT trap), the launcher again if it stops first")
    if job.command is not None:
        script = job_script(job.command, job.env, job.poweroff_minutes, deadline, secrets=job.secrets,
                            root=job.root_job)
        print(f"== job (detached, in ~/{REMOTE_REPO}{'; the command as root' if job.root_job else ''}) ==\n{script}\n")
        print(f"== how a detached step is launched (the job, verbatim) ==\n{detached(script, 'job', REMOTE_REPO)}\n")
    if job.uploads:
        print("== upload (detached, after the job, whatever its exit code) ==")
        for src, dest in job.uploads:
            print(f"{remote_path(src)} -> {dest}")
        print(f"token: {job.hf_token_file} -> {REMOTE_HF_TOKEN} on the VM over ssh stdin, mode 600, before the job "
              f"(so it can save as it goes); deleted when the upload step ends")


def all_regions(args, regions: list[Region]) -> list[Region]:
    """Every region of the regions file, even when ``--regions`` narrowed the choice: a leftover can be anywhere."""
    return load_regions(args.regions_file)[1] if getattr(args, "regions_file", None) else regions


def launcher_alive(entry: dict) -> bool:
    """Is the launcher that made this reservation still running? Checkable on the same host only. A different
    host (e.g. the launching machine came back as a new container) means that launcher is gone."""
    pid = entry.get("pid")
    if not pid or entry.get("host") != socket.gethostname():
        return False
    try:
        args = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")
    except OSError:
        return False
    return any(os.path.basename(a) == "nebius_job.py" for a in args)


def find_record(entry: dict, ledger: Path) -> dict | None:
    """The job.json of a reservation's VM, or None if it can't be found from here.

    - ``out`` is absolute in reservations made since 2026-10-12. An older, relative one is tried against the current
      directory, the repo (where the launcher usually runs) and the ledger's folder.
    - A record counts only if it names this VM: an ``--out`` reused by a later run holds that run's record."""
    out = entry.get("out")
    if not out:
        return None
    path = Path(out).expanduser()
    for folder in dict.fromkeys([path] if path.is_absolute() else [Path.cwd() / path, REPO / path,
                                                                   ledger.parent / path]):
        try:
            rec = json.loads((folder / "job.json").read_text())
        except (OSError, ValueError):
            continue
        if isinstance(rec, dict) and rec.get("name") == entry.get("job"):
            return rec
    return None


def reaped_cost(entry: dict, rec: dict, why: str) -> tuple[float, str]:
    """(USD, note) to settle a reservation at when its launcher can't: the prices that launcher sampled (``rec``, its
    job.json), conservatively (each interval at its higher bound, the unsampled rest at the reservation's hourly worst
    case), up to now or the VM's ``--hours`` shutdown, whichever is first. With no samples, the reserved worst case."""
    hours = rec.get("hours") or entry.get("hours") or 0
    start = (rec.get("times") or {}).get("create_start") or entry.get("created")
    if not (rec.get("price_samples") and start and hours):
        return entry["usd"], f"{why}: no price samples, so at the worst case"
    end = min(time.time(), _seconds(start) + hours * 3600)
    usd = estimated_cost(rec["price_samples"], datetime.fromtimestamp(end, UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                         conservative=True, cap_hourly=entry["usd"] / hours)  # reserved: the worst rate × hours
    return usd, f"{why}: from its sampled prices (conservative), up to its --hours shutdown"


def entry_regions(entry: dict, regions: list[Region]) -> list[Region]:
    """Where a reservation's VM can be: its project's region once it was created, else any region."""
    return [r for r in regions if r.project == entry["project"]] if entry.get("project") else list(regions)


def vm_left(nb: Nebius, name: str) -> bool:
    """Is there an instance named ``name``, or its boot disk, in ``nb``'s project? InfraError if it can't be told."""
    return any(i.name == name for i in nb.instances()) or any(d.name == f"{name}-boot" for d in nb.disks())


def settle_kept(budget: Budget, name: str, why: str) -> None:
    """Settle the reservation of the kept VM ``name`` (``up``, ``run --keep``), which is gone now (``reaped_cost``)."""
    for e in [e for e in budget.data["entries"]
              if e.get("job") == name and e.get("kind") == "reserved" and e.get("kept")]:
        usd, note = reaped_cost(e, find_record(e, budget.path) or {}, why)
        budget.settle(name, usd, note)
        log(f"budget: kept VM {name} settled at ${usd:.2f} ({note})")


def reap(args, budget: Budget, regions: list[Region]) -> None:
    """Clean up after launchers that died mid-run (e.g. their machine restarted), and settle kept VMs that are gone.

    For each reservation whose launcher is gone:
    - delete its VM once the VM has stopped (it powers itself off after its job, or at ``--hours``), or once it is
      overdue: 30 min past its ``--hours`` cap (wedged), or, per its job.json, an hour old with no job started;
    - leave a VM that is still running alone, for a later reap. If its job.json can't be found from here, whether its
      job still runs is unknown, so only the ``--hours`` cap makes it overdue;
    - settle the reservation at the estimate from the prices that launcher sampled, up to the VM's ``--hours``
      deadline; with no samples, at the reserved worst case.

    A kept VM (``up``, ``run --keep``) is never deleted here. Its reservation is settled the same way once the VM is
    gone from its region (deleted by hand; ``down`` with the ledger settles it at once). vast.ai entries are left
    alone: hub destroys such an instance at its end, and the entry counts for nothing here."""
    for e in [e for e in budget.data["entries"] if e.get("kind") == "reserved" and e.get("provider") != "vast"]:
        name = e["job"]
        if not e.get("kept") and launcher_alive(e):
            continue
        places = entry_regions(e, regions)
        if not places:
            log(f"reap: {name} is in project {e['project']}, which is not among the regions here; left as it is")
            continue
        if e.get("kept"):
            try:
                gone = not any(vm_left(Nebius(args.nebius, r.project), name) for r in places)
            except InfraError as err:
                log(f"reap: could not check kept VM {name} ({err}); it stays reserved")
                continue
            if gone:
                settle_kept(budget, name, "kept VM gone; settled by reap")
            continue
        running = False
        rec = find_record(e, budget.path)
        if rec is None and e.get("out"):
            log(f"reap: no job.json of {name} under {e['out']} as seen from here ({os.getcwd()}): whether its job "
                "still runs is unknown, so only its --hours cap can make it overdue")
        known, rec = rec is not None, rec or {}
        times, hours = rec.get("times") or {}, rec.get("hours") or e.get("hours") or 0
        start = times.get("create_start") or e.get("created")
        now = time.time()
        overdue = start is not None and (
            (bool(hours) and now > _seconds(start) + hours * 3600 + 1800)  # 30 min past its cap: wedged
            or (known and "job_start" not in times and now > _seconds(start) + 3600))  # an hour and no job: orphan
        for region in places:
            nb = Nebius(args.nebius, region.project)
            here = [i for i in nb.instances() if i.name == name]
            alive = any(i.state in ALIVE_STATES for i in here)
            if alive and not (overdue or e.get("teardown_failed")):
                log(f"reap: {name} in {region.name} is {here[0].state} without its launcher; it powers itself off "
                    "after its job or at --hours, and a later reap deletes it")
                running = True
                break
            if here or any(d.name == f"{name}-boot" for d in nb.disks()):
                log(f"reap: deleting {name} in {region.name}, left by a launcher that died")
                try:
                    teardown(nb, name)
                except InfraError as err:
                    log(f"reap: could not delete {name} yet ({err}); a later reap tries again")
                    running = True
                    break
        if running:
            continue
        usd, note = reaped_cost(e, rec, "launcher died; settled by reap")
        budget.settle(name, usd, note)
        log(f"reap: {name} settled at ${usd:.2f} ({note})")


def cmd_vm(args, cmd: list[str]) -> int:
    """``run`` and ``up``: check every region's price, create the VM in the cheapest qualifying one (the next
    one if creation fails there), run the SSH part, and always tear down unless kept.

    ``run --attempts N`` relaunches, up to N VMs in all. It does so after a spot preemption, or after a job that
    exits 75: it stopped early, e.g. at its deadline, and can resume.
    - Each new VM goes to the region that is cheapest then.
    - Each is checked against the budget like the first.
    - The job must resume from what it saved; ``extract_probe_features.py --store`` does.
    - Attempt k > 1 writes to ``--out``/retry-k/.

    SIGTERM and SIGHUP are handled like Ctrl-C (SIGHUP not under nohup), and none of them interrupts a delete: one
    that arrives while a VM is being deleted is acted on once it is gone, with no further attempt (exit 130)."""
    handlers = {} if args.dry_run else install_launcher_signals()
    try:
        return vast_run(args, cmd) if getattr(args, "provider", "nebius") == "vast" else vm_attempts(args, cmd)
    finally:
        restore_signals(handlers)


def vm_attempts(args, cmd: list[str]) -> int:
    """``cmd_vm`` inside its signal handlers."""
    if args.cmd == "run" and (given := changed_flags(args) & set(VAST_ONLY)):
        raise SystemExit(f"{flag_names(given)}: only with --provider vast")
    if args.attempts is None:
        args.attempts = 3
    base = vm_spec(args)
    args.max_gpu_price = spot_bid(args)  # the flag, else the environment's (spot only); job.json records this one
    first_job = job_spec(args, cmd, base.name)
    check_code(first_job.allow_dirty)  # before anything is priced, reaped or created
    if args.cmd == "run" and not args.keep:
        first_job = replace(first_job, poweroff_minutes=POWEROFF_AFTER_JOB_MINUTES)
    key = _require(args.ssh_key, "--ssh-key", "NEBIUS_SSH_KEY")
    pub = Path(key).expanduser().with_name(Path(key).name + ".pub")
    if not args.dry_run and not pub.is_file():
        raise SystemExit(f"public key {pub} not found (it goes into cloud-init)")
    tenant, regions = resolve_regions(args)
    if args.budget_file and not Path(args.budget_file).expanduser().exists():
        raise SystemExit(f"budget ledger {args.budget_file} not found: refusing to spend without it "
                         "(create it with {\"limit_usd\": N, \"entries\": []})")
    budget = Budget(args.budget_file) if args.budget_file else None
    if budget is not None and not args.dry_run:
        reap(args, budget, all_regions(args, regions))
    attempts = max(1, args.attempts) if args.cmd == "run" else 1
    first: Record | None = None
    code = EXIT_INFRA
    for k in range(1, attempts + 1):
        spec = base if k == 1 else replace(base, name=f"{base.name}-r{k}")
        job = first_job if k == 1 else replace(first_job, out=first_job.out / f"retry-{k}")
        if k > 1 and budget is not None and budget.limit is not None and args.max_gpu_price and spec.spot:
            # a relaunch only has to finish what is left: shrink --hours to what the budget still covers
            budget._load()
            hourly = args.max_gpu_price * gpu_count(spec.preset)
            fits = (budget.limit - budget.committed()) / hourly
            if fits < MIN_RELAUNCH_HOURS:
                log(f"budget: only {fits:.2f} h left at ${hourly:g}/h; not relaunching. What the job saved stays "
                    "saved, and the same command resumes it later.")
                return code
            if fits < spec.hours:
                log(f"budget: attempt {k} gets --hours {fits:.2f} (not {spec.hours:g}), all the budget still covers")
                spec = replace(spec, hours=round(fits - 0.005, 2))
        try:
            code, rec = vm_attempt(args, spec, job, key, pub, tenant, regions, budget, k)
        except BudgetExceeded as e:
            if k == 1:
                raise
            log(f"{e}. Not relaunching: what the job saved stays saved, and the same command resumes it later.")
            return code
        if k == 1:
            first = rec
        elif first is not None:
            first.retries.append(str(job.out))
            first.save(first_job.out)
        resumable = code == EXIT_UNFINISHED or (code == EXIT_STOPPED and rec is not None and rec.preempted)
        if args.dry_run or k == attempts or not resumable:
            return code
        why = "was preempted" if code == EXIT_STOPPED else f"stopped early so it can resume (exit {EXIT_UNFINISHED})"
        log(f"attempt {k} {why}; starting attempt {k + 1} of {attempts} on a fresh VM")
    return code


def vm_attempt(args, spec: VmSpec, job: JobSpec, key: str, pub: Path, tenant: str | None, regions: list[Region],
               budget: Budget | None, attempt: int) -> tuple[int, Record | None]:
    """One VM: price check (waiting up to ``--wait-hours`` for a region to qualify), budget, create in the cheapest
    qualifying region, the SSH part, teardown, and the ledger. Returns the exit code and the record."""
    by_name = {r.name: r for r in regions}
    bid = args.max_gpu_price if spec.spot else None
    if args.max_gpu_price is not None and not spec.spot:
        raise SystemExit("--max-gpu-price is a spot bid; it does not apply to --on-demand")
    caps = [c for c in (args.max_price, bid * gpu_count(spec.preset) if bid else None) if c is not None]
    cap = min(caps) if caps else None  # whole-VM USD/h
    log(f"checking {'spot' if spec.spot else 'on-demand'} prices for {spec.platform}/{spec.preset} "
        f"in {', '.join(by_name)}" + (f" (cap ${cap:g}/h)" if cap is not None else " (no cap)")
        + (f"; bid ${bid:g}/GPU-h" if bid else ""))
    offers = survey(args.nebius, regions, tenant, spec.platform, spec.preset, spec.spot)
    order = candidates(offers, cap)
    print(offer_table(offers, cap, order[0].region if order else None), flush=True)
    checks = 0 if args.dry_run else max(0, round(args.wait_hours * 3600 / PRICE_RECHECK_SECONDS))
    while not order and checks > 0:
        log(f"no region qualifies now (cap, capacity, quota or price); checking again in "
            f"{PRICE_RECHECK_SECONDS // 60} min ({checks} more checks, --wait-hours)")
        time.sleep(PRICE_RECHECK_SECONDS)
        checks -= 1
        offers = survey(args.nebius, regions, tenant, spec.platform, spec.preset, spec.spot)
        order = candidates(offers, cap)
        if order:
            print(offer_table(offers, cap, order[0].region), flush=True)
    worst_hourly = None
    if budget is not None and order:
        worst_hourly = bid * gpu_count(spec.preset) if bid else \
            Nebius(args.nebius, by_name[order[0].region].project).estimate_hourly(spec.platform, spec.preset, False)
    worst = worst_hourly * spec.hours if worst_hourly is not None else None
    what = f"{spec.name} ({machine_label(spec)}, ≤{spec.hours:g} h)"
    if budget is not None:
        log(budget.summary() + (f"; this job's worst case ${worst:.2f}" if worst is not None else ""))
    if args.dry_run:
        if not order:
            print("== nothing would be created: no region qualifies (cap, capacity, quota or price) ==")
            return EXIT_INFRA, None
        if budget is not None:
            budget.check(worst, what)
            print(f"== budget: ok, would reserve ${worst:.2f} ==")
        if args.cmd == "run":
            print(f"== attempts: up to {args.attempts} VMs (relaunch after a spot preemption or a job exit of "
                  f"{EXIT_UNFINISHED}) ==")
        if bid:
            print(f"== pricing: spot with a max bid of ${bid:g}/GPU-h (policy {policy_name(spec.platform, bid)}, "
                  f"reused or created in the chosen region) ==")
        elif not spec.spot:
            print(f"== pricing: on-demand, ${order[0].price:.2f}/h in {order[0].region} (a fixed price: no bid) ==")
        pubkey = pub.read_text() if pub.is_file() else f"<missing {pub}>"
        return dry_run(args.nebius, in_region(spec, by_name[order[0].region]), job, pubkey,
                       keep=args.cmd == "up" or args.keep), None
    if budget is not None and order:
        budget.check(worst, what)
    keep_always = args.cmd == "run" and args.keep
    rec = Record(mode=args.cmd, name=spec.name, spot=spec.spot, hours=spec.hours, command=job.command,
                 flags={k: v for k, v in vars(args).items() if k not in ("func", *VAST_ONLY)},
                 max_price=cap, offers=[asdict(o) for o in offers], attempt=attempt, **job_fields(job))
    if not order:
        rec.errors.append("no region qualifies (cap, capacity, quota or price); nothing created")
        rec.save(job.out)
        log("no region qualifies; nothing created")
        return EXIT_INFRA, rec
    nb: Nebius | None = None
    inst: Instance | None = None
    remote: Remote | None = None
    sampler: PriceSampler | None = None
    code = EXIT_INFRA
    created = time.time()

    def check_vm() -> None:
        if inst is None or nb is None:
            return
        try:
            state = nb.get(inst.id).state
        except InfraError as e:
            log(f"could not read the VM state: {e}")
            return
        if state not in ALIVE_STATES:
            raise VmStopped(state)

    if budget is not None:
        # absolute: `reap` may run from another directory, and must find this VM's job.json there
        budget.reserve(spec.name, worst, what, pid=os.getpid(), host=socket.gethostname(),
                       out=os.path.abspath(job.out), hours=spec.hours)
    try:
        cloud_init = render_cloud_init(spec.user, pub.read_text(), spec.hours)
        for offer in order:
            spec = in_region(spec, by_name[offer.region])
            nb = Nebius(args.nebius, spec.project_id)
            if bid:
                spec = replace(spec, pricing_policy_id=ensure_policy(nb, spec.platform, bid))
            log(f"creating {spec.name} in {offer.region} at ${offer.price:.2f}/h: {spec.platform}/{spec.preset}, "
                f"{'spot price' if spec.spot else 'on-demand'}, {spec.disk_gib} GiB disk, "
                f"auto-shutdown after {spec.hours:g} h")
            rec.region = offer.region
            rec.times["create_start"] = utc()
            rec.creates.append({"region": offer.region, "price": offer.price, "start": rec.times["create_start"]})
            rec.save(job.out)
            t_create = time.time()
            try:
                inst = nb.create(create_argv(args.nebius, spec, cloud_init))
                rec.creates[-1]["minutes"] = round((time.time() - t_create) / 60, 2)
                if budget is not None:
                    budget.annotate(spec.name, region=offer.region, project=spec.project_id,
                                    created=rec.times["create_start"])
                break
            except InfraError as e:
                log(f"creation failed in {offer.region}: {e}")
                rec.errors.append(f"create in {offer.region}: {e}")
                rec.creates[-1].update(error=str(e)[-600:], minutes=round((time.time() - t_create) / 60, 2))
                teardown(nb, spec.name)  # leave nothing behind before trying the next region
        if inst is None:
            raise InfraError("could not create the VM in any qualifying region")
        price_spec = spec
        sampler = PriceSampler(lambda: nb.estimate_hourly(price_spec.platform, price_spec.preset, price_spec.spot),
                               rec.price_samples, on_sample=lambda: rec.save(job.out))
        sampler.start(rec.times["create_start"], offer.price)
        rec.instance_id = inst.id
        inst = wait_running(nb, inst)
        rec.ip, rec.times["running"] = inst.ip, utc()
        log(f"{inst.id} RUNNING at {inst.ip} ({time.time() - created:.0f}s)")
        remote = Remote(f"{spec.user}@{inst.ip}", job.out / "known_hosts", str(Path(key).expanduser()))
        rec.host = remote.host
        rec.save(job.out)
        job_code = remote_steps(remote, job, rec, check_vm)
        code = 0 if job_code in (None, 0) else job_code
        if code:
            log(f"job failed with exit code {code}")
    except VmStopped as e:
        code = vm_stopped(rec, str(e), created, spec.hours)
    except KeyboardInterrupt as e:
        log(f"interrupted ({e or 'Ctrl-C'})")
        rec.errors.append(f"interrupted ({e or 'Ctrl-C'})")
        code = EXIT_INTERRUPTED
    except Exception as e:  # noqa: BLE001 — anything else is an infrastructure failure, still tear down
        traceback.print_exc()
        rec.errors.append(f"{type(e).__name__}: {e}")
        code = EXIT_INFRA
        try:  # SSH failing during bootstrap, push or setup is often the VM being stopped under us
            check_vm()
        except VmStopped as stopped:
            code = vm_stopped(rec, str(stopped), created, spec.hours)
        except Exception:  # noqa: BLE001
            pass
    finally:
        with signals_held() as held:  # Ctrl-C, SIGTERM and SIGHUP wait until the VM is deleted
            if remote is not None and rec.stopped_state is None:
                try:
                    held.cut_collection = True  # SIGTERM/SIGHUP skip the rest of this best-effort step
                    try:
                        remove_secrets(remote, job.secrets)  # if the job step's trap didn't (a kept VM keeps its RAM)
                        collect(remote, job, rec)
                    finally:
                        held.cut_collection = False
                except LauncherSignal as e:
                    log(f"collection cut short ({e}): deleting the VM first")
                    rec.errors.append(f"collection cut short by {e}")
            keep = (keep_always or (args.cmd == "up" and code == 0)) and inst is not None  # no VM: nothing to keep
            if keep:
                rec.kept = True
                if budget is not None:
                    budget.annotate(spec.name, kept=True)
                log(f"VM kept. ssh -i {key} -o UserKnownHostsFile={job.out / 'known_hosts'} {rec.host}")
                log(f"auto-shutdown {spec.hours:g} h after boot stops it but does NOT delete it "
                    f"(extend: ssh … 'sudo shutdown -h +MINUTES'); "
                    f"delete: python3 scripts/nebius_job.py down --name {spec.name}"
                    + (f" --budget-file {budget.path}" if budget is not None else ""))
            else:
                if sampler is not None:
                    sampler.stop()
                    if inst is not None:
                        sampler.sample_now()  # the price at the end, so a late rise is counted
                if nb is not None:
                    try:
                        teardown(nb, spec.name, rec.instance_id)
                        rec.teardown_verified = True
                    except Exception as e:  # noqa: BLE001
                        rec.teardown_verified = False
                        rec.errors.append(f"teardown: {e}")
                        log(f"!!! TEARDOWN FAILED: {e}\n"
                            f"!!! the VM may still bill; `reap` retries it (or: nebius_job.py down --name {spec.name})")
                        code = EXIT_INFRA  # never relaunch on top of a VM we couldn't delete
                rec.times["deleted"] = utc()
            end = rec.times.get("deleted") or utc()
            rec.estimated_cost_usd = estimated_cost(rec.price_samples, end)
            settle_usd = estimated_cost(rec.price_samples, end, conservative=True, cap_hourly=worst_hourly)
            if budget is not None and not rec.kept:
                if rec.teardown_verified is False:  # still reserved at the worst case until reap deletes it
                    budget.annotate(spec.name, teardown_failed=True)
                elif settle_usd is not None:
                    budget.settle(spec.name, settle_usd, f"conservative estimate from the sampled "
                                                         f"{'spot' if spec.spot else 'on-demand'} prices")
                elif inst is None:
                    budget.settle(spec.name, 0.0, "no VM was created")
                else:  # created but no price samples: keep the worst case
                    budget.settle(spec.name, worst, "no price samples; settled at the worst case")
                log(budget.summary())
            if rec.estimated_cost_usd is not None:
                log(f"estimated cost ${rec.estimated_cost_usd:.2f} in {rec.region} "
                    "(" + ("so far; the VM is still running" if rec.kept else
                           f"from the sampled {'spot' if spec.spot else 'on-demand'} prices") + ")")
            rec.save(job.out)
            log(f"record: {job.out / 'job.json'}")
        if held.signum is not None:  # acted on now that the VM is deleted: no further attempt
            what = signal.Signals(held.signum).name
            log(f"{what} received while cleaning up: stopping here")
            rec.errors.append(f"{what} while the VM was being cleaned up; the launcher stopped after it")
            rec.save(job.out)
            if rec.teardown_verified is not False:  # a failed delete keeps its own exit code (2): never relaunched
                code = EXIT_INTERRUPTED
    return code, rec


def vm_stopped(rec: Record, state: str, created: float, hours: float) -> int:
    """Classify a VM that left RUNNING. If the job had already ended, its own code stands: 75 is resumed, and 0 is
    done (it saved its own results; only the final sync or collection was cut). Otherwise it is a spot preemption,
    or the --hours cap if that is when it happened."""
    rec.stopped_state = state
    if rec.exit_code in (0, EXIT_UNFINISHED):
        log(f"the VM went {state} after the job had exited {rec.exit_code}")
        rec.errors.append(f"VM {state} after the job ended (final upload/collection incomplete)")
        return rec.exit_code
    rec.preempted = time.time() - created < hours * 3600 - 300
    why = "spot preemption" if rec.preempted else "the --hours auto-shutdown"
    log(f"the VM went {state} mid-run ({why})")
    rec.errors.append(f"VM {state} ({why})")
    return EXIT_STOPPED


NEBIUS_ONLY = ("platform", "preset", "spot", "on_demand", "max_gpu_price", "regions", "regions_file", "project_id",
               "subnet_id", "image_family", "image_project", "user", "ssh_key", "nebius", "keep", "root_job",
               "sandbox_python")
CARRY_SECRETS = ("secret", "upload", "hf_token_file")
VAST_ONLY = ("gpu", "gpus", "offer", "min_cuda", "max_cost", "purpose", "tag", "image", "reserve_minutes", "hub_gpu",
             "rsync")


def changed_flags(args) -> set[str]:
    """The ``run`` flags whose value is not their default (with the same environment): those given on the command
    line, unless given their default value."""
    defaults = vars(build_parser().parse_args(["run", "--hours", "1", "--out", "-"]))
    return {k for k, v in vars(args).items() if k in defaults and k not in ("hours", "out") and v != defaults[k]}


def flag_names(dests) -> str:
    return ", ".join(f"--{d.replace('_', '-')}" for d in sorted(dests))


def vast_run(args, cmd: list[str]) -> int:
    """``run --provider vast``: the job on a fresh vast.ai instance through hub-gpu, ended and verified gone whatever
    happens (module docstring). Every refusal comes before anything is started."""
    changed = changed_flags(args)
    if mixed := changed & set(NEBIUS_ONLY):
        raise SystemExit(f"--provider vast: {flag_names(mixed)} are Nebius's (vast.ai: --gpu, --gpus, --offer, "
                         "--image; no --keep, no --root-job)")
    if carry := changed & set(CARRY_SECRETS):
        raise SystemExit(f"--provider vast: {flag_names(carry)} refused: no secret goes to a vast.ai instance, someone "
                         "else's machine (an HF bucket upload needs a token); results come back with --fetch")
    if args.attempts not in (None, 1):
        raise SystemExit("--provider vast: no --attempts: a vast.ai instance is not preempted, and what one saved "
                         "stays on it (relaunch by hand, --push-ing the fetched results back)")
    args.attempts = 1
    args.gpus = 1 if args.gpus is None else args.gpus
    args.min_cuda = VAST_MIN_CUDA if args.min_cuda is None else args.min_cuda
    if args.gpus < 1 or args.hours <= 0:
        raise SystemExit("--provider vast: --gpus and --hours must be positive")
    reserve = (min(VAST_RESERVE_MINUTES, args.hours * 15) if args.reserve_minutes is None else args.reserve_minutes)
    if not 0 <= reserve * 120 < args.hours * 3600:
        raise SystemExit(f"--reserve-minutes {reserve:g} leaves the job no time: keep it under half of --hours")
    if os.environ.get("NEBIUS_MAX_GPU_PRICE", "").strip():
        log(f"--provider vast: ignoring NEBIUS_MAX_GPU_PRICE={os.environ['NEBIUS_MAX_GPU_PRICE'].strip()}, a Nebius "
            "spot bid (vast.ai: --max-price caps the instance's price)")
    name = vast_name(args.name)
    job = job_spec(args, cmd, name)
    keys = [f"--env {k} (it looks like a key)" for k, v in job.env if secret_env(k, v)]
    keys += [f"--push {local} ({why})" for local, _ in job.pushes if (why := secret_push(local))]
    if keys:
        raise SystemExit(f"--provider vast: no secret goes to a vast.ai instance, someone else's machine: "
                         f"{'; '.join(keys)}")
    code = check_code(job.allow_dirty)
    job = replace(job, provider="vast", reserve=reserve * 60, rsync=args.rsync)
    if shutil.which(args.rsync) is None and not args.dry_run:
        raise SystemExit(f"rsync ({args.rsync}) not found on this machine: --fetch from vast.ai needs it (--rsync or "
                         "RSYNC_BIN; on hub, rsync in .hub/apt-packages)")
    if args.budget_file and not Path(args.budget_file).expanduser().exists():
        raise SystemExit(f"budget ledger {args.budget_file} not found: refusing to spend without it "
                         "(create it with {\"limit_usd\": N, \"entries\": []})")
    budget = Budget(args.budget_file) if args.budget_file else None
    return vast_attempt(args, HubGpu(args.hub_gpu), name, job, code, budget)


def vast_survey(hub: HubGpu, args, disk: int, tried: set = frozenset()) -> tuple[list[dict], list[dict]]:
    """The offers for the instance and the ones to try, in order: those that qualify, cheapest first, but the
    ``tried`` ones; or ``--offer``."""
    offers = hub.offers(gpu=args.gpu, gpus=args.gpus, disk_gb=disk, max_price=args.max_price)
    if args.offer is None:
        return offers, [o for o in vast_candidates(offers, args.min_cuda, args.max_price) if o["offer"] not in tried]
    mine = next((o for o in offers if o["offer"] == args.offer), None)
    if mine is None:
        raise SystemExit(f"--offer {args.offer} is not among the 50 cheapest offers now for {args.gpus}× "
                         f"{args.gpu or 'any GPU'} with {disk} GB of disk" + (f" under ${args.max_price:g}/h" if
                                                                             args.max_price is not None else "")
                         + ": give the --gpu and --gpus it has (hub-gpu offers), or another one (offers come and go "
                           "within minutes)")
    if why := offer_skip(mine, args.min_cuda, args.max_price):
        raise SystemExit(f"--offer {args.offer}: {why} (--min-cuda, --max-price)")
    return offers, [mine]


def vast_attempt(args, hub: HubGpu, name: str, job: JobSpec, launch_code: dict, budget: Budget | None) -> int:
    """The instance: hub-gpu limits, the offer (waiting up to ``--wait-hours`` for one), its worst case, the start (the
    next offer if it fails), the SSH part, the fetch, the end and its check, the ledger. Returns the exit code."""
    lim = hub.limits()
    if problem := limits_problem(lim, args.gpus, args.hours):
        raise SystemExit(f"hub-gpu limits: {problem}")
    disk = disk_gb(args.disk_gib)
    log(f"checking vast.ai offers: {args.gpus}× {args.gpu or 'any GPU'}, {disk} GB of disk, CUDA ≥ {args.min_cuda:g}"
        + (f", at most ${args.max_price:g}/h" if args.max_price is not None else " (no price cap)"))
    offers, order = vast_survey(hub, args, disk)
    why = "--offer" if args.offer is not None else "cheapest"
    print(vast_offer_table(offers, args.min_cuda, args.max_price, order[0]["offer"] if order else None, why),
          flush=True)
    checks = 0 if args.dry_run or args.offer is not None else max(0, round(args.wait_hours * 3600 /
                                                                         PRICE_RECHECK_SECONDS))
    while not order and checks > 0:
        log(f"no offer qualifies now (GPU, CUDA, price, disk); checking again in {PRICE_RECHECK_SECONDS // 60} min "
            f"({checks} more checks, --wait-hours)")
        time.sleep(PRICE_RECHECK_SECONDS)
        checks -= 1
        offers, order = vast_survey(hub, args, disk)
        if order:
            print(vast_offer_table(offers, args.min_cuda, args.max_price, order[0]["offer"]), flush=True)
    rec = Record(mode="run", provider="vast", name=name, spot=False, hours=args.hours, command=job.command,
                 flags={k: v for k, v in vars(args).items() if k not in ("func",)}, max_price=args.max_price,
                 offers=offers[:16], **job_fields(job))
    if not order:
        if args.dry_run:
            print("== nothing would be started: no offer qualifies (GPU, CUDA, price, disk) ==")
            return EXIT_INFRA
        rec.errors.append("no offer qualifies (GPU, CUDA, price, disk); nothing started")
        rec.save(job.out)
        log("no offer qualifies; nothing started")
        return EXIT_INFRA
    worst = float(order[0]["usd_per_hour"]) * args.hours
    if problem := limits_problem(lim, args.gpus, args.hours, worst, args.max_cost):
        raise SystemExit(f"{name}: {problem}")
    log(f"worst case ${worst:.2f} (${float(order[0]['usd_per_hour']):.3f}/h × {args.hours:g} h); hub-gpu limits: "
        f"${float(lim.get('room_usd') or 0):.2f} left in any 24 hours"
        + (f"; --max-cost ${args.max_cost:g}" if args.max_cost is not None else ""))
    if name in {r.get("name") for r in hub.instances()}:
        raise SystemExit(f"{name} is taken among this project's instances: run again for another suffix")
    purpose, tags = vast_purpose(args.purpose, job.command), vast_tags(args.tag, launch_code)
    what = f"{name} ({args.gpus}× {order[0].get('gpu')} on vast.ai, ≤{args.hours:g} h)"
    if args.dry_run:
        return vast_dry_run(args, job, name, order[0], purpose, tags, disk, worst, lim, budget)
    if budget is not None:
        budget.record(name, worst, what, pid=os.getpid(), host=socket.gethostname(), out=os.path.abspath(job.out),
                      hours=args.hours)
        log(budget.summary())
    rec.vast = {"limits": lim, "disk_gb": disk, "min_cuda": args.min_cuda, "purpose": purpose, "tags": tags,
                "image": args.image, "reserve_minutes": round(job.reserve / 60, 2)}
    instance = remote = started = ends = None
    price = float(order[0]["usd_per_hour"])
    code = EXIT_INFRA
    t0 = time.time()

    def check_vm() -> None:
        if instance is not None:
            vast_state(hub, instance)  # VmStopped if it went away

    tried: set = set()
    try:
        while started is None and len(tried) < VAST_STARTS:
            if tried:  # refused (taken meanwhile): once more, with the cheapest of a fresh listing
                if args.offer is not None:
                    break
                offers, order = vast_survey(hub, args, disk, tried)
                if not order:
                    log("no other offer qualifies")
                    break
            offer = order[0]
            price = float(offer["usd_per_hour"])
            if tried and (problem := limits_problem(lim, args.gpus, args.hours, price * args.hours, args.max_cost)):
                log(f"not trying offer {offer['offer']}: {problem}")
                break
            tried.add(offer["offer"])
            log(f"starting {name}: offer {offer['offer']}, {offer.get('gpus')}× {offer.get('gpu')} in "
                f"{offer.get('where')} at ${price:.3f}/h, {disk} GB of disk, for {args.hours:g} h (then destroyed)")
            rec.region = offer.get("where")
            rec.times["create_start"] = utc()
            rec.creates.append({"offer": offer["offer"], "price": price, "where": offer.get("where"),
                                "start": rec.times["create_start"]})
            rec.save(job.out)
            t = time.time()
            try:
                started = hub.start(offer["offer"], name=name, purpose=purpose, tags=tags, hours=args.hours,
                                    disk_gb=disk, image=args.image)
                rec.creates[-1]["minutes"] = round((time.time() - t) / 60, 2)
                rec.vast["offer"] = offer
            except InfraError as e:
                log(f"start failed: {e}")
                rec.errors.append(f"start of offer {offer['offer']}: {e}")
                rec.creates[-1].update(error=str(e)[-600:], minutes=round((time.time() - t) / 60, 2))
                vast_teardown(hub, name)  # if the start did make an instance after all, end it before the next
        if started is None:
            raise InfraError(f"could not start an instance: {len(tried)} offer(s) refused"
                             + (f", the last with: {rec.creates[-1].get('error')}" if rec.creates else ""))
        instance = started["instance"]
        price = float(started.get("usd_per_hour") or price)
        ends = float(started.get("deadline") or _seconds(rec.times["create_start"]) + args.hours * 3600)
        job = replace(job, ends=ends)  # the job's deadline, and how long the fetch may take
        rec.instance_id = str(instance)
        rec.price_samples.append([rec.times["create_start"], price])
        rec.vast.update(start=started, ends_utc=datetime.fromtimestamp(ends, UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
        rec.save(job.out)
        if budget is not None:
            budget.annotate(name, instance=instance, usd=round(price * args.hours, 2), ends=rec.vast["ends_utc"])
        log(f"instance {instance}: ${price:.3f}/h, destroyed at {rec.vast['ends_utc']} "
            f"(${float(started.get('cost_usd') or price * args.hours):.2f} if it runs to then)")
        row = vast_wait_running(hub, instance)
        target, port = vast_ssh(row)
        remote = Remote(target, job.out / "known_hosts", str(Path(HUB_GPU_KEY).expanduser()), port=port)
        rec.host, rec.ip, rec.times["running"] = target, target.rpartition("@")[2], utc()
        rec.vast["ssh"] = {"host": target, "port": port, "state": row.get("state")}
        rec.save(job.out)
        log(f"instance {instance} is {row.get('state')} ({time.time() - t0:.0f}s), SSH at ssh -p {port} {target}")
        vast_wait_ssh(hub, remote, instance, bool(started.get("ssh_key_attached", True)), until=ends - job.reserve)
        job_code = remote_steps(remote, job, rec, check_vm)
        code = 0 if job_code in (None, 0) else job_code
        if code:
            log(f"job failed with exit code {code}")
    except VmStopped as e:
        code = vast_stopped(rec, str(e), ends)
    except KeyboardInterrupt as e:
        log(f"interrupted ({e or 'Ctrl-C'})")
        rec.errors.append(f"interrupted ({e or 'Ctrl-C'})")
        code = EXIT_INTERRUPTED
    except Exception as e:  # noqa: BLE001 — anything else is an infrastructure failure: still fetch and end
        traceback.print_exc()
        rec.errors.append(f"{type(e).__name__}: {e}")
        code = EXIT_INFRA
        if ends is not None and time.time() > ends - job.reserve:
            log("the instance's end is near (--reserve-minutes): fetching what the job made so far, then ending it")
        try:  # SSH failing is often the instance going away under us
            check_vm()
        except VmStopped as stopped:
            code = vast_stopped(rec, str(stopped), ends)
        except Exception:  # noqa: BLE001
            pass
    finally:
        with signals_held() as held:  # Ctrl-C, SIGTERM and SIGHUP wait until the instance is ended
            if remote is not None and rec.stopped_state is None:
                try:
                    held.cut_collection = True  # SIGTERM/SIGHUP skip the rest of this best-effort step
                    try:
                        collect(remote, job, rec)
                    finally:
                        held.cut_collection = False
                except LauncherSignal as e:
                    log(f"collection cut short ({e}): ending the instance first")
                    rec.errors.append(f"collection cut short by {e}")
            if rec.creates:  # a start was tried: end whatever it made
                try:
                    rec.vast["end"] = vast_teardown(hub, name, instance)
                    rec.teardown_verified = True
                except Exception as e:  # noqa: BLE001
                    rec.teardown_verified = False
                    rec.errors.append(f"teardown: {e}")
                    log(f"!!! END FAILED: {e}\n!!! the instance bills until vast.ai destroys it"
                        + (f" at {rec.vast.get('ends_utc')}" if ends else "") + f": hub-gpu end {name}")
                    code = EXIT_INFRA
                rec.times["deleted"] = utc()
                if not rec.price_samples and rec.vast.get("end"):  # started, though the answer never came
                    rec.price_samples.append([rec.times["create_start"],
                                              float(rec.vast["end"].get("usd_per_hour") or price)])
            end = rec.times.get("deleted") or utc()
            if ends is not None:  # it can't have run past its end
                end = min(end, datetime.fromtimestamp(ends, UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
            rec.estimated_cost_usd = estimated_cost(rec.price_samples, end)
            if budget is not None:
                if rec.teardown_verified is False:  # still at its worst case: it may run until its end
                    budget.annotate(name, teardown_failed=True)
                elif rec.estimated_cost_usd is not None:
                    budget.settle(name, rec.estimated_cost_usd, "vast.ai: its price × the time from its start to its "
                                                                "verified end")
                else:
                    budget.settle(name, 0.0, "no instance was started")
                log(budget.summary())
            if rec.estimated_cost_usd is not None:
                log(f"estimated cost ${rec.estimated_cost_usd:.2f} (${price:.3f}/h from its start to its verified end; "
                    "hub-gpu limits shows what was committed)")
            rec.save(job.out)
            log(f"record: {job.out / 'job.json'}")
        if held.signum is not None:
            what = signal.Signals(held.signum).name
            log(f"{what} received while cleaning up: stopping here")
            rec.errors.append(f"{what} while the instance was being ended; the launcher stopped after it")
            rec.save(job.out)
            if rec.teardown_verified is not False:
                code = EXIT_INTERRUPTED
    return code


def vast_stopped(rec: Record, why: str, ends: float | None) -> int:
    """Classify an instance that went away (``vast_gone``). If the job had already ended, its own code stands, as on
    Nebius. Otherwise it was destroyed at its end (``--hours``), if that is when, or it disappeared or failed mid-job:
    exit 3 either way (nothing relaunches on vast.ai)."""
    rec.stopped_state = why
    if rec.exit_code in (0, EXIT_UNFINISHED):
        log(f"the instance is {why}, after the job had exited {rec.exit_code}")
        rec.errors.append(f"instance {why} after the job ended (collection incomplete)")
        return rec.exit_code
    what = ("destroyed at its end (--hours)" if ends is not None and time.time() > ends - 120 else
            "it disappeared or failed mid-job")
    log(f"!!! the instance is {why}: {what}")
    rec.errors.append(f"instance {why} ({what})")
    return EXIT_STOPPED


def vast_dry_run(args, job: JobSpec, name: str, offer: dict, purpose: str, tags: list[str], disk: int, worst: float,
                 lim: dict, budget: Budget | None) -> int:
    """Print what ``run --provider vast`` would do with ``offer``. Nothing is started and nothing goes over SSH; the
    only hub-gpu calls (before this) are the read-only limits, offers and list."""
    mine = lim["limits"]
    print(f"== limits (hub-gpu limits) ==\n${float(mine['usd_per_day']):g} a day, {mine['gpus']} GPUs at once, timed "
          f"runs of at most {float(mine['hours']):g} h; ${float(lim['room_usd']):.2f} left in any 24 hours, "
          f"{lim.get('gpus_in_use', 0)} GPUs in use\nthis instance: ${float(offer['usd_per_hour']):.3f}/h × "
          f"{args.hours:g} h = ${worst:.2f} at most" + (f" (--max-cost ${args.max_cost:g})" if args.max_cost is not None
                                                        else "") + ": ok")
    argv = [args.hub_gpu, "--json", *start_args(offer["offer"], name=name, purpose=purpose, tags=tags,
                                                 hours=args.hours, disk_gb=disk, image=args.image)]
    print(f"== start (dry run: nothing is started) ==\n{shlex.join(argv)}\nvast.ai destroys it {args.hours:g} h after "
          "its start, with whatever is on it")
    key = Path(HUB_GPU_KEY).expanduser()
    print(f"== ssh ==\nroot@<host> -p <port> from hub-gpu list, with hub-gpu's key {key}"
          + ("" if key.is_file() else " (hub-gpu makes it at its first start)")
          + f"; known_hosts {job.out / 'known_hosts'}")
    deadline = int(time.time() + args.hours * 3600 - job.reserve)
    print_remote_plan(job, "root@<host>", vast_bootstrap_script(), deadline)
    print(f"(the job's NEBIUS_JOB_DEADLINE, {deadline} here: the instance's end less {job.reserve / 60:g} min kept for "
          f"the fetch; the launcher stops waiting for the job {job.reserve / 120:g} min before the end)")
    print(f"== fetch (rsync over ssh, resumable: every {VAST_FETCH_EVERY / 60:g} min while the job runs, skipping "
          f"{' '.join(IN_PROGRESS)}, then once more) ==")
    for path in job.fetches:
        print(f"{remote_path(path)} -> {fetch_dest(path, job.out) / os.path.basename(path.rstrip('/'))}")
    print(f"~/{REMOTE_LOGS}/ -> {job.out / 'remote_logs'}; job.json -> {job.out / 'job.json'}")
    found = shutil.which(job.rsync)
    print(f"rsync here: {found}" if found else f"!! no rsync here ({job.rsync}): a real run refuses to start (--rsync, "
          "RSYNC_BIN; on hub: .hub/apt-packages)")
    print("== no secrets ==\nnone goes to the instance, someone else's machine: no --secret, no HF token (--hf "
          "downloads are unauthenticated), no --upload; results come back only through --fetch")
    print(f"== teardown ==\n{name}: hub-gpu end, then verified ended in hub-gpu list (after errors, Ctrl-C, SIGTERM "
          "and SIGHUP too)")
    if budget is not None:
        print(f"== budget: a vast.ai entry of at most ${worst:.2f} would be recorded in {budget.path}, not counted "
              f"against its limit ({budget.summary()}) ==")
    return 0


def cmd_exec(args, cmd: list[str]) -> int:
    """The SSH part only, on an existing box (an ``up`` VM, any CUDA machine, or any box with ``--no-gpu``)."""
    job = job_spec(args, cmd, args.name)
    if job.command is None:
        raise SystemExit("exec needs a job command after --")
    check_code(job.allow_dirty)
    rec = Record(mode="exec", host=args.host, command=job.command,
                 flags={k: v for k, v in vars(args).items() if k not in ("func",)}, **job_fields(job))
    remote = Remote(args.host, job.out / "known_hosts", str(Path(args.ssh_key).expanduser()) if args.ssh_key else None)
    code = EXIT_INFRA
    try:
        job.out.mkdir(parents=True, exist_ok=True)
        job_code = remote_steps(remote, job, rec)
        code = job_code or 0
    except KeyboardInterrupt:
        rec.errors.append("interrupted (Ctrl-C); the remote job keeps running")
        code = EXIT_INTERRUPTED
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        rec.errors.append(f"{type(e).__name__}: {e}")
    finally:
        if job.uploads:  # the upload step removes it, but not if we stopped before that
            with contextlib.suppress(Exception):
                remote.run(f'rm -f "{REMOTE_HF_TOKEN}"', timeout=60, check=False)
        remove_secrets(remote, job.secrets)  # the job step's trap removes them, but not if we stopped before it
        collect(remote, job, rec)
        rec.save(job.out)
        log(f"record: {job.out / 'job.json'}")
    return code


def _clients(args) -> list[tuple[Region, Nebius]]:
    _, regions = resolve_regions(args, need_subnet=False)
    return [(region, Nebius(args.nebius, region.project)) for region in regions]


def cmd_down(args, cmd: list[str]) -> int:
    """Delete a VM (by name or id) and its boot disk in whichever region it is, then verify. With the ledger, a kept
    VM's reservation (``up``, ``run --keep``) is then settled from the prices its launcher sampled (``reaped_cost``)."""
    if not (args.name or args.id):
        raise SystemExit("down needs --name or --id")
    found = []
    for region, nb in _clients(args):
        name = args.name
        if args.id and not name:
            try:
                name = nb.get(args.id).name
            except InfraError:
                continue  # not in this region
        here = (args.id and not args.name) or any(i.name == name for i in nb.instances()) \
            or any(d.name == f"{name}-boot" for d in nb.disks())
        if here:
            log(f"{region.name}: deleting {name}")
            teardown(nb, name, args.id)
            found.append(name)
    if not found:
        log(f"nothing named {args.name or args.id!r} in any region; nothing to delete")
    elif args.budget_file and Path(args.budget_file).expanduser().exists():
        budget = Budget(args.budget_file)
        for name in dict.fromkeys(found):
            settle_kept(budget, name, "kept VM deleted; settled by down")
    return 0


def cmd_list(args, cmd: list[str]) -> int:
    for region, nb in _clients(args):
        instances, disks = nb.instances(), nb.disks()
        print(f"{region.name} ({region.project}): {len(instances)} instances, {len(disks)} disks")
        for i in instances:
            print(f"  {i.name:32} {i.id:40} {i.state:10} {i.ip or '-':16} {i.platform:14} "
                  f"{'spot' if i.preemptible else 'on-demand':9} {i.created_at}")
        for d in disks:
            print(f"  {d.name:32} {d.id:40} {d.state:10} {d.size_gib} GiB")
    return 0


def cmd_budget(args, cmd: list[str]) -> int:
    if not args.budget_file:
        raise SystemExit("--budget-file (or $NEBIUS_BUDGET_FILE) is required")
    b = Budget(args.budget_file)
    for e in b.data["entries"]:
        print(f"  {e['when'][:16]}  {e['kind']:8}  ${e['usd']:7.2f}  {e.get('what', '')}"
              + ("  (vast.ai: not counted)" if e.get("provider") == "vast" else ""))
    print(b.summary())
    return 0


def cmd_reap(args, cmd: list[str]) -> int:
    """Settle the reservations of launchers that died, deleting their stopped VMs (run/up do this first anyway)."""
    if not args.budget_file:
        raise SystemExit("--budget-file (or $NEBIUS_BUDGET_FILE) is required")
    _, regions = resolve_regions(args, need_subnet=False)
    budget = Budget(args.budget_file)
    reap(args, budget, all_regions(args, regions))
    log(budget.summary())
    return 0


def cmd_prices(args, cmd: list[str]) -> int:
    """Every region's current price and capacity for a VM; read-only."""
    tenant, regions = resolve_regions(args, need_subnet=False)
    spot = not args.on_demand
    offers = survey(args.nebius, regions, tenant, args.platform, args.preset, spot)
    order = candidates(offers, args.max_price)
    print(f"{'spot' if spot else 'on-demand'} {args.platform}/{args.preset} at {utc()}"
          + (f", cap ${args.max_price:g}/h" if args.max_price is not None else ""))
    print(offer_table(offers, args.max_price, order[0].region if order else None))
    return 0 if order else EXIT_INFRA


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    nebius = argparse.ArgumentParser(add_help=False)
    nebius.add_argument("--regions-file", default=os.environ.get("NEBIUS_REGIONS_FILE"),
                        help="JSON {tenant, regions: {NAME: {project, subnet, image_project}}} (env NEBIUS_REGIONS_FILE)")
    nebius.add_argument("--regions", help="comma-list: only these regions of the regions file")
    nebius.add_argument("--project-id", default=os.environ.get("NEBIUS_PROJECT_ID"),
                        help="a single project instead of a regions file (env NEBIUS_PROJECT_ID)")
    nebius.add_argument("--nebius", default=os.environ.get("NEBIUS_BIN", "nebius"), help="CLI binary (env NEBIUS_BIN)")

    machine = argparse.ArgumentParser(add_help=False)
    machine.add_argument("--platform", default="gpu-b300-sxm",
                         help="gpu-b300-sxm (default), or a CPU-only one, e.g. cpu-d3 (with --on-demand)")
    machine.add_argument("--preset", default="1gpu-24vcpu-346gb",
                         help="1gpu-24vcpu-346gb (default); cpu-d3: e.g. 16vcpu-64gb, 32vcpu-128gb")
    pricing = machine.add_mutually_exclusive_group()
    pricing.add_argument("--spot", action="store_true", help="preemptible at the current spot price (default)")
    pricing.add_argument("--on-demand", action="store_true", help="regular, non-preemptible VM")
    machine.add_argument("--max-price", type=float, default=None, metavar="USD",
                         help="skip regions whose price for the whole VM is above this, per hour (default: no cap)."
                              " Checked at launch; a spot price can still rise during the run. vast.ai: offers")

    remote = argparse.ArgumentParser(add_help=False)
    remote.add_argument("--out", required=True, help="local output dir; {stamp} and {name} are expanded")
    remote.add_argument("--push", action="append", default=[], metavar="LOCAL:REMOTE",
                        help="extra input (file or dir); REMOTE is relative to the remote repo unless absolute or ~/")
    remote.add_argument("--hf", action="append", default=[], metavar="ORG/REPO[@REV][=NAME]",
                        help=f"public HF repo to download into {RAM_DIR}/NAME")
    remote.add_argument("--upload", action="append", default=[], metavar="REMOTE_PATH=hf://buckets/NS/BUCKET/PREFIX",
                        help="after the job, sync this file/dir from the VM straight to an HF bucket; {stamp}/{name} "
                             "expand in the destination. The token is on the VM from the job's start, so the job can "
                             "save its own progress as it goes")
    remote.add_argument("--hf-token-file", default=os.environ.get("HF_TOKEN_FILE", "~/.cache/huggingface/token"),
                        help="HF token on this machine, on the VM with --upload only, from the job's start to the "
                             "upload's end (env HF_TOKEN_FILE)")
    remote.add_argument("--fetch", action="append", default=[], metavar="REMOTE_PATH",
                        help="file or dir to bring back into --out (relative to the remote repo unless absolute or ~/)")
    remote.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                        help="exported for the job (no secrets: --secret)")
    remote.add_argument("--secret", action="append", default=[], metavar="NAME[=ENV]",
                        help=f"the key in {SECRETS_DIR}/NAME on this machine (`hub-secret set NAME` puts it there) is "
                             "the job's $ENV (default: NAME upper-cased). It goes over ssh stdin into an owner-only "
                             f"file in {REMOTE_SECRETS} right before the job, which deletes it when it ends; never on "
                             "a command line, in a log or in job.json (names only). Refused if missing or empty")
    remote.add_argument("--root-job", action="store_true",
                        help="run the job as root, like the RL runs (the terminal env's read_only_verifier needs it). "
                             f"The setup installs uv's CPython --sandbox-python into {UV_PYTHON_DIR} (where the env's "
                             "sandbox user, nobody, can run it), syncs the venv on it, makes $HOME private (chmod "
                             "750) and checks all that as root. The job runs via sudo -E -H with its environment; "
                             "call .venv/bin/python (or uv run --no-sync) in it")
    remote.add_argument("--sandbox-python", default=None, metavar="VERSION",
                        help=f"with --root-job: the exact CPython of the venv and the sandbox (default "
                             f"{DEFAULT_SANDBOX_PYTHON}, the RL runs')")
    remote.add_argument("--extra-gpu", action="store_true", help="uv sync --extra gpu (flash-attn, causal-conv1d, …)")
    remote.add_argument("--poll-interval", type=float, default=15.0, help="seconds between job log polls")
    remote.add_argument("--allow-dirty", action="store_true",
                        help="push a working tree with uncommitted changes (refused otherwise); each push then saves "
                             f"their patch in its --out as {provenance.PATCH_FILE}, with the untracked files listed in "
                             f"{provenance.UNTRACKED_FILE}")

    vm = argparse.ArgumentParser(add_help=False)
    vm.add_argument("--subnet-id", default=os.environ.get("NEBIUS_SUBNET_ID"),
                    help="with --project-id only (env NEBIUS_SUBNET_ID)")
    vm.add_argument("--hours", type=float, required=True,
                    help="in-VM auto-shutdown after this long (spending cap); vast.ai: the instance is destroyed then")
    vm.add_argument("--ssh-key", default=os.environ.get("NEBIUS_SSH_KEY"),
                    help="private key; KEY.pub goes into cloud-init (env NEBIUS_SSH_KEY)")
    vm.add_argument("--name", default="md-job",
                    help="VM name prefix (a random suffix is added); vast.ai: at most 17 characters")
    vm.add_argument("--user", default="md", help="login user created on the VM")
    vm.add_argument("--max-gpu-price", type=float, default=None,  # else NEBIUS_MAX_GPU_PRICE, for spot: spot_bid
                    metavar="USD", help="spot bid per GPU-hour through a Nebius pricing policy: never charged more; "
                                        "the VM doesn't run while the market is above it (env NEBIUS_MAX_GPU_PRICE, "
                                        "which --on-demand ignores; the flag with --on-demand is refused)")
    vm.add_argument("--budget-file", default=os.environ.get("NEBIUS_BUDGET_FILE"),
                    help="spending ledger; refuse a VM whose worst case would exceed its limit (env NEBIUS_BUDGET_FILE)."
                         " A vast.ai instance is recorded there, not counted against the limit")
    vm.add_argument("--disk-gib", type=int, default=200,
                    help="boot disk (network SSD; weights go to RAM, not here). vast.ai: the instance's disk, where "
                         "the weights go if the container can't mount a RAM disk")
    vm.add_argument("--image-family", default="ubuntu24.04-cuda13.0",
                    help="ubuntu24.04-cuda13.0 (default); for a CPU VM, ubuntu24.04-driverless")
    vm.add_argument("--image-project", default="project-e03public-images",
                    help="with --project-id only: the region's public-image project (default: uk-south1's)")
    vm.add_argument("--wait-hours", type=float, default=2.0,
                    help="when no region qualifies (price above the bid or cap, no capacity), re-check every 15 min "
                         "for this long before giving up (vast.ai: no offer)")
    vm.add_argument("--dry-run", action="store_true", help="print the plan; create nothing")

    vast = argparse.ArgumentParser(add_help=False)
    vast.add_argument("--provider", choices=("nebius", "vast"), default="nebius",
                      help="nebius (default), or vast: a vast.ai instance through hub-gpu (the vast.ai flags below)")
    vast.add_argument("--gpu", help="vast.ai: the GPU model, as hub-gpu offers names it, e.g. B200 (default: any)")
    vast.add_argument("--gpus", type=int, help="vast.ai: how many GPUs (default 1)")
    vast.add_argument("--offer", type=int,
                      help="vast.ai: start this offer of hub-gpu offers, not the cheapest (give its --gpu and --gpus)")
    vast.add_argument("--min-cuda", type=float, metavar="VERSION",
                      help=f"vast.ai: skip offers whose driver runs an older CUDA (default {VAST_MIN_CUDA:g}: the "
                           "locked torch is a CUDA 13.0 build; from 13 on, GPUs older than Turing are skipped too)")
    vast.add_argument("--max-cost", type=float, metavar="USD",
                      help="vast.ai: refuse an instance whose price × --hours is above this (hub-gpu limits are "
                           "checked anyway)")
    vast.add_argument("--purpose", help="vast.ai: the instance's purpose line, which the owner sees (default: the job)")
    vast.add_argument("--tag", action="append", default=[],
                      help=f"vast.ai: a tag, e.g. experiment=v2 (repeatable, up to {VAST_MAX_TAGS - 2}; the launcher "
                           "adds launcher= and commit=)")
    vast.add_argument("--image", help="vast.ai: the docker image (default: hub-gpu's, vastai/pytorch)")
    vast.add_argument("--reserve-minutes", type=float, metavar="MIN",
                      help=f"vast.ai: kept before the instance's end for --fetch; the job's NEBIUS_JOB_DEADLINE is "
                           f"that much earlier, and the launcher stops waiting for the job halfway into it (default "
                           f"{VAST_RESERVE_MINUTES:g}, at most a quarter of --hours)")
    vast.add_argument("--hub-gpu", default=os.environ.get("HUB_GPU_BIN", "hub-gpu"),
                      help="vast.ai: the hub-gpu CLI (env HUB_GPU_BIN)")
    vast.add_argument("--rsync", default=os.environ.get("RSYNC_BIN", "rsync"),
                      help="vast.ai: this machine's rsync, for --fetch (env RSYNC_BIN)")

    p = sub.add_parser("run", parents=[nebius, machine, vm, remote, vast],
                       help="cheapest region → create → set up → job → fetch → delete")
    p.add_argument("--keep", action="store_true", help="do not delete the VM at the end")
    p.add_argument("--attempts", type=int, default=None,
                   help=f"VMs to use in all (default 3): after a spot preemption, or a job exit of {EXIT_UNFINISHED} "
                        "(stopped early, can resume), a fresh VM runs the job again in the then-cheapest region. "
                        "vast.ai: 1, no relaunch")
    p.set_defaults(func=cmd_vm)
    p = sub.add_parser("up", parents=[nebius, machine, vm, remote],
                       help="cheapest region → create + set up (+ downloads), keep the VM")
    p.set_defaults(func=cmd_vm, keep=True, attempts=1)
    p = sub.add_parser("exec", parents=[remote], help="set up + job + fetch on an existing SSH box")
    p.add_argument("--host", required=True, help="USER@HOST")
    p.add_argument("--no-gpu", action="store_true",
                   help="a box without a GPU: the setup runs uv sync and shows the Python version, no torch/CUDA check")
    p.add_argument("--ssh-key", default=os.environ.get("NEBIUS_SSH_KEY"))
    p.add_argument("--name", default="exec", help="only used for {name} in --out")
    p.set_defaults(func=cmd_exec)
    p = sub.add_parser("down", parents=[nebius], help="delete a VM and its boot disk, then verify")
    p.add_argument("--name")
    p.add_argument("--id")
    p.add_argument("--budget-file", default=os.environ.get("NEBIUS_BUDGET_FILE"),
                   help="the spending ledger: settle the deleted VM's reservation if it was kept (up, run --keep)")
    p.set_defaults(func=cmd_down)
    p = sub.add_parser("list", parents=[nebius], help="instances and disks in every region")
    p.set_defaults(func=cmd_list)
    p = sub.add_parser("prices", parents=[nebius, machine], help="every region's current price + capacity")
    p.set_defaults(func=cmd_prices)
    p = sub.add_parser("budget", help="the spending ledger: limit, spent, reserved")
    p.add_argument("--budget-file", default=os.environ.get("NEBIUS_BUDGET_FILE"), required=False)
    p.set_defaults(func=cmd_budget)
    p = sub.add_parser("reap", parents=[nebius],
                       help="settle reservations of launchers that died; delete their stopped VMs")
    p.add_argument("--budget-file", default=os.environ.get("NEBIUS_BUDGET_FILE"), required=False)
    p.set_defaults(func=cmd_reap)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1:]
    args = build_parser().parse_args(argv)
    if args.cmd == "up" and cmd:
        raise SystemExit("up takes no job command (use run, or exec against the kept VM)")
    if args.cmd == "run" and not cmd:
        raise SystemExit("run needs a job command after --")
    return args.func(args, cmd)


if __name__ == "__main__":
    sys.exit(main())
