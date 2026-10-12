#!/usr/bin/env python3
"""Run a job on a fresh Nebius GPU VM (or any SSH box) and always clean up.

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
The job gets PYTHONUNBUFFERED=1, NEBIUS_JOB_RAM=/mnt/ram (where ``--hf`` puts weights) and each
``--env K=V``. ``{stamp}`` and ``{name}`` in ``--out`` are expanded.

What ``run`` does (``exec`` is the SSH part only, so a non-Nebius box works too):
1. Pick the region (above). Create the VM there, wait for RUNNING with a public IP, then wait for SSH.
2. Bootstrap: RAM disk, lingering, uv.
3. Start the ``--hf`` downloads, detached.
4. Push the repo, with its ``CODE_VERSION.json`` (below), and the ``--push`` inputs.
5. ``uv sync --locked``. If torch can't see the GPU, fall back to scripts/setup_box.sh.
6. Wait for the downloads, then run the job, detached and polled.
   - With ``--upload``, the HF token reaches the VM before the job, so the job can save its own progress as it goes
     (``extract_probe_features.py --store``). The token stays until the upload step ends.
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
- No secrets go to the VM. Only public HF repos are downloaded (no HF_TOKEN), and the repo is pushed
  as ``git ls-files`` plus untracked-but-not-ignored files, so .env*, .venv and .git never travel.

Exit status: 0 on success; the job's own exit code if it failed; 2 for an infrastructure failure;
3 if the VM stopped mid-run (spot preemption or the --hours cap); 75 if the job stopped early to resume
later and no attempt was left; 130 on Ctrl-C, SIGTERM or SIGHUP (also when it arrived while the VM was deleted).
Stdlib only, so plain ``python3`` (3.11+) runs it.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.util
import json
import os
import re
import secrets
import shlex
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


def setup_script(extra_gpu: bool) -> str:
    extra = " --extra gpu" if extra_gpu else ""
    return f"""set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked{extra}
if ! uv run --no-sync python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "torch cannot see the GPU with the locked wheel: retargeting via scripts/setup_box.sh"
  bash scripts/setup_box.sh{extra}
fi
uv run --no-sync python -c 'import torch as t; print("torch", t.__version__, t.version.cuda, t.cuda.get_device_name(0))'
"""


def job_script(command: str, env: list[tuple[str, str]], poweroff_minutes: int | None = None,
               deadline_fallback: int | None = None) -> str:
    """The job step.

    - It exports ``NEBIUS_JOB_DEADLINE``, the VM's scheduled shutdown (the ``--hours`` cap) in epoch seconds, and
      saves it for the upload step. A job can then stop cleanly before the cap; extract_probe_features.py does.
      If systemd's record of the shutdown can't be read, ``deadline_fallback`` is used instead: the launcher's
      estimate, VM creation + ``--hours``, a little before the real shutdown.
    - With ``poweroff_minutes``, the VM powers itself off that long after the job ends, unless a sooner shutdown is
      already scheduled. So if the launcher has died, the VM stops on its own instead of running until ``--hours``.
    """
    exports = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in env)
    head = (f'export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1 NEBIUS_JOB_RAM={RAM_DIR} '
            f'HF_XET_HIGH_PERFORMANCE=1 HF_TOKEN_PATH="{REMOTE_HF_TOKEN}"\n{exports}'
            f'usec=$(sudo -n cat {SCHEDULED_SHUTDOWN} 2>/dev/null | sed -n "s/^USEC=//p")\n'
            + (f'[ -n "$usec" ] || usec={int(deadline_fallback)}000000\n' if deadline_fallback else "")
            + 'if [ -n "$usec" ]; then\n'
            '  export NEBIUS_JOB_DEADLINE=$((usec / 1000000))\n'
            f'  echo "$NEBIUS_JOB_DEADLINE" > "$HOME/{REMOTE_LOGS}/deadline"\n'
            'fi\n')
    if poweroff_minutes is None:
        return head + command
    m = int(poweroff_minutes)
    return (f"{head}(\n{command}\n)\nrc=$?\n"
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
    """Price and capacity of the VM in every region. Read-only: the cost calculator and capacity advice."""
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
            if not offered:
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
        return round(sum(e["usd"] for e in self.data["entries"]), 2)

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
        return f"budget: ${self.committed():.2f} spent or reserved ({lim}{left})"


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

    def argv(self, command: str) -> list[str]:
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"UserKnownHostsFile={self.known_hosts}", "-o", "ConnectTimeout=15",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]
        if self.key:
            argv += ["-i", self.key, "-o", "IdentitiesOnly=yes"]
        return argv + [self.host, command]

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
         check_vm=None, max_failures: int = 12) -> int:
    """Wait for a detached step to finish. Streams new log lines to stdout and appends them to
    ``out/<step>.log`` (so the log survives a preemption); returns the step's exit code."""
    offset, failures, pending, shown = 0, 0, "", ""
    t0 = time.time()
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


def vm_deadline(rec: Record) -> float | None:
    """When the in-VM ``--hours`` shutdown hits (creation + hours; the real one is a little later)."""
    if rec.hours and rec.times.get("create_start"):
        return _seconds(rec.times["create_start"]) + rec.hours * 3600
    return None


def remote_steps(remote: Remote, job: JobSpec, rec: Record, check_vm=None) -> int | None:
    """Everything over SSH up to the job's end. Returns the job's exit code (None: no job)."""
    wait_ssh(remote, check_vm)
    rec.times["ssh_ready"] = utc()
    log(f"ssh ready: {remote.host}; bootstrapping (RAM disk, lingering, uv)")
    cp = remote.run(bootstrap_script(), timeout=1200, check=False)
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
    start(remote, "setup", setup_script(job.extra_gpu), cwd=REMOTE_REPO)
    if (code := poll(remote, "setup", job.out, interval=10, timeout=3600, check_vm=check_vm)) != 0:
        raise InfraError(f"setup (uv sync / CUDA check) failed with {code}; see setup.log")
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
    log(f"job: {job.command}")
    fallback = None  # if the VM's own record of its shutdown can't be read: creation + --hours, slightly early
    if rec.hours and rec.times.get("create_start"):
        fallback = int(_seconds(rec.times["create_start"]) + rec.hours * 3600)
    start(remote, "job", job_script(job.command, job.env, job.poweroff_minutes, fallback), cwd=REMOTE_REPO)
    rec.times["job_start"] = utc()
    rec.save(job.out)
    cap = vm_deadline(rec)  # past the cap + 10 min the VM can only be gone or wedged: stop waiting
    code = poll(remote, "job", job.out, interval=job.poll_interval, check_vm=check_vm,
                timeout=None if cap is None else max(60.0, cap + 600 - time.time()))
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


# --------------------------------------------------------------------------------------- commands


def _require(value: str | None, flag: str, env: str) -> str:
    if not value:
        raise SystemExit(f"{flag} (or ${env}) is required")
    return value


def vm_spec(args) -> VmSpec:
    """The VM, minus where it goes (``in_region`` fills that in)."""
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,40}", args.name):
        raise SystemExit(f"--name must be lowercase letters, digits and dashes, got {args.name!r}")
    return VmSpec(name=f"{args.name}-{secrets.token_hex(3)}", project_id="", subnet_id="",
                  platform=args.platform, preset=args.preset, spot=not args.on_demand, disk_gib=args.disk_gib,
                  image_family=args.image_family, image_project="", hours=args.hours, user=args.user)


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
    return JobSpec(out=expand_out(args.out, name, stamp), command=job_command(cmd) if cmd else None,
                   pushes=pushes, hf=hf, fetches=list(args.fetch), uploads=uploads, hf_token_file=token_file,
                   env=env, extra_gpu=args.extra_gpu, poll_interval=args.poll_interval, allow_dirty=args.allow_dirty)


def dry_run(nebius: str, spec: VmSpec, job: JobSpec, pubkey: str, keep: bool) -> int:
    """Print what ``run``/``up`` would do in the chosen region. Nothing is created and nothing goes over SSH;
    the only cloud calls (before this) are the read-only price and capacity checks."""
    user_data = render_cloud_init(spec.user, pubkey, spec.hours)
    argv = create_argv(nebius, spec, "<CLOUD_INIT>")
    print("== create (dry run: nothing is created) ==")
    print(shlex.join(argv))
    print(f"== cloud-init (--cloud-init-user-data) ==\n{user_data}")
    files = repo_files()
    size = sum((REPO / f).stat().st_size for f in files if (REPO / f).is_file())
    print(f"== push ==\n{len(files)} repo files, {size / 1e6:.1f} MB uncompressed: "
          f"{REPO} -> {spec.user}@<ip>:{REMOTE_REPO}/")
    for local, dest in job.pushes:
        print(f"{local} -> {remote_path(dest)}")
    code = code_version(REPO)
    print(f"== code version (now; taken again at each push) -> {REMOTE_REPO}/{provenance.CODE_VERSION_FILE} and "
          f"job.json ==\n{json.dumps(code)}")
    if code["dirty"] and code["source"] == "git":
        print(f"uncommitted changes (--allow-dirty): their patch -> {job.out / provenance.PATCH_FILE}, the untracked "
              f"files in it -> {job.out / provenance.UNTRACKED_FILE}")
    print(f"== bootstrap (over ssh) ==\n{bootstrap_script()}")
    if job.hf:
        print(f"== downloads (detached, started right after bootstrap) ==\n{download_script(job.hf)}\n")
    print(f"== setup (detached, in ~/{REMOTE_REPO}) ==\n{setup_script(job.extra_gpu)}")
    if job.command is not None:
        script = job_script(job.command, job.env, job.poweroff_minutes)
        print(f"== job (detached, in ~/{REMOTE_REPO}) ==\n{script}\n")
        print(f"== how a detached step is launched (the job, verbatim) ==\n{detached(script, 'job', REMOTE_REPO)}\n")
    if job.uploads:
        print("== upload (detached, after the job, whatever its exit code) ==")
        for src, dest in job.uploads:
            print(f"{remote_path(src)} -> {dest}")
        print(f"token: {job.hf_token_file} -> {REMOTE_HF_TOKEN} on the VM over ssh stdin, mode 600, before the job "
              f"(so it can save as it goes); deleted when the upload step ends")
    print("== fetch ==")
    for path in job.fetches:
        print(f"{remote_path(path)} -> {fetch_dest(path, job.out) / os.path.basename(path.rstrip('/'))}")
    print(f"~/{REMOTE_LOGS}/ -> {job.out / 'remote_logs'}; job.json -> {job.out / 'job.json'}")
    end = "kept (prints the ssh command)" if keep else "deleted, then verified gone (instance and boot disk)"
    print(f"== teardown ==\n{spec.name}: {end}; in-VM auto-shutdown after {spec.hours:g} h")
    return 0


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
    gone from its region (deleted by hand; ``down`` with the ledger settles it at once)."""
    for e in [e for e in budget.data["entries"] if e.get("kind") == "reserved"]:
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
        return vm_attempts(args, cmd)
    finally:
        restore_signals(handlers)


def vm_attempts(args, cmd: list[str]) -> int:
    """``cmd_vm`` inside its signal handlers."""
    base = vm_spec(args)
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
    what = f"{spec.name} ({gpu_count(spec.preset)}× {spec.platform}, ≤{spec.hours:g} h)"
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
        pubkey = pub.read_text() if pub.is_file() else f"<missing {pub}>"
        return dry_run(args.nebius, in_region(spec, by_name[order[0].region]), job, pubkey,
                       keep=args.cmd == "up" or args.keep), None
    if budget is not None and order:
        budget.check(worst, what)
    keep_always = args.cmd == "run" and args.keep
    rec = Record(mode=args.cmd, name=spec.name, spot=spec.spot, hours=spec.hours, command=job.command,
                 flags={k: v for k, v in vars(args).items() if k not in ("func",)},
                 max_price=cap, offers=[asdict(o) for o in offers], attempt=attempt)
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
                    budget.settle(spec.name, settle_usd, "conservative estimate from the sampled spot prices")
                elif inst is None:
                    budget.settle(spec.name, 0.0, "no VM was created")
                else:  # created but no price samples: keep the worst case
                    budget.settle(spec.name, worst, "no price samples; settled at the worst case")
                log(budget.summary())
            if rec.estimated_cost_usd is not None:
                log(f"estimated cost ${rec.estimated_cost_usd:.2f} in {rec.region} "
                    f"({'so far; the VM is still running' if rec.kept else 'from the sampled spot prices'})")
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


def cmd_exec(args, cmd: list[str]) -> int:
    """The SSH part only, on an existing box (an ``up`` VM or any CUDA machine)."""
    job = job_spec(args, cmd, args.name)
    if job.command is None:
        raise SystemExit("exec needs a job command after --")
    check_code(job.allow_dirty)
    rec = Record(mode="exec", host=args.host, command=job.command,
                 flags={k: v for k, v in vars(args).items() if k not in ("func",)})
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
        print(f"  {e['when'][:16]}  {e['kind']:8}  ${e['usd']:7.2f}  {e.get('what', '')}")
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
    machine.add_argument("--platform", default="gpu-b300-sxm")
    machine.add_argument("--preset", default="1gpu-24vcpu-346gb")
    pricing = machine.add_mutually_exclusive_group()
    pricing.add_argument("--spot", action="store_true", help="preemptible at the current spot price (default)")
    pricing.add_argument("--on-demand", action="store_true", help="regular, non-preemptible VM")
    machine.add_argument("--max-price", type=float, default=None, metavar="USD",
                         help="skip regions whose price for the whole VM is above this, per hour (default: no cap)."
                              " Checked at launch; a spot price can still rise during the run")

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
                        help="exported for the job (no secrets)")
    remote.add_argument("--extra-gpu", action="store_true", help="uv sync --extra gpu (flash-attn, causal-conv1d, …)")
    remote.add_argument("--poll-interval", type=float, default=15.0, help="seconds between job log polls")
    remote.add_argument("--allow-dirty", action="store_true",
                        help="push a working tree with uncommitted changes (refused otherwise); each push then saves "
                             f"their patch in its --out as {provenance.PATCH_FILE}, with the untracked files listed in "
                             f"{provenance.UNTRACKED_FILE}")

    vm = argparse.ArgumentParser(add_help=False)
    vm.add_argument("--subnet-id", default=os.environ.get("NEBIUS_SUBNET_ID"),
                    help="with --project-id only (env NEBIUS_SUBNET_ID)")
    vm.add_argument("--hours", type=float, required=True, help="in-VM auto-shutdown after this long (spending cap)")
    vm.add_argument("--ssh-key", default=os.environ.get("NEBIUS_SSH_KEY"),
                    help="private key; KEY.pub goes into cloud-init (env NEBIUS_SSH_KEY)")
    vm.add_argument("--name", default="md-job", help="VM name prefix (a random suffix is added)")
    vm.add_argument("--user", default="md", help="login user created on the VM")
    vm.add_argument("--max-gpu-price", type=float, default=(float(os.environ["NEBIUS_MAX_GPU_PRICE"])
                                                             if os.environ.get("NEBIUS_MAX_GPU_PRICE") else None),
                    metavar="USD", help="spot bid per GPU-hour through a Nebius pricing policy: never charged more; "
                                        "the VM doesn't run while the market is above it (env NEBIUS_MAX_GPU_PRICE)")
    vm.add_argument("--budget-file", default=os.environ.get("NEBIUS_BUDGET_FILE"),
                    help="spending ledger; refuse a VM whose worst case would exceed its limit (env NEBIUS_BUDGET_FILE)")
    vm.add_argument("--disk-gib", type=int, default=200, help="boot disk (network SSD; weights go to RAM, not here)")
    vm.add_argument("--image-family", default="ubuntu24.04-cuda13.0")
    vm.add_argument("--image-project", default="project-e03public-images",
                    help="with --project-id only: the region's public-image project (default: uk-south1's)")
    vm.add_argument("--wait-hours", type=float, default=2.0,
                    help="when no region qualifies (price above the bid or cap, no capacity), re-check every 15 min "
                         "for this long before giving up")
    vm.add_argument("--dry-run", action="store_true", help="print the plan; create nothing")

    p = sub.add_parser("run", parents=[nebius, machine, vm, remote],
                       help="cheapest region → create → set up → job → fetch → delete")
    p.add_argument("--keep", action="store_true", help="do not delete the VM at the end")
    p.add_argument("--attempts", type=int, default=3,
                   help=f"VMs to use in all: after a spot preemption, or a job exit of {EXIT_UNFINISHED} (stopped "
                        "early, can resume), a fresh VM runs the job again in the then-cheapest region")
    p.set_defaults(func=cmd_vm)
    p = sub.add_parser("up", parents=[nebius, machine, vm, remote],
                       help="cheapest region → create + set up (+ downloads), keep the VM")
    p.set_defaults(func=cmd_vm, keep=True, attempts=1)
    p = sub.add_parser("exec", parents=[remote], help="set up + job + fetch on an existing SSH box")
    p.add_argument("--host", required=True, help="USER@HOST")
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
