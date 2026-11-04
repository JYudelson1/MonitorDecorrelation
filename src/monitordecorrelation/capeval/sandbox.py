"""Docker sandbox for the agentic coding benchmarks: one container per episode, a fresh one per grading.

Model-generated commands run via ``docker exec`` in a long-lived per-episode container with **no
network**, as an unprivileged user, in ``/work``. Grading never reuses the agent's container: the graded
files are copied into a fresh container together with the hidden tests (which the agent never had access
to). Every container carries ``capeval.run=<run id>`` and ``capeval.owner=<host>:<pid>:<start time>``.

Why many parallel sandboxes cannot take the box down
----------------------------------------------------
Hundreds of containers run at once (most of the time they only wait for the model), so every resource
a container can consume is bounded per container AND in total:

- **Memory / CPU / processes, per container** (kernel-enforced cgroups): ``--memory`` = ``--memory-swap``
  = ``MEM_LIMIT_MB`` (no swap), ``--cpus 1``, ``--pids-limit``. Zombies are reaped (``--init``).
- **Disk: none.** The root filesystem is read-only, every writable path (``/work``, ``/tmp``,
  ``/dev/shm``) is a size-capped tmpfs, and logging is off (``--log-driver none``; the default json-file
  driver would write anything a grading run prints to the host disk). A runaway ``dd`` hits "No space
  left on device" inside its own container. (tmpfs is RAM: it is in the memory budget below.)
- **No work outside a command's slot.** Each agent command runs as ``timeout … bash -c <cmd>`` followed
  by ``kill -9 -1`` as the agent user, so a ``cmd &`` it left behind dies with it (the instructions tell
  the model background processes do not persist). The container's own keep-alive (``sleep``) runs as
  root and is untouched. So CPU and memory are only ever used by commands holding an *exec permit*.
- **Box-wide caps, across processes.** Two cross-process semaphores (``flock`` slot files under the
  repo's ``globalsem.sem_dir()``, crash-safe: the kernel drops a dead process's locks) cap, over every
  capeval run on this machine: live containers (``capeval_containers``) and concurrent commands +
  gradings (``capeval_exec``). Their sizes come from the machine (``box_limits``): exec permits =
  cores − 2 (each command is held to one core); containers =
  (RAM − ``HOST_RESERVE_MB`` − exec permits × ``MEM_LIMIT_MB``) / ``IDLE_FOOTPRINT_MB``, i.e. even if every
  active command used its full memory limit and every idle container filled its tmpfs, the total fits in
  RAM. A run's own ``--max-containers`` / ``--max-exec`` are clamped to these, and at startup further to
  what is actually free now (``plan_limits``), with the reason logged.
- **Docker itself.** At most ``LIFECYCLE_CONCURRENCY`` ``docker run`` / ``rm`` / file-transfer calls at a
  time, so the daemon is never flooded; containers run ``--network none`` (no veth, no IP exhaustion).
- **No leaks.** Episode containers are ``--rm`` and die on their own after ``CONTAINER_LIFETIME_S``;
  ``cleanup()`` removes a run's containers on exit / Ctrl-C; and ``reap_orphans()`` (at startup)
  removes containers whose owner process is gone (a SIGKILLed run), which would otherwise hold memory
  for hours.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import io
import os
import random
import shutil
import socket
import subprocess
import tarfile
import tempfile
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from monitordecorrelation.globalsem import sem_dir

# The sandbox image: Python 3.11 + pytest + the libraries CanItEdit's problems import (numpy, pandas,
# scipy, scikit-learn, z3, autograd, flask, torch-cpu) + sortedcontainers (LeetCode's environment has it).
DOCKERFILE = """\
FROM python:3.11-slim
RUN pip install --no-cache-dir "pytest==8.4.2" "numpy<2.4" "pandas<3" "scipy<1.17" "scikit-learn<1.8" \\
        "sympy<1.15" z3-solver autograd flask sortedcontainers \\
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN useradd -m -u 1000 agent && mkdir -p /work /grade && chmod 777 /work /grade
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_INDEX=1
"""
IMAGE_TAG = "capeval-sandbox:" + hashlib.sha256(DOCKERFILE.encode()).hexdigest()[:12]
CONTAINER_LIFETIME_S = 6 * 3600
MAX_CAPTURE_BYTES = 256_000  # bytes of command output kept by the harness (the model sees less)
MAX_FILE_BYTES = 2_000_000   # a graded file larger than this is treated as missing
SANDBOX_UID = 1000

# ---- per-container limits (kernel-enforced)
MEM_LIMIT_MB = 2048     # RAM (= RAM + swap) of one container — commands, its tmpfs files and all
PIDS_LIMIT = 512
WORK_TMPFS_MB = 64      # /work (the agent's files; graders' /grade)
TMP_TMPFS_MB = 32       # /tmp (also $HOME)
SHM_MB = 16             # /dev/shm
# ---- box-wide budget
IDLE_OVERHEAD_MB = 32   # host-side cost of an idle container (containerd shim, init, sleep), generous
IDLE_FOOTPRINT_MB = WORK_TMPFS_MB + TMP_TMPFS_MB + SHM_MB + IDLE_OVERHEAD_MB  # worst case per idle container
HOST_RESERVE_MB = 8192  # the harness itself, dockerd, the OS
LIFECYCLE_CONCURRENCY = 8

# After each agent command: kill everything the agent user still runs (its background jobs). ``kill -1``
# skips the calling shell; the root-owned init + keep-alive ``sleep`` cannot be signalled by the agent.
_EXEC_WRAPPER = 'timeout -k 5 "$1" bash -c "$2"; rc=$?; kill -9 -1 2>/dev/null; exit $rc'
_READ_SCRIPT = ('f="/work/$1"; [ -f "$f" ] || exit 3; [ "$(wc -c < "$f")" -le "$2" ] || exit 4; '
                'printf "CAPEVAL_FILE\\n"; cat "$f"')
_GRADE_WRAPPER = 'cp -r /src/. /grade/ && cd /grade && exec "$@"'


@dataclass
class ExecResult:
    output: str
    returncode: int | None
    timed_out: bool
    n_bytes: int


# ---------------------------------------------------------------------------------------------
# Machine budget
# ---------------------------------------------------------------------------------------------
def _meminfo_mb(key: str) -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) // 1024
    raise RuntimeError(f"/proc/meminfo has no {key}")


def _cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 2


def box_limits() -> dict:
    """Machine-wide caps shared by every capeval run on this box (constant for a given machine, so all
    runs size the shared semaphores identically)."""
    total = _meminfo_mb("MemTotal")
    exec_n = max(2, _cpu_count() - 2)
    containers = (total - HOST_RESERVE_MB - exec_n * MEM_LIMIT_MB) // IDLE_FOOTPRINT_MB
    return {"mem_total_mb": total, "exec": exec_n, "containers": max(0, containers)}


def plan_limits(requested_containers: int, requested_exec: int, *, other_live_containers: int = 0
                ) -> tuple[int, int, list[str]]:
    """(containers, exec, notes): the requested per-run caps clamped to the box-wide caps and to the
    memory free right now (a worst-case budget: every exec permit at its full memory limit, every
    container — this run's and other runs' — with full tmpfs)."""
    box = box_limits()
    notes = []
    exec_n = max(1, min(requested_exec, box["exec"]))
    if exec_n < requested_exec:
        notes.append(f"--max-exec {requested_exec} -> {exec_n} (box-wide cap: cores − 2)")
    if box["containers"] < 1:
        raise RuntimeError(f"not enough RAM for even one sandbox ({box['mem_total_mb']} MB total; need "
                           f"{HOST_RESERVE_MB} reserve + {box['exec']}×{MEM_LIMIT_MB} exec + {IDLE_FOOTPRINT_MB})")
    containers = min(requested_containers, box["containers"])
    if containers < requested_containers:
        notes.append(f"--max-containers {requested_containers} -> {containers} (box-wide memory cap: "
                     f"{box['mem_total_mb']} MB RAM)")
    avail = _meminfo_mb("MemAvailable")
    room = (avail - HOST_RESERVE_MB - exec_n * MEM_LIMIT_MB
            - other_live_containers * IDLE_FOOTPRINT_MB) // IDLE_FOOTPRINT_MB
    if room < containers:
        new = max(1, room)
        notes.append(f"--max-containers {containers} -> {new} (only {avail} MB of RAM free now"
                     + (f", {other_live_containers} containers of other capeval runs live" if other_live_containers
                        else "") + ")")
        containers = new
    return containers, exec_n, notes


class BoxSemaphore:
    """A semaphore shared by every process on the machine: ``n`` ``flock``-ed slot files under the repo's
    ``globalsem.sem_dir()/<name>`` (same mechanism as ``globalsem.slot``, so a dead holder's permits are
    freed by the kernel), acquired by polling without ever blocking the event loop."""

    def __init__(self, name: str, n: int) -> None:
        self.n = n
        self.dir = sem_dir() / name
        self.dir.mkdir(parents=True, exist_ok=True)

    async def acquire(self) -> int:
        delay = 0.02
        while True:
            for i in random.sample(range(self.n), self.n):
                fd = os.open(self.dir / str(i), os.O_CREAT | os.O_RDWR, 0o666)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    os.close(fd)
                    continue
                return fd
            await asyncio.sleep(delay * (1.0 + random.random()))
            delay = min(delay * 1.5, 1.0)

    @staticmethod
    def release(fd: int) -> None:
        os.close(fd)  # drops the flock

    @asynccontextmanager
    async def slot(self):
        fd = await self.acquire()
        try:
            yield
        finally:
            self.release(fd)


# ---------------------------------------------------------------------------------------------
# Ownership (for reaping containers of dead runs)
# ---------------------------------------------------------------------------------------------
def _proc_start(pid: int) -> str | None:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[19]  # field 22: start time since boot
    except (OSError, IndexError):
        return None


OWNER = f"{socket.gethostname()}:{os.getpid()}:{_proc_start(os.getpid())}"


def _owner_alive(owner: str) -> bool:
    try:
        host, pid, start = owner.rsplit(":", 2)
    except ValueError:
        return False
    if host != socket.gethostname():
        return True  # a run on another machine sharing this daemon: not ours to judge
    return pid.isdigit() and _proc_start(int(pid)) == start


def list_capeval_containers() -> list[tuple[str, str]]:
    """(container id, owner label) of every capeval container on the daemon, running or not."""
    out = subprocess.run(["docker", "ps", "-a", "--filter", "label=capeval.run", "--format",
                          '{{.ID}}\t{{.Label "capeval.owner"}}'], capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"docker ps failed: {out.stderr[-1000:]}")
    return [tuple(line.split("\t", 1)) for line in out.stdout.splitlines() if "\t" in line]


def reap_orphans() -> tuple[int, int]:
    """Remove capeval containers whose owner process is gone (or that predate owner labels).
    -> (removed, live containers of OTHER running capeval processes)."""
    dead, others = [], 0
    for cid, owner in list_capeval_containers():
        if not owner or not _owner_alive(owner):
            dead.append(cid)
        elif owner != OWNER:
            others += 1
    for i in range(0, len(dead), 100):
        subprocess.run(["docker", "rm", "-f", *dead[i:i + 100]], capture_output=True, timeout=600)
    return len(dead), others


# ---------------------------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------------------------
async def _run_capture(argv: list[str], timeout: float, cap: int = MAX_CAPTURE_BYTES, on_timeout=None,
                       stdin_bytes: bytes | None = None) -> ExecResult:
    """Run ``argv`` on the host with stdout+stderr merged, keeping at most ``cap`` bytes (the rest is
    drained and counted). ``timed_out`` here means the HOST-side timeout fired (``on_timeout`` is then
    awaited, e.g. to remove a container)."""
    proc = await asyncio.create_subprocess_exec(
        *argv, stdin=asyncio.subprocess.PIPE if stdin_bytes is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    buf = bytearray()
    total = 0

    async def feed() -> None:
        assert proc.stdin is not None
        try:
            proc.stdin.write(stdin_bytes)
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            proc.stdin.close()

    async def drain() -> None:
        nonlocal total
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(65536)
            if not chunk:
                break
            total += len(chunk)
            if len(buf) < cap:
                buf.extend(chunk[: cap - len(buf)])
        await proc.wait()

    timed_out = False
    try:
        if stdin_bytes is not None:
            await asyncio.wait_for(asyncio.gather(feed(), drain()), timeout)
        else:
            await asyncio.wait_for(drain(), timeout)
    except asyncio.TimeoutError:
        timed_out = True
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        if on_timeout is not None:
            await on_timeout()
    return ExecResult(buf.decode("utf-8", errors="replace"), proc.returncode, timed_out, total)


def _tar(files: dict[str, str | bytes]) -> bytes:
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w") as tf:
        for rel, content in files.items():
            data = content.encode() if isinstance(content, str) else content
            ti = tarfile.TarInfo(rel)
            ti.size = len(data)
            ti.mode = 0o644
            tf.addfile(ti, io.BytesIO(data))
    return bio.getvalue()


# ---------------------------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------------------------
class Container:
    """A running per-episode container (its ``/work`` is a tmpfs inside it)."""

    def __init__(self, mgr: "SandboxManager", name: str) -> None:
        self.mgr = mgr
        self.name = name

    async def put_files(self, files: dict[str, str | bytes]) -> None:
        if not files:
            return
        async with self.mgr.lifecycle_sem:
            res = await _run_capture(["docker", "exec", "-i", "--user", self.mgr.agent_user, self.name,
                                      "tar", "-x", "-C", "/work"], 120, stdin_bytes=_tar(files))
        if res.returncode != 0:
            raise RuntimeError(f"copying files into the sandbox failed ({res.returncode}): {res.output[-1000:]}")

    async def exec(self, command: str, timeout: float) -> ExecResult:
        """Run ``command`` with ``bash -c`` in ``/work``; a command still running after ``timeout`` s is
        killed (``timed_out``; returncode 124/137); anything it left running is killed afterwards."""
        argv = ["docker", "exec", "--user", self.mgr.agent_user, "-w", "/work", self.name,
                "sh", "-c", _EXEC_WRAPPER, "sh", f"{timeout:g}", command]
        async with self.mgr.exec_slot():
            res = await _run_capture(argv, timeout + 60)
        if res.returncode in (124, 137) and not res.timed_out:
            res.timed_out = True
        return res

    async def read(self, rel: str) -> str | None:
        """Contents of ``/work/<rel>``, or None if it is missing, not a regular file, or too large."""
        argv = ["docker", "exec", "--user", self.mgr.agent_user, self.name,
                "sh", "-c", _READ_SCRIPT, "sh", rel, str(MAX_FILE_BYTES)]
        async with self.mgr.lifecycle_sem:
            res = await _run_capture(argv, 120, cap=MAX_FILE_BYTES + 64)
        if res.returncode != 0 or not res.output.startswith("CAPEVAL_FILE\n"):
            return None
        return res.output[len("CAPEVAL_FILE\n"):]


class SandboxManager:
    def __init__(self, *, n_exec: int, max_containers: int, root_dir: str | None = None) -> None:
        self.image = IMAGE_TAG
        self.run_id = uuid.uuid4().hex[:10]
        self.label = f"capeval.run={self.run_id}"
        self.requested = (max_containers, n_exec)
        self.root_dir = root_dir or tempfile.mkdtemp(prefix="capeval_sandbox_")
        os.makedirs(self.root_dir, exist_ok=True)
        if os.getuid() == 0:
            self.agent_user, self.uid, self.gid = f"{SANDBOX_UID}:{SANDBOX_UID}", SANDBOX_UID, SANDBOX_UID
        else:
            self.agent_user, self.uid, self.gid = f"{os.getuid()}:{os.getgid()}", os.getuid(), os.getgid()
        self.limits: dict = {}
        self.notes: list[str] = []
        self._ready = False

    # ------------------------------------------------------------------ setup
    def prepare(self) -> None:
        """Build the image if needed, reap dead runs' containers, and size every cap (sync; call once
        before any container is used). ``self.notes`` says what was clamped and why."""
        self.ensure_image()
        removed, others = reap_orphans()
        if removed:
            self.notes.append(f"removed {removed} leftover sandbox containers of dead capeval runs")
        containers, exec_n, notes = plan_limits(*self.requested, other_live_containers=others)
        self.notes += notes
        box = box_limits()
        self.limits = {"max_containers": containers, "max_exec": exec_n, "box": box,
                       "per_container": {"memory_mb": MEM_LIMIT_MB, "cpus": 1, "pids": PIDS_LIMIT,
                                         "work_tmpfs_mb": WORK_TMPFS_MB, "tmp_tmpfs_mb": TMP_TMPFS_MB,
                                         "shm_mb": SHM_MB, "disk": "none (read-only rootfs, tmpfs only)"}}
        self.container_sem = asyncio.Semaphore(containers)
        self.exec_sem = asyncio.Semaphore(exec_n)
        self.lifecycle_sem = asyncio.Semaphore(LIFECYCLE_CONCURRENCY)
        self.box_containers = BoxSemaphore("capeval_containers", box["containers"])
        self.box_exec = BoxSemaphore("capeval_exec", box["exec"])
        self._ready = True

    def ensure_image(self) -> None:
        """Build the sandbox image if it is not there yet (tag = hash of DOCKERFILE)."""
        have = subprocess.run(["docker", "image", "inspect", self.image], capture_output=True)
        if have.returncode == 0:
            return
        print(f"[capeval] building sandbox image {self.image} (one-time, a few minutes)…", flush=True)
        proc = subprocess.run(["docker", "build", "-t", self.image, "-"], input=DOCKERFILE.encode(),
                              capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError("docker build failed:\n" + proc.stdout.decode()[-3000:]
                               + proc.stderr.decode()[-3000:])

    @asynccontextmanager
    async def exec_slot(self):
        """One command / grading at a time per permit: this run's cap, then the box-wide one."""
        async with self.exec_sem, self.box_exec.slot():
            yield

    # ------------------------------------------------------------------ docker args
    def _run_args(self, name: str, user: str) -> list[str]:
        return ["docker", "run", "--name", name, "--label", self.label, "--label", f"capeval.owner={OWNER}",
                "--init", "--network", "none", "--log-driver", "none", "--read-only",
                "--tmpfs", f"/tmp:rw,size={TMP_TMPFS_MB}m,mode=1777", "--shm-size", f"{SHM_MB}m",
                "--cpus", "1", "--memory", f"{MEM_LIMIT_MB}m", "--memory-swap", f"{MEM_LIMIT_MB}m",
                "--pids-limit", str(PIDS_LIMIT), "--user", user,
                "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONUNBUFFERED=1"]

    async def _rm(self, name: str) -> None:
        async with self.lifecycle_sem:
            await _run_capture(["docker", "rm", "-f", name], 120)

    # ------------------------------------------------------------------ episode containers
    @asynccontextmanager
    async def container(self, files: dict[str, str | bytes]):
        """A running container whose ``/work`` holds ``files``; removed on exit."""
        assert self._ready, "SandboxManager.prepare() first"
        async with self.container_sem:
            fd = await self.box_containers.acquire()
            name = f"capeval-{self.run_id}-{uuid.uuid4().hex[:10]}"
            # The keep-alive runs as root so the agent's post-command `kill -9 -1` cannot stop it.
            argv = self._run_args(name, "0:0") + [
                "-d", "--rm", "--tmpfs", f"/work:rw,size={WORK_TMPFS_MB}m,mode=1777", "-w", "/work",
                self.image, "sleep", str(CONTAINER_LIFETIME_S)]
            try:
                async with self.lifecycle_sem:
                    res = await _run_capture(argv, 300)
                if res.returncode != 0:
                    raise RuntimeError(f"docker run failed ({res.returncode}): {res.output[-2000:]}")
                box = Container(self, name)
                await box.put_files(files)
                yield box
            finally:
                await self._rm(name)
                self.box_containers.release(fd)

    # ------------------------------------------------------------------ one-shot (grading)
    def _make_dir(self, files: dict[str, str | bytes]) -> str:
        d = tempfile.mkdtemp(dir=self.root_dir)
        for rel, content in files.items():
            path = os.path.join(d, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(content.encode() if isinstance(content, str) else content)
        if os.getuid() == 0:
            for dirpath, dirnames, filenames in os.walk(d):
                os.chown(dirpath, self.uid, self.gid)
                for name in filenames:
                    os.chown(os.path.join(dirpath, name), self.uid, self.gid)
        return d

    async def run_once(self, files: dict[str, str | bytes], argv: list[str], *, timeout: float,
                       ro_mounts: dict[str, str] | None = None, as_root: bool = False) -> ExecResult:
        """Run ``argv`` in a FRESH container, in ``/grade`` (a tmpfs) holding a copy of ``files`` (+
        read-only ``ro_mounts`` host->container), killed after ``timeout`` s; the container is removed.
        ``as_root`` runs it as the container's root (a grader that must read files the code it runs
        cannot, demoting that code itself); otherwise as the unprivileged sandbox user."""
        assert self._ready, "SandboxManager.prepare() first"
        gdir = self._make_dir(files)
        name = f"capeval-{self.run_id}-g{uuid.uuid4().hex[:10]}"
        mounts: list[str] = ["-v", f"{gdir}:/src:ro"]
        for host, cont in (ro_mounts or {}).items():
            mounts += ["-v", f"{host}:{cont}:ro"]
        full = self._run_args(name, "0:0" if as_root else self.agent_user) + ["--rm"] + mounts + [
            "--tmpfs", f"/grade:rw,size={WORK_TMPFS_MB}m,mode=1777", "-w", "/grade", self.image,
            "timeout", "-k", "5", f"{timeout:g}", "sh", "-c", _GRADE_WRAPPER, "sh"] + argv
        try:
            async with self.exec_slot():
                res = await _run_capture(full, timeout + 120, on_timeout=lambda: self._rm(name))
        finally:
            shutil.rmtree(gdir, ignore_errors=True)
        if res.returncode in (124, 137):
            res.timed_out = True
        return res

    # ------------------------------------------------------------------ cleanup
    def cleanup(self) -> None:
        """Remove every container of this run (sync; safe to call at exit)."""
        try:
            ids = subprocess.run(["docker", "ps", "-aq", "--filter", f"label={self.label}"],
                                 capture_output=True, text=True, timeout=60).stdout.split()
            for i in range(0, len(ids), 100):
                subprocess.run(["docker", "rm", "-f", *ids[i:i + 100]], capture_output=True, timeout=300)
        except Exception:  # noqa: BLE001 — best effort at shutdown
            pass
        shutil.rmtree(self.root_dir, ignore_errors=True)
