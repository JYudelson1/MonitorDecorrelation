"""scripts/nebius_job.py: the pure parts of the GPU-job launcher, plus its lifecycle with the cloud faked.

Offline. Nothing here talks to Nebius or opens SSH. The lifecycle tests replace the ``Nebius`` client
and the SSH steps with fakes, and check that the VM is deleted whatever happens to the job.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
import types
from datetime import UTC, datetime
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "nebius_job", Path(__file__).resolve().parents[1] / "scripts" / "nebius_job.py")
nj = importlib.util.module_from_spec(_spec)
sys.modules["nebius_job"] = nj  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(nj)

PUBKEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestKeyOnly hub: test key"
CLEAN = {"commit": "c0de1" * 8, "dirty": False, "diff_sha256": None, "source": "git"}
REAL_CODE_VERSION = nj.code_version


@pytest.fixture(autouse=True)
def committed_code(monkeypatch):
    """Every test pushes a clean, committed tree unless it says otherwise: the real checkout may be anything."""
    monkeypatch.setattr(nj, "code_version", lambda repo: dict(CLEAN))

# Trimmed from the real `nebius compute instance create --format json` output of 2026-10-09.
CREATED = {
    "metadata": {"id": "computeinstance-e03mk85f6b0jqnhxp2", "parent_id": "project-x",
                 "name": "probe-nvfp4-test-cf3b11", "created_at": "2026-10-09T18:41:21.452855Z"},
    "spec": {"resources": {"platform": "gpu-b300-sxm", "preset": "1gpu-24vcpu-346gb"},
             "recovery_policy": "FAIL", "preemptible": {"on_preemption": "STOP"}, "follows_spot_price": {}},
    "status": {"state": "RUNNING", "network_interfaces": [{
        "name": "eth0", "ip_address": {"address": "10.64.0.38/32"},
        "public_ip_address": {"address": "89.124.37.250/32"}}]},
}


def _spec_vm(**kw) -> nj.VmSpec:
    return nj.VmSpec(name="md-job-abc123", project_id="project-x", subnet_id="vpcsubnet-y", **kw)


# ------------------------------------------------------------------------------------- cloud-init


def test_cloud_init_has_key_ram_disk_linger_and_cap():
    text = nj.render_cloud_init("md", PUBKEY, hours=2)
    assert text.startswith("#cloud-config\n")
    assert json.dumps(PUBKEY) in text  # quoted: the comment's ": " would otherwise break the YAML
    assert "mount -t tmpfs -o size=90%,mode=1777 tmpfs /mnt/ram" in text
    assert "loginctl enable-linger md" in text
    assert "shutdown -h +120" in text
    assert "/dev/shm" not in text  # logind's RemoveIPC wipes it when the last session ends
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(text)
    assert doc["users"][0]["name"] == "md"
    assert doc["users"][0]["ssh_authorized_keys"] == [PUBKEY]
    assert doc["runcmd"][-1] == "shutdown -h +120"


def test_cloud_init_rounds_hours_and_rejects_bad_users():
    assert "shutdown -h +90" in nj.render_cloud_init("md", PUBKEY, hours=1.5)
    with pytest.raises(ValueError):
        nj.render_cloud_init("md; rm -rf /", PUBKEY, hours=1)


# ------------------------------------------------------------------------------------- create argv


def _flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_create_argv_spot():
    argv = nj.create_argv("nebius", _spec_vm(), "#cloud-config\n")
    assert argv[:4] == ["nebius", "compute", "instance", "create"]
    assert "--follows-spot-price" in argv and "--on-demand" not in argv
    assert _flag(argv, "--preemptible-on-preemption") == "stop"
    assert _flag(argv, "--recovery-policy") == "FAIL"  # else the in-VM shutdown cap restarts the VM
    assert _flag(argv, "--boot-disk-managed-disk-name") == "md-job-abc123-boot"  # required by the API
    assert _flag(argv, "--boot-disk-managed-disk-size-gibibytes") == "200"
    assert _flag(argv, "--cloud-init-user-data") == "#cloud-config\n"
    assert _flag(argv, "--format") == "json"
    nic = json.loads(_flag(argv, "--network-interfaces"))
    assert nic == [{"name": "eth0", "ip_address": {}, "public_ip_address": {}, "subnet_id": "vpcsubnet-y"}]


def test_create_argv_on_demand():
    argv = nj.create_argv("nebius", _spec_vm(spot=False, disk_gib=300), "x")
    assert "--on-demand" in argv
    assert "--follows-spot-price" not in argv and "--preemptible-on-preemption" not in argv
    assert _flag(argv, "--recovery-policy") == "FAIL"
    assert _flag(argv, "--boot-disk-managed-disk-size-gibibytes") == "300"


# ------------------------------------------------------------------------------------- parsing


def test_parse_instance_running_and_starting():
    inst = nj.parse_instance(CREATED)
    assert (inst.id, inst.name, inst.state, inst.ip) == (
        "computeinstance-e03mk85f6b0jqnhxp2", "probe-nvfp4-test-cf3b11", "RUNNING", "89.124.37.250")
    assert inst.platform == "gpu-b300-sxm" and inst.preemptible
    starting = {"metadata": {"id": "i", "name": "n"}, "spec": {},
                "status": {"state": "STARTING", "network_interfaces": [{"public_ip_address": {}}]}}
    inst = nj.parse_instance(starting)
    assert inst.state == "STARTING" and inst.ip is None and not inst.preemptible


def test_parse_items_and_disks():
    assert nj.parse_items("") == [] and nj.parse_items("{}") == []
    disks = [nj.parse_disk(d) for d in nj.parse_items(json.dumps({"items": [
        {"metadata": {"id": "computedisk-1", "name": "md-job-abc123-boot"}, "spec": {"size_gibibytes": "200"},
         "status": {"state": "READY"}}]}))]
    assert disks == [nj.Disk("computedisk-1", "md-job-abc123-boot", "READY", "200")]


def test_parse_hf_push_and_paths(tmp_path):
    assert nj.parse_hf("thinkingmachines/Inkling-Small-NVFP4") == (
        "thinkingmachines/Inkling-Small-NVFP4", None, "Inkling-Small-NVFP4")
    assert nj.parse_hf("org/m@b6a9953=nvfp4") == ("org/m", "b6a9953", "nvfp4")
    with pytest.raises(ValueError):
        nj.parse_hf("not-a-repo")
    assert nj.parse_push("../x/eps.jsonl:data/eps.jsonl") == ("../x/eps.jsonl", "data/eps.jsonl")
    with pytest.raises(ValueError):
        nj.parse_push("no-destination")
    assert nj.remote_path("results") == "md/results"
    assert nj.remote_path("~/nebius_job") == "nebius_job"
    assert nj.remote_path("/mnt/ram/x") == "/mnt/ram/x"
    assert nj.fetch_dest("data/out/feats", tmp_path) == tmp_path / "data/out"
    assert nj.fetch_dest("/mnt/ram/x", tmp_path) == tmp_path
    assert nj.expand_out(str(tmp_path / "{stamp}-{name}"), "md-job-1").name.endswith("-md-job-1")


def test_poll_output_parsing_and_progress_bars():
    assert nj.parse_poll(f"\n{nj.SEP}\nhello\n".encode()) == (None, b"hello\n")
    assert nj.parse_poll(f"7\n{nj.SEP}\n".encode()) == (7, b"")
    assert nj.parse_poll(f"0\n{nj.SEP}\nline {nj.SEP}\n".encode()) == (0, f"line {nj.SEP}\n".encode())
    cmd = nj.poll_command("job", 10)
    assert cmd.index("job.exit") < cmd.index("job.log")  # exit first: then the log read is complete
    assert "tail -c +11" in cmd
    assert nj.display_lines("a 10%\ra 50%\ra 100%\nnext\r\nlast") == ["a 100%", "next", "last"]


# ------------------------------------------------------------------------------------- what travels


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_repo_files_ship_only_tracked_and_unignored(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / ".gitignore").write_text("data/\n.venv/\n")
    for rel in ("src/a.py", "notes/new.md", ".env", ".env.local", "data/big.bin", ".venv/lib/x.py",
                "src/__pycache__/a.cpython-311.pyc", "tracked_but_deleted.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    git("add", "src/a.py", ".gitignore", "tracked_but_deleted.py")
    git("add", "-f", ".env")  # even a tracked .env must not travel
    (tmp_path / "tracked_but_deleted.py").unlink()
    files = nj.repo_files(tmp_path)
    assert files == [".gitignore", "notes/new.md", "src/a.py"]


# ------------------------------------------------------------------------------------- detached steps


@pytest.mark.skipif(shutil.which("setsid") is None, reason="needs setsid")
def test_detached_step_returns_at_once_and_records_the_exit_code(tmp_path):
    """The launch line must hand the SSH channel back immediately. The naive ``cd x && nohup … &``
    keeps it open until the step ends (that hung our ssh twice); the control below shows this test
    catches that."""
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    (tmp_path / "md").mkdir()

    def launch(line: str) -> float:
        t0 = time.time()
        subprocess.run(["bash", "-c", line], cwd=tmp_path, env=env, capture_output=True, timeout=30, check=False)
        return time.time() - t0

    line = nj.detached("sleep 2; pwd; echo done; exit 7", "job", cwd="md")
    assert "< /dev/null" in line and "setsid nohup bash -lc" in line and line.rstrip().endswith("&")
    assert launch(line) < 1.0
    exit_file = tmp_path / nj.REMOTE_LOGS / "job.exit"
    deadline = time.time() + 20
    while not exit_file.exists() and time.time() < deadline:
        time.sleep(0.1)
    assert exit_file.read_text().strip() == "7"  # `exit` inside the script still records its code
    log = (tmp_path / nj.REMOTE_LOGS / "job.log").read_text()
    assert log.splitlines() == [str(tmp_path / "md"), "done"]

    naive = f"cd {tmp_path} && nohup sleep 2 > {tmp_path}/naive.log 2>&1 < /dev/null & echo started"
    assert launch(naive) > 1.5

    # a missing working directory fails the step instead of running the script somewhere else
    exit_file.unlink()
    launch(nj.detached("echo ran-anyway", "job", cwd="no-such-dir"))
    while not exit_file.exists() and time.time() < deadline + 20:
        time.sleep(0.1)
    assert exit_file.read_text().strip() == "1"
    assert "ran-anyway" not in (tmp_path / nj.REMOTE_LOGS / "job.log").read_text()


# ------------------------------------------------------------------------------------- dry run


def _keyfile(tmp_path: Path) -> Path:
    key = tmp_path / "id_test"
    key.write_text("PRIVATE (unused)")
    key.with_name("id_test.pub").write_text(PUBKEY + "\n")
    return key


def test_dry_run_makes_no_subprocess_calls(tmp_path, monkeypatch, capsys, fake_cloud):
    """Beyond the (faked) read-only price and capacity checks, a dry run runs nothing."""
    def forbidden(*a, **k):
        raise AssertionError(f"dry run must not run anything: {a[:1]}")

    for name in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, forbidden)
    monkeypatch.setattr(nj, "repo_files", lambda repo=nj.REPO: ["pyproject.toml"])
    out = tmp_path / "out" / "{stamp}-{name}"
    code = nj.main(["run", "--dry-run", "--hours", "2", "--project-id", "project-x", "--subnet-id", "vpcsubnet-y",
                    "--ssh-key", str(_keyfile(tmp_path)), "--out", str(out), "--hf", "org/model",
                    "--push", f"{tmp_path}:data/in", "--fetch", "results", "--env", "A=b c",
                    "--", "uv", "run", "python", "-c", "print(1)"])
    text = capsys.readouterr().out
    assert code == 0
    assert "compute instance create" in text and "--follows-spot-price" in text
    assert "#cloud-config" in text and PUBKEY in text and "shutdown -h +120" in text
    assert "1 repo files" in text
    assert "hf download org/model --local-dir /mnt/ram/model" in text
    assert "uv sync --locked" in text
    assert "print(1)" in text and "export A='b c'" in text
    assert "md/results ->" in text
    assert "default" in text and "1.00" in text  # the price table (single-project mode)
    assert not (tmp_path / "out").exists()  # nothing written either
    assert not any(c.alive or c.deleted for c in FakeNebius.clouds.values())  # nothing created


# ------------------------------------------------------------------------------------- lifecycle


class FakeNebius:
    """Records calls; the VM is RUNNING at once and vanishes when deleted (boot disk included). State is per
    project and shared by every client of that project, like the real cloud; prices come from ``PRICES``."""
    last: FakeNebius
    clouds: dict[str, FakeNebius.Cloud] = {}
    PRICES = {"project-x": 1.0}
    CAPACITY: list[dict] = []
    FAIL_CREATE: set[str] = set()  # projects whose create fails (e.g. no capacity after all)

    class Cloud:
        def __init__(self):
            self.alive: dict[str, nj.Instance] = {}
            self.deleted: list[str] = []
            self.created = 0
            self.policies: list[dict] = []
            self.create_argv: list[str] = []

    def __init__(self, binary, project_id):
        self.project_id = project_id
        self.cloud = FakeNebius.clouds.setdefault(project_id, FakeNebius.Cloud())
        FakeNebius.last = self

    @property
    def alive(self):
        return self.cloud.alive

    @property
    def deleted(self):
        return self.cloud.deleted

    def create(self, argv):
        self.cloud.created += 1
        self.cloud.create_argv = list(argv)
        if self.project_id in FakeNebius.FAIL_CREATE:
            raise nj.InfraError("nebius compute instance failed (1): not enough resources")
        inst = nj.Instance("computeinstance-1", _flag(argv, "--name"), "RUNNING", "1.2.3.4")
        self.alive[inst.id] = inst
        return inst

    def get(self, iid):
        if iid not in self.alive:
            raise nj.InfraError(f"nebius compute instance get failed (1): {iid} not found")
        return self.alive[iid]

    def instances(self):
        return list(self.alive.values())

    def disks(self):
        return [nj.Disk("computedisk-1", f"{i.name}-boot", "READY") for i in self.alive.values()]

    def delete_instance(self, iid):
        self.deleted.append(iid)
        self.alive.pop(iid, None)

    def delete_disk(self, did):
        raise AssertionError("the managed boot disk goes with the instance")

    def estimate_hourly(self, platform, preset, spot):
        if self.project_id not in FakeNebius.PRICES:
            raise nj.InfraError("nebius billing v1alpha1 failed (1): no price")
        return FakeNebius.PRICES[self.project_id] * (1 if spot else 10)

    def capacity(self, tenant):
        return FakeNebius.CAPACITY

    def pricing_policies(self):
        return list(self.cloud.policies)

    def create_pricing_policy(self, name, platform, max_gpu_price):
        pid = f"pricingpolicy-{len(self.cloud.policies) + 1}"
        self.cloud.policies.append({"metadata": {"id": pid, "name": name},
                                    "spec": {"compute_instance_spec": {"v1": {"platform": platform}},
                                             "pricing": {"max_price_v1": {"max_price": str(max_gpu_price)}}},
                                    "status": {"state": "STATE_ACTIVE"}})
        return pid


def _run_args(tmp_path: Path, *extra: str) -> list[str]:
    return ["--hours", "2", "--project-id", "project-x", "--subnet-id", "vpcsubnet-y",
            "--ssh-key", str(_keyfile(tmp_path)), "--out", str(tmp_path / "out"), *extra]


@pytest.fixture
def fake_cloud(monkeypatch):
    monkeypatch.setattr(FakeNebius, "clouds", {})
    monkeypatch.setattr(FakeNebius, "PRICES", {"project-x": 1.0})
    monkeypatch.setattr(FakeNebius, "CAPACITY", [])
    monkeypatch.setattr(FakeNebius, "FAIL_CREATE", set())
    monkeypatch.setattr(nj, "Nebius", FakeNebius)
    monkeypatch.setattr(nj, "collect", lambda remote, job, rec: None)
    monkeypatch.setattr(nj.time, "sleep", lambda s: None)


@pytest.mark.parametrize("failure, expected", [
    (RuntimeError("boom"), nj.EXIT_INFRA),
    (KeyboardInterrupt(), nj.EXIT_INTERRUPTED),
    (nj.VmStopped("STOPPED"), nj.EXIT_STOPPED),
])
def test_run_deletes_the_vm_whatever_happens(tmp_path, monkeypatch, fake_cloud, failure, expected):
    def remote_steps(remote, job, rec, check_vm=None):
        raise failure

    monkeypatch.setattr(nj, "remote_steps", remote_steps)
    code = nj.main(["run", *_run_args(tmp_path, "--attempts", "1"), "--", "true"])
    assert code == expected
    assert FakeNebius.last.deleted == ["computeinstance-1"] and not FakeNebius.last.alive
    rec = json.loads((tmp_path / "out" / "job.json").read_text())
    assert rec["teardown_verified"] is True and rec["instance_id"] == "computeinstance-1"
    assert rec["ip"] == "1.2.3.4" and rec["errors"]
    assert rec["preempted"] is isinstance(failure, nj.VmStopped)


def test_run_success_reports_the_jobs_exit_code(tmp_path, monkeypatch, fake_cloud):
    monkeypatch.setattr(nj, "remote_steps", lambda remote, job, rec, check_vm=None: 5)
    assert nj.main(["run", *_run_args(tmp_path), "--", "false"]) == 5
    assert FakeNebius.last.deleted == ["computeinstance-1"]
    monkeypatch.setattr(nj, "remote_steps", lambda remote, job, rec, check_vm=None: 0)
    assert nj.main(["run", *_run_args(tmp_path), "--", "true"]) == 0
    rec = json.loads((tmp_path / "out" / "job.json").read_text())
    assert rec["command"] == "true" and "deleted" in rec["times"] and not rec["kept"]


def test_up_keeps_the_vm_only_when_setup_succeeds(tmp_path, monkeypatch, fake_cloud):
    monkeypatch.setattr(nj, "remote_steps", lambda remote, job, rec, check_vm=None: None)
    assert nj.main(["up", *_run_args(tmp_path)]) == 0
    assert FakeNebius.last.deleted == [] and FakeNebius.last.alive
    assert json.loads((tmp_path / "out" / "job.json").read_text())["kept"] is True

    def broken(remote, job, rec, check_vm=None):
        raise nj.InfraError("uv sync failed")

    monkeypatch.setattr(nj, "remote_steps", broken)
    assert nj.main(["up", *_run_args(tmp_path)]) == nj.EXIT_INFRA
    assert FakeNebius.last.deleted == ["computeinstance-1"]


def test_teardown_mops_up_an_orphaned_boot_disk(monkeypatch):
    monkeypatch.setattr(nj.time, "sleep", lambda s: None)
    monkeypatch.setattr(FakeNebius, "clouds", {})

    class OrphanDisk(FakeNebius):
        def __init__(self):
            super().__init__("nebius", "p")
            self.disk = nj.Disk("computedisk-9", "md-job-x-boot", "READY")

        def disks(self):
            return [self.disk] if self.disk else []

        def delete_disk(self, did):
            assert did == "computedisk-9"
            self.disk = None

    nb = OrphanDisk()  # creation died after the disk was made: no instance, a leftover disk
    nj.teardown(nb, "md-job-x")
    assert nb.disk is None


def test_teardown_raises_if_something_survives(monkeypatch):
    monkeypatch.setattr(nj.time, "sleep", lambda s: None)
    monkeypatch.setattr(FakeNebius, "clouds", {})
    clock = iter(range(0, 10_000, 100))
    monkeypatch.setattr(nj.time, "time", lambda: next(clock))

    class Stuck(FakeNebius):
        def __init__(self):
            super().__init__("nebius", "p")
            self.alive["i-1"] = nj.Instance("i-1", "md-job-y", "DELETING", None)

        def delete_instance(self, iid):
            self.deleted.append(iid)  # accepted, but the instance never goes away

    with pytest.raises(nj.InfraError, match="teardown incomplete"):
        nj.teardown(Stuck(), "md-job-y")


def test_cli_rejects_missing_or_misplaced_job_commands(tmp_path):
    with pytest.raises(SystemExit):
        nj.main(["run", *_run_args(tmp_path)])  # run needs `-- CMD`
    with pytest.raises(SystemExit):
        nj.main(["up", *_run_args(tmp_path), "--", "true"])  # up takes none


# ------------------------------------------------------------------------------------- regions + prices

REGIONS = {"tenant": "tenant-t", "regions": {
    "uk-south1": {"project": "project-e03", "subnet": "vpcsubnet-e03", "image_project": "project-e03public-images"},
    "eu-west2": {"project": "project-e04", "subnet": "vpcsubnet-e04", "image_project": "project-e04public-images"},
    "us-north1": {"project": "project-u02", "subnet": "vpcsubnet-u02", "image_project": "project-u02public-images"},
    "us-central1": {"project": "project-u00", "subnet": "vpcsubnet-u00", "image_project": "project-u00public-images"},
}}
# 2026-10-09's B300 spot prices; us-central1 has a price but no B300
PRICES = {"project-e03": 7.69, "project-e04": 1.29, "project-u02": 0.99, "project-u00": 2.0}


def _advice(region, platform="gpu-b300-sxm", preset="1gpu-24vcpu-346gb", level="HIGH", available=8):
    """One row of `nebius capacity resource-advice list` (shape of the real 2026-10-09 output)."""
    return {"spec": {"region": region, "compute_instance": {"platform": platform, "preset": {"name": preset}}},
            "status": {"preemptible": {"availability_level": f"AVAILABILITY_LEVEL_{level}", "available": available,
                                       "limit": 8},
                       "on_demand": {"availability_level": "AVAILABILITY_LEVEL_LOW", "limit": 32}}}


@pytest.fixture
def regions_file(tmp_path, monkeypatch, fake_cloud):
    path = tmp_path / "regions.json"
    path.write_text(json.dumps(REGIONS))
    monkeypatch.setattr(FakeNebius, "PRICES", dict(PRICES))
    monkeypatch.setattr(FakeNebius, "CAPACITY", [_advice("uk-south1"), _advice("eu-west2"), _advice("us-north1"),
                                                 _advice("us-central1", platform="gpu-b200-sxm")])
    return path


def _region_args(tmp_path: Path, regions_file: Path, *extra: str) -> list[str]:
    return ["--hours", "2", "--regions-file", str(regions_file), "--ssh-key", str(_keyfile(tmp_path)),
            "--out", str(tmp_path / "out"), *extra]


def test_capacity_of_reads_the_requested_mode():
    rows = [_advice("us-north1"), _advice("uk-south1", level="LIMIT_REACHED", available=0)]
    assert nj.capacity_of(rows, "us-north1", "gpu-b300-sxm", "1gpu-24vcpu-346gb", spot=True) == (True, "HIGH", 8)
    assert nj.capacity_of(rows, "us-north1", "gpu-b300-sxm", "1gpu-24vcpu-346gb", spot=False) == (True, "LOW", None)
    assert nj.capacity_of(rows, "uk-south1", "gpu-b300-sxm", "1gpu-24vcpu-346gb", spot=True) == (
        True, "LIMIT_REACHED", 0)
    assert nj.capacity_of(rows, "eu-west2", "gpu-b300-sxm", "1gpu-24vcpu-346gb", spot=True) == (False, None, None)
    assert nj.capacity_of(rows, "us-north1", "gpu-b300-sxm", "8gpu-192vcpu-2768gb", spot=True)[0] is False


def test_candidates_are_cheapest_first_within_the_cap_capacity_and_quota():
    offers = [nj.Offer("uk-south1", 7.69, "HIGH", 8), nj.Offer("eu-west2", 1.29, "HIGH", 8),
              nj.Offer("us-north1", 0.99, "HIGH", 8), nj.Offer("us-central1", note="not offered here"),
              nj.Offer("full", 0.5, "LIMIT_REACHED", 8), nj.Offer("no-quota", 0.5, "HIGH", 0),
              nj.Offer("unpriced", None, "HIGH", 8), nj.Offer("tie-low", 1.29, "LOW", 8)]
    assert [o.region for o in nj.candidates(offers, None)] == ["us-north1", "eu-west2", "tie-low", "uk-south1"]
    assert [o.region for o in nj.candidates(offers, 1.0)] == ["us-north1"]
    assert nj.candidates(offers, 0.5) == []
    assert offers[0].skip_reason(5.0) == "above the $5/h cap"
    table = nj.offer_table(offers, None, "us-north1")
    assert "<- cheapest" in table and "not offered here" in table and "quota used up" in table


def test_estimated_cost_holds_each_price_until_the_next_sample():
    samples = [["2026-10-09T18:00:00Z", 1.0], ["2026-10-09T18:30:00Z", 3.0]]
    assert nj.estimated_cost(samples, "2026-10-09T19:00:00Z") == 2.0  # 0.5 h at $1 + 0.5 h at $3
    assert nj.estimated_cost([], "2026-10-09T19:00:00Z") is None


def test_price_sampler_records_until_stopped():
    samples: list = []
    sampler = nj.PriceSampler(lambda: 2.0, samples, interval=0.05)
    sampler.start("2026-10-09T18:00:00Z", 1.0)
    time.sleep(0.3)
    sampler.stop()
    n = len(samples)
    assert samples[0] == ["2026-10-09T18:00:00Z", 1.0] and n >= 3 and samples[-1][1] == 2.0
    time.sleep(0.15)
    assert len(samples) == n  # stopped means stopped


def test_run_uses_the_cheapest_region_and_falls_back_when_creation_fails(tmp_path, monkeypatch, regions_file):
    monkeypatch.setattr(FakeNebius, "FAIL_CREATE", {"project-u02"})  # the cheapest region has no room after all
    monkeypatch.setattr(nj, "remote_steps", lambda remote, job, rec, check_vm=None: 0)
    assert nj.main(["run", *_region_args(tmp_path, regions_file), "--", "true"]) == 0
    clouds = FakeNebius.clouds
    assert clouds["project-u02"].created == 1 and not clouds["project-u02"].alive  # tried, nothing left
    assert clouds["project-e04"].deleted == ["computeinstance-1"] and not clouds["project-e04"].alive
    assert clouds["project-e03"].created == 0  # never reached the expensive region
    rec = json.loads((tmp_path / "out" / "job.json").read_text())
    assert rec["region"] == "eu-west2" and rec["teardown_verified"] is True
    assert {o["region"]: o["price"] for o in rec["offers"]} == {
        "uk-south1": 7.69, "eu-west2": 1.29, "us-north1": 0.99, "us-central1": None}
    assert rec["price_samples"][0][1] == 1.29 and rec["estimated_cost_usd"] is not None
    assert any("create in us-north1" in e for e in rec["errors"])


def test_run_creates_nothing_when_no_region_is_under_the_cap(tmp_path, monkeypatch, regions_file):
    monkeypatch.setattr(nj, "remote_steps", lambda remote, job, rec, check_vm=None: 0)
    code = nj.main(["run", *_region_args(tmp_path, regions_file, "--max-price", "0.5"), "--", "true"])
    assert code == nj.EXIT_INFRA
    assert all(c.created == 0 for c in FakeNebius.clouds.values())
    rec = json.loads((tmp_path / "out" / "job.json").read_text())
    assert rec["max_price"] == 0.5 and rec["instance_id"] is None and "no region qualifies" in rec["errors"][0]


def test_run_can_be_pinned_to_regions(tmp_path, monkeypatch, regions_file):
    monkeypatch.setattr(nj, "remote_steps", lambda remote, job, rec, check_vm=None: 0)
    assert nj.main(["run", *_region_args(tmp_path, regions_file, "--regions", "uk-south1"), "--", "true"]) == 0
    assert FakeNebius.clouds["project-e03"].created == 1
    assert json.loads((tmp_path / "out" / "job.json").read_text())["region"] == "uk-south1"
    with pytest.raises(SystemExit, match="not in"):
        nj.main(["run", *_region_args(tmp_path, regions_file, "--regions", "mars-north1"), "--", "true"])


def test_prices_lists_every_region(tmp_path, capsys, regions_file):
    assert nj.main(["prices", "--regions-file", str(regions_file)]) == 0
    text = capsys.readouterr().out
    lines = {line.split()[0]: line for line in text.splitlines()[2:] if line.strip()}
    assert "0.99" in lines["us-north1"] and "<- cheapest" in lines["us-north1"]
    assert "7.69" in lines["uk-south1"] and "not offered here" in lines["us-central1"]
    assert nj.main(["prices", "--regions-file", str(regions_file), "--max-price", "0.5"]) == nj.EXIT_INFRA


def test_down_and_list_cover_every_region(tmp_path, capsys, regions_file):
    stray = FakeNebius("nebius", "project-e04")
    stray.alive["computeinstance-9"] = nj.Instance("computeinstance-9", "md-job-q1w2e3", "RUNNING", None)
    assert nj.main(["list", "--regions-file", str(regions_file)]) == 0
    assert "md-job-q1w2e3" in capsys.readouterr().out
    assert nj.main(["down", "--regions-file", str(regions_file), "--name", "md-job-q1w2e3"]) == 0
    assert FakeNebius.clouds["project-e04"].deleted == ["computeinstance-9"]
    assert not any(c.alive for c in FakeNebius.clouds.values())
    with pytest.raises(SystemExit, match="exclude each other"):
        nj.main(["list", "--regions-file", str(regions_file), "--project-id", "project-x"])


# ------------------------------------------------------------------------------------- SSH side, locally


class LocalRemote(nj.Remote):
    """Runs the "remote" commands in a local bash whose $HOME is a temp dir: the real quoting, tar
    pipelines and detached steps, minus SSH."""

    def __init__(self, home: Path):
        super().__init__("md@local", home / "known_hosts")
        self.home = home

    def argv(self, command: str) -> list[str]:
        home = shlex.quote(str(self.home))
        return ["bash", "-c", f"cd {home} && export HOME={home} && {command}"]


def test_push_and_fetch_through_a_local_shell(tmp_path, monkeypatch):
    home, repo = tmp_path / "home", tmp_path / "repo"
    home.mkdir()
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("A")
    (repo / "it's here.md").write_text("R")  # a quote in a file name survives the pipeline
    monkeypatch.setattr(nj, "REPO", repo)
    r = LocalRemote(home)
    nj.push_repo(r, ["src/a.py", "it's here.md"])
    assert (home / "md" / "src" / "a.py").read_text() == "A"
    assert (home / "md" / "it's here.md").read_text() == "R"

    inputs = tmp_path / "inputs"
    (inputs / "sub").mkdir(parents=True)
    (inputs / "sub" / "x.jsonl").write_text("X")
    nj.push_path(r, str(inputs), "data/in")  # a directory
    nj.push_path(r, str(inputs / "sub" / "x.jsonl"), "data/one/x.jsonl")  # a file
    assert (home / "md" / "data" / "in" / "sub" / "x.jsonl").read_text() == "X"
    assert (home / "md" / "data" / "one" / "x.jsonl").read_text() == "X"
    with pytest.raises(nj.InfraError):
        nj.push_path(r, str(tmp_path / "nope"), "x")

    out = tmp_path / "out"
    assert nj.fetch_path(r, "data/in", nj.fetch_dest("data/in", out)) == "ok"
    assert (out / "data" / "in" / "sub" / "x.jsonl").read_text() == "X"
    assert nj.fetch_path(r, "missing/thing", out) == "missing"
    (home / nj.REMOTE_LOGS).mkdir()
    (home / nj.REMOTE_LOGS / "job.log").write_text("L")
    assert nj.fetch_path(r, f"~/{nj.REMOTE_LOGS}", out / "remote_logs") == "ok"
    assert (out / "remote_logs" / nj.REMOTE_LOGS / "job.log").read_text() == "L"


@pytest.mark.skipif(shutil.which("setsid") is None, reason="needs setsid")
def test_start_and_poll_through_a_local_shell(tmp_path, capsys):
    home, out = tmp_path / "home", tmp_path / "out"
    (home / "md").mkdir(parents=True)
    out.mkdir()
    r = LocalRemote(home)
    nj.start(r, "job", r"for i in 1 2 3; do echo line $i; sleep 0.3; done; printf 'p 10%%\rp 100%%\n'; exit 4",
             cwd="md")
    assert nj.poll(r, "job", out, interval=0.1) == 4
    remote_log = (home / nj.REMOTE_LOGS / "job.log").read_bytes()
    assert (out / "job.log").read_bytes() == remote_log  # streamed copy == remote file, byte for byte
    printed = capsys.readouterr().out
    assert "job| line 1" in printed and "job| line 3" in printed and "job| p 100%" in printed
    assert "p 10%" not in printed.replace("p 100%", "")  # progress updates collapse to the last


def test_poll_checks_the_vm_when_ssh_drops(tmp_path, monkeypatch):
    monkeypatch.setattr(nj.time, "sleep", lambda s: None)

    class Down(nj.Remote):
        def run(self, command, **kw):
            return subprocess.CompletedProcess([], nj.SSH_DOWN, b"", b"Connection refused")

    calls = []

    def check_vm():
        calls.append(1)
        if len(calls) == 2:
            raise nj.VmStopped("STOPPED")

    with pytest.raises(nj.VmStopped):
        nj.poll(Down("md@h", tmp_path / "kh"), "job", tmp_path, check_vm=check_vm)
    assert len(calls) == 2
    with pytest.raises(nj.InfraError, match="lost SSH"):  # no VM to ask (exec mode): give up eventually
        nj.poll(Down("md@h", tmp_path / "kh"), "job", tmp_path, max_failures=3)


# ------------------------------------------------------------------------------------- uploads to an HF bucket

SENTINEL = "hf_SENTINELtokenNEVERtoBEprinted0123456789"


def test_parse_upload_and_destinations(tmp_path):
    assert nj.parse_upload("results=hf://buckets/probemaxxing/probe-data/runs/{stamp}-{name}/") == (
        "results", "hf://buckets/probemaxxing/probe-data/runs/{stamp}-{name}")
    assert nj.parse_upload("/mnt/ram/f=hf://buckets/o/b") == ("/mnt/ram/f", "hf://buckets/o/b")
    for bad in ("results", "results=s3://x/y", "=hf://buckets/o/b", "r=hf://buckets/only-namespace"):
        with pytest.raises(ValueError):
            nj.parse_upload(bad)
    script = nj.upload_script([("results", "hf://buckets/o/b/runs/x"), ("~/nebius_job", "hf://buckets/o/b/logs")])
    assert f"trap 'rm -f \"{nj.REMOTE_HF_TOKEN}\"' EXIT" in script and nj.REMOTE_HF_TOKEN.startswith(nj.RAM_DIR)
    assert 'buckets sync results hf://buckets/o/b/runs/x' in script and '"$HOME"/nebius_job' in script


def test_upload_destination_shares_the_out_stamp_and_needs_a_token(tmp_path):
    token = tmp_path / "token"
    args = nj.build_parser().parse_args(
        ["exec", "--host", "md@h", "--out", str(tmp_path / "{stamp}-{name}"), "--hf-token-file", str(token),
         "--upload", "results=hf://buckets/o/b/runs/{stamp}-{name}"])
    with pytest.raises(SystemExit, match="needs a Hugging Face token"):
        nj.job_spec(args, ["true"], "exec")
    token.write_text(SENTINEL)
    job = nj.job_spec(args, ["true"], "exec")
    (src, dest), = job.uploads
    assert src == "results" and dest.endswith(job.out.name) and job.hf_token_file == token


def test_dry_run_shows_the_upload_plan_but_never_the_token(tmp_path, monkeypatch, capsys, fake_cloud):
    monkeypatch.setattr(nj, "repo_files", lambda repo=nj.REPO: ["pyproject.toml"])
    token = tmp_path / "token"
    token.write_text(SENTINEL)
    assert nj.main(["run", "--dry-run", *_run_args(tmp_path, "--hf-token-file", str(token), "--upload",
                                                     "results=hf://buckets/o/b/runs/{name}"), "--", "true"]) == 0
    text = capsys.readouterr().out
    assert "== upload" in text and "hf://buckets/o/b/runs/md-job-" in text and "mode 600" in text
    assert SENTINEL not in text


def _fake_hf(home: Path, exit_code: int = 0) -> Path:
    """A stand-in for the repo venv's ``hf``: refuses to run without an owner-only token at $HF_TOKEN_PATH (where the
    launcher points huggingface_hub), records its args."""
    hf = home / "md" / ".venv" / "bin" / "hf"
    hf.parent.mkdir(parents=True, exist_ok=True)
    hf.write_text(f"""#!/bin/bash
tok="$HF_TOKEN_PATH"
[ -s "$tok" ] || {{ echo "no token"; exit 3; }}
[ "$(stat -c %a "$tok")" = 600 ] || {{ echo "token mode $(stat -c %a "$tok")"; exit 4; }}
echo "$*" >> "$HOME/hf_calls.txt"
exit {exit_code}
""")
    hf.chmod(0o755)
    return home / "hf_calls.txt"


@pytest.mark.skipif(shutil.which("setsid") is None, reason="needs setsid")
@pytest.mark.parametrize("exit_code", [0, 1])
def test_upload_sends_the_token_privately_uses_it_and_deletes_it(tmp_path, monkeypatch, exit_code):
    monkeypatch.setattr(nj.time, "sleep", lambda s: None)
    monkeypatch.setattr(nj, "REMOTE_HF_TOKEN", str(tmp_path / "ram" / ".hf_token"))  # stands in for /mnt/ram
    home, out = tmp_path / "home", tmp_path / "out"
    (home / "md" / "results").mkdir(parents=True)
    (home / "md" / "results" / "a.json").write_text("{}")
    (home / "md" / "report.md").write_text("r")
    out.mkdir()
    calls = _fake_hf(home, exit_code)
    token = tmp_path / "token"
    token.write_text(SENTINEL)
    job = nj.JobSpec(out=out, uploads=[("results", "hf://buckets/o/b/runs/x"),
                                       ("report.md", "hf://buckets/o/b/runs/x/report.md")], hf_token_file=token)
    rec = nj.Record(mode="exec")
    nj.upload(LocalRemote(home), job, rec)
    assert calls.read_text().splitlines() == ["buckets sync results hf://buckets/o/b/runs/x",
                                              "buckets cp report.md hf://buckets/o/b/runs/x/report.md"]
    assert not (tmp_path / "ram" / ".hf_token").exists()  # deleted, also after a failed upload
    want = "ok" if exit_code == 0 else "failed (1); see upload.log"
    assert rec.uploaded == {"hf://buckets/o/b/runs/x": want, "hf://buckets/o/b/runs/x/report.md": want}
    for f in out.rglob("*"):
        if f.is_file():
            assert SENTINEL not in f.read_text(errors="replace"), f  # job.json, upload.log, …


# ------------------------------------------------------------------------------------- price bid + budget


def test_gpu_count_policy_name_and_bid_argv():
    assert nj.gpu_count("8gpu-192vcpu-2768gb") == 8 and nj.gpu_count("1gpu-24vcpu-346gb") == 1
    assert nj.policy_name("gpu-b300-sxm", 2.5) == "md-cap-gpu-b300-sxm-2p500"
    argv = nj.create_argv("nebius", _spec_vm(pricing_policy_id="pricingpolicy-x"), "x")
    assert _flag(argv, "--spot-pricing-policy-id") == "pricingpolicy-x" and "--follows-spot-price" not in argv
    assert _flag(argv, "--preemptible-on-preemption") == "stop"


def test_ensure_policy_reuses_or_creates(fake_cloud):
    nb = FakeNebius("nebius", "project-p")
    first = nj.ensure_policy(nb, "gpu-b300-sxm", 2.5)
    assert nj.ensure_policy(nb, "gpu-b300-sxm", 2.5) == first  # reused
    assert nj.ensure_policy(nb, "gpu-b300-sxm", 3.0) != first  # a different bid is a different policy
    assert len(FakeNebius.clouds["project-p"].policies) == 2


def test_budget_checks_reserves_and_settles(tmp_path):
    b = nj.Budget(tmp_path / "budget.json")
    b.data = {"limit_usd": 50, "entries": [{"when": "t", "job": "old", "kind": "spent", "usd": 45.0}]}
    b.save()  # every change re-reads the file first
    b.check(5.0, "fits exactly")
    with pytest.raises(SystemExit, match="budget"):
        b.check(5.01, "too much")
    b.reserve("j1", 5.0, "a job")
    assert nj.Budget(tmp_path / "budget.json").committed() == 50.0  # persisted
    b.settle("j1", 1.234, "estimate")
    again = nj.Budget(tmp_path / "budget.json")
    assert again.committed() == 46.23 and again.data["entries"][-1]["kind"] == "spent"
    assert "$3.77 left" in again.summary()


def _ledger(tmp_path, spent: float) -> Path:
    path = tmp_path / "budget.json"
    path.write_text(json.dumps({"limit_usd": 50, "entries": [
        {"when": "2026-10-09T19:22Z", "job": "manual", "kind": "spent", "usd": spent, "what": "earlier test"}]}))
    return path


def test_run_with_bid_and_budget_reserves_then_settles(tmp_path, monkeypatch, regions_file):
    monkeypatch.setattr(nj, "remote_steps", lambda remote, job, rec, check_vm=None: 0)
    ledger = _ledger(tmp_path, 5.2)
    assert nj.main(["run", *_region_args(tmp_path, regions_file, "--max-gpu-price", "2.5",
                                         "--budget-file", str(ledger)), "--", "true"]) == 0
    cloud = FakeNebius.clouds["project-u02"]  # us-north1, $0.99 <= the $2.50 bid
    assert _flag(cloud.create_argv, "--spot-pricing-policy-id") == "pricingpolicy-1"
    assert len(cloud.policies) == 1 and not FakeNebius.clouds["project-e03"].policies
    entries = json.loads(ledger.read_text())["entries"]
    assert [e["kind"] for e in entries] == ["spent", "spent"]  # the reservation was settled
    assert entries[1]["usd"] <= 2.5 * 1 * 2  # never above bid x GPUs x hours
    rec = json.loads((tmp_path / "out" / "job.json").read_text())
    assert rec["max_price"] == 2.5 and rec["region"] == "us-north1"


def test_bid_skips_regions_above_it(tmp_path, monkeypatch, regions_file):
    monkeypatch.setattr(nj, "remote_steps", lambda remote, job, rec, check_vm=None: 0)
    monkeypatch.setattr(FakeNebius, "FAIL_CREATE", {"project-u02", "project-e04"})  # only uk-south1 ($7.69) left
    code = nj.main(["run", *_region_args(tmp_path, regions_file, "--max-gpu-price", "2.5"), "--", "true"])
    assert code == nj.EXIT_INFRA and FakeNebius.clouds["project-e03"].created == 0


def test_budget_refuses_before_creating_anything(tmp_path, monkeypatch, regions_file):
    monkeypatch.setattr(nj, "remote_steps", lambda remote, job, rec, check_vm=None: 0)
    ledger = _ledger(tmp_path, 46.0)  # $4 left; this job's worst case is 2.5 x 1 GPU x 2 h = $5
    with pytest.raises(SystemExit, match="budget"):
        nj.main(["run", *_region_args(tmp_path, regions_file, "--max-gpu-price", "2.5",
                                      "--budget-file", str(ledger)), "--", "true"])
    assert all(c.created == 0 and not c.policies for c in FakeNebius.clouds.values())
    assert len(json.loads(ledger.read_text())["entries"]) == 1  # nothing reserved


def test_budget_without_a_bid_assumes_the_on_demand_price(tmp_path, monkeypatch, capsys, regions_file):
    ledger = _ledger(tmp_path, 0.0)
    assert nj.main(["run", "--dry-run", *_region_args(tmp_path, regions_file, "--budget-file", str(ledger)),
                    "--", "true"]) == 0
    out = capsys.readouterr().out
    assert "would reserve $19.80" in out  # fake on-demand = 10 x the $0.99 spot, x 2 h
    assert len(json.loads(ledger.read_text())["entries"]) == 1  # a dry run reserves nothing
    assert not any(c.policies or c.created for c in FakeNebius.clouds.values())


# ------------------------------------------------------------------------------------- relaunch, deadline, reap


def test_a_preempted_run_relaunches_on_a_fresh_vm_within_the_budget(tmp_path, monkeypatch, regions_file):
    outs = []

    def remote_steps(remote, job, rec, check_vm=None):
        outs.append(job.out)
        if len(outs) == 1:
            raise nj.VmStopped("STOPPED")  # preempted mid-job
        return 0

    monkeypatch.setattr(nj, "remote_steps", remote_steps)
    ledger = _ledger(tmp_path, 5.2)
    code = nj.main(["run", *_region_args(tmp_path, regions_file, "--max-gpu-price", "2.5",
                                         "--budget-file", str(ledger)), "--", "true"])
    out = tmp_path / "out"
    assert code == 0 and outs == [out, out / "retry-2"]
    first = json.loads((out / "job.json").read_text())
    second = json.loads((out / "retry-2" / "job.json").read_text())
    assert first["preempted"] is True and first["attempt"] == 1 and first["retries"] == [str(out / "retry-2")]
    assert second["attempt"] == 2 and second["name"] == first["name"] + "-r2" and not second["preempted"]
    cloud = FakeNebius.clouds["project-u02"]  # us-north1 both times: still the cheapest under the bid
    assert cloud.created == 2 and not cloud.alive and len(cloud.deleted) == 2
    entries = json.loads(ledger.read_text())["entries"]
    assert [(e["job"], e["kind"]) for e in entries[1:]] == [(first["name"], "spent"), (second["name"], "spent")]
    assert entries[1]["region"] == "us-north1" and entries[1]["pid"] == os.getpid() and entries[1]["out"] == str(out)


@pytest.mark.parametrize("codes, attempts, expected, vms", [
    ([75, 0], 3, 0, 2),  # stopped early so it can resume: relaunched, then done
    ([75, 75, 75], 3, 75, 3),  # never finishes: stops after --attempts VMs
    ([5], 3, 5, 1),  # a real failure is not retried
    ([75], 1, 75, 1),  # --attempts 1: no relaunch
])
def test_only_resumable_endings_are_relaunched(tmp_path, monkeypatch, fake_cloud, codes, attempts, expected, vms):
    seq = iter(codes)
    monkeypatch.setattr(nj, "remote_steps", lambda remote, job, rec, check_vm=None: next(seq))
    assert nj.main(["run", *_run_args(tmp_path, "--attempts", str(attempts)), "--", "true"]) == expected
    assert FakeNebius.clouds["project-x"].created == vms and not FakeNebius.clouds["project-x"].alive


def test_the_hours_cap_is_not_mistaken_for_a_preemption(tmp_path, monkeypatch, fake_cloud):
    def remote_steps(remote, job, rec, check_vm=None):
        raise nj.VmStopped("STOPPED")

    monkeypatch.setattr(nj, "remote_steps", remote_steps)
    # a 3-minute cap: stopping now is the cap, not a preemption, so the job is too slow and is not relaunched
    assert nj.main(["run", *_run_args(tmp_path, "--hours", "0.05"), "--", "true"]) == nj.EXIT_STOPPED
    assert FakeNebius.clouds["project-x"].created == 1


def test_the_budget_stops_relaunching(tmp_path, monkeypatch, capsys, regions_file):
    def remote_steps(remote, job, rec, check_vm=None):
        raise nj.VmStopped("STOPPED")

    monkeypatch.setattr(nj, "remote_steps", remote_steps)
    monkeypatch.setattr(nj, "estimated_cost", lambda samples, end, **kw: 4.5)
    ledger = _ledger(tmp_path, 41.0)  # $9 left: the first VM's $5 worst case fits, and it then costs $4.50
    code = nj.main(["run", *_region_args(tmp_path, regions_file, "--max-gpu-price", "2.5",
                                         "--budget-file", str(ledger)), "--", "true"])
    # the first VM costs $4.50, leaving $4.50 = 1.8 h at $2.50/h: a second VM with --hours 1.79; then nothing is left
    assert code == nj.EXIT_STOPPED and FakeNebius.clouds["project-u02"].created == 2
    assert "not relaunching" in capsys.readouterr().out
    entries = json.loads(ledger.read_text())["entries"]
    assert [(e["kind"], e["usd"]) for e in entries] == [("spent", 41.0), ("spent", 4.5), ("spent", 4.5)]


def test_waits_for_a_region_to_come_under_the_bid(tmp_path, monkeypatch, regions_file):
    monkeypatch.setattr(FakeNebius, "PRICES", {k: 9.0 for k in PRICES})  # everywhere above the $2.50 bid
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        FakeNebius.PRICES["project-u02"] = 0.99  # 15 minutes later, us-north1 is cheap again

    monkeypatch.setattr(nj.time, "sleep", sleep)
    monkeypatch.setattr(nj, "remote_steps", lambda remote, job, rec, check_vm=None: 0)
    assert nj.main(["run", *_region_args(tmp_path, regions_file, "--max-gpu-price", "2.5"), "--", "true"]) == 0
    assert sleeps[0] == nj.PRICE_RECHECK_SECONDS and FakeNebius.clouds["project-u02"].created == 1
    monkeypatch.setattr(nj.time, "sleep", lambda s: None)
    monkeypatch.setattr(FakeNebius, "PRICES", {k: 9.0 for k in PRICES})
    code = nj.main(["run", *_region_args(tmp_path, regions_file, "--max-gpu-price", "2.5", "--wait-hours", "1"),
                    "--", "true"])
    assert code == nj.EXIT_INFRA  # gave up after an hour of checks


def test_job_script_exports_the_deadline_and_can_power_off_after_the_job():
    plain = nj.job_script("python x.py", [("A", "b c")])
    assert "NEBIUS_JOB_DEADLINE" in plain and "shutdown -h" not in plain and plain.endswith("python x.py")
    script = nj.job_script("python x.py; exit 7", [], poweroff_minutes=15)
    assert "shutdown -h +15" in script and script.endswith("exit $rc") and "(\npython x.py; exit 7\n)" in script
    assert "usec=1760000000000000" in nj.job_script("true", [], deadline_fallback=1760000000)


@pytest.mark.skipif(shutil.which("setsid") is None, reason="needs setsid")
def test_without_systemds_record_the_job_gets_the_launchers_deadline(tmp_path):
    home, out = tmp_path / "home", tmp_path / "out"
    (home / "md").mkdir(parents=True)
    out.mkdir()
    _fake_sudo(home, tmp_path / "no-such-file")
    r = LocalRemote(home)
    nj.start(r, "job", nj.job_script('echo "deadline=$NEBIUS_JOB_DEADLINE"', [], deadline_fallback=1760000000),
             cwd="md")
    assert nj.poll(r, "job", out, interval=0.1) == 0
    assert "deadline=1760000000" in (out / "job.log").read_text()


def _fake_sudo(home: Path, scheduled: Path) -> Path:
    """A stand-in for sudo in the job's PATH ($HOME/.local/bin): ``cat`` reads ``scheduled`` (systemd's pending
    shutdown), ``shutdown`` is recorded, anything else fails."""
    sudo = home / ".local" / "bin" / "sudo"
    sudo.parent.mkdir(parents=True, exist_ok=True)
    sudo.write_text(f"""#!/bin/bash
[ "$1" = -n ] && shift
case "$1" in
  cat) cat {shlex.quote(str(scheduled))} ;;
  shutdown) echo "$*" >> "$HOME/shutdowns.txt" ;;
  *) exit 1 ;;
esac
""")
    sudo.chmod(0o755)
    return home / "shutdowns.txt"


@pytest.mark.skipif(shutil.which("setsid") is None, reason="needs setsid")
@pytest.mark.parametrize("left, powers_off", [(7200, True), (300, False)])
def test_the_job_step_gets_its_deadline_keeps_its_exit_code_and_powers_off(tmp_path, left, powers_off):
    home, out = tmp_path / "home", tmp_path / "out"
    (home / "md").mkdir(parents=True)
    out.mkdir()
    deadline = int(time.time()) + left
    scheduled = tmp_path / "scheduled"
    scheduled.write_text(f"USEC={deadline}000000\nWARN_WALL=1\nMODE=poweroff\n")
    shutdowns = _fake_sudo(home, scheduled)
    r = LocalRemote(home)
    script = nj.job_script('echo "deadline=$NEBIUS_JOB_DEADLINE"; exit 7', [], poweroff_minutes=15)
    nj.start(r, "job", script, cwd="md")
    assert nj.poll(r, "job", out, interval=0.1) == 7
    assert f"deadline={deadline}" in (out / "job.log").read_text()
    assert (home / nj.REMOTE_LOGS / "deadline").read_text().strip() == str(deadline)
    if powers_off:
        assert shutdowns.read_text().split() == ["shutdown", "-h", "+15"]
    else:  # --hours comes before the 15 minutes: leave it
        assert not shutdowns.exists()


@pytest.mark.skipif(shutil.which("setsid") is None, reason="needs setsid")
def test_the_upload_step_puts_back_the_hours_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(nj.time, "sleep", lambda s: None)
    monkeypatch.setattr(nj, "REMOTE_HF_TOKEN", str(tmp_path / "ram" / ".hf_token"))
    home, out = tmp_path / "home", tmp_path / "out"
    (home / "md" / "results").mkdir(parents=True)
    out.mkdir()
    _fake_hf(home)
    shutdowns = _fake_sudo(home, tmp_path / "unused")
    (home / nj.REMOTE_LOGS).mkdir()
    (home / nj.REMOTE_LOGS / "deadline").write_text(str(int(time.time()) + 3600))
    token = tmp_path / "token"
    token.write_text(SENTINEL)
    job = nj.JobSpec(out=out, uploads=[("results", "hf://buckets/o/b/x")], hf_token_file=token)
    nj.upload(LocalRemote(home), job, nj.Record(mode="run"))
    restore, rearm = shutdowns.read_text().splitlines()
    verb, flag, minutes = restore.split()  # first, back to the --hours cap (an hour away here)
    assert (verb, flag) == ("shutdown", "-h") and 58 <= int(minutes.lstrip("+")) <= 60
    assert rearm == "shutdown -h +15"  # then, the job being over, a power-off soon unless the launcher deletes it


def test_with_uploads_the_token_reaches_the_vm_before_the_job(tmp_path, monkeypatch):
    events = []

    class Ok(nj.Remote):
        def run(self, command, **kw):
            return subprocess.CompletedProcess([], 0, b"", b"")

    monkeypatch.setattr(nj, "repo_files", lambda: [])
    monkeypatch.setattr(nj, "push_repo", lambda remote, files: None)
    monkeypatch.setattr(nj, "start", lambda remote, step, script, cwd=None: events.append(("start", step, script)))
    monkeypatch.setattr(nj, "poll", lambda remote, step, out, **kw: 0)
    monkeypatch.setattr(nj, "send_token", lambda remote, token_file: events.append(("token", token_file)))
    monkeypatch.setattr(nj, "upload", lambda remote, job, rec, check_vm=None: events.append(("upload", None)))
    job = nj.JobSpec(out=tmp_path, command="python x.py", uploads=[("r", "hf://buckets/o/b/x")],
                     hf_token_file=tmp_path / "t", poweroff_minutes=15)
    assert nj.remote_steps(Ok("md@h", tmp_path / "kh"), job, nj.Record(mode="run")) == 0
    assert [e[:2] for e in events] == [("start", "setup"), ("token", tmp_path / "t"), ("start", "job"),
                                       ("upload", None)]
    assert "shutdown -h +15" in events[2][2]
    events.clear()
    assert nj.remote_steps(Ok("md@h", tmp_path / "kh"), nj.JobSpec(out=tmp_path, command="true"),
                           nj.Record(mode="run")) == 0
    assert [e[0] for e in events] == ["start", "start"]  # no uploads: the token never travels


def test_launcher_alive_needs_the_same_host_and_a_live_launcher():
    me = {"pid": os.getpid(), "host": socket.gethostname()}
    assert nj.launcher_alive(me) is False  # pytest is not a launcher
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "nebius_job.py", "run"])
    try:
        until = time.time() + 10  # until the child has exec'd (before that, its cmdline is pytest's)
        while b"nebius_job.py" not in Path(f"/proc/{other.pid}/cmdline").read_bytes() and time.time() < until:
            time.sleep(0.05)
        assert nj.launcher_alive({"pid": other.pid, "host": socket.gethostname()}) is True
        assert nj.launcher_alive({"pid": other.pid, "host": "another-pod"}) is False
    finally:
        other.kill()
        other.wait()
    assert nj.launcher_alive({"pid": other.pid, "host": socket.gethostname()}) is False
    assert nj.launcher_alive({}) is False


def test_reap_cleans_up_after_a_launcher_that_died(tmp_path, monkeypatch, capsys, regions_file):
    start = "2026-10-09T20:00:00Z"
    dead_out = tmp_path / "dead-run"
    dead_out.mkdir()
    an_hour_later = "2026-10-09T21:00:00Z"
    (dead_out / "job.json").write_text(json.dumps({  # a real job.json names its VM
        "name": "md-dead", "times": {"create_start": start, "job_start": start}, "hours": 1.5,
        "price_samples": [[start, 7.92], [an_hour_later, 9.0]]}))

    def reserved(job, usd, **info):
        return {"when": start, "job": job, "kind": "reserved", "usd": usd, "what": job, **info}

    ledger = tmp_path / "budget.json"
    ledger.write_text(json.dumps({"limit_usd": 50, "entries": [
        reserved("md-dead", 30.0, pid=999999, host="hub", out=str(dead_out), hours=1.5, project="project-u02"),
        reserved("md-alive", 30.0, pid=os.getpid(), host=socket.gethostname()),
        reserved("md-kept", 20.0, kept=True, project="project-e03"),
        reserved("md-running", 10.0, pid=999998, host="hub", project="project-e04"),
        reserved("md-never-made", 5.0, pid=999997, host="hub"),
    ]}))
    nb_u, nb_e = FakeNebius("nebius", "project-u02"), FakeNebius("nebius", "project-e04")
    nb_u.alive["computeinstance-9"] = nj.Instance("computeinstance-9", "md-dead", "STOPPED", None)
    nb_e.alive["computeinstance-8"] = nj.Instance("computeinstance-8", "md-running", "RUNNING", "1.2.3.4")
    nb_k = FakeNebius("nebius", "project-e03")  # a kept VM, stopped at its --hours: never reaped
    nb_k.alive["computeinstance-7"] = nj.Instance("computeinstance-7", "md-kept", "STOPPED", None)
    monkeypatch.setattr(nj, "launcher_alive", lambda e: e["job"] == "md-alive")
    assert nj.main(["reap", "--regions-file", str(regions_file), "--budget-file", str(ledger)]) == 0
    entries = {e["job"]: e for e in json.loads(ledger.read_text())["entries"]}
    # conservative: the first hour at the higher bound ($9), the unsampled last half hour at the bid rate ($30/1.5 h)
    assert (entries["md-dead"]["kind"], entries["md-dead"]["usd"]) == ("spent", 19.0)
    assert nb_u.deleted == ["computeinstance-9"] and not nb_u.alive
    assert entries["md-alive"]["kind"] == entries["md-kept"]["kind"] == entries["md-running"]["kind"] == "reserved"
    assert nb_e.alive and not nb_e.deleted  # still running: left for a later reap
    assert nb_k.alive and not nb_k.deleted
    assert (entries["md-never-made"]["kind"], entries["md-never-made"]["usd"]) == ("spent", 5.0)  # no samples
    assert "md-running in eu-west2 is RUNNING" in capsys.readouterr().out


@pytest.mark.skipif(shutil.which("setsid") is None, reason="needs setsid")
@pytest.mark.parametrize("fail, expected", [(False, 0), (True, 1)])
def test_downloads_run_at_once_and_fail_if_any_fails(tmp_path, fail, expected, monkeypatch):
    monkeypatch.setattr(nj, "RAM_DIR", str(tmp_path / "ram"))
    home, out = tmp_path / "home", tmp_path / "out"
    home.mkdir()
    out.mkdir()
    uvx = home / ".local" / "bin" / "uvx"
    uvx.parent.mkdir(parents=True)
    uvx.write_text(f"""#!/bin/bash
# fake `uvx --from X hf download REPO [--revision R] --local-dir DEST`: logs start/end so overlap is visible
repo="$5"; dest="${{@: -1}}"
echo "start $repo $(date +%s.%N)" >> "$HOME/downloads.txt"
mkdir -p "$dest" && echo w > "$dest/weights"
sleep 0.5
echo "end $repo $(date +%s.%N)" >> "$HOME/downloads.txt"
[ "$repo" = org/bad ] && exit 3 || exit 0
""")
    uvx.chmod(0o755)
    hf = [("org/one", None, "one"), ("org/bad" if fail else "org/two", "abc123", "two")]
    r = LocalRemote(home)
    nj.start(r, "download", nj.download_script(hf))
    assert nj.poll(r, "download", out, interval=0.1) == expected
    events = (home / "downloads.txt").read_text().split("\n")
    starts = [float(e.split()[2]) for e in events if e.startswith("start")]
    ends = [float(e.split()[2]) for e in events if e.startswith("end")]
    assert len(starts) == 2 and max(starts) < min(ends)  # both were running at the same time
    assert ("one: " in (out / "download.log").read_text()) and (tmp_path / "ram" / "one" / "weights").exists()



def test_conservative_cost_charges_each_interval_at_its_higher_bound():
    samples = [["2026-10-09T18:00:00Z", 1.0], ["2026-10-09T18:30:00Z", 3.0]]
    assert nj.estimated_cost(samples, "2026-10-09T19:00:00Z") == 2.0
    # 0.5 h at max(1, 3) + 0.5 h with no closing sample at the cap
    assert nj.estimated_cost(samples, "2026-10-09T19:00:00Z", conservative=True, cap_hourly=4.0) == 3.5
    assert nj.estimated_cost(samples, "2026-10-09T19:00:00Z", conservative=True, cap_hourly=2.5) == 2.5  # capped


def test_a_failed_teardown_keeps_the_money_reserved_and_never_relaunches(tmp_path, monkeypatch, regions_file):
    def remote_steps(remote, job, rec, check_vm=None):
        raise nj.VmStopped("STOPPED")  # a preemption, which would normally relaunch

    def stuck(nb, name, instance_id=None, timeout=300):
        raise nj.InfraError("instance is STOPPING; cannot delete")

    monkeypatch.setattr(nj, "remote_steps", remote_steps)
    monkeypatch.setattr(nj, "teardown", stuck)
    ledger = _ledger(tmp_path, 5.2)
    code = nj.main(["run", *_region_args(tmp_path, regions_file, "--max-gpu-price", "2.5",
                                         "--budget-file", str(ledger)), "--", "true"])
    assert code == nj.EXIT_INFRA and FakeNebius.clouds["project-u02"].created == 1  # no second VM
    entry = json.loads(ledger.read_text())["entries"][-1]
    assert entry["kind"] == "reserved" and entry["teardown_failed"] is True and entry["usd"] == 5.0
    rec = json.loads((tmp_path / "out" / "job.json").read_text())
    assert rec["teardown_verified"] is False


@pytest.mark.parametrize("job_code, expected, vms", [(75, 0, 2), (0, 0, 1)])
def test_a_vm_that_stops_after_the_job_keeps_the_jobs_verdict(tmp_path, monkeypatch, fake_cloud, job_code,
                                                               expected, vms):
    calls = []

    def remote_steps(remote, job, rec, check_vm=None):
        calls.append(1)
        if len(calls) == 1:
            rec.exit_code = job_code  # the job ended; then the VM stopped during the final upload
            raise nj.VmStopped("STOPPED")
        return 0

    monkeypatch.setattr(nj, "remote_steps", remote_steps)
    assert nj.main(["run", *_run_args(tmp_path), "--", "true"]) == expected
    assert FakeNebius.clouds["project-x"].created == vms  # 75 is resumed on a fresh VM; 0 is done


def test_run_refuses_to_spend_without_its_ledger(tmp_path, fake_cloud):
    with pytest.raises(SystemExit, match="ledger"):
        nj.main(["run", *_run_args(tmp_path, "--budget-file", str(tmp_path / "missing.json")), "--", "true"])
    assert not FakeNebius.clouds or all(c.created == 0 for c in FakeNebius.clouds.values())


@pytest.mark.skipif(shutil.which("setsid") is None, reason="needs setsid")
def test_upload_paths_are_never_shell_expanded(tmp_path, monkeypatch):
    monkeypatch.setattr(nj.time, "sleep", lambda s: None)
    monkeypatch.setattr(nj, "REMOTE_HF_TOKEN", str(tmp_path / "ram" / ".hf_token"))
    home, out = tmp_path / "home", tmp_path / "out"
    (home / "md").mkdir(parents=True)
    out.mkdir()
    _fake_hf(home)
    token = tmp_path / "token"
    token.write_text(SENTINEL)
    evil = "$(touch $HOME/pwned)`touch $HOME/pwned2`"
    job = nj.JobSpec(out=out, uploads=[(evil, "hf://buckets/o/b/x")], hf_token_file=token)
    nj.upload(LocalRemote(home), job, nj.Record(mode="run"))
    assert not (home / "pwned").exists() and not (home / "pwned2").exists()
    assert "does not exist" in (out / "upload.log").read_text()



def test_a_relaunch_gets_the_hours_the_budget_still_covers(tmp_path, monkeypatch, regions_file):
    seen = []

    def remote_steps(remote, job, rec, check_vm=None):
        seen.append(rec.hours)
        if len(seen) == 1:
            raise nj.VmStopped("STOPPED")
        return 0

    monkeypatch.setattr(nj, "remote_steps", remote_steps)
    monkeypatch.setattr(nj, "estimated_cost", lambda samples, end, **kw: 1.0)
    ledger = _ledger(tmp_path, 42.0)  # $8 left; 1 GPU at $2.50 x 2 h = $5 for the first attempt
    code = nj.main(["run", *_region_args(tmp_path, regions_file, "--max-gpu-price", "2.5",
                                         "--budget-file", str(ledger)), "--", "true"])
    assert code == 0 and seen[0] == 2.0
    assert 2.7 <= seen[1] < 2.8 or seen[1] == 2.0  # $7 left after the first attempt's $1: 2.8 h > 2 h, so 2 h
    entries = json.loads(ledger.read_text())["entries"]
    assert sum(e["usd"] for e in entries) <= 50


def test_no_relaunch_when_the_budget_covers_less_than_half_an_hour(tmp_path, monkeypatch, capsys, regions_file):
    def remote_steps(remote, job, rec, check_vm=None):
        raise nj.VmStopped("STOPPED")

    monkeypatch.setattr(nj, "remote_steps", remote_steps)
    monkeypatch.setattr(nj, "estimated_cost", lambda samples, end, **kw: 3.9)
    ledger = _ledger(tmp_path, 45.0)  # $5 left; the first attempt costs $3.90, leaving $1.10 = 0.44 h at $2.50/h
    code = nj.main(["run", *_region_args(tmp_path, regions_file, "--max-gpu-price", "2.5",
                                         "--budget-file", str(ledger)), "--", "true"])
    assert code == nj.EXIT_STOPPED and FakeNebius.clouds["project-u02"].created == 1
    assert "not relaunching" in capsys.readouterr().out


# ------------------------------------------------------------------------------------- code version


def _git(repo: Path, *args: str) -> str:
    cp = subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com",
                         "-c", "commit.gpgsign=false", *args], check=True, capture_output=True, text=True)
    return cp.stdout.strip()


def test_uncommitted_changes_are_refused_before_anything_is_created(tmp_path, monkeypatch, fake_cloud):
    monkeypatch.setattr(nj, "remote_steps", lambda remote, job, rec, check_vm=None: 0)
    dirty = {**CLEAN, "dirty": True, "diff_sha256": "d" * 64}
    for stamp, why in ((dirty, "uncommitted changes"), (nj.provenance.unknown(), "can't be told")):
        monkeypatch.setattr(nj, "code_version", lambda repo, s=stamp: dict(s))
        for argv in (["run", *_run_args(tmp_path), "--", "true"],
                     ["run", "--dry-run", *_run_args(tmp_path), "--", "true"], ["up", *_run_args(tmp_path)],
                     ["exec", "--host", "md@h", "--out", str(tmp_path / "out"), "--", "true"]):
            with pytest.raises(SystemExit, match=why):
                nj.main(argv)
        assert not FakeNebius.clouds and not (tmp_path / "out").exists()  # not even a price was asked
    assert nj.main(["run", *_run_args(tmp_path, "--allow-dirty"), "--", "true"]) == 0
    assert FakeNebius.clouds["project-x"].created == 1 and not FakeNebius.clouds["project-x"].alive


def test_the_dry_run_shows_the_code_version(tmp_path, monkeypatch, capsys, fake_cloud):
    monkeypatch.setattr(nj, "repo_files", lambda repo=nj.REPO: ["pyproject.toml"])
    assert nj.main(["run", "--dry-run", *_run_args(tmp_path), "--", "true"]) == 0
    text = capsys.readouterr().out
    assert "== code version" in text and "md/CODE_VERSION.json" in text and json.dumps(CLEAN) in text
    assert "code.diff" not in text and not (tmp_path / "out").exists()


def test_the_code_version_goes_up_with_the_repo_before_setup(tmp_path, monkeypatch):
    events = []

    class Ok(nj.Remote):
        def run(self, command, **kw):
            if "CODE_VERSION.json" in command:
                events.append(("code", kw["stdin"].read()))
            return subprocess.CompletedProcess([], 0, b"", b"")

    monkeypatch.setattr(nj, "repo_files", lambda: ["a.py"])
    monkeypatch.setattr(nj, "push_repo", lambda remote, files: events.append(("repo", files)))
    monkeypatch.setattr(nj, "start", lambda remote, step, script, cwd=None: events.append(("start", step)))
    monkeypatch.setattr(nj, "poll", lambda remote, step, out, **kw: 0)
    rec = nj.Record(mode="exec")
    assert nj.remote_steps(Ok("md@h", tmp_path / "kh"), nj.JobSpec(out=tmp_path, command="true"), rec) == 0
    assert events[:2] == [("repo", ["a.py"]), ("code", (json.dumps(CLEAN, indent=1) + "\n").encode())]
    assert events[2:] == [("start", "setup"), ("start", "job")]
    assert rec.code == CLEAN == json.loads((tmp_path / "job.json").read_text())["code"]
    events.clear()  # the tree changed since launch: refused at the push, before anything goes up
    monkeypatch.setattr(nj, "code_version", lambda repo: {**CLEAN, "dirty": True, "diff_sha256": "d" * 64})
    with pytest.raises(nj.InfraError, match="--allow-dirty"):
        nj.remote_steps(Ok("md@h", tmp_path / "kh"), nj.JobSpec(out=tmp_path, command="true"), nj.Record(mode="exec"))
    assert events == []


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_a_dirty_push_saves_the_patch_that_rebuilds_what_was_pushed(tmp_path, monkeypatch):
    """A throwaway repo's real code version: refused while dirty without --allow-dirty. With it, the patch saved in
    --out rebuilds exactly the files pushed, and the VM's copy (no .git) reads the stamp from CODE_VERSION.json."""
    repo, home, out = tmp_path / "repo", tmp_path / "home", tmp_path / "out"
    (repo / "src").mkdir(parents=True)
    home.mkdir()
    _git(repo, "init", "-q")
    (repo / "src" / "a.py").write_text("x = 1\n")
    (repo / "run.sh").write_text("echo hi\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "first")
    monkeypatch.setattr(nj, "REPO", repo)
    monkeypatch.setattr(nj, "code_version", REAL_CODE_VERSION)
    rec = nj.Record(mode="exec")
    head = _git(repo, "rev-parse", "HEAD")
    assert nj.stamp_push(nj.JobSpec(out=out), rec) == {"commit": head, "dirty": False, "diff_sha256": None,
                                                       "source": "git"}

    (repo / "src" / "a.py").write_text("x = 2\n")
    (repo / "src" / "new.py").write_text("y = 3\n")
    (repo / ".env.local").write_text(f"TOKEN={SENTINEL}\n")  # untracked and not ignored: never pushed or recorded
    with pytest.raises(nj.InfraError, match="uncommitted changes"):
        nj.stamp_push(nj.JobSpec(out=out), rec)
    assert not (out / "code.diff").exists()
    code = nj.stamp_push(nj.JobSpec(out=out, allow_dirty=True), rec)
    assert code["dirty"] is True and code["commit"] == head and rec.code == code
    assert json.loads((out / "job.json").read_text())["code"] == code
    patch = (out / "code.diff").read_bytes()
    assert hashlib.sha256(patch).hexdigest() == code["diff_sha256"] and SENTINEL.encode() not in patch
    assert (out / "code_untracked.txt").read_text() == "src/new.py\n"

    r = LocalRemote(home)
    files = nj.repo_files(repo)
    assert files == ["run.sh", "src/a.py", "src/new.py"]
    nj.push_repo(r, files)
    nj.push_code_version(r, code)
    vm_copy = home / "md"
    assert json.loads((vm_copy / "CODE_VERSION.json").read_text()) == code
    assert nj.provenance.code_version(vm_copy) == {**code, "source": "file"}  # what code on the VM stamps

    rebuilt = tmp_path / "rebuilt"
    subprocess.run(["git", "clone", "-q", str(repo), str(rebuilt)], check=True, capture_output=True)
    _git(rebuilt, "apply", str(out / "code.diff"))
    for f in files:
        assert (rebuilt / f).read_bytes() == (vm_copy / f).read_bytes(), f

    _git(repo, "add", "src")
    _git(repo, "commit", "-q", "-m", "second")
    assert nj.stamp_push(nj.JobSpec(out=out), rec)["dirty"] is False  # .env.local doesn't count
    assert not (out / "code.diff").exists() and not (out / "code_untracked.txt").exists()  # not this push's


def test_what_never_travels_is_never_recorded():
    for path in (".env", ".env.local", "a/.envrc", ".venv/lib/x.py", "src/__pycache__/m.cpython-311.pyc", "m.pyc",
                 ".git/config", ".gitignore", "src/env.py", "data/x.json"):
        assert nj.shippable(path) is not nj.provenance.never_recorded(path), path


# ------------------------------------------------------------------------------------- teardown, signals, ledger


def _stamp(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_teardown_retries_a_failed_listing(monkeypatch):
    """One failed `instance list` (an API hiccup) must not skip the delete: it is retried like the delete itself."""
    monkeypatch.setattr(nj.time, "sleep", lambda s: None)
    monkeypatch.setattr(FakeNebius, "clouds", {})

    class Flaky(FakeNebius):
        failures = 0

        def instances(self):
            if self.failures:
                self.failures -= 1
                raise nj.InfraError("nebius compute instance failed (1): 503 Service Unavailable")
            return super().instances()

    nb = Flaky("nebius", "p")
    nb.alive["i-1"] = nj.Instance("i-1", "md-job-z", "RUNNING", None)
    nb.failures = 2
    nj.teardown(nb, "md-job-z")  # no id, as after a failed create: only the listing finds the VM
    assert nb.deleted == ["i-1"] and not nb.alive

    class Down(FakeNebius):  # the listing never works: the VM it knows is deleted all the same, then it raises
        def instances(self):
            raise nj.InfraError("nebius compute instance failed (1): unavailable")

    clock = iter(range(0, 100_000, 100))
    monkeypatch.setattr(nj.time, "time", lambda: next(clock))
    nb = Down("nebius", "p2")
    nb.alive["i-2"] = nj.Instance("i-2", "md-job-w", "RUNNING", None)
    with pytest.raises(nj.InfraError):
        nj.teardown(nb, "md-job-w", "i-2")
    assert nb.deleted == ["i-2"]


def test_the_ledger_records_out_as_an_absolute_path(tmp_path, monkeypatch, regions_file):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(nj, "remote_steps", lambda remote, job, rec, check_vm=None: 0)
    ledger = _ledger(tmp_path, 0.0)
    args = _region_args(tmp_path, regions_file, "--max-gpu-price", "2.5", "--budget-file", str(ledger))
    args[args.index("--out") + 1] = "runs/rel"  # relative, as in the docs' examples
    assert nj.main(["run", *args, "--", "true"]) == 0
    out = json.loads(ledger.read_text())["entries"][-1]["out"]
    assert os.path.isabs(out) and Path(out).resolve() == (tmp_path / "runs" / "rel").resolve()


def test_reap_never_takes_a_record_it_cannot_find_for_no_job(tmp_path, monkeypatch, capsys, regions_file):
    """VMs two hours old (past "an hour and no job") whose launcher died. Reap deletes one only if its own job.json
    says no job started, or if it is 30 min past its --hours cap. An old ledger entry's relative --out is found from
    another directory (here via the repo). A record that can't be found, or that belongs to another VM (a reused
    --out), means "unknown", never "no job"."""
    now = time.time()
    start, long_ago = _stamp(now - 2 * 3600), _stamp(now - 5 * 3600)
    repo = tmp_path / "repo"
    monkeypatch.setattr(nj, "REPO", repo)
    for name, times in (("a", {"create_start": start, "job_start": start}), ("idle", {"create_start": start})):
        (repo / "runs" / name).mkdir(parents=True)
        (repo / "runs" / name / "job.json").write_text(json.dumps({"name": f"md-{name}", "hours": 4, "times": times}))
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")

    def reserved(job, out, created):
        return {"when": created, "job": job, "kind": "reserved", "usd": 10.0, "what": job, "pid": 999999,
                "host": "a-pod-long-gone", "out": out, "hours": 4, "project": "project-u02", "created": created}

    ledger = tmp_path / "budget.json"
    ledger.write_text(json.dumps({"limit_usd": 50, "entries": [
        reserved("md-a", "runs/a", start),  # relative (before 2026-10-12): found via the repo; its job runs
        reserved("md-lost", "runs/lost", start),  # no record anywhere: unknown, so left running
        reserved("md-other", "runs/a", start),  # that folder's record is md-a's, not its own: unknown
        reserved("md-idle", "runs/idle", start),  # its record: no job an hour after creation, an orphan
        reserved("md-wedged", "runs/lost", long_ago),  # unknown, but 5 h old with --hours 4: past its cap
    ]}))
    nb = FakeNebius("nebius", "project-u02")
    for k, name in enumerate(("md-a", "md-lost", "md-other", "md-idle", "md-wedged")):
        nb.alive[f"computeinstance-{k}"] = nj.Instance(f"computeinstance-{k}", name, "RUNNING", "1.2.3.4")
    assert nj.main(["reap", "--regions-file", str(regions_file), "--budget-file", str(ledger)]) == 0
    assert {i.name for i in nb.alive.values()} == {"md-a", "md-lost", "md-other"}
    entries = {e["job"]: e for e in json.loads(ledger.read_text())["entries"]}
    assert {job for job, e in entries.items() if e["kind"] == "reserved"} == {"md-a", "md-lost", "md-other"}
    assert entries["md-idle"]["usd"] == entries["md-wedged"]["usd"] == 10.0  # no price samples: the worst case
    assert "no job.json of md-lost" in capsys.readouterr().out


SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


class SignalEscaped(Exception):
    """A signal reached the test's own handler: the launcher's were not in place."""


@pytest.fixture
def guarded_signals():
    """For tests that send themselves signals: handlers that fail the test instead of killing pytest, and the
    originals put back afterwards."""
    saved = {s: signal.getsignal(s) for s in SIGNALS}

    def escaped(signum, frame):
        raise SignalEscaped(signal.Signals(signum).name)

    for s in SIGNALS:
        signal.signal(s, escaped)
    yield escaped
    for s, handler in saved.items():
        signal.signal(s, handler)


@pytest.mark.parametrize("sig", SIGNALS)
def test_a_signal_during_the_delete_waits_for_it_then_ends_the_run(tmp_path, monkeypatch, fake_cloud,
                                                                   guarded_signals, sig):
    """Ctrl-C, SIGTERM (the launcher's machine shutting down) or SIGHUP (its terminal closing) while the VM is being
    deleted: the delete completes, and then the launcher stops (130) instead of relaunching the resumable job."""
    real_teardown = nj.teardown

    def teardown(nb, name, instance_id=None, timeout=300):
        signal.raise_signal(sig)
        real_teardown(nb, name, instance_id, timeout)

    monkeypatch.setattr(nj, "teardown", teardown)
    monkeypatch.setattr(nj, "remote_steps", lambda remote, job, rec, check_vm=None: nj.EXIT_UNFINISHED)
    assert nj.main(["run", *_run_args(tmp_path), "--", "true"]) == nj.EXIT_INTERRUPTED
    cloud = FakeNebius.clouds["project-x"]
    assert cloud.created == 1 and cloud.deleted == ["computeinstance-1"] and not cloud.alive  # deleted, no relaunch
    rec = json.loads((tmp_path / "out" / "job.json").read_text())
    assert rec["teardown_verified"] is True and any(signal.Signals(sig).name in e for e in rec["errors"])
    assert all(signal.getsignal(s) is guarded_signals for s in SIGNALS)  # the launcher's handlers are gone again


@pytest.mark.parametrize("sig, rest_fetched", [(signal.SIGTERM, False), (signal.SIGHUP, False),
                                               (signal.SIGINT, True)])
def test_sigterm_or_sighup_while_fetching_deletes_the_vm_first(tmp_path, monkeypatch, fake_cloud, guarded_signals,
                                                               sig, rest_fetched):
    """The machine or terminal is going away: the best-effort fetches stop so the VM is deleted at once. A second
    Ctrl-C lets them finish, as before; either way the launcher stops after the delete."""
    fetched = []

    def collect(remote, job, rec):
        signal.raise_signal(sig)
        fetched.append("the rest")

    monkeypatch.setattr(nj, "collect", collect)
    monkeypatch.setattr(nj, "remote_steps", lambda remote, job, rec, check_vm=None: 0)
    assert nj.main(["run", *_run_args(tmp_path), "--", "true"]) == nj.EXIT_INTERRUPTED
    assert bool(fetched) is rest_fetched and FakeNebius.clouds["project-x"].deleted == ["computeinstance-1"]
    rec = json.loads((tmp_path / "out" / "job.json").read_text())
    assert rec["teardown_verified"] is True
    assert any("collection cut short" in e for e in rec["errors"]) is not rest_fetched


def test_under_nohup_a_hangup_does_not_stop_the_run(tmp_path, monkeypatch, fake_cloud, guarded_signals):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)  # as nohup leaves it

    def remote_steps(remote, job, rec, check_vm=None):
        signal.raise_signal(signal.SIGHUP)  # the terminal closes mid-job
        return 0

    monkeypatch.setattr(nj, "remote_steps", remote_steps)
    assert nj.main(["run", *_run_args(tmp_path), "--", "true"]) == 0
    assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN and FakeNebius.clouds["project-x"].deleted
    assert signal.getsignal(signal.SIGTERM) is guarded_signals


def test_a_kept_vms_reservation_is_settled_once_it_is_deleted(tmp_path, regions_file):
    """`up` and `run --keep` leave the VM reserved at its worst case. `down` with the ledger settles it from the prices
    its launcher sampled, the unsampled rest at the worst rate, up to its --hours shutdown. `reap` settles a kept VM
    that is gone (deleted by hand), and never deletes one that is still there."""
    t0 = time.time() - 3 * 3600  # created 3 h ago with --hours 2: it stopped at its cap an hour ago
    start, half = _stamp(t0), _stamp(t0 + 1800)
    kept_out = tmp_path / "kept"
    kept_out.mkdir()
    (kept_out / "job.json").write_text(json.dumps({"name": "md-kept", "hours": 2, "kept": True,
                                                   "times": {"create_start": start},
                                                   "price_samples": [[start, 1.0], [half, 1.5]]}))

    def kept(job, **info):
        return {"when": start, "job": job, "kind": "reserved", "usd": 5.0, "what": job, "pid": os.getpid(),
                "host": socket.gethostname(), "hours": 2, "created": start, "kept": True, **info}

    ledger = tmp_path / "budget.json"
    ledger.write_text(json.dumps({"limit_usd": 50, "entries": [
        kept("md-kept", out=str(kept_out), project="project-u02"),
        kept("md-by-hand", project="project-e04"),  # deleted in the console: gone
        kept("md-still-here", project="project-e03"),
    ]}))
    nb_u, nb_e3 = FakeNebius("nebius", "project-u02"), FakeNebius("nebius", "project-e03")
    nb_u.alive["computeinstance-7"] = nj.Instance("computeinstance-7", "md-kept", "STOPPED", None)
    nb_e3.alive["computeinstance-8"] = nj.Instance("computeinstance-8", "md-still-here", "STOPPED", None)
    assert nj.main(["down", "--regions-file", str(regions_file), "--name", "md-kept",
                    "--budget-file", str(ledger)]) == 0
    assert not nb_u.alive
    entries = {e["job"]: e for e in json.loads(ledger.read_text())["entries"]}
    # 0.5 h at max($1.00, $1.50), then 1.5 h unsampled at the worst rate ($5 reserved for 2 h)
    assert (entries["md-kept"]["kind"], entries["md-kept"]["usd"]) == ("spent", 0.75 + 3.75)
    assert entries["md-by-hand"]["kind"] == entries["md-still-here"]["kind"] == "reserved"  # down settles its own
    assert nj.main(["reap", "--regions-file", str(regions_file), "--budget-file", str(ledger)]) == 0
    entries = {e["job"]: e for e in json.loads(ledger.read_text())["entries"]}
    assert (entries["md-by-hand"]["kind"], entries["md-by-hand"]["usd"]) == ("spent", 5.0)  # no record: worst case
    assert entries["md-still-here"]["kind"] == "reserved" and nb_e3.alive and not nb_e3.deleted


def test_run_keep_without_a_vm_leaves_nothing_reserved(tmp_path, monkeypatch, regions_file):
    monkeypatch.setattr(FakeNebius, "FAIL_CREATE", {"project-u02", "project-e04"})  # both regions under the bid
    ledger = _ledger(tmp_path, 0.0)
    code = nj.main(["run", "--keep", *_region_args(tmp_path, regions_file, "--max-gpu-price", "2.5",
                                                    "--budget-file", str(ledger)), "--", "true"])
    assert code == nj.EXIT_INFRA
    entry = json.loads(ledger.read_text())["entries"][-1]
    assert (entry["kind"], entry["usd"]) == ("spent", 0.0) and not entry.get("kept")
    assert json.loads((tmp_path / "out" / "job.json").read_text())["kept"] is False


# ------------------------------------------------------------------------------------- CPU VMs, keys, root jobs

KEY = "tk-SECRETkey-NEVERtoBEseen-0123456789abcdef"  # a distinctive fake key: it must never show anywhere
CPU = ("--platform", "cpu-d3", "--preset", "32vcpu-128gb", "--on-demand", "--image-family", "ubuntu24.04-driverless")
# the pre-RL rollouts this was built for (episode workers as given: N)
ROLLOUT = [".venv/bin/python", "experiments/eval_terminal_monitors_baseline.py", "--config",
           "experiments/configs/terminal_verifier_control.json", "--n-prompts", "512", "--samples", "64",
           "--episode-workers", "N", "--run-name", "pre-rl-probe-v2", "--set", "monitors=[]", "seed=1000000"]
GPU_SETUP = """set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked
if ! uv run --no-sync python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "torch cannot see the GPU with the locked wheel: retargeting via scripts/setup_box.sh"
  bash scripts/setup_box.sh
fi
uv run --no-sync python -c 'import torch as t; print("torch", t.__version__, t.version.cuda, t.cuda.get_device_name(0))'
"""


@pytest.fixture
def keys(tmp_path, monkeypatch):
    """This machine's key folder (hub's ~/.secrets) with one key, tinker_api_key; the VM's RAM-disk key folder under
    tmp_path; and no spot bid from the environment."""
    folder = tmp_path / "secrets"
    folder.mkdir(mode=0o700)
    (folder / "tinker_api_key").write_text(KEY)
    monkeypatch.setattr(nj, "SECRETS_DIR", str(folder))
    monkeypatch.setattr(nj, "REMOTE_SECRETS", str(tmp_path / "ram" / "secrets"))
    monkeypatch.delenv("NEBIUS_MAX_GPU_PRICE", raising=False)
    return folder


def _strings(obj):
    """Every string value in a JSON document (not its keys)."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def test_a_cpu_preset_missing_from_the_capacity_advice_is_still_priced(regions_file, monkeypatch, capsys):
    """The advice lists GPU presets only: a CPU preset it doesn't list is priced in every region (on demand), its
    capacity and quota unknown. One it does list keeps its advice; a GPU preset it doesn't list is still skipped."""
    monkeypatch.setattr(FakeNebius, "CAPACITY", [*FakeNebius.CAPACITY,
                                                 _advice("eu-west2", platform="cpu-d3", preset="32vcpu-128gb")])
    _, regions = nj.load_regions(str(regions_file))
    offers = {o.region: o for o in nj.survey("nebius", regions, "tenant-t", "cpu-d3", "32vcpu-128gb", spot=False)}
    assert {r: o.price for r, o in offers.items()} == pytest.approx(
        {"uk-south1": 76.9, "eu-west2": 12.9, "us-north1": 9.9, "us-central1": 20.0})  # the fake's on-demand: 10x
    assert all(not o.note for o in offers.values())
    assert (offers["us-north1"].level, offers["us-north1"].available) == (None, None)
    assert offers["eu-west2"].level == "LOW"  # listed: its on-demand advice
    assert [o.region for o in nj.candidates(list(offers.values()), None)][0] == "us-north1"
    gpu = {o.region: o for o in nj.survey("nebius", regions, "tenant-t", "gpu-b300-sxm", "1gpu-24vcpu-346gb", True)}
    assert gpu["us-central1"].note == "gpu-b300-sxm/1gpu-24vcpu-346gb not offered here"
    assert gpu["us-central1"].price is None
    assert nj.main(["prices", "--regions-file", str(regions_file), *CPU[:4], "--on-demand"]) == 0
    lines = {line.split()[0]: line for line in capsys.readouterr().out.splitlines()[2:] if line.strip()}
    assert "<- cheapest" in lines["us-north1"] and "?" in lines["us-north1"] and "not offered" not in lines["uk-south1"]


def test_on_demand_ignores_the_environments_spot_bid_and_refuses_the_flag(tmp_path, monkeypatch, capsys,
                                                                          regions_file):
    monkeypatch.setenv("NEBIUS_MAX_GPU_PRICE", "2.5")  # what a wrapper (hub's bin/gpu-job) exports by default
    monkeypatch.setattr(nj, "remote_steps", lambda remote, job, rec, check_vm=None: 0)
    assert nj.main(["run", *_region_args(tmp_path, regions_file, *CPU), "--", "true"]) == 0
    assert "--on-demand: ignoring NEBIUS_MAX_GPU_PRICE=2.5, a spot bid" in capsys.readouterr().out
    cloud = FakeNebius.clouds["project-u02"]  # us-north1, the cheapest on demand too
    argv = cloud.create_argv
    assert "--on-demand" in argv and "--spot-pricing-policy-id" not in argv and "--follows-spot-price" not in argv
    assert (_flag(argv, "--resources-platform"), _flag(argv, "--resources-preset")) == ("cpu-d3", "32vcpu-128gb")
    assert _flag(argv, "--boot-disk-managed-disk-source-image-family-image-family") == "ubuntu24.04-driverless"
    assert not any(c.policies for c in FakeNebius.clouds.values())
    rec = json.loads((tmp_path / "out" / "job.json").read_text())
    assert rec["spot"] is False and rec["flags"]["max_gpu_price"] is None and rec["max_price"] is None
    with pytest.raises(SystemExit, match="spot bid; it does not apply to --on-demand"):  # given on the command line
        nj.main(["run", *_region_args(tmp_path, regions_file, *CPU, "--max-gpu-price", "2.5"), "--", "true"])
    with pytest.raises(SystemExit, match="CPU VMs run --on-demand"):
        nj.main(["run", *_region_args(tmp_path, regions_file, *CPU[:4]), "--", "true"])
    created = cloud.created
    assert nj.main(["run", *_region_args(tmp_path, regions_file), "--", "true"]) == 0  # spot keeps the bid
    assert cloud.created == created + 1 and _flag(cloud.create_argv, "--spot-pricing-policy-id") == "pricingpolicy-1"
    assert json.loads((tmp_path / "out" / "job.json").read_text())["flags"]["max_gpu_price"] == 2.5


def test_the_budget_holds_an_on_demand_cpu_vm_at_its_fixed_price(tmp_path, monkeypatch, capsys, regions_file):
    """The worst case is the on-demand price x --hours, and neither the ledger nor job.json claims a GPU."""
    monkeypatch.delenv("NEBIUS_MAX_GPU_PRICE", raising=False)
    monkeypatch.setattr(nj, "repo_files", lambda repo=nj.REPO: ["pyproject.toml"])
    monkeypatch.setattr(FakeNebius, "PRICES", {**PRICES, "project-u02": 0.106})  # us-north1: $1.06/h on demand
    ledger = _ledger(tmp_path, 0.0)
    args = _region_args(tmp_path, regions_file, *CPU, "--hours", "3", "--budget-file", str(ledger))
    assert nj.main(["run", "--dry-run", *args, "--", "true"]) == 0
    text = capsys.readouterr().out
    assert "would reserve $3.18" in text and "== pricing: on-demand, $1.06/h in us-north1" in text
    monkeypatch.setattr(nj, "remote_steps", lambda remote, job, rec, check_vm=None: 0)
    assert nj.main(["run", *args, "--", "true"]) == 0
    entry = json.loads(ledger.read_text())["entries"][-1]
    assert entry["what"].endswith(" (cpu-d3/32vcpu-128gb, ≤3 h)") and "×" not in entry["what"]
    assert entry["kind"] == "spent" and entry["usd"] <= 3.18
    assert entry["note"] == "conservative estimate from the sampled on-demand prices"
    text = capsys.readouterr().out
    assert "from the sampled on-demand prices" in text and "sampled spot" not in text
    rec = json.loads((tmp_path / "out" / "job.json").read_text())
    assert not [s for s in _strings(rec) if "gpu" in s.lower()], "job.json names a GPU"


def test_the_setup_without_a_gpu_skips_the_torch_check_and_the_gpu_one_is_unchanged(tmp_path, monkeypatch, capsys,
                                                                                     fake_cloud):
    cpu = nj.setup_script(False, gpu=False)
    assert cpu.splitlines()[2] == "uv sync --locked" and "sys.version.split()[0]" in cpu
    assert not any(word in cpu for word in ("import torch", "torch.cuda", "setup_box", "nvidia", "sudo"))
    assert nj.setup_script(False) == GPU_SETUP  # byte for byte what it was
    assert nj.setup_script(True) == GPU_SETUP.replace("uv sync --locked", "uv sync --locked --extra gpu").replace(
        "setup_box.sh\n", "setup_box.sh --extra gpu\n")
    p = nj.build_parser()
    exec_args = ["exec", "--host", "md@h", "--out", str(tmp_path)]
    job = nj.job_spec(p.parse_args([*exec_args, "--no-gpu"]), ["true"], "exec")
    assert job.gpu is False and nj.job_setup_script(job) == cpu
    assert nj.job_spec(p.parse_args([*exec_args]), ["true"], "exec").gpu is True
    assert nj.job_spec(p.parse_args(["run", "--hours", "1", "--out", str(tmp_path), *CPU]), ["true"], "x").gpu is False
    for argv in ([*exec_args, "--no-gpu", "--extra-gpu"], ["run", "--hours", "1", "--out", str(tmp_path), *CPU,
                                                           "--extra-gpu"]):
        with pytest.raises(SystemExit, match="--extra-gpu needs a GPU"):
            nj.job_spec(p.parse_args(argv), ["true"], "x")
    monkeypatch.setattr(nj, "repo_files", lambda repo=nj.REPO: ["pyproject.toml"])
    assert nj.main(["run", "--dry-run", *_run_args(tmp_path, *CPU), "--", "true"]) == 0
    text = capsys.readouterr().out
    assert cpu in text and "import torch" not in text and "--on-demand" in text
    assert nj.main(["run", "--dry-run", *_run_args(tmp_path), "--", "true"]) == 0  # a GPU run: as before
    text = capsys.readouterr().out
    assert GPU_SETUP in text and "== keys" not in text and "sudo -n -E" not in text and "\ntrap " not in text


def test_secret_flags_parse_and_a_missing_or_empty_key_is_refused(tmp_path, monkeypatch, capsys, fake_cloud, keys):
    s = nj.parse_secret("tinker_api_key")
    assert (s.name, s.env, s.path) == ("tinker_api_key", "TINKER_API_KEY", keys / "tinker_api_key")
    assert nj.parse_secret("wandb.key=WANDB_API_KEY").env == "WANDB_API_KEY"
    for bad in ("", "../x", ".hidden", "a b", "x/y", "=ENV"):
        with pytest.raises(ValueError, match="--secret wants NAME"):
            nj.parse_secret(bad)
    with pytest.raises(ValueError, match="name one: --secret tinker-key=ENV"):  # TINKER-KEY is no variable name
        nj.parse_secret("tinker-key")
    with pytest.raises(ValueError, match="not a key") as err:  # a key typed after = (as for --env) is not echoed
        nj.parse_secret(f"tinker_api_key={KEY}")
    assert KEY not in str(err.value)
    with pytest.raises(ValueError, match="given twice"):
        nj.job_secrets(["tinker_api_key", "tinker_api_key=OTHER"], set())
    for specs, taken in ((["tinker_api_key"], {"TINKER_API_KEY"}), (["a=X", "b=X"], set()), (["path=PATH"], set())):
        with pytest.raises(ValueError, match="already set"):
            nj.job_secrets(specs, taken)
    exec_args = ["exec", "--host", "md@h", "--out", str(tmp_path / "o")]

    def spec(*extra):
        return nj.job_spec(nj.build_parser().parse_args([*exec_args, *extra]), ["true"], "exec")

    assert spec("--secret", "tinker_api_key").secrets == [s]
    with pytest.raises(SystemExit, match="--env"):  # an --env of the key's variable is ambiguous
        spec("--secret", "tinker_api_key", "--env", "TINKER_API_KEY=x")
    with pytest.raises(SystemExit, match=r"nope: .*nope is missing; put the key there with `hub-secret set nope`"):
        spec("--secret", "nope")
    (keys / "folder").mkdir()
    for name, content in (("blank", ""), ("spaces", "\n  \n"), ("folder", None)):
        if content is not None:
            (keys / name).write_text(content)
        with pytest.raises(SystemExit, match="not a file" if content is None else "is empty"):
            spec("--secret", name)
    with pytest.raises(SystemExit, match="missing"):  # before anything is priced or created
        nj.main(["run", *_run_args(tmp_path, "--secret", "nope"), "--", "true"])
    assert not FakeNebius.clouds
    monkeypatch.setattr(nj, "repo_files", lambda repo=nj.REPO: ["pyproject.toml"])
    assert nj.main(["run", "--dry-run", *_run_args(tmp_path, "--secret", "nope"), "--", "true"]) == 0  # shown instead
    assert "!! missing: a real run refuses it (hub-secret set nope)" in capsys.readouterr().out
    assert nj.main(["up", "--dry-run", *_run_args(tmp_path, "--secret", "tinker_api_key")]) == 0
    text = capsys.readouterr().out
    assert "up runs no job, so --secret tinker_api_key is not sent" in text and "== keys: not sent" in text


def test_no_key_reaches_the_dry_run_the_scripts_job_json_or_the_logs(tmp_path, monkeypatch, capsys, fake_cloud,
                                                                     keys):
    """The rollout job's whole launch, dry and (faked) real: the key is never printed, written or put in a script."""
    monkeypatch.setattr(nj, "repo_files", lambda repo=nj.REPO: ["pyproject.toml"])
    upload = "data/runs/pre-rl-probe-v2=hf://buckets/probemaxxing/inkling-probes/inputs/pre-rl-probe-v2/{stamp}"
    token = tmp_path / "hf_token"
    token.write_text(SENTINEL)
    argv = [*_run_args(tmp_path, *CPU, "--hours", "3", "--secret", "tinker_api_key", "--root-job", "--upload",
                       upload, "--hf-token-file", str(token), "--fetch", "data/runs/pre-rl-probe-v2"), "--", *ROLLOUT]
    assert nj.main(["run", "--dry-run", *argv]) == 0
    out = capsys.readouterr()
    text = out.out + out.err
    assert KEY not in text and SENTINEL not in text
    assert (f"{keys / 'tinker_api_key'} -> {nj.REMOTE_SECRETS}/tinker_api_key on the VM, the job's $TINKER_API_KEY"
            in text)
    assert nj.job_setup_script(nj.JobSpec(out=tmp_path, gpu=False, root_job=True)) in text
    assert "; the command as root) ==" in text and "sudo -n -E -H env \"PATH=$PATH\" bash -c" in text
    assert "--set '\"'\"'monitors=[]'\"'\"' seed=1000000" in text  # the command, quoted inside the root shell
    scripts, removed = [], []

    def remote_steps(remote, job, rec, check_vm=None):
        scripts.append(nj.job_script(job.command, job.env, job.poweroff_minutes, secrets=job.secrets,
                                     root=job.root_job))
        return 0

    monkeypatch.setattr(nj, "remote_steps", remote_steps)
    monkeypatch.setattr(nj, "remove_secrets", lambda remote, secrets: removed.append([s.name for s in secrets]))
    assert nj.main(["run", *argv]) == 0
    assert removed == [["tinker_api_key"]]  # the launcher's own cleanup, before the VM is deleted
    assert KEY not in scripts[0] and f'TINKER_API_KEY="$(cat {nj.REMOTE_SECRETS}/tinker_api_key)"' in scripts[0]
    rec = json.loads((tmp_path / "out" / "job.json").read_text())
    assert (rec["root_job"], rec["sandbox_python"], rec["secrets"]) == (True, "3.11.16",
                                                                        {"tinker_api_key": "TINKER_API_KEY"})
    assert rec["flags"]["secret"] == ["tinker_api_key"] and rec["flags"]["root_job"] is True
    out = capsys.readouterr()
    assert KEY not in out.out + out.err
    for f in tmp_path.rglob("*"):  # job.json, known_hosts, …
        if f.is_file() and f.parent != keys:
            assert KEY not in f.read_text(errors="replace"), f


@pytest.mark.skipif(shutil.which("setsid") is None, reason="needs setsid")
@pytest.mark.parametrize("exit_code", [0, 7])
def test_keys_travel_privately_reach_the_job_and_are_deleted_when_it_ends(tmp_path, keys, exit_code):
    home, out = tmp_path / "home", tmp_path / "out"
    (home / "md").mkdir(parents=True)
    out.mkdir()
    (keys / "wandb_api_key").write_text("wandb-" + KEY)
    secrets = nj.job_secrets(["tinker_api_key", "wandb_api_key=WANDB_API_KEY"], set())
    r = LocalRemote(home)
    nj.send_secrets(r, secrets)
    folder = Path(nj.REMOTE_SECRETS)
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    assert {f.name: stat.S_IMODE(f.stat().st_mode) for f in folder.iterdir()} == {"tinker_api_key": 0o600,
                                                                                  "wandb_api_key": 0o600}
    assert (folder / "tinker_api_key").read_text() == KEY  # byte for byte
    command = (f'printf %s "$TINKER_API_KEY" | sha256sum; printf %s "$WANDB_API_KEY" | sha256sum; '
               f'ls {shlex.quote(str(folder))}; exit {exit_code}')
    script = nj.job_script(command, [], secrets=secrets)
    assert KEY not in script
    nj.start(r, "job", script, cwd="md")
    assert nj.poll(r, "job", out, interval=0.1) == exit_code
    log = (out / "job.log").read_text()
    for value in (KEY, "wandb-" + KEY):
        assert hashlib.sha256(value.encode()).hexdigest() in log  # the job had each key in its variable
    assert "tinker_api_key" in log  # the files were there during the job …
    assert not any(folder.iterdir())  # … and gone once it ended, whatever its exit code
    assert KEY not in log and KEY not in (home / nj.REMOTE_LOGS / "job.log").read_text()
    nj.start(r, "job", nj.job_script("echo command-ran", [], secrets=secrets), cwd="md")
    assert nj.poll(r, "job", out, interval=0.1) == 1  # a key missing on the VM: the command never runs
    assert "command-ran" not in (out / "job.log").read_text()


def test_a_key_folder_that_is_a_link_is_refused(tmp_path, keys):
    home, elsewhere = tmp_path / "home", tmp_path / "elsewhere"
    home.mkdir()
    elsewhere.mkdir()
    Path(nj.REMOTE_SECRETS).parent.mkdir(parents=True)
    Path(nj.REMOTE_SECRETS).symlink_to(elsewhere)
    with pytest.raises(nj.InfraError) as err:
        nj.send_secrets(LocalRemote(home), nj.job_secrets(["tinker_api_key"], set()))
    assert not any(elsewhere.iterdir()) and KEY not in str(err.value)


@pytest.mark.parametrize("ending", ["error", "ctrl-c"])
def test_the_launcher_removes_the_keys_if_it_stops_before_the_job_step_does(tmp_path, monkeypatch, fake_cloud, keys,
                                                                            ending):
    """exec, and a kept VM (run --keep): the keys reached the box, then the launcher stopped before the job step
    could delete them (an error, Ctrl-C). Its cleanup deletes them."""
    home = tmp_path / "home"
    (home / "md").mkdir(parents=True)
    monkeypatch.setattr(nj, "Remote", lambda host, known_hosts, key=None: LocalRemote(home))
    sent = []

    def remote_steps(remote, job, rec, check_vm=None):
        nj.send_secrets(remote, job.secrets)
        sent.append(sorted(p.name for p in Path(nj.REMOTE_SECRETS).iterdir()))
        raise RuntimeError("boom") if ending == "error" else KeyboardInterrupt()

    monkeypatch.setattr(nj, "remote_steps", remote_steps)
    expected = nj.EXIT_INFRA if ending == "error" else nj.EXIT_INTERRUPTED
    code = nj.main(["exec", "--host", "md@h", "--out", str(tmp_path / "exec"), "--secret", "tinker_api_key",
                    "--root-job", "--no-gpu", "--", "true"])
    assert code == expected and sent == [["tinker_api_key"]] and not Path(nj.REMOTE_SECRETS).exists()
    rec = json.loads((tmp_path / "exec" / "job.json").read_text())
    assert rec["secrets"] == {"tinker_api_key": "TINKER_API_KEY"} and rec["root_job"] is True
    code = nj.main(["run", "--keep", *_run_args(tmp_path, *CPU, "--secret", "tinker_api_key"), "--", "true"])
    assert code == expected
    assert FakeNebius.clouds["project-x"].alive and len(sent) == 2  # kept, and the keys …
    assert not Path(nj.REMOTE_SECRETS).exists()  # … are gone from it


def test_keys_reach_the_vm_right_before_the_job(tmp_path, monkeypatch, keys):
    events = []

    class Ok(nj.Remote):
        def run(self, command, **kw):
            return subprocess.CompletedProcess([], 0, b"", b"")

    monkeypatch.setattr(nj, "repo_files", lambda: [])
    monkeypatch.setattr(nj, "push_repo", lambda remote, files: None)
    monkeypatch.setattr(nj, "start", lambda remote, step, script, cwd=None: events.append(("start", step, script)))
    monkeypatch.setattr(nj, "poll", lambda remote, step, out, **kw: 0)
    monkeypatch.setattr(nj, "send_token", lambda remote, token_file: events.append(("token",)))
    monkeypatch.setattr(nj, "send_secrets", lambda remote, secrets: events.append(("keys", [s.name for s in secrets])))
    monkeypatch.setattr(nj, "upload", lambda remote, job, rec, check_vm=None: events.append(("upload",)))
    job = nj.JobSpec(out=tmp_path, command="python x.py", uploads=[("r", "hf://buckets/o/b/x")],
                     hf_token_file=tmp_path / "t", secrets=nj.job_secrets(["tinker_api_key"], set()), gpu=False,
                     root_job=True)
    assert nj.remote_steps(Ok("md@h", tmp_path / "kh"), job, nj.Record(mode="run")) == 0
    assert [e[:2] for e in events] == [("start", "setup"), ("token",), ("keys", ["tinker_api_key"]), ("start", "job"),
                                       ("upload",)]
    assert events[0][2] == nj.setup_script(False, gpu=False, sandbox_python="3.11.16")
    assert nj.ROOT_SUDO in events[3][2] and f"$(cat {nj.REMOTE_SECRETS}/tinker_api_key)" in events[3][2]


def test_the_root_job_setup_installs_the_python_makes_the_repo_private_and_checks_as_root():
    text = nj.setup_script(False, gpu=False, sandbox_python="3.11.16")
    lines = text.splitlines()

    def at(s):
        return next(i for i, line in enumerate(lines) if line.startswith(s))

    assert (at("umask 022") < at('sudo -n install -d -m 755 -o "$(id -un)" /opt/uv-python')
            < at("export UV_PYTHON_INSTALL_DIR=/opt/uv-python") < at("uv python install 3.11.16")
            < at('case "$(readlink -f .venv/bin/python 2>/dev/null)" in /opt/uv-python/*) ;; *) rm -rf .venv')
            < at("uv sync --locked --managed-python --python 3.11.16") < at('chmod 750 "$HOME"')
            < at(f"{nj.ROOT_SUDO} .venv/bin/python -c "))
    assert "torch" not in text and "uv sync --locked\n" not in text  # only the pinned sync
    check = shlex.split(text[text.index(nj.ROOT_SUDO):])
    assert check == [*shlex.split(nj.ROOT_SUDO), ".venv/bin/python", "-c", nj.ROOT_CHECK, "3.11.16",
                     "$PWD/pyproject.toml"]
    gpu = nj.setup_script(True, sandbox_python="3.12.9")
    assert "uv sync --locked --extra gpu --managed-python --python 3.12.9" in gpu and "torch.cuda.is_available()" in gpu
    assert gpu.index("torch.cuda") < gpu.index('chmod 750 "$HOME"')
    p = nj.build_parser()

    def spec(*extra):
        return nj.job_spec(p.parse_args(["exec", "--host", "md@h", "--out", "/tmp/x", *extra]), ["true"], "exec")

    assert (spec("--root-job").root_job, spec("--root-job").sandbox_python) == (True, "3.11.16")
    assert spec("--root-job", "--sandbox-python", "3.12.9").sandbox_python == "3.12.9"
    assert nj.job_fields(spec()) == {"root_job": False, "sandbox_python": None, "secrets": {}}
    with pytest.raises(SystemExit, match="does not apply without --root-job"):
        spec("--sandbox-python", "3.11.16")
    with pytest.raises(SystemExit, match="exact CPython version"):
        spec("--root-job", "--sandbox-python", "3.11")


def test_the_root_check_wants_the_exact_python_and_a_private_repo(monkeypatch, capsys):
    """ROOT_CHECK's own logic, run here with the env's sandbox check faked (the real one needs root)."""
    fake = types.ModuleType("monitordecorrelation.envs.terminal_verifier")
    fake.SANDBOX_USER = "nobody"
    fake._sandbox_ids = lambda: (65534, 65534, "/opt/uv-python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11")
    monkeypatch.setitem(sys.modules, "monitordecorrelation.envs.terminal_verifier", fake)
    readable = []

    def run(argv, **kw):  # the sandbox user's attempt to read the repo
        assert (kw["user"], kw["group"], kw["extra_groups"], argv[0]) == (65534, 65534, [], fake._sandbox_ids()[2])
        return subprocess.CompletedProcess(argv, 0 if readable else 1)

    monkeypatch.setattr(subprocess, "run", run)
    here = sys.version.split()[0]

    def check(version):
        monkeypatch.setattr(sys, "argv", ["-c", version, "/home/md/md/pyproject.toml"])
        exec(nj.ROOT_CHECK, {"__name__": "__main__"})

    check(here)
    assert "nobody cannot read the repo" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="not --sandbox-python 3.0.0"):
        check("3.0.0")
    readable.append(True)
    with pytest.raises(SystemExit, match="nobody can read /home/md/md/pyproject.toml"):
        check(here)
    env_source = (nj.REPO / "src" / "monitordecorrelation" / "envs" / "terminal_verifier.py").read_text()
    assert "\ndef _sandbox_ids() ->" in env_source and '\nSANDBOX_USER = "nobody"' in env_source  # what it imports


def _fake_root_sudo(home: Path, root_home: Path, scheduled: Path) -> None:
    """sudo in the step's PATH for a root job: ``cat`` reads ``scheduled``, ``shutdown`` is recorded, and anything else
    (``-n -E -H env PATH=… bash -c BODY``) runs here (tests aren't root) with FAKE_SUDO_ROOT=1 and what real sudo does
    to the environment with -E: PATH becomes its secure_path and PYTHONPATH is dropped (env_delete); -H sets
    HOME=root_home. Without -E it refuses (real sudo would reset the whole environment). Every call is recorded."""
    sudo = home / ".local" / "bin" / "sudo"
    sudo.parent.mkdir(parents=True, exist_ok=True)
    sudo.write_text(f"""#!/bin/bash
echo "$*" >> "$HOME/sudo_calls.txt"
opts=" "
while [[ "$1" == -* ]]; do opts+="$1 "; shift; done
case "$1" in
  cat) exec cat {shlex.quote(str(scheduled))} ;;
  shutdown) echo "$*" >> "$HOME/shutdowns.txt"; exit 0 ;;
esac
[[ "$opts" == *" -E "* ]] || {{ echo "fake sudo: wants -E"; exit 99; }}
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH
[[ "$opts" == *" -H "* ]] && export HOME={shlex.quote(str(root_home))}
export FAKE_SUDO_ROOT=1
exec "$@"
""")
    sudo.chmod(0o755)


@pytest.mark.skipif(shutil.which("setsid") is None, reason="needs setsid")
def test_a_root_job_runs_its_command_through_sudo_with_the_jobs_environment(tmp_path, keys):
    """The detached step stays the login user's (deadline, power-off, exit code); only the command goes through sudo,
    with the step's PATH, the --env values, the key, NEBIUS_JOB_DEADLINE and UV_NO_SYNC=1, and root's HOME."""
    home, root_home, out = tmp_path / "home", tmp_path / "root", tmp_path / "out"
    (home / "md").mkdir(parents=True)
    root_home.mkdir()
    out.mkdir()
    deadline = int(time.time()) + 7200
    scheduled = tmp_path / "scheduled"
    scheduled.write_text(f"USEC={deadline}000000\nMODE=poweroff\n")
    _fake_root_sudo(home, root_home, scheduled)
    secrets = nj.job_secrets(["tinker_api_key"], set())
    r = LocalRemote(home)
    nj.send_secrets(r, secrets)
    command = ('echo "root=$FAKE_SUDO_ROOT home=$HOME pwd=$PWD a=$A pp=$PYTHONPATH uv=$UV_NO_SYNC '
               'deadline=$NEBIUS_JOB_DEADLINE unbuffered=$PYTHONUNBUFFERED"; command -v sudo; '
               'printf %s "$TINKER_API_KEY" | sha256sum; exit 7')
    script = nj.job_script(command, [("A", "b c"), ("PYTHONPATH", "/x")], poweroff_minutes=15, secrets=secrets,
                           root=True)
    assert KEY not in script
    nj.start(r, "job", script, cwd="md")
    assert nj.poll(r, "job", out, interval=0.1) == 7  # the command's own exit code
    log = (out / "job.log").read_text()
    assert (f"root=1 home={root_home} pwd={home / 'md'} a=b c pp=/x uv=1 deadline={deadline} unbuffered=1"
            in log)
    assert str(home / ".local" / "bin" / "sudo") in log  # the step's PATH, not sudo's secure_path
    assert hashlib.sha256(KEY.encode()).hexdigest() in log and KEY not in log
    calls = (home / "sudo_calls.txt").read_text()
    assert f"-n -E -H env PATH={home / '.local' / 'bin'}:" in calls and KEY not in calls
    assert (home / "shutdowns.txt").read_text().split() == ["shutdown", "-h", "+15"]  # the step's power-off, as ever
    assert (home / nj.REMOTE_LOGS / "deadline").read_text().strip() == str(deadline)  # in the login user's home
    assert not any(Path(nj.REMOTE_SECRETS).iterdir())  # and the key is gone
