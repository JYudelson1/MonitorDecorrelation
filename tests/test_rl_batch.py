"""scripts/rl_batch.py (the RL batch driver and its RL-machine supervisor), scripts/tunnel.sh, setup_vllm.sh's restart
mode and experiments/check_probe_long_episodes.py: offline.

Nothing here talks to Nebius, vast.ai, W&B or tinker, or opens SSH. The supervisor runs real processes of a fake
run_experiment.py; tunnel.sh runs with a fake ssh; setup_vllm.sh with fake GPUs, vLLM, uv and curl; the long-episode
check against a fake probe server on localhost; the driver's machine flows with the launcher faked.
"""

from __future__ import annotations

import hashlib
import http.server
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rl_batch", ROOT / "scripts" / "rl_batch.py")
rb = importlib.util.module_from_spec(_spec)
sys.modules["rl_batch"] = rb
_spec.loader.exec_module(rb)
nj = rb.nj

V2 = "data/probes/Inkling-Small/v2"
IID, OOD = f"{V2}/terminal_prerl_v2/with_reasoning/followup", f"{V2}/diverse_deception/with_reasoning/followup"


# --------------------------------------------------------------------------------------------- a fake repo


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A git repo with the real configs and the v2 manifest, and two fake v2 probe folders (committed): the plan's
    world. rl_batch.REPO points at it."""
    r = tmp_path / "repo"
    (r / "experiments" / "configs").mkdir(parents=True)
    for name in ("terminal_verifier_probe_iid.json", "terminal_verifier_probe_ood.json",
                 "terminal_verifier_control_resumable.json", "terminal_verifier_q3_cot.json"):
        shutil.copy(ROOT / "experiments" / "configs" / name, r / "experiments" / "configs" / name)
    (r / "data" / "probe_datasets" / "terminal_prerl_v2").mkdir(parents=True)
    shutil.copy(ROOT / rb.V2_MANIFEST, r / rb.V2_MANIFEST)
    for i, p in enumerate((IID, OOD)):
        d = r / p
        d.mkdir(parents=True)
        (d / "probe.npz").write_bytes(os.urandom(64) + bytes([i]))
        (d / "meta.json").write_text(json.dumps({"view": "with_reasoning", "effort": 0.5, "reader_info": {
            "loader": "bf16", "revision": rb.READER.split("@")[1]}}))
        (d / "golden.jsonl").write_text("{}\n")
    _git(r, "init", "-q")
    _git(r, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    _git(r, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x")
    monkeypatch.setattr(rb, "REPO", r)
    return r


def _roster_file(tmp_path: Path, **over) -> Path:
    data = json.loads((ROOT / "experiments" / "rl_batches" / "probe_rows_v2.json").read_text())
    data.update(over)
    f = tmp_path / "roster.json"
    f.write_text(json.dumps(data))
    return f


# --------------------------------------------------------------------------------------------------- roster


def test_the_shipped_roster_loads_and_states_every_run(repo, tmp_path):
    """The shipped roster loads against the real configs: every run's seed, name and early stop stated."""
    path = ROOT / "experiments" / "rl_batches" / "probe_rows_v2.json"
    roster = rb.load_roster(path)
    assert roster.wandb_entity == "jyudel-redwood-research" and roster.n_steps == 90
    assert all("stop_after_zero_behavior_steps" in r for r in json.loads(path.read_text())["runs"])
    assert [(r.run_name, r.seed) for r in roster.rows] == [
        ("probe-iid-seed-219", 219), ("probe-iid-seed-220", 220), ("probe-iid-seed-221", 221),
        ("probe-ood-seed-222", 222), ("probe-ood-seed-223", 223), ("probe-ood-seed-224", 224)]
    assert {r.stop_after_zero_behavior_steps for r in roster.rows} == {None}


@pytest.mark.parametrize("change, error", [
    (lambda d: d["runs"][0].pop("stop_after_zero_behavior_steps"), "stated"),
    (lambda d: d["runs"][0].update(seed=10_001), "0 to 10,000"),
    (lambda d: d["runs"][0].update(seed=-1), "0 to 10,000"),
    (lambda d: d["runs"][0].update(seed=True), "0 to 10,000"),
    (lambda d: d["runs"][1].update(run_name=d["runs"][0]["run_name"]), "unique"),
    (lambda d: d["runs"][0].update(colour="red"), "a run has"),
    (lambda d: d.update(sets=["seed=3"]), "roster's own field"),
    (lambda d: d["runs"][0].update(sets=["n_steps=5"]), "roster's own field"),
    (lambda d: d["runs"][0].update(sets=["no-equals"]), "key=value"),
    (lambda d: d["runs"][0].update(config="experiments/configs/missing.json"), "not found"),
    (lambda d: d.update(n_steps=0), "n_steps"),
    (lambda d: d.pop("wandb_entity"), "a roster has"),
    (lambda d: d["runs"][0].update(stop_after_zero_behavior_steps=0), "positive"),
])
def test_a_roster_states_everything_and_nothing_else(repo, tmp_path, change, error):
    data = json.loads((ROOT / "experiments" / "rl_batches" / "probe_rows_v2.json").read_text())
    change(data)
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match=error):
        rb.load_roster(f)


def test_a_run_gets_exactly_what_the_roster_states(repo, tmp_path):
    roster = rb.load_roster(_roster_file(tmp_path, sets=["kl_coef=0.0001"], n_steps=360))
    row = rb.dataclasses.replace(roster.rows[0], sets=["monitors.probe_iid.threshold=0.6"],
                                 stop_after_zero_behavior_steps=16)
    argv = rb.run_argv(roster, row)
    assert argv[:4] == [".venv/bin/python", "experiments/run_experiment.py", "--config",
                        "experiments/configs/terminal_verifier_probe_iid.json"]
    assert argv[4:] == ["--set", "seed=219", "run_name=probe-iid-seed-219", "wandb_project=terminal-env-inkling-small",
                        "n_steps=360", "stop_after_zero_behavior_steps=16", "kl_coef=0.0001",
                        "monitors.probe_iid.threshold=0.6"]
    null = rb.run_argv(roster, roster.rows[0])
    assert "stop_after_zero_behavior_steps=null" in null  # stated even when null: the config's value never decides
    reh = rb.rehearsal_roster(roster, None, None)
    assert reh.n_steps == 2 and "eval_every=1" in reh.sets and reh.wandb_project.endswith("-rehearsal")
    assert all(r.run_name.startswith("rehearsal-") for r in reh.rows)


def test_the_rehearsal_logs_to_its_own_wandb_entity_and_project(repo, tmp_path, capsys):
    """--rehearsal-entity (e.g. the owner's own instead of the team's) and --rehearsal-project change the rehearsal's
    W&B entity and project and nothing else, and the supervisor holds its runs to them; without them, the roster's
    entity and <project>-rehearsal."""
    f = _roster_file(tmp_path)
    roster = rb.load_roster(f)
    reh = rb.rehearsal_roster(roster, None, None)
    assert (reh.wandb_entity, reh.wandb_project) == ("jyudel-redwood-research", "terminal-env-inkling-small-rehearsal")
    assert rb.rehearsal_roster(roster, "md-rehearsal", None).wandb_entity == "jyudel-redwood-research"
    assert rb.rehearsal_roster(roster, None, "own-entity") == rb.dataclasses.replace(reh, wandb_entity="own-entity")
    own = rb.rehearsal_roster(roster, "md-rehearsal", "own-entity")
    assert own == rb.dataclasses.replace(reh, wandb_entity="own-entity", wandb_project="md-rehearsal")
    vm = rb.vm_plan(type("B", (), {"settings": rb.Settings()})(), rb.build_plan(own, True))
    assert (vm["entity"], vm["project"]) == ("own-entity", "md-rehearsal")  # what the supervisor checks the runs against
    assert rb.main(["plan", "--roster", str(f), "--offline", "--rehearsal", "--rehearsal-entity", "own-entity"]) == 0
    assert "W&B own-entity/terminal-env-inkling-small-rehearsal (REHEARSAL)" in capsys.readouterr().out


@pytest.mark.parametrize("extra", [["--rehearsal-entity", "own-entity"], ["--rehearsal-project", "md-rehearsal"]])
def test_the_rehearsals_options_are_refused_without_it(repo, tmp_path, extra):
    """Given without --rehearsal they would be dropped silently, and the ~$750 batch would start instead."""
    with pytest.raises(SystemExit, match="give --rehearsal too"):
        rb.main(["plan", "--roster", str(_roster_file(tmp_path)), "--offline", *extra])
    with pytest.raises(SystemExit, match="--compare-runs"):
        rb.main(["start", "--roster", str(_roster_file(tmp_path)), "--out", str(tmp_path / "b"), "--compare-runs", "x"])


def test_the_judge_compare_scores_its_sample_once_per_pass_into_its_own_folder(tmp_path, monkeypatch):
    """Two passes: the same runs and sample, each into judge_compare/pass<i>/ (a pass reuses only its own folder's
    verdicts, so the second is a retest); then the whole folder comes back."""
    pushed, ran, fetched = [], [], []
    monkeypatch.setattr(rb.nj, "push_path", lambda rl, d, dest: pushed.append((d, dest)))
    monkeypatch.setattr(rb, "run_text", lambda rl, cmd, timeout: (ran.append(cmd), (0, "ok"))[1])
    monkeypatch.setattr(rb.nj, "fetch_rsync", lambda rl, path, dest, rsync, timeout: fetched.append(path))
    ops = type("O", (), {"rl": lambda self: "rl"})()
    batch = type("B", (), {"out": tmp_path, "settings": rb.Settings()})()
    rec = rb.judge_compare(batch, ops, ["/x/weak-out-seed-200", "/x/control-seed-216"], passes=2, workers=128)
    assert [dest for _, dest in pushed] == [f"{rb.REMOTE_BATCH}/compare/weak-out-seed-200",
                                            f"{rb.REMOTE_BATCH}/compare/control-seed-216"]
    assert len(ran) == 2 and all("--pos 4 --neg 4 --workers 128" in c for c in ran)
    assert [c.split("--out-dir ")[1].split()[0] for c in ran] == [f"{rb.REMOTE_BATCH}/judge_compare/pass1",
                                                                  f"{rb.REMOTE_BATCH}/judge_compare/pass2"]
    assert fetched == [f"{rb.REMOTE_BATCH}/judge_compare"] and rec["exit"] == 0
    assert rec["report"] == [str(tmp_path / "judge_compare" / f"pass{i}" / "compare_report.json") for i in (1, 2)]
    ran.clear()
    one = rb.judge_compare(batch, ops, ["/x/weak-out-seed-200"])  # the rehearsal's: one pass, the compare's workers
    assert len(ran) == 1 and "--workers" not in ran[0] and ran[0].endswith(
        f"--out-dir {rb.REMOTE_BATCH}/judge_compare --pos 4 --neg 4")
    assert one["report"] == str(tmp_path / "judge_compare" / "compare_report.json")


# ----------------------------------------------------------------------------------------------------- plan


def test_the_plan_pins_the_committed_probes(repo, tmp_path):
    plan = rb.build_plan(rb.load_roster(_roster_file(tmp_path)))
    assert list(plan.probes) == sorted([IID, OOD]) and plan.judges
    for p, sha in plan.probes.items():
        want = hashlib.sha256((repo / p / "probe.npz").read_bytes() + (repo / p / "meta.json").read_bytes())
        assert sha == want.hexdigest()  # whitebox/probe_server.probe_sha, the server's own identity of a probe
    assert [r.probes for r in plan.runs] == [[IID]] * 3 + [[OOD]] * 3
    assert all(r.judges == [8001, 8002] for r in plan.runs)
    vm = rb.vm_plan(type("B", (), {"settings": rb.Settings()})(), plan)
    assert vm["probes"] == plan.probes and vm["entity"] == "jyudel-redwood-research"
    assert [h["url"] for h in vm["health"]] == ["http://127.0.0.1:8178/v1/meta", "http://127.0.0.1:8001/v1/models",
                                                "http://127.0.0.1:8002/v1/models"]
    assert vm["health"][0]["shas"] == sorted(set(plan.probes.values()))


def test_the_plan_refuses_what_the_machines_cannot_serve(repo, tmp_path):
    roster = rb.load_roster(_roster_file(tmp_path))
    (repo / IID / "meta.json").write_text('{"changed": true}')  # e.g. deploy_probes.py rewrote it after the commit
    with pytest.raises(SystemExit, match="uncommitted changes"):
        rb.build_plan(roster)
    _git(repo, "checkout", "--", IID)
    cfg_path = repo / "experiments" / "configs" / "terminal_verifier_probe_iid.json"
    cfg = json.loads(cfg_path.read_text())
    for edit, error in [(lambda c: c["monitors"][0].update(server_url="http://127.0.0.1:9999"), "serves probes at"),
                        (lambda c: c["monitors"][1].update(base_url="http://localhost:8009/v1"), "is not one the batch"),
                        (lambda c: c["env_options"].update(require_explanation=False), "rendered as the v2 probes"),
                        (lambda c: c.update(thinking_effort=0.7), "rendered as the v2 probes")]:
        bad = json.loads(json.dumps(cfg))
        edit(bad)
        cfg_path.write_text(json.dumps(bad))
        with pytest.raises(SystemExit, match=error):
            rb.build_plan(roster)
    cfg_path.write_text(json.dumps(cfg))
    shutil.rmtree(repo / OOD)
    with pytest.raises(SystemExit, match="not deployed here"):
        rb.build_plan(roster)


def test_the_estimate_is_the_probe_servers_hours_for_the_probe_rows(repo, tmp_path):
    """6 probe rows × 90 steps: every valid train and eval episode through one copy at ~0.36 s; the judges see the
    evals only, a third of what Vlad's engines served. 360 steps: four times the hours."""
    plan = rb.build_plan(rb.load_roster(_roster_file(tmp_path)))
    est = rb.estimate(plan)
    episodes = 6 * (90 * 128 + 31 * 256) * 0.97
    assert est["probe_server_hours"] == pytest.approx(episodes * rb.PROBE_SECONDS_PER_EPISODE / 3600, abs=0.1)
    assert 10 < est["batch_hours"] < 12 and est["batch_hours"] == est["probe_server_hours"]
    assert est["judge_calls"][8001] == pytest.approx(6 * 31 * 256 * 0.97 * 2)
    assert 0.25 < est["judge_load"][8001] < 0.4 and est["judge_load"][8001] == est["judge_load"][8002]
    long = rb.estimate(rb.build_plan(rb.load_roster(_roster_file(tmp_path, n_steps=360))))
    assert 40 < long["batch_hours"] < 48
    reh = rb.estimate(rb.build_plan(rb.rehearsal_roster(rb.load_roster(_roster_file(tmp_path)), None, None), True))
    assert 0.4 < reh["batch_hours"] < 0.8  # 2 steps and 3 evals of 6 probe runs: ~35 probe-server minutes


def test_leases_keep_half_a_lease_and_stop_at_72_hours():
    now = 1_000_000.0
    row = {"started": now - 3600, "deadline": now + 3 * 3600}  # 3 h left of a 4 h lease: fine
    assert rb.lease_hours_for(row, 4.0, now) is None
    row["deadline"] = now + 1.5 * 3600  # less than half a lease left: a lease from now, in quarter hours
    assert rb.lease_hours_for(row, 4.0, now) == 5.0
    row["deadline"] = now + 600
    assert rb.lease_hours_for({**row, "started": now - 70 * 3600}, 4.0, now) == 72.0
    assert rb.lease_hours_for({**row, "started": now - 72 * 3600, "deadline": now + 600}, 4.0, now) is None
    assert rb.lease_hours_for({**row, "ended": now}, 4.0, now) is None


VAST_BANNER = (b"Warning: Permanently added '[210.157.233.86]:14438' (ED25519) to the list of known hosts.\r\n"
               b"Welcome to vast.ai. If authentication fails, try again after a few seconds, and double check your "
               b"ssh key.\nHave fun!\n")  # what ssh to a vast.ai instance wrote to stderr on 2026-10-27


def test_ssh_s_own_lines_never_reach_what_a_command_answered(monkeypatch):
    """vast.ai's login banner (stderr, every connection) followed the JSON of every curl on an instance: the watch read
    "not JSON" for healthy servers and would have restarted them. The command's own stderr stays."""
    class Remote:
        def __init__(self, out: bytes, err: bytes):
            self.out, self.err = out, err

        def run(self, command, timeout, check, stdin=None):
            return subprocess.CompletedProcess(["ssh"], 0, self.out, self.err)

    models = json.dumps({"data": [{"id": "Qwen/Qwen3.5-35B-A3B-FP8"}]}).encode()
    assert rb.run_text(Remote(models, VAST_BANNER), "curl …") == (0, models.decode())
    assert rb.curl_json(Remote(models, VAST_BANNER), "http://127.0.0.1:8004/v1/models") == (
        True, {"data": [{"id": "Qwen/Qwen3.5-35B-A3B-FP8"}]})
    assert rb.judges_ok(Remote(models, VAST_BANNER), (8004,)) is None
    assert rb.run_text(Remote(b"", VAST_BANNER + b"curl: (7) Failed to connect\n"), "curl …") == (
        0, "curl: (7) Failed to connect\n")
    report = '{"ok": true, "end_to_end": {"x": {"ok": true}}}'
    assert rb.json_in(report + "\n" + VAST_BANNER.decode()) == {"ok": True, "end_to_end": {"x": {"ok": True}}}
    assert rb.json_in('[{"tokens": 8012, "status": 200}]\nHave fun!\n') == [{"tokens": 8012, "status": 200}]
    with pytest.raises(ValueError):
        rb.json_in(VAST_BANNER.decode())


def test_the_tunnel_key_opens_the_servers_ports_and_nothing_else():
    line = rb.authorized_line("ssh-ed25519 AAAAkey rl-batch x", [8002, 8001])
    assert line == ('restrict,port-forwarding,permitopen="127.0.0.1:8001",permitopen="127.0.0.1:8002",'
                    'command="/bin/false" ssh-ed25519 AAAAkey rl-batch x')
    conf = rb.tunnel_conf("root@ssh5.vast.ai", 31234, rb.tunnel_forwards("judges"), "judges.57.known_hosts")
    assert "FORWARDS='8001:127.0.0.1:8001 8002:127.0.0.1:8002'" in conf and "PORT=31234" in conf
    assert rb.tunnel_forwards("probe") == ["8178:127.0.0.1:8178"]


def test_authorizing_the_tunnel_key_appends_its_line_once_after_an_unended_last_line(tmp_path):
    """vast.ai's authorized_keys ends without a newline (the plumbing test, 2026-10-23): the key goes on a line of its
    own, the existing key stays intact, and a second authorization adds nothing."""
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    keys = home / ".ssh" / "authorized_keys"
    keys.write_text("ssh-ed25519 AAAAhubgpu hub-gpu")  # no newline at its end
    line = rb.authorized_line("ssh-ed25519 AAAAtunnel rl-batch x", [8178])
    remote = type("Local", (nj.Remote,), {"argv": lambda self, c: ["bash", "-c", f"export HOME={home}; {c}"]})(
        "root@local", tmp_path / "kh")
    rb.authorize(remote, line)
    rb.authorize(remote, line)
    assert keys.read_text() == f"ssh-ed25519 AAAAhubgpu hub-gpu\n{line}\n"
    with pytest.raises(nj.InfraError):
        rb.authorize(remote, "")  # nothing to authorize: refused, the file untouched
    assert keys.read_text() == f"ssh-ed25519 AAAAhubgpu hub-gpu\n{line}\n"


def test_the_probe_server_starts_only_with_the_pinned_probes(tmp_path):
    """The server step's first lines refuse a probe folder whose sha256 isn't the pinned one: a folder rewritten since
    the pin would answer the runs' requests with 409, fatal to them."""
    d = tmp_path / "md" / IID
    d.mkdir(parents=True)
    (d / "probe.npz").write_bytes(b"npz")
    (d / "meta.json").write_text("{}")
    sha = rb.probe_sha(d)
    cmd = rb.probe_server_command({IID: sha}, rehearsal=False)
    assert "\nexec .venv/bin/python experiments/inkling_probe_server.py" in cmd and "--max-batch-tokens 1" in cmd \
           and f"--probes {IID}" in cmd
    assert "ALLOC_CONF" not in cmd  # the server's own default (probe_server.default_allocator_conf), not set twice
    head = cmd.split("\nmkdir -p results")[0]
    ok = subprocess.run(["bash", "-c", head + "\necho reached"], cwd=tmp_path / "md", capture_output=True, text=True)
    assert ok.returncode == 0 and "the pinned one" in ok.stdout and "reached" in ok.stdout
    (d / "meta.json").write_text('{"x": 1}')
    bad = subprocess.run(["bash", "-c", head + "\necho reached"], cwd=tmp_path / "md", capture_output=True, text=True)
    assert bad.returncode == 1 and "NOT the pinned" in bad.stdout and "reached" not in bad.stdout
    assert "check_probe_server.py" in rb.probe_server_command({IID: sha}, rehearsal=True)


def test_the_judges_run_pinned_on_localhost_and_restart_on_their_own():
    assert rb.JUDGES_COMMAND == ("VLLM_VERSION=0.30.0 VLLM_HOST=127.0.0.1 VLLM_RESTART=1 exec bash "
                                 "scripts/setup_vllm.sh")
    for port, model in ((8001, "Qwen/Qwen3-30B-A3B-FP8"), (8004, "Qwen/Qwen3.5-35B-A3B-FP8"),
                        (8002, "Qwen/Qwen3.5-35B-A3B-FP8")):
        assert f"http://127.0.0.1:{port}/v1/models | grep -qF {model}" in rb.JUDGES_HEALTH


# ----------------------------------------------------------------------------------------------- tunnel.sh


def _fake_bin(folder: Path, name: str, body: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    f = folder / name
    f.write_text("#!/bin/bash\n" + body + "\n")
    f.chmod(0o755)
    return f


def _wait(test, timeout=15.0, every=0.1):
    t0 = time.time()
    while time.time() - t0 < timeout:
        got = test()
        if got:
            return got
        time.sleep(every)
    return None


@pytest.mark.skipif(shutil.which("setsid") is None, reason="needs setsid")
def test_the_tunnel_loop_reconnects_follows_its_conf_and_stops(tmp_path):
    """scripts/tunnel.sh with a fake ssh that records its arguments and runs until killed: the loop keeps one ssh up,
    starts another when one dies (a forced reconnect), reads its conf again each time (a new target), and stops on
    SIGTERM with its ssh."""
    fake = tmp_path / "bin"
    calls = tmp_path / "calls"
    _fake_bin(fake, "ssh", f'echo "$*" >> {calls}; exec sleep 300')
    state = tmp_path / "state"
    state.mkdir()
    (state / "probe.conf").write_text(rb.tunnel_conf("root@host-a", 1111, rb.tunnel_forwards("probe"), "probe.1.kh"))
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}"}
    loop = subprocess.Popen(["bash", str(ROOT / "scripts" / "tunnel.sh"), str(state), "probe"], env=env,
                            start_new_session=True)
    try:
        assert _wait(lambda: (state / "probe.ssh.pid").exists() and calls.exists())
        args = calls.read_text().splitlines()[0]
        for want in ("-N", "-o ExitOnForwardFailure=yes", "-o ServerAliveInterval=10", "-o ServerAliveCountMax=3",
                     "-o BatchMode=yes", f"-o UserKnownHostsFile={state}/probe.1.kh", f"-o IdentityFile={rb.TUNNEL_KEY}",
                     "-p 1111", "-L 8178:127.0.0.1:8178", "root@host-a"):
            assert want in args, want
        assert int((state / "probe.pid").read_text()) == loop.pid
        assert "ssh running since" in (state / "probe.state").read_text()
        first = int((state / "probe.ssh.pid").read_text())
        (state / "probe.conf").write_text(rb.tunnel_conf("root@host-b", 2222, rb.tunnel_forwards("probe"), "p.2.kh"))
        os.kill(first, signal.SIGTERM)  # the ssh dies (a dropped link, or the driver retargeting): a new one starts
        assert _wait(lambda: len(calls.read_text().splitlines()) >= 2, 20)
        assert "-p 2222" in calls.read_text().splitlines()[1] and "root@host-b" in calls.read_text().splitlines()[1]
        log = (state / "probe.log").read_text()
        assert "ssh exited" in log and "reconnecting in 2s" in log
        second = int(_wait(lambda: (state / "probe.ssh.pid").exists() and (state / "probe.ssh.pid").read_text()))
        assert second != first
    finally:
        loop.send_signal(signal.SIGTERM)
        loop.wait(timeout=10)
    assert "stopped since" in (state / "probe.state").read_text()
    assert not (state / "probe.pid").exists() and not (state / "probe.ssh.pid").exists()
    assert _wait(lambda: rb.pid_ended(second), 5)  # its ssh went with it


# ---------------------------------------------------------------------------------------------- supervisor

FAKE_RUN = r'''
import json, os, sys, time
from pathlib import Path
argv = sys.argv[1:]
sets = dict(a.split("=", 1) for a in argv[argv.index("--set") + 1:] if "=" in a)
name, resume = sets["run_name"], "--resume" in argv
behave = json.loads((Path(os.environ["FAKE_BEHAVIOR"]) / f"{name}.json").read_text())
d = Path("data/runs") / name
(d / "resume_states").mkdir(parents=True, exist_ok=True)
attempt_f = d / "attempts"
attempt = int(attempt_f.read_text()) + 1 if attempt_f.exists() else 1
attempt_f.write_text(str(attempt))
print(f"fake run {name} attempt {attempt} resume={resume} sets={sorted(sets)}", flush=True)
print(f"wandb: View run at https://wandb.ai/{behave.get('entity', 'ent')}/{behave.get('project', 'proj')}/runs/r{attempt}", flush=True)
if not resume:
    (d / "resume_states" / "step_000000.json").write_text(json.dumps({"steps_done": 0, "eval_done": False}))
with open(d / "metrics.jsonl", "a") as f:
    f.write(json.dumps({"step": 0, "time/total_s": 12.5, "time/score_s": 3.0}) + "\n")
time.sleep(behave.get("sleep", 0))
if attempt <= behave.get("fail", 0):
    sys.exit(7)
(d / "final_checkpoint.txt").write_text("tinker://x/weights/final\n")
(d / "resume_states" / "step_000002.json").write_text(json.dumps({"steps_done": 2, "eval_done": True}))
'''


@pytest.fixture
def sup_env(tmp_path, monkeypatch):
    repo = tmp_path / "vmrepo"
    (repo / "data" / "runs").mkdir(parents=True)
    script = tmp_path / "fake" / "experiments" / "run_experiment.py"  # the path a new supervisor recognizes a run by
    script.parent.mkdir(parents=True)
    script.write_text(FAKE_RUN)
    behave = tmp_path / "behave"
    behave.mkdir()
    for k, v in {"TINKER_API_KEY": "t" * 20, "WANDB_API_KEY": "w" * 40, "WANDB_ENTITY": "ent", "WANDB_MODE": "online",
                 "FAKE_BEHAVIOR": str(behave)}.items():
        monkeypatch.setenv(k, v)
    return repo, script, behave


def _sup(repo: Path, script: Path, runs: dict, health=lambda: None, **plan) -> "rb.Supervisor":
    p = {"batch": "b", "entity": "ent", "project": "proj", "need_root": False, "max_attempts": 3, "retry_s": 0.05,
         "runs": [{"name": n, "argv": [sys.executable, str(script), "--config", "c.json", "--set", f"run_name={n}",
                                       "seed=1"]} for n in runs], **plan}
    return rb.Supervisor(p, repo / "rl_batch" / "state", repo, poll_s=0.05, health=health)


def _behave(behave: Path, **runs) -> None:
    for name, b in runs.items():
        (behave / f"{name}.json").write_text(json.dumps(b))


def test_the_supervisor_runs_every_run_resumes_a_failed_one_and_reports(sup_env):
    repo, script, behave = sup_env
    _behave(behave, a={}, b={"fail": 1})
    sup = _sup(repo, script, ["a", "b"])
    assert sup.run() == 0
    st = json.loads((repo / "rl_batch" / "state" / "status.json").read_text())
    assert st["phase"] == "done"
    a, b = st["runs"]["a"], st["runs"]["b"]
    assert (a["state"], a["attempts"], a["modes"]) == ("done", 0, ["fresh"])
    assert (b["state"], b["attempts"], b["modes"]) == ("done", 1, ["fresh", "resume"])  # from its step-0 save
    assert a["wandb"] == "https://wandb.ai/ent/proj/runs/r1" and a["steps"] == 1 and a["step_s_median"] == 12.5
    log = (repo / "data" / "runs" / "b" / "run.log").read_text()
    assert "resume=False" in log and "resume=True" in log and "--resume" in log
    assert (repo / "data" / "runs" / "a" / "QUEUE_DONE").exists()
    assert isinstance(st["probe_queue"], list) and st["probe_queue"]


def test_the_supervisor_gives_up_after_its_attempts_and_moves_a_folder_without_a_save_aside(sup_env):
    repo, script, behave = sup_env
    _behave(behave, c={"fail": 9})
    d = repo / "data" / "runs" / "c"
    d.mkdir()
    (d / "config.json").write_text("{}")  # an earlier attempt that died before its first save
    sup = _sup(repo, script, ["c"])
    assert sup.run() == 1
    st = json.loads((repo / "rl_batch" / "state" / "status.json").read_text())
    assert (st["phase"], st["runs"]["c"]["state"], st["runs"]["c"]["attempts"]) == ("failed", "failed", 3)
    assert st["runs"]["c"]["modes"] == ["aside", "resume", "resume"]
    assert (repo / "data" / "runs" / "c.attempt-1" / "config.json").exists()


def test_the_supervisor_waits_for_the_servers(sup_env):
    repo, script, behave = sup_env
    _behave(behave, a={})
    answers = iter(["the probe server: refused", "the judge at :8001: timed out", None])
    sup = _sup(repo, script, ["a"], health=lambda: next(answers, None))
    assert sup.run() == 0
    log = (repo / "rl_batch" / "state" / "supervisor.log").read_text()
    assert "waiting for the servers" in log and "the probe server: refused" in log
    assert json.loads((repo / "rl_batch" / "state" / "status.json").read_text())["runs"]["a"]["attempts"] == 0


def test_the_supervisor_stops_a_run_logging_to_another_wandb_entity(sup_env):
    repo, script, behave = sup_env
    _behave(behave, a={"entity": "someone-else", "sleep": 30}, b={})
    sup = _sup(repo, script, ["a", "b"])
    t0 = time.time()
    assert sup.run() == 1 and time.time() - t0 < 25
    st = json.loads((repo / "rl_batch" / "state" / "status.json").read_text())
    assert st["runs"]["a"]["state"] == "failed" and "someone-else/proj, not ent/proj" in st["runs"]["a"]["note"]
    assert st["runs"]["b"]["state"] == "done"


@pytest.mark.parametrize("env, error", [({"WANDB_API_KEY": ""}, "WANDB_API_KEY not set"),
                                        ({"TINKER_API_KEY": ""}, "TINKER_API_KEY"),
                                        ({"WANDB_ENTITY": "own-entity"}, "not the roster's"),
                                        ({"WANDB_MODE": "offline"}, "not online")])
def test_the_supervisor_refuses_without_the_keys_the_entity_or_online_wandb(sup_env, monkeypatch, env, error):
    repo, script, behave = sup_env
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    sup = _sup(repo, script, ["a"])
    assert sup.run() == 2
    assert error in (repo / "rl_batch" / "state" / "supervisor.log").read_text()
    assert json.loads((repo / "rl_batch" / "state" / "status.json").read_text())["phase"] == "refused"


def test_the_supervisor_refuses_a_probe_that_isnt_the_pinned_one(sup_env):
    repo, script, behave = sup_env
    d = repo / "probe"
    d.mkdir()
    (d / "probe.npz").write_bytes(b"a")
    (d / "meta.json").write_text("{}")
    assert _sup(repo, script, ["a"], probes={"probe": rb.probe_sha(d)}).check_env() is None
    assert "not the pinned" in _sup(repo, script, ["a"], probes={"probe": "0" * 64}).check_env()


def test_the_supervisor_restarts_and_stops_runs_on_request(sup_env):
    """restart-<run>: stopped and started again at once with --resume, not counted as a failure; stop-<run>: stopped
    for good."""
    repo, script, behave = sup_env
    _behave(behave, a={"sleep": 2}, b={"sleep": 60})
    sup = _sup(repo, script, ["a", "b"])
    control = repo / "rl_batch" / "state" / "control"

    def requests():
        _wait(lambda: (repo / "data" / "runs" / "a" / "metrics.jsonl").exists(), 20)
        time.sleep(0.3)
        (control / "restart-a").touch()
        _wait(lambda: "restart a" in (repo / "rl_batch" / "state" / "supervisor.log").read_text(), 20)
        (control / "stop-b").touch()

    threading.Thread(target=requests, daemon=True).start()
    assert sup.run() == 1  # b stopped
    st = json.loads((repo / "rl_batch" / "state" / "status.json").read_text())
    assert (st["runs"]["a"]["state"], st["runs"]["a"]["attempts"], st["runs"]["a"]["restarts"]) == ("done", 0, 1)
    assert st["runs"]["a"]["modes"] == ["fresh", "resume"]
    assert (st["runs"]["b"]["state"], st["runs"]["b"]["note"]) == ("stopped", "stop-b")


def test_a_new_supervisor_takes_over_the_runs_still_running(sup_env):
    repo, script, behave = sup_env
    _behave(behave, a={"sleep": 3})
    first = _sup(repo, script, ["a"])
    first.state.mkdir(parents=True)
    first.round()  # starts a; this supervisor then "dies" (its child stays a zombie here until reaped)
    second = _sup(repo, script, ["a"])
    assert second.run() == 0
    st = json.loads((repo / "rl_batch" / "state" / "status.json").read_text())
    assert st["runs"]["a"]["adopted"] and st["runs"]["a"]["state"] == "done" and st["runs"]["a"]["modes"] == []
    assert "started by an earlier supervisor" in (repo / "rl_batch" / "state" / "supervisor.log").read_text()
    assert (repo / "data" / "runs" / "a" / "attempts").read_text() == "1"  # it ran once: not started a second time
    first.procs["a"].wait()


def test_wandb_check_reads_the_runs_url():
    assert rb.wandb_check("wandb: 🚀 View run at https://wandb.ai/e/p/runs/ab12\n", "e", "p") == (
        "https://wandb.ai/e/p/runs/ab12", None)
    url, why = rb.wandb_check("https://wandb.ai/own-entity/p/runs/x1", "e", "p")
    assert why == "W&B logs to own-entity/p, not e/p"
    assert rb.wandb_check("wandb: W&B syncing is set to `offline` in this directory.", "e", "p")[1].startswith(
        "W&B is offline")
    assert rb.wandb_check("nothing yet", "e", "p") == (None, None)


def test_a_run_is_done_only_with_its_final_state(tmp_path):
    d = tmp_path / "r"
    assert rb.run_start_mode(d) == "fresh" and not rb.run_done(d)
    d.mkdir()
    (d / "run.log").write_text("x")
    assert rb.run_start_mode(d) == "fresh"
    (d / "config.json").write_text("{}")
    assert rb.run_start_mode(d) == "aside"
    (d / "resume_states").mkdir()
    (d / "resume_states" / "step_000000.json").write_text('{"eval_done": false}')
    assert rb.run_start_mode(d) == "resume" and not rb.run_done(d)
    (d / "final_checkpoint.txt").write_text("x")
    assert not rb.run_done(d)
    (d / "resume_states" / "step_000090.json").write_text('{"eval_done": true}')
    assert rb.run_done(d)


def test_tcp_connections_counts_requests_waiting_on_a_port():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    port = srv.getsockname()[1]
    clients = [socket.create_connection(("127.0.0.1", port)) for _ in range(3)]
    try:
        if not Path("/proc/net/tcp").exists():
            pytest.skip("no /proc/net/tcp")
        assert rb.tcp_connections(port) == 3
    finally:
        for c in clients:
            c.close()
        srv.close()


# ------------------------------------------------------------------------------ setup_vllm.sh, restart mode


FAKE_VLLM = r'''
case "$1" in --version) echo 0.32.0; exit 0 ;; esac
port=""; prev=""
for a in "$@"; do [ "$prev" = "--port" ] && port=$a; prev=$a; done
echo "$port $*" >> "$FAKE_CALLS"
n=$(grep -c "^$port " "$FAKE_CALLS")
if [ "$port" = "$DIE_PORT" ] && [ "$n" -le "${DIE_TIMES:-1}" ]; then sleep 1; echo "engine on $port died"; exit 3; fi
exec sleep 300
'''
FAKE_VLLM_PY = r'''
case "$*" in
  *"import vllm"*) exit 0 ;;
  *'m.version("vllm")'*) echo "${FAKE_VLLM_VERSION:-0.32.0}"; exit 0 ;;
  *'m.version("cuda-toolkit")'*) echo 13.0.0; exit 0 ;;
  -) cat > /dev/null; echo "13.0 $FAKE_TK 13.0.0"; exit 0 ;;
esac
exit 0
'''


@pytest.fixture
def vllm_box(tmp_path):
    """A repo copy with setup_vllm.sh and fakes on PATH: 4 H200s, vLLM 0.32.0 and its python, a CUDA 13.0 toolkit,
    uv, c++ and a curl whose health checks pass."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "setup_vllm.sh", repo / "scripts" / "setup_vllm.sh")
    fake, tk = tmp_path / "bin", tmp_path / "tk"
    (tk / "bin").mkdir(parents=True)
    (tk / "include").mkdir()
    (tk / "lib").mkdir()
    (tk / "include" / "cuda_runtime_api.h").write_text("#define CUDART_VERSION 13000\n")
    _fake_bin(tk / "bin", "nvcc", 'echo "Cuda compilation tools, release 13.0, V13.0.88"')
    _fake_bin(fake, "nvidia-smi", 'case "$*" in *query-gpu*) for i in 0 1 2 3; do echo "$i, NVIDIA H200, 143771"; '
                                  'done ;; *) echo "| CUDA Version: 13.0 |" ;; esac')
    _fake_bin(fake, "vllm", FAKE_VLLM)
    _fake_bin(fake, "python", FAKE_VLLM_PY)
    _fake_bin(fake, "uv", 'case "$1" in run) echo "lb $*" >> "$FAKE_CALLS"; exec sleep 300 ;; '
                          'pip) echo "vllm==0.32.0" ;; esac')
    _fake_bin(fake, "c++", "exit 0")
    _fake_bin(fake, "curl", "exit 0")
    _fake_bin(fake, "ldconfig", "exit 0")
    calls = tmp_path / "calls"
    env = {"PATH": f"{fake}:/usr/bin:/bin", "HOME": str(tmp_path), "FAKE_CALLS": str(calls), "FAKE_TK": str(tk),
           "VLLM_VERSION": "0.32.0"}
    return repo, env, calls


def _setup_vllm(repo: Path, env: dict) -> subprocess.Popen:
    return subprocess.Popen(["bash", str(repo / "scripts" / "setup_vllm.sh")], cwd=repo, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, text=True)


@pytest.mark.skipif(shutil.which("setsid") is None, reason="needs setsid")
def test_setup_vllm_restarts_a_dead_server_and_keeps_its_log(vllm_box):
    """VLLM_RESTART=1: the engine on :8004 dies; it alone is started again (its log kept as .log.1), the others and the
    balancer keep running; every server listens on VLLM_HOST, the balancer too; the vLLM env's packages are written."""
    repo, env, calls = vllm_box
    proc = _setup_vllm(repo, {**env, "VLLM_RESTART": "1", "VLLM_HOST": "127.0.0.1", "DIE_PORT": "8004"})
    try:
        assert _wait(lambda: calls.exists() and sum(line.startswith("8004 ")
                                                    for line in calls.read_text().splitlines()) >= 2, 40, 0.5)
        time.sleep(1)
        lines = calls.read_text().splitlines()
        assert [line.split()[0] for line in lines if not line.startswith("lb ")].count("8001") == 1
        assert all("--host 127.0.0.1" in line for line in lines if not line.startswith("lb "))
        assert any("load_balancer.py --listen-port 8002 --backend-ports 8003 8004 8005 --listen-host 127.0.0.1" in line
                   for line in lines)
        assert sum(line.startswith("lb ") for line in lines) == 1
        assert (repo / "logs" / "vllm" / "qwen3.5-35b-8004.log.1").read_text().strip() == "engine on 8004 died"
        assert (repo / "logs" / "vllm" / "freeze.txt").read_text().strip() == "vllm==0.32.0"
        assert proc.poll() is None  # still serving
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        out = proc.communicate(timeout=30)[0]
    assert "restarting qwen3.5-35b-8004 (1 of at most 10" in out and "vllm 0.32.0" in out


def test_setup_vllm_without_restart_stops_everything_and_refuses_another_version(vllm_box):
    repo, env, calls = vllm_box
    proc = _setup_vllm(repo, {**env, "DIE_PORT": "8004"})
    out = proc.communicate(timeout=60)[0]
    assert proc.returncode == 1 and "qwen3.5-35b-8004 exited (code 3)" in out and "restarting" not in out
    proc = _setup_vllm(repo, {**env, "FAKE_VLLM_VERSION": "0.31.0"})
    out = proc.communicate(timeout=60)[0]
    assert proc.returncode == 1 and "vllm 0.31.0 is not VLLM_VERSION=0.32.0" in out


@pytest.mark.skipif(shutil.which("setsid") is None, reason="needs setsid")
def test_setup_vllm_makes_its_venv_on_uvs_own_python_with_its_headers(vllm_box):
    """Triton (vLLM's JIT) compiles against the venv Python's Python.h: on the Nebius image uv took the system
    python3.12, which has none, and every Qwen3.5 server died at its start (2026-10-28). The venv is made with
    --managed-python; one whose Python lacks the headers is made again; one that has them is used as it is."""
    repo, env, calls = vllm_box
    fake = Path(env["PATH"].split(":")[0])
    (fake / "vllm").unlink()  # no vllm on PATH: setup_vllm.sh uses its own .venv-vllm
    tpl = repo.parent / "tpl"
    _fake_bin(tpl, "python", FAKE_VLLM_PY)  # uv's own: Python.h there (the fake answers the check with 0)
    _fake_bin(tpl, "vllm", FAKE_VLLM)
    _fake_bin(fake, "uv", f'echo "uv $*" >> "$FAKE_CALLS"\ncase "$1 $2" in\n'
                          f'  "venv "*) mkdir -p .venv-vllm/bin && cp {tpl}/python .venv-vllm/bin/python ;;\n'
                          f'  "pip install") cp {tpl}/vllm .venv-vllm/bin/vllm ;;\n'
                          '  "pip freeze") echo "vllm==0.32.0" ;;\n'
                          '  run*) echo "lb $*" >> "$FAKE_CALLS"; exec sleep 300 ;;\nesac')
    venv = repo / ".venv-vllm" / "bin"
    _fake_bin(venv, "vllm", FAKE_VLLM)
    _fake_bin(venv, "python", 'case "$*" in *sysconfig*) exit 1 ;; esac\n' + FAKE_VLLM_PY)  # the system's: none

    def run() -> str:
        calls.unlink(missing_ok=True)
        proc = _setup_vllm(repo, env)
        try:
            assert _wait(lambda: calls.exists() and "8005 " in calls.read_text(), 40, 0.5)
        finally:
            os.killpg(proc.pid, signal.SIGTERM)
        return proc.communicate(timeout=30)[0]

    out = run()
    assert "has no Python.h (vLLM's Triton JIT needs it): making it again on uv's own Python" in out
    assert "uv venv --no-config --managed-python --python 3.12 --seed .venv-vllm" in calls.read_text()
    out = run()  # its Python has the headers now: used as it is
    assert "using vllm from .venv-vllm/bin/vllm" in out and "uv venv" not in calls.read_text()


@pytest.mark.skipif(shutil.which("setsid") is None, reason="needs setsid")
def test_setup_vllm_starts_each_server_with_the_vllm_envs_tools_on_its_path(vllm_box):
    """FlashInfer JIT-compiles kernels as an engine starts (sm_103, 2026-10-28) and runs its env's ninja by name: the
    servers, started as .venv-vllm/bin/vllm without that folder on their PATH, failed ("No such file or directory:
    'ninja'"). Each server gets the vLLM env's bin folder first on its PATH, absolute (ninja runs from a build folder)."""
    repo, env, calls = vllm_box
    fake = Path(env["PATH"].split(":")[0])
    (fake / "vllm").unlink()  # vLLM from its own env, as on a fresh machine
    venv = repo / ".venv-vllm" / "bin"
    _fake_bin(venv, "python", FAKE_VLLM_PY)  # its headers there (the fake says so): the env is used as it is
    _fake_bin(venv, "ninja", "exit 0")
    _fake_bin(venv, "vllm", 'case "$1" in --version) echo 0.32.0; exit 0 ;; esac\nport=""; prev=""\n'
                            'for a in "$@"; do [ "$prev" = "--port" ] && port=$a; prev=$a; done\n'
                            'cd /  # FlashInfer runs ninja from its build folder\n'
                            'echo "$port ninja=$(command -v ninja)" >> "$FAKE_CALLS"\nexec sleep 300')
    proc = _setup_vllm(repo, env)
    try:
        assert _wait(lambda: calls.exists() and sum(" ninja=" in x for x in calls.read_text().splitlines()) >= 4,
                     40, 0.5)
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.communicate(timeout=30)
    found = dict(x.split(" ninja=") for x in calls.read_text().splitlines() if " ninja=" in x)
    assert set(found) == {"8001", "8003", "8004", "8005"} and set(found.values()) == {str(venv / "ninja")}


def test_setup_vllm_gives_up_on_a_server_that_keeps_dying(vllm_box):
    repo, env, calls = vllm_box
    proc = _setup_vllm(repo, {**env, "VLLM_RESTART": "1", "VLLM_RESTART_MAX": "1", "DIE_PORT": "8003",
                              "DIE_TIMES": "5"})
    out = proc.communicate(timeout=90)[0]
    assert proc.returncode == 1 and "died 2 times (VLLM_RESTART_MAX=1): stopping everything" in out


def test_the_load_balancer_listens_on_localhost_by_default(tmp_path):
    port = socket.socket()
    port.bind(("127.0.0.1", 0))
    free = port.getsockname()[1]
    port.close()
    proc = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "load_balancer.py"), "--listen-port", str(free),
                             "--backend-ports", str(free + 1), "--report-every", "0"], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        line = proc.stdout.readline()
        assert f"listening on 127.0.0.1:{free}" in line
    finally:
        proc.terminate()
        proc.wait(timeout=10)


# ------------------------------------------------------------------------------------- long-episode check


class _FakeProbeServer(http.server.BaseHTTPRequestHandler):
    """A probe server that can't hold an episode above LIMIT tokens even in query chunks (413), as the real one
    answers then."""
    LIMIT = 10_000
    seen: list = []

    def do_GET(self):  # noqa: N802
        self._send(200, {"probes": {}})

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        ep = body["episodes"][0]
        n = sum(len(t.get("output") or "") for t in ep["turns"]) // 5  # ~tokens of the padding, for the fake
        _FakeProbeServer.seen.append(n)
        if n > self.LIMIT:
            self._send(413, {"error": "episode 0 ran out of GPU memory: OutOfMemoryError: CUDA out of memory",
                             "unscorable": [{"key": ep.get("key"), "view": "with_reasoning", "n_tokens": n}]})
        else:
            self._send(200, {"results": [{"scores": {body["probes"][0]: {"score": 0.1, "logit": -4.0}},
                                          "n_tokens": {"with_reasoning": n}}], "timing": {"forward_s": 0.4}})

    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_the_long_episode_check_pads_episodes_to_length_and_reports_each_answer(tmp_path):
    pytest.importorskip("tml_renderers")
    spec = importlib.util.spec_from_file_location("long_eps", ROOT / "experiments" / "check_probe_long_episodes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeProbeServer)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    out = tmp_path / "long.json"
    probe = ROOT / "data" / "probes" / "Inkling-Small" / "terminal_prerl" / "with_reasoning" / "followup"
    try:
        code = mod.main(["--url", f"http://127.0.0.1:{srv.server_address[1]}", "--probe", str(probe), "--tokens",
                         "2000,6000", "--out", str(out)])
    finally:
        srv.shutdown()
    report = json.loads(out.read_text())
    assert code == 0 and report["ok"] and [r["target_tokens"] for r in report["results"]] == [2000, 6000]
    for r in report["results"]:
        assert r["target_tokens"] <= r["tokens"] < r["target_tokens"] + 30  # the fewest filler rows that reach it
        assert r["server_alive_after"]
    assert report["results"][0]["status"] == 200 and report["results"][0]["score"] == 0.1
    _FakeProbeServer.LIMIT = 500  # now the second length runs out of memory: an answer, reported, not a failure
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeProbeServer)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        code = mod.main(["--url", f"http://127.0.0.1:{srv.server_address[1]}", "--probe", str(probe), "--tokens",
                         "6000", "--out", str(out)])
    finally:
        srv.shutdown()
        _FakeProbeServer.LIMIT = 10_000
    row = json.loads(out.read_text())["results"][0]
    assert code == 0 and row["status"] == 413 and "out of memory" in row["error"]


# ------------------------------------------------------------------------------- the driver, launcher faked


class FakeLaunch:
    """rl_batch.launch faked: each call recorded; ``up`` writes the job.json a kept machine would; a role in FAIL
    fails."""
    FAIL: set = set()

    def __init__(self):
        self.calls = []

    def __call__(self, args, logfile, wait=True):
        self.calls.append(list(args))
        code = 0
        if args[0] == "up":
            out = Path(args[args.index("--out") + 1])
            role = out.name
            out.mkdir(parents=True, exist_ok=True)
            code = 2 if role in self.FAIL else 0
            vast = "--provider" in args
            rec = {"name": f"dennis-{role}-abc123", "instance_id": "9001" if vast else "computeinstance-x",
                   "host": "root@ssh5.vast.ai" if vast else "md@203.0.113.9", "kept": code == 0, "hours": 4,
                   "times": {"create_start": rb.utc()}, "code": {"commit": "c0de"},
                   "vast": {"ssh": {"host": "root@ssh5.vast.ai", "port": 31000}, "ends_utc": rb.utc()}}
            (out / "job.json").write_text(json.dumps(rec))
        if wait:
            return code
        return type("P", (), {"poll": lambda self: code, "returncode": code, "wait": lambda self: code})()


def _batch(tmp_path, repo, tmp=None) -> tuple:
    roster = rb.load_roster(_roster_file(tmp_path))
    plan = rb.build_plan(roster)
    batch = rb.Batch(tmp_path / "batch")
    batch.data.update(roster=rb.dataclasses.asdict(roster), settings=rb.dataclasses.asdict(rb.Settings()),
                      plan={"runs": [rb.dataclasses.asdict(r) for r in plan.runs], "probes": plan.probes,
                            "judges": plan.judges, "rehearsal": False})
    batch.save()
    return batch, plan


def test_the_machines_come_up_together_with_their_names_tags_and_servers(repo, tmp_path, monkeypatch):
    fake = FakeLaunch()
    monkeypatch.setattr(rb, "launch", fake)
    monkeypatch.setattr(rb.time, "sleep", lambda s: None)
    batch, plan = _batch(tmp_path, repo)
    found = {"judge_gpu": "H200 NVL", "lease_hours": 4.0}
    rb.machines_up(batch, plan, rb.Settings(), found, rl_hours=20)
    ups = {Path(c[c.index("--out") + 1]).name: c for c in fake.calls}
    assert set(ups) == {"rl", "probe", "judges"}
    rl = ups["rl"]
    for flag, value in (("--platform", "cpu-d3"), ("--preset", "32vcpu-128gb"), ("--image-family",
                                                                                 "ubuntu24.04-driverless"),
                        ("--hours", "20"), ("--name", "dennis-rl")):
        assert rl[rl.index(flag) + 1] == value
    assert "--on-demand" in rl and "--root-job" in rl and "--secret" not in rl  # keys go with the job, later
    probe, judges = ups["probe"], ups["judges"]
    assert probe[probe.index("--gpu") + 1] == "B200" and probe[probe.index("--gpus") + 1] == "4"
    assert probe[probe.index("--hf") + 1] == rb.READER and probe[probe.index("--name") + 1] == "dennis-probe-srv"
    assert judges[judges.index("--gpu") + 1] == "H200 NVL" and judges[judges.index("--name") + 1] == "dennis-judges"
    for up in (probe, judges):
        tags = [up[i + 1] for i, a in enumerate(up) if a == "--tag"]
        assert tags == ["owner=dennis", "experiment=probes-v2-rl", f"step={'probe-server' if up is probe else 'judges'}"]
        assert up[up.index("--hours") + 1] == "4" and "--secret" not in up and "--health" in up
        assert len(up[up.index("--name") + 1]) + 7 <= 24  # hub-gpu's names: the launcher adds -XXXXXX
    assert probe[-1].startswith("check() {") and judges[-1] == rb.JUDGES_COMMAND
    assert (probe[probe.index("--max-price") + 1], judges[judges.index("--max-price") + 1]) == ("60", "30")
    assert all(batch.machines[r]["kept"] for r in ("rl", "probe", "judges"))
    assert batch.machines["probe"]["port"] == 31000


def test_a_machine_that_fails_to_come_up_fails_the_start(repo, tmp_path, monkeypatch):
    fake = FakeLaunch()
    monkeypatch.setattr(FakeLaunch, "FAIL", {"judges"})
    monkeypatch.setattr(rb, "launch", fake)
    monkeypatch.setattr(rb.time, "sleep", lambda s: None)
    batch, plan = _batch(tmp_path, repo)
    with pytest.raises(nj.InfraError, match="up failed for judges"):
        rb.machines_up(batch, plan, rb.Settings(), {"judge_gpu": "H200", "lease_hours": 4.0}, 20)


def test_the_end_ends_every_machine_and_says_so(repo, tmp_path, monkeypatch):
    fake = FakeLaunch()
    monkeypatch.setattr(rb, "launch", fake)
    monkeypatch.setattr(rb.time, "sleep", lambda s: None)
    batch, plan = _batch(tmp_path, repo)
    rb.machines_up(batch, plan, rb.Settings(), {"judge_gpu": "H200", "lease_hours": 4.0}, 20)
    fake.calls.clear()
    assert rb.end_all(batch, None, plan, "test") is True
    downs = sorted((c[c.index("--name") + 1], "--provider" in c) for c in fake.calls if c[0] == "down")
    assert downs == [("dennis-judges-abc123", True), ("dennis-probe-abc123", True), ("dennis-rl-abc123", False)]
    assert batch.phase == "ended" and all(m.get("ended") for m in batch.machines.values())
    monkeypatch.setattr(rb, "launch", lambda args, logfile, wait=True: 2)  # a down that can't verify
    for m in batch.machines.values():
        m.pop("ended")
    assert rb.end_all(batch, None, plan, "again") is False and batch.phase == "end-failed"


def _watching(monkeypatch, batch, statuses, exits, raise_first=False):
    """watch's world faked: the supervisor's status and exit code per round, every periodic action recorded."""
    seen = {"sync": 0, "health": 0, "lease": 0, "start_job": []}
    statuses, exits = iter(statuses), iter(exits)
    state = {"raised": not raise_first}

    def vm_status(ops):
        if not state["raised"]:
            state["raised"] = True
            raise nj.InfraError("ssh hiccup")
        return next(statuses)

    monkeypatch.setattr(rb, "vm_status", vm_status)
    monkeypatch.setattr(rb, "job_exit", lambda ops: next(exits))
    monkeypatch.setattr(rb, "sync", lambda ops, runs: seen.__setitem__("sync", seen["sync"] + 1) or {})
    monkeypatch.setattr(rb, "health_round", lambda *a: seen.__setitem__("health", seen["health"] + 1))
    monkeypatch.setattr(rb, "extend_leases", lambda ops: seen.__setitem__("lease", seen["lease"] + 1))
    monkeypatch.setattr(rb, "start_job", lambda ops, plan, k, check_nobody=False: seen["start_job"].append(k)
                        or batch.data.setdefault("job", {}).update(starts=k))
    monkeypatch.setattr(rb.time, "sleep", lambda s: None)
    return seen


def test_the_watch_survives_a_bad_round_and_ends_with_the_supervisor(repo, tmp_path, monkeypatch):
    batch, plan = _batch(tmp_path, repo)
    running = {"phase": "running", "runs": {}}
    seen = _watching(monkeypatch, batch, [running, running, {"phase": "done", "runs": {}}], [None, None, 0],
                     raise_first=True)
    assert rb.watch(batch, object(), plan) == "done"
    assert any("watch round failed (1 in a row): InfraError: ssh hiccup" in e["what"] for e in batch.data["events"])
    assert seen["health"] == 1 and seen["lease"] == 1  # once per their own period, not per round


def test_the_watch_starts_a_supervisor_that_died_with_runs_unfinished(repo, tmp_path, monkeypatch):
    batch, plan = _batch(tmp_path, repo)
    batch.data["job"] = {"step": rb.JOB_STEP, "starts": 1}
    seen = _watching(monkeypatch, batch, [{"phase": "supervisor-stopped"}, {"phase": "done"}], [143, 0])
    assert rb.watch(batch, object(), plan) == "done" and seen["start_job"] == [2]
    seen = _watching(monkeypatch, batch, [{"phase": "failed"}], [1])
    assert rb.watch(batch, object(), plan) == "failed" and seen["start_job"] == []  # its runs failed: over


def test_the_watch_gives_up_on_an_rl_machine_it_cannot_reach(repo, tmp_path, monkeypatch):
    batch, plan = _batch(tmp_path, repo)
    clock = iter(range(0, 10**6, 600))  # ten minutes a round
    monkeypatch.setattr(rb.time, "time", lambda: float(next(clock)))
    _watching(monkeypatch, batch, [None] * 1000, [None] * 1000)
    assert rb.watch(batch, object(), plan) == "rl-gone"
    assert any("unreachable for 30 min" in e["what"] for e in batch.data["events"])


def test_a_stall_the_health_checks_cannot_see_is_reported_once_per_stretch(repo, tmp_path):
    """No run finishing a step or an eval for stall_minutes (45) is said once, with what each running run waits for;
    progress starts the clock again."""
    batch, plan = _batch(tmp_path, repo)
    wait = "[probe_iid] no answer from the probe server for 180s (HTTP 503: OutOfMemoryError); still retrying"

    def runs(a_steps):
        return {"phase": "running", "runs": {"a": {"state": "running", "steps": a_steps, "evals": 1, "last_warning": wait},
                                             "b": {"state": "done", "steps": 90, "evals": 31}}}

    assert not rb.stalled(batch, runs(3), 0.0) and not rb.stalled(batch, runs(3), 44 * 60.0)
    assert rb.stalled(batch, runs(3), 46 * 60.0) and rb.stalled(batch, runs(3), 50 * 60.0)
    said = [e for e in batch.data["events"] if "no run has finished a step" in e["what"]]
    assert len(said) == 1 and said[0]["waiting"] == {"a": wait}
    assert not rb.stalled(batch, runs(4), 51 * 60.0)  # a step: the clock starts again
    assert not rb.stalled(batch, runs(4), 95 * 60.0) and rb.stalled(batch, runs(4), 97 * 60.0)
    assert len([e for e in batch.data["events"] if "no run has finished a step" in e["what"]]) == 2


def test_the_supervisor_shows_what_a_waiting_run_waits_for(sup_env):
    repo, script, behave = sup_env
    sup = _sup(repo, script, ["a"])
    d = repo / "data" / "runs" / "a"
    d.mkdir()
    (d / "run.log").write_text("step 3: sampling\n[12:00:01] ⚠️  q35_cot: attempt 7 failed (ConnectError: refused); "
                               "retrying\n[probe_iid] no answer from the probe server for 360s (URLError: refused); "
                               "still retrying\nother line\n")
    assert sup.last_warning(rb.RunState("a", [])).startswith("[probe_iid] no answer from the probe server for 360s")
    (d / "run.log").write_text("all fine\n")
    assert sup.last_warning(rb.RunState("a", [])) is None


def test_a_manual_request_waits_for_the_driver_that_holds_the_batch(repo, tmp_path):
    batch, plan = _batch(tmp_path, repo)
    args = type("A", (), {"out": str(batch.out)})()
    done = []
    with rb.batch_lock(batch.out):  # a driver watches
        assert rb.ask_or_do(args, "sync", lambda b: done.append(1) or 0) == 0
        assert not done and (batch.out / "requests" / "sync").exists()
    assert rb.requests(batch) == ["sync"] and not (batch.out / "requests" / "sync").exists()
    assert rb.ask_or_do(args, "sync", lambda b: done.append(1) or 0) == 0 and done == [1]  # no driver: done at once


# ---------------------------------------------------------------------- Nebius servers, the launcher faked (2026-10-28)

NEBIUS = {"probe_provider": "nebius", "judges_provider": "nebius"}


def test_a_nebius_server_comes_up_spot_on_a_vm_of_its_own_with_its_server(repo, tmp_path):
    """--probe-provider / --judges-provider nebius: up of a kept GPU VM (no vast.ai flag), spot under its role's bid,
    its server started and healthy: the probe server on 8× B200 as two reader copies (GPUs 0-3 and 4-7, behind the
    balancer at :8178, each port in its health check), the judges in Vlad's layout on GPUs 0-3 of 8× H200 (VLLM_GPUS);
    the RL machine kept out of us-north1 and the regions without CPU quota. vast.ai stays the default, unchanged."""
    plan = rb.build_plan(rb.load_roster(_roster_file(tmp_path)))
    s = rb.Settings(**NEBIUS, probe_regions="me-west1,us-central1", budget_file="/ledger.json")
    probe = rb.up_args("probe", s, tmp_path / "p", 8, plan)
    assert probe[:2] == ["up", "--platform"] and "--provider" not in probe and "--tag" not in probe
    for flag, value in (("--platform", "gpu-b200-sxm-a,gpu-b200-sxm"), ("--preset", "8gpu-160vcpu-1792gb"),
                        ("--disk-gib", "650"), ("--hours", "8"), ("--name", "dennis-probe-srv"),
                        ("--max-gpu-price", "4.5"), ("--regions", "me-west1,us-central1"), ("--hf", rb.READER),
                        ("--budget-file", "/ledger.json"), ("--health-timeout", "3600"), ("--wait-hours", "0.5")):
        assert probe[probe.index(flag) + 1] == value, flag
    assert probe[probe.index("--health") + 1] == " && ".join(
        f"curl -sf -m 20 http://127.0.0.1:{p}/v1/meta > /dev/null" for p in (8179, 8180, 8178))
    cmd = probe[-1]
    assert probe[-2] == "--" and "CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python experiments/inkling_probe_server.py" \
        in cmd and "CUDA_VISIBLE_DEVICES=4,5,6,7 " in cmd and "--port 8179" in cmd and "--port 8180" in cmd
    assert "scripts/load_balancer.py --listen-port 8178 --backend-ports 8179 8180 --listen-host 127.0.0.1" in cmd
    judges = rb.up_args("judges", s, tmp_path / "j", 8, plan)
    for flag, value in (("--platform", "gpu-h200-sxm"), ("--preset", "8gpu-128vcpu-1600gb"), ("--disk-gib", "250"),
                        ("--max-gpu-price", "2.5"), ("--health", rb.JUDGES_HEALTH), ("--name", "dennis-judges")):
        assert judges[judges.index(flag) + 1] == value, flag
    assert judges[-1] == f"VLLM_GPUS=0,1,2,3 {rb.JUDGES_COMMAND}" and "--regions" not in judges and "--hf" not in judges
    on_demand = rb.up_args("judges", rb.dataclasses.replace(s, judges_on_demand=True, judges_gpu_ids="4,5,6,7"),
                           tmp_path / "j", 8, plan)
    assert "--on-demand" in on_demand and "--max-gpu-price" not in on_demand
    assert on_demand[-1] == f"VLLM_GPUS=4,5,6,7 {rb.JUDGES_COMMAND}"
    rl = rb.rl_up_args(s, tmp_path / "rl", 8)
    assert rl[rl.index("--regions") + 1] == "eu-west2,uk-south1,uk-south2,eu-west1"
    cache = tmp_path / "cache"
    (cache / "flashinfer").mkdir(parents=True)  # no vllm/: only what is there goes
    cached = rb.up_args("judges", rb.dataclasses.replace(s, judges_cache=str(cache)), tmp_path / "j", 8, plan)
    pushes = [cached[i + 1] for i, a in enumerate(cached) if a == "--push"]
    assert pushes == [f"{cache.resolve() / 'flashinfer'}:~/.cache/flashinfer"]
    assert cached.index("--push") < cached.index("--")
    assert "--push" not in rb.up_args("probe", rb.dataclasses.replace(s, judges_cache=str(cache)), tmp_path / "p", 8,
                                      plan)
    rb.check_settings(rb.dataclasses.replace(s, judges_cache=str(cache)))
    for bad in (rb.Settings(judges_cache=str(cache)), rb.dataclasses.replace(s, judges_cache=str(tmp_path / "none"))):
        with pytest.raises(SystemExit, match="--judges-cache"):  # vast.ai judges, or nothing to push
            rb.check_settings(bad)
    vast = rb.Settings()  # the default: both on vast.ai, as before
    assert rb.up_args("probe", vast, tmp_path / "p", 8, plan, gpu="B200", lease=4)[-1] == rb.probe_server_command(
        plan.probes, False) and rb.up_args("judges", vast, tmp_path / "j", 8, plan, gpu="H200")[-1] == rb.JUDGES_COMMAND
    assert rb.probe_copies(vast) == 1 and rb.probe_copies(s) == 2
    one = rb.estimate(plan, 1)
    assert rb.estimate(plan, 2)["probe_server_hours"] == pytest.approx(one["probe_server_hours"] / 2, abs=0.1)
    for bad, error in (({"probe_copies": 3}, "split evenly"), ({"judges_gpu_ids": "0"}, "serves on 2 or 4"),
                       ({"probe_gpu_ids": "0,1,2,9"}, "distinct GPUs"), ({"judges_platform": "cpu-d3"}, "a GPU VM"),
                       ({"probe_provider": "aws"}, "one of vast, nebius")):
        with pytest.raises(SystemExit, match=error):
            rb.check_settings(rb.dataclasses.replace(s, **bad))
    rb.check_settings(rb.dataclasses.replace(s, probe_copies=1, probe_gpu_ids="0,1,2,3"))  # a copy on GPUs 0-3
    assert rb.probe_server_command(plan.probes, False, 1, [0, 1, 2, 3]).splitlines()[-1].startswith(
        "CUDA_VISIBLE_DEVICES=0,1,2,3 exec .venv/bin/python experiments/inkling_probe_server.py")


FAKE_PROBE_PY = r'''#!/bin/bash
# .venv/bin/python, faked: what each probe server process would be, recorded with its GPUs
echo "$CUDA_VISIBLE_DEVICES|$*" >> "$CALLS"
port=""; out=""; prev=""
for a in "$@"; do [ "$prev" = "--port" ] && port=$a; [ "$prev" = "--out" ] && out=$a; prev=$a; done
case "$1" in
  experiments/check_probe_server.py) echo '{"ok": true}' > "$out"; exit 0 ;;
  experiments/inkling_probe_server.py) echo $$ > "pid.$port"; echo "golden check passed; serving on :$port"
    if [ "$port" = "$DIE_PORT" ]; then sleep 1; echo "out of memory"; exit 5; fi; exec sleep 60 ;;
  scripts/load_balancer.py) echo $$ > pid.balancer; echo "[lb] listening"; exec sleep 60 ;;
esac
'''


def test_the_two_copy_probe_server_runs_both_copies_behind_the_balancer_and_stops_with_any(tmp_path):
    """The probe server's step with two copies, run in a local bash with a fake python: the pinned shas checked once
    before anything starts; (rehearsal) check_probe_server.py on each copy's GPUs, a report each; copy 0 on GPUs 0-3 at
    :8179, copy 1 on GPUs 4-7 at :8180, the balancer at :8178 before both, each line of theirs in the step's log; when
    one dies, the others are stopped and the step exits with its code (the health checks then restart it whole)."""
    md = tmp_path / "md"
    d = md / IID
    d.mkdir(parents=True)
    (d / "probe.npz").write_bytes(b"npz")
    (d / "meta.json").write_text("{}")
    _fake_bin(md / ".venv" / "bin", "python", FAKE_PROBE_PY.split("\n", 1)[1])
    calls = tmp_path / "calls"
    env = {**os.environ, "CALLS": str(calls), "DIE_PORT": "8180"}
    cmd = rb.probe_server_command({IID: rb.probe_sha(d)}, True, 2, list(range(8)))
    t0 = time.time()
    cp = subprocess.run(["bash", "-c", cmd], cwd=md, env=env, capture_output=True, text=True, timeout=60)
    assert cp.returncode == 5 and time.time() - t0 < 30, cp.stdout + cp.stderr
    lines = calls.read_text().splitlines()
    checks = sorted(line for line in lines if "check_probe_server.py" in line)
    assert [c.split("|")[0] for c in checks] == ["0,1,2,3", "4,5,6,7"]
    assert all(json.loads((md / "results" / f"check_probe_server_v2-{k}.json").read_text())["ok"] for k in (0, 1))
    servers = {line.split("--port ")[1].split()[0]: line.split("|")[0] for line in lines if "inkling_probe" in line}
    assert servers == {"8179": "0,1,2,3", "8180": "4,5,6,7"}
    assert any("load_balancer.py --listen-port 8178 --backend-ports 8179 8180" in line for line in lines)
    assert "[copy 1 :8180] out of memory" in cp.stdout and "[balancer :8178] [lb] listening" in cp.stdout
    assert "a probe server process exited (5): stopping the others" in cp.stdout
    for f in ("pid.8179", "pid.balancer"):  # stopped with it
        assert _wait(lambda f=f: rb.pid_ended(int((md / f).read_text())), 10), f
    (d / "meta.json").write_text('{"x": 1}')  # not the pinned probe: refused before any of them starts
    calls.unlink()
    cp = subprocess.run(["bash", "-c", cmd], cwd=md, env=env, capture_output=True, text=True, timeout=60)
    assert cp.returncode == 1 and "NOT the pinned" in cp.stdout and not calls.exists()


@pytest.mark.skipif(shutil.which("setsid") is None, reason="needs setsid")
def test_setup_vllm_serves_on_the_gpus_vllm_gpus_names(vllm_box):
    """nvidia-smi ignores CUDA_VISIBLE_DEVICES, so an 8-GPU VM was refused ("need exactly 2 or 4 GPUs, found 8", the
    B300 judges check, 2026-10-28). VLLM_GPUS (else a CUDA_VISIBLE_DEVICES of indices) picks the GPUs, in order: Qwen3
    on the first, the three Qwen3.5 on the next; each checked; every engine on its own; any other GPU left alone."""
    repo, env, calls = vllm_box
    fake = Path(env["PATH"].split(":")[0])
    _fake_bin(fake, "nvidia-smi", 'case "$*" in *query-gpu*) for i in 0 1 2 3 4 5 6 7; do echo "$i, NVIDIA H200, '
                                  '143771"; done ;; *) echo "| CUDA Version: 13.0 |" ;; esac')
    _fake_bin(fake, "vllm", 'case "$1" in --version) echo 0.32.0; exit 0 ;; esac\nport=""; prev=""\n'
                            'for a in "$@"; do [ "$prev" = "--port" ] && port=$a; prev=$a; done\n'
                            'echo "$port gpu=$CUDA_VISIBLE_DEVICES" >> "$FAKE_CALLS"\nexec sleep 300')

    def served(extra: dict, engines: int) -> dict:
        calls.unlink(missing_ok=True)
        proc = _setup_vllm(repo, {**env, **extra})
        try:
            assert _wait(lambda: calls.exists() and sum(" gpu=" in x for x in calls.read_text().splitlines())
                         >= engines, 40, 0.5), proc.stdout.read() if proc.poll() is not None else "still waiting"
            return dict(x.split(" gpu=") for x in calls.read_text().splitlines() if " gpu=" in x)
        finally:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.communicate(timeout=30)

    assert served({"VLLM_GPUS": "4,5,6,7"}, 4) == {"8001": "4", "8003": "5", "8004": "6", "8005": "7"}
    assert any(x.startswith("lb ") for x in calls.read_text().splitlines())  # and the balancer before :8003-8005
    assert served({"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, 4) == {"8001": "0", "8003": "1", "8004": "2", "8005": "3"}
    assert served({"VLLM_GPUS": "6,7"}, 2) == {"8001": "6", "8002": "7"}
    for extra, error in (({}, "need exactly 2 or 4 GPUs, found 8"), ({"VLLM_GPUS": "1,9"}, "lists no GPU 9"),
                         ({"VLLM_GPUS": "1,1,2,3"}, "names GPU 1 twice"), ({"VLLM_GPUS": "a"}, "not a comma list")):
        proc = _setup_vllm(repo, {**env, **extra})
        out = proc.communicate(timeout=60)[0]
        assert proc.returncode == 1 and error in out, out


NEBIUS_REGIONS = {"tenant": "tenant-t", "regions": {
    r: {"project": f"project-{r}", "subnet": f"vpcsubnet-{r}", "image_project": f"images-{r}"}
    for r in ("us-north1", "me-west1", "us-central1", "eu-west2", "uk-south1", "uk-south2", "eu-west1")}}


class NebiusWorld:
    """The launcher and the Nebius CLI faked together. Each ``up`` makes a VM of its own (a new name, address and
    instance id, in us-north1) and keeps it, unless its role still has failures in ``fail`` (then exit 3: preempted
    while it booted, nothing left); a ``down`` deletes it; ``state`` is what the CLI lists for a VM (RUNNING unless
    set); every call recorded; the clock is the test's."""

    def __init__(self):
        self.calls, self.vms, self.state, self.fail, self.n, self.wired = [], {}, {}, {}, 0, []

    def launch(self, args, logfile, wait=True):
        self.calls.append(list(args))
        code = 0
        if args[0] == "up":
            out = Path(args[args.index("--out") + 1])
            role, nebius = out.name.split("-r")[0], "--provider" not in args
            self.n += 1
            failing = self.fail.get(role, 0) > 0
            self.fail[role] = self.fail.get(role, 0) - 1
            name = f"dennis-{role}-{self.n:06x}"
            flag = lambda f: args[args.index(f) + 1] if f in args else None  # noqa: E731
            rec = {"name": name, "instance_id": f"computeinstance-{self.n}" if nebius else str(9000 + self.n),
                   "host": f"md@203.0.113.{self.n}" if nebius else "root@ssh5.vast.ai", "kept": not failing,
                   "hours": float(flag("--hours")), "times": {"create_start": rb.utc()}, "code": {"commit": "c0de"},
                   "region": "us-north1" if nebius else "Texas, US", "spot": nebius and "--on-demand" not in args,
                   "platform": (flag("--platform") or "").split(",")[0] or None,
                   "flags": {"platform": flag("--platform"), "preset": flag("--preset")},
                   "teardown_verified": True if failing else None,
                   "vast": {"ssh": {"host": "root@ssh5.vast.ai", "port": 31000}}}
            out.mkdir(parents=True, exist_ok=True)
            (out / "job.json").write_text(json.dumps(rec))
            code = 3 if failing else 0
            if not failing:
                self.vms[name] = rec
        elif args[0] == "down":
            self.vms.pop(args[args.index("--name") + 1], None)
        if wait:
            return code
        return type("P", (), {"poll": lambda self: code, "returncode": code, "wait": lambda self: code, "pid": None})()

    def nebius(self, binary, project):
        world = self

        class Client:
            def instances(self):
                return [nj.Instance(v["instance_id"], n, world.state.get(n, "RUNNING"), v["host"].split("@")[1])
                        for n, v in world.vms.items()]

            def stop_instance(self, iid):
                world.calls.append(["stop", iid])
                world.state.update({n: "STOPPED" for n, v in world.vms.items() if v["instance_id"] == iid})

            def estimate_hourly(self, platform, preset, spot):
                return 30.4

        return Client()


@pytest.fixture
def nebius_world(tmp_path, monkeypatch):
    """The launcher, the Nebius CLI, the tunnels and the servers' own health faked (``NebiusWorld``); a clock of the
    test's (time.sleep moves it); the regions file in NEBIUS_REGIONS_FILE. ``world.real`` keeps what was faked."""
    regions = tmp_path / "regions.json"
    regions.write_text(json.dumps(NEBIUS_REGIONS))
    monkeypatch.setenv("NEBIUS_REGIONS_FILE", str(regions))
    monkeypatch.setenv("NEBIUS_JOB_NAME_PREFIX", "dennis-")
    world = NebiusWorld()
    world.real = {"server_health": rb.server_health, "through_tunnels": rb.through_tunnels, "wire": rb.wire}
    monkeypatch.setattr(rb, "launch", world.launch)
    monkeypatch.setattr(rb.nj, "Nebius", world.nebius)
    monkeypatch.setattr(rb, "wire", lambda batch, ops, plan, roles=None, timeout=600: world.wired.append(list(roles)))
    monkeypatch.setattr(rb, "server_health", lambda ops, role, pinned: None)
    monkeypatch.setattr(rb, "through_tunnels", lambda ops, probes, roles: {r: None for r in roles})
    world.clock = {"t": 1_800_000_000.0}
    monkeypatch.setattr(rb.time, "time", lambda: world.clock["t"])
    monkeypatch.setattr(rb.time, "sleep", lambda s: world.clock.update(t=world.clock["t"] + s))
    return world


def _nebius_batch(tmp_path: Path, repo: Path, **settings) -> tuple:
    roster = rb.load_roster(_roster_file(tmp_path))
    plan = rb.build_plan(roster)
    s = rb.Settings(**{**NEBIUS, **settings})
    batch = rb.Batch(tmp_path / "batch")
    batch.data.update(roster=rb.dataclasses.asdict(roster), settings=rb.dataclasses.asdict(s), hours=8.0,
                      plan={"runs": [rb.dataclasses.asdict(r) for r in plan.runs], "probes": plan.probes,
                            "judges": plan.judges, "rehearsal": False})
    batch.save()
    return batch, plan, s


def _events(batch) -> list[str]:
    return [e["what"] for e in batch.data["events"]]


def test_a_nebius_server_whose_up_fails_at_the_start_gets_a_new_vm(repo, tmp_path, nebius_world):
    """A Nebius server VM preempted while it boots (2 of 2 8-GPU VMs on 2026-10-10) fails its up: the start tries a
    new VM after a backoff (1 min, then 2), up to max_replacements, instead of ending the batch; one that never comes
    up fails the start."""
    world = nebius_world
    world.fail = {"probe": 2}
    batch, plan, s = _nebius_batch(tmp_path, repo)
    rb.machines_up(batch, plan, s, {}, rl_hours=8)
    probe_ups = [Path(c[c.index("--out") + 1]).name for c in world.calls
                 if c[0] == "up" and "probe-srv" in c[c.index("--name") + 1]]
    assert probe_ups == ["probe", "probe-r1", "probe-r2"]
    m = batch.machines["probe"]
    assert m["kept"] and m["replacements"] == 2 and (m["provider"], m["project"]) == ("nebius", "project-us-north1")
    assert all(batch.machines[r]["kept"] and batch.machines[r]["provider"] == "nebius" for r in ("rl", "judges"))
    assert [e for e in _events(batch) if "a new VM in" in e] == [
        "probe: a new VM in 1 min (new VM 1 of at most 8)", "probe: a new VM in 2 min (new VM 2 of at most 8)"]
    assert not [c for c in world.calls if c[0] == "down"]  # each failed up deleted its own VM (verified)
    world.fail = {"probe": 99}
    batch, plan, s = _nebius_batch(tmp_path, repo, max_replacements=2)
    with pytest.raises(nj.InfraError, match="up failed for probe"):
        rb.machines_up(batch, plan, s, {}, rl_hours=8)


def test_a_preempted_nebius_server_gets_a_new_vm_every_time_up_to_its_cap(repo, tmp_path, nebius_world):
    """The health round finds a Nebius server's VM STOPPED (a spot preemption: on-preemption stop) by the Nebius CLI:
    it is ended (down, verified; the ledger settles it), a new VM is due after a backoff, its up runs on its own while
    the watch goes on, and once it has kept the VM the tunnel moves there. Again at the next preemption (not "at most
    once"), up to max_replacements; then no new VM, said loudly. The other server is never touched."""
    world = nebius_world
    batch, plan, s = _nebius_batch(tmp_path, repo, max_replacements=2, budget_file="/ledger.json")
    rb.machines_up(batch, plan, s, {}, 8)
    ops, bad_since = rb.Ops(batch), batch.data.setdefault("bad_since", {})
    judges = batch.machines["judges"]["name"]
    for k in (1, 2):
        old = batch.machines["probe"]["name"]
        world.state[old] = "STOPPED"
        rb.health_round(batch, ops, plan, ["probe", "judges"], bad_since)
        assert world.calls[-1] == ["down", "--name", old, "--budget-file", "/ledger.json"]
        assert batch.machines["probe"]["replacing"]["k"] == k and batch.data["retired"][-1]["name"] == old
        rb.replacements_round(batch, ops, plan)  # its backoff isn't over: nothing yet
        assert len(world.wired) == k - 1
        world.clock["t"] += rb.backoff_seconds(s, k) + 1
        rb.replacements_round(batch, ops, plan)  # its up started, on its own
        up = world.calls[-1]
        assert up[0] == "up" and Path(up[up.index("--out") + 1]).name == f"probe-r{k}"
        assert up[up.index("--hours") + 1] == "8"  # the hours left of the batch's cap, the RL machine's
        rb.replacements_round(batch, ops, plan)  # it exited: the new VM taken, its tunnel moved there
        m = batch.machines["probe"]
        assert m["name"] != old and m["kept"] and not m.get("replacing") and m["replacements"] == k
        assert world.wired[-1] == ["probe"] and len(world.wired) == k
    world.state[batch.machines["probe"]["name"]] = "STOPPED"
    rb.health_round(batch, ops, plan, ["probe", "judges"], bad_since)
    assert batch.machines["probe"].get("abandoned") and any("!!! probe: no new VM (2 new ones already)" in e
                                                             for e in _events(batch))
    rb.health_round(batch, ops, plan, ["probe", "judges"], bad_since)  # nothing more is tried
    assert [c[0] for c in world.calls].count("up") == 3 + 2 and judges in world.vms


def test_a_nebius_vm_stopped_at_its_own_hours_is_the_batchs_end_not_a_preemption(repo, tmp_path, nebius_world):
    world = nebius_world
    batch, plan, s = _nebius_batch(tmp_path, repo)
    rb.machines_up(batch, plan, s, {}, 8)
    world.clock["t"] += 8 * 3600
    world.state[batch.machines["probe"]["name"]] = "STOPPED"
    rb.health_round(batch, rb.Ops(batch), plan, ["probe", "judges"], {})
    assert batch.machines["probe"]["capped"] and not batch.machines["probe"].get("replacing")
    assert any("at its --hours cap: not replaced" in e for e in _events(batch))


def test_the_leases_of_nebius_servers_are_their_shutdown_timers(repo, tmp_path, nebius_world, monkeypatch):
    """Like the RL machine's: every Nebius VM's shutdown moved to a lease from now, never past its --hours (the
    ledger's reservation); a VM being replaced, or ended, has none."""
    world = nebius_world
    batch, plan, s = _nebius_batch(tmp_path, repo)
    rb.machines_up(batch, plan, s, {}, 8)
    sent = []
    monkeypatch.setattr(rb, "run_text", lambda remote, command, timeout=120, stdin_text=None: sent.append(
        (remote.host, command)) or (0, "0"))
    rb.extend_leases(rb.Ops(batch))
    assert sorted(h for h, _ in sent) == sorted(m["host"] for m in batch.machines.values())
    assert {c for _, c in sent} == {"sudo -n shutdown -h +240 > /dev/null 2>&1; echo $?"}
    assert all(m["shutdown_at"] for m in batch.machines.values())
    world.clock["t"] += 7.9 * 3600
    sent.clear()
    batch.machines["judges"]["ended"] = rb.utc()
    rb.extend_leases(rb.Ops(batch))
    assert {h for h, _ in sent} == {batch.machines[r]["host"] for r in ("rl", "probe")}
    assert {c for _, c in sent} == {"sudo -n shutdown -h +5 > /dev/null 2>&1; echo $?"}  # its cap: the reservation


def test_the_end_ends_every_nebius_vm_one_being_made_and_what_a_replacement_left(repo, tmp_path, nebius_world,
                                                                                   monkeypatch):
    """The end: an up still making a new VM is stopped first (SIGTERM: it deletes what it made), then every machine
    ended with the launcher's Nebius down and the ledger (the VM that up named too, and a replaced one whose end was
    not verified then), and CURRENT_BATCH no longer names the batch."""
    world = nebius_world
    batch, plan, s = _nebius_batch(tmp_path, repo, budget_file="/ledger.json")
    rb.machines_up(batch, plan, s, {}, 8)
    pointer = tmp_path / "gpu_jobs" / "CURRENT_BATCH"
    rb.claim_pointer(batch, pointer)
    assert pointer.read_text() == f"{batch.out.resolve()}\n"
    ops = rb.Ops(batch)
    batch.data["retired"] = [{"role": "probe", "provider": "nebius", "name": "dennis-probe-srv-old", "why": "STOPPED"},
                             {"role": "probe", "provider": "nebius", "name": "dennis-probe-srv-older", "ended": "x"}]
    pending = batch.out / "launch" / "probe-r1"
    pending.mkdir(parents=True)
    (pending / "job.json").write_text(json.dumps({"name": "dennis-probe-srv-new"}))  # its VM being made
    proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
    ops.procs["probe"] = proc
    batch.machines["probe"] = {"role": "probe", "provider": "nebius", "replacements": 1, "replacing": {
        "k": 1, "out": str(pending), "pid": proc.pid, "started": rb.utc(), "not_before": 0, "why": "STOPPED"}}
    monkeypatch.setattr(rb, "job_exit", lambda ops: 0)
    monkeypatch.setattr(rb, "sync", lambda ops, runs: {})
    monkeypatch.setattr(rb.nj, "fetch_rsync", lambda *a, **k: "ok")
    assert rb.end_all(batch, ops, plan, "test") is True
    assert proc.wait(timeout=10) == -signal.SIGTERM
    downs = [c for c in world.calls if c[0] == "down"]
    assert sorted(c[c.index("--name") + 1] for c in downs) == sorted(
        [batch.machines["rl"]["name"], batch.machines["judges"]["name"], "dennis-probe-srv-new", "dennis-probe-srv-old"])
    assert all("--provider" not in c and c[-2:] == ["--budget-file", "/ledger.json"] for c in downs)
    assert batch.phase == "ended" and not pointer.exists()


def test_the_watchdog_keeps_nebius_servers_up_replaces_a_preempted_one_and_ends_them_with_the_batch(
        repo, tmp_path, nebius_world, monkeypatch):
    """`watchdog` (the hourly scheduled job on CURRENT_BATCH's batch) with no driver: a round of the watch. It moves
    every Nebius VM's shutdown timer (the servers' too) and samples their spot price into their job.json (for the
    ledger); a preempted server VM is ended and its new one's up started on its own, then taken up by the next run;
    once the supervisor is done, every VM is ended and the pointer removed. A driver holding the batch: nothing."""
    world = nebius_world
    batch, plan, s = _nebius_batch(tmp_path, repo)
    rb.machines_up(batch, plan, s, {}, 8)
    batch.set_phase("running")
    pointer = tmp_path / "CURRENT_BATCH"
    rb.claim_pointer(batch, pointer)
    sent, state = [], {"exit": None, "status": {"phase": "running", "runs": {}}}
    monkeypatch.setattr(rb, "run_text", lambda remote, command, timeout=120, stdin_text=None: sent.append(
        (remote.host, command)) or (0, "0"))
    monkeypatch.setattr(rb, "vm_status", lambda ops: state["status"])
    monkeypatch.setattr(rb, "job_exit", lambda ops: state["exit"])
    monkeypatch.setattr(rb, "sync", lambda ops, runs: {})
    monkeypatch.setattr(rb.nj, "fetch_rsync", lambda *a, **k: "ok")
    watchdog = ["watchdog", "--out", str(batch.out)]
    assert rb.main(watchdog) == 0
    hosts = {m["host"] for m in rb.Batch(batch.out).machines.values()}
    assert {h for h, c in sent if c.startswith("sudo -n shutdown -h +")} == hosts
    for role in ("probe", "judges"):
        samples = json.loads((Path(batch.machines[role]["out"]) / "job.json").read_text())["price_samples"]
        assert [p for _, p in samples] == [30.4]
    with rb.batch_lock(batch.out):  # a driver watches: the watchdog leaves it alone
        sent.clear()
        assert rb.main(watchdog) == 0 and not sent
    old = batch.machines["probe"]["name"]
    world.state[old] = "STOPPED"
    world.clock["t"] += 3600
    assert rb.main(watchdog) == 0  # found preempted: ended, a new VM due
    assert ["down", "--name", old] in world.calls and rb.Batch(batch.out).machines["probe"]["replacing"]
    world.clock["t"] += 3600
    assert rb.main(watchdog) == 0  # its up started, on its own (this run ends before it does)
    assert Path(world.calls[-1][world.calls[-1].index("--out") + 1]).name == "probe-r1"
    world.clock["t"] += 3600
    assert rb.main(watchdog) == 0  # the next run finds it done: the new VM taken, the tunnel moved there
    new = rb.Batch(batch.out).machines["probe"]
    assert new["name"] != old and new["kept"] and world.wired == [["probe"]]
    state.update(exit=0, status={"phase": "done", "runs": {}})
    assert rb.main(watchdog) == 0
    downs = {c[c.index("--name") + 1] for c in world.calls if c[0] == "down"}
    assert downs >= {m["name"] for m in rb.Batch(batch.out).machines.values()} and not world.vms
    assert rb.Batch(batch.out).phase == "ended" and not pointer.exists()


def test_the_watchdog_ends_a_batch_whose_start_died(repo, tmp_path, nebius_world, monkeypatch):
    """A "starting" batch that no driver holds: its start died (hub restarted the sandbox); nothing can finish it, so
    the watchdog ends what it started, an up still running included."""
    world = nebius_world
    batch, plan, s = _nebius_batch(tmp_path, repo)
    rb.machines_up(batch, plan, s, {}, 8)
    batch.set_phase("starting")
    monkeypatch.setattr(rb, "job_exit", lambda ops: 0)
    monkeypatch.setattr(rb, "sync", lambda ops, runs: {})
    monkeypatch.setattr(rb.nj, "fetch_rsync", lambda *a, **k: "ok")
    assert rb.main(["watchdog", "--out", str(batch.out)]) == 0
    assert not world.vms and rb.Batch(batch.out).phase == "ended"
    assert any("its start died" in e for e in _events(rb.Batch(batch.out)))


def test_runs_from_pushes_each_runs_folder_and_refuses_one_without_a_save(repo, tmp_path, monkeypatch, capsys):
    """start --runs-from DIR: each roster run's folder DIR/<run_name>/ resumes from its newest resume state (logged),
    a finished run is taken as done; a run without a save is refused unless --allow-fresh (then it starts fresh,
    nothing pushed), a save of another run always; each folder pushed to the RL machine's data/runs/ and checked
    there before the supervisor starts; the estimate counts only the steps left."""
    plan = rb.build_plan(rb.load_roster(_roster_file(tmp_path)))
    rows, src = [r.row for r in plan.runs], tmp_path / "old-runs"

    def run(row, *steps, seed=None, done=False):
        d = src / row.run_name / "resume_states"
        d.mkdir(parents=True)
        for k in steps:
            (d / f"step_{k:06d}.json").write_text(json.dumps({"steps_done": k, "eval_done": done and k == steps[-1],
                                                               "config": {"run_name": row.run_name,
                                                                          "seed": row.seed if seed is None else seed}}))

    run(rows[0], 54, 63)
    run(rows[1], 81, 90, done=True)
    (src / rows[1].run_name / "final_checkpoint.txt").write_text("tinker://x/final\n")
    run(rows[2], 45)
    (src / rows[3].run_name).mkdir()  # a folder without a save; rows[4]: no folder
    run(rows[5], 9, seed=999)
    with pytest.raises(SystemExit) as e:
        rb.runs_from(plan, src)
    msg = str(e.value)
    assert f"{rows[3].run_name}: no resume_states/step_*.json in its folder" in msg and "--allow-fresh" in msg
    assert f"{rows[4].run_name}: no folder" in msg and "is another run's (its seed is 999)" in msg
    with pytest.raises(SystemExit, match="another run's"):
        rb.runs_from(plan, src, allow_fresh=True)
    shutil.rmtree(src / rows[5].run_name)
    run(rows[5], 9)
    found = rb.runs_from(plan, src, allow_fresh=True)
    assert found[rows[0].run_name] == {"from": str(src / rows[0].run_name), "resume_state": "step_000063.json",
                                       "steps_done": 63, "eval_done": False, "done": False}
    assert found[rows[1].run_name]["done"] and found[rows[4].run_name] == {"fresh": True, "why": "no folder"}
    out = capsys.readouterr().out
    assert f"{rows[0].run_name} resumes from step_000063.json (63 steps done)" in out and "starts fresh" in out
    pushed = []
    monkeypatch.setattr(rb.nj, "push_path", lambda remote, local, dest: pushed.append((local, dest)))
    monkeypatch.setattr(rb, "run_text", lambda remote, command, timeout=120, stdin_text=None: (
        0, "step_000054.json\nstep_000063.json\nstep_000081.json\nstep_000090.json\nstep_000045.json\nstep_000009.json"))
    batch = rb.Batch(tmp_path / "b")
    batch.machines["rl"] = {"host": "md@203.0.113.9", "out": str(tmp_path)}
    rb.push_runs(batch, rb.Ops(batch), found)
    assert pushed == [(str(src / r.run_name), f"data/runs/{r.run_name}") for r in (rows[0], rows[1], rows[2], rows[5])]
    done = {n: v["steps_done"] for n, v in found.items() if "steps_done" in v}
    # steps left 27, 0, 45, 90, 90, 81 (with their evals): 7.0 of the 11.3 probe-server hours from scratch
    assert rb.estimate(plan, 1)["probe_server_hours"] == 11.3 and rb.estimate(plan, 1, done)["probe_server_hours"] == 7.0
    monkeypatch.setattr(rb, "run_text", lambda remote, command, timeout=120, stdin_text=None: (0, ""))
    with pytest.raises(nj.InfraError, match="not on the RL machine"):
        rb.push_runs(batch, rb.Ops(batch), {rows[0].run_name: found[rows[0].run_name]})
    for extra, error in ((["--allow-fresh"], "give --runs-from"), (["--runs-from", str(tmp_path / "none")],
                                                                    "not a folder")):
        with pytest.raises(SystemExit, match=error):  # refused before anything is made
            rb.main(["start", "--roster", str(_roster_file(tmp_path)), "--out", str(tmp_path / "b2"), *extra])
    assert not (tmp_path / "b2").exists()


def test_a_judges_replacement_on_vast_asks_for_every_judge_gpu_name(repo, tmp_path, monkeypatch):
    """A new judges instance asks the launcher for every name of --judge-gpu, in order (it used to pass the first
    name alone whenever none had an offer at that moment, and the launcher then waited for it alone)."""
    fake = FakeLaunch()
    monkeypatch.setattr(rb, "launch", fake)
    monkeypatch.setattr(rb.time, "sleep", lambda s: None)
    monkeypatch.setattr(rb, "wire", lambda *a, **k: None)
    batch, plan = _batch(tmp_path, repo)
    rb.machines_up(batch, plan, rb.Settings(), {"judge_gpu": "H200", "lease_hours": 4.0}, 20)
    ops = rb.Ops(batch)
    ops.hub = None  # no offers asked here: the launcher asks for each name itself, also while it waits
    assert rb.restart_server(batch, ops, plan, "judges", new_instance=True)
    up = [c for c in fake.calls if c[0] == "up"][-1]
    assert up[up.index("--gpu") + 1] == "H200,H200 NVL" and Path(up[up.index("--out") + 1]).name == "judges-r1"
    assert ["down", "--provider", "vast", "--name", "dennis-judges-abc123", "--hub-gpu", "hub-gpu"] in fake.calls


def test_a_tunnel_down_once_is_repaired_only_if_still_down_a_round_later(repo, tmp_path, nebius_world, monkeypatch):
    """Healthy judges read "down through the tunnel (:8001: curl exit 28)" during eval bursts (2026-10-27): the checks
    through the tunnels now wait 60 s, and a tunnel is repaired only when two rounds in a row find it down."""
    world = nebius_world
    batch, plan, s = _nebius_batch(tmp_path, repo)
    rb.machines_up(batch, plan, s, {}, 8)
    batch.data["tunnel_pub"] = "ssh-ed25519 AAAAtunnel rl-batch x"
    down, repairs = {"judges": ":8001: curl exit 28"}, []
    monkeypatch.setattr(rb, "through_tunnels", lambda ops, probes, roles: {r: down.get(r) for r in roles})
    monkeypatch.setattr(rb, "authorize", lambda remote, line: repairs.append(("authorize", remote.host)))
    monkeypatch.setattr(rb, "tunnel_state", lambda rl, name: {"loop": True})
    monkeypatch.setattr(rb, "kick_tunnel", lambda rl, name: repairs.append(("kick", name)))
    ops, roles = rb.Ops(batch), ["probe", "judges"]
    rb.health_round(batch, ops, plan, roles, {})
    assert not repairs
    rb.health_round(batch, ops, plan, roles, {})
    assert repairs == [("authorize", batch.machines["judges"]["host"]), ("kick", "judges")]
    down.clear()
    rb.health_round(batch, ops, plan, roles, {})
    down["judges"] = ":8002: curl exit 28"
    rb.health_round(batch, ops, plan, roles, {})  # once again, after a good round: not yet
    assert len(repairs) == 2
    asked = []
    monkeypatch.setattr(rb, "curl_json", lambda remote, url, timeout=20: asked.append((url, timeout)) or (True, {}))
    world.real["through_tunnels"](ops, plan.probes, roles)
    assert [u for u, _ in asked] == ["http://127.0.0.1:8178/v1/meta", "http://127.0.0.1:8001/v1/models"]
    assert {t for _, t in asked} == {rb.TUNNEL_CHECK_SECONDS} == {60}


def test_the_probe_servers_health_on_its_machine_asks_each_copy_and_the_balancer(repo, tmp_path, nebius_world,
                                                                                monkeypatch):
    """Two copies behind the balancer: on its machine each copy's port and the balancer's must serve every pinned sha
    (a copy that lost one is down), through the tunnel the balancer's."""
    batch, plan, s = _nebius_batch(tmp_path, repo)
    rb.machines_up(batch, plan, s, {}, 8)
    every = {"probes": {sha: {} for sha in plan.probes.values()}}
    answers, asked = {8179: every, 8180: {"probes": {}}, 8178: every}, []

    def curl(remote, url, timeout=20):
        port = int(url.split(":")[2].split("/")[0])
        asked.append((remote.host, port))
        return True, answers[port]

    monkeypatch.setattr(rb, "curl_json", curl)
    host = batch.machines["probe"]["host"]
    why = nebius_world.real["server_health"](rb.Ops(batch), "probe", plan.probes)
    assert why.startswith("the server at :8180 doesn't serve") and asked == [(host, 8179), (host, 8180)]
    answers[8180] = every
    assert nebius_world.real["server_health"](rb.Ops(batch), "probe", plan.probes) is None
    assert asked[2:] == [(host, 8179), (host, 8180), (host, 8178)]


def test_the_rehearsal_saves_every_step_and_stops_a_nebius_server_vm_for_a_new_one(repo, tmp_path, nebius_world,
                                                                                  monkeypatch):
    """The rehearsal saves at every step (its kill-and-resume drill resumes from a save), and on Nebius stops a server
    VM as a preemption would: the health rounds must end it, make a new VM and move the tunnel there, while the runs
    wait (no failed attempt)."""
    reh = rb.rehearsal_roster(rb.load_roster(_roster_file(tmp_path)), None, None)
    assert reh.sets[-2:] == ["eval_every=1", "save_every=1"]
    world = nebius_world
    batch, plan, s = _nebius_batch(tmp_path, repo)
    rb.machines_up(batch, plan, s, {}, 8)
    old = dict(batch.machines["probe"])
    monkeypatch.setattr(rb, "vm_status", lambda ops: {"phase": "running", "runs": {"a": {"attempts": 0}}})
    items = {}
    rb.preemption_drill(batch, rb.Ops(batch), plan, lambda name, ok, **info: items.update({name: (ok, info)}))
    ((name, (ok, info)),) = items.items()
    assert ok and info["old"] == old["name"] and info["new"] == batch.machines["probe"]["name"] != old["name"]
    assert ["stop", old["instance"]] in world.calls and ["down", "--name", old["name"]] in world.calls
    assert world.wired == [["probe"]] and old["name"] not in world.vms


def test_the_kill_and_resume_drill_kills_a_run_while_it_runs_after_a_save(monkeypatch):
    """The drill restarts the first run once it runs with a save, and needs it back with --resume; a run that is no
    longer running (it ended before the drill, as in the 2026-10-27 rehearsal) is not killed, and the item fails."""
    plan = type("P", (), {"runs": [type("R", (), {"row": rb.Row("rehearsal-a", "c.json", 1, None)})()]})()
    clock = {"t": 0.0}
    monkeypatch.setattr(rb.time, "time", lambda: clock["t"])
    monkeypatch.setattr(rb.time, "sleep", lambda s: clock.update(t=clock["t"] + s))
    for running, want in ((True, True), (False, False)):
        st = {"state": "running" if running else "done", "modes": ["fresh"]}
        restarted, items = [], {}
        monkeypatch.setattr(rb, "vm_status", lambda ops, st=st: {"runs": {"rehearsal-a": st}})
        monkeypatch.setattr(rb, "run_text", lambda remote, command, timeout=120, stdin_text=None: (
            0, "step_000000.json\nstep_000001.json\n"))

        def control(ops, what, st=st, restarted=restarted):
            restarted.append(what)
            st["modes"].append("resume")

        monkeypatch.setattr(rb, "control", control)
        ops = type("O", (), {"rl": lambda self: "rl"})()
        rb.kill_and_resume(ops, plan, lambda name, ok, **info: items.update(ok=ok, **info))
        assert items["ok"] is want and restarted == (["restart-rehearsal-a"] if running else [])
        assert items["resumed_from"] == ("step_000001.json" if running else None)


def test_current_batch_names_the_batch_from_its_start_until_its_machines_are_gone(repo, tmp_path):
    """output/gpu_jobs/CURRENT_BATCH (the watchdog's scheduled job reads it): the batch's folder, absolute, written as
    it starts; another start is refused while it names a batch that has not ended; removed only by its own batch."""
    jobs = tmp_path / "gpu_jobs"
    pointer = jobs / "CURRENT_BATCH"
    a, b = rb.Batch(jobs / "a"), rb.Batch(jobs / "b")
    a.data["phase"] = "running"
    a.save()
    rb.claim_pointer(a, pointer)
    assert pointer.read_text() == f"{a.out.resolve()}\n" and a.data["pointer"] == str(pointer.resolve())
    with pytest.raises(SystemExit, match=r"has not ended \(running\)"):
        rb.check_pointer(pointer, b.out)
    with pytest.raises(SystemExit, match="has not ended"):  # start refuses before anything
        rb.main(["start", "--roster", str(_roster_file(tmp_path)), "--out", str(jobs / "c")])
    assert not (jobs / "c").exists()
    a.data["phase"] = "ended"
    a.save()
    b.save()
    rb.claim_pointer(b, pointer)
    rb.release_pointer(a)  # it names b now: left alone
    assert pointer.read_text() == f"{b.out.resolve()}\n"
    rb.release_pointer(b)
    assert not pointer.exists()
