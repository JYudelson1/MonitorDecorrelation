"""scripts/run_probe_pipeline.sh: valid bash; a dry run prints every stage's commands in order and creates nothing;
finished steps are skipped when a run is resumed, and a step rerun after a failure may replace its own outputs; stages
are selectable and bad arguments are refused."""

from __future__ import annotations

import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "run_probe_pipeline.sh")
ORDER = ["scripts/fetch_features.py", "-c", "experiments/fit_probes.py", "experiments/verify_fits.py",
         "experiments/compare_readers.py", "experiments/ood_refits.py", "experiments/learning_curves.py",
         "experiments/probe_analyses.py", "experiments/deploy_probes.py", "experiments/probe_report.py"]
# the scripts that refuse an existing output unless --overwrite (deploy and collect manage their folders themselves)
OVERWRITING = {"experiments/fit_probes.py", "experiments/compare_readers.py", "experiments/ood_refits.py",
               "experiments/learning_curves.py", "experiments/probe_analyses.py"}


def dry(tmp_path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", SCRIPT, "--dry-run", "--run", str(tmp_path / "run"), "--scratch",
                           str(tmp_path / "scratch"), "--python", "python3", "--docs", str(tmp_path / "doc.md"), *args],
                          capture_output=True, text=True, cwd=tmp_path, timeout=60)


def commands(out: str) -> list[list[str]]:
    """The printed commands, split into words, without the interpreter."""
    return [line[2:].split()[1:] for line in out.splitlines() if line.startswith("+ ")]


def test_script_is_valid_bash():
    cp = subprocess.run(["bash", "-n", SCRIPT], capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr


def test_dry_run_prints_every_stage_in_order_and_creates_nothing(tmp_path):
    cp = dry(tmp_path)
    assert cp.returncode == 0, cp.stderr
    cmds = commands(cp.stdout)
    first = [c[0] for c in cmds]
    assert sorted(set(first), key=first.index) == ORDER
    assert [first.count(s) for s in ORDER] == [4, 2, 4, 1, 2, 4, 2, 2, 2, 1]  # sets × readers, readouts, ...
    run, s = tmp_path / "run", tmp_path / "scratch"
    lines = [" ".join(c) for c in cmds]
    expected = [
        f"scripts/fetch_features.py hf://buckets/probemaximizer/inkling-probes-bulk/features/bf16 "
        f"{s}/features/followup/bf16 --keep followup,counts,answer_logp --workers 4",
        f"scripts/fetch_features.py hf://buckets/probemaxxing/inkling-probes/features/nvfp4 {s}/features/mean_router/"
        "nvfp4 --keep mean_action,mean_reasoning,counts,answer_logp,router_logits,router_topk,router_margin "
        "--workers 4",
        f"experiments/fit_probes.py --reader bf16 --features {s}/features/mean_router/bf16 --out {run}/probes/bf16 "
        "--readouts mean,router,router_all --skip-baselines --jobs 6 --overwrite",
        f"experiments/verify_fits.py --probes {run}/probes --out {run}/verify_fits "
        f"--check nvfp4/followup={s}/features/followup/nvfp4 --check bf16/followup={s}/features/followup/bf16 "
        f"--check nvfp4/mean,router,router_all={s}/features/mean_router/nvfp4 "
        f"--check bf16/mean,router,router_all={s}/features/mean_router/bf16",
        f"experiments/compare_readers.py --probes {run}/probes --features {s}/features/followup "
        f"--diagnostics {s}/diagnostics --out {run}/compare/followup --readouts followup --overwrite",
        f"experiments/ood_refits.py --probe {run}/probes/nvfp4/diverse_deception/without_reasoning/mean "
        f"--features {s}/features/mean_router/nvfp4 --out {run}/ood_refits/nvfp4/mean --refits 50 --overwrite",
        f"experiments/learning_curves.py --probes {run}/probes/bf16 --features {s}/features/followup/bf16 "
        f"--out {run}/learning_curves/bf16 --overwrite",
        f"experiments/probe_analyses.py dataset-stats --out {run}/analyses --overwrite",
        f"experiments/deploy_probes.py --probes {run}/probes/bf16 --features {s}/features/mean_router/nvfp4 "
        f"--diagnostics {s}/diagnostics/nvfp4 --serving-reader nvfp4 --readouts mean --out data/probes/Inkling-Small",
        f"experiments/probe_report.py collect --scratch {run} --compare followup={run}/compare/followup "
        f"--compare mean_router={run}/compare/mean_router --verify-fits {run}/verify_fits --analyses {run}/analyses "
        "--out data/probe_results",
    ]
    for line in expected:
        assert line in lines, line
    assert all((c[-1] == "--overwrite") == (c[0] in OVERWRITING) for c in cmds), lines
    assert f"# no {tmp_path}/doc.md: nothing to render" in cp.stdout
    assert not list(tmp_path.iterdir())


def test_resume_skips_finished_steps_and_stages_are_selectable(tmp_path):
    (tmp_path / "run" / ".done").mkdir(parents=True)
    (tmp_path / "run" / ".done" / "compare-followup").write_text("2026-10-12T00:00:00Z\n")
    (tmp_path / "doc.md").write_text("# doc\n")
    cp = dry(tmp_path, "--stages", "compare,report", "--features-of", f"followup={tmp_path}/old_features")
    assert cp.returncode == 0, cp.stderr
    assert "# done already: compare-followup" in cp.stdout
    cmds = commands(cp.stdout)
    assert [c[0] for c in cmds] == ["experiments/compare_readers.py", "experiments/probe_report.py",
                                    "experiments/probe_report.py"]
    assert cmds[0][cmds[0].index("--out") + 1].endswith("compare/mean_router")
    assert cmds[2][:2] == ["experiments/probe_report.py", "render"] and str(tmp_path / "doc.md") in cmds[2]
    cp = dry(tmp_path, "--stages", "fit", "--readers", "bf16", "--features-of", f"followup={tmp_path}/old_features")
    assert [c[c.index("--features") + 1] for c in commands(cp.stdout)] == [
        f"{tmp_path}/old_features/bf16", f"{tmp_path}/scratch/features/mean_router/bf16"]
    cp = dry(tmp_path, "--stages", "compare", "--readers", "bf16")
    assert "# compare needs two readers; skipped" in cp.stdout and not commands(cp.stdout)


def test_bad_arguments_are_refused(tmp_path):
    for args in (["--stages", "fit,nonsense"], ["--set", "nameonly"], ["--refits"], ["--bogus", "1"],
                 ["--set", "a=mean", "--stages", "analyses"]):
        cp = dry(tmp_path, *args)
        assert cp.returncode == 2, (args, cp.stdout)
        assert cp.stderr.startswith("run_probe_pipeline.sh: "), args
    assert not list(tmp_path.iterdir())
