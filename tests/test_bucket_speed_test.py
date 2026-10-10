"""scripts/bucket_speed_test.py on a local store with tiny files: it measures every phase, reads two files back
and leaves nothing behind."""

from __future__ import annotations

import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_speed_test_measures_each_phase_and_cleans_up(tmp_path):
    store, ram, out = tmp_path / "store", tmp_path / "ram", tmp_path / "out" / "speed.json"
    store.mkdir()
    cp = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "bucket_speed_test.py"),
                         "--store", str(store), "--out", str(out), "--file-mb", "0.2", "--files", "3",
                         "--seconds", "30", "--parallel", "2", "--ram", str(ram)],
                        capture_output=True, text=True, cwd=REPO, timeout=300)
    assert cp.returncode == 0, cp.stderr[-3000:]
    summary = json.loads(out.read_text())
    assert [p["phase"] for p in summary["phases"]] == ["single", "parallel2"]
    assert [p["files"] for p in summary["phases"]] == [3, 6]
    assert all(p["mb_per_s"] > 0 for p in summary["phases"])
    assert summary["download"]["files"] == 2 and summary["leftovers"] == 0
    assert not list(store.rglob("*.bin")) and not list(store.rglob("*.done")) and not list(ram.iterdir())
