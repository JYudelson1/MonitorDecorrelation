"""The pre-registration in the repo is the one the Inkling probes were fitted under.

docs/inkling_probes_preregistration.md is section 5b of the original probe plan, byte for byte. Its sha256 is the
``prereg_sha256`` that experiments/fit_probes.py writes into every fitted probe's meta.json.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "docs" / "inkling_probes_preregistration.md"
PREREG_SHA256 = "58eb57820b6cc88b193519ca83ab3ff2268a3f0805778ba8e273e87f2b0aaff6"


def test_preregistration_file_has_the_recorded_hash():
    assert hashlib.sha256(PREREG.read_bytes()).hexdigest() == PREREG_SHA256


def test_fit_probes_hashes_the_preregistration_file():
    spec = importlib.util.spec_from_file_location("_fit_probes", REPO / "experiments" / "fit_probes.py")
    fit_probes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fit_probes)
    assert fit_probes.prereg_sha() == PREREG_SHA256


def test_deployed_inkling_probes_record_this_preregistration():
    metas = sorted((REPO / "data" / "probes" / "Inkling-Small").glob("*/*/*/meta.json"))
    assert metas
    for path in metas:
        assert json.loads(path.read_text())["prereg_sha256"] == PREREG_SHA256, path
