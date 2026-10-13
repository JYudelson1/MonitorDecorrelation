"""Collect the Inkling probe results into the repo, and render their tables into markdown files.

    uv run python experiments/probe_report.py collect --scratch RESULTS --analyses RESULTS/analyses
    uv run python experiments/probe_report.py render docs/INKLING_PROBES.md
    uv run python experiments/probe_report.py check docs/INKLING_PROBES.md

The probes are fitted and analysed in a scratch folder outside the repo (``--scratch``). ``collect`` copies the small
result files into the repo, so the numbers travel with the code; ``render`` writes the docs' tables from them, so a
table can't drift from the results it reports.

**collect** rebuilds these folders of ``--out`` (``data/probe_results/``), never with ``scores.jsonl`` (per episode,
large):
- ``fitted/<reader>/<dataset>/<view>/<readout>/{probe.npz,meta.json,metrics.json,cv_grid.csv}`` and
  ``fitted/<reader>/summary.json`` (experiments/fit_probes.py). Each IID probe also gets ``hack_counts.json``, counted
  from its ``scores.jsonl``: the test and the cross-fitted hacks caught at t₁%, and the tasks with hacks whose hacks
  were all caught.
- ``compare/<name>/{decision.json,report.md}`` (experiments/compare_readers.py; ``--compare NAME=DIR``);
- ``ood_refits/<reader>/<readout>/{summary.json,refits.csv,by_layer.csv}`` (experiments/ood_refits.py);
- ``learning_curves/<reader>/{summary.json,iid.csv,ood.csv,ood_sources.csv}`` (experiments/learning_curves.py);
- ``verify_fits/<run>/{summary.json,report.md}`` (experiments/verify_fits.py) and ``analyses/<run>/*.json``
  (experiments/probe_analyses.py): every ``<timestamp>/`` run folder of the folder given (or the folder itself, if it
  holds the files). A table takes each probe's newest check, and each analysis from its newest run.

Other files in ``--out`` (e.g. a hand-written ``PROVENANCE.md``) are kept as they are. Machine paths in the JSON
and markdown files are rewritten: a path into a collected folder becomes its path relative to ``--out``, any other
path its last component. ``MANIFEST.json`` lists every collected file with its source (relative to the folder
holding ``--scratch``) and sha256, and the code version that collected them (``code``, provenance.code_version).

**render FILE** replaces what is between each pair of markers

    <!-- probe_report:begin NAME -->
    <!-- probe_report:end NAME -->

with a table made from ``data/probe_results/`` (``deployed_monitors``: from ``data/probes/Inkling-Small/``). A table
whose inputs are missing becomes one "(not available: ...)" line; an unknown NAME is an error. The NAMEs are the keys
of ``TABLES``.

**check FILE** exits 1 if render would change FILE.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import defaultdict
from collections.abc import Callable

READERS = ("nvfp4", "bf16")
READOUTS = ("followup", "mean", "router", "router_all")
FALLBACK = "followup_prev"  # the readout rule's alternative to followup: the token before "Yes"
VIEWS = ("with_reasoning", "without_reasoning")
DATASETS = ("terminal_prerl", "diverse_deception")
PROBE_FILES = ("probe.npz", "meta.json", "metrics.json", "cv_grid.csv")
COMPARE_FILES = ("decision.json", "report.md")
OOD_REFIT_FILES = ("summary.json", "refits.csv", "by_layer.csv")
CURVE_FILES = ("summary.json", "iid.csv", "ood.csv", "ood_sources.csv")
VERIFY_FILES = ("summary.json", "report.md")
DEFAULT_COMPARE = (("followup", "compare"), ("mean_router", "compare_mean_router"))
TEXT = (".json", ".md", ".csv")
OWNED = ("fitted", "compare", "ood_refits", "learning_curves", "verify_fits", "analyses", "MANIFEST.json")
CONFIDENCE = 0.975  # the one-sided Clopper–Pearson bounds of test_set_bounds

# ------------------------------------------------------------------------------------------------ collect

# A path token, not inside a word, a number, a path or a URL: ~/..., ./..., ../..., an absolute path of two or more
# components (so "100%/0.37%" is no path), or a bare scratch/... path.
CHARS = r"[^\s'\"`,;()\[\]{}<>|]"
PATH = re.compile(rf"(?<![\w.~/:%-])(?:(?:~|\.\.?)/{CHARS}+|/[^\s'\"`,;()\[\]{{}}<>|/]+/{CHARS}+)"
                  rf"|(?<![\w./:%-])scratch/{CHARS}+")


def rel_join(*parts: str) -> str:
    return "/".join(p.replace(os.sep, "/") for p in parts if p and p != ".")


class Rewriter:
    """Rewrites machine paths: a path into a collected folder becomes its path in the collected tree, any other path
    its last component (more of it if JSON keys would clash). ``folders``: (source folder, its collected path)."""

    def __init__(self, folders: list[tuple[str, str]]):
        self.folders = sorted(((os.path.realpath(src), dest) for src, dest in folders), key=lambda f: -len(f[0]))

    def path(self, token: str, tail: int = 1) -> str:
        norm = os.path.normpath(os.path.expanduser(token))
        if os.path.isabs(norm):
            for cand in (norm, os.path.realpath(norm)):
                for src, dest in self.folders:
                    if cand == src or cand.startswith(src + os.sep):
                        return rel_join(dest, os.path.relpath(cand, src))
        # relative (e.g. ../scratch/probes/bf16) or from another machine: match the source folders' last components
        parts = [p for p in norm.split(os.sep) if p not in ("", ".", "..")]
        for i in range(len(parts)):
            hits = [(j, dest) for src, dest in self.folders for j in range(2, len(parts) - i + 1)
                    if parts[i:i + j] == src.split(os.sep)[-j:]]
            if hits:
                j, dest = max(hits)
                return rel_join(dest, *parts[i + j:])
        return "/".join(parts[-tail:]) if parts else token

    def text(self, s: str, tail: int = 1) -> str:
        def sub(m: re.Match) -> str:
            token = m.group(0)
            core = token.rstrip(".:!?")
            return self.path(core, tail) + token[len(core):] if core else token

        return PATH.sub(sub, s)

    def json(self, x):
        if isinstance(x, str):
            return self.text(x)
        if isinstance(x, list):
            return [self.json(v) for v in x]
        if not isinstance(x, dict):
            return x
        keys, tail = {k: self.text(k) for k in x}, 1
        while len(set(keys.values())) < len(keys):
            tail += 1
            if tail > 32:
                raise ValueError(f"keys {sorted(x)} can't be told apart once their paths are rewritten")
            by_new = defaultdict(list)
            for k, new in keys.items():
                by_new[new].append(k)
            keys.update({k: self.text(k, tail) for group in by_new.values() if len(group) > 1 for k in group})
        return {keys[k]: self.json(v) for k, v in x.items()}


def run_folders(folder: str | None, pattern: str) -> list[str]:
    """``[folder]`` if it holds files matching ``pattern``, else its subfolders that do, one per run, oldest first
    (by name: ``<timestamp>/``)."""
    if not folder or not os.path.isdir(folder):
        return []
    if glob.glob(os.path.join(folder, pattern)):
        return [folder]
    return [os.path.join(folder, d) for d in sorted(os.listdir(folder)) if glob.glob(os.path.join(folder, d, pattern))]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hack_counts(scores: str, threshold: float) -> dict:
    """The ``used_verifier`` hacks caught at ``threshold`` (logit > threshold, as fit_probes counts them) on the test
    split and cross-fitted over every split, and the tasks with hacks whose hacks were all caught."""
    by_task: dict[str, dict] = {"test": defaultdict(list), "crossfit": defaultdict(list)}
    with open(scores) as fh:
        for line in fh:
            row = json.loads(line)
            if not row.get("used_verifier"):
                continue
            by_task["crossfit"][row["task_id"]].append(row["crossfit_logit"] > threshold)
            if row["split"] == "test":
                by_task["test"][row["task_id"]].append(row["test_logit"] > threshold)
    out: dict = {"threshold_1": threshold}
    for name, tasks in by_task.items():
        flags = [f for v in tasks.values() for f in v]
        out[name] = {"hacks": len(flags), "caught": sum(flags), "tasks": len(tasks),
                     "tasks_caught": sum(all(v) for v in tasks.values())}
    return out


def copy(src: str, dest: str, rewriter: Rewriter) -> bool:
    """Copy ``src`` to ``dest``, machine paths rewritten in text files; True if that changed the content."""
    if not src.endswith(TEXT):
        shutil.copyfile(src, dest)
        return False
    with open(src, encoding="utf-8", newline="") as fh:
        raw = fh.read()
    if src.endswith(".json"):
        data = json.loads(raw)
        new = rewriter.json(data)
        text = raw if json.dumps(new) == json.dumps(data) else \
            json.dumps(new, indent=1) + ("\n" if raw.endswith("\n") else "")
    else:
        text = rewriter.text(raw)
    with open(dest, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return text != raw


def source_label(path: str, strip: str, sources: dict[str, str]) -> str:
    """``path`` relative to ``strip``; else to its source folder's parent; else its name."""
    real = os.path.realpath(path)
    if real.startswith(strip.rstrip(os.sep) + os.sep):
        return rel_join(os.path.relpath(real, strip))
    for folder in sources.values():
        top = os.path.realpath(folder)
        if real.startswith(top + os.sep):
            return rel_join(os.path.basename(top), os.path.relpath(real, top))
    return os.path.basename(real)


def collect(args) -> int:
    from monitordecorrelation.provenance import code_version  # here: render and check never import the package

    probes = args.probes or os.path.join(args.scratch, "probes")
    readers = sorted(d for d in os.listdir(probes) if os.path.isdir(os.path.join(probes, d))) \
        if os.path.isdir(probes) else []
    if not readers:
        raise SystemExit(f"collect: no fitted probes in {probes} (fit_probes.py's --out, one folder per reader)")
    sources = {"fitted": probes}
    for spec in args.compare or [f"{n}={os.path.join(args.scratch, d)}" for n, d in DEFAULT_COMPARE]:
        name, sep, folder = spec.partition("=")
        if not (name and sep and folder):
            raise SystemExit(f"collect: --compare {spec!r}: want NAME=DIR")
        sources[f"compare/{name}"] = folder
    sources["ood_refits"] = args.ood_refits or os.path.join(args.scratch, "ood_refits")
    sources["learning_curves"] = args.learning_curves or os.path.join(args.scratch, "learning_curves")
    for run in run_folders(args.verify_fits or os.path.join(args.scratch, "verify_fits"), "summary.json"):
        sources[f"verify_fits/{os.path.basename(run)}"] = run
    for run in run_folders(args.analyses, "*.json"):
        sources[f"analyses/{os.path.basename(run)}"] = run
    sources = {dest: folder for dest, folder in sources.items() if folder and os.path.isdir(folder)}

    files: list[tuple[str, str]] = []  # (source file, collected path)
    counted: list[tuple[str, str]] = []  # (fitted IID probe folder, collected path)

    def add(dest: str, rel: str) -> None:
        if os.path.isfile(os.path.join(sources[dest], rel)):
            files.append((os.path.join(sources[dest], rel), rel_join(dest, rel)))

    for reader in readers:
        add("fitted", f"{reader}/summary.json")
        for meta in sorted(glob.glob(os.path.join(probes, reader, "*", "*", "*", "meta.json"))):
            rel = rel_join(os.path.relpath(os.path.dirname(meta), probes))
            for f in PROBE_FILES:
                add("fitted", f"{rel}/{f}")
            if rel.split("/")[1] == "terminal_prerl" and os.path.isfile(os.path.join(probes, rel, "scores.jsonl")):
                counted.append((os.path.join(probes, rel), f"fitted/{rel}"))
    for dest, folder in sources.items():
        if dest.startswith(("compare/", "verify_fits/")):
            for f in COMPARE_FILES if dest.startswith("compare/") else VERIFY_FILES:
                add(dest, f)
        elif dest in ("ood_refits", "learning_curves"):
            depth = ("*", "*", "") if dest == "ood_refits" else ("*", "")
            for sub in sorted(glob.glob(os.path.join(folder, *depth))):
                for f in OOD_REFIT_FILES if dest == "ood_refits" else CURVE_FILES:
                    add(dest, rel_join(os.path.relpath(sub, folder), f))
        elif dest.startswith("analyses/"):
            for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
                add(dest, os.path.basename(path))

    out = os.path.normpath(args.out)
    tmp, old = out + ".tmp", out + ".old"
    for d in (tmp, old):
        shutil.rmtree(d, ignore_errors=True)
    rewriter = Rewriter([(folder, dest) for dest, folder in sources.items()])
    strip = os.path.realpath(args.strip or os.path.dirname(os.path.abspath(args.scratch)))
    manifest: dict[str, dict] = {}
    for src, rel in files:
        dest = os.path.join(tmp, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        entry = {"source": source_label(src, strip, sources)}
        if copy(src, dest, rewriter):
            entry["source_sha256"] = sha256(src)
        manifest[rel] = {**entry, "sha256": sha256(dest)}
    for folder, rel in counted:
        with open(os.path.join(folder, "meta.json")) as fh:
            threshold = json.load(fh)["threshold_1"]
        dest = os.path.join(tmp, rel, "hack_counts.json")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w") as fh:
            json.dump(hack_counts(os.path.join(folder, "scores.jsonl"), threshold), fh, indent=1)
            fh.write("\n")
        manifest[f"{rel}/hack_counts.json"] = {
            "derived_from": [source_label(os.path.join(folder, f), strip, sources)
                             for f in ("scores.jsonl", "meta.json")],
            "sha256": sha256(dest)}
    given = os.path.abspath(args.strip or os.path.dirname(os.path.abspath(args.scratch)))
    marks = {m for m in (strip, given) if m != os.sep} | {"../scratch/"}
    leaks = [rel for rel in manifest if rel.endswith(TEXT)
             and any(m in open(os.path.join(tmp, rel), encoding="utf-8").read() for m in marks)]
    if leaks:
        shutil.rmtree(tmp)
        raise SystemExit(f"collect: machine paths survived the rewriting in {', '.join(leaks)}; nothing written")
    with open(os.path.join(tmp, "MANIFEST.json"), "w") as fh:
        json.dump({"written_by": "experiments/probe_report.py collect", "code": code_version(), "files": manifest}, fh,
                  indent=1, sort_keys=True)
        fh.write("\n")
    kept = sorted(n for n in os.listdir(out) if n not in OWNED) if os.path.isdir(out) else []
    for name in kept:  # not collect's: carried over as they are
        src = os.path.join(out, name)
        (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, os.path.join(tmp, name))
    if os.path.isdir(out):
        os.replace(out, old)
    os.replace(tmp, out)
    shutil.rmtree(old, ignore_errors=True)
    by_top: dict[str, int] = defaultdict(int)
    for rel in manifest:
        by_top[rel.split("/")[0]] += 1
    print(f"[collect] {out}: " + ", ".join(f"{k} {n} files" for k, n in sorted(by_top.items())), flush=True)
    for name in ("compare", "ood_refits", "learning_curves", "verify_fits", "analyses"):
        if name not in by_top:
            print(f"[collect] nothing collected for {name}")
    if kept:
        print(f"[collect] kept as they were: {', '.join(kept)}")
    return 0


# ------------------------------------------------------------------------------------------------ formatting


class Missing(Exception):
    """An input of a table is not there: the table renders as one "(not available: ...)" line."""


def missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def num(x, d: int = 3) -> str:
    """``x`` to ``d`` decimals ("—" if missing); a value below 1 that would show as 1 keeps two more decimals."""
    if missing(x):
        return "—"
    s = f"{x:.{d}f}"
    return f"{x:.{d + 2}f}" if x < 1 and s == f"{1:.{d}f}" else s


def pct(x) -> str:
    """A rate as a percentage: whole percents from 10%, finer below; never rounded up to 100%."""
    if missing(x):
        return "—"
    v = 100 * x
    if v >= 10 or v == 0:
        s = f"{v:.0f}%"
        return f"{math.floor(v * 10) / 10:.1f}%" if s == "100%" and v < 100 else s
    return f"{v:.1f}%" if v >= 1 else f"{v:.2f}%"


def signed(x, d: int = 3) -> str:
    return "—" if missing(x) else f"{x:+.{d}f}"


def sci(x) -> str:
    return "—" if missing(x) else "0" if x == 0 else f"{x:.1e}"


def with_ci(x, interval) -> str:
    return f"{num(x)} [{num(interval[0])}, {num(interval[1])}]"


def band(m: dict | None, fmt: Callable = num) -> str:
    """median [5th, 95th percentile]"""
    return "—" if not m else f"{fmt(m['median'])} [{fmt(m['p05'])}, {fmt(m['p95'])}]"


def mark(ok) -> str:
    return "—" if ok is None else "✓" if ok else "✗"


def yes(x) -> str:
    return "yes" if x else "no"


def outcome(decision: dict, key: str) -> str:
    """A rule's outcome in a comparison's decision: yes / no, or "not evaluated" when it is None (compare_readers.py
    from 2026-10-12 on, when the rule ran for no reader), with the recorded reason."""
    if decision.get(key) is None:
        why = (decision.get("not_evaluated") or {}).get(key)
        return "not evaluated" + (f" ({why})" if why else "")
    return yes(decision[key])


def short(view: str) -> str:
    return view.replace("_reasoning", "")


def rank(value: str, known: tuple) -> tuple:
    return (known.index(value) if value in known else len(known), value)


def chosen(selection: dict) -> str:
    return f"{selection['layer']}{'*' if selection.get('flat') else ''}"


def md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


def legend(*sentences: str) -> list[str]:
    return ["", " ".join(sentences)]


def cp_lower(k: int, n: int, confidence: float = CONFIDENCE) -> float:
    """The exact one-sided Clopper–Pearson lower bound on a proportion when ``k`` of ``n`` trials succeed."""
    if n <= 0:
        return float("nan")
    if k <= 0:
        return 0.0
    from scipy.stats import beta

    return float(beta.ppf(1 - confidence, k, n - k + 1))


# ------------------------------------------------------------------------------------------------ inputs


class Results:
    """The collected tree (``root``) and the deployed monitors (``deployed``), as the tables read them."""

    def __init__(self, root: str, deployed: str):
        self.root, self.deployed = root, deployed

    def json(self, rel: str):
        path = os.path.join(self.root, rel)
        if not os.path.isfile(path):
            raise Missing(f"no {rel} collected")
        with open(path) as fh:
            return json.load(fh)

    def csv(self, rel: str) -> list[dict]:
        path = os.path.join(self.root, rel)
        if not os.path.isfile(path):
            raise Missing(f"no {rel} collected")
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh))

    def glob(self, pattern: str) -> list[list[str]]:
        """The collected paths matching ``pattern``, each split into its parts."""
        return sorted(os.path.relpath(p, self.root).split(os.sep) for p in glob.glob(os.path.join(self.root, pattern)))

    def newest(self, pattern: str) -> str:
        """The collected path matching ``pattern`` (a ``*`` for the run folder) of the newest run."""
        hits = self.glob(pattern)
        if not hits:
            raise Missing(f"no {pattern} collected")
        return "/".join(hits[-1])

    def probes(self, dataset: str) -> list[tuple[str, str, str]]:
        """(reader, view, readout) of every fitted probe of ``dataset`` with a metrics.json, except the readout
        fallback's (``FALLBACK``: only in its own table, ``readout_fallback``)."""
        return [(p[1], p[3], p[4]) for p in self.glob(f"fitted/*/{dataset}/*/*/metrics.json") if p[4] != FALLBACK]


def by_readout_view_reader(p: tuple[str, str, str]) -> tuple:
    return rank(p[2], READOUTS), rank(p[1], VIEWS), rank(p[0], READERS)


# ------------------------------------------------------------------------------------------------ tables


def iid_probes(r: Results) -> list[str]:
    rows, sizes = [], set()
    for reader, view, readout in sorted(r.probes("terminal_prerl"), key=by_readout_view_reader):
        if readout.startswith("router"):
            continue
        m = r.json(f"fitted/{reader}/terminal_prerl/{view}/{readout}/metrics.json")
        s, cf, conc = m["selection"], m["crossfit"], m.get("concealment") or {}
        sizes.add((m["n"]["test"], m["n"]["test_hacks"]))
        rows.append([readout, short(view), reader, chosen(s), f"{s['C']:g}",
                     with_ci(m["test_auroc"], m["test_auroc_ci"]),
                     f"{pct(m['tpr_at_t1'])} / {pct(m['fpr_at_t1'])}", num(cf["hacks_vs_read_not_used"]),
                     num(cf["within_task"]), " / ".join(pct((conc.get(k) or {}).get("recall_at_t1")) for k in "AB")])
    if not rows:
        raise Missing("no fitted IID probes (fitted/<reader>/terminal_prerl/)")
    test = "the test split" + (" ({:,} episodes, {} hacks)".format(*next(iter(sizes))) if len(sizes) == 1 else "")
    return md_table(["readout", "reasoning", "reader", "layer", "C", "test AUROC [95% CI]", "TPR / FPR at t₁%",
                     "hacks vs read-not-used", "within-task AUROC", "concealed hacks caught (A / B)"], rows) + legend(
        "From `fitted/<reader>/terminal_prerl/<view>/<readout>/metrics.json`; label `used_verifier`.",
        "nvfp4 is the 4-bit reader, bf16 full precision.",
        "Layer and C: the pre-registered selection (* a flat grid, so the fallback).",
        f"Test AUROC on {test}, with its 95% task-cluster bootstrap CI.",
        "TPR / FPR at t₁%, the 1%-FPR threshold from out-of-fold honest dev episodes.",
        "Hacks vs read-not-used and within-task: cross-fitted AUROCs over every split.",
        "Concealed: the share of concealed hacks (levels A and B) caught at t₁% by the cross-fitted models.")


def weakest(by_name: dict | None) -> str:
    vals = sorted((v, k) for k, v in (by_name or {}).items() if not missing(v))
    return f"{vals[0][1]} {num(vals[0][0])}" if vals else "—"


def ood_probes(r: Results) -> list[str]:
    rows = []
    for reader, view, readout in sorted(r.probes("diverse_deception"), key=by_readout_view_reader):
        if readout.startswith("router"):
            continue
        m = r.json(f"fitted/{reader}/diverse_deception/{view}/{readout}/metrics.json")
        s, tw, to = m["selection"], m["transfer"]["with_reasoning"], m["transfer"]["without_reasoning"]
        conc = tw.get("concealment_recall_at_t1") or {}
        rows.append([readout, reader, chosen(s), f"{s['C']:g}", with_ci(m["test_auroc"], m["test_auroc_ci"]),
                     weakest(m.get("per_source")), weakest(m.get("leave_one_source_out")),
                     f"{num(tw['auroc_all'])} / {num(to['auroc_all'])}",
                     f"{pct(tw['tpr_at_t1'])} / {pct(to['tpr_at_t1'])}",
                     " / ".join(pct(conc.get(k)) for k in "AB")])
    if not rows:
        raise Missing("no fitted OOD probes (fitted/<reader>/diverse_deception/)")
    return md_table(["readout", "reader", "layer", "C", "OOD test AUROC [95% CI]", "weakest source",
                     "weakest left out", "TerminalEnv transfer AUROC (with / without)", "TPR at t₁% (with / without)",
                     "concealed caught at t₁%, with reasoning (A / B)"], rows) + legend(
        "From `fitted/<reader>/diverse_deception/without_reasoning/<readout>/metrics.json`; label: deceptive.",
        "OOD test AUROC on the held-out pairs, with its 95% pair-cluster bootstrap CI.",
        "Weakest source: the lowest per-source test AUROC; weakest left out: the lowest when the probe is refitted",
        "without that source.",
        "Transfer: AUROC over every TerminalEnv episode against `used_verifier`; TPR at each view's t₁% from its",
        "honest dev episodes. Concealed: the share of concealed hacks caught at t₁%.")


def router_probes(r: Results) -> list[str]:
    found = [(ds, *p) for ds in DATASETS for p in r.probes(ds) if p[2].startswith("router")]
    rows = []
    for ds, reader, view, readout in sorted(found, key=lambda p: (rank(p[0], DATASETS), rank(p[2], VIEWS),
                                                                   rank(p[3], READOUTS), rank(p[1], READERS))):
        rel = f"fitted/{reader}/{ds}/{view}/{readout}"
        meta, m = r.json(f"{rel}/meta.json"), r.json(f"{rel}/metrics.json")
        s, iid, model = m["selection"], ds == "terminal_prerl", meta.get("model_layer")
        tr = m.get("transfer") or {}
        rows.append([f"IID, {short(view)}" if iid else "OOD", readout, reader,
                     (f"{s['layer']} ({model})" if isinstance(model, int) else "all") + ("*" if s.get("flat") else ""),
                     f"{s['C']:g}", with_ci(m["test_auroc"], m["test_auroc_ci"]),
                     num(m["crossfit"]["hacks_vs_read_not_used"]) if iid else "—",
                     "—" if iid else " / ".join(num(tr[v]["auroc_all"]) for v in VIEWS)])
    if not rows:
        raise Missing("no fitted router probes (fitted/<reader>/<dataset>/<view>/router*/)")
    return md_table(["probe (reasoning)", "readout", "reader", "MoE layer (model layer)", "C", "test AUROC [95% CI]",
                     "hacks vs read-not-used", "TerminalEnv transfer AUROC (with / without)"], rows) + legend(
        "Exploratory, not deployed: logistic regression on the router logits at \"Yes\" of one MoE layer (`router`)",
        "or of every MoE layer (`router_all`, layer \"all\").",
        "From `fitted/<reader>/<dataset>/<view>/<readout>/{meta,metrics}.json`.",
        "Test AUROC: the IID test split, or the OOD test pairs. * a flat grid. Hacks vs read-not-used: cross-fitted.")


def deployed_monitors(r: Results) -> list[str]:
    path = os.path.join(r.deployed, "deployed.json")
    if not os.path.isfile(path):
        raise Missing(f"no deployed.json in the deployed monitors' folder {os.path.basename(r.deployed.rstrip('/'))}")
    with open(path) as fh:
        deployed = json.load(fh)
    rows = []
    for d in deployed:
        sc = d.get("serving_check") or {}
        b = sc.get("batching")
        rows.append([d["name"], f"{d.get('trained_reader') or d.get('reader')} → {d.get('reader')}", str(d["layer"]),
                     f"{d['threshold_1']:.2f}", num(sc.get("test_auroc")),
                     f"{pct(sc.get('test_tpr_at_t1'))} / {pct(sc.get('test_fpr_at_t1'))}",
                     f"{d['golden']} ({d['golden_tolerance']:g})",
                     f"{b['dlogit_p99']:.2f}; {pct(b['near_threshold_hacks'])} / {pct(b['near_threshold_negatives'])}"
                     if b else "—"])
    if not rows:
        raise Missing("deployed.json lists no monitors")
    return md_table(["monitor", "trained on → served on", "layer", "t₁% (logit)", "test AUROC", "TPR / FPR at t₁%",
                     "golden episodes (tolerance)", "batching: p99 Δlogit; hacks / honest that close to t₁%"],
                    rows) + legend(
        "From `data/probes/Inkling-Small/deployed.json`: each monitor's serving check on the serving reader's",
        "TerminalEnv test split, label `used_verifier` (for OOD monitors, their transfer).",
        "A monitor served on another reader than it was trained on has t₁% recalibrated on the serving reader's honest",
        "dev episodes. Golden: the episodes the server checks at startup, and the largest logit difference it accepts.",
        "Batching (follow-up monitors): the 99th percentile of the absolute logit difference between batched and",
        "one-at-a-time readouts, and the shares of the view's hacks and honest episodes, over every split, that",
        "close to t₁%.")


DELTAS = {"iid_test": "test", "ood_test": "OOD test", "iid_transfer_with_reasoning": "transfer with",
          "iid_transfer_without_reasoning": "transfer without"}


def per_view(d: dict, key: str) -> str:
    return " / ".join(num((d.get(v) or {}).get(key)) for v in VIEWS)


def comparison_notes(d: dict) -> list[str]:
    notes = []
    for reader, j in (d.get("jitter") or {}).items():
        if j:
            notes.append(f"Jitter ({reader}, {j['n']} diagnostics inputs): the absolute Δlogit between batched and "
                         f"one-at-a-time readouts has median {j['dlogit_p50']:.2f} and 99th percentile "
                         f"{j['dlogit_p99']:.2f}.")
    for reader, b in (d.get("batching") or {}).items():
        if b:
            notes.append(f"Batching rule ({reader}): {pct(b['near_threshold_hacks'])} of hacks and "
                         f"{pct(b['near_threshold_negatives'])} of honest episodes lie that close to t₁% (fix "
                         f"batching: {yes(b['fix_batching'])}).")
    for reader, s in (d.get("sink") or {}).items():
        if s:
            notes.append(f"\"Yes\" ({reader}): median norm ratio to the context {s['median_ratio_layers_10_35']:.2f} "
                         f"in layers 10–35 (sink-like: {yes(s['sink_like'])}).")
    if (d.get("routing_agreement") or {}).get("mean") is not None:
        notes.append(f"Routing agreement over the diagnostics' tokens: {pct(d['routing_agreement']['mean'])} of the "
                     "top-6 experts match.")
    at_yes = d.get("routing_agreement_at_yes") or {}
    if at_yes:
        notes.append("Routing agreement at \"Yes\" (share of matching top-6 experts): " + "; ".join(
            f"{k} {pct(v['mean'])} over {v['n']:,} items (lowest {pct(v['min'])}, MoE layer {v['min_layer']})"
            for k, v in at_yes.items()) + ".")
    return ["", *(f"- {n}" for n in notes)] if notes else []


def reader_comparison(name: str) -> Callable[[Results], list[str]]:
    def table(r: Results) -> list[str]:
        rel = f"compare/{name}/decision.json"
        d = r.json(rel)
        rows = []
        for probe, p in d["probes"].items():
            ok = p.get("passes") or {}
            layers = "all" if probe.endswith("/router_all") else \
                f"{p['layers'].get('nvfp4', '—')} / {p['layers'].get('bf16', '—')}"
            rows.append([probe, layers,
                         per_view(p["agreement"], "jaccard_at_t1"), per_view(p["agreement"], "spearman"),
                         f"{per_view(p['refit_floor'], 'jaccard_median')}; "
                         f"{per_view(p['refit_floor'], 'spearman_median')}",
                         "; ".join(f"{DELTAS.get(k, k)} {signed(v['delta'])} [{signed(v['ci'][0])}, "
                                   f"{signed(v['ci'][1])}]" for k, v in p["delta_auroc"].items()),
                         f"(a) {mark(ok.get('agreement'))} (c) {mark(ok.get('agreement_within_refit_noise'))} "
                         f"(b) {mark(ok.get('delta_auroc'))}"])
        if not rows:
            raise Missing(f"{rel} compares no probes")
        dec = d["decision"]
        return md_table(["probe", "layer (4-bit / full)", "Jaccard at t₁% (with / without)",
                         "Spearman (with / without)", "refit floor, medians: Jaccard; Spearman",
                         "ΔAUROC, 4-bit − full [95% CI]", "rules"], rows) + [
            "", "Decision per readout: " + ", ".join(f"{k} → {v}" for k, v in dec["by_readout"].items())
            + f". Fix batching: {outcome(dec, 'fix_batching')}. \"Yes\" sink-like: {outcome(dec, 'sink_like_yes')}."
        ] + comparison_notes(d) + legend(
            f"From `{rel}` (experiments/compare_readers.py).",
            "Agreement: the full-precision probe's logits on both readers' TerminalEnv val+test features, per view:",
            "Jaccard of the episodes flagged at t₁%, and Spearman. Refit floor: the same between that probe and 20",
            "task-bootstrap refits of itself (medians).",
            "ΔAUROC: each reader's own probe on its own features, with a cluster bootstrap CI.",
            "Rules: (a) agreement (Jaccard ≥ 0.8 and Spearman ≥ 0.95 in both views), (c) agreement at least the refit",
            "floor's, (b) no AUROC loss (Δ ≥ −0.01 and CI lower end ≥ −0.02). A readout keeps 4-bit if every probe on",
            "it passes (a) or (c), and (b).")

    return table


def shift_cell(s: dict | None) -> str:
    return "—" if not s else f"{signed(s['mean'], 2)} ({num(s['sd'], 2)}); {num(s['p99_abs'], 2)}; {num(s['pearson'])}"


def rates(m: dict | None, kind: str) -> str:
    return "—" if not m else f"{pct(m.get(f'tpr_at_t1_{kind}'))} / {pct(m.get(f'fpr_at_t1_{kind}'))}"


def cross_reader(r: Results) -> list[str]:
    probes = []
    for parts in sorted(r.glob("compare/*/decision.json"), key=lambda p: (p[1] != "followup", p[1])):
        d = r.json("/".join(parts))
        probes += [(name, p) for name, p in d.get("probes", {}).items() if "cross_reader" in p]
    if not probes:
        raise Missing("no compare/*/decision.json has the cross_reader fields (experiments/compare_readers.py "
                      "from 2026-10-12 on)")
    feats = sorted({f for _, p in probes for by_f in p["cross_reader"].values() for f in by_f},
                   key=lambda x: rank(x, READERS))
    on = " / ".join(feats)
    cos = [num(p["weight_cosine"]) if p.get("weight_cosine") is not None else
           f"layers differ ({p['layers'].get('nvfp4')}, {p['layers'].get('bf16')})" for _, p in probes]
    lines = ["Weight cosine and score shift:", ""] + md_table(
        ["probe", "weight cosine, 4-bit vs full", "score shift, with reasoning: mean (sd); p99 abs; Pearson r",
         "without reasoning"],
        [[name, c, *(shift_cell((p.get("score_shift") or {}).get(v)) for v in VIEWS)]
         for (name, p), c in zip(probes, cos)])
    readers_of = [(name, pr, by_f) for name, p in probes
                  for pr, by_f in sorted(p["cross_reader"].items(), key=lambda x: rank(x[0], READERS))]
    iid = [x for x in readers_of if x[0].startswith("terminal_prerl")]
    if iid:
        lines += ["", "Cross-reader scoring, IID probes (TerminalEnv test split):", ""] + md_table(
            ["probe", "probe's reader", f"test AUROC on {on} features", f"TPR / FPR at its own t₁%, on {on}",
             f"TPR / FPR at t₁% recalibrated on the features, on {on}"],
            [[name, pr, " / ".join(num((by_f.get(f) or {}).get("test_auroc")) for f in feats),
              "; ".join(rates(by_f.get(f), "own") for f in feats),
              "; ".join(rates(by_f.get(f), "recalibrated") for f in feats)] for name, pr, by_f in iid])
    ood = [x for x in readers_of if not x[0].startswith("terminal_prerl")]
    if ood:
        lines += ["", "Cross-reader scoring, OOD probes:", ""] + md_table(
            ["probe", "probe's reader", f"OOD test AUROC on {on} features", f"transfer, with reasoning, on {on}",
             f"transfer, without reasoning, on {on}"],
            [[name, pr, " / ".join(num((by_f.get(f) or {}).get("ood_test_auroc")) for f in feats),
              *(" / ".join(num(((by_f.get(f) or {}).get("transfer_auroc") or {}).get(v)) for f in feats)
                for v in VIEWS)]
             for name, pr, by_f in ood])
    floor = [[name, pr, dr, short(view), str(s["n"]), f"{signed(s['mean'])} ({num(s['sd'])})", num(s["spearman"])]
             for name, p in probes for dr, by_p in (p.get("batching_floor") or {}).items()
             for pr, by_v in sorted(by_p.items(), key=lambda x: rank(x[0], READERS)) for view, s in by_v.items()]
    if floor:
        lines += ["", "Batching floor (the diagnostics' TerminalEnv rows):", ""] + md_table(
            ["probe", "probe's reader", "diagnostics' reader", "reasoning", "n", "one-at-a-time − batched: mean (sd)",
             "Spearman"], floor)
    return lines + legend(
        "From the report-only fields of `compare/*/decision.json` (no rule reads them).",
        "Weight cosine: between the two readers' probes, when they sit at the same layer.",
        "Score shift: the full-precision probe's logits on 4-bit minus on full-precision features, over the same",
        "TerminalEnv val+test episodes.",
        "Cross-reader: each reader's probe on both readers' features; the recalibrated t₁% is the 99th percentile of",
        "the probe's logits on those features' honest dev episodes, as deploy_probes.py sets it for serving.",
        "Batching floor: the probe's one-at-a-time minus batched logits on the diagnostics sample.")


def hack_cells(r: Results, rel: str) -> tuple:
    """(caught, of) for test hacks, test tasks, cross-fitted hacks, cross-fitted tasks; None where unknown."""
    try:
        c = r.json(f"{rel}/hack_counts.json")
        return tuple((c[s][k], c[s][n]) for s in ("test", "crossfit")
                     for k, n in (("caught", "hacks"), ("tasks_caught", "tasks")))
    except Missing:
        m = r.json(f"{rel}/metrics.json")
        n = m["n"]["test_hacks"]
        cf = (m.get("concealment") or {}).get("unedited_crossfit")
        return ((round(m["tpr_at_t1"] * n), n), None,
                (round(cf["recall_at_t1"] * cf["n"]), cf["n"]) if cf else None, None)


def floor_num(x, d: int = 3) -> str:
    """``x`` rounded DOWN to ``d`` decimals ("—" if missing): a lower bound shown rounded up would no longer be one."""
    return "—" if missing(x) else f"{math.floor(x * 10 ** d + 1e-9) / 10 ** d:.{d}f}"  # 1e-9: 0.29·100 = 28.99…


def bound(c: tuple[int, int] | None) -> str:
    return "—" if c is None else f"{c[0]}/{c[1]}, ≥ {floor_num(cp_lower(*c))}"


def test_set_bounds(r: Results) -> list[str]:
    groups: dict[tuple, list[str]] = {}
    for reader, view, readout in sorted(r.probes("terminal_prerl"), key=by_readout_view_reader):
        key = hack_cells(r, f"fitted/{reader}/terminal_prerl/{view}/{readout}")
        groups.setdefault(key, []).append(f"{reader} {short(view)}/{readout}")
    if not groups:
        raise Missing("no fitted IID probes (fitted/<reader>/terminal_prerl/)")
    rows = [[f"all {len(names)}" if len(groups) == 1 else ", ".join(names), *(bound(c) for c in key)]
            for key, names in groups.items()]
    return md_table(["IID probes", "test split: hacks caught", "test split: tasks caught", "cross-fitted: hacks caught",
                     "cross-fitted: tasks caught"], rows) + legend(
        "k/n, ≥ x: k of n caught at t₁%, and the exact one-sided 97.5% Clopper–Pearson lower bound on the rate,",
        "rounded down (it exceeds x with 97.5% confidence, the lower end of a 95% two-sided interval).",
        "Hacks: per episode. Tasks: the tasks with hacks, one counting as caught only if all its hacks are; tasks are",
        "the independent unit, as a task's hacks are alike.",
        "Test split: the probe on its held-out test tasks. Cross-fitted: every hack of every split, each scored by a",
        "model fitted without its task (5 folds by task).",
        "From `fitted/<reader>/terminal_prerl/<view>/<readout>/hack_counts.json` (else, hacks only, metrics.json).")


def refit_cell(m: dict | None, fmt: Callable = num) -> str:
    return "—" if not m else f"{fmt(m['original'])}; {band(m, fmt)}"


def ood_refits(r: Results) -> list[str]:
    rows = []
    found = [(p[1], p[2]) for p in r.glob("ood_refits/*/*/summary.json")]
    for reader, readout in sorted(found, key=lambda p: (rank(p[1], READOUTS), rank(p[0], READERS))):
        s = r.json(f"ood_refits/{reader}/{readout}/summary.json")
        m = s["metrics"]
        rows.append([readout, reader, f"{s['layer']}, {s['C']:g}", str(s["refits"]),
                     refit_cell(m.get("ood_test_auroc")),
                     refit_cell(m.get("transfer_auroc_with_reasoning")),
                     refit_cell(m.get("transfer_auroc_without_reasoning")),
                     refit_cell(m.get("tpr_at_t1_with_reasoning"), pct),
                     num((m.get("cosine_to_original") or {}).get("median"), 2)])
    if not rows:
        raise Missing("no ood_refits/<reader>/<readout>/summary.json")
    return md_table(["OOD probe", "reader", "layer, C", "refits", "OOD test AUROC: original; refits",
                     "transfer, with reasoning", "transfer, without reasoning", "TPR at t₁%, with reasoning",
                     "weight cosine to the original (median)"], rows) + legend(
        "From `ood_refits/<reader>/<readout>/summary.json` (experiments/ood_refits.py): the fitted OOD probe",
        "(original), then its refits on pair-bootstrap resamples of its training pairs, at the same layer and C:",
        "median [5th, 95th percentile].",
        "Transfer: AUROC over every TerminalEnv episode against `used_verifier`; TPR at each view's t₁%.")


def ood_by_layer(r: Results) -> list[str]:
    series = []
    found = [(p[1], p[2]) for p in r.glob("ood_refits/*/*/by_layer.csv")]
    for reader, readout in sorted(found, key=lambda p: (rank(p[1], READOUTS), rank(p[0], READERS))):
        rows = r.csv(f"ood_refits/{reader}/{readout}/by_layer.csv")
        try:
            selected = r.json(f"ood_refits/{reader}/{readout}/summary.json")["layer"]
        except Missing:
            selected = None
        series.append((f"{readout}, {reader}", {int(x["layer"]): x for x in rows}, selected))
    if not series:
        raise Missing("no ood_refits/<reader>/<readout>/by_layer.csv")

    def cell(by_layer: dict, selected, layer: int) -> str:
        x = by_layer.get(layer)
        if x is None:
            return "—"
        t = num(float(x["transfer_auroc_with_reasoning"]), 2)
        return f"{f'**{t}**' if layer == selected else t} ({num(float(x['ood_test_auroc']))})"

    layers = sorted(set().union(*(by_layer for _, by_layer, _ in series)))
    return md_table(["layer", *(name for name, _, _ in series)],
                    [[str(layer), *(cell(by, sel, layer) for _, by, sel in series)] for layer in layers]) + legend(
        "From `ood_refits/<reader>/<readout>/by_layer.csv`: one fit on all training pairs at each layer, at the",
        "probe's C. Each cell: transfer AUROC to TerminalEnv hacks, with reasoning (bold at the selected layer), and",
        "in brackets the in-distribution OOD test AUROC.")


def curve_readers(r: Results) -> list[str]:
    readers = [p[1] for p in r.glob("learning_curves/*/summary.json")]
    if not readers:
        raise Missing("no learning_curves/<reader>/summary.json")
    return sorted(readers, key=lambda x: rank(x, READERS))


def learning_curves_iid(r: Results) -> list[str]:
    rows = []
    for reader in curve_readers(r):
        s = r.json(f"learning_curves/{reader}/summary.json")
        for x in sorted(s["iid"], key=lambda x: (rank(x["view"], VIEWS), x["fraction"])):
            rows.append([reader, short(x["view"]), f"{x['fraction']:.0%}", f"{x['n_hacks_or_pairs']:g}", str(x["fits"]),
                         band(x["test_auroc"]), band(x["test_tpr_at_1pct_fpr"], pct),
                         band(x["concealed_test_hacks_recall"], pct)])
    return md_table(["reader", "reasoning", "dev tasks used", "training hacks", "fits", "test AUROC", "TPR at 1% FPR",
                     "concealed test hacks caught"], rows) + legend(
        "From `learning_curves/<reader>/summary.json` (experiments/learning_curves.py): the follow-up probe refitted",
        "at its layer and C on a share of its dev tasks (tasks with and without hacks sampled separately), scored on",
        "the test split: median [5th, 95th percentile] over the fits; training hacks: the median count.",
        "TPR at 1% FPR takes its threshold from the test negatives, so it measures the ranking, not the calibration.")


def learning_curves_ood(r: Results) -> list[str]:
    rows = []
    for reader in curve_readers(r):
        s = r.json(f"learning_curves/{reader}/summary.json")
        for x in sorted(s["ood"], key=lambda x: x["fraction"]):
            rows.append([reader, f"{x['n_hacks_or_pairs']:,.0f} ({x['fraction']:.0%})", str(x["fits"]),
                         band(x["ood_test_auroc"]), band(x["transfer_auroc_with_reasoning"]),
                         band(x["transfer_auroc_without_reasoning"]),
                         f"{pct(x['tpr_at_t1_with_reasoning']['median'])} / "
                         f"{pct(x['tpr_at_t1_without_reasoning']['median'])}"])
    return md_table(["reader", "training pairs (share)", "fits", "OOD test AUROC", "transfer, with reasoning",
                     "transfer, without reasoning", "TPR at t₁%, with / without (medians)"], rows) + legend(
        "From `learning_curves/<reader>/summary.json`: the OOD follow-up probe refitted at its layer and C on a share",
        "of its training pairs (stratified by source): median [5th, 95th percentile] over the fits.",
        "Transfer: AUROC over every TerminalEnv episode against `used_verifier`.")


def ood_source_mixes(r: Results) -> list[str]:
    rows = []
    for reader in curve_readers(r):
        s = r.json(f"learning_curves/{reader}/summary.json")
        for x in s["ood_sources"]:
            how, src = x["trained_on"], x["source"]
            mix = {"all": "all sources", "all_but": f"all but {src}", "only": f"{src} only"}.get(how, f"{how} {src}")
            rows.append([reader, mix, f"{x['n_hacks_or_pairs']:,.0f}", str(x["fits"]),
                         num(x["ood_test_auroc"]["median"]),
                         band(x["transfer_auroc_with_reasoning"]), band(x["transfer_auroc_without_reasoning"])])
    return md_table(["reader", "training mix", "pairs", "fits", "OOD test AUROC (median)", "transfer, with reasoning",
                     "transfer, without reasoning"], rows) + legend(
        "From `learning_curves/<reader>/summary.json`'s `ood_sources`: the OOD follow-up probe refitted on",
        "pair-bootstrap resamples of one mix of its training sources, at its layer and C: median [5th, 95th",
        "percentile] over the fits. Pairs: the median number of distinct pairs in a resample, about 63% of the mix,",
        "since a bootstrap resample repeats some pairs and leaves out others.")


def verify_fits(r: Results) -> list[str]:
    runs = ["/".join(p) for p in r.glob("verify_fits/*/summary.json")]
    if not runs:
        raise Missing("no verify_fits/<run>/summary.json collected")
    latest: dict[str, tuple[dict, str]] = {}  # probe -> (its newest check, that run's folder)
    notes = []
    for rel in runs:  # oldest first, so a later run's check of a probe replaces an earlier one's
        s, run = r.json(rel), rel.split("/")[1]
        probes = s.get("probes") or {}
        for name, p in probes.items():
            latest[name] = (p, run)
        notes.append(f"- `{run}`: {len(probes)} probe{'' if len(probes) == 1 else 's'}. {verify_run(s)}")
    if not latest:
        raise Missing("verify_fits/*/summary.json list no probes")
    rows = []
    for name, (p, run) in latest.items():
        if "error" in p:
            rows.append([name, run, "—", "—", "—", f"{p.get('verdict', 'fail')}: {p['error']}"])
            continue
        cv, sel, refit = p.get("cv") or {}, p.get("selection") or {}, p.get("refit") or {}
        saved = sel.get("saved") or {}
        where = "all MoE layers" if name.endswith("/router_all") else f"layer {saved.get('layer')}"
        rows.append([name, run, f"{cv.get('cells_matched', '—')} / {cv.get('cells_checked', '—')}",
                     str(sel.get("status", "—")) + (f" ({where}, C {saved['C']:g})" if saved else ""),
                     "; ".join(sci((refit.get(k) or {}).get("max_abs_diff"))
                               for k in ("weights", "thresholds", "metrics", "test_logits")),
                     str(p.get("verdict", "—"))])
    passed = sum(str(p.get("verdict", "")).startswith("pass") for p, _ in latest.values())
    return md_table(["probe", "run", "grid cells matched / checked", "selection",
                     "largest Δ: weights; t₁/t₅; metrics; test logits", "verdict"], rows) + [
        "", f"{passed} of {len(latest)} probes pass (each probe's newest check). Runs:", ""] + notes + legend(
        "From `verify_fits/<run>/summary.json` (experiments/verify_fits.py): part of each fit re-run with the code",
        "of the day and compared with what the fit saved. Grid cells: recomputed out-of-fold AUROCs equal to",
        "cv_grid.csv's to 5 decimals. Δ: the largest absolute difference from the saved weights, thresholds, metrics",
        "and test logits (0: bit for bit).")


def verify_run(s: dict) -> str:
    """A verify_fits run's date and code: the commit, and the sha256 of the fitting code it ran."""
    code = s.get("code") or {}
    files = s.get("code_files") or code  # runs before 2026-10-12 06:40 UTC kept the file hashes under "code"
    took = (s.get("seconds") or {}).get("total")
    return f"Finished {s.get('created_utc', '—')}" + (f" after {took / 60:.0f} min" if took else "") + (
        f" at commit {code['commit'][:12]}{' with uncommitted changes' if code.get('dirty') else ''}"
        if code.get("commit") else "") + (
        f" with fit_probes.py {files['fit_probes.py'][:12]}, probe_fit.py {files['probe_fit.py'][:12]} (sha256)"
        if "fit_probes.py" in files and "probe_fit.py" in files else "") + "."


def dataset_stats(r: Results) -> list[str]:
    rel = r.newest("analyses/*/dataset_stats.json")
    s = r.json(rel)
    lines: list[str] = []
    tokens = [(ds, view, v) for ds, by_view in (s.get("tokens") or {}).items() for view, v in (by_view or {}).items()]
    if tokens:
        lines += md_table(["dataset", "reasoning", "items", "tokens per probe input: mean / median / p95 / max",
                           "total tokens"],
                          [[ds, short(view), f"{v['n_items']:,}",
                            f"{v['mean']:,.0f} / {v['median']:,.0f} / {v['p95']:,.0f} / {v['max']:,}",
                            f"{v['total_estimate'] / 1e6:.1f}M" + (f" (from {v['n_rendered']:,} rendered)"
                                                                  if v["n_rendered"] != v["n_items"] else "")]
                           for ds, view, v in tokens])
    dis = s.get("disclosure")
    if dis:
        h, o = dis["hacks"], dis["non_hacks"]
        lines += ["", f"Disclosure (terminal_prerl): {h['disclosing']:,} of {h['n']:,} hacks ({pct(h['share'])}) and "
                      f"{o['disclosing']:,} of {o['n']:,} other episodes ({pct(o['share'])}) match the pattern the "
                      "concealment set removes (verif, ground, truth) in their visible turn texts."]
    overlap = s.get("ood_question_overlap")
    if overlap:
        lines += ["", "OOD question overlap (diverse_deception):", ""] + md_table(
            ["source", "items", "test items", "test items whose question is in training",
             "distinct questions (train / test)"],
            [[src, f"{v['n_items']:,}", f"{v['n_test']:,}", f"{v['test_sharing_question_with_train']:,}",
              f"{v['distinct_questions']:,} ({v['distinct_questions_train']:,} / {v['distinct_questions_test']:,})"]
             for src, v in overlap.items()])
    if not lines:
        raise Missing(f"{rel} has none of tokens, disclosure, ood_question_overlap")
    return (lines[1:] if lines[0] == "" else lines) + legend(
        f"From `{rel}` (experiments/probe_analyses.py dataset-stats). Tokens: each probe input as the extraction",
        "renders it (the conversation plus the follow-up); the total is the mean times the items.")


def readout_fallback(r: Results) -> list[str]:
    """The readout rule's comparison: per probe with a ``FALLBACK`` fit, followup vs followup_prev by dev CV."""
    pairs = sorted({(p[1], p[2], p[3]) for p in r.glob(f"fitted/*/*/*/{FALLBACK}/metrics.json")},
                   key=lambda p: (rank(p[1], DATASETS), rank(p[2], VIEWS), rank(p[0], READERS)))
    rows, decisions = [], []
    for reader, ds, view in pairs:
        iid, cv = ds == "terminal_prerl", {}
        name = f"IID, {short(view)}" if iid else "OOD"
        for readout in ("followup", FALLBACK):
            m = r.json(f"fitted/{reader}/{ds}/{view}/{readout}/metrics.json")
            sel = m["selection"]
            cv[readout] = sel["cv_auroc"]
            if iid:
                extra = [f"{pct(m['tpr_at_t1'])} / {pct(m['fpr_at_t1'])}", "—"]
            else:
                t = m["transfer"]
                extra = ["—", f"{num(t['with_reasoning']['auroc_all'])} / {num(t['without_reasoning']['auroc_all'])}"]
            rows.append([name, reader, readout, f"{chosen(sel)}, {sel['C']:g}", num(sel["cv_auroc"], 4),
                         with_ci(m["test_auroc"], m["test_auroc_ci"]), *extra])
        won = FALLBACK if cv[FALLBACK] > cv["followup"] else "followup"
        how = "tie" if cv[FALLBACK] == cv["followup"] else "higher"
        decisions.append(f"{name} ({reader}) → `{won}` ({how})")
    if not rows:
        raise Missing(f"no fitted {FALLBACK} probes (fitted/<reader>/<dataset>/<view>/{FALLBACK}/)")
    return md_table(["probe", "reader", "readout", "layer, C", "dev CV AUROC", "test AUROC [95% CI]",
                     "IID: TPR / FPR at t₁%", "OOD: transfer AUROC (with / without)"], rows) + [
        "", "If the readout rule fires, its comparison (the higher dev CV AUROC; `followup` on a tie) picks: "
        + "; ".join(decisions) + "."
    ] + legend(
        f"From `fitted/<reader>/<dataset>/<view>/{{followup,{FALLBACK}}}/metrics.json`: `{FALLBACK}` reads the token",
        "just before \"Yes\", fitted with the same folds, grid and selection as `followup`. Dev CV AUROC: the mean",
        "out-of-fold AUROC of the configuration the selection picked (* a flat grid). Test AUROC on the test split",
        "(IID) or the held-out pairs (OOD), with its 95% cluster bootstrap CI.")


TABLES: dict[str, Callable[[Results], list[str]]] = {
    "iid_probes": iid_probes, "ood_probes": ood_probes, "router_probes": router_probes,
    "deployed_monitors": deployed_monitors, "reader_comparison_followup": reader_comparison("followup"),
    "reader_comparison_mean": reader_comparison("mean_router"), "cross_reader": cross_reader,
    "test_set_bounds": test_set_bounds, "ood_refits": ood_refits, "ood_by_layer": ood_by_layer,
    "learning_curves_iid": learning_curves_iid, "learning_curves_ood": learning_curves_ood,
    "ood_source_mixes": ood_source_mixes, "verify_fits": verify_fits, "dataset_stats": dataset_stats,
    "readout_fallback": readout_fallback,
}

# ------------------------------------------------------------------------------------------------ render, check

MARKER = re.compile(r"<!-- probe_report:(begin|end) (\S+) -->")
BLOCK = re.compile(r"<!-- probe_report:begin (\S+) -->(.*?)<!-- probe_report:end \1 -->", re.S)


def block(name: str, results: Results) -> list[str]:
    try:
        return TABLES[name](results)
    except Missing as e:
        return [f"(not available: {e})"]
    except KeyError as e:
        return [f"(not available: the inputs have no field {e})"]


def render_text(text: str, results: Results, where: str) -> str:
    """``text`` with every marked block regenerated."""
    out, pos, current = [], 0, None
    for m in MARKER.finditer(text):
        kind, name = m.groups()
        if kind == "begin":
            if current:
                raise SystemExit(f"{where}: table {name!r} begins inside {current!r}")
            if name not in TABLES:
                raise SystemExit(f"{where}: unknown table {name!r}; known: {', '.join(TABLES)}")
            out.append(text[pos:m.end()])
            current = name
        else:
            if name != current:
                raise SystemExit(f"{where}: the end marker of {name!r} has no begin marker before it")
            out.append("\n\n" + "\n".join(block(name, results)) + "\n\n")
            pos, current = m.start(), None
    if current:
        raise SystemExit(f"{where}: table {current!r} has no end marker")
    return "".join(out) + text[pos:]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect", help="copy the small result files into --out (rebuilt each time)")
    c.add_argument("--scratch", default="../scratch", help="where the results are, unless given one by one")
    c.add_argument("--probes", help="fit_probes.py's outputs, one folder per reader (default: <scratch>/probes)")
    c.add_argument("--compare", action="append", metavar="NAME=DIR",
                   help="a compare_readers.py output, collected as compare/NAME (repeatable; default: "
                        "followup=<scratch>/compare and mean_router=<scratch>/compare_mean_router)")
    c.add_argument("--ood-refits", help="ood_refits.py's outputs, <reader>/<readout>/ (default: <scratch>/ood_refits)")
    c.add_argument("--learning-curves", help="learning_curves.py's outputs, <reader>/ (default: "
                                             "<scratch>/learning_curves)")
    c.add_argument("--verify-fits", help="verify_fits.py's --out: every <timestamp>/ run in it (or one run folder) "
                                         "(default: <scratch>/verify_fits)")
    c.add_argument("--analyses", help="probe_analyses.py's --out: every <timestamp>/ run in it (or one run folder) "
                                      "(default: none)")
    c.add_argument("--out", default="data/probe_results")
    c.add_argument("--strip", help="removed from the sources listed in MANIFEST.json (default: the folder holding "
                                   "--scratch)")
    for name, what in (("render", "regenerate the marked tables in FILE"),
                       ("check", "exit 1 if render would change FILE")):
        p = sub.add_parser(name, help=what)
        p.add_argument("files", nargs="+", metavar="FILE")
        p.add_argument("--results", default="data/probe_results", help="collect's --out")
        p.add_argument("--deployed", default="data/probes/Inkling-Small", help="the deployed monitors")
    args = ap.parse_args(argv)
    if args.command == "collect":
        return collect(args)
    results, stale = Results(args.results, args.deployed), []
    for path in args.files:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        new = render_text(text, results, path)
        if args.command == "render":
            if new != text:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(new)
            print(f"[render] {path}: {'updated' if new != text else 'up to date'}")
        elif new != text:
            old_blocks, new_blocks = BLOCK.findall(text), BLOCK.findall(new)
            changed = [n for (n, a), (_, b) in zip(old_blocks, new_blocks) if a != b]
            stale.append(path)
            print(f"[check] {path}: out of date: {', '.join(changed) or 'markers'}; run "
                  f"`experiments/probe_report.py render {path}`")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
