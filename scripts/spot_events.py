#!/usr/bin/env python3
"""Every spot event of our Nebius GPU runs, from the launcher's records (``job.json`` of each attempt).

    python3 scripts/spot_events.py output/gpu_jobs            # prints the table and the summary
    python3 scripts/spot_events.py output/gpu_jobs --md out.md

**Per VM attempt, one row:**
- when it was launched, which VM, which region and at what price;
- each creation failure, classified as capacity (a schedule timeout or NotEnoughResources), quota (e.g. public IPs) or
  other;
- the minutes until the VM ran, and how long it ran;
- how it ended: finished, preempted, the --hours cap, or failed;
- the estimated cost.

**The summary** gives counts by kind and region, VM-hours, preemptions per VM-hour, the minutes lost to waiting for
capacity, and the cost of attempts that were preempted. These answer the question "are spot preemptions too much
trouble?".

Stdlib only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone


def seconds(stamp: str | None) -> float | None:
    if not stamp:
        return None
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def kind_of(error: str) -> str:
    if re.search(r"NotEnoughResources|schedule timeout|insufficient hardware", error, re.I):
        return "capacity"
    if re.search(r"Quota|quota", error):
        m = re.search(r"quota ([\w.\-]+)", error) or re.search(r"violations:\s*([\w.\-]+)", error)
        return f"quota ({m.group(1)})" if m else "quota"
    return "other"


def rows(root: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(root, "**", "job.json"), recursive=True)):
        try:
            rec = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("mode") not in ("run", "up"):
            continue
        t = rec.get("times") or {}
        start, running, deleted = seconds(t.get("create_start")), seconds(t.get("running")), seconds(t.get("deleted"))
        if rec.get("creates"):  # the launcher's own per-attempt record (since 2026-10-10 ~02:20)
            failures = [{"region": c["region"], "kind": kind_of(c.get("error") or ""), "minutes": c.get("minutes")}
                        for c in rec["creates"] if c.get("error")]
            start = seconds(rec["creates"][0]["start"]) or start
        else:
            creates = [e for e in rec.get("errors") or [] if e.startswith("create in ")]
            failures = [{"region": e[len("create in "):].split(":", 1)[0], "kind": kind_of(e)} for e in creates]
        if rec.get("preempted"):
            end = "preempted"
        elif rec.get("stopped_state"):
            end = "after the job" if rec.get("exit_code") in (0, 75) else "--hours cap"
        elif rec.get("exit_code") == 0:
            end = "finished"
        elif rec.get("exit_code") == 75:
            end = "stopped early (75)"
        elif rec.get("instance_id") and not t.get("deleted"):
            end = "running"
        elif rec.get("instance_id"):
            end = f"failed ({rec.get('exit_code')})"
        else:
            end = "never created"
        offers = {o["region"]: o.get("price") for o in rec.get("offers") or []}
        out.append({
            "launched": t.get("create_start") or "", "vm": rec.get("name"), "attempt": rec.get("attempt", 1),
            "region": rec.get("region") if rec.get("instance_id") else None,
            "price": offers.get(rec.get("region")), "create_failures": failures,
            "minutes_to_running": round((running - start) / 60, 1) if start and running else None,
            "minutes_running": round(((deleted or running) - running) / 60, 1) if running else None,
            "end": end, "cost": rec.get("estimated_cost_usd"), "path": os.path.relpath(path, root),
            "gpus": next((int(m.group(1)) for m in [re.match(r"(\d+)gpu", (rec.get("flags") or {}).get("preset", ""))]
                          if m), 1),
        })
    return out


def summary(table: list[dict]) -> dict:
    created = [r for r in table if r["region"]]
    vm_hours = sum((r["minutes_running"] or 0) / 60 for r in created)
    gpu_hours = sum((r["minutes_running"] or 0) / 60 * r["gpus"] for r in created)
    preempted = [r for r in created if r["end"] == "preempted"]
    fails: dict = {}
    for r in table:
        for f in r["create_failures"]:
            fails.setdefault(f["kind"], {}).setdefault(f["region"], 0)
            fails[f["kind"]][f["region"]] += 1
    return {"attempts": len(table), "vms_created": len(created), "vm_hours": round(vm_hours, 2),
            "gpu_hours": round(gpu_hours, 2), "preemptions": len(preempted),
            "preemptions_per_vm_hour": round(len(preempted) / vm_hours, 2) if vm_hours else None,
            "minutes_waiting_for_a_vm": round(sum(r["minutes_to_running"] or 0 for r in created), 1),
            "cost_of_preempted_attempts": round(sum(r["cost"] or 0 for r in preempted), 2),
            "creation_failures": fails}


def markdown(table: list[dict], s: dict) -> str:
    lines = ["| launched (UTC) | VM | region | $/h | creation failures | min to run | min ran | end | cost |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        fails = ", ".join(f"{f['region']}: {f['kind']}" for f in r["create_failures"]) or "-"
        lines.append(f"| {r['launched'][5:16].replace('T', ' ')} | {r['vm']} | {r['region'] or '-'} | "
                     f"{r['price'] if r['price'] is not None else '-'} | {fails} | {r['minutes_to_running'] or '-'} | "
                     f"{r['minutes_running'] if r['minutes_running'] is not None else '-'} | {r['end']} | "
                     f"{'-' if r['cost'] is None else f'${r[chr(99) + chr(111) + chr(115) + chr(116)]:.2f}'} |")
    lines += ["", f"**Summary:** {s['vms_created']} VMs from {s['attempts']} attempts; {s['vm_hours']} VM-hours "
                  f"({s['gpu_hours']} GPU-hours); {s['preemptions']} preemptions "
                  f"({s['preemptions_per_vm_hour']} per VM-hour); {s['minutes_waiting_for_a_vm']} min waiting for VMs "
                  f"to run; ${s['cost_of_preempted_attempts']} spent on attempts that were preempted.",
              f"Creation failures: {json.dumps(s['creation_failures'])}"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="the launcher's output folder, e.g. output/gpu_jobs")
    ap.add_argument("--md", help="also write the table here")
    ap.add_argument("--json", help="also write the rows and summary here")
    args = ap.parse_args()
    table = rows(args.root)
    s = summary(table)
    text = markdown(table, s)
    print(text)
    if args.md:
        open(args.md, "w").write(text + "\n")
    if args.json:
        json.dump({"rows": table, "summary": s}, open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
