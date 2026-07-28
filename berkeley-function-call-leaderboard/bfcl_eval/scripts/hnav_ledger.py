"""Append-only experiment ledger for the H-Nav autonomous alternative search.

Every hypothesis in the (max 10) alternative loop gets ledger entries as it
moves through phases; entries are never rewritten, only appended -- the full
decision history stays auditable. File:
    gov_logs/hnav_autonomous/experiment_ledger.jsonl

Phases (monotonic per experiment_id):
    proposed -> falsifier_running -> (rejected | full_run) ->
    (validated | promising | inconclusive | rejected)

Usage:
    from hnav_ledger import append_entry, load_ledger, latest_status, assert_monotonic
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

LEDGER_PATH = Path("gov_logs/hnav_autonomous/experiment_ledger.jsonl")

STATUSES = ("proposed", "falsifier_running", "rejected", "full_run",
            "promising", "validated", "inconclusive")
_ORDER = {s: i for i, s in enumerate(
    ("proposed", "falsifier_running", "full_run",
     "promising", "validated", "inconclusive", "rejected"))}
# terminal states: nothing may follow them
_TERMINAL = {"rejected", "validated", "inconclusive"}


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def append_entry(entry, ledger_path=LEDGER_PATH):
    """Validate + append one ledger entry; returns the stored record.

    Required: experiment_id, hypothesis_number (1..10), name, status.
    Recommended: mechanism, pre_registered_hypothesis, primary_metric,
    primary_test, development_data, held_out_data, controls,
    negative_controls, commands, artifacts, result_summary, effect_size,
    confidence_interval, p_value, adjusted_p_value, cost_summary, decision,
    next_action.
    """
    for k in ("experiment_id", "hypothesis_number", "name", "status"):
        if k not in entry:
            raise ValueError(f"ledger entry missing required field {k!r}")
    if entry["status"] not in STATUSES:
        raise ValueError(f"bad status {entry['status']!r}; allowed: {STATUSES}")
    n = entry["hypothesis_number"]
    if not (isinstance(n, int) and 1 <= n <= 10):
        raise ValueError("hypothesis_number must be an int in 1..10")

    ledger_path = Path(ledger_path)
    existing = load_ledger(ledger_path)
    prior = [e for e in existing if e["experiment_id"] == entry["experiment_id"]]
    if prior:
        last = prior[-1]["status"]
        if last in _TERMINAL:
            raise ValueError(
                f"experiment {entry['experiment_id']} already terminal ({last})")
        if _ORDER[entry["status"]] < _ORDER[last] and entry["status"] != "rejected":
            raise ValueError(
                f"status regression {last} -> {entry['status']} "
                f"for {entry['experiment_id']}")
    # exhaustion guard: at most 10 distinct completed alternatives
    distinct = {e["experiment_id"] for e in existing} | {entry["experiment_id"]}
    if len(distinct) > 10:
        raise ValueError("more than 10 distinct alternatives is not allowed")

    rec = dict(entry)
    rec["ts"] = datetime.now(timezone.utc).isoformat()
    rec["git_head"] = _git_head()
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return rec


def load_ledger(ledger_path=LEDGER_PATH):
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        return []
    out = []
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def latest_status(ledger_path=LEDGER_PATH):
    """{experiment_id: last entry} in append order."""
    latest = {}
    for e in load_ledger(ledger_path):
        latest[e["experiment_id"]] = e
    return latest


def assert_monotonic(entries):
    """Re-validate a loaded ledger: per-experiment phases only advance."""
    seen = {}
    for e in entries:
        eid, st = e["experiment_id"], e["status"]
        if eid in seen:
            last = seen[eid]
            if last in _TERMINAL:
                raise AssertionError(f"{eid}: entry after terminal {last}")
            if _ORDER[st] < _ORDER[last] and st != "rejected":
                raise AssertionError(f"{eid}: regression {last} -> {st}")
        seen[eid] = st
    return True


if __name__ == "__main__":
    entries = load_ledger()
    assert_monotonic(entries)
    for eid, e in latest_status().items():
        print(f"{eid:28s} #{e['hypothesis_number']:<2d} {e['status']:18s} "
              f"{e.get('decision', '')}")
    print(f"{len(entries)} entries, {len(latest_status())} experiments")
