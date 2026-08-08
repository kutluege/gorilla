"""Step-budget guard (P0.4 of the RAG program pre-registration).

Every intervention in the RAG program adds model steps to some entries.
MAXIMUM_STEP_LIMIT = 20; exceeding it sets force_quit, which aborts ALL
remaining turns of the entry -- for a question entry that is a guaranteed
failure, and for a prereq entry it kills the rest of the chain's writes.
An arm that looks worse on accuracy may simply be tripping the step cap.

This instrument emits, per (result tree, backend, entry): the max step count
over turns and whether the entry was force-quit, plus per-arm totals.

PRE-REGISTERED RULE (frozen before any RAG campaign): any arm whose
force_quit rate exceeds 3x its within-campaign baseline arm's rate has its
accuracy comparison INVALIDATED until re-run with the offending policy
budget-capped. The rule is a validity gate, not a tuning knob.

    python bfcl_eval/scripts/step_budget_guard.py \
        --result-dirs result_hnav_stage4/rep01/baseline/Qwen_Qwen3-4B-Instruct-2507-FC \
        --out gov_logs/hnav_autonomous/step_budget_baseline.json
"""

import argparse
import glob as globmod
import json
from collections import defaultdict
from pathlib import Path

FORCE_QUIT_MARK = "exceeded the maximum number of steps"


def scan_result_file(path: Path):
    """Yield (entry_id, max_steps, force_quit) per entry in a result file."""
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        turns = rec.get("result") or []
        max_steps = 0
        fq = False
        for turn in turns:
            steps = turn if isinstance(turn, list) else [turn]
            max_steps = max(max_steps, len(steps))
            for s in steps:
                if isinstance(s, str) and FORCE_QUIT_MARK in s:
                    fq = True
        # the harness also records force-quit in inference_log if present
        il = rec.get("inference_log")
        if il and FORCE_QUIT_MARK in json.dumps(il):
            fq = True
        yield rec.get("id", "?"), max_steps, fq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dirs", nargs="+", required=True,
                    help="model-slug-level result dirs (globs ok)")
    ap.add_argument("--backends", default="kv,vector")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dirs = sorted(set(p for g in args.result_dirs for p in globmod.glob(g)))
    backends = args.backends.split(",")
    report = {"dirs": {}, "rule": "arm force_quit rate > 3x within-campaign "
              "baseline rate invalidates the accuracy comparison"}
    for d in dirs:
        d = Path(d)
        per_dir = {}
        for backend in backends:
            base = d / "agentic" / "memory" / backend
            counts = defaultdict(int)
            fq_entries = []
            step_hist = defaultdict(int)
            for kind, fname in (
                ("prereq", f"BFCL_v4_memory_{backend}_prereq_result.json"),
                ("question", f"BFCL_v4_memory_{backend}_result.json"),
            ):
                f = base / fname
                if not f.exists():
                    continue
                for eid, max_steps, fq in scan_result_file(f):
                    counts[f"{kind}_entries"] += 1
                    step_hist[max_steps] += 1
                    if fq:
                        counts[f"{kind}_force_quit"] += 1
                        fq_entries.append(eid)
            per_dir[backend] = {
                **counts,
                "force_quit_entries": fq_entries,
                "max_steps_histogram": dict(sorted(step_hist.items())),
            }
        report["dirs"][str(d)] = per_dir
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for d, per in report["dirs"].items():
        for b, v in per.items():
            print(f"{d} {b}: prereq_fq={v.get('prereq_force_quit', 0)}"
                  f"/{v.get('prereq_entries', 0)}"
                  f" question_fq={v.get('question_force_quit', 0)}"
                  f"/{v.get('question_entries', 0)}")
    print(f"[step-budget] wrote {out}")


if __name__ == "__main__":
    main()
