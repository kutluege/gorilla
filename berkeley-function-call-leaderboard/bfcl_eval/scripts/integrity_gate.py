"""Benchmark-integrity gate (directive §14 / §3.1).

Proves the benchmark surface is untouched before a campaign launches and for
the final report: `bfcl_eval/data/**` and `bfcl_eval/eval_checker/**` must be
byte-identical to their state at the H-Nav baseline commit, and the working
tree must carry no modifications to them either.

    python bfcl_eval/scripts/integrity_gate.py \
        --since b0b7d12 --out gov_logs/hnav_autonomous/integrity_gate.json

Exit 1 on any diff -- suitable as a launcher pre-flight gate.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BFCL = "berkeley-function-call-leaderboard"
PROTECTED = [f"{BFCL}/bfcl_eval/data", f"{BFCL}/bfcl_eval/eval_checker"]
# Grader + gold files hashed individually for the report.
HASH_FILES = [
    f"{BFCL}/bfcl_eval/data/BFCL_v4_memory.json",
    f"{BFCL}/bfcl_eval/data/possible_answer/BFCL_v4_memory.json",
    f"{BFCL}/bfcl_eval/eval_checker/agentic_eval/agentic_checker.py",
    f"{BFCL}/bfcl_eval/eval_checker/eval_runner.py",
    f"{BFCL}/bfcl_eval/eval_checker/eval_runner_helper.py",
]
GIT_ROOT = REPO_ROOT.parent  # gorilla/


def run_git(*args):
    r = subprocess.run(["git", *args], cwd=GIT_ROOT, capture_output=True,
                       text=True)
    return r.returncode, r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None,
                    help="commit to diff against (default: HEAD only, i.e. "
                         "working-tree check)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = {"protected": PROTECTED, "clean": True, "checks": {}}
    _, head = run_git("rev-parse", "HEAD")
    report["head"] = head

    # 1. working tree must not modify protected paths
    _, wt = run_git("status", "--short", "--", *PROTECTED)
    report["checks"]["working_tree"] = wt or "clean"
    if wt:
        report["clean"] = False

    # 2. committed history since --since must not touch protected paths
    if args.since:
        code, diff = run_git("diff", "--stat", f"{args.since}..HEAD", "--",
                             *PROTECTED)
        report["checks"][f"diff_{args.since}..HEAD"] = diff or "clean"
        if code != 0 or diff:
            report["clean"] = False

    # 3. content hashes of the grader + gold files
    hashes = {}
    for rel in HASH_FILES:
        p = GIT_ROOT / rel
        if p.exists():
            hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        else:
            hashes[rel] = "MISSING"
            report["clean"] = False
    report["hashes_sha256_16"] = hashes

    if args.out:
        out = REPO_ROOT / args.out if not Path(args.out).is_absolute() \
            else Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["clean"] else 1)


if __name__ == "__main__":
    main()
