"""H-Nav Stage 2 gate: verify the Stage 1 headline numbers regenerate.

Re-runs ``evaluate_hnav_stage1.py`` with the exact Stage 1 arguments (same
seed, same explicit input paths -- never globs that could pick up
``features_diff_oracle_*``) into a scratch report, then compares it against
(a) the committed ``gov_logs/hnav_stage1_report.json`` and (b) the
hard-coded EXPECTED headline constants below.  Writes
``gov_logs/hnav_stage2/stage1_verification.json`` and exits nonzero on any
mismatch: expansion work must stop until the discrepancy is diagnosed.

Usage (CWD = berkeley-function-call-leaderboard):
    python bfcl_eval/scripts/verify_stage1.py             # full regen (~6-7 min)
    python bfcl_eval/scripts/verify_stage1.py --check-only  # committed report vs EXPECTED only
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

COMMITTED_REPORT = Path("gov_logs/hnav_stage1_report.json")
REGEN_REPORT = Path("gov_logs/hnav_stage2/stage1_report_regen.json")
REGEN_MD = Path("gov_logs/hnav_stage2/stage1_report_regen.md")
OUT_PATH = Path("gov_logs/hnav_stage2/stage1_verification.json")

# Pinned reproduction command (identical to HNAV_STAGE1_IMPLEMENTATION_SUMMARY.md section 5,
# step 5).  --diff-features listed EXPLICITLY: the features_diff_* glob also matches
# features_diff_oracle_* and silently shadowed the real features once already.
EVAL_ARGS = [
    "bfcl_eval/scripts/evaluate_hnav_stage1.py",
    "--logs",
    "gov_logs/me_harvest/rep0*_v1_shadow",
    "gov_logs/me_ablations/rep0*_geometry_only",
    "gov_logs/me_ablations/rep0*_gm_v1",
    "gov_logs/me_ablations/rep0*_joint_entropy_diag",
    "--labels",
    "gov_logs/me_harvest/outcomes_hnav.jsonl",
    "gov_logs/me_ablations/outcomes_hnav_*.jsonl",
    "--diff-features",
    "gov_logs/me_harvest/features_diff.jsonl",
    "gov_logs/me_ablations/features_diff_geometry_only.jsonl",
    "gov_logs/me_ablations/features_diff_gm_v1.jsonl",
    "gov_logs/me_ablations/features_diff_joint_entropy_diag.jsonl",
    "--extra-features",
    "gov_logs/me_harvest/features_v2.jsonl",
    "--counterfactual",
    "gov_logs/hnav_counterfactual.json",
    "--label", "must_write",
    "--seed", "12345",
    "--n-boot", "2000",
    "--n-perm", "10000",
]

# Headline constants as reported in HNAV_STAGE1_IMPLEMENTATION_SUMMARY.md and the
# Stage 2 directive.  Guards against the committed report and a regen drifting together.
EXPECTED = {
    "n_labeled": 4575,
    "n_positive": 159,
    "p_hat": 0.408544,
    "target_distribution.must_write": 159,
    "target_distribution.inert_superseded": 3905,
    "mechanical_claim.n_near_duplicate_updates": 220,
    "mechanical_claim.median_sim_max": 0.9438,
    "mechanical_claim.median_diff_sim_max": 0.196219,
    "mechanical_claim.n_must_write_among_them": 0,
    "nested_models.dAUC_diff_given_geometry_margin.mean": 0.076378,
    "nested_models.dAUC_diff_given_geometry_margin.ci95": [0.014724, 0.138193],
    "hypotheses.H1.verdict": "FAIL",
    "hypotheses.H2.verdict": "PASS",
    "hypotheses.H3.verdict": "FAIL",
    "verdict.decision": "PIVOT to action-side H_act",
}

# Bootstrap / permutation quantities are seeded, so a same-input regen should be
# bit-identical; the loose tolerance only cushions cross-platform float noise.
FLOAT_TOL = 5e-3
# Sections skipped in the report-vs-regen deep compare.  git_head legitimately moves
# (the regen runs after the preservation commits); data_manifest paths carry shas that
# are compared explicitly instead.
SKIP_KEYS = {"git_head"}


def dig(obj, dotted):
    cur = obj
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def close(a, b, tol=FLOAT_TOL):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= tol
        except (TypeError, ValueError):
            return False
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    return a == b


def deep_compare(a, b, path="", diffs=None, tol=FLOAT_TOL):
    """Recursive compare with float tolerance; returns list of mismatch strings."""
    if diffs is None:
        diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in SKIP_KEYS and not path:
                continue
            p = f"{path}.{k}" if path else k
            if k not in a:
                diffs.append(f"{p}: missing in committed")
            elif k not in b:
                diffs.append(f"{p}: missing in regen")
            else:
                deep_compare(a[k], b[k], p, diffs, tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: list len {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                deep_compare(x, y, f"{path}[{i}]", diffs, tol)
    else:
        if not close(a, b, tol):
            diffs.append(f"{path}: {a!r} != {b!r}")
    return diffs


def check_expected(report, source_name):
    rows = []
    for dotted, want in EXPECTED.items():
        try:
            got = dig(report, dotted)
            ok = close(got, want, tol=1e-4 if isinstance(want, float) else FLOAT_TOL)
        except (KeyError, TypeError):
            got, ok = None, False
        rows.append({"key": dotted, "expected": want, "observed": got,
                     "source": source_name, "match": bool(ok)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true",
                    help="skip the regen subprocess; compare committed report vs EXPECTED only")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    committed = json.loads(COMMITTED_REPORT.read_text(encoding="utf-8"))
    checks = check_expected(committed, "committed_report")
    result = {
        "committed_report": str(COMMITTED_REPORT),
        "reproduction_command": [sys.executable] + EVAL_ARGS
        + ["--out", str(REGEN_REPORT), "--md", str(REGEN_MD)],
        "float_tolerance": FLOAT_TOL,
        "expected_checks": checks,
        "regen_ran": False,
        "regen_vs_committed_diffs": None,
    }

    if not args.check_only:
        REGEN_REPORT.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable] + EVAL_ARGS + ["--out", str(REGEN_REPORT), "--md", str(REGEN_MD)]
        print("[verify_stage1] regenerating:", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            result["regen_error"] = f"evaluate_hnav_stage1 exited {proc.returncode}"
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
            print("[verify_stage1] FAIL: regen subprocess failed")
            sys.exit(1)
        regen = json.loads(REGEN_REPORT.read_text(encoding="utf-8"))
        result["regen_ran"] = True
        diffs = deep_compare(committed, regen)
        result["regen_vs_committed_diffs"] = diffs
        checks.extend(check_expected(regen, "regen_report"))

    ok = all(c["match"] for c in checks) and not result.get("regen_vs_committed_diffs")
    result["match"] = bool(ok)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    n_bad = sum(1 for c in checks if not c["match"])
    print(f"[verify_stage1] expected-value checks: {len(checks) - n_bad}/{len(checks)} pass")
    if result["regen_vs_committed_diffs"] is not None:
        print(f"[verify_stage1] regen-vs-committed diffs: {len(result['regen_vs_committed_diffs'])}")
        for d in (result["regen_vs_committed_diffs"] or [])[:20]:
            print("   ", d)
    for c in checks:
        if not c["match"]:
            print(f"    MISMATCH {c['source']}:{c['key']} expected {c['expected']} got {c['observed']}")
    print(f"[verify_stage1] verdict: {'MATCH' if ok else 'MISMATCH'} -> {args.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
