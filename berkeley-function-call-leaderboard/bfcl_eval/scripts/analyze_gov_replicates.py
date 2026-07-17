"""
Governance replicate analyzer  --  Plan 2 Step 2 (v2 SS8.3).

Consumes the runner's JSONL manifest plus per-replicate result/score trees
(via parse_wrra) and emits, per backend:

  - survival-conditional pairing: a (replicate, scenario) unit is DROPPED
    when the prereq chain died in EITHER arm; the drop count is a primary
    statistic, not a footnote;
  - exact McNemar on the matched surviving question pairs;
  - scenario-level bootstrap CI on the accuracy delta (resampling
    (replicate, scenario) units, so per-question correlation within a
    scenario never fakes precision);
  - Holm correction across the per-backend McNemar p-values;
  - the student pre-registered exclusion. `--include-student` is a labeled
    sensitivity switch only: the output is stamped include_student=true and
    must never be reported as the primary result.

Usage:
  python bfcl_eval/scripts/analyze_gov_replicates.py \
      --manifest result_gov_replicates/manifest.jsonl \
      --out gov_logs/replicate_analysis.json
  # sensitivity only:
  ... --include-student
"""

import argparse
import json
import random
import sys
from math import comb
from pathlib import Path

BFCL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BFCL_ROOT))

from bfcl_eval.scripts.parse_wrra import parse_arm  # noqa: E402

STUDENT_SCENARIO = "student"
BASELINE_ARM = "baseline"
GOVERNED_ARM = "governed"


# ---------------------------------------------------------------------------
# Statistics (pure, unit-tested)
# ---------------------------------------------------------------------------

def mcnemar_exact(b, c):
    """Exact two-sided McNemar p on the discordant counts.

    b = baseline-only correct, c = governed-only correct. Under H0 the
    discordant pairs are Binomial(n=b+c, 0.5); p = 2 * P(X <= min(b, c)),
    clipped at 1. n == 0 -> p = 1.0 (no evidence either way).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def holm(pvals):
    """Holm step-down adjustment. {label: p} -> {label: p_adjusted}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running_max = {}, 0.0
    for i, (label, p) in enumerate(items):
        running_max = max(running_max, (m - i) * p)
        adjusted[label] = min(1.0, running_max)
    return adjusted


def bootstrap_delta_ci(units, n_boot=10000, seed=12345, alpha=0.05):
    """Percentile CI for (governed - baseline) accuracy, resampling units.

    `units` is a list of (replicate, scenario) aggregates:
        {"b_correct": int, "g_correct": int, "n": int}
    Each bootstrap draw resamples the units with replacement and computes the
    pooled accuracy delta over the drawn units.
    Returns (delta_observed, lo, hi) or (None, None, None) with no data.
    """
    units = [u for u in units if u["n"] > 0]
    if not units:
        return None, None, None

    def pooled_delta(sample):
        n = sum(u["n"] for u in sample)
        return (sum(u["g_correct"] for u in sample)
                - sum(u["b_correct"] for u in sample)) / n

    observed = pooled_delta(units)
    rng = random.Random(seed)
    draws = sorted(
        pooled_delta([units[rng.randrange(len(units))]
                      for _ in range(len(units))])
        for _ in range(n_boot)
    )
    lo = draws[int((alpha / 2) * n_boot)]
    hi = draws[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return observed, lo, hi


# ---------------------------------------------------------------------------
# Survival-conditional pairing
# ---------------------------------------------------------------------------

def index_records(records):
    """wrra records of one (replicate, arm) -> per-backend scenario/question maps."""
    scen, quest = {}, {}
    for r in records:
        if r.get("record") == "wrra_scenario":
            scen[(r["backend"], r["scenario"])] = r
        elif r.get("record") == "wrra_question":
            quest.setdefault((r["backend"], r["scenario"]), {})[r["id"]] = r
    return scen, quest


def pair_replicate(rep, base_records, gov_records, include_student=False):
    """Survival-conditional pairing for one replicate.

    Returns (pairs, dropped_units, excluded):
      pairs         [{replicate, backend, scenario, id, b, g}]  (b/g: bool)
      dropped_units [{replicate, backend, scenario, reason, dead_in, n_questions}]
      excluded      {"student_units": int, "unmatched_questions": int,
                     "unscored_questions": int}
    """
    b_scen, b_quest = index_records(base_records)
    g_scen, g_quest = index_records(gov_records)
    pairs, dropped = [], []
    excluded = {"student_units": 0, "unmatched_questions": 0,
                "unscored_questions": 0}

    for key in sorted(set(b_scen) | set(g_scen)):
        backend, scenario = key
        if scenario == STUDENT_SCENARIO and not include_student:
            excluded["student_units"] += 1
            continue

        b_rec, g_rec = b_scen.get(key), g_scen.get(key)
        dead_in = [
            arm for arm, rec in ((BASELINE_ARM, b_rec), (GOVERNED_ARM, g_rec))
            if rec is None or rec["chain_dead"]
        ]
        n_q = max(
            b_rec["question_entries"] if b_rec else 0,
            g_rec["question_entries"] if g_rec else 0,
        )
        if dead_in:
            dropped.append({
                "replicate": rep, "backend": backend, "scenario": scenario,
                "reason": "chain_dead", "dead_in": dead_in,
                "n_questions": n_q,
            })
            continue

        bq, gq = b_quest.get(key, {}), g_quest.get(key, {})
        for qid in sorted(set(bq) | set(gq)):
            if qid not in bq or qid not in gq:
                excluded["unmatched_questions"] += 1
                continue
            b_ok, g_ok = bq[qid]["correct"], gq[qid]["correct"]
            if b_ok is None or g_ok is None:
                excluded["unscored_questions"] += 1
                continue
            pairs.append({
                "replicate": rep, "backend": backend, "scenario": scenario,
                "id": qid, "b": bool(b_ok), "g": bool(g_ok),
            })
    return pairs, dropped, excluded


def analyze(records_by_rep, include_student=False, seed=12345, n_boot=10000):
    """records_by_rep: {rep: {"baseline": [records], "governed": [records]}}."""
    all_pairs, all_drops = [], []
    excluded = {"student_units": 0, "unmatched_questions": 0,
                "unscored_questions": 0}
    for rep in sorted(records_by_rep):
        arms = records_by_rep[rep]
        pairs, drops, excl = pair_replicate(
            rep, arms[BASELINE_ARM], arms[GOVERNED_ARM],
            include_student=include_student,
        )
        all_pairs.extend(pairs)
        all_drops.extend(drops)
        for k in excluded:
            excluded[k] += excl[k]

    backends = sorted({p["backend"] for p in all_pairs}
                      | {d["backend"] for d in all_drops})
    per_backend, mcnemar_ps = {}, {}
    for backend in backends:
        pairs = [p for p in all_pairs if p["backend"] == backend]
        drops = [d for d in all_drops if d["backend"] == backend]
        b = sum(1 for p in pairs if p["b"] and not p["g"])
        c = sum(1 for p in pairs if p["g"] and not p["b"])
        p_mcnemar = mcnemar_exact(b, c)
        mcnemar_ps[backend] = p_mcnemar

        unit_keys = sorted({(p["replicate"], p["scenario"]) for p in pairs})
        units = []
        for rep, scenario in unit_keys:
            sub = [p for p in pairs
                   if p["replicate"] == rep and p["scenario"] == scenario]
            units.append({
                "b_correct": sum(p["b"] for p in sub),
                "g_correct": sum(p["g"] for p in sub),
                "n": len(sub),
            })
        delta, lo, hi = bootstrap_delta_ci(units, n_boot=n_boot, seed=seed)

        n = len(pairs)
        per_backend[backend] = {
            "n_pairs": n,
            "n_units": len(unit_keys),
            "n_units_dropped": len(drops),
            "dropped_units": drops,
            "acc_baseline": (sum(p["b"] for p in pairs) / n) if n else None,
            "acc_governed": (sum(p["g"] for p in pairs) / n) if n else None,
            "discordant_baseline_only": b,
            "discordant_governed_only": c,
            "mcnemar_p": p_mcnemar,
            "bootstrap_delta": delta,
            "bootstrap_ci95": [lo, hi],
        }

    adjusted = holm(mcnemar_ps) if mcnemar_ps else {}
    for backend, p_adj in adjusted.items():
        per_backend[backend]["mcnemar_p_holm"] = p_adj

    return {
        "include_student": include_student,
        "primary_result": not include_student,
        "n_replicates": len(records_by_rep),
        "seed": seed,
        "n_boot": n_boot,
        "total_units_dropped": len(all_drops),
        "excluded": excluded,
        "per_backend": per_backend,
    }


# ---------------------------------------------------------------------------
# Manifest-driven CLI
# ---------------------------------------------------------------------------

def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def completed_replicates(manifest_records):
    """Replicates whose evaluate phase exited 0 for BOTH arms."""
    done = {}
    for r in manifest_records:
        if (r.get("event") == "cmd_end" and r.get("phase") == "evaluate"
                and r.get("exit_code") == 0):
            done.setdefault(r["replicate"], set()).add(r["arm"])
    return sorted(rep for rep, arms in done.items()
                  if {BASELINE_ARM, GOVERNED_ARM} <= arms)


def gather_records(manifest_records, backends):
    starts = [r for r in manifest_records if r.get("event") == "run_start"]
    if not starts:
        raise SystemExit("manifest has no run_start record")
    cfg = starts[-1]
    reps = completed_replicates(manifest_records)
    if not reps:
        raise SystemExit("manifest shows no replicate completed in both arms")

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else BFCL_ROOT / p

    records_by_rep = {}
    for rep in reps:
        result_root = resolve(cfg["result_root"]) / f"rep{rep:02d}"
        score_root = resolve(cfg["score_root"]) / f"rep{rep:02d}"
        records_by_rep[rep] = {
            arm: parse_arm(result_root, score_root, cfg["models"][arm],
                           backends, arm=arm)
            for arm in (BASELINE_ARM, GOVERNED_ARM)
        }
    return cfg, reps, records_by_rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--backends", default="kv,vector")
    ap.add_argument("--include-student", action="store_true",
                    help="SENSITIVITY ONLY: lift the pre-registered exclusion")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default=None, help="Write the JSON summary here")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = BFCL_ROOT / manifest_path
    manifest_records = load_manifest(manifest_path)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    cfg, reps, records_by_rep = gather_records(manifest_records, backends)
    summary = analyze(records_by_rep, include_student=args.include_student,
                      seed=args.seed, n_boot=args.n_boot)
    summary["manifest"] = str(manifest_path)
    summary["run_id"] = cfg.get("run_id")
    summary["git_head"] = cfg.get("git_head")
    summary["served_models"] = cfg.get("served_models")
    summary["replicates_analyzed"] = reps

    label = "SENSITIVITY (student included)" if args.include_student else "primary"
    print(f"[gov-analyze] {label} | replicates: {reps} | "
          f"units dropped: {summary['total_units_dropped']}")
    for backend, s in summary["per_backend"].items():
        print(
            f"  {backend}: n_pairs={s['n_pairs']} "
            f"drops={s['n_units_dropped']} "
            f"acc B={s['acc_baseline']:.4f} G={s['acc_governed']:.4f} | "
            f"b/c={s['discordant_baseline_only']}/{s['discordant_governed_only']} "
            f"McNemar p={s['mcnemar_p']:.4f} (Holm {s['mcnemar_p_holm']:.4f}) | "
            f"dAcc={s['bootstrap_delta']:+.4f} "
            f"CI95=[{s['bootstrap_ci95'][0]:+.4f}, {s['bootstrap_ci95'][1]:+.4f}]"
            if s["n_pairs"] else
            f"  {backend}: no surviving pairs (drops={s['n_units_dropped']})"
        )

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = BFCL_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[gov-analyze] summary -> {out}")


if __name__ == "__main__":
    main()
