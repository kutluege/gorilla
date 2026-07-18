"""
Governance replicate analyzer  --  Plan 2 Step 2, extended in Plan 3 Step 7
(multi-arm, G13) and Step 8 (unscored-replicate contract, G15).

Consumes the runner's JSONL manifest plus per-replicate result/score trees
(via parse_wrra) and emits, per (backend x arm-pair):

  - JOINT survival-conditional pairing: a (replicate, backend, scenario) unit
    is DROPPED when the prereq chain died in ANY analyzed arm -- so every
    arm-pair is compared over the SAME surviving subset (risk R6); the drop
    count is a primary statistic;
  - exact McNemar on the matched surviving question pairs, each non-reference
    arm paired against the designated REFERENCE arm (default `baseline`);
  - scenario-level bootstrap CI on the accuracy delta;
  - Holm correction across ALL (backend x arm-pair) McNemar p-values -- with
    two arms this reduces exactly to the Plan 2 per-backend correction;
  - the student pre-registered exclusion (`--include-student` = labeled
    sensitivity switch only);
  - G15 contract: a replicate whose evaluate produced no score file is an
    ERROR, never silently absorbed -- `--allow-unscored` is the explicit,
    stamped opt-out for salvage analyses.

Usage:
  python bfcl_eval/scripts/analyze_gov_replicates.py \
      --manifest result_gov_replicates/manifest.jsonl \
      --out gov_logs/replicate_analysis.json
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

    b = reference-only correct, c = other-only correct. Under H0 the
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
    """Percentile CI for (other - reference) accuracy, resampling units.

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
# Joint survival-conditional pairing (N arms, reference-based)
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


def _pair_rep_generic(rep, records_by_arm, ref_arm, include_student=False):
    """One replicate, N arms, JOINT survival.

    Returns (pairs_by_other, dropped_units, excluded):
      pairs_by_other {other_arm: [{replicate, backend, scenario, id, b, g}]}
          (b = reference correct, g = other-arm correct)
      dropped_units  [{replicate, backend, scenario, reason, dead_in,
                       n_questions}]  -- ONE entry per unit (joint drop)
      excluded       {"student_units", "unmatched_questions",
                      "unscored_questions"}
    """
    arm_labels = list(records_by_arm)
    idx = {arm: index_records(records_by_arm[arm]) for arm in arm_labels}
    others = [a for a in arm_labels if a != ref_arm]
    pairs_by_other = {a: [] for a in others}
    dropped = []
    excluded = {"student_units": 0, "unmatched_questions": 0,
                "unscored_questions": 0}

    keys = sorted(set().union(*(set(idx[a][0]) for a in arm_labels)))
    for key in keys:
        backend, scenario = key
        if scenario == STUDENT_SCENARIO and not include_student:
            excluded["student_units"] += 1
            continue
        dead_in = [
            arm for arm in arm_labels
            if idx[arm][0].get(key) is None or idx[arm][0][key]["chain_dead"]
        ]
        n_q = max((idx[arm][0][key]["question_entries"]
                   for arm in arm_labels if idx[arm][0].get(key)), default=0)
        if dead_in:
            dropped.append({
                "replicate": rep, "backend": backend, "scenario": scenario,
                "reason": "chain_dead", "dead_in": dead_in,
                "n_questions": n_q,
            })
            continue
        ref_q = idx[ref_arm][1].get(key, {})
        for other in others:
            oth_q = idx[other][1].get(key, {})
            for qid in sorted(set(ref_q) | set(oth_q)):
                if qid not in ref_q or qid not in oth_q:
                    excluded["unmatched_questions"] += 1
                    continue
                b_ok, g_ok = ref_q[qid]["correct"], oth_q[qid]["correct"]
                if b_ok is None or g_ok is None:
                    excluded["unscored_questions"] += 1
                    continue
                pairs_by_other[other].append({
                    "replicate": rep, "backend": backend, "scenario": scenario,
                    "id": qid, "b": bool(b_ok), "g": bool(g_ok),
                })
    return pairs_by_other, dropped, excluded


def pair_replicate(rep, base_records, gov_records, include_student=False):
    """Two-arm compatibility wrapper (Plan 2 shape): (pairs, dropped, excluded)."""
    pairs_by_other, dropped, excluded = _pair_rep_generic(
        rep,
        {BASELINE_ARM: base_records, GOVERNED_ARM: gov_records},
        BASELINE_ARM,
        include_student=include_student,
    )
    return pairs_by_other[GOVERNED_ARM], dropped, excluded


def comparison_stats(pairs, seed, n_boot):
    b = sum(1 for p in pairs if p["b"] and not p["g"])
    c = sum(1 for p in pairs if p["g"] and not p["b"])
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
    return {
        "n_pairs": n,
        "n_units": len(unit_keys),
        "acc_baseline": (sum(p["b"] for p in pairs) / n) if n else None,
        "acc_governed": (sum(p["g"] for p in pairs) / n) if n else None,
        "discordant_baseline_only": b,
        "discordant_governed_only": c,
        "mcnemar_p": mcnemar_exact(b, c),
        "bootstrap_delta": delta,
        "bootstrap_ci95": [lo, hi],
    }


def analyze(records_by_rep, include_student=False, seed=12345, n_boot=10000,
            ref_arm=BASELINE_ARM):
    """records_by_rep: {rep: {arm_label: [wrra records]}}. N arms; every
    non-reference arm is compared against `ref_arm` over the JOINTLY surviving
    units. Holm runs across backend x arm-pair."""
    reps = sorted(records_by_rep)
    if not reps:
        raise SystemExit("no replicates to analyze")
    arm_labels = list(records_by_rep[reps[0]])
    if ref_arm not in arm_labels:
        raise SystemExit(f"reference arm '{ref_arm}' not in arms {arm_labels}")
    others = [a for a in arm_labels if a != ref_arm]

    pairs_by_other = {a: [] for a in others}
    all_drops = []
    excluded = {"student_units": 0, "unmatched_questions": 0,
                "unscored_questions": 0}
    for rep in reps:
        rep_pairs, drops, excl = _pair_rep_generic(
            rep, records_by_rep[rep], ref_arm, include_student=include_student)
        for other in others:
            pairs_by_other[other].extend(rep_pairs[other])
        all_drops.extend(drops)
        for k in excluded:
            excluded[k] += excl[k]

    backends = sorted(
        {p["backend"] for ps in pairs_by_other.values() for p in ps}
        | {d["backend"] for d in all_drops}
    )
    comparisons, pvals = {}, {}
    for other in others:
        for backend in backends:
            pairs = [p for p in pairs_by_other[other] if p["backend"] == backend]
            drops = [d for d in all_drops if d["backend"] == backend]
            key = f"{backend}|{other}_vs_{ref_arm}"
            stats = comparison_stats(pairs, seed, n_boot)
            stats.update({
                "backend": backend, "arm": other, "reference": ref_arm,
                "n_units_dropped": len(drops),
                "dropped_units": drops,
            })
            comparisons[key] = stats
            pvals[key] = stats["mcnemar_p"]

    adjusted = holm(pvals) if pvals else {}
    for key, p_adj in adjusted.items():
        comparisons[key]["mcnemar_p_holm"] = p_adj

    summary = {
        "include_student": include_student,
        "primary_result": not include_student,
        "n_replicates": len(reps),
        "arms": arm_labels,
        "reference_arm": ref_arm,
        "seed": seed,
        "n_boot": n_boot,
        "total_units_dropped": len(all_drops),
        "excluded": excluded,
        "comparisons": comparisons,
        "holm_m": len(pvals),
    }
    if len(others) == 1:
        # Plan 2 two-arm shape: per_backend[backend] with numbers identical to
        # the pre-G13 analyzer (Holm over backend x one pair == over backends).
        summary["per_backend"] = {
            comparisons[k]["backend"]: comparisons[k] for k in comparisons
        }
    return summary


# ---------------------------------------------------------------------------
# Manifest-driven CLI
# ---------------------------------------------------------------------------

def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def completed_replicates(manifest_records, arms=None):
    """Replicates whose evaluate phase exited 0 for EVERY analyzed arm."""
    required = set(arms) if arms else {BASELINE_ARM, GOVERNED_ARM}
    done = {}
    for r in manifest_records:
        if (r.get("event") == "cmd_end" and r.get("phase") == "evaluate"
                and r.get("exit_code") == 0):
            done.setdefault(r["replicate"], set()).add(r["arm"])
    return sorted(rep for rep, got in done.items() if required <= got)


def unscored_replicates(records_by_rep):
    """G15: (rep, arm, backend) triples whose questions ALL lack scores --
    the signature of a missing score file for an otherwise-complete arm."""
    out = []
    for rep in sorted(records_by_rep):
        for arm, records in records_by_rep[rep].items():
            per_backend = {}
            for r in records:
                if r.get("record") == "wrra_question":
                    per_backend.setdefault(r["backend"], []).append(r["correct"])
            for backend, corrects in sorted(per_backend.items()):
                if corrects and all(c is None for c in corrects):
                    out.append((rep, arm, backend))
    return out


def enforce_scored(unscored, allow_unscored):
    """G15: a replicate without a score is NOT complete. Raises SystemExit
    naming every unscored (rep, arm, backend) unless explicitly allowed."""
    if unscored and not allow_unscored:
        detail = ", ".join(f"rep{r:02d}/{a}/{b}" for r, a, b in unscored)
        raise SystemExit(
            f"[gov-analyze] G15: unscored replicate arm(s): {detail}. "
            "A replicate without a score is NOT complete -- run `bfcl "
            "evaluate` for it, or pass --allow-unscored for an explicitly "
            "stamped salvage analysis."
        )


def gather_records(manifest_records, backends):
    starts = [r for r in manifest_records if r.get("event") == "run_start"]
    if not starts:
        raise SystemExit("manifest has no run_start record")
    cfg = starts[-1]
    arms = [a["label"] for a in cfg.get("arms", [])] or list(cfg["models"])
    models = {a["label"]: a["model"] for a in cfg.get("arms", [])} or cfg["models"]
    reps = completed_replicates(manifest_records, arms=arms)
    if not reps:
        raise SystemExit("manifest shows no replicate completed in all arms")

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else BFCL_ROOT / p

    records_by_rep = {}
    for rep in reps:
        records_by_rep[rep] = {}
        for arm in arms:
            result_root = resolve(cfg["result_root"]) / f"rep{rep:02d}" / arm
            score_root = resolve(cfg["score_root"]) / f"rep{rep:02d}" / arm
            records_by_rep[rep][arm] = parse_arm(
                result_root, score_root, models[arm], backends, arm=arm)
    return cfg, reps, records_by_rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--backends", default="kv,vector")
    ap.add_argument("--reference-arm", default=BASELINE_ARM)
    ap.add_argument("--include-student", action="store_true",
                    help="SENSITIVITY ONLY: lift the pre-registered exclusion")
    ap.add_argument("--allow-unscored", action="store_true",
                    help="G15 opt-out: analyze despite missing score files "
                         "(salvage only; the output is stamped)")
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

    unscored = unscored_replicates(records_by_rep)
    enforce_scored(unscored, args.allow_unscored)

    summary = analyze(records_by_rep, include_student=args.include_student,
                      seed=args.seed, n_boot=args.n_boot,
                      ref_arm=args.reference_arm)
    summary["manifest"] = str(manifest_path)
    summary["run_id"] = cfg.get("run_id")
    summary["git_head"] = cfg.get("git_head")
    summary["served_models"] = cfg.get("served_models")
    summary["replicates_analyzed"] = reps
    summary["allow_unscored"] = bool(unscored) and args.allow_unscored
    summary["unscored"] = [f"rep{r:02d}/{a}/{b}" for r, a, b in unscored]

    label = "SENSITIVITY (student included)" if args.include_student else "primary"
    print(f"[gov-analyze] {label} | arms={summary['arms']} "
          f"(ref={summary['reference_arm']}) | replicates: {reps} | "
          f"units dropped (joint): {summary['total_units_dropped']} | "
          f"Holm m={summary['holm_m']}")
    for key, s in summary["comparisons"].items():
        if s["n_pairs"]:
            print(
                f"  {key}: n_pairs={s['n_pairs']} drops={s['n_units_dropped']} "
                f"acc ref={s['acc_baseline']:.4f} arm={s['acc_governed']:.4f} | "
                f"b/c={s['discordant_baseline_only']}/{s['discordant_governed_only']} "
                f"McNemar p={s['mcnemar_p']:.4f} (Holm {s['mcnemar_p_holm']:.4f}) | "
                f"dAcc={s['bootstrap_delta']:+.4f} "
                f"CI95=[{s['bootstrap_ci95'][0]:+.4f}, {s['bootstrap_ci95'][1]:+.4f}]"
            )
        else:
            print(f"  {key}: no surviving pairs (drops={s['n_units_dropped']})")

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
