"""
H-Nav Stage 1.2: counterfactual suppressed-update evaluation.

At the frozen thresholds the gate almost never suppresses (34 NOOPs in 4584
logged decisions) because the residual condition ``r < delta`` essentially never
fires. Measuring write-side headroom from those 34 events alone is hopeless, so
this sweeps the Stage-0 NOOP region over ``(sim_high, delta)`` and asks, for
every cell: *if geometry had been trusted this much, what would suppression have
cost?*

Region membership replicates ``governance_filter.decide`` exactly
(:724-733)::

    r < delta  and  sim_max > sim_high  and  len(verbatim_misses) == 0
                                        and  preflight_ok

Labels come from ``label_outcomes_hnav.py``; nothing is recomputed here, so the
whole sweep re-runs in seconds.

Per cell:
  n_region / coverage              how much of the write stream is suppressed
  n_harmful_noop                   must_write rows suppressed -- the cost
  n_correct_noop                   rows that were safe to suppress
  n_damage_avoided                 must_suppress rows correctly suppressed
  suppression_precision  (+Wilson) safe / decided -- the deployable quantity
  expected_dAcc                    (damage_avoided - harmful) * p_hat / n_q,
                                   using the measured proxy conversion factor
                                   from hnav_answer_index --validate

Counts in the tight cells are small; Wilson intervals are reported instead of
bootstrap CIs, and no cell is a "finding" without the Holm correction applied by
``evaluate_hnav_stage1.py``.

Run:
  python bfcl_eval/scripts/counterfactual_noop.py \
      --labels "gov_logs/me_harvest/outcomes_hnav.jsonl" \
               "gov_logs/me_ablations/outcomes_hnav_*.jsonl" \
      --proxy gov_logs/hnav_proxy_validation.json \
      --out gov_logs/hnav_counterfactual.json
"""

import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

SIM_HIGH_GRID = [0.800, 0.825, 0.850, 0.875, 0.900, 0.925, 0.950, 0.975]
DELTA_GRID = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# Scenario excluded from every headline number (its chains never survive).
DEAD_SCENARIOS = ("student",)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.959963984540054):
    """Wilson score interval for a binomial proportion. Correct at n=0 and at
    k in {0, n}, where the normal approximation is not."""
    if n <= 0:
        return None, None, None
    p = k / n
    d = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(p, 6), round(max(0.0, center - half), 6), round(min(1.0, center + half), 6)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_labels(patterns: List[str], include_student=False) -> List[dict]:
    rows, seen = [], set()
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if not include_student and r.get("scenario") in DEAD_SCENARIOS:
                        continue
                    key = (r.get("arm"), r.get("replicate"), r.get("candidate_id"))
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(r)
    return rows


def in_region(row: dict, sim_high: float, delta: float, veto: bool,
              preflight: bool) -> bool:
    """``governance_filter.decide``'s NOOP condition, byte-for-byte.

    Both side conditions are sweep axes rather than hardcoded filters, because
    each removes most of the population and the two questions differ:

    ``preflight=True, veto=True``  -- the DEPLOYABLE region: what the gate could
        actually suppress. Measured on the harvest, ``preflight_ok`` is False
        for 57 % of decisions (the backend would reject the call anyway), so
        this region is very small; that smallness is itself a Stage-1 finding.
    ``preflight=False, veto=False`` -- the GEOMETRIC region: everything
        whole-blob cosine calls a duplicate. This is the population the
        diff-aware falsifier has to discriminate on, so it carries the H2
        signal claim.
    """
    sim_max, r = row.get("sim_max"), row.get("r")
    if sim_max is None or r is None:
        return False
    if preflight and not row.get("preflight_ok"):
        return False
    if veto and row.get("n_verbatim_misses", 0) != 0:
        return False
    return r < delta and sim_max > sim_high


# ---------------------------------------------------------------------------
# Cell evaluation
# ---------------------------------------------------------------------------


def cell_stats(region: List[dict], n_total: int, p_hat: Optional[float],
               n_questions: int) -> dict:
    counts = defaultdict(int)
    for r in region:
        counts[r["hnav_target"]] += 1
    n_uncertain = counts["uncertain"]
    n_harmful = counts["must_write"]
    n_damage_avoided = counts["must_suppress"]
    n_safe = counts["may_suppress"] + counts["inert_superseded"] + n_damage_avoided
    decided = n_harmful + n_safe
    prec, prec_lo, prec_hi = wilson(n_safe, decided)
    harm, harm_lo, harm_hi = wilson(n_harmful, decided)
    n_lineage = sum(r.get("must_write_lineage", 0) for r in region)
    out = {
        "n_region": len(region),
        "coverage": round(len(region) / n_total, 6) if n_total else 0.0,
        "n_decided": decided,
        "n_uncertain": n_uncertain,
        "n_harmful_noop": n_harmful,
        "n_correct_noop": n_safe,
        "n_damage_avoided": n_damage_avoided,
        "n_inert_superseded": counts["inert_superseded"],
        "n_harmful_lineage": n_lineage,
        "suppression_precision": prec,
        "suppression_precision_wilson95": [prec_lo, prec_hi],
        "harmful_rate": harm,
        "harmful_rate_wilson95": [harm_lo, harm_hi],
    }
    if p_hat is not None and n_questions:
        out["expected_dAcc"] = round(
            (n_damage_avoided - n_harmful) * p_hat / n_questions, 6
        )
        out["expected_dAcc_lineage"] = round(
            (n_damage_avoided - n_lineage) * p_hat / n_questions, 6
        )
    return out


def breakdown(region: List[dict], keys) -> dict:
    groups = defaultdict(list)
    for r in region:
        groups[tuple(r.get(k) for k in keys)].append(r)
    out = {}
    for g, rs in sorted(groups.items(), key=lambda x: str(x[0])):
        c = defaultdict(int)
        for r in rs:
            c[r["hnav_target"]] += 1
        safe = c["may_suppress"] + c["inert_superseded"] + c["must_suppress"]
        decided = safe + c["must_write"]
        p, lo, hi = wilson(safe, decided)
        out["|".join(str(x) for x in g)] = {
            "n": len(rs), "n_harmful": c["must_write"], "n_safe": safe,
            "suppression_precision": p, "wilson95": [lo, hi],
        }
    return out


# ---------------------------------------------------------------------------
# Shadow-vs-live agreement on the writes that were REALLY suppressed
# ---------------------------------------------------------------------------


def live_suppressions(rows: List[dict]) -> dict:
    """The label's verdict on decisions the gate actually applied as NOOP.

    This is the only place where the first-order counterfactual meets reality.
    n is tiny (30 across the whole campaign), so it is a sanity check, never an
    estimate.
    """
    live = [r for r in rows if r.get("suppressed")]
    c = defaultdict(int)
    for r in live:
        c[r["hnav_target"]] += 1
    return {
        "n_live_suppressions": len(live),
        "by_target": dict(c),
        "by_arm": {
            a: len([r for r in live if r.get("arm") == a])
            for a in sorted({r.get("arm") for r in live})
        },
        "harmful_noop": sum(1 for r in live if r["hnav_label"] == "harmful_noop"),
        "note": "first-order counterfactual sanity check only; n is far too "
                "small to estimate a rate",
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--proxy", default="gov_logs/hnav_proxy_validation.json")
    ap.add_argument("--n-questions", type=int, default=155,
                    help="questions per backend in the memory category")
    ap.add_argument("--include-student", action="store_true")
    ap.add_argument("--out", default="gov_logs/hnav_counterfactual.json")
    args = ap.parse_args()

    rows = load_labels(args.labels, include_student=args.include_student)
    if not rows:
        raise SystemExit("[counterfactual_noop] no label rows matched")

    p_hat = None
    proxy_path = Path(args.proxy)
    if proxy_path.exists():
        with open(proxy_path, "r", encoding="utf-8") as f:
            p_hat = (json.load(f).get("p_hat") or {}).get("pooled")

    n_total = len(rows)
    cells = {}
    for preflight in (True, False):
        for veto in (True, False):
            for sh in SIM_HIGH_GRID:
                for dl in DELTA_GRID:
                    region = [r for r in rows if in_region(r, sh, dl, veto, preflight)]
                    if not region:
                        continue
                    key = (f"sim_high={sh:.3f}|delta={dl:.2f}"
                           f"|veto={int(veto)}|preflight={int(preflight)}")
                    cell = cell_stats(region, n_total, p_hat, args.n_questions)
                    cell["by_backend"] = breakdown(region, ["backend"])
                    cell["by_op_family"] = breakdown(region, ["op_family"])
                    cell["by_fate"] = breakdown(region, ["fate"])
                    cells[key] = cell

    overall = defaultdict(int)
    for r in rows:
        overall[r["hnav_target"]] += 1

    doc = {
        "n_decisions": n_total,
        "p_hat": p_hat,
        "n_questions_per_backend": args.n_questions,
        "include_student": bool(args.include_student),
        "target_distribution": dict(overall),
        "fate_distribution": {
            (k or "unresolved"): sum(1 for r in rows if r.get("fate") == k)
            for k in ("terminal", "superseded", "gone", None)
        },
        "live_suppressions": live_suppressions(rows),
        "grid": {"sim_high": SIM_HIGH_GRID, "delta": DELTA_GRID},
        "cells": cells,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)

    print(f"[counterfactual_noop] {n_total} labeled decisions, p_hat={p_hat}")
    print(f"[counterfactual_noop] targets: {dict(overall)}")
    print(f"[counterfactual_noop] {len(cells)} non-empty cells -> {out_path}")
    for suffix, title in (("|veto=1|preflight=1", "DEPLOYABLE (gate-faithful)"),
                          ("|veto=0|preflight=0", "GEOMETRIC (falsifier population)")):
        sel = [(k, c) for k, c in cells.items() if k.endswith(suffix)]
        sel.sort(key=lambda x: -x[1]["n_region"])
        print(f"[counterfactual_noop] frontier -- {title}:")
        print(f"  {'cell':30s} {'cov':>6s} {'n':>5s} {'harm':>5s} {'prec':>6s} "
              f"{'wil_lo':>7s} {'E[dAcc]':>8s}")
        for key, c in sel[:6]:
            lo = c["suppression_precision_wilson95"][0]
            print(f"  {key[:30]:30s} {c['coverage']:6.3f} {c['n_region']:5d} "
                  f"{c['n_harmful_noop']:5d} {c['suppression_precision'] or 0:6.3f} "
                  f"{lo if lo is not None else 0:7.3f} "
                  f"{c.get('expected_dAcc', 0):8.4f}")


if __name__ == "__main__":
    main()
