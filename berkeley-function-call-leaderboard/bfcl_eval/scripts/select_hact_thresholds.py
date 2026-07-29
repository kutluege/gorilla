"""Stage 4 threshold selection for H_act gate policies (runs ONLY on GO/PARTIAL).

Dataset partition discipline (gov_logs/hnav_shadow/DATASET_PARTITION.md):
dev = shadow reps 01-02, val = rep03. A small pre-justified grid per gate
signal is swept on dev; the dev-chosen threshold is evaluated ONCE on val and
frozen to gov_logs/hnav_stage4/hact_calibration.json (sha recorded in the
confirmatory campaign manifest via HACT_CALIB). Refuses to overwrite an
existing frozen file (calibrate_margin_entropy --freeze precedent).

Gate semantics evaluated offline: "intervene" on a candidate write when the
uncertainty signal exceeds tau (or vote margin falls below tau). Metrics per
tau: coverage (interventions / candidates), harmful-event recall
(must_suppress caught), false-intervention rate (interventions on must_write
or benign necessary writes), Wilson intervals.

Usage:
    python bfcl_eval/scripts/select_hact_thresholds.py \
        --logs gov_logs/hnav_shadow/rep01_hact_shadow gov_logs/hnav_shadow/rep02_hact_shadow \
        --val-logs gov_logs/hnav_shadow/rep03_hact_shadow \
        --outcomes gov_logs/hnav_shadow/outcomes_hnav_shadow.jsonl \
        --out gov_logs/hnav_stage4/hact_calibration.json [--freeze]
"""

import argparse
import glob as globmod
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from calibrate_margin_entropy import git_head, load_decisions, sha_file  # noqa: E402
from counterfactual_noop import wilson  # noqa: E402
from evaluate_hact_shadow import (  # noqa: E402
    attach_labels_by_replicate,
    join_hact,
    load_hact_rows,
)

# Pre-justified grids (PREREGISTRATION.md discipline: small, fixed, recorded).
GRIDS = {
    # intervene when h_act_target >= tau
    "hact_gate": {"feature": "h_act_target", "direction": "ge",
                  "grid": [0.5, 0.7, 0.9, 1.1, 1.3]},
    # intervene when vote_margin_target <= tau
    "vote_margin_gate": {"feature": "vote_margin_target", "direction": "le",
                         "grid": [0.125, 0.25, 0.375, 0.5]},
    # factorized: op entropy leg (reconsider op) -- target leg reuses hact_gate
    "hact_op_gate": {"feature": "h_act_op", "direction": "ge",
                     "grid": [0.3, 0.5, 0.7, 0.9]},
}


def load_labeled(log_dirs, outcomes):
    rows, _ = load_decisions(log_dirs, escalated_only=False)
    hact, _, _, _ = load_hact_rows(log_dirs)
    joined, _ = join_hact(rows, hact)
    return attach_labels_by_replicate(joined, outcomes, "must_suppress"), joined


def intervention_stats(rows, feature, direction, tau):
    """rows must carry row['label'] (must_suppress) and row['outcomes']."""
    n = len(rows)
    hit = [r for r in rows
           if (r["features"].get(feature) is not None)
           and ((r["features"][feature] >= tau) if direction == "ge"
                else (r["features"][feature] <= tau))]
    n_int = len(hit)
    harmful = [r for r in rows if r["label"] == 1]
    caught = sum(1 for r in hit if r["label"] == 1)
    # false intervention: intervening on a write the counterfactual says was
    # necessary (must_write) -- the outcomes dict is attached by attach_hnav_labels
    false_int = sum(1 for r in hit
                    if (r.get("outcomes") or {}).get("must_write"))
    rec_p, rec_lo, rec_hi = wilson(caught, len(harmful)) if harmful else (None,) * 3
    fir_p, fir_lo, fir_hi = wilson(false_int, n_int) if n_int else (None,) * 3
    return {
        "tau": tau, "n": n, "n_interventions": n_int,
        "coverage": n_int / n if n else None,
        "n_harmful": len(harmful), "harmful_caught": caught,
        "harmful_recall": rec_p, "harmful_recall_wilson95": [rec_lo, rec_hi],
        "n_false_interventions": false_int,
        "false_intervention_rate": fir_p,
        "false_intervention_wilson95": [fir_lo, fir_hi],
    }


def select_on_dev(dev_rows, spec, max_fir=0.20, min_recall=0.30):
    """Pick tau: max harmful recall subject to false-intervention rate <=
    max_fir; require recall >= min_recall; tie-break lower coverage.
    Returns (tau|None, table)."""
    table = [intervention_stats(dev_rows, spec["feature"], spec["direction"], t)
             for t in spec["grid"]]
    ok = [s for s in table
          if s["false_intervention_rate"] is not None
          and s["false_intervention_rate"] <= max_fir
          and (s["harmful_recall"] or 0.0) >= min_recall]
    if not ok:
        return None, table
    best = sorted(ok, key=lambda s: (-(s["harmful_recall"] or 0), s["coverage"]))[0]
    return best["tau"], table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True, help="dev arm dirs (reps 01-02)")
    ap.add_argument("--val-logs", nargs="+", required=True, help="val arm dirs (rep 03)")
    ap.add_argument("--outcomes", nargs="+", required=True)
    ap.add_argument("--max-fir", type=float, default=0.20)
    ap.add_argument("--min-recall", type=float, default=0.30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()

    dev_dirs = sorted(set(p for g in args.logs for p in globmod.glob(g)))
    val_dirs = sorted(set(p for g in args.val_logs for p in globmod.glob(g)))
    overlap = set(dev_dirs) & set(val_dirs)
    if overlap:
        sys.exit(f"[thresholds] dev/val overlap forbidden: {overlap}")

    dev_rows, _ = load_labeled(dev_dirs, args.outcomes)
    val_rows, _ = load_labeled(val_dirs, args.outcomes)
    # Twin exclusion: the campaign's replicate-invariant HACT_SEED makes some
    # candidates byte-identical across replicates (same candidate_id). A val
    # row whose id also appears in dev would validate the threshold on the
    # data that selected it; drop them and report the count.
    dev_ids = {r["candidate_id"] for r in dev_rows}
    n_val_twins = sum(1 for r in val_rows if r["candidate_id"] in dev_ids)
    val_rows = [r for r in val_rows if r["candidate_id"] not in dev_ids]
    print(f"[thresholds] dev rows={len(dev_rows)} val rows={len(val_rows)} "
          f"(val twins excluded={n_val_twins})")

    result = {"git_head": git_head(), "max_fir": args.max_fir,
              "min_recall": args.min_recall,
              "n_val_twins_excluded": n_val_twins,
              "dev_dirs": dev_dirs, "val_dirs": val_dirs,
              "outcomes_sha": [sha_file(p) for pat in args.outcomes
                               for p in sorted(globmod.glob(pat))],
              "gates": {}}
    for name, spec in GRIDS.items():
        tau, dev_table = select_on_dev(dev_rows, spec, args.max_fir, args.min_recall)
        entry = {"feature": spec["feature"], "direction": spec["direction"],
                 "grid": spec["grid"], "dev_table": dev_table,
                 "tau": tau, "val": None, "deployable": False}
        if tau is not None:
            val_stats = intervention_stats(val_rows, spec["feature"],
                                           spec["direction"], tau)
            entry["val"] = val_stats
            entry["deployable"] = (
                val_stats["false_intervention_rate"] is not None
                and val_stats["false_intervention_rate"] <= args.max_fir)
        result["gates"][name] = entry
        print(f"[thresholds] {name}: tau={tau} "
              f"deployable={entry['deployable']}")

    out = Path(args.out)
    if args.freeze and out.exists():
        sys.exit(f"[thresholds] refusing to overwrite frozen {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[thresholds] wrote {out} (frozen={args.freeze})")


if __name__ == "__main__":
    main()
