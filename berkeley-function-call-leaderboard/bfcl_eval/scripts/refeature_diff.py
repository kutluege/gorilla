"""
H-Nav Stage 1.3 driver: marginal-diff features for every gov2 decision.

Structurally a clone of ``refeature_entropy_v2.py`` (walk_log -> per-decision
compute -> JSONL keyed on candidate_id), with two deliberate differences:

1. It does NOT filter to escalated decisions. The NOOP region the falsifier has
   to discriminate on is resolved at Stage 0, so restricting to escalations
   would drop exactly the population of interest.
2. It carries an ABTT whitener (``replay_admission.Embedder``) so
   ``diff_sim_max`` lands in the SAME space as the logged Stage-0 ``sim_max``
   and the two are directly comparable.

Two output files, and the split is a leakage boundary, not a convenience:

  features_diff.jsonl         model features. Nothing here has seen a benchmark
                              question or answer.
  features_diff_oracle.jsonl  ground-truth-derived diagnostics (``oracle_*``).
                              Never registered in ``FEATURE_KEYS``; used only
                              for the oracle-ceiling section of the report,
                              which answers "how much headroom could ANY online
                              feature have had?".

Run:
  python bfcl_eval/scripts/refeature_diff.py \
      --gov-logs "gov_logs/me_harvest/rep0*_v1_shadow" \
      --out gov_logs/me_harvest/features_diff.jsonl \
      --oracle-out gov_logs/me_harvest/features_diff_oracle.jsonl
"""

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bfcl_eval.model_handler.middleware import diff_signals  # noqa: E402
from bfcl_eval.model_handler.middleware.governance_filter import GovConfig  # noqa: E402
from bfcl_eval.model_handler.middleware.retrieval_sim import default_encode  # noqa: E402
from bfcl_eval.scripts.label_outcomes import scenario_of, walk_log  # noqa: E402


def feature_row(rec, tiers, whiten, encode):
    backend, tier = rec["backend"], rec["tier"]
    ref = rec.get("candidate_ref")
    new_text = rec.get("candidate_text") or ""
    old_text = (tiers.get(tier) or {}).get(str(ref)) if ref is not None else None
    feats = diff_signals.compute(
        backend, tier, ref, old_text, new_text, tiers, whiten=whiten, encode=encode
    )
    feats["candidate_id"] = rec.get("candidate_id")
    return feats


def oracle_row(rec, tiers):
    """Ground-truth-derived diagnostics. Imported here and NOWHERE in
    ``diff_signals`` so the model-feature module stays answer-blind."""
    from bfcl_eval.scripts.hnav_answer_index import carries, scenario_questions

    backend, tier = rec["backend"], rec["tier"]
    ref = rec.get("candidate_ref")
    new_text = rec.get("candidate_text") or ""
    old_text = (tiers.get(tier) or {}).get(str(ref)) if ref is not None else None
    parts = diff_signals.marginal_diff(old_text, new_text)
    qs = scenario_questions(scenario_of(rec.get("test_id", "")))
    marginal = [
        q for q in qs
        if carries(new_text, q["gold"]) and not carries(old_text, q["gold"])
    ]
    return {
        "candidate_id": rec.get("candidate_id"),
        "oracle_carries_gold": int(any(carries(new_text, q["gold"]) for q in qs)),
        "oracle_diff_carries_gold": int(
            any(carries(parts.added_text, q["gold"]) for q in qs)
        ),
        "oracle_n_marginal_gold": len(marginal),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gov-logs", required=True, help="glob of governed arm dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--oracle-out", default=None)
    ap.add_argument("--no-whiten", action="store_true",
                    help="skip the ABTT channel (diff_sim_max stays None)")
    args = ap.parse_args()

    dirs = sorted(glob.glob(args.gov_logs))
    if not dirs:
        sys.exit(f"no logs match {args.gov_logs!r}")

    whiten = None
    if not args.no_whiten:
        from bfcl_eval.scripts.replay_admission import Embedder

        whiten = Embedder(GovConfig.from_env()).whiten_one

    rows, oracle_rows = [], []
    n_seen = n_desync = 0
    for d in dirs:
        log_path = Path(d)
        if log_path.is_dir():
            log_path = log_path / "governance_log.jsonl"
        for rec, tiers, synced in walk_log(log_path):
            n_seen += 1
            if not synced or not rec.get("candidate_id"):
                n_desync += 1
                continue
            rows.append(feature_row(rec, tiers, whiten, default_encode))
            if args.oracle_out:
                oracle_rows.append(oracle_row(rec, tiers))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.oracle_out:
        o = Path(args.oracle_out)
        o.parent.mkdir(parents=True, exist_ok=True)
        with open(o, "w", encoding="utf-8") as f:
            for row in oracle_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    cov = {
        k: sum(1 for r in rows if r.get(k) is not None)
        for k in diff_signals.ALL_FEATURES
    }
    thin = {k: v for k, v in cov.items() if v < len(rows)}
    print(f"[refeature_diff] decisions={n_seen} featured={len(rows)} "
          f"skipped={n_desync}")
    print(f"[refeature_diff] partial coverage: {thin or 'none (all features full)'}")
    if rows:
        both = [r for r in rows
                if r.get("diff_sim_max") is not None and r.get("has_old")]
        print(f"[refeature_diff] diff_sim_max present on {len(both)} update rows")
    print(f"[refeature_diff] wrote {out}"
          + (f" and {args.oracle_out}" if args.oracle_out else ""))


if __name__ == "__main__":
    main()
