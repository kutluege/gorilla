"""
Offline store audit (Plan v2 Step 9, plan SS7.4).

Re-scores a committed backend memory snapshot (``memory_snapshot/<scenario>_final.json``)
under a chosen admission policy WITHOUT touching it: each stored item is
leave-one-out re-evaluated as if it were a fresh candidate against the rest of
the store (Stage 0 geometry + Stage 1 margin-entropy signals), producing a
report of duplicates, shadowed items, and interference risks. Report-only by
construction -- this script never writes to any store, snapshot, or sidecar.

Historical rehydration is deliberately NOT regated at runtime (plan SS7.4:
replay of committed state must stay idempotent); this audit mode is the
explicit offline instrument for re-evaluating old stores.

Run:
  python bfcl_eval/scripts/audit_store.py \
      --snapshot .../memory_snapshot/customer_final.json --backend kv \
      [--policy geometry_margin_entropy_v1] [--out report.json]
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bfcl_eval.model_handler.middleware.governance_filter import (  # noqa: E402
    GovConfig,
    GovernanceCache,
    MemoryItem,
    ThresholdState,
    WriteCandidate,
    compute_signals,
    decide,
    load_abtt,
    preflight_would_succeed,
)
from bfcl_eval.model_handler.middleware.semantic_entropy import _get_encoder  # noqa: E402


def load_snapshot(path: Path, backend: str):
    """-> [(tier, ref, text, args)] in snapshot order."""
    with open(path, "r", encoding="utf-8") as f:
        snap = json.load(f)
    entries = []
    for tier, snap_key in (("core", "core_memory"), ("archival", "archival_memory")):
        tier_data = snap.get(snap_key, {})
        if backend == "kv":
            for key, value in tier_data.items():
                from bfcl_eval.model_handler.middleware.governance_filter import (
                    kv_composite_text,
                )

                entries.append(
                    (tier, str(key), kv_composite_text(key, value),
                     {"key": str(key), "value": str(value)})
                )
        else:
            for vec_id, text in tier_data.get("store", {}).items():
                entries.append((tier, str(int(vec_id)), str(text), {"text": str(text)}))
    return entries


def audit(snapshot_path: Path, backend: str, cfg: GovConfig) -> dict:
    encoder = _get_encoder()
    if encoder is None:
        sys.exit("[audit_store] sentence-transformers encoder unavailable")
    abtt = load_abtt(cfg.artifact_path, expected_d=cfg.d)

    def whiten_one(text):
        import numpy as np

        raw = np.asarray(encoder.encode([text], normalize_embeddings=False))[0]
        return abtt.apply(raw)

    entries = load_snapshot(snapshot_path, backend)
    rows = []
    for i, (tier, ref, text, args) in enumerate(entries):
        # Leave-one-out mirror: everything except the audited item.
        cache = GovernanceCache(backend)
        for j, (t2, r2, x2, _) in enumerate(entries):
            if j == i:
                continue
            cache.put(
                MemoryItem(ref=r2, text=x2, emb_whitened=whiten_one(x2),
                           turn_written=-1, tier=t2)
            )
        thresholds = ThresholdState(cfg)
        thresholds.seed(cache)
        op = f"{tier}_memory_add"
        candidate = WriteCandidate(
            op=op, kind="add", tier=tier, backend=backend, text=text,
            args=args, ref=ref if backend == "kv" else None,
            raw_call=f"{op}(...)",
        )
        preflight_ok = preflight_would_succeed(candidate, cache)
        v_w = whiten_one(text)
        signals = compute_signals(v_w, candidate, cache, thresholds, cfg)
        decision, reason = decide(candidate, signals, cache, cfg, preflight_ok)
        row = {
            "tier": tier,
            "ref": ref,
            "text": text,
            "sim_max": round(signals.sim_max, 4),
            "r": round(signals.r, 4),
            "tau_t": round(signals.tau_t, 4),
            "verbatim_misses": signals.verbatim_misses,
            "s0_decision": decision.value,
            "s0_reason": reason,
        }
        if cfg.policy in ("geometry_margin_entropy_v1", "geometry_margin_entropy_risk_v1"):
            from bfcl_eval.model_handler.middleware.admission_policy import (
                apply_option_a_rule,
                compute_margin_entropy_signals,
            )
            from bfcl_eval.model_handler.middleware.governance_filter import GovDecision

            if decision == GovDecision.ESCALATE:
                me = compute_margin_entropy_signals(candidate, signals, cache, cfg, "")
                action, code, _, flagged = apply_option_a_rule(
                    candidate, signals, me, cache, cfg, "", preflight_ok
                )
                row["s1_action"] = action
                row["s1_reason_code"] = code
                row["s1_flagged"] = flagged
                row["s1_me"] = me.to_log()
        rows.append(row)

    dup = [r for r in rows if r["s0_decision"] == "NOOP"]
    return {
        "snapshot": str(snapshot_path),
        "backend": backend,
        "policy": cfg.policy,
        "n_items": len(rows),
        "n_duplicate_like": len(dup),
        "items": rows,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--backend", required=True, choices=("kv", "vector"))
    ap.add_argument("--policy", default="geometry_only")
    ap.add_argument("--out", default=None, help="report JSON path (default: stdout)")
    args = ap.parse_args()

    cfg = GovConfig()
    cfg.policy = args.policy
    report = audit(Path(args.snapshot), args.backend, cfg)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload)
        print(f"[audit_store] wrote {args.out} ({report['n_items']} items, "
              f"{report['n_duplicate_like']} duplicate-like)")
    else:
        print(payload)


if __name__ == "__main__":
    main()
