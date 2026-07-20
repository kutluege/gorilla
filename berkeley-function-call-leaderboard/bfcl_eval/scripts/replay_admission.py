"""
Admission replay harness (Plan v2 Step 7, plan SS14).

Re-runs the decisions of a committed ``governance_log.jsonl`` through a chosen
GOV_POLICY, reconstructing per-scenario memory state by chain-carry exactly as
``replay_geometry_deltaH.py`` does (rehydrate -> decisions -> observes, with
checkpoint verification and explicit desync accounting).

Per policy:

* ``legacy_full``     -- recomputes Stage 0 (embeddings, whitening, signals,
  decision) from scratch and re-derives the FINAL decision by re-executing the
  deterministic ``_resolve_escalation`` mapping over the LOGGED stage1/stage2
  blocks. The NLI forward pass is not re-run (DeBERTa is deterministic but
  heavyweight, and Stage 2's probes derive from ``user_text``, which legacy
  logs do not record) -- the logged verdict tables are the oracle for exactly
  that component; everything downstream of them is recomputed and compared
  bit-for-bit against the logged decision/reason.
* ``geometry_only``   -- recomputes Stage 0 and compares Stage-0 fields; every
  ESCALATE maps to ADD.
* ``geometry_margin_entropy_v1`` -- recomputes Stage 0 (must match the logged
  signals bit-for-bit) and produces valid gov2 records under the v1 rule.
  ``user_text`` is taken from the log when present (gov2 logs); for legacy
  logs the user-channel probes are unavailable and KV falls back to its
  key-channel probes while Vector locate/strong stay indeterminate.

Acceptance (SS17.2 replay list): legacy_full replay over the 5 governed A/B
logs reproduces logged (decision, reason) exactly; determinism: two runs are
byte-identical.

Run:
  python bfcl_eval/scripts/replay_admission.py \
      --logs gov_logs/replicates/rep01_governed --policy legacy_full \
      --env GOV_DELTA=0.32 --env GOV_SIM_HIGH=0.95 --env GOV_S2_MARGIN=0.01 \
      --env GOV_NLI_ENABLED=1 --env GOV_NLI_SHADOW=0 \
      --env GOV_S2_ENABLED=1 --env GOV_S2_SHADOW=0
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from bfcl_eval.model_handler.middleware.governance_filter import (  # noqa: E402
    GovConfig,
    GovDecision,
    GovernanceCache,
    KV_WRITE_OPS,
    MemoryItem,
    Stage0Signals,
    ThresholdState,
    VECTOR_WRITE_OPS,
    WriteCandidate,
    build_candidate,
    compute_signals,
    decide,
    kv_composite_text,
    load_abtt,
    preflight_would_succeed,
)
from bfcl_eval.model_handler.middleware.semantic_entropy import _get_encoder  # noqa: E402

STAGE0_FINAL_REASONS = (
    "redundant",
    "novel",
    "empty_memory",
    "noop_blocked_by_preflight",
)


def scenario_of(test_id: str) -> str:
    parts = (test_id or "").split("-")
    return parts[1] if len(parts) >= 2 else str(test_id)


# ---------------------------------------------------------------------------
# Embedding cache (raw MiniLM; whitening applied per config artifact)
# ---------------------------------------------------------------------------


class Embedder:
    def __init__(self, cfg: GovConfig):
        self.encoder = _get_encoder()
        if self.encoder is None:
            sys.exit("[replay_admission] sentence-transformers encoder unavailable")
        self.abtt = load_abtt(cfg.artifact_path, expected_d=cfg.d)
        self._raw: dict = {}

    def whiten_one(self, text: str) -> np.ndarray:
        if text not in self._raw:
            self._raw[text] = np.asarray(
                self.encoder.encode([text], normalize_embeddings=False)
            )[0]
        return self.abtt.apply(self._raw[text])


# ---------------------------------------------------------------------------
# Per-(backend, scenario) chain replay state
# ---------------------------------------------------------------------------


class Chain:
    def __init__(self):
        self.tiers = {"core": {}, "archival": {}}  # tier -> {ref: text}
        self.synced = False
        self.started = False
        self.cache = None  # GovernanceCache mirror, rebuilt at rehydrate
        self.thresholds = None

    def total(self):
        return len(self.tiers["core"]) + len(self.tiers["archival"])


def rebuild_cache(chain: Chain, backend: str, cfg: GovConfig, emb: Embedder):
    cache = GovernanceCache(backend)
    for tier in ("core", "archival"):
        for ref, text in chain.tiers[tier].items():
            cache.put(
                MemoryItem(
                    ref=str(ref),
                    text=text,
                    emb_whitened=emb.whiten_one(text),
                    turn_written=-1,
                    tier=tier,
                )
            )
    chain.cache = cache
    chain.thresholds = ThresholdState(cfg)
    chain.thresholds.seed(cache)


def reconstruct_candidate(rec: dict) -> WriteCandidate:
    backend = rec["backend"]
    op = rec["op"]
    table = KV_WRITE_OPS if backend == "kv" else VECTOR_WRITE_OPS
    kind, tier, _params = table[op]
    text = rec.get("candidate_text") or ""
    ref = rec.get("candidate_ref")
    if backend == "kv":
        key = str(ref)
        prefix = f"{key.replace('_', ' ')}: "
        value = text[len(prefix):] if text.startswith(prefix) else text
        args = {"key": key, "value": value}
        raw_call = f"{op}(key={key!r}, value={value!r})"
        assert kv_composite_text(key, value) == text or not text.startswith(prefix)
    elif kind == "add":
        args = {"text": text}
        raw_call = f"{op}(text={text!r})"
        ref = None
    else:
        args = {"vec_id": ref, "new_text": text}
        raw_call = f"{op}(vec_id={ref}, new_text={text!r})"
    return WriteCandidate(
        op=op, kind=kind, tier=tier, backend=backend,
        text=text, args=args, ref=None if ref is None else str(ref),
        raw_call=raw_call,
    )


# ---------------------------------------------------------------------------
# legacy_full: re-derive the final decision from logged stage blocks
# ---------------------------------------------------------------------------


def rederive_legacy_escalation(rec, cfg: GovConfig, cache, preflight_ok):
    """Deterministic re-execution of GovernanceSession._resolve_escalation over
    the logged stage1/stage2 outcome fields. Returns (decision, reason)."""
    if not cfg.nli_enabled:
        return "ADD", "escalate:stage0_escalate_fallback"
    s1 = rec.get("stage1")
    if s1 is None:
        return None, None  # cannot re-derive without the logged block
    outcome, reason = s1.get("outcome"), s1.get("reason")
    rewritten_call = s1.get("rewritten_call")
    low_confidence = False
    shadowed_by = "stage1_shadow" if cfg.nli_shadow else None

    if outcome == "ESCALATE":
        s2 = rec.get("stage2")
        if cfg.s2_enabled and s2 is not None:
            outcome, reason = s2.get("outcome"), s2.get("reason")
            rewritten_call = s2.get("rewritten_call")
            low_confidence = bool(s2.get("low_confidence", False))
            if shadowed_by is None and cfg.s2_shadow:
                shadowed_by = "stage2_shadow"
        else:
            outcome, reason = "ADD", "stage1_all_neutral_no_stage2"

    if shadowed_by is not None:
        return "ADD", f"escalate:{shadowed_by}:{reason}"
    if outcome == "NOOP" and not preflight_ok:
        return "ADD", f"escalate:{reason}_preflight_blocked"
    if outcome == "REWRITE":
        rc = build_candidate(rec["backend"], rewritten_call) if rewritten_call else None
        if rc is None or not preflight_would_succeed(rc, cache):
            return "ADD", f"escalate:{reason}_rewrite_preflight_blocked"
    return outcome, f"escalate:{reason}"


# ---------------------------------------------------------------------------
# Replay driver
# ---------------------------------------------------------------------------


def replay(log_path: Path, cfg: GovConfig, policy: str):
    emb = Embedder(cfg)
    chains = defaultdict(Chain)
    rows = []
    counters = defaultdict(int)
    boundary_cfgs = {}

    if policy not in ("legacy_full", "geometry_only", "geometry_margin_entropy_v1"):
        sys.exit(f"[replay_admission] unsupported policy for replay: {policy}")

    with open(log_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                counters["unparseable_lines"] += 1
                continue
            event = rec.get("event")
            backend = rec.get("backend", "")
            key = (backend, scenario_of(rec.get("test_id", "")))
            chain = chains[key]

            if event == "rehydrate":
                n = int(rec.get("items_loaded", 0))
                if n == 0:
                    chain.tiers = {"core": {}, "archival": {}}
                    chain.synced = True
                    chain.started = True
                    counters["checkpoints_reset"] += 1
                elif chain.synced and chain.total() == n:
                    counters["checkpoints_verified"] += 1
                else:
                    chain.synced = False
                    chain.started = True
                    counters["checkpoints_desync"] += 1
                if chain.synced:
                    rebuild_cache(chain, backend, cfg, emb)

            elif event == "observe":
                tier, ref, text = rec.get("tier"), str(rec.get("ref")), rec.get("text", "")
                if tier in ("core", "archival"):
                    chain.tiers[tier][ref] = text
                    if chain.synced and chain.cache is not None:
                        chain.cache.put(
                            MemoryItem(
                                ref=ref, text=text,
                                emb_whitened=emb.whiten_one(text),
                                turn_written=int(rec.get("step_idx", -1)),
                                tier=tier,
                            )
                        )

            elif event == "observe_remove":
                tier, ref = rec.get("tier"), str(rec.get("ref"))
                if tier in ("core", "archival"):
                    chain.tiers[tier].pop(ref, None)
                    if chain.synced and chain.cache is not None:
                        chain.cache.drop(tier, ref)

            elif event == "observe_clear":
                tier = rec.get("tier")
                if tier in ("core", "archival"):
                    chain.tiers[tier] = {}
                    if chain.synced and chain.cache is not None:
                        chain.cache.clear_tier(tier)

            elif event == "decision":
                counters["decisions_total"] += 1
                row = {
                    "line_no": line_no,
                    "test_id": rec.get("test_id"),
                    "step_idx": rec.get("step_idx"),
                    "backend": backend,
                    "op": rec.get("op"),
                    "logged_decision": rec.get("decision"),
                    "logged_reason": rec.get("reason"),
                    "computable": False,
                    "drop_reason": None,
                }
                if not chain.synced or chain.cache is None:
                    row["drop_reason"] = "desync"
                    counters["dropped_desync"] += 1
                    rows.append(row)
                    continue
                if rec.get("n_items") is not None and int(rec["n_items"]) != chain.cache.total_size():
                    row["drop_reason"] = "n_items_mismatch"
                    counters["dropped_n_items_mismatch"] += 1
                    chain.synced = False
                    rows.append(row)
                    continue
                row["computable"] = True
                counters["decisions_computable"] += 1
                replay_one(rec, chain, cfg, emb, policy, row, counters)
                rows.append(row)

    return rows, counters


def replay_one(rec, chain: Chain, cfg: GovConfig, emb: Embedder, policy, row, counters):
    candidate = reconstruct_candidate(rec)
    cache = chain.cache
    preflight_ok = preflight_would_succeed(candidate, cache)

    if policy in ("legacy_full", "geometry_only"):
        v_w = emb.whiten_one(candidate.text)
        signals = compute_signals(v_w, candidate, cache, chain.thresholds, cfg)
        s0_decision, s0_reason = decide(candidate, signals, cache, cfg, preflight_ok)
    else:
        from bfcl_eval.model_handler.middleware.geometry_gate import GeometryGate

        geo = GeometryGate(cfg).evaluate(
            candidate, cache, chain.thresholds, preflight_ok, emb.whiten_one
        )
        signals, s0_decision, s0_reason = geo.signals, geo.decision, geo.reason

    row["sim_max"] = round(signals.sim_max, 4)
    row["r"] = round(signals.r, 4)
    row["tau_t"] = round(signals.tau_t, 4)
    row["preflight_ok"] = preflight_ok
    sig_match = (
        row["sim_max"] == rec.get("sim_max")
        and row["r"] == rec.get("r")
        and row["tau_t"] == rec.get("tau_t")
        and preflight_ok == rec.get("preflight_ok")
    )
    row["stage0_signals_match"] = sig_match
    counters["stage0_signals_match" if sig_match else "stage0_signals_mismatch"] += 1

    # Stage-0 outcome comparison (both stage0-final and escalated records).
    logged_reason = rec.get("reason") or ""
    logged_stage0_final = logged_reason in STAGE0_FINAL_REASONS
    if policy == "geometry_margin_entropy_v1" and s0_reason in (
        "exact_duplicate", "unretrievable_novel"
    ):
        # New-policy-only outcomes: comparable only against gov2 logs; against
        # legacy logs they are the documented allowed deviation (plan Step 2).
        row["stage0_new_reason"] = s0_reason
        counters["stage0_new_reason_codes"] += 1
        stage0_match = None
    elif logged_stage0_final:
        stage0_match = (
            s0_decision.value == rec.get("decision") and s0_reason == logged_reason
        )
    else:
        stage0_match = s0_decision == GovDecision.ESCALATE
    row["stage0_match"] = stage0_match
    if stage0_match is True:
        counters["stage0_match"] += 1
    elif stage0_match is False:
        counters["stage0_mismatch"] += 1

    # Final decision per policy.
    if policy == "geometry_only":
        final_decision = "ADD" if s0_decision == GovDecision.ESCALATE else s0_decision.value
        final_reason = (
            "escalate:stage0_escalate_fallback"
            if s0_decision == GovDecision.ESCALATE
            else s0_reason
        )
        row["replayed_decision"] = final_decision
        row["replayed_reason"] = final_reason
        return

    if policy == "legacy_full":
        if s0_decision != GovDecision.ESCALATE:
            final_decision, final_reason = s0_decision.value, s0_reason
        else:
            final_decision, final_reason = rederive_legacy_escalation(
                rec, cfg, cache, preflight_ok
            )
            if final_decision is None:
                row["drop_reason"] = "no_stage_blocks"
                counters["legacy_no_stage_blocks"] += 1
                return
            if hasattr(final_decision, "value"):
                final_decision = final_decision.value
        row["replayed_decision"] = final_decision
        row["replayed_reason"] = final_reason
        match = final_decision == rec.get("decision") and final_reason == rec.get("reason")
        row["decision_match"] = match
        counters["decision_match" if match else "decision_mismatch"] += 1
        return

    # geometry_margin_entropy_v1: produce a valid gov2-style replay record.
    from bfcl_eval.model_handler.middleware.admission_policy import (
        apply_option_a_rule,
        compute_margin_entropy_signals,
    )

    if s0_decision != GovDecision.ESCALATE:
        row["replayed_action"] = (
            "NOOP" if s0_decision == GovDecision.NOOP else "ADD"
        )
        row["replayed_reason"] = s0_reason
        row["replayed_stage"] = "GEOMETRY"
        return
    user_text = rec.get("user_text") or ""
    me = compute_margin_entropy_signals(candidate, signals, cache, cfg, user_text)
    action, code, rewritten, flagged = apply_option_a_rule(
        candidate, signals, me, cache, cfg, user_text, preflight_ok
    )
    row["replayed_action"] = action
    row["replayed_reason"] = code
    row["replayed_stage"] = "MARGIN_ENTROPY"
    row["s1_me"] = me.to_log()
    row["flagged"] = flagged
    row["rewritten_call"] = rewritten
    counters[f"v1_action_{action}"] += 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_cfg(env_pairs) -> GovConfig:
    import os

    saved = {}
    try:
        for pair in env_pairs or []:
            k, _, v = pair.partition("=")
            saved[k] = os.environ.get(k)
            os.environ[k] = v
        return GovConfig.from_env()
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--logs", required=True, help="arm dir (or the .jsonl itself)")
    ap.add_argument("--policy", default="legacy_full")
    ap.add_argument("--env", action="append", default=[],
                    help="GOV_* env override KEY=VAL (repeatable); applied for "
                         "config construction only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    log_path = Path(args.logs)
    if log_path.is_dir():
        log_path = log_path / "governance_log.jsonl"
    if not log_path.exists():
        sys.exit(f"log not found: {log_path}")

    env_pairs = list(args.env)
    if not any(p.startswith("GOV_POLICY=") for p in env_pairs):
        env_pairs.append(f"GOV_POLICY={args.policy}")
    cfg = build_cfg(env_pairs)

    arm = log_path.parent.name
    out_dir = (
        Path(args.out)
        if args.out
        else log_path.parent.parent / f"replay_admission_{args.policy}_{arm}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, counters = replay(log_path, cfg, args.policy)
    summary = {
        "arm": arm,
        "log": str(log_path),
        "policy": args.policy,
        "env": env_pairs,
        "accounting": dict(counters),
    }
    with open(out_dir / "replay_decisions.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out_dir / "replay_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    total = counters.get("decisions_total", 0)
    print(
        f"[replay_admission] arm={arm} policy={args.policy} decisions={total} "
        f"computable={counters.get('decisions_computable', 0)} "
        f"s0_sig_match={counters.get('stage0_signals_match', 0)}/"
        f"{counters.get('stage0_signals_mismatch', 0)} "
        f"s0_match={counters.get('stage0_match', 0)}/{counters.get('stage0_mismatch', 0)} "
        f"final_match={counters.get('decision_match', 0)}/"
        f"{counters.get('decision_mismatch', 0)}"
    )
    for k in sorted(counters):
        if k.startswith("v1_action_"):
            print(f"[replay_admission]   {k} = {counters[k]}")
    print(f"[replay_admission] wrote {out_dir}")


if __name__ == "__main__":
    main()
