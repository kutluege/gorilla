"""
Stage-0 geometry gate (Plan v2 Step 2, plan SS8).
================================================

A stable interface around the *unchanged* Stage-0 decision core
(``governance_filter.compute_signals`` / ``decide`` — deliberately NOT moved:
the causally validated component stays byte-identical where the legacy policy
runs it), plus the two evidence-scoped hardenings of plan SS8.2:

1. **Exact-duplicate fast path** (before any embedding): normalized
   byte-equality of the candidate's composite text against stored items ->
   NOOP without encoding. Removes the only observed false-negative class of
   geometry (identical re-writes at whitened sim 0.99 < sim_high from ABTT
   noise). Only fires when preflight predicts backend acceptance — a synthetic
   success must never mask a real backend error (same guard as every NOOP).

2. **Retrievability floor** on the confident-ADD branch: the candidate's own
   key/keyword probe must rank it top-``FLOOR_TOP_N`` in the provisional tier
   store (one ``simulate_*`` call). A "novel" write its own probe cannot find
   is malformed/off-topic -> ESCALATE, not ADD.

Thresholds are untouched (SIM_HIGH / DELTA / SAGE tau survive as configured);
the ambiguity band is re-estimated only via the SS13 calibration protocol.

Used by the new admission policies (Step 4/5). The legacy policy keeps calling
the core functions directly, so ``GOV_POLICY=legacy_full`` behavior cannot
drift by construction.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from bfcl_eval.model_handler.middleware.governance_filter import (
    GovConfig,
    GovDecision,
    GovernanceCache,
    Stage0Signals,
    ThresholdState,
    WriteCandidate,
    _normalize_for_match,
    compute_signals,
    decide,
)

FLOOR_TOP_N = 3  # plan SS8.2: own-probe must rank the candidate top-3
FLOOR_K = 5


@dataclass
class GeometryResult:
    decision: GovDecision
    reason: str  # legacy-style reason ("redundant"/"novel"/"ambiguous"/...)
    reason_code: str  # closed-vocabulary gov2 code (plan SS17.1)
    signals: Stage0Signals
    v_w: Optional[np.ndarray]  # None on the no-encoding fast path
    fast_path: bool = False
    floor_rank: Optional[int] = None  # rank from the retrievability floor probe


_REASON_CODE_MAP = {
    ("NOOP", "redundant"): "s0_noop",
    ("ADD", "novel"): "s0_confident_add",
    ("ADD", "empty_memory"): "s0_confident_add",
    ("ADD", "noop_blocked_by_preflight"): "s0_noop_preflight_blocked",
    ("ESCALATE", "ambiguous"): "s0_escalate",
}


def _exact_duplicate(candidate: WriteCandidate, cache: GovernanceCache) -> bool:
    """Normalized byte-equality of the candidate composite text against stored
    items (both tiers — Stage 0 judges redundancy against everything stored)."""
    norm = _normalize_for_match(candidate.text)
    if not norm:
        return False
    return any(
        _normalize_for_match(it.text) == norm for it in cache.all_items()
    )


def retrievability_floor(
    candidate: WriteCandidate, cache: GovernanceCache
) -> Optional[int]:
    """Rank of the candidate under its own key/keyword probe in the provisional
    tier store (plan SS8.2 point 2). None when the tier is empty (trivially
    retrievable — the floor cannot bind on a 1-item corpus)."""
    from bfcl_eval.model_handler.middleware.entropy_metrics import rank_self
    from bfcl_eval.model_handler.middleware.probe_gen import content_tokens_ordered
    from bfcl_eval.model_handler.middleware.retrieval_sim import (
        simulate_kv,
        simulate_vector,
    )

    tier_items = cache.items[candidate.tier]
    if not tier_items:
        return None
    if candidate.backend == "kv":
        key = str(candidate.args["key"])
        corpus = list(tier_items.keys()) + [key]
        probe = key.replace("_", " ")
        ranked = simulate_kv(corpus, probe, k=FLOOR_K)
        return rank_self(ranked, key, k=FLOOR_K)
    texts = [it.text for it in tier_items.values()]
    corpus = texts + [candidate.text]
    toks = content_tokens_ordered(candidate.text, limit=8)
    probe = " ".join(toks) if toks else candidate.text
    ranked = simulate_vector(corpus, probe, k=FLOOR_K)
    return rank_self(ranked, len(corpus) - 1, k=FLOOR_K)


class GeometryGate:
    """Stage 0 behind a stable interface: ``evaluate(candidate, ...) ->
    GeometryResult``. ``whiten_fn`` is injected (the session's encoder+ABTT
    pipeline) and only called when the fast path misses, so exact duplicates
    cost zero model calls."""

    def __init__(self, cfg: GovConfig):
        self.cfg = cfg

    def evaluate(
        self,
        candidate: WriteCandidate,
        cache: GovernanceCache,
        thresholds: ThresholdState,
        preflight_ok: bool,
        whiten_fn: Callable[[str], np.ndarray],
    ) -> GeometryResult:
        # -- SS8.2(1): exact-duplicate fast path, no encoding ------------------
        if preflight_ok and cache.total_size() > 0 and _exact_duplicate(candidate, cache):
            sig = Stage0Signals(
                n_items=cache.total_size(), rho=cache.density(candidate.tier)
            )
            # Keep the SAGE EMA sequence identical to the non-fast-path order
            # of events: every governed decision advances the tier threshold.
            sig.tau_t = thresholds.update(candidate.tier, sig.rho)
            sig.sim_max = 1.0  # exact duplicate by definition
            sig.r = 0.0
            return GeometryResult(
                decision=GovDecision.NOOP,
                reason="exact_duplicate",
                reason_code="s0_exact_dup",
                signals=sig,
                v_w=None,
                fast_path=True,
            )

        # -- unchanged decision core ------------------------------------------
        v_w = whiten_fn(candidate.text)
        signals = compute_signals(v_w, candidate, cache, thresholds, self.cfg)
        decision, reason = decide(candidate, signals, cache, self.cfg, preflight_ok)

        # -- SS8.2(2): retrievability floor on the confident-ADD branch --------
        floor_rank = None
        if decision == GovDecision.ADD and reason == "novel":
            floor_rank = retrievability_floor(candidate, cache)
            if floor_rank is not None and floor_rank > FLOOR_TOP_N:
                return GeometryResult(
                    decision=GovDecision.ESCALATE,
                    reason="unretrievable_novel",
                    reason_code="s0_unretrievable_escalate",
                    signals=signals,
                    v_w=v_w,
                    floor_rank=floor_rank,
                )

        reason_code = _REASON_CODE_MAP.get((decision.value, reason), "s0_escalate")
        return GeometryResult(
            decision=decision,
            reason=reason,
            reason_code=reason_code,
            signals=signals,
            v_w=v_w,
            floor_rank=floor_rank,
        )
