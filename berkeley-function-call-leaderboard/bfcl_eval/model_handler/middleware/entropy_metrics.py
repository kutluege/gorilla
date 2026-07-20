"""
Margin-entropy signal functions (Plan v2 Step 3, plan SS9.1 / SS12).
===================================================================

Pure functions over the outputs of ``retrieval_sim`` — the only *new*
mathematics of the two-stage architecture: ``nmargin``, ``n_eff_norm``,
``disp``, ``churn``, ``delta_h_self``, plus the rank/aggregation helpers the
Stage-1 rule consumes. Everything is deterministic given its inputs; no model
calls happen here (``delta_h_self`` delegates to ``retrieval_sim``, whose
encoder is the shared MiniLM singleton).

No imports from the BFCL harness — unit-testable stand-alone (SS14 Step 3).
"""

import math
from typing import List, Optional, Sequence, Tuple

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Scalar signal functions
# ---------------------------------------------------------------------------


def nmargin(scores: Sequence[float], backend: str) -> float:
    """Normalized top1-top2 margin (plan SS9.1).

    KV (BM25 is scale-free across store sizes): ``(s1 - s2) / (|s1| + eps)``.
    Vector (cosine already bounded): raw margin ``s1 - s2``.
    Fewer than 2 scores -> 0.0 (matches ``retrieval_sim.margin``).
    """
    if len(scores) < 2:
        return 0.0
    s1, s2 = float(scores[0]), float(scores[1])
    if backend == "kv":
        return (s1 - s2) / (abs(s1) + _EPS)
    return s1 - s2


def n_eff_norm(h: float, n_items: int, k: int = 5) -> float:
    """Store-size-normalized effective retrieval-target count (plan SS9.1):
    ``exp(H) / min(k, n_items + 1)`` — the denominator is the provisional
    corpus size clipped at top-k, so the value lies in [1/k, 1] for a
    well-formed H over ``min(k, n_items+1)`` scores.
    """
    denom = max(1, min(int(k), int(n_items) + 1))
    return float(math.exp(h)) / denom


def disp(scores: Sequence[float]) -> float:
    """NQC-style dispersion: population std of the top-k scores over their mean
    (plan SS9.1 — logged v1, ablations 13/14 decide). 0.0 on degenerate input."""
    s = [float(x) for x in scores]
    if not s:
        return 0.0
    mean = sum(s) / len(s)
    if abs(mean) < _EPS:
        return 0.0
    var = sum((x - mean) ** 2 for x in s) / len(s)
    return math.sqrt(var) / abs(mean)


def churn(top_before: Sequence, top_after: Sequence) -> float:
    """Jaccard *distance* of the pre/post top-N retrieved-identity sets
    (plan SS9.1: rank-displacement summary; N=3 at the call site). Both empty
    -> 0.0 (nothing displaced)."""
    a, b = set(top_before), set(top_after)
    if not a and not b:
        return 0.0
    union = a | b
    return 1.0 - len(a & b) / len(union)


# ---------------------------------------------------------------------------
# Rank helpers (replace the brittle binary all-top-1 of the old Stage 2)
# ---------------------------------------------------------------------------


def rank_self(ranked: Sequence[Tuple[float, object]], target, k: int = 5) -> int:
    """1-indexed rank of ``target`` (a KV key string or a Vector corpus index)
    in a ``[(score, identity), ...]`` ranking; ``k + 1`` when absent from the
    top-k (a bounded sentinel, monotone with 'worse')."""
    for pos, (_, identity) in enumerate(ranked, start=1):
        if identity == target:
            return pos
    return int(k) + 1


def median(values: Sequence[float]) -> Optional[float]:
    """Plain median (average of middle pair on even n); None on empty input.
    Stage-1 aggregates per-channel signals by median, never min-pooling across
    channels (plan SS9.1)."""
    if not values:
        return None
    s = sorted(float(v) for v in values)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


# ---------------------------------------------------------------------------
# dH_self: the candidate's effect on its own retrieval neighborhood
# ---------------------------------------------------------------------------


def delta_h_self(
    backend: str,
    corpus: Sequence[str],
    candidate_entry: str,
    probes: Sequence[str],
    k: int = 5,
    temperature: float = 1.0,
) -> dict:
    """Mean over the candidate-channel ``probes`` of H_after - H_before with the
    candidate provisionally appended (plan SS9.1 dH_self). A thin, explicitly
    named wrapper over ``retrieval_sim.delta_h_for_probes`` (which already
    isolates the provisional insert via a list copy); the distinction from
    dH_neighbor is purely the probe source (candidate-locating decision probes
    vs neighbors' own cached probes)."""
    from bfcl_eval.model_handler.middleware.retrieval_sim import delta_h_for_probes

    res = delta_h_for_probes(
        backend, corpus, candidate_entry, probes, k=k, temperature=temperature
    )
    return {
        "per_probe": res["per_probe"],
        "dH_self_mean": res["dH_mean"],
        "H_after_mean": res["H_after_mean"],
    }


# ---------------------------------------------------------------------------
# Top-set extraction for churn
# ---------------------------------------------------------------------------


def top_identity_set(ranked: Sequence[Tuple[float, object]], n: int = 3) -> List:
    """The identities of the top-n entries of a ``[(score, identity), ...]``
    ranking (order-insensitive consumer: ``churn`` takes sets)."""
    return [identity for _, identity in ranked[: int(n)]]
