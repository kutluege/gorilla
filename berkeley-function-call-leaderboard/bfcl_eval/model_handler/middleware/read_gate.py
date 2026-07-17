"""
SS6 read-time mechanism  --  Plan 3 Step 5 (G9).

One-shot margin-triggered disambiguation over a retrieve RESULT. It never
writes memory (read-only by construction: the only thing it can change is the
tool-result payload string), it is NOT hierarchical descent, and it performs
at most ONE refinement round.

Flow (per gated read call):
  1. single pass through the real path -- the backend's own ranking (KV:
     BM25+ over keys via *_key_search; Vector: cosine via *_retrieve);
  2. margin = top1 - top2 over the returned scores; margin > threshold ->
     return top-1 (the expected majority case);
  3. else the ambiguous set = top-1's delta-neighborhood, bounded to 2-4
     entries; ONE refinement round: append to the query the first token from
     the ambiguous candidates' DISTINCT vocabulary that also appears in the
     query; re-rank WITHIN the set only (simulate_kv / raw-MiniLM
     simulate_vector -- never the whitened space);
  4. resolved -> return the refined top-1; still ambiguous (or no usable
     token) -> return the ENTIRE bounded set and let the model choose.

Gated ops are the ranked-retrieval calls only; exact-key KV ``*_retrieve`` has
no ranking to disambiguate. Config: ``GOV_READ_ENABLED`` (default 0),
``GOV_READ_SHADOW`` (default 1), ``GOV_READ_MARGIN``, ``GOV_READ_SET_MAX``
(clamped 2..4).
"""

import json
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

READ_GATED_OPS = {
    "kv": {"core_memory_key_search", "archival_memory_key_search"},
    "vector": {"core_memory_retrieve", "archival_memory_retrieve"},
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return {t for t in _TOKEN_RE.findall(str(text).lower().replace("_", " "))
            if len(t) > 2}


@dataclass
class ReadGateOutcome:
    action: str  # "pass" | "top1_pass" | "refined_top1" | "return_set"
    margin: Optional[float] = None
    ambiguous_refs: List[str] = field(default_factory=list)
    refinement_token: Optional[str] = None
    refined_query: Optional[str] = None
    new_payload: Optional[str] = None  # serialized replacement result, live mode
    refinement_rounds: int = 0

    def to_log(self) -> dict:
        return {
            "action": self.action,
            "margin": self.margin,
            "ambiguous_refs": self.ambiguous_refs,
            "refinement_token": self.refinement_token,
            "refined_query": self.refined_query,
            "refinement_rounds": self.refinement_rounds,
            "payload_rewritten": self.new_payload is not None,
        }


def parse_ranking(backend: str, result: dict):
    """-> [(score, ref, text), ...] descending, or None if not a ranked payload.
    KV: {"ranked_results": [[score, key], ...]} (ref == text == key).
    Vector: {"result": [{"id", "similarity_score", "text"}, ...]}."""
    try:
        if backend == "kv":
            rr = result["ranked_results"]
            out = [(float(s), str(k), str(k)) for s, k in rr]
        else:
            out = [
                (float(e["similarity_score"]), str(e["id"]), str(e["text"]))
                for e in result["result"]
            ]
    except (KeyError, TypeError, ValueError):
        return None
    return sorted(out, key=lambda t: -t[0])


def serialize_ranking(backend: str, ranked) -> str:
    if backend == "kv":
        return json.dumps({"ranked_results": [[s, ref] for s, ref, _ in ranked]})
    return json.dumps({
        "result": [
            {"id": int(ref), "similarity_score": s, "text": text}
            for s, ref, text in ranked
        ]
    })


def distinct_refinement_token(query: str, candidates) -> Optional[str]:
    """First token that is DISTINCT to exactly one ambiguous candidate AND
    appears in the query -- the SS6 refinement vocabulary rule."""
    q_toks = _tokens(query)
    cand_toks = [_tokens(text) for _, _, text in candidates]
    for i, toks in enumerate(cand_toks):
        others = set().union(*(t for j, t in enumerate(cand_toks) if j != i)) \
            if len(cand_toks) > 1 else set()
        for tok in sorted(toks - others):
            if tok in q_toks:
                return tok
    return None


def default_rerank(backend: str, candidates, query: str):
    """Re-rank WITHIN the ambiguous set via the backend-faithful simulators."""
    from bfcl_eval.model_handler.middleware.retrieval_sim import (
        simulate_kv,
        simulate_vector,
    )

    texts = [text for _, _, text in candidates]
    if backend == "kv":
        ranked = simulate_kv(texts, query, k=len(texts))
        order = {t: s for s, t in ranked}
        rescored = [(order.get(text, 0.0), ref, text)
                    for _, ref, text in candidates]
    else:
        ranked = simulate_vector(texts, query, k=len(texts))
        order = {idx: s for s, idx in ranked}
        rescored = [(order.get(i, 0.0), candidates[i][1], candidates[i][2])
                    for i in range(len(candidates))]
    return sorted(rescored, key=lambda t: -t[0])


def gate_read(backend: str, query: str, result: dict, margin_thr: float,
              set_max: int, rerank_fn: Callable = default_rerank) -> ReadGateOutcome:
    """Pure gate decision for one ranked read result (never touches memory)."""
    set_max = max(2, min(4, int(set_max)))
    ranked = parse_ranking(backend, result)
    if not ranked or len(ranked) < 2:
        return ReadGateOutcome(action="pass")
    margin = ranked[0][0] - ranked[1][0]
    if margin > margin_thr:
        return ReadGateOutcome(
            action="top1_pass", margin=margin,
            new_payload=serialize_ranking(backend, ranked[:1]),
        )
    # delta-neighborhood of top-1, bounded 2..set_max
    top_score = ranked[0][0]
    ambiguous = [r for r in ranked if top_score - r[0] <= margin_thr][:set_max]
    if len(ambiguous) < 2:
        ambiguous = ranked[:2]
    refs = [ref for _, ref, _ in ambiguous]
    token = distinct_refinement_token(query, ambiguous)
    if token is None:
        return ReadGateOutcome(
            action="return_set", margin=margin, ambiguous_refs=refs,
            new_payload=serialize_ranking(backend, ambiguous),
        )
    refined_query = f"{query} {token}"
    rescored = rerank_fn(backend, ambiguous, refined_query)  # the ONE round
    new_margin = rescored[0][0] - rescored[1][0] if len(rescored) > 1 else 0.0
    if new_margin > 0:
        return ReadGateOutcome(
            action="refined_top1", margin=margin, ambiguous_refs=refs,
            refinement_token=token, refined_query=refined_query,
            refinement_rounds=1,
            new_payload=serialize_ranking(backend, rescored[:1]),
        )
    return ReadGateOutcome(
        action="return_set", margin=margin, ambiguous_refs=refs,
        refinement_token=token, refined_query=refined_query,
        refinement_rounds=1,
        new_payload=serialize_ranking(backend, ambiguous),
    )
