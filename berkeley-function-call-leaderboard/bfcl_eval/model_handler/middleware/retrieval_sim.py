"""
Retrieval simulation core (Stage 2 / offline replay).
====================================================

Backend-faithful re-implementations of the two BFCL memory retrieval paths, used
both by the live Stage 2 escalation layer and by the offline replay instrument
(``bfcl_eval/scripts/replay_geometry_deltaH.py``). Build once, share everywhere
(STAGE0 analysis SS6/SS8.1).

Fidelity contracts (verified against the actual backends):

* ``simulate_kv``   -- ``rank_bm25.BM25Plus`` over **key names only**, tokenization
  byte-identical to ``memory_kv._similarity_search`` (memory_kv.py:71-88):
  ``text.replace('_', ' ').lower().split()``. Values are never scored -- the real
  ``*_memory_key_search`` ops pass only the key list.
* ``simulate_vector`` -- probe embedded with **raw MiniLM,
  ``normalize_embeddings=True``**, cosine by dot product against stored-text
  embeddings. The real path is FAISS ``IndexFlatIP`` over L2-normalized float32
  vectors (memory_vector.py:244-256, 309-333) -- an *exact* inner-product search,
  so numpy dot on the same normalized embeddings yields the same ranking at
  N <= 57. The ABTT-whitened space is Stage 0's decision space and NEVER enters
  this simulation.

Per-tier fidelity: the real ops search a single tier's store. Callers must pass
one tier's keys/texts (plus any provisional candidate), not the merged matrix
Stage 0 uses for redundancy.

Pure logic, no BFCL imports. ``rank_bm25`` is imported lazily so the module can
be imported (e.g. for probe-only use) without it.
"""

import math
import threading
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

# Reuse the lazily-loaded, thread-locked MiniLM singleton (CPU) -- the same
# encoder instance the vector backend and Stage 0 use.
from bfcl_eval.model_handler.middleware.semantic_entropy import _get_encoder

# ---------------------------------------------------------------------------
# KV: BM25Plus over key names (byte-identical to memory_kv._similarity_search)
# ---------------------------------------------------------------------------


def kv_tokenize(text: str) -> List[str]:
    """Byte-identical to memory_kv.py:83/85: ``replace('_',' ').lower().split()``."""
    return text.replace("_", " ").lower().split()


def simulate_kv(keys: Sequence[str], probe: str, k: int = 5) -> List[Tuple[float, str]]:
    """Rank ``keys`` for ``probe`` exactly as ``memory_kv._similarity_search`` does.

    Returns the same ``ranked_results`` shape: ``[(score, key), ...]`` sorted by
    score descending (Python's stable sort, same tie behavior as the backend).
    """
    if not keys:
        return []
    from rank_bm25 import BM25Plus  # lazy: repo dependency via memory_kv

    tokenized_corpus = [kv_tokenize(text) for text in keys]
    bm25 = BM25Plus(tokenized_corpus)
    tokenized_query = kv_tokenize(probe)
    scores = bm25.get_scores(tokenized_query)
    ranked_results = sorted(zip(scores, keys), key=lambda x: x[0], reverse=True)
    return [(float(s), t) for s, t in ranked_results[:k]]


# ---------------------------------------------------------------------------
# Vector: raw-MiniLM cosine (exact equivalent of faiss.IndexFlatIP at N <= 57)
# ---------------------------------------------------------------------------

_EMB_CACHE: dict = {}
_EMB_LOCK = threading.Lock()


def default_encode(texts: List[str]) -> np.ndarray:
    """L2-normalized MiniLM embeddings with a global text->vector cache.

    Matches ``VectorStore._embed`` (memory_vector.py:251-256):
    ``ENCODER.encode(texts, normalize_embeddings=True)`` as float32.
    """
    encoder = _get_encoder()
    if encoder is None:
        raise RuntimeError(
            "[retrieval_sim] sentence-transformers encoder unavailable; "
            "simulate_vector cannot run without embeddings."
        )
    missing = [t for t in texts if t not in _EMB_CACHE]
    if missing:
        vecs = encoder.encode(missing, normalize_embeddings=True)
        with _EMB_LOCK:
            for t, v in zip(missing, np.asarray(vecs, dtype=np.float32)):
                _EMB_CACHE[t] = v
    return np.stack([_EMB_CACHE[t] for t in texts]).astype(np.float32)


def simulate_vector(
    texts: Sequence[str],
    probe: str,
    encode: Optional[Callable[[List[str]], np.ndarray]] = None,
    k: int = 5,
) -> List[Tuple[float, int]]:
    """Rank stored ``texts`` for ``probe`` exactly as ``VectorStore.retrieve`` does.

    Returns ``[(score, index), ...]`` (index into ``texts``), score = inner product
    of L2-normalized embeddings (== cosine), sorted descending, at most
    ``min(k, len(texts))`` entries -- mirroring memory_vector.py:309-333.
    """
    if not texts:
        return []
    encode = encode or default_encode
    corpus = encode(list(texts))
    q = encode([probe])[0]
    scores = corpus @ q
    top = min(k, len(texts))
    order = np.argsort(-scores, kind="stable")[:top]
    return [(float(scores[i]), int(i)) for i in order]


# ---------------------------------------------------------------------------
# Score-distribution metrics (SS8.1: margin is the decision metric; entropy /
# n_eff are logged diagnostics only -- temperature affects logged H, never a
# decision).
# ---------------------------------------------------------------------------


def margin(scores: Sequence[float]) -> float:
    """top1 - top2 of a ranked (descending) score list; 0.0 when < 2 scores."""
    if len(scores) < 2:
        return 0.0
    return float(scores[0] - scores[1])


def entropy(scores: Sequence[float], temperature: float = 1.0, k: int = 5) -> float:
    """Softmax entropy (nats) over the top-k scores. Deterministic, no sampling."""
    s = np.asarray(sorted(scores, reverse=True)[:k], dtype=np.float64)
    if s.size == 0:
        return 0.0
    t = max(float(temperature), 1e-9)
    z = s / t
    z -= z.max()  # numerical stability
    p = np.exp(z)
    p /= p.sum()
    return float(-np.sum(p * np.log(np.clip(p, 1e-12, None))))


def n_eff(h: float) -> float:
    """Effective number of retrieval targets: exp(H) for H in nats."""
    return float(math.exp(h))


# ---------------------------------------------------------------------------
# Delta-H (the SS8.4 / SS4.4 "dH_komsu" estimator): change in a neighbor's
# own-probe top-k retrieval entropy when a candidate is provisionally added.
# ---------------------------------------------------------------------------


def probe_scores(
    backend: str,
    corpus: Sequence[str],
    probe: str,
    encode: Optional[Callable] = None,
    k: int = 5,
) -> List[float]:
    """Ranked (descending) top-k scores of ``probe`` against ``corpus`` under the
    given backend's retrieval model. KV corpus = key names; Vector corpus = texts."""
    if backend == "kv":
        return [s for s, _ in simulate_kv(corpus, probe, k=k)]
    return [s for s, _ in simulate_vector(corpus, probe, encode=encode, k=k)]


def delta_h_for_probes(
    backend: str,
    corpus: Sequence[str],
    candidate_entry: str,
    probes: Sequence[str],
    encode: Optional[Callable] = None,
    k: int = 5,
    temperature: float = 1.0,
) -> dict:
    """For one neighbor's probe list: H_before over ``corpus``, H_after over
    ``corpus + [candidate_entry]``, per probe; plus the mean dH.

    ``candidate_entry`` is the candidate's *corpus representation*: its key name
    for KV, its stored text for Vector.
    """
    corpus_after = list(corpus) + [candidate_entry]
    rows = []
    for p in probes:
        h_before = entropy(
            probe_scores(backend, corpus, p, encode=encode, k=k), temperature, k
        )
        h_after = entropy(
            probe_scores(backend, corpus_after, p, encode=encode, k=k), temperature, k
        )
        rows.append(
            {
                "probe": p,
                "H_before": round(h_before, 6),
                "H_after": round(h_after, 6),
                "dH": round(h_after - h_before, 6),
            }
        )
    mean_dh = sum(r["dH"] for r in rows) / len(rows) if rows else 0.0
    mean_h_after = sum(r["H_after"] for r in rows) / len(rows) if rows else 0.0
    return {
        "per_probe": rows,
        "dH_mean": round(mean_dh, 6),
        "H_after_mean": round(mean_h_after, 6),
        "n_eff_after_mean": round(n_eff(mean_h_after), 6),
    }
