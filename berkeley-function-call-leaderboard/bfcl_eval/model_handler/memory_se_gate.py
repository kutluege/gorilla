"""Semantic-entropy gating middleware for BFCL memory (measurement spike, log-only).

This module is pure logic. It clusters K sampled answers by meaning, computes the
semantic entropy over those clusters, and persists per-question samples to a
sidecar JSONL so the entropy-vs-correctness signal can be measured offline
(see ``bfcl_eval/analysis/se_report.py``).

It reuses the all-MiniLM encoder already loaded by the vector memory backend, so
no new heavy dependency is introduced. The encoder is imported lazily so that the
default (gating off) code path never pays for it.

Nothing here changes the recorded benchmark output: the caller logs the entropy
but keeps the deterministic temp-0 generation as the answer.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from math import log
from pathlib import Path
from typing import Optional

# Lazy handle to the shared encoder; only loaded if gating is actually exercised.
_ENCODER = None
_ENCODER_LOCK = threading.Lock()


def _get_encoder():
    """Return the shared all-MiniLM encoder (loaded once by the vector backend)."""
    global _ENCODER
    if _ENCODER is None:
        with _ENCODER_LOCK:
            if _ENCODER is None:
                from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.memory_vector import (
                    ENCODER,
                )

                _ENCODER = ENCODER
    return _ENCODER


# Same characters the BFCL scorer (agentic_checker.standardize_string) ignores.
_PUNCT_RE = re.compile(r"[\,\.\/\-\_\*\^\(\)]")


def normalize_answer(s: str) -> str:
    """Lowercase and strip the punctuation the contains-match scorer ignores.

    Mirrors ``agentic_checker.standardize_string`` so clustering sees answers the
    way the scorer does (e.g. "April 1, 2024" == "April 1 2024").
    """
    if not isinstance(s, str):
        s = str(s)
    return _PUNCT_RE.sub("", s).strip().lower().replace("'", '"')


def cluster_semantic_equivalence(samples, threshold: float = 0.70, encoder=None):
    """Greedy single-linkage clustering of short answers by cosine similarity.

    Returns a partition of ``range(len(samples))`` as a list of index lists.
    Embeddings are L2-normalized, so the dot product is the cosine similarity.
    """
    if not samples:
        return []
    enc = encoder or _get_encoder()
    embs = enc.encode(
        [normalize_answer(s) for s in samples], normalize_embeddings=True
    )
    clusters: list[list[int]] = []
    for i, e in enumerate(embs):
        placed = False
        for c in clusters:
            if float(e @ embs[c[0]]) >= threshold:
                c.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    return clusters


def semantic_entropy(samples, threshold: float = 0.70, encoder=None):
    """H_sem = -sum_c p(c) log p(c), with p(c) = |c| / K.

    Returns ``(entropy, n_clusters, clusters)``. Entropy is in nats; 0.0 means all
    K samples collapsed into a single meaning. Returns ``(inf, 0, [])`` for an
    empty sample set.
    """
    if not samples:
        return float("inf"), 0, []
    clusters = cluster_semantic_equivalence(
        samples, threshold=threshold, encoder=encoder
    )
    k = len(samples)
    h = -sum((len(c) / k) * log(len(c) / k) for c in clusters)
    return h, len(clusters), clusters


def prompt_cache_key(formatted_prompt: str, temperature: float, k: int, seed) -> str:
    """Stable cache key for a (prompt, sampling-config) so re-runs reuse samples."""
    payload = json.dumps(
        {"p": formatted_prompt, "t": temperature, "k": k, "s": seed},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_SIDECAR_LOCK = threading.Lock()


def write_se_sidecar(model_result_dir, record: dict) -> Optional[Path]:
    """Persist one scored question's gate record to its own sidecar file.

    One file per test id keeps concurrent inference threads from clashing without
    a shared-file lock dance. Written with mode "w" so re-runs are idempotent
    (memory questions produce a single final answer). Returns the path written,
    or ``None`` when no result directory is available.
    """
    if not model_result_dir:
        return None
    out_dir = Path(model_result_dir) / "se_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{record['id']}.jsonl"
    line = json.dumps(record, ensure_ascii=False)
    with _SIDECAR_LOCK:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(line + "\n")
    return out_file
