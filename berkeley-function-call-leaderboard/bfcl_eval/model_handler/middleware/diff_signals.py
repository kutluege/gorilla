"""
Marginal-diff signals (H-Nav Stage 1.3, the coupling falsifier).
================================================================

Stage 0 compares the candidate's WHOLE composite text against whole stored
items. When an update appends one clause to a long record, most of the text is
unchanged, so ``sim_max`` stays high and the gate can read a genuine update as a
duplicate. Measured on the harvest: 47 of 52 near-duplicate update/replace
decisions had ``sim_max >= 0.90`` while carrying real new content ("Water
reservoir area is slightly misaligned...", "No return needed -- damaged unit
will remain with him").

This module scores the MARGINAL content instead of the blob: what did this write
actually add, and is that addition novel with respect to the item it folds into
and to the rest of the store?

Same discipline as ``entropy_metrics``: pure functions over plain dicts and
strings, no BFCL harness imports, no ground truth (the falsifier must never see
benchmark answers -- ``test_diff_signals.py`` enforces that with an AST scan).
Embedding-dependent signals take injected callables, so the module is testable
without a model and reusable at runtime if Stage 2 ever wires it in.

Feature names are prefixed ``diff_`` and registered in
``calibrate_margin_entropy.DIFF_FEATURES``.
"""

import difflib
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from bfcl_eval.model_handler.middleware.probe_gen import content_tokens_ordered
from bfcl_eval.model_handler.middleware.recsum_blobdiff import split_propositions

_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")
_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}"
    r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)\b"
)
_VN_NEIGHBORS = 8

# Feature names in a fixed order -- the design matrix depends on it.
LEXICAL_FEATURES = (
    "has_old",
    "diff_edit_ratio",
    "diff_added_frac",
    "diff_n_added_clauses",
    "diff_n_dropped_clauses",
    "diff_novel_tok_frac_target",
    "diff_novel_tok_frac_store",
    "diff_n_new_values",
    "diff_n_new_values_unseen_in_store",
    "diff_n_new_entities",
    "diff_n_new_dates",
)
SEMANTIC_FEATURES = (
    "diff_sim_max",
    "diff_sim_max_raw",
    "diff_cos_target",
    "diff_cos_store_max",
    "diff_dSvn",
    "diff_rank_self_post",
)
ALL_FEATURES = LEXICAL_FEATURES + SEMANTIC_FEATURES


# ---------------------------------------------------------------------------
# The diff itself
# ---------------------------------------------------------------------------


@dataclass
class DiffParts:
    added_text: str
    dropped_text: str
    added_clauses: List[str] = field(default_factory=list)
    dropped_clauses: List[str] = field(default_factory=list)
    has_old: bool = True
    edit_ratio: float = 1.0


def marginal_diff(old: Optional[str], new: str) -> DiffParts:
    """Content ``new`` introduces over ``old`` (and content it drops).

    Character-level opcodes give the added/dropped spans (this is what the
    semantic signals score); clause-level set difference over
    ``split_propositions`` gives the countable structural signals. A fresh add
    (``old is None``) is entirely marginal by definition.
    """
    new = str(new or "")
    if old is None:
        return DiffParts(
            added_text=new,
            dropped_text="",
            added_clauses=split_propositions(new),
            dropped_clauses=[],
            has_old=False,
            edit_ratio=1.0,
        )
    old = str(old)
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    added, dropped = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            added.append(new[j1:j2])
        if tag in ("delete", "replace"):
            dropped.append(old[i1:i2])
    old_props = {p.lower() for p in split_propositions(old)}
    new_props = split_propositions(new)
    return DiffParts(
        added_text=" ".join(s.strip() for s in added if s.strip()),
        dropped_text=" ".join(s.strip() for s in dropped if s.strip()),
        added_clauses=[p for p in new_props if p.lower() not in old_props],
        dropped_clauses=[
            p for p in split_propositions(old)
            if p.lower() not in {q.lower() for q in new_props}
        ],
        has_old=True,
        edit_ratio=1.0 - sm.ratio(),
    )


# ---------------------------------------------------------------------------
# Lexical signals
# ---------------------------------------------------------------------------


def _values(text: str) -> List[str]:
    """Concrete values (numbers, money, dates, quoted spans) via the same
    extractor Stage 0's verbatim channel uses."""
    from bfcl_eval.model_handler.middleware.governance_filter import (
        extract_verbatim_values,
    )

    return extract_verbatim_values(text or "")


def _entities(text: str) -> List[str]:
    return _ENTITY_RE.findall(text or "")


def lexical_features(parts: DiffParts, old: Optional[str], new: str,
                     store_texts: Sequence[str]) -> Dict[str, float]:
    old_s = old or ""
    store_blob = " ||| ".join(store_texts)
    tok_new = set(content_tokens_ordered(new))
    tok_old = set(content_tokens_ordered(old_s))
    tok_added = set(content_tokens_ordered(parts.added_text))
    tok_store = set(content_tokens_ordered(store_blob))

    new_values = [v for v in _values(parts.added_text) if v not in _values(old_s)]
    unseen = [v for v in new_values if v.lower() not in store_blob.lower()]
    old_ents = set(_entities(old_s))
    new_ents = {e for e in _entities(parts.added_text) if e not in old_ents}
    old_dates = set(_DATE_RE.findall(old_s))
    new_dates = {d for d in _DATE_RE.findall(parts.added_text) if d not in old_dates}

    return {
        "has_old": float(parts.has_old),
        "diff_edit_ratio": round(parts.edit_ratio, 6),
        "diff_added_frac": round(len(parts.added_text) / max(1, len(new)), 6),
        "diff_n_added_clauses": float(len(parts.added_clauses)),
        "diff_n_dropped_clauses": float(len(parts.dropped_clauses)),
        "diff_novel_tok_frac_target": round(
            len(tok_new - tok_old) / max(1, len(tok_new)), 6
        ),
        "diff_novel_tok_frac_store": round(
            len(tok_added - tok_store) / max(1, len(tok_added)), 6
        ),
        "diff_n_new_values": float(len(new_values)),
        "diff_n_new_values_unseen_in_store": float(len(unseen)),
        "diff_n_new_entities": float(len(new_ents)),
        "diff_n_new_dates": float(len(new_dates)),
    }


# ---------------------------------------------------------------------------
# Semantic signals
# ---------------------------------------------------------------------------


def _cos_max(vec, mat) -> float:
    import numpy as np

    if mat is None or len(mat) == 0:
        return 0.0
    v = np.asarray(vec, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return 0.0
    M = np.asarray(mat, dtype=np.float64)
    norms = np.linalg.norm(M, axis=1)
    norms[norms < 1e-9] = 1.0
    return float(np.max((M @ v) / (norms * n)))


def semantic_features(
    parts: DiffParts,
    backend: str,
    tier_items: Dict[str, str],
    store_texts: Sequence[str],
    old: Optional[str],
    new: str,
    ref,
    whiten: Optional[Callable[[str], "object"]] = None,
    encode: Optional[Callable[[List[str]], "object"]] = None,
) -> Dict[str, Optional[float]]:
    """Embedding-space signals over the marginal diff.

    ``diff_sim_max`` is the headline falsifier: Stage 0's own quantity, computed
    on ``added_text`` in the SAME ABTT-whitened space, so it is directly
    comparable to the logged ``sim_max``. ``whiten`` / ``encode`` are injected;
    when absent the corresponding features are None and the caller decides
    (the driver always supplies them; unit tests inject fakes).
    """
    import numpy as np

    from bfcl_eval.model_handler.middleware.entropy_metrics import (
        rank_self,
        vn_entropy,
    )

    out: Dict[str, Optional[float]] = {k: None for k in SEMANTIC_FEATURES}
    added = parts.added_text.strip()
    if not added:
        # A write that adds no characters is maximally redundant by every
        # marginal measure; encode that explicitly rather than leaving None.
        out.update({
            "diff_sim_max": 1.0, "diff_sim_max_raw": 1.0, "diff_cos_target": 1.0,
            "diff_cos_store_max": 1.0, "diff_dSvn": 0.0,
        })

    if whiten is not None and store_texts and added:
        v = whiten(added)
        M = np.stack([whiten(t) for t in store_texts])
        out["diff_sim_max"] = round(_cos_max(v, M), 6)

    if encode is not None and added:
        if store_texts:
            emb = encode(list(store_texts) + [added, old or ""])
            store_emb, added_emb, old_emb = emb[:-2], emb[-2], emb[-1]
            out["diff_sim_max_raw"] = round(_cos_max(added_emb, store_emb), 6)
            out["diff_cos_store_max"] = out["diff_sim_max_raw"]
            out["diff_cos_target"] = (
                round(float(np.dot(added_emb, old_emb)), 6) if old else None
            )
            sims = store_emb @ added_emb
            order = np.argsort(-sims)[:_VN_NEIGHBORS]
            neigh = store_emb[order]
            out["diff_dSvn"] = round(
                vn_entropy(np.vstack([neigh, added_emb[None, :]])) - vn_entropy(neigh),
                6,
            )
        else:
            out.update({"diff_sim_max_raw": 0.0, "diff_cos_store_max": 0.0,
                        "diff_cos_target": None, "diff_dSvn": 0.0})

    # Is the marginal content itself retrievable in the post-write store?
    if added and tier_items:
        from bfcl_eval.model_handler.middleware.retrieval_sim import (
            simulate_kv,
            simulate_vector,
        )

        probe_toks = content_tokens_ordered(added, limit=8)
        probe = " ".join(probe_toks) if probe_toks else added
        if backend == "kv":
            key = str(ref) if ref is not None else None
            corpus = list(tier_items.keys())
            if key and key not in corpus:
                corpus = corpus + [key]
            if key:
                out["diff_rank_self_post"] = float(
                    rank_self(simulate_kv(corpus, probe, k=5), key, k=5)
                )
        else:
            texts = [t for r, t in tier_items.items() if str(r) != str(ref)]
            corpus = texts + [new]
            out["diff_rank_self_post"] = float(
                rank_self(simulate_vector(corpus, probe, k=5), len(corpus) - 1, k=5)
            )
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute(
    backend: str,
    tier: str,
    ref,
    old_text: Optional[str],
    new_text: str,
    decision_tiers: Dict[str, Dict[str, str]],
    whiten: Optional[Callable] = None,
    encode: Optional[Callable] = None,
) -> Dict[str, Optional[float]]:
    """All marginal-diff features for one write candidate.

    ``decision_tiers`` is the decision-time store ({tier: {ref: text}}) as
    reconstructed by ``label_outcomes.walk_log``. Stage 0 judges redundancy
    against BOTH tiers, so the store-scoped signals do too; the retrievability
    signal stays per-tier, matching the real backends.
    """
    parts = marginal_diff(old_text, new_text)
    store_texts = [
        t
        for tname in ("core", "archival")
        for r, t in (decision_tiers.get(tname) or {}).items()
        if not (tname == tier and str(r) == str(ref))
    ]
    feats: Dict[str, Optional[float]] = {}
    feats.update(lexical_features(parts, old_text, new_text, store_texts))
    feats.update(
        semantic_features(
            parts, backend, decision_tiers.get(tier) or {}, store_texts,
            old_text, new_text, ref, whiten=whiten, encode=encode,
        )
    )
    return feats
