"""
Semantic-Entropy Gate
=====================

Sampling-based uncertainty gate for BFCL V4 Memory categories (thesis "Idea 1").

Given N sampled completions for one generation step, this module:

  1. converts each completion into a :class:`Candidate` (decoded tool calls or free text),
  2. clusters the candidates by semantic equivalence (union-find over pairwise similarity),
  3. computes the cluster entropy ``H = -sum(p_c * log2(p_c))`` and majority fraction ``m``,
  4. commits the majority cluster's medoid -- unless the majority action is a *destructive*
     memory operation under high uncertainty, in which case it falls back to the largest
     non-destructive cluster ("safe default"),
  5. returns a full audit record for the JSONL log (write-time-uncertainty analysis).

Pure logic: no BFCL imports, so it is unit-testable without a server or the harness.
The gate only ever *selects among* sampled candidates -- it never synthesizes text, which
would break the handler's decode / chat-template assumptions.

All behavior is driven by ``SE_*`` environment variables (see :class:`SEConfig`), so a
single registry entry covers every ablation arm.
"""

import difflib
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field

# Exact tool names from bfcl_eval/eval_checker/multi_turn_eval/func_source_code/
# memory_{kv,vector,rec_sum}.py. Destructive = data is gone afterwards; overwrite =
# data is replaced (recoverable only if the new value preserves it).
DESTRUCTIVE_OPS = {
    "core_memory_remove",
    "core_memory_clear",
    "archival_memory_remove",
    "archival_memory_clear",
    "memory_clear",  # rec_sum
}
OVERWRITE_OPS = {
    "core_memory_replace",
    "archival_memory_replace",  # kv
    "core_memory_update",
    "archival_memory_update",  # vector
    "memory_update",
    "memory_replace",  # rec_sum (memory_update overwrites the whole blob!)
}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


@dataclass
class SEConfig:
    num_samples: int = 5  # 1 disables the gate entirely
    temperature: float = 0.7  # sampling temperature for the N-sample call
    sim_threshold: float = 0.80  # tool-call candidates: same cluster if cosine >= this
    text_sim_threshold: float = 0.85  # unused for merging (kept for ablations); text
    # clustering is content-token based -- embedding cosine both under-merges short
    # paraphrases (0.445 for "the user is 35 years old." / "you are 35 years old.")
    # and over-merges template twins with different answers ("...coffee." / "...tea.").
    destructive_max_entropy: float = 0.7  # bits
    destructive_min_majority: float = 0.6
    overwrite_max_entropy: float = 1.2  # looser tier for replace/update
    text_jaccard_threshold: float = 0.6  # content-token overlap that also merges text pairs
    gate_destructive: bool = True  # False -> pure majority vote (ablation E2)
    gate_recall: bool = True  # False -> gate prereq entries only (ablation E3)
    log_dir: str = ""  # "" -> current working directory
    log_file: str = "semantic_entropy_log.jsonl"
    verbose: bool = False

    @classmethod
    def from_env(cls) -> "SEConfig":
        return cls(
            num_samples=int(os.getenv("SE_NUM_SAMPLES", "5")),
            temperature=float(os.getenv("SE_TEMPERATURE", "0.7")),
            sim_threshold=float(os.getenv("SE_SIM_THRESHOLD", "0.80")),
            text_sim_threshold=float(os.getenv("SE_TEXT_SIM_THRESHOLD", "0.85")),
            destructive_max_entropy=float(os.getenv("SE_DESTRUCTIVE_MAX_ENTROPY", "0.7")),
            destructive_min_majority=float(os.getenv("SE_DESTRUCTIVE_MIN_MAJORITY", "0.6")),
            overwrite_max_entropy=float(os.getenv("SE_OVERWRITE_MAX_ENTROPY", "1.2")),
            text_jaccard_threshold=float(os.getenv("SE_TEXT_JACCARD_THRESHOLD", "0.6")),
            gate_destructive=_env_flag("SE_GATE_DESTRUCTIVE", True),
            gate_recall=_env_flag("SE_GATE_RECALL", True),
            log_dir=os.getenv("SE_LOG_DIR", ""),
            log_file=os.getenv("SE_LOG_FILE", "semantic_entropy_log.jsonl"),
            verbose=_env_flag("SE_VERBOSE", False),
        )


@dataclass
class Candidate:
    text: str  # raw completion text
    kind: str  # "tool_call" | "text"
    calls: list = None  # e.g. ["core_memory_add(key='age', value='35')"]
    func_names: tuple = ()  # sorted tool names, () for text
    cluster_text: str = ""  # normalized string used for similarity


_THINK_SPLIT = "</think>"


def _strip_think(text: str) -> str:
    if _THINK_SPLIT in text:
        return text.split(_THINK_SPLIT)[-1].lstrip("\n")
    return text


def _normalize_call_string(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[\"']", "", s)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def build_candidate(raw_text: str, calls) -> Candidate:
    """``calls`` is the handler's ``decode_execute`` output (list of call strings), or
    None/[] when decoding failed or produced nothing -- those candidates are free text."""
    if calls:
        func_names = tuple(sorted(c.split("(", 1)[0].strip() for c in calls))
        cluster_text = _normalize_call_string("; ".join(calls))
        return Candidate(
            text=raw_text,
            kind="tool_call",
            calls=list(calls),
            func_names=func_names,
            cluster_text=cluster_text,
        )
    cleaned = _strip_think(raw_text).strip()
    return Candidate(
        text=raw_text,
        kind="text",
        calls=None,
        func_names=(),
        cluster_text=re.sub(r"\s+", " ", cleaned.lower()),
    )


# ---------------------------------------------------------------------------
# Similarity: lazy MiniLM singleton (already a repo dependency via the vector
# memory backend), with a difflib fallback if sentence-transformers is broken.
# ---------------------------------------------------------------------------

_ENCODER = None
_ENCODER_FAILED = False
_ENCODER_LOCK = threading.Lock()


def _get_encoder():
    global _ENCODER, _ENCODER_FAILED
    if _ENCODER is not None or _ENCODER_FAILED:
        return _ENCODER
    with _ENCODER_LOCK:
        if _ENCODER is None and not _ENCODER_FAILED:
            try:
                from sentence_transformers import SentenceTransformer

                _ENCODER = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            except Exception as e:  # pragma: no cover - environment-dependent
                print(f"[SE] sentence-transformers unavailable ({e}); falling back to difflib")
                _ENCODER_FAILED = True
    return _ENCODER


# ---------------------------------------------------------------------------
# NLI scorer: lazy DeBERTa-MNLI singleton (CPU), mirroring _get_encoder().
# Used by the Stage 1 governance escalation handler (governance_filter.py).
# ---------------------------------------------------------------------------

_NLI = None
_NLI_FAILED = False
_NLI_LOCK = threading.Lock()

DEFAULT_NLI_MODEL = "microsoft/deberta-large-mnli"


class NliScorer:
    """Thin wrapper: probs(premise, hypothesis) -> (p_contra, p_neutral, p_entail).

    Label order is resolved from the model config's id2label, never assumed.
    """

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
        self._idx = {name: i for i, name in id2label.items()}
        for needed in ("contradiction", "neutral", "entailment"):
            if needed not in self._idx:
                raise RuntimeError(
                    f"[NLI] model labels {id2label} missing '{needed}'; "
                    f"not an MNLI-style classifier."
                )

    def probs(self, premise: str, hypothesis: str) -> tuple:
        return self.probs_batch([(premise, hypothesis)])[0]

    def probs_batch(self, pairs: list) -> list:
        """Batch all pairs of one escalation in a single forward pass (CPU
        efficiency: k=3 bidirectional -> one batch of 6)."""
        import torch

        premises = [p for p, _ in pairs]
        hypotheses = [h for _, h in pairs]
        enc = self._tokenizer(
            premises, hypotheses, return_tensors="pt",
            padding=True, truncation=True, max_length=512,
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        c, n, e = (self._idx["contradiction"], self._idx["neutral"], self._idx["entailment"])
        return [(float(row[c]), float(row[n]), float(row[e])) for row in probs]


def _get_nli_model(model_name: str = None, device: str = None):
    """Lazy, thread-locked NLI singleton. Returns None if transformers/weights
    are unavailable (callers must degrade gracefully -- e.g. Stage 1 falls back
    to the Stage 0 ADD behavior and logs the outage)."""
    global _NLI, _NLI_FAILED
    if _NLI is not None or _NLI_FAILED:
        return _NLI
    with _NLI_LOCK:
        if _NLI is None and not _NLI_FAILED:
            try:
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )

                name = model_name or os.getenv("GOV_NLI_MODEL", DEFAULT_NLI_MODEL)
                dev = device or os.getenv("GOV_NLI_DEVICE", "cpu")
                tokenizer = AutoTokenizer.from_pretrained(name)
                model = AutoModelForSequenceClassification.from_pretrained(name)
                model.to(dev).eval()
                _NLI = NliScorer(model, tokenizer, dev)
            except Exception as e:  # pragma: no cover - environment-dependent
                print(f"[NLI] model unavailable ({e}); Stage 1 will fall back to ADD")
                _NLI_FAILED = True
    return _NLI


def _similarity_matrix(texts: list) -> list:
    """NxN similarity in [0, 1]-ish (cosine for embeddings, ratio for difflib)."""
    n = len(texts)
    encoder = _get_encoder()
    if encoder is not None:
        embs = encoder.encode(texts, normalize_embeddings=True)
        return [[float(sum(a * b for a, b in zip(embs[i], embs[j]))) for j in range(n)] for i in range(n)]
    return [
        [difflib.SequenceMatcher(None, texts[i], texts[j]).ratio() for j in range(n)]
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Clustering + entropy
# ---------------------------------------------------------------------------

# Function words stripped before comparing answer *content*. The BFCL agentic grader
# does substring matching on standardized answers, so two answers are "the same" when
# they carry the same content tokens, however they are phrased. Raw sentence-embedding
# cosine under-merges short paraphrases (measured: "the user is 35 years old." vs
# "you are 35 years old." = 0.445 with MiniLM), hence this second criterion.
_STOPWORDS = frozenset(
    "a an the is are was were be been being am i you he she it we they your my mine "
    "their his her its our do does did don doesn didn not no yes to of in on at for "
    "with and or but that this these those there have has had as by from about so "
    "very just also s t re ve ll d m".split()
)


def _content_tokens(text: str) -> frozenset:
    return frozenset(
        t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS
    )


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 1.0 if a == b else 0.0
    return len(a & b) / len(a | b)


def _text_equivalent(a: "Candidate", b: "Candidate", cfg: SEConfig) -> bool:
    """Two free-text answers count as the same when their content tokens overlap
    strongly, or one's content is contained in the other's -- mirroring the grader's
    substring criterion ("strawberry matcha" == "strawberry matcha latte")."""
    ta, tb = _content_tokens(a.cluster_text), _content_tokens(b.cluster_text)
    if _jaccard(ta, tb) >= cfg.text_jaccard_threshold:
        return True
    return bool(ta and tb) and (ta <= tb or tb <= ta)


def _union_find_clusters(candidates: list, sim: list, cfg: SEConfig) -> list:
    n = len(candidates)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = candidates[i], candidates[j]
            if a.kind != b.kind:
                continue  # tool calls never cluster with free text
            if a.kind == "tool_call":
                if a.func_names != b.func_names:
                    continue  # a different tool name never merges
                if sim[i][j] >= cfg.sim_threshold:
                    union(i, j)
            else:
                if _text_equivalent(a, b, cfg):
                    union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    # Deterministic order: largest first, ties by earliest member index.
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def _entropy(cluster_sizes: list, n: int) -> float:
    return -sum((s / n) * math.log2(s / n) for s in cluster_sizes if s > 0)


def _medoid(cluster: list, sim: list) -> int:
    if len(cluster) == 1:
        return cluster[0]
    best, best_score = cluster[0], -1.0
    for i in cluster:
        score = sum(sim[i][j] for j in cluster if j != i) / (len(cluster) - 1)
        if score > best_score:
            best, best_score = i, score
    return best


def _op_class(candidate: Candidate) -> str:
    if any(f in DESTRUCTIVE_OPS for f in candidate.func_names):
        return "destructive"
    if any(f in OVERWRITE_OPS for f in candidate.func_names):
        return "overwrite"
    return "safe" if candidate.kind == "tool_call" else "text"


# ---------------------------------------------------------------------------
# Decision procedure
# ---------------------------------------------------------------------------


def choose(candidates: list, cfg: SEConfig) -> tuple:
    """Select the index of the candidate to commit; return (index, audit_record).

    Never synthesizes a response -- only picks among the sampled candidates.
    """
    start = time.time()
    n = len(candidates)
    if n == 1:
        return 0, {
            "n": 1,
            "entropy": 0.0,
            "majority_fraction": 1.0,
            "num_clusters": 1,
            "decision": "single_candidate",
            "chosen_index": 0,
            "op_class": _op_class(candidates[0]),
            "wall_time_ms": 0.0,
        }

    sim = _similarity_matrix([c.cluster_text for c in candidates])
    clusters = _union_find_clusters(candidates, sim, cfg)
    sizes = [len(c) for c in clusters]
    entropy = _entropy(sizes, n)
    majority_fraction = sizes[0] / n

    majority = clusters[0]
    rep_idx = _medoid(majority, sim)
    rep_class = _op_class(candidates[rep_idx])

    decision = "commit_majority"
    fallback = False
    forced_destructive = False
    chosen = rep_idx

    if cfg.gate_destructive and rep_class in ("destructive", "overwrite"):
        if rep_class == "destructive":
            allowed = (
                entropy <= cfg.destructive_max_entropy
                and majority_fraction >= cfg.destructive_min_majority
            )
        else:  # overwrite: looser tier, entropy-only
            allowed = entropy <= cfg.overwrite_max_entropy
        if allowed:
            decision = f"commit_{rep_class}"
        else:
            # Fall back to the largest cluster whose medoid is not destructive/overwrite
            # (an add / retrieve / plain answer -- the "safe default").
            fallback_cluster = None
            for cl in clusters[1:]:
                m = _medoid(cl, sim)
                if _op_class(candidates[m]) not in ("destructive", "overwrite"):
                    fallback_cluster = cl
                    break
            if fallback_cluster is not None:
                chosen = _medoid(fallback_cluster, sim)
                decision = f"fallback_from_{rep_class}"
                fallback = True
            else:
                # Every cluster wants a destructive/overwrite op: execute the majority
                # anyway (a synthetic refusal would corrupt the conversation) but flag it.
                decision = f"forced_{rep_class}"
                forced_destructive = True

    cluster_of = {}
    for ci, cl in enumerate(clusters):
        for i in cl:
            cluster_of[i] = ci

    record = {
        "n": n,
        "entropy": round(entropy, 4),
        "majority_fraction": round(majority_fraction, 4),
        "num_clusters": len(clusters),
        "cluster_sizes": sizes,
        "op_class": rep_class,
        "decision": decision,
        "fallback": fallback,
        "forced_destructive": forced_destructive,
        "chosen_index": chosen,
        "chosen_kind": candidates[chosen].kind,
        "chosen_calls": candidates[chosen].calls,
        "candidates": [
            {
                "kind": c.kind,
                "func_names": list(c.func_names),
                "cluster": cluster_of[i],
                "text": c.text[:300],
            }
            for i, c in enumerate(candidates)
        ],
        "wall_time_ms": round((time.time() - start) * 1000, 1),
    }
    return chosen, record


# ---------------------------------------------------------------------------
# JSONL audit log
# ---------------------------------------------------------------------------

_LOG_LOCK = threading.Lock()


def log_record(record: dict, cfg: SEConfig) -> None:
    path = os.path.join(cfg.log_dir or ".", cfg.log_file)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with _LOG_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as e:  # logging must never break inference
        print(f"[SE] failed to append audit record: {e}")
