"""
Stage 0 Geometric Governance Filter
===================================

Write-time governance for BFCL V4 Memory categories (KV and Vector backends only;
``rec_sum`` is explicitly out of scope). Intercepts the model's memory *write* tool
calls (add / replace / update) and decides, per call:

  NOOP     -- the information is already stored: suppress the write and show the
              model a synthetic backend-format success result.
  ADD      -- genuinely new information: let the write execute unchanged.
  ESCALATE -- ambiguous: deferred to Stage 1 (NLI) / Stage 2 (Retrieval Entropy),
              which are NOT implemented in this phase. The current fallback is the
              base behavior (perform the write). See :func:`handle_escalation`.

Three signals, computed in a *whitened* embedding space (MiniLM + ABTT):

  1. ``sim_max`` -- max cosine similarity to any stored memory item.
  2. ``r``       -- residual norm of the candidate outside the subspace spanned by
                    the stored items (via QR). The part of the candidate that memory
                    cannot explain.
  3. Verbatim-value gate -- concrete values (numbers, dates, names, emails, ...)
                    extracted by regex and searched literally in the stored texts.

The whitened space is used ONLY here. The Vector backend's own raw-MiniLM retrieval
path is never touched, and the two spaces are never mixed.

Pure logic: no BFCL imports (the MiniLM encoder singleton is shared with
``semantic_entropy.py``), so it is unit-testable without a server or the harness.
All behavior is driven by ``GOV_*`` environment variables (see :class:`GovConfig`),
so a single registry entry covers every ablation arm.
"""

import ast
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

# Reuse the lazily-loaded, thread-locked MiniLM singleton (CPU). MiniLM is already a
# repo dependency via the vector memory backend.
from bfcl_eval.model_handler.middleware.semantic_entropy import _get_encoder

DEFAULT_ARTIFACT_PATH = str(
    Path(__file__).resolve().parent / "artifacts" / "abtt_minilm_l6_d16.npz"
)

# Backend capacity constants -- mirror memory_kv.py / memory_vector.py exactly.
MAX_CORE_SIZE = 7
MAX_CORE_ENTRY_LENGTH = 300
MAX_ARCHIVAL_SIZE = 50
MAX_ARCHIVAL_ENTRY_LENGTH = 2000

_KV_KEY_PATTERN = re.compile(r"^[a-z]+(_[a-z0-9]+)*$")  # memory_kv._is_valid_key_format

# Shadow ids handed out for suppressed Vector adds. Deliberately out-of-band (real ids
# are small and sequential): if the model later updates/removes a shadow id the backend
# errors loudly, which is recoverable -- unlike returning the predicted next real id,
# which the next genuine add would silently collide with.
SHADOW_ID_START = 9000


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GovConfig:
    """All knobs are environment-driven (one registry entry covers every arm).

    Stage 0 (geometric filter):
      GOV_ENABLED, GOV_DRY_RUN, GOV_D, GOV_TAU0, GOV_TAU_MIN, GOV_LAMBDA, GOV_ALPHA,
      GOV_DELTA, GOV_SIM_HIGH, GOV_ARTIFACT_PATH, GOV_LOG_DIR, GOV_LOG_FILE,
      GOV_VERBOSE, GOV_STRICT_THRESHOLDS (degenerate (sim_high, delta) pair raises
      instead of warning; see :meth:`validate`).

    Stage 1 (NLI escalation; default OFF, shadow-first):
      GOV_NLI_ENABLED  -- load the NLI scorer and run the escalation handler.
      GOV_NLI_SHADOW   -- compute + log the full Stage 1 decision but never intervene.
      GOV_NLI_K        -- number of top-sim neighbors to check (default 3).
      GOV_TAU_ENTAIL, GOV_TAU_CONTRA, GOV_DELTA_SPEC -- decision-table thresholds.

    Stage 2 (retrieval-entropy escalation; default OFF):
      GOV_S2_ENABLED   -- run the probe-based retrieval simulation on Stage 1's
                          all-neutral escalations.
      GOV_S2_MARGIN    -- min top1-vs-top2 margin (all probes agreeing on the same
                          stored item) required to call the candidate a duplicate.
                          Placeholder default; Plan 2 calibrates from the replay
                          margin distribution.
      GOV_S2_CANON_LLM -- allow one LLM canonicalization attempt for rewrites
                          (default off: deterministic canonicalization only).
      GOV_PROBE_PARAPHRASE -- enable the paraphrase probe channel (default off:
                          template probes only; see probe_gen.py).
    """

    enabled: bool = True
    dry_run: bool = False  # shadow mode: full pipeline + logging, but never suppress
    d: int = 16  # ABTT top-directions removed (must match the artifact)
    # SAGE-style adaptive threshold initials (tau applies to the residual r).
    tau0: float = 0.25
    tau_min: float = 0.025
    lam: float = 2.0
    alpha: float = 0.9  # EMA smoothing
    delta: float = 0.025  # r ~ 0 cutoff for the NOOP branch
    # NOT from SAGE -- similarity floor for NOOP in the whitened space. ABTT lowers
    # pairwise cosines substantially, so this needs calibration from a GOV_DRY_RUN=1
    # shadow run before trusting it.
    sim_high: float = 0.80
    strict_thresholds: bool = False  # degenerate (sim_high, delta) raises, not warns
    artifact_path: str = DEFAULT_ARTIFACT_PATH
    log_dir: str = ""  # "" -> current working directory
    log_file: str = "governance_log.jsonl"
    verbose: bool = False
    # -- Stage 1 (NLI) ------------------------------------------------------
    nli_enabled: bool = False
    nli_shadow: bool = True  # shadow-first deploy: log everything, change nothing
    nli_k: int = 3
    tau_entail: float = 0.75
    tau_contra: float = 0.75
    delta_spec: float = 0.10
    # -- Stage 2 (retrieval entropy) ----------------------------------------
    s2_enabled: bool = False
    s2_margin: float = 0.05  # placeholder; calibrate from replay margins (Plan 2)
    s2_canon_llm: bool = False

    @classmethod
    def from_env(cls) -> "GovConfig":
        cfg = cls(
            enabled=_env_flag("GOV_ENABLED", True),
            dry_run=_env_flag("GOV_DRY_RUN", False),
            d=int(os.getenv("GOV_D", "16")),
            tau0=float(os.getenv("GOV_TAU0", "0.25")),
            tau_min=float(os.getenv("GOV_TAU_MIN", "0.025")),
            lam=float(os.getenv("GOV_LAMBDA", "2.0")),
            alpha=float(os.getenv("GOV_ALPHA", "0.9")),
            delta=float(os.getenv("GOV_DELTA", "0.025")),
            sim_high=float(os.getenv("GOV_SIM_HIGH", "0.80")),
            strict_thresholds=_env_flag("GOV_STRICT_THRESHOLDS", False),
            artifact_path=os.getenv("GOV_ARTIFACT_PATH", DEFAULT_ARTIFACT_PATH),
            log_dir=os.getenv("GOV_LOG_DIR", ""),
            log_file=os.getenv("GOV_LOG_FILE", "governance_log.jsonl"),
            verbose=_env_flag("GOV_VERBOSE", False),
            nli_enabled=_env_flag("GOV_NLI_ENABLED", False),
            nli_shadow=_env_flag("GOV_NLI_SHADOW", True),
            nli_k=int(os.getenv("GOV_NLI_K", "3")),
            tau_entail=float(os.getenv("GOV_TAU_ENTAIL", "0.75")),
            tau_contra=float(os.getenv("GOV_TAU_CONTRA", "0.75")),
            delta_spec=float(os.getenv("GOV_DELTA_SPEC", "0.10")),
            s2_enabled=_env_flag("GOV_S2_ENABLED", False),
            s2_margin=float(os.getenv("GOV_S2_MARGIN", "0.05")),
            s2_canon_llm=_env_flag("GOV_S2_CANON_LLM", False),
        )
        cfg.validate()
        return cfg

    def noop_residual_bound(self) -> float:
        """Max residual a single-neighbor candidate can have while still clearing
        the ``sim_max > sim_high`` gate: ``sqrt(1 - sim_high**2)`` (unit vectors)."""
        return math.sqrt(max(0.0, 1.0 - self.sim_high**2))

    def validate(self) -> bool:
        """Startup guard against a geometrically degenerate (sim_high, delta) pair.

        Geometry (single stored neighbor, all vectors unit-norm): the residual of
        the candidate against that neighbor is ``r = sqrt(1 - sim**2)``, so
        ``sim_max > sim_high`` already forces ``r < bound = sqrt(1 - sim_high**2)``.

          * ``delta >= bound``: the residual gate never binds -- the NOOP branch
            fires at the advertised ``sim_high``. OK.
          * ``delta <  bound``: the residual gate binds FIRST and silently raises
            the effective single-neighbor similarity bar to ``sqrt(1 - delta**2)``
            -- a config that *looks* active but cannot fire at its advertised
            threshold (v2 SS2.3: (0.95, 0.30) is degenerate because
            0.30 < bound = 0.312; the proposed fix (0.95, 0.32) clears it).

        NOTE(inequality direction): Plan 1 Step 1 flags that the v2 doc's prose is
        internally ambiguous about the sign. The direction encoded here (warn when
        ``delta < bound``) is the only one consistent with ALL of v2's worked
        numbers -- (0.95, 0.30) degenerate, (0.95, 0.32) fine, (0.80, 0.40) warns.
        Plan 2's calibration step must re-check this against live margins.

        Returns True when the pair is healthy; warns (or raises under
        ``strict_thresholds``) and returns False otherwise.
        """
        bound = self.noop_residual_bound()
        if self.delta < bound:
            effective_sim = math.sqrt(max(0.0, 1.0 - self.delta**2))
            msg = (
                f"[GOV] WARNING: degenerate thresholds sim_high={self.sim_high} "
                f"delta={self.delta}: delta < single-neighbor bound "
                f"sqrt(1-sim_high^2)={bound:.4f}, so the residual gate binds first "
                f"and the effective one-neighbor NOOP threshold is "
                f"sim>{effective_sim:.4f}, not the advertised {self.sim_high}. "
                f"Raise GOV_DELTA to >= {bound:.4f} (or lower GOV_SIM_HIGH)."
            )
            if self.strict_thresholds:
                raise ValueError(msg)
            print(msg)
            return False
        return True


# ---------------------------------------------------------------------------
# ABTT (All-But-The-Top) whitening transform -- fitted offline by
# bfcl_eval/scripts/compute_abtt.py on BFCL's own prereq-conversation pool.
# ---------------------------------------------------------------------------


class AbttTransform:
    def __init__(self, mu: np.ndarray, u_top: np.ndarray, meta: dict):
        self.mu = mu.astype(np.float64)
        self.u_top = u_top.astype(np.float64)  # (dim, d), orthonormal columns
        self.meta = meta

    def apply(self, v: np.ndarray) -> np.ndarray:
        """Whiten a single raw embedding and L2-normalize. Returns float64 (dim,)."""
        vc = np.asarray(v, dtype=np.float64) - self.mu
        w = vc - self.u_top @ (self.u_top.T @ vc)
        norm = np.linalg.norm(w)
        if norm < 1e-12:
            return np.zeros_like(w)
        return w / norm

    def apply_batch(self, X: np.ndarray) -> np.ndarray:
        Xc = np.asarray(X, dtype=np.float64) - self.mu
        W = Xc - (Xc @ self.u_top) @ self.u_top.T
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return W / norms


_ABTT_CACHE: dict = {}
_ABTT_LOCK = threading.Lock()


def load_abtt(path: str, expected_d: Optional[int] = None) -> AbttTransform:
    """Load the committed ABTT artifact; fail loudly if missing or mismatched."""
    with _ABTT_LOCK:
        if path in _ABTT_CACHE:
            return _ABTT_CACHE[path]
        if not os.path.exists(path):
            raise RuntimeError(
                f"[GOV] ABTT artifact not found at {path}. Run "
                f"`python bfcl_eval/scripts/compute_abtt.py` once to generate it "
                f"(or set GOV_ARTIFACT_PATH)."
            )
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"])) if "meta" in data else {}
        d = int(data["d"])
        if expected_d is not None and d != expected_d:
            raise RuntimeError(
                f"[GOV] ABTT artifact at {path} was fitted with D={d}, "
                f"but GOV_D={expected_d}. Refit or adjust GOV_D."
            )
        transform = AbttTransform(mu=data["mu"], u_top=data["u_top"], meta=meta)
        _ABTT_CACHE[path] = transform
        return transform


# ---------------------------------------------------------------------------
# Metadata cache -- middleware-side mirror of the backend memory store.
# Rehydrated from the backend's on-disk snapshot at conversation start, and kept in
# sync by observing *successful* execution results. Intentionally has NO usage-count
# field.
# ---------------------------------------------------------------------------


@dataclass
class MemoryItem:
    ref: str  # KV key, or str(vec_id) for Vector
    text: str  # KV: "key words: value" composite; Vector: stored text
    emb_whitened: np.ndarray  # L2-normalized, ABTT space
    turn_written: int
    tier: str  # "core" | "archival"
    category: str = ""
    low_confidence: bool = False
    # Stage 2 probe cache: generated at write time, persisted to the
    # <scenario>_gov_state.json sidecar. Empty until GOV_S2_ENABLED runs.
    probes: list = field(default_factory=list)
    probe_provenance: str = ""  # "write_time" | "sidecar" | "stored_text"


def kv_composite_text(key: str, value: str) -> str:
    """KV items are embedded as 'key words: value' -- keys carry real semantics in
    this benchmark; underscores become spaces so MiniLM tokenizes them naturally."""
    return f"{str(key).replace('_', ' ')}: {value}"


class GovernanceCache:
    def __init__(self, backend: str):
        self.backend = backend  # "kv" | "vector"
        self.items: dict[str, dict[str, MemoryItem]] = {"core": {}, "archival": {}}
        self._version = 0
        self._matrix_cache: Optional[tuple[int, np.ndarray, list]] = None
        self._q_cache: Optional[tuple[int, np.ndarray]] = None

    # -- capacity / density ------------------------------------------------

    @staticmethod
    def capacity(tier: str) -> int:
        return MAX_CORE_SIZE if tier == "core" else MAX_ARCHIVAL_SIZE

    @staticmethod
    def max_entry_length(tier: str) -> int:
        return MAX_CORE_ENTRY_LENGTH if tier == "core" else MAX_ARCHIVAL_ENTRY_LENGTH

    def size(self, tier: str) -> int:
        return len(self.items[tier])

    def total_size(self) -> int:
        return len(self.items["core"]) + len(self.items["archival"])

    def density(self, tier: str) -> float:
        return self.size(tier) / self.capacity(tier)

    # -- mutation ----------------------------------------------------------

    def _touch(self):
        self._version += 1

    def put(self, item: MemoryItem):
        self.items[item.tier][item.ref] = item
        self._touch()

    def drop(self, tier: str, ref: str):
        if ref in self.items[tier]:
            del self.items[tier][ref]
            self._touch()

    def clear_tier(self, tier: str):
        if self.items[tier]:
            self.items[tier] = {}
            self._touch()

    # -- geometry ----------------------------------------------------------

    def all_items(self) -> list:
        return list(self.items["core"].values()) + list(self.items["archival"].values())

    def matrix(self) -> tuple[np.ndarray, list]:
        """(n, dim) whitened matrix over BOTH tiers + the corresponding items.
        Redundancy is judged against everything stored, regardless of tier."""
        if self._matrix_cache is not None and self._matrix_cache[0] == self._version:
            return self._matrix_cache[1], self._matrix_cache[2]
        items = self.all_items()
        if items:
            M = np.stack([it.emb_whitened for it in items])
        else:
            M = np.zeros((0, 0))
        self._matrix_cache = (self._version, M, items)
        return M, items

    def basis_q(self) -> Optional[np.ndarray]:
        """Orthonormal basis Q (dim, k) of the whitened memory vectors.

        Full ``np.linalg.qr`` recompute per cache version: with n <= 57 items in a
        384-d space this costs tens of microseconds, so true *incremental* QR is not
        needed yet -- but this function boundary is where a Householder-update
        implementation would drop in.
        """
        if self._q_cache is not None and self._q_cache[0] == self._version:
            return self._q_cache[1]
        M, _ = self.matrix()
        if M.shape[0] == 0:
            self._q_cache = (self._version, None)
            return None
        Q, _ = np.linalg.qr(M.T, mode="reduced")  # (dim, k)
        self._q_cache = (self._version, Q)
        return Q

    def memory_texts(self) -> list:
        return [it.text for it in self.all_items()]

    # -- rehydration / observation ------------------------------------------

    def rehydrate(self, snapshot: Optional[dict], encode_fn, abtt: AbttTransform):
        """Populate the mirror from a backend snapshot dict (format per backend's
        ``_flush_memory_to_local_file``). ``snapshot=None`` -> start empty."""
        self.items = {"core": {}, "archival": {}}
        self._touch()
        if not snapshot:
            return
        entries = []  # (tier, ref, text)
        for tier, snap_key in (("core", "core_memory"), ("archival", "archival_memory")):
            tier_data = snapshot.get(snap_key, {})
            if self.backend == "kv":
                for key, value in tier_data.items():
                    entries.append((tier, str(key), kv_composite_text(key, value)))
            else:  # vector: {"next_id": N, "store": {id: text}}
                for vec_id, text in tier_data.get("store", {}).items():
                    entries.append((tier, str(int(vec_id)), str(text)))
        if not entries:
            return
        raw = encode_fn([text for _, _, text in entries])
        whitened = abtt.apply_batch(raw)
        for (tier, ref, text), emb in zip(entries, whitened):
            self.items[tier][ref] = MemoryItem(
                ref=ref,
                text=text,
                emb_whitened=emb,
                turn_written=-1,  # written in a previous conversation
                tier=tier,
            )
        self._touch()


# ---------------------------------------------------------------------------
# Adaptive thresholds (SAGE-style), per tier, per conversation.
# ---------------------------------------------------------------------------


class ThresholdState:
    def __init__(self, cfg: GovConfig):
        self.cfg = cfg
        self._tau: dict[str, Optional[float]] = {"core": None, "archival": None}

    def tau_star(self, rho: float) -> float:
        return self.cfg.tau_min + self.cfg.tau0 * math.exp(-self.cfg.lam * rho)

    def seed(self, cache: GovernanceCache):
        for tier in ("core", "archival"):
            self._tau[tier] = self.tau_star(cache.density(tier))

    def update(self, tier: str, rho: float) -> float:
        star = self.tau_star(rho)
        prev = self._tau[tier]
        self._tau[tier] = star if prev is None else self.cfg.alpha * prev + (1 - self.cfg.alpha) * star
        return self._tau[tier]


# ---------------------------------------------------------------------------
# Verbatim-value gate (regex only; GLiNER deliberately deferred -- see TODO below).
# ---------------------------------------------------------------------------

# TODO(GLiNER): swap/augment with a lightweight NER pass if regex recall proves
# insufficient in shadow-run logs. Regex keeps this phase < 1 ms and dependency-free.
_VALUE_PATTERNS = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),  # emails
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),  # ISO dates
    re.compile(r"\b\d{1,2}[/.]\d{1,2}[/.]\d{2,4}\b"),  # 03/05/2024
    re.compile(
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
        r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+\d{1,2}"
        r"(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b",
        re.IGNORECASE,
    ),  # March 5th, 2024
    re.compile(r"[$€£]\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:%|percent|dollars?|usd|eur)\b", re.IGNORECASE),
    re.compile(r"\b\d[\d,]*(?:\.\d+)?\s?(?:mg|ml|kg|km|lbs?|hours?|hrs?|minutes?|mins?|days?|weeks?|months?|years?|am|pm)\b", re.IGNORECASE),
    re.compile(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b"),  # phone-like digit runs
    re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"),  # Capitalized name bigrams+
    re.compile(r"\b\d[\d,]*(?:\.\d+)?\b"),  # bare numbers (last: most generic)
]

_STOPWORDS = frozenset(
    "a an the is are was were be been being am i you he she it we they your my mine "
    "their his her its our do does did don doesn didn not no yes to of in on at for "
    "with and or but that this these those there have has had as by from about so "
    "very just also user likes like s t re ve ll d m".split()
)


def _normalize_for_match(text: str) -> str:
    text = text.lower()
    text = text.replace("_", " ").replace(",", "")
    return re.sub(r"\s+", " ", text).strip()


def extract_verbatim_values(text: str) -> list:
    """Concrete values in the candidate text, most-specific pattern first, deduped.
    Spans already claimed by an earlier (more specific) pattern are not re-matched."""
    claimed: list = []  # (start, end)
    values: list = []
    for pattern in _VALUE_PATTERNS:
        for m in pattern.finditer(text):
            span = (m.start(), m.end())
            if any(s < span[1] and span[0] < e for s, e in claimed):
                continue
            claimed.append(span)
            val = m.group(0).strip()
            if val and val not in values:
                values.append(val)
    return values


def verbatim_present(values: list, memory_texts: list) -> tuple:
    """Literal (normalized) substring search -- NO embeddings. Returns (hits, misses)."""
    blob = _normalize_for_match(" ||| ".join(memory_texts))
    hits, misses = [], []
    for val in values:
        if _normalize_for_match(val) in blob:
            hits.append(val)
        else:
            misses.append(val)
    return hits, misses


def _content_tokens(text: str) -> frozenset:
    return frozenset(
        t for t in re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
        if len(t) > 2 and t not in _STOPWORDS
    )


# ---------------------------------------------------------------------------
# Signals + decision
# ---------------------------------------------------------------------------


class GovDecision(str, Enum):
    NOOP = "NOOP"
    ADD = "ADD"
    ESCALATE = "ESCALATE"


@dataclass
class WriteCandidate:
    op: str  # e.g. "core_memory_add"
    kind: str  # "add" | "replace" | "update"
    tier: str  # "core" | "archival"
    backend: str  # "kv" | "vector"
    text: str  # candidate text (KV: "key words: value" composite)
    args: dict  # parsed call kwargs
    ref: Optional[str] = None  # KV key / Vector vec_id (for replace/update)
    raw_call: str = ""


@dataclass
class Stage0Signals:
    sim_max: float = 0.0
    r: float = 1.0
    tau_t: float = 0.0
    verbatim_values: list = field(default_factory=list)
    verbatim_hits: list = field(default_factory=list)
    verbatim_misses: list = field(default_factory=list)
    verbatim_mode: str = "values"  # "values" | "content_tokens"
    n_items: int = 0
    rho: float = 0.0
    latency_ms: float = 0.0
    # Full similarity vector M @ v_w (same order as cache.matrix() items). Stage 1
    # ranks escalation neighbors by argsort over this -- no re-embedding.
    sims: list = field(default_factory=list)


def compute_signals(
    v_w: np.ndarray,
    candidate: WriteCandidate,
    cache: GovernanceCache,
    thresholds: ThresholdState,
    cfg: GovConfig,
) -> Stage0Signals:
    start = time.perf_counter()
    sig = Stage0Signals(n_items=cache.total_size(), rho=cache.density(candidate.tier))
    sig.tau_t = thresholds.update(candidate.tier, sig.rho)

    M, _ = cache.matrix()
    if M.shape[0] > 0:
        sims = M @ v_w
        sig.sims = [float(s) for s in sims]
        sig.sim_max = float(np.max(sims))
        Q = cache.basis_q()
        resid = v_w - Q @ (Q.T @ v_w)
        sig.r = float(np.linalg.norm(resid))

    memory_texts = cache.memory_texts()
    sig.verbatim_values = extract_verbatim_values(candidate.text)
    if sig.verbatim_values:
        sig.verbatim_hits, sig.verbatim_misses = verbatim_present(
            sig.verbatim_values, memory_texts
        )
    else:
        # Conservative extension: with no structured values to anchor on (e.g. "likes
        # tea" vs "likes coffee"), fall back to content-token containment so that a
        # geometric near-duplicate with genuinely new content words can never NOOP.
        sig.verbatim_mode = "content_tokens"
        tokens = _content_tokens(candidate.text)
        blob_tokens = _content_tokens(" ".join(memory_texts))
        sig.verbatim_hits = sorted(tokens & blob_tokens)
        sig.verbatim_misses = sorted(tokens - blob_tokens)

    sig.latency_ms = round((time.perf_counter() - start) * 1000, 3)
    return sig


def handle_escalation(
    candidate: WriteCandidate,
    signals: Stage0Signals,
    cache: GovernanceCache,
    cfg: GovConfig,
) -> tuple:
    """Ambiguous-region handler.

    # ------------------------------------------------------------------
    # TODO(Stage 1 -- NLI; see plan Section 3): entailment check between the
    #   candidate and its top-sim_max neighbors. Stored item entails candidate
    #   -> NOOP; candidate contradicts a stored item -> route to update.
    # TODO(Stage 2 -- Retrieval Entropy; see plan Section 4): entropy of the
    #   retrieval distribution under candidate perturbations; high entropy
    #   -> genuinely novel -> ADD.
    # ------------------------------------------------------------------

    Current phase fallback: base behavior (perform the write). This means Stage 0
    alone can only suppress clearly-redundant writes and never blocks novel
    information, so any measured A/B delta is attributable purely to NOOPs.
    """
    return GovDecision.ADD, "stage0_escalate_fallback"


def decide(
    candidate: WriteCandidate,
    signals: Stage0Signals,
    cache: GovernanceCache,
    cfg: GovConfig,
    preflight_ok: bool,
) -> tuple:
    """Returns (decision, reason)."""
    if signals.n_items == 0:
        return GovDecision.ADD, "empty_memory"

    verbatim_all_present = len(signals.verbatim_misses) == 0
    has_new_verbatim = len(signals.verbatim_misses) > 0

    if (
        signals.r < cfg.delta
        and signals.sim_max > cfg.sim_high
        and verbatim_all_present
    ):
        if not preflight_ok:
            # The real call would be rejected by the backend (dup key, full store,
            # ...). A synthetic success must never mask a real backend error.
            return GovDecision.ADD, "noop_blocked_by_preflight"
        return GovDecision.NOOP, "redundant"

    if signals.r > signals.tau_t and has_new_verbatim:
        return GovDecision.ADD, "novel"

    decision, reason = handle_escalation(candidate, signals, cache, cfg)
    return decision, f"escalate:{reason}"


# ---------------------------------------------------------------------------
# Backend adapters: gated ops, call parsing, preflight, synthetic results,
# success observation.
# ---------------------------------------------------------------------------

# op -> (kind, tier, ordered param names)
KV_WRITE_OPS = {
    "core_memory_add": ("add", "core", ("key", "value")),
    "archival_memory_add": ("add", "archival", ("key", "value")),
    "core_memory_replace": ("replace", "core", ("key", "value")),
    "archival_memory_replace": ("replace", "archival", ("key", "value")),
}
VECTOR_WRITE_OPS = {
    "core_memory_add": ("add", "core", ("text",)),
    "archival_memory_add": ("add", "archival", ("text",)),
    "core_memory_update": ("update", "core", ("vec_id", "new_text")),
    "archival_memory_update": ("update", "archival", ("vec_id", "new_text")),
}

# Ops observed (never gated) to keep the cache mirror in sync.
KV_OBSERVED_OPS = {
    "core_memory_remove": ("remove", "core", ("key",)),
    "archival_memory_remove": ("remove", "archival", ("key",)),
    "core_memory_clear": ("clear", "core", ()),
    "archival_memory_clear": ("clear", "archival", ()),
}
VECTOR_OBSERVED_OPS = {
    "core_memory_remove": ("remove", "core", ("vec_id",)),
    "archival_memory_remove": ("remove", "archival", ("vec_id",)),
    "core_memory_clear": ("clear", "core", ()),
    "archival_memory_clear": ("clear", "archival", ()),
}

# Read-only decoy executed in place of a suppressed write. Exists on both backends,
# always succeeds, mutates nothing. Its real output is replaced by the synthetic
# success before it reaches the chat history.
DECOY_CALL = "core_memory_retrieve_all()"

# Exact success strings from memory_kv.py -- used both to build synthetic results
# and to recognize genuine successes when observing.
KV_ADD_SUCCESS = {"core": "Key-value pair added.", "archival": "Key added."}
KV_REPLACE_SUCCESS = "Key replaced."
KV_REMOVE_SUCCESS = "Key removed."
KV_CLEAR_SUCCESS = {"core": "Short term memory cleared.", "archival": "Long term memory cleared."}
VECTOR_CLEAR_SUCCESS = "Memory cleared."


def parse_call(call: str):
    """Parse a decoded call string like ``core_memory_add(key='age', value='35')``
    into (func_name, kwargs) using ast -- never eval. Returns None if unparseable."""
    try:
        node = ast.parse(call.strip(), mode="eval").body
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            return None
        positional = [ast.literal_eval(a) for a in node.args]
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
        return node.func.id, positional, kwargs
    except Exception:
        return None


def _bind_args(positional: list, kwargs: dict, param_names: tuple) -> Optional[dict]:
    bound = dict(kwargs)
    for i, val in enumerate(positional):
        if i >= len(param_names):
            return None
        bound.setdefault(param_names[i], val)
    if any(p not in bound for p in param_names):
        return None
    return bound


def build_candidate(backend: str, call: str) -> Optional[WriteCandidate]:
    """Return a WriteCandidate if ``call`` is a gated write op, else None."""
    parsed = parse_call(call)
    if parsed is None:
        return None
    name, positional, kwargs = parsed
    table = KV_WRITE_OPS if backend == "kv" else VECTOR_WRITE_OPS
    if name not in table:
        return None
    kind, tier, params = table[name]
    args = _bind_args(positional, kwargs, params)
    if args is None:
        return None
    if backend == "kv":
        text = kv_composite_text(args["key"], args["value"])
        ref = str(args["key"])
    else:
        text = str(args["text"] if kind == "add" else args["new_text"])
        ref = None if kind == "add" else str(args["vec_id"])
    return WriteCandidate(
        op=name, kind=kind, tier=tier, backend=backend,
        text=text, args=args, ref=ref, raw_call=call,
    )


def preflight_would_succeed(candidate: WriteCandidate, cache: GovernanceCache) -> bool:
    """Predict (from the mirror) whether the backend would ACCEPT this write.
    NOOP is only ever allowed when the answer is yes."""
    tier_items = cache.items[candidate.tier]
    max_len = cache.max_entry_length(candidate.tier)
    if candidate.backend == "kv":
        key, value = str(candidate.args["key"]), str(candidate.args["value"])
        if len(value) > max_len:
            return False
        if candidate.kind == "add":
            return (
                _KV_KEY_PATTERN.match(key) is not None
                and key not in tier_items
                and len(tier_items) < cache.capacity(candidate.tier)
            )
        return key in tier_items  # replace
    # vector
    if candidate.kind == "add":
        return (
            len(str(candidate.args["text"])) <= max_len
            and len(tier_items) < cache.capacity(candidate.tier)
        )
    return (  # update
        str(candidate.args["vec_id"]) in tier_items
        and len(str(candidate.args["new_text"])) <= max_len
    )


def synthetic_success(candidate: WriteCandidate, shadow_id: Optional[int]) -> str:
    """Backend-format success result (json.dumps compact, matching
    multi_turn_utils' serialization of dict results)."""
    if candidate.backend == "kv":
        if candidate.kind == "add":
            return json.dumps({"status": KV_ADD_SUCCESS[candidate.tier]})
        return json.dumps({"status": KV_REPLACE_SUCCESS})
    if candidate.kind == "add":
        return json.dumps({"id": shadow_id})
    return json.dumps({"status": f"ID {candidate.args['vec_id']} updated."})


# ---------------------------------------------------------------------------
# JSONL audit log
# ---------------------------------------------------------------------------

_LOG_LOCK = threading.Lock()


def log_gov_record(record: dict, cfg: GovConfig) -> None:
    path = os.path.join(cfg.log_dir or ".", cfg.log_file)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with _LOG_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as e:  # logging must never break inference
        print(f"[GOV] failed to append audit record: {e}")


# ---------------------------------------------------------------------------
# Session -- one per memory test entry. Orchestrates HOOK 1 (govern_calls, at the
# handler's decode_execute boundary) and HOOK 2 (patch_results, at the handler's
# _add_execution_results_prompting boundary).
# ---------------------------------------------------------------------------


class GovernanceSession:
    def __init__(
        self,
        cfg: GovConfig,
        backend: str,
        test_id: str,
        snapshot: Optional[dict],
        snapshot_path: str = "",
        snapshot_missing: bool = False,
    ):
        self.cfg = cfg
        self.backend = backend
        self.test_id = test_id
        self.abtt = load_abtt(cfg.artifact_path, expected_d=cfg.d)
        self.encoder = _get_encoder()
        if self.encoder is None:
            raise RuntimeError(
                "[GOV] sentence-transformers encoder unavailable; the governance "
                "filter cannot run without embeddings."
            )
        self.cache = GovernanceCache(backend)
        self.cache.rehydrate(snapshot, self._encode_batch, self.abtt)
        self.thresholds = ThresholdState(cfg)
        self.thresholds.seed(self.cache)
        self.step = 0
        self._shadow_next = SHADOW_ID_START
        self._issued_shadow_ids: set = set()
        # idx -> {"original_call", "synthetic", "candidate"} for the current step
        self._pending: dict = {}

        log_gov_record(
            {
                "ts": time.time(),
                "event": "rehydrate",
                "test_id": test_id,
                "backend": backend,
                "snapshot_path": snapshot_path,
                "snapshot_missing": snapshot_missing,
                "items_loaded": self.cache.total_size(),
                "dry_run": cfg.dry_run,
                # ABTT provenance (Plan 1 Step 2c): lets a replay assert it
                # reconstructed the same whitening this session used.
                "abtt_d": int(self.abtt.u_top.shape[1]),
                "abtt_artifact_path": cfg.artifact_path,
                "abtt_meta": self.abtt.meta,
            },
            cfg,
        )

    # -- embedding (raw space; whitening happens explicitly via abtt) --------

    def _encode_batch(self, texts: list) -> np.ndarray:
        return np.asarray(self.encoder.encode(texts, normalize_embeddings=False))

    def _whiten_one(self, text: str) -> np.ndarray:
        return self.abtt.apply(self._encode_batch([text])[0])

    # -- HOOK 1: pre-execution ------------------------------------------------

    def govern_calls(self, calls: list) -> list:
        """Inspect decoded call strings; rewrite NOOP'd writes to a read-only decoy.
        Never removes a call and never changes list length (an empty decoded list
        would short-circuit the harness step loop before results are injected)."""
        self.step += 1
        self._pending = {}
        governed = list(calls)
        for idx, call in enumerate(calls):
            candidate = build_candidate(self.backend, call)
            if candidate is None:
                continue
            self._govern_one(idx, candidate, governed)
        return governed

    def _govern_one(self, idx: int, candidate: WriteCandidate, governed: list):
        v_w = self._whiten_one(candidate.text)
        preflight_ok = preflight_would_succeed(candidate, self.cache)
        signals = compute_signals(v_w, candidate, self.cache, self.thresholds, self.cfg)
        decision, reason = decide(candidate, signals, self.cache, self.cfg, preflight_ok)

        shadow_id = None
        synthetic = None
        if decision == GovDecision.NOOP and not self.cfg.dry_run:
            if candidate.backend == "vector" and candidate.kind == "add":
                shadow_id = self._shadow_next
                self._shadow_next += 1
                self._issued_shadow_ids.add(shadow_id)
            synthetic = synthetic_success(candidate, shadow_id)
            governed[idx] = DECOY_CALL
            self._pending[idx] = {
                "original_call": candidate.raw_call,
                "synthetic": synthetic,
            }

        if self.cfg.verbose:
            print(
                f"[GOV] {self.test_id} step={self.step} {candidate.op} "
                f"sim_max={signals.sim_max:.3f} r={signals.r:.3f} tau={signals.tau_t:.3f} "
                f"-> {decision.value} ({reason})"
            )

        log_gov_record(
            {
                "ts": time.time(),
                "event": "decision",
                "test_id": self.test_id,
                "step_idx": self.step,
                "call_idx": idx,
                "backend": candidate.backend,
                "op": candidate.op,
                "tier": candidate.tier,
                "candidate_ref": candidate.ref,
                # Full text, never truncated: the offline replay reconstructs memory
                # state from this field verbatim (Plan 1 Step 2b).
                "candidate_text": candidate.text,
                "sim_max": round(signals.sim_max, 4),
                "r": round(signals.r, 4),
                "tau_t": round(signals.tau_t, 4),
                "delta": self.cfg.delta,
                "sim_high": self.cfg.sim_high,
                "verbatim_mode": signals.verbatim_mode,
                "verbatim_values": signals.verbatim_values,
                "verbatim_hits": signals.verbatim_hits,
                "verbatim_misses": signals.verbatim_misses,
                "n_items": signals.n_items,
                "rho": round(signals.rho, 4),
                "decision": decision.value,
                "reason": reason,
                "preflight_ok": preflight_ok,
                "dry_run": self.cfg.dry_run,
                "synthetic_result": synthetic,
                "shadow_id": shadow_id,
                "latency_ms": signals.latency_ms,
            },
            self.cfg,
        )

    # -- HOOK 2: post-execution -----------------------------------------------

    def patch_results(self, execution_results: list, decoded_calls: list) -> list:
        """Substitute synthetic successes at suppressed indices, restore the original
        call strings in ``decoded_calls`` (in place -- it is the same list the parent
        zips against for the tool-message ``name``), and observe genuine successes to
        keep the cache mirror in sync. Returns the patched results list."""
        new_results = list(execution_results)
        for idx, info in self._pending.items():
            if idx < len(new_results):
                new_results[idx] = info["synthetic"]
            if idx < len(decoded_calls):
                decoded_calls[idx] = info["original_call"]

        for idx, call in enumerate(decoded_calls):
            if idx in self._pending:
                continue  # suppressed: backend state unchanged
            if idx < len(new_results):
                self._observe(call, execution_results[idx])

        self._pending = {}
        return new_results

    def _observe(self, call: str, raw_result: str):
        """Update the mirror from a genuinely-executed op, ONLY on an exact success
        signature. Anything else (error dicts, 'Error during execution: ...' strings)
        is treated as not-a-success, so the mirror can never over-count."""
        parsed = parse_call(call)
        if parsed is None:
            return
        name, positional, kwargs = parsed
        write_table = KV_WRITE_OPS if self.backend == "kv" else VECTOR_WRITE_OPS
        observe_table = KV_OBSERVED_OPS if self.backend == "kv" else VECTOR_OBSERVED_OPS
        table = write_table if name in write_table else observe_table if name in observe_table else None
        if table is None:
            return
        kind, tier, params = table[name]
        args = _bind_args(positional, kwargs, params)
        if args is None:
            return
        try:
            result = json.loads(raw_result)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(result, dict) or "error" in result:
            return

        if self.backend == "kv":
            self._observe_kv(kind, tier, args, result)
        else:
            self._observe_vector(kind, tier, args, result)

    def _observe_kv(self, kind: str, tier: str, args: dict, result: dict):
        status = result.get("status")
        if kind == "add" and status == KV_ADD_SUCCESS[tier]:
            self._put_item(tier, str(args["key"]), kv_composite_text(args["key"], args["value"]))
        elif kind == "replace" and status == KV_REPLACE_SUCCESS:
            self._put_item(tier, str(args["key"]), kv_composite_text(args["key"], args["value"]))
        elif kind == "remove" and status == KV_REMOVE_SUCCESS:
            self.cache.drop(tier, str(args["key"]))
            self._log_observe_removal("observe_remove", tier, ref=str(args["key"]))
        elif kind == "clear" and status == KV_CLEAR_SUCCESS[tier]:
            self.cache.clear_tier(tier)
            self._log_observe_removal("observe_clear", tier)

    def _observe_vector(self, kind: str, tier: str, args: dict, result: dict):
        status = result.get("status", "")
        if kind == "add" and "id" in result:
            self._put_item(tier, str(int(result["id"])), str(args["text"]))
        elif kind == "update" and status == f"ID {args['vec_id']} updated.":
            self._put_item(tier, str(args["vec_id"]), str(args["new_text"]))
        elif kind == "remove" and status == f"ID {args['vec_id']} removed from store.":
            self.cache.drop(tier, str(args["vec_id"]))
            self._log_observe_removal("observe_remove", tier, ref=str(args["vec_id"]))
        elif kind == "clear" and status == VECTOR_CLEAR_SUCCESS:
            self.cache.clear_tier(tier)
            self._log_observe_removal("observe_clear", tier)

    def _log_observe_removal(self, event: str, tier: str, ref: Optional[str] = None):
        """Destructive mutations must appear in the live log stream (Plan 1 Step 2a);
        without them, an offline replay can only infer removals from result files."""
        record = {
            "ts": time.time(),
            "event": event,
            "test_id": self.test_id,
            "step_idx": self.step,
            "backend": self.backend,
            "tier": tier,
            "n_items": self.cache.total_size(),
        }
        if ref is not None:
            record["ref"] = ref
        log_gov_record(record, self.cfg)

    def _put_item(self, tier: str, ref: str, text: str):
        self.cache.put(
            MemoryItem(
                ref=ref,
                text=text,
                emb_whitened=self._whiten_one(text),
                turn_written=self.step,
                tier=tier,
            )
        )
        log_gov_record(
            {
                "ts": time.time(),
                "event": "observe",
                "test_id": self.test_id,
                "step_idx": self.step,
                "backend": self.backend,
                "tier": tier,
                "ref": ref,
                "text": text,  # full text -- replay needs the verbatim value
                "n_items": self.cache.total_size(),
            },
            self.cfg,
        )
