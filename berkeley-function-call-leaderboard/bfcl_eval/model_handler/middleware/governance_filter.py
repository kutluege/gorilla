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
      GOV_NLI_TAU_ENTAIL / GOV_TAU_ENTAIL, GOV_NLI_TAU_CONTRA / GOV_TAU_CONTRA,
      GOV_NLI_DELTA_SPEC / GOV_DELTA_SPEC -- decision-table thresholds (both
                          spellings accepted; the NLI-prefixed form wins).
      GOV_NLI_MODEL, GOV_NLI_DEVICE -- scorer weights/device (read by
                          semantic_entropy._get_nli_model; defaults
                          microsoft/deberta-large-mnli on cpu).

    Stage 2 (retrieval-entropy escalation; default OFF):
      GOV_S2_ENABLED   -- run the probe-based verification loop on Stage 1's
                          all-neutral escalations (+ write-time probe cache and
                          the <scenario>_gov_state.json sidecar).
      GOV_S2_SHADOW    -- compute + log the full Stage 2 result but never intervene.
      GOV_S2_MARGIN    -- min top1-vs-top2 margin (with every probe retrieving the
                          provisional candidate as top-1) required to accept
                          without canonicalization. Placeholder default; Plan 2
                          calibrates from the replay margin distribution.
      GOV_S2_CANON_LLM -- allow one LLM canonicalization attempt when the
                          deterministic one fails (default off; callback wired
                          by the handler in Plan 2).
      GOV_S2_T_KV, GOV_S2_T_VEC -- per-backend softmax temperature for the
                          *logged* entropy H only; never a decision input.
      GOV_S2_PROBES_N  -- cap on decision probes (read by probe_gen.ProbeConfig).
      GOV_PROBE_PARAPHRASE -- enable the paraphrase probe channel (default off:
                          template probes only; Plan-1 stub, see probe_gen.py).
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
    s2_shadow: bool = True  # shadow-first deploy, mirroring nli_shadow
    s2_margin: float = 0.05  # placeholder; calibrate from replay margins (Plan 2)
    s2_canon_llm: bool = False
    s2_t_kv: float = 1.0  # softmax T for logged H only -- NEVER a decision input
    s2_t_vec: float = 1.0

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
            # Both spellings accepted: Plan 1 uses GOV_TAU_*, the STAGE0 doc
            # SS7.5 uses GOV_NLI_TAU_* -- the NLI-prefixed form wins if both set.
            tau_entail=float(
                os.getenv("GOV_NLI_TAU_ENTAIL", os.getenv("GOV_TAU_ENTAIL", "0.75"))
            ),
            tau_contra=float(
                os.getenv("GOV_NLI_TAU_CONTRA", os.getenv("GOV_TAU_CONTRA", "0.75"))
            ),
            delta_spec=float(
                os.getenv("GOV_NLI_DELTA_SPEC", os.getenv("GOV_DELTA_SPEC", "0.10"))
            ),
            s2_enabled=_env_flag("GOV_S2_ENABLED", False),
            s2_shadow=_env_flag("GOV_S2_SHADOW", True),
            s2_margin=float(os.getenv("GOV_S2_MARGIN", "0.05")),
            s2_canon_llm=_env_flag("GOV_S2_CANON_LLM", False),
            s2_t_kv=float(os.getenv("GOV_S2_T_KV", "1.0")),
            s2_t_vec=float(os.getenv("GOV_S2_T_VEC", "1.0")),
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
    REWRITE = "REWRITE"  # Stage 1/2 outcome: call rewritten to a different backend op


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


def decide(
    candidate: WriteCandidate,
    signals: Stage0Signals,
    cache: GovernanceCache,
    cfg: GovConfig,
    preflight_ok: bool,
) -> tuple:
    """Pure Stage 0 signal -> decision logic. Returns (decision, reason).

    ESCALATE is now a real outcome (STAGE0 doc SS7.1): the *session* orchestrates
    Stage 1 (NLI) / Stage 2 (retrieval entropy) resolution, because escalation
    needs the whitened embedding, the NLI scorer, and the rewrite machinery --
    none of which belong in this dependency-free function. When both stages are
    disabled the session's fallback is ADD with reason
    ``escalate:stage0_escalate_fallback``, preserving the pre-Stage-1 behavior
    bit for bit.
    """
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

    return GovDecision.ESCALATE, "ambiguous"


# ---------------------------------------------------------------------------
# Stage 1 (NLI) escalation resolver -- STAGE0 doc SS7, Plan 1 Step 5.
# Pure logic with an injected scorer; the DeBERTa singleton lives in
# semantic_entropy._get_nli_model() and is only touched when GOV_NLI_ENABLED.
# ---------------------------------------------------------------------------


@dataclass
class Stage1Result:
    outcome: GovDecision  # NOOP | ADD | REWRITE | ESCALATE (to Stage 2)
    reason: str
    precheck: Optional[str] = None
    candidates: list = field(default_factory=list)  # [{ref, tier, sim}]
    pairs: list = field(default_factory=list)  # [{ref, fwd:[c,n,e], bwd:[c,n,e], verdict}]
    target_ref: Optional[str] = None
    target_tier: Optional[str] = None
    rewritten_call: Optional[str] = None
    superseded_text: Optional[str] = None  # full text -- supersede is in-place
    # replace until SS5 placement machinery exists (risk R3); this log line is
    # the only thing preventing data loss.
    latency_ms: float = 0.0
    # The evaluated MemoryItems in desc-sim order -- Stage 2's neighbor list
    # (SS7.7: "the neighbor list Stage 2 re-scores is exactly the candidate list
    # Stage 1 evaluated"). Not serialized into the log.
    neighbor_items: list = field(default_factory=list)

    def to_log(self) -> dict:
        d = {
            "outcome": self.outcome.value,
            "reason": self.reason,
            "precheck": self.precheck,
            "candidates": self.candidates,
            "pairs": self.pairs,
            "target_ref": self.target_ref,
            "target_tier": self.target_tier,
            "rewritten_call": self.rewritten_call,
            "superseded_text": self.superseded_text,
            "latency_ms": self.latency_ms,
        }
        return d


def select_stage1_candidates(
    candidate: WriteCandidate,
    signals: Stage0Signals,
    cache: GovernanceCache,
    k: int,
) -> list:
    """Top-k neighbors by the ALREADY-computed whitened sims (argsort, no
    re-embedding) + the canonical-key/target channel (STAGE0 doc SS7.3):
    KV -- force-include the item stored under candidate.ref in either tier;
    Vector -- on update ops, force-include the update's target item."""
    _, items = cache.matrix()
    order = np.argsort(-np.asarray(signals.sims)) if signals.sims else []
    selected = []
    seen = set()
    for idx in list(order)[:k]:
        it = items[int(idx)]
        selected.append((it, float(signals.sims[int(idx)])))
        seen.add((it.tier, it.ref))
    forced = []
    if candidate.backend == "kv" and candidate.ref is not None:
        for tier in ("core", "archival"):
            it = cache.items[tier].get(str(candidate.ref))
            if it is not None and (tier, it.ref) not in seen:
                forced.append(it)
    elif candidate.backend == "vector" and candidate.kind == "update":
        it = cache.items[candidate.tier].get(str(candidate.ref))
        if it is not None and (candidate.tier, it.ref) not in seen:
            forced.append(it)
    for it in forced:
        sim = None
        if signals.sims:
            for j, other in enumerate(items):
                if other is it:
                    sim = float(signals.sims[j])
                    break
        selected.insert(0, (it, sim if sim is not None else 0.0))
    return selected


def build_rewrite_call(candidate: WriteCandidate, target: MemoryItem) -> str:
    """Rewrite the model's write into a supersede of ``target`` (SS7.5)."""
    if candidate.backend == "kv":
        value = str(candidate.args["value"])
        return f"{target.tier}_memory_replace(key={target.ref!r}, value={value!r})"
    return (
        f"{target.tier}_memory_update(vec_id={int(target.ref)}, "
        f"new_text={candidate.text!r})"
    )


def stage1_resolve(
    candidate: WriteCandidate,
    signals: Stage0Signals,
    cache: GovernanceCache,
    cfg: GovConfig,
    preflight_ok: bool,
    nli_scorer,
) -> Stage1Result:
    """SS7.5 decision table over the top-k neighbors, first decisive verdict wins.

    The KV canonical-key pre-check runs FIRST and resolves at dict-lookup cost
    with no NLI call -- per SS5.4 / SS8.4 it, not geometry, carries most of KV's
    redundancy load.
    """
    start = time.perf_counter()
    res = Stage1Result(outcome=GovDecision.ADD, reason="stage1_no_neighbors")

    # -- canonical-key pre-check (KV), idempotent-update pre-check (Vector) ----
    tier_items = cache.items[candidate.tier]
    if candidate.backend == "kv" and candidate.ref is not None:
        stored = tier_items.get(str(candidate.ref))
        if stored is not None:
            same_value = _normalize_for_match(stored.text) == _normalize_for_match(
                candidate.text
            )
            if candidate.kind == "replace" and same_value:
                # Idempotent replace: backend would succeed and change nothing.
                res.outcome = GovDecision.NOOP
                res.reason = "stage1_precheck_idempotent_replace"
                res.precheck = "kv_idempotent_replace"
            elif candidate.kind == "add":
                # add on an existing key would error "Key name must be unique."
                # -> rewrite to the corresponding replace (idempotent when the
                # value matches; a legitimate update when it differs).
                res.outcome = GovDecision.REWRITE
                res.reason = (
                    "stage1_precheck_idempotent_key"
                    if same_value
                    else "stage1_precheck_add_existing_key"
                )
                res.precheck = "kv_add_existing_key"
                res.target_ref = stored.ref
                res.target_tier = stored.tier
                res.superseded_text = stored.text
                res.rewritten_call = build_rewrite_call(candidate, stored)
            if res.precheck is not None:
                res.latency_ms = round((time.perf_counter() - start) * 1000, 3)
                return res
    elif candidate.backend == "vector" and candidate.kind == "update":
        stored = tier_items.get(str(candidate.ref))
        if stored is not None and _normalize_for_match(stored.text) == _normalize_for_match(candidate.text):
            res.outcome = GovDecision.NOOP
            res.reason = "stage1_precheck_idempotent_update"
            res.precheck = "vector_idempotent_update"
            res.latency_ms = round((time.perf_counter() - start) * 1000, 3)
            return res

    # -- NLI over top-k neighbors ------------------------------------------
    selected = select_stage1_candidates(candidate, signals, cache, cfg.nli_k)
    res.candidates = [
        {"ref": it.ref, "tier": it.tier, "sim": round(sim, 4)} for it, sim in selected
    ]
    res.neighbor_items = [it for it, _ in selected]
    if not selected:
        res.latency_ms = round((time.perf_counter() - start) * 1000, 3)
        return res
    if nli_scorer is None:
        res.outcome = GovDecision.ADD
        res.reason = "stage1_nli_unavailable"
        res.latency_ms = round((time.perf_counter() - start) * 1000, 3)
        return res

    # One batched forward pass for all 2k directed pairs (SS7.4).
    pairs = []
    for it, _ in selected:
        pairs.append((it.text, candidate.text))  # fwd: stored entails candidate
        pairs.append((candidate.text, it.text))  # bwd: candidate entails stored
    probs = nli_scorer.probs_batch(pairs)

    decisive = None
    for i, (it, sim) in enumerate(selected):
        fwd_c, fwd_n, fwd_e = probs[2 * i]
        bwd_c, bwd_n, bwd_e = probs[2 * i + 1]
        verdict = "neutral"
        if fwd_e >= cfg.tau_entail and bwd_e >= cfg.tau_entail:
            verdict = "equivalent"
        elif fwd_e >= cfg.tau_entail:
            verdict = "derivable"
        elif bwd_e >= cfg.tau_entail and (bwd_e - fwd_e) > cfg.delta_spec:
            verdict = "more_specific"
        elif bwd_e >= cfg.tau_entail:
            verdict = "specificity_inconclusive"
        elif max(fwd_c, bwd_c) >= cfg.tau_contra:
            verdict = "contradiction"
        res.pairs.append(
            {
                "ref": it.ref,
                "tier": it.tier,
                "premise_fwd": it.text,
                "fwd": [round(fwd_c, 4), round(fwd_n, 4), round(fwd_e, 4)],
                "bwd": [round(bwd_c, 4), round(bwd_n, 4), round(bwd_e, 4)],
                "verdict": verdict,
            }
        )
        if decisive is None and verdict != "neutral":
            decisive = (it, verdict)

    if decisive is None:
        res.outcome = GovDecision.ESCALATE
        res.reason = "stage1_all_neutral"
    else:
        target, verdict = decisive
        if verdict in ("equivalent", "derivable"):
            res.outcome = GovDecision.NOOP
            res.reason = f"stage1_{verdict}"
        elif verdict == "specificity_inconclusive":
            res.outcome = GovDecision.ADD
            res.reason = "stage1_keep_both"
        else:  # more_specific | contradiction -> supersede target
            res.outcome = GovDecision.REWRITE
            res.reason = f"stage1_{verdict}"
            res.target_ref = target.ref
            res.target_tier = target.tier
            res.superseded_text = target.text
            res.rewritten_call = build_rewrite_call(candidate, target)
    res.latency_ms = round((time.perf_counter() - start) * 1000, 3)
    return res


# ---------------------------------------------------------------------------
# Stage 2 (retrieval entropy) escalation resolver -- STAGE0 doc SS8, Plan 1 Step 6.
# Consumes retrieval_sim + probe_gen (lazy imports: Stage 0 never pays for them).
# ---------------------------------------------------------------------------


@dataclass
class Stage2Result:
    outcome: GovDecision  # ADD (accept / accept-low-confidence) | REWRITE
    reason: str  # stage2_accept | stage2_accept_rewritten | stage2_accept_low_confidence
    probes: list = field(default_factory=list)
    per_probe: list = field(default_factory=list)
    min_margin: Optional[float] = None
    worst_probe: Optional[str] = None
    canonicalization: dict = field(default_factory=dict)
    low_confidence: bool = False
    rewritten_call: Optional[str] = None
    dH_neighbor: list = field(default_factory=list)  # decision-inert diagnostics
    dH_mean: Optional[float] = None
    latency_ms: float = 0.0

    def to_log(self) -> dict:
        return {
            "outcome": self.outcome.value,
            "reason": self.reason,
            "probes": self.probes,
            "per_probe": self.per_probe,
            "min_margin": self.min_margin,
            "worst_probe": self.worst_probe,
            "canonicalization": self.canonicalization,
            "low_confidence": self.low_confidence,
            "rewritten_call": self.rewritten_call,
            "dH_neighbor": self.dH_neighbor,
            "dH_mean": self.dH_mean,
            "latency_ms": self.latency_ms,
        }


def _sanitize_kv_key(key: str) -> str:
    key = re.sub(r"[^a-z0-9_]+", "_", key.lower()).strip("_")
    key = re.sub(r"_+", "_", key)
    return key


def canonicalize_candidate(
    candidate: WriteCandidate,
    rival_entries: list,
    verbatim_values: list,
    user_text: str,
    max_len: int,
) -> Optional[tuple]:
    """One-shot deterministic canonicalization (SS8.2): pick the highest-value
    discriminative token (verbatim values first, then content tokens of the
    user text / candidate value) absent from every rival entry; KV appends it
    to the key (validated against _KV_KEY_PATTERN), Vector prepends a clause
    (re-checked against the tier's max_entry_length -- the customer-6 lesson).

    Returns (rewritten_call, token) or None when no valid rewrite exists.
    """
    from bfcl_eval.model_handler.middleware.probe_gen import content_tokens_ordered

    rival_blob = _normalize_for_match(" ||| ".join(rival_entries))
    pool = list(verbatim_values)
    pool += content_tokens_ordered(user_text or "")
    if candidate.backend == "kv":
        pool += content_tokens_ordered(str(candidate.args.get("value", "")))
    else:
        pool += content_tokens_ordered(candidate.text)
    token = None
    for cand_tok in pool:
        norm = _normalize_for_match(str(cand_tok))
        if norm and norm not in rival_blob:
            token = str(cand_tok)
            break
    if token is None:
        return None

    if candidate.backend == "kv":
        old_key = str(candidate.args["key"])
        new_key = _sanitize_kv_key(f"{old_key}_{token}")
        if not _KV_KEY_PATTERN.match(new_key) or new_key == old_key:
            return None
        value = str(candidate.args["value"])
        if len(value) > max_len:
            return None
        return f"{candidate.tier}_memory_add(key={new_key!r}, value={value!r})", token
    new_text = f"Regarding {token}: {candidate.text}"
    if len(new_text) > max_len:
        return None
    return f"{candidate.tier}_memory_add(text={new_text!r})", token


def stage2_resolve(
    candidate: WriteCandidate,
    signals: Stage0Signals,
    cache: GovernanceCache,
    cfg: GovConfig,
    user_text: str,
    neighbors: list,
    canonicalize_llm_fn=None,
) -> Stage2Result:
    """SS8.1 verification loop: decision probes (from user text ONLY) must
    retrieve the provisional candidate as top-1 on every probe with
    min-margin > GOV_S2_MARGIN. One deterministic canonicalization attempt on
    failure, re-tested on the same probes; final fallback writes as-is with a
    low_confidence flag. Stage 2 never suppresses.

    Simulation is per-tier and backend-faithful (retrieval_sim); the whitened
    space never enters. ``dH_neighbor`` / ``N_eff`` are logged, decision-inert.
    """
    from bfcl_eval.model_handler.middleware.probe_gen import (
        ProbeConfig,
        generate_decision_probes,
        generate_item_probes,
    )
    from bfcl_eval.model_handler.middleware.retrieval_sim import (
        delta_h_for_probes,
        margin as sim_margin,
        entropy as sim_entropy,
        n_eff as sim_n_eff,
        simulate_kv,
        simulate_vector,
    )

    start = time.perf_counter()
    res = Stage2Result(outcome=GovDecision.ADD, reason="stage2_accept_low_confidence")
    probe_cfg = ProbeConfig.from_env()
    probes = generate_decision_probes(user_text, signals.verbatim_values, probe_cfg)
    res.probes = [p.to_dict() for p in probes]
    temperature = cfg.s2_t_kv if candidate.backend == "kv" else cfg.s2_t_vec

    tier_items = cache.items[candidate.tier]
    rival_refs = list(tier_items.keys())
    rival_texts = [tier_items[r].text for r in rival_refs]

    def candidate_entry(cand: WriteCandidate) -> str:
        return str(cand.args["key"]) if cand.backend == "kv" else cand.text

    def test_probes(entry: str) -> tuple:
        """Returns (all_top1_is_candidate, min_margin, per_probe_rows)."""
        corpus = (rival_refs if candidate.backend == "kv" else rival_texts) + [entry]
        cand_idx = len(corpus) - 1
        rows, min_m, all_top1 = [], None, True
        for p in probes:
            if candidate.backend == "kv":
                ranked = simulate_kv(corpus, p.text, k=5)
                top1_is_cand = bool(ranked) and ranked[0][1] == entry
            else:
                ranked = simulate_vector(corpus, p.text, k=5)
                top1_is_cand = bool(ranked) and ranked[0][1] == cand_idx
            scores = [s for s, _ in ranked]
            m = sim_margin(scores)
            h = sim_entropy(scores, temperature, k=5)
            rows.append(
                {
                    "probe": p.text,
                    "channel": p.channel,
                    "top1_is_candidate": top1_is_cand,
                    "margin": round(m, 6),
                    "H": round(h, 6),
                    "n_eff": round(sim_n_eff(h), 6),
                }
            )
            all_top1 = all_top1 and top1_is_cand
            min_m = m if min_m is None else min(min_m, m)
        return all_top1, min_m, rows

    if probes:
        ok, min_m, rows = test_probes(candidate_entry(candidate))
        res.per_probe = rows
        res.min_margin = round(min_m, 6) if min_m is not None else None
        if rows:
            worst = min(rows, key=lambda r: r["margin"])
            res.worst_probe = worst["probe"]
        if ok and min_m is not None and min_m > cfg.s2_margin:
            res.outcome = GovDecision.ADD
            res.reason = "stage2_accept"
        else:
            # One-shot canonicalization, deterministic first (SS8.2).
            rewritten = canonicalize_candidate(
                candidate,
                rival_refs if candidate.backend == "kv" else rival_texts,
                signals.verbatim_values,
                user_text,
                cache.max_entry_length(candidate.tier),
            )
            if rewritten is None and cfg.s2_canon_llm and canonicalize_llm_fn is not None:
                llm_call = canonicalize_llm_fn(candidate, rival_texts)
                rewritten = (llm_call, "<llm>") if llm_call else None
            if rewritten is not None:
                new_call, token = rewritten
                new_cand = build_candidate(candidate.backend, new_call)
                if new_cand is not None and preflight_would_succeed(new_cand, cache):
                    ok2, min_m2, rows2 = test_probes(candidate_entry(new_cand))
                    res.canonicalization = {
                        "applied": True,
                        "token": token,
                        "rewritten_call": new_call,
                        "retry_min_margin": round(min_m2, 6) if min_m2 is not None else None,
                    }
                    if ok2 and min_m2 is not None and min_m2 > cfg.s2_margin:
                        res.outcome = GovDecision.REWRITE
                        res.reason = "stage2_accept_rewritten"
                        res.rewritten_call = new_call
                else:
                    res.canonicalization = {
                        "applied": False,
                        "token": token,
                        "rewritten_call": new_call,
                        "blocked_by": "parse_or_preflight",
                    }
            else:
                res.canonicalization = {"applied": False, "token": None}
            if res.outcome != GovDecision.REWRITE:
                res.outcome = GovDecision.ADD
                res.reason = "stage2_accept_low_confidence"
                res.low_confidence = True
    else:
        # No user text captured -> no anti-circular probe source; never decide
        # from candidate text (SS4.1 invariant). Accept with the flag.
        res.reason = "stage2_no_probe_source"
        res.low_confidence = True

    # -- decision-inert diagnostics: dH on the neighbors' own cached probes ----
    corpus = rival_refs if candidate.backend == "kv" else rival_texts
    entry = candidate_entry(candidate)
    dhs = []
    for it in neighbors:
        item_probes = list(it.probes) or [
            p.text
            for p in generate_item_probes(
                candidate.backend, it.text, ref=it.ref, source="stored_text"
            )
        ]
        if not item_probes or not corpus:
            continue
        d = delta_h_for_probes(
            candidate.backend, corpus, entry, item_probes, k=5, temperature=temperature
        )
        res.dH_neighbor.append({"ref": it.ref, "dH": d["dH_mean"], "n_eff": d["n_eff_after_mean"]})
        dhs.append(d["dH_mean"])
    if dhs:
        res.dH_mean = round(sum(dhs) / len(dhs), 6)

    res.latency_ms = round((time.perf_counter() - start) * 1000, 3)
    return res


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
        nli_scorer=None,
        sidecar_path: Optional[str] = None,
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
        # idx -> {"mode": "noop"|"rewrite", ...} for the current step
        self._pending: dict = {}
        # Stage 1/2 state. The scorer is injectable for tests; when None and
        # GOV_NLI_ENABLED, the DeBERTa singleton is loaded lazily on first use.
        self._nli_scorer = nli_scorer
        self._nli_loaded = nli_scorer is not None
        self.user_text: str = ""  # latest user turn (Stage 2 probe source)
        self._low_conf_texts: set = set()  # (tier, text) to flag at observe time
        # Probe-cache sidecar (Stage 2): <scenario>_gov_state.json next to the
        # backend snapshot. Only touched when GOV_S2_ENABLED.
        if sidecar_path is not None:
            self.sidecar_path = sidecar_path
        elif snapshot_path and snapshot_path.endswith("_final.json"):
            self.sidecar_path = snapshot_path[: -len("_final.json")] + "_gov_state.json"
        else:
            self.sidecar_path = ""
        if cfg.s2_enabled:
            self._load_probe_sidecar()

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

    def _get_nli_scorer(self):
        if not self._nli_loaded:
            self._nli_loaded = True
            from bfcl_eval.model_handler.middleware.semantic_entropy import (
                _get_nli_model,
            )

            self._nli_scorer = _get_nli_model()
        return self._nli_scorer

    def _govern_one(self, idx: int, candidate: WriteCandidate, governed: list):
        v_w = self._whiten_one(candidate.text)
        preflight_ok = preflight_would_succeed(candidate, self.cache)
        signals = compute_signals(v_w, candidate, self.cache, self.thresholds, self.cfg)
        decision, reason = decide(candidate, signals, self.cache, self.cfg, preflight_ok)

        # -- escalation cascade (Stage 1 NLI -> Stage 2 retrieval entropy) ----
        stage1_log = stage2_log = None
        rewritten_call = None
        low_confidence = False
        if decision == GovDecision.ESCALATE:
            decision, reason, stage1_log, stage2_log, rewritten_call, low_confidence = (
                self._resolve_escalation(candidate, signals, preflight_ok)
            )

        shadow_id = None
        synthetic = None
        applied = "none"
        if not self.cfg.dry_run:
            if decision == GovDecision.NOOP:
                if candidate.backend == "vector" and candidate.kind == "add":
                    shadow_id = self._shadow_next
                    self._shadow_next += 1
                    self._issued_shadow_ids.add(shadow_id)
                synthetic = synthetic_success(candidate, shadow_id)
                governed[idx] = DECOY_CALL
                self._pending[idx] = {
                    "mode": "noop",
                    "original_call": candidate.raw_call,
                    "synthetic": synthetic,
                }
                applied = "noop"
            elif decision == GovDecision.REWRITE:
                governed[idx] = rewritten_call
                self._pending[idx] = {
                    "mode": "rewrite",
                    "original_call": candidate.raw_call,
                    "rewritten_call": rewritten_call,
                }
                applied = "rewrite"
                if low_confidence:
                    rc = build_candidate(self.backend, rewritten_call)
                    if rc is not None:
                        self._low_conf_texts.add((rc.tier, rc.text))
            elif low_confidence:
                self._low_conf_texts.add((candidate.tier, candidate.text))

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
                "applied": applied,
                "original_call": candidate.raw_call if applied == "rewrite" else None,
                "rewritten_call": rewritten_call,
                "low_confidence": low_confidence,
                "stage1": stage1_log,
                "stage2": stage2_log,
                "latency_ms": signals.latency_ms,
            },
            self.cfg,
        )

    def _resolve_escalation(
        self, candidate: WriteCandidate, signals: Stage0Signals, preflight_ok: bool
    ) -> tuple:
        """Run Stage 1 (and Stage 2 on all-neutral) and map their outcome to the
        final applied decision, honoring shadow modes and preflight guards.

        Returns (decision, reason, stage1_log, stage2_log, rewritten_call,
        low_confidence). With both stages disabled this is bit-identical to the
        pre-Stage-1 fallback: (ADD, "escalate:stage0_escalate_fallback").
        """
        cfg = self.cfg
        if not cfg.nli_enabled:
            return GovDecision.ADD, "escalate:stage0_escalate_fallback", None, None, None, False

        s1 = stage1_resolve(
            candidate, signals, self.cache, cfg, preflight_ok, self._get_nli_scorer()
        )
        stage1_log = s1.to_log()
        stage2_log = None
        outcome, reason = s1.outcome, s1.reason
        rewritten_call = s1.rewritten_call
        low_confidence = False
        shadowed_by = "stage1_shadow" if cfg.nli_shadow else None

        if outcome == GovDecision.ESCALATE:
            if cfg.s2_enabled:
                s2 = stage2_resolve(
                    candidate,
                    signals,
                    self.cache,
                    cfg,
                    self.user_text,
                    neighbors=s1.neighbor_items,
                    canonicalize_llm_fn=None,  # injected by the handler in Plan 2
                )
                stage2_log = s2.to_log()
                outcome, reason = s2.outcome, s2.reason
                rewritten_call = s2.rewritten_call
                low_confidence = s2.low_confidence
                if shadowed_by is None and cfg.s2_shadow:
                    shadowed_by = "stage2_shadow"
            else:
                outcome, reason = GovDecision.ADD, "stage1_all_neutral_no_stage2"

        # -- shadow discipline: compute + log everything, change nothing -------
        if shadowed_by is not None:
            stage1_log["shadowed"] = True
            if stage2_log is not None:
                stage2_log["shadowed"] = True
            return (
                GovDecision.ADD,
                f"escalate:{shadowed_by}:{reason}",
                stage1_log,
                stage2_log,
                None,
                False,
            )

        # -- live application guards ------------------------------------------
        if outcome == GovDecision.NOOP and not preflight_ok:
            # Same absolute protection as Stage 0 NOOPs (SS7.5): a synthetic
            # success must never mask a real backend error.
            return (
                GovDecision.ADD,
                f"escalate:{reason}_preflight_blocked",
                stage1_log,
                stage2_log,
                None,
                low_confidence,
            )
        if outcome == GovDecision.REWRITE:
            rc = build_candidate(self.backend, rewritten_call) if rewritten_call else None
            if rc is None or not preflight_would_succeed(rc, self.cache):
                # Never rewrite into a guaranteed error -- fall back to ADD.
                return (
                    GovDecision.ADD,
                    f"escalate:{reason}_rewrite_preflight_blocked",
                    stage1_log,
                    stage2_log,
                    None,
                    low_confidence,
                )
        return (
            outcome,
            f"escalate:{reason}",
            stage1_log,
            stage2_log,
            rewritten_call if outcome == GovDecision.REWRITE else None,
            low_confidence,
        )

    # -- HOOK 2: post-execution -----------------------------------------------

    def patch_results(self, execution_results: list, decoded_calls: list) -> list:
        """Substitute synthetic successes at suppressed (NOOP) indices, restore
        their original call strings in ``decoded_calls`` (in place -- it is the
        same list the parent zips against for the tool-message ``name``), and
        observe genuine successes to keep the cache mirror in sync.

        Rewrite-mode entries (Stage 1/2) are the opposite (SS7.5): the REAL
        backend result flows through untouched and the executed (rewritten) call
        is NOT restored -- ``_observe`` must see the executed call so the mirror
        stays truthful; the original call is preserved in the decision log
        record. Returns the patched results list."""
        new_results = list(execution_results)
        for idx, info in self._pending.items():
            if info.get("mode", "noop") != "noop":
                continue
            if idx < len(new_results):
                new_results[idx] = info["synthetic"]
            if idx < len(decoded_calls):
                decoded_calls[idx] = info["original_call"]

        for idx, call in enumerate(decoded_calls):
            if idx in self._pending and self._pending[idx].get("mode", "noop") == "noop":
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
        item = MemoryItem(
            ref=ref,
            text=text,
            emb_whitened=self._whiten_one(text),
            turn_written=self.step,
            tier=tier,
        )
        if (tier, text) in self._low_conf_texts:
            item.low_confidence = True
            self._low_conf_texts.discard((tier, text))
        if self.cfg.s2_enabled:
            # Write-time probe cache (SS8.3): cheap, deterministic templates.
            from bfcl_eval.model_handler.middleware.probe_gen import (
                generate_item_probes,
            )

            item.probes = [
                p.text
                for p in generate_item_probes(self.backend, text, ref=ref)
            ]
            item.probe_provenance = "write_time"
        self.cache.put(item)
        if self.cfg.s2_enabled:
            self._persist_probe_sidecar()
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
                "low_confidence": item.low_confidence,
            },
            self.cfg,
        )

    # -- Stage 2 probe-cache sidecar (SS8.3) ----------------------------------

    def _load_probe_sidecar(self):
        """Attach persisted probes to rehydrated items; regenerate (flagged
        ``stored_text`` -- diagnostics only, never accept/reject) on cache miss."""
        from bfcl_eval.model_handler.middleware.probe_gen import generate_item_probes

        stored: dict = {}
        if self.sidecar_path and os.path.exists(self.sidecar_path):
            try:
                with open(self.sidecar_path, "r", encoding="utf-8") as f:
                    stored = json.load(f).get("tiers", {})
            except (json.JSONDecodeError, OSError) as e:
                print(f"[GOV] probe sidecar unreadable ({e}); regenerating probes")
        for tier in ("core", "archival"):
            for ref, item in self.cache.items[tier].items():
                entry = stored.get(tier, {}).get(ref)
                if entry is not None and entry.get("text") == item.text:
                    item.probes = list(entry.get("probes", []))
                    item.probe_provenance = "sidecar"
                else:
                    item.probes = [
                        p.text
                        for p in generate_item_probes(
                            self.backend, item.text, ref=ref, source="stored_text"
                        )
                    ]
                    item.probe_provenance = "stored_text"

    def _persist_probe_sidecar(self):
        if not self.sidecar_path:
            return
        payload = {
            "backend": self.backend,
            "tiers": {
                tier: {
                    ref: {
                        "text": it.text,
                        "probes": it.probes,
                        "probe_provenance": it.probe_provenance,
                    }
                    for ref, it in self.cache.items[tier].items()
                }
                for tier in ("core", "archival")
            },
        }
        try:
            os.makedirs(os.path.dirname(self.sidecar_path) or ".", exist_ok=True)
            with open(self.sidecar_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except OSError as e:  # persistence must never break inference
            print(f"[GOV] failed to persist probe sidecar: {e}")
