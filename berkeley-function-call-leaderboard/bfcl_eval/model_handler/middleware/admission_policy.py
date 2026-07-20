"""
Memory-mutation admission policies (Plan v2 Step 4, plan SS9).
=============================================================

The concrete :class:`MemoryMutationAdmissionBoundary` and the policy registry:

  legacy_full                    -- delegate escalations to the old NLI ->
                                    Stage-2 cascade (handled by the session's
                                    unchanged code path; the boundary is never
                                    constructed for this policy).
  geometry_only                  -- Stage 0 only; every ESCALATE falls back to
                                    ADD (the validated stage0_only arm).
  geometry_margin_entropy_v1     -- Stage 0 -> joint Margin-Entropy rule
                                    (Option A, plan SS9.2). Deterministic,
                                    LLM-free, NLI-free.
  geometry_margin_entropy_risk_v1-- Option B: calibrated monotone logistic risk
                                    (plan SS9.4); requires GOV_ME_CALIB.

Design invariants (plan SS9.2):
  (i)  the interference term only moves a decision between ADD and
       SAFE_REWRITE/ABSTAIN when margin/locate is already indeterminate or the
       candidate shadows an existing memory — never a standalone rejector;
  (ii) NOOP requires the same preflight guard as Stage 0;
  (iii) the default on total ambiguity is a flagged ADD, not suppression.

ABSTAIN semantics in v1 (plan SS9.3): observational — the session executes the
write and tags the stored item; ABSTAIN exists as a distinct logged action so
risk-coverage curves can be computed offline.

No NLI import can occur on any code path in this module (test: no_online_nli).
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Optional

from bfcl_eval.model_handler.middleware.geometry_gate import GeometryGate, GeometryResult
from bfcl_eval.model_handler.middleware.governance_filter import (
    GOV_POLICIES,
    GovConfig,
    GovDecision,
    GovernanceCache,
    Stage0Signals,
    ThresholdState,
    WriteCandidate,
    build_candidate,
    canonicalize_candidate,
    preflight_would_succeed,
    select_stage1_candidates,
)
from bfcl_eval.model_handler.middleware.memory_mutation import (
    AdmissionDecision,
    MemoryMutationCandidate,
)

# Single source of truth for the registry lives in governance_filter (read by
# GovConfig.from_env validation); re-exported here as the public name.
POLICIES = GOV_POLICIES

# Candidate-locating decision-probe channels (plan SS9.1): value_question is
# interference-only and never enters the locate/strong aggregation (the SS4.2
# fix). Paraphrase probes are user-sentence paraphrases -> locating.
_LOCATE_CHANNEL_PREFIXES = (
    "template:user_identity",
    "template:user_keywords",
    "paraphrase:",
)
# KV locate/strong runs on key-channel item probes instead (plan SS10): BM25
# over key names makes natural-language probes structurally degenerate.
_KV_LOCATE_CHANNELS = ("template:key", "template:key_value")

SMALLSTORE_N = 5  # KV entropy features inactive below this tier size (plan SS10)
CHURN_TOP_N = 3


@dataclass
class MarginEntropySignals:
    """Full Stage-1 signal record (log schema SS17.1 ``s1`` block)."""

    per_probe: list = field(default_factory=list)
    dH_self: Optional[float] = None
    dH_neighbor: list = field(default_factory=list)
    dH_mean: Optional[float] = None
    n_eff_norm: Optional[float] = None
    disp: Optional[float] = None
    churn: Optional[float] = None
    rank_self_med: Optional[float] = None
    nmargin_med: Optional[float] = None
    locate: bool = False
    strong: bool = False
    interf: bool = False
    dup_risk: bool = False
    smallstore: bool = False
    probes_n: int = 0
    latency_ms: float = 0.0

    def to_log(self) -> dict:
        return {
            "per_probe": self.per_probe,
            "dH_self": self.dH_self,
            "dH_neighbor": self.dH_neighbor,
            "dH_mean": self.dH_mean,
            "n_eff_norm": self.n_eff_norm,
            "disp": self.disp,
            "churn": self.churn,
            "rank_self_med": self.rank_self_med,
            "nmargin_med": self.nmargin_med,
            "locate": self.locate,
            "strong": self.strong,
            "interf": self.interf,
            "dup_risk": self.dup_risk,
            "smallstore": self.smallstore,
            "probes_n": self.probes_n,
            "latency_ms": self.latency_ms,
        }


def compute_margin_entropy_signals(
    candidate: WriteCandidate,
    signals: Stage0Signals,
    cache: GovernanceCache,
    cfg: GovConfig,
    user_text: str,
) -> MarginEntropySignals:
    """Deterministic Stage-1 signal computation (plan SS9.1). All simulation is
    per-tier and backend-faithful via ``retrieval_sim``; provisional inserts are
    list copies — no store or mirror is ever touched."""
    from bfcl_eval.model_handler.middleware.entropy_metrics import (
        churn as em_churn,
        disp as em_disp,
        median,
        n_eff_norm as em_n_eff_norm,
        nmargin as em_nmargin,
        rank_self as em_rank_self,
        top_identity_set,
    )
    from bfcl_eval.model_handler.middleware.probe_gen import (
        ProbeConfig,
        generate_decision_probes,
        generate_item_probes,
    )
    from bfcl_eval.model_handler.middleware.retrieval_sim import (
        delta_h_for_probes,
        entropy as sim_entropy,
        simulate_kv,
        simulate_vector,
    )

    start = time.perf_counter()
    out = MarginEntropySignals()
    backend = candidate.backend
    temperature = cfg.s2_t_kv if backend == "kv" else cfg.s2_t_vec
    tier_items = cache.items[candidate.tier]
    rival_refs = list(tier_items.keys())
    rival_texts = [tier_items[r].text for r in rival_refs]
    n_tier = len(rival_refs)
    out.smallstore = backend == "kv" and n_tier < SMALLSTORE_N

    entry = str(candidate.args["key"]) if backend == "kv" else candidate.text
    corpus_pre = rival_refs if backend == "kv" else rival_texts
    corpus_post = list(corpus_pre) + [entry]
    cand_identity = entry if backend == "kv" else len(corpus_post) - 1

    def rank_probe(corpus, probe_text):
        if backend == "kv":
            return simulate_kv(corpus, probe_text, k=5)
        return simulate_vector(corpus, probe_text, k=5)

    # -- probe set: user-text decision probes + (KV) key-channel item probes ---
    probe_cfg = ProbeConfig.from_env()
    probes = list(generate_decision_probes(user_text, signals.verbatim_values, probe_cfg))
    if backend == "kv":
        probes += generate_item_probes(
            "kv", candidate.text, ref=str(candidate.args["key"])
        )

    locate_ranks, locate_nmargins, locate_disps, locate_h_posts, dh_self_terms = (
        [], [], [], [], []
    )
    for p in probes:
        ranked_post = rank_probe(corpus_post, p.text)
        scores_post = [s for s, _ in ranked_post]
        scores_pre = [s for s, _ in rank_probe(corpus_pre, p.text)] if corpus_pre else []
        h_pre = sim_entropy(scores_pre, temperature, k=5) if scores_pre else 0.0
        h_post = sim_entropy(scores_post, temperature, k=5)
        rank = em_rank_self(ranked_post, cand_identity, k=5)
        nm = em_nmargin(scores_post, backend)
        row = {
            "probe": p.text,
            "channel": p.channel,
            "rank_self": rank,
            "margin": round(scores_post[0] - scores_post[1], 6) if len(scores_post) > 1 else 0.0,
            "nmargin": round(nm, 6),
            "H_pre": round(h_pre, 6),
            "H_post": round(h_post, 6),
        }
        out.per_probe.append(row)
        if backend == "kv":
            is_locating = p.channel in _KV_LOCATE_CHANNELS
        else:
            is_locating = p.channel.startswith(_LOCATE_CHANNEL_PREFIXES)
        if is_locating:
            locate_ranks.append(rank)
            locate_nmargins.append(nm)
            locate_disps.append(em_disp(scores_post))
            locate_h_posts.append(h_post)
            dh_self_terms.append(h_post - h_pre)
    out.probes_n = len(probes)

    if dh_self_terms:
        out.dH_self = round(sum(dh_self_terms) / len(dh_self_terms), 6)
    if locate_h_posts:
        mean_h_post = sum(locate_h_posts) / len(locate_h_posts)
        out.n_eff_norm = round(em_n_eff_norm(mean_h_post, n_tier, k=5), 6)
    if locate_disps:
        out.disp = round(median(locate_disps), 6)
    out.rank_self_med = median(locate_ranks)
    out.nmargin_med = round(median(locate_nmargins), 6) if locate_nmargins else None

    # -- interference: neighbors' own probes (dH) + rank churn -----------------
    neighbors = [it for it, _ in select_stage1_candidates(candidate, signals, cache, cfg.nli_k)]
    dhs, churns = [], []
    for it in neighbors:
        item_probes = list(it.probes) or [
            p.text
            for p in generate_item_probes(backend, it.text, ref=it.ref, source="stored_text")
        ]
        if not item_probes or not corpus_pre:
            continue
        d = delta_h_for_probes(
            backend, corpus_pre, entry, item_probes, k=5, temperature=temperature
        )
        probe_churns = []
        for ip in item_probes:
            pre_top = top_identity_set(rank_probe(corpus_pre, ip), n=CHURN_TOP_N)
            post_top = top_identity_set(rank_probe(corpus_post, ip), n=CHURN_TOP_N)
            probe_churns.append(em_churn(pre_top, post_top))
        n_churn = round(sum(probe_churns) / len(probe_churns), 6) if probe_churns else 0.0
        out.dH_neighbor.append({"ref": it.ref, "dH": d["dH_mean"], "churn": n_churn})
        dhs.append(d["dH_mean"])
        churns.append(n_churn)
    if dhs:
        out.dH_mean = round(sum(dhs) / len(dhs), 6)
    if churns:
        out.churn = round(sum(churns) / len(churns), 6)

    out.latency_ms = round((time.perf_counter() - start) * 1000, 3)
    return out


def apply_option_a_rule(
    candidate: WriteCandidate,
    s0: Stage0Signals,
    me: MarginEntropySignals,
    cache: GovernanceCache,
    cfg: GovConfig,
    user_text: str,
    preflight_ok: bool,
) -> tuple:
    """The v1 joint rule (plan SS9.2). Returns
    ``(action, reason_code, rewritten_call, flagged)``.

    ``flagged`` marks ADD-with-flag outcomes (bounded interference / total
    ambiguity) for the session's ``_flagged_texts`` tagging.
    """
    r_thr = cfg.me_r
    m_thr = cfg.me_margin_kv if candidate.backend == "kv" else cfg.me_margin_vec
    d_thr = cfg.me_dh_kv if candidate.backend == "kv" else cfg.me_dh_vec
    c_thr = cfg.me_churn_kv if candidate.backend == "kv" else cfg.me_churn_vec

    locate = me.rank_self_med is not None and me.rank_self_med <= r_thr
    strong = me.nmargin_med is not None and me.nmargin_med >= m_thr
    interf = (me.dH_mean is not None and me.dH_mean >= d_thr) or (
        me.churn is not None and me.churn >= c_thr
    )
    suffix = ""
    if me.smallstore:
        # KV entropy features are structurally degenerate below SMALLSTORE_N
        # items (plan SS10): the interference term is deactivated, flagged.
        interf = False
        suffix = "_smallstore"
    dup_risk = s0.sim_max >= (cfg.sim_high - cfg.me_dup_eps)
    me.locate, me.strong, me.interf, me.dup_risk = locate, strong, interf, dup_risk

    if locate and strong and not interf:
        return "ADD", "s1_confident" + suffix, None, False
    if locate and strong and interf:
        rewritten = (
            None
            if cfg.me_rewrite_disabled  # ablation arm 7 (plan SS18)
            else _safe_rewrite(candidate, cache, s0, user_text)
        )
        if rewritten is not None:
            return "SAFE_REWRITE", "s1_bounded_interference" + suffix, rewritten, False
        return "ADD", "s1_bounded_interference" + suffix, None, True
    if not locate and dup_risk:
        if not preflight_ok:
            # Same absolute guard as Stage 0: a synthetic success must never
            # mask a real backend error.
            return "ADD", "s1_shadowed_duplicate_preflight_blocked" + suffix, None, True
        return "NOOP", "s1_shadowed_duplicate" + suffix, None, False
    if not locate and not strong and interf:
        return "ABSTAIN", "s1_high_risk" + suffix, None, True
    return "ADD", "s1_ambiguous_default" + suffix, None, True


def _safe_rewrite(
    candidate: WriteCandidate,
    cache: GovernanceCache,
    s0: Stage0Signals,
    user_text: str,
) -> Optional[str]:
    """Non-destructive dual-representation rewrite (plan SS11): KV *keys* only
    (deterministic entity-suffix; the value is stored byte-identically), adds
    only. Vector has no rewrites in v1. Returns a call string or None."""
    if candidate.backend != "kv" or candidate.kind != "add":
        return None
    tier_items = cache.items[candidate.tier]
    rewritten = canonicalize_candidate(
        candidate,
        list(tier_items.keys()),
        s0.verbatim_values,
        user_text,
        cache.max_entry_length(candidate.tier),
    )
    if rewritten is None:
        return None
    new_call, _token = rewritten
    new_cand = build_candidate(candidate.backend, new_call)
    if new_cand is None or not preflight_would_succeed(new_cand, cache):
        return None
    # Belt-and-braces on the SS11 contract: the rewrite may only change the key.
    if str(new_cand.args.get("value")) != str(candidate.args.get("value")):
        return None
    return new_call


# ---------------------------------------------------------------------------
# NC arm: shuffled-entropy negative control (plan SS18). dH values are permuted
# WITHIN (backend, store-size bin) at decision time: each decision swaps its
# dH_mean/dH_self for a value drawn from the rolling pool of previously seen
# values in its bin, selected by a seeded hash of the candidate -- deterministic
# for a fixed log order, destroys the dH<->candidate pairing, preserves the
# marginal distribution.
# ---------------------------------------------------------------------------

_NC_POOLS: dict = {}


def _store_size_bin(n_items: int) -> int:
    return min(int(n_items) // 5, 4)  # bins: 0-4, 5-9, 10-14, 15-19, 20+


def nc_shuffle_dh(me: MarginEntropySignals, candidate: WriteCandidate,
                  n_items: int, seed: int) -> None:
    key = (candidate.backend, _store_size_bin(n_items))
    pool = _NC_POOLS.setdefault(key, [])
    own = (me.dH_mean, me.dH_self)
    if pool:
        digest = hashlib.sha256(
            f"{seed}|{candidate.raw_call}|{len(pool)}".encode("utf-8")
        ).digest()
        idx = int.from_bytes(digest[:4], "big") % len(pool)
        swapped = pool[idx]
        me.dH_mean = swapped[0]
        me.dH_self = swapped[1]
    if own[0] is not None or own[1] is not None:
        pool.append(own)
        del pool[:-200]  # bounded rolling pool


# ---------------------------------------------------------------------------
# Option B: calibrated monotone logistic risk (plan SS9.4)
# ---------------------------------------------------------------------------

_RISK_FEATURES = ("nmargin_med", "rank_self_med", "dH_mean", "n_eff_norm", "sim_max", "backend_kv")


def load_risk_model(path: str) -> dict:
    """Load and sanity-check the frozen calibration JSON for risk_v1."""
    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)
    for key in ("intercept", "coefs", "threshold_high", "threshold_low"):
        if key not in model:
            raise ValueError(f"risk calibration {path} missing field {key!r}")
    unknown = set(model["coefs"]) - set(_RISK_FEATURES)
    if unknown:
        raise ValueError(f"risk calibration has unknown features: {sorted(unknown)}")
    return model


def risk_score(model: dict, candidate: WriteCandidate, s0: Stage0Signals, me: MarginEntropySignals) -> float:
    import math

    feats = {
        "nmargin_med": me.nmargin_med if me.nmargin_med is not None else 0.0,
        "rank_self_med": me.rank_self_med if me.rank_self_med is not None else 6.0,
        "dH_mean": me.dH_mean if me.dH_mean is not None else 0.0,
        "n_eff_norm": me.n_eff_norm if me.n_eff_norm is not None else 1.0,
        "sim_max": s0.sim_max,
        "backend_kv": 1.0 if candidate.backend == "kv" else 0.0,
    }
    z = float(model["intercept"]) + sum(
        float(w) * feats[name] for name, w in model["coefs"].items()
    )
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------------------
# The boundary
# ---------------------------------------------------------------------------


class MemoryMutationAdmissionBoundary:
    """``evaluate(candidate, ...) -> AdmissionDecision`` (plan SS7.1).

    Constructed per session for the NEW policies only; ``legacy_full`` keeps the
    session's original escalation path and never builds a boundary, so legacy
    behavior cannot drift. The cache is read-only during evaluation (provisional
    simulation copies; asserted by the integration tests via snapshot hashing).
    """

    def __init__(self, cfg: GovConfig):
        if cfg.policy == "legacy_full":
            raise ValueError("legacy_full does not use the admission boundary")
        if cfg.policy not in POLICIES:
            raise ValueError(f"unknown GOV_POLICY {cfg.policy!r}; known: {POLICIES}")
        self.cfg = cfg
        self.gate = GeometryGate(cfg)
        self.risk_model = None
        if cfg.policy == "geometry_margin_entropy_risk_v1":
            if not cfg.me_calib:
                raise ValueError(
                    "GOV_POLICY=geometry_margin_entropy_risk_v1 requires GOV_ME_CALIB "
                    "(frozen calibration JSON, plan SS13.3)"
                )
            self.risk_model = load_risk_model(cfg.me_calib)

    def evaluate(
        self,
        mmc: MemoryMutationCandidate,
        cache: GovernanceCache,
        thresholds: ThresholdState,
        whiten_fn,
        user_text: str = "",
        write_candidate: Optional[WriteCandidate] = None,
    ) -> AdmissionDecision:
        candidate = write_candidate or build_candidate(mmc.backend, mmc.raw_call)
        if candidate is None:
            raise ValueError(f"unparseable mutation candidate: {mmc.raw_call!r}")
        t0 = time.perf_counter()
        preflight_ok = preflight_would_succeed(candidate, cache)
        geo: GeometryResult = self.gate.evaluate(
            candidate, cache, thresholds, preflight_ok, whiten_fn
        )
        s0_ms = round((time.perf_counter() - t0) * 1000, 3)
        diagnostics = {
            "s0": {
                "sim_max": round(geo.signals.sim_max, 4),
                "r": round(geo.signals.r, 4),
                "tau_t": round(geo.signals.tau_t, 4),
                "rho": round(geo.signals.rho, 4),
                "n_items": geo.signals.n_items,
                "verbatim_mode": geo.signals.verbatim_mode,
                "verbatim_values": geo.signals.verbatim_values,
                "verbatim_hits": geo.signals.verbatim_hits,
                "verbatim_misses": geo.signals.verbatim_misses,
                "decision": geo.decision.value,
                "reason": geo.reason,
                "fast_path": geo.fast_path,
                "floor_rank": geo.floor_rank,
            },
            "signals": geo.signals,
            "preflight_ok": preflight_ok,
            "escalated": False,
            "latency_ms": {"s0": s0_ms, "s1": 0.0},
            "flagged": False,
        }

        # -- Stage 0 resolves (or expansions, which are Stage-0-only) ----------
        if geo.decision != GovDecision.ESCALATE or mmc.source == "placement_expansion":
            if mmc.source == "placement_expansion" and geo.decision == GovDecision.NOOP:
                action, reason_code = "NOOP", "expansion_noop"
            elif geo.decision == GovDecision.NOOP:
                action, reason_code = "NOOP", geo.reason_code
            else:  # ADD (incl. escalate-on-expansion: never escalate a
                # governance-generated write — circular; plan SS7.3)
                action = "ADD"
                reason_code = (
                    geo.reason_code
                    if geo.decision == GovDecision.ADD
                    else "s0_escalate"
                )
            self._stamp_features_sha(diagnostics)
            return AdmissionDecision(
                action=action,
                stage="GEOMETRY",
                confidence=None,
                reason_code=reason_code,
                rewritten_call=None,
                diagnostics=diagnostics,
            )

        diagnostics["escalated"] = True

        # -- geometry_only: validated fallback-ADD semantics -------------------
        if self.cfg.policy == "geometry_only":
            self._stamp_features_sha(diagnostics)
            return AdmissionDecision(
                action="ADD",
                stage="GEOMETRY",
                confidence=None,
                reason_code="s0_escalate",
                rewritten_call=None,
                diagnostics=diagnostics,
            )

        # -- Stage 1: joint margin-entropy admission ---------------------------
        t1 = time.perf_counter()
        me = compute_margin_entropy_signals(
            candidate, geo.signals, cache, self.cfg, user_text
        )
        if self.cfg.me_shuffle_dh:
            original = {"dH_mean": me.dH_mean, "dH_self": me.dH_self}
            nc_shuffle_dh(me, candidate, geo.signals.n_items, self.cfg.me_shuffle_seed)
            diagnostics["nc_shuffle"] = {"original": original, "seed": self.cfg.me_shuffle_seed}
        confidence = None
        if self.risk_model is not None:
            confidence = risk_score(self.risk_model, candidate, geo.signals, me)
            action, reason_code, rewritten_call, flagged = self._apply_risk_rule(
                confidence, candidate, geo.signals, me, cache, user_text, preflight_ok
            )
        else:
            action, reason_code, rewritten_call, flagged = apply_option_a_rule(
                candidate, geo.signals, me, cache, self.cfg, user_text, preflight_ok
            )
        diagnostics["s1"] = me.to_log()
        diagnostics["latency_ms"]["s1"] = round((time.perf_counter() - t1) * 1000, 3)
        diagnostics["flagged"] = flagged
        self._stamp_features_sha(diagnostics)
        return AdmissionDecision(
            action=action,
            stage="MARGIN_ENTROPY",
            confidence=confidence,
            reason_code=reason_code,
            rewritten_call=rewritten_call,
            diagnostics=diagnostics,
        )

    def _apply_risk_rule(
        self, risk, candidate, s0, me, cache, user_text, preflight_ok
    ) -> tuple:
        """Option B action mapping (plan SS9.4): same action space as Option A;
        the shadowed-duplicate NOOP region is geometric (identical to A) so the
        validated duplicate semantics never depend on the learned model."""
        locate = me.rank_self_med is not None and me.rank_self_med <= self.cfg.me_r
        dup_risk = s0.sim_max >= (self.cfg.sim_high - self.cfg.me_dup_eps)
        me.locate, me.dup_risk = locate, dup_risk
        if not locate and dup_risk:
            if not preflight_ok:
                return "ADD", "s1_shadowed_duplicate_preflight_blocked", None, True
            return "NOOP", "s1_shadowed_duplicate", None, False
        if risk >= float(self.risk_model["threshold_high"]):
            return "ABSTAIN", "s1_high_risk", None, True
        if risk <= float(self.risk_model["threshold_low"]):
            return "ADD", "s1_confident", None, False
        return "ADD", "s1_ambiguous_default", None, True

    @staticmethod
    def _stamp_features_sha(diagnostics: dict) -> None:
        """Hash-stamp the online feature vector at decision time (plan SS13.2
        leakage control): the calibration pipeline asserts features never
        change after the fact and never include question/GT fields."""
        s0 = diagnostics["s0"]
        s1 = diagnostics.get("s1") or {}
        feature_view = {
            "s0": {k: s0.get(k) for k in ("sim_max", "r", "tau_t", "rho", "n_items")},
            "s1": {
                k: s1.get(k)
                for k in (
                    "rank_self_med", "nmargin_med", "dH_self", "dH_mean",
                    "n_eff_norm", "disp", "churn", "smallstore",
                )
            },
        }
        blob = json.dumps(feature_view, sort_keys=True, default=str).encode("utf-8")
        diagnostics["decision_features_sha"] = hashlib.sha256(blob).hexdigest()[:16]
