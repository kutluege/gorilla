# TWO_STAGE_GEOMETRY_MARGIN_ENTROPY_PLAN
## Replacing the Geometry → NLI → Stage-2 cascade with Geometry → joint Margin–Entropy admission

**Status:** planning document — no code is to be written until this plan is reviewed.
**Date:** 2026-07-20.
**Repo root for all paths:** `berkeley-function-call-leaderboard/` unless prefixed `../`.
**Predecessor documents:** `EVAL_RESULTS.md` (2026-07-19 campaign), `../PLAN_3_IMPLEMENTATION_CASCADE_COMPLETION.md`, `gov_logs/plan3_calibration.md`, `../EXPERIMENTS_OVERVIEW.md`.
**Every code fact in this plan was re-verified against the working tree at commit `cd4cb81` during the audit on 2026-07-20** (file:line citations throughout). Every experimental number was taken from the committed artifacts named in §2.

---

## 1. Executive summary

The 2026-07-18/19 campaign (5× A/B + 8-arm × 3-replicate ablations, one vLLM instance,
zero failed phases) settled three things:

1. The full **Geometry → NLI → margin Stage-2** cascade is *accuracy-neutral* on the
   official BFCL score (KV −0.7 pp p=0.80; Vector 0.0 p=1.00).
2. **Geometry is load-bearing**: the only Holm-significant effect in the whole campaign is
   removing it (`geo_off`: KV −10.6 pp, McNemar p=0.0001, Holm p=0.0013, and ~1.8× runtime).
3. The logged-but-inert **entropy signals carry information** the deciding margin ignores
   (≈5.5–6 pp scenario-accuracy separation *within both margin strata* — association, not
   causation).

This plan replaces the cascade with a **two-stage architecture**:

```text
Candidate memory mutation
        |
        v
Stage 0: Geometry  (unchanged decision core, extracted & hardened)
        |
        |-- clearly redundant ----------------> NOOP
        |-- clearly novel and safe ------------> ADD
        `-- geometrically ambiguous ----------> Stage 1
                                                  |
                                   Joint Margin + Entropy decision
                                   (deterministic provisional simulation)
                                                  |
                    +-----------------------------+---------------------------+
                    |                             |                           |
                   ADD                     SAFE_REWRITE                 ABSTAIN/NOOP
                                     (dual-representation only)
```

**NLI is removed from the online decision path entirely** (§1.3 binding). It survives only
as (a) the offline paraphrase-tolerant judge, (b) an optional offline calibration oracle.
The new Stage 1 is deterministic, LLM-free, and — unlike the old Stage 2, which was
*accept-only by construction* (§5.2) — has a real ABSTAIN/NOOP branch, a bounded
SAFE_REWRITE branch that never destroys the raw candidate, and a calibrated joint
margin–entropy rule whose central research question is pre-registered:

> **Does entropy identify harmful writes beyond what geometry and retrieval margin
> already explain?** (tested by ablation + shuffled-entropy negative control, §19)

The minimal thesis-schedule implementation (§23, Phases 0–8) reuses ~80 % of existing,
already-verified machinery (`retrieval_sim.py` is backend-faithful and deterministic;
`probe_gen.py` decision probes are leakage-safe by construction; the Stage-0 core and its
ABTT artifact are untouched) and deletes nothing until historical manifests remain
reproducible under a `GOV_POLICY=legacy_full` selector (§16).

---

## 2. Binding experimental evidence

All artifacts committed on branch `claude/nihai-plan-v2-cascade-thresholds-k34u67`.

| Fact | Value | Artifact |
|---|---|---|
| A/B primary, KV | ΔAcc −0.0067, b/c 34/31, McNemar p=0.8043, CI95 [−0.0505, +0.0396] | `gov_logs/ab5_analysis.json` |
| A/B primary, Vector | ΔAcc 0.0000, b/c 35/35, p=1.0000, CI95 [−0.0366, +0.0430] | same |
| `geo_off` (NLI-first), KV | ΔAcc −0.1057, p=0.0001, **Holm p=0.0013**, CI95 [−0.1778, −0.0377] | `gov_logs/ablations_analysis.json` |
| `geo_off` runtime | 73–80 min/arm-run vs 41–45 governed_full, 33–35 baseline | `result_gov_ablations/manifest.jsonl` (`duration_s`) |
| `p_only`, KV | ΔAcc −0.0792, p=0.0046, Holm 0.0598 (marginal, negative) | `gov_logs/ablations_analysis.json` |
| `stage0_only` / `s2_off` | null (KV −1.5 pp both; Vector +1.1/+1.9 pp; all Holm 1.0) | same |
| G11 entropy separation | margin-confident: acc 0.1097 (dH high, n=90) vs 0.1646 (dH low, n=69); margin-low: 0.1128 vs 0.1717 | `gov_logs/g11_entropy_contribution.json` |
| Stage-2 outcome mix (5 governed reps) | 243 `accept_low_confidence` / 12 `accept` / 3 `accept_rewritten` of 258 | governed `governance_log.jsonl`, re-counted 2026-07-20 |
| Stage-2 margin distribution | min_margin p50=0.0186 > gate 0.01; 38.4 % below gate; **12.0 % exactly 0.0** | same recount |
| Canonicalization efficacy | 59 attempts, **35 (59 %) still below gate after rewrite**, admitted anyway | same recount |
| Stage-1 (NLI) live outcomes | 258/390 `all_neutral`→escalate; **89 of 95 `more_specific` rewrites preflight-blocked → ADD**; net: 20 NOOPs, 15 REWRITEs applied | same recount (`decision <= reason` cross-tab) |
| Chain survival | arm-independent; student dead 24/24 units; healthcare flaky in baseline and governed alike | `gov_logs/ab5_wrra/`, `gov_logs/ablations_wrra/` |
| §8.4 ordering verdict | Vector ρ(sim_max, ΔH)=0.757 CI95 [0.482, 0.899] ≥ 0.40 → geometry-first (binding) | `gov_logs/geometry_dH_verdict.json` |
| Paraphrase judge | A/B governed +11/+17 vs baseline +7/+10 (/775); ablations uniform +3..+9, ordering unchanged | `gov_logs/ab5_paraphrase_judge_summary.json`, `gov_logs/ablations_paraphrase_judge_summary.json` |

Binding design consequences (per the task brief, checked against the artifacts — no
inconsistency found): keep geometry first (§1.2); remove online NLI (§1.3); do not build
around chain revival (§1.6); the official score stays the sole primary metric; entropy is
a *conditional risk signal*, not a standalone classifier (§1.5).

---

## 3. Current code-path audit (verified 2026-07-20)

### 3.1 Interception and commit boundary

- `governance_filter.py:1428 GovernanceSession.govern_calls(calls)` — receives the
  **decoded call-string list of one harness step**, never shrinks it (an empty decoded
  list would short-circuit the harness loop), may grow it via placement expansions
  (`:1444-1447`).
- `governance_filter.py:1238 build_candidate(backend, call)` — parses a call string into a
  `WriteCandidate(op, kind, tier, backend, text, args, ref, raw_call)`. Gated ops are
  exactly `KV_WRITE_OPS` (`:1172` — add/replace on core/archival, params `(key, value)`)
  and `VECTOR_WRITE_OPS` (`:1178` — add `(text,)`, update `(vec_id, new_text)`).
  **Anything else returns `None` and passes through ungoverned**: reads, retrievals,
  removes/clears (observed-only, `:1186-1197`), malformed calls, unbindable args.
- Decision application (`:1478-1505`, only when `not cfg.dry_run`):
  - `NOOP` → call replaced by `DECOY_CALL = "core_memory_retrieve_all()"` (`:1202`),
    `_pending[idx] = {mode:"noop", synthetic}`; the synthetic backend-exact success string
    (`synthetic_success`, `:1291`) is patched over the decoy's real output in
    `patch_results` (`:1829`) before it reaches chat history.
  - `REWRITE` → call string replaced in place; `_pending[idx] = {mode:"rewrite", ...}`.
  - The **real mutation is executed by the harness after** `govern_calls` returns — there
    is *no* path that mutates memory before the decision. The middleware never touches the
    backend store directly.
- Mirror maintenance: `_observe` (`:1910`) parses each executed call + raw result and
  updates the mirror **only on the backend's exact success strings** (`KV_ADD_SUCCESS`
  etc., `:1206-1210`); removes/clears handled by `_observe_kv/_observe_vector/
  _log_observe_removal` (`:1939-1980`).
- Rehydration: `GovernanceCache.rehydrate` (`:439-467`) rebuilds the mirror from the
  backend snapshot (`memory_snapshot/<scenario>_final.json`), re-embeds + whitens, tags
  `turn_written=-1`. **It replays no decisions — rehydration is already idempotent and
  regate-free.** Probe sidecar reload: `_load_probe_sidecar` (`:2024`).

### 3.2 Decision flow (old cascade)

- `_govern_one` (`:1460`): whiten candidate (`_whiten_one`, `:1423`) → `preflight_would_succeed`
  (`:1263`, mirror-based prediction of backend acceptance) → `compute_signals` (`:608`,
  produces `sim_max, r, tau_t, rho, verbatim_*`, `n_items`) → `decide` (`:648`):
  NOOP iff `r < delta AND sim_max > sim_high AND verbatim ok AND preflight_ok`; ADD iff
  `r > tau_t AND verbatim new`; else ESCALATE.
- `_resolve_escalation` (`:1556`): `nli_enabled=0` → ADD `stage0_escalate_fallback`
  (`:1567-1568`). Else `stage1_resolve` (`:780`, DeBERTa NLI verdicts equivalent /
  derivable / contradiction / more_specific / all_neutral) → on `all_neutral`,
  `stage2_resolve` (`:1008`). Shadow discipline `:1600-1612`; live guards `:1614-1637`
  (NOOP requires preflight_ok; REWRITE re-parsed + preflighted, else falls back to ADD —
  the source of the 89 blocked stage-1 rewrites).
- `stage2_resolve` (`:1008-1163`): decision probes from **user text only**
  (`generate_decision_probes`, `probe_gen.py:236` — candidate text is *not a parameter*,
  the anti-circularity invariant); per-tier corpus + provisional candidate
  (`test_probes`, `:1056-1083`); **accept requires every probe top-1 = candidate AND
  min_margin > GOV_S2_MARGIN** (`:1092`); one-shot deterministic canonicalization
  (`canonicalize_candidate`, `:960`) retried on the same probes (`:1096-1128`);
  otherwise **ADD with `low_confidence=True` — Stage 2 never suppresses** (`:1021`,
  `:1131-1134`). `dH_neighbor`/`dH_mean` computed decision-inert (`:1141-1160`).

### 3.3 Simulation and probes (to be reused verbatim)

- `retrieval_sim.py` — backend-faithful, deterministic, pure:
  - KV: `simulate_kv` (`:52`) = `rank_bm25.BM25Plus` over **key names only**, tokenization
    byte-identical to `memory_kv._similarity_search` (`memory_kv.py:71-88`):
    `text.replace('_',' ').lower().split()`. Values are never scored by the real backend.
  - Vector: `simulate_vector` (`:99`) = inner product of L2-normalized MiniLM embeddings
    (`normalize_embeddings=True`, global text→vec cache `:74-96`), exact equivalent of the
    real FAISS `IndexFlatIP` (`memory_vector.py:244-256, 309-333`) at store sizes ≤ 57.
  - `margin` (`:129`) = top1 − top2 raw scores; `entropy` (`:136`) = softmax entropy
    (nats) over top-k=5 scores at temperature (default 1.0; decision-invariance to T
    regression-tested in `test_entropy_fields.py`); `n_eff` (`:149`) = exp(H);
    `delta_h_for_probes` (`:174`) = per-neighbor own-probe H_after − H_before with the
    candidate provisionally appended (list-copy — **the provisional insert is already an
    isolated copy; no store is mutated**).
- `probe_gen.py` — decision probes (`:236-257`): `template:user_identity` (the user turn),
  `template:user_value_question` ("what is {verbatim value}", ≤2),
  `template:user_keywords` (ordered content tokens, ≤8), optional
  `paraphrase:flan-t5-small` (beam, deterministic, `GOV_PROBE_PARAPHRASE`, off by
  default); capped by `GOV_S2_PROBES_N=4` **after** templates. Item probes (`:190-233`):
  KV `template:key` / `template:question`("what is {key}") / `template:key_value`; Vector
  `identity/question/keywords`; cached on `MemoryItem.probes`, persisted to the
  `<scenario>_gov_state.json` sidecar (`:2024, :2051`). **No probe uses benchmark
  questions or ground truth** — leakage-safe by construction.

### 3.4 Non-tool-call write sources (complete inventory)

Verified: the only paths that can put text into the store are (a) governed write ops,
(b) placement expansions (`_placement_pass`/`_apply_expansions`, `:1726, :1812` — the
archive-add half of an atomic move; only live when `GOV_P_ENABLED=1`), (c) Stage-1/2
rewrites (routed through the same `_pending` + preflight machinery). Rehydration creates
mirror entries but never backend entries. **There are no summary writes, no consolidation
writes, and no harness-side writes outside tool execution in the KV/Vector backends.**
Placement expansions currently bypass `_govern_one` (they are constructed post-decision) —
the new admission boundary closes this hole (§7.3).

---

## 4. Failure analysis of the old cascade

Why was the old pipeline inert on the primary metric? Five verified mechanisms:

1. **Stage 2 was accept-only.** All 258 escalations ended in ADD or REWRITE
   (`:1021` "Stage 2 never suppresses"); the stage could flag risk (`low_confidence`) but
   never act on it. A gate that cannot reject cannot prevent harmful writes.
2. **The accept test was mis-specified.** Accept required *every* probe to rank the
   candidate top-1 (`:1092`). The `user_value_question` channel ("what is Goldman Sachs")
   frequently ranks an *existing related memory* top-1 — correctly, since the store
   legitimately contains related facts — so 94 % of escalations "failed" the test while
   their median min_margin (0.0186) was actually above the gate. 12 % of min_margins were
   exactly 0.0 (KV: BM25 over key names scores a natural-language probe near-uniformly;
   Vector: ties). The binary all-top-1 conjunction over a min-aggregation collapses a rich
   score distribution into a nearly-constant signal → 243/258 identical
   `accept_low_confidence` outcomes. This is precisely the aleatoric/epistemic conflation
   of Yadkori et al. (§21): answer multiplicity read as error.
3. **Canonicalization did not repair and may harm.** 59 attempts, 35 still below gate
   after rewrite (admitted anyway); only 3 upgraded to REWRITE. Meanwhile the paraphrase
   judge shows the governed arm's answers are disproportionately *semantically right but
   keyword-wrong* (A/B Vector +17 judged vs baseline +10) — i.e. rewriting stored text
   risks destroying grader keywords for near-zero margin benefit.
4. **NLI mostly punted, and its actions were nullified.** 66 % `all_neutral`; of the 95
   `more_specific` rewrites 89 were preflight-blocked into plain ADD. Net live effect of
   the entire CPU NLI stage across 5 replicates: 20 NOOPs + 15 REWRITEs out of 1,787
   decisions (≈2 %), at 8–12 min extra wall-clock per run — and the `geo_off` arm shows
   what NLI-first costs when geometry is absent (−10.6 pp, 1.8× runtime).
5. **The informative signal was decision-inert.** dH_mean separates ~6 pp of scenario
   accuracy within both margin strata (G11) but was logged and ignored, per the old
   protocol's own rule ("entropy never decides").

Design consequences: the new Stage 1 (i) gets a real ABSTAIN/NOOP branch, (ii) replaces
min-over-probes/all-top-1 with per-channel margins + distribution statistics, (iii) makes
entropy *change* (not level) a first-class input under a pre-registered test, (iv) demotes
canonicalization to a non-destructive dual-representation rewrite, (v) drops online NLI.

---

## 5. Final architectural decision

**Adopted: two-stage Geometry → joint Margin–Entropy admission ("`geometry_margin_entropy_v1`").**

- Stage 0 is the existing `compute_signals` + `decide` core, extracted behind a stable
  interface, thresholds unchanged (SIM_HIGH=0.95, DELTA=0.32, SAGE τ_t) — they survived
  a 24-unit campaign without a single false-NOOP chain kill.
- Stage 1 runs **only** on Stage-0 ESCALATE (18–22 % of decisions in the live logs:
  390/1787 escalated). It is deterministic, uses the existing `retrieval_sim` +
  `probe_gen` machinery, and outputs one of `ADD | NOOP | SAFE_REWRITE | ABSTAIN`.
  ABSTAIN maps at the harness level to ADD-with-flag in v1 (never blocks the chain —
  §8.4) and exists as a distinct logged action so risk-coverage curves can be computed.
- Decision rule: **Option A (interpretable rule gate) first, Option B (small calibrated
  risk model, logistic on ≤6 monotone features) as the pre-registered comparison arm.**
  Option C (constrained Pareto rule) is recorded as future work — with ~250–400
  escalations per campaign, a constrained-optimization formulation cannot be validated
  out-of-sample and adds no interpretability over A. Sample-size arithmetic: 24 chains ×
  ~16 escalations/chain ≈ 390 gated events per 5-replicate campaign → a 6-parameter
  logistic at events-per-variable ≈ 30 is defensible (harvest arms add more, §13); a tree
  ensemble or NN is not.

---

## 6. The fifteen questions, answered

1. **Why geometry first?** It is the only component whose removal is Holm-significantly
   harmful (−10.6 pp KV) and it resolves ~78 % of writes at ~0.5 ms with zero model calls;
   the §8.4 correlational verdict (ρ=0.757) and the `geo_off` causal result agree.
2. **Why remove NLI online?** Net live effect ≈2 % of decisions, 89/95 of its rewrites
   nullified by preflight, 66 % punts, ~1.8× runtime when load-bearing, and the NLI-first
   ablation is the campaign's only significant harm. Offline roles remain.
3. **What was wrong with old Stage 2?** Accept-only; all-top-1 conjunction over
   min-margin collapsed the signal (94 % identical outcomes, 12 % exact-zero margins);
   canonicalization failed to repair (59 % of retries) and risks grader keywords; the
   informative entropy signal was inert. (§4, with counts.)
4. **Which entropy definition?** Softmax entropy over top-5 *retrieval scores* of
   deterministic probes (`retrieval_sim.entropy`, nats), used as **ΔH** (post-insertion
   minus pre-insertion) in two forms: candidate-probe ΔH_self and neighbor-own-probe
   ΔH_neighbor (existing `delta_h_for_probes`), plus n_eff = exp(H) as the store-size-
   comparable form. Generation/semantic entropy is **excluded** from the online path
   (cost; §21 SEP precedent for cheap proxies).
5. **Does entropy add information beyond geometry + margin?** Unknown at decision level —
   G11 shows a scenario-level association (~6 pp) that survives margin stratification but
   is confounded by scenario difficulty. This is *the* research question; §19's nested
   models + shuffled-entropy negative control answer it causally.
6. **Entropy for every write or only ambiguous ones?** Only Stage-0 ESCALATE. Rationale:
   cost (probe simulation ~6 ms vs 0.5 ms geometry), the G11 signal was measured on
   escalations, and gating clear duplicates/novelties on entropy would re-introduce the
   universal-rejection failure mode §6-of-the-brief forbids.
7. **Exact code boundary?** After call decoding and candidate normalization
   (`build_candidate`), before harness execution — i.e. exactly where `govern_calls` sits
   today, formalized as `MemoryMutationAdmissionBoundary.evaluate(candidate, snapshot)`
   (§7). Commit remains harness execution + `_observe` success confirmation.
8. **Tied to tool-call generation?** No. The boundary's input is a
   `MemoryMutationCandidate`, not a call string; tool calls are one *producer* (the only
   live one in this benchmark), placement expansions are the second, and future producers
   register the same way (§7.3).
9. **Non-tool mutations?** Inventory (§3.4): placement expansions get gated through the
   boundary in v1 (Stage 0 only, to avoid recursive escalation); rehydration is replay of
   committed state and is **not regated** (verified idempotent, `:439-467`); an explicit
   offline audit mode (`scripts/audit_store.py`, Phase 5) re-scores historical stores
   without touching them.
10. **KV vs Vector?** Shared architecture, backend-specific signal sets and calibrated
    parameters (§10): KV margins/entropy computed over BM25-on-keys are structurally
    degenerate for natural-language probes → KV Stage 1 leans on canonical-key features +
    key-channel probes; Vector uses the full margin/ΔH set.
11. **Canonicalization?** Constrained to non-destructive dual representation (§11):
    raw text is always stored verbatim; only KV *keys* may be normalized
    (entity-suffixing as today, value untouched); no generative rewriting of values/texts
    in v1. The old value-rewriting canonicalize is retired.
12. **Leakage-free calibration?** Scenario-level splits with chain-grouped
    cross-fitting on *harvest* arms (shadow mode), frozen thresholds before the
    confirmatory campaign; final test replicates never touched during tuning (§13).
13. **Required causal ablations?** §18 matrix (14 arms + shuffled-entropy control +
    nested predictive models).
14. **What justifies "entropy works"?** Pre-registered: geometry+margin+entropy beats
    geometry+margin out-of-sample on harmful-write prediction (ΔAUC > 0, chain-grouped
    CV, p<0.05) AND the live arm with entropy enabled is non-inferior on official score
    and superior on ≥1 pre-registered secondary; shuffled-entropy control must destroy
    the effect (§19.3).
15. **Minimal thesis-schedule implementation?** Phases 0–5 + ablation arms 1/2/3/5/15
    (§18) + one confirmatory 5× A/B ≈ 6–8 weeks of implementation + ~2 GPU days of
    runs; full matrix if time allows (§20, §23).

---

## 7. Trigger and admission-boundary design

### 7.1 New module: `middleware/memory_mutation.py`

```python
@dataclass(frozen=True)
class MemoryMutationCandidate:
    operation: str            # "add" | "replace" | "update"
    backend: str              # "kv" | "vector"
    tier: str                 # "core" | "archival"
    raw_key: str | None       # KV key as emitted by the model
    raw_value: str | None     # KV value / vector text as emitted
    normalized_key: str | None    # v1: sanitized key (existing _sanitize_kv_key)
    normalized_value: str | None  # v1: == raw_value (no value rewriting)
    source: str               # "tool_call" | "placement_expansion" | "audit"
    source_tool: str | None   # original op name, e.g. "core_memory_add"
    raw_call: str             # verbatim call string (for _pending bookkeeping)
    metadata: dict            # step_idx, call_idx, user_text ref, ...

@dataclass(frozen=True)
class AdmissionDecision:
    action: Literal["ADD", "NOOP", "SAFE_REWRITE", "ABSTAIN"]
    stage: Literal["GEOMETRY", "MARGIN_ENTROPY"]
    confidence: float | None      # calibrated risk when policy=B, else None
    reason_code: str              # closed vocabulary, §17
    rewritten_call: str | None    # only for SAFE_REWRITE
    diagnostics: dict             # full signal record, §17

class MemoryMutationAdmissionBoundary:
    def evaluate(self, candidate, store_snapshot) -> AdmissionDecision: ...
```

`store_snapshot` is the existing `GovernanceCache` (read-only during evaluation — the
provisional simulation already copies, `retrieval_sim.py:189`).

### 7.2 Producer wiring (runtime benchmark path)

`GovernanceSession.govern_calls` keeps its harness contract (list in → same-length-or-
longer list out, `_pending` bookkeeping) but becomes a thin producer:
`build_candidate` (existing, `:1238`) → wrap into `MemoryMutationCandidate(source=
"tool_call")` → `boundary.evaluate(...)` → apply exactly as today
(NOOP→decoy+synthetic, SAFE_REWRITE→call replacement, ADD/ABSTAIN→pass through). The
NOOP/decoy/synthetic-success and `patch_results` mechanics (`:1478-1505`, `:1829`) are
**unchanged** — they are the verified commit-boundary implementation.

Non-mutating traffic never reaches the boundary: reads/retrievals/removes/clears and
malformed calls already return `None` from `build_candidate` (§3.1); removes/clears stay
observed-only (destructive ops are the model's own prerogative; the old destructive guard
remains available behind `GOV_P_*`, out of scope for v1's headline arm).

### 7.3 Placement expansions

`_placement_pass` archive-adds (§3.4) are today un-gated. In v1 they are wrapped as
`MemoryMutationCandidate(source="placement_expansion")` and evaluated **Stage 0 only**
(escalation of a governance-generated write would be circular). Acceptance: an expansion
that Stage 0 calls duplicate is dropped and logged (`reason_code=expansion_noop`).

### 7.4 Rehydration and replay (decision)

**Default: no regating.** `rehydrate` (`:439-467`) rebuilds middleware state from
committed history; re-deciding there would make replay non-idempotent and desynchronize
the mirror from the real store (which already contains the entries). Decision verified
against code: rehydration constructs `MemoryItem`s directly, never routes through
`govern_calls`. For historical re-evaluation an explicit **offline audit mode** is added
(Phase 5: `scripts/audit_store.py --snapshot <file> --policy <name>` → report only, no
mutation). This matches the brief's preferred default.

---

## 8. Stage-0 specification (geometry, preserved + hardened)

### 8.1 Preserved (verbatim from audit)

- Embedding: MiniLM (shared encoder singleton) → ABTT whiten
  (`middleware/artifacts/abtt_minilm_l6_d16.npz`, loader `:299`).
- Signals (`compute_signals`, `:608`): `sim_max` (max whitened cosine vs tier items),
  `r` (residual norm vs span basis, `basis_q` `:416`), `tau_t` (SAGE-adaptive novelty
  threshold, `ThresholdState` `:475-`), `rho` (density), verbatim value extraction +
  presence (`:531-558`), `n_items`.
- Decisions (`decide`, `:648`): NOOP / ADD / ESCALATE exactly as today, incl. the
  degeneracy guard `GovConfig.validate()` (`:224`, NOOP region empty iff
  `delta < sqrt(1−sim_high²)`) and the preflight rule that a NOOP may never mask a real
  backend error (`:1263`, `:1615`).
- KV canonical-key behavior: KV NOOPs are effectively preflight-driven (duplicate key ⇒
  `preflight_ok=False` ⇒ never NOOP, real error surfaces) — confirmed by zero decided KV
  NOOPs across all harvests; retained.

### 8.2 Narrow corrections (evidence-scoped)

1. **Exact-duplicate fast path** (new, before embedding): normalized byte-equality
   (`_normalize_for_match`, `:525`) of (key, value)/text vs tier items → NOOP without
   encoding. Cheap, removes the only observed false-negative class of geometry (identical
   re-writes at sim 0.99 <
   sim_high because of ABTT noise).
2. **"Clearly novel and safe" is NOT just low similarity.** ADD additionally requires
   (a) verbatim values of the user turn present in the candidate (existing
   `verbatim_misses == []` logic), (b) candidate text length within backend limit
   (existing preflight), (c) **retrievability floor**: the candidate's own key/keyword
   probe must rank it top-3 in the provisional store (one `simulate_*` call, ~1 ms).
   A "novel" write that its own probe cannot find is malformed/off-topic → ESCALATE, not
   ADD. This is the §4-of-the-brief guard against low-sim ≠ safe.
3. **Ambiguity band calibration**: the ESCALATE band boundaries stay (SIM_HIGH, DELTA,
   τ_t) but are re-estimated out-of-sample per §13 (harvest arms), not from test outcomes.

Everything else in Stage 0 is frozen — it is the causally validated component.

---

## 9. Stage-1 specification: joint margin + entropy admission

Runs only on ESCALATE. All computation uses `retrieval_sim` (per-tier, backend-faithful,
provisional-copy isolation) and `probe_gen` decision + item probes. No LLM, no NLI, no
sampling; fully deterministic given the store snapshot.

### 9.1 Signals (per candidate; mathematical definitions)

Let `S_p = [s₁ ≥ s₂ ≥ …]` be top-5 scores of probe `p` against the provisional corpus
(tier items + candidate), and `S_p⁻` against the pre-insertion corpus.

| Signal | Definition | Notes |
|---|---|---|
| `rank_self(p)` | rank of the candidate under probe `p` | replaces the brittle binary all-top-1 |
| `margin(p)` | `s₁ − s₂` | existing `retrieval_sim.margin` |
| `nmargin(p)` | `(s₁ − s₂) / (s₁ + ε)` for KV (BM25 scale-free-ness); raw margin for Vector (cosine already bounded) | normalized margin per brief §5 |
| `H(p)`, `H⁻(p)` | softmax entropy (nats, T=1, k=5) post/pre insertion | existing `entropy` |
| `ΔH_self` | mean over **candidate-channel** probes of `H(p) − H⁻(p)` | new: candidate's effect on its own retrieval neighborhood |
| `ΔH_neighbor` | existing `delta_h_for_probes` mean over neighbors' cached own probes | interference measure |
| `n_eff` | `exp(H)` | store-size-comparable |
| `n_eff_norm` | `n_eff / min(k, n_items+1)` | store-size normalization (brief §6); in [1/k, 1] |
| `disp(p)` | std of top-5 scores / mean top-5 (NQC-style dispersion) | QPP-pedigree alternative to H; logged v1, ablation 13/14 decides |
| `churn` | Jaccard distance of pre/post top-3 sets over neighbor probes | rank-displacement summary |
| Stage-0 carry-over | `sim_max, r, tau_t, rho, n_items, verbatim_hits/misses` | conditioning features |

Channel structure (replaces min-over-all): probes are grouped
`identity/keywords` (candidate-locating) vs `value_question` (interference-probing);
`rank_self`/`margin` are aggregated **within channel** (median), never min-pooled across
channels. The `value_question` channel contributes only to interference features
(`ΔH_neighbor`, `churn`), not to the candidate-locating test — fixing failure §4.2.

### 9.2 Decision rule, Option A (v1 default `GOV_POLICY=geometry_margin_entropy_v1`)

```text
locate  := median rank_self over candidate-channel probes ≤ R      (default R=2)
strong  := median nmargin over candidate-channel probes ≥ M_b      (backend-calibrated)
interf  := ΔH_neighbor ≥ D_b  OR  churn ≥ C_b                      (backend-calibrated)
dup_risk:= sim_max ≥ sim_high − ε_dup                              (near-NOOP band)

if locate and strong and not interf:        ADD          (reason=s1_confident)
elif locate and strong and interf:          SAFE_REWRITE if a non-destructive rewrite
                                            exists (KV key-suffix only), else
                                            ADD_WITH_FLAG (reason=s1_bounded_interference)
elif not locate and dup_risk:               NOOP         (reason=s1_shadowed_duplicate;
                                            requires preflight_ok, same guard as Stage 0)
elif not locate and not strong and interf:  ABSTAIN      (reason=s1_high_risk)
else:                                       ADD_WITH_FLAG (reason=s1_ambiguous_default)
```

Design invariants: (i) the entropy/interference term can **only** move a decision between
ADD ↔ SAFE_REWRITE/ABSTAIN when margin/locate is already indeterminate or the candidate
shadows an existing memory — it is *conditional*, never a standalone rejector (brief §6);
(ii) NOOP requires the same preflight guard as Stage 0; (iii) the default on total
ambiguity is a flagged ADD, not suppression — write-parsimony must be earned by
calibration, not assumed.

### 9.3 ABSTAIN semantics in v1

The BFCL harness must receive an executable call list. `ABSTAIN` therefore applies as:
execute the write, tag the stored item `abstained=True` in the mirror + log
(`_low_conf_texts` mechanism generalized to `_flagged_texts`). ABSTAIN is thus v1-
observational — it exists to measure the risk-coverage curve (Geifman–El-Yaniv) that
justifies (or kills) a v2 where ABSTAIN suppresses. This is the honest consequence of the
campaign's null: we have no evidence yet that *any* suppression improves the score, so v1
must measure before it blocks. (`NOOP` in the shadowed-duplicate branch *does* suppress —
that region reproduces Stage 0's validated duplicate semantics at slightly wider band.)

### 9.4 Decision rule, Option B (pre-registered comparison arm)

Logistic regression `P(harmful | features)` on ≤6 features:
`{nmargin_med, rank_self_med, ΔH_neighbor, n_eff_norm, sim_max, backend}` with
monotonicity enforced by sign constraints (harm ↑ as margin ↓, rank ↑, ΔH ↑). Trained on
harvest labels (§13.3), isotonic-calibrated, threshold by Chow risk-ratio on the
calibration split. Ships behind `GOV_POLICY=geometry_margin_entropy_risk_v1`; identical
action space.

---

## 10. Backend-specific behavior

Shared architecture; per-backend signal subsets and calibrated constants
(`M_kv, D_kv, C_kv` vs `M_vec, D_vec, C_vec` — no shared thresholds unless ablation 9/10
shows equivalence).

**KV.** Real retrieval is BM25Plus **over key names only** (`memory_kv.py:71-88`) —
values are invisible to search. Consequences (all verified in data): natural-language
probes score near-uniformly (12 % exact-zero margins); H over BM25 scores is unstable at
n_items < 5 (softmax of near-identical scores ≈ uniform ⇒ H ≈ log k regardless of
content); harvest KV read-margins were 100 % degenerate-zero. Therefore KV Stage 1 uses:
key-channel probes only for locate/strong (`template:key`, `template:key_value`),
`nmargin` (scale-free), canonical-key features (existing `_sanitize_kv_key`, exact-key
collision, key-token Jaccard vs existing keys), and treats entropy features as **inactive
below `n_items = 5`** (`reason_code` suffix `_smallstore`). Value conflict on same
canonical key (same key, contradicting verbatim values) → SAFE_REWRITE via key-suffix
(today's entity-suffix mechanic, `canonicalize_candidate` `:960`, retained for keys only).

**Vector.** Real retrieval is exact IP over normalized MiniLM (`memory_vector.py:
244-256, 309-333`); simulation is exact at benchmark store sizes. Full signal set active.
Near-duplicate paraphrases are the main duplicate class (all 9 historical true NOOPs were
Vector, sim 0.95–1.0); hubness risk (a centroid-near candidate crowding top-k lists) is
exactly what `ΔH_neighbor`/`churn` measure — report k-occurrence skew as a diagnostic
(§21 Radovanović). ABTT stays Stage-0-only; simulation stays raw-MiniLM (fidelity
contract, `retrieval_sim.py:16-22`).

---

## 11. Canonicalization decision

**v1 policy: no generative canonicalization; no value/text rewriting; dual
representation.**

- Evidence: 59 % of canonicalize retries failed to clear the gate; judge asymmetry
  suggests keyword damage; only 3/1787 decisions ever benefited.
- Adopted (options 3+4+5 of the brief, merged):
  - `raw_value`/`raw_text` is **always** what the backend stores (grader sees model's own
    words — keyword-safe by construction).
  - `normalized_*` lives only in the middleware mirror + log for retrieval-side analysis
    (`MemoryItem.normalized_text`, computed by deterministic rules: lowercase, underscore
    split, entity extraction — no model).
  - KV SAFE_REWRITE may alter **keys only** (suffixing with a verbatim entity token —
    deterministic, preserves the value byte-identically).
  - Vector: no rewrites at all in v1 (option 6 — indexing over normalized text while
    returning raw — requires touching the real backend index and is deferred; recorded as
    v2 candidate).
- The old value-rewriting `canonicalize_candidate` path is retired from live policies and
  kept importable for `GOV_POLICY=legacy_full` replay only.

---

## 12. (folded into §9.1) Mathematical definitions
All formulas are stated in §9.1 against their implementing functions; the only *new*
math is `nmargin`, `n_eff_norm`, `disp`, `churn`, `ΔH_self` — each a pure function over
existing simulation outputs, to live in `middleware/entropy_metrics.py` with unit tests
per §17.

---

## 13. Data and calibration protocol (leakage-safe)

### 13.1 Units and splits

- **Calibration unit:** one governed write decision, grouped by **chain**
  (scenario × backend × replicate); all decisions of a chain stay in the same fold
  (write decisions within a chain are correlated through the shared store).
- **Split design:** leave-one-scenario-out CV over the 4 live scenarios
  (customer/finance/healthcare/notetaker; student is pre-registered dead), stratified by
  backend. Thresholds/model are chosen on CV folds of **harvest data only**, frozen in
  a committed `gov_logs/margin_entropy_calibration.json` (values + git hash + data
  manifest), and never revisited after the confirmatory campaign starts.
- **Data source for calibration:** new shadow harvests (Phase 3): full-signal logging
  arms (`GOV_DRY_RUN=1`) over ≥3 replicates — reusing the existing harvest machinery
  (`run_gov_replicates.py --arms-json`, `result_gov_harvest/` pattern). The 2026-07-18/19
  campaign logs (5 governed + 24 ablation govlogs, ~2.4 k decisions with full stage2
  blocks on 258+ escalations) serve as the *exploratory* set — allowed for feature
  development, **excluded** from threshold freezing (they overlap the reported test
  outcomes).

### 13.2 Labels (delayed credit assignment)

Per write decision, computed **post-hoc only** (§17 separates these fields):

1. `own_probe_fail`: candidate's item probes fail top-3 in the *final* store snapshot.
2. `neighbor_probe_fail`: any neighbor whose own probes were top-1 pre-insertion loses
   top-1 in the final snapshot (attributed to the write via replay ordering).
3. `harmful_write` (primary label): the chain survived AND ≥1 question whose
   scenario-matched retrieval touched the candidate's neighborhood flipped
   correct→incorrect relative to the paired baseline replicate (survival-conditional
   pairing reused from `analyze_gov_replicates.py`).
4. `useless_duplicate`: admitted candidate whose normalized text duplicates an item and
   whose removal in offline replay changes no probe ranking.
5. `latency_ms` per stage.

Leakage controls: labels use benchmark questions/answers **only offline, after the run**;
the online feature vector is hash-stamped into the log at decision time
(`decision_features_sha`), and the calibration pipeline asserts it never reads
question/GT fields for feature construction (unit-tested, §17). Probes remain built from
user text only (existing invariant, `probe_gen.py:236` — candidate text is not even a
parameter; ground truth appears nowhere in `probe_gen.py`, re-verified).

### 13.3 Threshold freezing ceremony

`M_b, D_b, C_b, R, ε_dup` (+ Option-B coefficients) chosen by risk-coverage on CV-harvest
(Geifman–El-Yaniv SGR at pre-declared risk level), committed with the calibration JSON +
a `CALIBRATION_FROZEN.md` note **before** the first confirmatory replicate is launched;
the confirmatory manifest records the calibration file's git hash.

---

## 14. File-by-file implementation plan

Each step: objective / files / functions / behavior / tests / logs / acceptance /
dependencies / risks / rollback. Rollback for every step is `GOV_POLICY=legacy_full`
(§16) plus git revert of the step's commit; steps are individually committable.

**Step 1 — `middleware/memory_mutation.py` (new).**
Objective: §7.1 dataclasses + boundary shell. Inspect: `governance_filter.py:580-607`
(WriteCandidate), `:1238-1260`. Modify: none (new file). Behavior: pure types + a
`from_write_candidate()` adapter. Tests: construction, adapter round-trip, frozen-ness.
Logs: none. Acceptance: mypy-clean, imported by nothing yet. Risk: none.

**Step 2 — `middleware/geometry_gate.py` (extraction).**
Objective: move Stage-0 pure logic behind `GeometryGate.evaluate(candidate, cache) ->
(decision, signals)`. Inspect/modify: `governance_filter.py` — relocate
`compute_signals` (`:608`), `decide` (`:648`), keep re-exports for old imports.
Add §8.2 fast-path + retrievability floor. Tests: existing `test_gov_offline.py` (23)
must pass unmodified via re-exports; new tests: exact-dup fast path, floor triggers
ESCALATE on unfindable candidate. Logs: unchanged Stage-0 fields + `reason_code=
s0_exact_dup|s0_unretrievable_escalate`. Acceptance: byte-identical decisions on a replay
of the 5 governed A/B logs (Step 12 replay harness) except the two new reason codes.
Dependencies: Step 1. Risk: silent behavior drift → mitigated by replay diff.

**Step 3 — `middleware/entropy_metrics.py` (new).**
Objective: `nmargin, n_eff_norm, disp, churn, delta_h_self` as pure functions over
`retrieval_sim` outputs. Inspect: `retrieval_sim.py:129-213`. Tests (§17 unit list):
ties, zero-margin, empty/singleton store, max/min entropy, monotonicity of n_eff_norm.
Acceptance: property-based tests green; no imports from BFCL harness. Dependencies: none.

**Step 4 — `middleware/admission_policy.py` (new).**
Objective: `MemoryMutationAdmissionBoundary` + Option-A rule (`§9.2`) + policy registry
`{legacy_full, geometry_only, geometry_margin_entropy_v1, geometry_margin_entropy_risk_v1}`.
Inspect: `_resolve_escalation` (`:1556-1645`) for the shadow/preflight guard patterns to
reuse. Behavior: v1 rule; `legacy_full` delegates to the existing `stage1_resolve`/
`stage2_resolve` untouched; `geometry_only` returns Stage-0 fallback ADD. Tests: rule
truth-table (16 combinations of locate/strong/interf/dup_risk), preflight-guard on NOOP,
ABSTAIN-tags-not-blocks. Logs: §17 schema. Acceptance: truth-table matches §9.2 spec
verbatim. Dependencies: Steps 1–3.

**Step 5 — wire `GovernanceSession` to the boundary.**
Objective: `govern_calls`/`_govern_one` route through `boundary.evaluate`; placement
expansions gated (§7.3). Modify: `governance_filter.py:1428-1505, 1726-1827`. Preserve:
`_pending`, decoy/synthetic, `patch_results`, `_observe`, shadow discipline (`GOV_ME_SHADOW`
new flag mirrors the old NLI/S2 shadow contract: compute+log, never intervene).
Tests: integration list §17; existing 468-check suite green with `GOV_POLICY=legacy_full`.
Acceptance: A/B smoke (Plan-2 Step-0 pattern) passes on both arms with v1 in shadow.
Risks: harness-contract regression → the never-shrink invariant is asserted in tests.

**Step 6 — config + logging.**
Objective: `GovConfig` additions (`GOV_POLICY`, `GOV_ME_*` per §16; read in `from_env`
`:170`), log schema v2 (§17) written by `log_gov_record` (`:1310`). Tests: env parsing,
schema-version stamping, old-parser compatibility (analyzer reads v1 logs unchanged).
Acceptance: `parse_wrra.py`, `analyze_gov_replicates.py`, `correlate_geometry_deltaH.py`
run unmodified on new logs. 

**Step 7 — replay harness (`scripts/replay_admission.py`, new).**
Objective: re-run any committed governance log's decisions through any policy
(reconstructing store state from `candidate_text` + observe events, as
`replay_geometry_deltaH.py` already does — inspect it first). Tests: §17 replay list.
Acceptance: `legacy_full` replay reproduces logged decisions bit-identically on the 5
governed A/B logs; determinism: two runs byte-identical. Dependencies: Steps 2–4.

**Step 8 — calibration pipeline (`scripts/calibrate_margin_entropy.py`, new).**
Objective: §13 splits, labels, risk-coverage threshold selection, Option-B fit,
`margin_entropy_calibration.json` + frozen note. Inspect: `analyze_gov_replicates.py`
(pairing), `parse_wrra.py` (chain/scenario mapping). Tests: no-GT-in-features assertion;
split integrity (no chain straddles folds); reproducible with fixed seed. Acceptance:
report with CV risk-coverage curves; thresholds committed.

**Step 9 — audit mode (`scripts/audit_store.py`, new).**
Objective: §7.4 offline re-scoring of a snapshot; report-only. Small; tests: never writes.

**Step 10 — runner/analyzer arms.** No code change expected (`run_gov_replicates.py` is
N-arm via `--arms-json` with per-arm `gov_env` — verified); ablation arms of §18 are
pure env configurations. Acceptance: `--dry-run` prints the §18 matrix correctly.

**Step 11 — docs/tests cleanup.** Update `EXPERIMENTS_OVERVIEW.md`, add
`test_admission_policy.py` to the suite list, mark NLI stage "offline-only" in module
docstrings. Old suites remain green throughout (binding constraint).

---

## 15. (folded into §14 Step 6 and §16) Configuration
## 16. Configuration and migration plan

```text
GOV_POLICY = legacy_full | geometry_only | geometry_margin_entropy_v1 |
             geometry_margin_entropy_risk_v1        (default: legacy_full until Phase 7)
GOV_ME_SHADOW   = 1|0    shadow-first discipline for the new stage (default 1)
GOV_ME_R        rank threshold R          (default 2)
GOV_ME_MARGIN_KV / GOV_ME_MARGIN_VEC     M_b   (from calibration JSON)
GOV_ME_DH_KV / GOV_ME_DH_VEC             D_b
GOV_ME_CHURN_KV / GOV_ME_CHURN_VEC       C_b
GOV_ME_DUP_EPS  ε_dup                    (default 0.02)
GOV_ME_CALIB    path to frozen calibration JSON (risk_v1 arm)
```

Migration rules:
- Old env vars keep working: `GOV_POLICY=legacy_full` reads `GOV_NLI_*`/`GOV_S2_*`
  exactly as today; new policies **ignore** `GOV_NLI_*` and assert at startup that no NLI
  model gets loaded (test: `no_online_nli`).
- No file deletions until Phase 8: `stage1_resolve`, `stage2_resolve`,
  `canonicalize_candidate` remain importable for legacy replay; historical manifests
  (`result_gov_replicates/manifest.jsonl`, `result_gov_ablations/manifest.jsonl`) must
  replay under `legacy_full` (Phase 0 acceptance).
- Manifest versioning: `run_start` gains `gov_policy` + `calibration_sha`; analyzer
  treats absence as `legacy_full`.
- Log schema versioning: every record gains `schema: "gov2"`; v1 parsers ignore unknown
  fields (JSONL append-only, verified analyzer-compatible).

---

## 17. Logging schema and test plan

### 17.1 Decision record (schema `gov2`; one JSONL line per candidate)

Online-decision fields (written at decision time, immutable):
`schema, run_id, replicate, scenario, chain_id (backend/scenario), test_id, step_idx,
call_idx, candidate_id (sha of raw_call+step), source, source_tool, backend, tier,
operation, raw_key, raw_value_sha (full text kept as today for replay: candidate_text),
normalized_key, store_size {core, archival}, s0: {sim_max, r, tau_t, rho, verbatim_*,
n_items, decision, reason}, escalated: bool, s1: {per_probe: [{probe, channel, rank_self,
margin, nmargin, H_pre, H_post}], dH_self, dH_neighbor: [...], dH_mean, n_eff_norm, disp,
churn, locate, strong, interf, dup_risk}, policy: {name, version, calibration_sha,
thresholds}, risk (Option B), action, stage, reason_code, rewritten_call,
latency_ms: {s0, s1, total}, decision_features_sha`.

Post-hoc outcome fields (separate file `outcomes.jsonl`, joined on `candidate_id` by the
calibration pipeline — **never read at runtime**): `own_probe_fail, neighbor_probe_fail,
harmful_write, useless_duplicate, question_flips: [...]`.

Reason-code vocabulary (closed): `s0_exact_dup, s0_noop, s0_confident_add,
s0_unretrievable_escalate, s0_escalate, s1_confident, s1_bounded_interference,
s1_shadowed_duplicate, s1_high_risk, s1_ambiguous_default, expansion_noop,
*_preflight_blocked, *_smallstore`.

### 17.2 Tests (minimum set; names indicative)

Unit (`test_admission_policy.py`, `test_entropy_metrics.py`): exact duplicate; near
duplicate (sim 0.96); clearly novel; ambiguous band; empty store; singleton store; tied
scores (margin 0, stable-sort rank); zero-margin; max-entropy (uniform scores) and
min-entropy (one dominant) distributions; ΔH sign both directions; KV conflicting value
same key → SAFE_REWRITE key-suffix, value byte-identical; vector paraphrase near-dup;
malformed candidate → boundary never called; read-only call bypass; rehydration
idempotence (rehydrate twice ⇒ identical mirror, zero decisions logged); small-store
entropy deactivation (n_items<5 ⇒ interf inactive, reason suffix); ABSTAIN tags, does
not block; NOOP requires preflight_ok.

Integration (`test_admission_integration.py`): tool call → candidate → decision → commit
(mirror updated only after observed success string); rejected candidate leaves store +
mirror untouched; provisional simulation leaves no residue (snapshot hash pre/post);
deterministic rerun ⇒ byte-identical decision records; raw/normalized linkage; placement
expansion gated Stage-0-only; historical rehydration does not regate.

Replay (`test_replay_admission.py`): the 5 governed A/B logs replayed under
`legacy_full` reproduce logged decisions exactly; under `geometry_only` reproduce Stage-0
fields exactly; under `geometry_margin_entropy_v1` produce valid gov2 records with
identical Stage-0 signals; determinism across two runs.

End-to-end (Plan-2 Step-0 smoke pattern, then full): score files exist for every unit;
exit 0 everywhere; per-(replicate, arm) `GOV_LOG_DIR` isolation; **assert no NLI weights
loaded in new-policy arms** (import hook counter); WRRA shows no new chain-death mode
(dead set ⊆ {student, healthcare} as in all 24 historical units).

---

## 18. Ablation matrix (Phase 6)

Arms (all 3 replicates, `--arms-json`, baseline always included; numbering = brief §11):

| # | Arm | Config |
|---|---|---|
| 1 | baseline | GOV off |
| 2 | geometry_only | `GOV_POLICY=geometry_only` |
| 3 | geo + old margin logic | `legacy_full` with `GOV_NLI_ENABLED=0, GOV_S2_ENABLED=1` (≙ 2026 s2-only path) |
| 4 | geo + entropy-only rule | v1 with margin terms disabled (`M_b=−∞`) — entropy/interference decides alone (diagnostic arm; expected to underperform, tests brief §6) |
| 5 | geo + joint rule (Option A) | `geometry_margin_entropy_v1` |
| 6 | geo + risk model (Option B) | `geometry_margin_entropy_risk_v1` |
| 7 | joint, no rewrites | v1 with SAFE_REWRITE disabled |
| 8 | joint + dual-representation rewrite | v1 default |
| 9/10 | shared vs backend-specific thresholds | v1 with `M_kv=M_vec` etc. vs calibrated |
| 11/12 | candidate-own probes only vs +neighbor-own | v1 with ΔH_neighbor masked vs full |
| 13/14 | entropy without/with store-size normalization | `n_eff` vs `n_eff_norm` in `interf` |
| NC | **shuffled-entropy negative control** | v1, ΔH values permuted within (backend, store-size bin) at decision time, fixed seed |

Conditional predictive tests (offline, on harvest labels, chain-grouped CV): nested
models `geometry` ⊂ `geometry+margin` ⊂ `geometry+entropy` ⊂ `geometry+margin+entropy`;
report ΔAUC/ΔBrier per nesting step with grouped-bootstrap CIs. Entropy's claim lives or
dies on `geometry+margin+entropy` vs `geometry+margin` **and** the NC arm nullifying.

Priority order under time pressure: 1, 2, 5, 3, NC, 6, 11/12 — the rest as budget allows.

---

## 19. Statistical evaluation protocol

- Primary: official unmodified BFCL score; survival-conditional pairing; exact McNemar;
  scenario-level bootstrap CI (n=2000, seed 12345); Holm over backend × arm-pair;
  N=5 sequential replicates for the confirmatory A/B (v1 vs baseline vs geometry_only —
  pre-registered primary comparison: **v1 vs geometry_only**), 3 for ablations; student
  excluded; manifests pin config + calibration sha; same-server-instance rule +
  `RESUME_PROTOCOL.md` apply; seed statement verbatim ("N=5 replicates, sequential, no
  fixed seed; run-to-run variation stems from server-side nondeterminism at temperature
  0.001; replicate ≠ seed").
- Secondary/diagnostic: paraphrase judge; chains/W-R-R-A; duplicate-admission rate;
  harmful-write rate (§13.2 labels); own-probe success; neighbor interference; entropy
  calibration (reliability diagram of Option-B risk); latency/stage; **online NLI calls
  (must be 0)**; % resolved at Stage 0; % escalated.
- Success criteria (pre-declared):
  - *Minimum engineering:* official ΔAcc CI within a non-inferiority margin of −2 pp per
    backend vs baseline; dead-scenario set ⊆ {student, healthcare}; runtime <
    governed_full's 44 min/arm-run; zero online NLI; bit-identical replay.
  - *Scientific:* arm 5 ≥ arm 3 (joint vs margin-only) on harmful-write rate with
    non-inferior official score; nested-model ΔAUC(entropy | geometry+margin) > 0 at
    p<0.05 grouped; NC arm shows no such gain.
  - *Strong thesis:* Holm-corrected official-score improvement of arm 5 or 6 over
    baseline or geometry_only in ≥1 backend, replicated direction in the other, with NC
    and ablations 11–14 attributing the gain to the entropy term.
- No claim from pooled raw counts or single replicates (binding).

---

## 20. Runtime and compute estimates

Per arm-run (384 entries, measured 2026-07 campaign): baseline 33–35 min; Stage-0-only
≈35 min; old full cascade 41–53 min (NLI-dominated); **v1 projected ≈36–38 min**
(Stage-1 cost ≈ old stage2_resolve ≈ 6 ms × ~80 escalations ≈ negligible; NLI removed).
Confirmatory 5× A/B (3 arms × 5 reps) ≈ 9–10 h; §18 matrix (15 arms × 3 reps, most
~36 min) ≈ 28–30 h; harvests ≈ 4 h; total ≈ 2 GPU-days on the tunnel machine.
CPU-side calibration/replay: < 2 h, offline. Fits one further 48 h window plus slack.

---

## 21. Research grounding (what transfers, what does not)

Full annotated review in the 2026-07-20 literature survey (agent report, to be committed
as `docs/lit_review_write_gating.md` in Phase 0). Load-bearing citations:

- **Write-gating gap:** all verified uncertainty-gated RAG systems gate *read* — FLARE
  (arXiv:2305.06983, token-prob threshold), DRAGIN (2403.10081, entropy×attention),
  Self-RAG (2310.11511, learned reflection tokens), Adaptive-RAG (2403.14403, external
  complexity classifier), SeaKR (2406.19215, Gram-determinant of hidden states across 20
  samples). None gates *write*; existing write policies are per-write LLM judgments —
  Mem0 (2504.19413, ADD/UPDATE/DELETE/NOOP judge; we adopt its action taxonomy,
  deterministically), A-Mem (2502.12110, neighbor-retrieval-at-write skeleton),
  Generative Agents (2304.03442, LLM importance at write). SeaKR's
  "insert-candidate-measure-uncertainty-change" is the nearest conceptual ancestor of our
  provisional ΔH — ours is deterministic and retrieval-side. **The deterministic,
  calibrated, LLM-free write gate is the thesis's novelty claim** (one more targeted
  search before the defense is pre-registered in Phase 8).
- **Entropy caution:** Yadkori et al. (2406.02543) — entropy conflates aleatoric
  multiplicity with epistemic error; margins disambiguate → the joint rule's principled
  defense, and exactly the observed `user_value_question` failure (§4.2). Kuhn/Farquhar
  semantic entropy (2302.09664; Nature 2024) motivates meaning-level uncertainty but its
  sampling+NLI recipe is what our ablation rejected; SEP (2406.15927) legitimizes cheap
  deterministic proxies; AdaRAGUE (2501.12835) shows simple uncertainty ≥ complex
  pipelines at scale.
- **Score-distribution statistics pedigree:** NQC (Shtok et al., TOIS 2012) and
  successors — our `disp`/entropy features are QPP statistics applied at write time;
  borrow their background-score normalization (our `n_eff_norm`).
- **Abstention:** Chow (1970) cost-ratio thresholds; Geifman–El-Yaniv (NeurIPS 2017)
  risk-coverage/SGR → §13.3 threshold ceremony; Kamath et al. (ACL 2020) → calibrate
  across category shift (our backend/scenario strata).
- **Interference:** hubness (Radovanović JMLR 2010) mechanizes how a write degrades
  neighbors; embedding-capacity limits (2508.21038) and more-documents harm (2503.04388)
  justify write-parsimony as such.
- **ABTT:** Mu & Viswanath (ICLR 2018) — cite primary for the Stage-0 whitening.

Nothing is copied wholesale from read-time RAG; every borrowed element is re-derived on
deterministic retrieval-side quantities (the brief's §17 constraint).

---

## 22. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Entropy proves non-causal (NC arm nullifies) | Pre-registered honest outcome: v1 collapses to geometry+margin (still a publishable negative + the simplification/runtime win); thesis claims scoped in §24 |
| New rule suppresses a needed write (chain death) | v1 suppresses only in the shadowed-duplicate region (Stage-0-validated semantics); WRRA new-death-mode gate in §19; shadow-first rollout |
| Small calibration sample (KV escalations are few) | Backend-stratified harvests sized in Phase 3 (target ≥150 escalations/backend); Option B capped at 6 monotone features; fall back to Option A if EPV < 20 |
| Replay drift vs old logs | Step-2/7 bit-identity acceptance on committed logs before any live run |
| KV entropy degenerate at small stores | explicit `_smallstore` deactivation + ablations 13/14 |
| Harness contract regression (list shrink, synthetic results) | invariants unit-tested; `legacy_full` escape hatch; never-delete rule §16 |
| Server nondeterminism swamps effects again | paired design + pre-registered primary comparison (v1 vs geometry_only) maximizes power; non-inferiority margin declared up front |
| 529-style infra flakiness in long runs | existing detached-launcher + monitor + `RESUME_PROTOCOL.md` |

---

## 23. Ordered implementation checklist

**Phase 0 — audit and reproduction (≈3 days)**
- [ ] Commit this plan + `docs/lit_review_write_gating.md`.
- [ ] Replay guard: run existing suites (16 suites / 468 checks) green at `cd4cb81`.
- [ ] `legacy_full` replay of the 5 governed A/B logs reproduces decisions (Step 7 harness prototype on current code paths).
- [ ] Freeze exploratory dataset list (which logs may inform features, §13.1).

**Phase 1 — centralized mutation boundary (≈4 days)**
- [ ] Steps 1, 4-shell, 5: types, boundary, `govern_calls` wiring, placement gating.
- [ ] Integration tests green; smoke run with `GOV_POLICY=legacy_full` byte-compatible.

**Phase 2 — geometry extraction and stabilization (≈3 days)**
- [ ] Step 2: `geometry_gate.py`, fast path, retrievability floor; replay bit-identity.

**Phase 3 — offline margin–entropy analysis (≈1 week, needs tunnel)**
- [ ] Step 3 metrics module; Step 6 logging schema.
- [ ] Shadow harvest arms (≥3 reps, both backends) with full gov2 logging.
- [ ] Nested predictive analysis + NC shuffle offline; **go/no-go gate G1**.

**Phase 4 — joint policy implementation (≈4 days)**
- [ ] Step 4 full Option-A rule; ABSTAIN tagging; KV/Vector signal subsets.
- [ ] Truth-table + unit tests; shadow smoke.

**Phase 5 — replay and calibration (≈4 days)**
- [ ] Steps 7, 8, 9; thresholds frozen + committed (`CALIBRATION_FROZEN.md`); Option-B fit.

**Phase 6 — ablations (≈30 h compute + 2 days analysis)**
- [ ] §18 matrix (priority order); analyzer on ablation manifest; **gate G2**.

**Phase 7 — full BFCL evaluation (≈10 h compute + 2 days)**
- [ ] Confirmatory 5× A/B (baseline / geometry_only / v1 [/ risk_v1 if G2 passes]).
- [ ] Full §19 stats; EVAL_RESULTS_V2.md.

**Phase 8 — thesis analysis and reporting (≈1 week)**
- [ ] Risk-coverage curves, entropy-calibration plots, novelty-claim final search.
- [ ] Update `EXPERIMENTS_OVERVIEW.md`; retire legacy env docs; final push.

## 23.1 Go/no-go gates

- **G1 (after Phase 3):** nested-model ΔAUC(entropy | geometry+margin) > 0 with grouped
  CI excluding 0 on harvest data, and NC shuffle ≈ 0. Fail → drop entropy terms; ship
  geometry+margin v1 (still Phases 4–7, reduced matrix), thesis pivots to the
  simplification + negative-result narrative.
- **G2 (after Phase 6):** arm 5 non-inferior to arms 2 and 3 on official score AND
  superior on ≥1 pre-registered secondary. Fail → confirmatory campaign runs
  geometry_only as the headline arm.
- **G3 (before Phase 7 launch):** calibration frozen + committed; no NLI import in new
  arms; replay bit-identity green.

---

## 24. Thesis claims that would and would not be justified

**Justified if the plan's gates pass:**
- "A deterministic, LLM-free, two-stage write-admission gate matches the official BFCL
  score of heavier cascades at ~20 % lower runtime with zero online model calls beyond
  the embedder" (engineering claim; needs only §19 minimum success).
- "Geometric filtering is the load-bearing component of write-time governance"
  (already Holm-significant from the 2026-07 campaign; cite `geo_off`).
- "Retrieval-score entropy adds out-of-sample predictive value for harmful writes beyond
  geometry and margin" (only if G1 + NC pass).
- "Joint margin–entropy admission reduces harmful-write rate at non-inferior official
  accuracy" (only if G2 passes).

**Not justified (regardless of outcomes):**
- Any claim that governance *revives* memory chains (§1.6 — survival is benchmark noise).
- Any official-score improvement claim from pooled counts, single replicates, or
  uncorrected multiple comparisons.
- "Entropy detects hallucinated memories" — the gate observes retrieval geometry, not
  truthfulness; a fluent wrong memory with clean margins passes.
- Generalization beyond BFCL v4 KV/Vector stores (rec_sum was excluded throughout).
- Any NLI-related negative claim stronger than "in this pipeline, at this model scale,
  online NLI cost exceeded its benefit" — DeBERTa-large on CPU with 66 % neutral punts is
  not evidence against NLI as such.

---

*End of plan. No implementation code is to be written until this document is reviewed —
the first executable item is Phase 0's replay guard.*
