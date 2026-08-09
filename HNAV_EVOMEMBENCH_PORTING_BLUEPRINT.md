# H-Nav → EvoMemBench Porting Blueprint

**Reverse-engineered from code** in `berkeley-function-call-leaderboard/` (branch `claude/nihai-plan-v2-cascade-thresholds-k34u67`).
All paths below are relative to `berkeley-function-call-leaderboard/` unless prefixed with `gorilla/`.
Every claim is labeled: **[CODE-VERIFIED]** (read directly from the implementation, cited), **[REPORT-VERIFIED]** (from committed reports/artifacts, cited), **[INFERENCE]** (my synthesis from verified facts), **[EVOMEMBENCH-HYPOTHESIS]** (a proposal that must be checked against the real EvoMemBench repo).

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Current H-Nav definition](#2-current-h-nav-definition)
3. [Architecture diagram](#3-architecture-diagram)
4. [BFCL write-path implementation](#4-bfcl-write-path-implementation)
5. [BFCL read/retrieval-path implementation](#5-bfcl-readretrieval-path-implementation)
6. [GeometryGate deep dive](#6-geometrygate-deep-dive)
7. [Retrieval entropy deep dive](#7-retrieval-entropy-deep-dive)
8. [H-Nav intervention policy](#8-h-nav-intervention-policy)
9. [Relevant negative and positive BFCL findings](#9-relevant-negative-and-positive-bfcl-findings)
10. [Benchmark-independent H-Nav core](#10-benchmark-independent-h-nav-core)
11. [BFCL-specific dependencies](#11-bfcl-specific-dependencies)
12. [Hypothesized EvoMemBench mapping](#12-hypothesized-evomembench-mapping)
13. [Proposed EvoMemBench H-Nav architecture](#13-proposed-evomembench-h-nav-architecture)
14. [Detailed pseudocode](#14-detailed-pseudocode)
15. [Integration checklist for a future EvoMemBench agent](#15-integration-checklist-for-a-future-evomembench-agent)
16. [Stage-0 headroom analysis](#16-stage-0-headroom-analysis)
17. [Proposed experiment arms and controls](#17-proposed-experiment-arms-and-controls)
18. [Failure taxonomy](#18-failure-taxonomy)
19. [Required logs/artifacts](#19-required-logsartifacts)
20. [Statistical evaluation requirements](#20-statistical-evaluation-requirements)
21. [Thesis claims each possible outcome would support](#21-thesis-claims-each-possible-outcome-would-support)
22. [Open questions requiring the EvoMemBench repository](#22-open-questions-requiring-the-evomembench-repository)

---

## 1. Executive summary

**What H-Nav is.** [CODE-VERIFIED] H-Nav is not one module but a research program materialized as a middleware package (`bfcl_eval/model_handler/middleware/`, ~6 300 lines) plus a stack of thin handler subclasses hooking exactly four seams of BFCL's `@final` multi-turn loop. Its runtime pieces: (1) a **Stage-0 geometric write gate** in a whitened (MiniLM+ABTT) embedding space — max-similarity, QR-residual novelty, adaptive SAGE thresholds, and a literal verbatim-value guard deciding NOOP/ADD/ESCALATE; (2) a **margin/retrieval-entropy admission stage** that re-simulates the benchmark's own retrievers (BM25-over-keys, MiniLM cosine) over deterministic user-text probes, computing rank-of-self, normalized margins, entropy deltas (dH), and rank churn to resolve escalations into ADD/NOOP/SAFE_REWRITE/ABSTAIN; (3) a margin-triggered **read gate** that trims or disambiguates ranked retrieval results; (4) **placement/eviction/destructive guards**; (5) **action-side H_act** — sampled-completion vote entropy over canonicalized actions at four resolutions, plus token-logprob features; and (6) deterministic **capture/read scaffolds** that inject a real archival write on no-write turns and a real archival read on no-read question turns.

**What happened in BFCL.** [REPORT-VERIFIED] The signal machinery largely *works* (the marginal-diff detector finds real geometry defects, ΔAUC +0.076 CI [+0.015, +0.138]; vote features predict no-call steps at AUC 0.998) but the *decision classes it polices are nearly empty on this benchmark*: `must_write` ≈ 3.5% and `must_suppress` ≈ 0.7% of writes, so every gating policy failed its pre-registered gate (Stage 1 H1 FAIL, Stage 3 NO_GO, Stage 4 inconclusive/underpowered). The loss decomposition showed the real failure is a **capture→read pipeline**: 77–81% of questions never get their gold fact into memory (though ~88% of those facts were stated verbatim in conversation), and gold reaching *archival* memory converts to correct answers at ≈0.017/0.000 (kv/vector) versus 0.556/0.701 for core, because the agent almost never reads archival (1/155 question entries). The deterministic scaffolds attacking that pipeline are the surviving lever (C1 interim: read scaffold vector +8.4pp on 1/5 replicates, non-confirmatory). The headline lesson for porting: **"method has no signal" and "benchmark rarely generates the situations the signal is for" are different failures, and BFCL's negatives are overwhelmingly the latter** — which is exactly the question EvoMemBench must be able to separate on day one (§16).

**What this document provides.** A code-level reconstruction of every mechanism with file/line citations (§2–§8), the honest findings ledger (§9), a core-vs-glue decomposition with a per-component portability table (§10–§11), a benchmark-agnostic integration checklist and target architecture (§12–§13, §15), implementation-grade pseudocode (§14), and a pre-registered-style experiment design for EvoMemBench — headroom gate first (§16), arms/controls (§17), failure taxonomy (§18), logging/statistics requirements (§19–§20), and the thesis claims each outcome supports (§21).

---

## 2. Current H-Nav definition

**H-Nav is not one module.** [CODE-VERIFIED] It is an umbrella research program whose runtime artifacts are a *stack of handler subclasses* over one shared middleware package. The name "H-Nav" ("hierarchical/heuristic navigation" of memory state) covers, in the code:

| # | Sub-system | Runtime entry | Middleware | Status in code |
|---|---|---|---|---|
| 1 | **Stage-0 geometric write governance** ("Gov") | `bfcl_eval/model_handler/local_inference/qwen_gov.py` (`QwenGovHandler`, registry `-FC-GOV`) | `middleware/governance_filter.py` (2 428 lines), `middleware/geometry_gate.py` | Fully implemented, live-capable; shadow-first discipline |
| 2 | **Margin/retrieval-entropy admission** (Plan v2 "gov2") | same handler, `GOV_POLICY=geometry_margin_entropy_v1` / `_risk_v1` | `middleware/admission_policy.py`, `middleware/entropy_metrics.py`, `middleware/retrieval_sim.py`, `middleware/probe_gen.py` | Fully implemented; NLI removed from the online path |
| 3 | **Legacy NLI + Stage-2 cascade** | same handler, `GOV_POLICY=legacy_full` | `governance_filter.stage1_resolve` / `stage2_resolve`, `middleware/semantic_entropy.py` (NLI singleton) | Implemented but **retired to offline/legacy replay** (governance_filter.py:745-751, 970-976) |
| 4 | **Placement / eviction / destructive guard (SS5)** | same handler, `GOV_P_*` | `middleware/placement.py` + session `_destructive_pass`/`_placement_pass` (governance_filter.py:1962-2114) | Implemented, shadow-first, never promoted to a headline arm |
| 5 | **Read-time gate (SS6)** | same handler, `GOV_READ_*` | `middleware/read_gate.py` + session `_gate_reads` (governance_filter.py:2211-2258) | Implemented, shadow-first |
| 6 | **Action-side H_act sampling** (H-Nav Stage 2/3) | `qwen_hact.py` (`QwenHactHandler`, `-FC-HACT`) | `middleware/hact_sampler.py`, `middleware/action_space.py` | Implemented policies: `shadow`, `random_select`, `majority` (qwen_hact.py:63); gate policies never built because Stage 3 returned NO_GO |
| 7 | **Marginal-diff falsifier features** (Stage 1.3) | offline only | `middleware/diff_signals.py` (17 features) | Implemented, offline-only (consumed by `bfcl_eval/scripts/refeature_diff.py` / calibration) |
| 8 | **Deterministic write scaffold** (alt5R) | `qwen_scaffold.py` (`QwenScaffoldHandler`, `-FC-SCAF`) | none (self-contained) | Implemented, live |
| 9 | **RAG read/write scaffolds** (methods M1-M10) | `qwen_rag.py` (`QwenRagHandler`, `-FC-RAG`) | none (self-contained; reuses `slug_key`) | Implemented, live; C1 campaign running blind |
| 10 | **Semantic-entropy gate** (thesis Idea 1, predecessor) | `qwen_se.py` (`-FC-SE`) | `middleware/semantic_entropy.py` | Implemented; separate project, shares the MiniLM/NLI singletons |
| 11 | **MIG retrieval reranker** (separate project) | `qwen_mig.py` (`-FC-MIG`) | `middleware/mig_reranker.py` | Implemented; relevant as the precedent for tool-result rewriting |
| 12 | **Offline counterfactual labeling + statistics** | scripts | `bfcl_eval/scripts/label_outcomes_hnav.py`, `hnav_answer_index.py`, `evaluate_hact_shadow.py`, `select_hact_thresholds.py`, `calibrate_margin_entropy.py`, `falsifier_*.py`, `rag_falsifiers.py` | Implemented; this is where the ground-truth targets (`must_write`, `must_suppress`, ...) are defined |

**The load-bearing conceptual definition** [INFERENCE from the above]: H-Nav = *detect difficult memory states with cheap deterministic signals (geometry of the embedded store + simulated-retrieval score distributions + sampled-action disagreement), then intervene selectively at one of four hook points (write admission, write placement, read disambiguation, turn-level capture/read scaffolding), under a shadow-first, counterfactually-labeled evaluation discipline.*

Sub-systems 1-2 are the "geometry side" (task item A), the retrieval-simulation machinery inside 2-3 plus 5 is the "retrieval-uncertainty side" (task item B), and the applied decisions in 1, 2, 4, 5, 6, 8, 9 are the "navigation/intervention policy" (task item C).

---

## 3. Architecture diagram

[CODE-VERIFIED] Derived from the actual class hierarchy and hook points (`qwen_gov.py:91-224`, `qwen_hact.py:97-381`, `qwen_scaffold.py:145-261`, `qwen_rag.py:242-673`, `governance_filter.py:1392-2258`).

```
                         BFCL harness (base_handler.py @final multi-turn loop)
   ┌────────────────────────────────────────────────────────────────────────────────────┐
   │  per test entry:  _pre_query_processing_prompting → [ per turn: user msg →         │
   │   ( per step ≤ MAXIMUM_STEP_LIMIT: _query_prompting → _parse_query_response        │
   │     → decode_execute → execute tools → _add_execution_results_prompting ) ]        │
   └────────────────────────────────────────────────────────────────────────────────────┘
        ▲ hooks (all overridden points are non-@final methods of the loop)
        │
   QwenFCHandler (qwen_fc.py)  — Qwen chat template, <tool_call> JSON → call strings
        │
   QwenGovHandler (qwen_gov.py, -FC-GOV)
        │   _pre_query_processing_prompting : build GovernanceSession
        │       (rehydrate mirror cache from backend snapshot _final.json)
        │   add_first_turn/_add_next_turn    : stash session.user_text  (probe source)
        │   decode_execute                   : session.govern_calls(calls)     [HOOK 1]
        │   _add_execution_results_prompting : session.patch_results(...)      [HOOK 2]
        │
        ├── QwenHactHandler (qwen_hact.py, -FC-HACT)
        │       _query_prompting : primary request (baseline params + logprobs + seed)
        │                          + n-1 exploration samples @ HACT_TEMP
        │                          → canonicalize → vote features → hact record
        │
        ├── QwenScaffoldHandler (qwen_scaffold.py, -FC-SCAF)
        │       decode_execute : no-call prereq turn → inject archival_memory_add(user turn)
        │
        └── QwenRagHandler (qwen_rag.py, -FC-RAG)
                decode_execute : no-call question turn → inject archival read (M1)
                                 no-call prereq turn  → packed archival write (M2)
                _add_execution_results_prompting : payload rerank/pad/nudge (M3/M9/M10)

   GovernanceSession (governance_filter.py:1392)          middleware/, pure logic
   ├─ govern_calls (HOOK 1)                                ──────────────────────
   │   ├─ build_candidate(call) → WriteCandidate            parse_call (ast, never eval)
   │   ├─ legacy path: _govern_one                          compute_signals + decide
   │   │    └─ ESCALATE → stage1_resolve (NLI, legacy)      semantic_entropy.NliScorer
   │   │                 → stage2_resolve (probes, legacy)  probe_gen + retrieval_sim
   │   ├─ gov2 path:  _govern_one_admission
   │   │    └─ MemoryMutationAdmissionBoundary.evaluate     admission_policy.py
   │   │         ├─ GeometryGate.evaluate (Stage 0)         geometry_gate.py
   │   │         └─ compute_margin_entropy_signals          entropy_metrics + retrieval_sim
   │   │              + apply_option_a_rule / risk rule
   │   ├─ _destructive_pass / _placement_pass (SS5)         placement.py
   │   └─ apply: NOOP→decoy call, REWRITE→in-place call swap, expansions inserted
   ├─ patch_results (HOOK 2)
   │   ├─ substitute synthetic successes at NOOP indices; restore original call strings
   │   ├─ _observe: mirror-cache update on exact backend success strings only
   │   └─ _gate_reads (SS6): margin-triggered read disambiguation                read_gate.py
   └─ JSONL audit log (gov2 schema)                          log_gov_record

   Memory backends (never modified):
     memory_kv.py      KV: BM25Plus over KEY NAMES; core 7×300 chars, archival 50×2000
     memory_vector.py  Vector: all-MiniLM-L6-v2 + faiss IndexFlatIP (cosine), same caps
```

**The decision space of the write gate** [CODE-VERIFIED, `memory_mutation.py:25-26`]: `ADMISSION_ACTIONS = ("ADD", "NOOP", "SAFE_REWRITE", "ABSTAIN")`, stages `("GEOMETRY", "MARGIN_ENTROPY")`, with a closed reason-code vocabulary (`memory_mutation.py:30-43`). There is **no DELETE action** in the admission vocabulary — deletion is governed on the *observed* channel by the destructive guard (§8).

---

## 4. BFCL write-path implementation

End-to-end trace of one governed write, naming every function. [CODE-VERIFIED]

**Step 0 — session construction.** `QwenGovHandler._pre_query_processing_prompting` (qwen_gov.py:92-100) calls `_build_session` → `_load_snapshot` (qwen_gov.py:129-162), which reads the backend's rolling snapshot `<model_result_dir>/<agentic/memory/<backend>>/memory_snapshot/<scenario>_final.json` (same path the backend's `MemoryAPI._prepare_snapshot` uses). `GovernanceSession.__init__` (governance_filter.py:1392-1491) loads the ABTT artifact (`load_abtt`, fails loudly if missing), gets the MiniLM encoder singleton (`semantic_entropy._get_encoder`), and `GovernanceCache.rehydrate` (governance_filter.py:492-520) embeds every stored item into the whitened space. KV items are embedded as the composite `"key words: value"` (`kv_composite_text`, governance_filter.py:399-402 — underscores become spaces because keys carry real semantics). `ThresholdState.seed` initializes the SAGE-style adaptive residual threshold per tier.

**Step 1 — raw turn → candidate.** The model emits `<tool_call>{"name": "core_memory_add", "arguments": {...}}</tool_call>`; `QwenFCHandler.decode_execute` (qwen_fc.py:36-46) converts it to a call string like `core_memory_add(key='age', value='35')`. `QwenGovHandler.decode_execute` (qwen_gov.py:199-204) forwards the list to `session.govern_calls` (governance_filter.py:1503-1540), which increments the step counter, flushes any pending H_act record, and per call runs `build_candidate(backend, call)` (governance_filter.py:1302-1324): `parse_call` (ast-based, governance_filter.py:1277-1288) + op tables `KV_WRITE_OPS` / `VECTOR_WRITE_OPS` (governance_filter.py:1236-1247) → a `WriteCandidate{op, kind, tier, backend, text, args, ref}`. Non-write calls pass through untouched.

**Step 2 — features.** Under a gov2 policy, `_govern_one_admission` (governance_filter.py:1655-1839) builds a `MemoryMutationCandidate` (`memory_mutation.from_write_candidate`) and calls `MemoryMutationAdmissionBoundary.evaluate` (admission_policy.py:441-550):
- `preflight_would_succeed(candidate, cache)` (governance_filter.py:1327-1352) predicts backend acceptance from the mirror (dup key, capacity, length caps).
- `GeometryGate.evaluate` (geometry_gate.py:121-175): exact-duplicate fast path → else whiten candidate text (`session._whiten_one` = MiniLM encode → `AbttTransform.apply`), `compute_signals` (governance_filter.py:661-698: `sim_max`, residual `r`, `tau_t`, verbatim values), `decide` (governance_filter.py:701-738), then the retrievability floor on confident ADDs. See §6 for formulas.
- If Stage 0 says ESCALATE (and policy ≠ `geometry_only`): `compute_margin_entropy_signals` (admission_policy.py:120-250) simulates per-tier retrieval with/without the provisional candidate over deterministic probes; `apply_option_a_rule` (admission_policy.py:253-306) or the calibrated risk rule (admission_policy.py:552-569) maps signals to `ADD | NOOP | SAFE_REWRITE | ABSTAIN`. See §7.

**Step 3 — apply.** Back in `_govern_one_admission` (governance_filter.py:1719-1749), and only when *not* shadowed (`GOV_ME_SHADOW=0` and `GOV_DRY_RUN=0`):
- `NOOP` → the call at that index is replaced by the read-only decoy `core_memory_retrieve_all()` (`DECOY_CALL`, governance_filter.py:1266) and a `_pending[idx] = {mode:"noop", synthetic: synthetic_success(...)}` entry is recorded. Vector adds get an out-of-band shadow id (9000+, governance_filter.py:64-68) so a later model reference to it errors loudly instead of colliding.
- `SAFE_REWRITE` → the call string is replaced in place by `ad.rewritten_call` (KV key-suffix rewrite only; the value is byte-identical, admission_policy.py:309-337).
- `ADD` / `ABSTAIN` → the write executes unchanged; ABSTAIN only tags the stored item (`_flagged_texts`) — **ABSTAIN is observational in v1** (admission_policy.py:26-28).
- SS5 passes (`_destructive_pass`, `_placement_pass`) may additionally expand one call into `[archival_memory_add(victim), core_memory_remove(victim)]` (atomic move) via `_apply_expansions` (governance_filter.py:2097-2114).

**Step 4 — execution + result patching.** The harness executes the (possibly rewritten) call list against the real backend. `QwenGovHandler._add_execution_results_prompting` (qwen_gov.py:207-223) calls `session.patch_results` (governance_filter.py:2179-2209): at NOOP indices the decoy's real output is replaced by the synthetic backend-format success string (`synthetic_success`, governance_filter.py:1355-1364 — byte-exact copies of `memory_kv.py`/`memory_vector.py` success messages) and the original call string is restored into `model_responses_decoded` so the chat history shows the model's own call; at rewrite indices the *real* result flows through and the *executed* call stays visible (the mirror must observe the truth).

**Step 5 — state observation.** `_observe` (governance_filter.py:2260-2313) parses each executed call + result and updates the mirror cache **only on exact success signatures** (e.g. `{"status": "Key-value pair added."}`), embedding the new/updated item (`_put_item`, governance_filter.py:2331-2375). Removes/clears (never gated in the base path) are observed too, so the mirror can never over-count.

**Step 6 — logging.** Every decision emits a full `gov2` JSONL record (governance_filter.py:1783-1839): all Stage-0 signals, the full `s1` margin-entropy block, `user_text` verbatim, thresholds, the decision-feature SHA (`_stamp_features_sha`, admission_policy.py:571-589 — a leakage control asserting features never change post hoc), applied action, synthetic result, latencies.

**Not implemented on the write path** [CODE-VERIFIED]: no LLM ever rewrites *values* (v1 normalization contract, `memory_mutation.py:62-67`); no online NLI under gov2 policies (asserted by `_get_nli_scorer`, governance_filter.py:1542-1555); Vector has **no** SAFE_REWRITE (`_safe_rewrite` returns None for vector, admission_policy.py:317-318); nothing ever *deletes* memory as an admission action.

---

## 5. BFCL read/retrieval-path implementation

**5.1 The native read path (untouched by default).** [CODE-VERIFIED from `retrieval_sim.py:10-29` docstring, which pins the backend line numbers; full backend detail in §11]
- KV: `*_memory_key_search(query, k)` ranks **key names only** with `rank_bm25.BM25Plus` (tokenization `text.replace('_',' ').lower().split()`, memory_kv.py:71-88); values are fetched by exact `*_memory_retrieve(key)`. Result payload `{"ranked_results": [[score, key], ...]}`.
- Vector: `*_memory_retrieve(query, top_k)` embeds with raw MiniLM (`normalize_embeddings=True`) and does exact inner-product search over L2-normalized vectors (faiss `IndexFlatIP`, memory_vector.py:244-256, 309-333). Result payload `{"result": [{"id", "similarity_score", "text"}, ...]}` — **scores are shown to the model**.
- Core memory is additionally auto-dumped into the system prompt at entry start (the `"Core Memory from previous interactions:"` block that `qwen_rag.py:268-276` scrapes back out) — so core reads are free, archival reads require an explicit tool call. [CODE-VERIFIED: `add_memory_instruction_system_prompt`, model_handler/utils.py:610-650, invoked once per entry at base_handler.py:134-145/:431-442 and never refreshed mid-entry] Note also the KV/vector API asymmetry: KV has **no** `archival_memory_retrieve_all` (archival reachable only by exact key, key listing, or key search), while vector does expose it — one call dumps the whole archival store. [CODE-VERIFIED: memory_kv.py tool surface vs memory_vector.py:225]

**5.2 Read-time gate (SS6)** — the only in-place H-Nav read intervention. `session._gate_reads` (governance_filter.py:2211-2258) runs inside `patch_results` on the *results* of gated read ops (`READ_GATED_OPS`, read_gate.py:33-36: KV `*_key_search`, Vector `*_retrieve`; exact-key KV retrieve is not gated — nothing to disambiguate). `gate_read` (read_gate.py:132-172):
1. parse the ranked payload; margin = top1−top2 score;
2. `margin > GOV_READ_MARGIN` (default 0.05) → **return only top-1** (payload trimmed);
3. else ambiguous set = top-1's δ-neighborhood (score within `margin_thr`), bounded 2..`GOV_READ_SET_MAX` (clamped 2..4); pick one *distinct refinement token* (a token unique to exactly one candidate that also appears in the query, read_gate.py:97-108); **one** re-rank round within the set only, via the backend-faithful simulators (`default_rerank`);
4. resolved → return the refined top-1; else → return the entire bounded set and let the model choose.
It is read-only by construction: only the payload string can change, and only when live (`GOV_READ_ENABLED=1, GOV_READ_SHADOW=0`, not dry-run).

**5.3 Read scaffold (RAG M1)** — the intervention that actually attacks the measured failure mode. `QwenRagHandler.decode_execute` (qwen_rag.py:494-563): on a memory **question** entry whose model response decodes to *no* tool calls, inject exactly one archival read — `archival_memory_retrieve(query=<probe>, top_k=K)` (vector) or `archival_memory_key_search(query=<probe>, k=K)` (KV) — where the probe is the current user turn text verbatim (M4 `core_augmented` optionally appends content words of the in-context core dump, qwen_rag.py:349-356). The result enters context through the normal execution-result path; the model answers on the next step; the injection fires at most once per turn. Controls implemented in the same class: `read_irrelevant` (frozen generic probe, qwen_rag.py:141), `instruction_only` (system-prompt nudge, qwen_rag.py:145-149), payload pad (M3), agent-side BM25/random rerank of a widened pool (M9), evidence-format answer instruction (M5), result nudge ± null (M10), KV two-stage key→value reads (M8).

**5.4 Retrieval-uncertainty *measurement* on the read path** happens in simulation, not on live reads: `retrieval_sim.simulate_kv` / `simulate_vector` are backend-faithful re-implementations (retrieval_sim.py:52-119) used by the write-side margin/entropy signals (§7), the read gate's re-rank round, the retrievability floor, and the offline replay instrument. **There is no live "retrieval entropy triggers a different read strategy" mechanism**: the read gate triggers on *margin* only, and the entropy quantities are logged diagnostics ("never a decision input", retrieval_sim.py:123-126, GovConfig comments governance_filter.py:127-129). [CODE-VERIFIED]

**5.5 MIG reranker (adjacent project, precedent only).** `QwenMIGHandler` widens `top_k`/`k` at `decode_execute` and rewrites retrieval result payloads at `_add_execution_results_prompting` scoring candidates by LLM-judge utility or logprob information gain `MIG(m) = logp(a_hat|S+{m}) − logp(a_hat|S)` (mig_reranker.py:27-37). It is not part of H-Nav's decision stack but proves the tool-result-boundary hook is sufficient for arbitrary read-side reordering/trimming. [CODE-VERIFIED]

---

## 6. GeometryGate deep dive

All quantities live in the **whitened space**: raw `all-MiniLM-L6-v2` (384-d, CPU singleton shared with the vector backend) → ABTT: subtract corpus mean `mu`, remove projections onto the top `D=16` covariance eigendirections `U_top`, L2-normalize (`AbttTransform.apply`, governance_filter.py:321-341). The artifact `middleware/artifacts/abtt_minilm_l6_d16.npz` was fitted offline by `bfcl_eval/scripts/compute_abtt.py` on **BFCL's own memory-prereq conversation pool, sentence-segmented** (deliberately not an external corpus); measured there: median pairwise cosine 0.096 raw → ~0.00 after ABTT; top-16 directions carry ~34% of variance (compute_abtt.py:1-20). The whitened space is used **only** for governance decisions; the vector backend's raw-MiniLM retrieval space is never touched and the two are never mixed (governance_filter.py:25-27). [CODE-VERIFIED]

**Representations** [CODE-VERIFIED]:
- *Existing items*: `MemoryItem{ref, text, emb_whitened, turn_written, tier, low_confidence/abstained/flagged, probes}` (governance_filter.py:380-397), mirrored across **both tiers** in `GovernanceCache` — redundancy is judged against everything stored regardless of tier (governance_filter.py:456-458).
- *Incoming candidate*: whole composite text (`WriteCandidate.text`; KV = `"key words: value"`), whitened once.
- *Whole blob*: implicitly, the row-space of the stored matrix `M` (n×384) and its QR basis `Q` (governance_filter.py:469-485; full recompute per cache version — n ≤ 57 makes incremental QR unnecessary, but the function boundary is where it would drop in).
- *Marginal difference*: **not** in the online gate — it lives in `diff_signals.py` (offline, §7.6/§9): `marginal_diff(old,new)` via `difflib.SequenceMatcher` char opcodes + clause set-difference over `recsum_blobdiff.split_propositions`.

**Signals** (`compute_signals`, governance_filter.py:661-698):
1. `sim_max = max(M @ v_w)` — max whitened cosine to any stored item (unit vectors ⇒ dot = cosine). The full sims vector is kept for neighbor selection.
2. `r = ‖v_w − Q Qᵀ v_w‖` — residual norm of the candidate outside the subspace *spanned by all stored items* (QR). "The part of the candidate memory cannot explain."
3. Adaptive threshold `tau_t` (SAGE-style, per tier, per conversation; `ThresholdState`, governance_filter.py:528-544): `tau* (ρ) = tau_min + tau0·exp(−λρ)` with density ρ = size/capacity, EMA-smoothed `tau ← α·tau_prev + (1−α)·tau*`; defaults `tau0=0.25, tau_min=0.025, λ=2.0, α=0.9` (GovConfig).
4. **Verbatim-value gate**: regex extraction of concrete values (emails, ISO/slash/prose dates, money/percent, unit-suffixed numbers, phone-like runs, capitalized name bigrams, bare numbers — most-specific-first with span claiming; governance_filter.py:553-598), then *literal normalized substring* search in the concatenated stored texts (`verbatim_present`) → hits/misses. If no structured values exist, a conservative fallback compares **content tokens** (len>2, non-stopword) so a geometric near-duplicate with genuinely new content words can never NOOP (governance_filter.py:687-695).

**Decision rule** (`decide`, governance_filter.py:701-738), thresholds `delta=GOV_DELTA` (default 0.025), `sim_high=GOV_SIM_HIGH` (default 0.80):

```
if store empty:                                ADD  "empty_memory"
if r < delta AND sim_max > sim_high AND all verbatim values already present:
    if backend would reject the write:         ADD  "noop_blocked_by_preflight"
    else:                                      NOOP "redundant"
if r > tau_t AND some verbatim value is new:   ADD  "novel"
else:                                          ESCALATE "ambiguous"
```

`GovConfig.validate` (governance_filter.py:273-312) guards against a degenerate `(sim_high, delta)` pair: with one stored neighbor `r = sqrt(1−sim²)`, so `delta < sqrt(1−sim_high²)` makes the residual gate bind first and silently raise the effective NOOP similarity bar — it warns (or raises under `GOV_STRICT_THRESHOLDS`).

**Plan-v2 hardenings** (`geometry_gate.py`): (1) **exact-duplicate fast path** — normalized byte-equality against any stored item (both tiers) → NOOP with `sim_max=1.0, r=0.0`, *zero* encoder calls; added because ABTT noise made identical rewrites land at whitened sim ≈0.99 < `sim_high` — the only observed false-negative class of geometry (geometry_gate.py:10-15). Preflight-guarded like every NOOP. (2) **Retrievability floor** on confident ADDs — the candidate's own key/keyword probe must rank it top-`FLOOR_TOP_N=3` (of `FLOOR_K=5`) in the provisional tier store via the backend-faithful simulator; a "novel" write its own probe cannot find is malformed/off-topic → ESCALATE, not ADD (geometry_gate.py:82-165).

**KV vs Vector differences** [CODE-VERIFIED]:

| Aspect | KV | Vector |
|---|---|---|
| Candidate text embedded | composite `"key words: value"` | the stored text itself |
| Identity/ref | key string | integer `vec_id` (str in the mirror) |
| Gated ops | `core/archival_memory_add`, `*_replace` | `*_add`, `*_update` |
| Stage-1 pre-check (legacy) | canonical-key: `add` on existing key → REWRITE to `replace`; idempotent replace → NOOP (governance_filter.py:856-886) | idempotent update (same normalized text) → NOOP (governance_filter.py:887-894) |
| Retrieval simulated over | **key names** (BM25Plus) | stored texts (raw-MiniLM cosine) |
| Locate/strong probe channels | `template:key`, `template:key_value` item probes — NL probes are structurally degenerate against BM25-over-keys (admission_policy.py:70-72) | user-text decision-probe channels `template:user_identity`, `template:user_keywords`, `paraphrase:*` (admission_policy.py:65-69) |
| nmargin normalization | `(s1−s2)/(|s1|+ε)` (BM25 scale-free) | raw `s1−s2` (cosine bounded) (entropy_metrics.py:26-38) |
| Smallstore guard | interference term deactivated below `SMALLSTORE_N=5` tier items (admission_policy.py:74, 280-283) | none |
| SAFE_REWRITE | key-suffix append only, value byte-identical | **none in v1** (admission_policy.py:317-318) |
| NOOP synthetic | `{"status": "Key-value pair added."}` / `"Key added."` / `"Key replaced."` | `{"id": <shadow id ≥ 9000>}` / `{"status": "ID <n> updated."}` |

**ADD/UPDATE/NOOP/DELETE implications** [CODE-VERIFIED]: ADD executes unchanged; NOOP suppresses with a synthetic success (the model believes it wrote); SAFE_REWRITE/REWRITE changes *which* op executes (add→replace supersede in legacy; key-suffix add in gov2; archival routing / atomic move in SS5); DELETE is never an admission outcome — destructive ops are the model's own and are *guarded* (blocked/expanded) by G7, not issued by governance except as the remove half of an atomic move.

---

## 7. Retrieval entropy deep dive

**Five distinct uncertainty families exist in this repository and must not be conflated** [CODE-VERIFIED]:

| Family | Distribution over | Where computed | Used for decisions? |
|---|---|---|---|
| (a) Retrieval score entropy / margins | simulated top-k retrieval scores of a probe against a (provisional) store | `retrieval_sim.py`, `entropy_metrics.py`, consumed by `admission_policy.py` and legacy `stage2_resolve` | Margins and ranks: YES (Option A rule); softmax entropy H: **logged only** in legacy Stage 2; `dH`/`churn` deltas: YES (interference term) |
| (b) Semantic entropy | clusters of N sampled *completions* (union-find over call-string cosine / content-token equivalence) | `semantic_entropy.py` (`choose`), `qwen_se.py` | YES in the SE gate (destructive/overwrite commit thresholds) — separate project |
| (c) Action entropy H_act | canonical-action *signatures* of N sampled completions at 4 resolutions | `action_space.py` (`vote_features`), `hact_sampler.py`, `qwen_hact.py` | Shadow-logged; gate policies never implemented (Stage 3 NO_GO) |
| (d) Vote disagreement (non-entropy) | same votes: `modal_share`, `vote_margin`, `n_unique`, `n_no_call`, `n_parse_fail`, `primary_matches_modal` | `action_space.vote_features` | Used as *controls* in the Stage-3 nested ladder (M4); `majority` policy uses modal counts |
| (e) Token-logprob uncertainty | per-token logprobs of the primary completion (`lp_mean/min/ppl`, top-k margins, tool-span slice) | `hact_sampler.logprob_features` (hact_sampler.py:99-150) | Features only (M3 block); never a live gate |

### 7.1 Family (a): the write-side margin/entropy machinery (the live one)

**Candidate scores** come from *backend-faithful simulation*, never from the live backend: `simulate_kv(keys, probe, k=5)` reproduces `memory_kv._similarity_search` byte-identically (BM25Plus over key names, same tokenizer, same stable sort); `simulate_vector(texts, probe, k=5)` reproduces `VectorStore.retrieve` (raw-MiniLM normalized embeddings, dot product = cosine, exact at N ≤ 57) (retrieval_sim.py:52-119). Provisional inserts are list copies — the store is never touched (admission_policy.py:127-129).

**Probability construction / entropy** (`retrieval_sim.entropy`, retrieval_sim.py:136-146): softmax over the top-k (=5) scores at temperature T (`GOV_S2_T_KV` / `GOV_S2_T_VEC`, default 1.0): `p_i = softmax(s_i/T)`, `H = −Σ p log p` (nats), `n_eff = exp(H)`. **Absolute H is deliberately decision-inert** — the config comments pin "softmax T for logged H only — NEVER a decision input" (governance_filter.py:164) and every record stores the T used (governance_filter.py:992-995). The G1-fail analysis found raw-score softmax entropy scale-degenerate (cosine compression → H ≈ log k always; BM25 ties → H = log k exactly), which motivated two v2 replacements (entropy_metrics.py:133-148): `zscore_entropy` (softmax over z-scored top-k scores; affine-invariant; all-tied → log n by definition) and `vn_entropy` (von Neumann entropy of the neighborhood's normalized embedding Gram matrix ρ = K/tr(K); `effective_rank = exp(S)` — a deterministic retrieval-side analog of SeaKR's Gram-determinant uncertainty, giving KV a semantic signal BM25-over-keys cannot). Both are implemented as pure functions; **in the online path they are not consumed** — they feed the offline feature sets (`refeature_entropy_v2.py`, `diff_dSvn` in diff_signals). [CODE-VERIFIED]

**Relative (delta) entropy is the decision-relevant form.** `delta_h_for_probes` (retrieval_sim.py:174-213): for a probe list, `dH = H_after − H_before` where "after" appends the candidate's corpus representation (KV: its key; Vector: its text). Two variants distinguished only by probe source (entropy_metrics.py:106-129): `dH_self` (candidate-locating decision probes — the candidate's effect on its own retrieval neighborhood) and `dH_neighbor` (each top-k neighbor's own cached item probes — does the candidate blur an existing item's retrievability).

**The full Stage-1 signal record** (`MarginEntropySignals`, admission_policy.py:78-117), computed in `compute_margin_entropy_signals` (admission_policy.py:120-250): per probe `rank_self` (1-indexed; k+1 sentinel when absent), raw margin, `nmargin` (backend-normalized), `H_pre`, `H_post`; aggregates by **median over locating channels** (never min-pooling): `rank_self_med`, `nmargin_med`, `dH_self`, `n_eff_norm = exp(H̄_post)/min(k, n+1)`, `disp` (NQC-style std/mean), plus the interference pair `dH_mean` (mean neighbor dH) and `churn` (Jaccard *distance* of pre/post top-3 identity sets, `CHURN_TOP_N=3`).

**Probes** are deterministic templates (probe_gen.py): *decision probes* from the **user's turn text only** — hard anti-circularity invariant: model-generated candidate text/args are not even parameters of `generate_decision_probes` (probe_gen.py:236-257) — channels `template:user_identity` (the sentence itself), `template:user_value_question` ("what is <value>", interference-only, excluded from locate aggregation per the SS4.2 fix), `template:user_keywords`, optional beam-search paraphrases from flan-t5-small (a non-Qwen family, the structural loop-break; default off); *item probes* from stored items, backend-matched (KV: key words / "what is <key words>" / key+value tokens; Vector: identity text / "what do you know about ..." / keyword bag), cached on `MemoryItem.probes` and persisted to the `<scenario>_gov_state.json` sidecar.

**How it affects intervention** — the Option A rule (§8). Detection semantics [INFERENCE from the rule structure]: `locate` (median rank ≤ R) + `strong` (median nmargin ≥ M) detect *findable and unambiguous*; `interf` (dH_mean ≥ D or churn ≥ C) detects *this write blurs existing retrieval*; `¬locate ∧ dup_risk` detects *shadowed duplicate* (the candidate cannot be found because an existing near-duplicate absorbs its probes); `¬locate ∧ ¬strong ∧ interf` detects *high-risk ambiguity*. So the mechanism detects **uncertain and shadowed** retrieval; genuinely *missing* retrieval (no candidate at all) is only touched by the retrievability floor and, on the read side, by the scaffolds.

### 7.2 Legacy Stage 2 (retired to `legacy_full`)

`stage2_resolve` (governance_filter.py:1072-1227): decision probes must retrieve the provisional candidate **top-1 on every probe** with min top1−top2 margin > `GOV_S2_MARGIN` (default 0.05, placeholder). Failure → one deterministic canonicalization attempt (`canonicalize_candidate`, governance_filter.py:1024-1069: pick the highest-value discriminative token absent from every rival — verbatim values first, then user-text/value content tokens; KV appends it to the key, Vector prepends "Regarding <token>: ") re-tested on the same probes; final fallback writes as-is with `low_confidence=True`. Stage 2 never suppresses. The binary all-top-1 rule was replaced by the rank/median machinery above ("replace the brittle binary all-top-1 of the old Stage 2", entropy_metrics.py:76-78). [CODE-VERIFIED]

### 7.3 Family (b): semantic entropy (SE gate — do not merge into H-Nav)

`semantic_entropy.choose` (semantic_entropy.py:369-461): N=5 samples at T=0.7 → candidates (tool-call vs text) → union-find clustering (tool calls: same sorted func-name tuple AND normalized-call-string cosine ≥ 0.80; texts: content-token Jaccard ≥ 0.6 or containment — chosen because MiniLM cosine under-merges short paraphrases, measured 0.445 for a trivial pair) → cluster entropy `H = −Σ (s/n) log2(s/n)` (bits) + majority fraction. Destructive majority requires `H ≤ 0.7` and majority ≥ 0.6; overwrite requires `H ≤ 1.2`; else fall back to the largest non-destructive cluster's medoid; if none exists, execute anyway but flag `forced_destructive`. Commits the medoid; never synthesizes text.

### 7.4 Family (c)+(d): action-side H_act

`action_space.canonicalize` maps each decoded call to a `CanonicalAction` at four resolutions (R1 tool / R2 op `kind:tier` / R3 op+target / R4 op+target+content-fingerprint sha256[:16] of superficially-normalized content); a sample's signature is the **order-insensitive** join of its per-call signatures; empty decode → `"no_call"`, decode crash → `parse_fail` (action_space.py:81-159). `vote_features` (action_space.py:174-206): Shannon entropy over signature counts (nats), normalized variant `h/log(n_unique)`, modal share, vote margin `(top1−top2)/n`, uniqueness/no-call/parse-fail counts, `primary_matches_modal`. The runtime (`qwen_hact.py`) issues a **primary** request byte-identical to baseline (+`logprobs=5`, deterministic seed) and `n=HACT_N−1` exploration samples at `HACT_TEMP=0.7`; exploration samples are decoded with the *ungoverned* grandparent parser (they must never touch the stateful governance session, qwen_hact.py:23-26, 224-229). Records join gov2 rows on `(test_id, step_idx)` via the session flush (governance_filter.py:1511-1527); no-call/parse-crash steps are flushed as **orphans** — "orphans ARE data (the parser-failure / no-call prediction target), not loss" (qwen_hact.py:30-35, 349-367).

### 7.5 Family (e): token logprobs

`logprob_features` (hact_sampler.py:99-150): over the primary completion — `lp_n_tokens, lp_sum, lp_mean, lp_min, lp_ppl=exp(−lp_mean)`, top-k margin mean/min (top1−top2 of each token's top-k dict), and the same stats restricted to the first `<tool_call>...</tool_call>` span located by cumulative token offsets. Graceful degradation when the server rejects/omits logprobs.

### 7.6 Marginal-diff signals (offline Stage 1.3)

`diff_signals.compute` (diff_signals.py:292-324): 17 features over `marginal_diff(old,new)` — lexical (`has_old`, `diff_edit_ratio` = 1−SequenceMatcher ratio, added fraction, added/dropped clause counts, novel-token fractions vs target and store, new values/values-unseen-in-store/entities/dates) and semantic (`diff_sim_max` — Stage 0's own quantity computed on the **added text only**, in the same whitened space, directly comparable to logged `sim_max`; raw-MiniLM variants; `diff_dSvn` — vn-entropy delta of the 8-neighbor Gram when the added text joins; `diff_rank_self_post` — is the marginal content itself retrievable post-write). Motivation measured on the harvest: 47/52 near-duplicate update/replace decisions had `sim_max ≥ 0.90` while carrying real new content — whole-blob geometry reads a genuine update as a duplicate (diff_signals.py:8-16). [CODE-VERIFIED]

---

## 8. H-Nav intervention policy

The complete decision process, derived from the real implementation. [CODE-VERIFIED]

### 8.1 Write admission (gov2, the current path)

```
WriteCandidate (parsed write call)
   │ preflight_ok := mirror predicts backend acceptance
   ▼
GeometryGate.evaluate
   ├─ preflight_ok ∧ store nonempty ∧ normalized-exact duplicate ──────────► NOOP (s0_exact_dup)
   ├─ store empty ─────────────────────────────────────────────────────────► ADD (s0_confident_add)
   ├─ r<δ ∧ sim_max>sim_high ∧ verbatim all present ∧ preflight_ok ────────► NOOP (s0_noop)
   │                                        └ ¬preflight_ok ───────────────► ADD (s0_noop_preflight_blocked)
   ├─ r>τ_t ∧ new verbatim ∧ own-probe rank ≤ 3 ───────────────────────────► ADD (s0_confident_add)
   │                        └ own-probe rank > 3 ──────────────────────────► ESCALATE (s0_unretrievable_escalate)
   └─ else ────────────────────────────────────────────────────────────────► ESCALATE (s0_escalate)

ESCALATE, policy = geometry_only ──────────────────────────────────────────► ADD (s0_escalate)   [validated fallback]
ESCALATE, policy = geometry_margin_entropy_v1  → compute_margin_entropy_signals, then Option A:
   locate := rank_self_med ≤ R (default 2)         strong := nmargin_med ≥ M (kv .10 / vec .05)
   interf := dH_mean ≥ D (.05) ∨ churn ≥ C (.34)   (interf forced False on KV smallstore < 5)
   dup_risk := sim_max ≥ sim_high − dup_eps (.02)
   ├─ locate ∧ strong ∧ ¬interf ───────────────────────────────────────────► ADD (s1_confident)
   ├─ locate ∧ strong ∧ interf ── rewrite available ───────────────────────► SAFE_REWRITE (s1_bounded_interference)
   │                            └ else ────────────────────────────────────► ADD + flag
   ├─ ¬locate ∧ dup_risk ── preflight_ok ──────────────────────────────────► NOOP (s1_shadowed_duplicate)
   │                      └ ¬preflight_ok ─────────────────────────────────► ADD + flag
   ├─ ¬locate ∧ ¬strong ∧ interf ──────────────────────────────────────────► ABSTAIN (s1_high_risk)  [observational: write executes, item tagged]
   └─ else ────────────────────────────────────────────────────────────────► ADD (s1_ambiguous_default) + flag
policy = ..._risk_v1: same NOOP region (geometric, identical to A); calibrated logistic risk over
   (nmargin_med, rank_self_med, dH_mean, n_eff_norm, sim_max, backend_kv) → ABSTAIN if ≥ thr_high,
   confident ADD if ≤ thr_low, flagged ADD between (admission_policy.py:552-569).
```

Invariants pinned in code comments (admission_policy.py:19-28): interference alone never rejects; NOOP always requires the preflight guard; total ambiguity defaults to a *flagged ADD, not suppression*.

**Mechanics of each action** (governance_filter.py:1719-1749, 2179-2209): NOOP → decoy call executes (`core_memory_retrieve_all()`), synthetic backend-format success replaces its output, original call string restored in chat history — the model's world-model stays consistent while the store is protected; SAFE_REWRITE → rewritten call executes for real, model sees the real result of the *rewritten* call; ADD/ABSTAIN → unchanged. Fallbacks: any boundary exception degrades to ungoverned pass-through with an `admission_error` log line — "governance must never break inference" (governance_filter.py:1660-1697).

### 8.2 Legacy escalation cascade (`legacy_full`)

Stage 1 NLI decision table over top-k=3 neighbors, both directions batched (governance_filter.py:911-965): `equivalent`/`derivable` → NOOP; `more_specific`/`contradiction` → REWRITE into an in-place supersede of the target (`build_rewrite_call`: KV `replace`, Vector `update` — the log line with `superseded_text` is "the only thing preventing data loss", governance_filter.py:764-767); `specificity_inconclusive` → keep both (ADD); all-neutral → Stage 2 (§7.2). Shadow modes at every stage log the full would-be decision and apply ADD.

### 8.3 SS5 placement / destructive guard (per-call passes after admission)

`_destructive_pass` (governance_filter.py:1962-2009) + `placement.plan_destructive` (placement.py:201-231): `*_memory_clear` **never passes** (decoy + synthetic success); core remove → archive-then-remove expansion when the archive-add preflights OK, else blocked; archival remove → last-copy NLI check (`last_copy_entailed`: some remaining entry entails the victim at τ=0.75, else blocked with `critical_information_loss` risk). `_placement_pass` (governance_filter.py:2011-2095): deterministic category routing (`classify_category`: identity-pattern substrings → core, else event_detail → rewritten to archival); core-full identity add → atomic move of the lowest-value core item (`value(m_i)=w1·(1−max_{j≠i} cos) + w2/(1+Δt)` in whitened space, no access-frequency term, placement.py:65-92); verbatim final check — extracted user-turn values missing from the outgoing write are appended `" (v1, v2)"` (length-capped). Expansions re-enter the boundary **Stage-0 only** (escalating a governance-generated write would be circular; duplicates dropped as `expansion_noop`, governance_filter.py:2116-2177).

### 8.4 Read gate (SS6)

§5.2. Trigger: ranked-read result with top1−top2 margin ≤ `GOV_READ_MARGIN`. Action ladder: `top1_pass` (trim to top-1) / `refined_top1` (one refinement round) / `return_set` (bounded ambiguous set). Affects only the tool-result payload; never memory; never the query the model issued.

### 8.5 Turn-level scaffolds (the interventions that won)

- **Write capture (SCAF / RAG M2)**: trigger = gated prereq turn whose decode produced **no calls**; action = inject one real `archival_memory_add` — per-turn verbatim (SCAF), or packed ~2000-char blobs (RAG `pack`, buffer flushed when the next turn would overflow; the chain-final tail is lost by causality and pre-registered as the live-vs-offline gap, qwen_rag.py:36-42, 419-462). KV keys are deterministic content slugs (`slug_key` — an opaque key would be unretrievable by construction under BM25-over-keys, qwen_scaffold.py:80-94). Controls: `shuffled_control`/`pack_shuffled` (same volume, scrambled words), `abstractive` (M6 LLM summarization, predicted to lose), `per_turn`.
- **Read injection (RAG M1)**: trigger = memory question turn whose decode produced no calls; action = inject one archival read probed with the user turn; model answers next step (§5.3). Controls: `read_irrelevant`, `instruction_only`, M3 pad, M9 rerank, M10 nudge/null.
- **Eviction (M7, vector)**: cap-rejected packed write retries as `[archival_memory_remove(victim), archival_memory_add(text)]`, victim = highest Jaccard-overlap incumbent (`redundancy`) or seeded `random` control (qwen_rag.py:464-524).

### 8.6 H_act policies

Implemented (qwen_hact.py:261-302): `shadow` (return primary untouched), `random_select` (seeded uniform choice among the N samples — a *control* arm), `majority` (replace primary with the modal R3-signature sample; degenerate signatures `no_call`/`parse_fail` can never form a winning bloc). Gate policies (`vote_margin_gate`, `hact_gate`, `factorized`, `full`) are named in config and threshold grids exist (`select_hact_thresholds.py` GRIDS: h_act_target ≥ {0.5..1.3}, vote_margin ≤ {0.125..0.5}) but the runtime raises if selected — they were to land after a GO/PARTIAL Stage-3 verdict which never came. [CODE-VERIFIED]

---

## 9. Relevant negative and positive BFCL findings

All numbers below are [REPORT-VERIFIED] from committed artifacts (paths given); mechanism claims cross-checked against code. Primary sources: `gorilla/HNAV_ACTION_SIDE_AUTONOMOUS_RESEARCH_REPORT.md` (living report, §1a scorecard), `gorilla/HNAV_STAGE1_IMPLEMENTATION_SUMMARY.md`, `gov_logs/HNAV_STAGE1.md`, `gov_logs/hnav_shadow/{PREREGISTRATION.md, STAGE3_REPORT.md, stage3_gate.json, SANITY_CHECK.md}`, `gov_logs/hnav_stage4/{INTEGRITY.md, VERDICT.md, cost_accounting.json, flip_tables.json}`, `gov_logs/hnav_autonomous/*`, `gov_logs/hnav_rag/*`, `gov_logs/ME_ABLATIONS_PHASE6_G2.md`.

### 9.1 Stage-by-stage verdict ladder

| Stage | Question | Verdict | Key numbers |
|---|---|---|---|
| Gov Plan-v2 G1 | does retrieval entropy add out-of-sample predictive value over geometry+margin? | **FAIL** — entropy terms dropped | raw-score softmax entropy is scale-degenerate (motivated `zscore_entropy`/`vn_entropy`, entropy_metrics.py:133-148) |
| Gov Plan-v2 G2 | is `gm_v1` (geometry+margin-entropy) non-inferior to `geometry_only` at −2pp? | **FAIL** (power/width failure: point estimates ≈0, CIs ±5–9pp at n=3; no secondary win) | gm_v1 vs geometry_only: kv +0.0167 [−0.0208,+0.0681], vec −0.0069 [−0.0814,+0.0596]; **all governed arms ≤ baseline** (e.g. geometry_only kv −0.0375); online NLI calls: 0 in all 12 governed runs |
| H-Nav Stage 1 (write-side coupling) | does the whole-blob-vs-marginal-diff geometry defect cost accuracy? | **PIVOT to action-side H_act** (H1 FAIL, H2 PASS, H3 FAIL) | see 9.2 |
| Stage 2 | build H_act observability (no verdict by design) | delivered | canonicalization, hact1 schema, logprob features |
| Stage 3 (shadow H_act gate) | does H_act predict harmful writes (T1=`must_suppress`)? | **NO_GO** | see 9.3 |
| Stage 4 (live A0/A1/A2 + alt10 rescue) | does majority-vote write rescue lift accuracy? | **promising but inconclusive / underpowered** (pre-committed label) | see 9.4 |
| Autonomous loop alts 1/7/9 | factorized entropy; parser robustness; backend-specific calibration | all **rejected** at (near-)zero compute | alt7: 0/3276 parse failures — no headroom; alt1: falsified by the Stage-3 ladder itself; alt9: 2–7 positives per backend — unfittable |
| alt5R write scaffold | offline falsifier | GO then **superseded before launch** by the tier-conditional correction | predicted +0.077/+0.171 → corrected standalone ≈ 0 (−0.006 kv / −0.018 vec); 15 GPU-h saved |
| RAG C1 (read scaffold M1) | live, 5 replicates, blind | running; interim rep01 only | vector read_verbatim +8.39pp vs co-run baseline (above the predicted +2.4..+5.6 band); ordering holds on vector (35 ≥ 32 > 22), fails on kv; **non-confirmatory by pre-registration** |

### 9.2 Stage 1 in detail — which geometry hypotheses failed and which survived

Labels over 4 575 decisions (student excluded): `inert_superseded` 85.4%, `may_suppress` 8.7%, **`must_write` 3.5% (159)**, **`must_suppress` 0.9% (41)**, `uncertain` 1.6%. Each store ref is written 3.32× on average (max 52); only the last write survives to question time — 47.0% of writes are `gone`, 38.3% `superseded`, 14.5% `terminal`. `must_write` concentrates in terminal/add (0.270) and is **exactly 0 in every non-terminal bucket**.

- **H1 (deployable headroom) FAIL**: widest deployable NOOP cell covers 2.64% of writes vs the 5% bar; harmful suppressions 5/121, Wilson95 [0.018, **0.093**] vs the 0.05 ceiling; the *entire* deployable frontier has negative expected ΔAcc (−0.005…−0.011). `preflight_ok` false for 57% of decisions.
- **H2 (the diff signal is real) PASS**: AUC nests geometry 0.5243 → +margin 0.5269 → **+diff 0.6027**; ΔAUC(diff | geometry+margin) = **+0.0764 [+0.0147, +0.1382]**; NC shuffle −0.0058 [−0.0205, +0.0089]; survives removal of `has_old` (+0.0713). The measured mechanism: 220 near-duplicate update/replace decisions at blob `sim_max ≥ 0.90` have median marginal `diff_sim_max` 0.1962 (median drop 0.745) — **the gate really does misread genuine updates as duplicates** — but `must_write` among those 220 = **0**: the defect sits entirely on decisions that cannot change an answer. "Fixing the geometry defect cannot recover accuracy that was never at risk."
- **H3 (cheap override) FAIL**: false-override rate 0.664 vs bar 0.20.
- **Oracle ceiling**: ground-truth-feature AUC 0.6164 vs answer-blind 0.6027 — only ~0.014 AUC is available to *any* online feature; write-side prediction is saturated.
- Live reality check (n=30 actual suppressions across arms): 19 `inert_superseded`, 8 `may_suppress`, 3 `must_write` → 3 harmful NOOPs.

### 9.3 Stage 3 in detail — which uncertainty hypotheses failed

T1 = `must_suppress`, n=1 253 decisions, **9 positives (0.72%)** vs `min_positives=25`. Nested OOF AUC ladder: the best nest is **M0 (metadata only, 0.586)**; geometry features *lower* AUC (M1/M2 0.462); every entropy nest ≤ 0.50. Headline `dAUC(M8−M4)` (H_act beyond vote controls) = **−0.013 [−0.037, +0.001]**; all univariate permutations Holm p = 1.0. The single positive T1 increment was **token logprobs beyond geometry+margin: +0.098 [+0.027, +0.173]** (M3−M2) — which vanishes once vote controls enter (M4−M3 = −0.056). T2 (predicting no-call/parse-fail steps): votes AUC **0.9983**, logprobs 0.927; entropy beyond votes **+0.0000045**. T3: chain-mean H_act is backend-stratified (kv 0.20–0.60, vector 0.02–0.11), descriptive only. Verdict NO_GO; gate policy arms A3–A6 were correctly never built (the runtime still raises on them, qwen_hact.py:80-84). Crucially: "the gate was not abandoned because the numbers were disappointing — it was abandoned because the *class it predicts does not exist* on this benchmark" (LAST_SESSION_UPDATES.md §3).

### 9.4 Stage 4 + alt10 — the closest thing to a positive, and why it is not validated

Survival-conditional matched pairs (n=290/backend): a2(majority) vs baseline **vector +0.0345 [−0.0107, +0.0786]**, Holm p 0.736, direction positive 3/3 replicates and matching the falsifier's +0.035 prediction almost exactly; the *primary* contrast a2 vs a1(random-select, equal compute) ≈ null (+0.014 vector) — most of the gain is generic extra-sampling. The pre-registered negative control (shuffled sample-to-step assignment) **did not collapse** (~70% of the offline rescue metric survives; the shuffled variant still clears the 0.02 GO bar) → a substantial fraction of "rescue" is non-content-specific carriage of short gold strings (median 7 chars). Cost: 234 extra calls / ~558k completion tokens per net recovered answer, wall clock 2.48×. Mechanistic strata (A4): rescued vector questions move from `not_carried` into `carried_retrievable`; kv rescues land `carried_not_retrievable` — consistent with the BM25-over-key-names defect.

### 9.5 The loss decomposition and the tier-conditional finding (the program's pivotal results)

- 77–81% of question instances (kv 81.0% / vector 77.3%) have the gold fact **absent from the final store**; p(correct | not carried) ≈ 0.043–0.049. ~88% of the uncarried gold facts were stated **verbatim** in the prereq conversation — capture failure, not knowledge failure.
- Tier-conditional conversion (pooled, `tier_conditional_pooled.json`): gold in **core** → p(correct) 0.556 (kv) / 0.701 (vector); gold **archival-only** → **0.0167 / 0.0000**; not carried → 0.049 / 0.043. Only 6/155 vector question entries issue any memory call; exactly **1** touches archival. "Archival memory is a write-only black hole at question time."
- Consequence (frozen methodology): every Δanswerability→ΔAcc conversion must be tier-conditional and reported as a range over the read-conversion factor c — the tier-blind p̂=0.409 conversion over-predicted alt5R by the entire effect (+0.171 → ≈0).
- Strict regrade: 17–19% of lenient passes are context-field leaks (kv 0.1118→0.0925; vector 0.1226→0.0989) — grader-robustness caveat on all lenient numbers.
- Corrected baseline: vector reproduces the official leaderboard exactly (0.1226); kv is 4.9pp below official, fully explained by the dead `student` chain (0 writes in all 30 cells; a behavioral, not harness, failure — 811-char monologue turns answered with monologues and zero tool calls).

### 9.6 Method-failure vs benchmark-headroom-failure (the central distinction)

Explicit in the reports and enforced here: three mechanisms (write gating, action uncertainty, and the geometry-defect fix) "all failed for the same reason: the classes they police (`must_write` 3.5%, `must_suppress` 0.7%) are too rare to move a score" (LAST_SESSION_UPDATES.md §11). The strongest defensible claim (REPORT §27.12): on BFCL v4 Memory the binding constraint is the capture→read pipeline; capture and read are complements — either alone ≈ 0. **Method-failures proper** (signal genuinely absent): scalar H_act on T1 (AUC ≤ 0.5 vs controls); retrieval-score softmax entropy (scale-degenerate, G1); backend-specific calibration (unfittable at 2–7 positives). **Headroom-failures** (signal present, class empty): marginal-diff geometry (H2 PASS, 0 must_write among detections); H_act gating (its target class has 9 positives); write-side governance generally (oracle ceiling 0.014 AUC). **Survivors**: vote features for no-call prediction (0.998 AUC → alt10/scaffolds); token logprobs as a cheap T1 feature (+0.098 before controls); the diff detector as diagnostics; deterministic capture/read scaffolding; the counterfactual-labeling + tier-conditional methodology itself.

### 9.7 Discrepancies found (code vs docs / internal)

1. **Flip counts unreconciled** [REPORT-VERIFIED]: `flip_tables.json` reports a2-vs-baseline vector positives 33; `cost_accounting.json` and REPORT §26 report 28 (negatives agree at 41). No artifact reconciles the subsets; a port must define one flip denominator.
2. **Grader docstring vs regex** [CODE-VERIFIED]: `agentic_checker.py` docstring claims whitespace is stripped; the regex `[\,\.\/\-\_\*\^\(\)]` does not include `\s` — whitespace is preserved (matching works for dates only because commas are stripped).
3. **Vector core-dump dead branch** [CODE-VERIFIED]: `memory_vector._dump_core_memory_to_context` checks `if not self.core_memory:` on a `VectorStore` with no `__len__`, so the "no content" branch is dead and an empty core dumps as `{}` into the system prompt.
4. **Stage-3 shadow scores void as replicates** [REPORT-VERIFIED]: pinned-seed pseudo-replication made the hact arm "effectively ONE trajectory draw" (52% of (test_id, step) keys byte-identical across replicates); SANITY_CHECK.md voids those scores as auxiliary baselines — decision-neutrality itself HELD (all 1 253 decisions `applied:"none"`).
5. **`server_created` is not a server identity** [REPORT-VERIFIED]: C1 Amendment 1 correction — vLLM 0.9.1's `/v1/models` `created` is a request timestamp; server-instance continuity is asserted, not verified, across campaigns.
6. **NC semantics repaired mid-program** [REPORT-VERIFIED]: Stage-3's PARTIAL clause originally depended on NC2 (entropy shuffle), which cannot collapse an entropy-free delta — corrected pre-unblinding to NC2b (vote shuffle). Port the corrected form.
7. **T2 undercount** [REPORT-VERIFIED]: in the shadow campaign the orphan flush was skipped when decode *raised*, undercounting the parse-crash subclass of T2 positives; fixed in the current `qwen_hact.decode_execute` (flush-then-reraise, qwen_hact.py:369-381).
8. **governance_filter.py module docstring is stale** [CODE-VERIFIED]: lines 12-14 still describe Stages 1/2 as "NOT implemented in this phase"; both are implemented below in the same file (and retired to legacy by Plan v2). Trust the code and the section-level comments, not the header.

---

## 10. Benchmark-independent H-Nav core

Layer-1 abstractions, **derived from types that already exist in the code** (this decomposition is not aspirational — the middleware was explicitly built producer-agnostic; `memory_mutation.py:9-16` names three producers, only one of which is BFCL tool calls).

| Abstraction | Existing type/function | What it needs from the benchmark |
|---|---|---|
| `MemoryCandidate` | `MemoryMutationCandidate` (memory_mutation.py:59-78): operation/backend/tier/raw_key/raw_value/normalized_key/normalized_value/source/raw_call/metadata | a conversion point from the host's write API |
| `AdmissionDecision` | `memory_mutation.AdmissionDecision`: action ∈ {ADD, NOOP, SAFE_REWRITE, ABSTAIN}, stage, confidence, reason_code (closed vocab), rewritten_call, diagnostics | an application point that can suppress/replace a write and fake its success |
| `MemoryStoreMirror` | `GovernanceCache` (+`MemoryItem`) | item enumeration + a rehydration source; capacity/length constants |
| `GeometryFeatures` | `Stage0Signals` + `compute_signals` + `AbttTransform` + `ThresholdState` | an embedding fn; a whitening artifact fitted on in-domain text |
| `RetrievalSimulator` | `simulate_kv`/`simulate_vector` + `probe_scores`/`delta_h_for_probes` | a *faithful offline replica of the host's retriever* (the single hardest porting requirement) |
| `RetrievalUncertainty` | `MarginEntropySignals` + `entropy_metrics.{nmargin,n_eff_norm,disp,churn,rank_self,median,zscore_entropy,vn_entropy}` | nothing beyond the simulator |
| `ProbeGenerator` | `probe_gen.{generate_decision_probes,generate_item_probes,ProbeConfig}` | the current user/query text (anti-circularity source) |
| `DiffFeatures` | `diff_signals.{marginal_diff,compute}` | old-text lookup for update ops |
| `ReadGate` / `NavigationDecision(read)` | `read_gate.{gate_read,ReadGateOutcome,parse_ranking,serialize_ranking}` | ranked scores of the host retriever + a payload rewrite point |
| `ActionUncertainty` | `action_space.{canonicalize,vote_features_all}` + `hact_sampler.{HactConfig,logprob_features,build_hact_record}` | N-sample generation + an action-string parser |
| `PlacementPolicy` | `placement.{classify_category,value_scores,pick_eviction_victim,plan_destructive,verbatim_final_rewrite}` | tier semantics + call builders |
| `CaptureScaffold` / `ReadScaffold` | the policy cores of `qwen_scaffold.py` / `qwen_rag.py` (turn-state machine + `slug_key` + pack buffer + probe builder) | "no action was taken this turn" signal + a call-injection point |
| `AuditLog` | `log_gov_record` + gov2/hact1/scaffold/rag record schemas | a filesystem |
| `CounterfactualLabeler` | `label_outcomes_hnav.py` (S_with/S_without worlds, necessity/damage, fate ∈ {terminal, superseded, gone}, targets must_write/must_suppress/may_suppress/inert_superseded/uncertain) + `hnav_answer_index` (top-3 carrier criterion, conversion factor p̂) | final stores, question set, the host grader |

### Component-portability table

| Component | Current BFCL implementation | H-Nav core or BFCL-specific? | Portable unchanged? | Required adaptation |
|---|---|---|---|---|
| ABTT whitening (`AbttTransform`, `compute_abtt.py`) | fitted on BFCL prereq conversations, MiniLM-384d, D=16 | Core method, BFCL-specific *artifact* | Method yes, artifact **no** | Refit `mu`/`U_top` on the target corpus + target embedder; keep D a validated knob |
| `compute_signals` (sim_max, QR residual r, τ_t) | governance_filter.py:661-698 | Core | **Yes** (pure numpy over the mirror) | none beyond embedding dim |
| `decide` Stage-0 rule + `GovConfig.validate` | governance_filter.py:701-738, 273-312 | Core | **Yes** | thresholds must be re-calibrated (shadow run first) |
| Exact-dup fast path + retrievability floor | geometry_gate.py | Core | Mostly | floor needs the host retrieval simulator |
| Verbatim-value gate (regex + literal search) | governance_filter.py:553-610 | Core (English-regex biased) | Yes | extend patterns per domain/language |
| `GovernanceCache` mirror + `_observe` | governance_filter.py:405-520, 2260-2375 | Core pattern, BFCL success-string matching | Pattern yes, matching **no** | rewrite `_observe` against the host's write-result contract (or use return values directly if the host is a library, not a tool protocol) |
| `preflight_would_succeed` | governance_filter.py:1327-1352 | BFCL-specific (caps, key regex) | No | re-derive from host store constraints |
| `simulate_kv` / `simulate_vector` | retrieval_sim.py | BFCL-specific fidelity contracts | **No** | reimplement per host retriever; verify byte/rank fidelity like retrieval_sim's docstring does |
| `entropy_metrics` (nmargin, n_eff_norm, disp, churn, rank_self, median, zscore_entropy, vn_entropy) | entropy_metrics.py | Core | **Yes** (pure) | choose backend-appropriate nmargin normalization |
| `delta_h_for_probes` (dH_self / dH_neighbor) | retrieval_sim.py:174-213 | Core | Yes given a simulator | none |
| `probe_gen` templates + paraphrase channel | probe_gen.py | Core, templates English/BFCL-flavored | Mostly | rewrite templates for host query style; keep anti-circularity invariant |
| Option A rule / risk_v1 | admission_policy.py:253-306, 552-569 | Core | **Yes** | thresholds via the SS13-style calibration protocol, not copied |
| `MemoryMutationCandidate` / `AdmissionDecision` / boundary | memory_mutation.py, admission_policy.py:416-550 | Core | **Yes** | write a host adapter producing candidates |
| NOOP mechanics (decoy call + synthetic success + shadow ids) | governance_filter.py:1266, 1355-1364, 64-68 | BFCL-specific mechanism, core *idea* | No | host equivalent: intercept the write API return path; if the agent sees tool results, fabricate the host's success shape; if not, simply drop the write |
| SAFE_REWRITE (KV key suffix) / canonicalize_candidate | admission_policy.py:309-337, governance_filter.py:1024-1069 | Core idea, KV-shaped | Partially | redefine "non-destructive dual representation" for the host schema |
| Read gate | read_gate.py | Core (pure over (score, ref, text) lists) | **Yes** | payload parse/serialize per host result shape |
| Placement/eviction/destructive guard | placement.py + session passes | Core logic, BFCL tiers | Mostly | map to host tiers; drop if host has one tier |
| NLI machinery (NliScorer, last-copy, blobdiff) | semantic_entropy.py:186-250, placement.py:177-186, recsum_blobdiff.py | Core, offline-only by Plan v2 decision | Yes | keep offline (judge/labeler), do not put back online without cause |
| `action_space` canonicalization + vote features | action_space.py | Core, parser reuses BFCL op tables | Mostly | swap `parse_call`+op tables for the host action grammar |
| H_act sampling runtime | qwen_hact.py | BFCL/vLLM-specific (completions API, seeds, logprobs) | No | reimplement against host LLM client; keep primary/exploration split + orphan discipline |
| Write scaffold / RAG read scaffold policy | qwen_scaffold.py, qwen_rag.py | Core policy, BFCL hook | Policy yes, hook no | host needs a "turn ended with no memory action" hook + call injection |
| MIG reranker | mig_reranker.py | Separate project | n/a | precedent only |
| Counterfactual labeler + answer index | label_outcomes_hnav.py, hnav_answer_index.py | Core *methodology*, BFCL grader/store formats | Methodology yes | rebuild S_with/S_without over host snapshots + host grader |
| Statistics harness (nested AUC ladder, NC shuffle, Holm, grouped bootstrap, Wilson) | evaluate_hact_shadow.py, calibrate_margin_entropy.py, analyze_gov_replicates.py, counterfactual_noop.py | Core | **Yes** (feature-name changes only) | group bootstrap by the host's dependency unit (scenario/chain) |
| Handlers / registry / env-var config surface | qwen_*.py, GOV_*/HACT_*/SCAF_*/RAG_* | BFCL-specific | No | one adapter class per host; keep the env-driven single-registry pattern |

---

## 11. BFCL-specific dependencies

Everything below must be treated as **glue** a port must replace. [CODE-VERIFIED unless noted]

1. **The `@final` multi-turn loop + hook surface** (`base_handler.py`): H-Nav never patches the loop; it exists entirely in the overridable methods listed in §3. The loop (`inference_multi_turn_prompting`, base_handler.py:393-682; FC twin :94-391): per turn, a `while True` step loop calls `_query_prompting → _parse_query_response_prompting → _add_assistant_message_prompting → decode_execute → execute_multi_turn_func_call → _add_execution_results_prompting`; it breaks on an empty decode, a decode exception, or the step cap (`count > MAXIMUM_STEP_LIMIT` post-increment ⇒ effectively 21 steps/turn, then force-quit which fails the entry). Tool execution is delegated to `execute_multi_turn_func_call` — the handler never runs tools itself. **Only memory-prereq entries flush memory to disk** (base_handler.py:369-376/:660-667); answer-entry writes are discarded in RAM. A port must find equivalent seams and know its own step budget before injecting scaffold steps.
2. **Tool-call string protocol**: decoded calls are Python-call strings parsed by `ast` (`parse_call`); the op tables (`KV_WRITE_OPS`, `VECTOR_WRITE_OPS`, `*_OBSERVED_OPS`, governance_filter.py:1236-1261) hard-code the six memory tool names per backend and their parameter orders.
3. **Backend success strings**: `_observe` matches exact strings (`"Key-value pair added."`, `"ID <n> updated."`, ...; governance_filter.py:1270-1274) — brittle by design (never over-count).
4. **Snapshot mechanics** (`memory_api_metaclass.py:24-76`, `@final _prepare_snapshot`): layout `<result_dir>/<model>/agentic/memory/<backend>/memory_snapshot/` with per-prereq checkpoints under `prereq_checkpoints/` and the rolling head `<scenario>_final.json`; first prereq entry of a chain starts clean (`is_first_memory_prereq_entry` → `None`); a *missing* head for a later entry is a soft failure (memory silently starts empty — the gov session mirrors that warn-and-empty behavior, qwen_gov.py:151-160). Vector snapshots persist text only; embeddings are re-computed on load (`load_from_snapshot`, memory_vector.py:356-372). Sidecar `<scenario>_gov_state.json` carries the probe caches.
5. **Tiers and caps**: core 7 entries × 300 chars, archival 50 × 2000 (constants mirrored at governance_filter.py:56-60); KV key grammar `^[a-z]+(_[a-z0-9]+)*$`.
6. **Retrieval quirks the signals are tuned to**: KV BM25Plus over *key names only* (values unscored), Vector exact IP over MiniLM embeddings with scores exposed to the model.
7. **Test-ID grammar / entry structure**: ids are `memory_<backend>[_prereq]_<idx>-<scenario>-<local_idx>` (rewritten from raw `memory_*` by `process_memory_test_case`, utils.py:714-752); `is_memory`, `is_memory_prereq`, `is_first_memory_prereq_entry` ("prereq" in id and endswith "-0"), `extract_memory_backend_type` drive every gate. Structure: per scenario, a chain of prereq (fill) conversations (37 per backend; 9–13 user turns each; never graded — their only product is the flushed snapshot) followed by 155 single-question answer entries (`BFCL_v4_memory.json`; 5 scenarios: student 50 / customer 30 / finance 25 / healthcare 25 / notetaker 25), linked by `depends_on`. 155 × 3 backends = 465 scored entries.
8. **Grader** (`agentic_checker.py`, 49 lines): both sides are "standardized" (strip `, . / - _ * ^ ( )`, lowercase, `'`→`"` — whitespace **kept**, contra the docstring), then correctness = any ground-truth alternative found as a `\b`-bounded literal substring anywhere in **the last non-tool-call message** of the run (eval_runner.py:109-148; no such message ⇒ `agentic:no_last_message`). Disjunctive alternatives; binary per entry. This is why packed *verbatim* capture beats abstractive summaries (gold answers are short entities, median 7 chars), why SE text-clustering uses content-token containment, and why the strict answer-field regrade exists as a guard.
9. **Prompting stack**: Qwen chat template + `<tool_call>` JSON extraction (qwen_fc.py); H_act's logprob features assume a vLLM legacy-completions response shape.
10. **Memory system prompt** (`constants/default_prompts.py:91-99`, `MEMORY_BACKEND_INSTRUCTION_CORE_ARCHIVAL`): "You have access to an advanced memory system, consisting of two memory types 'Core Memory' and 'Archival Memory'. ... You should actively manage your memory data ... The Core memory is limited in size, but always visible to you in context. The Archival Memory has a much larger capacity, but will be held outside of your immediate context" — followed by the core-memory dump, injected **once per entry before turn 0** (`add_memory_instruction_system_prompt`, model_handler/utils.py:610-650; base_handler.py:134-145) and never refreshed mid-entry. Notably it discloses no numeric caps and gives no instruction to write during prereq turns — the under-writing the scaffolds compensate for is downstream of this prompt. `MAXIMUM_STEP_LIMIT=20` (effective 21) bounds injected extra steps. Answer entries additionally get the `{'answer': ..., 'context': ...}` format instruction (default_prompts.py:79-80) — the field the strict regrade keys on.
11. **ABTT artifact**: fitted on the BFCL prereq-conversation pool (§6) — in-domain by construction, invalid for any other corpus.

---

## 12. Hypothesized EvoMemBench mapping

No EvoMemBench APIs are invented here; this is a benchmark-agnostic checklist of *what must be located* in the real repo, with the BFCL analogue that defines the requirement. All items [EVOMEMBENCH-HYPOTHESIS] with BFCL anchors [CODE-VERIFIED].

| # | Must locate in EvoMemBench | BFCL analogue that defines the contract |
|---|---|---|
| 1 | **Incoming-candidate conversion point** — where a proposed memory write (add/update/delete, whatever the API) is representable *before* it mutates the store | `decode_execute` → `build_candidate` → `from_write_candidate` |
| 2 | **Memory-store representation + enumeration** — how to list current items (id, text, tier/collection, timestamps) and how the store persists across episodes/time steps | `GovernanceCache.rehydrate` from `<scenario>_final.json`; `MemoryItem` |
| 3 | **Write-decision hook** — a seam where a write can be (a) executed, (b) suppressed with a plausible success, or (c) replaced by a rewritten write, without corrupting the agent's dialogue state | `govern_calls` + `patch_results` (decoy + synthetic success) |
| 4 | **Embedding interface** — the embedder the *benchmark's own retrieval* uses, and whether H-Nav may reuse it (BFCL reuses MiniLM but decides in a separate whitened space) | `semantic_entropy._get_encoder` + `AbttTransform` |
| 5 | **Pre-truncation retrieval scores** — access to the ranked candidate list *with scores* before it is cut to top-k / stuffed into the prompt | KV `ranked_results`, Vector `similarity_score`; `read_gate.parse_ranking` |
| 6 | **Query construction point** — where the retrieval query is formed from the task/turn, so probes can be derived from *user/task text only* (anti-circularity) | `session.user_text` stashing (qwen_gov.py:168-196) |
| 7 | **Context-injection point** — where retrieved items enter the prompt (to trim/reorder/annotate) | `_add_execution_results_prompting`; MIG precedent |
| 8 | **Evaluator** — per-question correctness, and whether an *offline* re-grade over a reconstructed store is possible (required for counterfactual labels) | `agentic_checker` + `hnav_answer_index` + `label_outcomes_hnav` |
| 9 | **"No action taken" signal** — detectable end-of-turn with no memory write/read, for the scaffolds | empty `decode_execute` result (qwen_scaffold.py:232-244) |
| 10 | **Store evolution driver** — what makes memory *evolve* (updates, contradictions over time, drift); this determines which failure classes (§18) are actually populated | BFCL analogue is weak (mostly appends + self-overwrites; see §9) — this is precisely why EvoMemBench is attractive |
| 11 | **Retriever replica feasibility** — can the retriever be run offline on an arbitrary provisional corpus (list of texts) cheaply? If not, `dH/churn/rank_self` signals are impossible as designed | `retrieval_sim` fidelity contracts |
| 12 | **Dependency unit for statistics** — the correlated-unit grouping (BFCL: scenario chain) for grouped bootstrap/permutation | `evaluate_hact_shadow` chain-grouped bootstrap |

---

## 13. Proposed EvoMemBench H-Nav architecture

[EVOMEMBENCH-HYPOTHESIS], decomposition [CODE-VERIFIED] as the factoring the BFCL code already has. Principle: **benchmark → thin adapter → H-Nav core**, with the core identical to the modules in §10.

```
EvoMemBench runtime                     Adapter layer (to write)              H-Nav core (port of middleware/)
───────────────────                     ────────────────────────              ────────────────────────────────
agent write API ────────────────────►  CandidateAdapter                       hnav.types
                                        .to_candidate(api_call)                 MemoryCandidate, AdmissionDecision
store state / persistence ──────────►  StoreMirrorAdapter                     hnav.mirror.StoreMirror
                                        .enumerate(), .on_write_result()        (embeds via EmbeddingAdapter)
benchmark embedder ─────────────────►  EmbeddingAdapter (.encode)             hnav.geometry
                                                                                Whitener(mu,U_top,D) [refit!]
                                                                                GeometryFeatures, stage0_decide
benchmark retriever ────────────────►  RetrieverReplica (.rank(corpus,q,k))   hnav.retrieval
                                        [must pass a fidelity test]             probes, margins, dH, churn,
                                                                                rank_self, zscore/vn entropy
write path seam ────────────────────►  WriteInterceptor                       hnav.admission
                                        .apply(AdmissionDecision)               AdmissionBoundary(policy)
read result seam ───────────────────►  ReadInterceptor                        hnav.readgate.gate_read
                                        .rewrite_payload(...)
LLM client ─────────────────────────►  SamplerAdapter (primary+explore)       hnav.action
                                                                                canonicalize (host grammar),
                                                                                vote_features, logprob_features
turn loop ──────────────────────────►  TurnObserver (.no_action_taken)        hnav.scaffold
                                                                                CapturePolicy, ReadPolicy
everything ─────────────────────────►  —                                      hnav.audit (JSONL, schemas §19)
offline: snapshots + grader ────────►  EvalAdapter                            hnav.labels (S_with/S_without,
                                                                                must_write/... targets)
                                       hnav.controller.HNavController orchestrates (see §14.5)
```

Invariants to preserve, in priority order [INFERENCE from what the BFCL discipline caught]: (1) shadow-first — every new signal ships with a mode that computes+logs but never intervenes; (2) mirror never over-counts — observe only confirmed write successes; (3) suppression never masks a real backend error (preflight guard); (4) probes never derive from model-generated content (anti-circularity); (5) interventions degrade to pass-through on any internal error; (6) decision features are hash-stamped at decision time (leakage control); (7) deterministic seeds everywhere sampling exists.

---

## 14. Detailed pseudocode

Faithful to the BFCL implementation; host-specific calls are marked `ADAPTER`. [CODE-VERIFIED formulas; structure INFERENCE]

### 14.1 Geometry feature extraction

```python
class Whitener:                      # governance_filter.AbttTransform
    # mu: (d,) corpus mean; U: (d, D) top-D covariance eigendirections, orthonormal.
    # REFIT on the host corpus with the host embedder (compute_abtt.py recipe:
    # in-domain text pool, sentence-segmented, min length filter, record file hashes).
    def apply(self, v):
        w = (v - self.mu); w -= self.U @ (self.U.T @ w)
        n = norm(w);  return zeros_like(w) if n < 1e-12 else w / n

def geometry_features(candidate, mirror, thresholds, cfg, whiten, retriever_replica):
    # -- exact-duplicate fast path (no encoding)  [geometry_gate.py:129-146]
    if preflight_ok(candidate, mirror) and mirror.size > 0 \
       and any(norm_text(it.text) == norm_text(candidate.text) for it in mirror.items):
        thresholds.update(candidate.tier, mirror.density(candidate.tier))  # keep EMA sequence
        return Signals(sim_max=1.0, r=0.0, decision_hint="NOOP:exact_dup")

    v_w = whiten(ADAPTER.encode([candidate.text])[0])
    M   = mirror.whitened_matrix()            # (n, d), BOTH tiers
    sims = M @ v_w
    sim_max = max(sims)
    Q = qr_basis(M.T)                         # orthonormal basis of stored items
    r = norm(v_w - Q @ (Q.T @ v_w))           # residual: what memory cannot explain
    rho   = mirror.density(candidate.tier)
    tau_t = thresholds.update(candidate.tier, rho)   # tau*=tau_min+tau0*exp(-lam*rho); EMA alpha

    values = extract_verbatim_values(candidate.text)      # regex ladder, most-specific first
    if values:  hits, misses = literal_search(values, mirror.texts())   # NO embeddings
    else:       hits, misses = content_token_overlap(candidate.text, mirror.texts())

    # old/new + marginal-diff extension (offline in BFCL — diff_signals.compute):
    old = mirror.text_at(candidate.ref) if candidate.op in ("update","replace") else None
    diff = marginal_diff(old, candidate.text)             # difflib opcodes + clause set-diff
    diff_sim_max = cos_max(whiten(ADAPTER.encode([diff.added_text])[0]),
                           whitened_store) if diff.added_text else 1.0
    # + lexical counts (new values/entities/dates, novel-token fractions),
    # + diff_dSvn = vn_entropy(neigh ∪ added) - vn_entropy(neigh)   (8 nearest, raw space)
    # + diff_rank_self_post via retriever_replica
    return Signals(sim_max, r, tau_t, rho, values, hits, misses, sims, diff_features)
```

### 14.2 Write navigation (outputs restricted to what current H-Nav supports)

```python
# Action space is EXACTLY: ADD | NOOP | SAFE_REWRITE | ABSTAIN   [memory_mutation.py:25]
def write_navigate(candidate, mirror, cfg, sig, me):        # me may be None (no escalation)
    pf = preflight_ok(candidate, mirror)
    # ---- Stage 0 [governance_filter.decide + geometry_gate floor]
    if sig.decision_hint == "NOOP:exact_dup":               return NOOP("s0_exact_dup")
    if mirror.size == 0:                                    return ADD("s0_confident_add")
    if sig.r < cfg.delta and sig.sim_max > cfg.sim_high and not sig.misses:
        return NOOP("s0_noop") if pf else ADD("s0_noop_preflight_blocked")
    if sig.r > sig.tau_t and sig.misses:
        rank = retrievability_floor(candidate, mirror)      # own-probe rank in provisional tier
        if rank is None or rank <= 3:                       return ADD("s0_confident_add")
        escalate = True                                     # "s0_unretrievable_escalate"
    else: escalate = True                                   # "s0_escalate"
    if cfg.policy == "geometry_only":                       return ADD("s0_escalate")
    # ---- Stage 1 Option A [admission_policy.apply_option_a_rule]
    locate = me.rank_self_med is not None and me.rank_self_med <= cfg.R
    strong = me.nmargin_med  is not None and me.nmargin_med  >= cfg.M[backend]
    interf = (me.dH_mean or -inf) >= cfg.D[backend] or (me.churn or -inf) >= cfg.C[backend]
    if backend_smallstore(candidate, mirror): interf = False
    dup_risk = sig.sim_max >= cfg.sim_high - cfg.dup_eps
    if locate and strong and not interf:                    return ADD("s1_confident")
    if locate and strong and interf:
        rw = safe_rewrite(candidate, mirror, sig)           # identity-only rewrite; value byte-identical
        return SAFE_REWRITE(rw) if rw else ADD_flagged("s1_bounded_interference")
    if not locate and dup_risk:
        return NOOP("s1_shadowed_duplicate") if pf else ADD_flagged(...)
    if not locate and not strong and interf:                return ABSTAIN("s1_high_risk")  # observational
    return ADD_flagged("s1_ambiguous_default")
```

### 14.3 Retrieval uncertainty — exact current formulations

```python
# MAIN (live, decision-relevant):  [entropy_metrics.py + admission_policy.py]
def retrieval_uncertainty(candidate, mirror, user_text, retriever_replica, cfg):
    probes = decision_probes(user_text, verbatim_values)      # user text ONLY (anti-circular)
    if backend == "kv": probes += item_probes(key=candidate.key)   # key channel (BM25 reality)
    pre  = mirror.tier_corpus(candidate.tier)                 # per-tier, backend-faithful
    post = pre + [candidate.corpus_repr]                      # KV: key name; Vector: text
    rows = []
    for p in probes:
        rp, ro = retriever_replica.rank(post, p, k=5), retriever_replica.rank(pre, p, k=5)
        rows.append(dict(rank_self = rank_of(candidate, rp, sentinel=k+1),
                         margin    = rp[0].s - rp[1].s,
                         nmargin   = (rp[0].s-rp[1].s)/(abs(rp[0].s)+eps) if kv else rp[0].s-rp[1].s,
                         H_pre     = softmax_entropy(scores(ro), T), H_post = softmax_entropy(scores(rp), T),
                         locating  = p.channel in LOCATE_CHANNELS[backend]))
    L = [r for r in rows if r.locating]
    agg = dict(rank_self_med = median(r.rank_self for r in L),      # median, never min-pool
               nmargin_med   = median(r.nmargin  for r in L),
               dH_self       = mean(r.H_post - r.H_pre for r in L),
               n_eff_norm    = exp(mean(r.H_post for r in L)) / min(k, len(post)),
               disp          = median(std(scores)/|mean(scores)| for r in L))
    # interference: neighbors' OWN cached probes
    for nb in topk_neighbors(sig.sims, k=3, forced=[candidate.ref]):
        dH_nb = mean(H(post,p) - H(pre,p) for p in nb.probes)
        churn_nb = mean(jaccard_distance(top3_ids(pre,p), top3_ids(post,p)) for p in nb.probes)
    agg.dH_mean, agg.churn = mean(dH_nb), mean(churn_nb)
    return agg

# ALTERNATIVES (implemented, offline / logged-only — port as diagnostics first):
#   softmax H at temperature T, n_eff=exp(H)      — decision-inert by pinned design
#   zscore_entropy(scores,k)                       — affine-invariant fix for scale degeneracy
#   vn_entropy(embeddings) / effective_rank        — semantic neighborhood rank (works even
#                                                    when the host retriever is lexical)
#   legacy Stage-2 rule: all-top-1 AND min_margin > tau, one canonicalization retry
#   read-side: margin = top1-top2 of the LIVE result (the only live read trigger)
```

### 14.4 Read navigation

```python
def read_navigate(query, ranked_payload, cfg, retriever_replica):   # read_gate.gate_read
    ranked = parse_ranking(payload)                  # [(score, ref, text)] desc
    if len(ranked) < 2:                              return PASS
    margin = ranked[0].s - ranked[1].s
    if margin > cfg.read_margin:                     return TOP1(payload=[ranked[0]])   # normal
    amb = [r for r in ranked if ranked[0].s - r.s <= cfg.read_margin][: clamp(cfg.set_max,2,4)]
    tok = first_token_distinct_to_one_candidate_and_present_in_query(query, amb)
    if tok is None:                                  return RETURN_SET(amb)             # let model choose
    rescored = retriever_replica.rank([r.text for r in amb], f"{query} {tok}")          # ONE round
    return REFINED_TOP1(rescored[0]) if rescored[0].s > rescored[1].s else RETURN_SET(amb)

def read_scaffold(turn, cfg):                        # qwen_rag.py policy core
    if turn.is_question and turn.decoded_actions == [] and not turn.already_injected:
        probe = turn.user_text (+ core_context_words if cfg.core_augmented)
        inject one host retrieval call(query=probe, k=cfg.top_k); mark injected
        # optional post-hoc payload transforms: rerank fetch_k→top_k, pad control, nudge
```

### 14.5 Combined HNavController

```python
class HNavController:
    # state: mirror, thresholds, whitener, retriever_replica, cfg, audit, turn_state
    def on_episode_start(self, store_snapshot):
        self.mirror.rehydrate(store_snapshot, encode, whiten); self.thresholds.seed(self.mirror)
    def on_user_turn(self, text): self.user_text = text          # BEFORE any model output
    def on_model_actions(self, actions):                          # HOOK 1
        out = list(actions)
        for i, a in enumerate(actions):
            cand = CandidateAdapter.to_candidate(a)
            if cand is None: continue
            sig = geometry_features(...); me = retrieval_uncertainty(...) if escalated
            dec = write_navigate(...)
            self.audit.log_decision(...)                          # gov2-equivalent, ALWAYS
            if not self.cfg.shadow: out[i] = apply(dec, a)        # decoy / rewrite / passthrough
        out = destructive_guard(out); out = placement_pass(out)   # optional SS5 layer
        if out == [] and self.cfg.scaffold: out = capture_or_read_scaffold(self.turn_state)
        return out
    def on_action_results(self, actions, results):                # HOOK 2
        results = substitute_synthetic_successes(results)
        for a, r in zip(actions, results): self.mirror.observe(a, r)   # exact-success only
        if self.cfg.read_gate: results = [read_navigate(...) for gated reads]
        return results
    def on_generation(self, prompt):                              # optional H_act layer
        primary = ADAPTER.generate(prompt, seed=stable_seed(...), logprobs=k)
        explores = ADAPTER.generate(prompt, n=N-1, T=explore_T, seed=...)
        votes = vote_features_all([canonicalize(s) for s in [primary]+explores])
        self.audit.log_hact(votes, logprob_features(primary))     # join key: (episode, step)
        return policy_select(primary, explores, votes)            # shadow → primary untouched
```

### 14.6 Logging event schema (designed for causal/failure analysis)

One JSONL stream per sub-system, joinable on `(episode_id, step_idx, call_idx)`; every intervention logs *what would have happened* and *what did*. Modeled on the gov2 record (governance_filter.py:1783-1839), hact1 (hact_sampler.py:155-213), scaffold/rag logs.

```jsonc
// decision (write admission) — one per candidate
{"event":"decision","schema":"gov2-port","candidate_id":"sha16(episode|step|call|raw)",
 "episode_id":..., "step_idx":..., "call_idx":..., "backend":..., "op":..., "tier":...,
 "candidate_ref":..., "candidate_text":"FULL TEXT (replay reconstructs state from this)",
 "s0":{"sim_max":..., "r":..., "tau_t":..., "rho":..., "n_items":..., "verbatim_hits":[...],
        "verbatim_misses":[...], "fast_path":false, "floor_rank":null},
 "s1_me":{"rank_self_med":..., "nmargin_med":..., "dH_self":..., "dH_mean":..., "n_eff_norm":...,
          "disp":..., "churn":..., "locate":..., "strong":..., "interf":..., "dup_risk":...,
          "smallstore":..., "per_probe":[{"probe":...,"channel":...,"rank_self":...,"nmargin":...,
          "H_pre":...,"H_post":...}]},
 "action":"ADD|NOOP|SAFE_REWRITE|ABSTAIN","reason_code":"closed vocabulary","stage":...,
 "confidence":null_or_risk, "escalated":..., "user_text":"verbatim (probe replay)",
 "policy":{"name":...,"shadow":...,"thresholds":{...},"calibration":sha_or_null},
 "preflight_ok":..., "applied":"none|noop|rewrite", "synthetic_result":..., "rewritten_call":...,
 "decision_features_sha":"sha16 of the online feature view (leakage control)",
 "latency_ms":{"s0":...,"s1":...,"total":...}}
// observe — mirror truth after every confirmed mutation (incl. removals!)
{"event":"observe|observe_remove|observe_clear","episode_id":...,"step_idx":...,
 "tier":...,"ref":...,"text":"full","n_items":...}
// read_gate / read_inject / pack_write / hact / placement — analogous, see §19
```

Non-negotiable properties [CODE-VERIFIED as the properties BFCL replay depends on]: full untruncated `candidate_text` and stored `text` (offline replay reconstructs decision-time stores from the log alone — `label_outcomes.walk_log`); destructive mutations logged in-stream; `user_text` logged verbatim; shadow rows identical in shape to live rows.

---

## 15. Integration checklist for a future EvoMemBench agent

Ordered; each item names its acceptance test. [EVOMEMBENCH-HYPOTHESIS with CODE-VERIFIED contracts]

1. **Locate the 12 mapping points of §12.** Deliverable: a table like §12 with real file/function names. No H-Nav code before this exists.
2. **Build `RetrieverReplica` and prove fidelity.** Test: for ≥200 sampled (corpus, query) pairs from real runs, replica ranking == benchmark ranking (allow documented tie-break differences). BFCL precedent: retrieval_sim.py docstring's line-pinned fidelity contract.
3. **Refit the whitener** on an in-domain text pool with the benchmark's embedder (`compute_abtt.py` recipe; record corpus hashes in the artifact meta). Test: median pairwise cosine ≈ 0 post-ABTT; `load`-time D mismatch fails loudly.
4. **Build `StoreMirror` + observation.** Test: after replaying N episodes, mirror state == actual store state, byte-for-byte per item (BFCL: exact-success matching; adapt to host's write-result contract).
5. **Implement `preflight_would_succeed`** from host store constraints. Test: prediction matches actual accept/reject on a fuzzed write set.
6. **Wire HOOK 1/HOOK 2 equivalents** with pass-through defaults. Test: hooks active + all interventions disabled ⇒ metrics byte-identical to baseline (BFCL: `GOV_ENABLED=0` ≡ `-FC`).
7. **Ship shadow mode first** (`shadow=1` default everywhere). Run the full benchmark once; verify zero behavioral diff; collect the signal harvest.
8. **Port the counterfactual labeler** (S_with/S_without over host snapshots + host grader; fate/target vocabulary of label_outcomes_hnav.py:41-60). Test: hand-audit ≥30 labels.
9. **Run the Stage-0 headroom analysis (§16) on the harvest.** GO/NO_GO before building any live arm.
10. **Calibrate thresholds** on shadow data only, freeze to a hashed JSON, pass via config (SS13.3 pattern: `GOV_ME_CALIB`, `HACT_CALIB`; refuse-to-overwrite discipline of select_hact_thresholds.py).
11. **Only then** enable live arms per §17, with the audit schema of §14.6/§19 from day one.
12. Throughout: **never** read gold answers/future turns in any online code path (BFCL enforces with AST scans — test_diff_signals.py-style forbidden-token tests; replicate).

---

## 16. Stage-0 headroom analysis

**Purpose.** Before any live H-Nav arm runs on EvoMemBench, a shadow harvest + offline analysis must answer: *does this benchmark generate the failure classes H-Nav polices, at rates where a policy can move the score?* This is the single most important lesson of the BFCL program — every gating negative there was a headroom failure, detected only after full campaigns. The gate below reuses thresholds **already justified and pre-registered in this research**, with their origins.

### 16.1 The pre-registered thresholds and where they come from

| Threshold | Value | Origin (verbatim source) |
|---|---|---|
| Minimum coverage for a deployable decision cell | **≥ 5% of writes** | `H1_MIN_COVERAGE = 0.05`, `bfcl_eval/scripts/evaluate_hnav_stage1.py:68` (Stage 1 pre-registration; quoted in gov_logs/HNAV_STAGE1.md §3 and latest/ORIGINAL_DIRECTIVE.md §1.1) [CODE-VERIFIED constant] |
| Correct-intervention precision | **≥ 0.90**, Wilson95 lower ≥ 0.80 | `H1_MIN_PRECISION = 0.90`, `H1_MIN_PRECISION_LO = 0.80` (same file :69-70) |
| Harmful-intervention rate ceiling | **Wilson95 upper ≤ 0.05** | `H1_MAX_HARM_HI = 0.05` (same file :71). BFCL Stage 1 failed this at UB 0.093 |
| Expected-ΔAcc falsifier bar (offline GO) | **≥ +0.02 on ≥ 1 backend** | `"go_criterion": "expected_dacc ... >= 0.02"` in gov_logs/hnav_autonomous/falsifier_write_rescue.json and falsifier_write_scaffold.json; ledger alt10 entry [REPORT-VERIFIED] |
| Minimum positives for any multivariate claim | **≥ 25** | `MIN_POSITIVES_FOR_MODEL = 25` (evaluate_hnav_stage1.py:64, EPV gate); Stage 3 gate criterion `min_positives: 25` (stage3_gate.json — failed at n=9); PREREGISTRATION.md: "with fewer than 25 positives no multivariate claim is made ... the univariate permutation family is then the only T1 evidence" |
| Cheap-falsifier false-override ceiling | **≤ 0.20** | `H3_MAX_FALSE_OVERRIDE = 0.20` (evaluate_hnav_stage1.py:72; BFCL failed at 0.664) |
| NC collapse requirement | **|NC delta mean| < 0.02 or CI covers 0** | Stage-3 gate (PREREGISTRATION.md "Decision gate"); reuse with the NC2b correction (§9.7.6) |
| Confirmatory success bar | CI excluding zero **AND** consistent direction in ≥ 4/5 replicates | scaffold PREREGISTRATION H-S1; PREREGISTRATION_C1 §Analysis; H-C2a |
| Step-budget invalidation | arm force-quit rate > max(3× baseline, 2 entries/arm-replicate) | CORRECTED_BASELINE.md §3 / step_budget_baseline.json |
| Non-inferiority margin for "does no harm" arms | −2pp | ME_ABLATIONS_PHASE6_G2.md §4 (Plan v2 G2) |

[INFERENCE] These transfer because none is BFCL-anatomy-specific: coverage/precision/harm bars price *any* selective intervention; 0.02 expected-ΔAcc ≈ 3 questions on a 155-question set — rescale to ~2% of the EvoMemBench question count; min-positives 25 is an events-per-variable statistical floor, not a benchmark fact.

### 16.2 The Stage-0 headroom protocol for EvoMemBench

1. **Run the baseline agent N≥3 replicates in pure shadow** (adapters wired, zero interventions) with full logging (§19) and per-episode snapshots.
2. **Loss decomposition first** (the BFCL-pivotal instrument): for every question, classify {gold not captured} / {captured, not retrievable} / {captured+retrievable, not read} / {read, not used} / {used, wrong answer anyway}, with per-stratum p(correct) and, if tiers exist, tier-conditional conversion factors. This one table decides which H-Nav components can matter at all.
3. **Label the write stream counterfactually** (port of label_outcomes_hnav): fate {terminal, superseded, gone}; targets {must_write, must_suppress, may_suppress, inert_superseded, uncertain} via S_with/S_without regrade.
4. **Measure base rates.** GO for the *admission-gate* arm family requires: `must_suppress` ≥ 25 positives in the harvest **and** an operating cell with coverage ≥ 5%, precision ≥ 0.90 (Wilson lo ≥ 0.80), harm UB ≤ 0.05, expected ΔAcc ≥ +2% of question count. Same structure for each other arm family with its own target class (read-gate: ambiguous-read events whose resolution flips answers; scaffolds: no-capture/no-read turns carrying gold).
5. **Falsify offline before going live**: for each candidate intervention, an offline replay must predict expected ΔAcc ≥ the 0.02-equivalent bar *using tier-conditional conversion with a range over the read-conversion factor c* — never a pooled scalar (the alt5R lesson: +0.171 pooled → ≈0 corrected).
6. **NO_GO consequences are pre-committed**: a family whose class is < 25 positives gets no live arm — its result is reported as "benchmark does not generate this failure class" (a legitimate finding, per Outcome F of the directive), not silently tuned until something passes.

### 16.3 Expected-headroom priors from BFCL [INFERENCE — verify against EvoMemBench data, do not assume]

- If EvoMemBench genuinely evolves facts over time (updates/contradictions), `must_write`-like and conflict classes should be far denser than BFCL's 3.5%/0.7%, reopening exactly the components BFCL could not test (diff-aware admission, supersede rewrites, read-time recency).
- If its stores grow larger than ~57 items, retrieval ambiguity (read-gate class) and capacity classes (eviction) inflate.
- If its agent reliably reads memory before answering, the BFCL-dominant capture→read class may be near-empty there — in which case the scaffold arms are dropped and the thesis rests on the gating/read-disambiguation classes. The headroom gate decides; not the port's preferences.

---

## 17. Proposed experiment arms and controls

[EVOMEMBENCH-HYPOTHESIS; arm structure mirrors the pre-registration discipline used across gov_logs/hnav_shadow/PREREGISTRATION.md and gov_logs/hnav_rag/PREREGISTRATION_C1.md]

Objective: test **which failure classes are repaired**, not aggregate accuracy alone. Every arm ships with its shadow twin and identical logging.

| Arm | Configuration | Question it answers |
|---|---|---|
| **A0** baseline | adapters wired, all interventions off | reference distribution; also proves hook-neutrality (must equal no-adapter run) |
| **A0'** replicate baseline | A0 re-run(s), different seeds | server/sampling noise floor for all deltas (BFCL: in-campaign baseline arm, qwen_hact.py:19-21) |
| **A1** geometry-only | Stage-0 gate live (`geometry_only` policy), no margin/entropy stage | does pure geometric dedup/admission help/harm on this benchmark? |
| **A2** retrieval-uncertainty-only | Stage-0 forced pass-through except ESCALATE→Option A (or: read gate only), geometry NOOP disabled | does the margin/dH/churn machinery carry signal absent geometry suppression? |
| **A3** full H-Nav | geometry + margin-entropy + read gate (+ scaffolds if the loss decomposition warrants) | joint effect |
| **NC1** shuffled-entropy | A3 with dH permuted within (backend, store-size bin), seeded (`nc_shuffle_dh`, admission_policy.py:349-371) | destroys signal-candidate pairing, preserves marginals — any surviving "gain" is confounded |
| **NC2** irrelevant-probe / shuffled-capture | scaffold controls: frozen generic read probe; word-scrambled writes (qwen_rag.py IRRELEVANT_PROBE, `pack_shuffled`) | separates mechanism (extra step/volume) from information |
| **Ablations** | A3 minus SAFE_REWRITE (`me_rewrite_disabled`, plan SS18 arm 7); A3 with random-select at matched sampling cost; read-gate-only; per-threshold sweeps only on pre-registered grids | attribute the effect to a component |

Justified additions [INFERENCE]: if EvoMemBench populates update/conflict classes (§18), add **A2b: diff-aware admission** (diff_signals features promoted online — the exact component BFCL could not test for lack of update headroom) and **A4: capture+read scaffolds** only if the benchmark's loss decomposition shows capture/read pipeline losses like BFCL's.

Per-arm reporting: per-failure-class baseline acc, arm acc, positive/negative flips, coverage, intervention precision, token/latency cost (BFCL: flip_tables.json + cost_accounting.json shapes).

---

## 18. Failure taxonomy

Labels adapted to the real architecture (write-side vocabulary = the counterfactual targets of label_outcomes_hnav.py; read-side = the mechanisms of §5/§8). For each: (1) can current H-Nav detect it, (2) signal, (3) intervention, (4) tested in BFCL?, (5) BFCL headroom?, (6) why EvoMemBench may test it better. Headroom entries reference §9/§16.

### Write-side

| Failure class | Detect? | Signal | Intervention | Tested in BFCL? | BFCL headroom | Why EvoMemBench better |
|---|---|---|---|---|---|---|
| Redundant write (exact) | **Yes** | normalized byte-equality fast path | NOOP | Yes (live-capable; validated arm) | Low: widest deployable NOOP cell = 2.6% of writes vs 5% bar; frontier expected ΔAcc negative |  An evolving store accumulates re-statements over time → class should be denser |
| Redundant write (paraphrase) | **Yes** | `sim_max>sim_high ∧ r<δ ∧` verbatim containment | NOOP | Yes | Same cell: coverage 2.6% < 5%; harm Wilson UB 0.093 > 0.05 ceiling | same |
| Shadowed duplicate (new item unfindable behind near-dup) | **Yes** | `¬locate ∧ dup_risk` | NOOP | Yes (gov2 shadow) | rare (escalation subset) | Longer horizons + larger stores → more shadowing |
| Critical-delta update misread as duplicate | **Partially** | offline only: `diff_sim_max` low while blob `sim_max` high; `diff_n_new_values_unseen_in_store>0` | none online (diff features never promoted) | Offline falsifier only | 47/52 near-dup updates carried real new content — the *detector* works; the *decision class* (suppression harm) was rare | If EvoMemBench has genuine temporal updates, promote diff features online — the marquee port |
| Destructive overwrite (replace/update drops content) | **Yes** | legacy: NLI blob-diff (`recsum_blobdiff`), dropped-clause counts; SE gate op-class | legacy REWRITE-to-supersede logs `superseded_text`; SE fallback-to-safe; not in gov2 | SE project tested; gov2 does not gate overwrites beyond geometry | Refs are overwritten 3.32×/ref, 38% of writes superseded — yet `must_suppress` only 0.7%: self-overwrites are mostly benign here | Evolving stores make old-value loss consequential at question time |
| Conflict / contradiction | **Legacy only** | NLI `contradiction` verdict (τ=0.75) | supersede REWRITE | Shadow-tested in legacy; retired offline in Plan v2 | low observed | Contradictions-over-time are EvoMemBench's presumed core; consider reviving as *offline judge* first |
| Stale fact | **No** (online) | only `turn_written` recency exists (placement value formula) | none | No | n/a | Needs benchmark with time-indexed validity; add recency-aware read reranking [EVOMEMBENCH-HYPOTHESIS] |
| Incorrect NOOP by the *gate* (false suppression) | **Yes (evaluation-side)** | counterfactual `must_write ∧ suppressed` | threshold retreat; preflight+verbatim guards bound it | Yes — the harm ceiling metric | must_write = 3.5% (159/4 575), concentrated in terminal adds (0.27) | same labeler works anywhere with offline regrade |
| Model never writes (capture failure) | **Yes** | empty decode on a prereq/turn | scaffold write (SCAF/M2 pack) | Yes — the dominant BFCL loss | **Huge**: 77–81% of questions' gold not carried; ~88% of it stated verbatim in-conversation | If EvoMemBench agents also under-write, same scaffold applies; else class is empty there |
| Unretrievable write (bad key/off-topic) | **Yes** | retrievability floor rank>3; `diff_rank_self_post` | ESCALATE; slug keys in scaffolds | Yes | small | KV-like lexical stores elsewhere |
| Memory growth / capacity pressure | **Yes** | ρ (density), preflight capacity fail | SAGE τ tightening; M7 eviction; SS5 atomic move | Partially (M7 in C-campaigns; SS5 never live-headline) | caps rarely bound at baseline (under-writing) | Long-horizon accumulation makes eviction real |

### Retrieval-side

| Failure class | Detect? | Signal | Intervention | Tested in BFCL? | BFCL headroom | Why EvoMemBench better |
|---|---|---|---|---|---|---|
| Dominant wrong memory (top-1 confidently wrong) | **No** | margin is *high* — margin-triggered gate passes it | none | No | n/a | Needs answer-aware signals (MIG-style) or provenance checks [EVOMEMBENCH-HYPOTHESIS] |
| Multiple plausible memories (ambiguity) | **Yes** | live top1−top2 margin ≤ τ | read gate: refine-or-return-set | Shadow-tested (SS6) | Thin: baseline archival stores hold ~2–19 entries, often ≤ top_k, so the retrieved set is the whole store (PREREGISTRATION_C1 limitation) | Denser stores → more real ambiguity |
| Old-vs-new conflict at read | **Partially** | ambiguity margin fires, but gate is recency-blind | return-set lets the model choose | No targeted test | low | Time-versioned stores; extend refinement with recency |
| High retrieval entropy / distributed facts | **Partially** | H/zscore/vn logged; n_eff_norm; never a live trigger | none live; MIG/`return_set` adjacent | Logged only (deliberate: scale degeneracy) | — | Port `zscore_entropy`/`vn_entropy` as candidate triggers, calibrate there |
| Low-confidence top-1 (correct but weak) | **Yes** | margin small | top1_pass after refinement | Shadow | small | same |
| Distractor contamination of context | **Yes (adjacent)** | none in H-Nav proper; MIG utility scores | payload trim/rerank (MIG, M9, pad control) | Yes (MIG project; M3/M9 arms) | Bounded: M9 ceiling +2.3pp pre-declared (recall@5 already 0.927) | Bigger k / noisier stores amplify |
| Model never reads (read failure) | **Yes** | no-call question turn | read scaffold (M1) | Yes — the second dominant BFCL loss (archival conversion 0.0167/0.0000 kv/vec vs core 0.556/0.701; 1/155 entries ever reads archival) | **Huge** in BFCL; C1 interim vector +8.4pp (1 rep, non-confirmatory) | If EvoMemBench forces explicit retrieval, class may vanish; check first |
| Stale retrieval (index lags store) | **No** | n/a (BFCL indexes are rebuilt per call) | none | No | n/a | Only meaningful if EvoMemBench has async indexing |

---

## 19. Required logs/artifacts

[CODE-VERIFIED as the artifact set the BFCL analysis pipeline actually consumed; port all of it]

1. **Decision log** (gov2-equivalent, §14.6) — one row per write candidate, shadow and live identical in shape.
2. **Observation log** — every confirmed store mutation incl. removes/clears (offline replay depends on it, governance_filter.py:2315-2329).
3. **Action-uncertainty log** (hact1-equivalent) — per gated generation: seeds, all sample texts+parsed calls, vote features at all resolutions, logprob features, token counts, orphan/parse-crash flags; join key to decision rows.
4. **Read log** — read_gate outcomes; injected-read events with probe text; payload transforms.
5. **Capture log** — scaffold/pack writes with mode, buffer state, cap hits, eviction events.
6. **Store snapshots per episode boundary** (the `_final.json` analogue) — required by the counterfactual labeler.
7. **Frozen calibration JSONs** with SHA recorded in the campaign manifest (`hact_calibration.json` / `margin_entropy_calibration.json` pattern; refuse-to-overwrite).
8. **Whitening artifact** with corpus file hashes + fit metadata in `meta` (compute_abtt.py pattern); logged per session (`abtt_meta` in the rehydrate record).
9. **Pre-registration documents** frozen and committed *before* each campaign (PREREGISTRATION.md pattern), plus an experiment ledger (`experiment_ledger.jsonl` / `hnav_ledger.py`: one row per hypothesis with verdict).
10. **Integrity/cost accounting** per campaign (INTEGRITY.md, cost_accounting.json patterns): git heads, config dumps, token/latency totals, replicate seeds.
11. **Outcome labels file** (`outcomes_hnav.jsonl` analogue) — per write: fate, hnav_target, necessity/damage predicates, lineage diagnostics.

---

## 20. Statistical evaluation requirements

[CODE-VERIFIED from evaluate_hact_shadow.py:1-120, select_hact_thresholds.py, analyze_gov_replicates.holm, counterfactual_noop.wilson; these are implemented, reusable procedures]

1. **Grouping**: all resampling grouped by the dependency unit (BFCL: scenario chain). Chain-grouped bootstrap for CIs; within-chain permutation for p-values.
2. **Out-of-fold discipline**: leave-one-scenario-out OOF predictions for every AUC; never in-sample.
3. **Nested feature ladders**: incremental blocks (metadata → geometry → margin/entropy → logprobs → vote controls → entropy features → interactions), so each family is judged *beyond* its controls — the headline contrast is entropy-beyond-own-vote-controls (M8−M4), not entropy-vs-nothing.
4. **Multiplicity**: Holm correction over the pre-registered delta family only.
5. **Negative controls as gates**: the NC shuffle must collapse the gain (|NC dAUC| < 0.02 or CI covers 0) for a GO — a significant delta that survives its own negative control is required, not optional.
6. **Replicates**: ≥3 shadow replicates; replicate-holdout sign consistency required; dev/val partition frozen in a committed doc (DATASET_PARTITION.md pattern: dev=reps 1-2, val=rep 3, evaluated ONCE).
7. **Minimum support**: `n_positives ≥ 25` for any multivariate/gated claim (`MIN_POSITIVES_FOR_MODEL = 25`, evaluate_hnav_stage1.py:64; Stage 3 failed this at 9 positives); below it, only the pre-registered univariate within-chain permutation family is admissible evidence.
8. **Effect accounting**: coverage (interventions/candidates), harmful-event recall, false-intervention rate on must_write, all with Wilson intervals; expected ΔAcc via the measured carrier→correct conversion factor (p̂ ≈ 0.42 in BFCL, hnav_answer_index --validate) rather than raw event counts.
9. **Flip tables**: per-question positive/negative flips vs baseline, by failure class (flip_tables.json pattern), with replicate-noise-floor context from A0'.
10. **Cost**: full token/latency cost of sampling/probing reported alongside every gain (cost_accounting.json pattern); the harness-visible accounting stays baseline-identical while the record carries true cost (qwen_hact.py:253-259).

---

## 21. Thesis claims each possible outcome would support

The thesis question: *can H-Nav improve an evolving RAG / agent-memory system by detecting difficult memory states and intervening selectively?* Possible outcomes and the claim each supports [INFERENCE, calibrated by the BFCL outcome vocabulary — ORIGINAL_DIRECTIVE.md §28 pre-blesses the negative outcomes as valid thesis results]:

1. **Strong positive.** ≥1 pre-registered arm beats baseline with CI excluding zero, direction consistent in ≥4/5 replicates, survives its equal-compute control AND its negative control collapses, with per-class flip tables showing the gains concentrated in the targeted failure classes. Claim: *H-Nav's geometric/uncertainty signals detect and repair specific memory-failure classes in an evolving store, improving end-task performance.* (BFCL never reached this bar; its closest was alt10's +0.0345 failing conditions 3 and 7.)
2. **Mechanistic positive without end-task transfer.** Detection AUCs clear their gates (≥25 positives, NC-robust, Holm-significant) and targeted per-class error rates drop, but aggregate accuracy CI includes zero (e.g. repaired classes are a small share of total loss, or interventions trade classes — like BFCL's finance net-negative flips). Claim: *H-Nav is a validated failure-class detector-and-repairer whose end-task value is bounded by the benchmark's class mix* — with the measured class-mix table as the quantitative bridge. This still answers thesis step (4) affirmatively and step (5) negatively-with-mechanism.
3. **Benchmark-limited outcome (BFCL replay).** The Stage-0 headroom gate returns NO_GO for the gating families (classes < 25 positives / coverage < 5%), and only scaffold-like levers matter. Claim: *the failure classes H-Nav targets are rare in this benchmark too; the loss decomposition localizes where memory systems actually fail* — a benchmark-characterization contribution, plus the transferable methodology (counterfactual labeling, tier-conditional conversion, shadow-first discipline). Two independent benchmarks agreeing on class rarity would itself be a publishable claim about agent-memory benchmarks vs deployed systems.
4. **Clean negative (method failure).** Classes are dense (headroom GO) but the signals do not predict them out-of-sample (ladder ≤ controls, permutation p ≥ 0.05, or NC fails to collapse the gain). Claim: *geometric and retrieval-uncertainty signals, as constituted, do not detect these failure classes even where they are frequent* — the strongest possible falsification of the H-Nav hypothesis, and only obtainable on a benchmark with headroom (which is why the BFCL negatives could NOT support this claim; do not conflate).
5. **Partial/asymmetric outcomes** (expected, plan for them): e.g. read-side works, write-side doesn't (or vice versa); vector-analog works, lexical-analog doesn't (BFCL's kv/vector asymmetry — kv rescues were carried-but-unretrievable). Claim scoped per component and per retrieval regime; the component-portability table (§10) plus per-class results make these claims precise rather than anecdotal.

In all outcomes, report: base rates, coverage, precision/harm with Wilson intervals, cost per recovered answer, and the negative-control behavior — BFCL's record shows the label "promising but inconclusive" must be available and pre-committed, and that offline falsifier GOs are predictions, not results.

---

## 22. Open questions requiring the EvoMemBench repository

[EVOMEMBENCH-HYPOTHESIS — all of these gate design decisions above]

1. **What evolves?** Are there genuine fact *updates*/contradictions over time (populating the critical-delta and conflict classes BFCL lacked), or mostly append-and-query? This decides whether diff_signals gets promoted online (§18 marquee port).
2. **Retriever access**: can the benchmark's retriever be invoked offline on arbitrary provisional corpora? Is it lexical, dense, hybrid? (Determines the RetrieverReplica and which nmargin normalization applies; hybrid needs a new normalization decision.)
3. **Score visibility**: does the agent (and can the middleware) see pre-truncation ranked scores? Without them the read gate and all margin signals need a replica detour on every read.
4. **Write API semantics**: is there an update-in-place with stable ids (enables old-text lookup for diffs), or append-only? Is there a delete? What are capacity/length constraints (preflight)?
5. **Success-result contract**: does the agent observe write results (needed for synthetic-success NOOP mechanics), or are writes fire-and-forget (then suppression is trivially just dropping the call)?
6. **Turn/episode structure**: what is the dependency unit for grouped statistics; is there a per-episode snapshot boundary for the counterfactual labeler; what is the step budget for injected scaffold steps?
7. **Grader**: exact-match/substring vs LLM-judge? (Substring made verbatim capture optimal in BFCL — an LLM-judge grader could reverse the M6 abstractive-vs-verbatim prediction and changes the SE text-equivalence criterion.)
8. **Does the agent under-write / under-read at baseline?** Run the loss decomposition FIRST (§16 step 1). If EvoMemBench agents reliably write and read, the scaffold arms are pointless and the admission/read-gate arms are the story; if not, BFCL's conclusion may transfer.
9. **Embedder reuse**: which embedding model does the benchmark use, and is reusing it for governance acceptable, with ABTT refit on what corpus?
10. **Memory pressure**: do stores actually hit capacity (making eviction/placement live), and over what horizon?
11. **Is there a notion of memory tiers** (working vs archival)? If not, the tier-conditional machinery (routing, atomic moves, per-tier τ) collapses to one tier — simplify, don't emulate.
12. **Latency/cost budget**: are ~10-40 ms/decision CPU signal computations and optional N-sample generation acceptable in the benchmark's harness?
