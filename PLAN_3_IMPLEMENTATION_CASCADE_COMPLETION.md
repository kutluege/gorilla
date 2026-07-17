# Plan 3 — Completing the Cascade (Placement, Read-Time, Multi-Arm Ablations)

> **Sequence:** This is the **third** of the ordered plans. It extends the instruments Plan 1
> (`PLAN_1_IMPLEMENTATION_OFFLINE.md`) and Plan 2 Steps 1–2 (`PLAN_2_EVALUATION_TUNNELED_GPU.md`,
> commits `88d42a8` + `df9d64a`) produced. Plan 2 Steps 0/3/4/5 (tunnel-dependent) are **still
> pending**; Plan 3 is therefore split into an **offline phase (3A)** that can be built and
> tested now, and a **live phase (3B)** that slots in after Plan 2's live steps run. Nothing in
> 3A blocks on the tunnel; nothing in 3B is claimed before its live data exists.

Derived from the Plan 3 continuation brief (G1–G15) and reconciled against the actual repository
state on branch `claude/nihai-plan-v2-cascade-thresholds-k34u67`.

---

## 0. Reality check — per-gap audit (what already exists, with citations)

> ⚠️ **Master-plan availability.** `kaskad_nihai_plan.md` ("Nihai Plan v2") is **not present in
> the repository** on any branch — it exists only as a referenced document. Its §5 value
> formula, §5.1–5.6 contracts, §6 read-time wording, and §7 w₁ grid are recoverable **only**
> from the Plan 3 continuation brief supplied by the author, which this plan therefore treats
> as the binding spec for those sections. Where a spec below cites "brief GX", that is the
> authority. Recovering/committing the master plan itself remains desirable but is not a
> blocker for 3A.

All paths under `berkeley-function-call-leaderboard/` unless noted. `GF` =
`bfcl_eval/model_handler/middleware/governance_filter.py`.

| Gap | Status in repo | Evidence (file:line) | Remaining work |
|---|---|---|---|
| G1 entropy diagnostics | **PARTIAL (≈95%)** — H, N_eff, ΔH_neighbor computed + logged, decision-inert | formulas `retrieval_sim.py:136-213`; ΔH loop `GF:1110-1129`; per-probe H/n_eff `GF:1039-1048`; only margin decides `GF:1061,1087`; inert comment `GF:904` | Log the softmax **temperature** per record (`Stage2Result.to_log`, `GF:908-922` has no T field); `test_entropy_fields.py`; decision-path regression proof |
| G2 §8.4 ρ gate | **PARTIAL** — offline replay instrument + committed verdicts exist; **live-log correlator absent** | Spearman impl `replay_geometry_deltaH.py:61-87`; committed ρ in `gov_logs/replay_*/replay_summary.json` (KV .48/.61/.66, Vector .41/.59/−.16); the 3 committed live logs **predate Stage 2** (0 `stage2` fields) | New `correlate_geometry_deltaH.py` reading live `stage2.dH_neighbor` from decision records, joined to `sim_max`/`r`, surviving-chains-only (via `parse_wrra`); needs a live Stage-2 shadow run (3B) for the real verdict |
| G3 paraphrase probes | **PARTIAL** — channel plumbed, generator is a stub | stub `probe_gen.py:83-97`; channels all `template:*` `probe_gen.py:123-161`; user-text-only provenance `probe_gen.py:144-163`, `GF:1014` | Real CPU paraphraser (flan-t5-small, `GOV_PROBE_MODEL`), lazy singleton, template fallback preserved |
| G4 probe representativeness | **ABSENT** | — | Offline validator: BFCL answer-key questions vs probes over the same corpus (`bfcl_eval/data/BFCL_v4_memory.json` + possible_answer) |
| G5 CANONICALIZE | **MOSTLY EXISTS** — deterministic one-shot + same-probe retest + low_confidence all real | `canonicalize_candidate` `GF:931-976`; same probes `GF:1080`; single round by construction `GF:1064-1103`; LLM path dead (`canonicalize_llm_fn=None`, `GF:1527`) | `test_canonicalize.py` acceptance (single round, no second generation); leave LLM path unwired (brief G5 forbids generate-race anyway) |
| G6 placement/eviction | **ABSENT** | no value formula / routing / eviction / last-copy check anywhere; `verbatim_*` fields are the Stage 0 redundancy gate (`GF:574-577,646-663`), not §5's final check; only `MemoryItem.turn_written` (`GF:315`) exists as recency datum | New `middleware/placement.py` + wiring (Step 4) |
| G7 destructive guard | **ABSENT as intervention** — observe-only | remove/clear never gated: WRITE tables `GF:1141-1152`, observed-only tables `GF:1154-1166`; `build_candidate` returns None → passthrough `GF:1214,1374-1386`; mirror-sync + log only `GF:1651-1685` | Gate remove/clear: archive-then-remove expansion, clear → safe rewrite (Step 4) |
| G8 rec_sum blob-diff | **exclusion EXISTS**, surrogate absent | `qwen_gov.py:54,111-112`; no blob-diff code | Thesis declares rec_sum out of scope → standalone module + offline tests only, **not wired into any arm** (Step 4, lowest priority) |
| G9 read-time | **ABSENT in governance**; prior art in MIG | writes-only gating `qwen_gov.py:198-223`; read interception prior art `middleware/mig_reranker.py:50-52,187` | New read-time module (Step 5), shadow-first |
| G10 calibration grids | **ABSENT** (and §7 grid-on-accuracy superseded by geometry-first calibration, E11) | — | Instruments in Step 6; live refinement in 3B |
| G11 ablations | **ABSENT** (arms become runner configs once G13 lands) | — | Step 9 (3B) |
| G12 ABTT ablation | **ABSENT**; fit code reusable | corpus = BFCL prereq pool `compute_abtt.py:32-72`; eigen path corpus-agnostic `:106-140`; no identity-ablation code | Optional: general-corpus loader + replay rerun (Step 9, optional) |
| G13 multi-arm | **ABSENT** — two arms hard-coded | runner: constants `run_gov_replicates.py:50-51`, seq `:66-72`, env branch `:103-124`, models dict `:191-192`; analyzer: constants `analyze_gov_replicates.py:41-42`, `pair_replicate` binary `:119-169`, both-arms check `:260`, arms loop `:281-284`, Holm backend-only `:198,227-229` | Step 7: arm-spec list; reference-arm pairwise stats; Holm over backend×arm-pair; joint survival across **all** arms |
| G14 seed | **CONFIRMED null end-to-end** | manifest `run_gov_replicates.py:212`; no CLI seed (`bfcl_eval/__main__.py`); no seed in vLLM call (`base_oss_handler.py:354-377`) | **Decision (Step 8): honest statement, no seed plumbing.** Rationale below |
| G15 generate→evaluate | **MOSTLY EXISTS** | evaluate runs immediately after generate per arm `run_gov_replicates.py:227-273`; exit code in manifest `:255-259`; **score path absent** from `cmd_end`; analyzer silently degrades on missing scores (`parse_wrra.py:167-180` → `correct=None` → `excluded["unscored_questions"]`, `analyze_gov_replicates.py:162-165` — nothing errors) | Step 8: score-file path + existence check in manifest; analyzer errors on unscored replicates; `test_evaluate_integration.py`; three-layer metric table |

### 0.1 Conflict table vs `GOV_CASCADE_PLAN_RECOMMENDATIONS.md` (no silent conflicts)

| Recommendation | Plan 3 position | Conflict? | Resolution (binding for this plan) |
|---|---|---|---|
| §1 **Stage −1 write-compliance watchdog** ("mandatory for the thesis") | Absent from G1–G15 | **YES — omission** | **Adopt option (a) log-and-report only.** Already 90% satisfied: `parse_wrra.py` dead-chain detection + survival-conditional pairing in the analyzer ARE the log-and-report watchdog, run offline. Step 4 adds a live `write_compliance` warning event to the governance log when a prereq chain ends with 0 resolved writes (detection only). Options (b) nudge / (c) re-roll are **rejected**: they change model behavior, i.e. constitute a new experimental arm that would contaminate the governance A/B (R4-class conflation). If wanted later, they are a separate arm in the G13 multi-arm runner. |
| §1/§9 reorder Month 1: archival/write-compliance first | Plan 3 keeps §5 in Step 4, after G1/G2 | Partial (priority framing) | Keep Plan 3 order — G2's go/no-go must run early (its own R1). The §5 block is Step 4, immediately after; the standalone mandatory-archiving ablation (Step 9) preserves the "most guaranteed gain" claim as its own measured arm. |
| Entropy's role (recs treat ΔH as gate correlate only) | Identical: decision-inert, margin decides | No | DoD asserts "no accept/reject path reads H/N_eff/ΔH" (regression test, Step 1). |
| §2.2/§2.3/§6: per-backend thresholds; Vector has no keys → canonical-key is KV-only | Adopted since Plan 1 | No | Unchanged. |
| §2.4 never grid-search on accuracy | Adopted | No, but tension with brief G10's grids | G10 grids run on **decision-log geometry / eviction simulation recall**, accuracy is validation-only (Step 6). |
| §8.2 SAGE arm | Dropped (Plan 2) | No | Stays dropped; documented negative result. |
| §3(4) sidecar cached probes | Exists (Plan 1) | No | Reused by Step 1 ΔH and Step 4 value formula. |

### 0.2 Standing facts Plan 3 inherits

- Baseline reference (to be committed in Step 0): **3/5 chains alive** (healthcare + student
  dead), KV accuracy **0.0710 (11/155)**, Vector **0.1161** — from `parse_wrra.py` on the
  committed `result/`+`score/` trees.
- §8.4 offline verdict: geometry↔ΔH **marginal for Vector** (arm-dependent −0.16..+0.59) →
  **the NLI-first fallback branch must be treated as live**, not hypothetical (Step 2).
- Both registry arms send the same served-model id; one vLLM serves every arm (G13 included).
- Middleware log is append-mode + CWD-relative → per-replicate/arm `GOV_LOG_DIR` isolation is
  a hard contract for every new logged field (R7).

---

## 1. Objective & scope

**Objective:** close every gap (G1–G15) between the implemented cascade and the full
architecture: complete the Stage 2 diagnostics and probe channels, decide the §8.4 gate from
live logs, build the §5 placement/eviction/destructive-guard block and the §6 read-time
mechanism (both shadow-first), generalize the runner/analyzer to multi-arm ablations, harden
the generate→evaluate contract, and report with the three metric layers cleanly separated.

**In scope (3A — offline, buildable now):** Steps 0–8 code + standalone tests.
**In scope (3B — needs the tunnel, after Plan 2 Steps 0/3/4/5):** Step 2's live verdict,
Step 6's live refinement, Step 9's ablation runs + final report.
**Out of scope:** rec_sum governance in any arm (thesis-excluded; G8 is a standalone module),
fine-tuned components, any change to the BFCL backend/grader/score files, Stage −1 options
(b)/(c).

**Binding constraints (unchanged):** entropy never decides — margin decides; no bare deletes —
archive-then-remove, atomic; whitened space for Stage 0 + value formula, raw MiniLM for
Stage 2 sim + read-time; pooled calibration, per-scenario reporting; student pre-registered
excluded; `--num-threads 1` sequential; existing 7 suites (282 checks) stay green; new tests
are standalone print-style scripts; `parse_wrra.py`/`run_gov_replicates.py`/
`analyze_gov_replicates.py` are extended, never rewritten.

---

## 2. Prerequisites (verify before Step 1)

1. Branch `claude/nihai-plan-v2-cascade-thresholds-k34u67` at `df9d64a` or later; 7 suites green.
2. DeBERTa (`microsoft/deberta-large-mnli`) and MiniLM in the local HF cache (both confirmed).
3. For Step 3: `flan-t5-small` will be downloaded on first use (~300 MB) — same lazy-singleton
   discipline as DeBERTa.
4. For 3B only: tunnel up, Plan 2 Steps 0/3 complete (a Stage-2-enabled shadow log exists).

---

## 3. Ordered work items

Each step is independently committable and lists files, change, acceptance.

### Step 0 — Scope lock + pre-replicate reference commit  · brief Step 0
- This document **is** the scope lock and the §0.1 conflict table.
- Run `parse_wrra.py` on **both** committed arms and commit the outputs:
  `gov_logs/wrra_baseline.jsonl` (model `...-FC`) and `gov_logs/wrra_governed.jsonl`
  (model `...-FC-GOV`), from the default `result/`+`score/` trees. These pin the 3/5 //
  0.0710 reference every later §5/§6 claim compares against.
- **Acceptance:** both JSONL files committed; summary lines reproduced in the commit message;
  conflict table present in this file (no silent conflict left).

### Step 1 — Finish G1: temperature in Stage 2 records + tests  · brief G1
- **Files:** `GF` (`Stage2Result` + `to_log`, `GF:894-922`; populate at `stage2_resolve`
  around `GF:1016`), new `bfcl_eval/scripts/test_entropy_fields.py`.
- **Change:** add `temperature` (the per-backend softmax T actually used) and `probes_n` to
  `Stage2Result`/`to_log` — the only §4.4 fields missing. H/N_eff/ΔH_neighbor are already
  computed (`retrieval_sim.py:136-213`, `GF:1110-1129`); do **not** touch their math or the
  decision path.
- **Acceptance:** `test_entropy_fields.py` — (a) H/N_eff/ΔH match hand-computed values on
  synthetic score lists (call `entropy`/`n_eff`/`delta_h_for_probes` directly); (b) every
  Stage 2 log record carries `H`, `n_eff`, `min_margin`, `dH_neighbor`, `dH_mean`,
  `temperature`; (c) **regression:** identical synthetic input → identical decision + reason
  with the new fields present vs a pre-change expectation table; (d) existing 7 suites green.

### Step 2 — G2 correlator + go/no-go protocol  · brief G2 — **GO/NO-GO**
- **Files:** new `bfcl_eval/scripts/correlate_geometry_deltaH.py`, new
  `bfcl_eval/scripts/test_correlate_geometry.py`.
- **Change:** read live `governance_log.jsonl` decision records that carry
  `stage2.dH_neighbor`; join each record's Stage 0 `sim_max`/`r` to its own `stage2.dH_mean`;
  **filter to surviving chains** by joining `test_id`→scenario against a `wrra_*.jsonl` file
  (dead chain ⇒ excluded); Spearman ρ (reuse/port `spearman()` from
  `replay_geometry_deltaH.py:61-87`) + bootstrap CI per backend; emit JSON verdict with the
  ρ≥0.40 branch decision.
- **Protocol:** the committed replay verdict (KV +0.48..+0.66; Vector arm-dependent) stands as
  the *provisional* verdict; the *binding* verdict is computed in 3B from the Plan 2 Step 3
  shadow harvest (first Stage-2-emitting live log). **If Vector ρ < 0.40 → NLI-first for
  Vector:** Stage 1 sees Vector traffic first; cost note recomputed (~450 DeBERTa CPU
  calls/run measured is affordable, so the fallback is viable); KV keeps canonical-key-first
  regardless (zero decided NOOPs).
- **Acceptance:** synthetic-log tests: correlation matches hand-checked ranks; dead-chain
  records excluded; ρ + CI + branch decision in the JSON output; provisional-vs-binding
  distinction stated in the artifact.

### Step 3 — G3 paraphrase channel + G4 representativeness + G5 test  · brief G3/G4/G5
- **Files:** `probe_gen.py` (replace `_paraphrase_stub`, `probe_gen.py:83-97`), new
  `bfcl_eval/scripts/validate_probe_representativeness.py`, new
  `bfcl_eval/scripts/test_canonicalize.py`, extend `test_retrieval_sim_offline.py` probe checks.
- **Change (G3):** lazy CPU singleton for `GOV_PROBE_MODEL` (default
  `google/flan-t5-small` — different family from Qwen), generating 2–3 paraphrases of
  `user_text` **only** (provenance invariant `probe_gen.py:150-151` untouched); channel tag
  `paraphrase:flan-t5-small`; `GOV_PROBE_PARAPHRASE=0` (default) keeps template-only —
  shadow-first discipline; total probes still capped by `GOV_S2_PROBES_N`.
- **Change (G4):** validator pairs BFCL answer-key questions (`bfcl_eval/data/BFCL_v4_memory.json`
  + `possible_answer/`) with their scenario's prereq source sentences; for each, does the
  *real* question rank the same entry top-1 as the generated probes do over the same simulated
  corpus (`simulate_kv`/`simulate_vector`)? Report per-backend agreement — a thesis finding,
  not a gate.
- **Change (G5):** no production code — `test_canonicalize.py` proves: exactly one
  canonicalization round, same probe list on retest (`GF:1080`), no second candidate
  generation path exists, `low_confidence` set on failure.
- **Acceptance:** paraphrase channel emits ≤3 paraphrase probes with correct channel tags and
  degrades cleanly to templates when the model is unavailable; representativeness report
  written; canonicalize tests green; provenance assertion (no `candidate.text` in probe
  inputs) covered by an explicit test.

### Step 4 — §5 block: placement, eviction, destructive guard  · brief G6/G7/G8
- **Files:** new `bfcl_eval/model_handler/middleware/placement.py`; wiring in `GF`
  (`govern_calls`/`_govern_one`, gated tables `GF:1141-1166`) and `qwen_gov.py`; new
  `bfcl_eval/scripts/test_placement.py`, `test_eviction.py`, `test_destructive_guard.py`.
- All new behavior behind **`GOV_P_ENABLED` (default 0) + `GOV_P_SHADOW` (default 1)** —
  shadow-run-first protocol, same as Stages 1/2.
- **G6a verbatim final check:** post-decision, pre-write: every extracted verbatim value
  (reuse `extract_verbatim_values`/`verbatim_present`, `GF:509-535`) must appear verbatim in
  the outgoing key or value; else rewrite via the existing `_pending` `"rewrite"` mode
  (`GF:1430-1441` — real result flows, executed call observed; no new plumbing).
- **G6b routing:** middleware-side deterministic classifier (there is **no fact-extraction
  prompt in this pipeline** to add a field to — the model emits calls directly, so the brief's
  "one added prompt field" is realized as a middleware classifier; this design deviation is
  recorded here): `identity` (name/age/pronoun/preference patterns over user_text +
  candidate) → core; `event/detail` → archival; logged as `placement.category`. Core-full ⇒
  atomic move (G6c value-ranked victim): **expand** the governed call list — insert
  `archival_memory_add(victim)` *before* the core write, then `core_memory_remove(victim)` —
  with `_pending` index remapping; archive-add failure aborts the remove (atomicity, R5).
- **G6c value formula:** `value(m_i) = w1*(1 − max_{j≠i} sim(m'_i, m'_j)) + w2*(1/(1+Δt))`
  in the **whitened** space (Stage 0 cache matrix), `Δt` = current step − `turn_written`
  (`GF:315`); start `w1=0.6, w2=0.4`; **no access-frequency term**.
- **G6d archival-full last-copy check:** before evicting from archival, NLI (existing
  `_get_nli_model` singleton) asks whether the victim's content is entailed by any remaining
  entry; if nowhere derivable → **no deletion**: allow overflow or reject candidate, log
  `risk:"critical_information_loss"`.
- **G7 destructive guard:** add remove/clear to a new gated table: bare
  `*_memory_remove` → archive-then-remove expansion (same mechanism as G6b);
  `*_memory_clear` → rewritten to the decoy read (`GF:1171`) + synthetic success, logged
  `destructive_blocked`; never enters the cascade.
- **G7-adjacent (Stage −1 log-and-report, §0.1 decision):** on the final prereq entry of a
  chain (`is_first_memory_prereq_entry` family, `bfcl_eval/utils.py`) with 0 mirror-observed
  resolved writes, emit a `write_compliance` warning event. Detection only, no intervention.
- **G8 rec_sum surrogate (lowest priority, standalone):** `middleware/recsum_blobdiff.py` —
  NLI entailment of old-blob propositions by the new blob for replace/clear/whole-update;
  offline tests only; **not wired into any registry arm** (rec_sum stays thesis-excluded).
- **Acceptance:** `test_placement.py` (routing categories, verbatim final check rewrites,
  value-formula hand-checks in whitened space); `test_eviction.py` (atomic order:
  archive-add precedes core-remove; simulated archive-add failure leaves core intact — no
  intermediate-state data loss); `test_destructive_guard.py` (clear NEVER passes under any
  config; last-copy deletion blocked; bare remove becomes archive+remove pair); shadow mode
  logs all of the above with zero call mutation; 7 suites still green.

### Step 5 — §6 read-time mechanism  · brief G9
- **Files:** new `bfcl_eval/model_handler/middleware/read_gate.py`; wiring in `qwen_gov.py`
  `_add_execution_results_prompting` (`qwen_gov.py:206-223`; prior art for read-result
  interception: `middleware/mig_reranker.py:50-52,187`); new `bfcl_eval/scripts/test_read_time.py`.
- **Change:** gated by `GOV_READ_ENABLED` (default 0) + `GOV_READ_SHADOW` (default 1).
  On a retrieve **result**: compute margin over the returned ranking (KV: BM25 scores via
  `simulate_kv`; Vector: raw-MiniLM cosine via `simulate_vector` — never whitened);
  `margin > threshold` → top-1 passthrough; else ambiguous set = top-1's δ-neighborhood
  (bounded 2–4); **one** refinement round: append to the query the overlapping distinct-vocab
  token from the ambiguous keys; re-rank within the set only; still ambiguous → return the
  whole bounded set to the model. Memory is never written.
- **Acceptance:** `test_read_time.py` — snapshot invariant (byte-identical memory before/
  after); at most one refinement round (call-count assertion); ambiguous set never exceeds 4;
  shadow mode = log-only, returned payload unchanged.

### Step 6 — G10 calibration instruments  · brief G10
- **Files:** new `bfcl_eval/scripts/calibrate_placement.py` (w₁ grid {0.4,0.5,0.6,0.7} via
  eviction **simulation** over committed snapshots → recall proxy), extend the Stage-1 shadow
  analysis for k ∈ {3,5,7,10} (re-scoring logged escalation candidates), read-time margin
  threshold from the Stage 2 margin distribution (same measurement).
- Grids run on **log geometry / simulation recall** only — accuracy is validation, never the
  search objective (§0.1). Pooled calibration, per-scenario reporting.
- **Acceptance:** calibration note (`gov_logs/plan3_calibration.md`) committed with the chosen
  values + supporting distributions; 3B refines the read-time threshold on live logs.

### Step 7 — G13 multi-arm runner + analyzer  · brief G13
- **Files:** `run_gov_replicates.py`, `analyze_gov_replicates.py`,
  `test_gov_replicates.py` (extend — existing checks must keep passing unchanged).
- **Runner:** replace the two-constant arm model with an ordered **arm-spec list**
  `[{label, model, gov_env}]` (default = today's baseline/governed pair — backward
  compatible); `--arm label=model[:GOV_K=V,...]` repeatable; sequence generalizes to
  B1,A1,C1,…,B2,… (`arm_sequence`, `run_gov_replicates.py:66-72`); per-(replicate,arm)
  `GOV_LOG_DIR`; manifest `run_start` records the full arm list.
- **Analyzer:** designated **reference arm** (default `baseline`); every other arm is paired
  against it. Joint survival: a (replicate, backend, scenario) unit drops if the chain died
  in **any** arm of the analysis set (otherwise arms compare over different subsets — R6);
  McNemar/bootstrap per (backend × arm-pair); **Holm across backend × arm-pair** (the `holm()`
  impl already takes arbitrary labels, `analyze_gov_replicates.py:63-71` — key it by
  `(backend, arm)`); `completed_replicates` requires **all** arms.
- **Acceptance:** extended `test_gov_replicates.py` — 3-arm sequence ordering; a dead chain in
  ONE arm drops the unit from ALL pairs; Holm m = backends×arm-pairs verified by hand; 2-arm
  results byte-identical to the pre-change analyzer (regression); all existing checks green.

### Step 8 — G15 hardening + G14 decision  · brief G14/G15
- **Files:** `run_gov_replicates.py`, `analyze_gov_replicates.py`, `parse_wrra.py` (no
  rewrite — additive), new `bfcl_eval/scripts/test_evaluate_integration.py`.
- **G15:** evaluate already chains after generate (`run_gov_replicates.py:227-273`). Add:
  (a) `cmd_end` for evaluate records the **score-file path(s) + existence check**; (b) the
  analyzer **raises** (with the replicate/arm named) when a both-arms-complete replicate has a
  missing score file, instead of silently swallowing into `unscored_questions`
  (`analyze_gov_replicates.py:162-165`); an explicit `--allow-unscored` opt-out exists for
  salvage analyses and stamps the output.
- **G14 decision (binding):** **no seed plumbing.** Rationale: no seed flag exists anywhere in
  the chain (CLI `bfcl_eval/__main__.py`; vLLM call `base_oss_handler.py:354-377`), adding one
  means modifying the shared harness for both arms mid-experiment, and at temperature 0.001
  sampling is near-greedy so a seed would be cosmetic. The report states verbatim: **"N=5
  replicates, sequential, no fixed seed; run-to-run variation stems from server-side
  nondeterminism at temperature 0.001; replicate ≠ seed."** The manifest's `seed: null` +
  temperature field is the audit trail.
- **Acceptance:** `test_evaluate_integration.py` — (a) generate failure ⇒ evaluate not called,
  replicate incomplete (existing fail-stop covers this — add the explicit check); (b) score
  path + existence in manifest; (c) unscored replicate ⇒ analyzer raises; `--allow-unscored`
  ⇒ stamped output; (d) `--dry-run` shows an evaluate after every generate (assert on plan
  text). Three-layer table (official BFCL score = primary / chains-W-R-R-A = diagnostic /
  paraphrase-tolerant judge = secondary) is the report template, stated in `EVAL_RESULTS.md`
  when 3B writes it.

### Step 9 — 3B: live verdict, ablations, final report  · brief G2/G11/G12
Requires the tunnel and Plan 2 Steps 0/3/4/5 complete. Order:
1. **G2 binding verdict** from the Plan 2 Step 3 shadow harvest (`correlate_geometry_deltaH.py`);
   ρ<0.40 for Vector ⇒ NLI-first reorder + recomputed cost note **before** any §5/§6 arm runs.
2. **Ablation arms** (all via the G13 runner, one mechanism at a time — R4): cascade
   (geometry-off / NLI-off+canonical-key / retrieval-off), read-time (top-1 / top-3 /
   margin-adaptive), **standalone mandatory-archiving arm** (G7 only, cascade off),
   ABTT-vs-identity re-confirm via replay; optional G12 general-corpus ABTT.
3. **Entropy contribution analysis** (diagnostic): margin-accepted decisions retro-checked
   against logged `N_eff`/`ΔH_neighbor` — did entropy flag failures margin missed? Never a
   gate change.
4. **Paraphrase-tolerant NLI/judge secondary metric** over final answers (DeBERTa entailment
   vs possible answers), reported beside — never instead of — the official score.
5. **`EVAL_RESULTS.md`** (extends Plan 2 Step 5's file): replicate tables, CIs, drop counts,
   three metric layers side by side per arm with primacy stated, chains_alive 3/5→? §5 claim
   vs the Step 0 committed reference, the seed statement, and the §0-diagram→code map (DoD 12).

---

## 4. Definition of Done (Plan 3)

1. Stage 2 records carry `H`, `n_eff`, `min_margin`, `dH_neighbor`, `dH_mean`, `temperature`;
   regression proves the decision path unchanged.
2. §8.4 Spearman ρ per backend from **live** Stage-2 logs, surviving chains only; the ≥0.40
   branch decision (or NLI-first fallback + cost note) committed.
3. Stage 2 complete: paraphrase channel (different-family CPU model), probe representativeness
   report, one-shot canonicalize proven by test.
4. §5 block live-able behind `GOV_P_*`: verbatim final check, routing, value formula
   (w₁=0.6/w₂=0.4 start), last-copy protection, delete/clear mandatory archiving; rec_sum
   surrogate exists standalone and unwired.
5. §6 read-time behind `GOV_READ_*`; proven memory-invariant, one refinement round, set ≤4.
6. w₁/k/read-margin calibrated on geometry with a committed note.
7. Runner + analyzer are N-arm; joint survival across all arms; Holm over backend×arm-pair;
   2-arm regression identical.
8. Ablations + statistics reported; entropy contribution analysis written as diagnostic;
   **replicate ≠ seed stated verbatim** in the report.
9. §5's effect measured against the **committed** Step 0 reference (3/5 chains alive).
10. Every arm×replicate has generate→evaluate complete; score path in manifest; unscored ⇒
    analyzer error (explicit opt-out only).
11. Final report: official BFCL score primary, W/R/R/A diagnostic, paraphrase-tolerant judge
    secondary — side by side per arm.
12. Every §0-diagram box maps to a code counterpart (table in `EVAL_RESULTS.md`).
13. Committed + pushed to `claude/nihai-plan-v2-cascade-thresholds-k34u67`.

---

## 5. Risks specific to Plan 3

- **R1 — ρ<0.40 reorders the plan.** Provisional evidence already points that way for Vector.
  Step 2 precedes Steps 4/5 for exactly this reason; KV is unaffected (canonical-key-first).
- **R2 — entropy scope creep.** ΔH/N_eff/H stay logged-only; Step 1's regression test is the
  tripwire; DoD 1 restates it.
- **R3 — calibration budget (~30 recall questions).** New thresholds (w₁/w₂, read margin, k)
  calibrate pooled on geometry/simulation, report per-scenario; accuracy is never the search
  objective.
- **R4 — conflating §5 and §6 effects.** Every 3B arm toggles ONE mechanism; the G13 runner
  makes that a config, not a code branch.
- **R5 — atomicity of core→archival moves.** Archive-add strictly precedes remove; a failed
  archive-add aborts the remove; `test_eviction.py` simulates the failure; a dedicated
  `placement_move` log channel keeps it distinguishable from cascade NOOPs in W/R/R/A.
- **R6 — multi-arm survival subsets.** Joint drop across ALL arms, enforced + tested in Step 7.
- **R7 — log isolation.** All new fields/events ride the existing per-(replicate,arm)
  `GOV_LOG_DIR` contract; Step 7 extends it to N arms; nothing writes outside it.
- **R8 — the seed claim.** Resolved by the Step 8 binding decision + verbatim report sentence.
- **R9 — unscored replicates.** Resolved by Step 8: analyzer raises, salvage is explicit.
- **R10 — diluting the official score.** The three-layer table with stated primacy is the
  report template; chains_alive is diagnostic by construction.
- **R11 (new) — call-list expansion.** G6b/G7 rewrite ONE call into TWO (archive+remove);
  `_pending` index remapping must survive; `patch_results`' rewrite passthrough
  (`GF:1587-1614`) is reused, but expansion is new surface — covered by dedicated tests
  before any live run.
- **R12 (new) — paraphraser drift.** A generative paraphraser can hallucinate probe content;
  probes remain user-text-derived, are length/content-filtered, and the template channel is
  never removed — `GOV_PROBE_PARAPHRASE=0` restores Plan-2 behavior byte-identically.

---

**Hand-off:** after 3A (Steps 0–8) everything is offline-green and shadow-gated; the moment the
tunnel is up, the order is: Plan 2 Step 0 smoke → Plan 2 Step 3 harvest → **Plan 3 Step 2
binding verdict** → Plan 2 Step 4 A/B → Plan 3 Step 9 ablations → joint `EVAL_RESULTS.md`.
