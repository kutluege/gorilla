# Stage 0 Results Analysis and Stage 1/2 Implementation Plans

**Scope.** Part A: evidence-verified analysis of everything the repo actually contains for the
BFCL v4 memory-governance work (Stage 0 geometric filter, MIG reranker track, SE track,
supporting audits). Part B: linked implementation plans for Stage 1 (NLI) and Stage 2
(Retrieval Entropy), grounded in the Stage 0 code that exists on this branch. All paths below
are relative to `berkeley-function-call-leaderboard/` unless stated otherwise. Every number was
re-read or recomputed from the files listed in the Appendix (§13); where the repo contradicts
prior chat claims or its own summary docs, the discrepancy is flagged inline and collected in
§2.5.

---

## 1. Executive summary

- **Stage 0 works as a mechanism and is inert as a treatment.** Across three full governed
  runs (~1,051 logged write decisions), the geometric filter suppressed **9 writes total**
  (1 shadow would-fire, 3 SAGE-default, 5 calibrated), all on the Vector backend, all
  verifiably redundant. Zero KV suppressions in any run. Decision latency 0.53 ms. The
  mechanism (mirror cache, decoy rewrite, synthetic success, preflight guard) is validated by
  the logs and by a forensic trace of the one pathological case (customer-6: 21 preflight-forced
  writes, every one confirmed as a genuine backend length-limit error the guard correctly
  refused to mask).
- **The A/B accuracy deltas are not a governance signal.** The KV spread
  (7.10% → 21.29% → 16.77%) occurred with *zero* KV suppressions; per-scenario decomposition
  shows it is fully explained by prereq-chain survival (healthcare stored 0 items in the
  baseline arm, 7–26 in governed arms). Four distinct baseline `-FC` results exist in this repo
  (KV: 7.10 / 11.61 / 13.55 / 15.48%), bounding single-run noise at ±8–10 points. The
  `student` scenario stored zero memories in **every run of every experiment from March through
  July 2026** and floors every arm by 50/155 questions.
- **The master plan's two key empirical premises need revision.** (1) Measured raw anisotropy
  is 0.096, not ~0.83 — ABTT's necessity narrative must change (§5.5). (2) The Vector backend
  has **no key field**, so Stage 1's canonical-key channel is KV-only (§7.3).
- **MIG decision: keep it out of the cascade.** MIG is a read-time reranker; its full-pipeline
  numbers (12.73% / 16.36% vs 14.55% baseline, n=55, vector-only) are — per its *own* report —
  invalid as a treatment comparison (write-phase confound) and its controlled experiment shows
  the real binding constraint is that the agent almost never retrieves at answer time. Report
  MIG as an independent comparison point; at most, revisit its judge scorer as one ablation arm
  of the plan's Section 6 read-time mechanism in Month 4 (§4.2).
- **Sequencing recommendation: run the Section 8.4 go/no-go gate before building Stage 1.**
  The gate's `ΔH_neighbor` quantity can be computed **offline, now, with no new GPU runs**, by
  replaying the ~1,051 logged decisions through a small retrieval-simulation module — and that
  module is a strict subset of Stage 2 (§6). Build it first; it settles whether geometry stays
  in front, feeds the ABTT ablation for free, and becomes Stage 2's core when Stages 1–2 land.
- **Do not commit `sim_high=0.95 / δ=0.30` as `GovConfig` defaults yet** (§5.3). They are
  supported by exactly 5 suppressions from one run, KV needs a per-backend treatment anyway,
  and the values are properly documented as the reproducible env-var profile.

---

## 2. Verified current state

### 2.1 Repository / branch state

- Branch `just_geo_filtering` and the working branch have **identical trees** (verified:
  `git diff --stat` is empty). Stage 0 core commit is `84d831d`; later commits (`b14b043`,
  `94afae4`) added experiment artifacts, logs, and reports.
- Stage 0 source: `bfcl_eval/model_handler/middleware/governance_filter.py` (939 lines),
  handler `bfcl_eval/model_handler/local_inference/qwen_gov.py`, ABTT fitting script
  `bfcl_eval/scripts/compute_abtt.py`, offline tests `bfcl_eval/scripts/test_gov_offline.py`.
- Registry: `Qwen/Qwen3-4B-Instruct-2507-FC-GOV` → `QwenGovHandler`
  (`bfcl_eval/constants/model_config.py:1705-1717`). Sibling experimental registrations:
  `-FC-MIG` (line 1669) and `-FC-SE` (line 1686).

### 2.2 Committed configuration defaults (Part A point 3)

From `governance_filter.py:83-101` (`GovConfig`):

| Field | Committed default | Calibrated run value (env only) |
|---|---|---|
| `sim_high` | **0.80** | 0.95 (`GOV_SIM_HIGH`) |
| `delta` | **0.025** | 0.30 (`GOV_DELTA`) |
| `tau0 / tau_min / lam / alpha` | 0.25 / 0.025 / 2.0 / 0.9 | unchanged |
| `d` (ABTT) | 16 | unchanged |

Confirmed: **the calibrated thresholds are not committed as defaults.** They exist only in the
documented reproduction command (`GOV_LOCAL_EXPERIMENTS_COMPARISON.md`, "Reproduction" section)
and in the calibrated run's per-decision log records (`sim_high: 0.95, delta: 0.3` on all 447
decisions in `gov_logs/governed_calibrated/governance_log.jsonl`). This matches the stated
intent of deferring the commit pending replicates. Note the committed `sim_high=0.80` default
is *also* not a SAGE value — SAGE has no similarity floor; the code comment at
`governance_filter.py:94-96` correctly flags it as needing shadow-run calibration.

### 2.3 Result/score directory reconciliation (Part A point 1)

No two result trees are byte-identical duplicates (verified by `diff -q`); each directory is a
distinct run. Authoritative mapping:

| Arm | Result dir | Score dir | Status |
|---|---|---|---|
| Baseline `-FC` (Stage-0 session) | `result/Qwen_Qwen3-4B-Instruct-2507-FC/` | `score/Qwen_Qwen3-4B-Instruct-2507-FC/` | authoritative baseline for the Stage-0 A/B |
| Governed, calibrated (0.95/0.30) | `result/Qwen_Qwen3-4B-Instruct-2507-FC-GOV/` | `score/Qwen_Qwen3-4B-Instruct-2507-FC-GOV/` | authoritative; lives only in the live `result/`+`score/` mix |
| Governed, SAGE defaults (0.80/0.025) | `result_gov_sage/` | `score_gov_sage/` | authoritative (note: no model-subdir level; arm identity from dir name only) |
| Shadow (`GOV_DRY_RUN=1`) | `result_gov_shadow_partial/` (kv 50, vector 25 entries) | **never scored** | intentionally aborted partial run |
| MIG judge | `result_mig_judge/` | `score_mig_judge/` | 55-entry vector-only partial |
| MIG logprob | `result_mig_logprob/` | `score_mig_logprob/` | 55-entry vector-only partial |
| MIG-era baseline | `result_baseline/` | `score_baseline/` | 55-entry vector-only partial; only valid comparator for the MIG arms |
| SE baseline / SE gated | `result_se_exp/{baseline,gated}/` | `score_se_exp/{baseline,gated}/` | earlier session, full 155×3 backends |

Stale/redundant: none of the trees is redundant, but `result/`+`score/` are a **mix of two
arms** (baseline + calibrated GOV) and must never be read as one experiment;
`tmp/smoke_archive_2026-07-01/` holds MIG smoke-run leftovers (healthcare-only) that should not
be confused with results.

### 2.4 Verified numbers (Part A point 2)

**Accuracy** (first line of each score file; exact paths in §13):

| Arm | memory_kv | memory_vector | memory_rec_sum |
|---|---|---|---|
| Baseline `-FC` (Stage-0 session) | 7.10% (11/155) | 11.61% (18/155) | — |
| GOV SAGE defaults | 21.29% (33/155) | 12.26% (19/155) | — |
| GOV calibrated | 16.77% (26/155) | 12.90% (20/155) | — |
| SE-session baseline `-FC` | 15.48% (24/155) | 17.42% (27/155) | 27.10% (42/155) |
| SE gated `-FC-SE` | 11.61% (18/155) | 16.13% (25/155) | 26.45% (41/155) |
| MIG-era baseline `-FC` (55 subset) | — | 14.55% (8/55) | — |
| MIG judge (55 subset) | — | 12.73% (7/55) | — |
| MIG logprob (55 subset) | — | 16.36% (9/55) | — |

These confirm the headline numbers in `GOV_LOCAL_EXPERIMENTS_COMPARISON.md` §1.1/§2 exactly.

**Governance decisions** (recomputed from `gov_logs/*/governance_log.jsonl`):

| Run | Decisions | NOOP | Escalate→fallback ADD | Preflight-forced ADD | Thresholds in log |
|---|---|---|---|---|---|
| shadow (partial, dry-run) | 262 (21 test_ids) | **1** would-fire (vector, sim=1.0, r=0.0) | 45 (kv 14, vec 31) | 0 | 0.80 / 0.025 |
| governed_sage | 342 | **3** (all vector; sim=1.0, r=0.0) | 81 (kv 23, vec 58) | 1 | 0.80 / 0.025 |
| governed_calibrated | 447 | **5** (all vector; sim 0.9526–1.0, r 0.0–0.295) | 68 (kv 14, vec 54) | 24 | 0.95 / 0.30 |

KV slice: SAGE run 166 KV decisions, 0 NOOPs (15 decisions at sim ≥ 0.80, all blocked by
verbatim misses or escalated); calibrated run 204 KV decisions, 0 NOOPs, **max KV sim_max =
0.9475 — below the 0.95 gate**. All 9 NOOPs across runs had empty `verbatim_misses` and
`preflight_ok=true`.

**Preflight guard forensics.** The calibrated run's 24 forced ADDs are dominated by one
pathology: `memory_vector_prereq_6-customer-6` steps 13–33 repeat an `archival_memory_update`
21 times. Cross-checking the result file
(`result/Qwen_Qwen3-4B-Instruct-2507-FC-GOV/agentic/memory/vector/BFCL_v4_memory_vector_prereq_result.json`)
shows exactly 21 genuine backend errors `"Entry length exceeds maximum length of 2000
characters."` for that entry — i.e., **every preflight-forced write would indeed have errored
at the backend**, and the guard correctly refused to fake a success over a real error
(step 12, whose text was within limits, was legitimately NOOPed). The guard is doing precisely
its designed job; the 21-step loop is a model pathology (retrying an oversized write until the
20-step force-quit), not a governance defect.

**Rehydration check.** `items_loaded` in `rehydrate` events varies normally (0 only for
first-prereq entries, up to 26 mid-chain) in all three logs — the mirror cache does carry
state across the prereq chain; cross-conversation redundancy detection was live.

**ABTT artifact** (`bfcl_eval/model_handler/middleware/artifacts/abtt_minilm_l6_d16.npz`,
embedded metadata): encoder all-MiniLM-L6-v2, D=16, **1,857 sentences** from 5 prereq files,
median pairwise cosine **0.0961 raw → −0.0029 after ABTT**, top-16 variance fraction
**33.7%**, created 2026-07-13.

**Offline tests.** `test_gov_offline.py` has 24 `check(...)` call sites of which two (the
missing-artifact pass/fail branches) are mutually exclusive → 23 executed assertions, matching
the documented "23/23 passing". Not re-run here (no sentence-transformers in this environment);
treated as documented-verified.

### 2.5 Discrepancies found (files win)

1. **Shadow-run stats in the comparison doc are stale.** `GOV_LOCAL_EXPERIMENTS_COMPARISON.md`
   §1.2 reports the shadow arm as "161 gated, 0 would-fire NOOPs, 23 escalations, 118 novel".
   The committed log (`gov_logs/shadow/governance_log.jsonl`) contains **262 decisions, 1
   would-fire NOOP** (`memory_vector_prereq_4-customer-4`, sim=1.0, r=0.0, dry_run=true),
   **45 escalations, 194 novel**. The doc's row was evidently snapshotted mid-run before the
   abort point. The doc should be corrected; the shadow arm *did* produce a would-fire NOOP.
2. **"All calibrated NOOPs at sim ≥ 0.99" (task-description claim) is wrong.** Two of the five
   (`memory_vector_prereq_20-finance-5`, steps 4 and 10) are at sim_max 0.9526 and 0.9679 with
   r ≈ 0.25–0.29 — inside the calibrated NOOP region but not near-byte-identical. The
   comparison doc's own "sim 0.95–1.0" phrasing is correct; the chat-level "≥ 0.99" claim is
   not. (They still look like genuine redundant rewrites from their verbatim-hit sets, but
   they are *paraphrase-level* duplicates, which slightly strengthens the case that the
   calibrated region does real work beyond byte-identity.)
3. **Master plan §1.2 anisotropy premise (~0.83 median cosine) is off by ~8×** — measured
   0.0961 on this corpus (artifact metadata + `compute_abtt.py` docstring). Already flagged in
   `GOV_CASCADE_PLAN_RECOMMENDATIONS.md` §2.1; consequences in §5.5 below.
4. **Master plan §3.1 states the Vector backend also has a key field ("vector'da da anahtar
   alanı var"). It does not.** `memory_vector.py` stores raw text with integer ids only. The
   canonical-key channel is structurally KV-only (Stage 1 plan §7.3 handles this).
5. **Preflight scale.** The task description implies one guard firing; the logs show 1 (SAGE)
   + 24 (calibrated). All verified correct (see forensics above), but the scale difference
   matters: the guard is a load-bearing component, not a corner case.
6. **Silent mirror mutations are unlogged.** `_observe` drops items on remove/clear via
   `cache.drop`/`clear_tier` without emitting any log event (only `_put_item` logs an
   `observe` record, `governance_filter.py:915-938`). No correctness bug found, but offline
   replay (§6) needs the full mutation stream — logging fix recommended.
7. Minor: logged `candidate_text` and `observe.text` are truncated to 300 chars
   (`governance_filter.py:818,934`), which matters for the replay instrument (full texts are
   recoverable from result files; see §6.2).

---

## 3. Experiment narrative (lab-notebook form)

One paragraph per distinct experiment/run found in the repo, chronological where determinable.

**E1 — First full local memory run and pipeline audit (~March 2026).** The starting question
was simply whether the local setup (Windows client → SSH tunnel → vLLM on the BSC cluster
serving Qwen3-4B-Instruct-2507) reproduces the official BFCL memory numbers. A full run of all
three memory backends was generated and scored, and two audit documents were written
(`memory_pipeline_audit_report.md`, `memory_score_discrepancy_report.md`) comparing local
scores against the leaderboard row. The audit traced every stage of the memory pipeline and
made the single most consequential discovery of the whole project: the `student` scenario
produces **zero** memory tool calls — the model chit-chats through all ten prerequisite turns —
so its memory ends empty and all 50 of its questions fail in every backend. This established
that large score gaps can come from generation/serving behavior rather than from anything the
evaluator or any middleware does.

**E2 — Clean re-benchmark on pinned upstream code.** To rule out local code drift as the cause
of E1's discrepancies, the benchmark was re-run from a clean checkout pinned to the official
commit (`f7cf735`) with unmodified code (`BFCL_PURE_REBENCHMARK_GUIDE.md`). The clean run scored
KV 11.61 / Vector 12.90 / Rec_Sum 34.19 (%) against official 16.13 / 12.26 / 24.52 — and the
student scenario was *still* empty. The reasoning: if a pure checkout reproduces the failure,
the cause is the serving environment or the model's tool-calling behavior, not local edits.
This run also quietly became a fourth independent baseline datapoint, later used to bound
run-to-run noise.

**E3 — MIG reranker: design and offline validation (June 2026).** The first intervention idea
was read-time: when the agent retrieves memories, rerank the retrieved pool by how much each
candidate actually helps answer the question (Memory Information Gain), instead of trusting
raw retrieval order. The reranker was built as a thin handler subclass intercepting tool
results (`mig_reranker.py`, `qwen_mig.py`), with two scorers — an LLM judge (0–5 utility
score per candidate) and an exact log-probability gain scorer using the vLLM echo trick — and
was validated mechanically before any benchmark run: static wiring checks, then a deterministic
harness (22/22 checks) proving pool-widening, parsing, scoring, and reserialization work
(`MIG_VALIDATION_REPORT.md`). The intent was to never burn GPU time on an implementation whose
plumbing was unproven.

**E4 — MIG controlled live experiment (2026-06-27).** Before running the full benchmark, the
reranker was tested in isolation on the question it is actually supposed to answer: *given a
polluted retrieval pool, does MIG keep the gold memory?* A purpose-built harness
(`mig_live_experiment.py`) constructed synthetic pools (gold source sentence + hard/random
distractors from the same scenario), ran four arms on each question (full pool, similarity
top-k, MIG-judge, MIG-logprob) against the live model, and scored with the real grader. Results
(`BFCL_MIG_EXPERIMENT_RESULTS.md`): the judge scorer kept the gold memory 100% of the time in
all four settings and matched or beat every other arm's answer accuracy (e.g., 0.900 vs 0.767
for similarity-trim on hard distractors). It also caught and fixed a real bug — logprob-MIG
with an empty-context draft ranks gold *last* (the draft is the model's wrong prior), fixed by
drafting from the pool. The mentality: prove the mechanism in a controlled setting where
retrieval is guaranteed to happen, before testing whether it matters in the wild.

**E5 — MIG full-pipeline A/B (2026-07-01).** The real `bfcl generate` test of MIG, on a
55-question vector-only subset (customer + healthcare) with sequential single-threaded arms:
baseline 14.55% (8/55), MIG-judge 12.73% (7/55), MIG-logprob 16.36% (9/55)
(`BFCL_MIG_EXPERIMENT_REPORT.md`, `result_mig_*`, `score_mig_*`). The report itself declares
the accuracy comparison **invalid as a MIG measurement**: MIG also intercepted during the
memory-*writing* prereq phase, so each arm built a different memory store; and across 55
answer turns the agent retrieved at answer time exactly once (judge) or zero times (logprob),
with gold-in-pool rate 0.00 on the one interception. The valid conclusion drawn was structural:
answer-time retrieval almost never happens, so *any* read-time reranker is starved of
opportunities — the binding constraint is retrieval frequency and recall, not ranking quality.

**E6 — Semantic-entropy gate A/B (session between MIG and Stage 0).** A different write-side
idea was tested at full scale: sample N completions per memory step, cluster them semantically,
commit the majority action, and require low cluster entropy for destructive operations
(`semantic_entropy.py`, `-FC-SE`). Full 155-entry runs on all three backends
(`result_se_exp/`, `score_se_exp/`, `BFCL_SE_EXPERIMENT_REPORT.md`) showed small negative,
statistically non-significant deltas everywhere (e.g., KV 15.48% → 11.61%, McNemar p=0.24).
The gate audit showed it did meaningfully intervene (21 destructive/overwrite majorities safely
diverted), but accuracy did not move. This run also produced the strongest baseline measured in
the repo (KV 15.48%), which later became the key evidence for cross-session baseline drift.

**E7 — ABTT corpus fitting (2026-07-13).** Prerequisite for Stage 0: fit the All-But-The-Top
whitening transform on BFCL's own prerequisite-conversation pool, per the master plan's §1.1
(never an external corpus, never a single user's memory). `compute_abtt.py` collected 1,857
user-turn sentences from the five scenario files, embedded them with MiniLM, and removed the
mean plus top-16 covariance eigendirections. The measurement mattered more than the artifact:
raw median pairwise cosine came out **0.096**, not the ~0.83 the design borrowed from the SAGE
literature — this corpus/encoder is far less anisotropic than assumed, immediately downgrading
ABTT from "necessary correction" to "modest recentering" (the top-16 directions still carry a
third of the variance). The transform was committed with full reproducibility metadata.

**E8 — Stage 0 offline test suite.** Before any server run, the governance filter was tested
end-to-end without the harness (`test_gov_offline.py`): fabricated KV/Vector snapshots, exact
duplicate re-adds (must NOOP), "likes tea" vs "likes coffee" (must never NOOP — content-token
fallback), preflight blocking on duplicate keys, decoy rewrite and synthetic-result patching,
dry-run behavior, cache observation, and sub-millisecond latency at full memory. 23/23
assertions pass. The mentality mirrors E3: prove the middleware's mechanics offline so that any
A/B difference later can only come from decisions, not bugs.

**E9 — Shadow (dry-run) partial run.** The first live deployment ran with `GOV_DRY_RUN=1`:
full pipeline, full logging, zero interventions — to observe the decision distribution the
thresholds would face before letting them act. It was deliberately aborted at roughly one-sixth
progress (21 test entries, 262 logged decisions; partial outputs in
`result_gov_shadow_partial/`, never scored) once it had served its calibration purpose. The
committed log shows the SAGE-style defaults would have fired exactly one NOOP (a sim=1.0
byte-duplicate), previewing the inertness finding.

**E10 — Governed run with SAGE defaults (+ concurrent baseline).** The first live A/B:
baseline `-FC` and governed `-FC-GOV` (sim_high=0.80, δ=0.025) were generated — concurrently,
sharing one vLLM server, which was later recognized as a methodological mistake (batching
nondeterminism at near-zero temperature). Outcome: 3 suppressions in 342 decisions (all Vector
byte-duplicates at sim=1.0), zero KV suppressions, KV accuracy 21.29% vs baseline 7.10% — a
spread obviously not caused by 3 suppressed Vector writes. This run generated the central
insight that prereq-chain survival (healthcare dying in the baseline arm, student dying
everywhere) dominates the accuracy metric.

**E11 — Threshold calibration analysis (log-based, no new run).** Rather than grid-searching
on noisy end-task accuracy, the ~600 logged decisions from E9–E10 were analyzed geometrically
(`GOV_CASCADE_PLAN_RECOMMENDATIONS.md` §2.2): the NOOP rule has a hard constraint
`r ≤ √(1−sim_max²)`, so δ=0.025 requires sim_max ≳ 0.9997 — the SAGE default was provably
near-inert by construction. The observed redundancy band (sim 0.95–0.99, r 0.13–0.30) yielded
the calibrated pair sim_high=0.95, δ=0.30, with the explicit rule that the two must be
calibrated jointly. The mentality: calibrate on decision-log geometry (dense, cheap), validate
on accuracy (sparse, noisy) — not the reverse.

**E12 — Governed run with calibrated thresholds (2026-07-13).** The second live A/B arm, run
sequentially (alone on the server) with `GOV_SIM_HIGH=0.95 GOV_DELTA=0.30`: 5 suppressions in
447 decisions (all Vector, sim 0.9526–1.0, including two paraphrase-level redundant rewrites),
still zero KV suppressions, 24 preflight-guard interventions (21 of them one model retry-loop
on an oversized write, all correctly predicted backend errors), KV 16.77%, Vector 12.90%. The
Vector arm completed the consistent-direction (but within-noise) 11.61 → 12.26 → 12.90 pattern.
The session closed with the two consolidation documents (`GOV_LOCAL_EXPERIMENTS_COMPARISON.md`,
`GOV_CASCADE_PLAN_RECOMMENDATIONS.md`) whose verdict this report re-verifies: *mechanism
validated, accuracy effect not yet measurable*.

---

## 4. Cross-mechanism comparison and the MIG decision

### 4.1 Comparison table (verified full-pipeline numbers only)

**Tier 1 — full 155-entry runs** (same grader, same categories; but note: *different sessions
are not directly comparable* — see baseline drift):

| Mechanism | Intervention point | memory_kv | memory_vector | Session |
|---|---|---|---|---|
| Baseline `-FC` | — | 7.10% | 11.61% | Stage-0 (2026-07-13) |
| Stage-0 GOV, SAGE defaults | write-time gate | 21.29% | 12.26% | Stage-0 |
| Stage-0 GOV, calibrated | write-time gate | 16.77% | 12.90% | Stage-0 |
| Baseline `-FC` | — | 15.48% | 17.42% | SE (earlier) |
| SE gate | generation-time vote | 11.61% | 16.13% | SE |

**Tier 2 — 55-entry vector-only partial runs** (customer+healthcare; internally comparable,
NOT comparable to Tier 1):

| Mechanism | memory_kv | memory_vector |
|---|---|---|
| MIG-era baseline `-FC` | **not run** | 14.55% (8/55) |
| MIG judge | **not run** | 12.73% (7/55) |
| MIG logprob | **not run** | 16.36% (9/55) |

**There is no `memory_kv` MIG run anywhere in the repo** — `result_mig_judge/`,
`result_mig_logprob/` and their score dirs contain only `vector/` (verified by full file
listing). The MIG column of any KV comparison is an explicit gap, not an omission.

Reading the table honestly: no mechanism in this repo has produced a defensible accuracy
improvement. The only consistent-direction signal is Stage-0 Vector (11.61 → 12.26 → 12.90 with
0/3/5 suppressions), which is within single-run noise (±8–10 pts documented across sessions;
the entire Vector spread here is 1.3 pts). The SE gate is flat-to-negative with proper paired
statistics. The MIG deltas are ±1–2 questions on n=55 **and** invalidated by the write-phase
confound documented in `BFCL_MIG_EXPERIMENT_REPORT.md` itself.

`result_se_exp`/`score_se_exp` (the "SE" track) is a separate semantic-entropy gating
experiment — included above as Tier 1 context because it is a full, clean A/B on the same
grader; it is not part of the write cascade and needs no further treatment here.

### 4.2 MIG-inclusion decision

**Decision: MIG does not belong in this pipeline. Report it as an independent, read-time
comparison point; do not merge it into the cascade architecture.**

Reasoning, from the evidence:

1. **Different intervention point, different question.** The cascade (Stages 0–2) governs
   *writes*; MIG selects among *already-retrieved* candidates at read time. Nothing MIG does
   changes what the cascade must decide, and nothing the cascade hands downstream (escalation
   contexts, similarity rankings) is an input MIG could consume. There is no seam in the
   Appendix A architecture where MIG slots in as a stage — and forcing it in would misrepresent
   what was tested.
2. **Its full-pipeline evidence is self-invalidated and starved.** Per its own report: the
   accuracy comparison is confounded (MIG altered prereq-phase writes, so arms answered from
   different memory states), and across 55 answer turns there were 0–1 answer-time retrievals
   with gold-in-pool 0.00. A read-time reranker cannot matter in a pipeline where the agent
   answers from the core-memory dump in the system prompt and almost never calls retrieval.
3. **Its controlled evidence is real but conditional.** E4 shows the judge scorer is genuinely
   good *when retrieval happens and the pool is polluted* (gold recall 1.000 in all cells).
   That condition is exactly the regime of the master plan's **Section 6** read-time mechanism
   — not of the write cascade. If, in Month 4, Section 6's margin-triggered refinement proves
   insufficient, MIG-judge is a ready-made fourth ablation arm for Section 8.2's read-time
   ablation ((a) top-1, (b) top-3, (c) margin-adaptive, (+d) MIG-judge). That is the only
   integration worth keeping on the table, and it is optional.
4. **What the MIG track *does* contribute now is methodology and infrastructure:** the
   sequential-arms/single-thread protocol (adopted in §5.2), the interception-subset analysis
   pattern (measure only where the mechanism fired — directly analogous to
   survival-conditioned scoring), the offline-validation-before-GPU discipline, and the
   documented proof that vLLM concurrent batching breaks A/B determinism. These carry forward
   regardless of MIG's fate.

For the thesis, MIG is best framed as: *"we also tested the read-time selection hypothesis and
found the binding constraint upstream of it (retrieval frequency/recall), redirecting effort to
write-time governance"* — a legitimate negative/diagnostic result with a clean controlled
sub-experiment, reported alongside but outside the cascade.

---

## 5. Findings and interpretation

### 5.1 What the Stage 0 A/B actually established

- The **mechanism** is sound: decisions are cheap (0.53 ms documented; latency fields in logs
  are sub-millisecond), the mirror stays in sync (rehydration verified in §2.4), suppression is
  invisible to the model (synthetic successes match backend formats byte-exactly), and the
  preflight guard has a 100%-verified record of predicting real backend rejections (§2.4).
- The **treatment effect on accuracy is unmeasurable at n=1 per arm**: 3–5 suppressions cannot
  move a 155-question metric whose run-to-run noise is ±8–10 points, and the confound
  (prereq-chain survival) is an order of magnitude larger than the treatment. The task
  description's tentative read is confirmed: the current A/B is *not* the plan's real test of
  whether geometry is worth keeping — Section 8.4's correlation gate is (see §6).

### 5.2 De-noising / replicate protocol (design; Part A point 5)

Design only — nothing here was run.

- **Arms:** baseline `-FC` and governed-calibrated `-FC-GOV` (drop the SAGE arm: it is
  provably near-inert (E11) and burns a replicate slot). Optionally add shadow-mode `-FC-GOV`
  with `GOV_DRY_RUN=1` as a *bias control* — it should be statistically indistinguishable from
  baseline, and any gap measures middleware overhead/prompt-side effects rather than
  suppression.
- **N:** 3 replicates per arm minimum (plan §8.3's "≥3 seed" line), 5 if budget allows —
  power analysis is honestly moot at this noise level; the point of replicates is to measure
  the noise, not to defeat it.
- **Execution discipline (adopted from the MIG protocol, E5):** strictly sequential — one arm
  at a time, `--num-threads 1`, dedicated vLLM server, no co-tenant runs. Both the MIG
  validation report and the SAGE/baseline concurrency incident (E10) independently show vLLM
  continuous batching flips borderline chit-chat-vs-write turns even at temperature 0.001.
  Record `git rev-parse HEAD`, `bfcl version`, server flags per replicate (per
  `BFCL_MEMORY_REPRODUCTION_GUIDE.md`).
- **Pairing and conditioning:** score per scenario×backend×replicate. Define a prereq chain as
  **dead** if the scenario's final snapshot has zero total entries OR its prereq entries show
  zero decoded memory-write calls (both signals already computable — snapshot counts from
  `memory_snapshot/*_final.json`, decoded-call audit per `audit_memory_pipeline.py`). Discard a
  scenario-replicate *pair* if the chain died in either arm; report the discard count as a
  primary statistic (write-compliance rate), not a footnote. McNemar on the paired surviving
  question set; bootstrap CI over scenario-level aggregates.
- **`student` policy: documented exclusion, not a fix.** Its death is deterministic across
  every run March→July (E1, E2, E5, E6, E9, E10, E12 — all arms, all backends), so it
  contributes zero information *and* zero variance to any A/B: excluding it up front removes
  50/155 dead questions from denominators and sharpens every percentage without touching
  pairing. Fixing it means changing the model's tool-calling behavior (prompt nudge or
  re-roll), which is a *different experiment* — that is precisely the
  "Stage −1 write-compliance watchdog" of `GOV_CASCADE_PLAN_RECOMMENDATIONS.md` §1, worth
  building as its own standalone ablation (option (a) log-and-report is mandatory
  instrumentation; option (b) corrective nudge is the candidate fix). Do not silently blend a
  student fix into the governance replicates — it would confound both.
- **MIG harness reusability:** the *pattern* is reusable, the *code* mostly is not.
  `mig_live_experiment.py` is a single-shot Q→A harness that bypasses the prereq pipeline
  entirely (pools built from gold `source` fields), so it cannot host a write-time governance
  A/B. Reuse from it: fixed-arms/paired-per-item design, deterministic seeding
  (`_stable_hash`), per-arm cost accounting, and the sequential-execution protocol its sibling
  report established. The replicate protocol above is therefore "MIG methodology, BFCL-native
  execution".

### 5.3 Threshold/default recommendation (Part A point 6)

**Do not commit `sim_high=0.95 / δ=0.30` as `GovConfig` defaults yet.** Grounds:

1. Evidence base is 5 suppressions from a single run; the plan explicitly deferred the commit
   pending replicates, and nothing since has added replicates.
2. The pair is Vector-calibrated. KV's entire observed distribution sits below it (max 0.9475
   in the calibrated run), so committing it as a global default silently encodes "KV geometry
   never fires" — a decision that should be made explicitly (per-backend `sim_high`, or
   accepting KV-inertness because Stage 1's canonical-key path covers KV; see §5.4), not
   inherited from a Vector calibration.
3. There is a latent footgun worth fixing at the same time: the committed default pair
   (0.80/0.025) is *geometrically near-empty* (δ=0.025 forces sim ≳ 0.9997), i.e., anyone
   running `-FC-GOV` without env vars gets a filter that almost never fires while looking
   enabled. Recommendation: when the replicate-backed values land, commit them **as a coupled
   pair with a startup validation** — warn (or fail) if `delta ≤ sqrt(1 − sim_high²)` leaves a
   degenerate NOOP region. Until then, keep defaults as-is and treat the documented env-var
   recipe in `GOV_LOCAL_EXPERIMENTS_COMPARISON.md` as the canonical calibrated profile.

### 5.4 KV vs Vector: structure, not artifact (Part A point 8)

The zero-KV-suppression result is a **genuine backend-structure mismatch**, with threshold
choice as only a secondary factor. Evidence from the logs:

- **KV redundancy arrives as same-key `replace` with changed values, which must not be
  suppressed.** The SAGE run contains KV decisions at sim_max = 1.0, r = 0.0
  (`memory_kv_prereq_13-healthcare-3`, `archival_memory_replace`) where the verbatim gate found
  *new* values (`misses: ['1.8','1.5']`, `['0.9 mg','1.2 mg',...]`) — the same key being
  legitimately updated with fresh lab results. Stage 0 correctly refused to NOOP these; they
  are exactly Stage 1 UPDATE-territory (canonical-key match → value diff), not duplicates.
- **KV byte-duplicates are pre-empted by the backend itself**: a duplicate `add` on an existing
  key errors at the backend (`preflight_would_succeed` returns False for `key in tier_items`,
  `governance_filter.py:655-662`), so the classic Vector failure mode (re-adding the same text
  under a fresh id) *cannot occur* in KV. The duplication channel geometry was built to catch
  is structurally absent.
- **The composite embedding depresses similarity**: `kv_composite_text()` embeds
  `"key words: value"` (`governance_filter.py:198-201`), so near-duplicate values under
  different keys land at sim ≈ 0.81–0.87 (the finance-1 escalation cluster) — below any
  reasonable global `sim_high`. Lowering the gate to catch these would put Vector's paraphrase
  band inside the NOOP region — a per-backend `sim_high` (KV ≈ 0.92 per the recommendations
  doc) is testable in a KV-only shadow sweep, but the log evidence suggests the payoff is
  small: the sim≥0.92 KV population is dominated by same-key replaces (updates), not
  suppressible duplicates.

**Consequence for Part B (§7):** Stage 1's canonical-key pre-check is not a "cheap extra
channel" for KV — it is KV's *primary* redundancy/update detector, and the Stage 1 plan
below weights it accordingly (it runs before any NLI call and resolves the dominant KV case
by itself). Geometry remains the primary signal only for Vector.

### 5.5 Anisotropy finding (Part A point 9)

Measured raw anisotropy is 0.0961 median pairwise cosine (artifact metadata; 2,000-sample
estimate in `compute_abtt.py:75-83`) vs the ~0.83 the design assumed from the SAGE-domain
literature. The assumed number comes from a different regime (token-level/CLS embeddings of
vanilla LMs); MiniLM sentence embeddings are contrastively trained and arrive largely isotropic
already. Implications:

1. **Thesis narrative must be reframed**: ABTT here is not "rescuing a collapsed similarity
   space" (the 8× premise is falsified by the repo's own measurement — state this openly; it
   reads as rigor, not weakness). The defensible claim is *domain-adapted recentering*: the
   top-16 directions still carry 33.7% of variance, and removing corpus-common components
   plausibly sharpens the *contrast* between duplicate and non-duplicate pairs even when the
   global mean cosine is low. That claim is currently **unquantified**.
2. **The ablation is worth it and nearly free.** Once the replay instrument (§6) exists, run it
   twice — with the committed ABTT transform and with the identity transform — over the same
   ~1,051 logged decisions, and compare (a) the separation of the NOOP band (sim_max of known
   duplicates vs the escalate band) and (b) the geometry↔ΔH Spearman ρ. No GPU, no new runs.
   This is also literally the master plan's own optional ablation (§1.1 "bonus bulgu"),
   upgraded from optional to recommended because ABTT's marginal value is now an open question
   rather than an assumption.

---

## 6. Gap analysis against Section 8.4 (the go/no-go gate)

**What 8.4 requires:** Spearman ρ between Stage 0 scores (`r`, `sim_max`) and "NLI/retrieval
based ΔH estimates", with ρ ≳ 0.4 keeping geometry as the front filter and ρ < 0.4 triggering
the NLI-first fallback. §4.4 defines the concrete estimator: `ΔH_komşu` — the change in
neighbors' *own-probe* retrieval entropy when a candidate is provisionally added.

**What exists:** Stage 0 logs ~1,051 decisions with full geometric signals
(`sim_max, r, tau_t, rho, verbatim_*`) and an event stream (`rehydrate`/`decision`/`observe`)
from which memory state at each decision is reconstructible. What does **not** exist: probe
generation, retrieval simulation, entropy computation — the Stage 1/2 machinery. So the gate is
currently unrunnable, which is backwards relative to the plan's own logic (the gate was a
Month-2 deliverable meant to be resolved *before* committing to the full cascade shape).

**Smallest thing that closes the gap — an offline replay instrument, and it is a strict subset
of Stage 2, not a throwaway:**

1. `middleware/retrieval_sim.py` (new, ~150 lines): backend-faithful scoring —
   `simulate_kv(keys, probe)` reimplementing `memory_kv._similarity_search` byte-for-byte
   (BM25Plus over key names, `key.replace('_',' ').lower().split()` tokenization,
   `memory_kv.py:71-88`) and `simulate_vector(texts, probe)` as raw-MiniLM
   `normalize_embeddings=True` cosine (exact equivalent of FAISS `IndexFlatIP` on ≤57 items,
   `memory_vector.py:251-256, 309-333`); plus `margin()`, `entropy()`, `n_eff()`.
2. `probe_gen.py` — template-channel probes only (deterministic, from the source user
   sentence; the paraphrase channel is a Stage 2 refinement the correlation study does not
   need).
3. `bfcl_eval/scripts/replay_geometry_deltaH.py` — a driver that walks each
   `governance_log.jsonl`, reconstructs the mirror state per decision (rehydrate + observe
   events; remove/clear gaps and 300-char truncations repaired from the result files, which
   contain every executed call and full texts), provisionally adds each logged candidate,
   computes per-neighbor `ΔH` on the neighbors' template probes, and emits
   (test_id, step, backend, sim_max, r, ΔH_neighbor, N_eff) rows → Spearman ρ per backend +
   scatter plots.

Everything in items 1–2 is Stage 2 production code (§8 consumes them unchanged); only the
~100-line replay driver is study-specific, and even it doubles as the regression harness for
Stage 2's simulation and the ABTT ablation vehicle (§5.5). Estimated effort: 1–2 days, zero GPU.

**Interpretation discipline for the gate:** run and report ρ per backend separately. Given
§5.4, expect the KV correlation to be weak or undefined over most of the range (geometry is
structurally secondary there); the decision the gate actually informs is *Vector's* front
filter. A plausible outcome is "geometry survives for Vector, NLI-first (canonical-key-first)
for KV" — which the cascade architecture accommodates without redesign, since Stage 1 already
runs per-backend logic. §10 sequences this.

---

## 7. Stage 1 (NLI) implementation plan

Spec: Appendix A §3 + Stage Map. Integration surface: the actual code in
`governance_filter.py` / `qwen_gov.py` at `84d831d` (current tree).

### 7.1 Where Stage 1 hooks in

Today: `decide()` (`governance_filter.py:518-547`) ends with
`decision, reason = handle_escalation(candidate, signals, cache, cfg)` and
`handle_escalation()` (`:494-515`) is the documented stub returning
`(GovDecision.ADD, "stage0_escalate_fallback")` — its docstring carries the Stage 1/Stage 2
TODOs verbatim.

**Planned restructure (small, mechanical):**

- `decide()` returns `(GovDecision.ESCALATE, "ambiguous")` instead of calling the stub. Pure
  signal→decision logic stays dependency-free and unit-testable.
- `GovernanceSession._govern_one()` (`:780-839`) becomes the orchestrator: on `ESCALATE` it
  calls `self.stage1.resolve(...)` (below); if Stage 1 escalates it calls
  `self.stage2.resolve(...)` (§8); if Stage 2 is absent/disabled the final fallback remains
  ADD, preserving today's semantics bit-for-bit when both stages are off.
- Rationale for moving orchestration to the session: `handle_escalation`'s current signature
  lacks two things Stage 1 needs — the whitened candidate embedding `v_w` (computed in
  `_govern_one` at `:781` and currently dropped after `compute_signals`) and an outcome channel
  richer than ADD/NOOP (call rewriting). The session already owns both the embedding and the
  `_pending`/rewrite machinery.

### 7.2 Reusing Stage 0's similarity ranking and cache (the "already computed, free" link)

The full similarity vector already exists transiently: `compute_signals()` computes
`M @ v_w` at `governance_filter.py:469` and keeps only the max. Change:

```python
@dataclass
class Stage0Signals:
    ...
    sims: Optional[np.ndarray] = None   # per-item whitened cosine, aligned with cache.matrix() items
```

with `sims = M @ v_w; sig.sims = sims; sig.sim_max = float(np.max(sims))` in
`compute_signals`. Stage 1's candidate ranking is then literally
`order = np.argsort(-signals.sims)`, and the aligned item list comes from the existing
version-cached `GovernanceCache.matrix()` (`:255-266`) — **no re-embedding, no re-scoring**.
The metadata cache entries themselves are the existing `MemoryItem` dataclass (`:187-196`):
`ref`, `text`, `emb_whitened`, `turn_written`, `tier`, plus the already-present (currently
unused) `low_confidence` field Stage 2 will start setting.

### 7.3 Candidate selection + canonical-key pre-check

`select_candidates(candidate, signals, cache, k)` (new, in `nli_gate.py`):

1. Top-k by `signals.sims` (default `k=3`, `GOV_NLI_K`; §7 of the plan wants k∈{3,5,7,10}
   sensitivity later).
2. **Canonical-key channel — KV only.** `candidate.ref` *is* the canonical key for KV
   (`build_candidate`, `:622-644`). If `candidate.ref in cache.items[tier]` for either tier,
   force-include that item regardless of sim rank. **Plan deviation, evidence-backed:**
   Appendix A §3.1 assumes Vector also has a key field; it does not (§2.5-4). For Vector the
   analogous channel is narrower: on `update` ops, force-include the *target* item
   `cache.items[candidate.tier].get(candidate.ref)` — the model has already named its
   candidate.
3. **Cheap pre-check (skips NLI entirely).** Two concrete cases, both straight from the logs:
   - *KV `add` on an existing key* → the backend would reject it (preflight already False).
     Today this falls through to ADD and errors genuinely. Stage 1 rewrites it to the
     corresponding `*_memory_replace(key=..., value=...)` — the plan's "exact canonical-key
     match → direct UPDATE". Outcome: `UPDATE_REWRITE` (§7.5).
   - *KV `replace` where the mirror's stored value equals the incoming value after
     `_normalize_for_match`* → byte-level idempotent write → NOOP via the existing suppression
     path (subject to preflight, as always).
   Per §5.4 this pre-check, not geometry, is expected to carry most of KV's redundancy load;
   it costs a dict lookup.

### 7.4 NLI engine

New module `bfcl_eval/model_handler/middleware/nli_gate.py`, dependency-light and
BFCL-import-free like `governance_filter.py` (unit-testable offline — extend
`test_gov_offline.py`'s fabricated-snapshot pattern).

- **Model:** `microsoft/deberta-large-mnli`, CPU, lazy thread-locked singleton exactly
  mirroring `_get_encoder()` in `semantic_entropy.py` (the pattern Stage 0 already reuses at
  `governance_filter.py:50`). Env: `GOV_NLI_MODEL`, `GOV_NLI_DEVICE`.
- **Call:** `nli_probs(premise: str, hypothesis: str) -> (p_contra, p_neutral, p_entail)`
  (single softmax'd forward pass; batch the 2k pairs per escalation in one call for CPU
  efficiency: k=3 → one batch of 6).
- **Bidirectional test per candidate pair** `(m_i.text, f=candidate.text)`:
  `fwd = nli_probs(m_i, f)` ("stored entails candidate") and `bwd = nli_probs(f, m_i)`.
- **Text caveat (KV):** `m_i.text` and `candidate.text` are the `"key words: value"`
  composites — keep them; keys carry real semantics in this benchmark and DeBERTa handles the
  colon form. Log the exact strings scored.

### 7.5 Decision mapping (Appendix A §3.2 → concrete outcomes)

Config: `GOV_NLI_TAU_ENTAIL=0.75`, `GOV_NLI_TAU_CONTRA=0.75`, `GOV_NLI_DELTA_SPEC=0.10`
(initials; calibration §7 protocol — from escalation-band replay, not end-task grid-search,
per E11's lesson). Candidates evaluated in descending `sims` order; first decisive verdict
short-circuits; every evaluated pair is logged.

| Verdict on (m_i, f) | Condition | Outcome |
|---|---|---|
| Equivalent | `fwd.entail ≥ τ_e` and `bwd.entail ≥ τ_e` | **NOOP** (existing decoy+synthetic path) |
| Derivable | `fwd.entail ≥ τ_e`, `bwd.entail < τ_e` | **NOOP** |
| More specific | `bwd.entail ≥ τ_e`, `fwd.entail < τ_e`, `bwd.entail − fwd.entail > δ_spec` | **UPDATE-replace** targeting `m_i.ref` |
| Specificity inconclusive | `bwd.entail ≥ τ_e` but margin ≤ δ_spec | **keep both** → ADD, reason `stage1_keep_both` (plan's safe side; logged as a named case) |
| Contradiction | `max(fwd.contra, bwd.contra) ≥ τ_c` | **UPDATE-contradiction** targeting `m_i.ref` (Zep supersede) |
| All k pairs neutral | none of the above fired on any candidate | **ESCALATE → Stage 2** |

**Outcome application — all via existing machinery:**

- **NOOP** reuses the Stage 0 suppression path unchanged: `preflight_would_succeed()` guard →
  `DECOY_CALL` rewrite in `governed[idx]` → `synthetic_success()` → `_pending[idx]` →
  `patch_results()` substitution (`:843-862`). Stage 1 NOOPs get the same absolute protection
  from the preflight guard.
- **UPDATE-replace / UPDATE-contradiction** introduce the one genuinely new capability:
  *call rewriting to a different backend op* (Stage 0 only rewrites to the read-only decoy).
  KV: `f"{tier}_memory_replace(key={m_i.ref!r}, value={new_value!r})"`; Vector:
  `f"{tier}_memory_update(vec_id={int(m_i.ref)}, new_text={candidate.text!r})"`. Mechanics:
  set `governed[idx]` to the rewritten call; record
  `_pending[idx] = {"mode": "rewrite", "original_call": ..., "rewritten_call": ...}`.
  In `patch_results`, a `"rewrite"` entry does **not** substitute a synthetic result (the real
  backend result flows through) and does **not** restore the original call string —
  `_observe()` must see the executed call so the mirror stays truthful; the original call is
  preserved in the log record instead. Preflight the *rewritten* op first; if it would fail,
  fall back to ADD (never rewrite into a guaranteed error).
  **Deliberate scope cut:** the plan's "old entry moves to archival" half of supersede depends
  on Section 5 placement machinery (not built). Until then, replace-in-place *is* the
  supersede, and the superseded value is logged in full (`superseded_text`) so the archival
  behavior can be replayed/added later without data loss. Flagged in §12.
- **ESCALATE** constructs the handoff object (§7.7) and calls Stage 2.

### 7.6 Config, shadow mode, logging

- `GOV_NLI_ENABLED` (default off until calibrated), `GOV_NLI_SHADOW=1` — full Stage 1
  computation and logging, zero interventions (mirrors `GOV_DRY_RUN`; per the recommendations
  doc's "add shadow mode to every stage"). First deployment is shadow-only on top of live
  Stage 0.
- Logging: extend the existing per-decision record (schema at `:807-839`) with a `stage1`
  object — `{precheck: str|null, candidates: [{ref, sim, tier}], pairs: [{ref, fwd: [c,n,e],
  bwd: [c,n,e], verdict}], outcome, target_ref, rewritten_call, superseded_text,
  latency_ms}`. Existing fields stay untouched so current log tooling keeps working.

### 7.7 What Stage 1 hands to Stage 2 (the contract)

```python
@dataclass
class EscalationContext:                      # new, in governance_filter.py
    candidate: WriteCandidate                 # op/kind/tier/backend/text/args/ref (:430-440)
    v_w: np.ndarray                           # whitened embedding (computed once in _govern_one)
    signals: Stage0Signals                    # incl. sims vector + verbatim fields
    neighbors: list[MemoryItem]               # Stage 1's evaluated candidates, desc. sim order
    nli_pairs: list[dict]                     # the all-neutral evidence (probs per pair)
    user_text: str                            # current user turn — Stage 2's probe source
    step: int                                 # session step counter
```

plus the live `GovernanceCache` (passed alongside, not copied — it *is* "the current memory
state at decision time", and Stage 2 must simulate against exactly it). `user_text` is captured
by two new thin overrides in `QwenGovHandler` — `_add_first_turn_message_prompting` and
`_add_next_turn_user_message_prompting` (both non-`@final`; the base loop calls the latter at
`base_handler.py:489`) — which stash the turn's user message on the session before the model
ever produces a call. This is the structural anti-circularity guarantee: probe text exists
before, and independent of, the model's generated key/value.

---

## 8. Stage 2 (Retrieval Entropy) implementation plan

Spec: Appendix A §4. Trigger: Stage 1 all-neutral, arriving as `EscalationContext` (§7.7).

### 8.1 Modules and entry point

- `bfcl_eval/model_handler/middleware/retrieval_sim.py` — the simulation core **shared with
  the §6 replay instrument** (build once, in §6, before Stage 1 even lands):
  - `simulate_kv(keys: list[str], probe: str) -> list[tuple[float, str]]` — `rank_bm25.BM25Plus`
    over key names only, tokenization byte-identical to `memory_kv._similarity_search`
    (`memory_kv.py:71-88`): `key.replace('_',' ').lower().split()`. Values are never scored —
    that is the BFCL KV reality the plan's §4.2 table encodes.
  - `simulate_vector(texts: list[str], probe: str, encode) -> list[tuple[float, int]]` — probe
    embedded with **raw MiniLM, `normalize_embeddings=True`**, cosine by dot product against
    stored-text embeddings. Exactness argument: the real path is FAISS `IndexFlatIP` over
    L2-normalized float32 vectors (`memory_vector.py:244-256, 309-333`) — an *exact* inner-
    product search, so numpy dot on the same normalized embeddings is the same ranking (no
    ANN approximation to worry about at N ≤ 57). **Never** the whitened space: whitening is
    Stage 0's decision space, not the backend's retrieval space (`governance_filter.py:25-26`
    already states the two-space separation; Stage 2 inherits it).
  - Per-tier fidelity: the real ops search one tier's store (`core_memory_key_search` searches
    core keys; `core_memory_retrieve` searches the core vector store). Simulation therefore
    runs against `cache.items[candidate.tier]` + the provisional candidate — not the merged
    matrix Stage 0 uses for redundancy.
  - `margin(scores)` = top1 − top2; `entropy(scores, T)` = softmax entropy; `n_eff` = `exp(H)`.
    Per-backend softmax temperature (`GOV_S2_T_KV`, `GOV_S2_T_VEC`) affects **logged H only** —
    the decision metric is margin, exactly per §4.4's scale-independence rationale.
- `bfcl_eval/model_handler/middleware/probe_gen.py`:
  - Channel (a), deterministic templates from `ctx.user_text` + the candidate's already-
    extracted `signals.verbatim_values` / content tokens (reusing `extract_verbatim_values` and
    `_content_tokens`, `governance_filter.py:383-416`): e.g. `"what is {attribute} of
    {entity}"`, `"{entity} {attribute}"` keyword probes. No model.
  - Channel (b), 2–3 paraphrases of `ctx.user_text` from a **separate small CPU model of a
    different family than Qwen** (env `GOV_PROBE_MODEL`; recommendation: a T5-family
    paraphraser, e.g. flan-t5-small with a paraphrase prompt — final choice is a calibration-
    time decision, flagged in §12). `GOV_PROBE_PARAPHRASE=0` degrades to template-only (the
    §6 replay mode).
  - Hard invariant, asserted in code: probe inputs are `ctx.user_text` and extraction fields
    only — `candidate.text`/`candidate.args` never enter probe text (the §4.1 structural
    circle-break). Total 3–5 probes; decision on **min** margin (worst case).
- `Stage2Probe.resolve(ctx, cache, session) -> Stage2Outcome` orchestrates:

```
probes = probe_gen(ctx)                              # 3–5, provenance-tagged
sim_state = tier keys/texts + provisional candidate   # backend-faithful, per-tier
for p in probes:  ranked_p = simulate(sim_state, p)
ok    = all(top1(ranked_p) is candidate) and min_margin > GOV_S2_MARGIN
if ok:                        -> ACCEPT (decision ADD; write proceeds unchanged)
else:                         -> CANONICALIZE once (8.2), re-test same probes once
      pass                    -> ACCEPT_REWRITTEN (rewrite call via §7.5 machinery)
      fail                    -> ACCEPT_LOW_CONFIDENCE (write as-is; flag + log)
```

No generate-and-race: one candidate, one repair attempt, same probes — §4.3's clarified
verification loop, encoded in the control flow (there is no code path that generates a second
candidate).

### 8.2 One-shot canonicalization

- **Deterministic first (v1):** pick the highest-value discriminative token — present in the
  candidate's value/user sentence, absent from every ambiguous rival's key (KV) or text
  (Vector); preference order: `verbatim_values` entities (drug names, units) → content tokens.
  KV: `new_key = f"{old_key}_{token}"`, validated against `_KV_KEY_PATTERN`
  (`governance_filter.py:62`) and length caps; Vector: prepend an entity+attribute clause to
  the value text, re-checked against `max_entry_length` (the customer-6 forensics in §2.4 is
  the cautionary tale: a rewrite that exceeds 2000 chars turns into a guaranteed backend
  error).
- **LLM canonicalization (v2, behind `GOV_S2_CANON_LLM=1`):** the session has no LLM client by
  design (pure middleware); `QwenGovHandler` injects a `canonicalize_fn` callback built on its
  own vLLM client (same pattern as MIG's lazily-constructed reranker,
  `qwen_mig.py:58-67`). Keeps `governance_filter.py` free of BFCL/network imports.
- The rewritten call goes through the §7.5 rewrite machinery, including preflighting the
  rewritten op.

### 8.3 Diagnostics: `ΔH_neighbor` and `N_eff` (the §8.4 feed)

Logged on **every** Stage 2 trigger, decision-inert per §4.4:

- Per probe: `{probe, channel, top1_ref, margin, H, n_eff}`; plus `min_margin`, `worst_probe`.
- `dH_neighbor`: for each `m_i` in `ctx.neighbors`, re-score *`m_i`'s own cached probes*
  against the tier state with vs without the provisional candidate:
  `[{ref, H_before, H_after, dH}]` (+ mean/max). This is §4.4's `ΔH_komşu`, computed by the
  exact same `retrieval_sim` calls — "ek maliyet yalnızca yeniden skorlama".
- **Neighbor probe cache** (the §4.4 premise "probes are generated when items are written"):
  `MemoryItem` gains `probes: list[str]` — template-channel probes generated at observe time
  for every genuinely written item (`_put_item`, `governance_filter.py:915-924`; cheap and
  deterministic). Persistence across conversations via a sidecar
  `<scenario>_gov_state.json` written next to the backend snapshot (path already known to the
  session via `snapshot_path`, `qwen_gov.py:116-149`) and read during `rehydrate()`; on a cache
  miss (pre-existing item, no sidecar) probes are regenerated from the stored text — flagged
  in the log as `probe_provenance: "stored_text"` since that channel is circularity-weaker
  (acceptable for diagnostics, never used for accept/reject).
- Config: `GOV_S2_ENABLED`, `GOV_S2_SHADOW`, `GOV_S2_MARGIN` (initial value set from the §6
  replay's margin distribution — same measurement, same code), `GOV_S2_PROBES_N=4`,
  `GOV_S2_T_KV`, `GOV_S2_T_VEC`.
- Log record: `stage2` object mirroring `stage1`'s style — `{probes: [...], per_probe: [...],
  min_margin, outcome, canonicalization: {applied, token|null, rewritten_call|null,
  retry_min_margin|null}, low_confidence, dH_neighbor: [...], latency_ms}`.

This section and §6 are the same machinery by construction: the replay instrument *is*
`retrieval_sim.py` + template-`probe_gen` run offline over logs; Stage 2 adds the paraphrase
channel, the decision rule, canonicalization, and the live probe cache. The go/no-go study can
therefore run before, during, or after Stage 1/2 development with no duplicated code.

---

## 9. Interconnection / data-flow summary (Stage 0 → 1 → 2)

```
model write call (add/replace/update)
        │  govern_calls() → build_candidate()                     [governance_filter.py:766]
        ▼
Stage 0  _govern_one(): v_w = _whiten_one(text)                   [:781]
         compute_signals(): sims = M @ v_w  (KEEP whole vector)   [:469 + new Stage0Signals.sims]
         decide(): NOOP | ADD | ESCALATE                          [:518]
        │ NOOP → preflight → DECOY_CALL + synthetic_success       [existing]
        │ ADD  → write unchanged                                  [existing]
        ▼ ESCALATE
Stage 1  select_candidates(): argsort(-signals.sims)[:k]          [reuses cache.matrix() items]
         + canonical-key channel (KV: candidate.ref lookup)
         pre-check: KV add-on-existing-key → UPDATE_REWRITE (no NLI)
         nli_gate: 2k DeBERTa-MNLI calls
        │ NOOP           → same decoy/synthetic path as Stage 0
        │ UPDATE-*       → rewrite governed[idx] to replace/update; real result flows
        │ keep-both      → ADD
        ▼ all-neutral ESCALATE
         EscalationContext{candidate, v_w, signals, neighbors, nli_pairs, user_text, step}
Stage 2  probe_gen(user_text, verbatim_values)  — NEVER candidate.text
         retrieval_sim per backend/tier: BM25Plus(keys) | raw-MiniLM cosine(texts)
         min-margin decision → ACCEPT | canonicalize-once → retry | low_confidence
         diagnostics: H, N_eff, ΔH_neighbor (neighbors' cached probes)
        ▼
        write (possibly rewritten), observe → _put_item(+probes) → sidecar persist
```

| Producer | Artifact | Consumer | Shared structure |
|---|---|---|---|
| Stage 0 `compute_signals` | `Stage0Signals.sims` + `v_w` | Stage 1 ranking; Stage 2 context | `Stage0Signals` (extended) |
| Stage 0 `GovernanceCache` | `matrix()`, `items`, `MemoryItem` | Stage 1 candidate texts; Stage 2 simulation state + probe cache | `MemoryItem` (+`probes`, existing `low_confidence`) |
| Stage 0 suppression machinery | preflight / decoy / `synthetic_success` / `_pending` / `patch_results` | Stage 1 NOOPs; Stage 1+2 call rewrites (`_pending` gains `mode`) | `GovernanceSession._pending` |
| handler (`qwen_gov.py`) | current user turn text | Stage 2 probes | `EscalationContext.user_text` |
| Stage 1 | `EscalationContext` | Stage 2 entry | defined once in `governance_filter.py` (§7.7 ≡ §8.1 input — same object) |
| Stage 2 | `low_confidence`, probe cache, `ΔH_neighbor` logs | §8.4 gate; Month-4 diagnostics; future Section 5 placement | JSONL `stage2` block; sidecar `<scenario>_gov_state.json` |

Consistency check between the two plans: Stage 1's handoff (§7.7) and Stage 2's consumed input
(§8.1) reference the same `EscalationContext` definition; both stages' rewrite outcomes go
through the same extended `_pending` mechanism with the same preflight rule; the neighbor list
Stage 2 re-scores is exactly the candidate list Stage 1 evaluated (no second ranking); and the
metadata cache schema is the single `MemoryItem` dataclass with two additive fields. No
reconciliation gaps found after cross-checking.

---

## 10. Recommendations against the master plan, section by section

- **§1.1 (ABTT):** Done and committed with reproducibility metadata. The anisotropy premise is
  falsified (0.096 vs ~0.83) — keep ABTT but rewrite its justification, and run the
  with/without-ABTT ablation on the replay instrument (§5.5). The plan's own optional
  domain-vs-general-corpus ablation can ride the same instrument later.
- **§1.2 (two-space separation):** Correctly implemented and documented in code
  (`governance_filter.py:25-26`); Stage 2's raw-space simulation (§8.1) completes the design.
- **§1.3 (metadata cache):** Exists as `MemoryItem`/`GovernanceCache`, deliberately without a
  usage-count field, per the plan. `kategori` (identity/event) does not exist yet — it belongs
  to Section 5 placement and should be added only when §5 is built.
- **§2 (Stage 0):** Built, tested, calibrated once. Outstanding: per-backend `sim_high`
  decision (§5.4), coupled-pair validation (§5.3), replicates before committing defaults.
- **§3 (Stage 1):** Not built. Plan in §7 above. Correction to carry into the thesis text: the
  canonical-key channel is KV-only (§2.5-4), and per §5.4 it should be presented as KV's
  *primary* redundancy detector rather than an auxiliary channel.
- **§4 (Stage 2):** Not built. Plan in §8 above. Its simulation core doubles as the §8.4
  instrument — build that first (§6).
- **§5 (placement/eviction/destructive-op protection):** Not built; nothing in `_observe` even
  logs remove/clear today (§2.5-6). Ranking *down* relative to the plan's Month-1 "most
  guaranteed win" framing: the logged evidence shows destructive ops are rare in these runs,
  while write-compliance failure (student; healthcare-in-baseline) is the dominant loss
  mechanism — `GOV_CASCADE_PLAN_RECOMMENDATIONS.md` §1 already argued this and the verified
  numbers back it. Keep §5 after Stage 1 (Stage 1's supersede path creates the first real
  consumer of archival moves); implement the small logging fix (observe remove/clear) now.
- **§6 (read-time mechanism):** Not built. The MIG track has *pre-paid* part of this work:
  read-time interception infrastructure exists (`mig_reranker.py` parse/reserialize), and —
  more importantly — MIG's evidence shows answer-time retrieval almost never happens, which
  caps §6's possible impact just as it capped MIG's. Before building §6, measure retrieval
  frequency in the replicate runs; if it stays near zero, §6 should be re-scoped (or the
  thesis should report *why* it cannot bind), and MIG-judge becomes an optional §8.2 ablation
  arm rather than new engineering (§4.2).
- **§7 (calibration):** The plan's grid-search-on-W/R/R/A approach is superseded by the E11
  lesson: calibrate on decision-log geometry (dense), validate on accuracy (sparse). Concretely
  outstanding: coupled (sim_high, δ) from replicated shadow logs; NLI thresholds
  (τ_entail, τ_contra, δ_spec) from a labeled sample of the ~194 logged escalations (they are
  already "exactly NLI-shaped"); Stage 2 margin from the §6 replay distribution. W/R/R/A
  remains the validation metric but requires the snapshot parser (§8.1 of the plan) which is
  **not built** — that is a real instrumentation gap.
- **§8.1–8.3 (evaluation instruments/statistics):** W/R/R/A parser missing; entry-count
  instrumentation effectively exists (snapshot counts + logs); §8.3's statistics need the
  survival-conditioning amendment and pre-registered student exclusion (§5.2). The §8.2
  cascade ablations become meaningful only after Stage 1/2 exist; the entropy-contribution
  retro-analysis is already designed into Stage 2's logging (§8.3).
- **§8.4 (go/no-go):** Runnable now via §6, before any Stage 1 code. Sequencing answer to the
  headline question: **resolve the gate first** — it is 1–2 days of offline work, its
  instrument is Stage 2 production code (zero waste under either outcome), and its result
  changes Stage 1's *shape* (geometry-first vs NLI/canonical-key-first ordering, likely split
  per backend per §5.4/§6) though not Stage 1's *necessity*. Building Stage 1 blind and
  re-ordering the cascade afterwards would cost more than the gate does.
- **§9 (timeline):** Month 1 partially done (baselines ✓, ABTT ✓, W/R/R/A parser ✗, mandatory
  archiving ✗ — deliberately deprioritized above); Month 2 done except the go/no-go study
  itself (the exact gap §6 closes); Month 3 (Stages 1–2 + calibration) is next and is where
  this report's Part B lands; Month 4's read-time work is partially pre-paid by MIG
  (infrastructure + the negative frequency finding) but its ablation/statistics work remains.
  Net: roughly on the Month-2/Month-3 boundary, with the go/no-go study as the one Month-2
  deliverable that must be back-filled, and with §5 consciously moved behind Stage 1.
- **§10 (cost):** Stage 0's measured 0.53 ms confirms the budget line; Stage 1's ~2k CPU-NLI
  calls per escalation and Stage 2's ms-scale simulation stay within the table's envelope; the
  only new cost item vs the plan is the optional paraphrase model, already envisioned there.

---

## 11. Prioritized next steps

1. **Build `retrieval_sim.py` + template `probe_gen` + the replay driver, and run the §8.4
   correlation study on the existing ~1,051 logged decisions** (per backend; ABTT-vs-identity
   ablation in the same pass). 1–2 days, no GPU. Its outcome fixes the cascade's front-end
   shape before any Stage 1 code is written. While in there, apply the three logging fixes:
   full candidate/observe text (or sidecar), observe-events for remove/clear, ABTT metadata in
   the rehydrate record.
2. **Run the de-noising replicate protocol (§5.2)** — 3× sequential baseline vs
   governed-calibrated, survival-conditioned pairing, student pre-registered as excluded —
   ideally overlapping with step 1 since it is GPU-bound while step 1 is not. This is what
   converts the Vector drift (11.61→12.90) from anecdote into a defensible (or refuted) claim
   and is the precondition for committing any `GovConfig` defaults.
3. **Implement Stage 1 per §7, shadow-mode first** (`GOV_NLI_SHADOW=1` on top of live
   Stage 0), with the gate result from step 1 deciding whether the canonical-key pre-check
   leads (KV certainly; Vector depending on ρ). Calibrate NLI thresholds from the logged
   escalation band before enabling interventions.
4. **Implement Stage 2 per §8** on top of the step-1 core (paraphrase channel,
   canonicalization, probe cache/sidecar, `ΔH_neighbor` live logging), shadow-mode first.
5. **Then** revisit: committing calibrated defaults (with the coupled-pair validation), the
   W/R/R/A snapshot parser, Section 5's archival machinery (now with Stage 1's supersede as its
   first consumer), and the Section 6 re-scoping decision informed by measured retrieval
   frequency (MIG-judge as optional ablation arm only).

---

## 12. Open risks and unresolved questions

- **The go/no-go gate could fail for Vector too** (ρ < 0.4 everywhere). The plan's fallback
  (NLI-first) is well-defined, and the Stage 1 plan here survives reordering (the pre-check and
  NLI engine are order-independent; only `decide()`'s branch ordering changes) — but the thesis
  narrative around Stage 0 would need honest restructuring toward "geometry as cheap
  short-circuit, not primary filter".
- **Replicates may show the Vector drift is pure noise.** Then Stage 0's accuracy case rests
  entirely on the gate correlation + mechanism validation until Stages 1–2 add suppression
  volume. That is a publishable but weaker position; the report should not promise more.
- **Stage 1 supersede without Section 5** replaces-in-place rather than archiving the old
  entry (§7.5). Information loss is prevented only by the `superseded_text` log. If a recall
  question later targets a superseded value, this design choice costs accuracy; monitor via
  the replicates' failure analysis.
- **Paraphrase-model choice for Stage 2 is unresolved** (family, size, offline availability on
  the BSC/Windows setups). Template-only mode is the guaranteed fallback; flagged as a
  calibration-time decision, and the §8.4 study deliberately does not depend on it.
- **KV per-backend `sim_high` (≈0.92) remains untested** — a KV-only shadow sweep is cheap but
  the §5.4 evidence suggests low yield; deprioritized, not closed.
- **Windows/encoding environment fragility** (cp1254 crashes, `PYTHONUTF8=1`) is a documented
  recurring tax on every run; the replicate protocol inherits the reproduction guide's
  mitigations but this remains an operational risk.
- **Open question for the author:** the SE track's negative result and the MIG track's
  invalidated result are currently only in their own reports — decide whether the thesis
  treats them as first-class negative results (recommended; they motivate the write-time
  focus) or as appendix material.
- **Open question for the author:** `result_gov_shadow_partial/` (unequal kv/vector entry
  counts, never scored) has served its purpose; decide whether to keep it in the repo with a
  README note or delete it to reduce future confusion. Same for `tmp/smoke_archive_2026-07-01/`.

---

## 13. Appendix — evidence files used

**Source code (read in full or in relevant part):**
- `bfcl_eval/model_handler/middleware/governance_filter.py` (all Stage 0 logic; line refs throughout)
- `bfcl_eval/model_handler/local_inference/qwen_gov.py` (handler hooks)
- `bfcl_eval/scripts/compute_abtt.py`; artifact `bfcl_eval/model_handler/middleware/artifacts/abtt_minilm_l6_d16.npz` (embedded JSON metadata parsed directly)
- `bfcl_eval/scripts/test_gov_offline.py` (assertion census)
- `bfcl_eval/model_handler/middleware/mig_reranker.py`, `bfcl_eval/model_handler/local_inference/qwen_mig.py`, `bfcl_eval/scripts/mig_live_experiment.py`, `bfcl_eval/scripts/analyze_mig_traces.py`, `mig_traces/exp_2026-07-01/`
- `bfcl_eval/model_handler/middleware/semantic_entropy.py` (+ `qwen_se.py` registration)
- `bfcl_eval/constants/model_config.py:1664-1717`, `bfcl_eval/constants/supported_models.py:142-145`
- `bfcl_eval/eval_checker/multi_turn_eval/func_source_code/memory_kv.py` (BM25Plus search :71-88; op semantics), `.../memory_vector.py` (FAISS IndexFlatIP :239-333; no key field)
- `bfcl_eval/model_handler/base_handler.py` (hook finality; `:489` user-message call site)
- `bfcl_eval/scripts/audit_memory_pipeline.py`, `bfcl_eval/scripts/analyze_memory_score_discrepancy.py` (student-scenario documentation)

**Logs (parsed programmatically):**
- `gov_logs/shadow/governance_log.jsonl` (497 lines), `gov_logs/governed_sage/governance_log.jsonl` (890), `gov_logs/governed_calibrated/governance_log.jsonl` (999)

**Results/scores (summary lines read; per-scenario failure ids recomputed):**
- `score/Qwen_Qwen3-4B-Instruct-2507-FC/agentic/memory/{kv,vector}/BFCL_v4_memory_*_score.json`
- `score/Qwen_Qwen3-4B-Instruct-2507-FC-GOV/agentic/memory/{kv,vector}/...`
- `score_gov_sage/agentic/memory/{kv,vector}/...`
- `score_baseline/Qwen_Qwen3-4B-Instruct-2507-FC/agentic/memory/vector/...`
- `score_mig_judge/...-FC-MIG/agentic/memory/vector/...`, `score_mig_logprob/...`
- `score_se_exp/{baseline,gated}/.../agentic/memory/{kv,vector,rec_sum}/...`
- `result/{...-FC,...-FC-GOV}/agentic/memory/{kv,vector}/BFCL_v4_memory_*_result.json` (+ `memory_snapshot/*_final.json` entry counts; customer-6 error forensics)
- `result_baseline/`, `result_gov_sage/`, `result_gov_shadow_partial/`, `result_mig_judge/`, `result_mig_logprob/`, `result_se_exp/` (entry counts, snapshot counts, category coverage)
- `tmp/smoke_archive_2026-07-01/` (identified as smoke leftovers only)

**Reports (read in full):**
`GOV_LOCAL_EXPERIMENTS_COMPARISON.md`, `GOV_CASCADE_PLAN_RECOMMENDATIONS.md`,
`BFCL_MIG_EXPERIMENT_REPORT.md`, `BFCL_MIG_EXPERIMENT_RESULTS.md`, `MIG_VALIDATION_REPORT.md`,
`BFCL_MIG_RERANKER_GUIDE.md`, `BFCL_MIG_RERANKER_IMPLEMENTATION_PLAN.md`,
`BFCL_SE_EXPERIMENT_REPORT.md`, `memory_pipeline_audit_report.md`,
`memory_score_discrepancy_report.md`, `BFCL_MEMORY_REPRODUCTION_GUIDE.md`,
`BFCL_PURE_REBENCHMARK_GUIDE.md`, `LOG_GUIDE.md`, `summarize1-4.md`

**Git:** `git log`/`git diff` between `just_geo_filtering` and the working branch (identical
trees); commit provenance for result dirs (`84d831d`, `b14b043`, `94afae4`).
