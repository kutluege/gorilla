# BFCL v4 Memory — Experiments Overview

**Last updated:** 2026-07-19
**Model under test:** `Qwen/Qwen3-4B-Instruct-2507` (served by vLLM, temperature 0.001)
**Benchmark:** BFCL v4 Memory (`memory_kv`, `memory_vector`, `memory_rec_sum`) — 155 answer questions + 37 prerequisite (memory-writing) entries per backend, across 5 scenarios (customer, student, finance, healthcare, notetaker)
**Grader:** unmodified official BFCL `agentic_checker`

This document explains, in plain terms, every method tried so far in this repo, what the results were, and how the methods compare. Source documents: `BFCL_MIG_EXPERIMENT_REPORT.md`, `BFCL_MIG_EXPERIMENT_RESULTS.md`, `MIG_VALIDATION_REPORT.md`, `BFCL_SE_EXPERIMENT_REPORT.md`, `GOV_LOCAL_EXPERIMENTS_COMPARISON.md`, `GOV_CASCADE_PLAN_RECOMMENDATIONS.md`, `STAGE0_RESULTS_ANALYSIS_AND_NEXT_STEPS.md`, `PLAN_1/2/3_*.md`, `gov_logs/plan3_calibration.md`.

---

## 0. The problem being attacked

In BFCL v4 Memory, an agent first goes through **prerequisite turns** where it must *write* facts into a memory store (key-value, vector, or recursive-summary), and later must *answer questions* by reading that memory back. A small model (Qwen3-4B) does this badly: baseline accuracy is roughly 7–17% on KV/Vector and ~25–34% on rec_sum, depending on the run.

Three ideas were tried, each targeting a different point in the pipeline:

| # | Method | Where it intervenes | Core idea in one line |
|---|---|---|---|
| 1 | **MIG reranker** (`-FC-MIG`) | **Read time** (retrieval results) | Widen the retrieval pool, then re-rank memories by how much each one changes the model's draft answer (information gain). |
| 2 | **Semantic-Entropy gate** (`-FC-SE`) | **Write time** (generation) | Sample the model N times per memory step, cluster the samples semantically, commit only the majority action; block destructive ops when the samples disagree (high entropy). |
| 3 | **Geometric Governance filter / cascade** (`-FC-GOV`) | **Write time** (middleware) | Before a `memory_add/update/replace` lands, check geometrically (embedding similarity + residual) whether it is redundant → suppress it as a NOOP; escalate ambiguous cases to later cascade stages (NLI, retrieval-entropy). |

---

## 1. Method 1 — MIG reranker (information-gain retrieval reranking)

### What was tried, simply

When the agent calls `archival_memory_retrieve`, the retriever returns a handful of memories ranked by embedding similarity. The MIG middleware widens that pool (e.g. to 20 candidates), then re-scores each candidate by asking: *"how much does adding this memory change the model's draft answer?"* Two scorers were built: a **judge** scorer (the model rates relevance) and a **logprob** scorer (log-probability of the draft answer with vs. without the memory). Only the top-k after reranking are given back to the agent.

### Experiments run

**(a) Offline/live validation (2026-07-01)** — 22/22 machinery checks passed. When the gold memory is in the pool, both scorers rank it #1. The implementation is correct.

**(b) Controlled retrieval experiment (2026-06-27)** — not the full benchmark; a purpose-built harness where each question gets a pool = the gold fact + hard distractors. Results (answer accuracy):

| Setting | full pool | similarity top-k | **MIG judge** | MIG logprob |
|---|---|---|---|---|
| customer, random distractors, P10/k3 | 0.867 | 0.833 | **0.867** | 0.833 |
| customer, hard distractors, P10/k3 | 0.867 | 0.767 | **0.900** | 0.867 |
| healthcare, hard, P10/k3 | 1.000 | 0.920 | **1.000** | 0.960 |
| customer, hard, tight budget P12/k1 | 0.867 | 0.733 | **0.867** | 0.833 |

Gold-selection recall@k for MIG-judge was **1.000 in every cell** (similarity baseline: 0.80–0.97). The judge scorer won every cell, and the margin grew as the budget tightened and distractors got harder.

**(c) Full-pipeline A/B (2026-07-01)** — 55-question vector-only subset of the real benchmark:

| Arm | memory_vector (55 entries) |
|---|---|
| Baseline `-FC` | 14.55% (8/55) |
| MIG judge | 12.73% (7/55) |
| MIG logprob | 16.36% (9/55) |

**These deltas are declared invalid as a treatment effect** in the report itself, for two reasons:
1. Across 55 answer entries, the agent invoked retrieval **0–1 times** — a reranker that is never called cannot move the score.
2. A confound: MIG was also active during the memory-*writing* phase, so each arm built a *different* memory store, making cross-arm comparison meaningless.
3. In the single genuine answer-time interception, the retriever missed the gold memory entirely (recall miss) — reranking can't fix a pool that doesn't contain the answer.

### Verdict

**Mechanism proven, benchmark effect ~zero.** MIG reliably picks the right memory *when the right memory is in the pool*, but on this benchmark the agent almost never retrieves at answer time, and when it does, recall (not ranking) is the bottleneck. Decision: keep MIG out of the governance cascade; report it as an independent read-time result.

---

## 2. Method 2 — Semantic-Entropy gate

### What was tried, simply

At every memory step, instead of taking the model's single output, sample it **N times**, cluster the samples by meaning, and commit the **majority** action. Destructive operations (delete/overwrite) are only allowed when the samples agree (low semantic entropy); otherwise a safe fallback runs (e.g. an add or a read instead of a delete).

### Experiment run — full 155-entry A/B on all 3 backends

| Backend | Baseline | SE-gated | Delta | McNemar p | 95% CI (delta) |
|---|---|---|---|---|---|
| memory_kv | 15.48% (24/155) | 11.61% (18/155) | −3.87% | 0.24 | [−9.03, +1.29] |
| memory_vector | 17.42% (27/155) | 16.13% (25/155) | −1.29% | 0.86 | [−8.39, +5.81] |
| memory_rec_sum | 27.10% (42/155) | 26.45% (41/155) | −0.65% | 1.00 | [−6.45, +5.16] |
| **overall** | **20.00%** | **18.06%** | **−1.94%** | 0.35 | [−5.59, +1.72] |

Gate activity across 2,024 gated steps: 1,821 majority commits; 31 destructive + 172 overwrite majorities proposed; **21 risky operations blocked** with a safe fallback; only 2 forced through with no safe alternative. Prereq steps were much more uncertain (mean entropy 1.36 bits, 38% unanimous) than recall steps (0.52 bits, 60% unanimous).

### Verdict

**The gate works as a safety mechanism (it really does catch and divert risky deletes), but it did not improve accuracy** — every delta is slightly negative and statistically indistinguishable from zero. The uncertainty signal is real (writes are measurably noisier than reads), but converting it into accuracy gains didn't happen at this model scale.

---

## 3. Method 3 — Geometric Governance filter (Stage 0 of a planned cascade)

### What was tried, simply

A write-time middleware sits in front of `memory_add/replace/update` (KV + Vector backends). Each incoming write is embedded (MiniLM + a one-time "ABTT" whitening fitted on the benchmark corpus), compared against existing memories, and classified:
- **NOOP** — near-duplicate of something already stored → suppress the write (rewrite the call into a decoy, return a synthetic success so the agent continues normally);
- **ESCALATE** — ambiguous (similar but not identical) → in Stage 0, falls back to a normal write; in the full cascade this goes to Stage 1 (NLI entailment) and Stage 2 (retrieval-entropy simulation);
- **Novel** → write normally.
A preflight guard refuses to mask genuine backend errors.

### Experiments run (2026-07-13, three full generation runs + shadow run)

**Headline accuracy:**

| Arm | memory_kv | memory_vector | Config |
|---|---|---|---|
| Baseline `-FC` | 7.10% (11/155) | 11.61% (18/155) | no middleware |
| GOV — SAGE defaults | 21.29% (33/155) | 12.26% (19/155) | sim_high=0.80, δ=0.025 |
| GOV — calibrated | 16.77% (26/155) | 12.90% (20/155) | sim_high=0.95, δ=0.30 |

**⚠️ The KV numbers must not be read at face value** — see §4 (the confound). The governed runs "winning" by +10–14 points on KV is dominated by whether the healthcare scenario's memory chain survived, not by governance.

**Governance decisions (the mechanism-level view, which *is* valid):**

| Arm | Gated decisions | NOOPs fired | Preflight blocks | Escalate→fallback |
|---|---|---|---|---|
| Shadow (partial) | 262 | 1 would-fire | 0 | 45 |
| GOV — SAGE defaults | 342 | **3** (all vector, sim=1.0) | 1 | 81 |
| GOV — calibrated | 447 | **5** (all vector, sim 0.95–1.0) | 24 | 68 |

Key mechanism findings:
- All 9 suppressions across all runs were **genuine redundant rewrites** (e.g. the same medication dosage re-written at sim 0.992; a byte-identical profile at sim 1.0). No false suppressions observed.
- **Zero KV suppressions ever** — KV's composite `"key: value"` embedding depresses similarity below any reasonable threshold; KV needs a canonical-key check instead of geometry.
- SAGE default thresholds are **near-inert by construction**: the geometric constraint `r ≈ √(1−sim_max²)` means δ=0.025 only fires at sim ≳ 0.9997. The calibrated pair (0.95, 0.30) fixes this but is itself borderline-degenerate (√(1−0.95²)=0.312 > 0.30) and is now guarded by `GovConfig.validate()`.
- Decision latency: **0.53 ms** per write at maximum memory size — well within budget.
- ~15–20% of gated writes land in the escalate band (sim 0.90–0.95) — "exactly NLI-shaped", motivating Stage 1.
- A premise of the original plan was falsified: the embedding space is *not* highly anisotropic on this corpus (median pairwise cosine 0.096, not ~0.83).

**§8.4 offline replay (go/no-go for the cascade, run 2026-07-17)** — replaying 1,051 logged decisions to test whether geometry (similarity) actually predicts information change (ΔH): 713/1,051 decisions computable (67.8%, not the 94.5% previously quoted). Correlation ρ(sim_max, ΔH): **KV +0.48 to +0.66 across arms (passes the ≥0.40 gate everywhere); Vector +0.41 and +0.59 in two arms but −0.16 in the calibrated arm** (a range-restriction artifact). Verdict: geometry stays as the front filter; the Vector duplicate band is cleanly separated (dup sim 0.99–1.00 vs non-dup 0.54–0.61).

**Plan 3 calibration (2026-07-18, live shadow harvest, 305 decisions):** Stage-2 margin threshold chosen `GOV_S2_MARGIN=0.01` (live p75); read-time margin `GOV_READ_MARGIN=0.0537` for Vector (KV margins are degenerate at exactly 0.0, so KV uses a bounded ambiguous-set mechanism instead); geometry-first ordering confirmed binding (Vector ρ(sim_max, ΔH)=0.757, CI [0.48, 0.90]).

**The decisive replicated campaign (2026-07-18/19, one 48 h vLLM instance)** — 5× survival-conditional A/B + 8-arm × 3-replicate ablations, full results in `berkeley-function-call-leaderboard/EVAL_RESULTS.md`:

| Test | Result |
|---|---|
| 5× A/B, official score (PRIMARY) | **Null.** KV ΔAcc −0.7 pp (McNemar p=0.80), Vector ΔAcc 0.0 (p=1.00, discordants 35/35); CI95 ≈ ±4–5 pp. Full cascade governs ~1,800 writes (30 NOOP, 18 canonicalizing rewrites) at zero measurable accuracy cost. |
| Ablations, 8 arms × 3 reps, Holm m=14 | **Only `geo_off` (NLI-first) survives correction: KV −10.6 pp, p=0.0001, Holm p=0.0013.** Removing the geometric front stage significantly hurts — causal confirmation of geometry-first. `p_only` KV −7.9 pp marginal (Holm 0.06); everything else null; no arm improves the official score. |
| Chain survival (diagnostic) | Arm-independent: student dead in every run of every arm; healthcare flickers in baseline and governed alike. The earlier single-run "governance revives chains" (3/5→4/5) did **not** reproduce — it was survival lottery. |
| Paraphrase-tolerant judge (SECONDARY) | Governed gains more than baseline under NLI-entailment tolerance (Vector 121 vs 112 /775) — canonicalized memories answer correctly but miss grader keywords. Reported beside, never instead of, the official score. |
| G11 entropy retro-check (diagnostic) | The inert entropy signal (dH_mean) separates ~6 pp of scenario accuracy *within* both margin strata — information the deciding margin gate ignores. Future-work lever, changed nothing in this campaign. |

### Verdict

**Mechanism validated; cascade is accuracy-neutral; the geometric stage is provably load-bearing.** The 5× replicated A/B resolves the earlier "not yet measurable" state: at these calibrations the cascade neither helps nor hurts the official score, while the ablation family shows that *removing* geometry (NLI-first ordering) is the one significantly harmful configuration (and ~1.8× slower per run). The earlier single-run KV gains (8.8% vs 22.5%, p=0.043) are now conclusively attributable to chain-survival noise.

---

## 4. The cross-cutting discovery: the benchmark is noisier than any treatment

This is arguably the most important empirical finding of the whole effort, because it reframes every accuracy table above.

1. **The same baseline model scored 7.10%, 11.61%, 13.55%, and 15.48% on memory_kv** in four nominally identical runs on different days/servers. Single-run deltas below ~±8–10 points on this benchmark are not evidence of anything.
2. **The dominant failure mode is that the model writes nothing.** The `student` scenario stored **zero** memories in every arm of every experiment — the model chit-chats through all 10 prereq turns without one tool call — so ~50 of 155 questions are dead weight in every run. `healthcare` died in the baseline governance run but survived in both governed runs; that accident alone explains most of the baseline's 7.10% vs the governed arms' 16–21%.
3. Consequence: **all future comparisons must be replicated (N≥5, sequential, single-threaded) and conditioned on memory-chain survival**, with the student scenario pre-registered as excluded. This protocol was executed in full on 2026-07-18/19 (5× A/B + 24 ablation units, zero failures) and confirmed the premise: within-arm replicate swings reached ±8–11 questions (governed KV 17→28 across identical replicates) — larger than every arm-vs-arm delta except `geo_off`.

---

## 5. Comparing the three methods

| | MIG reranker | SE gate | Governance filter |
|---|---|---|---|
| Intervention point | Read time (retrieval) | Write time (sampling) | Write time (middleware) |
| Extra cost | ~pool_size extra model calls per retrieval | N× model calls per memory step (most expensive) | ~0.5 ms per write, no extra model calls (cheapest) |
| Mechanism verified? | ✅ Yes — recall@k 1.000 when gold in pool (controlled exp) | ✅ Yes — 21 risky ops correctly diverted | ✅ Yes — 9/9 suppressions genuine duplicates, 0 false positives |
| Benchmark accuracy effect | ~None (invalid/neutral; retrieval fires 0–1× per 55 questions) | Slightly negative, not significant (−1.9% overall, p=0.35) | **Null, decisively measured** (5× A/B: KV −0.7 pp p=0.80, Vector 0.0 p=1.0); removing geometry is the only significant config (−10.6 pp, Holm p=0.0013) |
| Why it couldn't move the score | Agent rarely retrieves; when it does, recall (not ranking) misses gold | Uncertainty signal real, but majority-commit ≈ what the model did anyway | Genuine redundancy is rare (~2% of writes NOOP/rewritten); benefit shows only under paraphrase-tolerant judging (secondary) |
| Statistical rigor of the test | Weak (single partial run, confounded) — controlled exp was clean | Good (full 155×3, McNemar + CI) | **Strongest of the three** (5×+3× replicated, survival-conditional, exact McNemar, bootstrap CI, Holm m=14) |
| Status | Concluded; shelved from cascade | Concluded | **Concluded** — campaign complete (EVAL_RESULTS.md); future work: entropy-aware gating, keyword-preserving canonicalization |

### The narrative comparison

All three methods share one fate: **the mechanism works, but the bottleneck lies elsewhere.**

- **MIG** solves ranking, but the agent's problem is that it doesn't retrieve at all (and when it does, recall misses). Fixing ordering inside a pool that is empty or wrong changes nothing.
- **SE** detects uncertainty accurately, but converting "the model is unsure" into better memories mostly reproduced what the model would have done anyway, at N× inference cost — and the measured effect was mildly negative.
- **Governance** correctly and cheaply suppresses redundant writes, but genuine redundancy is rare (~1% of gated writes), so the ceiling of Stage 0 alone is tiny. Its real value so far is diagnostic: it produced the logging that exposed the survival-noise confound and the escalation band that motivates the NLI/entropy stages.

The honest meta-conclusion, stated in the source docs and worth repeating: on a 4B model, the binding constraints of BFCL v4 Memory are (1) **write compliance** — the model often never writes memories at all, and (2) **run-to-run variance** larger than any treatment tried. Any future method must first fix or control for those (Stage −1 watchdog, survival-conditioned replicates) before its accuracy effect can even be observed. The governance track is the only one that did control for them — and under that control, its effect is decisively null on the official score, with the geometric ordering of the cascade the one component whose removal measurably hurts.

---

## 6. What has actually been run vs. what is still a plan

| Item | Status |
|---|---|
| MIG validation + controlled experiment + pipeline A/B | ✅ Run, concluded |
| SE gate full A/B (155 × 3 backends) | ✅ Run, concluded |
| Stage 0 governance: shadow + SAGE + calibrated runs | ✅ Run (2026-07-13) |
| §8.4 geometry↔ΔH replay (go/no-go) | ✅ Run (2026-07-17) — geometry-first confirmed |
| Plan 1 (offline cascade components, Stage 1 NLI shadow, Stage 2 core) | ✅ Built, offline tests green |
| Plan 3 calibration instruments (w1 grid, k sensitivity, read margins, S2 margin) | ✅ Run (2026-07-18, offline + live harvest) |
| **Replicated 5× survival-conditional A/B (baseline vs governed)** | ✅ **Run (2026-07-18/19)** — null on official score; see `EVAL_RESULTS.md` |
| Read-time gate arm, placement/eviction arm, destructive-guard arm (one-at-a-time ablations) | ✅ **Run (2026-07-19)** — 8 arms × 3 reps; only `geo_off` significant (KV −10.6 pp, Holm p=0.0013); `p_only` marginal; read arms null |
| Paraphrase-tolerant NLI judge (secondary metric) + G11 entropy retro-check | ✅ Run (2026-07-19) — governed +9/775 Vector under paraphrase tolerance; entropy signal informative but inert by design |
