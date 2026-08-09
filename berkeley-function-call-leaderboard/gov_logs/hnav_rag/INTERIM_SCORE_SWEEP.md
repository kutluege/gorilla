# Interim score sweep — all memory score trees (2026-08-10)

## 0. Status of this document — read first

This is an **INTERIM, NON-CONFIRMATORY** look at a running campaign,
produced at the operator's request on 2026-08-10. C1 was pre-registered
blind-until-complete (`gov_logs/hnav_rag/PREREGISTRATION_C1.md`,
"No arm contrast is computed until all 5 replicates are complete";
`PREREGISTRATION_C2.md`, "BEFORE C1 unblinds"); this document records a
**deliberate, logged unblinding of the completed replicate(s) only**
(rep01). The confirmatory verdict remains the frozen 5-replicate paired
analysis (`bfcl_eval/scripts/analyze_rag_campaign.py` /
`analyze_gov_replicates.py` per the pre-registration) and **nothing here
overrides it**.

Single-replicate deltas on 155 questions carry historical noise of roughly
±5pp. Documented context: identical co-run baseline arms in the stage4
campaign scored 13, 27, and 12 correct on kv across three replicates — a
15-question (~9.7pp) swing between arms that differ only in seed/replicate
(`score_hnav_stage4/rep0*/baseline/...`, cross-referenced against
`gov_logs/hnav_stage4/VERDICT.md`, which froze its verdict at
"promising but inconclusive — underpowered by construction" for a +3.45pp
point estimate precisely because n=290 matched pairs cannot resolve effects
of this size). Treat every single-rep delta below accordingly.

Every number in this file was read from a score-file header
(`accuracy`, `correct_count`, `total_count` — first line of each
`BFCL_v4_memory_<backend>_score.json`); the full file list is in §5.
No result/log files were opened; no inferential statistics were run.

---

## 1. C1 rep01 (the headline) — `score_hnav_rag_c1/`

Only rep01 exists in the score tree as of this sweep (4 arms × 2 backends,
8 cells; later replicates not yet scored). All cells n=155. Deltas are vs
the **co-run rep01 baseline**, in percentage points (1 question = 0.645pp).

| arm | kv acc (correct/155) | Δkv (pp) | vector acc (correct/155) | Δvector (pp) |
|---|---|---:|---|---:|
| baseline (co-run) | 0.1419 (22) | — | 0.1419 (22) | — |
| read_verbatim (M1) | 0.1548 (24) | **+1.29** | **0.2258 (35)** | **+8.39** |
| read_irrelevant (control) | 0.0710 (11) | **−7.10** | 0.2065 (32) | **+6.45** |
| read_instruction (control) | 0.1806 (28) | **+3.87** | 0.1613 (25) | **+1.94** |

Arms beating the rep01 co-run baseline: on **vector**, all three
intervention arms (read_verbatim +8.39pp / +13 questions;
read_irrelevant +6.45pp / +10; read_instruction +1.94pp / +3). On **kv**,
read_instruction (+3.87pp / +6) and read_verbatim (+1.29pp / +2);
read_irrelevant **hurts** kv (−7.10pp / −11 questions).

### 1.1 read_verbatim vector vs the M1 prediction band

The user observed a ~+3–4pp gain for read_verbatim vector. The actual
rep01 number is **larger: +8.39pp** (35/155 vs 22/155). The pre-registered
M1 prediction band is **+2.4pp (read-conversion floor c=0.30) to +5.6pp
(c at the core conditional ≈0.70)** on vector
(`gov_logs/hnav_rag/PREREGISTRATION_C1.md` H-C1a:
"archival-only stratum (74/930 vector) converts at c ≥ 0.30 →
ΔAcc_vector ≥ +0.024"; band upper edge per
`HNAV_ACTION_SIDE_AUTONOMOUS_RESEARCH_REPORT.md` §1: "ΔAcc_vector +2.4pp
… with upside to +5.6pp (c at the core conditional ≈0.70)").

So: the rep01 point estimate is in the **predicted direction and at or
above the band's upper edge** (+8.39 vs +2.4…+5.6). It is consistent with
the pre-registered prediction in sign and exceeds it in magnitude —
**n=1 replicate, awaiting 4 more**; at ±5pp single-rep noise, a true
effect anywhere in the band (or partly outside it) is compatible with this
observation. No claim beyond "interim signal" is made.

### 1.2 read_verbatim vs read_irrelevant (the mechanism control)

The irrelevant-probe arm gains almost as much on vector (+6.45pp vs
+8.39pp; separation only +1.94pp / 3 questions). **This is EXPECTED in
C1** and was pre-registered. Verbatim from
`gov_logs/hnav_rag/PREREGISTRATION_C1.md`:

> **Recorded limitation:** at baseline archival sizes (≈2–19 entries,
> often ≤ 5) the retrieved set for any probe is most or all of the store,
> so C1's verbatim-vs-irrelevant contrast mostly isolates the
> *step/volume* effect, not probe *relevance*; relevance becomes
> discriminative in C2+ where capture fills the index.

The H-C1b ordering (`read_verbatim ≥ read_irrelevant > baseline`) holds on
vector in rep01 (35 ≥ 32 > 22). On kv it does not (irrelevant collapses to
11); the instruction-only nudge is the best kv arm (28). Per-backend
divergence is noted, not interpreted — that is the frozen analyzer's job.

---

## 2. Stage4 campaign — `score_hnav_stage4/` (already analyzed and final)

Final verdict already frozen in `gov_logs/hnav_stage4/VERDICT.md`:
**PROMISING BUT INCONCLUSIVE / underpowered** (a2 vs baseline vector
+0.0345 matched-pairs, CI [−0.011, +0.079], Holm p 0.736; primary a2-vs-a1
contrast ≈ null). Raw header accuracies below are for completeness only
(each cell n=155; pooled n=465).

| rep | arm | kv acc (correct) | vector acc (correct) |
|---|---|---|---|
| rep01 | baseline | 0.0839 (13) | 0.1161 (18) |
| rep01 | a1_random | 0.1419 (22) | 0.1484 (23) |
| rep01 | a2_majority | 0.1290 (20) | 0.1484 (23) |
| rep02 | baseline | 0.1742 (27) | 0.1097 (17) |
| rep02 | a1_random | 0.1290 (20) | 0.1742 (27) |
| rep02 | a2_majority | 0.1355 (21) | 0.1613 (25) |
| rep03 | baseline | 0.0774 (12) | 0.1419 (22) |
| rep03 | a1_random | 0.1548 (24) | 0.0839 (13) |
| rep03 | a2_majority | 0.1161 (18) | 0.1548 (24) |
| **pooled** | baseline | 0.1118 (52/465) | 0.1226 (57/465) |
| **pooled** | a1_random | 0.1419 (66/465) | 0.1355 (63/465) |
| **pooled** | a2_majority | 0.1269 (59/465) | 0.1548 (72/465) |

Cross-checks against `gov_logs/hnav_stage4/VERDICT.md`: the per-replicate
a2-vs-baseline vector directions quoted there (0.148>0.116, 0.161>0.110,
0.155>0.142) match these headers exactly, and the pooled baselines
(kv 0.1118, vector 0.1226) match the corrected-baseline table reproduced
in `HNAV_ACTION_SIDE_AUTONOMOUS_RESEARCH_REPORT.md` §6. Raw pooled
a2−baseline vector is +3.23pp here vs the VERDICT's +3.45pp because the
frozen analyzer uses survival-conditional matched pairs (n=290), not raw
pooling — the VERDICT number is the reportable one. **Already reported;
nothing new here.**

## 3. Shadow campaign — `score_hnav_shadow/` (already analyzed and final)

Stage 3 verdict already frozen: **NO_GO at the pre-registered gate**
(`gov_logs/hnav_shadow/STAGE3_REPORT.md`, `stage3_gate.json`; entropy adds
nothing over vote controls). The shadow arm records features without
intervening; accuracies below are for completeness (each cell n=155).

| rep | arm | kv acc (correct) | vector acc (correct) |
|---|---|---|---|
| rep01 | baseline | 0.1161 (18) | 0.1290 (20) |
| rep01 | hact_shadow | 0.0903 (14) | 0.1290 (20) |
| rep02 | baseline | 0.1226 (19) | 0.1613 (25) |
| rep02 | hact_shadow | 0.0968 (15) | 0.1290 (20) |
| rep03 | baseline | 0.1419 (22) | 0.1355 (21) |
| rep03 | hact_shadow | 0.0968 (15) | 0.1355 (21) |
| **pooled** | baseline | 0.1269 (59/465) | 0.1419 (66/465) |
| **pooled** | hact_shadow | 0.0946 (44/465) | 0.1312 (61/465) |

No shadow cell beats its co-run baseline. **Already reported; nothing new
here.**

## 4. Legacy one-offs — `score/` (non-comparable)

Single runs with **no co-run baseline** — different sessions and
configurations (Gov Stage 0 era). Listed for traceability only; **not
comparable** to any campaign cell and no deltas are computed.

| model | kv acc (correct/155) | vector acc (correct/155) |
|---|---|---|
| Qwen_Qwen3-4B-Instruct-2507-FC | 0.0710 (11) | 0.1161 (18) |
| Qwen_Qwen3-4B-Instruct-2507-FC-GOV | 0.1677 (26) | 0.1290 (20) |

---

## 5. Is there a +3–4 point method?

**VALIDATED: none.** No frozen analysis in this repository validates any
method at any effect size. The closest completed result, alt10 A2
(vector +3.45pp point estimate), is frozen as promising-but-inconclusive:
CI includes zero and the equal-compute control contrast is ≈ null
(`gov_logs/hnav_stage4/VERDICT.md`;
`HNAV_ACTION_SIDE_AUTONOMOUS_RESEARCH_REPORT.md` §1a: "NO. No method tried
so far has a validated +3–4pp improvement").

**INTERIM SIGNAL: C1 rep01 read_verbatim, vector +8.39pp over its co-run
baseline (35 vs 22 of 155)** — in the predicted direction and above the
pre-registered band (+2.4 to +5.6pp), with the H-C1b vector ordering
holding (verbatim 35 ≥ irrelevant 32 > baseline 22). Heavy caveats:
(a) n=1 replicate on 155 questions, against documented baseline-to-baseline
swings of up to 15 questions in this repo — a single-rep delta of this size
is within reach of noise plus a modest true effect; (b) the irrelevant-probe
control gains +6.45pp, so at C1's baseline store sizes most of the gain is
the pre-registered *step/volume* effect, not probe relevance — relevance
separation is designed to appear in C2+; (c) kv is mixed (verbatim +1.29pp,
irrelevant −7.10pp, instruction +3.87pp). This is a signal to keep the
campaign running, not a result. The frozen 5-replicate paired analysis
decides.

**PREDICTED (not demonstrated), per
`HNAV_ACTION_SIDE_AUTONOMOUS_RESEARCH_REPORT.md` §1/§1a:** M2 packed
capture × read (C2, queued, gated on C1 — the largest predicted effect in
the portfolio; offline answerability 0.187 → 0.82) and M5 quote-grounded
answers (C4, predicted ≈ +3pp on the strict regrade). Both are offline
falsifier predictions only; alt5R (+0.171 predicted → ≈0 after the
tier-conditional correction) is the standing warning about offline
predictions.

---

## 6. Traceability appendix — every score file read

Paths relative to `berkeley-function-call-leaderboard/`. Accuracy and
correct/total are the score-file header values, verbatim.

### score_hnav_rag_c1/ (8 files)

| file | accuracy | correct/total |
|---|---|---|
| `score_hnav_rag_c1/rep01/baseline/Qwen_Qwen3-4B-Instruct-2507-FC/agentic/memory/kv/BFCL_v4_memory_kv_score.json` | 0.14193548387096774 | 22/155 |
| `score_hnav_rag_c1/rep01/baseline/Qwen_Qwen3-4B-Instruct-2507-FC/agentic/memory/vector/BFCL_v4_memory_vector_score.json` | 0.14193548387096774 | 22/155 |
| `score_hnav_rag_c1/rep01/read_verbatim/Qwen_Qwen3-4B-Instruct-2507-FC-RAG/agentic/memory/kv/BFCL_v4_memory_kv_score.json` | 0.15483870967741936 | 24/155 |
| `score_hnav_rag_c1/rep01/read_verbatim/Qwen_Qwen3-4B-Instruct-2507-FC-RAG/agentic/memory/vector/BFCL_v4_memory_vector_score.json` | 0.22580645161290322 | 35/155 |
| `score_hnav_rag_c1/rep01/read_irrelevant/Qwen_Qwen3-4B-Instruct-2507-FC-RAG/agentic/memory/kv/BFCL_v4_memory_kv_score.json` | 0.07096774193548387 | 11/155 |
| `score_hnav_rag_c1/rep01/read_irrelevant/Qwen_Qwen3-4B-Instruct-2507-FC-RAG/agentic/memory/vector/BFCL_v4_memory_vector_score.json` | 0.2064516129032258 | 32/155 |
| `score_hnav_rag_c1/rep01/read_instruction/Qwen_Qwen3-4B-Instruct-2507-FC-RAG/agentic/memory/kv/BFCL_v4_memory_kv_score.json` | 0.18064516129032257 | 28/155 |
| `score_hnav_rag_c1/rep01/read_instruction/Qwen_Qwen3-4B-Instruct-2507-FC-RAG/agentic/memory/vector/BFCL_v4_memory_vector_score.json` | 0.16129032258064516 | 25/155 |

### score_hnav_stage4/ (18 files)

| file | accuracy | correct/total |
|---|---|---|
| `score_hnav_stage4/rep01/baseline/Qwen_Qwen3-4B-Instruct-2507-FC/agentic/memory/kv/BFCL_v4_memory_kv_score.json` | 0.08387096774193549 | 13/155 |
| `score_hnav_stage4/rep01/baseline/Qwen_Qwen3-4B-Instruct-2507-FC/agentic/memory/vector/BFCL_v4_memory_vector_score.json` | 0.11612903225806452 | 18/155 |
| `score_hnav_stage4/rep01/a1_random/Qwen_Qwen3-4B-Instruct-2507-FC-HACT/agentic/memory/kv/BFCL_v4_memory_kv_score.json` | 0.14193548387096774 | 22/155 |
| `score_hnav_stage4/rep01/a1_random/Qwen_Qwen3-4B-Instruct-2507-FC-HACT/agentic/memory/vector/BFCL_v4_memory_vector_score.json` | 0.14838709677419354 | 23/155 |
| `score_hnav_stage4/rep01/a2_majority/Qwen_Qwen3-4B-Instruct-2507-FC-HACT/agentic/memory/kv/BFCL_v4_memory_kv_score.json` | 0.12903225806451613 | 20/155 |
| `score_hnav_stage4/rep01/a2_majority/Qwen_Qwen3-4B-Instruct-2507-FC-HACT/agentic/memory/vector/BFCL_v4_memory_vector_score.json` | 0.14838709677419354 | 23/155 |
| `score_hnav_stage4/rep02/baseline/Qwen_Qwen3-4B-Instruct-2507-FC/agentic/memory/kv/BFCL_v4_memory_kv_score.json` | 0.17419354838709677 | 27/155 |
| `score_hnav_stage4/rep02/baseline/Qwen_Qwen3-4B-Instruct-2507-FC/agentic/memory/vector/BFCL_v4_memory_vector_score.json` | 0.10967741935483871 | 17/155 |
| `score_hnav_stage4/rep02/a1_random/Qwen_Qwen3-4B-Instruct-2507-FC-HACT/agentic/memory/kv/BFCL_v4_memory_kv_score.json` | 0.12903225806451613 | 20/155 |
| `score_hnav_stage4/rep02/a1_random/Qwen_Qwen3-4B-Instruct-2507-FC-HACT/agentic/memory/vector/BFCL_v4_memory_vector_score.json` | 0.17419354838709677 | 27/155 |
| `score_hnav_stage4/rep02/a2_majority/Qwen_Qwen3-4B-Instruct-2507-FC-HACT/agentic/memory/kv/BFCL_v4_memory_kv_score.json` | 0.13548387096774195 | 21/155 |
| `score_hnav_stage4/rep02/a2_majority/Qwen_Qwen3-4B-Instruct-2507-FC-HACT/agentic/memory/vector/BFCL_v4_memory_vector_score.json` | 0.16129032258064516 | 25/155 |
| `score_hnav_stage4/rep03/baseline/Qwen_Qwen3-4B-Instruct-2507-FC/agentic/memory/kv/BFCL_v4_memory_kv_score.json` | 0.07741935483870968 | 12/155 |
| `score_hnav_stage4/rep03/baseline/Qwen_Qwen3-4B-Instruct-2507-FC/agentic/memory/vector/BFCL_v4_memory_vector_score.json` | 0.14193548387096774 | 22/155 |
| `score_hnav_stage4/rep03/a1_random/Qwen_Qwen3-4B-Instruct-2507-FC-HACT/agentic/memory/kv/BFCL_v4_memory_kv_score.json` | 0.15483870967741936 | 24/155 |
| `score_hnav_stage4/rep03/a1_random/Qwen_Qwen3-4B-Instruct-2507-FC-HACT/agentic/memory/vector/BFCL_v4_memory_vector_score.json` | 0.08387096774193549 | 13/155 |
| `score_hnav_stage4/rep03/a2_majority/Qwen_Qwen3-4B-Instruct-2507-FC-HACT/agentic/memory/kv/BFCL_v4_memory_kv_score.json` | 0.11612903225806452 | 18/155 |
| `score_hnav_stage4/rep03/a2_majority/Qwen_Qwen3-4B-Instruct-2507-FC-HACT/agentic/memory/vector/BFCL_v4_memory_vector_score.json` | 0.15483870967741936 | 24/155 |

### score_hnav_shadow/ (12 files)

| file | accuracy | correct/total |
|---|---|---|
| `score_hnav_shadow/rep01/baseline/Qwen_Qwen3-4B-Instruct-2507-FC/agentic/memory/kv/BFCL_v4_memory_kv_score.json` | 0.11612903225806452 | 18/155 |
| `score_hnav_shadow/rep01/baseline/Qwen_Qwen3-4B-Instruct-2507-FC/agentic/memory/vector/BFCL_v4_memory_vector_score.json` | 0.12903225806451613 | 20/155 |
| `score_hnav_shadow/rep01/hact_shadow/Qwen_Qwen3-4B-Instruct-2507-FC-HACT/agentic/memory/kv/BFCL_v4_memory_kv_score.json` | 0.09032258064516129 | 14/155 |
| `score_hnav_shadow/rep01/hact_shadow/Qwen_Qwen3-4B-Instruct-2507-FC-HACT/agentic/memory/vector/BFCL_v4_memory_vector_score.json` | 0.12903225806451613 | 20/155 |
| `score_hnav_shadow/rep02/baseline/Qwen_Qwen3-4B-Instruct-2507-FC/agentic/memory/kv/BFCL_v4_memory_kv_score.json` | 0.12258064516129032 | 19/155 |
| `score_hnav_shadow/rep02/baseline/Qwen_Qwen3-4B-Instruct-2507-FC/agentic/memory/vector/BFCL_v4_memory_vector_score.json` | 0.16129032258064516 | 25/155 |
| `score_hnav_shadow/rep02/hact_shadow/Qwen_Qwen3-4B-Instruct-2507-FC-HACT/agentic/memory/kv/BFCL_v4_memory_kv_score.json` | 0.0967741935483871 | 15/155 |
| `score_hnav_shadow/rep02/hact_shadow/Qwen_Qwen3-4B-Instruct-2507-FC-HACT/agentic/memory/vector/BFCL_v4_memory_vector_score.json` | 0.12903225806451613 | 20/155 |
| `score_hnav_shadow/rep03/baseline/Qwen_Qwen3-4B-Instruct-2507-FC/agentic/memory/kv/BFCL_v4_memory_kv_score.json` | 0.14193548387096774 | 22/155 |
| `score_hnav_shadow/rep03/baseline/Qwen_Qwen3-4B-Instruct-2507-FC/agentic/memory/vector/BFCL_v4_memory_vector_score.json` | 0.13548387096774195 | 21/155 |
| `score_hnav_shadow/rep03/hact_shadow/Qwen_Qwen3-4B-Instruct-2507-FC-HACT/agentic/memory/kv/BFCL_v4_memory_kv_score.json` | 0.0967741935483871 | 15/155 |
| `score_hnav_shadow/rep03/hact_shadow/Qwen_Qwen3-4B-Instruct-2507-FC-HACT/agentic/memory/vector/BFCL_v4_memory_vector_score.json` | 0.13548387096774195 | 21/155 |

### score/ (4 files, legacy, non-comparable)

| file | accuracy | correct/total |
|---|---|---|
| `score/Qwen_Qwen3-4B-Instruct-2507-FC/agentic/memory/kv/BFCL_v4_memory_kv_score.json` | 0.07096774193548387 | 11/155 |
| `score/Qwen_Qwen3-4B-Instruct-2507-FC/agentic/memory/vector/BFCL_v4_memory_vector_score.json` | 0.11612903225806452 | 18/155 |
| `score/Qwen_Qwen3-4B-Instruct-2507-FC-GOV/agentic/memory/kv/BFCL_v4_memory_kv_score.json` | 0.16774193548387098 | 26/155 |
| `score/Qwen_Qwen3-4B-Instruct-2507-FC-GOV/agentic/memory/vector/BFCL_v4_memory_vector_score.json` | 0.12903225806451613 | 20/155 |

*Sweep performed 2026-08-10; 42 score files read; no other files were
modified. Reference documents consulted:
`gov_logs/hnav_rag/PREREGISTRATION_C1.md`,
`gov_logs/hnav_rag/PREREGISTRATION_C2.md`,
`gov_logs/hnav_stage4/VERDICT.md`, and
`HNAV_ACTION_SIDE_AUTONOMOUS_RESEARCH_REPORT.md` (repo root).*
