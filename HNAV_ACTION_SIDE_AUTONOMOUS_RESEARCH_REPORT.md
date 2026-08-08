# H-Nav Action-Side Autonomous Research Report

**Status:** LIVING DOCUMENT — sections marked `[PENDING <campaign>]` await
live results; every other number is final and traceable to a committed
artifact. Last updated 2026-08-08 (C1 READ campaign in flight).
**Branch:** `claude/nihai-plan-v2-cascade-thresholds-k34u67`
**Governing directive:** `latest/ORIGINAL_DIRECTIVE.md` (recovered verbatim)
**Machine-readable summary:** `berkeley-function-call-leaderboard/gov_logs/hnav_autonomous/final_summary.json` [PENDING final]

Paths below are relative to `berkeley-function-call-leaderboard/` unless
rooted.

---

## 1. Executive summary

- **Stage 1 verdict (write-side coupling): falsified.** The geometry defect
  is real (220 near-duplicate updates, whole-blob similarity 0.944 vs 0.196
  on the marginal diff) but lands on decisions that cannot change any
  answer: 0/220 were `must_write`; answer-critical writes are 159/4,575
  (3.5%) and predominantly fresh ADDs. H1 FAIL, H2 PASS (ΔAUC +0.076
  [+0.015, +0.138]), H3 FAIL. Verified regeneration:
  `gov_logs/hnav_stage2/stage1_verification.json`.
- **Stage 3 verdict (action-side `H_act` gating): NO_GO at its own
  pre-registered gate.** T1 (`must_suppress`) has 9 positives in 1,253
  (0.7% base rate, min bar 25). Entropy adds nothing over vote controls:
  dAUC(M8−M4) = −0.013, CI [−0.037, +0.001], Holm p = 1.0. Arms A3–A6 were
  correctly never built (§6.7). Artifact: `gov_logs/hnav_shadow/stage3_gate.json`.
- **Alternatives worked under the directive's ten-slot cap:** #7 parser
  robustness (rejected, 0/3,276 misparses), #1 factorized entropy
  (rejected, zero new compute), #9 backend-specific policies (rejected),
  #10 write-rescue (live campaign, **promising-but-inconclusive /
  underpowered** — see §8), #5R write scaffold (**superseded before
  launch** by the tier-conditional correction — see §9.5), plus the five
  remaining slots re-specified as the RAG program (§9.6–§9.10).
- **The pivotal finding of the whole program (2026-08-08):** stratifying
  the loss by *which memory tier carries the gold fact* shows the benchmark
  loss is a **capture→read pipeline failure**, not a governance or
  uncertainty failure. Gold in core memory (auto-dumped into the system
  prompt) converts at 0.556/0.701 (kv/vector); gold **only in archival
  memory converts at 0.017/0.000** because the agent issues an archival
  read on **1 of 155** question entries. Archival memory is a write-only
  black hole at question time. Artifact:
  `gov_logs/hnav_autonomous/tier_conditional_pooled.json`.
  Consequences: (a) the pre-registered alt5R scaffold prediction (+0.171
  vector) was mis-priced by a tier-blind conversion factor and corrects to
  ≈ 0 standalone — amended before launch, ~15 GPU-h saved; (b) capture and
  read must be tested **as an interaction**, which is what the successor
  RAG program does.
- **Best result so far:** [PENDING C1/C2] — offline falsifiers GO on all
  six new-method fronts; the read-conversion factor c (the one quantity no
  offline instrument can produce) is being measured live by C1.
- **Operational cost of the best H-Nav-proper method (alt10 A2):** 234
  extra model calls and ~558k extra completion tokens per net recovered
  answer at 2.48× wall clock — statistically arguable, operationally
  unattractive (§13).
- **Final recommendation:** [PENDING] — current draft: option 3/5 hybrid —
  *use simple deterministic read/write scaffolds rather than entropy; the
  action-uncertainty program is a defensible negative* (see §15).

## 2. Starting repository state

| item | value |
|---|---|
| branch | `claude/nihai-plan-v2-cascade-thresholds-k34u67` |
| session start commit | `2a3e0cc` (2026-08-08 session; program baseline `b0b7d12` Stage-1 preservation) |
| model | `Qwen/Qwen3-4B-Instruct-2507` via vLLM 0.9.1, SSH tunnel `localhost:8000/v1` |
| backends | BFCL v4 Memory `memory_kv`, `memory_vector` (rec_sum ungoverned) |
| environment | `gov_logs/hnav_stage2/repo_state.md`, `pip_freeze.txt` |
| benchmark data | `bfcl_eval/data/BFCL_v4_memory.json` sha256:16 `67867501782b23ca`; gold `a89d40924c82fac4` (integrity gate, §14) |

## 3. Verified Stage 1 findings

Regenerated exactly (`stage1_verification.json`, commit `b0b7d12`):
H1 FAIL (widest deployable cell 2.6% coverage vs 5% bar; harmful-rate
Wilson UB 0.093 vs 0.05 ceiling). H2 PASS (ΔAUC +0.076 [+0.015, +0.138],
shuffle −0.006, survives `has_old` removal). H3 FAIL (false-override 0.66
vs 0.20). 220 near-duplicates at ≥0.90 whole-blob similarity, median 0.944
→ 0.196 diff-only. Zero `must_write` among them. 159/4,575 answer-critical.
~85% of decisions `inert_superseded` (chain-overwrite structure).

## 4. Action-side architecture

`QwenHactHandler` (`-FC-HACT`): dual-request `_query_prompting` (primary at
harness temperature + N−1 exploration samples at 0.7), canonicalization at
4 resolutions (tool/op/op+target/full; commits `dfd6728`, `6b45c6b`),
token-logprob features on the primary (`lp_*`, tool-span variants),
`schema: hact1` records joined to governance decisions on
`(test_id, step_idx)`. Seeds: sha256-derived per (test_id, call_idx, salt),
replicate-shifted by the runner (fix `945837e`, adversarial review while
blind). Selection policies implemented: `shadow`, `random_select` (A1),
`majority` (A2).

## 5. Pre-registered evaluation protocol

- Shadow first; intervention only behind the §6.7 gate
  (`gov_logs/hnav_shadow/PREREGISTRATION.md`, frozen before the campaign).
- Paired inference: exact McNemar on survival-conditional matched
  questions + cluster bootstrap over (replicate, scenario) + Holm per
  declared family (`analyze_gov_replicates.py`).
- Threshold discipline: dev reps 01–02, val rep03, freeze-once
  (`adfb676`); no unconstrained sweeps.
- RAG program additions (2026-08-08, all frozen before their campaigns):
  `PREREGISTRATION_C1.md`, `PREREGISTRATION_C2.md`, tier-conditional
  conversion factors with ranges over the read-conversion factor c,
  student-INCLUDED headline denominator (155/backend) with
  student-excluded sensitivity, strict answer-field-only regrade beside
  every lenient metric, step-budget invalidation rule
  (force_quit > max(3× baseline, 2/arm-rep); baseline = 0/1,152).

## 6. Baseline reproduction

`gov_logs/hnav_autonomous/CORRECTED_BASELINE.md` (§3.3 report):

| | kv | vector |
|---|---|---|
| all 5 scenarios (465 q pooled, 3 reps) | 0.1118 | **0.1226** |
| student excluded (315 q) | 0.1556 | 0.1714 |
| official leaderboard | 0.1613 | 0.1226 |

**Vector reproduces official exactly; kv's −4.9pp gap is entirely the dead
`student` chain** — a *behavioural* zero-write failure (0 writes in all 30
cells across two campaigns; data files clean of encoding damage; the model
answers 811-char reflective monologues with monologues and no tool calls).
No harness bug: original and corrected baselines coincide; the correction
is to the interpretation and to the reproduction guide's vector
attribution, which is retracted. Strict-regrade reference: kv 0.0925,
vector 0.0989 (17–19% of lenient passes are context-field leaks).

## 7. Shadow-mode findings

`STAGE3_REPORT.md` / `stage3_gate.json`: T1 base rate 0.0072 (9/1,253);
every entropy nest at or below vote-only controls (M5 −0.009, M6 −0.002,
M8 −0.013 vs M4); Holm p = 1.0 throughout; replicate-holdout sign
inconsistent (+0.034/−0.010/null). Incidental positive carried forward:
vote features predict *no-call* (T2) at AUC 0.998 — the seed of alt10.
Negative controls behaved (shuffles collapse; NC13 oracle-path refusal).

## 8. Intervention results (stage4 campaign = A0/A1/A2 controls + alt10 live test)

`gov_logs/hnav_stage4/{INTEGRITY,VERDICT}.md`, n_pairs = 290/backend:

| contrast | backend | ΔAcc | 95% CI | McNemar p | Holm p |
|---|---|---:|---|---:|---:|
| a2 vs a1 (primary) | kv | +0.0000 | [−0.046, +0.048] | 1.000 | 1.000 |
| a2 vs a1 (primary) | vector | +0.0138 | [−0.042, +0.064] | 0.689 | 1.000 |
| a2 vs baseline | kv | +0.0241 | [−0.048, +0.092] | 0.410 | 1.000 |
| a2 vs baseline | vector | **+0.0345** | [−0.011, +0.079] | 0.184 | 0.736 |
| a1 vs baseline | vector | +0.0207 | [−0.055, +0.091] | 0.504 | 1.000 |

Verdict **promising-but-inconclusive / underpowered** (frozen): the live
vector point estimate reproduces the falsifier's +0.035 exactly with 3/3
replicate direction, but the CI includes zero at n = 290, the primary
equal-budget contrast is ~null (most of the gain is generic extra
sampling), and the pre-registered shuffled-assignment negative control did
**not** collapse (~70% of the offline "rescue" metric survives —
non-content-specific carriage of short golds). Strata moved in the
predicted direction (vector `not_carried` 347→333, `carried_retrievable`
112→124; kv rescues land `carried_NOT_retrievable` 19→43 — the
BM25-key defect). Cost: §13.

## 9. Autonomous alternatives (one subsection each)

### 9.1 alt7 — parser/serialization robustness — REJECTED
0/3,276 gated primary steps misparsed; all 1,910 orphans are deliberate
free-text no-calls. No headroom; killed with zero new code.

### 9.2 alt1 — factorized operation/target/content uncertainty — REJECTED
Already falsified by the Stage 3 ladder (M5/M6/M7/M8 ≈ 0 or negative over
vote controls; +0.0000 on T2 at every resolution). Zero new compute.

### 9.3 alt9 — backend-specific uncertainty policies — REJECTED
Nothing to specialize: T1 has 2–7 positives per backend; T2 at ceiling
globally. The real backend heterogeneity (vector rescues > kv) was folded
into the alt10 analysis.

### 9.4 alt10 — selective write-rescue — PROMISING-BUT-INCONCLUSIVE
Direction reversal of pre-specified #10 (delayed commitment → commit
more), recorded as a refinement at GO time. Full results in §8. Lesson:
the T2=0.998 signal is real but the rescued content is largely
non-specific; and a +0.035 effect cannot clear n=290 — power the campaign
or pre-commit the label, which we did.

### 9.5 alt5R — deterministic write scaffold — SUPERSEDED BEFORE LAUNCH
The falsifier's conversion factor (`P_HAT = 0.409`) was **tier-blind**;
the scaffold writes to archival, whose true conversion is 0.000 (vector) /
0.017 (kv). Corrected standalone prediction ≈ 0 on both backends
(`PREREGISTRATION.md` AMENDMENT 1, dated before any launch; ~15 GPU-h
saved). The capture *policy* survives as the `pack`/`per_turn` write side
of the RAG program, where it is tested as a capture×read interaction.
Lesson — the program's most transferable methodological result: **every
Δanswerability→ΔAcc conversion must be conditioned on the tier the write
lands in, and reported as a range over the read-conversion factor.**

### 9.6 alt4R (M1) — retrieve-then-generate read scaffold — [PENDING C1]
Slot #4 re-specified. Inject one archival read on no-call question turns
(`QwenRagHandler`, `-FC-RAG`; verbatim user turn as probe; controls:
identical-call irrelevant probe, instruction-only nudge). Falsifier
already satisfied: archival-only stratum n=74 converts at 0.000, carrier
recall@5 = 22/22. C1 measures the read-conversion factor c that prices
every capture method. Launched 2026-08-08, 4 arms × 5 reps,
server_created 1786201925.

### 9.7 alt3R (M2) — entry-budget-aware packed capture — [PENDING C2]
Slot #3 re-specified (placement/packing). The archival budget binds on
ENTRY COUNT (50), not entry length: greedy 2000-char packing lifts vector
carried-rate 0.748→0.884 and r@5 0.703→0.819 vs one-turn-per-entry, while
conventional sentence chunking collapses to 0.342
(`gov_logs/hnav_rag/falsifiers.json`). Fully causal buffer (chain tail
lost). Arms include `pack_shuffled` (volume/multiset control),
`per_turn+read` (retires alt5R by contrast), `pack_write` alone (tests
the ≈0 prediction). Gated on C1.

### 9.8 alt2R (M9) — agent-side rerank of widened retrieval — [PENDING C3]
Slot #2 re-specified. k=20 retrieve → BM25 rerank → top-5 into context:
offline recovers 97.1% carried-conditional recall vs r@20 99.3% at k=5
context cost. Ceiling stated in advance: ≤ +2.3pp. Controls:
random-rerank, k5+pad dilution.

### 9.9 alt6R (M6) — verbatim vs abstractive write — [PENDING C5, designed negative]
Slot #6 re-specified. Gold answers are median-7-char entities under an
exact substring grader; deterministic 4× compression already destroys
carried-rate (0.884→0.632). Prediction: MemGPT-style summarization LOSES
to packed verbatim at matched stored chars — locating compression at read
time, not write time.

### 9.10 alt8R (M7) — retrieval-aware capacity management — [PENDING C5]
Slot #8 re-specified. Post-packing, the 50-entry cap binds on student (56
needed): redundancy eviction beats FIFO by +6pp carried offline. Ceiling
~+2pp overall, stated in advance. Control: random-evict at matched count.

### Stage-5 extension (beyond the ten-slot ledger; `gov_logs/hnav_rag/ledger.jsonl`)
- **M3 read budget** (top_k ∈ {5,20,50}): recall half GO offline (+5.8pp);
  accuracy half unfalsifiable offline by construction — flagged likely
  null; the recall-vs-precision curve is the artifact. [PENDING C3]
- **M4 probe formulation** (verbatim vs model-authored vs core-augmented):
  flagged likely null; "query rewriting does not beat the verbatim
  question on a small index" is the citable negative. [PENDING C3]
- **M5 quote-grounded answers** (`{'answer','evidence','context'}`):
  attacks the 30% failure on core-carried gold; guarded by the strict
  regrade rule (lenient-only gain = grader artifact) and an
  `evidence_irrelevant` control. [PENDING C4]
- **M8 kv key/read asymmetry** (24-token descriptive keys + two-stage
  key_search→retrieve): kv-primary; +5.8pp r@5 offline at fixed index;
  explains kv archival's 0.017 conversion. [PENDING C6]
- **M10 evidence nudge** (length-matched nudge vs null appended to the
  injected read's result payload): 12.3% of vector questions abstain while
  gold is carried; sub-additive with M5 pre-declared. [PENDING C4]

## 10. Failure taxonomy
[PENDING C1–C6 flip tables; stage4 flip tables in
`gov_logs/hnav_stage4/flip_tables.json`: a2-vs-baseline +63/−41 spread
across all four live scenarios, finance net-negative 18/25.]

## 11. Ablation analysis
Extra budget vs self-consistency separated by stage4 (a1 captures budget:
+0.021–0.024; a2−a1 isolates majority voting: ≈0–0.014). Vote margin /
scalar / factorized entropy separated by the Stage 3 M0–M10 ladder (all
≈ 0 over vote controls). Capture / read / packing / rerank / instruction
each get a dedicated control arm in C1–C6. [PENDING live numbers]

## 12. Statistical interpretation (directive §8 vocabulary)
- **Rejected:** write-side coupling (Stage 1), `H_act` gating (Stage 3),
  alt1, alt7, alt9.
- **Promising but inconclusive / underpowered:** alt10 A2 vs baseline
  (vector +0.0345, CI incl. 0, 3/3 direction).
- **Not testable as pre-registered:** alt5R standalone (instrument
  mis-priced; amended, superseded).
- **Validated:** [PENDING C1–C6].

## 13. Operational analysis
alt10 A2: 3,981 extra exploration requests, 49.4M prompt / 9.5M completion
tokens (3 replicates), latency mean 3.5 s (p50 2.2, p95 10.4) per gated
step, wall 2.48× baseline → **234 extra calls / ~558k extra completion
tokens per net recovered answer** (`cost_accounting.json`). The RAG
scaffolds by contrast add ≤1 deterministic call per question turn and no
sampling. [PENDING C1 cost table]

## 14. Research integrity audit
- `integrity_gate.py` (launcher pre-flight + report): `bfcl_eval/data/**`
  and `bfcl_eval/eval_checker/**` byte-unchanged since `b0b7d12`; grader
  and gold hashes recorded in `gov_logs/hnav_autonomous/integrity_gate.json`.
- `test_leakage_audit.py`: AST audit over all 22 runtime modules (names,
  imports, **and data-path string literals**), 44 checks green; wildcard
  covers future `qwen_rag*` modules.
- Env isolation: runner scrubs `GOV_/HACT_/SCAF_/RAG_` (unit-tested);
  per-(replicate, arm) log dirs; arms declare env explicitly.
- Denominators: student-included 155/backend headline everywhere;
  student-excluded sensitivity; analyzer refuses unscored cells.
- Ledger: append-only, monotonic status validator; the alt10 entry is the
  first to carry the full §20 field set (`primary_test`, `effect_size`,
  CI, p, adjusted p, `cost_summary`).

## 15. Final recommendation
[PENDING C1–C6.] Draft under either C-outcome: **do not deploy an
entropy/uncertainty policy on this benchmark** (rejected at its own gate);
the deployable lever is the deterministic capture→read scaffolding, whose
live value C1/C2 are measuring. If the scaffolds also fail live, Outcome F
stands with the loss decomposition explaining *why* every policy family
had to fail: the classes policed are 0.7–3.5% of decisions, while 77–81%
of questions never get their gold into the store and what does reach
archival is never read back.

## 16. Reproduction guide
Runners: `tmp/run_hnav_shadow.ps1`, `tmp/run_hnav_stage4.ps1`,
`tmp/run_hnav_scaffold.ps1` (never launched, superseded),
`tmp/run_hnav_rag_c1.ps1`. Analysis: `analyze_rag_campaign.py --manifest
<campaign>/manifest.jsonl --out-dir <govlog>/analysis`; alt10 commands in
`VERDICT.md`; falsifiers `rag_falsifiers.py --out
gov_logs/hnav_rag/falsifiers.json`; every command of the earlier stages in
`latest/CONTINUATION_PLAN.md` §3–§5. Seeds and env per arms JSONs.

## 17. Artifact inventory
Handlers: `qwen_gov.py`, `qwen_hact.py`, `qwen_scaffold.py`, `qwen_rag.py`
(+ registry entries `-FC-GOV/-HACT/-SCAF/-RAG`). Scripts (new this phase):
`step_budget_guard.py`, `strict_regrade.py`, `integrity_gate.py`,
`test_leakage_audit.py`, `rag_falsifiers.py`, `analyze_rag_campaign.py`,
`test_qwen_rag.py` (48 checks). Artifacts: `gov_logs/hnav_stage4/*`
(integrity, verdict, analyses, answerability, NC, cost, flips, wrra),
`gov_logs/hnav_autonomous/*` (tier factors, corrected baseline, strict
regrade, step budget, integrity gate, ledger),
`gov_logs/hnav_rag/*` (preregs C1–C2, falsifiers, ledger). Campaign trees:
`result_hnav_{shadow,stage4,rag_c1}/`, `score_*`. Commits: `4351785..2a3e0cc`
(session 1, 11), `69564b7..f60a56f` (this session; pushed to fork).

## 18. Open questions
1. The read-conversion factor c for retrieved-into-context archival text —
   measured by C1; every capture method's value is linear in it.
2. Does packed capture pollute? (C2 negative flips; capacity exhaustion
   instrumented.)
3. Why does the model refuse to write on `student` narratives — prompt
   sensitivity or instruction-following failure? (Out of scope; the
   capture scaffold sidesteps rather than explains it.)
4. Whether logprob and resampling uncertainty measure the same phenomenon
   (§27.3) — small offline job on shadow logs, still owed.
5. Generalization beyond a 4B model and beyond BFCL's substring grader —
   the strict-regrade gap suggests grader-robustness work.

---

## §26 comparison table

| Approach | Main mechanism | Dev result | Held-out result | Δ acc (vector) | 95% CI | Adj. sig. | +flips | −flips | Added calls | Added tokens | Decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (A0) | — | 0.1226 | 0.1226 (reproduces official) | — | — | — | — | — | 0 | 0 | reference |
| A1 equal-compute random | random select of 8 samples | — | +0.0207 | +0.0207 | [−0.055, +0.091] | Holm 1.0 | 25 | 31* | +1/gated step | ~9.5M | control |
| A2 majority vote (alt10) | modal-write rescue on no-call | falsifier +0.035 | **+0.0345** | +0.0345 | [−0.011, +0.079] | Holm 0.736 | 28 | 18 | +1/gated step | ~9.5M | promising-inconclusive |
| vote-margin gate (A3) | — | — | **not run: Stage 3 gate NO_GO per §6.7** | — | — | — | — | — | — | — | not run |
| scalar `H_act` (A4) | — | — | not run: same reason | — | — | — | — | — | — | — | not run |
| factorized `H_act` (A5/alt1) | — | dAUC ≤ 0 all resolutions | — | — | — | Holm 1.0 | — | — | — | — | rejected |
| full action-side H-Nav (A6) | — | — | not run: same reason | — | — | — | — | — | — | — | not run |
| alt7 parser | — | 0/3,276 misparses | — | — | — | — | — | — | — | — | rejected |
| alt9 backend-specific | — | 2–7 positives/backend | — | — | — | — | — | — | — | — | rejected |
| alt5R write scaffold | verbatim auto-archive | +0.171 (tier-blind) → **≈0 corrected** | superseded | — | — | — | — | — | — | — | superseded pre-launch |
| M1 read scaffold | inject archival read | falsifier GO | [PENDING C1] | | | | | | +1 call/question | | |
| M2 packed capture | 2000-char greedy packing | carried 0.884 vs 0.748 | [PENDING C2] | | | | | | ~0 | | |
| M9 rerank | k20→BM25→top5 | 0.971 recall | [PENDING C3] | | | | | | 0 | | |
| M3 read budget | top_k curve | +5.8pp recall | [PENDING C3] | | | | | | 0 | | |
| M4 probe formulation | query construction | — | [PENDING C3] | | | | | | 0–1 | | |
| M5 quote-grounded | evidence field | gap 0.024 baseline | [PENDING C4] | | | | | | 0 | | |
| M10 evidence nudge | result-payload nudge | 12.3% abstain-carried | [PENDING C4] | | | | | | 0 | | |
| M6 abstractive write | summarize-then-archive | 0.632 vs 0.884 | [PENDING C5, designed negative] | | | | | | +1/prereq turn | | |
| M7 capacity mgmt | redundancy eviction | +6pp vs FIFO | [PENDING C5] | | | | | | 0 | | |
| M8 kv asymmetry | 24-tok keys + 2-stage read | +5.8pp r@5 | [PENDING C6] | | | | | | +2/question | | |

\* vector cells; a1's ±flips vs baseline from `analysis_secondary_vs_baseline.json`.

## §27 — the twelve questions

1. **Does action uncertainty predict incorrect memory-operation choices?**
   No. T1 AUC 0.586 at M0 and ≤ 0.50 for every entropy nest; 9 positives
   in 1,253.
2. **Does Shannon entropy add value beyond vote margin/disagreement?** No.
   dAUC(M8−M4) = −0.013 CI [−0.037, +0.001], Holm p = 1.0; +0.0000 on T2.
3. **Are token-logprob and resampling uncertainty the same phenomenon?**
   [PENDING — small offline job on `logprob_features` × `votes`.]
4. **Is operation uncertainty more important than target uncertainty?**
   Neither matters: M5 (op) −0.009, M6 (target) −0.002 vs M4.
5. **Does uncertainty-guided intervention improve final BFCL accuracy?**
   The only uncertainty-derived intervention that reached a live campaign
   (alt10) is promising-but-inconclusive, and its gain over the
   equal-budget control is ~null. [C1–C6 measure *non*-uncertainty
   scaffolds.]
6. **Is any gain statistically reliable?** Not yet: every CI to date
   includes zero. [PENDING C1–C6.]
7. **Does it generalize across KV and vector?** alt10: direction yes,
   mechanism differs (kv rescues land unretrievable — the key-index
   defect). [PENDING]
8. **Across scenarios?** alt10 flips spread over all four live scenarios;
   `student` is structurally dead at baseline (behavioural zero-write) and
   is the capture program's largest target. [PENDING]
9. **Is the gain worth its inference cost?** alt10: no — 234 calls/answer.
   The deterministic scaffolds cost ≤1 call/turn. [PENDING]
10. **Which mechanism actually produced the improvement?** For alt10-class
    gains: mostly generic extra sampling, not majority-vote specificity
    (a1 ≈ a2 on kv), and partly non-content-specific carriage (NC
    non-collapse). [PENDING for the RAG program.]
11. **Which plausible mechanisms were falsified?** Write-side diff
    coupling (Stage 1); action-side `H_act` gating (Stage 3); factorized
    uncertainty (alt1); backend-specific uncertainty policies (alt9);
    parser robustness (alt7); tier-blind answerability→accuracy conversion
    (the alt5R correction — falsified as an *instrument*).
12. **Strongest defensible thesis claim (draft):** On BFCL v4 Memory the
    binding constraint is the **capture→read pipeline**, not write
    governance and not action-choice uncertainty: 77–81% of questions
    never carry their gold fact; ~88% of those facts were stated verbatim
    in-conversation; the classes gating policies police are 0.7–3.5% of
    decisions; and of the facts that DO reach archival memory, the agent
    reads back approximately none (1/155 entries, conversion 0.000).
    Capture and read are complements — either alone is worth ≈ 0. [Live
    confirmation PENDING C1/C2; the shuffled-capture and irrelevant-probe
    controls decide which mechanism the claim may cite.]
