# H-Nav Stage 1 — implementation & results summary

**Date:** 2026-07-25 · **Branch:** `claude/nihai-plan-v2-cascade-thresholds-k34u67`
**Scope:** BFCL v4 Memory, KV + Vector, `Qwen/Qwen3-4B-Instruct-2507`
**Cost:** entirely offline on committed logs — no GPU, no vLLM tunnel, no new benchmark runs
**Verdict:** **PIVOT to action-side H_act.** Write-side coupling hypothesis falsified.

Generated artifact: `berkeley-function-call-leaderboard/gov_logs/HNAV_STAGE1.md`
(this file is the session record: what was built, what was measured, how to re-run).

---

## 1. Why this stage existed

Plan v2 ended with both gates failing — **G1** (entropy adds no out-of-sample predictive
value over geometry+margin) and **G2** (`gm_v1` not non-inferior to `geometry_only`).
The remaining live hypothesis was a *measurement* defect on the write side:

> When a small clause is appended to a long memory record, most of the text is unchanged,
> so whole-blob cosine stays high and GeometryGate can read a genuine update as a duplicate
> and suppress it.

The existing `harmful_write` label could not test this. A suppressed write never reaches
the final store, so no label computed *from* the final store can ask whether writing it
would have helped. Stage 1's job was to build a label that can, measure the true headroom,
and decide whether to keep investing in the write side.

**It was designed as a falsification exercise, not a search for a win.**

---

## 2. Two assets that made it cheap

1. **`bfcl_eval/data/possible_answer/BFCL_v4_memory.json`** — 155 rows of
   `{id, ground_truth[], source}`, 1:1 with the question set. Combined with the
   **production grader** `eval_checker/agentic_eval/agentic_checker.agentic_checker`
   (word-boundary regex over `standardize_string`), this gives a fully deterministic,
   offline "does this text carry a needed fact?" oracle — the real grader, not a
   re-implementation. Word boundaries matter: `"380 seconds"` does **not** carry gold `"38"`.

2. **The shadow harvest** (`GOV_DRY_RUN=1`) logged suppressions but applied none, so every
   would-be-suppressed candidate *is* in the final store. `S_without` is a clean deletion
   and the counterfactual is directly measurable. The three live ablation arms supply
   30 real suppressions as the reality check.

---

## 3. What was built

All under `berkeley-function-call-leaderboard/`.

| File | Responsibility |
|---|---|
| `bfcl_eval/scripts/hnav_answer_index.py` | Gold index, `carries()` via the production grader, `answerable()` via `retrieval_sim` top-3, `--validate` measuring the proxy conversion factor `p_hat` |
| `bfcl_eval/scripts/label_outcomes_hnav.py` | §1.1 corrected suppressed-update-aware label |
| `bfcl_eval/scripts/counterfactual_noop.py` | §1.2 `(sim_high, delta, veto, preflight)` sweep + headroom frontier |
| `bfcl_eval/model_handler/middleware/diff_signals.py` | §1.3 17 marginal-diff signals — pure, answer-blind, runtime-importable |
| `bfcl_eval/scripts/refeature_diff.py` | Feature driver → `features_diff.jsonl` + quarantined `features_diff_oracle.jsonl` |
| `bfcl_eval/scripts/evaluate_hnav_stage1.py` | §1.4 evaluation, hypotheses, verdict → `gov_logs/HNAV_STAGE1.md` |
| `bfcl_eval/scripts/test_hnav_labels.py` (51) | Grader semantics, fate model, necessity/damage, uncertainty triggers, cross-table, region≡`decide` |
| `bfcl_eval/scripts/test_diff_signals.py` (42) | Diff isolation, fresh-add defaults, the falsifier's core claim, AST leakage scan |
| `bfcl_eval/scripts/test_hnav_eval.py` (44) | PR-AUC, planted-signal recovery, NC control, fold integrity, few-positive guard |

**Reused rather than rewritten:** `label_outcomes.walk_log` (chain-carry reconstruction of
decision-time store state) is imported *unchanged*; so are `retrieval_sim.simulate_kv/
simulate_vector/default_encode`, `entropy_metrics`, `probe_gen.content_tokens_ordered`,
`recsum_blobdiff.split_propositions`, `governance_filter.extract_verbatim_values`,
`replay_admission.Embedder` (for the ABTT-whitened `diff_sim_max`), and
`analyze_gov_replicates.holm`.

**Modified:** `bfcl_eval/scripts/calibrate_margin_entropy.py` only, and only additively —
`DIFF_FEATURES` + `FEATURE_KEYS` entries, a new `pr_auc()`, `nests`/`delta_specs`/`delta_ref`
kwargs on `nested_auc_report()`, and `load_decisions(escalated_only=)`. Defaults preserve
the frozen margin-entropy path; its suite still passes 34/34. **Nothing on the runtime
handler path changed.**

### The label

`hnav_target` (arm-invariant, the calibration target):

| value | meaning |
|---|---|
| `must_write` | terminal write, necessity ∧ ¬damage — the only *retrievable* carrier of a needed fact |
| `must_suppress` | terminal write, damage ∧ ¬necessity — it destroyed or buried another answer |
| `may_suppress` | terminal write, neither — inert |
| `inert_superseded` | non-terminal: a later write replaced it, so suppression changes nothing |
| `uncertain` | not resolvable offline |

`hnav_label` = `hnav_target` × `suppressed`, giving the five values the brief asked for
(`harmful_noop` / `correct_noop` / `harmful_add_or_update` / `correct_add_or_update` /
`uncertain`). **`suppressed` derives from `applied == "noop"`, not `action`** — the shadow
harvest logs `action="NOOP", applied="none"`, meaning the candidate was written anyway.
That distinction is what makes the harvest a clean counterfactual, and it is unit-tested.

---

## 4. What we found

### 4.1 A structural fact nobody had measured: chains overwrite themselves

Each store ref receives **3.32 writes on average (max 52)**, and only the *terminal* write
reaches the snapshot the questions are answered against.

| fate | n | share |
|---|---|---|
| `terminal` | 665 | 14.5 % |
| `superseded` | 1753 | 38.3 % |
| `gone` | 2152 | 47.0 % |
| unresolved | 5 | 0.1 % |

**85 % of all governed write decisions are first-order inert** — suppressing them cannot
change any answer. This was modelled explicitly (`inert_superseded`) rather than dumped
into `uncertain`; the initial version discarded 78 % of the data before this was diagnosed.

### 4.2 Where the headroom actually sits

Over 4575 labeled decisions (student excluded): `must_write` 159 (3.5 %),
`must_suppress` 41, `may_suppress` 397, `inert_superseded` 3905, `uncertain` 73.

| fate / op family | n | must_write | rate |
|---|---|---|---|
| terminal / **add** | 555 | **150** | 0.270 |
| terminal / update_replace | 110 | 9 | 0.082 |
| every non-terminal bucket | 3910 | **0** | 0.000 |

### 4.3 The mechanism is real — and lands exactly where it costs nothing

On the 220 near-duplicate update/replace decisions (`sim_max ≥ 0.90`):

| | |
|---|---|
| median whole-blob `sim_max` | **0.944** |
| median marginal `diff_sim_max` | **0.196** |
| median drop | 0.745 (90.9 % of rows drop > 0.10) |
| **`must_write` among them** | **0** |

The write-side geometry defect described in the plan is **confirmed at scale**. But not one
of those decisions is a real loss. The losses are on **fresh adds**, where there is no
predecessor and therefore no marginal diff to compute. *Fixing the geometry defect cannot
recover accuracy that was never at risk.*

### 4.4 The pre-registered hypotheses

**H1 — write-side headroom is real: FAIL.**       
Widest deployable cell (`sim_high=0.800, delta=0.60`, verbatim veto + preflight honoured):
coverage **0.0264** of all writes (bar 0.05); correct-NOOP precision 0.959, Wilson95
[0.907, 0.982]; harmful 5/121, Wilson95 [0.018, **0.093**] (bar ≤ 0.05); expected ΔAcc
**−0.0105**. The *entire* deployable frontier has negative expected ΔAcc (−0.005 … −0.011).
`preflight_ok` is False for 57 % of decisions, so the preflight veto already removes most of
the near-duplicate region before geometry ever gets to suppress it.

**H2 — the marginal diff carries signal geometry lacks: PASS.**

| nest | AUC | PR-AUC |
|---|---|---|
| geometry | 0.5243 | 0.0343 |
| geometry+margin_entropy | 0.5269 | 0.0343 |
| geometry+margin_entropy+diff | **0.6027** | **0.0414** |
| diff_only | 0.4848 | 0.0308 |

Base rate (PR no-skill line) 0.0348. ΔAUC(diff | geometry+margin) = **+0.0764**,
CI95 [+0.0147, +0.1382]; NC-shuffled control −0.0058, CI95 [−0.0205, +0.0089].
Stable across all 12 leave-one-arm-out refits. **Robustness:** `must_write` concentrates in
fresh adds and `has_old` separates adds from updates perfectly, so the gain was re-measured
without it — `has_old` alone +0.0455 [+0.0098, +0.0818], diff content without `has_old`
**+0.0713 [+0.0105, +0.1321]**. The signal is genuine diff content, not the add/update
indicator.

**H3 — the falsifier is cheap: FAIL.** False-override rate **0.664** at τ=0.5 (bar 0.20).

**Oracle ceiling (run last, separately, so it cannot contaminate the analysis).**
Ground-truth-derived features reach AUC **0.6164** vs the answer-blind model's **0.6027**
(ΔAUC vs geometry+margin +0.0906 [+0.0582, +0.1194]). Roughly **0.014 of AUC** is left for
*any* online feature — write-side prediction is already saturated.

**Reality check.** Of the 30 suppressions the gate actually applied across the live arms
(geometry_only 5, gm_v1 5, joint_entropy_diag 20), the label calls 19 `inert_superseded`,
8 `may_suppress`, and only **3 `harmful_noop`** — independently corroborating the negative.

### 4.5 Proxy validity

"Answerable" = a top-3 retrieved item carries a gold answer. It is a strong **necessary**
condition, not a sufficient one, so event counts are converted to expected accuracy with a
measured factor rather than 1:1. Pooled over 12 arm-replicates:
**p_hat = 0.409** (kv 0.445, vector 0.386). Illustrative stratum (rep01/kv):
carried & retrievable 12/25 correct, carried but not retrievable 3/8, not carried 6/122.

---

## 5. Reproducing it

Interpreter `C:\Users\USER\miniconda3\envs\BFCL\python.exe`, CWD
`berkeley-function-call-leaderboard/`. `$HG = Qwen_Qwen3-4B-Instruct-2507-FC-GOV`.
Runtimes: labeller ≈ 10 s per arm-replicate, `refeature_diff` ≈ 20–120 s per arm,
`evaluate_hnav_stage1` ≈ 6–7 min at `--n-boot 2000 --n-perm 10000`.

```bash
# 0. tests
python bfcl_eval/scripts/test_hnav_labels.py
python bfcl_eval/scripts/test_diff_signals.py
python bfcl_eval/scripts/test_hnav_eval.py
python bfcl_eval/scripts/test_calibrate_margin_entropy.py     # regression

# 1. proxy validation, per arm-replicate (12x), --append
python bfcl_eval/scripts/hnav_answer_index.py --validate --arm harvest/rep01/v1_shadow \
  --result-dir result_gov_me_harvest/rep01/v1_shadow/$HG \
  --score-dir  score_gov_me_harvest/rep01/v1_shadow/$HG \
  --out gov_logs/hnav_proxy_validation.json --append

# 2. labels, per arm-replicate, --append
python bfcl_eval/scripts/label_outcomes_hnav.py \
  --gov-log gov_logs/me_harvest/rep01_v1_shadow \
  --result-dir result_gov_me_harvest/rep01/v1_shadow/$HG \
  --arm v1_shadow --replicate 01 \
  --out gov_logs/me_harvest/outcomes_hnav.jsonl --append

# 3. counterfactual sweep (seconds)
python bfcl_eval/scripts/counterfactual_noop.py \
  --labels "gov_logs/me_harvest/outcomes_hnav.jsonl" "gov_logs/me_ablations/outcomes_hnav_*.jsonl" \
  --proxy gov_logs/hnav_proxy_validation.json --out gov_logs/hnav_counterfactual.json

# 4. diff features, per arm
python bfcl_eval/scripts/refeature_diff.py --gov-logs "gov_logs/me_harvest/rep0*_v1_shadow" \
  --out gov_logs/me_harvest/features_diff.jsonl \
  --oracle-out gov_logs/me_harvest/features_diff_oracle.jsonl

# 5. evaluation + verdict  (list --diff-features EXPLICITLY, see gotcha below)
python bfcl_eval/scripts/evaluate_hnav_stage1.py \
  --logs "gov_logs/me_harvest/rep0*_v1_shadow" "gov_logs/me_ablations/rep0*_geometry_only" \
         "gov_logs/me_ablations/rep0*_gm_v1" "gov_logs/me_ablations/rep0*_joint_entropy_diag" \
  --labels "gov_logs/me_harvest/outcomes_hnav.jsonl" "gov_logs/me_ablations/outcomes_hnav_*.jsonl" \
  --diff-features "gov_logs/me_harvest/features_diff.jsonl" \
                  "gov_logs/me_ablations/features_diff_geometry_only.jsonl" \
                  "gov_logs/me_ablations/features_diff_gm_v1.jsonl" \
                  "gov_logs/me_ablations/features_diff_joint_entropy_diag.jsonl" \
  --extra-features "gov_logs/me_harvest/features_v2.jsonl" \
  --counterfactual gov_logs/hnav_counterfactual.json \
  --label must_write --seed 12345 --n-boot 2000 --n-perm 10000 \
  --out gov_logs/hnav_stage1_report.json --md gov_logs/HNAV_STAGE1.md

# 6. oracle ceiling — LAST and separate
python bfcl_eval/scripts/evaluate_hnav_stage1.py ... --oracle-ceiling \
  --oracle-features "gov_logs/me_*/features_diff_oracle*.jsonl" \
  --out gov_logs/hnav_stage1_oracle_ceiling.json
```

### Artifacts produced

`gov_logs/HNAV_STAGE1.md` (verdict) · `hnav_stage1_report.json` ·
`hnav_stage1_oracle_ceiling.json` · `hnav_counterfactual.json` (256 cells) ·
`hnav_proxy_validation.json` · `me_harvest/outcomes_hnav.jsonl` ·
`me_ablations/outcomes_hnav_<arm>.jsonl` · `features_diff*.jsonl` per arm.

---

## 6. Verification performed

- **All 24 offline suites green**, 137 checks new (51 + 42 + 44). No existing suite changed.
- **Determinism:** labeller and sweep both byte-identical on rerun.
- **Zero `chain_desync`** across all 4584 labeled rows. Remaining `uncertain` reasons:
  `both_effects` 49, `old_text_missing` 12, `store_empty` 5.
- **Region predicate ≡ `governance_filter.decide`**: six boundary cases (including
  `sim_max == sim_high` and `r == delta`, where the inequalities are strict) are asserted
  against the real `decide()` rather than a re-statement of it.
- **Damage prefilter is exact**: `--exact-damage` differs on 0/315 rows.
- **Leakage control, three layers:** the `FEATURE_KEYS` whitelist (extended with
  `oracle`/`gold` as forbidden tokens); an AST scan proving `diff_signals.py` never
  references benchmark ground truth and `refeature_diff.feature_row` is answer-blind; and
  `cv_splits`' chain-straddle assertion, mandatory here because every decision in a
  `(backend, scenario, replicate)` chain shares one final store.

### Bug caught mid-run

The glob `features_diff_*.jsonl` also matches `features_diff_oracle_*.jsonl`, which sorts
later; a last-writer-wins merge dict silently dropped the real features for three arms and
produced a spurious **H2 = FAIL** (with tell-tale AUC = 0.5000 in the per-replicate
breakdown). `merge_features` now *accumulates* records per candidate and reports the number
of rows that actually received a requested key, not the number that merely matched.

---

## 7. Limitations (binding — carried into the artifact)

- **First-order counterfactual.** Reverting a write from the final snapshot does not
  simulate how the agent would have behaved after seeing a synthetic success. Non-terminal
  writes are `inert_superseded` on that basis; the sensitivity target `must_write_lineage`
  (marginal gold still uniquely carried in the final store) bounds the under-count from
  above. The 30 live suppressions are the only empirical check, and n=30 supports a sanity
  check, never a rate.
- **Answerability is necessary, not sufficient** — hence `p_hat`, never 1:1.
- **Small cells.** Inside a NOOP-region cell: exact integers and Wilson intervals only; no
  multivariate fit below 25 positives, replaced by a within-chain label-permutation p-value
  on the single pre-registered univariate falsifier `diff_sim_max`, Holm-corrected across
  cells (0 cells survived Holm). The headline is the shape of the risk-coverage frontier,
  never a single cell.
- **Three replicates** support a min/max stability check, not an interval.

---

## 8. Where this leaves the project

The write-side lever is closed, with a clean and defensible negative rather than an
underpowered shrug: the defect is real, large, measurable, and located entirely on
decisions that cannot change an answer; the oracle ceiling shows even a ground-truth
feature could not rescue it.

Stages 2–4 of H-Nav were deliberately **not** implemented — the scope for this round was
Stage 1 plus the progression decision, so that Stages 2–4 get planned against evidence
rather than assumption. The pre-registered consequence of ¬H1 is to move to the
**action-side H_act** pathway: measuring how certain the agent is when it *selects* a
memory operation, which requires handler-side plumbing (top-k logprobs of the decision
tokens, or N sampled completions with parsed tool calls) that does not exist today —
`QwenGovHandler` never touches the completion request, and only `qwen_se.py` passes `n>1`.
That plumbing plus a live run is the next unit of work.

Related reading in this repo: `IDEA_PROACTIVE_RECALL_INJECTION.md` argues a neighbouring
point from the read side — that 94 % of question turns issue no retrieval call at all —
which is consistent with the finding here that write-side hygiene cannot move the score.

**Nothing in this session was committed.** New files are untracked;
`calibrate_margin_entropy.py` is the single modified tracked file.
