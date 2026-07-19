# EVAL_RESULTS — Write-time Governance Cascade on BFCL v4 Memory

**Campaign:** 2026-07-18 → 2026-07-19 (single 48 h vLLM instance, `created=1784379144`, vLLM 0.9.1)
**Model under test:** `Qwen/Qwen3-4B-Instruct-2507` (tunnel `http://localhost:8000/v1`), temperature 0.001, `--num-threads 1` strictly sequential
**Arms:** baseline `...-FC` vs governed `...-FC-GOV` (cascade: Stage 0 geometry → Stage 1 NLI → Stage 2 retrieval-entropy margin)
**Runs:** 5×A/B `govrep_20260718T125332Z` (git `8f4ea30`) + 8-arm × 3-replicate ablations (manifests committed under `result_gov_replicates/` and `result_gov_ablations/`)
**Seed statement (binding, G14):** N=5 replicates, sequential, no fixed seed; run-to-run variation stems from server-side nondeterminism at temperature 0.001; replicate ≠ seed.

The three metric layers below are never mixed: the **official BFCL score is the only primary
metric**; chains_alive/W-R-R-A is diagnostic; the paraphrase-tolerant judge is secondary.

---

## 0. Headline verdicts

1. **Primary (official BFCL score): the full cascade is a statistical null.** Governance
   neither improves nor degrades official accuracy in either backend (kv ΔAcc −0.7 pp,
   McNemar p=0.80; vector ΔAcc 0.0, p=1.00; CI95 ≈ ±4–5 pp). The cascade governs ~1,800
   write decisions per 5 replicates (30 NOOP dedups, 18 canonicalizing rewrites) at zero
   measurable accuracy cost.
2. **The geometric stage is load-bearing — the campaign's only Holm-significant effect.**
   Removing Stage 0 and letting NLI front the cascade (`geo_off`) costs
   **−10.6 pp on kv (McNemar p=0.0001, Holm p=0.0013, CI95=[−17.8, −3.8])**. This is the
   causal counterpart of the binding §8.4 correlational verdict (geometry-first,
   `gov_logs/geometry_dH_verdict.json`: vector ρ(sim_max, ΔH)=0.757, CI95=[0.482, 0.899] ≥ 0.40).
3. **No arm improves the official score.** The two nominally-positive arms (`s2_off` vector
   +1.9 pp, `read_set3` vector +3.0 pp) are far from significance. Placement/destructive-guard
   (`p_only`, kv −7.9 pp, Holm p=0.0598) and margin-adaptive read gating (`read_adaptive`,
   kv −6.0 pp, Holm p=0.35) trend negative.
4. **Chain survival is benchmark noise, not a governance effect** (§2). The secondary
   paraphrase-tolerant judge (§3) mildly favors the governed arm in the A/B (vector 121 vs
   112 /775), consistent with canonicalized memories producing semantically-correct but
   keyword-mismatched answers — but the effect is not arm-selective in the ablation grid,
   so it is reported as suggestive only. The inert entropy signal carries real information the margin
   gate ignores (§4).

---

## 1. Layer 1 — PRIMARY: official BFCL score (unmodified `bfcl evaluate`)

### 1.1 Headline 5× A/B (`gov_logs/ab5_analysis.json`)

Survival-conditional pairing, n_pairs=450 per backend (3 drops each; 6 joint), exact
McNemar, scenario-level bootstrap CI (n=2000, seed 12345), Holm m=2, student excluded
(pre-registered).

| backend | acc baseline | acc governed | ΔAcc | b/c | McNemar p | Holm p | CI95(ΔAcc) |
|---|---|---|---|---|---|---|---|
| kv | 0.2178 | 0.2111 | −0.0067 | 34/31 | 0.8043 | 1.0000 | [−0.0505, +0.0396] |
| vector | 0.1800 | 0.1800 | +0.0000 | 35/35 | 1.0000 | 1.0000 | [−0.0366, +0.0430] |

Per-replicate raw correct counts (/155, all scenarios incl. student):

| rep | base kv | gov kv | base vec | gov vec |
|---|---|---|---|---|
| 1 | 21 | 18 | 22 | 22 |
| 2 | 21 | 17 | 21 | 17 |
| 3 | 24 | 28 | 21 | 16 |
| 4 | 22 | 22 | 14 | 22 |
| 5 | 24 | 24 | 24 | 27 |

Within-arm replicate swings (governed kv 17→28) exceed every arm-vs-arm delta — the
replicate-variance structure, not the arm, dominates raw scores; hence the paired design.

### 1.2 Ablations, 8 arms × 3 replicates (`gov_logs/ablations_analysis.json`)

Reference arm `baseline`; joint survival across ALL 8 arms (4 units dropped); Holm m=14
over backend × arm-pair; n_pairs=265 per cell.

| arm (vs baseline) | kv ΔAcc | kv p (Holm) | vec ΔAcc | vec p (Holm) |
|---|---|---|---|---|
| governed_full | −0.0340 | 0.188 (1.00) | +0.0000 | 1.000 (1.00) |
| **geo_off** (NLI-first) | **−0.1057** | **0.0001 (0.0013)** | −0.0264 | 0.392 (1.00) |
| stage0_only | −0.0151 | 0.636 (1.00) | +0.0113 | 0.728 (1.00) |
| s2_off | −0.0151 | 0.659 (1.00) | +0.0189 | 0.511 (1.00) |
| read_adaptive | −0.0604 | 0.029 (0.35) | −0.0038 | 1.000 (1.00) |
| read_set3 | −0.0453 | 0.119 (1.00) | +0.0302 | 0.332 (1.00) |
| p_only | −0.0792 | 0.0046 (0.0598) | −0.0340 | 0.298 (1.00) |

Reading: only `geo_off` kv survives Holm. Directionally, the cascade's cost concentrates
where Stage 2's canonicalize-rewrite and the placement guard touch KV keys; its benefit
(if any) sits in vector paraphrase tolerance (§3). `stage0_only` ≈ `s2_off` ≈ baseline
supports "geometry does the safe work; later stages add cost without measurable official
gain at these calibrations."

Arm configs (exact `GOV_*` env) are pinned in `result_gov_ablations/manifest.jsonl`
(`run_start.arms`). **Note (binding):** `GOV_P_ENABLED` gates §5 placement **and** the §7
destructive guard together — there is no G7-only flag, so `p_only` measures their joint
effect.

---

## 2. Layer 2 — DIAGNOSTIC: chain survival + W-R-R-A (`parse_wrra`)

Sources: `gov_logs/ab5_wrra/rep0N_{arm}.jsonl`, `gov_logs/ablations_wrra/rep0N_{arm}.jsonl`.

Chains alive (of 5 scenarios; `student` dead in **every** run of **every** arm — writes
nothing at the prereq phase; excluded from primary analysis by pre-registration):

| arm (A/B, 5 reps) | kv alive /25 | vector alive /25 |
|---|---|---|
| baseline | 19 | 18 |
| governed | 17 | 18 |

The only other death mode is `healthcare`, which flickers on and off **arm-independently**
(it dies in baseline reps 2; governed reps 1, 2, 5 on kv — and in 9 of 24 ablation units
across all arm types including baseline). Governance neither kills chains (no governed-only
death mode) nor revives them.

**§5 chains_alive claim vs the committed reference:** the committed single-run Step 0
reference (`gov_logs/wrra_baseline.jsonl`: baseline 3/5 alive, kv 0.0710;
`gov_logs/wrra_governed.jsonl`: governed 4/5 alive, kv 0.1677) showed a +1 chain / +9.7 pp
governed advantage. Under 5 paired replicates this **does not reproduce**: survival
averages 3.4–3.8/5 in both arms and the accuracy gap is null (§1.1). The earlier
"governance revives chains" observation was within run-to-run noise — reported as such,
not claimed.

`geo_off`'s −10.6 pp is **not** chain death (its survival profile matches baseline):
NLI-first writes structurally worse memories (escalating every write burns steps and
admits near-duplicates that geometry would NOOP/canonicalize), which degrades read-time
answers while chains stay alive. W-R-R-A write/retrieve counts per scenario are in the
committed wrra files.

---

## 3. Layer 3 — SECONDARY: paraphrase-tolerant judge (never replaces the official score)

Judge = official-style keyword match **OR** DeBERTa-large-MNLI entailment (p>0.5) of any
ground-truth string by the model's final free-text message
(`gov_logs/ab5_paraphrase_judge_summary.json`; per-question records committed). The judge's
keyword-match reimplementation reproduces the official score files **exactly** (cross-check
passed on all 20 A/B run×backend cells).

Pooled over 5 replicates (/775):

| cell | official | judged | paraphrase gain |
|---|---|---|---|
| baseline / kv | 112 | 119 | +7 |
| baseline / vector | 102 | 112 | +10 |
| governed / kv | 109 | 120 | +11 |
| governed / vector | 104 | 121 | +17 |

Under paraphrase tolerance the governed arm gains more than baseline (kv: parity 120 vs
119; vector: +9 governed). Interpretation (secondary, uncorrected): Stage-2
canonicalization rewrites keys/values into forms that answer questions correctly but miss
the grader's exact keyword — the official metric slightly *understates* the governed arm's
semantic accuracy. This divergence is reported, not claimed as an improvement.

Ablation arms, pooled over 3 replicates (/465), same judge
(`gov_logs/ablations_paraphrase_judge_summary.json`):

| arm | kv official → judged | vector official → judged |
|---|---|---|
| baseline | 61 → 66 (+5) | 62 → 71 (+9) |
| governed_full | 52 → 58 (+6) | 55 → 61 (+6) |
| geo_off | 41 → 45 (+4) | 45 → 53 (+8) |
| stage0_only | 57 → 62 (+5) | 59 → 68 (+9) |
| s2_off | 64 → 71 (+7) | 57 → 66 (+9) |
| read_adaptive | 45 → 52 (+7) | 52 → 60 (+8) |
| read_set3 | 54 → 57 (+3) | 66 → 73 (+7) |
| p_only | 40 → 44 (+4) | 43 → 49 (+6) |

Paraphrase tolerance shifts every ablation arm roughly uniformly (+3 to +9) and does **not**
change the arm ordering: `geo_off` and `p_only` stay worst, `s2_off`/`read_set3` stay best.
Unlike the 5-replicate A/B (where the governed arm gained disproportionately), the 3-replicate
ablation grid shows no arm-selective paraphrase effect — consistent with the A/B's
+17-vs-+10 vector asymmetry being small and near noise-level. The secondary layer therefore
corroborates, rather than contradicts, the primary null and the geo_off harm.

---

## 4. G11 — entropy contribution analysis (DIAGNOSTIC ONLY; entropy never decides)

`gov_logs/g11_entropy_contribution.json`, over all 258 Stage-2-touched decisions in the
governed A/B logs. The margin decides (gate `GOV_S2_MARGIN=0.01`, live-calibrated p75);
`dH_neighbor`/`dH_mean`/`n_eff` are logged but inert.

Retro-check — among **margin-confident** accepts, do entropy-alarmed writes
(dH_mean > median 0.403) sit in worse scenarios?

| cell | n | mean scenario acc |
|---|---|---|
| margin-confident, dH high | 90 | 0.1097 |
| margin-confident, dH low | 69 | 0.1646 |
| margin-low, dH high | 26 | 0.1128 |
| margin-low, dH low | 70 | 0.1717 |

The entropy signal separates ~5.5–6 pp of scenario accuracy in **both** margin strata —
information the margin gate does not capture. Caveat: scenario-level association, plausibly
confounded (hard scenarios raise dH *and* lower accuracy); this motivates an entropy-aware
gate as future work, and per protocol it changed **nothing** in this campaign.

Stage-2 outcome mix: 243 `stage2_accept_low_confidence`, 12 `stage2_accept`,
3 `stage2_accept_rewritten` (plus 18 REWRITE decisions at the top level; 30 NOOPs from
geometry; 1739 ADDs of 1787 decisions across 5 governed replicates).

---

## 5. §8.4 cascade-ordering verdict (binding) and its causal confirmation

- **Correlational (binding, from the shadow harvest):** `gov_logs/geometry_dH_verdict.json`
  — vector ρ(sim_max, ΔH_neighbor)=0.757, CI95=[0.482, 0.899], gate ρ≥0.40 → **geometry-first**;
  kv n=4 (ρ=0.8, CI degenerate) but KV is canonical-key-first regardless. No NLI-first reorder.
- **Causal (this campaign):** the `geo_off` arm operationalizes the counterfactual
  (NLI-first) and loses −10.6 pp kv, Holm-significant (§1.2). Additionally NLI-first costs
  ~1.8× wall-clock per run (77–80 min vs 41–45 min) because every write escalates into the
  CPU NLI.

---

## 6. Methods (binding constraints, verbatim where required)

- Official scores come from unmodified `bfcl evaluate`; grader/score files never edited.
- Survival-conditional pairing: a question enters an arm-pair only if the chain is alive in
  both arms of that replicate; joint survival across ALL arms of a manifest counted once.
- Exact McNemar on discordant pairs; scenario-level bootstrap CI (n=2000, seed 12345);
  Holm-Bonferroni over backend × arm-pair (m=2 A/B; m=14 ablations).
- `student` excluded by pre-registration (writes nothing in every observed run;
  `--include-student` exists for sensitivity only).
- **N=5 replicates, sequential, no fixed seed; run-to-run variation stems from server-side
  nondeterminism at temperature 0.001; replicate ≠ seed.** (3 replicates for ablations.)
- All replicates of one manifest ran against the same vLLM instance (`created=1784379144`);
  every generate re-probed the tunnel; every phase exited 0; every evaluate's score files
  exist (manifest `score_files.exists` all true).
- Per-(replicate, arm) `GOV_LOG_DIR` isolation (append-mode middleware log, risk R7).
- Entropy never decides; the Stage-2 margin decides. Read gate and placement were OFF in
  the headline A/B (one-mechanism-per-arm, R4).
- Resume/interruption protocol: `gov_logs/RESUME_PROTOCOL.md` (not needed — zero failures).

## 7. §0 diagram → code map (DoD 12)

| Cascade node | Code |
|---|---|
| Stage 0 geometry (ABTT whiten, sim_max/r, NOOP/ADD/ESCALATE) | `bfcl_eval/model_handler/middleware/governance_filter.py` (`GovConfig`, `decide()`, `validate()`); artifact `middleware/artifacts/abtt_minilm_l6_d16.npz` |
| Stage 1 NLI resolve | `governance_filter.py` (`stage1_resolve`) + `middleware/semantic_entropy.py` (`NliScorer`, `_get_nli_model`, DeBERTa-large-mnli, CPU) |
| Stage 2 retrieval-entropy margin + one-shot canonicalize | `governance_filter.py` (`stage2_resolve`) + `middleware/retrieval_sim.py` + `middleware/probe_gen.py` (paraphrase channel: flan-t5-small, `GOV_PROBE_*`) |
| §5 placement, eviction, destructive guard (G6+G7, one flag) | `middleware/placement.py` (+ `middleware/recsum_blobdiff.py`, standalone) |
| §6 read-time gate (margin → bounded ambiguous set, one refinement) | `middleware/read_gate.py` |
| Stage −1 write-compliance watchdog (log-and-report only) | `governance_filter.py` (`write_compliance` events) |
| Handler wiring | `model_handler/local_inference/qwen_gov.py` (`QwenGovHandler`) |
| Runner / analyzer / diagnostics | `bfcl_eval/scripts/run_gov_replicates.py`, `analyze_gov_replicates.py`, `parse_wrra.py`, `correlate_geometry_deltaH.py`, `calibrate_placement.py` |

## 8. Honest summary for the thesis

The cascade **safely** governs writes (1,787 decisions, 0 chain kills, null official-score
impact) and its geometric first stage is **provably load-bearing** (only Holm-significant
effect in a 14-comparison family). But at these calibrations no configuration **improves**
the official BFCL score; the benchmark's replicate variance (±8 questions within an
identical arm) and the prereq-chain survival lottery dominate raw scores. The
paraphrase-tolerant secondary layer suggests the official metric slightly understates the
governed arm (vector +9/775), and the inert entropy signal demonstrably carries quality
information the margin gate ignores — both are future-work levers (entropy-aware gating,
canonicalization that preserves grader keywords), not claims.
