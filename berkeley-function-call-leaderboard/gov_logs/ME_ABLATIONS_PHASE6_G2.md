# Plan v2 Phase 6 — reduced ablation matrix, gate G2 verdict

**Verdict: G2 FAIL.** `gm_v1` (arm 5, `geometry_margin_entropy_v1`) is not
demonstrably non-inferior to `geometry_only` (arm 2) or `legacy_margin` (arm 3)
on the official score at the pre-declared −2 pp margin, and no pre-registered
secondary shows superiority. Per §23.1, **the Phase 7 confirmatory campaign
runs `geometry_only` as the headline arm.**

---

## 1. Run provenance

| | |
|---|---|
| Matrix | G1-fail reduced (§18 priority arms): baseline / geometry_only / gm_v1 / legacy_margin / joint_entropy_diag × 3 replicates |
| Arms spec | `gov_logs/me_ablation_arms.json` (`--arms-json @file`; byte-identical across launch and resume, verified) |
| Original run | `govrep_20260722T074659Z`, git head `2aff154`, launched 2026-07-22T07:46Z |
| Interruption | 2026-07-22T14:22Z, `tunnel_probe` connection-refused after rep03/baseline evaluate; runner fail-stopped per §8.3 |
| Resume | `govrep_20260723T075136Z`, `--start-replicate 3` (full rep03 re-run, all 5 arms), same manifest, completed 2026-07-23 |
| Server | vLLM 0.9.1, `Qwen/Qwen3-4B-Instruct-2507`, max_model_len 32000 — matches pinned `run_start` values |
| Instance continuity | **Asserted, not verified** (RESUME_PROTOCOL 2026-07-22 addendum: `/v1/models` `created` is stamped per-request on this server — two probes 5 s apart differed by 5 s — so API fingerprinting is unusable). Resume authorized by operator on 2026-07-23. |
| Temperature | Requested 0.001; vLLM `sampling_params` clamps T < 0.01 to **effective T = 0.01**. Applies uniformly to every arm and replicate of the campaign (same vLLM version throughout), so comparability is unaffected. |
| Post-resume checklist (§6) | All green: `run_end` present; 15/15 replicate×arm evaluates `exit_code 0` with both score files present; single inert `error` record (the 07-22 tunnel probe); fresh rep03 gov-log dirs only (no stale attempt mixed in). |
| Runtime | rep03 arm-runs 33.7–36.5 min each — under the 44 min/arm governed_full bound (§19 engineering criterion). |

## 2. Primary: official score (survival-conditional pairing, exact McNemar, scenario bootstrap, Holm m=8)

Analyzer: `analyze_gov_replicates.py`, seed 12345, n_boot 10000, student excluded,
joint drop of units dead in any arm (4 units). Outputs:
`me_ablations_analysis.json` (ref=baseline),
`me_ablations_analysis_ref_geometry_only.json`,
`me_ablations_analysis_ref_legacy_margin.json`.

### vs baseline (ref acc: kv 0.2167, vector 0.2172)

| arm | backend | dAcc | bootstrap CI95 | McNemar p (Holm) |
|---|---|---|---|---|
| geometry_only | kv | −0.0375 | [−0.1292, +0.0327] | 0.253 (1.000) |
| geometry_only | vector | −0.0241 | [−0.0807, +0.0250] | 0.419 (1.000) |
| gm_v1 | kv | −0.0208 | [−0.0766, +0.0375] | 0.542 (1.000) |
| gm_v1 | vector | −0.0310 | [−0.0862, +0.0172] | 0.281 (1.000) |
| legacy_margin | kv | −0.0333 | [−0.0894, +0.0167] | 0.332 (1.000) |
| legacy_margin | vector | −0.0241 | [−0.0621, +0.0034] | 0.419 (1.000) |
| joint_entropy_diag | kv | −0.0458 | [−0.1020, +0.0122] | 0.117 (0.938) |
| joint_entropy_diag | vector | −0.0241 | [−0.0456, −0.0033] | 0.427 (1.000) |

No arm differs from baseline after Holm; every point estimate is ≤ 0.

### G2 pairings (gm_v1 as candidate)

| comparison | backend | dAcc | bootstrap CI95 | CI lower vs −2 pp margin |
|---|---|---|---|---|
| gm_v1 vs geometry_only | kv | +0.0167 | [−0.0208, +0.0681] | −2.08 pp — **below margin** |
| gm_v1 vs geometry_only | vector | −0.0069 | [−0.0814, +0.0596] | −8.14 pp — **below margin** |
| gm_v1 vs legacy_margin | kv | +0.0125 | [−0.0531, +0.0913] | −5.31 pp — **below margin** |
| gm_v1 vs legacy_margin | vector | −0.0069 | [−0.0536, +0.0310] | −5.36 pp — **below margin** |

Point estimates are near zero (kv slightly positive, vector slightly negative),
but with n=3 replicates the scenario-bootstrap CIs are too wide to establish
non-inferiority at −2 pp in any backend. **Non-inferiority component: FAIL**
(not shown, as distinct from shown inferiority — see §5 interpretation).

## 3. Secondaries (§19; labeler `label_outcomes.py`, baseline-paired per replicate)

Harmful-write rate (harmful / gov2-labeled admission decisions):

| rep | geometry_only | gm_v1 | joint_entropy_diag |
|---|---|---|---|
| 1 | 35/297 = 0.118 | 47/460 = 0.102 | 58/452 = 0.128 |
| 2 | 36/388 = 0.093 | 35/314 = 0.111 | 39/442 = 0.088 |
| 3 | 28/405 = 0.069 | 24/331 = 0.073 | 41/444 = 0.092 |

gm_v1 beats geometry_only in **1 of 3 replicates** — direction not replicated;
no superiority claim (binding: no pooled-count claims). `legacy_margin` writes
pre-gov2 logs (no `candidate_id`), so its harmful-write rate is **not
computable** with the §13.2 labeler — recorded as a limitation.

Other pre-registered secondaries:
- **Online NLI calls: 0** in all 12 governed arm-runs (required).
- Useless-duplicate admissions: 0 everywhere.
- Escalation to Stage 1 (gm_v1): 16.9–24.3 % of decisions; median decision
  latency 34–40 ms vs 26–34 ms for geometry_only (Stage-1 overhead ≈ 5–8 ms).
- joint_entropy_diag REWRITE actions: 3–4 per replicate; no official-score or
  harmful-write advantage.

No secondary shows gm_v1 superior to either comparator. **Superiority
component: FAIL.**

## 4. Gate G2 (§23.1)

> G2: arm 5 non-inferior to arms 2 and 3 on official score AND superior on ≥1
> pre-registered secondary.

Both conjuncts fail ⇒ **G2 FAIL**. Pre-registered consequence: the Phase 7
confirmatory campaign runs **geometry_only** as the headline arm.

## 5. Interpretation (thesis narrative)

- Consistent with G1 (entropy added no out-of-sample predictive value, two
  independent estimators at two layers) and the entropy-v2 negative
  (`entropy_v2_report.json`): the margin–entropy Stage 1 neither helps nor
  demonstrably harms the official score; the geometry stage remains the
  load-bearing component (Holm-significant `geo_off` result, 2026-07 campaign).
- G2's failure is a *power/width* failure on the non-inferiority side (point
  estimates ≈ 0, CIs ± 5–9 pp at n=3) plus a genuine absence of secondary
  wins. The honest claim is the §24 simplification + negative-result
  narrative, not a joint-rule win.
- diag arm behaved as predicted by brief §6 (entropy-alone decisions
  underperform: worst kv point estimate of the matrix, −4.6 pp vs baseline).
