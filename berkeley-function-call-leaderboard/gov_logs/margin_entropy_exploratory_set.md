# Frozen exploratory dataset list — margin–entropy calibration (Plan §13.1)

**Frozen: 2026-07-21, at commit `cd4cb81` (branch `claude/nihai-plan-v2-cascade-thresholds-k34u67`).**

Per TWO_STAGE_GEOMETRY_MARGIN_ENTROPY_PLAN.md §13.1, the 2026-07-18/19 campaign logs are
the **exploratory set**: they MAY inform feature development (definitions of nmargin,
ΔH_self, churn, channel grouping, small-store cutoffs) and MUST NOT be used for threshold
freezing — they overlap the reported test outcomes of `EVAL_RESULTS.md`.

## Exploratory set (feature development allowed; threshold freezing FORBIDDEN)

- `gov_logs/replicates/rep01_governed` … `rep05_governed` (5× A/B governed arm,
  campaign `govrep_20260718T125332Z`)
- `gov_logs/ablations/rep0{1,2,3}_{governed_full,stage0_only,geo_off,s2_off,p_only,read_set3,read_adaptive}`
  (8-arm × 3-replicate ablation campaign; baseline arm has no governance log)
- `gov_logs/harvest/`, `gov_logs/shadow/`, `gov_logs/governed_calibrated/`,
  `gov_logs/governed_sage/` (pre-campaign harvests and shadow runs)
- Derived artifacts: `gov_logs/ab5_analysis.json`, `gov_logs/ablations_analysis.json`,
  `gov_logs/g11_entropy_contribution.json`, `gov_logs/geometry_dH_verdict.json`,
  `gov_logs/plan3_calibration.json`

## Calibration set (threshold freezing; to be produced in Phase 3)

- NEW shadow harvest arms (`GOV_DRY_RUN=1`, gov2 logging, ≥3 replicates, both backends),
  run AFTER the gov2 schema lands. Directory: `result_gov_me_harvest/` +
  `gov_logs/me_harvest/`. None exist at freeze time.

## Confirmatory set (never touched during tuning)

- The Phase 7 confirmatory 5× A/B replicates. Thresholds must already be committed in
  `gov_logs/margin_entropy_calibration.json` + `CALIBRATION_FROZEN.md` (with git hash)
  before the first confirmatory replicate is launched (gate G3).
