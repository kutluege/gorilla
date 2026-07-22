# Margin-entropy calibration freeze (plan SS13.3)

- Calibration file: `margin_entropy_calibration.json` (sha 268ebac77c26f7a1)
- Git head at freeze: `76672a413fb7aabbe8e653bac796b65bbe929370`
- Label: `harmful_write`; target risk: 0.15; seed: 12345
- Data manifest: [{"log": "gov_logs\\me_harvest\\rep01_v1_shadow\\governance_log.jsonl", "sha": "07724718ca0f29ec"}, {"log": "gov_logs\\me_harvest\\rep02_v1_shadow\\governance_log.jsonl", "sha": "dbd1e7d26018e551"}, {"log": "gov_logs\\me_harvest\\rep03_v1_shadow\\governance_log.jsonl", "sha": "a6bfa497456b7788"}]

Thresholds are FROZEN as of this commit. The confirmatory
campaign must reference this file's git hash in its manifest
(gate G3); no re-tuning after the first confirmatory replicate.

## Gate G1 verdict (binding, frozen with this file)

**FAIL — entropy terms dropped per plan SS23.1.** On 261 escalations / 25
harmful-write positives (3-replicate harvest `govrep_20260721T234427Z`,
chain-grouped leave-one-scenario-out CV): dAUC(entropy | geometry+margin) =
-0.032, CI95 [-0.085, +0.016]; NC-shuffled control -0.004 [-0.073, +0.039]
(indistinguishable from real). Geometry alone is the strongest predictor
(AUC 0.79). EPV 4.2 < 20 -> Option B not fitted (plan SS22 fallback).

Confirmatory v1 therefore ships as GEOMETRY+MARGIN: Option-A thresholds
R=3, M_kv=0.0, M_vec=0.01367 with the interference term inert
(GOV_ME_DH_*=1e9, GOV_ME_CHURN_*=1e9). One entropy-enabled arm
(calibrated D/C from the frozen grid: kv 0.1977/0.1667, vec 0.6913/0.5)
runs in the Phase 6 matrix as a live diagnostic of the same negative.
