# H-Nav action-side dataset partition (binding)

Extends `gov_logs/margin_entropy_exploratory_set.md` to the action-side program.

| Set | Data | Allowed uses | Forbidden uses |
|---|---|---|---|
| Exploratory (2026-07-18/19) | `gov_logs/replicates`, `gov_logs/ablations`, `gov_logs/harvest` | feature ideas, code debugging | any threshold or claim; contains NO action-side data |
| Calibration (2026-07-21/22) | `gov_logs/me_harvest`, `gov_logs/me_ablations` | Stage 1 write-side artifacts (frozen) | H_act anything — no logprobs/samples exist; backfill is impossible and must not be attempted |
| **Shadow / Stage-4-calibration (this campaign)** | `gov_logs/hnav_shadow`, `result_hnav_shadow` | Stage 3 shadow validation; Stage 4 threshold selection (dev = rep01–rep02, val = rep03) | being re-used as confirmatory evidence for any intervention |
| Confirmatory (future Stage 4 campaign) | `gov_logs/hnav_stage4`, `result_hnav_stage4` | one pre-registered paired analysis after thresholds are frozen + sha-pinned | any tuning, any peeking during threshold selection |

Rules: thresholds freeze BEFORE the confirmatory campaign starts (sha recorded
in its manifest); the shadow campaign's official scores may be cited only as
(a) the shadow-mode sanity check and (b) auxiliary ungoverned replicates;
`student` stays excluded from analysis (generation unchanged) per Plan v2.
