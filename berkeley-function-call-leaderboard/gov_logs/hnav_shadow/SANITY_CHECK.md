# Shadow campaign sanity check (pre-registered invariant)

Campaign `govrep_20260728T184333Z`, 3 replicates × {baseline, hact_shadow},
completed 2026-07-29 00:25Z, zero errors, no resumes.

## Decision-neutrality: HOLDS

All 1253 governance decisions in the hact_shadow arm have `applied: "none"`
(GOV_DRY_RUN=1 + HACT_POLICY=shadow applied nothing). The instrumentation
changed no decision.

## Score-level comparison: kv offset, attributed to seeded-trajectory draw

| rep | base kv | hact kv | base vec | hact vec |
|---|---|---|---|---|
| rep01 | 18/155 | 14/155 | 20/155 | 20/155 |
| rep02 | 19/155 | 15/155 | 25/155 | 20/155 |
| rep03 | 22/155 | 15/155 | 21/155 | 21/155 |

Diagnosis (all checks in this order, before any Stage 3 number was unblinded):

1. No dead chains: hact kv stores are healthy and often larger than baseline.
2. Flips go BOTH directions, scattered across scenarios (rep01 −13/+9,
   rep02 −7/+3, rep03 −11/+9): no targeted failure mode.
3. Pseudo-replication confirmed: the replicate-invariant HACT_SEED makes the
   three hact reps near-identical draws — correct-answer Jaccard across reps:
   vector 0.95–1.00, kv 0.61–0.88; 374/713 (52%) of (test_id, step) keys have
   byte-identical primary calls across ≥2 replicates. The hact arm is
   effectively ONE trajectory draw, so a single below-average kv draw appears
   as a "consistent" offset.
4. The primary request is parameter-identical to baseline except `logprobs`
   (does not affect sampling) and `seed` (fixes the draw, does not change the
   distribution).

**Verdict:** decision-neutrality (the property shadow mode must guarantee)
holds; the score offset is a pinned-seed trajectory-draw artifact under
pseudo-replication, not an intervention effect.

**Consequences (binding):**
- The hact_shadow arm's official scores are VOID as "auxiliary baseline
  replicates" (the dual-use is dropped).
- T1/T2 shadow analyses remain internally valid: features and counterfactual
  labels both derive from the hact arm's own trajectories.
- Effective replication for cross-replicate checks is < 3; NC12 and threshold
  validation use twin exclusion (see PREREGISTRATION.md corrections).
- Future campaigns use per-replicate seeds (runner offsets HACT_SEED by the
  replicate index; committed before any Stage 4 run).

## Instrumentation health

3276 hact rows (1366 joinable steps + 1910 orphans = no-call/parse-fail
steps); `logprobs_supported: true` throughout (zero degradation events);
100% of the 1253 gov decisions join a hact record on (test_id, step_idx).

## Label distribution (drives the gate)

`must_suppress` = **9**/1253 (rep01 2, rep02 7, rep03 0) — far below the
pre-registered EPV bar (25 positives). `must_write` = 50, `may_suppress` =
134, `inert_superseded` = 1052, `uncertain` = 8. The harmful-write event
class is as rare here (0.7%) as in Stage 1 (0.9%): rarity, not signal
quality, binds first.
