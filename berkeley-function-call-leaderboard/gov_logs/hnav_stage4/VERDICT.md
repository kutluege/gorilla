# alt10 write-rescue — live campaign verdict

**Analyzed:** 2026-08-08 (campaign finished 2026-07-29T10:48Z, sat unanalyzed).
**Verdict: PROMISING BUT INCONCLUSIVE (§8.3) — underpowered by construction.**
This label is frozen now and will not be revised in the final report.

## Headline numbers (analyzer, survival-conditional matched pairs, n_pairs=290/backend)

| contrast | backend | ΔAcc | 95% CI | McNemar p | Holm p |
|---|---|---:|---|---:|---:|
| **primary** a2_majority vs a1_random | kv | +0.0000 | [−0.046, +0.048] | 1.000 | 1.000 |
| **primary** a2_majority vs a1_random | vector | +0.0138 | [−0.042, +0.064] | 0.689 | 1.000 |
| secondary a2_majority vs baseline | kv | +0.0241 | [−0.048, +0.092] | 0.410 | 1.000 |
| secondary a2_majority vs baseline | **vector** | **+0.0345** | [−0.011, +0.079] | 0.184 | 0.736 |
| secondary a1_random vs baseline | kv | +0.0241 | [−0.054, +0.102] | 0.435 | 1.000 |
| secondary a1_random vs baseline | vector | +0.0207 | [−0.055, +0.091] | 0.504 | 1.000 |

Artifacts: `analysis_primary_a2_vs_a1.json`, `analysis_secondary_vs_baseline.json`.

## Why this is the honest label

1. **The live vector point estimate (+0.0345) reproduces the falsifier's prediction
   (+0.035) almost exactly** — and direction is positive in **3/3 replicates**
   (0.148>0.116, 0.161>0.110, 0.155>0.142). kv is mixed (2/3).
2. But the falsifier also predicted the campaign could not clear significance at
   n≈290 matched vector pairs, and it did not: CI includes zero, Holm p = 0.736.
   Pre-committed in the continuation plan: *"the correct label is §8.3 'promising
   but inconclusive' or 'underpowered', not validated."*
3. **The primary contrast is ~null.** a2 vs a1 is +0.014 vector / +0.000 kv, and
   a1 (random-select, equal budget) itself gains +0.021/+0.024 over baseline. So
   most of the gain over baseline is *generic extra-sampling/selection*, not
   majority-vote specificity. Interpretation constraint from INTEGRITY.md applies:
   A1 replaces the primary on ~90% of steps (a degraded agent), so even this
   attribution is conservative.

## Mechanism checks

- **A3 chain survival** (`wrra/`): `student` dead in all 18 cells; A2 neither
  kills nor revives chains (dead sets near-identical across arms; one healthcare
  death each in rep01-baseline-kv and rep03-a1-vector — those units are dropped by
  the analyzer's joint-survival rule). The accuracy delta is not a survival artifact.
- **A4 answerability strata** (`answerability.json`, pooled over 3 reps):
  A2 moves questions out of `not_carried` in the predicted direction —
  vector 347→333 with `carried_retrievable` 112→124; kv 376→344 but with the
  rescued mass landing disproportionately in `carried_NOT_retrievable` (19→43):
  kv rescues are carried but unretrievable, consistent with the BM25-over-key-names
  defect. Vector > kv on rescue retrievability, as the falsifier predicted.
- **A5 pre-registered negative control** (`nc_shuffled_assignment.json`) —
  **did not cleanly collapse.** Shuffling the sample-set→scenario assignment
  leaves vector rescued_modal at 6/6/7 vs the original 9/9/9 (~70% survives);
  kv 2/3/0 vs 2/4/4. A substantial fraction of the offline "rescue" metric is
  therefore **non-content-specific carriage** (short gold strings — median 7
  chars, often numbers — accidentally present in generic sampled text, plus
  redundantly-stated facts surviving the permutation). The falsifier's
  expected_dacc was optimistic in *attribution*, which is consistent with the
  near-null primary contrast. Reported as pre-registered, run, and failed-to-collapse.
- **A6 cost** (`cost_accounting.json`): a2 = 3,981 extra exploration requests,
  49.4M prompt + 9.5M completion tokens across 3 replicates, wall clock 2.48×
  baseline (83–86 min vs 33–35 min per replicate). Net recovered answers vs
  baseline (both backends pooled): 17. **Ratios: ~234 extra calls and ~558k extra
  completion tokens per net recovered answer.** Statistically arguable,
  operationally unattractive — the §22 sentence, now with numbers.
- **A7 flips** (`flip_tables.json`, descriptive; denominators are raw
  wrra-question pairs, not the analyzer's survival-conditional set): a2 vs
  baseline +63/−41 (spread over all four live scenarios; `finance` is net
  negative 18/25); a2 vs a1 +62/−60 (net ~2, matching the null primary).

## §8.1 scorecard

| condition | status |
|---|---|
| 1 improves reproduced fair baseline | point estimate yes (vector +0.0345) |
| 2 matched units | yes (290 pairs) |
| 3 CI reported | yes — **includes zero** |
| 4 consistent direction across replicates | vector 3/3, kv 2/3 |
| 5 not one scenario | yes (flips spread; finance net-negative noted) |
| 6 no backend harmed | yes |
| 7 survives equal-compute control | **no** — a2 vs a1 ≈ null |
| 8 survives self-consistency control | n/a (a2 *is* the self-consistency arm) |
| 9 no eval-only labels | yes |
| 10 reproducible | yes (manifest, seeds, commit `cc5dca7`) |

Conditions 3 and 7 fail → not validated. Vector direction consistency + exact
falsifier-to-live agreement → not rejected either. **Promising but inconclusive;
underpowered; gain attributable mostly to generic extra sampling rather than
majority-vote content selection.**
