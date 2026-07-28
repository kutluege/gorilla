# H-Nav Stage 1 -- corrected labels + coupling falsifier

**Verdict: PIVOT to action-side H_act** -- write-side headroom is not there: the deployable NOOP region is too small and too contaminated with unique carriers for any suppression policy to be net-positive on this benchmark. Note the hypothesised defect is REAL and large -- on the 220 near-duplicate updates the whole blob scores 0.944 while its marginal content scores 0.196 -- but NONE of them is must_write. The losses sit on fresh adds, where there is no predecessor and so no marginal diff to measure. Fixing the geometry defect cannot recover accuracy that was never at risk

## 1. Provenance

| | |
|---|---|
| Git head | `cb8f76b700f7b2ce4fb366f161091309621759ba` |
| Decisions labeled | 4575 (positives: 159) |
| Target | `must_write` |
| Arms | rep01_geometry_only, rep01_gm_v1, rep01_joint_entropy_diag, rep01_v1_shadow, rep02_geometry_only, rep02_gm_v1, rep02_joint_entropy_diag, rep02_v1_shadow, rep03_geometry_only, rep03_gm_v1, rep03_joint_entropy_diag, rep03_v1_shadow |
| Seed / bootstrap | 12345 / 2000 |
| Proxy conversion factor p_hat | 0.408544 |

## 2. Label distribution

| target | n |
|---|---|
| inert_superseded | 3905 |
| may_suppress | 397 |
| must_suppress | 41 |
| must_write | 159 |
| uncertain | 73 |

| fate | n |
|---|---|
| gone | 2152 |
| superseded | 1753 |
| terminal | 665 |
| unresolved | 5 |

### Where the headroom actually sits

A memory chain rewrites its own items, so only the *terminal* write per store ref reaches the snapshot the questions are answered against. Everything else is first-order inert.

| fate / op family | n | must_write | rate |
|---|---|---|---|
| terminal / add | 555 | 150 | 0.2703 |
| terminal / update_replace | 110 | 9 | 0.0818 |
| superseded / add | 348 | 0 | 0.0000 |
| superseded / update_replace | 1405 | 0 | 0.0000 |
| gone / add | 2006 | 0 | 0.0000 |
| gone / update_replace | 146 | 0 | 0.0000 |
| unresolved / add | 2 | 0 | 0.0000 |
| unresolved / update_replace | 3 | 0 | 0.0000 |

### The mechanical claim that motivated H-Nav

On the 220 near-duplicate update/replace decisions (`sim_max >= 0.90`), whole-blob similarity says *duplicate* while the marginal content does not: median `sim_max` 0.9438 vs median `diff_sim_max` 0.1962, a median drop of 0.7454, with 0.9091 of rows dropping more than 0.10. **The mechanism is real and large.** But **none** of those decisions is `must_write`. The write-side losses sit on fresh adds -- where there is no predecessor, hence no marginal diff to measure -- so measuring the defect better cannot recover accuracy that was never at risk.

## 3. H1 -- write-side headroom

**FAIL** -- no deployable cell meets coverage>=0.05, precision>=0.9 (Wilson lo>=0.8) and harmful Wilson hi<=0.05

Widest deployable cell (the gate-faithful region, i.e. what suppression could actually reach):

| | |
|---|---|
| cell | `sim_high=0.800\|delta=0.60\|veto=1\|preflight=1` |
| coverage of all writes | 0.0264 (bar 0.05) |
| correct-NOOP precision | 0.9587 Wilson95 [0.9069, 0.9822] (bar 0.9 / lo 0.8) |
| harmful suppressions | 5/121 Wilson95 [0.0178, 0.0931] (bar hi <= 0.05) |
| expected accuracy change | -0.0105 |

## 4. H2 -- does the marginal diff carry signal geometry lacks?

**PASS**

| nest | AUC | PR-AUC |
|---|---|---|
| geometry | 0.5243 | 0.0343 |
| geometry+margin_entropy | 0.5269 | 0.0343 |
| geometry+margin_entropy+diff | 0.6027 | 0.0414 |
| diff_only | 0.4848 | 0.0308 |

Base rate (PR-AUC no-skill line): 0.0348; PR-AUC gain from the diff features: 0.0071

- real `dAUC_diff_given_geometry_margin`: mean 0.0764 CI95 [0.0147, 0.1382]
- NC-shuffled `dAUC_diff_given_geometry_margin`: mean -0.0058 CI95 [-0.0205, 0.0089]

**Robustness -- is this just the add/update indicator?** `must_write` concentrates in fresh adds, and `has_old` separates adds from updates perfectly, so the gain is re-measured with that column removed:

- `dAUC_has_old_alone`: mean 0.0455 CI95 [0.0098, 0.0818]
- `dAUC_diff_content_only`: mean 0.0713 CI95 [0.0105, 0.1321]

The NC control permutes the diff features jointly within (backend, store-size bin), preserving their marginals and every geometry/margin feature. It must show no significant POSITIVE gain; it is expected to sit below zero, because 17 shuffled columns cost out-of-fold AUC rather than leaving it unchanged.

## 5. H3 -- is the falsifier cheap?

**FAIL** -- false_override_rate at tau=0.5 = 0.663793 (bar 0.2)

## 6. Breakdowns (univariate falsifier AUC, `diff_sim_max` scored in the hypothesized direction)

**backend**

| group | n | positives | AUC |
|---|---|---|---|
| kv | 2052 | 48 | 0.3356 |
| vector | 2523 | 111 | 0.5633 |

**op_family**

| group | n | positives | AUC |
|---|---|---|---|
| add | 2911 | 150 | 0.5286 |
| update_replace | 1664 | 9 | 0.5760 |

**fate**

| group | n | positives | AUC |
|---|---|---|---|
| None | 5 | 0 | n/a |
| gone | 2152 | 0 | n/a |
| superseded | 1753 | 0 | n/a |
| terminal | 665 | 159 | 0.5067 |

**item_length**

| group | n | positives | AUC |
|---|---|---|---|
| long | 2277 | 17 | 0.3704 |
| short | 2298 | 142 | 0.5179 |

**scenario_customer_vs_other**

| group | n | positives | AUC |
|---|---|---|---|
| customer | 2142 | 40 | 0.3430 |
| other | 2433 | 119 | 0.4909 |

**replicate**

| group | n | positives | AUC |
|---|---|---|---|
| rep01_geometry_only | 297 | 10 | 0.5012 |
| rep01_gm_v1 | 459 | 14 | 0.5754 |
| rep01_joint_entropy_diag | 452 | 11 | 0.5266 |
| rep01_v1_shadow | 315 | 18 | 0.4934 |
| rep02_geometry_only | 388 | 12 | 0.4864 |
| rep02_gm_v1 | 314 | 19 | 0.4758 |
| rep02_joint_entropy_diag | 442 | 14 | 0.4973 |
| rep02_v1_shadow | 378 | 11 | 0.4969 |
| rep03_geometry_only | 401 | 12 | 0.5337 |
| rep03_gm_v1 | 331 | 10 | 0.4858 |
| rep03_joint_entropy_diag | 444 | 17 | 0.4451 |
| rep03_v1_shadow | 354 | 11 | 0.7497 |

## 7. Reality check: the suppressions that really happened

The only place the first-order counterfactual meets reality. n = 30 across the live arms -- a sanity check, never a rate estimate.

| | |
|---|---|
| live suppressions | 30 |
| by arm | {'geometry_only': 5, 'gm_v1': 5, 'joint_entropy_diag': 20} |
| label verdict | {'inert_superseded': 19, 'may_suppress': 8, 'must_write': 3} |
| harmful_noop | 3 |

## 8. Limitations (binding)

- First-order counterfactual: reverting a write from the final snapshot does not simulate how the agent would have behaved after seeing a synthetic success. Non-terminal writes are labeled inert_superseded on that basis; must_write_lineage bounds the under-count from above.
- Answerability (top-3 retrieval of a gold-carrying item) is a NECESSARY not sufficient condition for a correct answer; event counts are converted to expected accuracy with the measured p_hat, never 1:1.
- Operating-point cells carry small counts: Wilson intervals only, no multivariate fit below 25 positives, Holm-corrected permutation p-values instead.
- Three replicates support a min/max stability check, not an interval.
