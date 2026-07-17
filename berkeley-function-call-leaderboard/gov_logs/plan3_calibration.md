# Plan 3 Step 6 -- G10 calibration note (offline instruments)

Source: `Qwen_Qwen3-4B-Instruct-2507-FC` committed trees, surviving chains ['kv/customer', 'kv/finance', 'kv/notetaker', 'vector/customer', 'vector/finance', 'vector/notetaker'].

## w1 grid (eviction simulation, answer-loss; lower = better)

| w1 | answer-bearing evicted | total | answer_loss |
|---|---|---|---|
| 0.4 | 18 | 51 | 0.3529 |
| 0.5 | 18 | 51 | 0.3529 |
| 0.6 | 18 | 51 | 0.3529 |
| 0.7 | 18 | 51 | 0.3529 |

## k sensitivity (related-coverage of the Stage-1 candidate set)

| k | mean coverage | cases |
|---|---|---|
| 3 | 0.9566 | 34 |
| 5 | 0.984 | 34 |
| 7 | 0.9893 | 34 |
| 10 | 0.9973 | 34 |

## read-time margin (real-question margins over snapshots)

- **kv**: answer-bearing top-1 n=10 (p25=0.0, p50=0.0); other top-1 n=70 (p50=0.0); **suggested GOV_READ_MARGIN = 0.0** (provisional)
- **vector**: answer-bearing top-1 n=27 (p25=0.0537, p50=0.1153); other top-1 n=53 (p50=0.0569); **suggested GOV_READ_MARGIN = 0.0537** (provisional)

Notes: pooled calibration over surviving scenarios; report per-scenario in 3B; wgrid isolates uniqueness (snapshot turn_written=-1 flattens recency); readmargin suggestion is provisional until the live shadow harvest.
