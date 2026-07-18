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

---

## 3B live refinement (2026-07-18 harvest, `gov_logs/harvest/rep01_harvest/governance_log.jsonl`)

Source: Plan 2 Step 3 full-cascade shadow harvest (`run_id govrep_20260718T104933Z`, manifest
`result_gov_harvest/manifest.jsonl`), 305 decisions, 32 Stage-2 escalations, 22 read_gate
shadow events; surviving chains 4/5 per backend (student dead — pre-registered excluded).

### Stage 2 margin (`GOV_S2_MARGIN`) — live distribution
Live escalation `min_margin`: p10=0.000, p25=0.0026, **p50=0.0050**, p75=0.0088.
The 0.05 placeholder sits far above the whole distribution → 30/32 escalations ended
`stage2_accept_low_confidence` (canonicalize attempted on essentially every escalation).
**Chosen for the A/B: `GOV_S2_MARGIN=0.01` (≈ live p75)** — the top quartile accepts without
canonicalization, the ambiguous bulk still canonicalizes. Pooled, log-geometry-only; accuracy
never consulted.

### Read-time margin (`GOV_READ_MARGIN`) — live shadow read_gate events
- vector (n=16): min=0.0014, p25=0.0109, p50=0.0333, p75=0.1498, max=0.2821 — the offline
  suggestion **0.0537** lies between p50 and p75 and stands for the read-time ablation arm.
- kv (n=4): every margin exactly 0.0 — degenerate, confirming the offline note. A KV margin
  threshold cannot separate anything; the KV read-arm relies on the bounded ambiguous set
  (`GOV_READ_SET_MAX`) + single refinement round instead. Recorded as a per-backend split
  decision: `GOV_READ_MARGIN=0.0537` is a *vector* threshold.
- Read gate stays **OFF** in the headline A/B (Plan 3 mechanisms are one-at-a-time ablation
  arms; R4).

### §8.4 binding verdict (cross-reference)
`gov_logs/geometry_dH_verdict.json`: vector ρ(sim_max, ΔH)=0.757, CI95=[0.482, 0.899] ≥ 0.40
→ **geometry-first confirmed (binding)**; kv n=4 (ρ=0.8, CI degenerate) but KV is
canonical-key-first regardless. No NLI-first reorder.
