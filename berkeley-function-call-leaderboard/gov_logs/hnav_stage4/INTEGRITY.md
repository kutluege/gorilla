# Stage-4 (alt10 write-rescue) campaign — integrity gate

**Verified:** 2026-08-08, before any unblinding of arm contrasts.
**Campaign:** `govrep_20260729T004515Z`, `result_hnav_stage4/manifest.jsonl`.

| # | assertion | result |
|---|---|---|
| 1 | all 18 `cmd_end` rows (3 arms × 3 reps × generate/evaluate) have `exit_code == 0`; zero `error` events | **PASS** — 18/18 exit 0, 0 errors |
| 2 | `run_start.git_head == cc5dca79baf9ed46542a6e25f77a5f3cb3ccec7e`; `served_models == ["Qwen/Qwen3-4B-Instruct-2507"]`; `vllm_version == 0.9.1`; `temperature == 0.001`; `num_threads == 1` | **PASS** — all exact |
| 3 | same-server-instance: one launcher PID (2052), one continuous window (launch 2026-07-29T03:45:15+03:00 → exit 13:48:25+03:00 code=0), `start_replicate == 1` (no resume) | **PASS** — with limitation below |
| 4 | score trees exist for all 9 (rep, arm) cells; every `cmd_end[phase=evaluate].score_files[].exists == true` | **PASS** — zero missing |
| 5 | `HACT_SEED` replicate-distinct (the `945837e` pseudo-replication fix): effective seeds 20260729 / 20260730 / 20260731 for reps 1/2/3, identical across the two instrumented arms within a replicate (paired by design), absent from baseline | **PASS** |

## Recorded limitation (assertion 3)

The `/v1/models` `created` field — the server-instance identity — was **not captured**
for this run. The evidence for same-server-instance is the continuity of the single
launcher process (one PID, one uninterrupted 10 h 03 m window, exit code 0, no
`--start-replicate` resume). This is weaker than an instance timestamp and is stated
as such; all subsequent campaigns record `created` at pre-flight.

## Analysis authorization

All five gates pass → the analyzer may run **without** `--allow-unscored`. Holm
family declared before unblinding (§6.5): **primary** family = {a2_majority vs
a1_random} × {kv, vector}, m = 2. The vs-baseline contrasts are **secondary**,
Holm-corrected within their own family, labelled as such.

Interpretation constraint recorded before unblinding: A1 is an *equal-sampling-budget*
control, not an equal-intervention control — it replaces the primary action on ~90%
of sampled steps vs A2's ~4%, so A1 is a degraded agent and `a2 > a1` is partly
"majority beats random". The contrast answering §8.1 items 7–8 is **a2 vs baseline**.
Both are reported.
