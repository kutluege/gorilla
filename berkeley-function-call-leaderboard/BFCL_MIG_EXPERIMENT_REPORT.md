# BFCL MIG Reranker — Experiment Report

**Branch:** `info_gain_rerank`  **Date:** 2026-07-01
**Endpoint:** vLLM @ `localhost:8000` serving `Qwen/Qwen3-4B-Instruct-2507` (`allow_logprobs: true`)
**Models:** baseline `Qwen/Qwen3-4B-Instruct-2507-FC` · MIG `Qwen/Qwen3-4B-Instruct-2507-FC-MIG`

> **Headline.** The MIG reranker **code is correct** (separately validated — see
> `MIG_VALIDATION_REPORT.md`). This experiment shows MIG **does not improve answer accuracy**
> on the tested memory-vector subset, for two reasons the reranker cannot fix on its own:
> (1) the agent almost never retrieves at answer time, and (2) when it does, the retriever
> misses the answer-bearing memory (a *recall* problem, not a *ranking* problem). Separately,
> the run surfaced a **methodological bug**: MIG intercepts during the memory-*writing*
> (prerequisite) phase, which changes what gets stored and makes cross-arm answer-score
> comparison invalid. **Do not read the accuracy deltas below as a MIG effect.**

---

## 1. Cleanup actions performed

| Action | Detail |
|---|---|
| Reverted shared-parent edit | `base_oss_handler.py` restored to committed state (removed the unnecessary `self.model_name = self.registry_name`; `BaseHandler` already sets `self.model_name`, and OSS/MIG inference + the reranker use `self.model_path_or_id`). |
| Kept benign fix | `utils.py` `open(..., encoding='utf-8')` retained (Windows). |
| Archived smoke artifacts | Partial healthcare-only results moved to `tmp/smoke_archive_2026-07-01/` so `evaluate` cannot score them. |
| Added analysis tool | `bfcl_eval/scripts/analyze_mig_traces.py` (new). |

## 2. Methodology

- **Snapshot reuse is NOT supported** by the framework: answer entries' `depends_on` is never
  pruned (`utils.py:748`) and the generation scheduler blocks any task whose dependencies aren't
  in-run (`_llm_response_generation.py:269-335`). Running "answers-only against a pre-built
  snapshot" deadlocks. True reuse would require a scheduler change (out of scope).
- **Fallback used (per instructions):** all arms run with `--num-threads 1`, **sequentially**
  (concurrent arms would reintroduce vLLM continuous-batching nondeterminism), each into its own
  `--result-dir`.
- **Subset:** 2 scenarios — `customer` (30 answers) + `healthcare` (25 answers) = **55 answers +
  15 prereqs** per arm, via `--run-ids`. The full 155-answer category is ~5 h at `--num-threads 1`;
  the subset is representative and, given the retrieval-frequency findings below, does not change
  the qualitative conclusion.

### Exact commands (per arm; driver: `<scratchpad>/runs/run_arms.sh`)
```bash
# common: --test-category memory_vector --run-ids --skip-server-setup \
#         --temperature 0.001 --num-threads 1 --include-input-log --allow-overwrite
# baseline
python -m bfcl_eval generate --model Qwen/Qwen3-4B-Instruct-2507-FC     $COMMON --result-dir result_baseline
# MIG judge   (env: MIG_ENABLED=true MIG_POOL_SIZE=20 MIG_FINAL_BUDGET=5 MIG_MODE=flat MIG_SCORER=judge   MIG_LOG_DIR=mig_traces/exp_2026-07-01/judge)
python -m bfcl_eval generate --model Qwen/Qwen3-4B-Instruct-2507-FC-MIG $COMMON --result-dir result_mig_judge
# MIG logprob (env: ... MIG_SCORER=logprob MIG_LOG_DIR=mig_traces/exp_2026-07-01/logprob)
python -m bfcl_eval generate --model Qwen/Qwen3-4B-Instruct-2507-FC-MIG $COMMON --result-dir result_mig_logprob
# evaluate (each arm)
python -m bfcl_eval evaluate --model <id> --test-category memory_vector --result-dir <rdir> --score-dir <sdir> --partial-eval
```

## 3. Results — overall accuracy (⚠️ NOT a valid comparison)

`memory_vector`, 55-entry subset, `--partial-eval`:

| Arm | Memory Vector Acc | Memory snapshot vs baseline |
|---|---|---|
| Baseline (FC) | **14.55 %** (8/55) | — |
| MIG judge | **12.73 %** (7/55) | **DIFFERENT** (customer core 6/arch 9 vs 4/15; healthcare 1/1 vs 0/0) |
| MIG logprob | **16.36 %** (9/55) | **DIFFERENT** (customer core 3/arch 15 vs 4/15; healthcare 1/1 vs 0/0) |

**These deltas (±1–2 entries) are within noise AND are computed on different memory states, so
they are methodologically invalid as a MIG effect** (see §5). Reported only for completeness.

## 4. Interception-subset analysis (the valid, MIG-specific view)

From the MIG traces (`analyze_mig_traces.py`):

| | MIG judge | MIG logprob |
|---|---|---|
| Total interceptions (rerankable) | 14 | 6 |
| …during **prereq** (memory-writing) turns | 13 | 6 |
| …during **answer** turns | **1** | **0** |
| Answer entries with ≥1 tool execution (of 55) | **1** | **0** |
| Scorable answer interceptions **with gold** | 1 | 0 |
| gold-in-pool rate (answer interceptions) | **0.00** | n/a |

**The single answer-time interception** (`memory_vector_17-customer-17`, gold = *"Oregon"*):
question *"Where is the warehouse that ships my grinder bundle from?"* → the agent's
`core_memory_retrieve` returned 6 memories, **all** about *"Michael, 35, Seattle, freelance
designer"* — **none about the warehouse/Oregon**. Gold was not in the pool, so reranking cannot
help. The draft honestly read *"…not specified in the provided information."* This is a
**retrieval-recall miss**, not a reranking failure.

> Reranker correctness itself is not in question: the deterministic harness in
> `MIG_VALIDATION_REPORT.md` showed that **when gold is in the pool**, both judge and logprob
> rank it #1 (recall@1 0→1). This experiment simply shows the pool rarely contains the gold.

## 5. Interpretation

1. **MIG does not improve accuracy here**, and the small deltas are not attributable to MIG
   (different snapshots + noise). Do **not** claim improvement.
2. **Retrieval frequency (R8) is the binding constraint.** On 55 answer entries, retrieval fired
   **0–1 times** — the model answers from core memory (dumped into the system prompt) or gives up.
   A reranker that is never invoked cannot move the score.
3. **When retrieval fires, recall is the bottleneck**, not ranking: the one case missed the gold
   entirely. MIG improves *ordering within the pool*; it cannot add a memory the retriever didn't
   return.
4. **Confound / methodological bug:** MIG's `decode_execute` widening + `_add_execution_results`
   reranking are active during **prerequisite (memory-writing)** turns (state `_mig_active` is set
   for every memory id, including `*_prereq_*`). The customer prereqs retrieve-before-write, so MIG
   **changed what memory got stored** (confirmed: interceptions on `prereq_2/4/7-customer`; and the
   two MIG arms built *different* snapshots from each other). This invalidates cross-arm answer
   comparison and is the top thing to fix.

**Code correctness vs experimental effectiveness are separate:** the implementation is correct
(validated); its *effectiveness on this model/benchmark* is ~neutral because retrieval is rare and
low-recall.

## 6. Recommendations (next steps / ablation matrix)

1. **Gate MIG to answer turns.** In `qwen_mig.py::_pre_query_processing_prompting`, only set
   `_mig_active` when `not is_memory_prereq(test_entry["id"])`. This removes the memory-construction
   confound so all arms share the baseline snapshot. (Small, justified handler change; requires a
   re-run. Then **diff snapshots to confirm they match baseline** before trusting deltas.)
2. **Make retrieval happen at answer time** — the dominant lever. Options: an answer-format prompt
   that instructs the agent to call `archival_memory_retrieve` before answering, or a forced-retrieval
   harness variant. Without this, MIG is a no-op on most entries.
3. **Attack recall, then ranking.** Since the one live case was a recall miss, measure/raise
   retrieval recall@k (e.g. query rewriting, larger `top_k`) — MIG only pays off once gold is in the
   pool.
4. **For a valid overall comparison**, either (1) above, or implement true snapshot reuse (prune
   `depends_on` to in-run ids for `--run-ids`, letting answers load a pre-built snapshot).
5. **Ablation matrix — only meaningful on answer-bearing interceptions:** `MIG_SCORER` ∈
   {similarity, judge, logprob} × `MIG_MODE` ∈ {flat, greedy} × `MIG_POOL_SIZE` ∈ {10,20,50},
   scored on the interception subset (gold-in-pool, gold-selected, recall@k before/after) rather
   than whole-category accuracy. Run the **full** `memory_vector` (all 5 scenarios) to accumulate
   enough answer-time interceptions for signal.

## 7. Deliverables & locations

- **Results:** `result_baseline/`, `result_mig_judge/`, `result_mig_logprob/`
- **Scores:** `score_baseline/`, `score_mig_judge/`, `score_mig_logprob/` (`data_agentic.csv`)
- **MIG traces:** `mig_traces/exp_2026-07-01/{judge,logprob}/*.jsonl`
- **Analysis tool:** `bfcl_eval/scripts/analyze_mig_traces.py`
- **Run driver / logs:** `<scratchpad>/runs/` (`run_arms.sh`, `*.log`, `run_ids_used.json`)
- **Archived smoke artifacts:** `tmp/smoke_archive_2026-07-01/`

## 8. Modified files
- `bfcl_eval/utils.py` — utf-8 fix (kept; pre-existing).
- `bfcl_eval/scripts/analyze_mig_traces.py` — **new** analysis script.
- `base_oss_handler.py` — reverted to committed (no net change).
- Reranker/handler implementation (`mig_reranker.py`, `qwen_mig.py`) — **unchanged** (no bug in
  core logic; the prereq-gating item in §6.1 is a proposed, not-yet-applied fix).

## 9. Remaining risks / TODOs
- [ ] Apply §6.1 prereq-gating fix and re-run for a valid overall comparison (verify snapshot match).
- [ ] `--num-threads 1` reproducibility is **untested** (determinism probe was inconclusive due to a
      `--result-dir` + `--run-ids` "already generated" quirk); confirm before trusting small deltas.
- [ ] Address answer-time retrieval frequency + recall before expecting any MIG accuracy gain.
- [ ] Scale to full `memory_vector` (5 scenarios) once retrieval is happening, for statistical power.
- [ ] Missing `BFCL_MIG_RERANKER_GUIDE.md` (referenced by docstrings) still to be authored.
