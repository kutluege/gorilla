# Memory Information Gain (MIG) Reranker — Implementation & Usage Guide

This document describes the MIG reranker that was added to BFCL V4 Memory, how it is wired
into the inference pipeline, how to run each experiment arm, and how to read the traces.

It implements the plan in the feasibility report: a **non-invasive** reranker that sits at
the **tool-result boundary**, touching zero benchmark data, zero memory-backend code, and
zero scoring code.

---

## 1. What was added

| File | Role | Status |
|------|------|--------|
| `bfcl_eval/model_handler/middleware/__init__.py` | New middleware package | New |
| `bfcl_eval/model_handler/middleware/mig_reranker.py` | `MIGReranker` + `MIGConfig` — pool widening, candidate parsing, scorers (similarity / judge / logprob), flat + greedy selection, tracing | New |
| `bfcl_eval/model_handler/local_inference/qwen_mig.py` | `QwenMIGHandler(QwenFCHandler)` — the two interception hooks + per-test state | New |
| `bfcl_eval/constants/model_config.py` | Registered `Qwen/Qwen3-4B-Instruct-2507-FC-MIG` (import + map entry) | Edited (registry only) |
| `bfcl_eval/constants/supported_models.py` | Added the id to `SUPPORTED_MODELS` | Edited (registry only) |

**Untouched:** all `data/`, all `func_source_code/memory_*.py`, all `eval_checker/` scoring.
The baseline id `Qwen/Qwen3-4B-Instruct-2507-FC` is byte-identical, so A/B comparison is clean.

---

## 2. How it hooks in

BFCL memory tasks run through the `@final` `inference_multi_turn_prompting` loop. The agent
itself emits memory-retrieval tool calls; BFCL executes them and injects the JSON result back
into the chat history. The reranker overrides exactly three (non-`@final`) methods:

1. **`decode_execute`** — after decoding the model's tool calls, *widen the retrieval pool*
   by rewriting `top_k` (vector) / `k` (kv key-search) up to `pool_size`. Only memory
   retrieval calls with a `query` argument are touched; everything else is a no-op. The kv
   exact-key `*_retrieve(key)` is left alone (it has no pool parameter).

2. **`_add_execution_results_prompting`** — parse the retrieved candidates out of the JSON
   result, score & trim them to `final_budget`, and reserialize in the exact same schema
   before they enter the chat history. The original (full, widened) result list is left
   untouched, so the verbose inference log still records the entire pool for offline
   gold-tracking / recall@k analysis.

3. **`_pre_query_processing_prompting`** — stash per-test state inside `inference_data`
   (one dict per test entry → thread-safe under `--num-threads > 1`). MIG only activates for
   memory categories; for everything else the handler is identical to `QwenFCHandler`.

### Backend handling (detected from the result *shape*, not the function name)

| Backend | Retrieval result | Reranked? |
|---------|------------------|-----------|
| `memory_vector` | `{"result": [{id, similarity_score, text}, …]}` | **Yes** — full support (primary target) |
| `memory_kv` | `{"ranked_results": [[score, key], …]}` from `*_key_search` | Yes — but keys are weak signal |
| `memory_kv` | `{"value": …}` from exact-key `*_retrieve` | No — pass-through (no candidate list) |
| `memory_rec_sum` | `{"memory_content": …}` whole blob | No — pass-through (chunking is future work) |

> **Note:** the live vector backend returns the singular key `"result"` (the tool *doc* says
> `"results"`). The parser accepts both. Verified in `func_source_code/memory_vector.py`.

The reranker **never returns an empty tool result** — if nothing scores above threshold it
keeps the single best candidate, so the agent is never stranded.

---

## 3. Configuration (all via `MIG_*` environment variables)

A single registry entry covers every arm; you switch arms with env vars.

| Env var | Default | Meaning |
|---------|---------|---------|
| `MIG_ENABLED` | `1` | Master switch. `0` → behaves like plain `-FC`. |
| `MIG_POOL_SIZE` | `20` | Widen retrieval `top_k`/`k` to this. |
| `MIG_FINAL_BUDGET` | `5` | Candidates kept after rerank. |
| `MIG_SCORER` | `judge` | `similarity` \| `judge` \| `logprob`. |
| `MIG_MODE` | `flat` | `flat` \| `greedy`. |
| `MIG_RERANK` | `1` | `0` → widen pool only, no rerank (ablation arm 2). |
| `MIG_HALT_TAU` | `0.0` | Greedy: stop when best marginal gain < this. |
| `MIG_MIN_GAIN` | `-inf` | Flat: drop candidates scoring below this (≥1 always kept). |
| `MIG_DRAFT` | `1` | Generate a draft answer to condition scoring (the MIG `a_hat`). |
| `MIG_MAX_DRAFT_TOKENS` | `96` | Draft length cap. |
| `MIG_DRAFT_TEMPERATURE` | `0.0` | Draft sampling temp. |
| `MIG_JUDGE_TEMPERATURE` | `0.0` | Judge sampling temp. |
| `MIG_LOG_DIR` | unset | If set, write per-test JSONL traces here. Off by default. |
| `MIG_VERBOSE` | `1` | One-line stdout summary per interception. |

> **Determinism:** vLLM clamps temperature to ≥ ~0.01 in some builds; keep temps at 0 and
> expect a tiny clamp, as noted in your existing rebenchmark guide.

---

## 4. Scorers

- **`similarity`** — identity trim; keep the backend's own ranking. Zero extra model calls.
  Used for the "widened pool, no rerank quality" arm (equivalent to `MIG_RERANK=0` but it
  still re-sorts by prior score).
- **`judge`** (v1 default) — an LLM utility judge rates each candidate 0–5 for how useful it
  is to answer the question, conditioned on the current draft answer (information-gain
  framing). Backend-agnostic, no token-offset arithmetic. ~`pool_size` judge calls + 1 draft
  per interception (flat); more in greedy.
- **`logprob`** (v2, **local vLLM only**) — exact information gain
  `MIG(m) = logp(a_hat | S∪{m}) − logp(a_hat | S)` via the Completions
  `echo=True, logprobs=1, max_tokens=0` trick. If the endpoint does not return prompt-token
  logprobs, the scorer detects this on the first call and **gracefully degrades to
  `similarity`** (logged once). Requires `MIG_DRAFT=1`.

---

## 5. Running the experiment arms

Prereq: a local vLLM server for `Qwen/Qwen3-4B-Instruct-2507` (the `-FC-MIG` id reuses the
same `model_name`, so the same served model works). Run the baseline from a clean checkout
per `BFCL_PURE_REBENCHMARK_GUIDE.md`.

```bash
# Arm 1 — Baseline (control, unmodified handler). Separate id, no MIG.
bfcl generate --model Qwen/Qwen3-4B-Instruct-2507-FC \
  --test-category memory_vector --num-threads 1 --include-input-log

# Arm 2 — Widened pool, NO rerank (isolates "more candidates" from "better ranking")
MIG_RERANK=0 MIG_POOL_SIZE=20 \
bfcl generate --model Qwen/Qwen3-4B-Instruct-2507-FC-MIG \
  --test-category memory_vector --num-threads 1 --include-input-log

# Arm 3 — Flat MIG (judge)
MIG_SCORER=judge MIG_MODE=flat MIG_POOL_SIZE=20 MIG_FINAL_BUDGET=5 \
MIG_LOG_DIR=./mig_traces \
bfcl generate --model Qwen/Qwen3-4B-Instruct-2507-FC-MIG \
  --test-category memory_vector --num-threads 1 --include-input-log

# Arm 4/5 — Greedy / hierarchical MIG with adaptive halt
MIG_SCORER=judge MIG_MODE=greedy MIG_HALT_TAU=2.0 MIG_FINAL_BUDGET=5 \
bfcl generate --model Qwen/Qwen3-4B-Instruct-2507-FC-MIG \
  --test-category memory_vector --num-threads 1 --include-input-log

# Arm (v2) — Exact logprob MIG (local vLLM only)
MIG_SCORER=logprob MIG_MODE=flat MIG_DRAFT=1 \
bfcl generate --model Qwen/Qwen3-4B-Instruct-2507-FC-MIG \
  --test-category memory_vector --num-threads 1 --include-input-log

# Pool-size sweep
for K in 10 20 50; do
  MIG_POOL_SIZE=$K MIG_SCORER=judge \
  bfcl generate --model Qwen/Qwen3-4B-Instruct-2507-FC-MIG \
    --test-category memory_vector --num-threads 1
done

# Score every arm with the unchanged official scorer
bfcl evaluate --model Qwen/Qwen3-4B-Instruct-2507-FC-MIG --test-category memory_vector
```

Backends: `memory_vector` (primary), `memory_kv` (secondary; reranks key-search keys),
`memory_rec_sum` (pass-through / no-op until chunking is implemented).

---

## 6. Tracing & offline analysis

With `MIG_LOG_DIR` set, each interception appends a JSON line to
`<MIG_LOG_DIR>/<test_id>.jsonl`:

```json
{"interception": 1, "question": "...", "fn_call": "archival_memory_retrieve(...)",
 "kind": "vector", "pool_size": 20, "kept_size": 5,
 "trace": {"note": "flat", "draft": "...", "scores": [...], "kept": [...]}}
```

Gold tracking (recall@k / selection precision) is computed offline by matching the
`source` gold sentence in `data/possible_answer/BFCL_v4_memory.json` against the candidate /
selected texts. Because the full widened pool is preserved in the standard inference log
(`--include-input-log`), you can measure both `gold_in_pool` (retrieval ceiling) and
`gold_selected` (reranker precision).

---

## 6b. Experiment findings (live Qwen3-4B, 2026-06-27)

A controlled live experiment (`bfcl_eval/scripts/mig_live_experiment.py`, full results in
`BFCL_MIG_EXPERIMENT_RESULTS.md`) on real memory data confirms the method works:

- **`scorer=judge` is the best arm in every cell** — gold-selection recall@k = **1.000**
  across scenarios/budgets vs 0.80–0.97 for the similarity baseline; answer accuracy at or
  above the noisy full pool, and well above similarity-trim under hard negatives / tight budgets.
- The experiment caught and fixed a **real bug**: flat `scorer=logprob` scored 0.000 because it
  drafted `a_hat` from empty context (wrong prior guess → gold contradicts it → ranked last).
  The fix — **`draft_from_pool=True`** (now default, env `MIG_DRAFT_FROM_POOL`) — restores
  recall to 0.88–1.00. A `None`-logprob poisoning bug was fixed alongside.
- **Recommendation:** use `scorer=judge` as the default. The reranker needs **no fine-tuning**
  (judge selection is already ~perfect); if you fine-tune, spend a LoRA on *answer
  generation + format compliance*, which is where the residual ~10–15% error lives. See
  `BFCL_MIG_EXPERIMENT_RESULTS.md` §5.

## 7. Validation status

- **Offline unit tests** (no GPU / server) — pool widening, multi-backend candidate parsing,
  reserialization round-trip, question extraction, `select()` fast paths, and a fake-client
  judge+flat path: **27/27 pass**. See the offline test under the session scratchpad.
- **Source compiles** — all new files and the edited registry files compile.
- **Pending (requires your env):** a live smoke test on a handful of `memory_vector` answer
  ids with `--num-threads 1 --include-input-log`, then inspect `mig_traces/*.jsonl`. The
  full registry import needs the standard BFCL deps (`anthropic`, `tree_sitter`, `openai`,
  `faiss`, `sentence-transformers`), which are not present in a minimal Python env.

---

## 8. Known limitations / future work

- `memory_rec_sum` is pass-through (no candidate list); chunking the blob into pseudo-
  candidates is the future extension.
- `memory_kv` reranks *keys* (BM25 key-search output); value-level MIG would need to also
  fetch values, which the handler can do by issuing extra retrieves but is not implemented in v1.
- `logprob` scorer requires the local vLLM Completions `echo`/`logprobs` path; hosted chat
  APIs do not expose prompt-token logprobs and will fall back to `similarity`.
- Exact token-offset alignment of the draft span under the Qwen chat template is the main
  thing to verify when first enabling the `logprob` scorer.
