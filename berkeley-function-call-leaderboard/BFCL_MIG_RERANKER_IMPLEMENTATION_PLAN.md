# BFCL Memory MIG Reranker — Implementation Plan

> **Status note (read first).** This branch (`info_gain_rerank`) **already contains a working
> MIG reranker implementation.** The core code (handler + reranker + registry) is committed
> (`9c2b7da`, `0bf8ec8`). This document therefore does double duty:
>
> 1. It is a **research/onboarding map** of the current BFCL Memory pipeline (so a second
>    agent does not need to redo the investigation), and
> 2. It is an **implementation plan** whose "Files to Add / Modify" sections describe what the
>    realized implementation looks like and what remains (running, validating, analyzing,
>    documenting).
>
> Where a section describes something that **already exists**, it is marked `[Implemented]`.
> Confirmed repository facts are marked `[Confirmed]`. Proposed/remaining work is marked
> `[Proposed]`. Unknowns are marked `[Needs verification]`.

---

## 1. Repository Findings

| Item | Value | Evidence |
|---|---|---|
| Baseline model ID | `Qwen/Qwen3-4B-Instruct-2507-FC` | [Confirmed] `constants/model_config.py:1636` |
| Baseline handler class | `QwenFCHandler` | [Confirmed] `constants/model_config.py:1642` |
| Handler file | `bfcl_eval/model_handler/local_inference/qwen_fc.py` | [Confirmed] |
| Handler parent | `QwenFCHandler(OSSHandler)` → `OSSHandler(BaseHandler)` | [Confirmed] `qwen_fc.py:10`, `base_oss_handler.py:22` |
| Inference mode | **Prompting mode** over the OpenAI **Completions** API (manual chat template, not native FC / not chat.completions) | [Confirmed] `base_oss_handler.py:319-366`, `qwen_fc.py:48-239` |
| MIG model ID | `Qwen/Qwen3-4B-Instruct-2507-FC-MIG` | [Implemented] `constants/model_config.py:1665`, `supported_models.py:143` |
| MIG handler class | `QwenMIGHandler(QwenFCHandler)` | [Implemented] `local_inference/qwen_mig.py:34` |
| MIG reranker module | `bfcl_eval/model_handler/middleware/mig_reranker.py` (`MIGConfig`, `MIGReranker`, `Candidate`) | [Implemented] |
| Endpoint config | Env vars read in `OSSHandler.__init__` | [Confirmed] `base_oss_handler.py:42-49` |
| CLI | `typer` app: `bfcl generate` / `bfcl evaluate` / `bfcl scores` (or `python -m bfcl_eval …`) | [Confirmed] `bfcl_eval/__main__.py` |
| Memory categories | `memory_kv`, `memory_vector`, `memory_rec_sum` | [Confirmed] `data/multi_turn_func_doc/memory_*.json` |
| Score outputs | `score/<model_dir>/agentic/memory/<backend>/BFCL_v4_memory_<backend>_score.json` + `score/data_agentic.csv` | [Confirmed] `eval_runner.py`, `BFCL_MEMORY_REPRODUCTION_GUIDE.md` |
| Referenced guide `BFCL_MIG_RERANKER_GUIDE.md` | **Missing** (docstrings reference it; file not present) | [Confirmed] gap — see §12 |

### Endpoint configuration (baseline and MIG share it) `[Confirmed]`

`OSSHandler.__init__` (`base_oss_handler.py:42-49`) builds the client from env vars:

```text
LOCAL_SERVER_ENDPOINT   (default: localhost)
LOCAL_SERVER_PORT       (default: from eval_config.LOCAL_SERVER_PORT)
REMOTE_OPENAI_BASE_URL  (optional; overrides host:port, e.g. https://host/v1)
REMOTE_OPENAI_API_KEY   (default: "EMPTY")
REMOTE_OPENAI_TOKENIZER_PATH (optional; tokenizer for remote endpoints)
```

`base_url = REMOTE_OPENAI_BASE_URL or http://{LOCAL_SERVER_ENDPOINT}:{LOCAL_SERVER_PORT}/v1`.
Because `QwenMIGHandler` inherits `__init__` unchanged, **the MIG variant automatically uses the
exact same BSC/local endpoint config as the baseline.** For the BSC tunnel, that is
`LOCAL_SERVER_ENDPOINT=localhost`, `LOCAL_SERVER_PORT=8000` (per `BFCL_PURE_REBENCHMARK_GUIDE.md`).

### Baseline generation / evaluation commands `[Confirmed pattern]`

From `BFCL_PURE_REBENCHMARK_GUIDE.md` (already used for the baseline run):

```powershell
# generation
python -m bfcl_eval generate `
  --model Qwen/Qwen3-4B-Instruct-2507-FC `
  --test-category memory_kv,memory_vector,memory_rec_sum `
  --skip-server-setup --temperature 0.001 --allow-overwrite

# evaluation
python -m bfcl_eval evaluate `
  --model Qwen/Qwen3-4B-Instruct-2507-FC `
  --test-category memory_kv,memory_vector,memory_rec_sum `
  --score-dir score_fresh
```

---

## 2. Current Memory Pipeline Flow `[Confirmed]`

Actual method names and files, question → score:

```text
answer entry (data/BFCL_v4_memory.json)                       # single user question, scenario, involved_classes=[MemoryAPI]
  └─ load_dataset_entry / process_memory_test_case            # utils.py:402 / 714  (splits prereq vs answer, sets depends_on)
  └─ populate_test_cases_with_predefined_functions            # utils.py:772        (attaches memory_<backend>.json tool docs)
  └─ process_agentic_test_case                                # utils.py:755        (adds agentic answer-format system prompt)
  └─ populate_initial_settings_for_memory_test_cases          # utils.py:837        (initial_config: model_result_dir, scenario, test_id)

generation (_llm_response_generation.py: dependency scheduler)
  prereq entries run first -> memory snapshot written by MemoryAPI._flush_memory_to_local_file
      -> result/<model>/agentic/memory/<backend>/memory_snapshot/<scenario>_final.json   # base_handler.py:371

answer entry inference: BaseHandler.inference (base_handler.py:68)
  └─ contain_multi_turn_interaction == True  -> OSSHandler.inference -> inference_multi_turn_prompting   # base_oss_handler.py:53, base_handler.py:393  (@final)
  └─ execute_multi_turn_func_call([], ...)   -> instantiate + _load_scenario (loads snapshot)            # base_handler.py:124, memory_api_metaclass.py:24
  └─ add_memory_instruction_system_prompt    -> core memory dumped into system prompt                    # model_handler/utils.py:610

  loop (<= MAXIMUM_STEP_LIMIT = 20):                                                                     # base_handler.py:211, default_prompts.py:1
     _query_prompting  -> client.completions.create(prompt=formatted_prompt, ...)                        # base_oss_handler.py:319
     _parse_query_response_prompting                                                                     # qwen_fc.py:250
     decode_execute    -> list of "fn(args)" strings                                                     # qwen_fc.py:36  <<< HOOK 1
     execute_multi_turn_func_call -> eval(func_call) on the memory backend instance -> JSON string       # multi_turn_utils.py:13,83
     _add_execution_results_prompting -> append {"role":"tool", ...} to chat history                     # base_oss_handler.py:411  <<< HOOK 2
     (break when model emits a non-decodable final answer)

scoring: bfcl evaluate -> agentic_runner -> _evaluate_single_agentic_entry                              # eval_runner.py:501, 72
  └─ take the LAST step that fails to decode as a tool call = final NL answer                            # eval_runner.py:104-148
  └─ agentic_checker: word-boundary substring match of any ground_truth after standardize_string         # agentic_checker.py:6,41
```

**Key architectural fact:** retrieval is **agent-driven via tool calls**, not an automatic
retrieval step. The candidate memories exist only as the **JSON return value of retrieval tool
calls**, executed in `execute_multi_turn_func_call` and injected by
`_add_execution_results_prompting`. That boundary is the entire integration surface.

---

## 3. Recommended Hook Point `[Confirmed / Implemented]`

Two overridable methods on the handler (the multi-turn loop itself is `@final` and is **not**
touched):

| Hook | Method | File | Purpose |
|---|---|---|---|
| **HOOK 1 (before execution)** | `decode_execute` | `qwen_mig.py:81` | Widen the retrieval pool: rewrite `top_k`/`k` up to `MIG_POOL_SIZE` on retrieval calls. |
| **HOOK 2 (after execution)** | `_add_execution_results_prompting` | `qwen_mig.py:100` | Parse the retrieved candidate list, rerank/trim to `MIG_FINAL_BUDGET`, reserialize before it enters chat history. |
| (support) | `_pre_query_processing_prompting` | `qwen_mig.py:70` | Stash per-test MIG state in `inference_data` (fresh per entry ⇒ thread-safe). |

**Why this is safer than modifying memory backend code:**

- The handler only ever sees **tool-result JSON strings**, the **decoded tool-call strings**, and
  the **chat history** — never the backend internals. Reranking at this layer is backend-agnostic
  and leaves `func_source_code/memory_*.py` byte-identical. `[Confirmed]`
- The handler has **no reference to the backend instance** (it lives in `multi_turn_utils`
  module globals, not passed to the handler). So pool expansion *must* be done by rewriting the
  model's tool-call arguments in `decode_execute`, not by calling the backend directly. This is
  exactly what `MIGReranker.widen_pool` does. `[Confirmed constraint]`
- The baseline model id keeps its own handler; the MIG variant is a **separate registry id**, so
  A/B comparison is clean and the official scorer is unchanged.

---

## 4. Files to Add `[Implemented]`

| File | Purpose | Status |
|---|---|---|
| `bfcl_eval/model_handler/middleware/__init__.py` | Package marker for the middleware namespace. | [Implemented] |
| `bfcl_eval/model_handler/middleware/mig_reranker.py` | `MIGConfig` (env-driven), `Candidate`, `MIGReranker` (pool widening, candidate parsing, flat/greedy selection, judge/logprob/similarity scorers, JSONL tracing). Backend-agnostic; holds no benchmark state. | [Implemented] |
| `bfcl_eval/model_handler/local_inference/qwen_mig.py` | `QwenMIGHandler(QwenFCHandler)`: overrides `decode_execute`, `_add_execution_results_prompting`, `_pre_query_processing_prompting`. Pass-through for non-memory categories. | [Implemented] |
| `BFCL_MIG_RERANKER_GUIDE.md` | Referenced by module docstrings but **not present**. Should hold the run recipe + experiment findings. | **[Proposed / remaining]** |

---

## 5. Files to Modify `[Implemented]`

| File | Why | What was added | Must NOT change |
|---|---|---|---|
| `bfcl_eval/constants/supported_models.py` | Register the new id so the CLI accepts it. | `"Qwen/Qwen3-4B-Instruct-2507-FC-MIG"` at line 143. | The baseline `-FC` entry (line 142). |
| `bfcl_eval/constants/model_config.py` | Route the new id to `QwenMIGHandler`. | `ModelConfig(...)` block at lines 1665+ (`model_handler=QwenMIGHandler`, `is_fc_model=True`, `model_name="Qwen/Qwen3-4B-Instruct-2507"`) + import of `QwenMIGHandler`. | The baseline `-FC` `ModelConfig` (lines 1636-1647). |

**No changes** were made (or should be made) to `data/`, `func_source_code/memory_*.py`,
`eval_checker/`, or `agentic_eval/`. `[Confirmed]`

> `[Needs verification]` Confirm `model_config.py` imports `QwenMIGHandler`
> (`from bfcl_eval.model_handler.local_inference.qwen_mig import QwenMIGHandler`). The registry
> block references the class; the import line should be checked to exist near the other handler
> imports (~line 61).

---

## 6. New Model ID and Registry Plan `[Implemented]`

| Field | Value |
|---|---|
| Proposed MIG model ID | `Qwen/Qwen3-4B-Instruct-2507-FC-MIG` |
| Mirrors baseline | `Qwen/Qwen3-4B-Instruct-2507-FC` |
| `model_name` (served checkpoint) | `Qwen/Qwen3-4B-Instruct-2507` (same served weights as baseline) |
| Handler | `QwenMIGHandler` |
| Endpoint config | Inherited unchanged from `OSSHandler.__init__` ⇒ same BSC/local endpoint as baseline. |

Because `model_name` is identical to the baseline and `__init__` is inherited, the MIG variant
hits the **same vLLM server / same `model_path_or_id`**. The only behavioral difference is the two
overridden hooks. All ablation behavior is chosen via `MIG_*` env vars, so **one registry entry
covers every experiment arm** (no per-arm code edits).

---

## 7. MIG Reranker Design (`MIGReranker`) `[Implemented]`

**Constructor:** `MIGReranker(client, model_id, config: MIGConfig, tokenizer=None)`. Pure utility
object; per-test state is passed in by the handler (thread-safe).

**Config (`MIGConfig`, all from `MIG_*` env vars):**

| Env var | Field | Default | Meaning |
|---|---|---|---|
| `MIG_ENABLED` | `enabled` | `True` | Master switch. |
| `MIG_POOL_SIZE` | `pool_size` | `20` | Widen retrieval `top_k`/`k` up to this. |
| `MIG_FINAL_BUDGET` | `final_budget` | `5` | Candidates kept after rerank. |
| `MIG_SCORER` | `scorer` | `judge` | `similarity` \| `judge` \| `logprob`. |
| `MIG_MODE` | `mode` | `flat` | `flat` \| `greedy`. |
| `MIG_RERANK` | `rerank_enabled` | `True` | If false → pool-widen only (ablation arm). |
| `MIG_HALT_TAU` | `halt_tau` | `0.0` | Greedy adaptive halt threshold. |
| `MIG_MIN_GAIN` | `min_gain` | `-inf` | Flat: drop below-threshold (keep ≥1). |
| `MIG_DRAFT` | `use_draft` | `True` | Condition scoring on a draft answer `a_hat`. |
| `MIG_DRAFT_FROM_POOL` | `draft_from_pool` | `True` | Seed `a_hat` from the retrieved pool (see risk R1). |
| `MIG_MAX_DRAFT_TOKENS` | `max_draft_tokens` | `96` | Draft length cap. |
| `MIG_DRAFT_TEMPERATURE` / `MIG_JUDGE_TEMPERATURE` | — | `0.0` | Determinism. |
| `MIG_LOG_DIR` | `log_dir` | `None` | JSONL trace dir (per-test). |
| `MIG_VERBOSE` | `verbose` | `True` | One-line stdout per interception. |

**Candidate parsing (by result *shape*, not function name):**

| Backend / call | Result shape | `kind` | Rerankable? |
|---|---|---|---|
| vector `*_retrieve` | `{"result": [{"id","similarity_score","text"}, ...]}` (also accepts `"results"`) | `vector` | Yes |
| kv `*_key_search` | `{"ranked_results": [[score, key], ...]}` | `kv_keys` | Weak (keys only) |
| kv exact `*_retrieve` | `{"value": ...}` | `None` | Pass-through |
| rec_sum `memory_retrieve` | `{"memory_content": ...}` | `None` | Pass-through |
| errors / non-JSON | anything else | `None` | Pass-through |

**Flat algorithm (`_select_flat`):** draft `a_hat` (optionally seeded from pool) → score every
candidate once → sort desc → keep top `final_budget` above `min_gain` → **never empty** (keep
best if all filtered).

**Greedy algorithm (`_select_greedy`):** loop: re-draft (first step seeds from remaining pool) →
score remaining pool → pick argmax → halt if `best < halt_tau` or budget hit → fallback top-1 by
prior score if nothing selected.

**Scorers:**
- `similarity`: keep backend order (used for the widened-pool-no-rerank arm).
- `judge`: LLM utility judge 0–5 via `completions.create(max_tokens=8)`, parsed + clamped.
- `logprob`: exact MIG `= logp(a_hat|S+{m}) − logp(a_hat|S)` via `echo=True, logprobs=1,
  max_tokens=0`; auto-degrades to similarity if the endpoint lacks echo logprobs
  (`_logprob_supported` latch).

**Fallbacks:** never returns empty; logprob→similarity on endpoint failure; judge/draft failures
caught and scored 0 / empty (best-effort).

**Logging:** `log_trace` writes `MIG_LOG_DIR/<test_id>.jsonl` with question, fn_call, kind,
pool/kept sizes, draft, per-candidate scores.

**Adding logprob-MIG later:** already implemented (`_mig_logprob`, `_answer_logprob`); the "later"
work is validating echo-logprob support on the BSC endpoint (risk R2) and tuning `draft_from_pool`.

---

## 8. Handler Design (`QwenMIGHandler`) `[Implemented]`

- **Parent:** `QwenFCHandler` (inherits prompt formatting, tool-call extraction, endpoint).
- **Overrides:** `_pre_query_processing_prompting` (stash `_mig_active`/`_mig_test_id`/`_mig_state`
  in `inference_data`, only for `is_memory(id)`), `decode_execute` (pool widening), 
  `_add_execution_results_prompting` (rerank).
- **Retrieval fns for pool widening** (`RETRIEVE_POOL_PARAM`): `archival_memory_retrieve`,
  `core_memory_retrieve` → `top_k`; `archival_memory_key_search`, `core_memory_key_search` → `k`.
  (kv exact `*_retrieve` takes `key` (no `query`) → skipped; `memory_retrieve` takes no args → skipped.)
- **Widening rule:** only widen (`max(current, pool_size)`), only when `"query"` in args, in place.
- **Rerank parsing/reserialization:** `parse_candidates` → `select` → `reserialize` into the exact
  `{"result": [...]}` / `{"ranked_results": [...]}` schema `execute_multi_turn_func_call` produced
  (via `json.dumps`), so the agent sees a schema-identical (just trimmed/reordered) result.
- **Non-memory / unsupported:** `_mig_active` false ⇒ exact parent behavior; `kind is None` ⇒
  return `raw` unchanged.
- **Original results preserved:** the full widened pool stays in `execution_results` (only the
  copy handed to the parent is trimmed), so the verbose inference log keeps the full pool for
  offline recall@k analysis.

---

## 9. Backend-Specific Plan `[Confirmed / Implemented]`

| Backend | V1 support | Result format | Reranking meaningful? | Behavior |
|---|---|---|---|---|
| `memory_vector` | **Yes (primary)** | `{"result":[{id,similarity_score,text}]}` | Yes — text candidates + scores | Widen `top_k`→pool; rerank to `final_budget`. |
| `memory_kv` | Partial | key-search `{"ranked_results":[[score,key]]}`; exact retrieve `{"value":...}` | Weak (keys are short snake_case; value not visible at rerank time) | Widen `k` on key-search; rerank keys; exact-retrieve passes through. |
| `memory_rec_sum` | No (pass-through) | `{"memory_content": <whole blob>}` | No candidate list | Pass-through unchanged (chunking is a future extension). |

**Headline conclusion:** MIG is fundamentally a **vector-memory reranker**; kv is a weak-signal
secondary; rec_sum is out of scope for reranking. Report all three but expect the effect to
concentrate on `memory_vector` (specifically `archival_memory_retrieve`, since core memory is
already fully dumped into the system prompt).

---

## 10. Logging and Analysis Plan `[Implemented + Proposed]`

**Implemented traces** (set `MIG_LOG_DIR`): per-test `<test_id>.jsonl`, one line per interception:
`interception`, `question`, `fn_call`, `kind`, `pool_size`, `kept_size`, `trace` (draft +
per-candidate `{text,prior,score}` + kept). Plus a one-line stdout summary per interception
(`MIG_VERBOSE`), captured in BFCL generation logs.

**Recommendation:** keep the **sidecar JSONL** (structured, easy to parse for the thesis) as the
primary trace; `handler_log` in the result file is fine for eyeballing but harder to aggregate.

**Proposed offline analysis** `[Proposed]` (new script, e.g.
`bfcl_eval/scripts/analyze_mig_traces.py`):
- Join traces with `data/possible_answer/BFCL_v4_memory.json` `source`/`ground_truth`.
- Compute **gold-in-pool** (was the answer-bearing memory retrieved at all?), **gold-selected**
  (did rerank keep it?), **rank-before/after**, recall@k, selection precision.
- Aggregate extra model calls (draft + per-candidate judge/logprob), latency, kept-size dist,
  greedy halt-step dist.

---

## 11. BFCL Run Plan `[Proposed — commands confirmed by CLI/guides]`

Environment (Windows/BSC, from the reproduction guides):

```powershell
Set-Location C:\Users\USER\Desktop\Tezim\BFCL\gorilla\berkeley-function-call-leaderboard
$env:BFCL_PROJECT_ROOT=(Get-Location).Path
$env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"
# .env already points at the BSC/local endpoint (LOCAL_SERVER_ENDPOINT=localhost, LOCAL_SERVER_PORT=8000)
```

### 1) Choose the MIG arm (env vars)

```powershell
$env:MIG_ENABLED="true"
$env:MIG_POOL_SIZE="20"
$env:MIG_FINAL_BUDGET="5"
$env:MIG_SCORER="judge"      # v1: judge | ablations: similarity, logprob
$env:MIG_MODE="flat"         # flat | greedy
$env:MIG_LOG_DIR="mig_traces\vector_judge_flat"
$env:MIG_VERBOSE="true"
```

### 2) Generate with the MIG model id (start with vector only for a fast loop)

```powershell
python -m bfcl_eval generate `
  --model Qwen/Qwen3-4B-Instruct-2507-FC-MIG `
  --test-category memory_vector `
  --skip-server-setup --temperature 0.001 --num-threads 1 `
  --include-input-log --allow-overwrite
```

### 3) Evaluate (fresh score dir to dodge the Excel file-lock issue)

```powershell
python -m bfcl_eval evaluate `
  --model Qwen/Qwen3-4B-Instruct-2507-FC-MIG `
  --test-category memory_vector `
  --score-dir score_mig
```

### 4) Locate scores & compare vs baseline

```powershell
Get-Content .\score_mig\Qwen_Qwen3-4B-Instruct-2507-FC-MIG\agentic\memory\vector\BFCL_v4_memory_vector_score.json -TotalCount 1
Get-Content .\score_mig\data_agentic.csv
# Baseline (already run) for the same backend:
Get-Content .\score_fresh\data_agentic.csv
```

The MIG result dir is `result\Qwen_Qwen3-4B-Instruct-2507-FC-MIG\agentic\memory\vector\`; scores
mirror under the chosen `--score-dir`. Compare the `Vector` column of `data_agentic.csv` between
baseline and MIG. `[Confirmed path shape]`

**Full run** (after the vector loop looks right): swap `--test-category` to
`memory_kv,memory_vector,memory_rec_sum`, drop `--num-threads 1`, keep `--temperature 0.001`, no
`--partial-eval`.

**Ablation matrix to run** `[Proposed]`: baseline · `similarity`+`MIG_RERANK=false` (pure pool
widening) · `judge`+`flat` · `judge`+`greedy` · `logprob`+`flat` · pool size {10,20,50}.

---

## 12. Validation Checklist `[Proposed]`

- [ ] Baseline files unchanged: `git diff --stat` shows no edits under `data/`,
      `func_source_code/memory_*.py`, `eval_checker/`, `agentic_eval/`.
- [ ] MIG registered separately: `Qwen/Qwen3-4B-Instruct-2507-FC-MIG` resolves; baseline `-FC` id
      still routes to `QwenFCHandler`.
- [ ] `model_config.py` imports `QwenMIGHandler` (see §5 `[Needs verification]`).
- [ ] Generation runs and prints `[MIG] QwenMIGHandler active ...` at startup.
- [ ] At least one `[MIG] ... pool=N -> kept=M` line appears (≥1 retrieval intercepted).
- [ ] Widened pool observed: a `*_retrieve`/`*_key_search` call executed with `top_k`/`k` ≥
      `MIG_POOL_SIZE` in the `--include-input-log` trace.
- [ ] Evaluation runs; `BFCL_v4_memory_vector_score.json` + `data_agentic.csv` exist under
      `--score-dir`.
- [ ] Reranker traces exist under `MIG_LOG_DIR`.
- [ ] Unsupported backends pass through: rec_sum / kv-exact-retrieve produce `kind=None`, results
      unchanged (verify a rec_sum run scores ~identically to baseline).
- [ ] Non-memory categories byte-identical to `QwenFCHandler` (spot-check a non-memory category if
      time permits).
- [ ] MIG vs baseline scores recorded with the exact env-var arm that produced them.

---

## 13. Risks and Open Questions

| ID | Risk | Evidence / status | Mitigation |
|---|---|---|---|
| R1 | **Wrong draft inverts logprob-MIG.** A zero-context draft is the model's prior guess, which is wrong exactly when memory is needed; the gold memory then contradicts the guess and scores lowest. | Already hit in the live experiment (`0bf8ec8`: "fix logprob wrong-draft failure"). | `draft_from_pool=True` seeds `a_hat` from the retrieved pool. Keep this on for logprob/judge; ablate it explicitly. |
| R2 | **BSC/local endpoint may not support echo logprobs.** | vLLM generally supports `echo`+`logprobs`, but `[Needs verification]` on this server build. | `_answer_logprob` latches `_logprob_supported=False` and degrades to similarity. Verify with a one-off `completions.create(echo=True, logprobs=1, max_tokens=0)` before trusting logprob arms. |
| R3 | **`memory_kv` returns only keys** at rerank time (values fetched later via exact key). Weak reranking signal. | `parse_candidates` → `kind="kv_keys"`; `memory_kv.py:197,320`. | Treat kv as secondary; report separately. Optional future: don't widen/rerank kv and just pass through. |
| R4 | **`memory_rec_sum` has no candidate list.** | `memory_rec_sum.py:136` returns whole blob. | Pass-through (implemented). Chunking is a future extension, not V1. |
| R5 | **Tool-call `top_k` rewriting fragility.** Widening depends on decoded arg dicts. | `widen_pool` guards: only known fns, requires `"query"` in args, only ever widens. | Guards in place; validate via input-log that widening actually took effect (checklist). |
| R6 | **Latency / extra model calls.** Flat = 1 draft + `pool_size` scoring calls per interception; greedy multiplies by picks. | By design. | Use `max_tokens=0` echo for logprob (cheap); server allows 100 concurrent (`eval_config.py:6`); cap `pool_size`; `--num-threads 1` only for debugging. |
| R7 | **Question extraction** relies on the last non-`<tool_response>` user message. | `extract_question` (`mig_reranker.py:276`). | Memory tasks are single-question, so robust; verify on a multi-interception trace. |
| R8 | **Effect may be small**: core memory is already fully in the system prompt for kv/vector; only archival retrieval benefits. | `add_memory_instruction_system_prompt` dumps core memory. | Focus analysis on `archival_memory_retrieve`; segment metrics by core vs archival. |
| R9 | **Missing `BFCL_MIG_RERANKER_GUIDE.md`** referenced by docstrings. | File absent. | Author it (run recipe + experiment findings) as part of remaining work (§4). |
| R10 | **Temperature clamp** (vLLM clamps <0.01 to 0.01). | Known from guides. | Record it; use `0.0` for MIG draft/judge/logprob side-calls (they're separate from the graded generation). |

---

## Appendix: Confirmed file:line index

```text
constants/model_config.py:1636         baseline -FC ModelConfig
constants/model_config.py:1665         MIG ModelConfig (QwenMIGHandler)
constants/supported_models.py:142-143  -FC and -FC-MIG ids
local_inference/qwen_fc.py:10,36,48,250  baseline handler (parent)
local_inference/qwen_mig.py:34,70,81,100 MIG handler (hooks)
middleware/mig_reranker.py:50,85,155,177,208,296,396,471  reranker internals
base_oss_handler.py:42-49,319-366,411  endpoint config, completions call, result injection
base_handler.py:68,124-145,211,296,371,393  inference dispatch, loop, snapshot flush
model_handler/utils.py:610             add_memory_instruction_system_prompt
multi_turn_eval/multi_turn_utils.py:13,83  execute_multi_turn_func_call / eval
func_source_code/memory_vector.py:134,209,309  retrieve methods
func_source_code/memory_kv.py:197,320  key_search methods
func_source_code/memory_rec_sum.py:136  whole-blob retrieve
eval_checker/eval_runner.py:72,104-148,501  agentic scoring extraction
agentic_eval/agentic_checker.py:6,41   substring match + standardize
utils.py:203,211,714,772,837           memory helpers / test-case assembly
constants/default_prompts.py:1,79-107  step limit + memory prompt templates
```
