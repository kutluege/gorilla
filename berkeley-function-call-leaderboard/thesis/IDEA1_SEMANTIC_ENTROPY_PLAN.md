# Implementation Plan — Idea 1: Semantic-Entropy Gate for BFCL Memory Categories

**Goal:** wrap Qwen3-4B-Instruct-2507 (served by vLLM on BSC, reached through an SSH
tunnel at `http://localhost:8000`) in a sampling-based uncertainty gate that
(a) samples N completions at every memory-relevant step, (b) clusters them semantically,
(c) commits the majority action — with *stricter* gating for destructive memory ops —
and (d) logs cluster entropy for the write-time-uncertainty ↔ forgetting correlation study.

No changes to datasets, graders, or scoring. The whole intervention lives in a new handler
subclass + one utility module + one registry entry.

**One important correction to the original sketch:** the override point is
`_query_prompting`, **not** `_query_FC`. All OSS/local models (including
`Qwen/Qwen3-4B-Instruct-2507-FC`) route through `OSSHandler.inference` →
`inference_multi_turn_prompting` → `_query_prompting`
(`bfcl_eval/model_handler/local_inference/base_oss_handler.py:52-64, 317-364`), which
calls the vLLM **completions** API with a manually formatted chat-template prompt. The
`-FC` suffix only changes how tool calls are decoded (`QwenFCHandler.decode_execute`
parses `<tool_call>` JSON), not the query path.

---

## Phase 0 — Connectivity and baseline (no code changes)

The repo already supports a remote OpenAI-compatible endpoint; you don't need any code to
point BFCL at the tunnel.

### 0.1 Serve on BSC (compute node)

```bash
vllm serve /gpfs/scratch/ehpc540/models/Qwen3-4B-Instruct-2507 \
  --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
  --port 8000
```

`--served-model-name` is **required**: the handler sends
`model="Qwen/Qwen3-4B-Instruct-2507"` (the `model_name` in the registry,
`bfcl_eval/constants/model_config.py:1660-1671`); without it, vLLM registers the model
under the filesystem path and every request 404s.

### 0.2 Tunnel (as you already do)

Your two-hop socket tunnel ends with the endpoint on `localhost:8000` locally. Verify from
the local machine:

```powershell
curl http://localhost:8000/v1/models
# must return the model id "Qwen/Qwen3-4B-Instruct-2507"
```

### 0.3 Environment variables for BFCL (local, PowerShell)

```powershell
.\venv\Scripts\Activate.ps1
$env:REMOTE_OPENAI_BASE_URL = "http://localhost:8000/v1"
$env:REMOTE_OPENAI_API_KEY  = "EMPTY"
# tokenizer+config are loaded locally by the handler (token counting / context length).
# Either let HF download Qwen/Qwen3-4B-Instruct-2507 (small files), or point to a local copy:
# $env:REMOTE_OPENAI_TOKENIZER_PATH = "C:\models\Qwen3-4B-Instruct-2507-tokenizer"
```

Mechanics (`base_oss_handler.py:41-49, 120-152, 257-265`): with `REMOTE_OPENAI_BASE_URL`
set and `--skip-server-setup` passed, the handler skips launching vLLM, loads only the
tokenizer/config (needed for `max_context_length` and input-token counting), and waits for
`GET {base_url}/models` to return 200 — i.e., your tunnel must be up before `bfcl generate`.

### 0.4 Baseline runs (the control arm of every later comparison)

```powershell
bfcl generate --model Qwen/Qwen3-4B-Instruct-2507-FC `
  --test-category memory `
  --skip-server-setup --num-threads 1
bfcl evaluate --model Qwen/Qwen3-4B-Instruct-2507-FC --test-category memory
```

Notes:
* `--num-threads 1` first time; the prereq chain is serialized by `depends_on` anyway, and
  a single 4B server gains little from concurrency. Raise later if stable.
* Re-running memory categories wipes/regenerates snapshots per category
  (`_llm_response_generation.py:145-156`) — never mix runs in one result dir.
* Keep the whole result folder: `result/.../agentic/memory/<backend>/memory_snapshot/`
  (incl. `prereq_checkpoints/`) is the raw material for the forgetting analysis.
* Expected duration: 5 scenarios × ~37 prereq conversations total × up to 20 steps/turn,
  ×3 backends, + 465 recall entries. Budget several hours per full memory pass.

**Deliverable of Phase 0:** baseline scores for `memory_kv`, `memory_vector`,
`memory_rec_sum` + preserved snapshots/logs. (Also run `multi_turn_base` once for a
multi-turn baseline if you want the gate evaluated there later.)

---

## Phase 1 — Utility module: clustering, entropy, gate policy

**New file:** `bfcl_eval/model_handler/semantic_entropy_utils.py`. Pure logic, no BFCL
imports (unit-testable offline).

### 1.1 Configuration (env vars, read at call time)

| Var | Default | Meaning |
|---|---|---|
| `SE_NUM_SAMPLES` | `5` | N completions per gated step; `1` disables the gate entirely |
| `SE_TEMPERATURE` | `0.7` | sampling temperature for the N-sample call |
| `SE_SIM_THRESHOLD` | `0.80` | cosine threshold: two tool-call candidates share a cluster |
| `SE_TEXT_SIM_THRESHOLD` | `0.85` | threshold for free-text (answer) candidates |
| `SE_DESTRUCTIVE_MAX_ENTROPY` | `0.7` | bits; destructive op executes only below this |
| `SE_DESTRUCTIVE_MIN_MAJORITY` | `0.6` | and only if majority-cluster fraction ≥ this |
| `SE_OVERWRITE_MAX_ENTROPY` | `1.2` | looser tier for replace/update ops |
| `SE_LOG_FILE` | `semantic_entropy_log.jsonl` | JSONL audit log (thread-lock the append) |

### 1.2 Operation classes

```python
DESTRUCTIVE_OPS = {"core_memory_clear", "archival_memory_clear",
                   "core_memory_remove", "archival_memory_remove", "memory_clear"}
OVERWRITE_OPS   = {"core_memory_replace", "archival_memory_replace",
                   "core_memory_update", "archival_memory_update",
                   "memory_update", "memory_replace"}   # memory_update = rec_sum full-blob overwrite!
```

Everything else (adds, retrieves, searches, plain text) is *safe*.

### 1.3 Candidate representation

Each of the N completions becomes:

```python
{"text": str,                  # raw completion
 "kind": "tool_call"|"text",   # decoded successfully as calls vs not
 "calls": list[str]|None,      # e.g. ["core_memory_add(key='age', value='35')"]
 "func_names": tuple[str]}     # sorted tool names, () for text
```

The handler produces `calls` by running each candidate through
`self.decode_execute(text, has_tool_call_tag=False)` in a try/except (decode failure or
empty ⇒ `kind="text"`). This reuses the exact decode path BFCL will apply to the chosen
candidate, so the gate reasons about precisely what would be executed.

### 1.4 Clustering (union-find over pairwise equivalence)

* Different `kind` ⇒ never same cluster.
* `tool_call` pair: same `func_names` **and** cosine(embedding of normalized joined call
  strings) ≥ `SE_SIM_THRESHOLD`. Normalization: lowercase, strip quotes/underscores,
  collapse whitespace — so `core_memory_add(key='user_age', value='35')` and
  `core_memory_add(key='age', value='35 years old')` cluster together, but a different
  tool name never merges.
* `text` pair: cosine ≥ `SE_TEXT_SIM_THRESHOLD`. (Optional upgrade, flag `SE_USE_NLI=1`:
  bidirectional entailment with `cross-encoder/nli-deberta-v3-base`, Farquhar-style;
  start with embeddings — cheaper and adequate for short factual answers.)
* Embedder: reuse `sentence-transformers` `all-MiniLM-L6-v2` on CPU — **already a repo
  dependency** (the vector backend uses it), so nothing new to install. Lazy-load a
  module-level singleton. Fallback if unavailable: `difflib.SequenceMatcher` ratio.

### 1.5 Entropy and decision

```
p_c = |cluster_c| / N          H = -Σ p_c log2 p_c        m = max_c p_c
```

Decision procedure (`choose(candidates, cfg) -> (chosen_index, record)`):

1. Majority cluster = largest (ties: lower H contribution first). Representative =
   medoid (max mean intra-cluster similarity).
2. If majority representative contains a **destructive** op: execute only if
   `H ≤ SE_DESTRUCTIVE_MAX_ENTROPY` **and** `m ≥ SE_DESTRUCTIVE_MIN_MAJORITY`.
   Otherwise **fall back to the largest non-destructive cluster's** representative
   (typically an `archival_memory_add` or a retrieve — the "safe default"). If *every*
   cluster is destructive, execute the majority anyway but log `forced_destructive: true`
   (do not fabricate a synthetic response — an unparseable injected message would corrupt
   the conversation).
3. **Overwrite** ops: same rule with the looser `SE_OVERWRITE_MAX_ENTROPY` threshold.
4. Safe ops / text answers: return the majority medoid (this alone is the
   semantic-self-consistency answer-selection for the 155 recall questions).
5. `record` = full audit: test_id, per-candidate `(kind, func_names, truncated text)`,
   cluster assignment, `H`, `m`, op class, decision, fallback flag, wall-time.

Rationale for the asymmetry: your observed Qwen failure is precisely "core memory full →
delete/clear without archiving". Making destruction require *consensus* while leaving
additive ops cheap targets that failure without throttling normal writes.

---

## Phase 2 — Handler subclass

**New file:** `bfcl_eval/model_handler/local_inference/qwen_se.py`
**Class:** `QwenSemanticEntropyHandler(QwenFCHandler)`.

Only two overrides:

### 2.1 Thread-safe test-id plumbing

`_query_prompting` receives only `inference_data`, but gating should apply only to memory
entries. `_pre_query_processing_prompting(test_entry)` *does* see the entry
(`base_oss_handler.py:366-376`) and its return dict flows into every later call:

```python
@override
def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
    inference_data = super()._pre_query_processing_prompting(test_entry)
    inference_data["se_test_id"] = test_entry["id"]     # dict-carried → no self.* races
    return inference_data
```

### 2.2 The gated query

```python
@override
def _query_prompting(self, inference_data: dict):
    cfg = se_config()
    test_id = inference_data.get("se_test_id", "")
    if cfg.num_samples <= 1 or not is_memory(test_id):   # bfcl_eval.utils.is_memory
        return super()._query_prompting(inference_data)

    # --- replicate base_oss_handler.py:317-364 with two changes: n=N, temperature=cfg.temperature ---
    formatted_prompt = self._format_prompt(inference_data["message"], inference_data["function"])
    inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}
    ... token-budget logic identical to base ...
    api_response = self.client.completions.create(
        model=self.model_path_or_id, prompt=formatted_prompt,
        n=cfg.num_samples, temperature=cfg.temperature,
        max_tokens=leftover_tokens_count, extra_body=extra_body, timeout=72000)

    # --- decode every choice, gate, put the chosen one first ---
    candidates = [make_candidate(c.text, try_decode(self, c.text)) for c in api_response.choices]
    chosen, record = choose(candidates, cfg)
    api_response.choices = ([api_response.choices[chosen]] +
                            api_response.choices[:chosen] + api_response.choices[chosen+1:])
    log_record({**record, "test_id": test_id, "n_calls_so_far": ...}, cfg)
    return api_response, elapsed
```

Why this is safe with zero further changes:

* `_parse_query_response_prompting` reads `api_response.choices[0].text`
  (`base_oss_handler.py:378-383`) — reordering choices is sufficient; the multi-turn loop,
  memory hooks, snapshot flushing, and grading all remain untouched.
* Token accounting stays **honest**: `usage.completion_tokens` from a single `n=N` call
  counts all N samples; `prompt_tokens` counts the (shared) prompt once. Your reported
  cost overhead is real, not hidden.
* Gating is scoped: prereq entries → tool-action gating on every step; recall entries →
  the same code path naturally does answer-level majority selection (candidates decode to
  `text` kind on the final answer step). Retrieval tool calls inside recall entries get
  majority-vote stabilization too, which is desirable.
* `n>1` on the completions endpoint is standard vLLM; all N samples share one prefill, so
  wall-clock overhead is far below ×N (expect ~×2–3 for N=5 on generation-heavy steps).

Optional variant to keep in your back pocket (flag `SE_ADAPTIVE=1`): sample 2 first; only
if they land in different clusters, sample the remaining N−2. Cuts cost roughly in half on
easy steps. Implement only if runtime hurts.

### 2.3 Registration

In `bfcl_eval/constants/model_config.py`, next to the existing entry at L1660:

```python
"Qwen/Qwen3-4B-Instruct-2507-SE-FC": ModelConfig(
    model_name="Qwen/Qwen3-4B-Instruct-2507",     # must equal --served-model-name
    display_name="Qwen3-4B-Instruct-2507 SE-Gated (FC)",
    url="https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507",
    org="Qwen", license="apache-2.0",
    model_handler=QwenSemanticEntropyHandler,
    input_price=None, output_price=None,
    is_fc_model=True, underscore_to_dot=False,
),
```

Distinct registry key ⇒ results land in a separate model folder
(`Qwen_Qwen3-4B-Instruct-2507-SE-FC/`) — baseline and gated runs never collide, and
`bfcl evaluate` compares them as two "models".

---

## Phase 3 — Offline validation before touching the server

Write a tiny script (`thesis/scripts/test_se_gate_offline.py`) that feeds hand-crafted
candidate sets straight into `choose()` — no server, no BFCL run:

1. 5× identical `core_memory_add` (paraphrased values) → 1 cluster, H=0, commit.
2. 3× `core_memory_add` + 2× `archival_memory_add` → 2 clusters, H≈0.97, commit majority
   (safe op, no block).
3. 2× `core_memory_remove` + 2× `archival_memory_add` + 1× `core_memory_clear` →
   destructive majority absent/uncertain → falls back to `archival_memory_add`. **This is
   the scenario that must pass — it is the thesis's failure-mode fix in miniature.**
4. 5 free-text answers, 4 saying "35", 1 saying "I don't know" → majority medoid "35".
5. Mixed tool-call + text candidates → kinds never cluster together.

Also verify: `SE_NUM_SAMPLES=1` makes the handler byte-identical to baseline (gate off).

Then one **smoke run** against the tunnel before the full matrix:

```powershell
bfcl generate --model Qwen/Qwen3-4B-Instruct-2507-SE-FC `
  --test-category memory_kv --run-ids ...   # or just let one scenario's prereqs run and Ctrl-C
```

Check: `semantic_entropy_log.jsonl` fills with records; snapshots appear; no decode errors.

---

## Phase 4 — Experiment matrix

| Run | Model key | Purpose |
|---|---|---|
| B1 | `...-FC`, temp 0.001 (default) | baseline |
| B2 | `...-FC`, `SE_NUM_SAMPLES=1` via SE handler | sanity: must equal B1 within noise |
| E1 | `...-SE-FC`, N=5, defaults | main result |
| E2 | E1 with gate thresholds ∞ (pure majority vote, no destructive blocking) | ablation: self-consistency only |
| E3 | E1 with N=5 but destructive gating only on prereqs (`SE` off for recall entries) | ablation: write-side vs answer-side benefit |
| E4 (opt) | N=10 | entropy-estimate quality vs cost |

All × {`memory_kv`, `memory_vector`, `memory_rec_sum`}. Because sampling is stochastic,
run E1 with ≥2 seeds if time allows (vLLM `seed` can go in `extra_body`) and report both.
Statistics: per-question paired outcomes (155 per backend) → McNemar's test baseline vs E1;
bootstrap CIs on accuracy deltas.

Multi-turn extension (optional, later): allow gating on `multi_turn_*` by widening the
`is_memory` check to an env-controlled category list — the state-based grader there will
show whether majority-action selection also reduces `instance_state_mismatch` failures.

---

## Phase 5 — Analysis (the thesis chapter)

Inputs: result dirs (baseline + gated), snapshots (`memory_snapshot/prereq_checkpoints/*.json`,
`{scenario}_final.json`), score files
(`score/.../agentic/memory/<backend>/BFCL_v4_memory_<backend>_score.json`), and
`semantic_entropy_log.jsonl`.

1. **Failure decomposition per question** (WRITE / RETENTION / RETRIEVE / ANSWER):
   ground-truth strings from `possible_answer/BFCL_v4_memory.json` (standardized the same
   way as `agentic_checker.standardize_string`) searched in: every prereq checkpoint →
   final snapshot → recall entry's tool-result log → final answer. Compare the
   distribution baseline vs gated: the destructive gate should specifically shrink
   RETENTION-loss; answer-side majority voting should shrink ANSWER-miss.
2. **Write-time entropy ↔ forgetting correlation:** map each question's `source` sentence
   to its prereq conversation (substring search over
   `data/memory_prereq_conversation/*.json`); join with the mean/max step entropy of that
   conversation from the SE log; point-biserial correlation against per-question outcome.
   This answers the headline research question: *does the model's own uncertainty at write
   time predict what it will forget?*
3. **Selective answering:** rank recall questions by answer-cluster entropy; plot accuracy
   vs coverage (risk–coverage curve) — entropy as a calibrated confidence signal.
4. **Gate audit:** counts of destructive ops proposed vs blocked vs forced, and what the
   fallback did instead — the qualitative table for the thesis ("the gate prevented X
   clears that would have destroyed facts later asked about").

---

## Pitfalls / troubleshooting

* **404 / model not found:** vLLM serving under the path name — re-serve with
  `--served-model-name Qwen/Qwen3-4B-Instruct-2507`.
* **Hang at "server is ready" check:** the handler polls `GET {REMOTE_OPENAI_BASE_URL}/models`
  before starting — tunnel down or wrong port.
* **Tokenizer download errors locally:** set `REMOTE_OPENAI_TOKENIZER_PATH` to a local
  folder containing the model's `tokenizer_config.json` + `config.json` (copy those small
  files from `/gpfs/scratch/ehpc540/models/Qwen3-4B-Instruct-2507`, no weights needed).
* **Windows + sentence-transformers:** first use downloads MiniLM (~90 MB); it's CPU-only
  here and already required by the `memory_vector` backend, so if baseline vector runs
  worked, the gate's embedder will too.
* **Don't reuse a result dir across code changes** — memory snapshots are stateful; stale
  `{scenario}_final.json` from an aborted run silently contaminates the next one (the
  harness warns with a wall of ⚠️ but continues with empty memory).
* **Never gate by rewriting candidate text** — only ever *select among sampled candidates*;
  synthesized messages break `decode_execute`/chat-template assumptions.

## Suggested order of work

1. Phase 0 baseline (can start today — zero code).
2. Phase 1 module + Phase 3 offline tests (pure local Python, no GPU/tunnel needed).
3. Phase 2 handler + registry (≈150 lines total incl. the copied query body).
4. Smoke run → full E1 → ablations → analysis.
