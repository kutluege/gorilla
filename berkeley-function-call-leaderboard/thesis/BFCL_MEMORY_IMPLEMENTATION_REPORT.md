# Implementing Semantic-Entropy / Ontology / Fact-Extraction / Admission-Control Ideas in BFCL v4

**Scope:** the memory (`memory_kv`, `memory_vector`, `memory_rec_sum`) and multi-turn
(`multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multi_turn_long_context`)
categories of the Berkeley Function-Calling Leaderboard (BFCL v4), as implemented in this
repository under `berkeley-function-call-leaderboard/`.

All file paths below are relative to `berkeley-function-call-leaderboard/`.

---

## Part A — How the benchmark actually works (research findings)

### A.1 The memory category pipeline

**Data.** `bfcl_eval/data/BFCL_v4_memory.json` contains 155 recall questions across 5
scenarios (`customer`, `student`, `finance`, `healthcare`, `notetaker`). Each entry is a
single-turn question, e.g.:

```json
{"id": "memory_1-customer-1", "question": [[{"role": "user", "content": "How old am I?"}]],
 "involved_classes": ["MemoryAPI"], "scenario": "customer"}
```

The answer key (`bfcl_eval/data/possible_answer/BFCL_v4_memory.json`) holds, per question,
a list of acceptable strings **and the source sentence** the fact came from:

```json
{"id": "memory_1-customer-1", "ground_truth": ["35", "thirty five"],
 "source": "I'm 35 years old, live in Seattle..."}
```

**Prerequisite conversations.** `bfcl_eval/data/memory_prereq_conversation/memory_{scenario}.json`
contains 5–10 long multi-turn conversations per scenario in which the *user monologues*
dense personal facts. The model must **proactively decide what to store** using memory
tools; nothing tells it which facts will be asked about later.

**Assembly.** `bfcl_eval/utils.py::process_memory_test_case` (L714–752) clones the dataset
per backend: ids are rewritten (`memory_…` → `memory_kv_…`), `involved_classes` is set to
`MemoryAPI_{kv|vector|rec_sum}`, and every prereq entry `depends_on` all previous prereq
entries of its scenario; every real question depends on the *entire* prereq chain.
`bfcl_eval/_llm_response_generation.py::generate_results` (L268–348) is a topological
scheduler over `depends_on`, so the write-phase conversations always run before the recall
questions.

**Persistence (snapshots).** Backends inherit from
`bfcl_eval/eval_checker/multi_turn_eval/func_source_code/memory_api_metaclass.py::MemoryAPI`.
`_prepare_snapshot` (L24–76) loads
`<result_dir>/<model>/agentic/memory/<backend>/memory_snapshot/{scenario}_final.json`;
after each **prereq** entry, `base_handler.py` (L369–376, L660–667) calls
`_flush_memory_to_local_file()`, which writes a per-entry checkpoint into
`memory_snapshot/prereq_checkpoints/{test_id}.json` **and** overwrites
`{scenario}_final.json`. Real recall entries load the final snapshot read-only and are never
flushed. The per-entry checkpoints are a complete audit trail of memory state after every
prereq conversation — very useful for diagnostics (see Part C).

**Inference.** Memory entries run through the same multi-turn loop as `multi_turn_*`
(`base_handler.py::inference_multi_turn_FC`, L94–390): per user turn, an inner step loop
queries the model, decodes tool calls (`decode_execute`), executes them via
`execute_multi_turn_func_call` (`multi_turn_eval/multi_turn_utils.py` L13–100 — backend
instances are cached in `globals()` per `(model, test_id, class)` so state persists across
steps/turns), feeds results back, and stops when the model emits a non-tool-call message.
`MAXIMUM_STEP_LIMIT = 20` steps/turn (`constants/default_prompts.py` L1). A memory system
prompt with the **current core-memory contents inlined** is injected at turn 0 via
`model_handler/utils.py::add_memory_instruction_system_prompt` (L610–650), using templates
in `constants/default_prompts.py` (L91–107).

**Scoring.** Memory is *not* state-checked. It is routed to
`eval_checker/eval_runner.py::agentic_runner` → `agentic_eval/agentic_checker.py`: the last
non-tool-call message is taken as the answer, and the entry passes iff any `ground_truth`
string appears as a whole word in the standardized (lowercased, punctuation-stripped)
response. Prereq entries are never scored. Leaderboard columns: Memory Summary + one column
per backend (`eval_runner_helper.py` L445–461); no per-scenario breakdown.

### A.2 The three memory backends

| Backend | Structure | Write API | Read API | Limits |
|---|---|---|---|---|
| `memory_kv.py` | two dicts: core + archival | `core/archival_memory_add(key, value)`, `_replace`, `_remove`, `_clear` | `_retrieve(key)` (exact key!), `_list_keys`, `_key_search` (BM25+ **over keys only**), `core_memory_retrieve_all` | core: 7 entries × 300 chars; archival: 50 × 2000; keys must be snake_case & unique |
| `memory_vector.py` | two FAISS stores (all-MiniLM-L6-v2) | `core/archival_memory_add(text)` → id, `_update(id, text)`, `_remove(id)`, `_clear` | `_retrieve(query, top_k)` cosine search, `_retrieve_all` | same 7×300 / 50×2000 |
| `memory_rec_sum.py` | one string blob | `memory_append(text)`, `memory_update(text)`, `memory_replace(old,new)`, `memory_clear` | `memory_retrieve()` (whole blob) | 10,000 chars total |

**Structural failure traps** (these explain the Qwen failures you observed):

1. **Core memory fills at 7 entries** and `*_add` returns an error string. There is no
   automatic spill; the model must itself decide to archive-then-remove. Weak models
   respond to `"Core memory is full"` with `core_memory_remove`/`core_memory_clear` —
   destroying facts without archiving. `*_clear` is irreversible and has no confirmation.
2. **KV archival retrieval searches keys only** (`memory_kv.py` L197–211, L320–334): BM25
   over key names. If facts were stored under uninformative keys (`note_1`, `misc`), they
   are unfindable at recall time even though they were stored.
3. **rec_sum `memory_update` replaces the entire blob** — a model that uses `update`
   instead of `append` silently erases all previous conversations' facts.
4. **Grading is verbatim substring** — paraphrases stored in memory ("mid-thirties") fail
   against ground truth ("35"). Storage must preserve numbers, names, units verbatim.

### A.3 Multi-turn categories

200 entries each; multi-API environments (`GorillaFileSystem`, `TwitterAPI`, trading,
travel, vehicle…) with per-turn ground-truth call sequences. Evaluation
(`multi_turn_eval/multi_turn_checker.py`) re-executes model calls and ground-truth calls on
fresh instances and requires, per turn: (i) **state match** — every public attribute of
every backend instance equal to ground truth (`state_checker` L162–194); (ii) **response
match** — accumulated model execution outputs must contain the turn's ground-truth
execution outputs as an unordered subsequence (`response_checker` L197–220); plus empty-turn
and irrelevance (miss_func/miss_param) checks. One wrong attribute fails the entry.

### A.4 Where a research "middleware" can hook in

The inference loops are `@final`, but everything they call is overridable. From least to
most invasive:

1. **Context injection** — wrap `add_memory_instruction_system_prompt` or override the
   backend's `_dump_core_memory_to_context`.
2. **Handler subclass** (e.g. `class QwenMemGovHandler(QwenFCHandler)`) overriding the
   non-final building blocks: `_pre_query_processing_FC`, `_compile_tools`,
   `add_first_turn_message_FC`, `_add_next_turn_user_message_FC`,
   `_add_assistant_message_FC`, `_add_execution_results_FC`, `_query_FC`, `decode_execute`.
   This is the natural seam for sampling-based validators and turn-level fact extraction.
3. **Decode layer** — `decode_execute` can rewrite/veto/augment memory tool calls before
   execution.
4. **New backend** — add `MemoryAPI_<name>` in
   `eval_checker/multi_turn_eval/func_source_code/`, register in
   `constants/executable_backend_config.py` (`CLASS_FILE_PATH_MAPPING`,
   `MULTI_TURN_FUNC_DOC_FILE_MAPPING`, `OMIT_STATE_INFO_CLASSES`) and
   `constants/category_mapping.py::ALL_AVAILABLE_MEMORY_BACKENDS`, add tool docs in
   `data/multi_turn_func_doc/memory_<name>.json`, and (if needed) a prompt branch in
   `add_memory_instruction_system_prompt`. The whole prereq/snapshot/scoring machinery is
   inherited for free — a new backend instantly becomes a new leaderboard category over the
   same 155 questions.

A crucial framing point: **BFCL's memory score is end-to-end and model-side.** Anything you
add on the agent side (validators, extractors, admission controllers) is legitimate "system
under test" improvement and needs no change to the grader — ideal for a thesis, because the
benchmark stays fixed while you compare `model` vs `model + your system`.

---

## Part B — Implementing the four ideas

### Idea 1 — Semantic entropy over answers and tool actions

**What it becomes in BFCL:** a sampling-based *uncertainty gate* inside a handler subclass,
applied at three decision points.

1. **Tool-action entropy during prereq (write) conversations.** At each step, instead of
   one completion, sample N (5–10) completions at temperature ~0.7 (vLLM `n>1` makes this
   cheap for local Qwen; `_query_FC` is the override point). Canonicalize each into an
   action signature: `(tool_name, normalized_args)` — exact match on tool name, semantic
   match on values (embedding similarity or bidirectional NLI between `value` strings;
   key names normalized). Cluster the N actions; compute entropy over clusters
   H = −Σ p_c log p_c.
   * Low entropy → commit the majority action.
   * High entropy → deliberate: either re-prompt with a reflection instruction, or fall
     back to the *safe default* (for memory ops the safe default is "write to archival
     with a descriptive key", never a destructive op).
   * **Asymmetric gating:** destructive ops (`*_clear`, `*_remove`, `memory_update` on
     rec_sum) require low entropy *and* majority agreement to execute; additive ops pass
     at higher entropy. This directly targets the observed failure (deleting under
     capacity pressure).
2. **Answer-level semantic entropy at recall time.** For each of the 155 questions, sample
   N final answers, cluster by bidirectional entailment (a DeBERTa-MNLI cross-encoder, as
   in Farquhar et al. 2024), output a representative of the largest cluster. Because
   grading is substring-based, majority-cluster selection converts "sometimes right" into
   "usually right", and cluster entropy is a calibrated confidence score you can report
   (selective-answering curves: accuracy vs coverage).
3. **Write-utility validation (links to Idea 4).** A candidate memory write reduces future
   semantic entropy iff the model, when asked probe questions with the write present vs
   absent, concentrates its answer distribution. Cheap proxy on BFCL: measure answer
   entropy on the recall questions with/without each memory entry (ablation at snapshot
   level — no re-running prereqs needed, just edit `{scenario}_final.json` copies).

**Code touch points:** new handler subclass + CLI flag (e.g.
`--memory-validator semantic-entropy --n-samples 5`); an NLI/embedding utility module;
no changes to data or graders. Cost: ×N inference on gated steps only (you can trigger
sampling *only* when a memory tool call or a recall answer is being produced).

**Research output:** correlation between write-time action entropy and later recall failure
(per fact, using the diagnostics of Part C) — i.e., *can the model's own uncertainty at
write time predict what it will forget?*

### Idea 2 — Vault-LD: markdown-wiki episodic memory + ontology/KG semantic memory

**What it becomes in BFCL:** a **fourth memory backend**, `memory_vault`, evaluated on the
identical 155 questions — turning BFCL-memory from a model benchmark into a *memory-system*
benchmark (hold the model constant, vary the backend).

Backend design (`MemoryAPI_vault`):

* **Episodic layer:** markdown notes with YAML-LD frontmatter. Tools:
  `note_write(path, frontmatter, body)`, `note_read(path)`, `note_search(query)` (BM25 over
  bodies + frontmatter values — fixing the kv backend's keys-only search).
* **Semantic layer:** triples extracted from frontmatter (`subject`, `predicate`, `object`,
  `time_scope`, `confidence`). Tools: `fact_assert(s, p, o, qualifiers)`,
  `fact_query(pattern)` (SPARQL-lite: wildcards over s/p/o), `entity_view(entity)` (all
  facts about an entity — the join the kv/vector backends cannot express).
* **Validation:** a mini-ontology per scenario (Person, Order, Course, Medication,
  Account…) with domain/range checks; `fact_assert` rejects type-violating triples with an
  instructive error (the model gets feedback instead of silent corruption).
* Snapshots serialize notes + triples to JSON — `_prepare_snapshot`/`_flush` inherited.

Registration is exactly the 4-step recipe of §A.4-4. Same limits philosophy: keep a bounded
"core" (the frontmatter index is always in context via `_dump_core_memory_to_context`) and
an unbounded-ish archive, so capacity-pressure behavior remains testable.

**Hypotheses to test:** (i) structured fields preserve verbatim values better than free
text (substring grading rewards this); (ii) entity-centric retrieval beats key-BM25 and
raw cosine similarity on multi-hop questions ("name one of two accessories I added…");
(iii) smaller models benefit more, because the schema externalizes organization they can't
do implicitly.

### Idea 3 — NER / fact extraction with importance scoring

**What it becomes in BFCL:** a *turn-level extraction middleware* in the handler, plus a
retrieval-time parameter binder. Three deployment variants of increasing autonomy:

* **V1 — Hint injection (model stays in control).** Before forwarding each prereq user
  turn, run an extractor (GLiNER/spaCy for entities + the same LLM with a structured
  extraction prompt for atomic facts) over the turn. Append a compact system hint:
  `Candidate facts worth persisting: [{fact, type, importance}, …]`. The model still issues
  its own memory calls. Implementation: override `_add_next_turn_user_message_FC` /
  `add_first_turn_message_FC`.
* **V2 — Co-writer.** The middleware issues its own memory tool calls (through the same
  `execute_multi_turn_func_call`, so they hit the real backend and appear in the tool
  transcript), writing normalized entries: snake_case, entity-bearing keys
  (`espresso_machine_delivery_days`, `kitchen_counter_sqft`). This single change attacks
  the kv backend's biggest structural trap: BM25-over-keys retrieval only works when keys
  carry the salient entity terms.
* **V3 — Retrieval-side binder.** At recall time, extract entities from the question,
  auto-fire `*_key_search`/`*_retrieve(query)` with entity-derived queries, and inject the
  top-k hits into context before the model answers. For multi-turn categories, the same
  entity store ranks previously-seen values (account ids, file names, amounts) as
  candidate tool-call parameters — directly relevant to `multi_turn_miss_param`.

**Scoring the importance:** type-based priors (numbers, dates, proper names, measurements
high; chit-chat low) × TF-IDF-style rarity × user-attachment (first-person possessives).
Ground truth for calibrating the scorer exists in the repo: the `source` field of every
possible answer tells you exactly which sentences mattered — you can train/tune the
importance scorer on some scenarios and test generalization on held-out ones.

**Constraint to respect:** extraction must store values *verbatim* (grading is substring
match); normalize keys, never values.

### Idea 4 — Entropy- and entailment-aware memory admission controller

**What it becomes in BFCL:** a **governed memory layer** between `decode_execute` and
execution — every memory tool call passes through an admission/eviction policy; plus an
optional benchmark-side extension.

Admission pipeline for a candidate write `f`:

1. **Decompose** (from Idea 3): atomic facts with entities, relation, value, time scope,
   confidence, provenance (turn id).
2. **Duplicate check:** embedding similarity vs existing entries; if near-duplicate → no-op
   (report "already known" to the model).
3. **Entailment check** (NLI both directions vs each related entry):
   * existing ⊨ f → reject (redundant; "Ali lives in Istanbul" already implies "Ali lives
     in Turkey");
   * f ⊨ existing, not vice versa → **replace** (f is strictly more specific);
   * contradiction → route to `*_replace`/`_update` with provenance note, never a bare add
     (prevents stale-fact answers).
4. **Utility score:** importance (Idea 3) + expected entropy reduction (Idea 1's probe) −
   storage/retrieval cost − temporal-decay penalty.
5. **Placement & eviction:** top-scoring facts in core (always in context), rest in
   archival. When core is full, **auto-archive the lowest-utility core entry, then add** —
   never error, never destroy. Destructive calls (`*_clear`, unarchived `_remove`) are
   intercepted and converted into archive-then-remove ("tombstoning"). This is the precise
   fix for the observed Qwen failure mode.

Two deployment roles, both valuable:

* **Agent-side (scored):** the controller runs inside the handler; BFCL score measures its
  end-to-end benefit. Model-agnostic — same controller wraps Qwen, Llama, GPT handlers.
* **Benchmark-side (diagnostic):** the controller's checks run passively over the
  transcript + snapshots and emit *memory-hygiene metrics* (redundancy rate, contradiction
  rate, destroyed-fact count, capacity-pressure mishandling), a new evaluation dimension
  BFCL currently lacks.

**Optional dataset extension (BFCL-conflict):** the current prereq conversations are almost
purely additive — facts rarely get revised. A small authored extension where later prereq
conversations *update* earlier facts ("actually we moved to Portland last month"), with
recall questions asking for the current value, would make the admission controller's
contradiction handling measurable. Same file formats, same pipeline; only new data files.

---

## Part C — Failure-mode diagnostics (foundation for the thesis)

BFCL reports one number per backend. The artifacts on disk allow a **decomposition of every
recall failure into four stages**, fully automatable:

| Stage | Test | Data used |
|---|---|---|
| **WRITE-miss** | no ground-truth string ever appears in any prereq checkpoint | `memory_snapshot/prereq_checkpoints/*.json` + `possible_answer.ground_truth` |
| **RETENTION-loss** | fact present in checkpoint k, absent from `{scenario}_final.json` (evicted/cleared/overwritten) | checkpoint sequence |
| **RETRIEVAL-miss** | fact present in final snapshot, but never surfaced in the recall entry's tool results | final snapshot + inference log |
| **ANSWER-miss** | fact retrieved into context, final answer still wrong | inference log + agentic checker |

The per-entry checkpoints even give **fact-survival curves** ("forgetting curves") across
the prereq chain — when in the 5–10 conversations does each fact die, and which tool call
killed it. This instrument is (a) a paper-worthy contribution by itself, (b) the metric
that shows *why* each of Ideas 1–4 helps (Idea 1 should cut ANSWER/WRITE misses, Idea 3
WRITE/RETRIEVAL misses, Idea 4 RETENTION losses), and (c) cheap — a standalone analysis
script over existing result folders, no re-inference needed.

---

## Part D — Proposed thesis (see also DEEP_RESEARCH_PROMPT.md)

**Title:** *Uncertainty-Gated Memory Governance for Tool-Calling LLM Agents: a
Write–Retain–Retrieve–Answer Analysis on the Berkeley Function-Calling Leaderboard*

**Claim:** small/mid open models don't fail BFCL-memory because they can't answer — they
fail because ungoverned memory tool use destroys or hides facts. A model-agnostic
middleware ("Memory Governor") that (1) extracts atomic facts, (2) admits them through
entailment/duplicate/contradiction checks, (3) places and evicts under budget with
mandatory archival, and (4) gates destructive/uncertain operations by tool-action semantic
entropy, recovers a large fraction of the gap — without touching model weights or the
benchmark grader.

**Experimental design:**
1. Baselines: Qwen (small + mid, FC and prompting modes) on `memory` + `multi_turn`.
2. Diagnostic instrument (Part C) → failure taxonomy of the baseline (thesis Chapter:
   "why memory benchmarks are failed").
3. Governor components added cumulatively (ablation: +extractor, +admission, +eviction,
   +entropy gate) — each maps to a specific failure stage, so the ablation table and the
   taxonomy table cross-validate.
4. Generalization: same governor on a second model family; plus `memory_vault` backend
   (Idea 2) as a backend-side comparison arm.
5. Calibration analysis: write-time entropy vs fact survival; answer entropy vs
   correctness (selective answering).

**Why it's a good thesis:** fixed public benchmark, clean baselines, a measurement
contribution (the decomposition) + a systems contribution (the governor) + a scientific
question (does uncertainty predict forgetting), all with modest compute (155×3 entries per
run; sampling only at gated steps).
