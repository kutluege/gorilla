# Deep Research Prompt — Memory Governance for Tool-Calling LLM Agents on BFCL

> **How to use this file:** give the entire contents to a research agent (deep-research
> tool, or a capable LLM with web access). It is self-contained: it explains the
> benchmark, the observed failure modes, the four candidate ideas, the proposed thesis
> direction, and the exact questions the research should answer.

---

## 1. Role and mission

You are a research assistant supporting a master's thesis on **improving the memory and
multi-turn tool-calling performance of LLM agents**, evaluated on the **Berkeley
Function-Calling Leaderboard (BFCL v4)** (`gorilla` repository,
`berkeley-function-call-leaderboard/`). Your job is to (a) map the state of the art
relevant to each idea below, (b) find what is genuinely novel vs already published,
(c) recommend concrete methods, models, and metrics, and (d) flag risks and evaluation
pitfalls. Cite primary sources (papers, benchmark reports, system docs) for every
substantive claim, prefer 2023–2026 work, and clearly separate established results from
speculation.

## 2. Benchmark context (facts about BFCL v4 — treat as ground truth)

* **Memory category:** 155 recall questions × 3 memory backends (`memory_kv`,
  `memory_vector`, `memory_rec_sum`), 5 scenarios (customer, student, finance, healthcare,
  notetaker). Before the questions run, the model holds 5–10 long "prerequisite"
  conversations per scenario in which a user volunteers dense personal facts; the model
  must *proactively* store facts using memory tools. Memory state is snapshotted to disk
  after every prereq conversation and reloaded for the next one and for the recall
  questions.
* **Backends and limits:** KV store (core: max 7 entries × 300 chars, always in context;
  archival: 50 × 2000, retrievable only by exact key or BM25 **over key names**); vector
  store (same limits, MiniLM+FAISS cosine retrieval); recursive-summary (single 10,000-char
  text blob with append/update/replace/clear).
* **Grading:** the final natural-language answer must contain a ground-truth string
  (whole-word substring match after normalization). No partial credit; storage/retrieval
  behavior is not directly scored — only end-to-end recall.
* **Multi-turn categories:** 200 entries each (base, miss_func, miss_param, long_context)
  over simulated APIs (file system, trading, travel, vehicle, Twitter…); graded by exact
  **backend state match** plus execution-result containment after every turn; up to 20
  model steps per user turn.
* **Observed baseline failures (Qwen, own experiments):** when core memory fills up
  (7-entry cap), the model deletes or clears entries **without archiving them first**;
  it also frequently fails to write facts at all; on `rec_sum` it overwrites the whole
  blob with `memory_update`. Failures are therefore mostly *memory-governance* failures,
  not question-answering failures.
* **Extension seams (already verified in the code):** model-side handler middleware
  (sampling, context injection, tool-call rewriting) needs no benchmark changes; new
  memory backends can be registered and automatically inherit the whole
  prereq/snapshot/scoring pipeline; per-conversation memory snapshots on disk permit a
  full write→retain→retrieve→answer failure decomposition.

## 3. Candidate ideas to research (from the thesis author)

**Idea 1 — Semantic entropy as a memory/tool-call validator.** Sample the model N times;
cluster answers by meaning (bidirectional entailment) and tool calls by
(tool name + semantically-normalized arguments). High cluster entropy ⇒ the model is
uncertain ⇒ gate the action (deliberate, choose safe default, or abstain); low entropy ⇒
commit majority action. Apply at write time (which memory op? which key/value?) and at
recall time (which answer?). Also: a fact is worth storing if it *reduces* semantic entropy
of answers to related future queries.

**Idea 2 — Vault-LD: markdown wiki as episodic memory + ontology/knowledge graph as
semantic memory.** Markdown notes with YAML-LD frontmatter combine human/LLM-readable text,
machine-queryable structured fields, and linked-data semantics; the note store gradually
becomes a queryable, validatable knowledge graph capable of inference. In BFCL terms: a new
`memory_vault` backend with note tools + triple/ontology tools, evaluated on the same 155
questions.

**Idea 3 — NER + fact extraction with importance scoring.** Decompose each user turn into
atomic facts (entities, numbers, measurements, dates, locations, relations, durable
preferences) with importance/confidence/usability scores; store structured, verbatim-value
entries under entity-bearing keys; at retrieval and tool-calling time, rank stored facts as
candidate tool parameters (e.g., reuse a stored "35 m² kitchen" as the area parameter of a
later cost-calculation call).

**Idea 4 — Entropy- and entailment-aware memory admission controller.** Before writing:
check semantic equivalence, logical entailment (reject facts already implied; replace with
strictly-more-specific facts), contradiction (route to update, not add), and a
decision-theoretic information-gain score (write only if it improves expected future
behavior, penalizing storage/retrieval cost, contradiction risk, low confidence, temporal
ambiguity). Result: a compact, high-value, temporally-grounded memory instead of a raw
transcript dump.

## 4. Synthesis under investigation (proposed thesis)

**"Uncertainty-Gated Memory Governance for Tool-Calling LLM Agents"** — a model-agnostic
middleware (the *Memory Governor*) combining: turn-level fact extraction (Idea 3) →
entailment/duplicate/contradiction admission (Idea 4) → budget-aware placement and eviction
with **mandatory archival before any deletion** → semantic-entropy gating of destructive or
uncertain operations (Idea 1); optionally compared against a structured `memory_vault`
backend (Idea 2). Measurement contribution: a **Write / Retain / Retrieve / Answer failure
decomposition** computed from BFCL's on-disk memory snapshots, including per-fact
"survival curves" across the prerequisite conversations. Scientific question: **does the
model's own uncertainty at write time predict which facts it will later fail to recall?**

## 5. Research questions (answer all; cite sources)

### A. Semantic entropy & uncertainty gating
1. State of the art on semantic entropy (Kuhn/Farquhar et al.), semantic-entropy probes,
   and cheaper single-pass approximations. How well do they transfer from free-form QA to
   **structured outputs (tool calls)**? Any prior work on uncertainty over function-call
   decisions, action-level self-consistency, or abstention in agents?
2. Practical clustering of tool calls: exact vs semantic argument matching, canonical
   argument normalization — what have agent-verification papers (e.g., tool-call
   verifiers, process reward models for agents) done?
3. Sampling cost control: adaptive sampling (only sample more when first two disagree),
   entropy from token logprobs vs cluster entropy — evidence on trade-offs.
4. Any published evidence that write-time uncertainty predicts downstream retrieval
   failure? (If none: this is our novelty claim — verify it is unclaimed.)

### B. Agent memory systems
5. Survey memory architectures for LLM agents: MemGPT/Letta, Mem0, Zep/Graphiti, A-MEM,
   HippoRAG(-2), Cognee, LangMem, MemoryBank, generative-agents reflection, MIRIX,
   MemOS, etc. For each: admission policy, eviction policy, conflict handling, structure
   (flat / KV / graph / hierarchical), and any *quantitative* results on memory benchmarks.
6. Which existing systems already do entailment- or contradiction-checked admission?
   (Mem0's ADD/UPDATE/DELETE/NOOP decision, Zep's temporal edges with validity intervals,
   knowledge-graph fact invalidation…) — precisely delineate what Idea 4 adds beyond them
   (decision-theoretic information-gain objective, uncertainty gating, mandatory-archival
   eviction under hard capacity budgets).
7. Memory benchmarks other than BFCL: LoCoMo, LongMemEval, MemBench, LTM-Benchmark,
   PrefEval — what failure taxonomies do they use? Has anyone published a
   write/retain/retrieve/answer decomposition or forgetting-curve analysis for agent
   memory? 
8. What is known about the BFCL v4 memory category itself (blog posts, papers, leaderboard
   analyses): published scores, known weaknesses of the category, top-model strategies.

### C. Structured / ontological memory (Idea 2)
9. Prior art on markdown/wiki-based agent memory (Obsidian-style vaults, Vault-LD,
   YAML-LD), JSON-LD in LLM pipelines, and ontology-validated LLM knowledge bases
   (SHACL-style validation of LLM-extracted triples). Feasibility and failure modes of
   letting small models write valid frontmatter/triples.
10. Evidence comparing structured (KG/entity-centric) vs unstructured (vector) memory for
    personal-fact recall — HippoRAG, GraphRAG, Zep results; when do graphs actually help
    (multi-hop? temporal? aggregation?), and do those question types occur in
    BFCL's 155 memory questions?

### D. Fact extraction & parameter binding (Idea 3)
11. Best current methods for open-schema atomic-fact extraction from conversation
    (LLM-based propositionizers, GLiNER-style zero-shot NER, dense-X-retrieval-style
    propositions). Accuracy/latency on consumer hardware; suitability for verbatim-value
    preservation (numbers, units, dates).
12. Importance/memorability scoring of facts: existing salience models for
    memory-augmented agents (e.g., generative-agents importance scores, Mem0 salience,
    MemoryBank forgetting curves). Anything trainable from question-source supervision
    (we have per-question `source` sentences in BFCL's answer key)?
13. Slot-filling / parameter reuse across turns: work on carrying entities from dialogue
    memory into API arguments (task-oriented dialogue state tracking meets function
    calling; relevant to BFCL `multi_turn_miss_param`).

### E. Evaluation methodology & thesis positioning
14. Methodological critiques of substring-match grading for memory recall; how other
    benchmarks grade paraphrase-tolerant recall; risks that middleware "teaches to the
    grader" (verbatim storage) and how to report this honestly.
15. Legitimacy and precedent of *harness-side* improvements on public leaderboards
    (contamination/fairness concerns, "scaffolding vs model" debates); how should the
    thesis report model-only vs model+governor scores?
16. Ablation and statistics for 155-question categories: minimum detectable effect,
    per-scenario variance, multiple-run protocols for stochastic sampling — recommend a
    statistically sound experiment plan (bootstrap CIs, McNemar's test on paired
    per-question outcomes, etc.).
17. Compute planning: for a 7B–32B Qwen on vLLM, estimate the cost multiplier of N-sample
    entropy gating restricted to memory-tool steps and recall answers, vs full best-of-N.

### F. Novelty scan (critical)
18. Search specifically for prior work combining **uncertainty estimation with memory
    write/eviction decisions** in LLM agents, and for **memory-governance middleware
    evaluated on BFCL**. List anything close (including 2025–2026 preprints), and state
    plainly which of our four ideas + the failure-decomposition instrument remain novel,
    partially novel, or already done.

## 6. Deliverable format

Produce a structured report with: (1) an executive summary with a novelty verdict per
idea; (2) per-question findings with citations; (3) a recommended method stack (extraction
model, NLI model, entropy estimator, graph store) with alternatives at low/medium compute;
(4) a risk register (evaluation pitfalls, benchmark quirks, reviewer objections); (5) a
prioritized experiment roadmap for a 6-month thesis timeline; (6) full bibliography.

## 7. Constraints and priorities

* Everything must remain **runnable on the BFCL harness in this repository** — prefer
  handler-middleware and new-backend designs over grader changes; any dataset extension
  (e.g., conflicting-facts prereqs) must reuse the existing file formats.
* Target models are open-weight (Qwen family first) served via vLLM on a single node;
  auxiliary models (NLI, NER, embedders) must run on CPU or a small GPU slice.
* The thesis's fixed comparison is: baseline model vs model+middleware vs model+new
  backend, all on unchanged BFCL scoring, plus the diagnostic decomposition as secondary
  metrics.
* When in doubt, prioritize: (A) and (B) questions > (F) > (D) > (C) > (E).
