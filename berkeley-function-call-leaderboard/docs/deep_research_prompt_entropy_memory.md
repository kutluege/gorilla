# Deep-Research Prompt — Entropy for Vector Memory Management in LLM Agents

Companion to `ENTROPY_COMPRESSION_PLAN_V3.md`. This prompt is written to be
pasted verbatim into a long-horizon deep-research tool. It deliberately does
**not** import this campaign's frozen rules or constraints: the goal is a broad,
literature-grounded view of whether entropy can improve vector memory
management at all, and at which stage — which then feeds back into (or
falsifies) the Plan-v3 design.

---

## The prompt

**Title: Entropy as an organizing principle for vector memory management in LLM agents**

**Role & goal.** You are a research analyst preparing a literature-grounded
design brief. The question: *Can information-theoretic (entropy-based) signals
meaningfully improve the management of an external vector memory used by an
LLM agent — and if so, at which lifecycle stage?* Do not assume any particular
system's constraints; survey the space broadly, then synthesize design
recommendations.

**Background context (one paragraph — context, not a constraint).** In one
experimental system (BFCL v4 Memory benchmark, small open model, BM25 and
FAISS/MiniLM memory backends), entropy features computed over retrieval-score
distributions (softmax entropy, ΔH interference, effective neighborhood size)
and over embedding Gram matrices (von Neumann entropy / Vendi effective rank)
repeatedly failed to predict *harmful writes* beyond geometric redundancy and
retrieval margin — out-of-sample, under negative-control shuffles, and across
two independent entropy definitions at two representational layers. The open
hypothesis is that entropy's proper role is **compression and capacity
management** (what to merge, evict, or summarize; how much information a store
effectively holds), not per-write harm prediction.

**Research areas to cover** (each with citations to papers, preprints, and
credible engineering writeups; distinguish peer-reviewed vs preprint vs blog):

1. **Entropy & compression foundations applied to memory.** Rate–distortion
   framing of memory stores; Shannon entropy vs semantic/embedding entropy;
   matrix-based entropy (von Neumann entropy, Vendi score, effective rank) as
   diversity/capacity measures for embedding collections; connections to
   determinantal point processes and coreset/prototype selection.
2. **Capacity of embedding-based retrieval.** Theoretical and empirical limits
   of dense retrieval stores (embedding capacity limits; "more documents hurt
   retrieval"; hubness in high dimensions). Does store entropy / effective rank
   predict retrieval degradation better than raw item count?
3. **Memory consolidation & forgetting in LLM agent systems.** Survey agent
   memory systems (MemGPT/Letta, Generative Agents reflection, HippoRAG, Mem0,
   A-Mem, MemoryBank, recursive summarization): what signals do they use to
   consolidate, summarize, or forget — anything information-theoretic, or only
   heuristics (recency, frequency, LLM-scored importance)? What evaluation
   evidence ties consolidation to downstream task accuracy?
4. **Model-side uncertainty coupled to store-side decisions.** Semantic entropy
   (Kuhn/Farquhar), aleatoric-vs-epistemic decomposition (e.g., Yadkori et
   al.), predictive-entropy-based selective generation — has any work coupled
   *model-side* entropy with *memory-side* decisions (write gating, retrieval
   reranking, summarization triggers)?
5. **Entropy in vector-database / ANN practice.** Quantization (PQ/OPQ/scalar),
   dimensionality reduction, deduplication and near-duplicate clustering at
   scale — where does entropy or information content explicitly drive index
   compression, and what recall/accuracy trade-offs are reported?
6. **Selective prediction & risk–coverage as the evaluation lens.**
   Geifman–El-Yaniv risk-coverage methodology applied to memory operations
   (write/merge/evict as abstainable decisions); precedents for evaluating
   memory compression by downstream QA accuracy rather than intrinsic metrics.
7. **Negative and null results.** Actively search for reports where
   entropy/uncertainty signals failed to add value over simpler geometric or
   frequency baselines (ablation tables count). The failure modes matter as
   much as the successes.

**Synthesis requirements:**

- A **stage-by-stage map** of the memory lifecycle (write → placement →
  consolidation → eviction → retrieval → summarization), annotated with: which
  entropy-family signals have evidence at each stage, the strength of that
  evidence (benchmarked / anecdotal / theoretical), and known failure
  conditions.
- An **explicit verdict** on the hypothesis: "entropy adds value at the
  compression/consolidation stage even where it adds none at per-write
  admission" — supported, contradicted, or untested by the literature.
- **Concrete candidate mechanisms ranked by evidence strength**, each with:
  the signal definition, the decision it drives, expected effect-size context,
  and the cheapest falsifying experiment.
- A short list of the **10–15 most load-bearing references** with one-line
  takeaways.

**Style constraints:** cite as you go; prefer 2023–2026 sources for
agent-memory systems but include the foundational older work (rate–distortion,
hubness, DPPs, selective prediction); flag anything that looks like
citation-farming or unreplicated claims.

---

## How the answer feeds back into Plan v3

- Area 2 evidence bears directly on hypothesis H-C and the deferred
  effective-rank budget (mechanism c).
- Area 3 tells us whether *any* published system drives consolidation with an
  information-theoretic signal — if none does, Plan v3's Stage-C is a genuine
  novelty claim candidate; if some do, their evaluations set the effect-size
  prior.
- Area 7 (nulls) calibrates expectations against this campaign's own G1/G2
  negatives and protects the thesis narrative from survivorship bias.
