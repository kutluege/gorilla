# DEEP RESEARCH PROMPT — Proactive Recall Injection for Agentic Memory Benchmarks

*Paste everything below this line into a deep-research tool. Companion context:
`IDEA_PROACTIVE_RECALL_INJECTION.md` in this repo.*

---

## Role and mission

You are researching methods for a thesis experiment on the Berkeley Function Calling
Leaderboard (BFCL) v4 Memory benchmark. In this benchmark an LLM agent (Qwen3-4B-Instruct,
served via vLLM, temperature ≈ 0) must (a) write facts into a persistent memory store
(key-value store searched by BM25 over key names only, or vector store searched by
normalized MiniLM embeddings) across 37 prerequisite conversation turns, then (b) answer
155 recall questions using that store via retrieval tool calls. The grader is a
case-insensitive keyword-substring match against gold answers.

**The measured failure mode that motivates this research:** across 5 replicated baseline
runs, the agent issued at least one retrieval call on only **5.9 % of question turns**
(92/1550), while ~730 facts sat successfully written in the store. Write-side
interventions (semantic-entropy write gating, geometric/NLI/margin write governance) were
all statistically null on the official score. The proposed intervention — **Proactive
Recall Injection (PRI)** — makes the *middleware* (the model handler that wraps the LLM,
where tool calls are decoded and results injected) perform retrieval deterministically on
every question turn, using the question text as the query, and injects the top-k results
into the model's context before it answers. Two supporting components: a deterministic
write scaffold (auto-write the user turn verbatim when a prerequisite turn produced zero
writes, guaranteeing a non-empty store) and KV dual-key indexing (expand model-invented
keys with salient value tokens so BM25-over-keys can match question vocabulary).

Your mission: survey the research and engineering landscape so we can design PRI's
details with evidence rather than guesses, and identify every prior method we should
borrow from, compare against, or cite.

## Constraints (binding — filter every recommendation through these)

- **No extra LLM calls at decision time.** The middleware may embed text with MiniLM and
  run BM25/dot-product retrieval, but may not call the main model or another LLM per
  question (deterministic, ~milliseconds budget). Methods requiring sampling, self-ask
  loops, or auxiliary LLM judges are citable as related work but not adoptable.
- **The benchmark is untouchable**: grader, datasets, question text, and score pipeline
  stay as-is. The intervention lives entirely in the model handler (a legitimate "agent
  design"), registered as a separate model id.
- **No ground-truth leakage**: retrieval queries may use only the question text the model
  already sees and the store the agent itself built. Gold answers/possible-answer files
  must not influence any online decision.
- **Small model**: Qwen3-4B — assume weak instruction-following, weak context attention,
  and no fine-tuning budget.
- Evaluation is fixed: 5 sequential replicates, McNemar on survival-conditional pairs,
  Holm correction, official score primary.

## Research questions

### Q1 — Where should injected memories enter the context, and in what form?
Survey evidence on how retrieved/injected content placement affects small-LLM answer
accuracy:
- fabricated tool-call rounds (assistant "called" search; result message follows) vs
  system-prompt suffix vs user-message prefix vs assistant-turn prefill;
- "lost in the middle" positional effects (Liu et al. 2023) and their interaction with
  chat templates — specifically any evidence for Qwen-family templates;
- verbatim quoting vs structured bullet lists vs JSON blobs for injected memories;
- whether labeling injected content ("Your memory returned: …") vs presenting it as a
  genuine tool result changes model trust/usage;
- prompt patterns that make small models *copy* retrieved spans (relevant because the
  grader is keyword-substring: verbatim copying is directly rewarded);
- risks: fabricated-round incoherence, template token collisions, context overflow.
Deliverable: a ranked shortlist of 3 injection formats with cited evidence and predicted
failure modes, plus a recommended smoke-test matrix.

### Q2 — How many memories to inject, and should injection be confidence-gated?
- Evidence on retrieval-set size vs answer quality at fixed context length
  ("More Documents, Same Length", arXiv:2503.04388; distractor-sensitivity literature) —
  optimal k for a ~20–60-item store;
- selective injection: inject only when top-1 margin/score clears a threshold vs always
  inject vs inject-all (retrieve_all for stores ≤ 20 items). Relate to selective
  prediction (Chow 1970; Geifman & El-Yaniv 2017) and query-performance-prediction
  statistics (NQC family) as the gating signal — we already have deterministic margin +
  score-entropy machinery;
- fallback ordering when the store is large: recency vs score vs tier (core vs archival);
- evidence on whether irrelevant injected memories cause *wrong* answers (harm) vs merely
  no help — the asymmetry determines whether gating is needed at all.
Deliverable: a k + gating recommendation with the evidence chain, and the ablation arms
needed to validate it.

### Q3 — Query construction from the question turn
The KV backend searches key names only; the vector backend searches stored text. Survey:
- query rewriting/expansion **without an LLM**: keyword extraction (RAKE/YAKE/TextRank),
  stopword-stripped token queries, entity extraction via deterministic NER, pseudo-
  relevance feedback (RM3-style) over the store itself;
- multi-query fusion: issuing {raw question, keyword query, entity query} and fusing by
  reciprocal-rank fusion — cost is trivial at these store sizes; evidence for RRF
  robustness;
- BM25-specific: matching question vocabulary to *key-name* vocabulary — token-overlap
  tricks, sub-token normalization (underscore splitting already matches the backend's
  tokenizer), and whether our dual-key write-time expansion is the better fix (compare
  against read-time query expansion — which side of the index should adapt?);
- embedding-side: query prefixing/instruction tricks that help MiniLM-class encoders
  match questions to statement-form memories (e.g., "query: " prefixes, hypothetical-
  document embeddings are LLM-based and thus excluded — note the exclusion explicitly).
Deliverable: a deterministic query-construction recipe per backend with citations.

### Q4 — Auto-writing fallback memories (the write scaffold)
- Prior art on middleware/framework-side automatic memory capture: MemGPT/Letta
  auto-archival, LangChain/LlamaIndex conversation-memory buffers, agent frameworks that
  persist raw turns vs extracted facts;
- verbatim-turn storage vs deterministic fact extraction (open IE without an LLM:
  Stanford OpenIE, dependency-pattern extractors) — quality/coverage trade-off for
  recall benchmarks whose graders are substring-based (verbatim storage preserves
  keywords by construction — argue for/against);
- key-name generation for KV without an LLM: slugification strategies, collision
  handling, and their interaction with BM25-over-keys retrieval;
- store-pollution risk: how do memory systems bound auto-capture (dedup, caps,
  importance thresholds)? Our existing geometric NOOP gate is available — cite analogous
  compositions;
- any published evidence on "guaranteed write" scaffolds changing downstream QA accuracy.
Deliverable: a scaffold policy (what to write, when, with what key) + pollution controls.

### Q5 — Prior art and novelty positioning
Map the landscape so the thesis can position PRI precisely:
- Systems where retrieval is *always on* vs agent-initiated: classic RAG (always-on) vs
  tool-calling agents (agent-initiated) — is there published work explicitly comparing
  the two regimes on the *same* agentic memory benchmark? BFCL v4 Memory results
  published anywhere (leaderboard entries, papers, blog posts) — what do top models do
  differently on these categories?
- Proactive/anticipatory retrieval literature (retrieve-before-generate, FLARE's
  anticipation flipped from "when" to "always at question turns");
- memory-augmented agent benchmarks (LoCoMo, MemBench, LongMemEval, letta-bench …):
  reported failure modes — is "agent never retrieves" documented elsewhere? Any
  benchmark-reported read-compliance statistics;
- handler/agent-side scaffolds published for BFCL specifically (any model whose handler
  injects retrieval automatically — check top BFCL v4 memory leaderboard model handlers
  in the gorilla repo itself, since handlers are open source);
- position: is "middleware-guaranteed memory I/O for tool-calling agents" novel, or does
  it reduce to known RAG-with-tools patterns? Name the closest 3 works and the exact
  delta.
Deliverable: related-work map + a defensible novelty statement (or an honest "this is
engineering, not novelty" verdict with the strongest citable framing).

### Q6 — Failure-mode anticipation
- When does injected context make small models *worse*? Sycophancy toward irrelevant
  context, distraction, copy-of-wrong-span errors; quantitative studies preferred;
- interaction with the model's own retrieval: if the model *does* call retrieve after
  injection, results double up — dedup policy;
- prereq turns: should injection also run during prerequisite (write-phase) turns (e.g.,
  to surface existing memories and reduce duplicate writes), or is that scope creep with
  measurable risk?
- what telemetry should be logged per injection to make post-hoc failure analysis
  possible (query, scores, k, injected ids, whether the answer used injected tokens)?
Deliverable: pre-registered failure-mode list with detection metrics for each.

## Output format

Produce a structured report:
1. **Executive recommendation** — the concrete PRI v1 design (injection surface, k,
   gating rule, query recipe per backend, scaffold policy) with confidence levels;
2. **Per-question findings** (Q1–Q6), each with: evidence summary, primary-source
   citations (title, authors, year, venue/arXiv id, URL), and an explicit
   transfers/does-not-transfer verdict under the constraints above;
3. **Ablation matrix** the evidence says we need (arms + expected effect direction);
4. **Risk register** — every identified failure mode with a detection metric;
5. **Novelty statement** — 3 closest works and the delta, phrased defensibly for a
   thesis;
6. **Open questions** the literature cannot answer, which only our experiment can.

Prefer primary papers and official repositories; mark clearly which claims are verified
from sources vs. inferred. Where the literature is silent on our exact setting (4B model,
substring grader, BM25-over-key-names), say so explicitly rather than extrapolating.
