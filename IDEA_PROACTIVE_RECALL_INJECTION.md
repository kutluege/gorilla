# IDEA: Proactive Recall Injection (PRI) — closing the read loop the agent never closes

**Date:** 2026-07-20 · **Status:** idea, pre-research · **Scope:** BFCL v4 Memory, KV + Vector backends, `Qwen/Qwen3-4B-Instruct-2507`
**Companion:** `DEEP_RESEARCH_PROMPT_PROACTIVE_RECALL.md` (the research prompt to explore this idea)

---

## 1. The empirical hole every previous method left untouched

Three methods have been tried in this repo — MIG reranking (read-time *ranking*), the
SE gate (write-time *sampling*), the governance cascade (write-time *filtering*). All
three were mechanism-verified and all three were accuracy-neutral. The 2026-07 replicated
campaign explains why: **they all govern actions the model takes, but the score is lost
on actions the model never takes.**

Hard numbers from the committed artifacts (5× baseline replicates, both backends):

| Fact | Value | Source |
|---|---|---|
| Question turns that issue ≥1 retrieval call | **92 / 1550 = 5.9 %** | `gov_logs/ab5_wrra/rep0*_baseline.jsonl` (`questions_with_retrieve`), recomputed 2026-07-20 |
| Memories successfully written per campaign | 731 resolved (of 1753 attempted) | same |
| Official baseline accuracy | 13–18 % (KV/Vector) | `gov_logs/ab5_analysis.json` |
| MIG post-mortem | retrieval invoked 0–1× per 55 questions → reranker never ran | `BFCL_MIG_EXPERIMENT_REPORT.md` |
| Write-side interventions (SE, GOV, 8 ablation arms) | all null on official score | `EVAL_RESULTS.md` |
| student scenario | writes nothing, dead in 24/24 units (~20 % of questions) | `gov_logs/ablations_wrra/` |

The agent *writes* memory tolerably (731 facts sit in the store) and *never reads it*:
94 % of questions are answered from the model's parametric guesswork or "I do not know".
The bottleneck is not ranking quality (MIG), not write quality (SE/GOV) — it is that
**the read loop is simply not executed**. No amount of write-side hygiene can lift a
score that never touches the store.

## 2. The idea in one sentence

**Make memory reading a middleware-guaranteed operation instead of a model-chosen one:
at every question turn, the handler itself retrieves from the store (using the question
text as the query) and injects the results into the model's context before it answers —
the model's only remaining job is answer synthesis.**

This is an agent-design change (a new registry model id, e.g. `...-FC-PRI`), not a
benchmark change: the grader, datasets, and score pipeline stay untouched, exactly like
the `-FC-GOV` / `-FC-SE` / `-FC-MIG` handlers before it. Architecturally it inverts the
previous work: instead of *gating* the model's memory I/O, it *supplies* the memory I/O
the model fails to perform.

## 3. Mechanism sketch (three components, independently ablatable)

### Component 1 — Proactive Recall Injection (the core)

At each question turn (detectable: the harness's question entries vs prereq entries are
distinct phases of the scenario):

1. Middleware takes the **user question text** as query.
2. Runs real-backend-faithful retrieval over the *current* store — the machinery already
   exists and is verified byte-identical: `retrieval_sim.simulate_kv` (BM25Plus over key
   names) and `simulate_vector` (normalized-MiniLM inner product), plus the live mirror
   cache (`GovernanceCache`) that already tracks store state per tier.
3. Injects the top-k results into the model's context before generation. Candidate
   injection surfaces (research question Q3 in the companion prompt):
   a. a synthetic tool-call round: fabricate an assistant `core_memory_search(...)` call +
      its real result in the message history (the middleware already fabricates synthetic
      tool results for NOOP decoys — `synthetic_success` / `patch_results` in
      `governance_filter.py:1291,1829` — the same mechanism, opposite direction);
   b. a system-prompt suffix ("Relevant memories: …");
   c. prefill of the assistant turn with a retrieval scratchpad.
4. The model answers with the memories in view.

Ceiling estimate: survival-conditional accuracy is ~0.20–0.22 while surviving stores
contain the answer content for a much larger fraction of questions (the paraphrase judge
found answers that were right-but-keyword-wrong even *without* retrieval; snapshots show
the facts present). If injection converts even a third of the "fact in store, never read"
questions, the effect dwarfs every ±1 pp seen so far.

### Component 2 — Deterministic Write Scaffold (rescuing dead chains)

The student scenario dies because the model makes zero write calls during prereq turns.
Middleware detects a prereq turn that produced no resolved write and **auto-writes a
deterministic fallback memory**: the user turn text itself (Vector: verbatim;
KV: key = deterministic slug of top content tokens, value = the turn text). This is a
legitimate agent memory policy (MemGPT-style auto-archiving), guarantees chain survival,
and feeds Component 1 (injection needs a non-empty store). Its own risk — store bloat —
is exactly what the existing Stage-0 geometry NOOP already controls; PRI composes with
the validated geometry gate rather than replacing it.

### Component 3 — KV dual-key indexing (backend-specific booster)

KV retrieval is BM25 **over key names only** (`memory_kv.py:71-88`) — values are
invisible to search, so question vocabulary must collide with model-invented key tokens.
At write time, middleware expands the key with 2–3 salient value tokens
(`key='talent_acquisition'` → `'talent_acquisition_goldman_sachs_15'` — the mechanism
already exists as the Stage-2 entity-suffix canonicalizer, retargeted from margin-repair
to recall-alignment; value stays byte-identical, so grader keywords are safe). This
directly widens the BM25 collision surface that Component 1's queries depend on.

## 4. Why this is different from everything tried before

| | MIG | SE gate | GOV cascade | **PRI** |
|---|---|---|---|---|
| Intervenes on | ranking of results the model asked for | write sampling | write admission | **the read action itself** |
| Requires the model to act first | yes (retrieve) | yes (write) | yes (write) | **no** |
| Verified failure mode it attacks | none (retrieval never fired) | none | none | the 94 % no-retrieval gap |
| Extra model calls | pool_size per retrieval | N× per step | 0 | **0** (deterministic retrieval + injection) |

## 5. Honest risks (to be resolved by the deep research)

1. **Context-window interference**: injected memories may distract more than help
   (the "More Documents, Same Length" effect, arXiv:2503.04388) — k must be small and
   possibly margin-gated (inject only when retrieval is confident; the margin–entropy
   machinery from the governance work is reusable here as an *injection* gate).
2. **Benchmark legitimacy**: must remain a handler-side agent policy; the injection must
   not leak ground truth (queries are the question text the model already sees — no
   leakage — but the framing needs a clean argument).
3. **Prompt-format sensitivity**: fabricated tool rounds vs system suffix may interact
   with Qwen's chat template; needs a smoke matrix.
4. **Auto-write pollution**: Component 2 could bury good memories under verbatim turns;
   compose with Stage-0 dedup and cap per-turn auto-writes at 1.
5. **The model may ignore injected content**: 4B models under-attend to context; may need
   an instruction line or answer-prefill nudge — variants to research.

## 6. Evaluation plan (inherits the validated protocol verbatim)

Arms: baseline / PRI-inject-only / PRI+scaffold / PRI+scaffold+dualkey (one mechanism per
arm, R4 discipline). 5× sequential replicates, survival-conditional pairing, exact
McNemar, scenario bootstrap, Holm; official score primary; W-R-R-A + paraphrase judge
secondary; runner/analyzer (`run_gov_replicates.py --arms-json`) needs zero changes.
Pre-registered primary comparison: **PRI-inject-only vs baseline** — the cleanest test of
"the read loop was the bottleneck".

*Full research questions, method survey targets, and prompt-engineering variants are
specified in the companion deep-research prompt.*
