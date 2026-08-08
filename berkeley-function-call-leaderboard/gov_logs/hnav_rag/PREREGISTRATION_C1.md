# RAG program, campaign C1 (READ) — pre-registration

**Frozen 2026-08-08, BEFORE the campaign runs.** Program context:
`gov_logs/hnav_autonomous/CORRECTED_BASELINE.md`,
`tier_conditional_pooled.json`, scaffold `PREREGISTRATION.md` AMENDMENT 1.
This is the first campaign of the RAG (capture→read) program — the Stage-5
extension beyond the original directive's ten-slot ledger, with method M1
also filling directive slot #4 (retrieval-aware action selection, §13,
re-specified on repository evidence per §9).

## Method under test — M1, deterministic retrieve-then-generate read scaffold

`Qwen/Qwen3-4B-Instruct-2507-FC-RAG` (`QwenRagHandler`): on a memory
**question** entry, when the model's response decodes to no tool calls and
the turn has not been read-scaffolded, inject exactly one archival read —
vector `archival_memory_retrieve(query=<verbatim user turn>, top_k=5)`,
kv `archival_memory_key_search(query=<verbatim user turn>, k=5)`. The
harness executes it; the model answers with the result in context.

**Motivation (frozen evidence):** gold carried only by archival converts at
0.000 (vector, 0/74) / 0.017 (kv, 1/60) vs core-carried 0.701 / 0.556,
because the agent issues an archival read on 1 of 155 question entries.
Recall@5 of the gold carrier with the verbatim question as probe is 22/22
on the existing baseline stores.

## Arms (5 replicates, `memory_kv,memory_vector`, temperature 0.001)

| arm | model | env | purpose |
|---|---|---|---|
| `baseline` | `-FC` | — | reproduced fair baseline, co-run |
| `read_verbatim` | `-FC-RAG` | `RAG_MODE=read_verbatim` | M1 |
| `read_irrelevant` | `-FC-RAG` | `RAG_MODE=read_irrelevant` | identical call, identical extra step and context growth, frozen generic probe with no question-specific content |
| `read_instruction` | `-FC-RAG` | `RAG_MODE=instruction_only` | frozen system-prompt nudge, no injection — separates deterministic mechanism from prompting |

`RAG_TOP_K` frozen at 5 (the backend default). Runner scrubs `RAG_*` from
inherited env (commit `11e5d16`); arms declare their env explicitly.

**Recorded limitation:** at baseline archival sizes (≈2–19 entries, often
≤ 5) the retrieved set for any probe is most or all of the store, so C1's
verbatim-vs-irrelevant contrast mostly isolates the *step/volume* effect,
not probe *relevance*; relevance becomes discriminative in C2+ where capture
fills the index. C1's primary product is the **read-conversion factor c** =
p(correct | archival-only-carried, read injected), which no offline
instrument can produce and which prices every later capture method.

## Hypotheses

- **H-C1a (primary).** `read_verbatim` > `baseline` on official accuracy,
  matched questions, per backend; vector is the primary backend. Effect
  prediction: archival-only stratum (74/930 vector) converts at c ≥ 0.30 →
  ΔAcc_vector ≥ +0.024.
- **H-C1b (mechanism).** `read_verbatim` ≥ `read_irrelevant` >
  `baseline` ordering; and `read_instruction` quantifies how much of the
  effect mere prompting recovers.
- **H-C1c (strata).** In the read arms, p(correct | archival-only) rises
  from ~0 toward the core conditional; measured via
  `hnav_answer_index.py --validate` by-tier on each arm.

## Analysis (frozen)

`analyze_gov_replicates.py --reference-arm baseline`, exact McNemar +
cluster bootstrap over (replicate, scenario), Holm across backend ×
arm-pair (m = 6). Headline: student-INCLUDED (`--include-student`),
student-excluded as sensitivity (analyzer default run also reported).
Per-replicate direction table mandatory. Success = CI excluding zero AND
consistent direction across ≥ 4 of 5 replicates on vector.
Step-budget guard: force_quit > max(3× baseline, 2/arm-rep) invalidates.
Strict answer-field-only regrade reported alongside the lenient metric for
every arm. Cost accounting per §22.

## Falsifier status (already run, offline, zero GPU)

Stratum exists (n=74 vector, p=0.000), carrier recall@5 = 22/22 → GO.
The unknown is c, and that is what this campaign measures.
