# MIG Reranker — Live Experiment Results & Fine-Tuning Recommendation

**Model:** `Qwen/Qwen3-4B-Instruct-2507` (served live via vLLM at `localhost:8000`, the same
model the `-FC-MIG` handler targets).
**Harness:** `bfcl_eval/scripts/mig_live_experiment.py`
**Date:** 2026-06-27

---

## 1. What this experiment is (and is not)

This is a **controlled retrieval experiment**, not the full `bfcl generate` run. (The machine
running this analysis lacks `faiss` / `sentence-transformers` / `transformers`, so the
agent-driven retrieval + snapshot generation can't execute locally — only the HTTP model
endpoint is reachable.) It exercises the **real** components end to end:

- real `MIGReranker` code (`parse_candidates` / `select` / `_judge` / `_mig_logprob` / `reserialize`),
- real BFCL agentic scorer (`agentic_checker`, ground-truth substring match),
- real benchmark data (`BFCL_v4_memory.json` questions + `possible_answer` gold `source`/`ground_truth`),
- the live Qwen model (completions + echo-logprobs).

**Per question:** build a noisy pool = the gold `source` sentence (contains the answer) +
distractor `source` sentences from *other* questions in the same scenario. With
`--distractors hard`, distractors are the **most lexically similar** other facts — the
realistic case where a retriever surfaces similar-looking but wrong entries.

**Arms:** `full_pool` (answer from all candidates, no trim), `sim_topk` (keep top-k by
similarity — `scorer=similarity`), `mig_judge_topk` (`scorer=judge`), `mig_logprob_topk`
(`scorer=logprob`). Budget `k` candidates kept from a pool of `P`.

**Metrics:** `gold_recall@k` (did the gold survive the trim?) and `answer_acc`
(ground-truth found in the model's final answer, exactly the official metric).

---

## 2. Headline results

### Answer accuracy

| setting (N) | full_pool | sim_topk | **mig_judge** | mig_logprob |
|---|---|---|---|---|
| customer, random distractors, P10/k3 (30) | 0.867 | 0.833 | **0.867** | 0.833 |
| customer, **hard** negatives, P10/k3 (30)  | 0.867 | 0.767 | **0.900** | 0.867 |
| healthcare, **hard** negatives, P10/k3 (25)| 1.000 | 0.920 | **1.000** | 0.960 |
| customer, **hard**, tight budget P12/**k1** (30) | 0.867 | 0.733 | **0.867** | 0.833 |

### Gold-selection recall@k

| setting | sim_topk | **mig_judge** | mig_logprob |
|---|---|---|---|
| customer, random, P10/k3   | 0.967 | **1.000** | 0.933 |
| customer, hard, P10/k3     | 0.900 | **1.000** | 1.000 |
| healthcare, hard, P10/k3   | 0.920 | **1.000** | 0.880 |
| customer, hard, P12/k1     | 0.800 | **1.000** | 0.967 |

**Cost (per run, ~25–30 s, 8 workers):** ~900–1260 model calls, ~0.17 s/call. The judge arm
adds `pool_size` calls + 1 draft + 1 answer per question; logprob adds `2·pool_size`
(`max_tokens=0`) calls + 1 draft + 1 answer.

---

## 3. Is the method working? — Yes.

1. **MIG-judge reranking is the best arm in every cell.** It keeps the answer-bearing memory
   **100% of the time** (gold_recall@k = 1.000 everywhere), versus 0.80–0.97 for the
   similarity baseline.
2. **It improves end-task accuracy where it matters.** Under hard negatives, the similarity
   trim actually *hurts* (customer 0.767, tight-budget 0.733 — worse than the noisy full pool),
   because a weak retriever drops the gold when forced to a small budget. MIG-judge recovers
   it and lifts accuracy to 0.900 / 0.867 — at or above the full pool, with a much smaller,
   cleaner context.
3. **The effect grows as the budget tightens** (P12→k1) and as distractors get harder —
   exactly the regime where context management matters.
4. **It never hurts** in the easy (random-distractor) case: judge matches full-pool accuracy
   while using 3 entries instead of 10.

### Important caveat (interpret honestly)

The similarity baseline here uses a **lexical** proxy; the real backend uses MiniLM
embeddings, which are stronger. So `sim_topk` numbers are a *pessimistic* stand-in and the
real-pipeline margin of judge-over-retriever will likely be **smaller** than shown. The
robust, well-supported claims are: (a) the judge reranker selects the gold near-perfectly,
(b) it does not hurt and helps under tight budgets / hard negatives, (c) it is training-free.
N is 25–30 per cell, so few-point differences are within noise — the *direction* is
consistent across four independent cells, which is the signal.

---

## 4. A real bug this experiment caught (and the fix shipped)

The **flat logprob-MIG** scorer initially scored **0.000** gold-recall — it ranked the gold
*last* every time. Root cause: it drafted `a_hat` with **empty context**, so `a_hat` was the
model's *prior guess* (e.g. "Caramel latte." when the answer was "strawberry matcha", or "I
don't have enough context"). Adding the gold memory then *lowers* the probability of that
wrong guess, so `MIG(gold) = logp(wrong | gold) − logp(wrong | ∅)` is strongly negative while
irrelevant distractors leave the wrong guess unchanged (MIG ≈ 0). Gold sinks to the bottom.
This is the "wrong-draft" failure mode flagged in the original risk table.

**Fix (now the default):** draft `a_hat` **conditioned on the candidate pool**
(`draft_from_pool=True`). The draft becomes the model's best answer given the evidence
("Strawberry matcha latte.", "Seven.") and logprob-MIG then measures which entry *supports*
that answer. Gold jumped to **rank 1/10** and recall to **0.88–1.00**. A second bug — a single
`None` logprob permanently disabling the scorer — was also fixed (now it degrades per-candidate
and only disables if the endpoint structurally lacks logprobs). Both fixes are in
`middleware/mig_reranker.py`; offline tests still pass 27/27.

**Takeaway:** prefer `scorer=judge` as the v1 default (simplest, best, no draft-quality
landmine). `scorer=logprob` is competitive *only* with the pool-conditioned draft.

---

## 5. Should you fine-tune? Where is the remaining error?

**The reranker itself does not need fine-tuning** — the zero-shot judge already hits
gold_recall@k = 1.0. Spending GPU to "train a better selector" is not where the points are.
The residual errors live in two other places:

1. **Answer generation from correct memory.** Even when the gold is in context, accuracy is
   ~0.87 (customer) — i.e. **~13% of the time the model has the right memory but still answers
   wrong or mis-formats** the `{'answer':…, 'context':…}` payload the checker scans.
2. **Retriever recall (not measured here).** In the real pipeline the embedding retriever may
   never surface the gold; the reranker can only reorder what it's given. This is the true
   ceiling and this experiment deliberately holds it at 100%.

### Recommendation — fine-tune the *generator/agent*, not the reranker. Worth it: **yes, with LoRA.**

| Option | Worth it? | Why |
|---|---|---|
| **LoRA SFT of the agent on memory QA** | **Yes — highest ROI** | Closes the "right memory → wrong/misformatted answer" gap (~10–15%) and enforces the exact agentic answer format the substring checker needs. Cheap on a 4B model (single GPU, hours). BFCL already supports LoRA adapters via vLLM (commit `dac44e7`), so you can eval the adapter with the existing `-FC-MIG` handler. |
| **Distill the judge into a tiny reranker (LoRA/cross-encoder)** | Only at scale | The judge costs `pool_size` LLM calls per retrieval. If latency/cost matters, distill judge scores into a small reranker so it's one forward pass. This is a **cost** optimization, not accuracy — skip it until the per-call cost actually bites. |
| **Fine-tune for selection quality** | **No** | Zero-shot judge is already 1.0; nothing to gain. |
| **Fine-tune the memory embedder (MiniLM)** | Separate track | Would raise the retriever ceiling, but it's the memory *backend* (off-limits for "pure BFCL") and a different project than the agent. |

**Concrete plan if you fine-tune:**
- **Data:** generate SFT traces from the BFCL memory data — `(retrieved memory entries, question) → {'answer': <ground_truth>, 'context': <short justification>}`. A few hundred to low-thousands of examples across the 5 scenarios; the gold `source` + `ground_truth` give you supervision for free.
- **Method:** LoRA (rank 8–16) SFT on `Qwen3-4B-Instruct-2507`, target the answer/format behavior. Keep temperature 0 at eval.
- **Eval:** run `bfcl generate`/`bfcl evaluate` on `memory_vector` with and without the adapter, *and* with/without MIG, as separate arms. The reranker (training-free) and the LoRA (training) are **complementary** — reranker fixes *what the model sees*, LoRA fixes *what it does with it*.
- **Expected outcome:** reranker gives you most of the selection benefit today for free;
  LoRA buys back the format/extraction gap. The combination should beat either alone.

**Bottom line:** the MIG reranker works and is the cheap win — ship it as-is (judge scorer).
Fine-tuning is worth it, but spend it on a small LoRA for *answer generation + format
compliance*, not on the reranker. Validate the real-pipeline numbers with `bfcl generate` on
`memory_vector` in your full environment before committing GPU to training — that run also
tells you whether the bottleneck is retriever recall (in which case raise `top_k`/improve the
embedder first).

---

## 6. Reproduce

```bash
# from berkeley-function-call-leaderboard/, with the vLLM server up on :8000
python bfcl_eval/scripts/mig_live_experiment.py \
  --scenario customer --n 30 --pool-size 10 --budget 3 \
  --distractors hard --workers 8 --out result_customer_hard.json

# arms / scorers / budgets are all flags; see --help
```
