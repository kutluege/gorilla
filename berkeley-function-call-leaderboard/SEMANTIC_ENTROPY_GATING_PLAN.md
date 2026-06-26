# Semantic Entropy Gating for BFCL V4 Memory — Implementation Plan

| | |
|---|---|
| **Branch** | `semantic-entropy` |
| **Status** | Design / planning (no production code yet) |
| **Scope** | Inference-time, training-free middleware between the LLM and the existing BFCL memory backends |
| **Target benchmark** | BFCL V4 Memory (`memory_kv`, `memory_vector`, `memory_rec_sum` × 5 scenarios) |
| **Horizon** | ~2–3 months, single researcher |
| **Repo root for paths below** | `berkeley-function-call-leaderboard/` (i.e. paths are relative to this directory) |

> This document is a self-contained, actionable plan. It encodes the feasibility analysis, the design decisions, the file-level change spec, pseudocode, the experiment matrix, calibration strategy, phased milestones, risks, and an ordered implementation checklist. **No benchmark behavior changes until a feature flag is explicitly turned on; the default path is byte-for-byte the current pipeline.**

---

## 0. TL;DR

Add a **training-free mathematical control layer** that measures *semantic uncertainty* (entropy over meaning-clusters of K sampled outputs) to decide when memory writes/updates/retrievals/answers are reliable.

```
H_sem(Y | x) = - Σ_c p(c | x) · log p(c | x)
```
where `x` = condition (user turn / write candidate / retrieval context / question), `Y` = K sampled outputs, `c` = a semantic-equivalence cluster, `p(c|x) = |c| / K`.

**Feasibility verdict: Option B — feasible with small inference-wrapper changes.** The only hard blocker is that the stack today emits exactly one deterministic completion per call. We add a `sample_k()` helper and a `memory_se_gate` module; everything else is additive.

**First deliverable:** read-path **answer-entropy gating in log-only mode** — a pure measurement study answering *"does semantic entropy predict correctness on BFCL memory?"* (AUROC). Only if the signal exists do we enable active gating, then attempt the riskier write gating.

---

## 1. Background & motivation

The benchmark scores a memory question by a **string contains-match** on the model's final natural-language message — not an LLM judge. Gold answers carry surface variants, e.g. `["35","thirty five"]`, `["513","five hundred thirteen","five hundred and thirteen"]`. This is precisely the regime where **semantic** entropy (clustering "35" with "thirty-five") is more informative than surface/string entropy.

Two properties of this repo make the method cheap and clean to build:

1. **An embedding model is already a dependency** — `SentenceTransformer("all-MiniLM-L6-v2")` + `faiss` are loaded by the vector backend. Embedding-based semantic clustering needs **zero new heavy dependencies** for v1.
2. **The benchmark values determinism** (`--temperature` default `0.001`). We respect that: the *recorded* answer/snapshot always comes from the existing deterministic path; sampling happens only inside the middleware to compute a confidence/entropy signal and (optionally) trigger a retry.

---

## 2. Key repo facts that constrain the design (grounded)

All verified by direct reads.

### 2.1 The agentic loop (primary integration site)
- `bfcl_eval/model_handler/base_handler.py` → `inference_multi_turn_FC` / `inference_multi_turn_prompting` (both `@final`). Per turn, per step: `_query_*` → `_parse_query_response_*` → `decode_execute` → `execute_multi_turn_func_call`.
- A response that **fails to decode** as a function call is the natural-language message — for the scored question this is the final answer.
- At the end of a **prereq** entry the handler calls `memory_instance._flush_memory_to_local_file()` (`base_handler.py:371` / `:662`), persisting the snapshot.

### 2.2 Single deterministic completion (the blocker)
- `bfcl_eval/model_handler/api_inference/claude.py` → `_query_FC`/`_query_prompting` build `kwargs={model, max_tokens, temperature, ...}` and call `self.client.messages.create(**kwargs)` **once**. No `n`, no `top_p`.
- `temperature` is a single fixed handler attribute (`self.temperature`), set once via `build_handler(model_name, args.temperature)`.
- Grep across `model_handler/`: **no** `n=` / `num_return_sequences` / `best_of`; only incidental `top_p` (qwen) and `do_sample:False` (nexus).
- The Anthropic Messages API has **no `n`** → K samples ⇒ K calls. (OpenAI-completion / vLLM expose `n`/`logprobs`, but BFCL never wires them.)
- Determinism is a *convention*, not an architectural constraint (`_llm_response_generation.py:178`: "*Since temperature is already set to 0.001, retrying … will not help*").

### 2.3 Memory state is a shared, dependency-ordered artifact
- `bfcl_eval/utils.py` → `process_memory_test_case()` chains prereqs via `depends_on` and makes **every scored question depend on all prereq entries**; `sort_key()` gives prereq priority 0, question priority 4. The DAG scheduler in `_llm_response_generation.py:generate_results()` runs prereqs first, flushing `{scenario}_final.json`, which all questions load.
- **Implication:** write/update gating in the prereq phase mutates state consumed by ~30 downstream scored questions per scenario (high leverage, high blast radius). Read-path gating touches only the single question being scored.

### 2.4 Memory injected into context + backend-specific tool names
- `bfcl_eval/model_handler/utils.py` → `add_memory_instruction_system_prompt()` injects `_dump_core_memory_to_context()` + scenario instructions into the first system message (the **retrieval-context insertion point**).
- Tool names **differ by backend**: KV/vector use `core_memory_add/replace/retrieve/...` + `archival_memory_*`; `rec_sum` uses `memory_append` (single string, no `replace`). ⇒ Write/update gating needs a per-backend method map; **read-path gating is backend-agnostic** (operates on question + final answer text).

### 2.5 Scoring & artifacts
- `bfcl_eval/eval_checker/eval_runner.py` → `_evaluate_single_agentic_entry()` finds the **last un-decodable message** and calls `agentic_checker()`.
- `bfcl_eval/eval_checker/agentic_eval/agentic_checker.py` → `agentic_checker()` = standardize (lowercase, strip punctuation) + word-boundary regex contains-match against `ground_truth` list. **No LLM judge.**
- Gold: `bfcl_eval/data/possible_answer/BFCL_v4_memory.json` → `{id, ground_truth:[...], source}`.
- Accuracy: `eval_runner_helper.py:save_eval_results()` → `accuracy=correct/total` → `BFCL_v4_{cat}_score.json`.
- Result JSON per entry already carries `result`, `input_token_count`, `output_token_count`, `latency`, `inference_log` (written via `{id, result, **metadata}` in `multi_threaded_inference`). Snapshots live under `result/{model}/agentic/memory/{backend}/memory_snapshot/`.
- Prereq entries are **excluded from scoring** (`eval_runner.py:799`).

---

## 3. Design overview

A thin middleware sits inside the handler's agentic loop. It never replaces the memory backends; it observes decoded calls and final answers, optionally samples K alternatives, clusters them by meaning, computes entropy, and gates a decision.

```
                 ┌─────────────────────── BaseHandler.inference_multi_turn_* ───────────────────────┐
 user turn ─────▶│  _query  ─▶ parse ─▶ decode_execute ─┬─(tool calls)─▶ [WRITE GATE] ─▶ execute    │
                 │                                       └─(NL final answer)─▶ [READ/ANSWER GATE]    │
                 └───────────────────────────────────────────────────────────────────────────────┘
                                         │ sample_k() (temp>0, cached)        │
                                         ▼                                    ▼
                            memory_se_gate.semantic_entropy() ──▶ cluster (all-MiniLM) ──▶ H_sem, n_clusters
                                         │
                          decision: accept / broaden-retrieval+retry / flag low-confidence / skip-write
```

Design invariants:
- **Recorded output is always the deterministic generation.** Sampling is internal to the gate.
- **Default flag = off** ⇒ baseline pipeline unchanged.
- **Backend-agnostic by default** (read path); write/update gating uses an explicit per-backend method map.
- **Reproducible** via a disk-backed sample cache keyed by prompt hash.

---

## 4. Intervention points & chosen sequencing

| # | Point | Phase | Backend-agnostic? | Blast radius | Accuracy risk | Build order |
|---|---|---|---|---|---|---|
| 4 | **Final-answer entropy gating** (sample K answers, compute H, log/retry/flag) | question | ✅ | none (read-only) | lowest | **1st (log-only) → 2nd (active)** |
| 3 | **Retrieval verification** (high H ⇒ broaden retrieval / abstain) | question | ✅ | none (read-only) | low | with #4 (active) |
| 1 | **Write gating** (gate `*_add`/`memory_append`) | prereq | ⚠️ needs method map | high (shared snapshot) | highest (recall) | 3rd (log-only first) |
| 2 | **Update gating** (block destructive `*_replace`) | prereq | ❌ KV/vector only | moderate | moderate | 4th (optional) |

**Sequencing rationale:** start where the signal is measurable with zero risk (#4 log-only), tie improvements directly to the scored metric (#3+#4 active), and only then take on the novel-but-risky shared-state mutation (#1), with #2 as an optional extension.

---

## 5. Sampling strategy & reproducibility

- **How K samples are produced:** a new `sample_k(inference_data, k, temperature)` helper loops the existing `_query_*` + `_parse_query_response_*` K times at `temperature>0`. (Anthropic has no `n`; for OSS/vLLM a single `n=k` call may be used as an optimization later.)
- **Default sampling params:** `K=5`, `temperature=0.7` (gate-only; the benchmark's `0.001` is untouched).
- **Cheaper alternatives (fallbacks):**
  - *Paraphrase ensembling* — at temp≈0, re-ask the question in K paraphrases; deterministic & reproducible, no temperature change.
  - *Logprob/NLL* — only on OpenAI-completion/vLLM handlers; not portable to Anthropic.
- **Reproducibility guarantees:**
  - Recorded answer/snapshot = deterministic path ⇒ official results stay reproducible.
  - **Disk-backed sample cache** keyed by `hash(messages, system, tools, temperature, k)` ⇒ re-runs reuse samples even against the unseedable Anthropic API.
  - Seed vLLM where available; report sampled metrics as mean±std over 3 seeds.

---

## 6. Semantic clustering

| Option | New deps | Difficulty | Runtime | Reliability | BFCL fit |
|---|---|---|---|---|---|
| **Embedding cosine** (all-MiniLM, greedy threshold) | none (installed) | low | ~ms CPU | good for short facts | **excellent** |
| **NLI bidirectional entailment** (DeBERTa-MNLI) | yes | medium | 10–100 ms/pair | higher | strong for clausal answers |
| **LLM-as-judge equivalence** | none (+tokens) | low–med | API latency | high but costly/variable | good but expensive |
| **Hybrid** (embed → NLI/LLM only on gray-band pairs) | partial | medium | adaptive | best | best long-term |

- **v1 (build now):** embedding clustering with the already-loaded `all-MiniLM-L6-v2`. Greedy single-linkage by cosine ≥ `0.70`. Normalize answers (lowercase, strip, optional number-word canonicalization "35"↔"thirty five").
- **v2 (camera-ready):** hybrid — embeddings for the easy majority; DeBERTa-MNLI bidirectional entailment for pairs whose cosine lands in `[0.55, 0.80]`, matching the canonical Kuhn-et-al. definition.

---

## 7. Entropy, thresholding & calibration (no fine-tuning)

`H_sem` is entropy over cluster mass (`p(c)=|c|/K`), reported in nats; `0.0` ⇔ full semantic agreement.

Calibration is **nonparametric** (pick a scalar τ on a held-out split; no gradient updates):
1. **No-threshold ranking mode (start here):** don't gate — just log H and report **AUROC of H vs. correctness**. Needs no τ. Decision gate for the whole project.
2. **Percentile threshold (v1 active):** on a held-out scenario, collect `(H, correct)` pairs, set τ to maximize accuracy at a fixed abstention budget (or Youden's J on the H→correct ROC).
3. **Adaptive cluster-count rule (companion):** accept if `n_clusters==1`; retry if `2 ≤ n_clusters < K/2`; flag/abstain if `n_clusters ≥ K/2`.
4. **Per-backend τ:** separate τ for kv/vector/rec_sum (rec_sum answers are longer ⇒ different H scale).
5. **NLL fallback:** OpenAI-completion/vLLM only.

**Held-out split:** reserve one scenario (e.g. `student`) for calibration; report on the remaining four. Describe as *nonparametric threshold calibration*, **not** training.

---

## 8. Module & file-level change spec

### 8.1 New files
| Path | Contents |
|---|---|
| `bfcl_eval/model_handler/memory_se_gate.py` | `normalize_answer`, `cluster_semantic_equivalence`, `semantic_entropy`, `should_write_memory`, `should_accept_retrieval_answer`, `SampleCache`, backend method maps (`WRITE_METHODS`, `UPDATE_METHODS`), `SEGateConfig`. Reuses the installed all-MiniLM encoder (lazy import to avoid a second model load). |
| `bfcl_eval/analysis/se_report.py` | Offline: read `inference_log` `se_gate` entries + `*_score.json` → entropy↔correctness table, AUROC, selective-prediction (accuracy vs coverage) curves. |
| `SEMANTIC_ENTROPY_GATING_PLAN.md` | This document. |

### 8.2 Modified files (minimal, flag-guarded)
| Path | Change |
|---|---|
| `bfcl_eval/model_handler/base_handler.py` | (a) `sample_k()` helper + `self._sample_cache`; (b) read-path hook after a step yields the final NL message (memory, non-prereq); (c) write-path hook after `decode_execute`, before `execute_multi_turn_func_call`. All guarded by `self.se_gate_enabled` (default `False`). |
| `bfcl_eval/model_handler/api_inference/claude.py` (+ other handlers as needed) | A tiny `_query_with_temperature(inference_data, temperature)` that mirrors `_query_*` but overrides temperature — used only by `sample_k()`. |
| `bfcl_eval/_llm_response_generation.py` | `get_args()`: add `--se-gate {off,logonly,read,write}`, `--se-k`, `--se-temp`, `--se-tau`, `--se-cluster {embed,nli,hybrid}`; thread into `build_handler()` and onto the handler instance. |

> Rationale for hooking in `base_handler.py` (not `multi_turn_utils.py`): `execute_multi_turn_func_call` just executes via `eval()`; the decoded calls and the final answer are only available in the handler loop, which is also where token/latency/log metadata is assembled.

---

## 9. Pseudocode

```python
# bfcl_eval/model_handler/memory_se_gate.py   (NEW — pure logic; reuses installed all-MiniLM)
from math import log

def normalize_answer(s: str) -> str:
    # lowercase, strip surrounding punctuation/space; optional number-word canonicalization
    ...

def cluster_semantic_equivalence(samples, threshold=0.70, encoder=ENCODER):
    """Greedy single-linkage clustering of short answers by cosine similarity."""
    embs = encoder.encode([normalize_answer(s) for s in samples],
                          normalize_embeddings=True)        # CPU; cached by hash
    clusters = []                                           # list[list[idx]]
    for i, e in enumerate(embs):
        for c in clusters:
            if float(e @ embs[c[0]]) >= threshold:          # unit vectors → dot = cosine
                c.append(i); break
        else:
            clusters.append([i])
    return clusters

def semantic_entropy(samples, **kw):
    """H_sem = -Σ_c p(c) log p(c), p(c)=|c|/K. Returns (H, n_clusters, clusters)."""
    if not samples:
        return float("inf"), 0, []
    clusters = cluster_semantic_equivalence(samples, **kw)
    K = len(samples)
    H = -sum((len(c)/K) * log(len(c)/K) for c in clusters)
    return H, len(clusters), clusters

def should_write_memory(user_turn, extracted_value, sampler, tau, k=5):
    """Gate a candidate memory write. Returns (allow, info)."""
    samples = sampler(extract_prompt(user_turn), k=k) + [extracted_value]
    H, n, _ = semantic_entropy(samples)
    return (H <= tau), {"entropy": H, "n_clusters": n, "k": k}

def should_accept_retrieval_answer(question, retrieved_ctx, sampler, tau, k=5):
    """Gate / verify the final answer. Returns (accept, confidence, info)."""
    samples = sampler(answer_prompt(question, retrieved_ctx), k=k)
    H, n, clusters = semantic_entropy(samples)
    majority = max(clusters, key=len)
    confidence = len(majority) / k
    return (H <= tau), confidence, {"entropy": H, "n_clusters": n,
                                    "majority_answer": samples[majority[0]]}
```

```python
# base_handler.py   (MODIFY — read-path hook, inside the step loop after decode_execute)
if (self.se_gate_enabled and is_memory(test_category)
        and not is_memory_prereq(test_entry_id)
        and is_final_nl_message(model_responses, decoded_model_responses)):
    sampler = lambda prompt, k: self.sample_k(inference_data, k, self.se_temp)
    accept, conf, info = should_accept_retrieval_answer(
        current_turn_message, current_context, sampler, self.se_tau, self.se_k)
    current_step_inference_log.append({"role": "se_gate", "content": info,
                                       "confidence": conf, "accepted": accept})
    if self.se_mode == "active" and not accept and not already_retried:
        inference_data = self._inject_broaden_retrieval_msg(inference_data)
        already_retried = True
        continue                       # re-enter step loop; do NOT break
    # log-only / accepted → fall through; recorded answer stays deterministic
```

```python
# base_handler.py   (MODIFY — write-path hook, before execute_multi_turn_func_call)
if self.se_gate_enabled and self.se_mode == "write" and is_memory_prereq(test_entry_id):
    kept = []
    for call in decoded_model_responses:
        if func_name_of(call) in WRITE_METHODS[backend_type]:
            allow, info = should_write_memory(current_turn_message, value_of(call),
                                              sampler, self.se_tau, self.se_k)
            current_step_inference_log.append({"role": "se_write_gate",
                                               "content": info, "call": func_name_of(call)})
            if not allow:
                continue               # skip / log-only; NEVER auto-delete
        kept.append(call)
    decoded_model_responses = kept
```

```python
# base_handler.py   (NEW helper; concrete _query in each handler)
def sample_k(self, inference_data, k, temperature):
    key = hash_prompt(inference_data, temperature, k)
    if key in self._sample_cache:
        return self._sample_cache[key]
    out = [self._parse_query_response_prompting(
               self._query_with_temperature(inference_data, temperature)[0])["model_responses"]
           for _ in range(k)]          # Anthropic has no n → loop
    self._sample_cache[key] = out      # persisted to snapshot sidecar
    return out
```

---

## 10. Configuration & CLI

```
--se-gate     {off, logonly, read, write}   default: off    # off = baseline, unchanged
--se-k        int                            default: 5
--se-temp     float                          default: 0.7
--se-tau      float | "auto"                  default: auto  # auto = load calibrated per-backend τ
--se-cluster  {embed, nli, hybrid}            default: embed
```
`off` must reproduce the current pipeline byte-for-byte (regression-test this).

---

## 11. Logging & artifacts (no schema break)

- **Per-decision:** append `{"role":"se_gate"|"se_write_gate", entropy, n_clusters, confidence, accepted, k, samples_hash}` into the existing `current_step_inference_log` → flows into `inference_log` in `metadata` → result JSON via `**metadata`. Never touches the scored `result` field.
- **Raw samples + cluster assignments:** JSONL sidecar at `result/{model}/agentic/memory/{backend}/memory_snapshot/se_samples/{test_id}.jsonl` (reuses the snapshot dir convention). Enables offline re-clustering without re-querying.
- **Confidence tags on memory (write gating):** parallel `se_confidence:{key:score}` map in the snapshot JSON; backward-compatible (loaders read only `core_memory`/`archival_memory`).
- **Aggregation:** `bfcl_eval/analysis/se_report.py` joins `se_gate` log entries with `*_score.json`. No change to `eval_runner`.

---

## 12. Experiment & ablation plan

**Substrate:** one strong FC model (a Claude handler) × {`memory_kv`, `memory_vector`, `memory_rec_sum`} × 5 scenarios. Baseline = current pipeline.

**Arms**
1. Baseline (temp 0.001).
2. Write gating only (#1).
3. Retrieval verification only (#3).
4. Final-answer entropy gating only (#4: log-only and active).
5. Write + retrieval (#1+#3).
6. K ∈ {3, 5, 8}.
7. Clustering: embed vs NLI vs LLM-judge vs hybrid.
8. Threshold: fixed vs percentile vs per-backend vs adaptive cluster-count.

**Metrics & where to get them**
| Metric | Source |
|---|---|
| Official accuracy | `save_eval_results` → `BFCL_v4_{cat}_score.json` |
| Write precision/recall | prereq `source` + `ground_truth` vs final `{scenario}_final.json` snapshot (approximate gold; document heuristic) |
| Update correctness | `*_replace` calls in `inference_log` vs gold post-snapshot |
| Retrieval success | gold string present in injected context / `*_retrieve/_search` results in `inference_log` |
| Answer correctness | official accuracy per question |
| Entropy↔correctness | AUROC offline from logged `entropy` + score files (`se_report.py`) |
| Abstention/retry rate | count `se_gate` entries with `accepted=False`/retry |
| Latency / tokens / extra calls | `latency`, `input_token_count`, `output_token_count` in result JSON; extra calls = `K × #gated decisions` |

**Headline comparisons:** accuracy Δ vs baseline **at matched token budget**; selective-prediction curves (accuracy vs coverage as τ sweeps); cost-normalized accuracy.

---

## 13. Project phases & milestones (~2–3 months)

| Phase | Weeks | Goal | Exit criterion |
|---|---|---|---|
| **P0 — Scaffolding** | 1 | `memory_se_gate.py` (entropy + embed clustering), `sample_k()` + cache, `--se-gate` flag, regression test that `off` == baseline. | `off` reproduces baseline; unit tests for `semantic_entropy` pass. |
| **P1 — Measurement spike (log-only #4)** | 2–3 | Run log-only on `memory_kv × customer`, then all backends × all scenarios. | **Go/No-Go: AUROC(H vs correct) clearly > 0.5.** `se_report.py` produces the table/curves. |
| **P2 — Calibration** | 4 | Percentile + per-backend τ on held-out scenario; adaptive cluster-count rule. | Calibrated τ checked in; documented as nonparametric. |
| **P3 — Active read path (#3+#4)** | 5–6 | Enable broaden-retrieval + selective retry; re-measure accuracy/coverage/cost. | Accuracy ≥ baseline at matched token budget, or a favorable selective-prediction curve. |
| **P4 — Ablations** | 7–8 | K sweep; embed vs hybrid-NLI; threshold variants. | Ablation tables complete. |
| **P5 — Write/update gating (#1/#2)** | 9–11 | Per-backend method map; log-only write gating → snapshot regen (`--allow-overwrite`) → measure write precision/recall + downstream accuracy → enable skip-on-high-H. | Write gating does not reduce downstream accuracy beyond a stated tolerance. |
| **P6 — Writeup** | 12 | Consolidate results, finalize plots, draft report. | Reproducible run scripts + results committed. |

Milestones can compress if P1's signal is strong; P5 is the contingency-heavy phase.

---

## 14. Risks & mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Entropy doesn't predict correctness on short answers | project-killing | **P1 AUROC study first**; abort/redesign if no signal. |
| Stochastic sampling → noisy/irreproducible | high | Recorded output deterministic; sample only in gate; disk cache by prompt hash; seed vLLM; report mean±std over 3 seeds. |
| Entropy rejects correct-but-diverse phrasing | medium | Cluster by meaning (embed/NLI). Recorded answer stays deterministic, so diversity affects only the retry/confidence decision; prefer broaden-retrieval over hard abstention. |
| Low entropy, confidently wrong | medium | Combine H with a grounding check (was the gold present in retrieved context?); report calibration (ECE), not just accuracy. |
| Write gating hurts recall | high | Default log-only; delay/flag rather than skip; never auto-delete; conservative τ for `add`; measure write-recall explicitly. |
| Retrieval/answer entropy too expensive | medium | Gate only the ~150 scored questions (not prereq turns); default K=3; cache; sample only on the final-answer step; report at matched token budget. |
| Backend-specific tool names | medium | Per-backend `WRITE_METHODS`/`UPDATE_METHODS` maps; read path stays text-only and backend-agnostic. `rec_sum` has no `replace` ⇒ #2 partly N/A (documented). |
| Shared-snapshot corruption (write gating) | high | Regenerate snapshots from scratch per condition with `--allow-overwrite`; never compare across mixed snapshot states. |
| all-MiniLM English-centric | low (today) | Swap to multilingual MiniLM / XNLI in hybrid path if the benchmark is extended. |

---

## 15. Definition of done

- `--se-gate off` is byte-for-byte identical to the current pipeline (regression test).
- `memory_se_gate.py` has unit tests for `semantic_entropy` (known inputs → known H) and clustering edge cases.
- P1 produces an entropy↔correctness AUROC table across backends/scenarios.
- A calibrated, documented, nonparametric τ exists (per backend).
- The active read-path gate has a measured accuracy/cost trade-off vs baseline at matched token budget.
- All experiments are reproducible from committed run scripts + cached samples.

---

## 16. Implementation checklist (ordered)

- [ ] **P0.1** Create `bfcl_eval/model_handler/memory_se_gate.py`: `normalize_answer`, `cluster_semantic_equivalence`, `semantic_entropy`, `should_write_memory`, `should_accept_retrieval_answer`, `SampleCache`, `WRITE_METHODS`/`UPDATE_METHODS`, `SEGateConfig`. Lazy-load all-MiniLM.
- [ ] **P0.2** Add `sample_k()` + `self._sample_cache` to `BaseHandler`; add `_query_with_temperature()` to `ClaudeHandler`.
- [ ] **P0.3** Add `--se-gate/--se-k/--se-temp/--se-tau/--se-cluster` to `get_args()`; thread through `build_handler()` onto the handler.
- [ ] **P0.4** Read-path hook (flag-guarded) in `inference_multi_turn_FC` and `_prompting`. Default `off`.
- [ ] **P0.5** Unit tests + regression test (`off` == baseline on a tiny memory subset).
- [ ] **P1.1** Run log-only #4 on `memory_kv × customer`; sanity-check logged H.
- [ ] **P1.2** Build `bfcl_eval/analysis/se_report.py` (join `inference_log` + `*_score.json` → AUROC + curves).
- [ ] **P1.3** Run log-only across all backends × scenarios. **Go/No-Go on AUROC.**
- [ ] **P2.1** Calibrate percentile + per-backend τ on held-out scenario; commit τ + adaptive cluster-count rule.
- [ ] **P3.1** Implement `_inject_broaden_retrieval_msg()` + single selective-retry loop; enable `--se-gate read`.
- [ ] **P3.2** Re-measure accuracy/coverage/cost vs baseline at matched token budget.
- [ ] **P4.1** Ablations: K∈{3,5,8}; embed vs hybrid-NLI (add DeBERTa-MNLI for gray band); threshold variants.
- [ ] **P5.1** Per-backend write-method map + write-path hook; **log-only** write gating; regenerate snapshots (`--allow-overwrite`).
- [ ] **P5.2** Measure write precision/recall + downstream question accuracy; then enable skip-on-high-H if safe.
- [ ] **P6.1** Consolidate results, plots, reproducible run scripts; draft report.

---

## Appendix A — File reference index

| Concern | File | Symbol |
|---|---|---|
| Agentic loop / integration site | `bfcl_eval/model_handler/base_handler.py` | `inference_multi_turn_FC`, `inference_multi_turn_prompting`, `decode_execute` |
| Single-completion query (sampling blocker) | `bfcl_eval/model_handler/api_inference/claude.py` | `_query_FC`, `_query_prompting`, `generate_with_backoff` |
| Memory → prompt injection | `bfcl_eval/model_handler/utils.py` | `add_memory_instruction_system_prompt` |
| Generation entry / DAG / CLI | `bfcl_eval/_llm_response_generation.py` | `get_args`, `build_handler`, `generate_results`, `multi_threaded_inference` |
| Prereq/question DAG & helpers | `bfcl_eval/utils.py` | `process_memory_test_case`, `sort_key`, `is_memory`, `is_memory_prereq` |
| Backends | `bfcl_eval/eval_checker/multi_turn_eval/func_source_code/memory_{kv,vector,rec_sum}.py` | `core_memory_*`, `archival_memory_*`, `memory_append`; `ENCODER` (all-MiniLM) |
| Backend base / snapshots | `.../func_source_code/memory_api_metaclass.py` | `_prepare_snapshot`, `_dump_core_memory_to_context`, `_flush_memory_to_local_file` |
| Tool execution | `bfcl_eval/eval_checker/multi_turn_eval/multi_turn_utils.py` | `execute_multi_turn_func_call` |
| Scoring | `bfcl_eval/eval_checker/eval_runner.py` | `_evaluate_single_agentic_entry`, `agentic_runner` |
| Contains-match | `bfcl_eval/eval_checker/agentic_eval/agentic_checker.py` | `agentic_checker`, `standardize_string` |
| Accuracy / cost / latency | `bfcl_eval/eval_checker/eval_runner_helper.py` | `save_eval_results`, `get_cost_latency_info` |
| Categories / scenarios | `bfcl_eval/constants/category_mapping.py` | `ALL_AVAILABLE_MEMORY_BACKENDS`, `MEMORY_CATEGORY`, `MEMORY_SCENARIO_NAME` |
| Gold answers | `bfcl_eval/data/possible_answer/BFCL_v4_memory.json` | `{id, ground_truth[], source}` |
| Scored questions | `bfcl_eval/data/BFCL_v4_memory.json` | — |
| Prereq conversations | `bfcl_eval/data/memory_prereq_conversation/memory_{scenario}.json` | — |
