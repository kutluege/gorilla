# MIG Reranker — Implementation Validation Report

**Branch:** `info_gain_rerank`  **Date:** 2026-07-01  **Endpoint:** vLLM @ `localhost:8000`
serving `Qwen/Qwen3-4B-Instruct-2507` (`allow_logprobs: true`).

**Verdict: the MIG reranker implementation is correct and functional.** Static wiring, override
contracts, and the full interception machinery (pool widening → candidate parsing → judge/logprob
scoring → selection → schema-preserving reserialization → JSONL tracing) all pass, validated
against the live model. Two flagged risks (R1, R2) are resolved in MIG's favour. The only caveats
are about *experiment methodology*, not the code.

---

## 1. Static validation — PASS

Confirmed by reading the committed code (see also the plan file):

- MIG id `Qwen/Qwen3-4B-Instruct-2507-FC-MIG` registered (`supported_models.py:143`); baseline
  `-FC` unchanged (`:142`).
- `QwenMIGHandler` imported (`model_config.py:62`) and wired (`:1665-1676`, `is_fc_model=True`,
  `model_name="Qwen/Qwen3-4B-Instruct-2507"`); baseline `-FC` still → `QwenFCHandler`.
- Runtime resolution verified: MIG id → `QwenMIGHandler`, baseline id → `QwenFCHandler`.
- All three overrides match parent signatures and call `super()`; the multi-turn loop is `@final`
  and untouched.
- `_rerank_one` passthrough is **byte-exact** for `kind=None` (returns `raw` unchanged), so the
  non-retrieval path (memory `add`, exact-key `retrieve`, rec_sum blob, errors) is identical to
  baseline. Confirmed both by code reading and empirically (see §3).

## 2. Live machinery validation — 22/22 PASS

A deterministic harness drove `MIGReranker` directly against the live endpoint (bypassing the
model's reluctance to retrieve — see §3). Script:
`<scratchpad>/mig_harness.py`.

| Path | Result |
|---|---|
| `widen_pool` — vector `top_k` 5→20, kv `k` 3→20, never-shrink (50 stays), kv exact `(key)` untouched, non-retrieval fn untouched | PASS (5/5) |
| `parse_candidates` — vector→`vector`, kv→`kv_keys`; passthrough for `{value}`, `{memory_content}`, `{error}`, empty `{result:[]}` | PASS (6/6) |
| `reserialize` — returns JSON **string**; vector→`{"result":[...]}`, kv→`{"ranked_results":[...]}`; raw dicts preserved verbatim | PASS (4/4) |
| **echo-logprob probe (R2)** | **PASS — endpoint supports `echo=True, logprobs=1, max_tokens=0`** (`token_logprobs` present) |
| `select` judge/flat (live) — non-empty, respects `final_budget`, ranks gold memory in | PASS; gold scored **5.0**, all others **0.0** |
| `select` logprob/flat (live) — `_logprob_supported=True`, gold ranked #1 | PASS; gold **+8.54**, distractors negative |

**R1 (wrong-draft inversion) is not occurring.** With `draft_from_pool=True`, the draft
`a_hat` = *"The patient is taking metformin 1000 mg twice daily for their diabetes."* and the
answer-bearing memory scored **highest** under both scorers — the failure mode the `0bf8ec8`
commit fixed stays fixed.

## 3. End-to-end smoke run — findings (methodology, not bugs)

Ran baseline `-FC` and MIG `-FC-MIG` over a 10-entry `memory_vector` slice (5 healthcare answers
+ 5 prereqs, via `--run-ids`).

- **MIG handler activates**: `[MIG] QwenMIGHandler active | ... scorer=judge mode=flat pool_size=20 ...`.
- **Zero interceptions**, because the model almost never issues a vector-retrieval tool call:
  across all 5 baseline answers there were **0** tool calls (answered from core memory, which is
  dumped into the system prompt); MIG had **1** retrieval, which returned `{"result": []}` and
  correctly passed through. This is **risk R8, strongly confirmed** — MIG's effect is confined to
  the rare `archival_memory_retrieve`-with-nonempty-pool case.
- **Run nondeterminism**: with `--num-threads 4`, vLLM continuous batching is non-deterministic
  even at `temperature≈0`. The two runs built **different memory snapshots** during prereq
  generation (baseline stored 2 core + 2 archival entries; MIG stored 0), so their answer-entry
  scores are **not comparable**. Verified this is *not* a MIG bug: the prereq `begin_of_turn_query`
  and first assistant response were **byte-identical** between runs, and the add-path is proven
  passthrough. The divergence is sampling/batching, not MIG.

### Recommendations for the real experiment
1. **Use `--num-threads 1`** (or a fixed seed / deterministic backend) for both baseline and MIG,
   **or** generate the baseline snapshot once and **reuse the same snapshot** for the MIG answer
   run, so the only variable is the reranker.
2. **Segment metrics to interception-bearing entries.** Report `archival_memory_retrieve` recall@k
   and gold-in-pool / gold-selected on the subset where retrieval actually fires; a whole-category
   average will be dominated by core-memory answers MIG never touches (R8).
3. **Consider forcing retrieval** for the study (e.g., a prompt variant that instructs the agent to
   call `archival_memory_retrieve`), otherwise interception counts will be very low on this model.
4. The **logprob arm is viable** on this endpoint (R2 resolved) — include it in the ablation matrix.

## 4. Cleanup findings

- **`base_oss_handler.py`** (uncommitted, shared parent): the added `self.model_name =
  self.registry_name` was **duplicated**; the duplicate line was removed (approved Step 0). Deeper
  finding: `BaseHandler.__init__` already sets `self.model_name = model_name` (the served
  checkpoint), and this line **overwrites** it with the registry id. It is **inert** for OSS/MIG
  inference (both use `self.model_path_or_id`; the reranker uses it too — `qwen_mig.py:63`) and for
  result paths (`model_name_underline_replaced` is computed earlier). **Recommendation:** revert
  `base_oss_handler.py` to its committed state — MIG does not need this line, and it sits on a
  shared parent. Left in place for now (only the dedup was in the approved scope).
- **`utils.py`** (uncommitted): `open(input_path, encoding='utf-8')` — benign Windows fix; keep.
- **Smoke artifacts**: partial healthcare-only results now exist under
  `result/Qwen_Qwen3-4B-Instruct-2507-FC{,-MIG}/agentic/memory/vector/`. Overwrite or delete before
  a full run so a later `evaluate` doesn't score the partial set.

## 5. Deferred (unchanged from plan)
Full ablation matrix; offline `analyze_mig_traces.py` (gold-in-pool / recall@k); the missing
`BFCL_MIG_RERANKER_GUIDE.md`.
