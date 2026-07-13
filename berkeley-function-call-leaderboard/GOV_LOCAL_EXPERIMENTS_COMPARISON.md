# Local BFCL v4 Memory Experiments — Consolidated Comparison

All experiments run locally on this machine against `Qwen/Qwen3-4B-Instruct-2507`
served by vLLM on the BSC cluster (SSH tunnel → `http://localhost:8000/v1`),
scored with the **unmodified** official BFCL grader (`agentic_checker`), temperature
0.001. Datasets, prompts, and backends identical across arms; the only variable per
comparison is the middleware.

Last updated: 2026-07-13 (Stage 0 governance session). Stage 0 code: branch
`just_geo_filtering`, commit `84d831d`.

---

## 1. Stage 0 Geometric Governance Filter (this session, 2026-07-13)

Three full generation runs (155 test + 37 prereq entries per backend), sequential
except where noted. Governance = write-time filter on `memory_add/replace/update`
(KV + Vector only): NOOP redundant writes via decoy-rewrite + synthetic success,
ESCALATE falls back to normal write (Stage 1 NLI / Stage 2 Retrieval Entropy not
yet implemented).

### 1.1 Headline scores

| Arm | memory_kv | memory_vector | Config |
|---|---|---|---|
| Baseline `-FC` | **7.10%** (11/155) | **11.61%** (18/155) | no middleware |
| GOV — SAGE defaults | **21.29%** (33/155) | **12.26%** (19/155) | `sim_high=0.80, δ=0.025` |
| GOV — calibrated | **16.77%** (26/155) | **12.90%** (20/155) | `sim_high=0.95, δ=0.30` |

⚠️ **Read §1.4 before interpreting the deltas.** The KV spread is dominated by
prereq-chain survival luck, not by governance.

Notes:
- The SAGE-defaults arm and the baseline ran **concurrently** on the same vLLM
  server (batching nondeterminism suspected); the calibrated arm ran alone.
- A fourth run (shadow mode, `GOV_DRY_RUN=1`) was intentionally aborted at ~1/6
  progress once it had produced enough calibration data (161 logged decisions);
  its partial outputs are archived in `result_gov_shadow_partial/`, never scored.

### 1.2 Governance decision statistics

| Arm | Gated decisions | NOOP | Blocked by preflight | Escalate→fallback ADD | Novel→ADD |
|---|---|---|---|---|---|
| Shadow (partial) | 161 | 0 would-fire | 0 | 23 | 118 |
| GOV — SAGE defaults | 342 | **3** (all vector, sim=1.0, r=0.0) | 1 | 81 | 241 |
| GOV — calibrated | 447 | **5** (all vector, sim 0.95–1.0) | 24 | 68 | 333 |

The 5 calibrated NOOPs are all genuine redundant rewrites (e.g., the same
Levothyroxine dosage text re-written at sim=0.992; a byte-identical customer
profile at sim=1.0). Zero KV suppressions in any arm — KV composite embeddings
(`"key words: value"`) depress similarity below `sim_high` even for redundant
value rewrites (see recommendations doc §2.3).

Latency: full decision path (QR residual + verbatim gate + thresholds) at maximum
memory (57 items) averages **0.53 ms** — within the <1 ms/write budget. Encoder
singleton shared with the vector backend; no new dependencies.

### 1.3 Per-scenario breakdown (correct answers / snapshot items at end of prereqs)

memory_kv (out of customer 30, student 50, finance 25, healthcare 25, notetaker 25):

| Arm | customer | student | finance | healthcare | notetaker | Σ |
|---|---|---|---|---|---|---|
| Baseline | 6 (17 items) | 1 (**0 items**) | 0 (19) | 3 (**0 items**) | 1 (7) | 11 |
| GOV SAGE | 4 (25) | 1 (**0**) | 7 (12) | 9 (7) | 12 (8) | 33 |
| GOV calibrated | 3 (7) | 0 (**0**) | 3 (17) | 8 (26) | 12 (9) | 26 |

memory_vector:

| Arm | customer | student | finance | healthcare | notetaker | Σ |
|---|---|---|---|---|---|---|
| Baseline | 5 (11 items) | 1 (**0 items**) | 5 (14) | 3 (**0 items**) | 4 (6) | 18 |
| GOV SAGE | 2 (19) | 1 (**0**) | 3 (26) | 7 (1) | 6 (8) | 19 |
| GOV calibrated | 4 (12) | 1 (**0**) | 2 (7) | 6 (2) | 7 (7) | 20 |

### 1.4 The dominant confound: prereq-chain survival

- **student** stored **zero** memories in every arm and backend: the model
  chit-chats through all 10 prereq turns without a single tool call. ~50/155
  questions are dead weight in every run.
- **healthcare** died in the baseline run (0 items on both backends) but survived
  in both governed runs — that alone explains most of the baseline's 7.10% KV
  score. This is run-to-run variance in whether borderline turns produce a tool
  call, not a middleware effect.
- With only 3–5 actual suppressions per governed run, the true Stage-0 treatment
  effect is bounded by a couple of entries; the observed inter-arm spread
  (±10 points) is survival noise. Conclusion: **mechanism validated, accuracy
  effect not yet measurable** without survival-conditioned, replicated runs (see
  `GOV_CASCADE_PLAN_RECOMMENDATIONS.md` §1).

### 1.5 ABTT preprocessing (one-time)

Fitted on the pooled 5-scenario prereq corpus (1,857 sentences, all-MiniLM-L6-v2,
D=16): median pairwise cosine **0.096 raw → −0.003 after ABTT**; top-16 directions
carry 33.7% of variance. Artifact committed:
`bfcl_eval/model_handler/middleware/artifacts/abtt_minilm_l6_d16.npz`.
(Note: the raw-anisotropy premise of ~0.83 median cosine did **not** hold on this
corpus.)

---

## 2. Earlier local experiments (prior sessions, same machine/model)

### 2.1 Semantic-Entropy gate (`-FC-SE`), full 155-entry runs

Sampling-based uncertainty gate at generation time (N samples, cluster, commit
majority; destructive ops require low entropy). Different session and server
instance than §1.

| Arm | memory_kv | memory_vector | memory_rec_sum |
|---|---|---|---|
| Baseline `-FC` (SE session) | 15.48% (24/155) | 17.42% (27/155) | 27.10% (42/155) |
| SE-gated `-FC-SE` | 11.61% (18/155) | 16.13% (25/155) | 26.45% (41/155) |

SE gating did not improve accuracy on any backend in that session.

### 2.2 MIG reranker (`-FC-MIG`), partial 55-entry vector-only runs

Retrieval-pool widening + information-gain reranking at the tool-result boundary.
Partial runs (55/155 vector entries), **not comparable** to full-run numbers above.

| Arm | memory_vector (55 entries) |
|---|---|
| Old partial baseline | 14.55% (8/55) |
| MIG — judge scorer | 12.73% (7/55) |
| MIG — logprob scorer | 16.36% (9/55) |

### 2.3 Cross-session baseline drift — the meta-finding

The **same** baseline model/config scored:

| Baseline `-FC` run | memory_kv | memory_vector |
|---|---|---|
| SE session (earlier) | 15.48% | 17.42% |
| Stage-0 session (2026-07-13) | 7.10% | 11.61% |

An 8-point KV swing between two nominally identical baseline runs (different day,
different server instance, concurrent load in the second). This is the same
survival-noise mechanism as §1.4 and is the single most important fact for
interpreting **any** of the tables in this document: single-run deltas below
~±8–10 points on this benchmark, at this model scale, are not evidence.

---

## 3. Where everything lives

| Artifact | Path |
|---|---|
| Stage 0 middleware + handler | `bfcl_eval/model_handler/middleware/governance_filter.py`, `bfcl_eval/model_handler/local_inference/qwen_gov.py` (commit `84d831d`) |
| Decision logs (JSONL) | `gov_logs/shadow/`, `gov_logs/governed_sage/`, `gov_logs/governed_calibrated/` |
| This session's results/scores | `result/`, `score/` (baseline + calibrated GOV); `result_gov_sage/`, `score_gov_sage/`; `result_gov_shadow_partial/` |
| Prior arms | `result_se_exp/`+`score_se_exp/`, `result_mig_*/`+`score_mig_*/`, `result_baseline/`+`score_baseline/` (partial, 55-entry) |
| Offline tests | `bfcl_eval/scripts/test_gov_offline.py` (23/23 passing) |
| Plan review | `GOV_CASCADE_PLAN_RECOMMENDATIONS.md` |

### Reproduction (calibrated governed arm)

```bash
# .env: LOCAL_SERVER_ENDPOINT=localhost / LOCAL_SERVER_PORT=8000 (BSC tunnel)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
GOV_DRY_RUN=0 GOV_SIM_HIGH=0.95 GOV_DELTA=0.30 GOV_LOG_DIR=./gov_logs/governed_calibrated \
bfcl generate --model Qwen/Qwen3-4B-Instruct-2507-FC-GOV \
  --test-category memory_kv,memory_vector --skip-server-setup \
  --temperature 0.001 --allow-overwrite

PYTHONUTF8=1 bfcl evaluate --model Qwen/Qwen3-4B-Instruct-2507-FC-GOV \
  --test-category memory_kv,memory_vector
```
