# Plan 2 — Live Evaluation via Tunneled BSC vLLM (GPU-backed)

> **Sequence:** This is the **second** of two ordered plans. It depends on every artifact
> Plan 1 (`PLAN_1_IMPLEMENTATION_OFFLINE.md`) produced: the governance code, the
> shadow-mode Stage 1, the replay verdict, and the calibratable knobs. **Do not start
> until Plan 1's Definition of Done is met.** Everything here needs the live model, which
> in this setup is reached **over an SSH tunnel to the BSC vLLM server** — there is no GPU
> on this machine (v2 §8.3: "the plan's only GPU-dependent line").

Derived from *Nihai Plan v2* (§7 calibration, §8.1–§8.4 evaluation, §11 risk register) and
grounded in the repository's remote-endpoint support
(`bfcl_eval/model_handler/local_inference/base_oss_handler.py:45-49, 120-141`).

---

## 0. The model is remote — tunnel first

This machine has no GPU. The model is served by vLLM on the BSC cluster and reached
through a forwarded local port. **Open and keep two terminals open** for the tunnel
(exactly as provided):

```bash
# Terminal A — reach the login node / compute node
ssh -L 8000:/home/boga/boga771710/socketdir/vllm.sock boga771710@alogin1.bsc.es
#   (equivalently, the socket form:  ssh -L ./vllm.sock:localhost:8000 as02r3b30)
# Terminal B — keep it open; this is the live vLLM endpoint on localhost:8000
```

**Smoke-test the tunnel before anything else** (the model at the socket is served under a
full path — use it verbatim as the `model` field):

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/apps/ai-hub/models/multimodal-models/Llama-4-Scout-17B-16E-Instruct",
    "messages":[{"role":"user","content":"What is the capital of Turkey?"}]
  }'
```

A 200 with a completion means the tunnel is live. If you get a connection refused / TLS /
407, the tunnel or proxy is down — re-open the SSH terminals; do **not** disable TLS or
unset `HTTPS_PROXY` (see `/root/.ccr/README.md`).

### 0.1 Point the BFCL harness at the tunnel (no local server spin-up)

The harness already supports an external OpenAI-compatible endpoint via env vars
(`base_oss_handler.py:45-49`) and skips its own vLLM launch with `--skip-server-setup`:

```bash
export REMOTE_OPENAI_BASE_URL="http://localhost:8000/v1"     # the tunnel
export REMOTE_OPENAI_API_KEY="EMPTY"                          # vLLM ignores it
# The governed handler is a Qwen subclass; the served model must match what the handler
# tokenizes. If the served model differs from the registry's HF id, supply the tokenizer
# explicitly so prompt formatting stays correct (base_oss_handler.py:120-141):
export REMOTE_OPENAI_TOKENIZER_PATH="/path/or/hf-id/matching/the/served/model"
```

Then every generation run below adds `--skip-server-setup` so the harness talks to
`localhost:8000` instead of trying to allocate a (non-existent) local GPU.

> **Model-identity note.** The governance handler `Qwen/Qwen3-4B-Instruct-2507-FC-GOV`
> (`model_config.py:1705-1711`) subclasses `QwenFCHandler`; its logic is model-agnostic but
> its prompt/tool formatting is Qwen-shaped. The A/B is only valid if **baseline and
> governed arms are served the same model** through the tunnel. If BSC serves Llama-4-Scout
> rather than Qwen3-4B, either (a) serve Qwen3-4B at the socket, or (b) add a governed
> subclass of the matching base handler and register `<model>-FC` / `<model>-FC-GOV` pairs.
> **Resolve which model the tunnel serves in Step 0 and fix the arm pair accordingly before
> spending replicate budget.**

---

## 1. Objective & scope

**Objective:** turn Plan 1's shadow-mode governance into a calibrated, statistically
defensible A/B result. Build the evaluation instruments the repo still lacks, calibrate
thresholds from live shadow logs (not end-task accuracy), run the 5× sequential
survival-conditional replicate experiment against the tunneled model, and deliver the
go/no-go on Vector drift and the `(0.95, 0.32)` pair.

**In scope (this plan):**
- §8.1 W/R/R/A snapshot parser (the repo's acknowledged instrumentation gap).
- §8.3 replicate runner (`run_gov_replicates.py`), analyzer (`analyze_gov_replicates.py`), JSONL manifest.
- Tunnel wiring + a live smoke run (1 scenario) proving governed and baseline arms both reach the model.
- §7 calibration: `(sim_high, δ)` coupled from live shadow logs; NLI thresholds from the labeled escalation sample; Stage 2 margin from the replay distribution.
- §8.2/§8.3 the two-arm A/B: `-FC` (baseline) vs `-FC-GOV` (governed-calibrated), 5× sequential, survival-conditional pairing, student pre-registered exclusion.
- §8.4 final live verdict updating Plan 1's offline replay verdict.

**Out of scope:**
- Any code Plan 1 owns (governance logic, Stage 1/2, replay core). This plan *runs* it.
- §5 placement/eviction; §6 read-time mechanism / MIG-judge arm (v2 §5, §6 — later).
- The dropped **SAGE arm** (v2 §8.2: structurally near-inert, a wasted replicate slot).

**Statistical guardrails (v2 §8.3, binding):**
- ≥3 (target **5**) replicates, **strictly sequential**, arms alternated within a replicate
  (B1,G1,B2,G2,…), `--num-threads 1`, dedicated vLLM server. Concurrency is forbidden
  (E10+MIG evidence: vLLM concurrent batching breaks A/B determinism).
- **Survival-conditional pairing:** if a prereq chain dies in *either* arm, drop that
  scenario-replicate pair; report the drop count as a primary statistic.
- **Student scenario is pre-registered-excluded** (zero writes every run); `--include-student`
  is a labeled sensitivity switch only.
- Calibrate on decision-log geometry (dense, cheap, deterministic); **validate** on
  end-task accuracy (sparse, ±8–10 pt noise). Never grid-search on accuracy (v2 §7, E11).

---

## 2. Prerequisites (verify before Step 1)

1. **Plan 1 Definition of Done met** — code merged, offline suites green, replay verdict written.
2. **Tunnel up** — §0 curl returns 200.
3. **Served model identified** — §0.1 note resolved; arm pair fixed.
4. **Baseline registry entry exists** — `Qwen/Qwen3-4B-Instruct-2507-FC` (baseline) and
   `-FC-GOV` (governed) both in `supported_models.py` / `model_config.py` (governed
   verified at `model_config.py:1705`, `supported_models.py:145`).
5. **ABTT artifact + encoder load** (Plan 1 §2). MiniLM/DeBERTa run CPU-side in the harness
   process; only the *policy model* is remote.

---

## 3. Ordered work items

### Step 0 — Tunnel + live smoke run  · v2 §8.3 (tunnel), §0 above
Prove both arms reach the model before building any statistics.

- Bring up the tunnel (§0) and export the `REMOTE_OPENAI_*` vars (§0.1).
- Run **one** memory scenario per arm with `--skip-server-setup`, e.g.:
  ```bash
  cd berkeley-function-call-leaderboard
  GOV_ENABLED=0 bfcl generate --model Qwen/Qwen3-4B-Instruct-2507-FC \
      --test-category <one_kv_memory_scenario> --skip-server-setup --num-threads 1
  GOV_ENABLED=1 GOV_DRY_RUN=1 bfcl generate --model Qwen/Qwen3-4B-Instruct-2507-FC-GOV \
      --test-category <one_kv_memory_scenario> --skip-server-setup --num-threads 1
  ```
  (Use the repo's actual CLI/entrypoint; if `bfcl` isn't the console script, invoke the
  generation module directly — confirm from `pyproject.toml` / `_llm_response_generation.py`.)
- **Acceptance:** both arms produce result files; the governed arm writes a
  `governance_log.jsonl` with `rehydrate` + `decision` events; no arm silently degrades
  (a missing ABTT artifact must raise, per `qwen_gov.py:111-114`).

### Step 1 — W/R/R/A snapshot parser  · v2 §8.1
The repo's real instrumentation gap: no parser for Write/Retrieve/Recall/Accuracy from the
memory snapshots. Must exist before replicate analysis.

- **New file `bfcl_eval/scripts/parse_wrra.py`:** read the per-scenario
  `memory_snapshot/<scenario>_final.json` (path convention verified in
  `qwen_gov.py:_load_snapshot`, 116-149) and the result/score files; emit per-arm,
  per-scenario Write count, Retrieve count, Recall, and end-task Accuracy as tidy JSONL.
- **New primary statistic — write-compliance / chain survival (v2 §8.1):** a chain is
  *dead* if the final snapshot has 0 entries **or** the prereq resolved 0 write calls.
  Report per arm (current single-run reference: baseline 3/5, governed 4/5 live).
- **Acceptance:** `test_parse_wrra.py` on a fixture snapshot tree; dead-chain detector
  flags a 0-entry final snapshot and a 0-write prereq.

### Step 2 — Replicate runner + analyzer + manifest  · v2 §8.3
- **New `bfcl_eval/scripts/run_gov_replicates.py`:** drive N replicates, **strictly
  sequential**, arms alternated within replicate (B1,G1,B2,G2,…), `--num-threads 1`,
  `--skip-server-setup` (tunnel). Emit a JSONL **manifest** per run: git HEAD, vLLM
  version (query `GET /v1/models` or the server banner), served-model id, server flags,
  all `GOV_*` env, seed, arm, replicate index, timestamps. A failed replicate **stops the
  runner** — never silently retried (v2 §8.3).
- **New `bfcl_eval/scripts/analyze_gov_replicates.py`:** consume manifests + Step 1's
  W/R/R/A + `gov_logs`. Implement: survival-conditional pairing (drop scenario-replicate
  pair if the chain died in either arm; report drop count), **McNemar** on matched
  surviving questions, scenario-level **bootstrap CI**, **Holm** correction. Implement the
  **student pre-registered exclusion**; `--include-student` is a labeled sensitivity flag
  only.
- **Acceptance:** `test_gov_replicates.py` (target v2's ~16 tests) on synthetic manifests:
  sequential ordering enforced; a dead chain in one arm drops the pair and increments the
  drop counter; McNemar/bootstrap/Holm produce correct numbers on a hand-checked table;
  `--include-student` toggles inclusion.

### Step 3 — Calibrate on log geometry  · v2 §7
Run a **governed shadow** pass over the tunnel to harvest dense decision-log geometry,
then set the knobs — **without touching end-task accuracy**.

- **Harvest:** `GOV_ENABLED=1 GOV_DRY_RUN=1 GOV_NLI_ENABLED=1 GOV_NLI_SHADOW=1` (Plan 1
  Step 5) across the memory suite via the tunnel. Full pipeline computes and logs; zero
  intervention. Produces `stage1` escalation records + Stage 0 geometry.
- **`(sim_high, δ)` — Vector:** set as a **coupled pair** from the shadow-log similarity
  distribution; test candidate `(0.95, 0.32)`; enforce the degenerate-region startup guard
  (Plan 1 Step 1). **KV geometric threshold is not calibrated** — KV is canonical-key gated
  (v2 §7, §8.4).
- **NLI `τ_entail, τ_contra, δ_spec`:** from a labeled sample of the ~194 logged
  escalations ("exactly NLI-shaped" data) — **not** grid-on-accuracy (v2 §7).
- **Stage 2 margin (`GOV_S2_MARGIN`):** start from the replay margin distribution (Plan 1
  Step 4), refine in live shadow.
- **Acceptance:** a committed calibration note fixing each value with its supporting
  log-geometry evidence, and confirmation the chosen `(sim_high, δ)` passes the startup
  guard (re-checking the Plan 1 §5 R1 inequality direction against the live distribution).

### Step 4 — The two-arm replicate A/B  · v2 §8.2, §8.3
The GPU-bound headline experiment.

- **Arms:** `-FC` (baseline, `GOV_ENABLED=0`) vs `-FC-GOV` (governed-calibrated, Step 3
  values baked into `GOV_*`). Optional bias-control shadow arm `GOV_DRY_RUN=1`. **SAGE arm
  dropped** (v2 §8.2).
- **Run:** `run_gov_replicates.py` with **N=5**, sequential, alternated, `--num-threads 1`,
  `--skip-server-setup`, student-excluded. Keep the tunnel terminals open for the entire
  run; a dropped tunnel = stopped runner (Step 2). This is long-running — launch it in the
  background and let completion notify; do not poll with `sleep`.
- **Primary question (v2 §8.3):** turn the single-run Vector drift (11.61 → 12.90) into
  evidence or noise, and validate `(0.95, 0.32)`. Single-run reference (pipeline demo, not
  evidence): survival-conditional KV baseline 8.8% vs governed 22.5% (McNemar p=0.043,
  suggestive); Vector p=1.0; bootstrap Δ KV +0.153 [−0.10,+0.44], Vector −0.011 [−0.12,+0.12].
- **Acceptance:** 5 completed replicates per arm with manifests; `analyze_gov_replicates.py`
  emits survival-conditional McNemar + bootstrap CI + Holm per backend, plus the drop count.

### Step 5 — Final verdict + cascade ablations  · v2 §8.2, §8.4
- **Update the §8.4 verdict** (Plan 1 Step 7's offline note) with the live replicate result:
  does the experiment confirm **Vector geometry-first (marginal)** and **KV
  canonical-key-first**, or overturn them? State it honestly, including whether the Vector
  CI still straddles the ρ=0.40 gate (v2 risk R1/R2).
- **Cascade ablations** (only now meaningful, since Stage 1/2 exist): governed-full vs
  Stage-0-only vs Stage-0+1; the ABTT-vs-identity ablation is already offline (v2 §1.1) —
  re-confirm via replay.
- **Acceptance:** a written results section in `STAGE0_RESULTS_ANALYSIS_AND_NEXT_STEPS.md`
  (or a new `EVAL_RESULTS.md`) with the replicate tables, CIs, drop counts, and the
  go/no-go decision, each figure traceable to a committed manifest + analysis artifact.

---

## 4. Definition of Done (Plan 2)

1. Tunnel-backed generation works for both arms with `--skip-server-setup` (Step 0).
2. W/R/R/A parser, replicate runner/analyzer, and manifest exist with passing offline
   tests (Steps 1–2).
3. Thresholds calibrated from live shadow-log geometry, not accuracy; `(sim_high, δ)`
   passes the startup guard (Step 3).
4. 5× sequential, survival-conditional, student-excluded A/B completed against the tunnel;
   McNemar + bootstrap CI + Holm reported per backend with drop counts (Step 4).
5. The §8.4 go/no-go verdict is updated from live data, with honest treatment of the
   marginal Vector gate and any divergence from the offline replay (Step 5).
6. Every reported number is traceable to a committed JSONL manifest / analysis artifact.
7. Committed and pushed to `claude/nihai-plan-v2-cascade-thresholds-k34u67`
   (`git push -u origin <branch>`).

---

## 5. Risks specific to Plan 2

- **R1 — served model ≠ handler model.** The governed handler is Qwen-shaped; if the tunnel
  serves Llama-4-Scout, prompt formatting diverges and the A/B is invalid. Resolve in Step 0
  (§0.1 note): serve the matching model, or add a matching governed subclass + registry pair.
- **R2 — tunnel drops mid-run.** A long sequential replicate loop is fragile to SSH
  timeouts; the runner stops on failure (by design). Keep both terminals open; consider
  `ServerAliveInterval` on the SSH session. Never work around a dead tunnel by disabling TLS.
- **R3 — Vector gate stays marginal.** If replicates render Vector drift as noise, the Stage 0
  accuracy claim leans on gate-correlation + mechanism validation until Stage 1/2 add
  suppression volume — a weaker but publishable position (v2 risk R1/R2). Report it as such;
  promise no more than the data supports.
- **R4 — determinism.** Any concurrency (multi-thread, vLLM batching across arms) breaks the
  paired comparison. Enforce `--num-threads 1` and sequential arms; the manifest records the
  server flags so a determinism violation is auditable after the fact (v2 §8.3, E10+MIG).
- **R5 — student contamination.** Mixing the student write-compliance fix into governance
  replicates conflates two effects. Keep the pre-registered exclusion; `--include-student`
  is sensitivity-only (v2 §5, §8.3).
