# Plan 1 — Implementation & Offline Validation (CPU-only, no live model)

> **Sequence:** This is the **first** of two ordered plans. It builds and unit-tests
> all governance code on CPU without touching the remote vLLM server. Plan 2
> (`PLAN_2_EVALUATION_TUNNELED_GPU.md`) consumes the code, calibration profile, and
> offline verdict produced here, and runs the live A/B experiment against the
> tunneled BSC vLLM. **Do not start Plan 2 until Plan 1's Definition of Done is met.**

Derived from *Nihai Plan v2 — Geometrik Filtre → NLI → Retrieval Entropisi Kaskadı*
(§1.3, §2, §3, §4, §5 log-fix, §8.4 offline replay) and reconciled against the actual
repository state on branch `claude/nihai-plan-v2-cascade-thresholds-k34u67`.

---

## 0. Reality check — plan claims vs. repository

The v2 plan narrates several instruments as *"built and tested"*. They are **not present
in the tree**. The companion `STAGE0_RESULTS_ANALYSIS_AND_NEXT_STEPS.md` §11 confirms
this. Grounded status:

| Component | v2 plan says | Actual repo state | This plan |
|---|---|---|---|
| Stage 0 filter | built, 939 lines, 23/23 tests | **Present** — `bfcl_eval/model_handler/middleware/governance_filter.py` (939 lines) | extend, don't rewrite |
| `qwen_gov.py` handler | built | **Present** — `bfcl_eval/model_handler/local_inference/qwen_gov.py` (176 lines), registered as `Qwen/Qwen3-4B-Instruct-2507-FC-GOV` | extend hooks |
| ABTT artifact | committed | **Present** — `.../middleware/artifacts/abtt_minilm_l6_d16.npz` + `scripts/compute_abtt.py` | reuse as-is |
| `retrieval_sim.py` | built, 28 tests | **MISSING** | **build in Step 3** |
| `probe_gen.py` | built, 12 tests | **MISSING** | **build in Step 3** |
| `replay_geometry_deltaH.py` | built, 28 tests | **MISSING** | **build in Step 4** |
| Stage 1 (NLI) | GO, next component | **TODO stub** — `handle_escalation()` returns `(ADD, "stage0_escalate_fallback")` at `governance_filter.py:494-515` | **build in Step 5** |
| Stage 2 (Retrieval Entropy) | core built | **not present** | **build in Steps 3–4 (core) + Step 6 (live)** |
| Decision logs | 1,051 decisions | **Present** — `gov_logs/shadow` (497), `gov_logs/governed_calibrated` (999), `gov_logs/governed_sage` (890 lines incl. rehydrate/observe) | replay source for Step 4/7 |

**Consequence:** the v2 §8.4 "GO" verdict rests on an instrument that must first be
(re)built here. Step 7 re-derives that verdict from the committed `gov_logs/`, honestly,
before Plan 2 spends any GPU time.

---

## 1. Objective & scope

**Objective:** land every governance component that runs on CPU and can be validated
offline — the three logging fixes, the retrieval-entropy simulation core (which doubles
as the §8.4 replay instrument), Stage 1 (NLI) in shadow mode, the Stage 2 live layer,
and the threshold startup guard — each with deterministic unit tests, and re-derive the
offline go/no-go verdict from the committed decision logs.

**In scope (this plan):**
- §5 logging fixes: `_observe` remove/clear events, full-text (untruncated) audit, rehydrate-time ABTT metadata.
- §1.3 cache additions: `MemoryItem.probes`, `Stage0Signals.sims`.
- §4 Stage 2 **core**: `retrieval_sim.py`, `probe_gen.py` (byte-parity with backends).
- §8.4 replay instrument: `replay_geometry_deltaH.py` over `gov_logs/`.
- §3 Stage 1 (NLI): DeBERTa-MNLI singleton, escalation handler, shadow mode (`GOV_NLI_SHADOW=1`).
- §4 Stage 2 **live**: margin decision, deterministic canonicalization, probe cache/sidecar.
- §2.3 startup threshold validation + `GOV_STRICT_THRESHOLDS`.
- Offline unit tests for all of the above.
- §8.4 offline replay verdict re-derived from `gov_logs/`.

**Out of scope (moves to Plan 2 or later):**
- Any run that needs the live vLLM model (all live A/B, calibration-from-live-logs, W/R/R/A snapshot parser, replicate runner/analyzer). → **Plan 2.**
- §5 placement/eviction/archival machinery (deferred behind Stage 1's supersede path; v2 §5).
- §6 read-time mechanism and MIG-judge arm (v2 §6, retrieval-frequency pre-conditioned).
- Paraphrase probe model wiring beyond the `GOV_PROBE_PARAPHRASE=0` template-only default (v2 §4, risk R5).

**Guardrails (unchanged from v2):**
- Middleware never touches the BFCL backend or grader; all logic stays at the `decode_execute` / `_add_execution_results_prompting` boundary (`qwen_gov.py:151-176`).
- No input is ever bare-deleted; supersede without §5 = in-place replace + `superseded_text` full-text log (v2 §3.2, risk R3).
- A synthetic success must never mask a real backend error: every rewrite passes preflight first, else falls back to ADD (`decide()` `noop_blocked_by_preflight`, `governance_filter.py:537-541`; customer-6 lesson).
- The whitened (ABTT) space stays Stage-0-only; Stage 2 simulation runs in the raw MiniLM / BM25 space, byte-faithful to the backends (v2 §1.2).

---

## 2. Prerequisites (verify once, before Step 1)

```bash
cd berkeley-function-call-leaderboard
# 1. Encoder + ABTT artifact load (no GPU needed; MiniLM is CPU)
python -c "from bfcl_eval.model_handler.middleware.governance_filter import load_abtt, DEFAULT_ARTIFACT_PATH; \
           t=load_abtt(DEFAULT_ARTIFACT_PATH, 16); print('ABTT ok', t.u_top.shape)"
# 2. Existing Stage 0 offline tests still green (regression baseline)
python bfcl_eval/scripts/test_gov_offline.py
# 3. Decision logs present (offline replay corpus for Steps 4 & 7)
wc -l gov_logs/*/governance_log.jsonl
# 4. rank_bm25 + faiss + sentence-transformers importable (backend parity in Step 3)
python -c "import rank_bm25, faiss, sentence_transformers; print('deps ok')"
```

All four must pass. If `transformers` cannot fetch DeBERTa-MNLI in this sandbox, note it
— Step 5 gates the model load behind `GOV_NLI_ENABLED` (default off), so the code lands
and unit-tests with a stub; the real weights are exercised in Plan 2.

---

## 3. Ordered work items

Each step is independently committable and lists: **files**, **change**, **acceptance**.
Steps are ordered by dependency; do them in sequence.

### Step 1 — Threshold startup validation (`GOV_STRICT_THRESHOLDS`)  · v2 §2.3
The calibrated pair `(sim_high=0.95, δ=0.30)` is geometrically near-degenerate: the
single-neighbor bound `√(1−sim_high²)=0.312 > 0.30`, so a config that *looks* active can
never fire. Make that impossible to ship silently.

- **File:** `governance_filter.py` — `GovConfig` (lines 83-119).
- **Change:** add `strict_thresholds: bool = False` (from `GOV_STRICT_THRESHOLDS`).
  Add `GovConfig.validate()` computing `bound = math.sqrt(max(0.0, 1.0 - sim_high**2))`;
  if `delta > bound` (NOOP branch geometrically unreachable — the *degenerate* case the
  v2 doc flags) emit a `print("[GOV] WARNING ...")`; raise `ValueError` when
  `strict_thresholds`. Call `validate()` at the end of `from_env()` and log the outcome.
- **Acceptance:** unit test — `(0.95, 0.32)` passes (0.32 < 0.312? no → **note:** 0.32 > 0.312, so this pair is *just* inside the fire-able region only if the rule is `delta ≤ bound`; assert the intended direction against the v2 numbers and pin it in the test so the sign can't silently flip). `(0.80, 0.40)` warns; same under strict raises. `GOV_STRICT_THRESHOLDS=1` env round-trips.

> ⚠️ **Resolve the inequality direction first.** v2 §2.3 says `(0.95, 0.30)` is degenerate
> because `√(1−0.95²)=0.312 > 0.30`, yet also proposes `(0.95, 0.32)` as the fix. Both can't
> be "δ below the bound." Treat `bound=√(1−sim_high²)` as the max δ that still admits a
> one-neighbor NOOP; encode the guard as "warn when δ makes the region vanish," write the
> test to lock whichever direction the calibration in Plan 2 actually wants, and leave a
> comment citing this ambiguity so Plan 2's calibration step re-checks it.

### Step 2 — Three logging fixes + cache metadata  · v2 §1.3, §5 (log-fix only)
Prerequisite for a faithful replay (Step 4) and for Plan 2's log-based calibration.

- **File:** `governance_filter.py`.
- **2a — observe remove/clear events.** `_observe_kv`/`_observe_vector` (lines 893-913)
  call `cache.drop` / `cache.clear_tier` with **no** `log_gov_record`. Add an
  `event:"observe_remove"` / `event:"observe_clear"` record (mirroring the `event:"observe"`
  put log at 925-938) so the live mutation stream is complete. (Replay compensated from
  result files; the live system must log it directly — v2 §5.)
- **2b — full text.** Drop the `[:300]` truncation on `candidate_text` (line 818) and
  `text` (line 933); either log full text or add a sidecar. Downstream replay needs the
  verbatim value.
- **2c — rehydrate ABTT metadata.** The `event:"rehydrate"` record (`GovernanceSession.__init__`,
  742-754) omits ABTT provenance. Add `abtt_d`, `abtt_artifact_path`, and `abtt.meta`
  fingerprint so a replay can assert it reconstructed the same whitening.
- **2d — `Stage0Signals.sims`.** Store the full similarity vector `M @ v_w` (not just
  `sim_max`) in `compute_signals` (line 469). Stage 1 candidate ranking (`argsort`) then
  comes free, no re-embedding (v2 §2.1, §3.1).
- **2e — `MemoryItem.probes`.** Add `probes: list[str] = field(default_factory=list)`
  to `MemoryItem` (lines 187-196); populated at write time in Step 6, persisted to the
  `<scenario>_gov_state.json` sidecar (v2 §1.3).
- **Acceptance:** extend `test_gov_offline.py` — a remove and a clear each emit exactly
  one observe event; a decision record round-trips full (untruncated) candidate text;
  rehydrate record carries `abtt_d==16`; `signals.sims` length == `n_items`.

### Step 3 — Retrieval-entropy simulation core (Stage 2 core)  · v2 §4
Two new modules under `bfcl_eval/model_handler/middleware/`, pure/offline, backend-faithful.

- **New file `retrieval_sim.py`:**
  - `simulate_kv(query, keys, k=5)` — BM25Plus over **key names only**, tokenized
    **byte-identically** to `memory_kv._similarity_search`:
    `text.replace("_"," ").lower().split()` (verified `memory_kv.py:83-86`). KV retrieval
    scores keys, not values.
  - `simulate_vector(query, texts, k=5)` — raw MiniLM `normalize_embeddings=True` cosine,
    equivalent to `faiss.IndexFlatIP` over normalized vectors (verified `memory_vector.py:244,251-256,309-320`).
    Reuse the `_get_encoder()` singleton from `semantic_entropy.py`.
  - Per-tier fidelity; **whitened space never enters the simulation** (v2 §1.2).
- **New file `probe_gen.py`:**
  - Deterministic template channel: given a stored item, emit template probes with
    per-probe provenance. `GOV_PROBE_PARAPHRASE=0` (default) → template-only.
    Paraphrase channel is a stub (wired in Plan-2-era work; v2 §4 risk R5).
- **Acceptance:** new `test_retrieval_sim_offline.py` (target the v2 counts: ~28 sim tests,
  ~12 probe tests). Parity assertions: (a) KV tokenization is byte-identical to
  `memory_kv._similarity_search` on a shared corpus; (b) `simulate_vector` ranking matches
  a direct `faiss.IndexFlatIP` build for N≤57 (cosine ≤5e-4); (c) probe provenance is
  complete and generation is deterministic (same input → same probes).

### Step 4 — Offline replay instrument (`replay_geometry_deltaH.py`)  · v2 §4, §8.4
This is the §8.4 go/no-go instrument **and** the Stage 2 production core's offline twin.

- **New file `bfcl_eval/scripts/replay_geometry_deltaH.py`:**
  - Read a `gov_logs/<arm>/governance_log.jsonl`; reconstruct per-scenario memory state
    by chain-carry (rehydrate → decisions → observes), pulling full texts from the log
    (Step 2b) and from result files where needed.
  - **Suppressed writes are not applied; destructive ops come from results** (v2 §4).
  - For each decision compute `ΔH_neighbor` = change in top-5 retrieval entropy of the
    stored item's neighbors when the candidate is (hypothetically) added, using Step 3's
    `simulate_kv`/`simulate_vector`. Emit `N_eff` and `ΔH_mean`.
  - Output a per-decision JSONL + a summary with, per backend: `ρ(sim_max, ΔH_mean)`,
    duplicate vs non-duplicate median similarity, NOOP-region size/purity, and the count
    of suppressed duplicates with `ΔH ≥ 0`.
- **Acceptance:** `test_replay_offline.py` (~28 reconstruction tests). State
  reconstruction is exact on a hand-built fixture log; running against the committed
  `gov_logs/` produces a summary without crashing and reports how many of the N logged
  decisions were computable (v2 quotes 938/1,051 ≈ 94.5% — reproduce the *fraction*, and
  **log the dropped remainder explicitly**, never silently).

### Step 5 — Stage 1 (NLI), shadow mode  · v2 §3
Replace the `handle_escalation` ADD-fallback with a real NLI stage that, in shadow mode,
computes and logs everything but changes nothing.

- **File `semantic_entropy.py`:** add a thread-locked lazy `_get_nli_model()` singleton
  (DeBERTa-large-MNLI, CPU) mirroring the `_get_encoder()` pattern (lines 153-171).
- **File `governance_filter.py`:**
  - Add `GOV_NLI_*` to `GovConfig`: `nli_enabled` (`GOV_NLI_ENABLED`, default **False**),
    `nli_shadow` (`GOV_NLI_SHADOW`, default **True** on first deploy), `nli_k`
    (`GOV_NLI_K=3`), `tau_entail=0.75`, `tau_contra=0.75`, `delta_spec=0.10` (v2 §3.1-3.2).
  - Rewrite `handle_escalation` (494-515): pick top-`k` neighbors from `signals.sims`
    (Step 2d, `argsort`, **no re-embedding**); for KV **only**, run the canonical-key
    pre-check *first* (exact key present + value equal after `_normalize_for_match` → NOOP;
    key present + value differs → UPDATE_REWRITE) — this is KV's **primary** redundancy
    detector, resolved at dict-lookup cost with **no** NLI call (v2 §3.1, §8.4 KV verdict).
    Otherwise run bidirectional NLI and map to the v2 §3.2 decision table
    (equivalent→NOOP, more-specific→UPDATE-replace, borderline→keep-both/ADD,
    contradiction→UPDATE-contradiction, all-neutral→escalate to Stage 2).
  - **Rewrite mechanics (v2 §3.2):** extend `_pending[idx]` with `mode:"rewrite"`;
    `patch_results` must **not** inject a synthetic result for rewrites (the real backend
    result flows) and must **not** restore the original call (`_observe` must see the
    executed call); the original call is preserved in the log record. Rewritten op passes
    preflight first; on failure, fall back to ADD (never a guaranteed error).
  - **Shadow discipline:** when `nli_shadow`, compute the full decision + log a `stage1`
    block, but leave `governed[idx]` untouched (zero intervention). Backend routing of
    "old entry to archive" stays **in-place replace + `superseded_text` full-text log**
    until §5 exists (v2 §3.2, risk R3).
- **Acceptance:** `test_stage1_offline.py` with a stubbed NLI scorer (inject
  entail/contra/neutral scores; no weights needed). Assert the full decision table; assert
  KV canonical-key pre-check fires before any NLI call; assert shadow mode never mutates
  `governed`; assert a rewrite that fails preflight degrades to ADD.

### Step 6 — Stage 2 live layer  · v2 §4 (remaining work)
On top of Step 3/4's core, add the live decision path (still gated off by default).

- **File `governance_filter.py`** (+ small `probe_gen.py` use):
  - `GOV_S2_*`: `s2_enabled` (default False), `s2_margin` (start from the replay margin
    distribution — Step 4), `s2_canon_llm` (`GOV_S2_CANON_LLM`, default off).
  - Decision rule = min-margin, single-candidate, single canonicalization attempt,
    **no generate-race-select** (v2 §4.2).
  - Canonicalization: deterministic first (KV `_KV_KEY_PATTERN` + length; Vector 2000-char
    re-check — customer-6 lesson); LLM callback behind `GOV_S2_CANON_LLM=1`.
  - Probe cache + `<scenario>_gov_state.json` sidecar persistence (`MemoryItem.probes`,
    Step 2e); cache-miss regeneration flagged `probe_provenance:"stored_text"`
    (diagnostic only — never accept/reject on it).
  - Log `ΔH_neighbor` / `N_eff` as **non-decision** fields (v2 §4.5).
- **Acceptance:** `test_stage2_offline.py` — margin rule picks the right branch on a fixture;
  canonicalization respects KV pattern + Vector length; sidecar round-trips probes; live
  path is a no-op when `s2_enabled` is False.

### Step 7 — Re-derive the offline §8.4 verdict  · v2 §8.4
Run Step 4's instrument over the committed logs and write an honest verdict note.

- **Run:** `python bfcl_eval/scripts/replay_geometry_deltaH.py --logs gov_logs/governed_calibrated`
  (and `shadow`, `governed_sage`), summarizing per backend.
- **Deliverable:** append a short `## §8.4 offline replay (re-derived)` section to
  `STAGE0_RESULTS_ANALYSIS_AND_NEXT_STEPS.md` reporting, **from the actual run** (not the
  v2 doc's quoted numbers): Vector `ρ(sim_max, ΔH_mean)`, KV `ρ`, duplicate/non-dup
  separation, NOOP purity, and the computable-decision fraction. State plainly whether the
  data reproduces v2's *Vector geometry-first / KV canonical-key-first* split or not.
- **Acceptance:** the verdict is backed by a committed run artifact under
  `gov_logs/replay_*/`, and any divergence from v2's quoted `ρ ≈ −0.386 / +0.343` is
  called out rather than smoothed over.

---

## 4. Definition of Done (Plan 1)

1. `test_gov_offline.py` + all new offline suites (`retrieval_sim`, `replay`, `stage1`,
   `stage2`) pass on CPU with **no live model and no GPU**.
2. Steps 1–2 (startup guard + logging fixes + cache metadata) merged; existing Stage 0
   behavior byte-unchanged when `GOV_NLI_ENABLED`/`GOV_S2_ENABLED` are off (regression
   check: a governed run's decisions match pre-change on a fixture).
3. Stage 1 shadow (`GOV_NLI_SHADOW=1`) and Stage 2 live path exist, are gated off by
   default, and never intervene while shadowed.
4. The §8.4 replay instrument reproduces the go/no-go summary from committed `gov_logs/`,
   with the computable/dropped fraction reported explicitly.
5. Every new `GOV_*` env var is documented in the `GovConfig` docstring and echoed at
   handler startup (`qwen_gov.py:72-76`).
6. Committed to `claude/nihai-plan-v2-cascade-thresholds-k34u67` with descriptive messages;
   pushed with `git push -u origin <branch>`.

**Hand-off to Plan 2:** the calibratable knobs (`sim_high`, `δ`, `τ_entail`, `τ_contra`,
`δ_spec`, `s2_margin`), the shadow-mode Stage 1, and the replay verdict are the inputs
Plan 2's live experiment calibrates and validates.

---

## 5. Risks specific to Plan 1

- **R1 — inequality-direction bug (Step 1).** The v2 `(0.95, 0.30)` vs `(0.95, 0.32)`
  numbers are internally tension-y; a wrong sign makes the guard warn on good configs or
  pass bad ones. Pin the direction with an explicit test and a comment; Plan 2 re-checks.
- **R2 — DeBERTa-MNLI unavailable in the sandbox.** Step 5 lands behind `GOV_NLI_ENABLED`
  with a stubbed scorer for tests; real weights run in Plan 2. Do not block the merge on
  weight download.
- **R3 — replay ≠ live faithfulness.** The offline replay approximates state from logs;
  Step 2's logging fixes narrow the gap but destructive-op reconstruction still leans on
  result files. Report the computable fraction; never silently drop the remainder.
- **R4 — supersede without §5** = in-place replace; only `superseded_text` prevents data
  loss. Tracked; §5 is out of scope here (v2 §5).
