# Plan v3 — Entropy as the Compression Controller: Stage-C Consolidation (2026-07-23)

Status: **pre-registered, not started.** No code from this plan exists yet; gates
G0/G1' precede any GPU spend. Companion research brief:
`berkeley-function-call-leaderboard/docs/deep_research_prompt_entropy_memory.md`.

---

## 1. Thesis and repositioning

Plan v2 falsified entropy **as a harmful-write predictor at admission time**:

- Gate G1 (offline, chain-grouped LOSO-CV): ΔAUC(entropy | geometry+margin)
  = −0.032, CI95 [−0.085, +0.016]; NC shuffle indistinguishable
  (`gov_logs/CALIBRATION_FROZEN.md`).
- Entropy v2 redefinition (affine-invariant z-score entropy + von Neumann /
  Vendi neighborhood entropy): −0.030 [−0.070, +0.002] — *the layer was not
  the problem* (`gov_logs/entropy_v2_report.json`).
- Gate G2 (live, Phase 6): non-inferiority not established at n=3 ⇒
  `geometry_only` promoted to the Phase-7 headline arm
  (`gov_logs/ME_ABLATIONS_PHASE6_G2.md`).

Plan v3 tests the surviving hypothesis: entropy was **misplaced, not wrong**.
Entropy is the mathematics of compressibility; its natural role in a memory
system is not judging *which single write is harmful* but deciding *how to
compress the store* — what is redundant, what to evict, what to merge, how much
information the store effectively holds. That stage does not currently exist:

- Dedup is drop-not-merge (NOOP suppression, `geometry_gate.py` exact-dup fast
  path).
- Eviction exists only in `placement.py` (`value_scores`,
  `pick_eviction_victim`), is purely geometric (uniqueness + recency), default
  OFF (`GOV_P_ENABLED=0`), and has **never run live in any arm**.
- Capacity theory (embedding-capacity limits arXiv:2508.21038,
  more-documents-harm arXiv:2503.04388, hubness JMLR 2010) is cited in
  `docs/lit_review_write_gating.md` as motivation for write-parsimony but never
  operationalized; the core=7 / archival=50 hard caps are the only capacity
  mechanism.
- `vn_entropy` / Vendi effective rank (`entropy_metrics.py`) — literally a
  store-compression measure — is implemented and unit-tested, and was only ever
  used as the (failed) harm predictor.

**Pre-registered hypothesis H-C.** At capacity-pressure points, choosing the
eviction victim by *minimum marginal von Neumann entropy loss* (the most
informationally redundant item) preserves later-needed information at least as
well as the geometric value score and strictly better than random, yielding a
non-inferior (−2 pp per backend) official BFCL score for the `hybrid` arm vs
`geometry_only`.

## 2. Feasibility census (exploratory tier, 2026-07-23)

Read-only census over all 12 Phase-6 governed logs
(`gov_logs/me_ablations/rep0*/governance_log.jsonl`, `decision` records,
fields `tier`/`rho`):

| (backend, tier)        | decisions at ρ ≥ 1.0 | total | max ρ |
|------------------------|----------------------|-------|-------|
| kv, core (cap 7)       | **647**              | 1743  | 1.00  |
| vector, core (cap 7)   | **235**              | 2122  | 1.00  |
| kv, archival (cap 50)  | 0                    | 388   | 0.66  |
| vector, archival       | 0                    | 416   | 0.32  |

Core capacity **binds hard and often**; archival never binds. Consequences:

- v1 compression targets **core-tier eviction-to-archival only**. The archival
  effective-rank budget (the direct operationalization of more-documents-harm)
  is deferred to v2; v1 only *logs* per-tier `vn_entropy`/`effective_rank`
  trajectories.
- The "capacity never binds → plan is moot" kill-risk is largely retired for
  core, but G0 (§6) still formalizes the per-scenario trigger census before any
  implementation work.
- Behavioral confound, pre-declared: today an add against a full core produces
  a backend error the model *sees* and reacts to (retry, self-remove, redirect
  to archival). Stage-C makes that add succeed via an atomic evict-first
  expansion — the intervention necessarily changes model-visible dialogue.
  Effects therefore bundle "no more full-core errors" with victim choice; the
  `geometry_compression` control arm (§5) shares the first component and
  isolates the second, which is the entropy claim.

This census is exploratory-tier evidence (feasibility only). No threshold is
frozen from it.

## 3. Architecture: Stage-C consolidation pass

A **new pass, not an extension of `_placement_pass`**. GOV_P bundles category
routing (G6b), verbatim rewrite (G6a), and the destructive guard (G7) with
eviction; enabling it wholesale would confound the ablation and break the
"existing arms stay bit-identical" invariant.

**Hook.** `governance_filter.py` `govern_calls`: after the admission loop and
the existing `p_enabled` block, `if cfg.c_enabled:
self._consolidation_pass(governed)`; the `_apply_expansions` condition relaxes
to `p_enabled or c_enabled`. With `GOV_C_ENABLED` unset, every existing arm is
byte-identical (legacy replay gate re-run to prove it, 1,787/1,787 standard).
The middleware's only actuator is the pre-execution call list, so Stage-C runs
inside `govern_calls` against the mirror, exactly like placement's
eviction-move; nothing hooks `_observe`.

**Trigger (v1): capacity pressure only.** For each proceeding (non-NOOP'd)
core `add` when `cache.size("core") >= cache.capacity("core")` — any category,
not just identity — plan the atomic expansion
`[archival_add(victim), core_remove(victim)]` via the existing `_expansions`
mechanism, guarded by `archive_preflight_ok` (`placement.py`). Eviction is
**move-not-delete**: content survives in archival — the key grader-safety
property. No below-cap proactive pruning in v1.

**Mechanism (a) — entropy-marginal eviction (LIVE after gates).** New pure
module `middleware/consolidation.py` (no model calls, no BFCL imports):

- `entropy_marginal_scores(items)`: over the store's existing `emb_whitened`
  vectors E (n ≤ 7 core), `ΔS_i = vn_entropy(E) − vn_entropy(E \ i)` using
  `entropy_metrics.vn_entropy`. Victim = argmin ΔS_i (least information lost =
  most redundant), deterministic tie-break by ref (mirroring
  `pick_eviction_victim`). Cost: ≤ 8 eigendecompositions of ≤ 7×7 Grams per
  event — microseconds.
- `pick_victim(items, mode)`, `mode ∈ {entropy, geometry, random}`: geometry
  delegates to `placement.pick_eviction_victim`; random is the seeded negative
  control (offline use only).
- Every consolidation event logs **both** the entropy-marginal and geometric
  scores for all items (gov2 append-only event `"consolidation"`; v1 parsers
  ignore unknown events) — live arms therefore yield paired counterfactual
  data without a live random arm.

**Mechanism (b) — merge-not-drop (SHADOW-ONLY in v1).** Online NLI is banned
in new policies, so `last_copy_entailed` is unavailable online; the online
merge rule must be an entailment *proxy*: bidirectional whitened cosine ≥
`GOV_C_TAU_MERGE` **and** normalized exact value-containment (victim's value
substring-contained in survivor's after normalization). Survivor value
byte-preserved; the victim is superseded, never rewritten — **no value
rewriting ever** (grader needs exact facts). v1 only *logs* planned merges
(`GOV_C_MERGE_ENABLED=0`); live enabling requires the G1'-merge sub-gate
(§6). Offline NLI may audit the proxy at calibration time.

**Mechanism (c) — effective-rank budget: DEFERRED.** Logged-only in v1 (§2).

**Config (all env-driven via `GovConfig.from_env`).**
`GOV_C_ENABLED` (default 0) | `GOV_C_SHADOW` (default 1, shadow-first) |
`GOV_C_VICTIM` ∈ `entropy|geometry|random` | `GOV_C_MERGE_ENABLED` (default 0)
| `GOV_C_TAU_MERGE` (frozen at calibration) | `GOV_C_SEED` |
`GOV_C_SHUFFLE` (seeded shuffled-ΔS negative control, analog of
`GOV_ME_SHUFFLE_DH`).

## 4. `entropy_only` admission policy: `admit_nondup`

New `GOV_POLICY="admit_nondup"` in `GOV_POLICIES` + a short branch in
`admission_policy.MemoryMutationAdmissionBoundary.evaluate`: run
`GeometryGate.evaluate` but honor **only the exact-duplicate fast path** →
NOOP; everything else → ADD (`reason_code="admit_nondup"`); retrievability
floor, residual gate, and escalation bypassed. Store health rests entirely on
Stage-C.

Pre-registered interpretation limits: this arm is **diagnostic only** and
excluded from the primary comparison. It can show whether compression keeps a
permissively-admitted store usable (an upper bound on compression's corrective
power); it cannot attribute anything to entropy (no geometric-compression twin
at this admission setting), and its predicted failure would **not** falsify
H-C. Full geometry-off was rejected deliberately: it replicates the one
Holm-significant harmful config of the campaign (geo_off: KV −10.6 pp,
McNemar p=0.0001, Holm p=0.0013) and would tell us nothing new.

## 5. Ablation matrix (arm manifest sketch)

`gov_logs/compression_arms.json`, consumed by `run_gov_replicates.py`
(sequential, arms alternated within replicate, `--num-threads 1`,
RESUME_PROTOCOL, manifest pins git HEAD + calibration sha):

| arm | admission | compression |
|---|---|---|
| `baseline` | ungoverned (`gov_env: null`) | — |
| `geometry_only` | `GOV_POLICY=geometry_only` (Phase-7 headline config) | off |
| `geometry_compression` | `GOV_POLICY=geometry_only` | `GOV_C_ENABLED=1 GOV_C_SHADOW=0 GOV_C_VICTIM=geometry` |
| `hybrid` | `GOV_POLICY=geometry_only` | `GOV_C_ENABLED=1 GOV_C_SHADOW=0 GOV_C_VICTIM=entropy` |
| `entropy_only` | `GOV_POLICY=admit_nondup` | `GOV_C_ENABLED=1 GOV_C_SHADOW=0 GOV_C_VICTIM=entropy` |

`GOV_C_MERGE_ENABLED`/`GOV_C_TAU_MERGE` set identically in the three
compression arms per the G1'-merge outcome (0 unless the sub-gate passes;
τ frozen at calibration). geometry_only admission thresholds = the frozen
Phase-6 values, unchanged. `geometry_compression` exists for **attribution**:
without it, a `hybrid` win only shows *compression helps*, not *entropy helps*
— the unfalsifiability failure mode that sank Plan v2's entropy claim. It also
doubles as the first live validation of the SS5 placement eviction machinery.
Random-victim stays offline-only: the paired per-event score logging (§3a)
buys the counterfactual without +3 GPU arm-runs.

## 6. Phases and gates (pre-registered pass/fail consequences)

**Phase 0 — trigger census + kill-gate G0 (CPU, ~0.5 d).**
New `bfcl_eval/scripts/replay_compression.py --census`, modeled on
`replay_geometry_deltaH.py` chain-carry reconstruction (with its explicit
dropped-decision accounting). Over the 3× `geometry_only` reps: per
scenario × backend × chain, core-add-at-cap events (Stage-C trigger points),
the model's observed reaction to full-core errors today, counterfactual
merge-trigger counts across a τ sweep, and store entropy/rank trajectories.
**G0 (kill):** ≥ 30 counterfactual eviction triggers per backend, spread over
≥ 3 of the 4 live scenarios. FAIL → plan killed or rescoped; written up as a
scoping negative. (§2 predicts PASS for core; the merge mechanism's fate is
genuinely open.)

**Phase 1 — offline counterfactual + gate G1' (CPU, ~1 d).**
`replay_compression.py --counterfactual`: at each trigger point in the
replayed chain, pick victims under {entropy, geometry, random, shuffled-ΔS
NC}; label each stored item **later-needed** = retrieved by / verbatim-required
for any later read or graded answer in the same chain (built from
`outcomes_*.jsonl` + `label_outcomes.py` machinery; labels offline post-hoc
only, never in the feature path — no-GT-in-features assertion reused from
`calibrate_margin_entropy.py`). Degradation proxy: own-probe rank shift of
later-needed items after simulated removal, via `retrieval_sim`
(backend-faithful; whitened space never enters the simulation).
Chain-grouped leave-one-scenario-out throughout. Development on exploratory
logs; the G1' statistic is computed on the Phase-3 calibration harvest before
freezing.
**G1' (go/no-go for live entropy arms):**
(i) entropy victims' later-needed hit rate < random NC, chain-grouped
bootstrap CI95 excluding 0; **and** (ii) entropy ≤ geometry on later-needed
hit rate (non-inferior; superiority is the interesting result). The NC shuffle
must show no effect.
FAIL(i) → entropy-compression dies offline; no entropy arms go live;
optionally run `geometry_compression` alone as placement validation.
FAIL(ii only) → same, and `geometry_compression` becomes the sole treatment.
**G1'-merge sub-gate:** merges go live only at ≥ 95% counterfactual precision
that dropped items are never needed verbatim later; else `GOV_C_MERGE_ENABLED`
stays 0 in all arms.

**Phase 2 — implementation, shadow-first (CPU, ~1 d).**
`middleware/consolidation.py` (pure); `governance_filter.py` (GovConfig `c_*`
fields, `GOV_POLICIES += ("admit_nondup",)`, `_consolidation_pass`,
`_log_consolidation`); `admission_policy.py` (`admit_nondup` branch). Tests:
`scripts/test_consolidation.py` (ΔS correctness on synthetic Grams,
determinism/tie-breaks, shadow no-op byte-identity, atomic expansion ordering,
no-online-NLI startup assertion for both new arms) + extension of
`test_replay_admission.py`; legacy replay bit-identity gate re-run
(`legacy_full` and `geometry_only`, 1,787/1,787).

**Phase 3 — shadow harvest + calibration freeze (GPU, ~2 h).**
One arm (`geometry_only` admission + `GOV_C_ENABLED=1 GOV_C_SHADOW=1`) × 3
reps — computes and logs everything, intervenes never. Recompute G1' on this
calibration tier; freeze `GOV_C_TAU_MERGE` (and confirm the argmin-ΔS rule
needs no threshold) via the freeze ceremony →
`gov_logs/compression_calibration.json`, sha pinned in arm manifests. No
re-tuning afterward.

**Phase 4 — live 5-arm matrix + gate G2' (GPU, ~9–10 h).**
§5 matrix × 3 reps ≈ 15 arm-runs (Phase-6 scale).
**G2' (pre-registered):** primary = official BFCL score, **hybrid vs
geometry_only**, non-inferiority −2 pp per backend on the CI lower bound.
Attribution secondary = hybrid vs geometry_compression, replicated direction
across backends. Mechanism secondaries = eviction counts, victim
later-needed rate (post-hoc), store effective-rank trajectories,
useless-dup admissions, zero online NLI calls.
PASS → `hybrid` earns a candidate slot alongside `geometry_only` in the
Phase-7 confirmatory 5× (that promotion decision belongs to the master
campaign). FAIL → `geometry_only` stays headline; entropy-compression is
reported in the §24-style negative-result narrative — at that point entropy
will have been tested and found wanting at *both* the admission and the
compression stages, which is itself a publishable scoping result.

**Phase 5 — analysis (CPU, ~0.5 d).**
`analyze_gov_replicates.py` unchanged for the primary; new small
`analyze_compression_events.py` for mechanism secondaries → `EVAL_RESULTS_V3.md`
+ `EXPERIMENTS_OVERVIEW.md` update.

## 7. Statistics (binding; identical to master plan §19)

Official unmodified BFCL score as sole primary; survival-conditional pairing;
exact McNemar per backend; scenario-level bootstrap CI n=2000 seed 12345; Holm
over backend × arm-pair (m recorded); non-inferiority margin −2 pp per backend
on the CI lower bound; N=3 for this ablation matrix (N=5 only in the
subsequent confirmatory); `student` excluded at analysis time; strictly
sequential, single-threaded, one vLLM instance, RESUME_PROTOCOL; manifests pin
git HEAD + calibration sha; verbatim seed statement; no claims from pooled raw
counts or single replicates.

## 8. Risks and limitations (pre-declared)

1. **Grader-exact-fact loss.** Structurally mitigated: eviction is
   archive-then-remove (content survives); merges are shadow-only behind a
   precision gate; no value rewriting ever; rank-budget pruning deferred.
2. **Effect size vs noise floor.** Compression touches only at-cap
   trajectories; expected |Δ| likely below the ±8–10 pp single-run noise floor
   — hence non-inferiority + mechanism-secondary framing and paired stats,
   not superiority claims.
3. **Attribution.** Solved by `geometry_compression`; the residual shared
   confound (removal of model-visible full-core errors) is equal across
   compression arms, keeping the victim-choice comparison clean.
4. **Capacity binds only in core.** All claims scoped to core-tier capacity
   management; archival evidence is logged trajectories only.
5. **`entropy_only` expected degradation** (geo_off precedent) — diagnostic
   only, excluded from primary, exact-dup fast path retained.
6. **Replay chain desync** — reuse `replay_geometry_deltaH.py`'s explicit
   dropped-decision accounting; report counts in every offline artifact.

## 9. Effort

| phase | resource | estimate |
|---|---|---|
| 0 census + G0 | CPU | ~0.5 d |
| 1 counterfactual + G1' | CPU | ~1 d |
| 2 implementation | CPU | ~1 d |
| 3 shadow harvest + freeze | GPU (tunnel) | ~2 h |
| 4 live matrix + G2' | GPU (tunnel) | ~9–10 h |
| 5 analysis | CPU | ~0.5 d |

Kill-gates G0/G1' sit entirely before any GPU spend on entropy arms.
