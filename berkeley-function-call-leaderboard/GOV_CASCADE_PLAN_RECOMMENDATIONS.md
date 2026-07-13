# Recommendations for the Final Cascade Plan
### (Geometric Filter → NLI → Retrieval Entropy, BFCL v4 Memory, KV + Vector)

Source of evidence: the Stage 0 implementation on branch `just_geo_filtering` (commit
`84d831d`) and three full generation runs through the BSC tunnel on 2026-07-13 —
one shadow (dry-run), one governed with SAGE defaults, one governed with calibrated
thresholds — plus a fresh baseline. ~900 logged write decisions across
`gov_logs/{shadow,governed_sage,governed_calibrated}/`. Everything below is grounded
in those logs and scores, not in design intuition.

---

## 1. The single biggest empirical finding: fix write-compliance before the cascade

**The dominant failure mode in every arm is scenarios where the model writes
*nothing*.** In all four runs (baseline, SAGE, calibrated, shadow), the *student*
scenario ended with an empty memory store on both backends — the model spends all
10 prereq turns chit-chatting ("That's an amazing journey…") without a single tool
call. That floors ~50/155 recall questions in **every** arm. *Healthcare* died the
same way in the baseline-KV run (0 items) but survived in the governed runs — pure
run-to-run luck that swung memory_kv accuracy from 7.1% to 16.8–21.3% with only 3–5
actual suppressions. **Between-run scenario survival moves scores by ±10 points;
no single cascade mechanism will produce a treatment effect that survives this
noise.**

Consequences for the plan:

- **Add a "Stage −1: write-compliance watchdog" to the middleware**, before any
  governance. It is trivially detectable: a prereq conversation that reaches its
  final turn with zero successful writes (the middleware mirror already knows this)
  is a guaranteed catastrophic failure for the whole scenario chain. Options, from
  cheap to invasive: (a) log-and-report only (minimum: makes runs comparable by
  exclusion); (b) inject one corrective user-style nudge message; (c) re-roll the
  turn at higher temperature. Even option (a) is mandatory for the thesis: without
  conditioning on chain survival, the §8.2 comparisons measure luck.
- **Re-prioritize Month 1** (§9). The plan calls mandatory archival "the most
  guaranteed win." In ~900 logged decisions, destructive ops were rare; empty-chain
  failures cost ~1/3 of the benchmark in every single run. The most guaranteed win
  is write-compliance detection + survival-conditioned scoring, and it is also the
  cheapest thing on the whole roadmap.
- **§8.3 statistics**: McNemar over pooled questions is not enough. Pair at the
  *scenario × backend* level, and pre-register the rule "scenario-runs where either
  arm's prereq chain died are excluded and reported separately." Otherwise ≥3 seeds
  mostly measures the chit-chat lottery.

## 2. Corrections to assumptions baked into the plan text

### 2.1 The anisotropy premise (§1.2) is factually wrong for this corpus
The plan justifies ABTT with "raw MiniLM is anisotropic (median pairwise cosine
~0.83)." Measured on the actual pooled prereq corpus (1,857 sentences, raw
mean-pooled MiniLM): **median pairwise cosine = 0.096**, dropping to ≈ −0.003 after
ABTT (D=16 removes 33.7% of variance). ABTT still does useful work (mean removal +
dominant-direction removal sharpens the near-duplicate band), but:
- Rewrite the rationale; a reviewer who checks the number will find it off by 8×.
- The planned ablation "domain-specific whitening superiority" (§1.1) may not
  deliver the expected narrative — on an already near-isotropic space the
  measurable effect of ABTT is subtle. Before promising that finding, measure what
  ABTT actually changes: the sim_max histogram separation between known-duplicate
  pairs and known-novel pairs, with vs. without ABTT. Claim only what that shows.

### 2.2 SAGE initials are geometrically incompatible with the NOOP rule (§2.2–2.3)
Hard constraint discovered in the logs: `r ≤ √(1 − sim_max²)`, and in practice
`r ≈ √(1 − sim_max²)` (candidates project mostly onto their nearest neighbor). With
SAGE `δ = 0.025`, the condition `r < δ` requires `sim_max ≳ 0.9997` — i.e., NOOP
fires **only on byte-identical strings** (SAGE-default run: 3 NOOPs, all sim = 1.0).
The empirical redundancy band sits at sim 0.95–0.99 with r 0.13–0.30. Working
calibration from this session: **sim_high = 0.95, δ = 0.30** (5 NOOPs, all genuine
redundant rewrites; e.g., the same Levothyroxine text re-written at sim 0.992).
Recommendations:
- Treat δ and τ_high as a *coupled pair*: the NOOP region is only non-empty when
  `δ > √(1 − τ_high²)`. Document this constraint in §2.3; grid-search over the
  pair, not each independently.
- SAGE's τ-formula remains fine for the *novelty* threshold (r > τ_t → ADD); it is
  the δ ("r ≈ 0") side that SAGE values cannot serve.

### 2.3 Per-backend thresholds are probably necessary
Even calibrated, **zero KV writes were suppressed** across all runs. Cause: KV
candidates are embedded as `"key words: value"` composites, and differing keys
depress similarity below 0.95 even when the value is redundant (historical KV
redundancy band: archival replaces at sim 0.92–0.97). Options: (a) per-backend
sim_high (KV ≈ 0.92); (b) embed KV value-only for the sim signal and keep the
composite for the verbatim gate; (c) accept that KV redundancy is mostly
same-key-replace, which Stage 1's canonical-key channel (§3.1) will catch anyway.
Decide with a KV-only shadow sweep, not by intuition.

### 2.4 Grid-search on end-task metrics (§7) will overfit noise
With ~30 recall questions per scenario-set and ±10-point survival noise, a grid
over 5+ hyperparameters against Retrieve+Answer accuracy fits the lottery. What
worked in practice this week: **calibrate from shadow-mode decision-log geometry**
(distributions of sim_max, r, verbatim status over hundreds of decisions — no extra
generation cost), then use end-task metrics only to *validate* the chosen operating
point. Recommend restructuring §7: every threshold gets its first value from
shadow logs; only marj (Stage 2) and the read-time threshold, which have no
shadow-derivable ground truth, get end-metric tuning.

## 3. Mechanisms the plan is missing (all hit in implementation)

These are not theoretical: each one either fired in production this week or was
required to make the middleware work at all. They belong in the plan text so
Stages 1–2 inherit them.

1. **Preflight safety rule.** A NOOP may only be issued when the middleware mirror
   predicts the real call would have *succeeded* (key not duplicate, id exists,
   store not full, length OK). Otherwise the synthetic success masks a real backend
   error and the model's world-model diverges from the store. This fired **24
   times** in the calibrated run alone. Must apply identically to Stage 1's
   NOOP/UPDATE routings and Stage 2's accept path.
2. **Decoy-rewrite, never call-removal.** An empty decoded call list short-circuits
   BFCL's step loop before results are injected (`base_handler.py:565-574`), and
   results zip 1:1 with decoded calls. Suppression must rewrite the call in place
   to a read-only decoy (`core_memory_retrieve_all()`) and patch the result
   afterward. Any future stage that suppresses or reroutes calls must reuse this
   mechanism — put it in §0 as an architectural invariant.
3. **Vector shadow-ids.** A suppressed vector `add` must return a synthetic
   `{"id": N}`. Returning the predicted real next-id silently collides with the
   next genuine add; out-of-band ids (9000+) fail loudly and recoverably if later
   referenced. Note the interaction with §3.2's "route contradiction to UPDATE":
   Stage 1 must never route an update onto a shadow id.
4. **Cross-conversation middleware state needs a sidecar file.** The mirror cache
   is rehydrated each conversation from the backend's snapshot — that works for
   embeddings (recomputable) but NOT for things the plan requires later:
   `turn_written`, `low_confidence` flags (§1.3), and especially Stage 2's
   **cached probe sets per entry** (§4.4 assumes probes persist so ΔH_komşu is
   re-scoring only). Recommend a middleware-owned sidecar
   (`<scenario>_gov_state.json`) written at the same prereq-flush moment, read-only
   with respect to BFCL's own files. Without this, §4.4's "cache'lendiğinden ek
   maliyeti yalnızca yeniden skorlama" is not implementable.
5. **Verbatim gate needs a defined no-values branch.** On preference-type sentences
   ("the user likes tea") the value regex extracts nothing, and the plan's rule
   becomes vacuous — the exact case where geometry alone would false-NOOP
   coffee→tea. The implemented fallback (content-token containment when no
   structured values exist) protected 439/566 decisions. Specify this branch in
   §2.1 rather than leaving it to the implementer.
6. **Sequential runs only.** Running two arms concurrently against one vLLM server
   introduced batching nondeterminism at temperature 0.001 and (likely) flipped
   borderline chit-chat-vs-write turns. Protocol: arms run sequentially, and the
   comparison notes vLLM version + flags. This belongs in §8 as a validity
   condition.

## 4. Good news the logs give the cascade design

- **The escalate band is exactly NLI-shaped.** 68–81 escalations per governed run
  (~15–20% of gated writes), populated overwhelmingly by verbatim-clean
  near-paraphrase updates/replaces at sim 0.90–0.95 — precisely the
  entailment/specificity cases §3.2 is designed for. The cascade's core bet (cheap
  stage resolves clear cases, ambiguity is a small band) is empirically confirmed.
- **Stage 1 volume is affordable.** At ~75 escalations × 2k NLI calls (k=3 → 6
  calls each) ≈ 450 DeBERTa CPU calls per full run. Fine.
- **Latency budget holds with margin.** Full decision path (QR residual + verbatim
  + thresholds) at maximum memory (57 items): **0.53 ms**, encode excluded. The
  <1 ms/write target (§2.3) is real; incremental QR remains unnecessary at this
  scale (full `np.linalg.qr` recompute is tens of µs — keep the function boundary,
  skip the complexity).
- **The audit-log design pays for itself.** Every recommendation in this document
  came out of the JSONL decision logs. Extend the same schema to Stages 1–2 from
  day one (verdicts, entailment probabilities, probe sets, margins), and add
  `GOV_DRY_RUN`-style **shadow mode to every stage** — it turns ablations (§8.2)
  into log post-processing instead of extra GPU runs.

## 5. A cheap, high-value addition: the offline replay harness

Every Stage 0 decision point is fully serializable (snapshot state + candidate
call). That means the entire cascade can be **replayed offline** against recorded
runs: feed logged snapshots + candidates through new stage logic and score the
decisions, without any model generation. Recommended as a Month-2 deliverable
alongside the go/no-go study — it converts threshold tuning, Stage 1 prompt/model
selection, and the §8.4 geometry↔ΔH correlation into pure-CPU iterations. The
existing `test_gov_offline.py` + fabricated-snapshot pattern is 70% of the way
there.

Related: the §8.4 go/no-go correlation does not need to wait for Month 2 — the
~900 logged decisions already contain (r, sim_max) per candidate, and ΔH_komşu is
computable offline from the stored snapshots by replaying probes. Do it now; if ρ
lands below 0.4, better to know before building Stage 1 around geometric
escalation.

## 6. Smaller notes

- §0 diagram uses generic op names (`memory_add`, `memory_search`) — actual gated
  ops differ per backend (KV: `core|archival_memory_add/replace`; Vector:
  `core|archival_memory_add/update`; reads are `*_retrieve*`/`*_key_search`). Keep
  the generic names for exposition but add the concrete op table; Stage 1's
  canonical-key channel and the read-time mechanism both depend on the real
  signatures (e.g., Vector has no keys at all — §3.1's "vector'da da anahtar alanı
  var" is incorrect; Vector entries are id+text only, so the canonical-key channel
  is KV-only unless keys are synthesized into stored text).
- Threshold EMA state (§2.3) currently resets per conversation, seeded from the
  density formula. If cross-conversation EMA matters, it rides the same sidecar
  file as item 3.4 above.
- rec_sum exclusion (§5.6) is cleanly achievable at the handler level (backend
  detection already routes rec_sum to pass-through); the NLI blob-diff proxy can
  reuse the same `GovernanceSession` skeleton with a different adapter table.
- Windows/console practicalities that cost time this week and will cost it again:
  BFCL's evaluator crashes on emoji under cp1254 (`PYTHONUTF8=1` required), and
  the repo's own baseline artifacts (`result_baseline/`) were partial (55/155,
  vector-only) — always verify entry counts before treating an old run as a
  baseline.

## 7. Session numbers for reference

| Arm | memory_kv | memory_vector | NOOPs | Preflight blocks | Escalations |
|---|---|---|---|---|---|
| Baseline `-FC` | 7.10% (11/155) | 11.61% (18/155) | — | — | — |
| GOV, SAGE defaults | 21.29% (33/155) | 12.26% (19/155) | 3 (all vector, sim=1.0) | 1 | 81 |
| GOV, calibrated (0.95/0.30) | 16.77% (26/155) | 12.90% (20/155) | 5 (all vector, sim 0.95–1.0) | 24 | 68 |

Score deltas across arms are dominated by prereq-chain survival (student dead in
all arms; healthcare dead only in baseline-KV), **not** by governance — which is
itself the strongest argument for §1 of this document.
