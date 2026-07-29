# H-Nav Stage 3 pre-registration — shadow-mode H_act validation

Frozen 2026-07-28, BEFORE the shadow campaign ran. Committed at the git head
recorded in `result_hnav_shadow/manifest.jsonl` (`run_start.git_head`). The
analysis in `evaluate_hact_shadow.py` implements this document verbatim; any
deviation must be reported as a protocol change, not silently applied.

## Campaign design

- Arms: `baseline` (`Qwen/Qwen3-4B-Instruct-2507-FC`, no governance) and
  `hact_shadow` (`...-FC-HACT` with `GOV_DRY_RUN=1`, `GOV_POLICY=geometry_only`,
  `HACT_POLICY=shadow`, `HACT_N=8`, `HACT_TEMP=0.7`, `HACT_LOGPROBS=5`,
  `HACT_GATE_RECALL=0`, `HACT_SEED=20260728`). Shadow mode changes NO behavior:
  the primary completion is returned unchanged and governance applies nothing.
- 3 replicates, `memory_kv,memory_vector`, temperature 0.001, no seed at the
  campaign level (replicate ≠ seed; server-side nondeterminism), runner
  `run_gov_replicates.py`, manifest is the log of record.
- Sanity requirement (not a hypothesis): the hact_shadow arm's official scores
  must be statistically indistinguishable from baseline (paired per-replicate
  deltas reported). A significant difference voids the campaign (shadow-mode
  invariant broken) and must be diagnosed before any analysis.

## Dataset partition

This campaign is the **calibration set** for any Stage 4 thresholds. The Stage 4
confirmatory campaign is a separate future run and must not be touched during
tuning. The 2026-07 exploratory/calibration sets (me_harvest, me_ablations)
contain no action-side data and cannot be used for H_act (no backfill).

## Targets

- **T1 (primary, candidate-level):** `hnav_target == must_suppress` from
  `label_outcomes_hnav.py` re-run on the new shadow logs — the write action
  caused counterfactual retrieval damage and carried no necessity, i.e. the
  action taken was wrong. Student scenario excluded (pre-registered, Plan v2).
- **T2 (step-level):** primary produced no governed call on a gated prereq step
  (orphan: no-call or parse failure). Features exclude everything derived from
  the primary's own action content (exploration-only votes; `lp_tool_*`
  excluded as collinear).
- **T3 (exploratory, descriptive only):** chain-level mean H_act vs official
  chain accuracy. Never gated on.

## Models and contrasts

Nested ladder M0..M10 as defined in `evaluate_hact_shadow.py` (`NESTS`).
Leave-one-scenario-out out-of-fold predictions; chain = (backend, scenario,
replicate); grouped chain bootstrap (n_boot 2000, seed 12345) for deltas.

Pre-registered delta family (Holm-corrected as one family):
`(M8-M4)` **headline** — Shannon entropy beyond its own vote-margin controls;
`(M5-M4)`, `(M6-M4)`, `(M7-M4)` — per-resolution increments;
`(M3-M2)` — token logprobs beyond geometry+margin;
`(M4-M3)` — vote controls beyond logprobs;
`(M10-M8)` — normalization + interactions beyond raw entropies;
`(M4-M0)` — the PARTIAL-GO criterion (self-consistency signal exists at all).

Univariate within-chain permutation family (Holm as one family, n_perm 10000):
`h_act_target`, `h_act_op`, `vote_margin_target`, `lp_tool_min` on T1.

Metrics per nest: ROC AUC, PR-AUC, Brier, ECE. Rare-event discipline: with
fewer than 25 positives no multivariate claim is made (EPV gate, Stage 1
precedent) — the univariate permutation family is then the only T1 evidence.

## Negative controls (all must behave)

NC2 within-(backend, op_add) joint shuffle of all entropy features (`bin_fn`);
NC3 seeded Gaussian-noise replacement of entropy features; NC4 global label
shuffle (AUC→~0.5); NC5 within-chain label shift (step t features vs step t+1
label); NC10 seed+1 stability; NC12 leave-one-replicate-out sign consistency;
NC13 oracle-path refusal; NC14 duplicate candidate/hact-row refusal; NC8
whitelist + lexical leakage guard over every merged feature vector; NC9
fold-straddle assertion inside `cv_splits`; NC11 permutation control on T2.

## Decision gate (mechanical, `decide_gate`)

- **GO** ⟺ T1 dAUC(M8−M4) bootstrap CI95 excludes 0 AND Holm-adjusted
  permutation p(`h_act_target`) < 0.05 AND NC2 collapses the headline
  (|mean| < 0.02 or CI covers 0) AND NC12 deltas all positive AND
  n_positives ≥ 25.
- **PARTIAL** ⟺ not GO, but dAUC(M4−M0) CI95 excludes 0 with NC2 collapsed
  and n_positives ≥ 25 → only vote-based intervention arms (A2 majority,
  A3 vote-margin gate) proceed; no entropy-threshold arms.
- **NO_GO** ⟺ neither → the autonomous alternative loop starts; the negative
  finding is reported as a finding.

## Protocol corrections (2026-07-29, pre-unblinding)

An adversarial multi-agent code review of the evaluator ran while the campaign
generated; the following corrections were applied AFTER the campaign completed
but BEFORE `evaluate_hact_shadow.py` was executed on any campaign data (the
evaluator had produced no numbers when these landed — this is a pre-unblinding
protocol fix, not a post-hoc adjustment):

1. **Replicate-aware joins.** `candidate_id` carries no replicate component,
   so identical decisions recur across replicates with potentially different
   counterfactual labels. All label joins and the NC14 dedup now key on
   `(replicate, candidate_id)`; the labeler is invoked with
   `--replicate <gov-log dir name>`.
2. **PARTIAL gate control corrected.** The original PARTIAL clause required
   "NC2 collapsed", but NC2 permutes only entropy features, which appear in
   neither M4 nor M0 — it cannot collapse an entropy-free delta, making
   PARTIAL structurally unreachable. The corrected clause uses **NC2b**: a
   joint within-(backend, op_add) shuffle of the VOTE_CONTROL_FEATURES block,
   the exact analogue of NC2 for the vote signal. GO is unchanged.
3. **Twin handling.** `HACT_SEED` was replicate-invariant in this campaign,
   so a fraction of candidates recur byte-identically across replicates.
   NC12 replicate-holdout now excludes training rows whose `candidate_id`
   appears in the held-out replicate; threshold validation (Stage 4) excludes
   val rows whose id appears in dev; twin counts and twin-label conflicts are
   reported. Future campaigns use per-replicate seeds (runner offsets
   HACT_SEED by the replicate index).
4. **Known T2 limitation (this campaign's data).** The orphan flush was
   skipped when the governed decode RAISED (malformed-but-matching tool_call
   JSON), so the parse-crash subclass of T2 positives is undercounted in this
   campaign's logs. Fixed for future runs (exception-safe flush with a
   `parse_crash` flag); for this campaign T2 covers the no-call subclass and
   the surviving parse-fail records only, and is interpreted accordingly.

## Stage 4 preview (bound by this gate)

On GO/PARTIAL: staged intervention campaign, first wave A0 (baseline), A1
(equal-budget random-select control), plus the two best-justified arms from
shadow evidence, 3 replicates. Primary contrast: each arm vs **A1** (equal
compute), secondary vs A0, via `analyze_gov_replicates.py` (survival-conditional
pairing, exact McNemar, bootstrap over (replicate, scenario), Holm, student
excluded). Thresholds selected on shadow reps 1–2, validated on rep 3, frozen +
sha-pinned BEFORE the confirmatory campaign.
