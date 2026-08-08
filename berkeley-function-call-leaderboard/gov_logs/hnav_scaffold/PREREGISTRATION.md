# alt5R write-scaffold campaign — pre-registration

Frozen 2026-07-29, BEFORE the campaign ran. Ledger: `alt5R_write_scaffold`
(hypothesis slot #5, replacing the pre-specified "adaptive sampling budget";
justification recorded in the ledger entry and below).

## Why this replaced pre-specified alternative #5

The directive permits replacing an alternative when repository evidence gives
a strong reason. Adaptive sampling is a cost optimization — it cannot add
accuracy, and Stage 3 showed the sampling signal it would ration (H_act) has
no predictive value here. The loss decomposition instead identified the
dominant loss channel, which no listed alternative targets:

| stratum | kv | vector | p(correct) kv / vector |
|---|---|---|---|
| gold not carried by final store | 81.0% | 77.3% | 0.049 / 0.043 |
| carried but not retrievable | 5.2% | 1.8% | 0.354 / 0.059 |
| carried + retrievable | 13.9% | 20.9% | 0.380 / 0.490 |

(6 shadow arm-replicates, 930 question instances per backend, production
grader.) **~88% of the uncarried gold facts were stated verbatim in the prereq
conversation** — the agent had the information and never wrote it.

## Policy under test

`Qwen/Qwen3-4B-Instruct-2507-FC-SCAF` (`QwenScaffoldHandler`): when a memory
**prereq** turn ends with no resolved write, archive the current user turn
text via one `archival_memory_add`. Runtime information only — the current
user turn. No gold, no future turns, no evaluation labels, no grader/dataset
changes. Capacity-capped at the backend's own limits (50 archival entries,
2000 chars); KV keys are deterministic content slugs so the entry is visible
to the backend's BM25-over-key-names retrieval.

This is an agent memory-policy change (MemGPT-style auto-archiving), of the
same kind as the `-FC-GOV` / `-FC-SE` / `-FC-MIG` handler variants before it.

## Arms (3 replicates, `memory_kv,memory_vector`, temperature 0.001)

| arm | model | purpose |
|---|---|---|
| `baseline` | `-FC` | reproduced fair baseline |
| `scaffold` | `-FC-SCAF`, `SCAF_MODE=user_turn` | the policy |
| `scaffold_shuffled` | `-FC-SCAF`, `SCAF_MODE=shuffled_control` | **content-value control**: identical write volume, identical word multiset, deterministically scrambled word order — isolates "the store got the FACT" from "the store got MORE TEXT" |

## Pre-registered hypotheses and tests

- **H-S1 (primary).** `scaffold` > `baseline` on official accuracy, matched
  questions, per backend. Test: exact McNemar on survival-conditional matched
  questions + cluster bootstrap over (replicate, scenario) via
  `analyze_gov_replicates.py`, Holm-corrected across the {backend} × {arm-pair}
  family. Success needs CI excluding zero AND consistent direction across the
  3 replicates.
- **H-S2 (mechanism).** `scaffold` > `scaffold_shuffled`. If the shuffled
  control matches the real scaffold, the gain is retrieval-surface volume, not
  fact capture, and must be reported as such.
- **H-S3 (mechanism check, offline).** Answerability (gold carried AND in
  top-3) rises in the `scaffold` arm's final stores vs `baseline`, in the
  direction the falsifier predicted (kv +18.9pp, vector +41.9pp answerability).

Pre-registered failure modes to report honestly if they occur: pollution
(scaffold entries crowding out genuine writes → accuracy drop on questions
that baseline answered), capacity exhaustion mid-chain, and turn-dynamics
cost (one extra model step per scaffolded turn — reported in the cost table).

## Expected effect (from the offline falsifier, NOT a result)

`gov_logs/hnav_autonomous/falsifier_write_scaffold.json`: expected
dAcc +0.077 (kv), +0.171 (vector) at p_hat = 0.409. The live campaign is the
test; the offline number assumes constant answerability→accuracy conversion
and unchanged trajectories, both of which the live run will violate to some
degree. Any live gain materially below this is itself informative about the
conversion factor.

---

## AMENDMENT 1 — tier-conditional conversion factor (2026-08-08, BEFORE launch)

The campaign pre-registered above has **not yet run**. Before launching it, a
tier-conditional re-analysis of the same 6 shadow arm-replicates
(`gov_logs/hnav_autonomous/tier_conditional_factors.json`, pooled in
`tier_conditional_pooled.json`) found that the conversion factor
`P_HAT = 0.408544` used by `falsifier_write_scaffold.py` is **tier-blind and
mis-priced for this policy**:

| gold carried in | kv n / p(correct) | vector n / p(correct) |
|---|---|---|
| core (auto-dumped into context) | 117 / 0.556 | 137 / **0.701** |
| **archival only** | 60 / **0.017** | 74 / **0.000** |
| not carried | 753 / 0.049 | 719 / 0.043 |

Independently verified from the result logs: only 6 of 155 vector question
entries issue any memory call, and exactly **1** ever touches archival. The
agent answers from the in-context core dump and never reads archival memory.

The scaffold writes to **archival**. Corrected standalone prediction
(`Δanswerable × (p_archival − p_not_carried)`):

| backend | falsifier Δanswerable | old expected ΔAcc | **corrected standalone ΔAcc** |
|---|---|---|---|
| kv | +0.189 | +0.077 | **−0.006 ≈ 0** |
| vector | +0.419 | +0.171 | **−0.018 ≈ 0** |

Range over a hypothetical read-conversion factor c (vector):
`ΔAcc = 0.419 × (c − 0.043)` → c=0.25: +0.087; c=0.50: +0.192; c=0.70: +0.276.

**Consequences, frozen now:**
1. The scaffold campaign as originally specified (capture without read) is
   predicted ≈ 0 by its own corrected instrument and will **not** be launched
   standalone. H-S3's predicted answerability movement remains testable and is
   retained.
2. The capture policy is folded into the successor RAG program as arm
   `pack+read` (capture × read interaction), pre-registered separately. The
   read-conversion factor c is the first quantity that program measures (C1).
3. Every future Δanswerability→ΔAcc conversion must use tier-conditional
   factors and report a range over c, never a single pooled scalar.
