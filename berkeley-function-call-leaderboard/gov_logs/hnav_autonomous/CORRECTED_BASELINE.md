# Corrected-baseline report (directive §3.3)

**Written:** 2026-08-08. Data: the three baseline arm-replicates of the stage4
campaign (`govrep_20260729T004515Z`, git `cc5dca7`, temp 0.001, one server
instance), pooled; per-question rows in `gov_logs/hnav_stage4/wrra/`.

## 1. Original vs corrected, the 2×2

| | kv | vector |
|---|---|---|
| **all 5 scenarios (465 q/backend, official denominator)** | 52/465 = **0.1118** | 57/465 = **0.1226** |
| **student excluded (315 q/backend, sensitivity)** | 49/315 = **0.1556** | 54/315 = **0.1714** |
| official leaderboard (`BFCL_MEMORY_REPRODUCTION_GUIDE.md`) | 0.1613 | 0.1226 |

**Vector reproduces the official number exactly (0.1226 vs 0.1226).** The
user's primary target has a trustworthy baseline; no correction is applied to
vector.

**kv is 4.9pp below official (0.1118 vs 0.1613)** and the gap is fully
explained by the dead `student` chain: student-excluded kv (0.1556) is within
noise of the official number. The guide's attribution of the *whole* tree gap
to student is **wrong for vector** (retracted here) and right for kv.

## 2. The student zero-write failure is behavioural, not a harness defect

Three lines of frozen evidence:

1. **Determinism across 30 cells.** `student_final.json` is `core=0, archival=0`
   in every cell of both campaigns — 6 shadow arm-replicates × 2 backends and
   9 stage4 (rep, arm) cells × 2 backends, including the HACT arms whose
   write-rescue fired on other scenarios. A harness/data defect would not
   produce writes on 4 scenarios and none on the 5th through 30 independent
   runs and three different handler stacks.
2. **The prereq data is clean.** All five
   `bfcl_eval/data/memory_prereq_conversation/memory_*.json` files contain
   **zero** U+FFFD replacement characters (the mojibake seen in console dumps
   is a Windows console-encoding artifact, not file corruption).
3. **The failure mode is visible in the transcripts.** Student user turns are
   the longest and least imperative in the suite (mean 811 chars vs notetaker
   145). Example, `memory_kv_prereq_22-student-0` step 0: the model answers a
   three-paragraph reflective monologue with an empathetic monologue of its
   own — *"That's an amazing journey… If you'd like, we could break down some
   of those advanced topics"* — and decodes **zero** tool calls. The model
   does not recognize a chatty narrative as a memory-writing occasion. That is
   an agent-policy failure, i.e. exactly the class of failure a capture
   intervention is allowed to fix.

## 3. Frozen conventions for the RAG program

- **Headline denominator: 155 questions/backend, student INCLUDED.** Student
  is 32% of the denominator and is where the capture mass sits; excluding it
  would hide the effect the program is designed to produce. Student-excluded
  is a mandatory sensitivity column (it is also the analyzer's default —
  passed explicitly as `--include-student` for the headline).
- **Reproduced fair baseline** = the stage4 baseline arms above. Every method
  arm is compared within-campaign against a co-run baseline arm, never against
  these frozen numbers across campaigns.
- **No bug fix is applied to the baseline**: the student failure is
  behavioural, so there is no "corrected baseline run" in the §3.3 sense —
  original and corrected coincide; the correction is to the *interpretation*
  (kv anomaly = dead student chain) and to the guide's vector attribution.
- **Step-budget guard reference** (`step_budget_baseline.json`): baseline
  force_quit = **0/1152 entries**. Pre-registered validity rule for all RAG
  arms, with a floor since the baseline is zero: an arm is invalidated if
  force_quit exceeds `max(3× baseline rate, 2 entries per arm-replicate)`;
  it must then be re-run with the offending policy budget-capped.
