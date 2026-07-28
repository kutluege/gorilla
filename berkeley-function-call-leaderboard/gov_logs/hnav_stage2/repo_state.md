# H-Nav Stage 2 — starting repository state

Recorded: 2026-07-28 (start of the action-side H_act research program).

## Git

- Branch: `claude/nihai-plan-v2-cascade-thresholds-k34u67`
- HEAD at program start: `cb8f76b700f7b2ce4fb366f161091309621759ba`
  (`[BFCL][GOV] Plan v2 Phase 6 COMPLETE: rep3 resume + full matrix analysis — GATE G2 FAIL, geometry_only becomes Phase 7 headline arm`)
- Uncommitted at start: modified `bfcl_eval/scripts/calibrate_margin_entropy.py` (4 additive Stage 1 edits);
  untracked = the entire H-Nav Stage 1 toolchain (6 scripts + `diff_signals.py` + 3 test suites),
  Stage 1 artifacts under `gov_logs/` (hnav_* reports, `outcomes_hnav*`, `features_diff*`),
  `HNAV_STAGE1_IMPLEMENTATION_SUMMARY.md`, two PRI idea docs, two tmp smoke dirs (left untracked),
  and a stray nested `berkeley-function-call-leaderboard/gorilla/` clone (added to `.gitignore`, never committed).
- Stage 1 preservation: commits 1 (code+tests) and 2 (artifacts) immediately after this snapshot;
  see `git log` from `cb8f76b` forward.

## Environment

- OS: Windows 11 Home 10.0.26200; no local GPU (binding constraint — all inference is remote).
- Python: 3.10.20 at `C:\Users\USER\miniconda3\envs\BFCL\python.exe` (conda env `BFCL`).
- Full `pip freeze`: `gov_logs/hnav_stage2/pip_freeze.txt`. Key packages:
  numpy 1.26.4, openai 2.28.0, rank-bm25 0.2.2, scipy 1.15.3,
  sentence-transformers 5.3.0, torch 2.10.0 (CPU), transformers 5.3.0.
- `PYTHONUTF8=1` required for all runs (RESUME_PROTOCOL).

## Model server (probed 2026-07-28)

- `GET http://localhost:8000/v1/models` (2-hop SSH tunnel to BSC cluster) →
  `Qwen/Qwen3-4B-Instruct-2507`, `max_model_len 32000`, `owned_by vllm`,
  **`allow_logprobs: true`**, `allow_sampling: true`.
- Model root on the serve host: `/gpfs/scratch/etur52/models/Qwen3-4B-Instruct-2507`
  — NOTE: different scratch path from the July campaigns (`/gpfs/scratch/ehpc540/...`),
  i.e. a different serve deployment of the same weights. Per the RESUME_PROTOCOL addendum
  (commit 76672a4) instance identity is unverifiable via API (`created` is per-request on
  some servers); instance continuity across campaigns is asserted, not verified. All new
  campaigns record their own manifest and are analyzed with in-campaign baselines only.
- Benchmark data version: `bfcl_eval/data/BFCL_v4_memory.json` + `possible_answer/BFCL_v4_memory.json`
  (155 rows), unchanged since Stage 1 (grader/benchmark files are never modified by this program).

## Governance configuration context

- Plan v2 outcome: G1 FAIL (entropy terms dropped), G2 FAIL (`gm_v1` not non-inferior);
  pre-registered headline arm for any governed comparison is `geometry_only`
  (`GOV_POLICY=geometry_only`, `GOV_DELTA=0.32`, `GOV_SIM_HIGH=0.95`).
- Frozen calibration: `gov_logs/margin_entropy_calibration.json` (sha `268ebac77c26f7a1`).
- H-Nav Stage 1 verdict: PIVOT to action-side H_act (see `HNAV_STAGE1_IMPLEMENTATION_SUMMARY.md`).
