# Resuming an interrupted gov replicate run (A/B or ablations)

**Yes — an interrupted run is resumable at replicate granularity.** The runner
(`bfcl_eval/scripts/run_gov_replicates.py`) fail-stops on any nonzero exit or
dead tunnel, appends an `error` record to the manifest, and prints the exact
resume point. Restarting is an explicit operator action via `--start-replicate`
(v2 §8.3: a failed replicate is never silently retried).

Written 2026-07-18 during the 48 h window, while run `govrep_20260718T125332Z`
(5× A/B, git head `8f4ea30`) was in flight.

---

## 0. First question: did the *server* restart, or just the connection?

The protocol requires **all replicates of one manifest against the same vLLM
server instance**. A dropped SSH tunnel that reconnects to the *same still-
running* vLLM process is fine — resume. A restarted vLLM / rebooted machine is
a **new instance** — the manifest is void; delete its trees and restart that
whole run from replicate 1.

Fingerprint check (PowerShell):

```powershell
(Invoke-RestMethod http://localhost:8000/v1/models).data[0].created
```

The `created` field is stamped at vLLM startup. For the current 48 h instance
it is **`1784379144`**. Same number ⇒ same instance ⇒ resume is legal.
Different number ⇒ new instance ⇒ restart the manifest from scratch.

## 1. Find where it stopped

The manifest is the log of record (append-only JSONL, one `cmd_start`/`cmd_end`
pair per replicate × arm × phase):

- 5× A/B: `result_gov_replicates/manifest.jsonl`
- Ablations: `result_gov_ablations/manifest.jsonl`

```powershell
Get-Content result_gov_replicates\manifest.jsonl -Tail 5
```

- A trailing `"event": "error"` record names the failed
  `replicate`/`arm`/`phase` (or `stage: tunnel_probe`). **Resume at that
  replicate.**
- If the launcher died with no `error` record (power loss, kill), the last
  `cmd_start` without a matching `cmd_end` is the in-flight replicate.
  **Resume at that replicate.**
- A `"event": "run_end"` record means the run actually finished — nothing to
  resume.

**Never resume at N+1 to "skip past" the in-flight replicate N.** A generate
that was interrupted mid-stream can leave connection-error entries baked into
rep N's result files; only a full re-run of rep N (all arms) is clean. The
runner's `--allow-overwrite` regenerates it entirely, so resuming at N costs
nothing but time (~45 min per arm-run).

## 2. Pre-resume cleanup (one manual step)

The middleware governance log is **append-mode** (risk R7). Re-running
replicate N without deleting its gov-log dirs would mix the aborted attempt's
decisions with the fresh ones. Delete the interrupted replicate's dirs only:

```powershell
# A/B example, resuming at replicate 3:
Remove-Item -Recurse -Force gov_logs\replicates\rep03_* -ErrorAction SilentlyContinue
# Ablations example, resuming at replicate 2:
Remove-Item -Recurse -Force gov_logs\ablations\rep02_* -ErrorAction SilentlyContinue
```

Result/score dirs need **no** cleanup: generate passes `--allow-overwrite` and
evaluate rewrites its score files. Do **not** delete the manifest — resume
appends to it, and the analyzer is built for that (see §5).

## 3. Resume the 5× A/B

Same command as the original launch **plus `--start-replicate N`** (same
gov-env, same default roots, same manifest path). Wrapper pattern (launch
detached via `Start-Process powershell -File <wrapper.ps1>`; PYTHONUTF8 is
required):

```powershell
$env:PYTHONUTF8 = "1"
$env:REMOTE_OPENAI_BASE_URL = "http://localhost:8000/v1"
Set-Location "C:\Users\USER\Desktop\Tezim\BFCL\gorilla\berkeley-function-call-leaderboard"
& C:\Users\USER\miniconda3\envs\BFCL\python.exe bfcl_eval\scripts\run_gov_replicates.py `
    --replicates 5 `
    --start-replicate N `
    --gov-env GOV_SIM_HIGH=0.95 `
    --gov-env GOV_DELTA=0.32 `
    --gov-env GOV_NLI_ENABLED=1 `
    --gov-env GOV_NLI_SHADOW=0 `
    --gov-env GOV_S2_ENABLED=1 `
    --gov-env GOV_S2_SHADOW=0 `
    --gov-env GOV_S2_MARGIN=0.01 `
    *>> tmp\ab5_console.log
```

## 4. Resume the ablations

Identical pattern with the ablation roots and the **exact same `--arms-json`**
(the arms list must be byte-identical across resume — it is re-recorded in the
new `run_start` and the analyzer reads the last one). Note the PS 5.1 quote
escape `.Replace('"','\"')` — without it the inner quotes are stripped:

```powershell
$env:PYTHONUTF8 = "1"
$env:REMOTE_OPENAI_BASE_URL = "http://localhost:8000/v1"
Set-Location "C:\Users\USER\Desktop\Tezim\BFCL\gorilla\berkeley-function-call-leaderboard"
$arms = '[' +
'{"label":"baseline","model":"Qwen/Qwen3-4B-Instruct-2507-FC","gov_env":null},' +
'{"label":"governed_full","model":"Qwen/Qwen3-4B-Instruct-2507-FC-GOV","gov_env":{"GOV_SIM_HIGH":"0.95","GOV_DELTA":"0.32","GOV_NLI_ENABLED":"1","GOV_NLI_SHADOW":"0","GOV_S2_ENABLED":"1","GOV_S2_SHADOW":"0","GOV_S2_MARGIN":"0.01"}},' +
'{"label":"geo_off","model":"Qwen/Qwen3-4B-Instruct-2507-FC-GOV","gov_env":{"GOV_SIM_HIGH":"1.5","GOV_DELTA":"0","GOV_TAU0":"99","GOV_TAU_MIN":"99","GOV_NLI_ENABLED":"1","GOV_NLI_SHADOW":"0","GOV_S2_ENABLED":"1","GOV_S2_SHADOW":"0","GOV_S2_MARGIN":"0.01"}},' +
'{"label":"stage0_only","model":"Qwen/Qwen3-4B-Instruct-2507-FC-GOV","gov_env":{"GOV_SIM_HIGH":"0.95","GOV_DELTA":"0.32","GOV_NLI_ENABLED":"0","GOV_S2_ENABLED":"0"}},' +
'{"label":"s2_off","model":"Qwen/Qwen3-4B-Instruct-2507-FC-GOV","gov_env":{"GOV_SIM_HIGH":"0.95","GOV_DELTA":"0.32","GOV_NLI_ENABLED":"1","GOV_NLI_SHADOW":"0","GOV_S2_ENABLED":"0"}},' +
'{"label":"read_adaptive","model":"Qwen/Qwen3-4B-Instruct-2507-FC-GOV","gov_env":{"GOV_SIM_HIGH":"1.5","GOV_DELTA":"0","GOV_TAU0":"0","GOV_TAU_MIN":"0","GOV_NLI_ENABLED":"0","GOV_S2_ENABLED":"0","GOV_READ_ENABLED":"1","GOV_READ_SHADOW":"0","GOV_READ_MARGIN":"0.0537","GOV_READ_SET_MAX":"4"}},' +
'{"label":"read_set3","model":"Qwen/Qwen3-4B-Instruct-2507-FC-GOV","gov_env":{"GOV_SIM_HIGH":"1.5","GOV_DELTA":"0","GOV_TAU0":"0","GOV_TAU_MIN":"0","GOV_NLI_ENABLED":"0","GOV_S2_ENABLED":"0","GOV_READ_ENABLED":"1","GOV_READ_SHADOW":"0","GOV_READ_MARGIN":"999","GOV_READ_SET_MAX":"3"}},' +
'{"label":"p_only","model":"Qwen/Qwen3-4B-Instruct-2507-FC-GOV","gov_env":{"GOV_SIM_HIGH":"1.5","GOV_DELTA":"0","GOV_TAU0":"0","GOV_TAU_MIN":"0","GOV_NLI_ENABLED":"0","GOV_S2_ENABLED":"0","GOV_P_ENABLED":"1","GOV_P_SHADOW":"0","GOV_P_W1":"0.6","GOV_P_W2":"0.4"}}' +
']'
& C:\Users\USER\miniconda3\envs\BFCL\python.exe bfcl_eval\scripts\run_gov_replicates.py `
    --replicates 3 `
    --start-replicate N `
    --result-root result_gov_ablations `
    --score-root score_gov_ablations `
    --gov-log-root gov_logs/ablations `
    --manifest result_gov_ablations/manifest.jsonl `
    --arms-json $arms.Replace('"', '\"') `
    *>> tmp\ablations_console.log
```

## 5. Why the appended manifest stays analyzable

`analyze_gov_replicates.py` is resume-aware by construction:

- `gather_records` takes the **last** `run_start` record for config, so the
  resume's header supersedes the aborted attempt's.
- `completed_replicates` unions every `cmd_end`/`evaluate`/`exit_code == 0`
  across the whole file — a replicate counts once it is complete in **every**
  arm, regardless of which attempt completed it. Duplicate records from the
  aborted attempt are inert.
- On-disk data for a re-run replicate is the fresh attempt's
  (`--allow-overwrite` + evaluate rewrite), so records and files agree.
- G15 still holds: any (rep, arm, backend) with no score file makes the
  analyzer raise. Verify after resume that the final `cmd_end` records carry
  `score_files: [... exists: true]` and that a `run_end` record is present.

## 6. Post-resume checklist

1. `run_end` record present in the manifest.
2. Every replicate × arm has an `evaluate` `cmd_end` with `exit_code: 0` and
   `exists: true` on both score files (memory_kv + memory_vector).
3. Server fingerprint (`created`) unchanged from the run's first `run_start`
   (`served_models` and `vllm_version` are also pinned there: vLLM 0.9.1,
   `Qwen/Qwen3-4B-Instruct-2507`).
4. Gov-log dirs `repNN_<arm>/governance_log.jsonl` exist for governed arms and
   contain exactly one attempt's decisions (step 2 above was done).

---

## Addendum 2026-07-22: the `created` fingerprint is server-dependent

On the 2026-07-21/22 tunnel machine (vLLM reported as 0.9.1 via /version, same
as the July campaign server), `/v1/models` `data[0].created` is stamped
**per-request** (three probes 3 s apart returned values 3 s apart), not at
vLLM startup. The §0 fingerprint check is therefore UNUSABLE on such servers:
a moving `created` does NOT prove a restart, and two differing values across a
tunnel drop prove nothing either way.

Operational rule when `created` is per-request (detect with two probes a few
seconds apart):
1. Instance identity cannot be verified via the API. Treat a manifest as
   same-instance only on direct operator confirmation that the vLLM process
   was not restarted; otherwise assume restart and void the manifest (the
   conservative default used for the 2026-07-22 me_harvest restart).
2. Record in the run report that instance continuity is asserted, not
   verified, for any resume performed under this rule.
