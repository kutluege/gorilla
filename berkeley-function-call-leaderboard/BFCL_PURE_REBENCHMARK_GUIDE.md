# Pure BFCL Re-Benchmark Guide

This guide explains how to re-benchmark a model in the closest practical form
to the official BFCL leaderboard run. The key rule is simple: use the official
BFCL code and official BFCL generation settings first, then debug serving
differences only after the clean harness is verified.

The local investigation tree is useful for reports and audits, but strict
comparison should be run from the clean reproduction checkout:

```text
C:\Users\USER\Desktop\Tezim\BFCL\bfcl_official_repro\gorilla\berkeley-function-call-leaderboard
```

For the Qwen memory investigation in this workspace, the official target is:

```text
BFCL commit: f7cf7359b7ac615a0b294831c5ba2bc95ee4a000
Package equivalent: bfcl-eval==2025.12.17
Model id in BFCL: Qwen/Qwen3-4B-Instruct-2507-FC
Official temperature: 0.001
```

Do not patch prompts, memory tools, model handlers, output parsers, or scoring
logic when the goal is official BFCL similarity.

## Repository Choice

Use the clean `bfcl_official_repro` worktree as the source of truth because it
is pinned to the official leaderboard checkpoint. The existing checkout below is
the investigation tree and may contain local edits, audit reports, temporary
logs, and old outputs:

```text
C:\Users\USER\Desktop\Tezim\BFCL\gorilla\berkeley-function-call-leaderboard
```

Keep those investigation artifacts, but do not use them for strict leaderboard
comparison. A dirty checkout can change generation behavior before you even get
to model-serving differences.

Create or recreate the clean worktree from the existing local clone:

```powershell
$ReproBase = "C:\Users\USER\Desktop\Tezim\BFCL\bfcl_official_repro"
$ExistingGorilla = "C:\Users\USER\Desktop\Tezim\BFCL\gorilla"

New-Item -ItemType Directory -Force -Path $ReproBase
Set-Location $ExistingGorilla
git cat-file -e f7cf735^{commit}
git worktree add --detach "$ReproBase\gorilla" f7cf735
```

Then confirm the clean checkout:

```powershell
$RepoRoot = "$ReproBase\gorilla\berkeley-function-call-leaderboard"
Set-Location $RepoRoot
git rev-parse HEAD
git status --short
```

Expected `HEAD`:

```text
f7cf7359b7ac615a0b294831c5ba2bc95ee4a000
```

For a pure run, `git status --short` should be empty except for deliberate new
score directories created after evaluation, such as `score_fresh/`.

## Pure BFCL Setup

Install BFCL from the clean checkout:

```powershell
Set-Location C:\Users\USER\Desktop\Tezim\BFCL\bfcl_official_repro\gorilla\berkeley-function-call-leaderboard
C:\Users\USER\miniconda3\envs\bfcl-official\python.exe -m pip install -e .
```

Create `.env` from BFCL's example and point it at the already-running local
OpenAI-compatible endpoint:

```powershell
Copy-Item .\bfcl_eval\.env.example .\.env -Force
```

Set these values in `.env`:

```text
LOCAL_SERVER_ENDPOINT=localhost
LOCAL_SERVER_PORT=8000
```

Preflight the endpoint:

```powershell
Invoke-RestMethod http://localhost:8000/v1/models
```

For this Qwen run, the endpoint may report the served model as:

```text
Qwen/Qwen3-4B-Instruct-2507
```

That is acceptable if BFCL is invoked with:

```text
Qwen/Qwen3-4B-Instruct-2507-FC
```

The `-FC` suffix is BFCL's function-calling handler/mode. It does not always
mean the vLLM server will expose a separate model id with the same suffix.

On Windows, set UTF-8 and Hugging Face offline mode before generation and
evaluation:

```powershell
$env:BFCL_PROJECT_ROOT=(Get-Location).Path
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUTF8="1"
```

Why these matter:

- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` keep
  `SentenceTransformer("all-MiniLM-L6-v2")` from making a Hugging Face network
  request during `memory_vector`.
- `PYTHONIOENCODING=utf-8` prevents console-printing failures on Windows.
- `PYTHONUTF8=1` makes Python default text file reads/writes use UTF-8, avoiding
  `UnicodeDecodeError` when BFCL reads result or score JSON files.

You can verify the vector encoder is available locally:

```powershell
C:\Users\USER\miniconda3\envs\bfcl-official\python.exe -c "from sentence_transformers import SentenceTransformer; m=SentenceTransformer('all-MiniLM-L6-v2', device='cpu'); print(m.get_sentence_embedding_dimension())"
```

Expected output:

```text
384
```

## Generation And Evaluation

Run memory generation with official BFCL temperature. The single-threaded
setting and input logs are intentionally debug-friendly; after the pipeline is
verified, a broader run can increase threads.

```powershell
Set-Location C:\Users\USER\Desktop\Tezim\BFCL\bfcl_official_repro\gorilla\berkeley-function-call-leaderboard

$env:BFCL_PROJECT_ROOT=(Get-Location).Path
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUTF8="1"

C:\Users\USER\miniconda3\envs\bfcl-official\python.exe -m bfcl_eval generate `
  --model Qwen/Qwen3-4B-Instruct-2507-FC `
  --test-category memory_kv,memory_vector,memory_rec_sum `
  --skip-server-setup `
  --temperature 0.001 `
  --num-threads 1 `
  --include-input-log `
  --allow-overwrite
```

Evaluate into a fresh score directory:

```powershell
C:\Users\USER\miniconda3\envs\bfcl-official\python.exe -m bfcl_eval evaluate `
  --model Qwen/Qwen3-4B-Instruct-2507-FC `
  --test-category memory_kv,memory_vector,memory_rec_sum `
  --score-dir score_fresh
```

Using `--score-dir score_fresh` avoids a common Windows problem where
`score\data_agentic.csv` is locked by Excel or another viewer. If you want to
write to the default `score/` directory, close Excel first.

Keep `--temperature 0.001`. In the observed vLLM logs, vLLM warns that
temperatures below `0.01` are clamped to `0.01`. That warning is expected and
should be recorded as part of the serving environment, not fixed by changing the
BFCL command.

Read the aggregated memory scores:

```powershell
Get-Content .\score_fresh\data_agentic.csv
```

## Validation And Debugging

For memory categories, generation has two phases:

1. Prereq conversations build memory state.
2. Answer rows query that memory state and are scored.

If the score differs from the official leaderboard, first inspect whether the
prereq phase actually produced memory tool calls and snapshots. For vector
memory:

```powershell
Get-Content .\result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\vector\memory_snapshot\student_final.json

Select-String -Path .\result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\vector\BFCL_v4_memory_vector_prereq_result.json `
  -Pattern "memory_vector_prereq_22-student-0","<tool_call>","core_memory_add","archival_memory_add"
```

Repeat the same idea for:

```text
memory_kv
memory_rec_sum
```

Important lesson from this workspace: an empty `student_final.json` was not a
snapshot-writer failure. The current served Qwen generated zero memory tool
calls for all `student` prereq rows across `memory_kv`, `memory_vector`, and
`memory_rec_sum`. That points upstream to generation or serving behavior.

The clean local run produced:

| Metric | Clean local run |
| --- | ---: |
| Memory Summary | `19.57%` |
| Memory KV | `11.61%` |
| Memory Vector | `12.90%` |
| Memory Recursive Summarization | `34.19%` |

The official screenshot target showed:

| Metric | Official screenshot |
| --- | ---: |
| Memory Summary | `17.63%` |
| Memory KV | `16.13%` |
| Memory Vector | `12.26%` |
| Memory Recursive Summarization | `24.52%` |

Because evaluation completed and the CSV matched the score JSON files, this
remaining difference is not a CSV aggregation issue. It is a generation or
serving mismatch.

When investigating such mismatches, record:

- BFCL commit or package version.
- Exact model id passed to BFCL.
- Exact model id returned by `/v1/models`.
- vLLM or SGLang version.
- Server launch command and model path.
- Tokenizer path and chat template behavior.
- Temperature, seed, max tokens, stop tokens, and any server-side defaults.
- Whether vLLM clamped temperature from `0.001` to `0.01`.
- Whether memory prereq rows contain raw `<tool_call>` tags.
- Whether final memory snapshots are populated before answer rows run.

Do not adjust prompts or force memory calls to match the official score. That
can be useful for a local experiment, but it is no longer pure BFCL.

## Common Failure Modes

### Hugging Face Client Closed During `memory_vector`

Symptom:

```text
RuntimeError: Cannot send a request, as the client has been closed.
```

Cause observed here: `SentenceTransformer("all-MiniLM-L6-v2")` attempted a
network metadata request while BFCL was running memory vector evaluation.

Fix for pure local reproduction when the model is already cached:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
```

### Windows `UnicodeDecodeError`

Symptom:

```text
UnicodeDecodeError: 'charmap' codec can't decode byte ...
```

Fix:

```powershell
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUTF8="1"
```

### `PermissionError` Writing `data_agentic.csv`

Symptom:

```text
PermissionError: [Errno 13] Permission denied: ...\score\data_agentic.csv
```

Fix: close Excel or write to a fresh score directory:

```powershell
--score-dir score_fresh
```

### Stale Or Mixed Outputs

Use isolated output directories or a clean official checkout. Memory snapshots
live under paths like:

```text
result\<model>\agentic\memory\<backend>\memory_snapshot
```

Do not compare official scores against a result tree that mixes old result JSONL
files with old memory snapshots.

## Pure Re-Benchmark Checklist

- Clean checkout is pinned to `f7cf7359b7ac615a0b294831c5ba2bc95ee4a000`.
- BFCL prompts, handlers, memory backends, and scoring are unmodified.
- `.env` points to the intended OpenAI-compatible endpoint.
- `/v1/models` returns the expected served checkpoint.
- `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `PYTHONIOENCODING`, and
  `PYTHONUTF8` are set.
- Generation uses `--temperature 0.001`.
- Evaluation writes to an unlocked score directory.
- `data_agentic.csv` is read from the same score directory used by evaluation.
- Memory discrepancies are debugged from prereq tool calls and memory snapshots,
  not from answer scores alone.
