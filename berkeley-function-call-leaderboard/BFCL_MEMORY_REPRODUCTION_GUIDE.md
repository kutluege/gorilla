# BFCL Memory Reproduction Guide

This guide explains how to regenerate and evaluate BFCL memory results for
`Qwen/Qwen3-4B-Instruct-2507-FC` in the closest practical way to the official
BFCL leaderboard harness.

The target is the BFCL website row, not the current local artifacts. The
leaderboard metadata says the public row was last updated on `2026-04-12` and
is reproducible from commit `f7cf735` / `bfcl-eval==2025.12.17`. The published
memory values for the screenshot target are:

| Metric | Official Value |
| --- | ---: |
| Memory Summary | `17.63` |
| KV | `16.13` |
| Vector | `12.26` |
| Recursive Sum | `24.52` |

## Why Regenerate Cleanly

Do not use the existing local `result/` and `score/` trees for official
comparison. The local audit found that the `student` memory prereq phase emitted
zero memory calls in all three backends:

| Backend | Student Memory Calls | Student Final Snapshot |
| --- | ---: | --- |
| `memory_kv` | `0` | empty core and archival memory |
| `memory_vector` | `0` | empty core and archival memory |
| `memory_rec_sum` | `0` | empty recursive summary |

That means the current local memory artifacts are affected by generation or
serving behavior, not only by score aggregation. The audit also found a stale
snapshot risk: memory snapshots live under
`result/<model>/agentic/memory/<backend>/memory_snapshot`, while local cleanup
can miss that path. A clean output root avoids mixing old result JSONL files
with old memory snapshots.

In the BSC remote-serving setup used for the local investigation, the intended
code change was connection-only: BFCL on Windows connects through SSH tunnels to
a vLLM/OpenAI-compatible server running on BSC. That kind of change should not
alter BFCL memory semantics by itself. If `student_final.json` is empty, and
especially if `healthcare_final.json` is also empty in the same backend/run,
treat the failure as a generation, endpoint, model-id, tokenizer/template, or
tool-call decoding problem before treating it as a score aggregation problem.

## Recommended Layout

Use a separate official reproduction checkout instead of the dirty local working
tree. This is not because the repository must be downloaded twice; it is because
official comparison needs a clean checkout at the exact upstream commit, with
fresh `result/` and `score/` directories.

```powershell
$ReproBase = "C:\Users\USER\Desktop\Tezim\BFCL\bfcl_official_repro"
$RepoRoot = "$ReproBase\gorilla\berkeley-function-call-leaderboard"
```

All commands below write only inside `$ReproBase`. They do not overwrite the
existing investigation outputs under:

```powershell
C:\Users\USER\Desktop\Tezim\BFCL\gorilla\berkeley-function-call-leaderboard\result
C:\Users\USER\Desktop\Tezim\BFCL\gorilla\berkeley-function-call-leaderboard\score
```

If you already have `gorilla` locally, prefer `git worktree` from the existing
clone. A worktree reuses the local Git object database and checks out another
copy of the files at `f7cf735`, so it avoids a second network clone while still
keeping the official reproduction isolated from local edits.

## Environment Setup

Create a fresh Python environment. BFCL recommends Python `3.10`.

```powershell
conda create -n bfcl-official python=3.10
conda activate bfcl-official
python -m pip install --upgrade pip
```

For the existing local benchmark environment in this workspace, activate:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Create and pin the official checkpoint. If you already have the local
`gorilla` repository, use the worktree option:

```powershell
$ExistingGorilla = "C:\Users\USER\Desktop\Tezim\BFCL\gorilla"
New-Item -ItemType Directory -Force -Path $ReproBase
Set-Location $ExistingGorilla
git cat-file -e f7cf735^{commit}
# If the previous command fails, fetch the official commit:
# git fetch origin f7cf735
git worktree add --detach "$ReproBase\gorilla" f7cf735
Set-Location $RepoRoot
git status --short
```

If you do not have the repository locally, clone it instead:

```powershell
New-Item -ItemType Directory -Force -Path $ReproBase
Set-Location $ReproBase
git clone https://github.com/ShishirPatil/gorilla.git
Set-Location $RepoRoot
git checkout f7cf735
git status --short
```

Do not run `git checkout f7cf735` in your existing investigation checkout unless
you have first committed, stashed, or otherwise saved local changes. Your current
checkout may contain audit scripts, generated reports, local handler edits, and
old result trees that are useful for investigation but unsuitable for strict
leaderboard comparison.

`git status --short` should be empty in the reproduction checkout before
generating official-comparison results.

Install BFCL from the pinned checkout. For the BSC tunnel setup, where vLLM is
already running remotely and BFCL uses `--skip-server-setup`, install the base
package only:

```powershell
python -m pip install -e .
python -m pip install soundfile
```

The `soundfile` package is needed because BFCL imports Qwen handlers through
`qwen-agent`, and `qwen-agent` imports it at startup.

Only install the vLLM extra if BFCL will launch/manage a local vLLM server on
the same machine:

```powershell
python -m pip install -e .[oss_eval_vllm]
python -m pip install anthropic
```

If you use SGLang instead of vLLM:

```powershell
python -m pip install -e .[oss_eval_sglang]
python -m pip install anthropic
```

As an alternative to a Git checkout, install the package version named by the
leaderboard metadata:

```powershell
python -m pip install "bfcl-eval==2025.12.17"
python -m pip install anthropic
```

Prefer the Git checkout when you want to inspect prompts, function docs,
snapshots, or local audit scripts.

If the `bfcl` console command is not available after installation, use
`python -m bfcl_eval` in its place. For example, `bfcl generate ...` becomes
`python -m bfcl_eval generate ...`.

On Windows, if `conda run -n bfcl-official python -m bfcl_eval --help` fails
while printing help because of console encoding, call the environment Python
directly:

```powershell
C:\Users\USER\miniconda3\envs\bfcl-official\python.exe -m bfcl_eval --help
```

## Configure The Output Root

Point BFCL at the clean reproduction checkout:

```powershell
$env:BFCL_PROJECT_ROOT = $RepoRoot
Set-Location $RepoRoot
```

Create a `.env` file:

```powershell
Copy-Item .\bfcl_eval\.env.example .\.env -Force
```

Edit `.env` for your model server. If BFCL reaches the server through your local
Windows tunnel on `localhost:8000`, configure it as a local endpoint:

```text
LOCAL_SERVER_ENDPOINT=localhost
LOCAL_SERVER_PORT=8000
```

For a direct remote OpenAI-compatible URL, use:

```text
REMOTE_OPENAI_BASE_URL=https://your-server.example/v1
REMOTE_OPENAI_API_KEY=your-api-key
REMOTE_OPENAI_TOKENIZER_PATH=C:\path\to\Qwen3-4B-Instruct-2507-FC
```

Keep the serving configuration fixed and record it: model path, backend,
temperature, max tokens, quantization, tokenizer path, server URL, and whether
the server applies any custom chat template.

## BSC Remote Serving Setup

This is the serving setup used in the local investigation. The benchmark runs
locally on Windows, while the model is served remotely on BSC and exposed
through SSH tunnels.

Record the BSC paths with each run:

```bash
# Serving work directory on BSC
/gpfs/scratch/ehpc540/users/ege/tez

# Target model path on BSC
MODEL_PATH="/gpfs/scratch/ehpc540/models/Qwen3-4B-Instruct-2507"
```

Confirm whether the served checkpoint is exactly the leaderboard target
`Qwen/Qwen3-4B-Instruct-2507-FC`. If the BSC path lacks the `-FC` suffix, record
that explicitly because it can change tool-call behavior and make the run useful
for debugging but not a strict leaderboard reproduction.

On BSC, create the socket directory before starting or tunneling to the server:

```bash
mkdir -p ~/socketdir
chmod 700 ~/socketdir
cd ~/socketdir
rm -f vllm.sock
```

Keep both SSH tunnel terminals open.

Terminal 1, from the BSC side where `as03r1b14` is reachable:

```bash
ssh -L ./vllm.sock:localhost:8000 as03r1b14
```

Terminal 2, from the local machine:

```bash
ssh -L 8000:/home/boga/boga771710/socketdir/vllm.sock boga771710@alogin1.bsc.es
```

Then verify the local tunnel before running BFCL:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/gpfs/scratch/ehpc540/models/Qwen3-4B-Instruct-2507",
    "messages": [
      {"role": "user", "content": "What is the capital of Turkey?"}
    ]
  }'
```

The `model` field in this curl request must match the model name accepted by
the running vLLM server. If the server was launched with a different served
model name, use that exact name in BFCL and in the smoke test.

BSC job controls:

```bash
# Show running jobs
squeue -u boga771710

# Stop one job
scancel <jobid>

# Stop all jobs for the user
scancel -u boga771710
```

For BFCL runs against this tunnel, use `--skip-server-setup`; BFCL should not
try to launch the model locally.

## Memory Vector Encoder Preflight

The `memory_vector` backend imports
`sentence-transformers/all-MiniLM-L6-v2` locally. This is separate from the
Qwen generation server. On the Windows/BSC tunnel setup, let BFCL load this
encoder from the local Hugging Face cache instead of probing the network during
generation:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
C:\Users\USER\miniconda3\envs\bfcl-official\python.exe -c "from sentence_transformers import SentenceTransformer; m=SentenceTransformer('all-MiniLM-L6-v2', device='cpu'); print(m.get_sentence_embedding_dimension())"
```

The preflight should print `384`. If it does not, download/cache
`sentence-transformers/all-MiniLM-L6-v2` before running `memory_vector`.

BFCL's official generation temperature is `0.001`; keep `--temperature 0.001`
for literal leaderboard reproduction. vLLM may log that it clamps this to
`0.01`; that warning is expected and does not mean the BFCL command is using
the wrong temperature.

## Full Memory Reproduction

Use all three memory categories for leaderboard comparison. Do not use
`--partial-eval` for final scores.

If BFCL should start and manage the local backend server:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
bfcl generate `
  --model Qwen/Qwen3-4B-Instruct-2507-FC `
  --test-category memory_kv,memory_vector,memory_rec_sum `
  --backend vllm `
  --num-gpus 1 `
  --gpu-memory-utilization 0.9 `
  --temperature 0.001 `
  --allow-overwrite
```

If you already have a vLLM/SGLang/OpenAI-compatible server running:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
bfcl generate `
  --model Qwen/Qwen3-4B-Instruct-2507-FC `
  --test-category memory_kv,memory_vector,memory_rec_sum `
  --skip-server-setup `
  --temperature 0.001 `
  --allow-overwrite
```

For the BSC tunnel, use the same command shape. The important parts are
`--skip-server-setup`, `.env` pointing to `localhost:8000`, and a served model
name that matches the remote vLLM server.

Evaluate the generated responses:

```powershell
bfcl evaluate `
  --model Qwen/Qwen3-4B-Instruct-2507-FC `
  --test-category memory_kv,memory_vector,memory_rec_sum
```

Show the score table:

```powershell
bfcl scores
```

Expected output locations:

```text
$RepoRoot\result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\kv
$RepoRoot\result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\vector
$RepoRoot\result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\rec_sum
$RepoRoot\score\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\kv
$RepoRoot\score\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\vector
$RepoRoot\score\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\rec_sum
```

## Debug Reproduction

Use this mode if the full run still disagrees with the official row or if
`student_final.json` is empty again. It is slower but preserves prompt and
handler logs.

Start with `memory_vector` because it exposes both core and archival memory and
is easy to inspect:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
bfcl generate `
  --model Qwen/Qwen3-4B-Instruct-2507-FC `
  --test-category memory_vector `
  --skip-server-setup `
  --temperature 0.001 `
  --num-threads 1 `
  --include-input-log `
  --allow-overwrite

bfcl evaluate `
  --model Qwen/Qwen3-4B-Instruct-2507-FC `
  --test-category memory_vector
```

Then repeat for `memory_kv` and `memory_rec_sum`:

```powershell
bfcl generate `
  --model Qwen/Qwen3-4B-Instruct-2507-FC `
  --test-category memory_kv,memory_rec_sum `
  --skip-server-setup `
  --temperature 0.001 `
  --num-threads 1 `
  --include-input-log `
  --allow-overwrite

bfcl evaluate `
  --model Qwen/Qwen3-4B-Instruct-2507-FC `
  --test-category memory_kv,memory_rec_sum
```

For a focused diagnosis, inspect both `student` and a known-working control such
as `customer` in the prereq result files:

```text
result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\vector\BFCL_v4_memory_vector_prereq_result.json
result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\kv\BFCL_v4_memory_kv_prereq_result.json
result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\rec_sum\BFCL_v4_memory_rec_sum_prereq_result.json
```

The important log fields are described in `LOG_GUIDE.md`:

- `inference_input`: the final prompt/input sent to the model, available only
  with `--include-input-log`.
- `handler_log`: decode success, empty response, decode failure, and force quit
  events.
- `tool`: executed function output.
- `state_info`: backend state after each turn.

## Post-Run Validation

First verify that the official checkpoint is still clean:

```powershell
git rev-parse HEAD
git status --short
```

`git rev-parse HEAD` should print `f7cf735...`, and `git status --short` should
not show code changes.

Check the `student` final snapshots:

```powershell
Get-Content .\result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\kv\memory_snapshot\student_final.json
Get-Content .\result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\vector\memory_snapshot\student_final.json
Get-Content .\result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\rec_sum\memory_snapshot\student_final.json
```

Also check `healthcare` in the same run because it is a useful cross-scenario
signal:

```powershell
Get-Content .\result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\kv\memory_snapshot\healthcare_final.json
Get-Content .\result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\vector\memory_snapshot\healthcare_final.json
Get-Content .\result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\rec_sum\memory_snapshot\healthcare_final.json
```

For a healthy reproduction run, `student` should not have zero memory calls
across every backend. If all three `student_final.json` files are empty again,
the issue is still in generation, model serving, prompt formatting, tool-call
decoding, or server/template routing.

If `healthcare_final.json` is also empty in the same backend, the signal is
stronger: the remote endpoint is probably not producing executable memory writes
for multiple scenarios. Re-check the BSC tunnel, served model name, exact model
checkpoint, tokenizer path, chat template, and raw `<tool_call>` output before
debugging the scorer.

Run the investigation audit script if it exists in the checkout. This script was
created during the local investigation and may not exist in a fresh upstream
`f7cf735` checkout. If it is missing, copy
`bfcl_eval\scripts\audit_memory_pipeline.py` from the investigation checkout
into the same path in the reproduction checkout, or run the script from the
investigation checkout while passing `--root $RepoRoot`.

```powershell
python .\bfcl_eval\scripts\audit_memory_pipeline.py `
  --root $RepoRoot `
  --model-dir Qwen_Qwen3-4B-Instruct-2507-FC `
  --output memory_pipeline_audit_report_repro.md
```

The key checks in the audit report are:

- `student` raw tool tags are nonzero.
- `student` decoded memory calls are nonzero.
- `student_final.json` is not empty for `kv` and `vector`.
- recursive-summary `student_final.json` has nonzero memory text.
- `customer` remains populated as a control scenario.

Finally, compare score headers and CSV values:

```powershell
Get-Content .\score\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\kv\BFCL_v4_memory_kv_score.json -TotalCount 1
Get-Content .\score\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\vector\BFCL_v4_memory_vector_score.json -TotalCount 1
Get-Content .\score\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\rec_sum\BFCL_v4_memory_rec_sum_score.json -TotalCount 1
Get-Content .\score\data_agentic.csv
```

Memory Summary is the unweighted average of `KV`, `Vector`, and `Recursive Sum`
in the agentic memory section. Compare the reproduced row with the official
target:

| Metric | Official Target |
| --- | ---: |
| Memory Summary | `17.63` |
| KV | `16.13` |
| Vector | `12.26` |
| Recursive Sum | `24.52` |

## Official Response Archive

The BFCL leaderboard links official model responses when available. If the
archive is available for `Qwen/Qwen3-4B-Instruct-2507-FC`, download it into a
separate comparison folder and compare ID by ID against your regenerated
artifacts. This is the strongest way to distinguish harness mismatch from
model-serving nondeterminism.

Do not copy official responses into your reproduction `result/` tree unless the
goal is to evaluate the official archive directly.

## Troubleshooting

### `ModuleNotFoundError: No module named 'anthropic'`

Install the missing dependency in the active environment:

```powershell
python -m pip install anthropic
```

Then retry `bfcl generate` or `bfcl evaluate`.

### Endpoint Configuration Problems

If BFCL should use an existing server, pass `--skip-server-setup` and set the
endpoint in `.env`. For a local server, use `LOCAL_SERVER_ENDPOINT` and
`LOCAL_SERVER_PORT`. For a remote OpenAI-compatible server, use
`REMOTE_OPENAI_BASE_URL` and `REMOTE_OPENAI_API_KEY`.

For the BSC tunnel described above, the endpoint is remote in reality but local
from BFCL's point of view:

```text
LOCAL_SERVER_ENDPOINT=localhost
LOCAL_SERVER_PORT=8000
```

Before running BFCL, verify that `curl http://localhost:8000/v1/chat/completions`
returns a normal response and that both SSH tunnel terminals are still open.

If the model outputs chatty prose instead of tool calls, confirm that:

- The endpoint is serving `Qwen/Qwen3-4B-Instruct-2507-FC`, not another Qwen
  checkpoint.
- The tokenizer path matches the served model.
- No external chat template is overriding BFCL's handler formatting.
- The server is not applying a forced system prompt.
- Temperature and decoding options match the BFCL command.

### Empty `student_final.json` Or `healthcare_final.json`

This is the specific failure seen in the local artifacts. Inspect the first
student prereq rows and their `handler_log` entries. If there are no raw
`<tool_call>` tags and no decoded memory calls, scoring is not the root cause;
the model did not produce executable memory writes during prereq generation.

Compare `student` and `healthcare` against a known-populated control such as
`customer`. If `customer` writes memory but `student` does not, inspect the
`inference_input` for prompt length, function docs, and tool availability. If
`student` and `healthcare` are both empty, or if no scenario writes memory,
focus on endpoint/model handler/tool-call template mismatch.

For the BSC setup, specifically verify:

- The tunnel points to the active vLLM job, not an old or stopped job.
- The curl smoke test uses the same served model name BFCL will use.
- The served checkpoint is the intended Qwen function-calling checkpoint.
- The server is not applying a chat template that strips or changes BFCL's tool
  instructions.
- The raw model response contains BFCL-decodable tool calls, not only prose.

### Stale Memory Snapshots

Use a clean checkout and a clean `result/` tree for official reproduction.
Avoid repeatedly regenerating into the existing local investigation directory.
If you must rerun inside the same clean reproduction checkout, delete the whole
model output directory first:

```powershell
Remove-Item -Recurse -Force .\result\Qwen_Qwen3-4B-Instruct-2507-FC
Remove-Item -Recurse -Force .\score\Qwen_Qwen3-4B-Instruct-2507-FC
```

Only run those commands inside the dedicated `$RepoRoot`, not inside the
original investigation checkout.

### Partial Eval Mismatch

Do not use `--partial-eval` for official comparison. Partial evaluation changes
the denominator by skipping missing IDs, so the result can differ from the
leaderboard even when the generated subset looks correct.

Use `--partial-eval` only for local debugging, and label those numbers as
debug-only.

### Dirty Checkout Mismatch

The previous local audit found memory-relevant diffs from the official target in
generation, local inference handling, and utilities. For official comparison,
run from a clean `f7cf735` checkout or from `bfcl-eval==2025.12.17`.

Before trusting a score, record:

```powershell
git rev-parse HEAD
git describe --always --dirty --tags
git status --short
bfcl version
```

Any dirty code state means the run is useful for investigation but not a strict
leaderboard reproduction.

## Acceptance Checklist

- Clean checkout at `f7cf735` or installed `bfcl-eval==2025.12.17`.
- Fresh environment with all imports available, including `anthropic`.
- Outputs written to the dedicated reproduction checkout, not the existing
  investigation `result/` or `score/` directories.
- Full memory categories generated: `memory_kv`, `memory_vector`,
  `memory_rec_sum`.
- Full memory categories evaluated without `--partial-eval`.
- `student` prereq rows show raw and decoded memory calls.
- `student_final.json` is nonempty for all memory backends.
- `healthcare_final.json` is checked as a cross-scenario signal, especially for
  BSC tunnel runs.
- `data_agentic.csv` memory values are compared with the official target row.
- Serving settings are recorded so the run can be repeated.
