# 🚀 BFCL Setup and Execution Summary: Local Qwen Model via Remote vLLM

This document summarizes the end-to-end troubleshooting and execution process for benchmarking a local Qwen model (`Qwen3-4B-Instruct-2507`) on the Berkeley Function Calling Leaderboard (BFCL). 

The setup involved a **Windows local machine** running the BFCL framework, connecting via an **SSH tunnel** to a **remote BSC Cloud compute node** running the model on vLLM.

---

## 1. Architecture & Port Forwarding

The physical execution was split between two environments:
1. **Remote Cloud (vLLM)**: The model was loaded on the BSC cloud under a SLURM job. Note: vLLM must be spun up with `hermes` tool call parsers and auto tool choice enabled.
2. **Local Windows Machine (BFCL)**: Ran the dataset generation, evaluation, and coordinated API calls.

**The Bridge (SSH Tunnel)**:
To securely route traffic from BFCL to the remote server, we used standard SSH port forwarding:
```bash
ssh -J <username>@alogin1 <username>@<compute_node> -L 8000:localhost:8000
```
*This allowed the local `http://localhost:8000/v1` address to pipe directly into the active vLLM instance on the cloud.*

---

## 2. Model Naming & Registry Conflicts

The most difficult hurdle involved how BFCL internalizes model names. 
We wanted to run the model in **Function Calling (FC)** mode using the BFCL alias: `Qwen/Qwen3-4B-Instruct-2507-FC`.

However, BFCL uses the base `model_name` string (`Qwen/Qwen3-4B-Instruct-2507`) for two conflicting purposes:
1. **Hugging Face Downloader**: It queries Hugging Face to download the tokenizer/config. (If it includes `-FC`, Hugging Face throws a `401/404` error because the repo does not exist).
2. **OpenAI API Client**: It sends the exact same string inside the API payload to the vLLM server.

**The Solution:**
Instead of altering BFCL's internal python logic to separate these strings, we aligned the server:
* **In the Cloud (`serve.sh`)**: Served the model specifically as its Hugging Face base name, omitting the `-FC` payload:
  `--served-model-name Qwen/Qwen3-4B-Instruct-2507`
* **In BFCL (`generate`)**: Called the benchmark using the `-FC` alias:
  `--model Qwen/Qwen3-4B-Instruct-2507-FC`
  
*This natively satisfied Hugging Face for tokenizers, while ensuring vLLM recognized the incoming exact API string.*

---

## 3. Silently Hanging Generators & Default Ports

When running `bfcl generate --skip-server-setup`, the process initially hung indefinitely without sending requests or throwing errors.

**The Cause:**
BFCL's `base_oss_handler.py` assumes local servers default to port `1053`. Since our server was on port `8000`, the handler was stuck in a `while` loop, pinging `1053` once per second forever, waiting for the server to "wake up."

**The Fix:**
Created a `.env` file in the `berkeley-function-call-leaderboard` root directory:
```env
LOCAL_SERVER_PORT=8000
OPENAI_API_KEY="dummy_key"
```

---

## 4. Windows-Specific Decoding Errors (Evaluation Phase)

During the `bfcl evaluate` command, the script spectacularly failed with `UnicodeDecodeError: 'charmap' codec can't decode byte 0x8d`.

**The Cause:**
On Windows, Python's `open()` function defaults to regional system encodings (e.g., CP1254) rather than `utf-8`. BFCL’s test response JSON files contain complex universal characters that these local maps cannot parse.

**The Fix:**
Patched BFCL's internal file loader:
* **File:** `bfcl_eval/utils.py` -> `_load_entries()`
* **Change:** Force UTF-8 explicitly:
  ```python
  def _load_entries(input_path: str) -> None:
      with open(input_path, encoding='utf-8') as f:
          file = f.readlines()
  ```

---

## 5. Final Execution Steps

With all environment and pipeline issues cleared, the actual benchmark commands executed smoothly:

**1. Generation (Querying the model and saving responses)**:
```bash
bfcl generate --model Qwen/Qwen3-4B-Instruct-2507-FC --skip-server-setup
```
*(Note: Seeing "Empty response from the model..." during multi-turn evaluation is completely normal behavior).*

**2. Evaluation (Scoring outputs against ground truth)**:
```bash
bfcl evaluate --model Qwen/Qwen3-4B-Instruct-2507-FC
```

**Results Highlights for Qwen3-4B-Instruct-2507-FC:**
* **Simple Python**: 95.50%
* **Multiple Tools**: 94.00%
* **Parallel Tools**: 93.00%
* **Irrelevance Finding**: 89.58%
* **Memory KV**: 13.55%
* **Web Search**: 0.00%