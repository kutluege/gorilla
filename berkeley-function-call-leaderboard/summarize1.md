# BFCL Benchmark Flow Summary

This document summarizes the end-to-end execution flow of the Berkeley Function Call Leaderboard (BFCL) benchmark, focusing on how test cases are generated and evaluated, specifically for models like Google Gemini and OpenAI.

## 1. High-Level Architecture

The benchmark consists of two completely decoupled phases:

1.  **Generation Phase (`bfcl generate`)**:
    *   Reads test cases from JSON files.
    *   Queries the LLM (either via FC API or Prompting).
    *   Saves the raw model output to a results file on disk.
    *   *No evaluation happens here.*

2.  **Evaluation Phase (`bfcl evaluate`)**:
    *   Reads the saved results file.
    *   Parses the output (AST decoding).
    *   Compares it against the ground truth using strict checkers (AST Checker).
    *   *No LLM interaction happens here.*

---

## 2. Decision Logic: FC Mode vs. Prompting Mode

The most critical decision—whether to use native Function Calling (FC) or text-based Prompting—is determined **before runtime** by the model name string passed to the CLI.

### The Decision Chain
1.  **User Input**: `bfcl generate --model gemini-2.5-flash-FC`
2.  **Registry Lookup**: `_llm_response_generation.py` looks up `"gemini-2.5-flash-FC"` in `model_config.py`.
3.  **Static Config**: The registry entry hardcodes `is_fc_model = True` or `False`.
4.  **Handler Build**: `build_handler()` instantiates the model handler (e.g., `GeminiHandler`) with this boolean flag frozen.
5.  **Inference Fork**: Inside `base_handler.inference()`, the code branches:
    *   `if self.is_fc_model:` -> call `_query_FC()`
    *   `else:` -> call `_query_prompting()`

---

## 3. Step-by-Step Flow: Generation Phase

Using `simple_python_0` as an example test case:

### Step A: Load Data
*   **Source**: `data/BFCL_v4_simple_python.json`
*   **Content**: A user question (`"Find the area of a triangle..."`) and a generic function schema (`"type": "dict"`).

### Step B: Compile Tools (FC Mode Only)
*   **Action**: `_compile_tools()` converts the generic BFCL schema into the model-specific API format.
*   **Example (Gemini)**: Converts `"type": "dict"` -> `"type": "object"`, wraps in `Tool` object.

### Step C: Construct Payload
*   **Prompting Mode**: Injects the function schema as raw JSON text into a massive system prompt (`DEFAULT_SYSTEM_PROMPT_FORMAT`).
*   **FC Mode**: Sends the curated `tools` array natively to the API.
*   **User Message**: Wraps the question in the model's expected object (e.g., `Content(role="user", parts=[Part(text=...)])`).

### Step D: Query LLM
*   **Input**: Normalized payload sent to the API provider (OpenAI, Google, etc.).
*   **Config**: Temperature set to `0.001` (near-deterministic). Auto-execution is **disabled** (the benchmark wants the tool call, not the result of the tool execution).

### Step E: Parse and Save
*   **Output**: The LLM returns either a structured tool call object (FC Mode) or a raw string (Prompting Mode).
*   **Normalization**: The handler converts this into a standard list of dictionaries: `[{"func_name": {"param": val}}]`.
*   **Storage**: Saved to `result/<model_registry_name>/<category>_result.json`.

---

## 4. Step-by-Step Flow: Evaluation Phase

Using `simple_python_0` results loaded from disk:

### Step A: Decode AST
*   **Prompting Mode**: The raw text string (e.g., `"[func(a=1)]"`) is parsed using Python's `ast.parse()`.
*   **FC Mode**: The already-structured JSON is passed through as-is.
*   **Result**: A clean Python dictionary.

### Step B: AST Checker (`ast_checker.py`)
The checker validates the output against `possible_answer/BFCL_v4_simple_python.json`.

1.  **Function Name**: Matches expected name? (e.g., `calculate_triangle_area`)
2.  **Required Params**: All mandatory keys present? (`base`, `height`)
3.  **Type Check**: Do values match the expected type?
    *   *Note*: Handles language-specifics (e.g., parsing strings to ints for Java/JS).
4.  **Value Check**: Do values match the allowed set?
    *   *Note*: Handles fuzzy matching (stripping spaces/punctuation).
    *   *Note*: `""` in ground truth means the parameter is optional.

### Step C: Scoring
*   Returns `{"valid": True}` or `False` with error details.
*   Aggregated into accuracy scores and leaderboard CSVs.

---

## 5. Key File References

| File | Purpose |
| :--- | :--- |
| `bfcl_eval/_llm_response_generation.py` | Main entry point for Generation Phase. Sets up threading and builds handlers. |
| `bfcl_eval/model_handler/base_handler.py` | The logic core. Contains `inference()`, which forks between `_query_FC` and `_query_prompting`. |
| `bfcl_eval/constants/model_config.py` | The static registry that decides if a model is FC or Prompting mode based on its name. |
| `bfcl_eval/model_handler/api_inference/*.py` | Model-specific implementations (e.g., `gemini.py`, `openai_response.py`). Handles API payload construction. |
| `bfcl_eval/eval_checker/ast_eval/ast_checker.py` | The strict validation logic for the Evaluation Phase. |
| `bfcl_eval/eval_checker/eval_runner.py` | Main entry point for Evaluation Phase. Routes tests to the correct checker. |

---

## 6. Key Code Block Explanations

### 6.1 `_llm_response_generation.py` — Block by Block

| Function | What It Does |
| :--- | :--- |
| `get_args()` | Parses CLI arguments (model name, test categories, temperature, threading, GPU settings, local model paths) using `argparse`. |
| `build_handler(model_name, temperature)` | Looks up the model in `MODEL_CONFIG_MAPPING` and instantiates its specific handler class with the correct temperature and `is_fc_model` flag. |
| `get_involved_test_entries(...)` | Determines which test cases to run. Loads from a specific ID list file if `--run-ids` is set; otherwise loads all entries for the given categories. |
| `collect_test_cases(...)` | Filters test cases to skip already-generated results (unless `--allow-overwrite`). Handles special setup for memory/web-search categories. Skips format sensitivity tests for FC models. |
| `multi_threaded_inference(...)` | Executes one test case. Calls `handler.inference()` inside a `try...except` block. On failure, logs the error and records the error string as the result so the benchmark does not crash. |
| `generate_results(...)` | Manages parallel execution. Starts a background writer thread for safe file I/O. Uses `ThreadPoolExecutor` with a priority queue (`ready_queue`) to enforce ordering for multi-turn/memory tests that have dependencies. |
| `main(args)` | Top-level entry point. Sets tokenizer env vars, validates models against the registry, iterates models, calls `collect_test_cases` → `generate_results`, and sorts final output files by test ID. |

---

### 6.2 `__main__.py` — CLI Commands (Typer)

| Command / Block | What It Does |
| :--- | :--- |
| `ExecutionOrderGroup` | Custom Typer group that overrides `list_commands` to display CLI help in a logical order instead of alphabetically. |
| `handle_multiple_input()` | Callback that converts a comma-separated CLI string (e.g., `"modelA,modelB"`) into a Python list. |
| `bfcl version` | Prints the installed package version. |
| `bfcl test-categories` | Prints a formatted table of all available test categories. |
| `bfcl models` | Prints all supported models from `MODEL_CONFIG_MAPPING`. |
| `bfcl generate` | Packs CLI options into a `SimpleNamespace`, loads `.env`, and calls `generation_main()`. |
| `bfcl results` | Scans the results directory and prints a table of models with generated results and timestamps. |
| `bfcl evaluate` | Loads `.env` and triggers `evaluation_main()` from `eval_runner.py`. |
| `bfcl scores` | Reads `data_overall.csv` from the scores directory and prints the leaderboard table. |

---

### 6.3 `base_handler.py` — The Mandatory Inference Sequence

`BaseHandler` is the blueprint every model handler must follow. It enforces a strict lifecycle for every test case.

**The `inference()` method is a router:**
- `if self.is_fc_model` → routes to `inference_single_turn_FC` or `inference_multi_turn_FC`
- `else` → routes to `inference_single_turn_prompting` or `inference_multi_turn_prompting`

**Single-Turn FC sequence (the core pattern):**
1. `_pre_query_processing_FC()` — Initialize the payload dict.
2. `_compile_tools()` — Convert BFCL JSON schema → model-specific tool format.
3. `add_first_turn_message_FC()` — Format the user prompt and add it to the payload.
4. `_query_FC()` — 🔥 The actual API call to the LLM provider.
5. `_parse_query_response_FC()` — Normalize the raw API response into BFCL standard format.

**Single-Turn Prompting sequence:**
- Same as above but **no `_compile_tools()` step**. Instead, function schemas are baked into the system prompt text during pre-processing.

**Multi-Turn loop:**
- Runs the single-turn sequence inside a `while True:` loop (capped by `MAXIMUM_STEP_LIMIT`).
- After each `_query_FC()`, it executes the requested function locally via `execute_multi_turn_func_call()`.
- The execution result is fed back into the model's chat history with `_add_execution_results_FC()`.
- Loop continues until the model stops calling tools.

**Abstract methods (`raise NotImplementedError`):**
- Methods like `_query_FC`, `_compile_tools`, `_parse_query_response_FC` are purposely unimplemented.
- This forces every specific handler (e.g., `GeminiHandler`, `OpenAIHandler`) to write custom API-specific logic for these exact steps.

---

### 6.4 `multi_threaded_inference` — How the Thread Wraps `handler.inference()`

```
ThreadPoolExecutor.submit(multi_threaded_inference, handler, test_case, ...)
    └── try:
            result = handler.inference(test_case, ...)
            return {"id": ..., "result": result, "metadata": ...}
        except Exception as e:
            tqdm.write(error + traceback)
            return {"id": ..., "result": error_string, "metadata": traceback}
```
- On **success**: returns the structured result dict.
- On **failure**: catches any exception (timeout, invalid JSON, rate limit), logs it, and returns the error as the result — **the benchmark never crashes due to a single test case failure**.

---

## 7. Model-Specific Handler Details

### 7.1 `gemini.py` — How Gemini Uses the Google SDK

Gemini does **not** use the OpenAI-compatible format. It uses Google's native `google-genai` SDK, requiring Google-specific objects.

**`_compile_tools()`**
- Calls `convert_to_tool(..., GORILLA_TO_OPENAPI, self.model_style)`.
- Converts the generic BFCL JSON schema into a `Tool(function_declarations=[...])` object.

**`_query_FC()`**
- Builds a `GenerateContentConfig` with temperature and thinking config.
- **CRITICAL:** Sets `AutomaticFunctionCallingConfig(disable=True)`. This prevents Google's backend from auto-executing the function. The benchmark only needs the tool call JSON, not the execution result.
- If `system_prompt` exists in `inference_data`, it is set as `config.system_instruction`.
- Fires the request via `self.client.models.generate_content(model, contents, config)`.

**`_parse_query_response_FC()`**
- Unpacks `api_response.candidates[0].content.parts`.
- A "part" can be text, a thought (reasoning), or a function call.
- Iterates through parts:
  - `part.function_call` → extracts `name` and `args`.
  - `part.thought` → aggregates reasoning text.
  - Otherwise → treated as plain text.

---

### 7.2 `qwen.py` — How Qwen Uses Streaming

Qwen reasoning models (QwQ, Qwen3) **only support streaming** responses. `QwenAPIHandler` inherits from `OpenAICompletionsHandler` but overrides the query and parse methods.

**`QwenAPIHandler.__init__()`**
- Uses the standard `openai.OpenAI` client pointed at Alibaba DashScope: `base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"`.

**`_query_FC()` (streaming enabled)**
- Calls `chat.completions.create(...)` with `stream=True`, `stream_options={"include_usage": True}`, and `extra_body={"enable_thinking": True}`.

**`_parse_query_response_FC()` (stream reconstruction)**
- Loops `for chunk in api_response:`.
- Concatenates `delta.reasoning_content` for the thinking trace.
- Concatenates `delta.content` for the text response.
- For tool calls: uses `tool_call.index` to correctly interleave parallel tool call chunks into a `tool_info` list.
- After the stream ends, reconstructs the standard OpenAI `tool_calls` array format.

**`QwenAgentThinkHandler`** (local vLLM variant)
- Uses `qwen_agent.llm.get_chat_model` pointing to `localhost:8000/v1`.
- `_query_FC` uses `self.llm.quick_chat_oai()`, a generator; iterates to get the final complete response.

---

## 8. System Prompt Handling

### 8.1 The Two Message Roles in Test Data

| Role | Who It Is | Purpose |
| :--- | :--- | :--- |
| `system` | The app/developer | Sets rules and context for the AI before the conversation starts |
| `user` | The human tester | Asks the actual question that triggers the function call |

Some test cases include a `system` role message; others go straight to `user`. Both are valid — the `system` message is optional.

---

### 8.2 System Prompt Flow in Gemini (FC Mode)

**Order of operations:**

1. **Raw data** contains `[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]` (or no system at all).
2. **`_pre_query_processing_FC()`** calls `extract_system_prompt()`, which finds and **removes** the `system` message from the list, saving it to `inference_data["system_prompt"]`.
3. **`_query_FC()`** checks:
   ```python
   if "system_prompt" in inference_data:
       config.system_instruction = inference_data["system_prompt"]
   ```
   If no system message existed, this block is skipped entirely.
4. **API call** sends:
   - `contents` → only user/model turns (no system message here)
   - `config.system_instruction` → system prompt in its dedicated Google SDK field

---

### 8.3 Critical Difference: FC Mode vs. Prompting Mode (System Prompt)

| Mode | No system prompt in test data | Has system prompt in test data |
| :--- | :--- | :--- |
| **FC Mode** | Nothing added. `config.system_instruction` is not set. | Extracted → injected into `config.system_instruction`. |
| **Prompting Mode** | **A full default system prompt is always built and inserted**, even if the test case has none. | Default prompt is prepended to the existing one. |

In **Prompting mode**, `system_prompt_pre_processing_chat_model()` always constructs a system prompt from `default_prompts.py` components:
- `persona` → "You are an expert in composing functions."
- `task` → "You are given a question and a set of possible functions..."
- `tool_call_format` → "You MUST put it in the format of `[func_name1(...)]`"
- `available_tools` → the actual JSON function schemas baked in as raw text

This means in Prompting mode, the model **always** receives the function schemas via the system prompt, regardless of whether the raw test case had a system message.

---

### 8.4 Concrete Example: `simple_python_0` in Gemini FC Mode

```json
{"role": "user", "content": "Find the area of a triangle with a base of 10..."}
```
- **No `system` role** in the test case.
- `extract_system_prompt()` returns `None`.
- `inference_data["system_prompt"]` is never set.
- `if "system_prompt" in inference_data:` → **skipped**.
- API receives: only the user message + the `calculate_triangle_area` tool definition.
- `config.system_instruction` → **not set**.
