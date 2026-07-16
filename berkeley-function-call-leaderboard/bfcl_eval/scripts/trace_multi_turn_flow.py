import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure `bfcl_eval` imports work when the script is run directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
    multi_turn_checker,
    multi_turn_irrelevance_checker,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    is_empty_execute_response,
)

VERSION_PREFIX = "BFCL_v4"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_PATH = PROJECT_ROOT / "result"
DATA_PATH = PACKAGE_ROOT / "data"
POSSIBLE_ANSWER_PATH = DATA_PATH / "possible_answer"


def _normalize_model_dir_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def _load_jsonl(file_path: Path) -> list[dict]:
    entries = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _find_entry_by_id(entries: list[dict], test_id: str) -> dict:
    for entry in entries:
        if entry["id"] == test_id:
            return entry
    raise ValueError(f"Cannot find entry with id '{test_id}'.")


def _extract_test_category_from_id(test_id: str) -> str:
    return test_id.rsplit("_", 1)[0]


def _extract_trace_from_inference_log(inference_log: list[Any]) -> list[dict]:
    """
    Convert the raw `inference_log` structure into a cleaner turn/step summary.
    """
    turn_traces: list[dict] = []
    turn_counter = 0

    for log_item in inference_log:
        if not isinstance(log_item, dict) or "begin_of_turn_query" not in log_item:
            continue

        turn_trace = {
            "turn_index": turn_counter,
            "begin_of_turn_query": log_item.get("begin_of_turn_query", []),
            "steps": [],
        }

        step_keys = [
            key
            for key in log_item.keys()
            if key.startswith("step_") and key.split("_", 1)[1].isdigit()
        ]
        step_keys.sort(key=lambda x: int(x.split("_", 1)[1]))

        for step_key in step_keys:
            step_idx = int(step_key.split("_", 1)[1])
            events = log_item[step_key]
            step_summary = {
                "step_index": step_idx,
                "model_input": None,
                "model_response_parsed": None,
                "decoded_model_response": None,
                "execution_results": [],
                "handler_messages": [],
            }

            for event in events:
                role = event.get("role")
                if role == "inference_input":
                    step_summary["model_input"] = event.get("content")
                elif role == "assistant":
                    step_summary["model_response_parsed"] = event.get("content")
                elif role == "handler_log":
                    handler_message = {
                        "content": event.get("content"),
                        "error": event.get("error"),
                    }
                    if "model_response_decoded" in event:
                        step_summary["decoded_model_response"] = event.get(
                            "model_response_decoded"
                        )
                        handler_message["model_response_decoded"] = event.get(
                            "model_response_decoded"
                        )
                    step_summary["handler_messages"].append(handler_message)
                elif role == "tool":
                    step_summary["execution_results"].append(event.get("content"))

            turn_trace["steps"].append(step_summary)

        turn_traces.append(turn_trace)
        turn_counter += 1

    return turn_traces


def _decode_result_from_inference_log(
    raw_result: list[list[Any]],
    trace_from_inference_log: list[dict],
) -> tuple[list[list[list[str]]], list[dict], str | None]:
    """
    Reconstruct eval-style decoded outputs from per-step `handler_log` entries
    inside `inference_log`.
    """
    if not trace_from_inference_log:
        return [], [], "Missing inference_log turn traces; cannot reconstruct decoded outputs."

    decode_trace = []
    decoded_multi_turn = []

    for turn_idx, single_turn_model_result_list in enumerate(raw_result):
        decoded_turn = []
        turn_trace = {"turn_index": turn_idx, "steps": []}
        turn_log = (
            trace_from_inference_log[turn_idx]
            if turn_idx < len(trace_from_inference_log)
            else {"steps": []}
        )

        for step_idx, model_result_item in enumerate(single_turn_model_result_list):
            step_from_log = (
                turn_log["steps"][step_idx]
                if step_idx < len(turn_log.get("steps", []))
                else {}
            )
            decoded = step_from_log.get("decoded_model_response")
            decode_error = None
            kept_for_eval = False

            if decoded is not None:
                if not is_empty_execute_response(decoded):
                    decoded_turn.append(decoded)
                    kept_for_eval = True
            else:
                decode_error = (
                    "No decoded response found in inference_log for this step. "
                    "Regenerate with normal inference log output."
                )

            turn_trace["steps"].append(
                {
                    "step_index": step_idx,
                    "raw_result": model_result_item,
                    "decoded": decoded,
                    "decode_error": decode_error,
                    "kept_for_eval": kept_for_eval,
                }
            )

        decode_trace.append(turn_trace)
        decoded_multi_turn.append(decoded_turn)

    return decoded_multi_turn, decode_trace, None


def _load_prompt_and_ground_truth(test_category: str, test_id: str) -> tuple[dict, list]:
    prompt_file = DATA_PATH / f"{VERSION_PREFIX}_{test_category}.json"
    ground_truth_file = POSSIBLE_ANSWER_PATH / f"{VERSION_PREFIX}_{test_category}.json"

    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    if not ground_truth_file.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {ground_truth_file}")

    prompt_entries = _load_jsonl(prompt_file)
    ground_truth_entries = _load_jsonl(ground_truth_file)

    prompt_entry = _find_entry_by_id(prompt_entries, test_id)
    ground_truth_entry = _find_entry_by_id(ground_truth_entries, test_id)

    return prompt_entry, ground_truth_entry["ground_truth"]


def _resolve_result_file(result_dir: Path, model_name: str, test_category: str) -> Path:
    model_dir_name = _normalize_model_dir_name(model_name)
    # This script is dedicated to multi-turn categories.
    return result_dir / model_dir_name / "multi_turn" / f"{VERSION_PREFIX}_{test_category}_result.json"


def _is_multi_turn(test_category: str) -> bool:
    return "multi_turn" in test_category


def _resolve_path_from_project_root(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _build_output(
    args: argparse.Namespace,
    test_category: str,
    prompt_entry: dict,
    ground_truth: list,
    raw_result: list[list[Any]],
    trace_from_inference_log: list[dict],
    decoded_trace: list[dict],
    decoded_result: list[list[list[str]]],
    decode_error: str | None,
) -> dict:
    checker_result = None
    irrelevance_result = None
    if decode_error is None:
        checker_result = multi_turn_checker(
            decoded_result,
            ground_truth,
            prompt_entry,
            test_category,
            args.model,
        )
        irrelevance_result = multi_turn_irrelevance_checker(decoded_result, ground_truth)

    return {
        "id": args.test_id,
        "model": args.model,
        "test_category": test_category,
        "question": prompt_entry["question"],
        "ground_truth": ground_truth,
        "raw_result": raw_result,
        "trace_from_inference_log": trace_from_inference_log,
        "decoded_trace_eval_logic": decoded_trace,
        "decoded_result_eval_logic": decoded_result,
        "decode_error": decode_error,
        "checker_result": checker_result,
        "irrelevance_result": irrelevance_result,
        "evaluation_llm_calls_model_again": False,
        "evaluation_note": (
            "Evaluation decodes and executes already-generated outputs; "
            "it does not call _query_* model APIs again."
        ),
    }


def _write_or_print_output(summary: dict, output_arg: str | None) -> None:
    summary = _make_json_serializable(summary)
    if output_arg is not None:
        output_path = _resolve_path_from_project_root(output_arg)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Trace written to: {output_path}")
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


def _make_json_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_json_serializable(item) for item in value]
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except (TypeError, ValueError):
        return str(value)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Trace one multi-turn BFCL entry: model input, parsed model response, "
            "decoded function calls, and evaluation outcome."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help=(
            "Model registry name, e.g. 'Qwen/Qwen3-8B-FC' or "
            "'gpt-4.1-2025-04-14-FC'."
        ),
    )
    parser.add_argument(
        "--test-id",
        required=True,
        type=str,
        help="Target multi-turn test id, e.g. 'multi_turn_miss_param_0'.",
    )
    parser.add_argument(
        "--result-dir",
        default=None,
        type=str,
        help=(
            "Optional custom result directory relative to project root. "
            "Defaults to <project_root>/result."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="Optional output path for the trace JSON (relative to project root).",
    )
    args = parser.parse_args()

    test_category = _extract_test_category_from_id(args.test_id)
    if not _is_multi_turn(test_category):
        raise ValueError(
            f"test-id '{args.test_id}' is not a multi-turn category (category='{test_category}')."
        )

    result_dir = (
        DEFAULT_RESULT_PATH
        if args.result_dir is None
        else _resolve_path_from_project_root(args.result_dir)
    )
    result_file = _resolve_result_file(result_dir, args.model, test_category)
    if not result_file.exists():
        raise FileNotFoundError(
            f"Result file not found: {result_file}. Run generation first."
        )

    all_results = _load_jsonl(result_file)
    result_entry = _find_entry_by_id(all_results, args.test_id)
    prompt_entry, ground_truth = _load_prompt_and_ground_truth(test_category, args.test_id)

    raw_result = result_entry["result"]
    inference_log = result_entry.get("inference_log", [])
    trace_from_inference_log = _extract_trace_from_inference_log(inference_log)

    decoded_result, decode_trace, decode_error = _decode_result_from_inference_log(
        raw_result,
        trace_from_inference_log,
    )

    summary = _build_output(
        args=args,
        test_category=test_category,
        prompt_entry=prompt_entry,
        ground_truth=ground_truth,
        raw_result=raw_result,
        trace_from_inference_log=trace_from_inference_log,
        decoded_trace=decode_trace,
        decoded_result=decoded_result,
        decode_error=decode_error,
    )
    _write_or_print_output(summary, args.output)


if __name__ == "__main__":
    main()
