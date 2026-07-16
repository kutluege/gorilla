"""
Practical BFCL flow tracer for real result files.

What it answers per step:
1) What went into the model (`inference_input`), if generation used --include-input-log
2) What handler logged
3) What model responded
4) What parser/decoder produced
5) What execution output returned

Example:
python trace_flow.py --model "Qwen/Qwen3-4B-Instruct-2507-FC" --test-id "multi_turn_miss_param_0" --output "tmp/flow_from_result.json"
"""

import argparse
import json
from pathlib import Path
from typing import Any


VERSION_PREFIX = "BFCL_v4"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = PROJECT_ROOT / "result"


def load_jsonl(file_path: Path) -> list[dict]:
    entries: list[dict] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def find_entry_by_id(entries: list[dict], test_id: str) -> dict:
    for entry in entries:
        if entry.get("id") == test_id:
            return entry
    raise ValueError(f"Entry '{test_id}' not found in result file.")


def test_category_from_id(test_id: str) -> str:
    return test_id.rsplit("_", 1)[0]


def model_dir(model_name: str) -> str:
    return model_name.replace("/", "_")


def resolve_result_file(result_dir: Path, model_name: str, test_id: str) -> Path:
    category = test_category_from_id(test_id)
    return result_dir / model_dir(model_name) / "multi_turn" / f"{VERSION_PREFIX}_{category}_result.json"


def parse_inference_log(inference_log: list[Any]) -> tuple[list[dict], dict]:
    """
    Transform raw inference_log into a compact step-by-step structure.
    """
    turns: list[dict] = []
    counters = {
        "total_steps": 0,
        "steps_with_model_input": 0,
        "steps_without_model_input": 0,
    }

    for item in inference_log:
        if not isinstance(item, dict) or "begin_of_turn_query" not in item:
            continue

        turn = {
            "begin_of_turn_query": item.get("begin_of_turn_query", []),
            "steps": [],
        }
        step_keys = [k for k in item if k.startswith("step_")]
        step_keys.sort(key=lambda x: int(x.split("_", 1)[1]))

        for step_key in step_keys:
            events = item.get(step_key, [])
            model_input = None
            model_response = None
            decoded = None
            exec_results: list[Any] = []
            handler_logs: list[dict] = []

            for ev in events:
                role = ev.get("role")
                if role == "inference_input":
                    model_input = ev.get("content")
                elif role == "assistant":
                    model_response = ev.get("content")
                elif role == "handler_log":
                    handler_logs.append(
                        {
                            "content": ev.get("content"),
                            "error": ev.get("error"),
                        }
                    )
                    if "model_response_decoded" in ev:
                        decoded = ev.get("model_response_decoded")
                elif role == "tool":
                    exec_results.append(ev.get("content"))

            counters["total_steps"] += 1
            if model_input is None:
                counters["steps_without_model_input"] += 1
            else:
                counters["steps_with_model_input"] += 1

            turn["steps"].append(
                {
                    "step": int(step_key.split("_", 1)[1]),
                    "model_input": model_input,
                    "handler_logs": handler_logs,
                    "model_response": model_response,
                    "parsed_output": decoded,
                    "execution_output": exec_results,
                }
            )

        turns.append(turn)

    return turns, counters


def make_json_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_serializable(v) for v in value]
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except Exception:
        return str(value)


def build_report(entry: dict) -> dict:
    inference_log = entry.get("inference_log", [])
    turns, counters = parse_inference_log(inference_log)

    report = {
        "id": entry.get("id"),
        "turn_count": len(turns),
        "flow": turns,
        "raw_result": entry.get("result"),
        "has_full_model_input_trace": counters["steps_without_model_input"] == 0,
        "summary": counters,
        "next_action_if_model_input_missing": (
            "Regenerate with --include-input-log, then rerun this script."
            if counters["steps_without_model_input"] > 0
            else "Model input was captured for all steps."
        ),
    }
    return make_json_serializable(report)


def main():
    parser = argparse.ArgumentParser(
        description="Trace real multi-turn BFCL result entry from saved result JSON."
    )
    parser.add_argument("--model", required=True, help="BFCL model registry name.")
    parser.add_argument("--test-id", required=True, help="BFCL test id.")
    parser.add_argument(
        "--result-dir",
        default=None,
        help="Result dir path. Relative to repo root if not absolute. Default: ./result",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path. Relative to repo root if not absolute.",
    )
    args = parser.parse_args()

    result_dir = (
        DEFAULT_RESULT_DIR
        if args.result_dir is None
        else (Path(args.result_dir) if Path(args.result_dir).is_absolute() else PROJECT_ROOT / args.result_dir)
    )
    result_file = resolve_result_file(result_dir, args.model, args.test_id)
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")

    entries = load_jsonl(result_file)
    entry = find_entry_by_id(entries, args.test_id)
    report = build_report(entry)

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Trace written to: {output_path}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

