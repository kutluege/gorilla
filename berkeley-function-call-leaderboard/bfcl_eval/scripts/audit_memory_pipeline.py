"""Read-only BFCL memory pipeline audit.

The goal is to explain memory-score discrepancies from persisted artifacts,
without overwriting result/score files or rerunning inference.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_MODEL_DIR = "Qwen_Qwen3-4B-Instruct-2507-FC"
OFFICIAL_COMMIT = "f7cf735"
OFFICIAL_PACKAGE = "bfcl-eval==2025.12.17"
SCENARIOS = ["customer", "healthcare", "finance", "student", "notetaker"]
BACKENDS = {
    "kv": {
        "category": "memory_kv",
        "class": "MemoryAPI_kv",
        "func_doc": "memory_kv.json",
        "result": "BFCL_v4_memory_kv_result.json",
        "prereq_result": "BFCL_v4_memory_kv_prereq_result.json",
        "score": "BFCL_v4_memory_kv_score.json",
    },
    "vector": {
        "category": "memory_vector",
        "class": "MemoryAPI_vector",
        "func_doc": "memory_vector.json",
        "result": "BFCL_v4_memory_vector_result.json",
        "prereq_result": "BFCL_v4_memory_vector_prereq_result.json",
        "score": "BFCL_v4_memory_vector_score.json",
    },
    "rec_sum": {
        "category": "memory_rec_sum",
        "class": "MemoryAPI_rec_sum",
        "func_doc": "memory_rec_sum.json",
        "result": "BFCL_v4_memory_rec_sum_result.json",
        "prereq_result": "BFCL_v4_memory_rec_sum_prereq_result.json",
        "score": "BFCL_v4_memory_rec_sum_score.json",
    },
}
MEMORY_CALL_RE = re.compile(
    r"\b(?:core_memory|archival_memory|memory)_(?:add|append|update|replace|clear|remove)\b"
)
TOOL_CALL_RE = re.compile(r"<tool_call>")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def scenario_from_id(entry_id: str) -> str:
    parts = entry_id.split("-")
    return parts[1] if len(parts) >= 3 else "unknown"


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(flatten_text(item) for item in value)
    if isinstance(value, dict):
        return "\n".join(flatten_text(item) for item in value.values())
    return str(value)


def collect_decoded_calls(log_value: Any) -> list[str]:
    calls: list[str] = []
    if isinstance(log_value, dict):
        decoded = log_value.get("model_response_decoded")
        if isinstance(decoded, list):
            for item in decoded:
                if isinstance(item, str):
                    calls.append(item)
                else:
                    calls.extend(collect_decoded_calls(item))
        for value in log_value.values():
            calls.extend(collect_decoded_calls(value))
    elif isinstance(log_value, list):
        for item in log_value:
            calls.extend(collect_decoded_calls(item))
    return calls


def count_snapshot_entries(path: Path, backend: str) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "core_entries": 0, "archival_entries": 0, "memory_chars": 0}
    data = read_json(path)
    if backend == "rec_sum":
        memory = data.get("memory", "")
        return {
            "exists": True,
            "core_entries": 0,
            "archival_entries": 0,
            "memory_chars": len(memory),
            "empty": len(memory) == 0,
        }
    core = data.get("core_memory", {})
    archival = data.get("archival_memory", {})
    core_store = core.get("store", core) if isinstance(core, dict) else {}
    archival_store = archival.get("store", archival) if isinstance(archival, dict) else {}
    core_entries = len(core_store) if isinstance(core_store, dict) else 0
    archival_entries = len(archival_store) if isinstance(archival_store, dict) else 0
    return {
        "exists": True,
        "core_entries": core_entries,
        "archival_entries": archival_entries,
        "memory_chars": 0,
        "empty": core_entries == 0 and archival_entries == 0,
    }


def git(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def category_id(backend_category: str, memory_id: str) -> str:
    return memory_id.replace("memory", backend_category, 1)


def score_by_scenario(score_rows: list[dict[str, Any]], result_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    failures = Counter(scenario_from_id(row["id"]) for row in score_rows[1:] if "id" in row)
    totals = Counter(scenario_from_id(row["id"]) for row in result_rows if "id" in row)
    stats: dict[str, dict[str, int]] = {}
    for scenario in SCENARIOS:
        total = totals[scenario]
        failed = failures[scenario]
        stats[scenario] = {
            "total": total,
            "failed": failed,
            "correct": total - failed,
        }
    return stats


def audit_backend(root: Path, model_dir: str, backend: str) -> dict[str, Any]:
    cfg = BACKENDS[backend]
    backend_root = root / "result" / model_dir / "agentic" / "memory" / backend
    score_root = root / "score" / model_dir / "agentic" / "memory" / backend
    prereq_rows = read_jsonl(backend_root / cfg["prereq_result"])
    result_rows = read_jsonl(backend_root / cfg["result"])
    score_rows = read_jsonl(score_root / cfg["score"])
    snapshot_dir = backend_root / "memory_snapshot"

    prereq_by_scenario: dict[str, dict[str, Any]] = {}
    for scenario in SCENARIOS:
        subset = [row for row in prereq_rows if scenario_from_id(row.get("id", "")) == scenario]
        raw_text = "\n".join(flatten_text(row.get("result")) for row in subset)
        decoded_calls = []
        for row in subset:
            decoded_calls.extend(collect_decoded_calls(row.get("inference_log")))
        memory_decoded_calls = [call for call in decoded_calls if MEMORY_CALL_RE.search(call)]
        checkpoint_dir = snapshot_dir / "prereq_checkpoints"
        checkpoints = sorted(checkpoint_dir.glob(f"*{scenario}*.json"))
        progression = [
            {
                "file": path.name,
                **count_snapshot_entries(path, backend),
            }
            for path in checkpoints
        ]
        final_snapshot = snapshot_dir / f"{scenario}_final.json"
        prereq_by_scenario[scenario] = {
            "prereq_rows": len(subset),
            "raw_tool_call_tags": len(TOOL_CALL_RE.findall(raw_text)),
            "raw_memory_call_mentions": len(MEMORY_CALL_RE.findall(raw_text)),
            "decoded_calls": len(decoded_calls),
            "decoded_memory_calls": len(memory_decoded_calls),
            "first_result_preview": flatten_text(subset[0].get("result"))[:700] if subset else "",
            "checkpoint_progression": progression,
            "final_snapshot": count_snapshot_entries(final_snapshot, backend),
        }

    return {
        "backend": backend,
        "category": cfg["category"],
        "result_rows": len(result_rows),
        "prereq_rows": len(prereq_rows),
        "score_header": score_rows[0] if score_rows else {},
        "score_by_scenario": score_by_scenario(score_rows, result_rows),
        "prereq_by_scenario": prereq_by_scenario,
    }


def inspect_prompt_tools(root: Path) -> dict[str, Any]:
    func_doc_dir = root / "bfcl_eval" / "data" / "multi_turn_func_doc"
    prereq_dir = root / "bfcl_eval" / "data" / "memory_prereq_conversation"
    prompt_info: dict[str, Any] = {}
    for backend, cfg in BACKENDS.items():
        functions = read_jsonl(func_doc_dir / cfg["func_doc"])
        names = [func.get("name", "") for func in functions]
        write_tools = [name for name in names if MEMORY_CALL_RE.search(name)]
        first_student = read_jsonl(prereq_dir / "memory_student.json")[0]
        first_customer = read_jsonl(prereq_dir / "memory_customer.json")[0]
        prompt_info[backend] = {
            "function_count": len(functions),
            "write_tool_count": len(write_tools),
            "write_tools": write_tools,
            "student_first_prereq_id_if_expanded": category_id(cfg["category"], first_student["id"]),
            "student_first_question_preview": flatten_text(first_student["question"])[:500],
            "customer_first_prereq_id_if_expanded": category_id(cfg["category"], first_customer["id"]),
            "customer_first_question_preview": flatten_text(first_customer["question"])[:500],
        }
    return prompt_info


def diff_against_official(root: Path) -> dict[str, Any]:
    paths = [
        "bfcl_eval/data/BFCL_v4_memory.json",
        "bfcl_eval/data/possible_answer/BFCL_v4_memory.json",
        "bfcl_eval/data/memory_prereq_conversation",
        "bfcl_eval/data/multi_turn_func_doc/memory_kv.json",
        "bfcl_eval/data/multi_turn_func_doc/memory_vector.json",
        "bfcl_eval/data/multi_turn_func_doc/memory_rec_sum.json",
        "bfcl_eval/eval_checker/multi_turn_eval/func_source_code/memory_api_metaclass.py",
        "bfcl_eval/eval_checker/multi_turn_eval/func_source_code/memory_kv.py",
        "bfcl_eval/eval_checker/multi_turn_eval/func_source_code/memory_vector.py",
        "bfcl_eval/eval_checker/multi_turn_eval/func_source_code/memory_rec_sum.py",
        "bfcl_eval/eval_checker/multi_turn_eval/multi_turn_utils.py",
        "bfcl_eval/model_handler/base_handler.py",
        "bfcl_eval/model_handler/local_inference/base_oss_handler.py",
        "bfcl_eval/model_handler/local_inference/qwen_fc.py",
        "bfcl_eval/model_handler/utils.py",
        "bfcl_eval/_llm_response_generation.py",
        "bfcl_eval/utils.py",
    ]
    return {
        "name_status": git(root, "diff", "--name-status", OFFICIAL_COMMIT, "--", *paths),
        "stat": git(root, "diff", "--stat", OFFICIAL_COMMIT, "--", *paths),
    }


def result_file_times(root: Path, model_dir: str) -> list[dict[str, str]]:
    base = root / "result" / model_dir / "agentic" / "memory"
    paths = sorted(base.rglob("*.json"))
    rows = []
    for path in paths:
        stat = path.stat()
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "bytes": str(stat.st_size),
            }
        )
    return rows


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def pct(correct: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{100 * correct / total:.2f}%"


def build_report(root: Path, model_dir: str) -> str:
    audits = {backend: audit_backend(root, model_dir, backend) for backend in BACKENDS}
    prompt_tools = inspect_prompt_tools(root)
    diffs = diff_against_official(root)
    file_times = result_file_times(root, model_dir)

    lines = [
        "# BFCL Memory Pipeline Audit",
        "",
        "## Version And Diff",
        f"- Local HEAD: `{git(root, 'rev-parse', 'HEAD')}`",
        f"- Local describe: `{git(root, 'describe', '--always', '--dirty', '--tags')}`",
        f"- Official target: `{OFFICIAL_COMMIT}` / `{OFFICIAL_PACKAGE}`",
        "- Official leaderboard: https://gorilla.cs.berkeley.edu/leaderboard",
        "",
        "Changed memory-relevant paths versus official target:",
        "",
        "```text",
        diffs["name_status"] or "(no diff in inspected paths)",
        "```",
        "",
        "Diff stat:",
        "",
        "```text",
        diffs["stat"] or "(no diff stat)",
        "```",
        "",
        "## Pipeline Trace",
        "- `load_dataset_entry(memory_*)` expands `BFCL_v4_memory.json` with prereq conversations from `bfcl_eval/data/memory_prereq_conversation`.",
        "- `populate_test_cases_with_predefined_functions` injects backend-specific function docs from `bfcl_eval/data/multi_turn_func_doc`.",
        "- `_llm_response_generation.py` schedules prereq dependencies, writes result JSONL, and only flushes memory snapshots at the end of each memory prereq entry.",
        "- `execute_multi_turn_func_call` keeps memory backend instances in module globals keyed by model, test id, and class; decoded calls mutate that instance.",
        "- `_prepare_snapshot` loads `<scenario>_final.json` for non-first prereq and answer entries, while `_flush_memory_to_local_file` writes both prereq checkpoints and the final scenario snapshot.",
        "",
        "Important stale-output risk: generation cleanup checks `model_result_dir / 'memory_snapshot' / test_category`, but actual memory snapshots are under `agentic/memory/<backend>/memory_snapshot`. A full overwrite may delete result files but leave old snapshots unless the current code path is corrected or snapshots are manually cleared in an isolated directory.",
        "",
        "## Backend Scenario Health",
    ]

    for backend, audit in audits.items():
        lines.extend(["", f"### {backend}", ""])
        rows = []
        for scenario in SCENARIOS:
            prereq = audit["prereq_by_scenario"][scenario]
            score = audit["score_by_scenario"][scenario]
            final = prereq["final_snapshot"]
            if backend == "rec_sum":
                final_state = f"memory_chars={final['memory_chars']}"
            else:
                final_state = f"core={final['core_entries']} archival={final['archival_entries']}"
            rows.append(
                [
                    scenario,
                    prereq["prereq_rows"],
                    prereq["raw_tool_call_tags"],
                    prereq["decoded_memory_calls"],
                    final_state,
                    f"{score['correct']}/{score['total']} ({pct(score['correct'], score['total'])})",
                ]
            )
        lines.extend(
            md_table(
                ["Scenario", "Prereq Rows", "Raw Tool Tags", "Decoded Memory Calls", "Final Snapshot", "Score"],
                rows,
            )
        )

    lines.extend(["", "## Tool Availability Check"])
    for backend, info in prompt_tools.items():
        lines.extend(
            [
                "",
                f"### {backend}",
                f"- Function docs loaded: `{info['function_count']}`",
                f"- Memory write tools available: `{info['write_tool_count']}`",
                f"- Write tools: `{', '.join(info['write_tools'])}`",
                f"- First expanded student prereq id: `{info['student_first_prereq_id_if_expanded']}`",
                f"- First student prompt preview: {info['student_first_question_preview']}",
                f"- First expanded customer prereq id: `{info['customer_first_prereq_id_if_expanded']}`",
                f"- First customer prompt preview: {info['customer_first_question_preview']}",
            ]
        )

    lines.extend(["", "## Deep Failure Notes"])
    for backend, audit in audits.items():
        student = audit["prereq_by_scenario"]["student"]
        lines.extend(
            [
                "",
                f"### {backend} student",
                f"- Student prereq rows: `{student['prereq_rows']}`",
                f"- Raw `<tool_call>` tags: `{student['raw_tool_call_tags']}`",
                f"- Decoded memory calls: `{student['decoded_memory_calls']}`",
                f"- First prereq result preview: {student['first_result_preview'] or '(empty)'}",
            ]
        )
        progression = student["checkpoint_progression"]
        compact = []
        for checkpoint in progression:
            if backend == "rec_sum":
                state = f"{checkpoint['file']}:{checkpoint['memory_chars']} chars"
            else:
                state = (
                    f"{checkpoint['file']}:c{checkpoint['core_entries']}"
                    f"/a{checkpoint['archival_entries']}"
                )
            compact.append(state)
        lines.append(f"- Checkpoint progression: `{'; '.join(compact)}`")

    healthcare_vector = audits["vector"]["prereq_by_scenario"]["healthcare"]
    lines.extend(
        [
            "",
            "### vector healthcare",
            "The final healthcare vector snapshot has populated core memory and empty archival memory. This is not equivalent to the student hard failure: answer-time prompts include core memory in context, while archival memory requires explicit retrieval. This needs score-level review to decide whether empty archival is acceptable or a retrieval weakness.",
            f"- Final snapshot: `{healthcare_vector['final_snapshot']}`",
            f"- Decoded memory calls: `{healthcare_vector['decoded_memory_calls']}`",
        ]
    )

    lines.extend(
        [
            "",
            "## Artifact Timing",
            "Memory artifact modification times can help decide whether result files were produced before or after local code edits, but they cannot prove the exact runtime code by themselves.",
            "",
        ]
    )
    time_rows = [
        [row["path"], row["modified"], row["bytes"]]
        for row in file_times
        if row["path"].endswith("_result.json") or row["path"].endswith("_final.json")
    ]
    lines.extend(md_table(["Path", "Modified", "Bytes"], time_rows[:60]))
    if len(time_rows) > 60:
        lines.append(f"\n_Only first 60 of {len(time_rows)} memory artifacts shown._")

    lines.extend(
        [
            "",
            "## Reproduction Commands",
            "Use an isolated result directory, include input logs, and run single-threaded to make prompt/tool availability auditable. Do not point these commands at the existing `result/` tree.",
            "",
            "```powershell",
            "python -m bfcl_eval generate --model Qwen/Qwen3-4B-Instruct-2507-FC --test-category memory_vector --result-dir tmp\\bfcl_memory_repro --allow-overwrite --include-input-log --num-threads 1",
            "python -m bfcl_eval evaluate --model Qwen/Qwen3-4B-Instruct-2507-FC --test-category memory_vector --result-dir tmp\\bfcl_memory_repro --score-dir tmp\\bfcl_memory_repro_score",
            "```",
            "",
            "Repeat with `memory_kv` and `memory_rec_sum`, then compare student and customer prereq checkpoints. To match the official row, run from commit `f7cf735` or package `bfcl-eval==2025.12.17` with the same model-serving backend and decoding settings.",
            "",
            "## Conclusion",
            "The empty student memory snapshots are caused upstream of scoring: the student prereq generations produced zero raw tool calls and zero decoded memory calls across `kv`, `vector`, and `rec_sum`. Memory tools are present in the function docs, so the immediate failure is model-generation behavior or prompt/model-serving mismatch, not missing backend APIs.",
            "",
            "The strongest next check is an isolated, single-threaded rerun with input logs at the official checkpoint. If student still emits no memory calls while customer does, compare the compiled prompts and model outputs. If official checkpoint reproduces the leaderboard values, regenerate all memory artifacts from that clean environment and discard the current affected local memory results.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=Path("memory_pipeline_audit_report.md"))
    args = parser.parse_args()

    root = args.root.resolve()
    report = build_report(root, args.model_dir)
    output = args.output if args.output.is_absolute() else root / args.output
    output.write_text(report, encoding="utf-8")
    sys.stdout.buffer.write(report.encode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
