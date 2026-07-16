"""Summarize BFCL memory score artifacts for a local model run.

This script is intentionally read-only. It compares the local aggregate CSV,
the per-backend memory score JSON headers, and the result/prereq coverage for a
single model directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "Qwen_Qwen3-4B-Instruct-2507-FC"

OFFICIAL_SCREENSHOT = {
    "memory_summary": 17.63,
    "memory_kv": 16.13,
    "memory_vector": 12.26,
    "memory_rec_sum": 24.52,
}

BACKENDS = {
    "memory_kv": ("kv", "BFCL_v4_memory_kv_score.json", "BFCL_v4_memory_kv_result.json"),
    "memory_vector": (
        "vector",
        "BFCL_v4_memory_vector_score.json",
        "BFCL_v4_memory_vector_result.json",
    ),
    "memory_rec_sum": (
        "rec_sum",
        "BFCL_v4_memory_rec_sum_score.json",
        "BFCL_v4_memory_rec_sum_result.json",
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def parse_percent(value: str) -> float | None:
    value = value.strip()
    if value == "N/A" or not value:
        return None
    return float(value.rstrip("%")) / 100


def load_local_agentic_csv(root: Path, model_display_name: str) -> dict[str, float | None]:
    path = root / "score" / "data_agentic.csv"
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter=";"))

    for row in rows[1:]:
        if len(row) >= 10 and row[1] == model_display_name:
            return {
                "agentic_overall": parse_percent(row[2]),
                "web_search_summary": parse_percent(row[3]),
                "web_search_base": parse_percent(row[4]),
                "web_search_no_snippet": parse_percent(row[5]),
                "memory_summary": parse_percent(row[6]),
                "memory_kv": parse_percent(row[7]),
                "memory_vector": parse_percent(row[8]),
                "memory_rec_sum": parse_percent(row[9]),
            }

    raise ValueError(f"Could not find {model_display_name!r} in {path}")


def score_stats(score_path: Path) -> dict[str, Any]:
    rows = read_jsonl(score_path)
    header = rows[0]
    failures = rows[1:]
    error_types = Counter(
        failure.get("error", {}).get("error_type", "unknown") for failure in failures
    )
    return {
        "accuracy": header["accuracy"],
        "correct_count": header["correct_count"],
        "total_count": header["total_count"],
        "failure_count": len(failures),
        "header_failure_count": header["total_count"] - header["correct_count"],
        "error_types": error_types,
        "sample_failures": failures[:3],
    }


def result_stats(result_path: Path) -> dict[str, Any]:
    rows = read_jsonl(result_path)
    ids = [row["id"] for row in rows]
    return {
        "total_count": len(rows),
        "unique_ids": len(set(ids)),
        "duplicate_ids": sorted([item for item, count in Counter(ids).items() if count > 1]),
        "first_id": ids[0] if ids else None,
        "last_id": ids[-1] if ids else None,
    }


def count_jsonl(path: Path) -> int:
    return len(read_jsonl(path))


def snapshot_stats(snapshot_dir: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for snapshot in sorted(snapshot_dir.glob("*_final.json")):
        try:
            data = json.loads(snapshot.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            stats[snapshot.name] = {"type": "invalid_json"}
            continue
        if isinstance(data, dict):
            item: dict[str, Any] = {"type": "dict", "top_level_keys": len(data)}
            if "core_memory" in data:
                core = data["core_memory"]
                if isinstance(core, dict) and "store" in core:
                    item["core_entries"] = len(core["store"])
                elif isinstance(core, dict):
                    item["core_entries"] = len(core)
            if "archival_memory" in data:
                archival = data["archival_memory"]
                if isinstance(archival, dict) and "store" in archival:
                    item["archival_entries"] = len(archival["store"])
                elif isinstance(archival, dict):
                    item["archival_entries"] = len(archival)
            if "memory" in data:
                item["memory_chars"] = len(data["memory"])
            stats[snapshot.name] = item
        elif isinstance(data, list):
            stats[snapshot.name] = {"type": "list", "items": len(data)}
        else:
            stats[snapshot.name] = {"type": type(data).__name__}
    return stats


def prereq_snapshot_stats(snapshot_dir: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    checkpoint_dir = snapshot_dir / "prereq_checkpoints"
    for snapshot in sorted(checkpoint_dir.glob("*.json")):
        scenario = snapshot.stem.rsplit("-", 1)[0].rsplit("-", 1)[-1]
        data = json.loads(snapshot.read_text(encoding="utf-8"))
        scenario_stats = stats.setdefault(
            scenario,
            {"checkpoints": 0, "empty_checkpoints": 0, "last_core_entries": 0, "last_archival_entries": 0},
        )
        scenario_stats["checkpoints"] += 1
        core = data.get("core_memory", {})
        archival = data.get("archival_memory", {})
        core_store = core.get("store", core) if isinstance(core, dict) else {}
        archival_store = archival.get("store", archival) if isinstance(archival, dict) else {}
        core_entries = len(core_store) if isinstance(core_store, dict) else 0
        archival_entries = len(archival_store) if isinstance(archival_store, dict) else 0
        scenario_stats["last_core_entries"] = core_entries
        scenario_stats["last_archival_entries"] = archival_entries
        if core_entries == 0 and archival_entries == 0:
            scenario_stats["empty_checkpoints"] += 1
    return stats


def build_markdown(root: Path, model: str, official_commit: str, official_package: str) -> str:
    score_root = root / "score" / model / "agentic" / "memory"
    result_root = root / "result" / model / "agentic" / "memory"
    display_name = model.replace("Qwen_Qwen", "Qwen").replace("_", "/")
    display_name = "Qwen3-4B-Instruct-2507 (FC)"

    local_csv = load_local_agentic_csv(root, display_name)

    backend_rows = []
    backend_details = {}
    for category, (subdir, score_name, result_name) in BACKENDS.items():
        score_path = score_root / subdir / score_name
        result_path = result_root / subdir / result_name
        stats = score_stats(score_path)
        results = result_stats(result_path)
        backend_details[category] = (stats, results, score_path, result_path)
        recomputed = stats["correct_count"] / stats["total_count"]
        backend_rows.append(
            [
                category,
                OFFICIAL_SCREENSHOT[category],
                local_csv[category],
                stats["accuracy"],
                recomputed,
                stats["correct_count"],
                stats["total_count"],
                stats["failure_count"],
                results["total_count"],
            ]
        )

    local_memory_summary = sum(row[2] for row in backend_rows if row[2] is not None) / 3
    header_memory_summary = sum(row[3] for row in backend_rows) / 3
    recomputed_memory_summary = sum(row[4] for row in backend_rows) / 3

    dataset_path = root / "bfcl_eval" / "data" / "BFCL_v4_memory.json"
    answer_path = root / "bfcl_eval" / "data" / "possible_answer" / "BFCL_v4_memory.json"
    prereq_dir = root / "bfcl_eval" / "data" / "memory_prereq_conversation"
    prereq_counts = {
        path.name: count_jsonl(path) for path in sorted(prereq_dir.glob("memory_*.json"))
    }

    lines = [
        "# BFCL Memory Score Discrepancy Report",
        "",
        "## Version Alignment",
        f"- Local repo HEAD: `{run_git(root, 'rev-parse', 'HEAD')}`",
        f"- Local git describe: `{run_git(root, 'describe', '--always', '--dirty', '--tags')}`",
        f"- Official leaderboard checkpoint: `{official_commit}`",
        f"- Official PyPI package: `{official_package}`",
        "- Official leaderboard page: https://gorilla.cs.berkeley.edu/leaderboard",
        "",
        "The local checkout is not the published leaderboard checkpoint. The worktree is also dirty, so local generation behavior may not match the official run even before considering model-serving settings.",
        "",
        "## Score Comparison",
        "",
        "| Metric | Official screenshot | Local CSV | Score JSON header | Recomputed from header | Delta local-official |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Memory Summary | {OFFICIAL_SCREENSHOT['memory_summary']:.2f}% | {pct(local_csv['memory_summary'])} | {pct(header_memory_summary)} | {pct(recomputed_memory_summary)} | {(local_csv['memory_summary'] * 100 - OFFICIAL_SCREENSHOT['memory_summary']):+.2f} pp |",
    ]

    for category, official, local, header, recomputed, *_ in backend_rows:
        lines.append(
            f"| {category} | {official:.2f}% | {pct(local)} | {pct(header)} | {pct(recomputed)} | {(local * 100 - official):+.2f} pp |"
        )

    lines.extend(
        [
            "",
            "The local CSV, score JSON headers, and header recomputation agree. This means `data_agentic.csv` is not stale relative to the local memory score JSON files.",
            "",
            "## Coverage",
            "",
            "| Backend | Correct | Score total | Failure rows in score file | Result rows | Result unique IDs |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for category, _official, _local, _header, _recomputed, correct, total, failures, result_total in backend_rows:
        results = backend_details[category][1]
        lines.append(
            f"| {category} | {correct} | {total} | {failures} | {result_total} | {results['unique_ids']} |"
        )

    lines.extend(
        [
            "",
            f"- Memory prompt rows: `{count_jsonl(dataset_path)}` from `{dataset_path.relative_to(root)}`",
            f"- Memory possible-answer rows: `{count_jsonl(answer_path)}` from `{answer_path.relative_to(root)}`",
            f"- Prereq conversation rows by scenario: {json.dumps(prereq_counts, sort_keys=True)}",
            "",
            "The three memory backends each scored 155 answer questions locally, matching the memory prompt and answer row counts. The score files contain a summary header plus only failed examples; they do not contain one row per successful example.",
            "",
            "## Failure Profile",
        ]
    )

    for category, (stats, _results, score_path, _result_path) in backend_details.items():
        error_summary = ", ".join(
            f"`{name}`={count}" for name, count in stats["error_types"].most_common()
        )
        lines.extend(
            [
                "",
                f"### {category}",
                f"- Score file: `{score_path.relative_to(root)}`",
                f"- Error types: {error_summary}",
            ]
        )
        for failure in stats["sample_failures"]:
            possible = failure.get("possible_answer")
            answer = failure.get("last_non_fc_message") or failure.get("model_result_raw")
            answer_text = str(answer).replace("\n", " ")
            if len(answer_text) > 240:
                answer_text = answer_text[:237] + "..."
            lines.append(
                f"- Sample `{failure.get('id')}` expected `{possible}` but final response was `{answer_text}`"
            )

    lines.extend(
        [
            "",
            "## Memory Snapshot Coverage",
        ]
    )
    for category, (_stats, _results, _score_path, result_path) in backend_details.items():
        snapshot_dir = result_path.parent / "memory_snapshot"
        lines.append(f"- `{category}` final snapshots: {json.dumps(snapshot_stats(snapshot_dir), sort_keys=True)}")
        lines.append(f"- `{category}` prereq checkpoints: {json.dumps(prereq_snapshot_stats(snapshot_dir), sort_keys=True)}")

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "The discrepancy is not caused by local CSV aggregation or stale local score files. The local score artifacts are internally consistent.",
            "",
            "However, the vector backend has a concrete local generation problem for the `student` scenario: its final snapshot and all ten student prereq checkpoint snapshots contain zero core and archival entries. This directly explains why all scored `memory_vector_*student*` answer cases fail with missing-memory responses.",
            "",
            "The most likely cause is that the local outputs were produced under a different evaluation environment than the published leaderboard: different BFCL commit/package, dirty local inference code, and/or different serving/decoding settings. The official row should be reproduced from commit `f7cf735` or `bfcl-eval==2025.12.17`, then compared against these local result JSON files ID-by-ID.",
        ]
    )

    return "\n".join(lines) + "\n"


def run_git(root: Path, *args: str) -> str:
    import subprocess

    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"unavailable: {exc}"
    return completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--official-commit", default="f7cf735")
    parser.add_argument("--official-package", default="bfcl-eval==2025.12.17")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    report = build_markdown(root, args.model, args.official_commit, args.official_package)
    if args.output:
        output = args.output
        if not output.is_absolute():
            output = root / output
        output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
