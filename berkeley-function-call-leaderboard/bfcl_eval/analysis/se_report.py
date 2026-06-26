"""Offline analysis for the semantic-entropy measurement spike.

Reads the per-question sidecar files written by the log-only SE gate
(``<result_dir>/<model>/**/se_samples/<id>.jsonl``), labels each question correct
or not by re-running the official BFCL scorer (``agentic_checker``) on the recorded
deterministic answer against the gold variants, and reports whether semantic
entropy predicts correctness:

  * AUROC of entropy vs. incorrectness (>0.5 means higher entropy => more likely wrong)
  * mean entropy split by correct / incorrect
  * a selective-prediction table (answer only when entropy <= tau): coverage & accuracy

This is self-contained: it does NOT require running the official eval, since it
recomputes correctness from the stored answer and the gold file directly.

Usage:
    python -m bfcl_eval.analysis.se_report --model "Qwen/Qwen3-4B-Instruct-2507"
    python -m bfcl_eval.analysis.se_report --result-dir ./result --model ... --backend memory_kv
    python -m bfcl_eval.analysis.se_report --samples-dir ./result/Qwen_Qwen3-4B-Instruct-2507/se_samples
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import bfcl_eval
from bfcl_eval.constants.eval_config import RESULT_PATH
from bfcl_eval.eval_checker.agentic_eval.agentic_checker import agentic_checker

_GOLD_FILE = (
    Path(bfcl_eval.__file__).parent / "data" / "possible_answer" / "BFCL_v4_memory.json"
)

# Run ids are rewritten per backend (memory -> memory_<backend>); reverse that to
# recover the gold id, e.g. "memory_kv_1-customer-1" -> "memory_1-customer-1".
_BACKEND_PREFIX_RE = re.compile(r"^memory_(kv|vector|rec_sum)_")


def run_id_to_gold_id(run_id: str) -> str:
    return _BACKEND_PREFIX_RE.sub("memory_", run_id)


def load_gold() -> dict:
    """Return {gold_id: ground_truth_list} from the memory possible-answer file."""
    gold = {}
    with open(_GOLD_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            gold[entry["id"]] = entry["ground_truth"]
    return gold


def find_sidecar_files(args) -> list[Path]:
    if args.samples_dir:
        base = Path(args.samples_dir)
        return sorted(base.glob("*.jsonl"))
    result_dir = Path(args.result_dir) if args.result_dir else RESULT_PATH
    model_dir = result_dir / args.model.replace("/", "_")
    return sorted(model_dir.glob("**/se_samples/*.jsonl"))


def load_records(files: list[Path]) -> list[dict]:
    records = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def auroc(scores: list[float], labels: list[int]):
    """AUROC with label 1 = positive (incorrect). None if a class is empty.

    Computed as the Mann-Whitney statistic: P(score(pos) > score(neg)) with ties
    counted as 0.5.
    """
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def analyze(records: list[dict], gold: dict, backend: str | None):
    rows = []
    missing_gold = 0
    for r in records:
        if backend and r.get("test_category") != backend:
            continue
        gid = run_id_to_gold_id(r["id"])
        if gid not in gold:
            missing_gold += 1
            continue
        correct = agentic_checker(r.get("deterministic_answer", ""), gold[gid])["valid"]
        rows.append(
            {
                "id": r["id"],
                "entropy": r.get("entropy"),
                "n_clusters": r.get("n_clusters"),
                "k": r.get("k"),
                "correct": bool(correct),
            }
        )
    return rows, missing_gold


def selective_prediction_table(rows: list[dict], taus: list[float]) -> list[dict]:
    """For each tau: coverage and accuracy when answering only if entropy <= tau."""
    valid = [r for r in rows if r["entropy"] is not None and r["entropy"] != float("inf")]
    n = len(valid)
    table = []
    for tau in taus:
        covered = [r for r in valid if r["entropy"] <= tau]
        cov = len(covered) / n if n else 0.0
        acc = (sum(r["correct"] for r in covered) / len(covered)) if covered else 0.0
        table.append({"tau": tau, "coverage": cov, "accuracy_on_covered": acc})
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default=None, help="Registry model name.")
    parser.add_argument(
        "--result-dir", type=str, default=None, help="Result root (default: BFCL RESULT_PATH)."
    )
    parser.add_argument(
        "--samples-dir",
        type=str,
        default=None,
        help="Directly point at an se_samples folder (overrides --result-dir/--model).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Filter to one backend category, e.g. memory_kv.",
    )
    args = parser.parse_args()

    if not args.samples_dir and not args.model:
        parser.error("Provide either --samples-dir or --model.")

    files = find_sidecar_files(args)
    if not files:
        print("No se_samples/*.jsonl files found. Did you run with --se-gate logonly?")
        return

    gold = load_gold()
    records = load_records(files)
    rows, missing_gold = analyze(records, gold, args.backend)

    if not rows:
        print(f"Found {len(records)} records but none matched gold. missing_gold={missing_gold}")
        return

    valid = [r for r in rows if r["entropy"] is not None and r["entropy"] != float("inf")]
    n = len(valid)
    n_correct = sum(r["correct"] for r in valid)
    scores = [r["entropy"] for r in valid]
    labels = [0 if r["correct"] else 1 for r in valid]  # positive = incorrect

    ent_correct = [r["entropy"] for r in valid if r["correct"]]
    ent_wrong = [r["entropy"] for r in valid if not r["correct"]]
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")

    au = auroc(scores, labels)

    print("=" * 64)
    print(f"Semantic-entropy measurement report  (backend={args.backend or 'all'})")
    print("=" * 64)
    print(f"sidecar files            : {len(files)}")
    print(f"records (after filter)   : {n}  (skipped no-gold: {missing_gold})")
    print(f"baseline accuracy        : {n_correct}/{n} = {n_correct / n:.3f}")
    print(f"mean entropy | correct   : {mean(ent_correct):.4f}  (n={len(ent_correct)})")
    print(f"mean entropy | incorrect : {mean(ent_wrong):.4f}  (n={len(ent_wrong)})")
    print(
        f"AUROC entropy->incorrect : "
        + (f"{au:.4f}" if au is not None else "n/a (one class empty)")
    )
    print(
        "  (>0.5 => higher entropy predicts wrong answers; the spike's go/no-go signal)"
    )
    print("-" * 64)
    print("Selective prediction (answer iff entropy <= tau):")
    print(f"{'tau':>6} {'coverage':>10} {'acc@covered':>12}")
    taus = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    for row in selective_prediction_table(valid, taus):
        print(f"{row['tau']:>6.2f} {row['coverage']:>10.3f} {row['accuracy_on_covered']:>12.3f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
