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
import ast
import json
import re
from pathlib import Path

import bfcl_eval
from bfcl_eval.constants.eval_config import RESULT_PATH
from bfcl_eval.eval_checker.agentic_eval.agentic_checker import agentic_checker
from bfcl_eval.model_handler.memory_se_gate import semantic_entropy

_GOLD_FILE = (
    Path(bfcl_eval.__file__).parent / "data" / "possible_answer" / "BFCL_v4_memory.json"
)

# Run ids are rewritten per backend (memory -> memory_<backend>); reverse that to
# recover the gold id, e.g. "memory_kv_1-customer-1" -> "memory_1-customer-1".
_BACKEND_PREFIX_RE = re.compile(r"^memory_(kv|vector|rec_sum)_")


def run_id_to_gold_id(run_id: str) -> str:
    return _BACKEND_PREFIX_RE.sub("memory_", run_id)


# The agentic memory format asks the model to answer as {"answer": ..., "context": ...}.
# Cluster on the "answer" field (the meaningful content) rather than the full blob,
# whose verbose "context" reasoning dilutes the semantic-entropy signal.
_ANSWER_RE = re.compile(r"""['"]answer['"]\s*:\s*['"](.*?)['"]""", re.DOTALL)


def extract_answer(text: str) -> str:
    """Best-effort pull of the 'answer' field; fall back to the raw text."""
    if not isinstance(text, str):
        text = str(text)
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1)
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(text)
            if isinstance(obj, dict) and "answer" in obj:
                return str(obj["answer"])
        except Exception:
            pass
    return text


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
        # entropy clustered on the extracted "answer" field, recomputed offline
        # from the stored raw samples (sharper than the full-blob entropy logged
        # at generation time).
        samples = r.get("samples") or []
        if samples:
            ent_ans, ncl_ans, _ = semantic_entropy([extract_answer(s) for s in samples])
        else:
            ent_ans, ncl_ans = float("inf"), 0
        rows.append(
            {
                "id": r["id"],
                "entropy_full": r.get("entropy"),  # logged at gen time (full blob)
                "entropy_answer": ent_ans,  # recomputed on the answer field
                "n_clusters_full": r.get("n_clusters"),
                "n_clusters_answer": ncl_ans,
                "k": r.get("k"),
                "correct": bool(correct),
            }
        )
    return rows, missing_gold


def selective_prediction_table(rows: list[dict], key: str, taus: list[float]) -> list[dict]:
    """For each tau: coverage and accuracy when answering only if entropy <= tau."""
    valid = [r for r in rows if r[key] is not None and r[key] != float("inf")]
    n = len(valid)
    table = []
    for tau in taus:
        covered = [r for r in valid if r[key] <= tau]
        cov = len(covered) / n if n else 0.0
        acc = (sum(r["correct"] for r in covered) / len(covered)) if covered else 0.0
        table.append({"tau": tau, "coverage": cov, "accuracy_on_covered": acc})
    return table


def _report_one(rows: list[dict], key: str, label: str):
    """Print AUROC + mean-by-correctness + selective table for one entropy variant."""
    valid = [r for r in rows if r[key] is not None and r[key] != float("inf")]
    n = len(valid)
    if not n:
        print(f"\n[{label}] no valid entropy values.")
        return
    scores = [r[key] for r in valid]
    labels = [0 if r["correct"] else 1 for r in valid]  # positive = incorrect
    ent_correct = [r[key] for r in valid if r["correct"]]
    ent_wrong = [r[key] for r in valid if not r["correct"]]
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    au = auroc(scores, labels)
    n_nonzero = sum(1 for s in scores if s not in (0.0, -0.0))
    print(f"\n--- {label} ---")
    print(f"  AUROC entropy->incorrect : " + (f"{au:.4f}" if au is not None else "n/a"))
    print(f"  entropy>0 count          : {n_nonzero}/{n}")
    print(f"  mean entropy | correct   : {mean(ent_correct):.4f}  (n={len(ent_correct)})")
    print(f"  mean entropy | incorrect : {mean(ent_wrong):.4f}  (n={len(ent_wrong)})")
    taus = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    print(f"  selective prediction (answer iff entropy <= tau):")
    print(f"    {'tau':>6} {'coverage':>10} {'acc@covered':>12}")
    for row in selective_prediction_table(valid, key, taus):
        print(f"    {row['tau']:>6.2f} {row['coverage']:>10.3f} {row['accuracy_on_covered']:>12.3f}")


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

    n = len(rows)
    n_correct = sum(r["correct"] for r in rows)

    print("=" * 64)
    print(f"Semantic-entropy measurement report  (backend={args.backend or 'all'})")
    print("=" * 64)
    print(f"sidecar files            : {len(files)}")
    print(f"records (after filter)   : {n}  (skipped no-gold: {missing_gold})")
    print(f"baseline accuracy        : {n_correct}/{n} = {n_correct / n:.3f}")
    print("AUROC > 0.5 => higher entropy predicts wrong answers (the go/no-go signal)")

    # Two clustering variants from the same samples:
    #  - full   : entropy logged at gen time on the whole {answer, context} blob
    #  - answer : recomputed here on just the extracted "answer" field (sharper)
    _report_one(rows, "entropy_full", "cluster on FULL response (logged at gen time)")
    _report_one(rows, "entropy_answer", "cluster on ANSWER field (recomputed offline)")
    print("=" * 64)


if __name__ == "__main__":
    main()
