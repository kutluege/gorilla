"""Repository-wide runtime leakage audit (directive §23, generalised).

Every RUNTIME module -- code importable while the agent is answering a
benchmark entry -- must be unable to see evaluation-only information:
gold answers, outcome labels, the production grader, or the offline
answerability oracle. The audit is AST-based (names, attributes, imports,
string literals for data paths) and FAILS the suite on any hit, per §23:
"Add a leakage audit that fails if runtime code accesses fields such as
benchmark_correct, outcome_label, gold answers, or prerequisite gold
mappings."

Offline analysis scripts (bfcl_eval/scripts/*) are exempt: they are
evaluation-side by definition. The runtime surface is the handler +
middleware tree.

Run:  python bfcl_eval/scripts/test_leakage_audit.py
"""

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Every module that executes during inference. New handlers/middleware MUST
# be added here; the wildcard sweep below catches forgotten ones.
RUNTIME_MODULES = [
    "bfcl_eval/model_handler/local_inference/qwen_gov.py",
    "bfcl_eval/model_handler/local_inference/qwen_hact.py",
    "bfcl_eval/model_handler/local_inference/qwen_scaffold.py",
    "bfcl_eval/model_handler/local_inference/qwen_se.py",
    "bfcl_eval/model_handler/local_inference/qwen_mig.py",
]
RUNTIME_GLOBS = [
    "bfcl_eval/model_handler/middleware/*.py",
    "bfcl_eval/model_handler/local_inference/qwen_rag*.py",  # RAG program
]

# Names/attributes/imports that mean "evaluation-only information".
FORBIDDEN_NAMES = {
    "possible_answer", "ground_truth", "agentic_checker",
    "hnav_answer_index", "scenario_questions", "gold_index",
    "correct_ids", "carries", "benchmark_correct", "outcome_label",
    "label_outcomes", "label_outcomes_hnav", "must_write", "must_suppress",
}
# Substrings that may not appear in any string literal (data-path leakage).
FORBIDDEN_LITERAL_SUBSTRINGS = [
    "possible_answer", "ground_truth", "BFCL_v4_memory.json",
]
# memory_prereq_conversation is the *conversation* the agent legitimately
# sees turn-by-turn; reading the FILE at runtime would still be lookahead:
FORBIDDEN_LITERAL_SUBSTRINGS.append("memory_prereq_conversation")

PASS = FAIL = 0


def check(label, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL {label}: {detail}")


def audit_file(path: Path):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = set()
    literals = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            for a in node.names:
                names.add(a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            literals.append(node.value)
    hits = FORBIDDEN_NAMES & names
    check(f"{path.name}: no eval-only names/imports", not hits, sorted(hits))
    lit_hits = sorted({sub for sub in FORBIDDEN_LITERAL_SUBSTRINGS
                       for lit in literals if sub in lit})
    check(f"{path.name}: no eval-data path literals", not lit_hits, lit_hits)


def main():
    files = [REPO_ROOT / m for m in RUNTIME_MODULES]
    for g in RUNTIME_GLOBS:
        files.extend(sorted(REPO_ROOT.glob(g)))
    seen = set()
    for f in files:
        if f in seen:
            continue
        seen.add(f)
        if not f.exists():
            continue
        print(f"[audit] {f.relative_to(REPO_ROOT)}")
        audit_file(f)
    print(f"\n{PASS} passed, {FAIL} failed  ({len(seen)} modules audited)")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
