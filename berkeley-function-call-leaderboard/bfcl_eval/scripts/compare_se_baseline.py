"""
Compare the semantic-entropy-gated run against the baseline and audit the gate.

Inputs (defaults match the experiment layout):
  --baseline-score  score_se_exp/baseline/<model>/agentic/memory/<backend>/..._score.json
  --gated-score     score_se_exp/gated/<model>/agentic/memory/<backend>/..._score.json
  --baseline-result / --gated-result   (to enumerate all question ids; score files only
                                        list the incorrect entries)
  --se-log          semantic_entropy_log.jsonl from the gated run

Output: a markdown report (default BFCL_SE_EXPERIMENT_REPORT.md) with per-backend
accuracy, McNemar's exact test on paired per-question outcomes, a bootstrap CI on the
accuracy delta, and a gate-decision audit.

Run:  python -m bfcl_eval.scripts.compare_se_baseline [--options]
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

BACKENDS = ["kv", "vector", "rec_sum"]
BASELINE_MODEL_DIR = "Qwen_Qwen3-4B-Instruct-2507-FC"
GATED_MODEL_DIR = "Qwen_Qwen3-4B-Instruct-2507-FC-SE"


def load_jsonl(path: Path) -> list:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def outcome_map(score_dir: Path, result_dir: Path, model_dir: str, backend: str) -> dict:
    """id -> True (correct) / False (wrong) for every recall question of one backend."""
    result_file = (
        result_dir / model_dir / "agentic" / "memory" / backend
        / f"BFCL_v4_memory_{backend}_result.json"
    )
    score_file = (
        score_dir / model_dir / "agentic" / "memory" / backend
        / f"BFCL_v4_memory_{backend}_score.json"
    )
    all_ids = {e["id"] for e in load_jsonl(result_file)}
    score_lines = load_jsonl(score_file)
    header = score_lines[0]  # {"accuracy": ..., "correct_count": ..., "total_count": ...}
    wrong_ids = {e["id"] for e in score_lines[1:]}
    outcomes = {qid: qid not in wrong_ids for qid in all_ids}
    # Sanity: the score header should agree with our reconstruction.
    if header.get("total_count") not in (None, len(outcomes)):
        print(
            f"[warn] {backend}: score total_count={header.get('total_count')} but "
            f"result file has {len(outcomes)} entries"
        )
    return outcomes


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def bootstrap_ci(paired: list, iters: int = 10000, seed: int = 0) -> tuple:
    """95% CI on accuracy delta (gated - baseline) over paired (base_ok, gated_ok)."""
    rng = random.Random(seed)
    n = len(paired)
    deltas = []
    for _ in range(iters):
        sample = [paired[rng.randrange(n)] for _ in range(n)]
        base = sum(1 for b, _ in sample if b) / n
        gated = sum(1 for _, g in sample if g) / n
        deltas.append(gated - base)
    deltas.sort()
    return deltas[int(0.025 * iters)], deltas[int(0.975 * iters)]


def audit_gate(se_log: Path) -> str:
    if not se_log.exists():
        return f"_SE log not found at `{se_log}` — skipping gate audit._\n"
    records = load_jsonl(se_log)
    lines = ["## Gate audit", ""]
    lines.append(f"Total gated steps: **{len(records)}**")

    by_phase = defaultdict(list)
    for r in records:
        by_phase["prereq" if r.get("is_prereq") else "recall"].append(r)

    for phase, recs in sorted(by_phase.items()):
        entropies = [r["entropy"] for r in recs]
        mean_h = sum(entropies) / len(entropies)
        unanimous = sum(1 for r in recs if r.get("num_clusters") == 1)
        lines.append(
            f"- **{phase}**: {len(recs)} steps | mean entropy {mean_h:.3f} bits | "
            f"unanimous {unanimous} ({unanimous / len(recs):.0%})"
        )

    decisions = Counter(r["decision"] for r in records)
    lines.append("")
    lines.append("| Decision | Count |")
    lines.append("|---|---|")
    for dec, cnt in decisions.most_common():
        lines.append(f"| {dec} | {cnt} |")

    destructive_proposed = [r for r in records if r.get("op_class") == "destructive"]
    blocked = [r for r in records if r.get("fallback")]
    forced = [r for r in records if r.get("forced_destructive")]
    lines.append("")
    lines.append(
        f"Destructive/overwrite majority proposed: **{len(destructive_proposed)}** destructive, "
        f"**{sum(1 for r in records if r.get('op_class') == 'overwrite')}** overwrite. "
        f"Blocked with safe fallback: **{len(blocked)}**. Forced through (no safe cluster): "
        f"**{len(forced)}**."
    )
    if blocked:
        lines.append("")
        lines.append("### Blocked destructive/overwrite steps (fallback actions)")
        lines.append("")
        lines.append("| Test id | H | m | Majority proposed | Fallback executed |")
        lines.append("|---|---|---|---|---|")
        for r in blocked:
            majority_names = next(
                (c["func_names"] for c in r.get("candidates", []) if c["cluster"] == 0),
                [],
            )
            lines.append(
                f"| {r.get('test_id')} | {r['entropy']} | {r['majority_fraction']} | "
                f"{', '.join(majority_names) or '?'} | "
                f"{', '.join(r.get('chosen_calls') or ['(text)'])[:80]} |"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--baseline-result", type=Path, default=root / "result_se_exp" / "baseline")
    parser.add_argument("--baseline-score", type=Path, default=root / "score_se_exp" / "baseline")
    parser.add_argument("--gated-result", type=Path, default=root / "result_se_exp" / "gated")
    parser.add_argument("--gated-score", type=Path, default=root / "score_se_exp" / "gated")
    parser.add_argument("--se-log", type=Path, default=None,
                        help="defaults to <gated-result>/semantic_entropy_log.jsonl")
    parser.add_argument("--output", type=Path, default=root / "BFCL_SE_EXPERIMENT_REPORT.md")
    args = parser.parse_args()

    se_log = args.se_log or (args.gated_result / "semantic_entropy_log.jsonl")

    report = [
        "# Semantic-Entropy Gate vs Baseline — BFCL V4 Memory",
        "",
        "Baseline: `Qwen/Qwen3-4B-Instruct-2507-FC` (greedy, temperature 0.001). "
        "Gated: `Qwen/Qwen3-4B-Instruct-2507-FC-SE` (N samples per memory step, semantic "
        "clustering, majority commit, destructive ops gated on cluster entropy).",
        "",
        "## Accuracy",
        "",
        "| Backend | Baseline | Gated | Delta | Discordant (b/c) | McNemar p | 95% CI (delta) |",
        "|---|---|---|---|---|---|---|",
    ]

    total_paired = []
    for backend in BACKENDS:
        try:
            base = outcome_map(args.baseline_score, args.baseline_result, BASELINE_MODEL_DIR, backend)
            gated = outcome_map(args.gated_score, args.gated_result, GATED_MODEL_DIR, backend)
        except FileNotFoundError as e:
            report.append(f"| {backend} | — | — | — | — | — | missing: {e.filename} |")
            continue
        common = sorted(set(base) & set(gated))
        if set(base) != set(gated):
            print(f"[warn] {backend}: id sets differ (base {len(base)}, gated {len(gated)}, common {len(common)})")
        paired = [(base[q], gated[q]) for q in common]
        total_paired.extend(paired)
        n = len(paired)
        base_acc = sum(1 for b, _ in paired if b) / n
        gated_acc = sum(1 for _, g in paired if g) / n
        b_disc = sum(1 for b, g in paired if b and not g)
        c_disc = sum(1 for b, g in paired if not b and g)
        p = mcnemar_exact(b_disc, c_disc)
        lo, hi = bootstrap_ci(paired)
        report.append(
            f"| {backend} | {base_acc:.2%} ({sum(1 for b, _ in paired if b)}/{n}) "
            f"| {gated_acc:.2%} ({sum(1 for _, g in paired if g)}/{n}) "
            f"| {gated_acc - base_acc:+.2%} | {b_disc}/{c_disc} | {p:.4f} | [{lo:+.2%}, {hi:+.2%}] |"
        )

    if total_paired:
        n = len(total_paired)
        base_acc = sum(1 for b, _ in total_paired if b) / n
        gated_acc = sum(1 for _, g in total_paired if g) / n
        b_disc = sum(1 for b, g in total_paired if b and not g)
        c_disc = sum(1 for b, g in total_paired if not b and g)
        p = mcnemar_exact(b_disc, c_disc)
        lo, hi = bootstrap_ci(total_paired)
        report.append(
            f"| **overall** | {base_acc:.2%} | {gated_acc:.2%} | {gated_acc - base_acc:+.2%} "
            f"| {b_disc}/{c_disc} | {p:.4f} | [{lo:+.2%}, {hi:+.2%}] |"
        )

    report.append("")
    report.append(
        "Discordant b/c = baseline-correct-gated-wrong / baseline-wrong-gated-correct. "
        "McNemar is an exact two-sided binomial test on the discordant pairs."
    )
    report.append("")
    report.append(audit_gate(se_log))

    args.output.write_text("\n".join(report), encoding="utf-8")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
