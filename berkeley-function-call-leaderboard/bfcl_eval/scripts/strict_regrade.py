"""Strict answer-field-only regrade (RAG program, pre-registered guard).

The production grader (`agentic_checker`) word-boundary-matches gold inside
the WHOLE final message. Question entries are instructed to answer as
``{'answer': ..., 'context': ...}``; gold that appears only in the `context`
field (or in surrounding prose) still passes. Any intervention that
increases the amount of verbatim memory text in the final message therefore
mechanically inflates the lenient metric.

This tool re-grades every question entry using ONLY the parsed `answer`
field. Pre-registered rule: an arm that improves the lenient metric but not
the strict metric is reported as a GRADER ARTIFACT, not a method.

Parsing: best-effort extraction of the answer value from the final message
(dict-literal or JSON with an 'answer' key, quoted or unquoted value). A
final message with no parseable answer field scores strict-incorrect unless
the whole-message match ALSO fails (then it is plain incorrect either way).

    python bfcl_eval/scripts/strict_regrade.py \
        --result-dirs "result_hnav_stage4/rep0*/baseline/Qwen_Qwen3-4B-Instruct-2507-FC" \
        --out gov_logs/hnav_autonomous/strict_regrade_baseline.json
"""

import argparse
import glob as globmod
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from bfcl_eval.eval_checker.agentic_eval.agentic_checker import (  # noqa: E402
    agentic_checker,
)
from hnav_answer_index import gold_index  # noqa: E402

# 'answer': <value>  -- value = quoted string (any quote) or bare token run
_ANSWER_RE = re.compile(
    r"""['"]answer['"]\s*:\s*(?:(['"])(?P<q>.*?)\1|(?P<bare>[^,}\n]+))""",
    re.S)


def extract_answer(final_message: str):
    m = _ANSWER_RE.search(final_message or "")
    if not m:
        return None
    return (m.group("q") if m.group("q") is not None
            else (m.group("bare") or "").strip())


def grade_dir(result_dir: Path, backend: str):
    f = (result_dir / "agentic" / "memory" / backend /
         f"BFCL_v4_memory_{backend}_result.json")
    gidx = gold_index()
    rows = {"n": 0, "lenient": 0, "strict": 0, "answer_field_missing": 0,
            "context_only_passes": []}
    if not f.exists():
        return None
    for line in open(f, encoding="utf-8"):
        rec = json.loads(line)
        qid = rec["id"]
        gold = gidx.get(qid.replace(f"memory_{backend}_", "memory_"))
        if not gold:
            continue
        rows["n"] += 1
        final = str((rec.get("result") or [[""]])[-1][-1])
        lenient = bool(agentic_checker(final, list(gold)).get("valid"))
        ans = extract_answer(final)
        if ans is None:
            rows["answer_field_missing"] += 1
            strict = False
        else:
            strict = bool(agentic_checker(ans, list(gold)).get("valid"))
        rows["lenient"] += int(lenient)
        rows["strict"] += int(strict)
        if lenient and not strict:
            rows["context_only_passes"].append(qid)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dirs", nargs="+", required=True)
    ap.add_argument("--backends", default="kv,vector")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dirs = sorted(set(p for g in args.result_dirs for p in globmod.glob(g)))
    report = {"dirs": {}}
    agg = defaultdict(lambda: [0, 0, 0])
    for d in dirs:
        per = {}
        for backend in args.backends.split(","):
            r = grade_dir(Path(d), backend)
            if r is None:
                continue
            per[backend] = r
            agg[backend][0] += r["n"]
            agg[backend][1] += r["lenient"]
            agg[backend][2] += r["strict"]
        report["dirs"][str(d)] = per
    report["pooled"] = {
        b: {"n": n, "lenient": l, "strict": s,
            "lenient_rate": round(l / n, 4) if n else None,
            "strict_rate": round(s / n, 4) if n else None,
            "gap": round((l - s) / n, 4) if n else None}
        for b, (n, l, s) in agg.items()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["pooled"], indent=2))
    print(f"[strict-regrade] wrote {out}")


if __name__ == "__main__":
    main()
