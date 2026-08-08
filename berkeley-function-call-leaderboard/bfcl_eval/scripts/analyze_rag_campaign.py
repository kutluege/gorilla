"""One-command post-campaign analysis for RAG-program campaigns.

Runs, in order, against a finished campaign manifest:
  1. analyzer, student-INCLUDED (headline)   -> analysis_student_incl.json
  2. analyzer, student-excluded (sensitivity)-> analysis_student_excl.json
  3. strict answer-field regrade per arm     -> strict_regrade.json
  4. step-budget guard per arm               -> step_budget.json
  5. by-tier answerability per (rep, arm)    -> answerability.json
  6. per-replicate accuracy direction table  -> per_replicate.json
  7. cost accounting (durations + rag_log)   -> cost.json

Refuses --allow-unscored: a campaign with unscored cells is not analyzable.

    python bfcl_eval/scripts/analyze_rag_campaign.py \
        --manifest result_hnav_rag_c1/manifest.jsonl \
        --out-dir gov_logs/hnav_rag_c1/analysis
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

PY = sys.executable


def sh(args_list):
    print("+", " ".join(str(a) for a in args_list))
    r = subprocess.run([str(a) for a in args_list], cwd=REPO_ROOT)
    if r.returncode != 0:
        raise SystemExit(f"step failed: {args_list}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--reference-arm", default="baseline")
    args = ap.parse_args()

    manifest = REPO_ROOT / args.manifest
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(manifest, encoding="utf-8")]
    rs = next(r for r in rows if r["event"] == "run_start")
    arms = {a["label"]: a["model"] for a in rs["arms"]}
    result_root = rs["result_root"]
    score_root = rs["score_root"]
    reps = sorted({r["replicate"] for r in rows
                   if r["event"] == "cmd_end" and r["phase"] == "evaluate"
                   and r["exit_code"] == 0})

    scripts = Path(__file__).parent

    # 1+2. paired analysis, both denominators
    sh([PY, scripts / "analyze_gov_replicates.py", "--manifest", args.manifest,
        "--reference-arm", args.reference_arm, "--include-student",
        "--out", out_dir / "analysis_student_incl.json"])
    sh([PY, scripts / "analyze_gov_replicates.py", "--manifest", args.manifest,
        "--reference-arm", args.reference_arm,
        "--out", out_dir / "analysis_student_excl.json"])

    # 3+4. strict regrade + step guard over every arm's result dirs
    slug = lambda m: m.replace("/", "_")
    all_dirs = [f"{result_root}/rep{rep:02d}/{arm}/{slug(model)}"
                for rep in reps for arm, model in arms.items()]
    sh([PY, scripts / "strict_regrade.py", "--result-dirs", *all_dirs,
        "--out", out_dir / "strict_regrade.json"])
    sh([PY, scripts / "step_budget_guard.py", "--result-dirs", *all_dirs,
        "--out", out_dir / "step_budget.json"])

    # 5. by-tier answerability per (rep, arm)
    ans_out = out_dir / "answerability.json"
    if ans_out.exists():
        ans_out.unlink()
    for rep in reps:
        for arm, model in arms.items():
            sh([PY, scripts / "hnav_answer_index.py", "--validate",
                "--arm", f"rep{rep:02d}_{arm}",
                "--result-dir",
                f"{result_root}/rep{rep:02d}/{arm}/{slug(model)}",
                "--score-dir",
                f"{score_root}/rep{rep:02d}/{arm}/{slug(model)}",
                "--out", ans_out, "--append"])

    # 6. per-replicate accuracy direction table (score-file headers)
    per_rep = defaultdict(dict)
    for rep in reps:
        for arm, model in arms.items():
            for backend in ("kv", "vector"):
                f = (REPO_ROOT / score_root / f"rep{rep:02d}" / arm /
                     slug(model) / "agentic" / "memory" / backend /
                     f"BFCL_v4_memory_{backend}_score.json")
                if not f.exists():
                    continue
                head = json.loads(open(f, encoding="utf-8").readline())
                per_rep[f"rep{rep:02d}"][f"{arm}|{backend}"] = {
                    "accuracy": head.get("accuracy"),
                    "correct": head.get("correct_count"),
                    "n": head.get("total_count")}
    (out_dir / "per_replicate.json").write_text(
        json.dumps(per_rep, indent=2, sort_keys=True), encoding="utf-8")

    # 7. cost accounting: wall clock per (arm, phase) + rag_log volumes
    dur = defaultdict(list)
    for r in rows:
        if r["event"] == "cmd_end":
            dur[f"{r['arm']}|{r['phase']}"].append(round(r["duration_s"] / 60, 1))
    gov_log_root = rs.get("gov_log_root", "")
    rag_counts = {}
    for rep in reps:
        for arm in arms:
            p = (REPO_ROOT / gov_log_root / f"rep{rep:02d}_{arm}" /
                 "rag_log.jsonl")
            if p.exists():
                events = defaultdict(int)
                for line in open(p, encoding="utf-8"):
                    events[json.loads(line).get("event", "?")] += 1
                rag_counts[f"rep{rep:02d}_{arm}"] = dict(events)
    (out_dir / "cost.json").write_text(json.dumps(
        {"generate_minutes_per_arm": {k: v for k, v in dur.items()
                                      if k.endswith("generate")},
         "rag_log_event_counts": rag_counts}, indent=2, sort_keys=True),
        encoding="utf-8")

    print(f"[analyze-rag] all artifacts in {out_dir}")


if __name__ == "__main__":
    main()
