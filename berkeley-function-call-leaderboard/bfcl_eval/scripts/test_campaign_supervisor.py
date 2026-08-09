"""Offline tests for campaign_supervisor.py (pure helpers only -- no
processes, no network). Run:
    python bfcl_eval/scripts/test_campaign_supervisor.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from campaign_supervisor import (  # noqa: E402
    clean_for_resume,
    completed_cells,
    last_stop_reason,
    resume_point,
)

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


LABELS = ["baseline", "read_verbatim", "read_irrelevant", "read_instruction"]


def write_manifest(rows):
    f = Path(tempfile.mkdtemp(prefix="sup_")) / "manifest.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                 encoding="utf-8")
    return f


def cell_rows(rep, arm, inf_errs=0, eval_ok=True, gen_exit=0):
    return [
        {"event": "cmd_end", "replicate": rep, "arm": arm,
         "phase": "generate", "exit_code": gen_exit,
         "inference_errors": inf_errs},
        {"event": "cmd_end", "replicate": rep, "arm": arm,
         "phase": "evaluate", "exit_code": 0,
         "score_files": [{"path": "x", "exists": eval_ok}]},
    ]


def test_completed_cells():
    rows = (cell_rows(1, "baseline")
            + cell_rows(1, "read_verbatim", inf_errs=3)     # contaminated
            + cell_rows(2, "baseline", eval_ok=False)       # missing score
            + cell_rows(2, "read_verbatim", gen_exit=1))    # failed gen
    m = write_manifest(rows)
    done = completed_cells(m)
    check("clean cell counted", (1, "baseline") in done)
    check("contaminated generate excluded", (1, "read_verbatim") not in done)
    check("missing score excluded", (2, "baseline") not in done)
    check("failed generate excluded", (2, "read_verbatim") not in done)
    check("missing manifest -> empty",
          completed_cells(Path("does/not/exist.jsonl")) == set())


def test_resume_point_arm_mode():
    done = {(1, "baseline"), (1, "read_verbatim")}
    check("arm mode resumes at first incomplete arm, keeping siblings",
          resume_point(5, LABELS, done, "arm") == (1, "read_irrelevant"))
    check("arm mode at a replicate boundary needs no start-arm",
          resume_point(5, LABELS, {(1, a) for a in LABELS}, "arm")
          == (2, None))
    all_done = {(r, a) for r in range(1, 6) for a in LABELS}
    check("complete campaign -> None",
          resume_point(5, LABELS, all_done, "arm") is None)


def test_resume_point_replicate_mode():
    done = {(1, "baseline"), (1, "read_verbatim")}
    check("replicate mode restarts the whole interrupted replicate",
          resume_point(5, LABELS, done, "replicate") == (1, None))
    done2 = {(1, a) for a in LABELS} | {(2, "baseline")}
    check("later replicate restarts from its own start",
          resume_point(5, LABELS, done2, "replicate") == (2, None))


def test_last_stop_reason():
    marker = Path(tempfile.mkdtemp(prefix="sup_")) / "killed.marker"
    m = write_manifest([{"event": "error", "stage": "tunnel_probe"}])
    check("tunnel_probe error -> outage",
          last_stop_reason(m, marker) == "outage")
    m2 = write_manifest([{"event": "error", "stage": "inference_errors"}])
    check("inference_errors -> outage",
          last_stop_reason(m2, marker) == "outage")
    m3 = write_manifest([{"event": "cmd_end", "replicate": 1,
                          "arm": "baseline", "phase": "generate",
                          "exit_code": 0}])
    check("no error events -> unknown (conservative replicate restart)",
          last_stop_reason(m3, marker) == "unknown")
    marker.write_text("wedge", encoding="utf-8")
    check("kill marker wins -> wedge", last_stop_reason(m, marker) == "wedge")


def test_clean_for_resume():
    tmp = Path(tempfile.mkdtemp(prefix="sup_"))
    cfg = {"result_root": tmp / "result", "score_root": tmp / "score",
           "gov_log_root": tmp / "govlog"}
    for a in LABELS:
        for root, sub in (("result_root", f"rep01/{a}"),
                          ("score_root", f"rep01/{a}"),
                          ("gov_log_root", f"rep01_{a}")):
            d = cfg[root] / sub
            d.mkdir(parents=True)
            (d / "x.json").write_text("{}", encoding="utf-8")
    removed = clean_for_resume(cfg, 1, "read_irrelevant", LABELS)
    check("arm resume removes the interrupted arm onward (3 dirs x 2 arms)",
          len(removed) == 6, removed)
    check("completed sibling arms kept",
          (cfg["result_root"] / "rep01/baseline/x.json").exists()
          and (cfg["result_root"] / "rep01/read_verbatim/x.json").exists())
    check("interrupted arm gone",
          not (cfg["result_root"] / "rep01/read_irrelevant").exists())
    removed2 = clean_for_resume(cfg, 1, None, LABELS)
    check("replicate restart removes every remaining arm tree",
          not (cfg["result_root"] / "rep01/baseline").exists(), removed2)


def main():
    for k, fn in sorted(globals().items()):
        if k.startswith("test_"):
            print(f"[{k}]")
            fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
