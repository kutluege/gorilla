"""
Offline tests for the generate -> evaluate -> score contract
(Plan 3 Step 8 / G15): evaluate runs immediately after every generate, its
exit code and score-file paths land in the manifest, a failed generate never
reaches evaluate, and an unscored replicate is an explicit analyzer ERROR.

Run:  python bfcl_eval/scripts/test_evaluate_integration.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bfcl_eval.scripts.analyze_gov_replicates import (  # noqa: E402
    enforce_scored,
)
from bfcl_eval.scripts.test_gov_replicates import (  # noqa: E402
    Harness,
    arm_records,
    make_cfg,
)

PASS, FAIL = 0, 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def phases_in_order(h):
    return [(r["arm"], r["phase"]) for r in h.manifest if r["event"] == "cmd_start"]


def run():
    print("[every generate is immediately followed by its evaluate]")
    h = Harness()
    h.go(make_cfg(n=2))
    seq = phases_in_order(h)
    check("gen/eval strictly interleaved per arm",
          all(seq[i] == (seq[i + 1][0], "generate")
              and seq[i + 1][1] == "evaluate"
              for i in range(0, len(seq), 2)), str(seq))
    check("no arm generates twice before evaluating",
          [p for _, p in seq] == ["generate", "evaluate"] * 4)

    print("[failed generate: evaluate NOT called, replicate incomplete]")
    h = Harness(fail_on_call=1)  # B1 generate fails immediately
    try:
        h.go(make_cfg(n=1))
        check("failed generate raises", False)
    except SystemExit:
        check("failed generate raises", True)
    check("evaluate never ran for the failed arm",
          len(h.cmds) == 1
          and all(r["phase"] == "generate" for r in h.manifest
                  if r["event"] == "cmd_end"))
    check("manifest marks the error at the generate stage",
          h.manifest[-1]["event"] == "error"
          and h.manifest[-1]["stage"] == "generate")

    print("[score paths + existence in the manifest]")
    h = Harness()
    h.go(make_cfg(n=1))
    ev_ends = [r for r in h.manifest
               if r["event"] == "cmd_end" and r["phase"] == "evaluate"]
    check("every evaluate cmd_end lists its expected score files",
          len(ev_ends) == 2 and all(len(r["score_files"]) == 2 for r in ev_ends))
    check("score records carry path + existence flag",
          all({"path", "exists"} <= set(s)
              for r in ev_ends for s in r["score_files"]))
    check("paths are arm-scoped (same-model arms cannot collide)",
          all("/baseline/" in r["score_files"][0]["path"]
              or "/governed/" in r["score_files"][0]["path"]
              for r in ev_ends), str(ev_ends[0]["score_files"]))

    print("[analyzer: unscored replicate is an explicit error]")
    unscored = [(1, "baseline", "kv")]
    try:
        enforce_scored(unscored, allow_unscored=False)
        check("unscored -> SystemExit", False)
    except SystemExit as e:
        check("unscored -> SystemExit", True)
        check("error names the exact replicate/arm/backend",
              "rep01/baseline/kv" in str(e), str(e))
        check("error tells the operator the remedy",
              "bfcl" in str(e) and "--allow-unscored" in str(e))
    try:
        enforce_scored(unscored, allow_unscored=True)
        check("--allow-unscored is the explicit opt-out", True)
    except SystemExit:
        check("--allow-unscored is the explicit opt-out", False)
    try:
        enforce_scored([], allow_unscored=False)
        check("fully scored passes silently", True)
    except SystemExit:
        check("fully scored passes silently", False)

    print("[metric-layer separation is declared, not implied]")
    # The three layers (official BFCL score primary / W-R-R-A diagnostic /
    # paraphrase-tolerant secondary) are a REPORTING contract; here we pin the
    # analyzer side: wrra question records carry ONLY reconstructed official
    # correctness -- no alternative metric field exists to be confused with it.
    q = arm_records("kv", {"customer": (False, {"q1": True})})
    qr = [r for r in q if r["record"] == "wrra_question"][0]
    check("wrra question record carries official correctness only",
          "correct" in qr and not any(k.startswith("judge") for k in qr))

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run())
