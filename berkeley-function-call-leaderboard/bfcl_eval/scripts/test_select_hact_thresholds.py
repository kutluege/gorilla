"""Tests for select_hact_thresholds.py (synthetic rows, no files).

Run: python bfcl_eval/scripts/test_select_hact_thresholds.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from select_hact_thresholds import (  # noqa: E402
    GRIDS,
    intervention_stats,
    select_on_dev,
)

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def row(h, label=0, must_write=False):
    return {"label": label, "features": {"h_act_target": h},
            "outcomes": {"must_write": must_write}}


def test_intervention_stats():
    rows = ([row(1.5, label=1)] * 3          # harmful, high H -> caught
            + [row(0.2, label=1)] * 1        # harmful, low H -> missed
            + [row(1.5, must_write=True)] * 2  # necessary, high H -> false int
            + [row(0.1)] * 14)               # benign, low H
    s = intervention_stats(rows, "h_act_target", "ge", 1.0)
    check("n", s["n"] == 20)
    check("interventions", s["n_interventions"] == 5, s)
    check("coverage", abs(s["coverage"] - 0.25) < 1e-12)
    check("harmful recall 3/4", abs(s["harmful_recall"] - 0.75) < 1e-6, s)
    check("false interventions 2", s["n_false_interventions"] == 2)
    check("FIR 2/5", abs(s["false_intervention_rate"] - 0.4) < 1e-6)
    check("wilson bounds ordered",
          s["harmful_recall_wilson95"][0] < 0.75 < s["harmful_recall_wilson95"][1])
    s2 = intervention_stats(rows, "h_act_target", "ge", 99.0)
    check("no interventions at huge tau", s2["n_interventions"] == 0
          and s2["false_intervention_rate"] is None)
    # None feature values never intervene
    rows_none = rows + [{"label": 1, "features": {"h_act_target": None},
                         "outcomes": {}}]
    s3 = intervention_stats(rows_none, "h_act_target", "ge", 1.0)
    check("None feature skipped", s3["n_interventions"] == 5)
    # 'le' direction (vote margin)
    vm = [{"label": 1, "features": {"vote_margin_target": 0.1}, "outcomes": {}},
          {"label": 0, "features": {"vote_margin_target": 0.9}, "outcomes": {}}]
    s4 = intervention_stats(vm, "vote_margin_target", "le", 0.25)
    check("le direction", s4["n_interventions"] == 1 and s4["harmful_caught"] == 1)


def test_select_on_dev():
    spec = {"feature": "h_act_target", "direction": "ge", "grid": [0.5, 1.0, 1.5]}
    rows = ([row(1.7, label=1)] * 6 + [row(0.7, label=1)] * 2
            + [row(1.7, must_write=True)] * 1 + [row(0.1)] * 41)
    tau, table = select_on_dev(rows, spec, max_fir=0.20, min_recall=0.30)
    check("grid fully swept", len(table) == 3)
    # tau=1.5: catches 6/8 harmful, FIR 1/7=0.14 (ok); tau=0.5: catches 8/8 but
    # FIR 1/9=0.11 also ok -> higher recall wins -> tau=0.5... verify mechanically:
    by_tau = {s["tau"]: s for s in table}
    best_ok = [s for s in table if s["false_intervention_rate"] is not None
               and s["false_intervention_rate"] <= 0.2
               and (s["harmful_recall"] or 0) >= 0.3]
    expect = sorted(best_ok, key=lambda s: (-(s["harmful_recall"] or 0),
                                            s["coverage"]))[0]["tau"]
    check("selection matches rule", tau == expect, (tau, expect))
    check("recall monotone in tau",
          by_tau[0.5]["harmful_recall"] >= by_tau[1.5]["harmful_recall"])
    # impossible constraints -> None
    tau2, _ = select_on_dev(rows, spec, max_fir=0.0, min_recall=0.99)
    check("infeasible -> None", tau2 is None)


def test_grids_reference_known_features():
    known = {"h_act_target", "vote_margin_target", "h_act_op"}
    check("grid features known", all(g["feature"] in known for g in GRIDS.values()))
    check("directions valid", all(g["direction"] in ("ge", "le")
                                  for g in GRIDS.values()))


def main():
    for k, fn in sorted(globals().items()):
        if k.startswith("test_"):
            print(f"[{k}]")
            fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
