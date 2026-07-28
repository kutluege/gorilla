"""
Offline tests for the margin-entropy calibration pipeline (Plan v2 Step 8).

Run:  python bfcl_eval/scripts/test_calibrate_margin_entropy.py

Synthetic-fixture coverage (SS17.2): dataset assembly (escalations only, dead
scenario excluded), the no-GT-in-features assertion, chain-grouped split
integrity, nested-model machinery recovering a planted entropy signal,
Option-A risk-coverage selection, Option-B sign constraints, freeze ceremony
(refuses overwrite), and fixed-seed determinism.
"""

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from bfcl_eval.scripts.calibrate_margin_entropy import (  # noqa: E402
    assert_no_gt_in_features,
    attach_outcomes,
    cv_splits,
    fit_option_b,
    load_decisions,
    nested_auc_report,
    option_a_grid,
)

TMP = Path(tempfile.mkdtemp(prefix="gov_calib_test_"))
PASS, FAIL = 0, 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


SCENARIOS = ("customer", "finance", "healthcare", "notetaker")


def build_fixture(seed=7):
    """Two harvest arms with gov2 logs + outcomes: harm is driven by high
    dH_mean at low margin, so entropy carries signal beyond margin."""
    rng = np.random.default_rng(seed)
    outcomes = []
    for rep in ("rep01", "rep02"):
        arm = TMP / rep
        arm.mkdir(parents=True, exist_ok=True)
        with open(arm / "governance_log.jsonl", "w", encoding="utf-8") as f:
            # one non-gov2 + one non-escalated + one dead-scenario line: all skipped
            f.write(json.dumps({"event": "decision", "test_id": "memory_kv_prereq_0-customer-0",
                                "backend": "kv", "decision": "ADD", "reason": "novel"}) + "\n")
            f.write(json.dumps({"event": "decision", "schema": "gov2", "escalated": False,
                                "test_id": "memory_kv_prereq_0-customer-0", "backend": "kv",
                                "candidate_id": "x0", "action": "ADD",
                                "reason_code": "s0_confident_add"}) + "\n")
            f.write(json.dumps({"event": "decision", "schema": "gov2", "escalated": True,
                                "test_id": "memory_kv_prereq_0-student-0", "backend": "kv",
                                "candidate_id": "dead1", "action": "ADD",
                                "reason_code": "s1_confident", "s1_me": {}}) + "\n")
            k = 0
            for backend in ("kv", "vector"):
                for scen in SCENARIOS:
                    for _ in range(14):
                        k += 1
                        cid = f"{rep}_{backend}_{scen}_{k}"
                        nmargin = float(rng.uniform(0, 0.5))
                        dh = float(rng.uniform(-0.1, 0.5))
                        rank = int(rng.integers(1, 6))
                        rec = {
                            "event": "decision", "schema": "gov2", "escalated": True,
                            "test_id": f"memory_{backend}_prereq_0-{scen}-0",
                            "backend": backend, "candidate_id": cid,
                            "action": "ADD", "reason_code": "s1_ambiguous_default",
                            "sim_max": float(rng.uniform(0.1, 0.9)),
                            "r": float(rng.uniform(0.1, 1.0)),
                            "tau_t": 0.1, "rho": 0.3, "n_items": int(rng.integers(3, 20)),
                            "s1_me": {
                                "rank_self_med": rank, "nmargin_med": nmargin,
                                "dH_self": dh * 0.5, "dH_mean": dh,
                                "n_eff_norm": float(rng.uniform(0.2, 1.0)),
                                "disp": 0.1, "churn": float(rng.uniform(0, 0.6)),
                                "smallstore": False,
                            },
                        }
                        f.write(json.dumps(rec) + "\n")
                        p_harm = 0.85 if (dh > 0.25 and nmargin < 0.25) else 0.08
                        outcomes.append({
                            "candidate_id": cid,
                            "harmful_write": int(rng.uniform() < p_harm),
                            "useless_duplicate": 0,
                        })
    with open(TMP / "outcomes.jsonl", "w", encoding="utf-8") as f:
        for o in outcomes:
            f.write(json.dumps(o) + "\n")


def main():
    build_fixture()
    print("[dataset assembly]")
    rows, manifest = load_decisions([str(TMP / "rep01"), str(TMP / "rep02")])
    check("escalated gov2 rows only", len(rows) == 2 * 2 * 4 * 14, str(len(rows)))
    check("dead scenario excluded", all(r["scenario"] != "student" for r in rows))
    check("manifest carries shas", len(manifest) == 2 and all(m["sha"] for m in manifest))
    check("chains keyed (backend, scenario, replicate)",
          rows[0]["chain"] == (rows[0]["backend"], rows[0]["scenario"], rows[0]["replicate"]))

    print("[leakage guards]")
    try:
        assert_no_gt_in_features(rows)
        check("clean features accepted", True)
    except AssertionError as e:
        check("clean features accepted", False, str(e))
    poisoned = json.loads(json.dumps(rows[0]))
    poisoned["features"]["question_overlap"] = 1.0
    try:
        assert_no_gt_in_features([poisoned])
        check("poisoned feature refused", False)
    except AssertionError:
        check("poisoned feature refused", True)

    print("[labels + splits]")
    rows = attach_outcomes(rows, TMP / "outcomes.jsonl", "harmful_write")
    check("all rows labeled", len(rows) == 2 * 2 * 4 * 14)
    n_folds = 0
    for held, train, test in cv_splits(rows):
        n_folds += 1
        check(f"fold {held}: no scenario leak",
              all(r["scenario"] != held for r in train)
              and all(r["scenario"] == held for r in test))
    check("one fold per scenario", n_folds == len(SCENARIOS))

    print("[nested models / gate-G1 machinery]")
    rep = nested_auc_report(rows, seed=12345, n_boot=300)
    check("all four nests scored",
          all(rep[n]["auc"] is not None for n in
              ("geometry", "geometry+margin", "geometry+entropy", "geometry+margin+entropy")),
          json.dumps({k: v for k, v in rep.items() if "auc" in str(v)[:50]}))
    d = rep.get("dAUC_entropy_given_geometry_margin")
    check("dAUC block present", d is not None)
    check("planted entropy signal recovered (dAUC>0)",
          d is not None and d["mean"] > 0, str(d))
    rep2 = nested_auc_report(rows, seed=12345, n_boot=300)
    check("fixed-seed determinism", json.dumps(rep) == json.dumps(rep2))

    print("[option A risk-coverage]")
    best, table = option_a_grid(rows, "kv", target_risk=0.15)
    check("kv thresholds selected", best is not None, str(best))
    if best:
        check("selected risk <= target", best["selective_risk"] <= 0.15, str(best))
        check("grid table emitted", len(table) > 0)

    print("[option B monotone fit]")
    model = fit_option_b(rows, target_risk=0.15, seed=12345)
    check("model fitted", model is not None)
    if model:
        check("sign: nmargin coef <= 0", model["coefs"]["nmargin_med"] <= 0, str(model["coefs"]))
        check("sign: rank coef >= 0", model["coefs"]["rank_self_med"] >= 0)
        check("sign: dH coef >= 0", model["coefs"]["dH_mean"] >= 0)
        check("thresholds present",
              "threshold_low" in model and "threshold_high" in model)
        check("isotonic mapping monotone",
              all(a <= b + 1e-9 for a, b in
                  zip(model["isotonic"]["values"], model["isotonic"]["values"][1:])))
        check("informative fit (train AUC > 0.6)", model["train_auc"] > 0.6,
              str(model["train_auc"]))

    print("[NC shuffle negative control]")
    from bfcl_eval.scripts.calibrate_margin_entropy import nc_shuffle_entropy_features

    nc_rows = nc_shuffle_entropy_features(rows, 12345)
    check("labels/margin features untouched",
          all(a["label"] == b["label"]
              and a["features"]["nmargin_med"] == b["features"]["nmargin_med"]
              for a, b in zip(rows, nc_rows)))
    check("entropy features actually permuted",
          any(a["features"]["dH_mean"] != b["features"]["dH_mean"]
              for a, b in zip(rows, nc_rows)))
    nc_rep = nested_auc_report(nc_rows, seed=12345, n_boot=300)
    nc_d = nc_rep.get("dAUC_entropy_given_geometry_margin")
    real_d = d
    check("NC destroys the planted entropy gain",
          nc_d is not None and real_d is not None
          and nc_d["mean"] < real_d["mean"]
          and nc_d["ci95"][0] <= 0.0, f"nc={nc_d} real={real_d}")
    nc_rows2 = nc_shuffle_entropy_features(rows, 12345)
    check("NC shuffle deterministic",
          json.dumps(nc_rows) == json.dumps(nc_rows2))

    print("[stage2 generalizations]")
    from calibrate_margin_entropy import brier, ece, reliability_curve
    import numpy as np

    # Brier / ECE known-value fixtures
    y01 = np.array([0.0, 1.0, 0.0, 1.0])
    check("brier perfect = 0", brier(y01, y01) == 0.0)
    check("brier anti-calibrated = 1", brier(y01, 1.0 - y01) == 1.0)
    check("brier constant 0.5", abs(brier(y01, np.full(4, 0.5)) - 0.25) < 1e-12)
    check("brier empty -> None", brier(np.array([]), np.array([])) is None)
    yc = np.array([0.0] * 8 + [1.0] * 2)
    pc = np.full(10, 0.2)   # perfectly calibrated single bin
    check("ece calibrated ~0", ece(yc, pc) < 1e-12, ece(yc, pc))
    check("ece miscalibrated", ece(yc, np.full(10, 0.9)) > 0.6)
    rc = reliability_curve(yc, pc, n_bins=10)
    check("reliability single non-empty bin", len(rc) == 1 and rc[0]["n"] == 10)
    check("reliability frac_pos", abs(rc[0]["frac_pos"] - 0.2) < 1e-12)

    # pairwise delta_specs (nest, ref) tuples == str form vs the same ref
    rep_str = nested_auc_report(
        rows, seed=12345, n_boot=200,
        delta_specs={"d": "geometry+margin+entropy"}, delta_ref="geometry+margin")
    rep_pair = nested_auc_report(
        rows, seed=12345, n_boot=200,
        delta_specs={"d": ("geometry+margin+entropy", "geometry+margin")})
    check("tuple delta_spec == str delta_spec",
          json.dumps(rep_str["d"]) == json.dumps(rep_pair["d"]))
    rep_multi = nested_auc_report(
        rows, seed=12345, n_boot=200,
        delta_specs={"a_vs_g": ("geometry+margin+entropy", "geometry"),
                     "gm_vs_g": ("geometry+margin", "geometry")})
    check("pairwise ladder computes both deltas",
          "a_vs_g" in rep_multi and "gm_vs_g" in rep_multi)

    # return_preds opt-in leaves default shape unchanged
    rep_default = nested_auc_report(rows, seed=12345, n_boot=100)
    check("no _oof_preds by default", "_oof_preds" not in rep_default)
    rep_preds = nested_auc_report(rows, seed=12345, n_boot=100, return_preds=True)
    check("_oof_preds attached on request",
          "_oof_preds" in rep_preds and "_y" in rep_preds
          and len(rep_preds["_y"]) == len(rows))

    # bin_fn override changes stratification but stays deterministic
    nc_default = nc_shuffle_entropy_features(rows, 777)
    nc_op = nc_shuffle_entropy_features(rows, 777,
                                        bin_fn=lambda r: r["backend"])
    check("bin_fn override deterministic",
          json.dumps(nc_op) == json.dumps(
              nc_shuffle_entropy_features(rows, 777, bin_fn=lambda r: r["backend"])))
    check("default bin_fn reproduces original binning",
          json.dumps(nc_default) == json.dumps(nc_shuffle_entropy_features(rows, 777)))

    print("[freeze ceremony]")
    import subprocess

    out = TMP / "calib" / "margin_entropy_calibration.json"
    cmd = [sys.executable, str(REPO_ROOT / "bfcl_eval/scripts/calibrate_margin_entropy.py"),
           "--logs", str(TMP / "rep0*"), "--outcomes", str(TMP / "outcomes.jsonl"),
           "--out", str(out), "--freeze"]
    r1 = subprocess.run(cmd, capture_output=True, text=True)
    check("freeze run exits 0", r1.returncode == 0, r1.stdout[-300:] + r1.stderr[-300:])
    check("calibration JSON written", out.exists())
    check("freeze note written", (out.parent / "CALIBRATION_FROZEN.md").exists())
    r2 = subprocess.run(cmd, capture_output=True, text=True)
    check("re-freeze refused", r2.returncode != 0, r2.stdout[-200:])

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
