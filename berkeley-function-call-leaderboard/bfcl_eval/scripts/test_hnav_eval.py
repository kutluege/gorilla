"""
Offline tests for the H-Nav Stage 1.4 evaluation harness.

Run:  python bfcl_eval/scripts/test_hnav_eval.py

Covers: PR-AUC correctness and its no-skill line, planted-signal recovery
through the nested-model machinery, the NC shuffle destroying that signal, the
leakage whitelist rejecting oracle features, chain-grouped fold integrity, the
few-positive guard, and the operating-point arithmetic (recall / precision /
false-override) with its Wilson intervals.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from bfcl_eval.scripts.calibrate_margin_entropy import (  # noqa: E402
    DIFF_FEATURES,
    GEOMETRY_FEATURES,
    MARGIN_FEATURES,
    assert_no_gt_in_features,
    auc,
    cv_splits,
    nc_shuffle_entropy_features,
    nested_auc_report,
    pr_auc,
)
from bfcl_eval.scripts.evaluate_hnav_stage1 import (  # noqa: E402
    MIN_POSITIVES_FOR_MODEL,
    operating_point,
    parse_cell,
    permutation_p,
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


# ---------------------------------------------------------------------------


def make_rows(n=240, seed=7, planted=True):
    """Synthetic decision rows where must_write depends on diff_sim_max only."""
    rng = np.random.default_rng(seed)
    scenarios = ["customer", "finance", "notetaker", "healthcare"]
    rows = []
    for i in range(n):
        scenario = scenarios[i % len(scenarios)]
        replicate = f"rep{(i // len(scenarios)) % 3 + 1:02d}"
        backend = "kv" if i % 2 == 0 else "vector"
        dsm = float(rng.uniform(0, 1))
        p = 1.0 - dsm if planted else 0.5
        label = int(rng.uniform() < p)
        feats = {f: float(rng.normal()) for f in GEOMETRY_FEATURES + MARGIN_FEATURES}
        feats["backend_kv"] = 1.0 if backend == "kv" else 0.0
        feats["n_items"] = float(rng.integers(1, 25))
        for f in DIFF_FEATURES:
            feats[f] = float(rng.normal())
        feats["diff_sim_max"] = dsm
        rows.append({
            "candidate_id": f"c{i}", "backend": backend, "scenario": scenario,
            "replicate": replicate, "chain": (backend, scenario, replicate),
            "features": feats, "label": label,
            "outcomes": {
                "hnav_target": "must_write" if label else "may_suppress",
                "op_family": "update_replace", "fate": "terminal",
                "cand_len": 100, "sim_max": 0.95, "r": 0.1,
                "n_verbatim_misses": 0, "preflight_ok": 1,
            },
        })
    return rows


NESTS = {
    "geometry": list(GEOMETRY_FEATURES),
    "geometry+margin_entropy": list(GEOMETRY_FEATURES + MARGIN_FEATURES),
    "geometry+margin_entropy+diff": list(
        GEOMETRY_FEATURES + MARGIN_FEATURES + DIFF_FEATURES
    ),
}
SPECS = {"dAUC_diff_given_geometry_margin": "geometry+margin_entropy+diff"}
REF = "geometry+margin_entropy"


def test_pr_auc():
    print("[pr_auc]")
    y = np.array([1.0, 1.0, 0.0, 0.0])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    check("perfect ranking -> 1.0", abs(pr_auc(y, p) - 1.0) < 1e-9, pr_auc(y, p))
    check("inverted ranking well below 1",
          pr_auc(y, -p) < 0.6, pr_auc(y, -p))
    # hand-computed: scores 0.9(pos) 0.8(neg) 0.7(pos) -> AP = (1*1/2)+(2/3*1/2)
    y2 = np.array([1.0, 0.0, 1.0])
    p2 = np.array([0.9, 0.8, 0.7])
    check("hand-computed AP", abs(pr_auc(y2, p2) - (0.5 + (2 / 3) * 0.5)) < 1e-9,
          pr_auc(y2, p2))
    check("degenerate (all positive) -> None", pr_auc(np.ones(4), np.arange(4.)) is None)
    check("degenerate (all negative) -> None", pr_auc(np.zeros(4), np.arange(4.)) is None)

    rng = np.random.default_rng(0)
    y3 = (rng.uniform(size=4000) < 0.3).astype(float)
    p3 = rng.uniform(size=4000)
    check("random scorer ~ base rate", abs(pr_auc(y3, p3) - y3.mean()) < 0.05,
          f"{pr_auc(y3, p3):.4f} vs base {y3.mean():.4f}")


def test_planted_signal_and_nc():
    print("[planted signal + NC control]")
    rows = make_rows()
    rep = nested_auc_report(rows, seed=12345, n_boot=300, nests=NESTS,
                            delta_specs=SPECS, delta_ref=REF)
    d = rep["dAUC_diff_given_geometry_margin"]
    check("diff nest beats reference AUC",
          rep["geometry+margin_entropy+diff"]["auc"] > rep[REF]["auc"],
          f"{rep['geometry+margin_entropy+diff']['auc']} vs {rep[REF]['auc']}")
    check("planted dAUC positive", d["mean"] > 0, d)
    check("planted dAUC CI excludes 0", d["ci95"][0] > 0, d["ci95"])
    check("pr_auc reported per nest", rep[REF]["pr_auc"] is not None)
    check("base rate reported", rep.get("base_rate") is not None)

    nc = nc_shuffle_entropy_features(rows, 12345, feats=DIFF_FEATURES)
    check("NC preserves labels",
          [r["label"] for r in nc] == [r["label"] for r in rows])
    check("NC preserves geometry features",
          all(a["features"]["sim_max"] == b["features"]["sim_max"]
              for a, b in zip(nc, rows)))
    check("NC actually permutes the diff features",
          any(a["features"]["diff_sim_max"] != b["features"]["diff_sim_max"]
              for a, b in zip(nc, rows)))
    nc_rep = nested_auc_report(nc, seed=12345, n_boot=300, nests=NESTS,
                               delta_specs=SPECS, delta_ref=REF)
    nc_d = nc_rep["dAUC_diff_given_geometry_margin"]
    check("NC destroys the planted gain", nc_d["mean"] < d["mean"],
          f"nc={nc_d['mean']:.4f} real={d['mean']:.4f}")
    # The control must show no POSITIVE gain. It is expected to sit BELOW zero:
    # the diff nest carries 17 extra features, so shuffling them into noise
    # costs out-of-fold AUC rather than leaving it unchanged.
    check("NC shows no positive gain", nc_d["ci95"][0] <= 0, nc_d["ci95"])

    nc2 = nc_shuffle_entropy_features(rows, 12345, feats=DIFF_FEATURES)
    check("NC shuffle deterministic",
          [r["features"]["diff_sim_max"] for r in nc]
          == [r["features"]["diff_sim_max"] for r in nc2])


def test_null_signal():
    print("[no planted signal]")
    rows = make_rows(planted=False, seed=11)
    rep = nested_auc_report(rows, seed=12345, n_boot=300, nests=NESTS,
                            delta_specs=SPECS, delta_ref=REF)
    d = rep["dAUC_diff_given_geometry_margin"]
    check("null dAUC CI covers 0", d["ci95"][0] <= 0 <= d["ci95"][1], d["ci95"])


def test_leakage_and_folds():
    print("[leakage + fold integrity]")
    rows = make_rows(n=40)
    assert_no_gt_in_features(rows)
    check("clean rows pass the whitelist", True)

    rows[0]["features"]["oracle_carries_gold"] = 1.0
    try:
        assert_no_gt_in_features(rows)
        check("oracle feature rejected", False, "no AssertionError raised")
    except AssertionError:
        check("oracle feature rejected", True)
    del rows[0]["features"]["oracle_carries_gold"]

    rows[0]["features"]["question_sim"] = 1.0
    try:
        assert_no_gt_in_features(rows)
        check("question-derived feature rejected", False)
    except AssertionError:
        check("question-derived feature rejected", True)
    del rows[0]["features"]["question_sim"]

    n_folds = 0
    for held, train, test in cv_splits(rows):
        n_folds += 1
        check(f"fold {held}: no chain straddles",
              not ({r["chain"] for r in train} & {r["chain"] for r in test}))
    check("one fold per scenario", n_folds == 4, n_folds)

    corrupted = make_rows(n=40)
    corrupted[0]["chain"] = corrupted[-1]["chain"]
    corrupted[0]["scenario"] = "customer"
    corrupted[-1]["scenario"] = "finance"
    try:
        list(cv_splits(corrupted))
        check("corrupted grouping caught", False, "assertion did not fire")
    except AssertionError:
        check("corrupted grouping caught", True)


def test_operating_point():
    print("[operating point arithmetic]")
    def row(dsm, label, target):
        return {"features": {"diff_sim_max": dsm}, "label": label,
                "chain": ("kv", "customer", "rep01"),
                "outcomes": {"hnav_target": target}}

    region = [
        row(0.1, 1, "must_write"),   # harmful, rescued at tau=0.5
        row(0.9, 1, "must_write"),   # harmful, still suppressed
        row(0.1, 0, "may_suppress"),  # safe, needlessly rescued
        row(0.9, 0, "may_suppress"),  # safe, correctly suppressed
        row(0.9, 0, "inert_superseded"),
        row(0.5, 0, "uncertain"),    # excluded from every denominator
    ]
    op = operating_point(region, 0.5)
    check("uncertain excluded from safe", op["n_safe"] == 3, op["n_safe"])
    check("harmful counted", op["n_harmful"] == 2)
    check("recall = 1/2", abs(op["harmful_noop_recall"] - 0.5) < 1e-9)
    # wilson() rounds the point estimate to 6 dp, so compare at that resolution
    check("false override = 1/3",
          abs(op["false_override_rate"] - 1 / 3) < 1e-6, op["false_override_rate"])
    check("precision = 2/3 among still-suppressed",
          abs(op["correct_noop_precision"] - 2 / 3) < 1e-6,
          op["correct_noop_precision"])
    check("wilson intervals present",
          all(op[f"{k}_wilson95"][0] is not None for k in
              ("harmful_noop_recall", "correct_noop_precision", "false_override_rate")))

    op_all = operating_point(region, 1.1)  # tau above every score: rescue all
    check("tau above range rescues everything",
          op_all["harmful_noop_recall"] == 1.0 and op_all["n_still_suppressed"] == 0)
    op_none = operating_point(region, 0.0)  # tau below every score: rescue none
    check("tau below range rescues nothing", op_none["harmful_noop_recall"] == 0.0)


def test_few_positive_guard_and_permutation():
    print("[few-positive guard + permutation test]")
    check("guard threshold is 25", MIN_POSITIVES_FOR_MODEL == 25)
    rng = np.random.default_rng(3)
    region = []
    for i in range(40):
        dsm = float(rng.uniform())
        label = int(dsm < 0.2)  # strong planted signal, few positives
        region.append({
            "features": {"diff_sim_max": dsm}, "label": label,
            "chain": ("kv", "customer", f"rep{i % 3 + 1:02d}"),
            "outcomes": {"hnav_target": "must_write" if label else "may_suppress"},
        })
    n_pos = sum(r["label"] for r in region)
    check("fixture is in the few-positive regime", n_pos < MIN_POSITIVES_FOR_MODEL, n_pos)
    perm = permutation_p(region, seed=12345, n_perm=500)
    check("permutation test runs", perm is not None)
    check("statistic recovers the planted direction",
          perm["statistic_auc"] > 0.7, perm)
    check("p-value significant", perm["p_value"] < 0.05, perm)
    check("p-value never 0 (add-one)", perm["p_value"] > 0)

    flat = [dict(r, label=0) for r in region]
    check("degenerate label -> None", permutation_p(flat, n_perm=50) is None)


def test_parse_cell():
    print("[cell key parsing]")
    check("round trip",
          parse_cell("sim_high=0.950|delta=0.32|veto=1|preflight=0")
          == (0.95, 0.32, True, False))
    check("both flags off",
          parse_cell("sim_high=0.800|delta=0.90|veto=0|preflight=0")
          == (0.8, 0.9, False, False))


def main():
    test_pr_auc()
    test_planted_signal_and_nc()
    test_null_signal()
    test_leakage_and_folds()
    test_operating_point()
    test_few_positive_guard_and_permutation()
    test_parse_cell()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
