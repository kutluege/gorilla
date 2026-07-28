"""Tests for verify_stage1.py comparison machinery (synthetic fixtures only)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import verify_stage1 as vs

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def test_dig():
    obj = {"a": {"b": {"c": 3}}, "x": [1, 2]}
    check("dig nested", vs.dig(obj, "a.b.c") == 3)
    check("dig top", vs.dig(obj, "x") == [1, 2])


def test_close():
    check("close int exact", vs.close(159, 159))
    check("close float within tol", vs.close(0.076378, 0.0763778, tol=5e-3))
    check("close float outside tol", not vs.close(0.076, 0.176, tol=5e-3))
    check("close str", vs.close("FAIL", "FAIL"))
    check("close str mismatch", not vs.close("FAIL", "PASS"))
    check("close list of floats", vs.close([0.0147, 0.1382], [0.014723, 0.138193], tol=5e-3))
    check("close list len mismatch", not vs.close([1.0], [1.0, 2.0]))
    check("close non-numeric vs float", not vs.close("x", 1.0))


def test_deep_compare_match():
    a = {"n": 4575, "nested": {"auc": 0.5243, "ci": [0.01, 0.13]}, "v": "PIVOT"}
    b = {"n": 4575, "nested": {"auc": 0.52431, "ci": [0.0100001, 0.13]}, "v": "PIVOT"}
    diffs = vs.deep_compare(a, b)
    check("deep_compare tolerant match", diffs == [], str(diffs))


def test_deep_compare_mismatch():
    a = {"n": 4575, "nested": {"auc": 0.52}}
    b = {"n": 4574, "nested": {"auc": 0.62}, "extra": 1}
    diffs = vs.deep_compare(a, b)
    check("deep_compare finds count diff", any(d.startswith("n:") for d in diffs), str(diffs))
    check("deep_compare finds float diff", any("nested.auc" in d for d in diffs))
    check("deep_compare finds missing key", any("extra" in d for d in diffs))


def test_deep_compare_skips_git_head():
    a = {"git_head": "aaa", "n": 1}
    b = {"git_head": "bbb", "n": 1}
    check("git_head skipped at top level", vs.deep_compare(a, b) == [])
    a2 = {"inner": {"git_head": "aaa"}}
    b2 = {"inner": {"git_head": "bbb"}}
    check("git_head NOT skipped when nested", vs.deep_compare(a2, b2) != [])


def test_check_expected():
    report = {
        "n_labeled": 4575, "n_positive": 159, "p_hat": 0.408544,
        "target_distribution": {"must_write": 159, "inert_superseded": 3905},
        "mechanical_claim": {"n_near_duplicate_updates": 220, "median_sim_max": 0.9438,
                             "median_diff_sim_max": 0.196219, "n_must_write_among_them": 0},
        "nested_models": {"dAUC_diff_given_geometry_margin": {
            "mean": 0.076378, "ci95": [0.014724, 0.138193]}},
        "hypotheses": {"H1": {"verdict": "FAIL"}, "H2": {"verdict": "PASS"},
                       "H3": {"verdict": "FAIL"}},
        "verdict": {"decision": "PIVOT to action-side H_act"},
    }
    rows = vs.check_expected(report, "fixture")
    check("all expected keys pass on faithful fixture", all(r["match"] for r in rows),
          str([r for r in rows if not r["match"]]))
    report["hypotheses"]["H2"]["verdict"] = "FAIL"
    rows = vs.check_expected(report, "fixture")
    bad = [r for r in rows if not r["match"]]
    check("tampered verdict detected", len(bad) == 1 and bad[0]["key"] == "hypotheses.H2.verdict")
    del report["n_labeled"]
    rows = vs.check_expected(report, "fixture")
    check("missing key detected as mismatch",
          any(r["key"] == "n_labeled" and not r["match"] for r in rows))


def main():
    for fn in [test_dig, test_close, test_deep_compare_match, test_deep_compare_mismatch,
               test_deep_compare_skips_git_head, test_check_expected]:
        print(f"[{fn.__name__}]")
        fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
