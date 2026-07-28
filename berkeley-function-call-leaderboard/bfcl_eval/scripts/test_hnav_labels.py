"""
Offline tests for the H-Nav corrected outcome label (Stage 1.1) and the
counterfactual NOOP sweep (Stage 1.2).

Run:  python bfcl_eval/scripts/test_hnav_labels.py

Covers: the production grader's word-boundary semantics, marginal-carrier
logic, necessity requiring uniqueness AND retrievability, damage on a dropped
clause, the fate model (terminal / superseded / gone), every uncertainty
trigger, ``suppressed`` deriving from ``applied`` rather than ``action``, the
full target x suppressed cross-table, byte-identical reruns, and the sweep
reproducing ``governance_filter.decide``'s NOOP condition at its boundary.

Synthetic fixtures only -- no committed logs are read, so the suite is stable
across campaigns.
"""

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bfcl_eval.model_handler.middleware.governance_filter import (  # noqa: E402
    GovConfig,
    GovDecision,
    GovernanceCache,
    Stage0Signals,
    ThresholdState,
    WriteCandidate,
    decide,
)
from bfcl_eval.scripts.counterfactual_noop import (  # noqa: E402
    in_region,
    load_labels,
    wilson,
)
from bfcl_eval.scripts.hnav_answer_index import carries, gold_index  # noqa: E402
from bfcl_eval.scripts.label_outcomes_hnav import (  # noqa: E402
    FATES,
    TARGETS,
    _LABEL_MAP,
    build_without,
    label_decision,
    resolve_final_ref,
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
# Fixtures
# ---------------------------------------------------------------------------


def rec(**kw):
    """A minimal gov2 decision record."""
    base = {
        "candidate_id": "c0",
        "test_id": "memory_kv_prereq_1-customer-1",
        "backend": "kv",
        "tier": "core",
        "op": "core_memory_add",
        "candidate_ref": "k1",
        "candidate_text": "k1: some stored text",
        "sim_max": 0.5, "r": 0.5, "delta": 0.32, "sim_high": 0.95,
        "verbatim_misses": [], "n_items": 3, "escalated": False,
        "reason_code": "s0_confident_add", "action": "ADD",
        "applied": "add", "preflight_ok": True,
    }
    base.update(kw)
    return base


def patch_questions(monkey):
    """Install a fake scenario question set on the label module."""
    import bfcl_eval.scripts.label_outcomes_hnav as mod
    mod.scenario_questions = monkey


def test_grader_semantics():
    print("[production grader semantics]")
    check("plain carriage", carries("The user is 38 years old", ["38"]))
    check("word boundary blocks substring",
          not carries("it took 380 seconds", ["38"]),
          "380 must not satisfy gold '38'")
    check("punctuation-insensitive", carries("Seattle, WA", ["seattle"]))
    check("alternative answers", carries("thirty five", ["35", "thirty five"]))
    check("empty text", not carries("", ["35"]))
    check("empty gold", not carries("anything", []))
    check("gold index loads", len(gold_index()) > 100, len(gold_index()))


def test_fate_model():
    print("[fate model]")
    finals = {"core": {"k1": "k1: NEW text"}, "archival": {}}
    check("terminal", resolve_final_ref("kv", "k1", "k1: NEW text", finals, "core")
          == ("k1", "terminal"))
    check("superseded", resolve_final_ref("kv", "k1", "k1: OLD text", finals, "core")
          == ("k1", "superseded"))
    check("gone (ref absent)",
          resolve_final_ref("kv", "k9", "x", finals, "core") == (None, "gone"))
    vec = {"core": {"0": "hello world"}, "archival": {}}
    check("vector add matched by text",
          resolve_final_ref("vector", None, "hello world", vec, "core")
          == ("0", "terminal"))
    check("vector add text gone",
          resolve_final_ref("vector", None, "other text", vec, "core")
          == (None, "gone"))
    check("all fates reachable", set(FATES) == {"terminal", "superseded", "gone"})

    w = build_without({"core": {"k1": "new", "k2": "keep"}, "archival": {}},
                      "core", "k1", True, None)
    check("add reverted by deletion", "k1" not in w["core"] and "k2" in w["core"])
    w2 = build_without({"core": {"k1": "new"}, "archival": {}},
                       "core", "k1", False, "old")
    check("update reverted to old text", w2["core"]["k1"] == "old")


def test_necessity_and_damage():
    print("[necessity / damage]")
    q = [{"qid": "memory_1-customer-1", "text": "how old is the user",
          "gold": ["38"]}]
    patch_questions(lambda s: q)

    # unique retrievable carrier -> must_write
    finals = {"core": {"age_key": "age key: the user is 38 years old",
                       "other_key": "other key: unrelated shipping note"},
              "archival": {}}
    row = label_decision(
        rec(candidate_ref="age_key", candidate_text="age key: the user is 38 years old",
            op="core_memory_add"),
        {"core": {"other_key": "other key: unrelated shipping note"}, "archival": {}},
        True, finals)
    check("unique carrier -> must_write", row["hnav_target"] == "must_write", row)
    check("necessity flagged", row["necessity"] == 1)
    check("marginal question counted", row["n_marginal_q"] == 1)

    # a second carrier makes it non-necessary
    finals2 = {"core": {"age_key": "age key: the user is 38 years old",
                        "dup_key": "dup key: user age is 38 confirmed"},
               "archival": {}}
    row2 = label_decision(
        rec(candidate_ref="age_key", candidate_text="age key: the user is 38 years old"),
        {"core": {"dup_key": "dup key: user age is 38 confirmed"}, "archival": {}},
        True, finals2)
    check("second carrier -> not must_write", row2["hnav_target"] != "must_write", row2)

    # old text already carried the gold -> not marginal -> not must_write
    row3 = label_decision(
        rec(op="core_memory_update", candidate_ref="age_key",
            candidate_text="age key: the user is 38 years old and lives in Seattle"),
        {"core": {"age_key": "age key: the user is 38 years old"}, "archival": {}},
        True,
        {"core": {"age_key": "age key: the user is 38 years old and lives in Seattle"},
         "archival": {}})
    check("old text already carried gold -> not marginal",
          row3["n_marginal_q"] == 0 and row3["hnav_target"] != "must_write", row3)

    # dropped clause destroys another question's answer -> must_suppress
    row4 = label_decision(
        rec(op="core_memory_update", candidate_ref="age_key",
            candidate_text="age key: the user likes espresso"),
        {"core": {"age_key": "age key: the user is 38 years old"}, "archival": {}},
        True,
        {"core": {"age_key": "age key: the user likes espresso"}, "archival": {}})
    check("dropped gold clause -> must_suppress",
          row4["hnav_target"] == "must_suppress" and row4["damage"] == 1, row4)

    # inert: content superseded by a later write
    row5 = label_decision(
        rec(candidate_ref="age_key", candidate_text="age key: the user is 38 years old"),
        {"core": {}, "archival": {}}, True,
        {"core": {"age_key": "age key: LATER overwritten value"}, "archival": {}})
    check("superseded -> inert_superseded",
          row5["hnav_target"] == "inert_superseded" and row5["fate"] == "superseded",
          row5)
    check("lineage diagnostic emitted",
          "lineage_gold_survives" in row5 and "lineage_unique_carrier" in row5)
    check("lineage never promotes to must_write", row5["must_write"] == 0)


def test_uncertainty_triggers():
    print("[uncertainty triggers]")
    q = [{"qid": "memory_1-customer-1", "text": "how old", "gold": ["38"]}]
    patch_questions(lambda s: q)
    finals = {"core": {"k1": "k1: text"}, "archival": {}}

    r1 = label_decision(rec(), {"core": {}, "archival": {}}, False, finals)
    check("chain_desync", r1["uncertain_reason"] == "chain_desync", r1)

    r2 = label_decision(rec(), {"core": {}, "archival": {}}, True,
                        {"core": {}, "archival": {}})
    check("store_empty", r2["uncertain_reason"] == "store_empty", r2)

    patch_questions(lambda s: [])
    r3 = label_decision(rec(), {"core": {}, "archival": {}}, True, finals)
    check("no_gold_for_scenario", r3["uncertain_reason"] == "no_gold_for_scenario", r3)
    patch_questions(lambda s: q)

    r4 = label_decision(
        rec(op="core_memory_update", candidate_ref="k1", candidate_text="k1: text"),
        {"core": {}, "archival": {}}, True, finals)
    check("old_text_missing", r4["uncertain_reason"] == "old_text_missing", r4)

    check("all declared reasons are reachable or documented",
          {"chain_desync", "store_empty", "no_gold_for_scenario",
           "old_text_missing"}.issubset(
              set(__import__("bfcl_eval.scripts.label_outcomes_hnav",
                             fromlist=["UNCERTAIN_REASONS"]).UNCERTAIN_REASONS)))


def test_suppressed_semantics():
    print("[suppressed derives from `applied`, not `action`]")
    q = [{"qid": "memory_1-customer-1", "text": "how old", "gold": ["38"]}]
    patch_questions(lambda s: q)
    finals = {"core": {"k1": "k1: the user is 38 years old"}, "archival": {}}
    shadow = label_decision(
        rec(action="NOOP", applied="none", candidate_ref="k1",
            candidate_text="k1: the user is 38 years old"),
        {"core": {}, "archival": {}}, True, finals)
    check("shadow NOOP counts as a WRITE", shadow["suppressed"] == 0, shadow)
    check("shadow NOOP of a needed fact -> correct_add_or_update",
          shadow["hnav_label"] == "correct_add_or_update", shadow)

    live = label_decision(
        rec(action="NOOP", applied="noop", candidate_ref="k1",
            candidate_text="k1: the user is 38 years old"),
        {"core": {}, "archival": {}}, True, finals)
    check("live NOOP counts as suppressed", live["suppressed"] == 1)
    check("live NOOP of a needed fact -> harmful_noop",
          live["hnav_label"] == "harmful_noop", live)


def test_cross_table():
    print("[target x suppressed cross-table]")
    expected = {
        ("must_write", True): "harmful_noop",
        ("must_write", False): "correct_add_or_update",
        ("must_suppress", True): "correct_noop",
        ("must_suppress", False): "harmful_add_or_update",
        ("may_suppress", True): "correct_noop",
        ("may_suppress", False): "correct_add_or_update",
        ("inert_superseded", True): "correct_noop",
        ("inert_superseded", False): "correct_add_or_update",
        ("uncertain", True): "uncertain",
        ("uncertain", False): "uncertain",
    }
    check("all 10 cells map as specified", _LABEL_MAP == expected,
          set(_LABEL_MAP.items()) ^ set(expected.items()))
    check("targets cover the map",
          {t for t, _ in _LABEL_MAP} == set(TARGETS))


def test_determinism():
    print("[determinism]")
    q = [{"qid": "memory_1-customer-1", "text": "how old", "gold": ["38"]}]
    patch_questions(lambda s: q)
    finals = {"core": {"k1": "k1: the user is 38 years old"}, "archival": {}}
    a = label_decision(rec(candidate_ref="k1",
                           candidate_text="k1: the user is 38 years old"),
                       {"core": {}, "archival": {}}, True, finals)
    b = label_decision(rec(candidate_ref="k1",
                           candidate_text="k1: the user is 38 years old"),
                       {"core": {}, "archival": {}}, True, finals)
    check("identical rows on rerun", json.dumps(a, sort_keys=True) ==
          json.dumps(b, sort_keys=True))


def test_region_matches_decide():
    """The sweep's region predicate must BE governance_filter.decide's NOOP
    condition -- including which inequalities are strict."""
    print("[NOOP region == governance_filter.decide]")
    cfg = GovConfig()
    cfg.sim_high, cfg.delta = 0.90, 0.50
    cache = GovernanceCache("kv")
    cand = WriteCandidate(backend="kv", op="core_memory_add", kind="add",
                          tier="core", ref="k", text="t",
                          args={"key": "k", "value": "v"})
    cases = [
        (0.95, 0.10, [], True, True),    # inside
        (0.90, 0.10, [], True, False),   # sim_max == sim_high -> NOT inside (strict >)
        (0.95, 0.50, [], True, False),   # r == delta -> NOT inside (strict <)
        (0.95, 0.10, ["x"], True, False),  # verbatim miss vetoes
        (0.95, 0.10, [], False, False),  # preflight blocks the NOOP
        (0.85, 0.10, [], True, False),   # below sim_high
    ]
    for sim_max, r, misses, preflight, expected in cases:
        sig = Stage0Signals(n_items=3, rho=0.5)
        sig.sim_max, sig.r, sig.tau_t = sim_max, r, 0.1
        sig.verbatim_misses = list(misses)
        d, _reason = decide(cand, sig, cache, cfg, preflight)
        real = d == GovDecision.NOOP
        row = {"sim_max": sim_max, "r": r, "n_verbatim_misses": len(misses),
               "preflight_ok": preflight}
        mine = in_region(row, cfg.sim_high, cfg.delta, True, True)
        check(f"sim={sim_max} r={r} misses={len(misses)} pf={preflight}",
              real == mine == expected, f"decide={real} sweep={mine} want={expected}")

    # the geometric variant must be a superset of the deployable one
    row = {"sim_max": 0.95, "r": 0.10, "n_verbatim_misses": 2, "preflight_ok": False}
    check("geometric region ignores both side conditions",
          in_region(row, 0.90, 0.50, False, False)
          and not in_region(row, 0.90, 0.50, True, True))


def test_wilson_and_loader():
    print("[wilson + label loader]")
    p, lo, hi = wilson(0, 10)
    check("wilson(0/10) upper ~ 0.278", abs(hi - 0.2775) < 0.01, hi)
    check("wilson(0/10) lower 0", lo == 0.0)
    p, lo, hi = wilson(10, 10)
    check("wilson(10/10) point 1.0", p == 1.0)
    check("wilson(10/10) lower < 1", lo < 1.0)
    check("wilson(0/0) -> None", wilson(0, 0) == (None, None, None))

    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "labels.jsonl"
        rows = [
            {"candidate_id": "a", "scenario": "customer", "arm": "x",
             "replicate": 1, "hnav_target": "must_write"},
            {"candidate_id": "b", "scenario": "student", "arm": "x",
             "replicate": 1, "hnav_target": "may_suppress"},
            {"candidate_id": "a", "scenario": "customer", "arm": "x",
             "replicate": 1, "hnav_target": "must_write"},  # duplicate
        ]
        f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        got = load_labels([str(f)])
        check("student excluded by default", all(r["scenario"] != "student" for r in got))
        check("duplicates dropped", len(got) == 1, len(got))
        check("student included on request",
              len(load_labels([str(f)], include_student=True)) == 2)


def main():
    test_grader_semantics()
    test_fate_model()
    test_necessity_and_damage()
    test_uncertainty_triggers()
    test_suppressed_semantics()
    test_cross_table()
    test_determinism()
    test_region_matches_decide()
    test_wilson_and_loader()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
