"""Tests for evaluate_hact_shadow.py -- synthetic planted-signal fixtures.

No logs, no model. Run: python bfcl_eval/scripts/test_evaluate_hact_shadow.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import evaluate_hact_shadow as ehs  # noqa: E402

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


SCENARIOS = ("customer", "finance", "healthcare", "notetaker")
REPS = ("rep01_hact_shadow", "rep02_hact_shadow", "rep03_hact_shadow")


def make_rows(n_per_cell=12, seed=7, planted=True, base_rate=0.15):
    """Synthetic joined+labeled T1 rows. planted: P(label) rises with h_act."""
    rng = np.random.default_rng(seed)
    rows = []
    for backend in ("kv", "vector"):
        for scen in SCENARIOS:
            for rep in REPS:
                for i in range(n_per_cell):
                    h = float(rng.uniform(0, math.log(8)))
                    if planted:
                        p = 0.03 + 0.5 * (h / math.log(8))
                    else:
                        p = base_rate
                    label = 1.0 if rng.uniform() < p else 0.0
                    feats = {
                        "backend_kv": 1.0 if backend == "kv" else 0.0,
                        "op_add": float(rng.integers(0, 2)),
                        "tier_core": 1.0,
                        "sim_max": float(rng.normal()), "r": float(rng.normal()),
                        "tau_t": 0.6, "rho": float(rng.uniform()),
                        "n_items": float(rng.integers(0, 30)),
                        "rank_self_med": None, "nmargin_med": None,
                        "dH_self": None, "dH_mean": None, "n_eff_norm": None,
                        "disp": None, "churn": None, "smallstore": 0.0,
                        "lp_mean": float(rng.normal(-0.5, 0.1)),
                        "lp_min": float(rng.normal(-2, 0.5)),
                        "lp_ppl": float(rng.uniform(1, 3)),
                        "lp_topk_margin_mean": float(rng.uniform(0, 5)),
                        "lp_topk_margin_min": float(rng.uniform(0, 1)),
                        "lp_tool_mean": float(rng.normal(-0.5, 0.1)),
                        "lp_tool_min": float(rng.normal(-2, 0.5)),
                        "modal_share_op": 1.0, "vote_margin_op": 1.0,
                        # vote controls: weakly correlated proxies of h (so the
                        # entropy increment beyond them stays clearly present)
                        "modal_share_target": max(0.125, 1.0 - h / math.log(8)
                                                  + float(rng.normal(0, 0.6))),
                        "vote_margin_target": float(rng.uniform(0, 1)),
                        "n_unique_target": float(rng.integers(1, 8)),
                        "n_parse_fail_target": 0.0, "n_no_call_target": 0.0,
                        "primary_matches_modal_target": float(rng.integers(0, 2)),
                        "h_act_tool": h * 0.5, "h_act_op": h * 0.8,
                        "h_act_target": h, "h_act_full": h * 1.1,
                        "h_act_norm_op": h / math.log(8),
                        "h_act_norm_target": h / math.log(8),
                        "h_act_norm_full": h / math.log(8),
                    }
                    feats["hx_target_kv"] = h * feats["backend_kv"]
                    feats["hx_target_add"] = h * feats["op_add"]
                    rows.append({
                        "candidate_id": f"{backend[:2]}{scen[:2]}{rep[-13]}{i:03d}"
                                        f"{len(rows):05d}",
                        "backend": backend, "scenario": scen, "replicate": rep,
                        "chain": (backend, scen, rep), "label": label,
                        "features": feats,
                    })
    return rows


def test_leakage_guard():
    rows = make_rows(n_per_cell=1)
    try:
        ehs.leakage_guard(rows)
        check("clean rows pass guard", True)
    except AssertionError as e:
        check("clean rows pass guard", False, str(e))
    bad = json.loads(json.dumps(rows[0]))
    bad["chain"] = rows[0]["chain"]
    bad["features"]["oracle_carries_gold"] = 1.0
    try:
        ehs.leakage_guard([bad])
        check("oracle key rejected", False)
    except AssertionError:
        check("oracle key rejected", True)
    bad2 = json.loads(json.dumps(rows[0]))
    bad2["features"]["question_sim"] = 0.5
    try:
        ehs.leakage_guard([bad2])
        check("non-whitelisted key rejected", False)
    except AssertionError:
        check("non-whitelisted key rejected", True)


def test_planted_signal_recovered():
    rows = make_rows(seed=7, planted=True)
    rep = ehs.ladder_report(rows, seed=12345, n_boot=300)
    head = rep[ehs.HEADLINE]
    check("headline dAUC positive", head["mean"] > 0, head)
    check("headline CI excludes 0", head["ci95"][0] > 0, head)
    check("M8 beats M0",
          rep["M8"]["auc"] > rep["M0"]["auc"], (rep["M8"], rep["M0"]))
    check("brier/ece attached", "brier" in rep["M8"] and "ece" in rep["M8"])


def test_null_no_signal():
    rows = make_rows(seed=11, planted=False)
    rep = ehs.ladder_report(rows, seed=12345, n_boot=300)
    head = rep[ehs.HEADLINE]
    check("null: CI covers 0", head["ci95"][0] <= 0.0 <= head["ci95"][1], head)


def test_nc_destroys_signal():
    rows = make_rows(seed=7, planted=True)
    from calibrate_margin_entropy import nc_shuffle_entropy_features
    feats = list(ehs.ENTROPY_HACT_ALL + ehs.ENTROPY_HACT_NORM + ehs.INTERACTIONS)
    nc_rows = nc_shuffle_entropy_features(
        rows, 12345, feats=feats,
        bin_fn=lambda r: (r["backend"], r["features"].get("op_add")))
    rep_nc = ehs.ladder_report(nc_rows, seed=12345, n_boot=300)
    rep_real = ehs.ladder_report(rows, seed=12345, n_boot=300)
    check("NC2 mean below real mean",
          rep_nc[ehs.HEADLINE]["mean"] < rep_real[ehs.HEADLINE]["mean"])
    check("NC2 collapsed by gate helper", ehs.nc_collapsed(rep_nc[ehs.HEADLINE]),
          rep_nc[ehs.HEADLINE])
    noise_rows = ehs.gaussian_noise_features(rows, feats, 12345)
    rep_noise = ehs.ladder_report(noise_rows, seed=12345, n_boot=300)
    check("NC3 noise collapsed", ehs.nc_collapsed(rep_noise[ehs.HEADLINE]),
          rep_noise[ehs.HEADLINE])
    shuf = ehs.global_label_shuffle(rows, 12345)
    rep_shuf = ehs.ladder_report(shuf, seed=12345, n_boot=300)
    check("NC4 all-shuffled AUC ~ 0.5",
          abs((rep_shuf["M8"]["auc"] or 0.5) - 0.5) < 0.08, rep_shuf["M8"])


def test_step_shift_and_permutation():
    rows = make_rows(seed=7, planted=True)
    shifted = ehs.shift_labels_within_chain(rows)
    same = sum(1 for a, b in zip(rows, shifted) if a["label"] == b["label"])
    check("NC5 moves labels", same < len(rows), f"{same}/{len(rows)}")
    check("NC5 preserves per-chain label counts",
          sum(r["label"] for r in rows) == sum(r["label"] for r in shifted))
    perm = ehs.within_chain_permutation_p(rows, "h_act_target",
                                          seed=1, n_perm=400)
    check("planted perm p small", perm["p_value"] < 0.02, perm)
    null_rows = make_rows(seed=13, planted=False)
    perm0 = ehs.within_chain_permutation_p(null_rows, "h_act_target",
                                           seed=1, n_perm=400)
    check("null perm p large", perm0["p_value"] > 0.05, perm0)


def test_replicate_holdout():
    rows = make_rows(seed=7, planted=True)
    out = ehs.replicate_holdout_delta(rows, "M8", "M4", seed=12345)
    check("3 held-out replicates", len(out) == 3, out)
    vals = [v for v in out.values() if v is not None]
    check("planted: positive in all holdouts", all(v > 0 for v in vals), out)


def test_gate_decisions():
    def fake_report(head_ci, part_ci, p_holm, holdout, n_pos, nc_mean=0.0):
        return {"t1": {
            "n_positive": n_pos,
            "ladder": {ehs.HEADLINE: {"mean": (head_ci[0] + head_ci[1]) / 2,
                                      "ci95": list(head_ci)},
                       ehs.PARTIAL_KEY: {"mean": (part_ci[0] + part_ci[1]) / 2,
                                         "ci95": list(part_ci)}},
            "permutation": {"h_act_target": {"p_value": p_holm}},
            "holm_adjusted": {"h_act_target": p_holm},
            "nc": {"nc2_shuffle_entropy": {
                ehs.HEADLINE: {"mean": nc_mean, "ci95": [-0.02, 0.02]},
                ehs.PARTIAL_KEY: {"mean": nc_mean, "ci95": [-0.02, 0.02]}},
                "nc12_replicate_holdout": holdout},
        }}

    g = ehs.decide_gate(fake_report((0.02, 0.10), (0.01, 0.05), 0.001,
                                    {"r1": 0.03, "r2": 0.02, "r3": 0.04}, 60))
    check("all-pass -> GO", g["verdict"] == "GO", g)
    g = ehs.decide_gate(fake_report((-0.01, 0.10), (0.01, 0.05), 0.001,
                                    {"r1": 0.03, "r2": 0.02, "r3": 0.04}, 60))
    check("headline CI covers 0 -> PARTIAL", g["verdict"] == "PARTIAL", g)
    g = ehs.decide_gate(fake_report((-0.01, 0.10), (-0.01, 0.05), 0.001,
                                    {"r1": 0.03, "r2": 0.02, "r3": 0.04}, 60))
    check("both fail -> NO_GO", g["verdict"] == "NO_GO", g)
    g = ehs.decide_gate(fake_report((0.02, 0.10), (0.01, 0.05), 0.2,
                                    {"r1": 0.03, "r2": 0.02, "r3": 0.04}, 60))
    check("big p -> not GO", g["verdict"] != "GO", g)
    g = ehs.decide_gate(fake_report((0.02, 0.10), (0.01, 0.05), 0.001,
                                    {"r1": 0.03, "r2": -0.02, "r3": 0.04}, 60))
    check("sign flip -> not GO", g["verdict"] != "GO", g)
    g = ehs.decide_gate(fake_report((0.02, 0.10), (0.01, 0.05), 0.001,
                                    {"r1": 0.03, "r2": 0.02, "r3": 0.04}, 10))
    check("too few positives -> not GO", g["verdict"] != "GO", g)


def test_t2_rows_and_join():
    hact = {}
    orphans = []
    tc = ("<tool_call>\n{\"name\": \"core_memory_add\", \"arguments\": "
          "{\"key\": \"a\", \"value\": \"1\"}}\n</tool_call>")
    for i in range(6):
        rec = {"schema": "hact1", "test_id": f"memory_kv_prereq_{i}-customer-0",
               "backend": "kv", "call_idx_in_entry": 0,
               "samples": [{"calls": ["core_memory_add(key='a', value='1')"]},
                           {"calls": ["core_memory_add(key='a', value='1')"]},
                           {"calls": None}],
               "votes": {}, "logprob_features": {"lp_mean": -0.5},
               "_replicate": "rep01_hact_shadow"}
        hact[("rep01_hact_shadow", rec["test_id"], i + 1)] = rec
    orphans.append({"schema": "hact1", "test_id": "memory_kv_prereq_9-customer-0",
                    "backend": "kv", "call_idx_in_entry": 0, "orphan": True,
                    "samples": [{"calls": None}, {"calls": None}],
                    "votes": {}, "logprob_features": {},
                    "_replicate": "rep01_hact_shadow"})
    orphans.append({"schema": "hact1", "test_id": "memory_kv_prereq_1-student-0",
                    "backend": "kv", "call_idx_in_entry": 0, "orphan": True,
                    "samples": [], "votes": {}, "logprob_features": {},
                    "_replicate": "rep01_hact_shadow"})
    rows = ehs.build_t2_rows(hact, orphans)
    check("t2 rows built (student excluded)", len(rows) == 7, len(rows))
    check("t2 one positive", sum(r["label"] for r in rows) == 1.0)
    check("t2 exploration-only votes computed",
          any(r["features"].get("h_act_target") is not None for r in rows))
    check("t2 excludes lp_tool features",
          all("lp_tool_mean" not in r["features"] for r in rows))

    # join respects (replicate, test_id, step_idx) and marks op metadata
    decisions = [{
        "candidate_id": "c1", "backend": "kv", "scenario": "customer",
        "replicate": "rep01_hact_shadow",
        "chain": ("kv", "customer", "rep01_hact_shadow"),
        "test_id": "memory_kv_prereq_0-customer-0", "step_idx": 1,
        "op": "core_memory_add", "tier": "core",
        "features": {"sim_max": 0.5, "backend_kv": 1.0},
    }, {
        "candidate_id": "c2", "backend": "kv", "scenario": "customer",
        "replicate": "rep01_hact_shadow",
        "test_id": "memory_kv_prereq_0-customer-0", "step_idx": 99,
        "op": "core_memory_replace", "tier": "core",
        "chain": ("kv", "customer", "rep01_hact_shadow"),
        "features": {"sim_max": 0.5, "backend_kv": 1.0},
    }]
    joined, missing = ehs.join_hact(decisions, hact)
    check("join hit + miss", len(joined) == 1 and missing == 1)
    jf = joined[0]["features"]
    check("op_add derived", jf["op_add"] == 1.0)
    check("lp merged", jf["lp_mean"] == -0.5)
    check("interaction built", "hx_target_kv" in jf)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"[{fn.__name__}]")
        fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
