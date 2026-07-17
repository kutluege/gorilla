"""
Offline tests for correlate_geometry_deltaH.py (Plan 3 Step 2 acceptance).

Run:  python bfcl_eval/scripts/test_correlate_geometry.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bfcl_eval.scripts.correlate_geometry_deltaH import (  # noqa: E402
    bootstrap_rho,
    collect_pairs,
    correlate,
    surviving_scenarios,
)
from bfcl_eval.scripts.replay_geometry_deltaH import spearman  # noqa: E402

PASS, FAIL = 0, 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def decision(backend, scenario, sim_max, r, dh, with_stage2=True):
    rec = {
        "event": "decision",
        "test_id": f"memory_{backend}_prereq_3-{scenario}-1",
        "backend": backend, "sim_max": sim_max, "r": r,
    }
    if with_stage2:
        rec["stage2"] = {"dH_mean": dh}
    elif dh is None:
        rec["stage2"] = {"dH_mean": None}
    return rec


def wrra_scen(backend, scenario, dead):
    return {"record": "wrra_scenario", "backend": backend,
            "scenario": scenario, "chain_dead": dead}


def run():
    print("[spearman sanity]")
    check("perfect monotone -> 1.0",
          abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9)
    check("perfect anti-monotone -> -1.0",
          abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9)
    check("hand-checked rho = 0.8",
          abs(spearman([1, 2, 3, 4], [1, 3, 2, 4]) - 0.8) < 1e-9)

    print("[survival filter + join]")
    wrra = [wrra_scen("kv", "customer", False),
            wrra_scen("kv", "student", True),
            wrra_scen("vector", "customer", False)]
    surviving = surviving_scenarios(wrra)
    check("surviving set built",
          surviving == {("kv", "customer"), ("vector", "customer")})

    logs = [
        decision("kv", "customer", 0.9, 0.1, 0.5),
        decision("kv", "customer", 0.5, 0.5, 0.1),
        decision("kv", "student", 0.99, 0.01, 0.9),      # dead chain
        decision("kv", "customer", 0.7, 0.3, None),      # dH null
        {"event": "rehydrate", "backend": "kv"},          # non-decision
        {"event": "decision", "test_id": "memory_kv_prereq_1-customer-0",
         "backend": "kv", "sim_max": 0.6, "r": 0.4},      # no stage2 at all
        decision("vector", "customer", 0.8, 0.2, 0.4),
    ]
    pairs, counters = collect_pairs(logs, surviving)
    check("dead-chain record excluded", counters["dead_chain"] == 1)
    check("null-dH and missing-stage2 counted separately",
          counters["no_dH"] == 1 and counters["no_stage2"] == 1)
    check("used = 3", counters["used"] == 3, str(counters))
    check("kv pairs joined from same record",
          pairs["kv"] == [(0.9, 0.1, 0.5), (0.5, 0.5, 0.1)])
    check("vector pairs separate", len(pairs["vector"]) == 1)

    print("[gate branching]")
    mono = {"kv": [(0.1 * i, 1 - 0.1 * i, 0.05 * i) for i in range(1, 9)]}
    v = correlate(mono, n_boot=200)["kv"]
    check("monotone sim_max->dH passes gate",
          v["rho_sim_max_vs_dH"] == 1.0 and v["gate_pass"]
          and v["branch"] == "geometry-first")
    check("r anti-correlates on same fixture", v["rho_r_vs_dH"] == -1.0)
    anti = {"vector": [(0.1 * i, 0.5, -0.05 * i) for i in range(1, 9)]}
    v2 = correlate(anti, n_boot=200)["vector"]
    check("anti-monotone fails gate -> nli-first-fallback",
          not v2["gate_pass"] and v2["branch"] == "nli-first-fallback")
    check("KV note pins canonical-key-first", "canonical-key-first" in v["note"])

    print("[bootstrap]")
    xs = [1, 2, 3, 4, 5, 6]
    ys = [1.1, 1.9, 3.2, 3.9, 5.1, 6.2]
    ci1 = bootstrap_rho(xs, ys, n_boot=300, seed=7)
    ci2 = bootstrap_rho(xs, ys, n_boot=300, seed=7)
    check("deterministic under fixed seed", ci1 == ci2)
    check("CI ordered and sane", ci1[0] <= ci1[1] <= 1.0)
    check("too few records -> None CI", bootstrap_rho([1, 2], [1, 2]) == (None, None))

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run())
