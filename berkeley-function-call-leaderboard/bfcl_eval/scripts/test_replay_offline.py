"""
Offline tests for the SS8.4 replay instrument (replay_geometry_deltaH.py).

Run:  python bfcl_eval/scripts/test_replay_offline.py

Covers (Plan 1 Step 4 acceptance): exact state reconstruction on a hand-built
fixture log (chain-carry, checkpoint verification, desync invalidation,
observe_remove/observe_clear application, NOOP non-application, n_items
cross-check), dH computation plumbing, drop accounting, Spearman helper.
"""

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bfcl_eval.scripts.replay_geometry_deltaH import (  # noqa: E402
    ChainState,
    median,
    replay_log,
    scenario_of,
    spearman,
    summarize,
)

PASS, FAIL = 0, 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def write_log(records):
    tmp = tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for rec in records:
        tmp.write(json.dumps(rec) + "\n")
    tmp.close()
    return Path(tmp.name)


def rehydrate(test_id, backend, n):
    return {"event": "rehydrate", "test_id": test_id, "backend": backend, "items_loaded": n}


def decision(test_id, backend, tier, ref, text, dec="ADD", reason="novel",
             n_items=0, sim_max=0.5, r=0.5, dry_run=False):
    return {
        "event": "decision", "test_id": test_id, "backend": backend, "tier": tier,
        "step_idx": 1, "op": f"{tier}_memory_add", "candidate_ref": ref,
        "candidate_text": text, "decision": dec, "reason": reason,
        "n_items": n_items, "sim_max": sim_max, "r": r,
        "sim_high": 0.95, "delta": 0.3, "verbatim_misses": [], "dry_run": dry_run,
    }


def observe(test_id, backend, tier, ref, text):
    return {"event": "observe", "test_id": test_id, "backend": backend,
            "tier": tier, "ref": ref, "text": text}


def run(records, **kw):
    path = write_log(records)
    params = {"neighbors_k": 5, "top_k": 5, "temperature": 1.0}
    params.update(kw)
    return replay_log(path, params["neighbors_k"], params["top_k"], params["temperature"])


def test_helpers():
    print("[helpers]")
    check("scenario_of", scenario_of("memory_kv_prereq_0-customer-0") == "customer")
    check("scenario_of non-prereq", scenario_of("memory_vector_12-finance-1") == "finance")
    check("spearman perfect", spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0)
    check("spearman inverse", spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0)
    check("spearman ties handled", spearman([1, 1, 2, 3], [1, 1, 2, 3]) == 1.0)
    check("spearman too short -> None", spearman([1, 2], [2, 1]) is None)
    check("spearman constant -> None", spearman([1, 1, 1], [1, 2, 3]) is None)
    check("median odd", median([3, 1, 2]) == 2)
    check("median even", median([1, 2, 3, 4]) == 2.5)
    check("median empty -> None", median([]) is None)
    cs = ChainState()
    check("chain starts unsynced/empty", not cs.synced and cs.total() == 0)


def test_basic_chain():
    print("[basic chain-carry]")
    t = "memory_kv_prereq_0-customer-0"
    t2 = "memory_kv_prereq_1-customer-0"
    records = [
        rehydrate(t, "kv", 0),
        decision(t, "kv", "core", "user_age", "user age: 35", n_items=0),
        observe(t, "kv", "core", "user_age", "user age: 35"),
        decision(t, "kv", "core", "user_name", "user name: Michael", n_items=1),
        observe(t, "kv", "core", "user_name", "user name: Michael"),
        rehydrate(t2, "kv", 2),  # checkpoint: carried state must be 2 items
        decision(t2, "kv", "core", "user_age", "user age: 35",
                 dec="NOOP", reason="redundant", n_items=2, sim_max=0.99, r=0.01),
    ]
    rows, counters = run(records)
    check("all decisions seen", counters["decisions_total"] == 3)
    check("all computable", counters["decisions_computable"] == 3)
    check("checkpoint verified once", counters["checkpoints_verified"] == 1)
    check("no desync", counters.get("checkpoints_desync", 0) == 0)
    check("first decision empty tier", rows[0]["drop_reason"] == "empty_tier")
    check("second decision has dH", rows[1]["dH_mean"] is not None, str(rows[1]))
    check("all rows validated", all(r["validated"] for r in rows), str(rows))
    dup_row = [r for r in rows if r["decision"] == "NOOP"][0]
    check("NOOP row computed against 2 items", dup_row["n_items_logged"] == 2)
    check("NOOP dH computed", dup_row["dH_mean"] is not None)


def test_noop_not_applied():
    print("[suppressed writes never applied]")
    t = "memory_kv_prereq_0-customer-0"
    t2 = "memory_kv_prereq_1-customer-0"
    records = [
        rehydrate(t, "kv", 0),
        decision(t, "kv", "core", "a_key", "a key: v", n_items=0),
        observe(t, "kv", "core", "a_key", "a key: v"),
        # Governed NOOP: suppressed, no observe event follows.
        decision(t, "kv", "core", "a_key", "a key: v",
                 dec="NOOP", reason="redundant", n_items=1),
        # Next rehydrate confirms the NOOP did NOT grow the store.
        rehydrate(t2, "kv", 1),
        decision(t2, "kv", "core", "b_key", "b key: w", n_items=1),
    ]
    rows, counters = run(records)
    check("checkpoint after NOOP verifies at n=1", counters["checkpoints_verified"] == 1)
    check("zero drops", counters.get("dropped_desync", 0) == 0)
    check("post-checkpoint decision computable",
          rows[-1]["computable"] and rows[-1]["dH_mean"] is not None)


def test_desync_invalidation():
    print("[desync retroactive invalidation]")
    t = "memory_kv_prereq_0-student-0"
    t2 = "memory_kv_prereq_1-student-0"
    t3 = "memory_kv_prereq_2-student-0"
    records = [
        rehydrate(t, "kv", 0),
        decision(t, "kv", "core", "k_one", "k one: 1", n_items=0),
        observe(t, "kv", "core", "k_one", "k one: 1"),
        decision(t, "kv", "core", "k_two", "k two: 2", n_items=1),
        observe(t, "kv", "core", "k_two", "k two: 2"),
        # An unlogged remove happened in the live run: snapshot says 1, we carry 2.
        rehydrate(t2, "kv", 1),
        decision(t2, "kv", "core", "k_three", "k three: 3", n_items=1),
        # Fresh restart resyncs the chain.
        rehydrate(t3, "kv", 0),
        decision(t3, "kv", "core", "k_four", "k four: 4", n_items=0),
    ]
    rows, counters = run(records)
    check("desync detected", counters["checkpoints_desync"] == 1)
    seg1 = [r for r in rows if r["test_id"] == t]
    check("pre-desync rows retroactively invalidated",
          all(r["validated"] is False and not r["computable"] for r in seg1), str(seg1))
    seg2 = [r for r in rows if r["test_id"] == t2]
    check("mid-desync decision dropped", seg2[0]["drop_reason"] == "desync")
    seg3 = [r for r in rows if r["test_id"] == t3]
    check("post-reset decision computable again", seg3[0]["computable"])
    check("dropped count explicit", counters["dropped_desync"] == 3, str(dict(counters)))


def test_remove_clear_events():
    print("[observe_remove / observe_clear]")
    t = "memory_vector_prereq_0-healthcare-0"
    t2 = "memory_vector_prereq_1-healthcare-0"
    records = [
        rehydrate(t, "vector", 0),
        observe(t, "vector", "core", "0", "Fact zero."),
        observe(t, "vector", "core", "1", "Fact one."),
        observe(t, "vector", "archival", "2", "Fact two."),
        {"event": "observe_remove", "test_id": t, "backend": "vector",
         "tier": "core", "ref": "1", "n_items": 2},
        {"event": "observe_clear", "test_id": t, "backend": "vector",
         "tier": "archival", "n_items": 1},
        rehydrate(t2, "vector", 1),  # 2 core - 1 removed, archival cleared -> 1
        decision(t2, "vector", "core", None, "A new vector fact.", n_items=1),
    ]
    rows, counters = run(records)
    check("remove+clear applied (checkpoint verifies)",
          counters["checkpoints_verified"] == 1, str(dict(counters)))
    check("decision after mutations computable",
          rows[0]["computable"] and rows[0]["dH_mean"] is not None, str(rows[0]))


def test_n_items_crosscheck():
    print("[n_items cross-check]")
    t = "memory_kv_prereq_0-finance-0"
    records = [
        rehydrate(t, "kv", 0),
        observe(t, "kv", "core", "k_a", "k a: 1"),
        # Logged mirror said 5 items; we carry 1 -> mid-entry drift detected.
        decision(t, "kv", "core", "k_b", "k b: 2", n_items=5),
        decision(t, "kv", "core", "k_c", "k c: 3", n_items=5),
    ]
    rows, counters = run(records)
    check("mismatch dropped", counters["dropped_n_items_mismatch"] == 1)
    check("chain desynced after mismatch", rows[1]["drop_reason"] == "desync")
    check("nothing usable", counters.get("decisions_computable", 0) == 0)


def test_multi_scenario_interleaving():
    print("[interleaved scenarios]")
    a, b = "memory_kv_prereq_0-customer-0", "memory_kv_prereq_0-student-0"
    a2, b2 = "memory_kv_prereq_1-customer-0", "memory_kv_prereq_1-student-0"
    records = [
        rehydrate(a, "kv", 0),
        rehydrate(b, "kv", 0),
        observe(a, "kv", "core", "cust_key", "cust key: 1"),
        observe(b, "kv", "core", "stud_key_x", "stud key x: 1"),
        observe(b, "kv", "core", "stud_key_y", "stud key y: 2"),
        rehydrate(a2, "kv", 1),  # customer carries 1
        rehydrate(b2, "kv", 2),  # student carries 2
        decision(a2, "kv", "core", "cust_two", "cust two: 2", n_items=1),
        decision(b2, "kv", "core", "stud_key_z", "stud key z: 3", n_items=2),
    ]
    rows, counters = run(records)
    check("both chains verified independently", counters["checkpoints_verified"] == 2)
    check("both decisions computable",
          all(r["computable"] and r["dH_mean"] is not None for r in rows), str(rows))


def test_vector_and_kv_backends_isolated():
    print("[same scenario name, different backends]")
    kv, vec = "memory_kv_prereq_0-customer-0", "memory_vector_prereq_0-customer-0"
    records = [
        rehydrate(kv, "kv", 0),
        rehydrate(vec, "vector", 0),
        observe(kv, "kv", "core", "only_kv", "only kv: 1"),
        rehydrate("memory_kv_prereq_1-customer-0", "kv", 1),
        rehydrate("memory_vector_prereq_1-customer-0", "vector", 0),
    ]
    rows, counters = run(records)
    check("backends keyed separately", counters["checkpoints_verified"] == 1
          and counters["checkpoints_reset"] == 3, str(dict(counters)))


def test_summary():
    print("[summarize]")
    t = "memory_kv_prereq_0-customer-0"
    records = [rehydrate(t, "kv", 0)]
    state_n = 0
    # Build an alternating dup/novel stream with varying sim_max.
    for i in range(6):
        ref, text = f"key_{i}", f"key {i}: value number {i * 7}"
        records.append(decision(t, "kv", "core", ref, text,
                                n_items=state_n, sim_max=0.2 + 0.1 * i, r=0.9 - 0.1 * i))
        records.append(observe(t, "kv", "core", ref, text))
        state_n += 1
    records.append(decision(t, "kv", "core", "key_0", "key 0: value number 0",
                            dec="NOOP", reason="redundant",
                            n_items=state_n, sim_max=0.99, r=0.01))
    rows, counters = run(records)
    summary = summarize(rows, counters)
    kvs = summary["backends"]["kv"]
    check("usable counted", summary["accounting"]["decisions_usable_for_correlation"] == 6,
          str(summary["accounting"]))
    check("rho computed", kvs["spearman_sim_max_vs_dH_mean"] is not None)
    check("dup median sim > nondup",
          kvs["duplicate_median_sim_max"] > kvs["nonduplicate_median_sim_max"])
    check("suppressed NOOP with dH>=0 counted",
          kvs["suppressed_noops"] == 1 and kvs["suppressed_noops_dH_ge_0"] in (0, 1))
    check("noop region size counted", kvs["noop_region_size"] == 1, str(kvs))
    check("vector side empty but present", summary["backends"]["vector"]["n_usable"] == 0)


def main():
    test_helpers()
    test_basic_chain()
    test_noop_not_applied()
    test_desync_invalidation()
    test_remove_clear_events()
    test_n_items_crosscheck()
    test_multi_scenario_interleaving()
    test_vector_and_kv_backends_isolated()
    test_summary()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
