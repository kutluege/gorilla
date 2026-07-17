"""
Offline tests for parse_wrra.py (Plan 2 Step 1 acceptance).

Builds a fixture result/score/snapshot tree in a temp dir and checks:
  - W/R counts (attempted vs resolved) from inference_log call/tool pairing
  - accuracy reconstruction from header+failures-only score files
  - dead-chain detector: 0-entry final snapshot, 0-write prereq, missing snapshot
  - store-aware snapshot counting (KV flat dict vs Vector {next_id, store})

Run:  python bfcl_eval/scripts/test_parse_wrra.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bfcl_eval.scripts.parse_wrra import (  # noqa: E402
    count_snapshot_entries,
    iter_call_pairs,
    parse_arm,
    parse_score_file,
    scenario_from_id,
    tool_result_ok,
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
# Fixture construction
# ---------------------------------------------------------------------------

def step(decoded_calls, tool_contents):
    """One step_N message list in real emission order."""
    msgs = [{"role": "assistant", "content": "<tool_call>...</tool_call>"}]
    msgs.append({
        "role": "handler_log",
        "content": "Successfully decoded model response.",
        "model_response_decoded": decoded_calls,
    })
    msgs.extend({"role": "tool", "content": c} for c in tool_contents)
    return msgs


def entry(test_id, turns):
    return {"id": test_id, "result": [["ans"]], "inference_log": turns}


OK_WRITE = json.dumps({"status": "Key-value pair added."})
ERR = json.dumps({"error": "Key already exists."})
OK_READ = json.dumps({"status": "ok", "value": "x"})


def build_fixture(root):
    slug = "Test_Model-FC"
    for backend in ("kv", "vector"):
        bdir = root / "result" / slug / "agentic" / "memory" / backend
        (bdir / "memory_snapshot").mkdir(parents=True)
        sdir = root / "score" / slug / "agentic" / "memory" / backend
        sdir.mkdir(parents=True)

        # --- prereq phase ---------------------------------------------------
        # customer: 3 writes attempted, 1 errors -> 2 resolved; 1 prereq read;
        #           1 destructive resolved; 1 unpaired call (no tool msg).
        # student:  read-only prereq -> zero resolved writes (dead chain rule 2).
        # finance:  writes fine, but final snapshot will be empty (dead rule 1).
        # notetaker: writes fine, snapshot file missing entirely (dead rule 3).
        prereq = [
            entry(f"memory_{backend}_prereq_0-customer-0", [
                {"step_0": step(["core_memory_add(key='a',value='1')"], [OK_WRITE])},
                {"step_0": step(["core_memory_add(key='b',value='2')",
                                 "core_memory_add(key='a',value='dup')"],
                                [OK_WRITE, ERR]),
                 "step_1": step(["core_memory_retrieve(key='a')"], [OK_READ])},
            ]),
            entry(f"memory_{backend}_prereq_1-customer-1", [
                {"step_0": step(["core_memory_remove(key='b')"], [OK_WRITE]),
                 "step_1": step(["archival_memory_add(key='c',value='3')"], [])},
            ]),
            entry(f"memory_{backend}_prereq_2-student-0", [
                {"step_0": step(["core_memory_retrieve_all()"], [OK_READ])},
            ]),
            entry(f"memory_{backend}_prereq_3-finance-0", [
                {"step_0": step(["core_memory_add(key='f',value='9')"], [OK_WRITE])},
            ]),
            entry(f"memory_{backend}_prereq_4-notetaker-0", [
                {"step_0": step(["core_memory_add(key='n',value='8')"], [OK_WRITE])},
            ]),
        ]
        with open(bdir / f"BFCL_v4_memory_{backend}_prereq_result.json", "w",
                  encoding="utf-8") as f:
            f.writelines(json.dumps(e) + "\n" for e in prereq)

        # --- question phase -------------------------------------------------
        questions = [
            entry(f"memory_{backend}_0-customer-0", [
                {"step_0": step(["core_memory_retrieve(key='a')"], [OK_READ])},
            ]),
            entry(f"memory_{backend}_1-customer-1", [
                {"step_0": step([], [])},
            ]),
            entry(f"memory_{backend}_2-student-0", [
                {"step_0": step(["core_memory_retrieve_all()"], [OK_READ])},
            ]),
            entry(f"memory_{backend}_3-finance-0", [
                {"step_0": step([], [])},
            ]),
        ]
        with open(bdir / f"BFCL_v4_memory_{backend}_result.json", "w",
                  encoding="utf-8") as f:
            f.writelines(json.dumps(e) + "\n" for e in questions)

        # --- snapshots ------------------------------------------------------
        if backend == "kv":
            alive = {"core_memory": {"a": "1"}, "archival_memory": {"c": "3"}}
            empty = {"core_memory": {}, "archival_memory": {}}
        else:
            alive = {"core_memory": {"next_id": 2, "store": {"0": "t0", "1": "t1"}},
                     "archival_memory": {"next_id": 0, "store": {}}}
            empty = {"core_memory": {"next_id": 0, "store": {}},
                     "archival_memory": {"next_id": 0, "store": {}}}
        snap = bdir / "memory_snapshot"
        json.dump(alive, open(snap / "customer_final.json", "w", encoding="utf-8"))
        json.dump(alive, open(snap / "student_final.json", "w", encoding="utf-8"))
        json.dump(empty, open(snap / "finance_final.json", "w", encoding="utf-8"))
        # notetaker_final.json deliberately absent

        # --- scores: kv gets header+failures; vector score file is MISSING --
        if backend == "kv":
            rows = [
                {"accuracy": 0.75, "correct_count": 3, "total_count": 4},
                {"id": f"memory_{backend}_1-customer-1", "valid": False,
                 "error": {"error_type": "agentic:answer_not_found"}},
            ]
            with open(sdir / f"BFCL_v4_memory_{backend}_score.json", "w",
                      encoding="utf-8") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
    return slug


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run():
    tmp = Path(tempfile.mkdtemp(prefix="wrra_fixture_"))
    try:
        slug = build_fixture(tmp)
        records = parse_arm(tmp / "result", tmp / "score", slug,
                            ["kv", "vector"], arm="baseline")
        scen = {(r["backend"], r["scenario"]): r
                for r in records if r["record"] == "wrra_scenario"}
        qs = {(r["backend"], r["id"]): r
              for r in records if r["record"] == "wrra_question"}
        summ = {r["backend"]: r
                for r in records if r["record"] == "wrra_arm_summary"}

        print("[helpers]")
        check("scenario_from_id prereq", scenario_from_id(
            "memory_kv_prereq_0-customer-0") == "customer")
        check("scenario_from_id question", scenario_from_id(
            "memory_vector_12-notetaker-3") == "notetaker")
        check("tool_result_ok success", tool_result_ok(OK_WRITE))
        check("tool_result_ok error", not tool_result_ok(ERR))
        check("tool_result_ok non-json", tool_result_ok("plain text status"))

        print("[call pairing]")
        pairs = list(iter_call_pairs([
            {"step_0": step(["core_memory_add(key='x',value='1')",
                             "core_memory_add(key='y',value='2')"],
                            [OK_WRITE, ERR])},
        ]))
        check("two calls paired in order",
              pairs == [("core_memory_add", True), ("core_memory_add", False)],
              str(pairs))

        print("[write/retrieve counts -- kv customer]")
        c = scen[("kv", "customer")]
        check("write attempted = 4", c["write_calls_attempted"] == 4,
              str(c["write_calls_attempted"]))
        check("write resolved = 2 (1 error, 1 unpaired)",
              c["write_calls_resolved"] == 2, str(c["write_calls_resolved"]))
        check("destructive resolved = 1", c["destructive_calls_resolved"] == 1)
        check("prereq retrieves = 1", c["retrieve_calls_prereq"] == 1)
        check("unpaired flagged", c["unpaired_calls"] == 1)
        check("question retrieves = 1", c["retrieve_calls_question"] == 1)
        check("recall = 1/2", abs(c["recall"] - 0.5) < 1e-9, str(c["recall"]))

        print("[dead-chain detector]")
        check("customer alive", not scen[("kv", "customer")]["chain_dead"])
        check("0-write prereq dead (student)",
              scen[("kv", "student")]["chain_dead"]
              and scen[("kv", "student")]["dead_reason"] == "zero_resolved_writes")
        check("0-entry snapshot dead (finance)",
              scen[("kv", "finance")]["chain_dead"]
              and scen[("kv", "finance")]["dead_reason"] == "empty_snapshot")
        check("missing snapshot dead (notetaker)",
              scen[("kv", "notetaker")]["chain_dead"]
              and scen[("kv", "notetaker")]["dead_reason"] == "missing_snapshot")
        check("summary dead count = 3", summ["kv"]["chains_dead"] == 3,
              str(summ["kv"]["dead_scenarios"]))

        print("[store-aware snapshot counting]")
        kv_c = scen[("kv", "customer")]
        check("kv flat-dict entries", (kv_c["snapshot_core_entries"],
                                       kv_c["snapshot_archival_entries"]) == (1, 1))
        vec_c = scen[("vector", "customer")]
        check("vector store entries", (vec_c["snapshot_core_entries"],
                                       vec_c["snapshot_archival_entries"]) == (2, 0))
        check("count_snapshot_entries missing -> None",
              count_snapshot_entries(tmp / "nope.json") is None)

        print("[accuracy reconstruction]")
        hdr, failed = parse_score_file(
            tmp / "score" / slug / "agentic" / "memory" / "kv"
            / "BFCL_v4_memory_kv_score.json")
        check("score header parsed", hdr["total_count"] == 4)
        check("failed set", failed == {"memory_kv_1-customer-1"})
        check("failed question marked incorrect",
              qs[("kv", "memory_kv_1-customer-1")]["correct"] is False)
        check("absent-from-score marked correct",
              qs[("kv", "memory_kv_0-customer-0")]["correct"] is True)
        check("customer accuracy = 1/2",
              abs(scen[("kv", "customer")]["accuracy"] - 0.5) < 1e-9)
        check("missing score file -> accuracy None",
              scen[("vector", "customer")]["accuracy"] is None
              and qs[("vector", "memory_vector_0-customer-0")]["correct"] is None)

        print("[record shape]")
        check("4 scenarios per backend",
              sum(1 for k in scen if k[0] == "kv") == 4)
        check("question records = 4 per backend",
              sum(1 for k in qs if k[0] == "kv") == 4)
        check("summary carries score header",
              summ["kv"]["score_header"]["accuracy"] == 0.75)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run())
