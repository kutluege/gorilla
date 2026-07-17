"""
Offline tests for the SS5 atomic core->archival move (Plan 3 Step 4 / G6b-G6c,
risk R5): archive-add strictly precedes core-remove, moves are planned only
when the archive-add preflights as accepting, and the mirror ends in the moved
state with no intermediate data loss.

Run:  python bfcl_eval/scripts/test_eviction.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bfcl_eval.model_handler.middleware.governance_filter import (  # noqa: E402
    GovConfig,
    GovernanceSession,
    KV_ADD_SUCCESS,
    KV_REMOVE_SUCCESS,
)

TMP = tempfile.mkdtemp(prefix="gov_evict_test_")
PASS, FAIL = 0, 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


class EntailScorer:
    def probs_batch(self, pairs):
        return [(0.05, 0.05, 0.90) for _ in pairs]


FULL_CORE = {f"user_fact_{i}": f"identity fact number {i}" for i in range(7)}


def make_session(log_file, archival=None, **cfg_overrides):
    cfg = GovConfig(log_dir=TMP, log_file=log_file, p_enabled=True, p_shadow=False)
    for k, v in cfg_overrides.items():
        setattr(cfg, k, v)
    return GovernanceSession(
        cfg=cfg, backend="kv",
        test_id="memory_kv_prereq_1-test-1",
        snapshot={"core_memory": dict(FULL_CORE),
                  "archival_memory": archival or {}},
        snapshot_path="<fabricated>",
        nli_scorer=EntailScorer(),
        sidecar_path=str(Path(TMP) / f"{log_file}_gov_state.json"),
    )


def read_events(log_file, event):
    with open(Path(TMP) / log_file, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if r["event"] == event]


IDENTITY_ADD = "core_memory_add(key='user_diet', value='vegetarian')"


def run():
    print("[live atomic move on a full core]")
    s = make_session("evict_live.jsonl")
    s.user_text = "I am vegetarian."
    governed = s.govern_calls([IDENTITY_ADD])
    check("one call expanded to three", len(governed) == 3, str(governed))
    arch_i = next((i for i, c in enumerate(governed)
                   if c.startswith("archival_memory_add(")), None)
    rem_i = next((i for i, c in enumerate(governed)
                  if c.startswith("core_memory_remove(")), None)
    add_i = next((i for i, c in enumerate(governed) if c == IDENTITY_ADD), None)
    check("archive-add strictly precedes core-remove (atomic order)",
          None not in (arch_i, rem_i, add_i) and arch_i < rem_i < add_i,
          str(governed))
    victim_key = governed[rem_i].split("key='")[1].split("'")[0]
    check("archive and remove target the SAME victim",
          f"key='{victim_key}'" in governed[arch_i])
    ev = read_events("evict_live.jsonl", "placement")
    check("eviction_move logged with victim ref",
          any(r["mechanism"] == "eviction_move" and r["victim_ref"] == victim_key
              and r["movable"] for r in ev), str(ev))

    print("[mirror ends in the moved state (no intermediate loss)]")
    results = [
        json.dumps({"status": KV_ADD_SUCCESS["archival"]}),
        json.dumps({"status": KV_REMOVE_SUCCESS}),
        json.dumps({"status": KV_ADD_SUCCESS["core"]}),
    ]
    s.patch_results(results, list(governed))
    check("victim now in archival", victim_key in s.cache.items["archival"])
    check("victim gone from core", victim_key not in s.cache.items["core"])
    check("new item landed in core", "user_diet" in s.cache.items["core"])
    check("no data lost: total = 7 moved-through + 1 new",
          s.cache.total_size() == 8, str(s.cache.total_size()))

    print("[simulated archive-add failure leaves core intact]")
    s2 = make_session("evict_fail.jsonl")
    s2.user_text = "I am vegetarian."
    governed2 = s2.govern_calls([IDENTITY_ADD])
    victim2 = governed2[1].split("key='")[1].split("'")[0]
    fail_results = [
        json.dumps({"error": "Archival memory is full."}),
        json.dumps({"error": "Key not removed."}),
        json.dumps({"status": KV_ADD_SUCCESS["core"]}),
    ]
    s2.patch_results(fail_results, list(governed2))
    check("failed archive-add: victim still in core (mirror truthful)",
          victim2 in s2.cache.items["core"])
    check("failed archive-add: nothing phantom in archival",
          victim2 not in s2.cache.items["archival"])

    print("[archival unavailable: move is NOT planned]")
    full_archival = {f"arch_{i}": f"archived note {i}" for i in range(50)}
    s3 = make_session("evict_blocked.jsonl", archival=full_archival)
    s3.user_text = "I am vegetarian."
    governed3 = s3.govern_calls([IDENTITY_ADD])
    check("no expansion when archival is full",
          governed3 == [IDENTITY_ADD], str(len(governed3)))
    ev3 = read_events("evict_blocked.jsonl", "placement")
    check("unmovable logged with archival_unavailable risk",
          any(r["mechanism"] == "eviction_move" and not r["movable"]
              and r["risk"] == "archival_unavailable" for r in ev3))

    print("[shadow: move planned, nothing mutated]")
    s4 = make_session("evict_shadow.jsonl", p_shadow=True)
    s4.user_text = "I am vegetarian."
    governed4 = s4.govern_calls([IDENTITY_ADD])
    check("shadow: call list unchanged", governed4 == [IDENTITY_ADD])
    ev4 = read_events("evict_shadow.jsonl", "placement")
    check("shadow: eviction_move logged as shadow",
          any(r["mechanism"] == "eviction_move" and r["shadow"] for r in ev4))

    print("[pending-index remap across expansions]")
    s5 = make_session("evict_remap.jsonl")
    s5.user_text = "I am vegetarian."
    governed5 = s5.govern_calls(
        ["core_memory_clear()", IDENTITY_ADD]  # clear blocks (NOOP) + move expands
    )
    check("final list: decoy + archive + remove + add",
          len(governed5) == 4 and governed5[0] == "core_memory_retrieve_all()",
          str(governed5))
    check("NOOP pending remapped to final index 0",
          set(s5._pending) == {0} and s5._pending[0]["mode"] == "noop",
          str(s5._pending))

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run())
