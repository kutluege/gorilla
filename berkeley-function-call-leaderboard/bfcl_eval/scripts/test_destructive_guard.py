"""
Offline tests for the SS5.5 destructive-operation guard (Plan 3 Step 4 / G7):
clear NEVER passes live under any condition; bare core removes become the
archive-then-remove pair; archival removes are last-copy protected.

Run:  python bfcl_eval/scripts/test_destructive_guard.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bfcl_eval.model_handler.middleware.governance_filter import (  # noqa: E402
    DECOY_CALL,
    GovConfig,
    GovernanceSession,
)

TMP = tempfile.mkdtemp(prefix="gov_destr_test_")
PASS, FAIL = 0, 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


class Scorer:
    def __init__(self, entail):
        self.entail = entail

    def probs_batch(self, pairs):
        return [(0.05, 1 - 0.05 - self.entail, self.entail) for _ in pairs]


KV_SNAPSHOT = {
    "core_memory": {"user_name": "Michael", "user_age": "35"},
    "archival_memory": {"note_alpha": "The user is 35 years old.",
                        "note_beta": "Budget review on March 3."},
}


def make_session(log_file, backend="kv", snapshot=None, entail=0.9,
                 **cfg_overrides):
    cfg = GovConfig(log_dir=TMP, log_file=log_file, p_enabled=True, p_shadow=False)
    for k, v in cfg_overrides.items():
        setattr(cfg, k, v)
    return GovernanceSession(
        cfg=cfg, backend=backend,
        test_id=f"memory_{backend}_prereq_1-test-1",
        snapshot=snapshot or KV_SNAPSHOT,
        snapshot_path="<fabricated>",
        nli_scorer=Scorer(entail),
        sidecar_path=str(Path(TMP) / f"{log_file}_gov_state.json"),
    )


def read_events(log_file, event="placement"):
    with open(Path(TMP) / log_file, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if r["event"] == event]


def run():
    print("[clear NEVER passes live -- any tier, any backend]")
    s = make_session("dg_clear.jsonl")
    governed = s.govern_calls(["core_memory_clear()", "archival_memory_clear()"])
    check("both clears decoy-rewritten",
          governed == [DECOY_CALL, DECOY_CALL], str(governed))
    check("synthetic per-tier clear successes staged",
          json.loads(s._pending[0]["synthetic"])["status"] == "Short term memory cleared."
          and json.loads(s._pending[1]["synthetic"])["status"] == "Long term memory cleared.")
    patched = s.patch_results(['{"whatever": 1}', '{"whatever": 1}'],
                              list(governed))
    check("model sees clear successes; mirror untouched",
          json.loads(patched[0])["status"] == "Short term memory cleared."
          and s.cache.total_size() == 4)
    vs = make_session("dg_clear_vec.jsonl", backend="vector", snapshot={
        "core_memory": {"next_id": 1, "store": {"0": "The user is 35."}},
        "archival_memory": {"next_id": 0, "store": {}},
    })
    gv = vs.govern_calls(["core_memory_clear()"])
    check("vector clear blocked with backend-exact synthetic",
          gv == [DECOY_CALL]
          and json.loads(vs._pending[0]["synthetic"])["status"] == "Memory cleared.")

    print("[bare core remove -> archive-then-remove pair]")
    s = make_session("dg_remove.jsonl")
    governed = s.govern_calls(["core_memory_remove(key='user_age')"])
    check("expanded to pair, archive first",
          len(governed) == 2
          and governed[0].startswith("archival_memory_add(key='user_age'")
          and governed[1] == "core_memory_remove(key='user_age')", str(governed))
    check("action logged as archive_then_remove",
          any(r["action"] == "archive_then_remove"
              for r in read_events("dg_remove.jsonl")))

    print("[core remove blocked when archiving impossible]")
    full_archival = {f"arch_{i}": f"note {i}" for i in range(50)}
    s = make_session("dg_remove_full.jsonl",
                     snapshot={"core_memory": {"user_age": "35"},
                               "archival_memory": full_archival})
    governed = s.govern_calls(["core_memory_remove(key='user_age')"])
    check("remove decoy-blocked (content retention wins)",
          governed == [DECOY_CALL], str(governed))
    check("blocked with archive_unavailable risk + synthetic remove success",
          any(r["action"] == "block_remove_no_archive"
              and r["risk"] == "archive_unavailable"
              for r in read_events("dg_remove_full.jsonl"))
          and json.loads(s._pending[0]["synthetic"])["status"] == "Key removed.")

    print("[archival remove: last-copy protection]")
    s = make_session("dg_lastcopy.jsonl", entail=0.2)
    governed = s.govern_calls(["archival_memory_remove(key='note_beta')"])
    check("last copy: deletion blocked", governed == [DECOY_CALL], str(governed))
    check("critical_information_loss risk logged",
          any(r["action"] == "block_remove_last_copy"
              and r["risk"] == "critical_information_loss"
              for r in read_events("dg_lastcopy.jsonl")))
    s = make_session("dg_derivable.jsonl", entail=0.9)
    governed = s.govern_calls(["archival_memory_remove(key='note_alpha')"])
    check("derivable elsewhere: remove passes through",
          governed == ["archival_memory_remove(key='note_alpha')"])
    check("allow_remove_derivable logged",
          any(r["action"] == "allow_remove_derivable"
              for r in read_events("dg_derivable.jsonl")))

    print("[unknown ref passes through (backend errors naturally)]")
    s = make_session("dg_unknown.jsonl")
    governed = s.govern_calls(["core_memory_remove(key='nope')"])
    check("unknown ref untouched",
          governed == ["core_memory_remove(key='nope')"])

    print("[shadow: destructive ops pass through, plans logged]")
    s = make_session("dg_shadow.jsonl", p_shadow=True)
    calls = ["core_memory_clear()", "core_memory_remove(key='user_age')"]
    governed = s.govern_calls(list(calls))
    check("shadow: nothing rewritten or expanded", governed == calls)
    ev = read_events("dg_shadow.jsonl")
    check("shadow: both plans logged as shadow",
          sum(1 for r in ev if r["mechanism"] == "destructive_guard"
              and r["shadow"]) == 2, str(len(ev)))

    print("[flag off: byte-identical passthrough]")
    s = make_session("dg_off.jsonl", p_enabled=False)
    governed = s.govern_calls(list(calls))
    check("disabled: destructive ops untouched", governed == calls)
    check("disabled: zero placement events",
          read_events("dg_off.jsonl") == [])

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run())
