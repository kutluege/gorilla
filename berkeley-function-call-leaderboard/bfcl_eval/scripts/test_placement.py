"""
Offline tests for the SS5 placement layer (Plan 3 Step 4 / G6a-G6c + G8).

Run:  python bfcl_eval/scripts/test_placement.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bfcl_eval.model_handler.middleware.governance_filter as gf  # noqa: E402
from bfcl_eval.model_handler.middleware.governance_filter import (  # noqa: E402
    GovConfig,
    GovernanceSession,
    MemoryItem,
    _normalize_for_match,
    build_candidate,
)
from bfcl_eval.model_handler.middleware.placement import (  # noqa: E402
    classify_category,
    kv_value_of,
    pick_eviction_victim,
    to_archival_call,
    value_scores,
    verbatim_final_rewrite,
)
from bfcl_eval.model_handler.middleware.recsum_blobdiff import (  # noqa: E402
    blobdiff_verdict,
    split_propositions,
)

TMP = tempfile.mkdtemp(prefix="gov_place_test_")
PASS, FAIL = 0, 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def item(ref, text, vec, turn=0, tier="core"):
    v = np.asarray(vec, dtype=np.float64)
    v = v / np.linalg.norm(v)
    return MemoryItem(ref=ref, text=text, emb_whitened=v, turn_written=turn, tier=tier)


class EntailScorer:
    def __init__(self, entail):
        self.entail = entail

    def probs_batch(self, pairs):
        return [(0.05, 1 - 0.05 - self.entail, self.entail) for _ in pairs]


def make_session(log_file, snapshot=None, **cfg_overrides):
    cfg = GovConfig(log_dir=TMP, log_file=log_file)
    for k, v in cfg_overrides.items():
        setattr(cfg, k, v)
    return GovernanceSession(
        cfg=cfg, backend="kv",
        test_id="memory_kv_prereq_1-test-1",
        snapshot=snapshot or {"core_memory": {"user_name": "Michael"},
                              "archival_memory": {}},
        snapshot_path="<fabricated>",
        nli_scorer=EntailScorer(0.9),
        sidecar_path=str(Path(TMP) / f"{log_file}_gov_state.json"),
    )


def read_events(log_file, event):
    with open(Path(TMP) / log_file, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if r["event"] == event]


def run():
    print("[G6b category classifier]")
    check("identity: name", classify_category("user name: Michael") == "identity")
    check("identity: allergy", classify_category("The user is allergic to nuts")
          == "identity")
    check("event_detail: meeting", classify_category(
        "meeting notes: budget review on March 3") == "event_detail")
    check("event_detail: order", classify_category(
        "order 5521 shipped to warehouse") == "event_detail")

    print("[G6c value formula (hand-checked)]")
    # Orthogonal pair, same age: uniqueness 1.0 each -> value = w1 + w2/(1+dt).
    a = item("a", "a", [1, 0], turn=0)
    b = item("b", "b", [0, 1], turn=0)
    vs = value_scores([a, b], current_step=1, w1=0.6, w2=0.4)
    check("orthogonal pair: 0.6*1 + 0.4*(1/2) = 0.8",
          all(abs(v - 0.8) < 1e-9 for v in vs), str(vs))
    # Near-duplicate pair loses uniqueness.
    c = item("c", "c", [1, 0.01], turn=0)
    vs2 = value_scores([a, c, b], current_step=1, w1=0.6, w2=0.4)
    check("near-duplicate scores below orthogonal",
          vs2[0] < 0.3 and vs2[1] < 0.3 and vs2[2] > 0.7, str(vs2))
    # Recency: newer identical-uniqueness item scores higher.
    old = item("old", "old", [1, 0], turn=0)
    new = item("new", "new", [0, 1], turn=9)
    vs3 = value_scores([old, new], current_step=10, w1=0.6, w2=0.4)
    check("recency favors the newer item", vs3[1] > vs3[0], str(vs3))
    check("victim = lowest value (the near-duplicate, tie-break by ref)",
          pick_eviction_victim([a, c, b], 1, 0.6, 0.4).ref == "a")
    check("empty items -> no victim", pick_eviction_victim([], 1, 0.6, 0.4) is None)

    print("[G6a verbatim final rewrite builder]")
    cand = build_candidate("kv", "core_memory_add(key='user_age', value='mid thirties')")
    rw = verbatim_final_rewrite(cand, ["35"], _normalize_for_match, 300)
    check("missing verbatim value appended",
          rw is not None and "35" in rw[0] and rw[1] == ["35"], str(rw))
    check("rewritten call stays parseable",
          build_candidate("kv", rw[0]) is not None)
    cand_ok = build_candidate("kv", "core_memory_add(key='user_age', value='35 years')")
    check("nothing missing -> None",
          verbatim_final_rewrite(cand_ok, ["35"], _normalize_for_match, 300) is None)
    check("length cap suppresses the rewrite",
          verbatim_final_rewrite(cand, ["35"], _normalize_for_match, 10) is None)
    check("kv_value_of recovers the raw value",
          kv_value_of(item("k", "user age: 35", [1, 0])) == "35")

    print("[G6b routing rewrite builder]")
    ev = build_candidate("kv", "core_memory_add(key='meeting_log', value='March 3 budget')")
    check("core add -> archival add",
          to_archival_call(ev) == "archival_memory_add(key='meeting_log', value='March 3 budget')")
    rep = build_candidate("kv", "core_memory_replace(key='k', value='v')")
    check("non-add never routed", to_archival_call(rep) is None)

    print("[session: shadow logs, never mutates]")
    s = make_session("place_shadow.jsonl", p_enabled=True, p_shadow=True)
    s.user_text = "The meeting about the budget is on March 3."
    calls = ["core_memory_add(key='meeting_log', value='budget meeting')"]
    governed = s.govern_calls(list(calls))
    check("shadow: calls unchanged", governed == calls, str(governed))
    routing = read_events("place_shadow.jsonl", "placement")
    check("shadow: routing planned + logged",
          any(r["mechanism"] == "routing" and r["shadow"] for r in routing))
    check("shadow: verbatim final planned + logged",
          any(r["mechanism"] == "verbatim_final" and "March 3" in str(r["missing_values"])
              for r in routing), str(routing))

    print("[session: live routing + verbatim rewrite]")
    s = make_session("place_live.jsonl", p_enabled=True, p_shadow=False)
    s.user_text = "The meeting about the budget is on March 3."
    governed = s.govern_calls(list(calls))
    check("live: event_detail core add rewritten to archival",
          governed[0].startswith("archival_memory_add("), str(governed))
    check("live: missing verbatim value appended to outgoing call",
          "March 3" in governed[0], str(governed))
    check("live: pending rewrite mode set (real result will flow)",
          s._pending[0]["mode"] == "rewrite")

    print("[flag off: byte-identical to Plan 2]")
    s = make_session("place_off.jsonl")  # p_enabled defaults False
    s.user_text = "The meeting about the budget is on March 3."
    governed = s.govern_calls(list(calls))
    check("disabled: calls unchanged", governed == calls)
    check("disabled: zero placement events",
          read_events("place_off.jsonl", "placement") == [])

    print("[G8 rec_sum blob-diff surrogate (standalone)]")
    old_blob = "The user is 35. The meeting is on March 3. Allergic to nuts."
    props = split_propositions(old_blob)
    check("propositions split", len(props) == 3, str(props))
    v = blobdiff_verdict(old_blob, "The user is 35.", EntailScorer(0.2))
    check("lost propositions -> reject with suggestion",
          not v.ok and len(v.lost) == 3 and "append" in v.suggestion)
    v2 = blobdiff_verdict(old_blob, "same content restated", EntailScorer(0.9))
    check("all entailed -> ok", v2.ok and v2.lost == [])
    v3 = blobdiff_verdict(old_blob, "", EntailScorer(0.9))
    check("clear (empty new blob) always rejected", not v3.ok and len(v3.lost) == 3)
    check("rec_sum stays out of governed backends",
          __import__("bfcl_eval.model_handler.local_inference.qwen_gov",
                     fromlist=["GOVERNED_BACKENDS"]).GOVERNED_BACKENDS
          == {"kv", "vector"})

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run())
