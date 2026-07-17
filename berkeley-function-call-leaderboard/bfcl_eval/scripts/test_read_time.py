"""
Offline tests for the SS6 read-time gate (Plan 3 Step 5 / G9): memory is
never altered (snapshot invariant), at most ONE refinement round runs, the
ambiguous set stays bounded 2-4, and shadow mode is log-only.

Run:  python bfcl_eval/scripts/test_read_time.py
"""

import copy
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bfcl_eval.model_handler.middleware.governance_filter import (  # noqa: E402
    GovConfig,
    GovernanceSession,
)
from bfcl_eval.model_handler.middleware.read_gate import (  # noqa: E402
    distinct_refinement_token,
    gate_read,
    parse_ranking,
    serialize_ranking,
)

TMP = tempfile.mkdtemp(prefix="gov_read_test_")
PASS, FAIL = 0, 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


KV_RESULT = {"ranked_results": [[3.2, "meeting_notes_march"],
                                [3.1, "meeting_notes_april"],
                                [1.0, "user_name"]]}
VEC_RESULT = {"result": [
    {"id": 0, "similarity_score": 0.82, "text": "Budget meeting on March 3."},
    {"id": 1, "similarity_score": 0.80, "text": "Budget meeting on April 9."},
    {"id": 2, "similarity_score": 0.30, "text": "The user likes coffee."},
]}


def counting_rerank(counter):
    def rerank(backend, candidates, query):
        counter.append(query)
        # deterministic: candidate containing the appended token wins by 1.0
        token = query.split()[-1]
        return sorted(
            [((1.0 if token in text.lower() else 0.0), ref, text)
             for _, ref, text in candidates],
            key=lambda t: -t[0],
        )
    return rerank


def run():
    print("[parsers + serializers round-trip]")
    kv_ranked = parse_ranking("kv", KV_RESULT)
    check("kv ranking parsed descending",
          [r[1] for r in kv_ranked] == ["meeting_notes_march",
                                        "meeting_notes_april", "user_name"])
    check("kv serialize round-trips",
          json.loads(serialize_ranking("kv", kv_ranked)) == KV_RESULT)
    vec_ranked = parse_ranking("vector", VEC_RESULT)
    check("vector ranking parsed", [r[1] for r in vec_ranked] == ["0", "1", "2"])
    check("unranked payload -> None", parse_ranking("kv", {"status": "x"}) is None)

    print("[refinement vocabulary rule]")
    cands = [(3.2, "a", "meeting notes march"), (3.1, "b", "meeting notes april")]
    check("distinct token overlapping the query picked",
          distinct_refinement_token("what was decided in the march meeting",
                                    cands) == "march")
    check("no overlap -> None",
          distinct_refinement_token("what about budgets", cands) is None)

    print("[gate: clear margin -> top-1]")
    out = gate_read("kv", "march meeting", {"ranked_results": [[5.0, "a"], [1.0, "b"]]},
                    margin_thr=0.5, set_max=4)
    check("clear margin returns top-1 only",
          out.action == "top1_pass"
          and json.loads(out.new_payload)["ranked_results"] == [[5.0, "a"]])
    check("zero refinement rounds on the clear path", out.refinement_rounds == 0)

    print("[gate: ambiguous -> ONE refinement, bounded set]")
    counter = []
    out = gate_read("kv", "what was decided in the march meeting", KV_RESULT,
                    margin_thr=0.5, set_max=4,
                    rerank_fn=counting_rerank(counter))
    check("ambiguous set bounded to the delta-neighborhood (2 entries)",
          out.ambiguous_refs == ["meeting_notes_march", "meeting_notes_april"])
    check("EXACTLY one refinement round", len(counter) == 1 and
          out.refinement_rounds == 1, str(counter))
    check("refined query = query + distinct token",
          counter[0].endswith(" march"))
    check("resolved -> refined top-1 returned",
          out.action == "refined_top1"
          and json.loads(out.new_payload)["ranked_results"][0][1]
          == "meeting_notes_march")

    print("[gate: unresolvable -> whole bounded set]")
    out = gate_read("kv", "what about budgets", KV_RESULT,
                    margin_thr=0.5, set_max=4)
    check("no usable token -> return_set with the bounded set",
          out.action == "return_set"
          and len(json.loads(out.new_payload)["ranked_results"]) == 2)
    wide = {"ranked_results": [[3.0 - 0.01 * i, f"key_{i}"] for i in range(8)]}
    out = gate_read("kv", "zzz", wide, margin_thr=0.5, set_max=9)
    check("set_max clamped to 4",
          len(json.loads(out.new_payload)["ranked_results"]) <= 4)
    out = gate_read("kv", "q", {"ranked_results": [[1.0, "only"]]},
                    margin_thr=0.5, set_max=4)
    check("singleton ranking passes untouched",
          out.action == "pass" and out.new_payload is None)

    print("[session: shadow logs, payload untouched, memory invariant]")
    def make_session(log_file, **cfg_overrides):
        cfg = GovConfig(log_dir=TMP, log_file=log_file, read_enabled=True,
                        read_margin=0.5)
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)
        return GovernanceSession(
            cfg=cfg, backend="kv",
            test_id="memory_kv_5-test-1",
            snapshot={"core_memory": {"meeting_notes_march": "budget +5%",
                                      "meeting_notes_april": "budget -2%"},
                      "archival_memory": {}},
            snapshot_path="<fabricated>",
            sidecar_path=str(Path(TMP) / f"{log_file}_gov_state.json"),
        )

    call = "core_memory_key_search(query='what was decided in the march meeting', k=5)"
    raw = json.dumps(KV_RESULT)

    s = make_session("read_shadow.jsonl", read_shadow=True)
    s.govern_calls([call])
    before = copy.deepcopy(s.cache.items)
    patched = s.patch_results([raw], [call])
    check("shadow: payload untouched", patched == [raw])
    with open(Path(TMP) / "read_shadow.jsonl", "r", encoding="utf-8") as f:
        events = [json.loads(l) for l in f if l.strip()]
    rg = [e for e in events if e["event"] == "read_gate"]
    check("shadow: read_gate event logged with outcome",
          len(rg) == 1 and rg[0]["shadow"] and rg[0]["action"] == "refined_top1",
          str(rg))
    check("memory invariant (mirror byte-identical)",
          {t: set(d) for t, d in s.cache.items.items()}
          == {t: set(d) for t, d in before.items()})

    s = make_session("read_live.jsonl", read_shadow=False)
    s.govern_calls([call])
    before = copy.deepcopy(s.cache.items)
    patched = s.patch_results([raw], [call])
    check("live: payload rewritten to the disambiguated top-1",
          json.loads(patched[0])["ranked_results"][0][1] == "meeting_notes_march"
          and len(json.loads(patched[0])["ranked_results"]) == 1, patched[0])
    check("live: memory still invariant",
          {t: set(d) for t, d in s.cache.items.items()}
          == {t: set(d) for t, d in before.items()})

    s = make_session("read_off.jsonl", read_enabled=False)
    s.govern_calls([call])
    patched = s.patch_results([raw], [call])
    check("disabled: byte-identical passthrough, no events",
          patched == [raw]
          and not (Path(TMP) / "read_off.jsonl").exists() or all(
              json.loads(l)["event"] != "read_gate"
              for l in open(Path(TMP) / "read_off.jsonl", encoding="utf-8")
              if l.strip()))

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run())
