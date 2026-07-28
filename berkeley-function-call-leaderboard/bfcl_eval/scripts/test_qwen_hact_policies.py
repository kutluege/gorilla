"""Tests for QwenHactHandler selection policies (A1 random_select, A2 majority).

Run: python bfcl_eval/scripts/test_qwen_hact_policies.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))

import tempfile  # noqa: E402

from test_qwen_hact_shadow import (  # noqa: E402
    FREE_TEXT,
    TOOL_CALL,
    TOOL_CALL_B,
    FakeClient,
    make_handler,
    make_inference_data,
    make_session,
)

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def run_policy(policy, primary_text=TOOL_CALL, explore_texts=None, n=4):
    tmp = tempfile.mkdtemp(prefix="hactpol_")
    h = make_handler(tmp, client=FakeClient(primary_text=primary_text,
                                            explore_texts=explore_texts))
    h.hact_config.policy = policy
    h.hact_config.num_samples = n
    h._gov_tls.session = make_session(tmp)
    resp, _ = h._query_prompting(make_inference_data("memory_kv_prereq_1-test-1"))
    return h, resp, h._gov_tls.session.pending_hact


def test_shadow_untouched():
    h, resp, rec = run_policy("shadow")
    check("shadow returns primary", resp.choices[0].text == TOOL_CALL)
    check("shadow has no policy_record", "policy_record" not in rec)


def test_majority_overrides_minority_primary():
    # primary = TOOL_CALL_B (key 'age'); exploration = 3x TOOL_CALL (key 'user_age')
    h, resp, rec = run_policy("majority", primary_text=TOOL_CALL_B,
                              explore_texts=[TOOL_CALL, TOOL_CALL, TOOL_CALL])
    check("majority swaps in modal sample", resp.choices[0].text == TOOL_CALL)
    pr = rec["policy_record"]
    check("policy_record majority", pr["policy"] == "majority")
    check("selected index nonzero", pr["selected_index"] > 0, pr)
    check("modal_count 3, primary_count 1",
          pr["modal_count"] == 3 and pr["primary_count"] == 1, pr)


def test_majority_keeps_tied_primary():
    # primary TOOL_CALL_B; exploration = [TOOL_CALL_B(dup? no: same sig), TOOL_CALL, FREE_TEXT]
    h, resp, rec = run_policy("majority", primary_text=TOOL_CALL_B,
                              explore_texts=[TOOL_CALL_B, TOOL_CALL, FREE_TEXT])
    check("tie -> primary kept", resp.choices[0].text == TOOL_CALL_B)
    check("selected index 0", rec["policy_record"]["selected_index"] == 0)


def test_majority_unanimous_keeps_primary():
    h, resp, rec = run_policy("majority", primary_text=TOOL_CALL,
                              explore_texts=[TOOL_CALL, TOOL_CALL, TOOL_CALL])
    check("unanimous -> primary kept", rec["policy_record"]["selected_index"] == 0)


def test_random_select_deterministic():
    h1, r1, rec1 = run_policy("random_select",
                              explore_texts=[TOOL_CALL_B, FREE_TEXT, TOOL_CALL])
    h2, r2, rec2 = run_policy("random_select",
                              explore_texts=[TOOL_CALL_B, FREE_TEXT, TOOL_CALL])
    check("random_select deterministic given (test_id, call_idx)",
          rec1["policy_record"]["selected_index"]
          == rec2["policy_record"]["selected_index"])
    check("returned text matches selection",
          r1.choices[0].text == r2.choices[0].text)
    idx = rec1["policy_record"]["selected_index"]
    check("selected index in range", 0 <= idx < 4, idx)


def test_random_select_covers_range():
    # across many (test_id, call_idx) pairs the selection must not be constant
    import tempfile as tf
    from test_qwen_hact_shadow import make_handler as mh, make_session as ms, \
        make_inference_data as mid
    seen = set()
    tmp = tf.mkdtemp(prefix="hactpol_")
    h = mh(tmp, client=FakeClient(explore_texts=[TOOL_CALL_B, FREE_TEXT, TOOL_CALL]))
    h.hact_config.policy = "random_select"
    h._gov_tls.session = ms(tmp)
    for i in range(12):
        d = mid("memory_kv_prereq_1-test-1")
        d["_hact_call_idx"] = i
        h._query_prompting(d)
        seen.add(h._gov_tls.session.pending_hact["policy_record"]["selected_index"])
        h._gov_tls.session.pending_hact = None
    check("random_select varies across steps", len(seen) >= 2, seen)


def test_policy_guard():
    from bfcl_eval.model_handler.local_inference.qwen_hact import (
        _IMPLEMENTED_POLICIES,
    )
    check("gate policies still guarded",
          "hact_gate" not in _IMPLEMENTED_POLICIES
          and "vote_margin_gate" not in _IMPLEMENTED_POLICIES)
    check("selection policies registered",
          "random_select" in _IMPLEMENTED_POLICIES
          and "majority" in _IMPLEMENTED_POLICIES)


def main():
    for k, fn in sorted(globals().items()):
        if k.startswith("test_"):
            print(f"[{k}]")
            fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
