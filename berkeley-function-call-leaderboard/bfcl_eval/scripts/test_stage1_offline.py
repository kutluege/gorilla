"""
Offline tests for Stage 1 (NLI escalation), stubbed scorer -- no weights needed.

Run:  python bfcl_eval/scripts/test_stage1_offline.py

Covers (Plan 1 Step 5 acceptance): the full SS7.5 decision table, the KV
canonical-key pre-check firing BEFORE any NLI call, shadow mode never mutating
``governed``, rewrite preflight fallback to ADD, and the rewrite mechanics in
patch_results (real result flows, executed call observed, no synthetic).
"""

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import bfcl_eval.model_handler.middleware.governance_filter as gf  # noqa: E402
from bfcl_eval.model_handler.middleware.governance_filter import (  # noqa: E402
    DECOY_CALL,
    GovConfig,
    GovDecision,
    GovernanceSession,
    build_candidate,
    compute_signals,
    stage1_resolve,
)

TMP = tempfile.mkdtemp(prefix="gov_stage1_test_")
PASS, FAIL = 0, 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


class StubScorer:
    """probs_batch keyed on (premise, hypothesis); default = neutral."""

    def __init__(self, table=None):
        self.table = table or {}
        self.calls = 0

    def probs_batch(self, pairs):
        self.calls += 1
        return [self.table.get((p, h), (0.05, 0.90, 0.05)) for p, h in pairs]


class ExplodingScorer:
    def probs_batch(self, pairs):
        raise AssertionError("NLI must not be called (pre-check should resolve)")


def make_cfg(**overrides) -> GovConfig:
    cfg = GovConfig(log_dir=TMP, log_file="test_stage1.jsonl",
                    nli_enabled=True, nli_shadow=False)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def make_session(backend, snapshot, scorer=None, **cfg_overrides):
    return GovernanceSession(
        cfg=make_cfg(**cfg_overrides),
        backend=backend,
        test_id=f"memory_{backend}_prereq_1-test-1",
        snapshot=snapshot,
        snapshot_path="<fabricated>",
        nli_scorer=scorer if scorer is not None else StubScorer(),
    )


VECTOR_SNAPSHOT = {
    "core_memory": {
        "next_id": 3,
        "store": {
            "0": "The user's name is Michael Rodriguez.",
            "1": "The user is 35 years old.",
            "2": "The user likes coffee.",
        },
    },
    "archival_memory": {"next_id": 0, "store": {}},
}

KV_SNAPSHOT = {
    "core_memory": {
        "user_name": "Michael Rodriguez",
        "user_age": "35",
        "favorite_drink": "coffee",
    },
    "archival_memory": {},
}


def resolve(session, call, scorer, preflight_ok=None):
    cand = build_candidate(session.backend, call)
    v_w = session._whiten_one(cand.text)
    if preflight_ok is None:
        preflight_ok = gf.preflight_would_succeed(cand, session.cache)
    signals = compute_signals(v_w, cand, session.cache, session.thresholds, session.cfg)
    return cand, signals, stage1_resolve(
        cand, signals, session.cache, session.cfg, preflight_ok, scorer
    )


def test_decision_table():
    print("[SS7.5 decision table]")
    s = make_session("vector", VECTOR_SNAPSHOT)
    cand_text = "The user drinks coffee every day."
    stored = "The user likes coffee."
    call = f"core_memory_add(text='{cand_text}')"

    # Equivalent: both directions entail.
    tbl = {(stored, cand_text): (0.02, 0.10, 0.88), (cand_text, stored): (0.02, 0.10, 0.88)}
    _, _, res = resolve(s, call, StubScorer(tbl))
    check("equivalent -> NOOP", res.outcome == GovDecision.NOOP
          and res.reason == "stage1_equivalent", res.reason)

    # Derivable: stored entails candidate only.
    tbl = {(stored, cand_text): (0.02, 0.10, 0.88)}
    _, _, res = resolve(s, call, StubScorer(tbl))
    check("derivable -> NOOP", res.outcome == GovDecision.NOOP
          and res.reason == "stage1_derivable", res.reason)

    # More specific: candidate entails stored, big margin.
    tbl = {(cand_text, stored): (0.02, 0.08, 0.90)}
    _, _, res = resolve(s, call, StubScorer(tbl))
    check("more_specific -> REWRITE", res.outcome == GovDecision.REWRITE
          and res.reason == "stage1_more_specific", res.reason)
    check("rewrite targets the stored item", res.target_ref == "2"
          and res.superseded_text == stored, str(res.target_ref))
    check("rewrite call is a vector update",
          res.rewritten_call == f"core_memory_update(vec_id=2, new_text={cand_text!r})",
          str(res.rewritten_call))

    # Specificity inconclusive: bwd entails, fwd below tau, margin <= delta_spec
    # (bwd_e - fwd_e = 0.78 - 0.70 = 0.08 <= 0.10).
    tbl = {(cand_text, stored): (0.02, 0.20, 0.78), (stored, cand_text): (0.02, 0.28, 0.70)}
    _, _, res = resolve(s, call, StubScorer(tbl))
    check("inconclusive -> keep both (ADD)", res.outcome == GovDecision.ADD
          and res.reason == "stage1_keep_both", res.reason)

    # Contradiction.
    tbl = {(stored, cand_text): (0.92, 0.05, 0.03)}
    _, _, res = resolve(s, call, StubScorer(tbl))
    check("contradiction -> REWRITE (supersede)", res.outcome == GovDecision.REWRITE
          and res.reason == "stage1_contradiction", res.reason)
    check("superseded_text logged in full", res.superseded_text == stored)

    # All neutral -> escalate to Stage 2.
    _, _, res = resolve(s, call, StubScorer())
    check("all neutral -> ESCALATE", res.outcome == GovDecision.ESCALATE
          and res.reason == "stage1_all_neutral", res.reason)
    check("every evaluated pair logged", len(res.pairs) == min(3, s.cache.total_size()))
    check("neighbors ranked by sims", len(res.candidates) >= 1
          and all("sim" in c for c in res.candidates))

    # Scorer unavailable -> ADD fallback, no crash.
    _, _, res = resolve(s, call, None)
    check("scorer None -> ADD fallback", res.outcome == GovDecision.ADD
          and res.reason == "stage1_nli_unavailable", res.reason)


def test_kv_precheck():
    print("[KV canonical-key pre-check (no NLI call)]")
    s = make_session("kv", KV_SNAPSHOT)

    # add on existing key, same value -> rewrite to idempotent replace.
    cand, _, res = resolve(
        s, "core_memory_add(key='user_age', value='35')", ExplodingScorer()
    )
    check("add-existing-key resolves without NLI", res.precheck == "kv_add_existing_key")
    check("idempotent add -> REWRITE to replace",
          res.outcome == GovDecision.REWRITE
          and res.reason == "stage1_precheck_idempotent_key"
          and res.rewritten_call == "core_memory_replace(key='user_age', value='35')",
          f"{res.reason} {res.rewritten_call}")

    # add on existing key, different value -> rewrite (legitimate update).
    _, _, res = resolve(
        s, "core_memory_add(key='user_age', value='36')", ExplodingScorer()
    )
    check("value-differs add -> REWRITE to replace",
          res.outcome == GovDecision.REWRITE
          and res.reason == "stage1_precheck_add_existing_key"
          and res.rewritten_call == "core_memory_replace(key='user_age', value='36')",
          str(res.rewritten_call))
    check("superseded value preserved", res.superseded_text == "user age: 35")

    # idempotent replace -> NOOP (preflight ok: key exists).
    _, _, res = resolve(
        s, "core_memory_replace(key='user_age', value='35')", ExplodingScorer()
    )
    check("idempotent replace -> NOOP",
          res.outcome == GovDecision.NOOP
          and res.reason == "stage1_precheck_idempotent_replace", res.reason)

    # New key -> pre-check does not fire; NLI runs (stub, all neutral).
    scorer = StubScorer()
    _, _, res = resolve(s, "core_memory_add(key='new_fact', value='hello world')", scorer)
    check("new key skips pre-check", res.precheck is None and scorer.calls == 1)


def test_vector_idempotent_update():
    print("[vector idempotent update pre-check]")
    s = make_session("vector", VECTOR_SNAPSHOT)
    _, _, res = resolve(
        s, "core_memory_update(vec_id=1, new_text='The user is 35 years old.')",
        ExplodingScorer(),
    )
    check("idempotent update -> NOOP without NLI",
          res.outcome == GovDecision.NOOP
          and res.precheck == "vector_idempotent_update", res.reason)


def force_escalate(monkey_target=gf):
    """Route every decide() through the escalation path for orchestration tests."""
    original = monkey_target.decide

    def fake_decide(candidate, signals, cache, cfg, preflight_ok):
        if signals.n_items == 0:
            return GovDecision.ADD, "empty_memory"
        return GovDecision.ESCALATE, "ambiguous"

    monkey_target.decide = fake_decide
    return original


def read_log(name):
    path = Path(TMP) / name
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_shadow_never_mutates():
    print("[shadow discipline]")
    original = force_escalate()
    try:
        tbl = {
            ("The user likes coffee.", "The user likes coffee daily."): (0.02, 0.10, 0.88),
            ("The user likes coffee daily.", "The user likes coffee."): (0.02, 0.10, 0.88),
        }
        s = make_session("vector", VECTOR_SNAPSHOT, scorer=StubScorer(tbl),
                         nli_shadow=True, log_file="test_stage1_shadow.jsonl")
        calls = ["core_memory_add(text='The user likes coffee daily.')"]
        governed = s.govern_calls(list(calls))
        check("shadow never rewrites", governed == calls, str(governed))
        check("shadow leaves no pending", not s._pending)
        rec = [r for r in read_log("test_stage1_shadow.jsonl") if r["event"] == "decision"][-1]
        check("shadow logs full stage1 block",
              rec["stage1"] is not None and rec["stage1"]["outcome"] == "NOOP"
              and rec["stage1"].get("shadowed") is True, str(rec.get("stage1")))
        check("shadow final decision is ADD",
              rec["decision"] == "ADD" and rec["reason"].startswith("escalate:stage1_shadow"),
              rec["reason"])
    finally:
        gf.decide = original


def test_live_noop_and_rewrite_mechanics():
    print("[live application mechanics]")
    original = force_escalate()
    try:
        stored = "The user likes coffee."
        cand_text = "The user likes coffee daily."
        tbl = {(stored, cand_text): (0.02, 0.10, 0.88), (cand_text, stored): (0.02, 0.10, 0.88)}
        s = make_session("vector", VECTOR_SNAPSHOT, scorer=StubScorer(tbl),
                         log_file="test_stage1_live.jsonl")
        governed = s.govern_calls([f"core_memory_add(text='{cand_text}')"])
        check("live equivalent NOOP -> decoy", governed[0] == DECOY_CALL, str(governed))
        check("NOOP pending mode", s._pending[0]["mode"] == "noop")
        s.patch_results(['{"result": []}'], list(governed))

        # Contradiction -> live rewrite to update; the REAL result flows and the
        # executed call is observed (mirror updates), no synthetic, no restore.
        tbl2 = {(stored, cand_text): (0.92, 0.05, 0.03)}
        s2 = make_session("vector", VECTOR_SNAPSHOT, scorer=StubScorer(tbl2),
                          log_file="test_stage1_live.jsonl")
        governed = s2.govern_calls([f"core_memory_add(text='{cand_text}')"])
        expected = f"core_memory_update(vec_id=2, new_text={cand_text!r})"
        check("live contradiction rewrites the call", governed[0] == expected, governed[0])
        check("rewrite pending mode", s2._pending[0]["mode"] == "rewrite")
        decoded = list(governed)
        real_result = '{"status": "ID 2 updated."}'
        patched = s2.patch_results([real_result], decoded)
        check("real result flows through (no synthetic)", patched[0] == real_result)
        check("executed call NOT restored", decoded[0] == expected, decoded[0])
        check("mirror observed the rewrite",
              s2.cache.items["core"]["2"].text == cand_text,
              s2.cache.items["core"]["2"].text)
        rec = [r for r in read_log("test_stage1_live.jsonl") if r["event"] == "decision"][-1]
        check("original call preserved in log",
              rec["original_call"] == f"core_memory_add(text='{cand_text}')"
              and rec["rewritten_call"] == expected, str(rec.get("original_call")))
        check("superseded_text in stage1 log",
              rec["stage1"]["superseded_text"] == stored)
    finally:
        gf.decide = original


def test_rewrite_preflight_fallback():
    print("[rewrite preflight fallback]")
    original = force_escalate()
    try:
        stored = "The user likes coffee."
        long_text = "The user has an elaborate coffee routine. " * 10  # > 300 chars
        assert len(long_text) > 300
        tbl = {(stored, long_text.strip()): (0.92, 0.05, 0.03)}
        s = make_session("vector", VECTOR_SNAPSHOT, scorer=StubScorer(tbl),
                         log_file="test_stage1_preflight.jsonl")
        call = f"core_memory_add(text={long_text.strip()!r})"
        governed = s.govern_calls([call])
        check("oversize rewrite degrades to ADD (call unchanged)",
              governed[0] == call, governed[0][:80])
        rec = [r for r in read_log("test_stage1_preflight.jsonl") if r["event"] == "decision"][-1]
        check("fallback reason recorded",
              rec["decision"] == "ADD"
              and rec["reason"].endswith("_rewrite_preflight_blocked"), rec["reason"])
    finally:
        gf.decide = original


def test_disabled_is_bit_compatible():
    print("[stage 1 disabled == pre-Stage-1 behavior]")
    original = force_escalate()
    try:
        s = make_session("vector", VECTOR_SNAPSHOT, scorer=ExplodingScorer(),
                         nli_enabled=False, log_file="test_stage1_off.jsonl")
        calls = ["core_memory_add(text='The user likes coffee daily.')"]
        governed = s.govern_calls(list(calls))
        check("disabled: call unchanged", governed == calls)
        rec = [r for r in read_log("test_stage1_off.jsonl") if r["event"] == "decision"][-1]
        check("disabled: legacy reason preserved",
              rec["reason"] == "escalate:stage0_escalate_fallback", rec["reason"])
        check("disabled: no stage1 block", rec["stage1"] is None)
    finally:
        gf.decide = original


def main():
    test_decision_table()
    test_kv_precheck()
    test_vector_idempotent_update()
    test_shadow_never_mutates()
    test_live_noop_and_rewrite_mechanics()
    test_rewrite_preflight_fallback()
    test_disabled_is_bit_compatible()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
