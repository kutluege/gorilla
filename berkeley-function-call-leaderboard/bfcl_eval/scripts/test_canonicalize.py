"""
Offline acceptance tests for the SS4.3 one-shot CANONICALIZE contract
(Plan 3 Step 3 / G5). The mechanism itself landed in Plan 1 Step 6; these
tests pin the contract: ONE correction, ONE re-test with the SAME probes, no
second generation round, low_confidence on failure.

Run:  python bfcl_eval/scripts/test_canonicalize.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bfcl_eval.model_handler.middleware.governance_filter as gf  # noqa: E402
from bfcl_eval.model_handler.middleware.governance_filter import (  # noqa: E402
    GovConfig,
    GovDecision,
    GovernanceSession,
)

TMP = tempfile.mkdtemp(prefix="gov_canon_test_")
PASS, FAIL = 0, 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


class NeutralScorer:
    def probs_batch(self, pairs):
        return [(0.05, 0.90, 0.05) for _ in pairs]


KV_SNAPSHOT = {
    "core_memory": {
        "user_name": "Michael Rodriguez",
        "user_age": "35",
        "favorite_drink": "coffee",
    },
    "archival_memory": {},
}


def force_escalate():
    original = gf.decide

    def fake_decide(candidate, signals, cache, cfg, preflight_ok):
        if signals.n_items == 0:
            return GovDecision.ADD, "empty_memory"
        return GovDecision.ESCALATE, "ambiguous"

    gf.decide = fake_decide
    return original


def run_write(log_file, s2_margin):
    """One escalated KV write; returns (governed, decision record, canon calls)."""
    calls = []
    real_canon = gf.canonicalize_candidate

    def counting_canon(*args, **kwargs):
        out = real_canon(*args, **kwargs)
        calls.append(out)
        return out

    original = force_escalate()
    gf.canonicalize_candidate = counting_canon
    try:
        cfg = GovConfig(
            log_dir=TMP, log_file=log_file,
            nli_enabled=True, nli_shadow=False,
            s2_enabled=True, s2_shadow=False,
        )
        cfg.s2_margin = s2_margin
        s = GovernanceSession(
            cfg=cfg, backend="kv",
            test_id="memory_kv_prereq_1-test-1",
            snapshot=KV_SNAPSHOT, snapshot_path="<fabricated>",
            nli_scorer=NeutralScorer(),
            sidecar_path=str(Path(TMP) / f"{log_file}_gov_state.json"),
        )
        s.user_text = "My favorite drink is espresso coffee these days."
        governed = s.govern_calls(
            ["core_memory_add(key='favorite_drink_espresso', value='espresso coffee')"]
        )
    finally:
        gf.decide = original
        gf.canonicalize_candidate = real_canon
    with open(Path(TMP) / log_file, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rec = [r for r in rows if r["event"] == "decision"][-1]
    return governed, rec, calls


def run():
    print("[margin cleared: canonicalize never invoked]")
    governed, rec, calls = run_write("canon_pass.jsonl", s2_margin=-1.0)
    check("accept without canonicalization", len(calls) == 0, str(len(calls)))
    check("reason stage2_accept", "stage2_accept" in rec["reason"], rec["reason"])
    check("canonicalization block empty",
          rec["stage2"]["canonicalization"] == {})

    print("[margin unreachable: exactly ONE correction round]")
    governed, rec, calls = run_write("canon_fail.jsonl", s2_margin=1e9)
    check("canonicalize called EXACTLY once", len(calls) == 1, str(len(calls)))
    s2 = rec["stage2"]
    canon = s2["canonicalization"]
    check("correction attempted and logged",
          canon.get("applied") in (True, False), str(canon))
    if canon.get("applied"):
        check("retry re-tested with the SAME probes (one probe list logged)",
              "retry_min_margin" in canon and isinstance(s2["probes"], list)
              and s2["probes_n"] == len(s2["probes"]))
    check("failure -> ADD with low_confidence (never suppressed)",
          rec["decision"] == "ADD" and s2["low_confidence"] is True,
          f"{rec['decision']}/{s2['low_confidence']}")
    check("write survives as-is (no drop)",
          governed == ["core_memory_add(key='favorite_drink_espresso', value='espresso coffee')"])

    print("[no second generation round exists]")
    # The only path that could produce a *new* candidate besides the single
    # deterministic call is the LLM hook -- assert it stays unwired in the
    # session path (canonicalize_llm_fn=None at the stage2_resolve call site).
    import inspect

    src = inspect.getsource(gf.GovernanceSession._resolve_escalation)
    check("stage2_resolve invoked with canonicalize_llm_fn=None",
          "canonicalize_llm_fn=None" in src)
    check("counting shows no retry loop (1 call even on failure)",
          len(calls) == 1)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run())
