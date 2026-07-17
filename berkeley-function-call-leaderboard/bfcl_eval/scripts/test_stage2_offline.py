"""
Offline tests for Stage 2 (retrieval-entropy escalation layer).

Run:  python bfcl_eval/scripts/test_stage2_offline.py

Covers (Plan 1 Step 6 acceptance): the min-margin rule picking the right branch
on fixtures, deterministic canonicalization respecting the KV key pattern and
the Vector length cap, probe sidecar round-trip, dH_neighbor diagnostics, the
no-user-text low-confidence path, shadow discipline, and the hard guarantee
that the live path is a no-op while GOV_S2_ENABLED is off.
"""

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import bfcl_eval.model_handler.middleware.governance_filter as gf  # noqa: E402
from bfcl_eval.model_handler.middleware.governance_filter import (  # noqa: E402
    GovConfig,
    GovDecision,
    GovernanceSession,
    WriteCandidate,
    build_candidate,
    canonicalize_candidate,
)

TMP = tempfile.mkdtemp(prefix="gov_stage2_test_")
PASS, FAIL = 0, 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


class NeutralScorer:
    """Always neutral -> Stage 1 escalates every candidate to Stage 2."""

    def probs_batch(self, pairs):
        return [(0.05, 0.90, 0.05) for _ in pairs]


def make_cfg(**overrides) -> GovConfig:
    cfg = GovConfig(
        log_dir=TMP, log_file="test_stage2.jsonl",
        nli_enabled=True, nli_shadow=False,
        s2_enabled=True, s2_shadow=False,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def make_session(backend, snapshot, **cfg_overrides):
    sidecar = cfg_overrides.pop("sidecar_path", str(Path(TMP) / f"{backend}_gov_state.json"))
    return GovernanceSession(
        cfg=make_cfg(**cfg_overrides),
        backend=backend,
        test_id=f"memory_{backend}_prereq_1-test-1",
        snapshot=snapshot,
        snapshot_path="<fabricated>",
        nli_scorer=NeutralScorer(),
        sidecar_path=sidecar,
    )


KV_SNAPSHOT = {
    "core_memory": {
        "user_name": "Michael Rodriguez",
        "user_age": "35",
        "favorite_drink": "coffee",
    },
    "archival_memory": {},
}

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


def force_escalate():
    original = gf.decide

    def fake_decide(candidate, signals, cache, cfg, preflight_ok):
        if signals.n_items == 0:
            return GovDecision.ADD, "empty_memory"
        return GovDecision.ESCALATE, "ambiguous"

    gf.decide = fake_decide
    return original


def read_log(name="test_stage2.jsonl"):
    path = Path(TMP) / name
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def last_decision(name="test_stage2.jsonl"):
    return [r for r in read_log(name) if r["event"] == "decision"][-1]


def test_canonicalize_unit():
    print("[canonicalize_candidate]")
    cand = build_candidate("kv", "core_memory_add(key='allergy_note', value='Penicillin reaction in 2019')")
    out = canonicalize_candidate(cand, ["penicillin_allergy", "user_name"], ["2019"],
                                 "allergic to penicillin since 2019", 300)
    check("kv picks first discriminative token", out is not None and out[1] == "2019", str(out))
    call, _ = out
    new_cand = build_candidate("kv", call)
    check("kv rewritten key valid snake_case",
          new_cand is not None and gf._KV_KEY_PATTERN.match(str(new_cand.args["key"])),
          call)
    check("kv key extends the original", str(new_cand.args["key"]) == "allergy_note_2019")

    # Token needing sanitization (capitalized name bigram).
    out = canonicalize_candidate(cand, ["allergy_note_2019_x"], ["Dr. Smith"],
                                 "seen by Dr. Smith", 300)
    check("kv token sanitized to pattern",
          out is not None and gf._KV_KEY_PATTERN.match(
              str(build_candidate("kv", out[0]).args["key"])), str(out))

    # No discriminative token anywhere -> None.
    out = canonicalize_candidate(cand, ["penicillin_reaction_2019_allergy_note"], [],
                                 "", 300)
    check("no discriminative token -> None", out is None, str(out))

    vec = build_candidate("vector", "archival_memory_add(text='Allergic to penicillin.')")
    out = canonicalize_candidate(vec, ["The user likes coffee."], ["2019"],
                                 "diagnosed in 2019", 2000)
    check("vector prepends clause", out is not None and out[0].startswith(
        "archival_memory_add(text='Regarding 2019:"), str(out))

    long_text = "x" * 1995
    vec_long = WriteCandidate(op="archival_memory_add", kind="add", tier="archival",
                              backend="vector", text=long_text, args={"text": long_text})
    out = canonicalize_candidate(vec_long, ["unrelated"], ["2019"], "in 2019", 2000)
    check("vector length cap enforced (customer-6)", out is None, str(out and out[0][:50]))


def test_accept_branch():
    print("[min-margin ACCEPT]")
    original = force_escalate()
    try:
        s = make_session("kv", KV_SNAPSHOT)
        s.user_text = "I am allergic to penicillin, diagnosed in 2019."
        call = "core_memory_add(key='allergy_diagnosed_2019', value='penicillin')"
        governed = s.govern_calls([call])
        check("accepted write proceeds unchanged", governed[0] == call, governed[0])
        rec = last_decision()
        check("stage2 block logged", rec["stage2"] is not None)
        check("outcome stage2_accept",
              rec["stage2"]["reason"] == "stage2_accept", rec["stage2"]["reason"])
        check("min_margin above threshold",
              rec["stage2"]["min_margin"] is not None
              and rec["stage2"]["min_margin"] > s.cfg.s2_margin,
              str(rec["stage2"]["min_margin"]))
        check("per-probe rows logged", len(rec["stage2"]["per_probe"]) >= 1)
        check("probes sourced from user text only",
              all(p["source"] == "user_text" for p in rec["stage2"]["probes"]),
              str(rec["stage2"]["probes"]))
        check("dH_neighbor diagnostics present",
              isinstance(rec["stage2"]["dH_neighbor"], list)
              and rec["stage2"]["dH_mean"] is not None, str(rec["stage2"]["dH_neighbor"]))
        check("not low confidence", rec["low_confidence"] is False)
    finally:
        gf.decide = original


def test_collision_canonicalization():
    print("[collision -> one-shot canonicalization]")
    original = force_escalate()
    try:
        snapshot = {
            "core_memory": {
                "penicillin_allergy": "reaction",
                "user_name": "Michael Rodriguez",
            },
            "archival_memory": {},
        }
        s = make_session("kv", snapshot, log_file="test_stage2_canon.jsonl")
        s.user_text = "My penicillin allergy was confirmed in 2019."
        call = "core_memory_add(key='allergy_note', value='Confirmed 2019')"
        governed = s.govern_calls([call])
        rec = last_decision("test_stage2_canon.jsonl")
        canon = rec["stage2"]["canonicalization"]
        check("canonicalization attempted", canon.get("applied") is not None, str(canon))
        outcome = rec["stage2"]["reason"]
        check("outcome is rewritten-accept or low-confidence-accept",
              outcome in ("stage2_accept_rewritten", "stage2_accept_low_confidence"),
              outcome)
        if outcome == "stage2_accept_rewritten":
            check("governed call rewritten consistently",
                  governed[0] == rec["stage2"]["rewritten_call"], governed[0])
            check("decision is REWRITE", rec["decision"] == "REWRITE")
        else:
            check("write proceeds as-is when retry fails", governed[0] == call, governed[0])
            check("low_confidence flagged", rec["low_confidence"] is True)
    finally:
        gf.decide = original


def test_no_user_text():
    print("[no probe source -> accept low-confidence]")
    original = force_escalate()
    try:
        s = make_session("vector", VECTOR_SNAPSHOT, log_file="test_stage2_nouser.jsonl")
        call = "core_memory_add(text='The user enjoys morning walks.')"
        governed = s.govern_calls([call])
        check("write proceeds unchanged", governed[0] == call)
        rec = last_decision("test_stage2_nouser.jsonl")
        check("reason stage2_no_probe_source",
              rec["stage2"]["reason"] == "stage2_no_probe_source", rec["stage2"]["reason"])
        check("low_confidence set", rec["low_confidence"] is True)
        # After the genuine success, the observed item carries the flag.
        s.patch_results(['{"id": 3}'], [call])
        item = s.cache.items["core"]["3"]
        check("observed item flagged low_confidence", item.low_confidence is True)
        check("write-time probes cached", item.probes and item.probe_provenance == "write_time",
              str(item.probes))
    finally:
        gf.decide = original


def test_sidecar_roundtrip():
    print("[probe sidecar round-trip]")
    original = force_escalate()
    try:
        sidecar = str(Path(TMP) / "roundtrip_gov_state.json")
        s = make_session("vector", VECTOR_SNAPSHOT, sidecar_path=sidecar,
                         log_file="test_stage2_sidecar.jsonl")
        s.user_text = "The user was born in Caracas."
        call = "core_memory_add(text='The user was born in Caracas.')"
        governed = s.govern_calls([call])
        s.patch_results(['{"id": 3}'], list(governed))
        check("sidecar written", Path(sidecar).exists())
        with open(sidecar, "r", encoding="utf-8") as f:
            payload = json.load(f)
        check("sidecar carries new item probes",
              payload["tiers"]["core"]["3"]["probes"], str(payload["tiers"]["core"].keys()))

        # A second session rehydrating the same state reuses sidecar probes.
        snapshot2 = {
            "core_memory": {"next_id": 4, "store": {
                **VECTOR_SNAPSHOT["core_memory"]["store"],
                "3": "The user was born in Caracas.",
            }},
            "archival_memory": {"next_id": 0, "store": {}},
        }
        s2 = make_session("vector", snapshot2, sidecar_path=sidecar,
                          log_file="test_stage2_sidecar.jsonl")
        item = s2.cache.items["core"]["3"]
        check("sidecar probes reloaded", item.probe_provenance == "sidecar"
              and item.probes == payload["tiers"]["core"]["3"]["probes"],
              item.probe_provenance)
        # Items absent from the sidecar are regenerated and flagged.
        other = s2.cache.items["core"]["0"]
        check("cache-miss regenerates as stored_text",
              other.probe_provenance in ("stored_text", "sidecar"), other.probe_provenance)

        # Text drift invalidates the sidecar entry.
        snapshot3 = {
            "core_memory": {"next_id": 4, "store": {"3": "The user was born in Lima."}},
            "archival_memory": {"next_id": 0, "store": {}},
        }
        s3 = make_session("vector", snapshot3, sidecar_path=sidecar,
                          log_file="test_stage2_sidecar.jsonl")
        check("drifted text -> stored_text provenance",
              s3.cache.items["core"]["3"].probe_provenance == "stored_text")
    finally:
        gf.decide = original


def test_shadow_and_disabled():
    print("[shadow + disabled discipline]")
    original = force_escalate()
    try:
        # s2 shadow: full computation and logging, zero intervention.
        s = make_session("kv", KV_SNAPSHOT, s2_shadow=True,
                         log_file="test_stage2_shadow.jsonl")
        s.user_text = "I am allergic to penicillin, diagnosed in 2019."
        call = "core_memory_add(key='allergy_diagnosed_2019', value='penicillin')"
        governed = s.govern_calls([call])
        check("s2 shadow never intervenes", governed[0] == call and not s._pending)
        rec = last_decision("test_stage2_shadow.jsonl")
        check("s2 shadow still logs stage2 block",
              rec["stage2"] is not None and rec["stage2"].get("shadowed") is True,
              str(rec["stage2"]))
        check("s2 shadow reason chain",
              rec["reason"].startswith("escalate:stage2_shadow"), rec["reason"])

        # s2 disabled: stage 1 all-neutral falls back to ADD, no stage2 block.
        s2 = make_session("kv", KV_SNAPSHOT, s2_enabled=False,
                          log_file="test_stage2_off.jsonl")
        s2.user_text = "irrelevant"
        governed = s2.govern_calls([call])
        rec = last_decision("test_stage2_off.jsonl")
        check("s2 disabled: no stage2 block", rec["stage2"] is None)
        check("s2 disabled: stage1 fallback reason",
              rec["reason"] == "escalate:stage1_all_neutral_no_stage2", rec["reason"])
        check("s2 disabled: write unchanged", governed[0] == call)
        item = next(iter(s2.cache.items["core"].values()))
        check("s2 disabled: no probes generated", item.probes == [])
    finally:
        gf.decide = original


def main():
    test_canonicalize_unit()
    test_accept_branch()
    test_collision_canonicalization()
    test_no_user_text()
    test_sidecar_roundtrip()
    test_shadow_and_disabled()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
