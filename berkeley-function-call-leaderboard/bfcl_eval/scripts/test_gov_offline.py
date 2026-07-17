"""
Offline tests for the Stage 0 geometric governance filter (no server, no harness).

Run:  python bfcl_eval/scripts/test_gov_offline.py

Covers:
  - fail-loud on missing ABTT artifact
  - rehydration from fabricated KV / Vector snapshots
  - NOOP on an exact duplicate Vector re-add (the classic duplicate-write failure)
  - never-NOOP when the candidate carries new content ("likes tea" vs "likes coffee")
  - ADD on genuinely new facts
  - preflight blocks NOOP when the backend would reject (duplicate KV key)
  - decoy rewrite + synthetic-result patching + decoded-call restoration (HOOK 2)
  - dry-run mode never rewrites
  - cache observation of genuine successes
  - non-encode decision path latency < 1 ms at full memory (57 items)
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bfcl_eval.model_handler.middleware.governance_filter import (  # noqa: E402
    DECOY_CALL,
    GovConfig,
    GovDecision,
    GovernanceSession,
    WriteCandidate,
    compute_signals,
    load_abtt,
)

TMP = tempfile.mkdtemp(prefix="gov_test_")
PASS, FAIL = 0, 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def make_cfg(**overrides) -> GovConfig:
    cfg = GovConfig(log_dir=TMP, log_file="test_gov.jsonl")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def make_session(backend: str, snapshot, **cfg_overrides) -> GovernanceSession:
    return GovernanceSession(
        cfg=make_cfg(**cfg_overrides),
        backend=backend,
        test_id=f"memory_{backend}_prereq_1-test-1",
        snapshot=snapshot,
        snapshot_path="<fabricated>",
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


def test_artifact():
    print("[artifact]")
    try:
        load_abtt(str(Path(TMP) / "missing.npz"))
        check("missing artifact raises", False)
    except RuntimeError:
        check("missing artifact raises", True)
    abtt = load_abtt(GovConfig().artifact_path)
    check("artifact loads", abtt.mu.shape == (384,) and abtt.u_top.shape == (384, 16))


def test_vector_decisions():
    print("[vector decisions]")
    s = make_session("vector", VECTOR_SNAPSHOT)
    check("rehydrated 3 items", s.cache.total_size() == 3)

    # Exact duplicate re-add -> NOOP with a decoy rewrite and shadow-id synthetic.
    calls = ["core_memory_add(text='The user is 35 years old.')"]
    governed = s.govern_calls(calls)
    check("exact dup rewritten to decoy", governed[0] == DECOY_CALL, str(governed))
    check("pending has synthetic", 0 in s._pending)
    if 0 in s._pending:
        synthetic = json.loads(s._pending[0]["synthetic"])
        check("shadow id out-of-band", synthetic.get("id", 0) >= 9000, str(synthetic))

    # HOOK 2: decoy result replaced, decoded call restored, cache unchanged.
    decoded = list(governed)
    results = ['{"result": [{"id": 0, "text": "..."}]}']
    patched = s.patch_results(results, decoded)
    check("synthetic patched in", json.loads(patched[0]).get("id", 0) >= 9000)
    check("decoded call restored", decoded[0] == calls[0], decoded[0])
    check("cache unchanged after NOOP", s.cache.total_size() == 3)

    # New content sharing the template -> must NOT be suppressed.
    governed = s.govern_calls(["core_memory_add(text='The user likes tea.')"])
    check("'likes tea' not suppressed", governed[0] != DECOY_CALL)

    # Genuinely new fact -> pass through, then a genuine success updates the cache.
    call = "archival_memory_add(text='Allergic to penicillin, diagnosed 2019.')"
    governed = s.govern_calls([call])
    check("new fact passes through", governed[0] == call)
    s.patch_results(['{"id": 0}'], [call])
    check("genuine add observed", s.cache.total_size() == 4)
    check("observed item in archival", s.cache.size("archival") == 1)


def test_kv_decisions():
    print("[kv decisions]")
    s = make_session("kv", KV_SNAPSHOT)
    check("rehydrated 3 items", s.cache.total_size() == 3)

    # Byte-identical re-add of an existing pair: geometry + verbatim say NOOP, but
    # the backend would reject the duplicate key -> preflight must force pass-through.
    governed = s.govern_calls(["core_memory_add(key='user_name', value='Michael Rodriguez')"])
    check("dup key passes through (preflight)", governed[0] != DECOY_CALL)

    # Same value under a NEW key: identical composite semantics are not guaranteed
    # geometrically; whatever the decision, it must never be a bare suppression of
    # novel content. Informational only:
    governed = s.govern_calls(["core_memory_add(key='name', value='Michael Rodriguez')"])
    print(f"    (info) new-key dup value suppressed={governed[0] == DECOY_CALL}")

    # Replace with a brand-new value -> never NOOP (verbatim miss on '37').
    governed = s.govern_calls(["core_memory_replace(key='user_age', value='37')"])
    check("value change not suppressed", governed[0] != DECOY_CALL)

    # Genuine replace success updates the mirror text.
    call = "core_memory_replace(key='user_age', value='37')"
    s.patch_results(['{"status": "Key replaced."}'], [call])
    check(
        "replace observed",
        s.cache.items["core"]["user_age"].text.endswith("37"),
        s.cache.items["core"]["user_age"].text,
    )

    # Clear observation empties the tier.
    s.patch_results(['{"status": "Short term memory cleared."}'], ["core_memory_clear()"])
    check("clear observed", s.cache.size("core") == 0)


def test_dry_run():
    print("[dry run]")
    s = make_session("vector", VECTOR_SNAPSHOT, dry_run=True)
    calls = ["core_memory_add(text='The user is 35 years old.')"]
    governed = s.govern_calls(calls)
    check("dry run never rewrites", governed == calls)
    check("dry run leaves no pending", not s._pending)


def test_failed_write_not_observed():
    print("[strict success matching]")
    s = make_session("kv", KV_SNAPSHOT)
    call = "core_memory_add(key='new_key', value='something')"
    s.patch_results(['{"error": "Core memory is full. Please clear some entries."}'], [call])
    check("error result not observed", s.cache.total_size() == 3)
    s.patch_results(["Error during execution: something broke"], [call])
    check("execution error not observed", s.cache.total_size() == 3)


def test_threshold_validation():
    print("[threshold startup validation]")
    # Direction pinned per Plan 1 Step 1 / R1: bound = sqrt(1 - sim_high^2) is the
    # max single-neighbor residual compatible with sim_max > sim_high, so
    # delta >= bound is healthy (residual gate never binds) and delta < bound is
    # the degenerate case (residual gate silently raises the effective sim bar).
    # This is the only direction consistent with ALL of v2's worked numbers:
    # (0.95, 0.30) degenerate, (0.95, 0.32) fine, (0.80, 0.40) warns.
    # Plan 2's calibration must re-check this against live margins.
    ok_cfg = make_cfg(sim_high=0.95, delta=0.32)
    check("(0.95, 0.32) passes", ok_cfg.validate() is True)
    check("(0.95, 0.30) warns (v2's degenerate pair)",
          make_cfg(sim_high=0.95, delta=0.30).validate() is False)
    check("(0.80, 0.40) warns", make_cfg(sim_high=0.80, delta=0.40).validate() is False)
    try:
        make_cfg(sim_high=0.80, delta=0.40, strict_thresholds=True).validate()
        check("strict raises on degenerate pair", False)
    except ValueError:
        check("strict raises on degenerate pair", True)
    check("strict passes healthy pair",
          make_cfg(sim_high=0.95, delta=0.32, strict_thresholds=True).validate() is True)

    import os
    from bfcl_eval.model_handler.middleware.governance_filter import GovConfig as GC
    old_env = {k: os.environ.get(k) for k in
               ("GOV_STRICT_THRESHOLDS", "GOV_SIM_HIGH", "GOV_DELTA")}
    try:
        os.environ["GOV_STRICT_THRESHOLDS"] = "1"
        os.environ["GOV_SIM_HIGH"] = "0.95"
        os.environ["GOV_DELTA"] = "0.32"
        cfg = GC.from_env()
        check("GOV_STRICT_THRESHOLDS=1 round-trips", cfg.strict_thresholds is True)
        os.environ["GOV_DELTA"] = "0.30"
        try:
            GC.from_env()
            check("strict from_env raises on (0.95, 0.30)", False)
        except ValueError:
            check("strict from_env raises on (0.95, 0.30)", True)
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _read_log_records(log_file: str):
    path = Path(TMP) / log_file
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_logging_fixes():
    print("[logging fixes (Plan 1 Step 2)]")
    log_file = "test_gov_step2.jsonl"
    s = make_session("kv", KV_SNAPSHOT, log_file=log_file)

    # 2c: rehydrate record carries ABTT provenance.
    recs = _read_log_records(log_file)
    rehydrates = [r for r in recs if r["event"] == "rehydrate"]
    check("rehydrate has abtt_d==16", rehydrates and rehydrates[-1].get("abtt_d") == 16)
    check("rehydrate has artifact path",
          rehydrates[-1].get("abtt_artifact_path", "").endswith(".npz"))

    # 2a: a remove and a clear each emit exactly one observe event.
    s.patch_results(['{"status": "Key removed."}'], ["core_memory_remove(key='user_age')"])
    recs = _read_log_records(log_file)
    removes = [r for r in recs if r["event"] == "observe_remove"]
    check("remove emits one observe_remove",
          len(removes) == 1 and removes[0]["ref"] == "user_age", str(removes))
    check("remove applied to mirror", "user_age" not in s.cache.items["core"])
    s.patch_results(['{"status": "Short term memory cleared."}'], ["core_memory_clear()"])
    recs = _read_log_records(log_file)
    clears = [r for r in recs if r["event"] == "observe_clear"]
    check("clear emits one observe_clear",
          len(clears) == 1 and clears[0]["tier"] == "core", str(clears))
    check("clear applied to mirror", s.cache.size("core") == 0)

    # 2b: decision + observe records carry full, untruncated text.
    long_text = "A long archival fact about the 2019 penicillin diagnosis. " * 12  # > 300
    assert len(long_text) > 300
    call = f"archival_memory_add(key='allergy_note', value='{long_text.strip()}')"
    s.govern_calls([call])
    s.patch_results(['{"status": "Key added."}'], [call])
    recs = _read_log_records(log_file)
    dec = [r for r in recs if r["event"] == "decision"][-1]
    obs = [r for r in recs if r["event"] == "observe"][-1]
    check("decision candidate_text untruncated",
          long_text.strip() in dec["candidate_text"] and len(dec["candidate_text"]) > 300,
          str(len(dec["candidate_text"])))
    check("observe text untruncated", len(obs["text"]) > 300, str(len(obs["text"])))

    # 2d: signals.sims has one entry per stored item.
    from bfcl_eval.model_handler.middleware.governance_filter import (
        build_candidate,
        compute_signals,
    )
    s2 = make_session("vector", VECTOR_SNAPSHOT, log_file=log_file)
    cand = build_candidate("vector", "core_memory_add(text='The user is 35 years old.')")
    sig = compute_signals(s2._whiten_one(cand.text), cand, s2.cache, s2.thresholds, s2.cfg)
    check("sims length == n_items", len(sig.sims) == s2.cache.total_size() == 3, str(sig.sims))
    check("sim_max == max(sims)", abs(sig.sim_max - max(sig.sims)) < 1e-12)

    # 2e: MemoryItem carries the probes field (empty until Stage 2 populates it).
    item = next(iter(s2.cache.items["core"].values()))
    check("MemoryItem.probes present+empty", item.probes == [] and item.probe_provenance == "")


def test_latency():
    print("[latency]")
    # Full memory: 7 core + 50 archival items.
    store = {str(i): f"Fact number {i} about topic {i % 9} with detail {i * 13}." for i in range(50)}
    snapshot = {
        "core_memory": {"next_id": 7, "store": {str(i): f"Core fact {i}." for i in range(7)}},
        "archival_memory": {"next_id": 50, "store": store},
    }
    s = make_session("vector", snapshot)
    candidate = WriteCandidate(
        op="archival_memory_add", kind="add", tier="archival", backend="vector",
        text="A brand new fact about quarterly earnings of 4.2 million.",
        args={"text": "A brand new fact about quarterly earnings of 4.2 million."},
    )
    v_w = s._whiten_one(candidate.text)  # encode excluded from the budget
    # Warm-up (fills matrix/QR caches), then measure the steady-state decision path.
    compute_signals(v_w, candidate, s.cache, s.thresholds, s.cfg)
    n_iter = 200
    start = time.perf_counter()
    for _ in range(n_iter):
        compute_signals(v_w, candidate, s.cache, s.thresholds, s.cfg)
    avg_ms = (time.perf_counter() - start) * 1000 / n_iter
    check(f"decision path < 1 ms (avg {avg_ms:.3f} ms @ 57 items)", avg_ms < 1.0)


def main():
    test_artifact()
    test_vector_decisions()
    test_kv_decisions()
    test_dry_run()
    test_failed_write_not_observed()
    test_threshold_validation()
    test_logging_fixes()
    test_latency()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
