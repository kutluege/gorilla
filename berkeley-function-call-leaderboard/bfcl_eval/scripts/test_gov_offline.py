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
    test_latency()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
