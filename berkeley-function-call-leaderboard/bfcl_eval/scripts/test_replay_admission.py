"""
Offline tests for the admission replay harness (Plan v2 Step 7, SS17.2 replay list).

Run:  python bfcl_eval/scripts/test_replay_admission.py

Fixture strategy: drive a LIVE GovernanceSession (per policy) over a small call
sequence so its governance log contains genuine rehydrate/decision/observe
events, then replay that log with replay_admission and assert (a) every
computable decision's Stage-0 signals match the live run bit-for-bit,
(b) legacy_full final decisions match, (c) two replays are byte-identical.
The 5-log campaign acceptance (1787/1787 legacy bit-identity) runs against the
committed gov_logs/replicates via the CLI -- this suite guards the machinery.
"""

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bfcl_eval.model_handler.middleware.governance_filter import (  # noqa: E402
    GovConfig,
    GovernanceSession,
)
from bfcl_eval.scripts.replay_admission import build_cfg, replay  # noqa: E402

TMP = tempfile.mkdtemp(prefix="gov_replay_adm_test_")
PASS, FAIL = 0, 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def make_cfg(policy, log_file, **overrides):
    cfg = GovConfig(log_dir=TMP, log_file=log_file, policy=policy)
    cfg.me_shadow = False
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def drive_session(policy, log_file, backend="vector"):
    """Fresh conversation (snapshot=None -> items_loaded=0 checkpoint), three
    write steps with genuine observe events, one duplicate re-write."""
    session = GovernanceSession(
        cfg=make_cfg(policy, log_file),
        backend=backend,
        test_id=f"memory_{backend}_scenario_prereq-0",
        snapshot=None,
        snapshot_path="<fabricated>",
    )
    session.user_text = "My name is Ada Lovelace and I work at Analytical Engines"
    steps = [
        ("core_memory_add(text='The user is named Ada Lovelace.')",
         json.dumps({"id": 0})),
        ("core_memory_add(text='The user works at Analytical Engines.')",
         json.dumps({"id": 1})),
        # near-duplicate of step 1 -- exercises NOOP / escalation paths
        ("core_memory_add(text='The user is named Ada Lovelace.')",
         json.dumps({"id": 2})),
    ]
    for call, result in steps:
        governed = session.govern_calls([call])
        session.patch_results([result], list(governed))
    return session


def replay_log(policy, log_file, extra_env=()):
    env = [f"GOV_POLICY={policy}", "GOV_ME_SHADOW=0"] + list(extra_env)
    cfg = build_cfg(env)
    return replay(Path(TMP) / log_file, cfg, policy)


def test_policy(policy, extra_env=()):
    print(f"[replay: {policy}]")
    log_file = f"{policy}.jsonl"
    drive_session(policy, log_file)
    rows, counters = replay_log(policy, log_file, extra_env)
    n = counters.get("decisions_total", 0)
    check("all decisions computable", n > 0 and counters.get("decisions_computable") == n,
          str(dict(counters)))
    check("stage-0 signals all match",
          counters.get("stage0_signals_mismatch", 0) == 0
          and counters.get("stage0_signals_match", 0) == n, str(dict(counters)))
    check("no stage-0 outcome mismatch", counters.get("stage0_mismatch", 0) == 0)
    check("final decisions all match",
          counters.get("decision_mismatch", 0) == 0
          and counters.get("decision_match", 0) == n, str(dict(counters)))
    if policy == "geometry_margin_entropy_v1":
        acted = sum(v for k, v in counters.items() if k.startswith("v1_action_"))
        stage0_final = sum(
            1 for r in rows if r.get("replayed_stage") == "GEOMETRY"
        )
        check("every escalation got a v1 action",
              acted + stage0_final == n, f"acted={acted} s0={stage0_final} n={n}")
    # Determinism: replay twice, byte-compare.
    rows2, _ = replay_log(policy, log_file, extra_env)
    check("deterministic replay",
          json.dumps(rows, sort_keys=True) == json.dumps(rows2, sort_keys=True))
    return rows


def test_desync_accounting():
    print("[desync accounting]")
    # A log that starts mid-chain (items_loaded > 0 with no prior state) must
    # drop its decisions EXPLICITLY, never silently score them.
    log_file = "desync.jsonl"
    path = Path(TMP) / log_file
    records = [
        {"event": "rehydrate", "test_id": "memory_kv_scenario_prereq-1",
         "backend": "kv", "items_loaded": 3, "dry_run": False},
        {"event": "decision", "test_id": "memory_kv_scenario_prereq-1",
         "backend": "kv", "op": "core_memory_add", "tier": "core",
         "candidate_ref": "user_name", "candidate_text": "user name: Ada",
         "n_items": 3, "decision": "ADD", "reason": "novel"},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    cfg = build_cfg(["GOV_POLICY=geometry_only"])
    rows, counters = replay(path, cfg, "geometry_only")
    check("mid-chain start counted as desync", counters.get("checkpoints_desync", 0) == 1)
    check("decision dropped explicitly", counters.get("dropped_desync", 0) == 1)
    check("dropped row carries reason", rows[0]["drop_reason"] == "desync")


def main():
    test_policy("legacy_full")  # NLI off in fixture: escalations -> fallback ADD
    test_policy("geometry_only")
    test_policy("geometry_margin_entropy_v1")
    test_desync_accounting()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
