"""Offline tests for QwenHactHandler shadow mode (no server, no harness run).

Fakes the OpenAI client + tokenizer on an ``object.__new__`` handler instance;
uses a REAL GovernanceSession (fabricated snapshot, tmp log dir) to exercise
the pending_hact stash -> govern_calls flush -> (test_id, step_idx) join, and
the orphan path. Run:  python bfcl_eval/scripts/test_qwen_hact_shadow.py
"""

import json
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bfcl_eval.model_handler.local_inference.qwen_hact import QwenHactHandler  # noqa: E402
from bfcl_eval.model_handler.middleware.governance_filter import (  # noqa: E402
    GovConfig,
    GovernanceSession,
)
from bfcl_eval.model_handler.middleware.hact_sampler import HactConfig  # noqa: E402

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


TOOL_CALL = (
    "<tool_call>\n{\"name\": \"core_memory_add\", "
    "\"arguments\": {\"key\": \"user_age\", \"value\": \"35\"}}\n</tool_call>"
)
TOOL_CALL_B = (
    "<tool_call>\n{\"name\": \"core_memory_add\", "
    "\"arguments\": {\"key\": \"age\", \"value\": \"35\"}}\n</tool_call>"
)
FREE_TEXT = "I have noted that."

KV_SNAPSHOT = {
    "core_memory": {"user_name": "Michael Rodriguez"},
    "archival_memory": {},
}


class FakeClient:
    """Records every completions.create kwargs; returns scripted responses."""

    def __init__(self, primary_text=TOOL_CALL, explore_texts=None, fail_on_logprobs=False):
        self.calls = []
        self.primary_text = primary_text
        self.explore_texts = explore_texts if explore_texts is not None \
            else [TOOL_CALL, TOOL_CALL_B, FREE_TEXT]
        self.fail_on_logprobs = fail_on_logprobs
        self.completions = SimpleNamespace(create=self._create)

    def _resp(self, texts):
        return SimpleNamespace(
            choices=[SimpleNamespace(text=t, logprobs=None) for t in texts],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20 * len(texts)),
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_on_logprobs and "logprobs" in kwargs:
            raise RuntimeError("logprobs not supported (scripted)")
        n = kwargs.get("n", 1)
        if n == 1:
            return self._resp([self.primary_text])
        return self._resp(self.explore_texts[: n])


def make_handler(tmp, hact_env=None, client=None):
    """Handler via object.__new__: skips server/tokenizer/GovConfig-env init."""
    h = object.__new__(QwenHactHandler)
    cfg = HactConfig(enabled=True, num_samples=4, temperature=0.7, logprobs_k=5,
                     gate_recall=False, policy="shadow", seed_base=99,
                     log_file="hact_log.jsonl", verbose=False)
    if hact_env:
        for k, v in hact_env.items():
            setattr(cfg, k, v)
    h.hact_config = cfg
    h._hact_logprob_supported = None
    h.temperature = 0.001
    h.model_path_or_id = "Qwen/Qwen3-4B-Instruct-2507"
    h.client = client if client is not None else FakeClient()
    h.tokenizer = SimpleNamespace(tokenize=lambda s: s.split())
    h.max_context_length = 32000
    h.gov_config = GovConfig(log_dir=tmp, log_file="test_gov.jsonl")
    h._gov_tls = threading.local()
    h._gov_tls.session = None
    return h


def make_session(tmp, backend="kv"):
    return GovernanceSession(
        cfg=GovConfig(log_dir=tmp, log_file="test_gov.jsonl", dry_run=True),
        backend=backend,
        test_id=f"memory_{backend}_prereq_1-test-1",
        snapshot=KV_SNAPSHOT,
        snapshot_path="<fabricated>",
    )


def make_inference_data(test_id):
    return {
        "function": [],
        "message": [{"role": "user", "content": "I am 35 years old."}],
        "_hact_test_id": test_id,
        "_hact_call_idx": 0,
    }


def read_hact_log(tmp):
    p = Path(tmp) / "hact_log.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


# ------------------------------------------------------------------ tests

def test_gate():
    tmp = tempfile.mkdtemp(prefix="hact_")
    h = make_handler(tmp)
    check("gate on for memory prereq kv",
          h._hact_gate_active("memory_kv_prereq_1-test-1"))
    check("gate on for vector prereq",
          h._hact_gate_active("memory_vector_prereq_2-test-1"))
    check("gate off for question turn (gate_recall=0)",
          not h._hact_gate_active("memory_kv_1-test-1"))
    check("gate off for rec_sum",
          not h._hact_gate_active("memory_rec_sum_prereq_1-test-1"))
    check("gate off for non-memory", not h._hact_gate_active("simple_20"))
    h.hact_config.gate_recall = True
    check("gate_recall=1 includes question turns",
          h._hact_gate_active("memory_kv_1-test-1"))
    h.hact_config.num_samples = 1
    check("n<=1 disables", not h._hact_gate_active("memory_kv_prereq_1-test-1"))
    h.hact_config.num_samples = 4
    h.hact_config.enabled = False
    check("HACT_ENABLED=0 disables", not h._hact_gate_active("memory_kv_prereq_1-test-1"))


def test_two_request_shape():
    tmp = tempfile.mkdtemp(prefix="hact_")
    h = make_handler(tmp)
    h._gov_tls.session = make_session(tmp)
    resp, latency = h._query_prompting(make_inference_data("memory_kv_prereq_1-test-1"))

    check("two requests issued", len(h.client.calls) == 2, len(h.client.calls))
    prim, expl = h.client.calls
    check("primary at handler temperature", prim["temperature"] == 0.001)
    check("primary requests logprobs=5", prim.get("logprobs") == 5)
    check("primary has deterministic seed", isinstance(prim.get("seed"), int))
    check("primary is single-sample", "n" not in prim)
    check("exploration at HACT_TEMP", expl["temperature"] == 0.7)
    check("exploration n = N-1", expl.get("n") == 3)
    check("exploration has distinct seed", expl.get("seed") != prim.get("seed"))
    check("same prompt both requests", prim["prompt"] == expl["prompt"])
    check("no logprobs on exploration", "logprobs" not in expl)
    check("returns primary response object", resp.choices[0].text == TOOL_CALL)
    check("latency is float", isinstance(latency, float))

    # deterministic seeds: same (test_id, call_idx) -> same seeds
    h2 = make_handler(tmp)
    h2._gov_tls.session = make_session(tmp)
    h2._query_prompting(make_inference_data("memory_kv_prereq_1-test-1"))
    check("seed deterministic across runs",
          h2.client.calls[0]["seed"] == prim["seed"])


def test_stash_and_flush_join():
    tmp = tempfile.mkdtemp(prefix="hact_")
    h = make_handler(tmp)
    session = make_session(tmp)
    h._gov_tls.session = session

    h._query_prompting(make_inference_data("memory_kv_prereq_1-test-1"))
    check("record stashed on session", getattr(session, "pending_hact", None) is not None)
    check("nothing logged before flush", read_hact_log(tmp) == [])

    # the harness would now call handler.decode_execute(primary_text) -> govern_calls
    calls = h.decode_execute(TOOL_CALL, False)
    check("primary decoded to one call", len(calls) == 1, calls)
    rows = read_hact_log(tmp)
    check("flush wrote one hact row", len(rows) == 1)
    row = rows[0]
    check("row has step_idx 1", row.get("step_idx") == 1, row.get("step_idx"))
    check("row not orphan", row.get("orphan") is False)
    check("row test_id joins", row["test_id"] == "memory_kv_prereq_1-test-1")
    check("stash cleared after flush", getattr(session, "pending_hact", None) is None)
    check("_log_file key stripped", "_log_file" not in row)
    check("votes present", "h_act_target" in row["votes"])
    check("full token cost recorded (primary+exploration)",
          row["input_tokens"] == 200 and row["output_tokens"] == 20 + 60)

    # gov decision log carries the same (test_id, step_idx)
    gov_rows = [json.loads(l) for l in
                (Path(tmp) / "test_gov.jsonl").read_text(encoding="utf-8").splitlines()
                if json.loads(l).get("event") == "decision"]
    check("gov decision at same step", any(
        r["test_id"] == row["test_id"] and r["step_idx"] == row["step_idx"]
        for r in gov_rows), f"{len(gov_rows)} decisions")


def test_orphan_flush():
    tmp = tempfile.mkdtemp(prefix="hact_")
    h = make_handler(tmp, client=FakeClient(primary_text=FREE_TEXT))
    session = make_session(tmp)
    h._gov_tls.session = session

    h._query_prompting(make_inference_data("memory_kv_prereq_1-test-1"))
    calls = h.decode_execute(FREE_TEXT, False)   # no tool calls -> govern_calls skipped
    check("no calls decoded", calls == [] or calls is None or len(calls) == 0)
    rows = read_hact_log(tmp)
    check("orphan flushed", len(rows) == 1)
    check("orphan step_idx None", rows[0]["step_idx"] is None)
    check("orphan flag true", rows[0]["orphan"] is True)
    check("no-call orphan not a parse crash", rows[0]["parse_crash"] is False)
    check("orphan votes still computed", "h_act_target" in rows[0]["votes"])
    check("stash cleared", getattr(session, "pending_hact", None) is None)


def test_orphan_flush_on_raising_decode():
    # Malformed-but-matching tool_call JSON makes QwenFCHandler.decode_execute
    # RAISE (not return empty); the pending record must still be flushed as a
    # parse-crash orphan before the exception propagates (T2 positive class).
    bad = "<tool_call>\n{\"name\": \"core_memory_add\"}\n</tool_call>"
    tmp = tempfile.mkdtemp(prefix="hact_")
    h = make_handler(tmp, client=FakeClient(primary_text=bad))
    session = make_session(tmp)
    h._gov_tls.session = session

    h._query_prompting(make_inference_data("memory_kv_prereq_1-test-1"))
    check("record stashed", getattr(session, "pending_hact", None) is not None)
    raised = False
    try:
        h.decode_execute(bad, False)
    except Exception:
        raised = True
    check("decode raised as the harness expects", raised)
    rows = read_hact_log(tmp)
    check("parse-crash orphan flushed", len(rows) == 1, len(rows))
    check("parse_crash flag set", rows[0].get("parse_crash") is True)
    check("orphan step_idx None", rows[0]["step_idx"] is None)
    check("stash cleared after crash flush",
          getattr(session, "pending_hact", None) is None)


def test_exploration_never_governed():
    tmp = tempfile.mkdtemp(prefix="hact_")
    h = make_handler(tmp)
    session = make_session(tmp)
    h._gov_tls.session = session

    h._query_prompting(make_inference_data("memory_kv_prereq_1-test-1"))
    check("session step untouched by sampling", session.step == 0, session.step)
    h.decode_execute(TOOL_CALL, False)
    check("exactly one governed step", session.step == 1, session.step)
    gov_rows = [json.loads(l) for l in
                (Path(tmp) / "test_gov.jsonl").read_text(encoding="utf-8").splitlines()]
    n_decisions = sum(1 for r in gov_rows if r.get("event") == "decision")
    check("one decision (primary only, not 4 samples)", n_decisions == 1, n_decisions)


def test_votes_capture_disagreement():
    tmp = tempfile.mkdtemp(prefix="hact_")
    # primary + explore: two 'user_age' adds, one 'age' add, one free text
    h = make_handler(tmp)
    session = make_session(tmp)
    h._gov_tls.session = session
    h._query_prompting(make_inference_data("memory_kv_prereq_1-test-1"))
    rec = session.pending_hact
    v = rec["votes"]
    check("4 samples voted", v["n_samples_target"] == 4)
    check("R2 disagreement only via no_call sample", v["n_unique_op"] == 2, v)
    check("R3 sees target split", v["n_unique_target"] == 3, v)
    check("primary signature recorded",
          rec["primary"]["signature_r3"] == "add:core|user_age")
    check("no_call counted", v["n_no_call_target"] == 1)
    check("primary is modal here", v["primary_matches_modal_target"] is True, v)


def test_logprob_degradation():
    tmp = tempfile.mkdtemp(prefix="hact_")
    h = make_handler(tmp, client=FakeClient(fail_on_logprobs=True))
    session = make_session(tmp)
    h._gov_tls.session = session
    h._query_prompting(make_inference_data("memory_kv_prereq_1-test-1"))
    check("degraded flag set", h._hact_logprob_supported is False)
    # 3 calls: failed primary w/ logprobs, retried primary, exploration
    check("retried without logprobs", len(h.client.calls) == 3, len(h.client.calls))
    check("retry has no logprobs kwarg", "logprobs" not in h.client.calls[1])
    rec = session.pending_hact
    check("record survives degradation",
          rec is not None and rec["logprobs_supported"] is False)
    check("lp features all None", all(v is None for v in rec["logprob_features"].values()))
    # next step must not request logprobs at all
    h._query_prompting(make_inference_data("memory_kv_prereq_1-test-1"))
    check("subsequent primary skips logprobs", "logprobs" not in h.client.calls[3])


def test_session_missing_fallback():
    tmp = tempfile.mkdtemp(prefix="hact_")
    h = make_handler(tmp)
    h._gov_tls.session = None
    h._query_prompting(make_inference_data("memory_kv_prereq_1-test-1"))
    rows = read_hact_log(tmp)
    check("logged directly without session", len(rows) == 1)
    check("session_missing marked", rows[0].get("session_missing") is True)


def test_ungated_passthrough_params():
    tmp = tempfile.mkdtemp(prefix="hact_")
    h = make_handler(tmp)
    h._gov_tls.session = None
    # question turn with gate_recall=0 -> base handler path (single plain request)
    h._query_prompting(make_inference_data("memory_kv_1-test-1"))
    check("single request when ungated", len(h.client.calls) == 1)
    req = h.client.calls[0]
    check("ungated request has no seed/logprobs/n",
          all(k not in req for k in ("seed", "logprobs", "n")), req.keys())
    check("ungated at handler temperature", req["temperature"] == 0.001)
    check("no hact row for ungated turn", read_hact_log(tmp) == [])


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"[{fn.__name__}]")
        fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
