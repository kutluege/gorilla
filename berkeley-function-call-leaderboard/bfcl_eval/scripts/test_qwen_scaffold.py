"""Offline tests for QwenScaffoldHandler (alt5R deterministic write scaffold).

No server, no harness. Verifies the injection policy, capacity/turn guards,
call well-formedness against the real backend APIs, and the shuffled-control
mode. Run: python bfcl_eval/scripts/test_qwen_scaffold.py
"""

import json
import re
import sys
import tempfile
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.memory_kv import (  # noqa: E402
    MemoryAPI_kv,
)
from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.memory_vector import (  # noqa: E402
    MemoryAPI_vector,
    VectorStore,
)
from bfcl_eval.model_handler.local_inference.qwen_scaffold import (  # noqa: E402
    QwenScaffoldHandler,
    slug_key,
)
from bfcl_eval.model_handler.middleware.governance_filter import (  # noqa: E402
    GovConfig,
    parse_call,
)

PASS, FAIL = 0, 0
USER_TEXT = ("My name is Michael, I'm 35 years old and I live in Seattle. "
             "I work as a freelance graphic designer.")


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


class FakeSession:
    """Pass-through governance session (govern_calls never modifies)."""

    def __init__(self, user_text=USER_TEXT):
        self.user_text = user_text

    def govern_calls(self, calls):
        return calls


def make_handler(tmp, **over):
    h = object.__new__(QwenScaffoldHandler)
    h.scaf_enabled = True
    h.scaf_max_entries = 50
    h.scaf_max_len = 2000
    h.scaf_gate_recall = False
    h.scaf_mode = "user_turn"
    h.scaf_log_file = "scaffold_log.jsonl"
    h.scaf_verbose = False
    for k, v in over.items():
        setattr(h, k, v)
    h._scaf_tls = threading.local()
    h._gov_tls = threading.local()
    h._gov_tls.session = FakeSession()
    h.gov_config = GovConfig(log_dir=tmp, log_file="gov.jsonl")
    # QwenGovHandler.decode_execute -> QwenFCHandler.decode_execute; with no
    # governance session attached the parent path just parses the text.
    return h


def prime(h, test_id="memory_kv_prereq_0-customer-0"):
    st = h._state()
    st.update({"turn": 1, "scaffolded_turn": -1, "test_id": test_id,
               "backend": h._scaf_backend(test_id), "n_written": 0,
               "used_keys": set(), "user_text": USER_TEXT})
    return st


def read_log(tmp):
    p = Path(tmp) / "scaffold_log.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


# ------------------------------------------------------------------ policy

def test_injects_on_no_call_prereq():
    tmp = tempfile.mkdtemp(prefix="scaf_")
    h = make_handler(tmp)
    prime(h)
    calls = h.decode_execute("I have noted that, thanks!", False)
    check("one call injected", len(calls) == 1, calls)
    check("archival add", calls[0].startswith("archival_memory_add("), calls[0])
    rows = read_log(tmp)
    check("logged", len(rows) == 1 and rows[0]["event"] == "scaffold_write")
    check("log records turn + len", rows[0]["turn"] == 1 and rows[0]["text_len"] > 0)


def test_does_not_touch_model_writes():
    tmp = tempfile.mkdtemp(prefix="scaf_")
    h = make_handler(tmp)
    prime(h)
    tc = ("<tool_call>\n{\"name\": \"core_memory_add\", \"arguments\": "
          "{\"key\": \"user_age\", \"value\": \"35\"}}\n</tool_call>")
    calls = h.decode_execute(tc, False)
    check("model's own call passes through untouched",
          len(calls) == 1 and calls[0].startswith("core_memory_add("), calls)
    check("no scaffold logged", read_log(tmp) == [])


def test_once_per_turn():
    tmp = tempfile.mkdtemp(prefix="scaf_")
    h = make_handler(tmp)
    st = prime(h)
    first = h.decode_execute("no calls", False)
    second = h.decode_execute("still no calls", False)
    check("first injects", len(first) == 1)
    check("second in SAME turn does not inject", second == [], second)
    st["turn"] = 2                       # next user turn
    third = h.decode_execute("no calls again", False)
    check("new turn injects again", len(third) == 1)
    check("two scaffold rows", len(read_log(tmp)) == 2)


def test_capacity_guard():
    tmp = tempfile.mkdtemp(prefix="scaf_")
    h = make_handler(tmp, scaf_max_entries=3)
    st = prime(h)
    n = 0
    for turn in range(1, 8):
        st["turn"] = turn
        if h.decode_execute("no calls", False):
            n += 1
    check("stops at max_entries", n == 3, n)
    check("counter matches", h._state()["n_written"] == 3)


def test_gating():
    tmp = tempfile.mkdtemp(prefix="scaf_")
    h = make_handler(tmp)
    check("prereq kv gated on", h._scaf_active("memory_kv_prereq_1-customer-0"))
    check("vector prereq gated on", h._scaf_active("memory_vector_prereq_1-customer-0"))
    check("question turn off (gate_recall=0)",
          not h._scaf_active("memory_kv_1-customer-0"))
    check("rec_sum out of scope",
          not h._scaf_active("memory_rec_sum_prereq_1-customer-0"))
    check("non-memory off", not h._scaf_active("simple_20"))
    h.scaf_gate_recall = True
    check("gate_recall=1 includes question turns",
          h._scaf_active("memory_kv_1-customer-0"))
    h.scaf_enabled = False
    check("disabled -> inert", not h._scaf_active("memory_kv_prereq_1-customer-0"))
    # disabled handler must pass decode through untouched
    prime(h)
    check("disabled injects nothing", h.decode_execute("no calls", False) == [])


def test_no_user_text():
    tmp = tempfile.mkdtemp(prefix="scaf_")
    h = make_handler(tmp)
    prime(h)
    h._state()["user_text"] = ""
    check("no user text -> no injection", h.decode_execute("no calls", False) == [])
    h._state()["user_text"] = "   "
    check("blank user text -> no injection", h.decode_execute("no calls", False) == [])


def test_works_without_governance_session():
    """The scaffold arm runs GOV_ENABLED=0, so there is NO gov session; the
    handler must keep its own user-text copy and still fire."""
    tmp = tempfile.mkdtemp(prefix="scaf_")
    h = make_handler(tmp)
    h._gov_tls.session = None
    prime(h)
    calls = h.decode_execute("no calls", False)
    check("fires with no gov session", len(calls) == 1, calls)
    # and the turn hooks are what populate it
    h2 = make_handler(tmp)
    h2._gov_tls.session = None
    st = h2._state()
    st.update({"test_id": "memory_kv_prereq_0-customer-0",
               "backend": "kv", "scaffolded_turn": -1})
    h2._record_user_text([{"role": "user", "content": "Turn text here."}])
    check("_record_user_text captures", st["user_text"] == "Turn text here.")


# ------------------------------------------------- call validity vs backends

def test_kv_call_executes():
    tmp = tempfile.mkdtemp(prefix="scaf_")
    h = make_handler(tmp)
    prime(h, "memory_kv_prereq_0-customer-0")
    call = h.decode_execute("no calls", False)[0]
    name, pos, kw = parse_call(call)
    check("kv call parses", name == "archival_memory_add" and "key" in kw and "value" in kw)
    check("kv key format valid",
          bool(re.match(r"^[a-z]+(_[a-z0-9]+)*$", kw["key"])), kw["key"])
    api = MemoryAPI_kv()          # direct state: _load_scenario needs harness dirs
    api.core_memory, api.archival_memory = {}, {}
    res = api.archival_memory_add(key=kw["key"], value=kw["value"])
    check("kv backend accepts the write", "error" not in json.dumps(res).lower(), res)
    check("value carries the user text", USER_TEXT[:30] in kw["value"])


def test_vector_call_executes():
    tmp = tempfile.mkdtemp(prefix="scaf_")
    h = make_handler(tmp)
    prime(h, "memory_vector_prereq_0-customer-0")
    call = h.decode_execute("no calls", False)[0]
    name, pos, kw = parse_call(call)
    check("vector call parses", name == "archival_memory_add" and "text" in kw)
    api = MemoryAPI_vector()
    api.core_memory = VectorStore(7, 300)
    api.archival_memory = VectorStore(50, 2000)
    res = api.archival_memory_add(text=kw["text"])
    check("vector backend accepts the write", "error" not in json.dumps(res).lower(), res)


def test_kv_keys_unique_and_meaningful():
    tmp = tempfile.mkdtemp(prefix="scaf_")
    h = make_handler(tmp)
    st = prime(h)
    keys = []
    for turn in range(1, 5):
        st["turn"] = turn
        call = h.decode_execute("no calls", False)[0]
        keys.append(parse_call(call)[2]["key"])
    check("keys unique across turns", len(set(keys)) == 4, keys)
    check("first key is content-derived (BM25-visible)",
          "michael" in keys[0] or "name" in keys[0], keys[0])
    check("slug deterministic", slug_key(USER_TEXT) == slug_key(USER_TEXT))


def test_truncation_and_shuffled_control():
    tmp = tempfile.mkdtemp(prefix="scaf_")
    long_text = "word " * 3000
    h = make_handler(tmp, scaf_max_len=100)
    prime(h, "memory_vector_prereq_0-customer-0")
    h._state()["user_text"] = long_text
    call = h.decode_execute("no calls", False)[0]
    check("text truncated to max_len", len(parse_call(call)[2]["text"]) <= 100)

    tmp2 = tempfile.mkdtemp(prefix="scaf_")
    hc = make_handler(tmp2, scaf_mode="shuffled_control")
    prime(hc, "memory_vector_prereq_0-customer-0")
    ctrl = parse_call(hc.decode_execute("no calls", False)[0])[2]["text"]
    check("control differs from the real text", ctrl != USER_TEXT)
    check("control preserves volume",
          abs(len(ctrl) - len(USER_TEXT)) <= 2, (len(ctrl), len(USER_TEXT)))
    check("control preserves the word multiset",
          sorted(ctrl.split()) == sorted(USER_TEXT.split()))
    hc2 = make_handler(tempfile.mkdtemp(prefix="scaf_"), scaf_mode="shuffled_control")
    prime(hc2, "memory_vector_prereq_0-customer-0")
    ctrl2 = parse_call(hc2.decode_execute("no calls", False)[0])[2]["text"]
    check("control deterministic", ctrl == ctrl2)


def test_no_leakage_in_module():
    """The scaffold may use runtime info only -- never gold/outcomes."""
    import ast
    src = (REPO_ROOT / "bfcl_eval/model_handler/local_inference/qwen_scaffold.py"
           ).read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            for a in node.names:
                names.add(a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name)
    forbidden = {"possible_answer", "ground_truth", "agentic_checker",
                 "hnav_answer_index", "scenario_questions", "gold_index",
                 "correct_ids", "carries"}
    hits = forbidden & names
    check("no ground-truth/eval imports or attributes", not hits, hits)


def main():
    for k, fn in sorted(globals().items()):
        if k.startswith("test_"):
            print(f"[{k}]")
            fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
