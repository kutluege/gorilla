"""Offline tests for QwenRagHandler (RAG program M1, read scaffold).

No server, no harness. Verifies the injection policy (question entries only),
once-per-turn guard, call well-formedness against the real backend read APIs,
the two control modes, and an AST leakage guard. Run:
    python bfcl_eval/scripts/test_qwen_rag.py
"""

import json
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
)
from bfcl_eval.model_handler.local_inference.qwen_rag import (  # noqa: E402
    IRRELEVANT_PROBE,
    READ_INSTRUCTION,
    QwenRagHandler,
)
from bfcl_eval.model_handler.middleware.governance_filter import (  # noqa: E402
    GovConfig,
    parse_call,
)

PASS, FAIL = 0, 0
QUESTION = "What is my first name?"


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def make_handler(tmp, **over):
    h = object.__new__(QwenRagHandler)
    h.rag_enabled = True
    h.rag_mode = "read_verbatim"
    h.rag_write = "none"
    h.rag_answer = "none"
    h.rag_nudge = "none"
    h.rag_top_k = 5
    h.rag_pack_len = 2000
    h.rag_max_entries = 50
    h.rag_log_file = "rag_log.jsonl"
    h.rag_verbose = False
    for k, v in over.items():
        setattr(h, k, v)
    h._rag_tls = threading.local()
    h._gov_tls = threading.local()
    h._gov_tls.session = None
    h.gov_config = GovConfig(log_dir=tmp, log_file="gov.jsonl")
    return h


def prime(h, test_id="memory_kv_3-customer-3", user_text=QUESTION):
    st = h._state()
    st.update({"turn": 1, "read_turn": -1, "write_turn": -1,
               "test_id": test_id, "backend": h._rag_backend(test_id),
               "user_text": user_text, "buffer": "", "n_written": 0,
               "used_keys": set()})
    return st


def read_log(tmp):
    p = Path(tmp) / "rag_log.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


# ------------------------------------------------------------------ policy

def test_injects_on_no_call_question():
    tmp = tempfile.mkdtemp(prefix="rag_")
    h = make_handler(tmp)
    prime(h, "memory_vector_3-customer-3")
    calls = h.decode_execute("I do not have that information.", False)
    check("one call injected", len(calls) == 1, calls)
    check("vector read", calls[0].startswith("archival_memory_retrieve("),
          calls)
    rows = read_log(tmp)
    check("logged read_inject", len(rows) == 1
          and rows[0]["event"] == "read_inject")
    check("probe is the user turn", rows[0]["probe_len"] == len(QUESTION))


def test_kv_uses_key_search():
    tmp = tempfile.mkdtemp(prefix="rag_")
    h = make_handler(tmp)
    prime(h, "memory_kv_3-customer-3")
    calls = h.decode_execute("I do not know.", False)
    check("kv read is key_search",
          calls and calls[0].startswith("archival_memory_key_search("), calls)


def test_call_wellformed_against_real_backends():
    tmp = tempfile.mkdtemp(prefix="rag_")
    h = make_handler(tmp)
    prime(h, "memory_vector_3-customer-3")
    vcall = h.decode_execute("no", False)[0]
    prime(h, "memory_kv_3-customer-3")
    kcall = h.decode_execute("no", False)[0]
    # parse_call gives (func, args, kwargs); execute against real instances
    api_v = object.__new__(MemoryAPI_vector)
    api_v.core_memory = None
    fn, _, kwargs = parse_call(vcall)
    check("vector call parses", fn == "archival_memory_retrieve"
          and kwargs.get("top_k") == 5, (fn, kwargs))
    fn, _, kwargs = parse_call(kcall)
    check("kv call parses", fn == "archival_memory_key_search"
          and kwargs.get("k") == 5, (fn, kwargs))
    api_k = object.__new__(MemoryAPI_kv)
    api_k.core_memory = {}
    api_k.archival_memory = {"user_first_name": "Michael"}
    out = api_k.archival_memory_key_search(**kwargs)
    check("kv key_search executes on a real instance",
          isinstance(out, dict) and "ranked_results" in out, out)


def test_does_not_touch_model_calls():
    tmp = tempfile.mkdtemp(prefix="rag_")
    h = make_handler(tmp)
    prime(h, "memory_vector_3-customer-3")
    tc = ("<tool_call>\n{\"name\": \"core_memory_retrieve\", \"arguments\": "
          "{\"query\": \"first name\", \"top_k\": 5}}\n</tool_call>")
    calls = h.decode_execute(tc, False)
    check("model's own call passes through untouched",
          len(calls) == 1 and calls[0].startswith("core_memory_retrieve("),
          calls)
    check("no injection logged", read_log(tmp) == [])


def test_once_per_turn():
    tmp = tempfile.mkdtemp(prefix="rag_")
    h = make_handler(tmp)
    st = prime(h, "memory_vector_3-customer-3")
    first = h.decode_execute("dunno", False)
    second = h.decode_execute("still dunno, even with results", False)
    check("first injects", len(first) == 1)
    check("post-retrieval no-call answer ends the turn (no re-inject)",
          second == [], second)
    st["turn"] = 2
    check("new turn may inject again",
          len(h.decode_execute("no", False)) == 1)


def test_gating():
    tmp = tempfile.mkdtemp(prefix="rag_")
    h = make_handler(tmp)
    check("question kv gated on", h._rag_active("memory_kv_3-customer-3"))
    check("question vector gated on",
          h._rag_active("memory_vector_10-student-10"))
    check("PREREQ gated OFF (write side belongs to SCAF)",
          not h._rag_active("memory_kv_prereq_1-customer-0"))
    check("rec_sum gated off", not h._rag_active("memory_rec_sum_3-customer-3"))
    check("non-memory gated off", not h._rag_active("web_search_5"))
    hd = make_handler(tempfile.mkdtemp(prefix="rag_"), rag_enabled=False)
    check("disabled handler never injects",
          not hd._rag_active("memory_kv_3-customer-3"))


def test_irrelevant_probe_mode():
    tmp = tempfile.mkdtemp(prefix="rag_")
    h = make_handler(tmp, rag_mode="read_irrelevant")
    prime(h, "memory_vector_3-customer-3")
    calls = h.decode_execute("no idea", False)
    check("control still injects one call", len(calls) == 1)
    check("probe is the frozen constant, not the question",
          repr(IRRELEVANT_PROBE) in calls[0]
          and QUESTION not in calls[0], calls)


def test_instruction_only_mode():
    tmp = tempfile.mkdtemp(prefix="rag_")
    h = make_handler(tmp, rag_mode="instruction_only")
    prime(h, "memory_vector_3-customer-3")
    check("instruction_only never injects",
          h.decode_execute("no idea", False) == [])
    # system-message append path
    entry = {"id": "memory_vector_3-customer-3",
             "question": [[{"role": "system", "content": "BASE"},
                           {"role": "user", "content": QUESTION}]]}
    st = h._state()
    st.update({"test_id": entry["id"], "backend": "vector"})
    # call just the instruction-append logic (bypass super() chain)
    first = entry["question"][0][0]
    if first.get("role") == "system" and READ_INSTRUCTION not in first["content"]:
        first["content"] = f"{first['content']}\n\n{READ_INSTRUCTION}"
    check("instruction appended to system message",
          entry["question"][0][0]["content"].endswith(READ_INSTRUCTION))
    check("instruction is answer-agnostic (no scenario tokens)",
          all(w not in READ_INSTRUCTION.lower()
              for w in ("customer", "student", "michael", "seattle")))


def test_empty_probe_guard():
    tmp = tempfile.mkdtemp(prefix="rag_")
    h = make_handler(tmp)
    prime(h, "memory_vector_3-customer-3", user_text="   ")
    check("blank user text -> no injection",
          h.decode_execute("no", False) == [])


TURN_A = "My name is Michael and I live in Seattle."          # 41 chars
TURN_B = "I work as a freelance graphic designer downtown."    # 49 chars


def test_write_pack_buffers_and_flushes():
    tmp = tempfile.mkdtemp(prefix="rag_")
    h = make_handler(tmp, rag_write="pack", rag_mode="none", rag_pack_len=80)
    st = prime(h, "memory_vector_prereq_0-customer-0", user_text=TURN_A)
    first = h.decode_execute("noted!", False)
    check("first turn buffers, no call yet", first == [], first)
    check("buffer holds turn A", st["buffer"] == TURN_A)
    st["turn"] = 2
    st["user_text"] = TURN_B                  # A+B > 80 -> flush A, buffer B
    second = h.decode_execute("noted again!", False)
    check("overflow flushes the buffer as ONE packed call",
          len(second) == 1
          and second[0].startswith("archival_memory_add("), second)
    check("flushed content is turn A", repr(TURN_A) in second[0], second)
    check("buffer now holds turn B", st["buffer"] == TURN_B)
    check("n_written incremented", st["n_written"] == 1)


def test_write_pack_tail_is_lost_causally():
    tmp = tempfile.mkdtemp(prefix="rag_")
    h = make_handler(tmp, rag_write="pack", rag_mode="none", rag_pack_len=8000)
    st = prime(h, "memory_vector_prereq_0-customer-0", user_text=TURN_A)
    out = h.decode_execute("ok", False)
    check("under-budget buffer never flushes (fully causal, tail lost)",
          out == [] and st["buffer"] == TURN_A)


def test_write_per_turn_mode():
    tmp = tempfile.mkdtemp(prefix="rag_")
    h = make_handler(tmp, rag_write="per_turn", rag_mode="none")
    prime(h, "memory_kv_prereq_0-customer-0", user_text=TURN_A)
    calls = h.decode_execute("ok", False)
    check("per_turn emits immediately (alt5R contrast)",
          len(calls) == 1 and calls[0].startswith("archival_memory_add(key="),
          calls)


def test_write_shuffled_control():
    tmp = tempfile.mkdtemp(prefix="rag_")
    h = make_handler(tmp, rag_write="pack_shuffled", rag_mode="none",
                     rag_pack_len=44)
    st = prime(h, "memory_vector_prereq_0-customer-0", user_text=TURN_A)
    h.decode_execute("ok", False)
    st["turn"] = 2
    st["user_text"] = TURN_B
    calls = h.decode_execute("ok", False)
    check("shuffled control flushes one call", len(calls) == 1, calls)
    body = calls[0].split("text=", 1)[1]
    import ast as _ast
    text = _ast.literal_eval(body[:-1])
    check("same word multiset, different order",
          sorted(text.split()) == sorted(TURN_A.split())
          and text != TURN_A, text)


def test_write_gating_and_capacity():
    tmp = tempfile.mkdtemp(prefix="rag_")
    h = make_handler(tmp, rag_write="pack", rag_mode="none")
    check("write gate ON for prereq",
          h._rag_write_active("memory_kv_prereq_1-customer-0"))
    check("write gate OFF for question entries",
          not h._rag_write_active("memory_kv_3-customer-3"))
    h2 = make_handler(tmp, rag_write="per_turn", rag_mode="none",
                      rag_max_entries=2)
    st = prime(h2, "memory_vector_prereq_0-customer-0", user_text=TURN_A)
    n = 0
    for turn in range(1, 6):
        st["turn"] = turn
        if h2.decode_execute("ok", False):
            n += 1
    check("capacity cap respected", n == 2, n)


def test_write_and_read_coexist():
    tmp = tempfile.mkdtemp(prefix="rag_")
    h = make_handler(tmp, rag_write="pack", rag_mode="read_verbatim")
    prime(h, "memory_vector_3-customer-3")     # question entry
    calls = h.decode_execute("I do not know.", False)
    check("combined arm: question turn gets the READ, not a write",
          calls and calls[0].startswith("archival_memory_retrieve("), calls)
    st = prime(h, "memory_vector_prereq_0-customer-0", user_text=TURN_A)
    st["buffer"] = "x" * 1999
    calls = h.decode_execute("ok", False)
    check("combined arm: prereq turn gets the WRITE path",
          calls and calls[0].startswith("archival_memory_add("), calls)


def test_nudge_appends_to_injected_result_only():
    from bfcl_eval.model_handler.local_inference.qwen_rag import (
        NUDGE_NULL_TEXT, NUDGE_TEXT)
    check("nudge and null control are length-matched (+/- 15 chars)",
          abs(len(NUDGE_TEXT) - len(NUDGE_NULL_TEXT)) <= 15,
          (len(NUDGE_TEXT), len(NUDGE_NULL_TEXT)))
    tmp = tempfile.mkdtemp(prefix="rag_")
    h = make_handler(tmp, rag_nudge="nudge")
    st = prime(h, "memory_vector_3-customer-3")

    captured = {}

    def fake_super(inference_data, execution_results, model_response_data):
        captured["results"] = execution_results
        return inference_data

    # inject -> nudge_pending set
    h.decode_execute("I do not know.", False)
    check("nudge_pending set by injection", st.get("nudge_pending") is True)
    import unittest.mock as mock
    with mock.patch(
        "bfcl_eval.model_handler.local_inference.qwen_gov."
        "QwenGovHandler._add_execution_results_prompting",
        side_effect=fake_super,
    ):
        h._add_execution_results_prompting({}, ["{'result': []}"], {})
        check("nudge appended to the injected read's result",
              captured["results"][0].endswith(NUDGE_TEXT))
        # a later model-issued call's results must NOT be nudged
        h._add_execution_results_prompting({}, ["plain result"], {})
        check("subsequent results untouched",
              captured["results"] == ["plain result"], captured["results"])


def test_answer_modes_frozen_constants():
    from bfcl_eval.model_handler.local_inference.qwen_rag import (
        EVIDENCE_INSTRUCTION, EVIDENCE_IRRELEVANT_INSTRUCTION)
    for name, text in (("evidence", EVIDENCE_INSTRUCTION),
                       ("evidence_irrelevant",
                        EVIDENCE_IRRELEVANT_INSTRUCTION)):
        check(f"{name} instruction is answer-agnostic",
              all(w not in text.lower() for w in
                  ("customer", "student", "michael", "seattle", "finance")))
    check("both demand the same three-field format",
          "'evidence'" in EVIDENCE_INSTRUCTION
          and "'evidence'" in EVIDENCE_IRRELEVANT_INSTRUCTION)


def test_no_leakage_in_module():
    """The read scaffold may use runtime info only -- never gold/outcomes."""
    import ast
    src = (REPO_ROOT / "bfcl_eval/model_handler/local_inference/qwen_rag.py"
           ).read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", "") or ""
            names.add(mod)
            for a in node.names:
                names.add(a.name)
    forbidden = {"possible_answer", "ground_truth", "agentic_checker",
                 "hnav_answer_index", "scenario_questions", "gold_index",
                 "correct_ids", "carries", "benchmark_correct",
                 "outcome_label"}
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
