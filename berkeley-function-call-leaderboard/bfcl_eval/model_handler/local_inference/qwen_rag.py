"""
QwenRagHandler
==============

Deterministic retrieve-then-generate READ scaffold (RAG program, method M1).

Why: the tier-conditional loss analysis (gov_logs/hnav_autonomous/
tier_conditional_factors.json) showed that gold facts carried by CORE memory
convert to correct answers at 0.556/0.701 (kv/vector) because core is
auto-dumped into the system prompt, while gold carried ONLY by ARCHIVAL
memory converts at 0.017/0.000 -- the agent reads archival memory on 1 of
155 question entries. Archival is a write-only black hole at question time.
Capture without read is worth ~0 (the corrected alt5R prediction); this
handler supplies the read half.

Mechanism (mirror of QwenScaffoldHandler, on the read channel): on a memory
QUESTION entry, when the model's response decodes to no tool calls and the
turn has not been read-scaffolded yet, ONE archival read call is injected:

    vector:  archival_memory_retrieve(query=<probe>, top_k=K)
    kv:      archival_memory_key_search(query=<probe>, k=K)

The harness executes it through the real backend, the result enters context
via the normal execution-result path, and the model answers on the next step.
The injection fires at most once per turn, so the model's post-retrieval
no-call answer terminates the turn normally and is what the grader sees.

Legitimacy: agent read policy only. The probe is the current user turn text
(or a frozen generic string in the control arm). No gold, no future turns,
no evaluation outcomes, no benchmark/grader changes. Separate registry id so
-FC stays byte-identical for A/B.

Env vars:
    RAG_ENABLED    master switch (default 0)
    RAG_MODE       read_verbatim (M1) | read_irrelevant (control: identical
                   call + step cost, frozen generic probe carrying no
                   question-specific information) | instruction_only
                   (control: system-prompt nudge, no injection -- separates
                   deterministic mechanism from prompting)
    RAG_TOP_K      k for the injected read (default 5, the backend default;
                   frozen for C1, varied only by the pre-registered C3 arms)
    RAG_LOG_FILE   audit log name inside GOV_LOG_DIR (default rag_log.jsonl)
    RAG_VERBOSE    print per-injection lines (default 0)
"""

import json
import os
import threading
import time
from pathlib import Path

from bfcl_eval.model_handler.local_inference.qwen_gov import (
    GOVERNED_BACKENDS,
    QwenGovHandler,
)
from bfcl_eval.utils import (
    extract_memory_backend_type,
    extract_test_category_from_id,
    is_memory,
    is_memory_prereq,
)
from overrides import override

MODES = ("read_verbatim", "read_irrelevant", "instruction_only")

# Frozen control probe: same call, same extra step, same context growth, but
# carries no information about the specific question. (Note, recorded at
# pre-registration: at baseline archival sizes <= top_k the retrieved SET is
# the whole store for any probe, so in C1 this control isolates step/volume
# effects; probe RELEVANCE becomes discriminative once capture fills the
# index in C2+.)
IRRELEVANT_PROBE = "Please recall any stored information relevant to this conversation."

# Frozen instruction-only control text (answer-agnostic, identical for every
# entry and backend).
READ_INSTRUCTION = (
    "Before answering any question about the user, first search your "
    "archival memory for relevant stored facts (archival_memory_retrieve or "
    "archival_memory_key_search), then answer using what you find."
)

_LOG_LOCK = threading.Lock()


def _flag(name, default="0"):
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _py_str(s):
    """Render a Python string literal for a decoded call string."""
    return repr(str(s))


class QwenRagHandler(QwenGovHandler):
    def __init__(self, model_name, temperature, registry_name, is_fc_model,
                 dtype="bfloat16", **kwargs) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model,
                         dtype=dtype, **kwargs)
        self.rag_enabled = _flag("RAG_ENABLED")
        self.rag_mode = os.getenv("RAG_MODE", "read_verbatim").strip().lower()
        if self.rag_mode not in MODES:
            raise ValueError(f"RAG_MODE must be one of {MODES}")
        self.rag_top_k = int(os.getenv("RAG_TOP_K", "5"))
        self.rag_log_file = os.getenv("RAG_LOG_FILE", "rag_log.jsonl")
        self.rag_verbose = _flag("RAG_VERBOSE")
        self._rag_tls = threading.local()
        print(f"[RAG] QwenRagHandler active | enabled={self.rag_enabled} "
              f"mode={self.rag_mode} top_k={self.rag_top_k}")

    # ---------------------------------------------------------- turn state

    def _state(self):
        st = getattr(self._rag_tls, "state", None)
        if st is None:
            st = {"turn": 0, "read_turn": -1, "test_id": "", "backend": "",
                  "user_text": ""}
            self._rag_tls.state = st
        return st

    def _rag_backend(self, test_id):
        try:
            return extract_memory_backend_type(
                extract_test_category_from_id(test_id, remove_prereq=True))
        except Exception:
            return ""

    def _record_user_text(self, messages):
        texts = [str(m.get("content", "")) for m in messages
                 if isinstance(m, dict) and m.get("role") == "user"]
        if texts:
            self._state()["user_text"] = texts[-1]

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        inference_data = super()._pre_query_processing_prompting(test_entry)
        test_id = test_entry.get("id", "")
        st = self._state()
        st.update({"turn": 0, "read_turn": -1, "test_id": test_id,
                   "backend": self._rag_backend(test_id), "user_text": ""})
        if (self.rag_mode == "instruction_only"
                and self._rag_entry_gated(test_id)):
            # Append the frozen nudge to the system message that
            # system_prompt_pre_processing_chat_model just built. Constant,
            # answer-agnostic, identical across entries and backends.
            try:
                first = test_entry["question"][0][0]
                if first.get("role") == "system" \
                        and READ_INSTRUCTION not in str(first.get("content")):
                    first["content"] = f"{first['content']}\n\n{READ_INSTRUCTION}"
                    self._log({"ts": time.time(), "event": "instruction_added",
                               "test_id": test_id, "backend": st["backend"]})
            except (KeyError, IndexError, TypeError):
                self._log({"ts": time.time(), "event": "instruction_skip",
                           "test_id": test_id, "reason": "no system message"})
        return inference_data

    def _bump_turn(self):
        self._state()["turn"] += 1

    @override
    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        self._bump_turn()
        self._record_user_text(first_turn_message)
        return super().add_first_turn_message_prompting(
            inference_data, first_turn_message)

    @override
    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        self._bump_turn()
        self._record_user_text(user_message)
        return super()._add_next_turn_user_message_prompting(
            inference_data, user_message)

    # ---------------------------------------------------------- the policy

    def _rag_entry_gated(self, test_id):
        """QUESTION entries of governed memory backends only -- the exact
        complement of the write scaffold's prereq gate."""
        if not test_id or not is_memory(test_id):
            return False
        if is_memory_prereq(test_id):
            return False
        return self._rag_backend(test_id) in GOVERNED_BACKENDS

    def _rag_active(self, test_id):
        if not self.rag_enabled:
            return False
        if self.rag_mode == "instruction_only":
            return False          # that mode never injects
        return self._rag_entry_gated(test_id)

    def _build_read_call(self, backend, probe):
        if backend == "kv":
            return (f"archival_memory_key_search(query={_py_str(probe)}, "
                    f"k={self.rag_top_k})")
        return (f"archival_memory_retrieve(query={_py_str(probe)}, "
                f"top_k={self.rag_top_k})")

    def _log(self, rec):
        log_dir = getattr(self.gov_config, "log_dir", "") or "."
        path = Path(log_dir) / self.rag_log_file
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @override
    def decode_execute(self, result, has_tool_call_tag):
        calls = super().decode_execute(result, has_tool_call_tag)
        st = self._state()
        test_id = st.get("test_id", "")
        if calls or not self._rag_active(test_id):
            return calls
        if st["read_turn"] == st["turn"]:
            return calls          # already injected this turn: the model's
            #                       post-retrieval no-call answer ends the turn
        probe = (IRRELEVANT_PROBE if self.rag_mode == "read_irrelevant"
                 else (st.get("user_text") or ""))
        if not probe.strip():
            return calls
        call = self._build_read_call(st["backend"], probe)
        st["read_turn"] = st["turn"]
        self._log({
            "ts": time.time(), "event": "read_inject", "test_id": test_id,
            "turn": st["turn"], "backend": st["backend"], "mode": self.rag_mode,
            "top_k": self.rag_top_k, "probe_len": len(probe), "call": call,
        })
        if self.rag_verbose:
            print(f"[RAG] {test_id} turn={st['turn']} -> {call[:90]}")
        return [call]
