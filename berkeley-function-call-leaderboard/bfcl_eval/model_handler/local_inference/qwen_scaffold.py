"""
QwenScaffoldHandler
===================

Deterministic write scaffold (autonomous-loop alternative alt5R). A prereq
user turn that ends without any resolved memory write gets the user turn text
archived automatically, MemGPT-style.

Why: the loss decomposition on the H-Nav shadow campaign showed ~80% of BFCL
v4 Memory questions have their gold fact absent from the final store, and ~88%
of those facts were stated verbatim in the prereq conversation. The agent had
the information and never wrote it. Write GATING (Stage 1) and action
uncertainty (Stage 3) both failed on this benchmark because they address
classes that are nearly empty; write CAPTURE is where the mass is.

Mechanism (one hook, `decode_execute`): when the model returns no calls on a
gated prereq turn and that turn has not been scaffolded yet, a single
``archival_memory_add`` call is APPENDED to the decoded call list. The harness
executes it through the normal backend path, so the write is real, capacity
limits are enforced by the backend, and the call + its result appear in the
result log exactly like a model-issued write (full auditability). The turn
then continues for one more model step, so the model sees the tool result.

Legitimacy: this is an agent memory policy, not a benchmark change. It uses
only information available at that moment (the current user turn text). It
never touches the grader, the dataset, gold answers, prereq mappings, or any
evaluation outcome. Its registry id is separate so the -FC baseline stays
byte-identical for A/B.

Env vars:
    SCAF_ENABLED      master switch (default 0)
    SCAF_MAX_ENTRIES  per-scenario scaffold write cap (default 50, the
                      backend's MAX_ARCHIVAL_SIZE)
    SCAF_MAX_LEN      truncate the archived text (default 2000, the backend's
                      MAX_ARCHIVAL_ENTRY_LENGTH)
    SCAF_GATE_RECALL  1 = every memory turn; 0 (default) = prereq turns only
    SCAF_MODE         user_turn (default) | shuffled_control (negative
                      control: archives a deterministically shuffled
                      cross-scenario text of equal volume)
    SCAF_LOG_FILE     audit log name inside GOV_LOG_DIR (default scaffold_log.jsonl)
    SCAF_VERBOSE      print per-injection lines (default 0)
"""

import hashlib
import json
import os
import re
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

MAX_ENTRIES_DEFAULT = 50      # memory_kv/memory_vector MAX_ARCHIVAL_SIZE
MAX_LEN_DEFAULT = 2000        # memory_kv/memory_vector MAX_ARCHIVAL_ENTRY_LENGTH
MODES = ("user_turn", "shuffled_control")

_STOP = {"the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for",
         "with", "is", "am", "are", "was", "were", "be", "been", "i", "my",
         "me", "you", "your", "it", "its", "this", "that", "so", "as", "at",
         "by", "we", "our", "us", "they", "them", "he", "she", "his", "her"}

_LOG_LOCK = threading.Lock()


def _flag(name, default="0"):
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def slug_key(text, n_tokens=6, salt=0):
    """Deterministic content-derived KV key.

    KV retrieval is BM25 over KEY NAMES only, so an opaque key would make the
    entry unretrievable by construction. Must satisfy
    memory_kv._is_valid_key_format: ^[a-z]+(_[a-z0-9]+)*$
    """
    toks = [t for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in _STOP and not t.isdigit()][:n_tokens]
    if not toks:
        toks = ["note"]
    key = "_".join(toks)
    if not re.match(r"^[a-z]+(_[a-z0-9]+)*$", key):
        key = "note"
    return f"{key}_{salt}" if salt else key


def _py_str(s):
    """Render a Python string literal for a decoded call string."""
    return repr(str(s))


class QwenScaffoldHandler(QwenGovHandler):
    def __init__(self, model_name, temperature, registry_name, is_fc_model,
                 dtype="bfloat16", **kwargs) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model,
                         dtype=dtype, **kwargs)
        self.scaf_enabled = _flag("SCAF_ENABLED")
        self.scaf_max_entries = int(os.getenv("SCAF_MAX_ENTRIES",
                                              str(MAX_ENTRIES_DEFAULT)))
        self.scaf_max_len = int(os.getenv("SCAF_MAX_LEN", str(MAX_LEN_DEFAULT)))
        self.scaf_gate_recall = _flag("SCAF_GATE_RECALL")
        self.scaf_mode = os.getenv("SCAF_MODE", "user_turn").strip().lower()
        if self.scaf_mode not in MODES:
            raise ValueError(f"SCAF_MODE must be one of {MODES}")
        self.scaf_log_file = os.getenv("SCAF_LOG_FILE", "scaffold_log.jsonl")
        self.scaf_verbose = _flag("SCAF_VERBOSE")
        self._scaf_tls = threading.local()
        print(f"[SCAF] QwenScaffoldHandler active | enabled={self.scaf_enabled} "
              f"mode={self.scaf_mode} max_entries={self.scaf_max_entries} "
              f"max_len={self.scaf_max_len} gate_recall={self.scaf_gate_recall}")

    # ---------------------------------------------------------- turn state

    def _state(self):
        st = getattr(self._scaf_tls, "state", None)
        if st is None:
            st = {"turn": 0, "scaffolded_turn": -1, "n_written": 0,
                  "test_id": "", "backend": "", "used_keys": set(),
                  "user_text": ""}
            self._scaf_tls.state = st
        return st

    def _record_user_text(self, messages):
        """Own copy of the current user turn text.

        Deliberately independent of the governance session: the scaffold arm
        runs with GOV_ENABLED=0 (nothing should suppress), and in that
        configuration QwenGovHandler builds no session at all.
        """
        texts = [str(m.get("content", "")) for m in messages
                 if isinstance(m, dict) and m.get("role") == "user"]
        if texts:
            self._state()["user_text"] = texts[-1]

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        inference_data = super()._pre_query_processing_prompting(test_entry)
        test_id = test_entry.get("id", "")
        st = self._state()
        st.update({"turn": 0, "scaffolded_turn": -1, "test_id": test_id,
                   "backend": self._scaf_backend(test_id)})
        # entry counter and key set persist across the prereq chain within a
        # scenario: reset only when the chain restarts (first prereq entry).
        if test_id.endswith("-0") and "prereq_0" in test_id:
            st["n_written"] = 0
            st["used_keys"] = set()
        return inference_data

    def _scaf_backend(self, test_id):
        try:
            return extract_memory_backend_type(
                extract_test_category_from_id(test_id, remove_prereq=True))
        except Exception:
            return ""

    def _bump_turn(self):
        st = self._state()
        st["turn"] += 1

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

    def _scaf_active(self, test_id):
        if not self.scaf_enabled or not test_id or not is_memory(test_id):
            return False
        if self._scaf_backend(test_id) not in GOVERNED_BACKENDS:
            return False
        if not self.scaf_gate_recall and not is_memory_prereq(test_id):
            return False
        return True

    def _scaffold_text(self, user_text, scenario_salt):
        body = str(user_text)[: self.scaf_max_len]
        if self.scaf_mode == "shuffled_control":
            # Negative control: same volume, deterministically scrambled word
            # order seeded by the turn -- carries no coherent fact.
            words = body.split()
            h = int(hashlib.sha256(
                f"{scenario_salt}|{len(words)}".encode()).hexdigest()[:8], 16)
            idx = sorted(range(len(words)), key=lambda i: (h * (i + 7)) % 9973)
            body = " ".join(words[i] for i in idx)
        return body

    def _build_call(self, backend, text):
        if backend == "kv":
            st = self._state()
            key = slug_key(text)
            n = 0
            while key in st["used_keys"]:
                n += 1
                key = slug_key(text, salt=n)
            st["used_keys"].add(key)
            return f"archival_memory_add(key={_py_str(key)}, value={_py_str(text)})"
        return f"archival_memory_add(text={_py_str(text)})"

    def _log(self, rec):
        log_dir = getattr(self.gov_config, "log_dir", "") or "."
        path = Path(log_dir) / self.scaf_log_file
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @override
    def decode_execute(self, result, has_tool_call_tag):
        calls = super().decode_execute(result, has_tool_call_tag)
        st = self._state()
        test_id = st.get("test_id", "")
        if calls or not self._scaf_active(test_id):
            return calls
        # The model produced no calls: this turn is about to end with nothing
        # written. Scaffold it once.
        if st["scaffolded_turn"] == st["turn"]:
            return calls          # already scaffolded this turn
        if st["n_written"] >= self.scaf_max_entries:
            return calls          # respect the backend's archival capacity
        user_text = st.get("user_text") or ""
        if not user_text.strip():
            return calls
        backend = st["backend"]
        text = self._scaffold_text(user_text, f"{test_id}|{st['turn']}")
        if not text.strip():
            return calls
        call = self._build_call(backend, text)
        st["scaffolded_turn"] = st["turn"]
        st["n_written"] += 1
        self._log({
            "ts": time.time(), "event": "scaffold_write", "test_id": test_id,
            "turn": st["turn"], "backend": backend, "mode": self.scaf_mode,
            "n_written": st["n_written"], "text_len": len(text), "call": call,
        })
        if self.scaf_verbose:
            print(f"[SCAF] {test_id} turn={st['turn']} -> {call[:90]}")
        return [call]
