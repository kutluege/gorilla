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

Write side (RAG program, method M2 -- entry-budget-aware packed capture):
the archival index budget is ENTRY COUNT (50), not entry length (2000
chars). One-turn-per-entry capture (alt5R) overflows the cap on long
scenarios and wastes slots on short turns; sentence chunking is
catastrophic (offline: carried 0.884 packed vs 0.342 sentences). RAG_WRITE
modes buffer no-write prereq turn texts and emit one `archival_memory_add`
per ~2000-char packed blob. Fully causal: the buffer contains only past
turns; the chain-final tail (< one blob) is lost, which is pre-registered
as the live-vs-offline carried-rate gap.

Env vars:
    RAG_ENABLED    master switch (default 0)
    RAG_MODE       none | read_verbatim (M1) | read_irrelevant (control:
                   identical call + step cost, frozen generic probe carrying
                   no question-specific information) | instruction_only
                   (control: system-prompt nudge, no injection -- separates
                   deterministic mechanism from prompting)
    RAG_WRITE      none (default) | pack (M2) | pack_shuffled (control:
                   identical blobs, deterministically scrambled word order)
                   | per_turn (alt5R granularity contrast)
    RAG_TOP_K      k for the injected read (default 5, the backend default;
                   frozen for C1, varied only by the pre-registered C3 arms)
    RAG_PACK_LEN   packed-entry char budget (default 2000, the backend's
                   MAX_ARCHIVAL_ENTRY_LENGTH)
    RAG_MAX_ENTRIES  per-scenario write cap (default 50, the backend's
                   MAX_ARCHIVAL_SIZE)
    RAG_LOG_FILE   audit log name inside GOV_LOG_DIR (default rag_log.jsonl)
    RAG_VERBOSE    print per-injection lines (default 0)
"""

import hashlib
import json
import os
import threading
import time
from pathlib import Path

from bfcl_eval.model_handler.local_inference.qwen_gov import (
    GOVERNED_BACKENDS,
    QwenGovHandler,
)
from bfcl_eval.model_handler.local_inference.qwen_scaffold import slug_key
from bfcl_eval.utils import (
    extract_memory_backend_type,
    extract_test_category_from_id,
    is_first_memory_prereq_entry,
    is_memory,
    is_memory_prereq,
)
from overrides import override

MODES = ("none", "read_verbatim", "read_irrelevant", "instruction_only")
WRITE_MODES = ("none", "pack", "pack_shuffled", "per_turn", "abstractive")
EVICT_MODES = ("none", "redundancy", "random")   # M7, vector archival

# M6 abstractive write: frozen summarization instruction. Answer-agnostic;
# the entity-preservation clause is the mainstream consolidation design
# under test (predicted to LOSE to packed verbatim: gold answers are
# median-7-char entities under an exact substring grader).
SUMMARIZE_PROMPT = (
    "Condense the following user messages into a compact summary. Keep "
    "every concrete fact exactly as stated (names, numbers, dates, places, "
    "preferences). Output only the summary.\n\nMessages:\n{body}\n\nSummary:"
)
ANSWER_MODES = ("none", "evidence", "evidence_irrelevant")
NUDGE_MODES = ("none", "nudge", "nudge_null")
RERANK_MODES = ("none", "bm25", "random")   # M9 (vector); kv passes through
PROBE_MODES = ("verbatim", "core_augmented")  # M4

# M3 pad control: frozen neutral filler used to match k=FETCH_K context
# volume while carrying only the top-TOP_K real entries' information.
PAD_FILLER_SENTENCE = (
    "This entry intentionally contains no conversation information and is "
    "padding for a controlled comparison of retrieval context volume. "
)

# M5 quote-grounded answer formulation (frozen constants, answer-agnostic,
# identical across entries and backends). The strict answer-field regrade is
# the pre-registered guard against grader-artifact gains.
EVIDENCE_INSTRUCTION = (
    "When you answer, respond in the format {'answer': ..., 'evidence': ..., "
    "'context': ...} where 'evidence' is the exact verbatim snippet from your "
    "memory that your answer came from."
)
EVIDENCE_IRRELEVANT_INSTRUCTION = (
    "When you answer, respond in the format {'answer': ..., 'evidence': ..., "
    "'context': ...} where 'evidence' is an exact verbatim snippet from your "
    "memory that is NOT related to the question."
)

# M10 evidence-conditioned nudge, appended to the INJECTED read's tool-result
# payload (precedent: mig_reranker rewrites result payloads agent-side).
# Treatment and control are length-matched; only the instruction differs.
NUDGE_TEXT = (
    "\n[Note: if any entry above contains the answer to the user's question, "
    "state that answer explicitly; only say you do not know if none does.]"
)
NUDGE_NULL_TEXT = (
    "\n[Note: the entries above were retrieved from the archival memory "
    "store as part of the current conversation session's record keeping.]"
)

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
        self.rag_write = os.getenv("RAG_WRITE", "none").strip().lower()
        if self.rag_write not in WRITE_MODES:
            raise ValueError(f"RAG_WRITE must be one of {WRITE_MODES}")
        self.rag_answer = os.getenv("RAG_ANSWER", "none").strip().lower()
        if self.rag_answer not in ANSWER_MODES:
            raise ValueError(f"RAG_ANSWER must be one of {ANSWER_MODES}")
        self.rag_nudge = os.getenv("RAG_NUDGE", "none").strip().lower()
        if self.rag_nudge not in NUDGE_MODES:
            raise ValueError(f"RAG_NUDGE must be one of {NUDGE_MODES}")
        self.rag_rerank = os.getenv("RAG_RERANK", "none").strip().lower()
        if self.rag_rerank not in RERANK_MODES:
            raise ValueError(f"RAG_RERANK must be one of {RERANK_MODES}")
        self.rag_probe = os.getenv("RAG_PROBE", "verbatim").strip().lower()
        if self.rag_probe not in PROBE_MODES:
            raise ValueError(f"RAG_PROBE must be one of {PROBE_MODES}")
        # FETCH_K: k used by the injected call; 0 = same as TOP_K. Rerank
        # and pad modes fetch FETCH_K then reduce to TOP_K agent-side.
        self.rag_fetch_k = int(os.getenv("RAG_FETCH_K", "0")) or None
        self.rag_pad = _flag("RAG_PAD")
        self.rag_key_tokens = int(os.getenv("RAG_KEY_TOKENS", "6"))
        self.rag_kv_two_stage = _flag("RAG_KV_TWO_STAGE")
        if (self.rag_rerank != "none" or self.rag_pad) and not self.rag_fetch_k:
            raise ValueError("RAG_RERANK/RAG_PAD require RAG_FETCH_K > TOP_K")
        self.rag_evict = os.getenv("RAG_EVICT", "none").strip().lower()
        if self.rag_evict not in EVICT_MODES:
            raise ValueError(f"RAG_EVICT must be one of {EVICT_MODES}")
        # M6 char-matched verbatim control: truncate each flushed blob to
        # this ratio of its length (frozen from the abstractive arm's
        # realized mean compression before the control arm runs; 0 = off).
        self.rag_pack_trunc_ratio = float(os.getenv("RAG_PACK_TRUNC_RATIO",
                                                    "0"))
        self.rag_summary_max_tokens = int(os.getenv("RAG_SUMMARY_MAX_TOKENS",
                                                    "512"))
        self.rag_top_k = int(os.getenv("RAG_TOP_K", "5"))
        self.rag_pack_len = int(os.getenv("RAG_PACK_LEN", "2000"))
        self.rag_max_entries = int(os.getenv("RAG_MAX_ENTRIES", "50"))
        self.rag_log_file = os.getenv("RAG_LOG_FILE", "rag_log.jsonl")
        self.rag_verbose = _flag("RAG_VERBOSE")
        self._rag_tls = threading.local()
        print(f"[RAG] QwenRagHandler active | enabled={self.rag_enabled} "
              f"mode={self.rag_mode} write={self.rag_write} "
              f"top_k={self.rag_top_k} pack_len={self.rag_pack_len}")

    # ---------------------------------------------------------- turn state

    def _state(self):
        st = getattr(self._rag_tls, "state", None)
        if st is None:
            st = {"turn": 0, "read_turn": -1, "write_turn": -1,
                  "retrieve_turn": -1, "evict_turn": -1, "test_id": "",
                  "backend": "", "user_text": "", "core_words": "",
                  "injected_probe": "", "pending_retrieves": None,
                  "evict_pending": None, "buffer": "", "n_written": 0,
                  "used_keys": set(), "written": []}
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
        st.update({"turn": 0, "read_turn": -1, "write_turn": -1,
                   "retrieve_turn": -1, "evict_turn": -1, "test_id": test_id,
                   "backend": self._rag_backend(test_id), "user_text": "",
                   "core_words": "", "injected_probe": "",
                   "pending_retrieves": None})
        st.pop("read_result_pending", None)
        st.pop("nudge_pending", None)
        st.pop("write_result_pending", None)
        # pack buffer, entry counter, key set and incumbent tracker persist
        # across the prereq chain within a scenario; reset only when the
        # chain restarts.
        if is_first_memory_prereq_entry(test_id):
            st["buffer"] = ""
            st["n_written"] = 0
            st["used_keys"] = set()
            st["written"] = []
            st["evict_pending"] = None
        if self.rag_probe == "core_augmented":
            # capture the core-memory dump the harness already injected into
            # the system message (runtime-visible information only)
            try:
                sys_msg = str(test_entry["question"][0][0].get("content", ""))
                marker = "Core Memory from previous interactions:"
                if marker in sys_msg:
                    import re as _re
                    dump = sys_msg.split(marker, 1)[1]
                    words = _re.findall(r"[A-Za-z0-9']+", dump)
                    st["core_words"] = " ".join(words[:120])
            except (KeyError, IndexError, TypeError):
                pass
        appends = []
        if self.rag_mode == "instruction_only":
            appends.append(READ_INSTRUCTION)
        if self.rag_answer == "evidence":
            appends.append(EVIDENCE_INSTRUCTION)
        elif self.rag_answer == "evidence_irrelevant":
            appends.append(EVIDENCE_IRRELEVANT_INSTRUCTION)
        if appends and self._rag_entry_gated(test_id):
            # Append the frozen instruction(s) to the system message that
            # system_prompt_pre_processing_chat_model just built. Constant,
            # answer-agnostic, identical across entries and backends.
            try:
                first = test_entry["question"][0][0]
                if first.get("role") == "system":
                    for text in appends:
                        if text not in str(first.get("content")):
                            first["content"] = f"{first['content']}\n\n{text}"
                    self._log({"ts": time.time(), "event": "instruction_added",
                               "test_id": test_id, "backend": st["backend"],
                               "n_appends": len(appends)})
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
        if self.rag_mode in ("none", "instruction_only"):
            return False          # those modes never inject reads
        return self._rag_entry_gated(test_id)

    def _build_read_call(self, backend, probe):
        k = self.rag_fetch_k or self.rag_top_k
        if backend == "kv":
            return (f"archival_memory_key_search(query={_py_str(probe)}, "
                    f"k={k})")
        return (f"archival_memory_retrieve(query={_py_str(probe)}, "
                f"top_k={k})")

    def _make_probe(self, st):
        """M4 probe construction. verbatim = the current user turn;
        core_augmented = user turn + content words of the in-context core
        dump (free: the dump is already in the system prompt)."""
        base = (st.get("user_text") or "").strip()
        if self.rag_probe == "core_augmented" and st.get("core_words"):
            return f"{base} {st['core_words']}".strip()
        return base

    # ------------------------------------------------------- write side (M2)

    def _rag_write_active(self, test_id):
        if not self.rag_enabled or self.rag_write == "none":
            return False
        if not test_id or not is_memory(test_id):
            return False
        if not is_memory_prereq(test_id):
            return False
        return self._rag_backend(test_id) in GOVERNED_BACKENDS

    def _shuffle_words(self, text, salt):
        """Deterministic word scramble (pack_shuffled control): identical
        volume and word multiset, no coherent fact."""
        words = text.split()
        h = int(hashlib.sha256(f"{salt}|{len(words)}".encode()
                               ).hexdigest()[:8], 16)
        idx = sorted(range(len(words)), key=lambda i: (h * (i + 7)) % 9973)
        return " ".join(words[i] for i in idx)

    def _build_write_call(self, backend, text):
        st = self._state()
        if backend == "kv":
            key = slug_key(text, n_tokens=self.rag_key_tokens)
            n = 0
            while key in st["used_keys"]:
                n += 1
                key = slug_key(text, n_tokens=self.rag_key_tokens, salt=n)
            st["used_keys"].add(key)
            return f"archival_memory_add(key={_py_str(key)}, value={_py_str(text)})"
        return f"archival_memory_add(text={_py_str(text)})"

    def _summarize(self, body, test_id):
        """M6: one synchronous model call condensing the buffered turns.
        Cost (tokens) is logged; the call goes to the same served model at
        the harness temperature."""
        prompt = SUMMARIZE_PROMPT.format(body=body)
        try:
            resp = self.client.completions.create(
                model=self.model_path_or_id,
                temperature=self.temperature,
                prompt=prompt,
                max_tokens=self.rag_summary_max_tokens,
                timeout=600,
            )
            text = (resp.choices[0].text or "").strip()
            usage = getattr(resp, "usage", None)
            self._log({"ts": time.time(), "event": "summarize",
                       "test_id": test_id, "in_len": len(body),
                       "out_len": len(text),
                       "prompt_tokens": getattr(usage, "prompt_tokens", None),
                       "completion_tokens": getattr(usage,
                                                    "completion_tokens", None)})
            return text[: self.rag_pack_len] or body[: self.rag_pack_len]
        except Exception as exc:
            # Summarization failure must not kill the chain: fall back to
            # the verbatim blob and log the degradation.
            self._log({"ts": time.time(), "event": "summarize_error",
                       "test_id": test_id, "error": str(exc)[:200]})
            return body[: self.rag_pack_len]

    def _write_policy(self, st, test_id):
        """Called on an empty decode of a gated prereq turn. Returns the
        packed/per-turn write call to inject, or None."""
        if st["write_turn"] == st["turn"]:
            return None           # once per turn
        if st["n_written"] >= self.rag_max_entries:
            return None           # respect the backend's archival capacity
        turn_text = (st.get("user_text") or "").strip()
        if not turn_text:
            return None
        st["write_turn"] = st["turn"]
        emit = None
        if self.rag_write == "per_turn":
            emit = turn_text[: self.rag_pack_len]
        else:                     # pack / pack_shuffled / abstractive
            buf = st["buffer"]
            if buf and len(buf) + 1 + len(turn_text) > self.rag_pack_len:
                emit = buf
                st["buffer"] = turn_text[: self.rag_pack_len]
            else:
                st["buffer"] = (f"{buf} {turn_text}".strip()
                                if buf else turn_text)[: self.rag_pack_len]
        if emit is None:
            return None
        if self.rag_write == "pack_shuffled":
            emit = self._shuffle_words(emit, f"{test_id}|{st['n_written']}")
        elif self.rag_write == "abstractive":
            emit = self._summarize(emit, test_id)
        if self.rag_pack_trunc_ratio:
            # M6 char-matched verbatim control
            emit = emit[: max(1, int(len(emit) * self.rag_pack_trunc_ratio))]
        call = self._build_write_call(st["backend"], emit)
        st["n_written"] += 1
        st["write_result_pending"] = {"text": emit, "add_index": 0}
        self._log({
            "ts": time.time(), "event": "pack_write", "test_id": test_id,
            "turn": st["turn"], "backend": st["backend"],
            "write_mode": self.rag_write, "n_written": st["n_written"],
            "text_len": len(emit), "buffer_len": len(st["buffer"]),
            "call": call,
        })
        if self.rag_verbose:
            print(f"[RAG:write] {test_id} turn={st['turn']} -> {call[:90]}")
        return call

    def _pick_victim(self, st):
        """M7: choose the incumbent to evict. redundancy = highest Jaccard
        token overlap with any other incumbent (the offline falsifier's
        drop_longest_dupe); random = seeded control at matched count."""
        import re as _re
        written = st.get("written") or []
        if not written:
            return None
        if self.rag_evict == "random":
            import random as _random
            rng = _random.Random(f"{st.get('test_id', '')}|evict")
            return rng.randrange(len(written))
        toks = [set(_re.findall(r"[a-z0-9]+", w["text"].lower()))
                for w in written]
        worst, worst_i = -1.0, 0
        for i, ti in enumerate(toks):
            best = max((len(ti & tj) / max(1, len(ti | tj))
                        for j, tj in enumerate(toks) if j != i), default=0.0)
            if best > worst:
                worst, worst_i = best, i
        return worst_i

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
        if calls:
            return calls
        # write side: packed capture on no-call prereq turns
        if self._rag_write_active(test_id):
            # M7: a cap-rejected blob retries as evict+add (vector only)
            if (st.get("evict_pending") is not None
                    and self.rag_evict != "none"
                    and st.get("backend") == "vector"
                    and st.get("evict_turn") != st["turn"]):
                victim_i = self._pick_victim(st)
                text = st.pop("evict_pending")
                st["evict_turn"] = st["turn"]
                if victim_i is not None:
                    v = st["written"].pop(victim_i)
                    remove = (f"archival_memory_remove"
                              f"(vec_id={int(v['ref'])})")
                    add = self._build_write_call("vector", text)
                    st["write_result_pending"] = {"text": text,
                                                  "add_index": 1}
                    self._log({"ts": time.time(), "event": "evict_write",
                               "test_id": test_id, "turn": st["turn"],
                               "evict_mode": self.rag_evict,
                               "victim_ref": v["ref"],
                               "victim_len": len(v["text"]),
                               "text_len": len(text)})
                    return [remove, add]
            call = self._write_policy(st, test_id)
            return [call] if call else calls
        # read side: retrieve-then-generate on no-call question turns
        if not self._rag_active(test_id):
            return calls
        # M8 two-stage kv read: key_search returned keys with no values;
        # inject the batch of value retrieves before letting the turn end.
        pending = st.get("pending_retrieves")
        if pending and st.get("retrieve_turn") != st["turn"] \
                and st["backend"] == "kv":
            st["pending_retrieves"] = None
            st["retrieve_turn"] = st["turn"]
            batch = [f"archival_memory_retrieve(key={_py_str(k)})"
                     for k in pending[: self.rag_top_k]]
            self._log({"ts": time.time(), "event": "two_stage_retrieve",
                       "test_id": test_id, "turn": st["turn"],
                       "n_keys": len(batch)})
            return batch
        if st["read_turn"] == st["turn"]:
            return calls          # already injected this turn: the model's
            #                       post-retrieval no-call answer ends the turn
        probe = (IRRELEVANT_PROBE if self.rag_mode == "read_irrelevant"
                 else self._make_probe(st))
        if not probe.strip():
            return calls
        call = self._build_read_call(st["backend"], probe)
        st["read_turn"] = st["turn"]
        st["injected_probe"] = probe
        st["read_result_pending"] = True
        if self.rag_nudge != "none":
            st["nudge_pending"] = True
        self._log({
            "ts": time.time(), "event": "read_inject", "test_id": test_id,
            "turn": st["turn"], "backend": st["backend"], "mode": self.rag_mode,
            "top_k": self.rag_top_k, "probe_len": len(probe), "call": call,
        })
        if self.rag_verbose:
            print(f"[RAG] {test_id} turn={st['turn']} -> {call[:90]}")
        return [call]

    # ---------------------------------------- result-payload transforms

    def _rerank_vector_payload(self, payload, probe):
        """M9: fetch_k entries -> agent-side rerank -> keep top_k. Returns
        the transformed payload string, or None if not applicable."""
        try:
            data = json.loads(payload)
            entries = data.get("result")
            if not isinstance(entries, list) or len(entries) <= self.rag_top_k:
                return None
        except (json.JSONDecodeError, AttributeError):
            return None
        if self.rag_rerank == "bm25":
            from rank_bm25 import BM25Plus
            toks = [str(e.get("text", "")).lower().split() for e in entries]
            bm = BM25Plus(toks)
            scores = bm.get_scores(str(probe).lower().split())
            order = sorted(range(len(entries)), key=lambda i: -scores[i])
        else:                                    # random control
            import random as _random
            order = list(range(len(entries)))
            _random.Random(f"{self._state().get('test_id', '')}").shuffle(order)
        data["result"] = [entries[i] for i in order[: self.rag_top_k]]
        return json.dumps(data)

    def _pad_vector_payload(self, payload):
        """M3 dilution control: keep top_k real entries; replace the rest of
        the fetched entries' texts with frozen neutral filler of matched
        length, so context volume equals fetch_k while information equals
        top_k."""
        try:
            data = json.loads(payload)
            entries = data.get("result")
            if not isinstance(entries, list) or len(entries) <= self.rag_top_k:
                return None
        except (json.JSONDecodeError, AttributeError):
            return None
        for e in entries[self.rag_top_k:]:
            want = max(len(str(e.get("text", ""))), 1)
            filler = (PAD_FILLER_SENTENCE
                      * (want // len(PAD_FILLER_SENTENCE) + 1))[:want]
            e["text"] = filler
        return json.dumps(data)

    @override
    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str],
        model_response_data: dict
    ) -> dict:
        st = self._state()
        wrp = st.pop("write_result_pending", None)
        if wrp and execution_results:
            idx = wrp.get("add_index", 0)
            payload = (execution_results[idx]
                       if idx < len(execution_results) else "")
            if "exceeds maximum size" in payload:
                # backend cap hit: the write did not land
                st["n_written"] = max(0, st["n_written"] - 1)
                if self.rag_evict != "none" and st.get("backend") == "vector":
                    st["evict_pending"] = wrp["text"]
                self._log({"ts": time.time(), "event": "write_capped",
                           "test_id": st.get("test_id", ""),
                           "turn": st["turn"],
                           "will_evict": self.rag_evict != "none"})
            elif st.get("backend") == "vector":
                try:
                    ref = json.loads(payload).get("id")
                    if ref is not None:
                        st.setdefault("written", []).append(
                            {"ref": ref, "text": wrp["text"]})
                except (json.JSONDecodeError, AttributeError, ValueError):
                    pass
        if st.pop("read_result_pending", False) and execution_results:
            execution_results = list(execution_results)
            payload = execution_results[0]
            probe = st.get("injected_probe", "")
            # M8: stash keys from the kv key_search result for the two-stage
            # value retrieves on the next step.
            if self.rag_kv_two_stage and st.get("backend") == "kv":
                try:
                    ranked = json.loads(payload).get("ranked_results") or []
                    st["pending_retrieves"] = [k for _, k in ranked]
                except (json.JSONDecodeError, AttributeError, ValueError):
                    pass
            # M9 / M3: transform the vector payload before it enters context.
            new_payload = None
            if st.get("backend") == "vector":
                if self.rag_rerank != "none":
                    new_payload = self._rerank_vector_payload(payload, probe)
                elif self.rag_pad:
                    new_payload = self._pad_vector_payload(payload)
            if new_payload is not None:
                execution_results[0] = new_payload
                self._log({"ts": time.time(), "event": "payload_transform",
                           "test_id": st.get("test_id", ""),
                           "turn": st["turn"], "rerank": self.rag_rerank,
                           "pad": self.rag_pad})
            if st.pop("nudge_pending", False):
                # M10: append the frozen nudge (or its length-matched null
                # control) to the injected read's payload. Precedent:
                # mig_reranker rewrites result payloads the same way.
                text = (NUDGE_TEXT if self.rag_nudge == "nudge"
                        else NUDGE_NULL_TEXT)
                execution_results[0] = f"{execution_results[0]}{text}"
                self._log({"ts": time.time(), "event": "result_nudge",
                           "test_id": st.get("test_id", ""),
                           "turn": st["turn"], "nudge_mode": self.rag_nudge})
        return super()._add_execution_results_prompting(
            inference_data, execution_results, model_response_data)
