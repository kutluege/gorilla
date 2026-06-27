"""
Memory Information Gain (MIG) Reranker
======================================

A backend-agnostic reranker that sits at the *tool-result boundary* of a BFCL memory
agent. When the agent emits a memory-retrieval tool call (e.g. ``archival_memory_retrieve``),
BFCL executes it and injects the JSON result back into the chat history. This module
intercepts that JSON result, scores each retrieved candidate by how much it helps answer
the user's question, and returns a trimmed/reordered candidate set so the agent sees a
cleaner, more relevant context.

Design constraints (see ``BFCL_MIG_RERANKER_GUIDE.md``):
  * Touches **zero** benchmark data, **zero** memory-backend code, **zero** scoring code.
  * Operates only on the JSON tool-result string + the decoded tool-call string + the
    user question (all available to the handler).
  * Never returns an empty tool result (that would strand the agent).
  * Pure utility object: it is given an OpenAI-compatible client + a model id and does the
    rest. It holds no benchmark state; per-test state is passed in by the handler.

Backend asymmetry handled here (detected from the *result shape*, not the function name,
because the same function name means different things on different backends):
  * ``memory_vector``  -> ``{"result": [{"id", "similarity_score", "text"}, ...]}``  (rerankable)
  * ``memory_kv``      -> ``{"ranked_results": [[score, key], ...]}`` from ``*_key_search`` (weak signal, supported)
  * ``memory_kv``      -> ``{"value": ...}`` from exact-key ``*_retrieve`` (no candidate list -> pass-through)
  * ``memory_rec_sum`` -> ``{"memory_content": ...}`` whole blob (no candidate list -> pass-through)

Scorers (config ``scorer``):
  * ``similarity`` : identity trim, keep the backend's own ranking (used for the
                     "widened-pool, no rerank" ablation arm).
  * ``judge``      : an LLM utility judge (0-5) over each candidate. v1 default, backend-agnostic.
  * ``logprob``    : exact information-gain via prompt-echo logprobs:
                     ``MIG(m) = logp(a_hat | S+{m}) - logp(a_hat | S)``. v2, local-vLLM only.

Modes (config ``mode``):
  * ``flat``   : score every candidate once against the current draft, keep the top ``final_budget``.
  * ``greedy`` : greedily add one candidate at a time, re-drafting/re-scoring after each pick,
                 with an adaptive halt when the best marginal gain drops below ``halt_tau``.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Function name -> name of the parameter that controls the size of the returned pool.
# These are the only retrieval functions that accept a pool-size argument; they are the
# targets for "pool expansion" in the handler's ``decode_execute`` override.
RETRIEVE_POOL_PARAM = {
    "archival_memory_retrieve": "top_k",  # vector backend (kv ``*_retrieve`` takes ``key`` -> ignored)
    "core_memory_retrieve": "top_k",      # vector backend
    "archival_memory_key_search": "k",    # kv backend
    "core_memory_key_search": "k",        # kv backend
}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class MIGConfig:
    """Configuration for the MIG reranker. Populated from ``MIG_*`` environment variables so
    that different ablation arms can be run against a single registry entry without code edits."""

    enabled: bool = True
    # Widen the retrieval pool: rewrite the agent's ``top_k``/``k`` up to this value.
    pool_size: int = 20
    # Number of candidates to keep after reranking.
    final_budget: int = 5
    # similarity | judge | logprob
    scorer: str = "judge"
    # flat | greedy
    mode: str = "flat"
    # If False, only pool widening happens (the "widened-pool, no rerank" arm).
    rerank_enabled: bool = True
    # Greedy mode: stop adding once the best marginal gain is below this threshold.
    halt_tau: float = 0.0
    # Flat mode: drop candidates whose score is below this threshold (kept >= 1 regardless).
    min_gain: float = float("-inf")
    # Condition scoring on a model-generated draft answer (the MIG "a_hat").
    use_draft: bool = True
    # Draft the answer conditioned on the retrieved candidate pool (not empty context).
    # This is critical: an empty-context draft is the model's *prior* guess, which is wrong
    # exactly when memory is needed -- and a wrong draft inverts logprob-MIG (the gold memory
    # contradicts the wrong guess and scores lowest). Conditioning the draft on the pool makes
    # a_hat the model's best answer given the retrieved evidence, which is what MIG should
    # measure support for. See BFCL_MIG_RERANKER_GUIDE.md ("Experiment findings").
    draft_from_pool: bool = True
    max_draft_tokens: int = 96
    draft_temperature: float = 0.0
    judge_temperature: float = 0.0
    # Directory for structured per-test JSONL traces. None disables file tracing.
    log_dir: Optional[str] = None
    # Print a one-line summary per interception to stdout (shows up in BFCL logs).
    verbose: bool = True

    @classmethod
    def from_env(cls) -> "MIGConfig":
        return cls(
            enabled=_env_bool("MIG_ENABLED", True),
            pool_size=_env_int("MIG_POOL_SIZE", 20),
            final_budget=_env_int("MIG_FINAL_BUDGET", 5),
            scorer=os.getenv("MIG_SCORER", "judge").strip().lower(),
            mode=os.getenv("MIG_MODE", "flat").strip().lower(),
            rerank_enabled=_env_bool("MIG_RERANK", True),
            halt_tau=_env_float("MIG_HALT_TAU", 0.0),
            min_gain=_env_float("MIG_MIN_GAIN", float("-inf")),
            use_draft=_env_bool("MIG_DRAFT", True),
            draft_from_pool=_env_bool("MIG_DRAFT_FROM_POOL", True),
            max_draft_tokens=_env_int("MIG_MAX_DRAFT_TOKENS", 96),
            draft_temperature=_env_float("MIG_DRAFT_TEMPERATURE", 0.0),
            judge_temperature=_env_float("MIG_JUDGE_TEMPERATURE", 0.0),
            log_dir=os.getenv("MIG_LOG_DIR") or None,
            verbose=_env_bool("MIG_VERBOSE", True),
        )


class Candidate:
    """A single rerankable retrieved entry, normalized across backends."""

    __slots__ = ("text", "prior_score", "raw", "kind")

    def __init__(self, text: str, prior_score: float, raw: Any, kind: str):
        self.text = text
        self.prior_score = prior_score  # backend-provided similarity / BM25 score
        self.raw = raw  # original object, used verbatim on reserialization
        self.kind = kind


class MIGReranker:
    """Backend-agnostic reranker. Stateless across test entries; the handler passes per-test
    state in explicitly so the object is safe to share across threads."""

    def __init__(
        self,
        client,
        model_id: str,
        config: MIGConfig,
        tokenizer=None,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.config = config
        self.tokenizer = tokenizer
        # Whether the echo-logprob call has been confirmed to work on this endpoint.
        # None = untested, True/False = result of the first attempt.
        self._logprob_supported: Optional[bool] = None

    # ------------------------------------------------------------------ #
    # Pool widening (called from the handler's decode_execute override)   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def widen_pool(tool_call: dict, pool_size: int) -> dict:
        """Rewrite a decoded tool-call dict in place so a retrieval call returns a larger pool.

        ``tool_call`` is ``{"name": <fn>, "arguments": {<args>}}``. Only retrieval functions
        that accept a pool-size argument are touched; everything else is left untouched. We
        only ever *widen* (never shrink) the pool, so an agent asking for more than
        ``pool_size`` is respected.
        """
        name = tool_call.get("name")
        param = RETRIEVE_POOL_PARAM.get(name)
        if param is None:
            return tool_call
        args = tool_call.get("arguments")
        if not isinstance(args, dict):
            return tool_call
        # The kv exact-key ``*_retrieve`` takes ``key`` (no ``query``); skip it. Only the
        # vector ``*_retrieve`` (query + top_k) and kv ``*_key_search`` (query + k) qualify.
        if "query" not in args:
            return tool_call
        current = args.get(param)
        try:
            current = int(current) if current is not None else 0
        except (TypeError, ValueError):
            current = 0
        args[param] = max(current, pool_size)
        return tool_call

    # ------------------------------------------------------------------ #
    # Candidate parsing                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def parse_candidates(raw_result: str) -> tuple[Optional[str], list[Candidate]]:
        """Parse a JSON tool-result string into (kind, candidates).

        Returns ``(None, [])`` for results that carry no rerankable candidate list
        (exact-key retrieve, whole-blob retrieve, errors, etc.) -> the handler passes those through.
        """
        if not isinstance(raw_result, str):
            return None, []
        try:
            data = json.loads(raw_result)
        except (json.JSONDecodeError, ValueError):
            return None, []
        if not isinstance(data, dict):
            return None, []

        # Vector backend: {"result": [{"id", "similarity_score", "text"}, ...]}
        # (Tool doc says "results" but the live backend returns the singular "result";
        #  accept both for robustness.)
        for key in ("result", "results"):
            if key in data and isinstance(data[key], list):
                cands = []
                for item in data[key]:
                    if isinstance(item, dict) and "text" in item:
                        cands.append(
                            Candidate(
                                text=str(item.get("text", "")),
                                prior_score=float(item.get("similarity_score", 0.0) or 0.0),
                                raw=item,
                                kind="vector",
                            )
                        )
                if cands:
                    return "vector", cands

        # KV backend key search: {"ranked_results": [[score, key], ...]}
        if "ranked_results" in data and isinstance(data["ranked_results"], list):
            cands = []
            for item in data["ranked_results"]:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    score, key = item
                    cands.append(
                        Candidate(
                            text=str(key),
                            prior_score=float(score or 0.0),
                            raw=[score, key],
                            kind="kv_keys",
                        )
                    )
            if cands:
                return "kv_keys", cands

        return None, []

    @staticmethod
    def reserialize(kind: str, kept: list[Candidate]) -> str:
        """Re-emit the kept candidates in the exact schema the agent expects, matching the
        JSON serialization produced by ``execute_multi_turn_func_call`` (``json.dumps``)."""
        if kind == "vector":
            return json.dumps({"result": [c.raw for c in kept]})
        if kind == "kv_keys":
            return json.dumps({"ranked_results": [c.raw for c in kept]})
        # Should not happen (callers only reserialize known kinds), but fail safe.
        return json.dumps({"result": [c.raw for c in kept]})

    # ------------------------------------------------------------------ #
    # Question extraction                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def extract_question(messages: list[dict]) -> str:
        """Pull the user question out of the running chat history. Memory tasks are
        single-question, so we take the last genuine user message (ignoring tool-response
        echoes that some templates fold into the user role)."""
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            stripped = content.strip()
            if stripped.startswith("<tool_response>") and stripped.endswith("</tool_response>"):
                continue
            return content
        return ""

    # ------------------------------------------------------------------ #
    # Top-level selection                                                 #
    # ------------------------------------------------------------------ #
    def select(
        self,
        question: str,
        candidates: list[Candidate],
        prior_selected_texts: Optional[list[str]] = None,
    ) -> tuple[list[Candidate], dict]:
        """Return (kept_candidates, trace) for a single retrieval interception."""
        prior_selected_texts = list(prior_selected_texts or [])
        if not candidates:
            return candidates, {"note": "no_candidates"}

        budget = max(1, self.config.final_budget)

        # Fast paths that need no model calls.
        if not self.config.rerank_enabled:
            return candidates[:budget], {"note": "rerank_disabled_trim"}
        if self.config.scorer == "similarity":
            # Backend already returns candidates in descending score order.
            kept = sorted(candidates, key=lambda c: c.prior_score, reverse=True)[:budget]
            return kept, {"note": "similarity_trim"}

        if self.config.mode == "greedy":
            return self._select_greedy(question, candidates, prior_selected_texts, budget)
        return self._select_flat(question, candidates, prior_selected_texts, budget)

    def _select_flat(self, question, candidates, prior_selected_texts, budget):
        a_hat = ""
        if self.config.use_draft:
            draft_ctx = list(prior_selected_texts)
            if self.config.draft_from_pool:
                # Seed the draft with the retrieved evidence so a_hat is the best answer given
                # the pool, not the model's (often wrong) zero-context prior.
                draft_ctx = draft_ctx + [c.text for c in candidates]
            a_hat = self._draft(question, draft_ctx)
        scored = []
        for cand in candidates:
            s = self.score(question, a_hat, prior_selected_texts, cand)
            scored.append((cand, s))
        scored.sort(key=lambda x: x[1], reverse=True)

        kept = [c for c, s in scored[:budget] if s > self.config.min_gain]
        if not kept:
            # Never strand the agent: keep the single best-scoring candidate.
            kept = [scored[0][0]]

        trace = {
            "note": "flat",
            "draft": a_hat,
            "scores": [
                {"text": c.text[:160], "prior": round(c.prior_score, 4), "score": round(s, 4)}
                for c, s in scored
            ],
            "kept": [c.text[:160] for c in kept],
        }
        return kept, trace

    def _select_greedy(self, question, candidates, prior_selected_texts, budget):
        pool = list(candidates)
        selected: list[Candidate] = []
        selected_texts = list(prior_selected_texts)
        steps = []
        while pool and len(selected) < budget:
            a_hat = ""
            if self.config.use_draft:
                draft_ctx = list(selected_texts)
                # On the first step nothing is selected yet; seed the draft with the remaining
                # pool so a_hat isn't the model's wrong zero-context prior (see _select_flat).
                if self.config.draft_from_pool and not selected:
                    draft_ctx = draft_ctx + [c.text for c in pool]
                a_hat = self._draft(question, draft_ctx)
            best_cand, best_score = None, float("-inf")
            round_scores = []
            for cand in pool:
                s = self.score(question, a_hat, selected_texts, cand)
                round_scores.append((cand, s))
                if s > best_score:
                    best_cand, best_score = cand, s
            steps.append(
                {
                    "draft": a_hat,
                    "picked": best_cand.text[:160] if best_cand else None,
                    "gain": round(best_score, 4),
                }
            )
            if best_cand is None or best_score < self.config.halt_tau:
                break
            selected.append(best_cand)
            selected_texts.append(best_cand.text)
            pool.remove(best_cand)

        if not selected:
            # Adaptive halt fired immediately; keep the top-1 by prior score.
            selected = [max(candidates, key=lambda c: c.prior_score)]

        trace = {"note": "greedy", "steps": steps, "kept": [c.text[:160] for c in selected]}
        return selected, trace

    # ------------------------------------------------------------------ #
    # Scorers                                                             #
    # ------------------------------------------------------------------ #
    def score(self, question: str, a_hat: str, selected_texts: list[str], cand: Candidate) -> float:
        scorer = self.config.scorer
        if scorer == "judge":
            return self._judge(question, a_hat, cand.text)
        if scorer == "logprob":
            if self._logprob_supported is False:
                # Endpoint doesn't support echo logprobs -> degrade the whole scorer to similarity.
                return cand.prior_score
            val = self._mig_logprob(question, a_hat, selected_texts, cand.text)
            if val is None:
                # This candidate's span was unmeasurable (but the endpoint works). MIG values
                # are in logp units (often negative), so falling back to the similarity scale
                # would wrongly outrank them -> rank this one last instead.
                return float("-inf")
            return val
        # similarity / unknown -> use the backend's own score
        return cand.prior_score

    def _draft(self, question: str, selected_texts: list[str]) -> str:
        """Generate a short draft answer conditioned on the question and already-selected
        memories. This is the MIG ``a_hat`` -- the model's current best guess."""
        context = ""
        if selected_texts:
            joined = "\n".join(f"- {t}" for t in selected_texts)
            context = f"Relevant remembered information:\n{joined}\n\n"
        system = (
            "You are a helpful assistant answering a question using remembered information "
            "about the user. Answer as concisely as possible. If you are unsure, give your "
            "single best guess in a few words."
        )
        user = f"{context}Question: {question}"
        prompt = self._chat_prompt(system, user)
        try:
            resp = self.client.completions.create(
                model=self.model_id,
                prompt=prompt,
                max_tokens=self.config.max_draft_tokens,
                temperature=self.config.draft_temperature,
                timeout=72000,
            )
            return self._clean(resp.choices[0].text)
        except Exception as e:  # noqa: BLE001 - draft is best-effort
            print(f"[MIG] draft generation failed: {e}")
            return ""

    def _judge(self, question: str, a_hat: str, cand_text: str) -> float:
        """LLM utility judge: rate 0-5 how useful a memory entry is for answering the question,
        accounting for what the current draft already covers (the information-gain framing)."""
        draft_line = f"Current draft answer (may be wrong or incomplete): {a_hat}\n" if a_hat else ""
        system = (
            "You are scoring how useful a single remembered memory entry is for correctly "
            "answering a question. Consider whether the entry contains the specific fact needed. "
            "Respond with ONLY a single integer from 0 to 5, where 0 means useless/irrelevant and "
            "5 means it directly contains the answer."
        )
        user = (
            f"Question: {question}\n"
            f"{draft_line}"
            f"Memory entry: {cand_text}\n\n"
            "Usefulness score (0-5):"
        )
        prompt = self._chat_prompt(system, user)
        try:
            resp = self.client.completions.create(
                model=self.model_id,
                prompt=prompt,
                max_tokens=8,
                temperature=self.config.judge_temperature,
                timeout=72000,
            )
            return self._parse_score(self._clean(resp.choices[0].text))
        except Exception as e:  # noqa: BLE001 - judge is best-effort
            print(f"[MIG] judge scoring failed: {e}")
            return 0.0

    def _mig_logprob(
        self, question: str, a_hat: str, selected_texts: list[str], cand_text: str
    ) -> Optional[float]:
        """Exact information gain: logp(a_hat | S+{cand}) - logp(a_hat | S).

        Uses the OpenAI Completions ``echo=True, logprobs=1, max_tokens=0`` trick to read
        prompt-token logprobs at zero generation cost. Returns ``None`` if the endpoint does
        not support echo logprobs (caller then degrades gracefully)."""
        if self._logprob_supported is False:
            return None
        if not a_hat:
            # Without a draft answer there is nothing to measure the gain on.
            return None
        with_prefix, answer = self._logprob_context(question, selected_texts + [cand_text], a_hat)
        without_prefix, _ = self._logprob_context(question, selected_texts, a_hat)
        lp_with = self._answer_logprob(with_prefix, answer)
        lp_without = self._answer_logprob(without_prefix, answer)
        # ``_answer_logprob`` sets ``self._logprob_supported`` when it can tell the endpoint
        # structurally lacks prompt logprobs. A ``None`` here without that flag set just means
        # this particular span had no countable tokens (transient) -> degrade this one candidate.
        if lp_with is None or lp_without is None:
            return None
        return lp_with - lp_without

    def _logprob_context(self, question: str, texts: list[str], a_hat: str) -> tuple[str, str]:
        """Build (prefix, answer) such that ``prefix + answer`` is the scored string and the
        answer span is exactly ``answer``."""
        ctx = ""
        if texts:
            joined = "\n".join(f"- {t}" for t in texts)
            ctx = f"Relevant remembered information:\n{joined}\n\n"
        prefix = f"{ctx}Question: {question}\nAnswer: "
        return prefix, a_hat

    def _answer_logprob(self, prefix: str, answer: str) -> Optional[float]:
        """Sum the logprobs of the ``answer`` tokens given ``prefix`` via prompt echo."""
        full = prefix + answer
        try:
            resp = self.client.completions.create(
                model=self.model_id,
                prompt=full,
                max_tokens=0,
                echo=True,
                logprobs=1,
                temperature=0.0,
                timeout=72000,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[MIG] echo-logprob call failed (will fall back to similarity): {e}")
            return None
        try:
            lp = resp.choices[0].logprobs
            token_logprobs = lp.token_logprobs
            text_offset = lp.text_offset
        except (AttributeError, IndexError, TypeError):
            # The endpoint returned no logprobs structure at all -> it doesn't support echo
            # logprobs. Disable the scorer once so callers degrade to similarity.
            self._logprob_supported = False
            return None
        if not token_logprobs or not text_offset:
            self._logprob_supported = False
            return None
        # We did get a logprobs structure, so the endpoint supports it.
        self._logprob_supported = True
        boundary = len(prefix)
        total = 0.0
        counted = 0
        for off, tlp in zip(text_offset, token_logprobs):
            if off >= boundary and tlp is not None:
                total += tlp
                counted += 1
        if counted == 0:
            return None
        return total

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _chat_prompt(system: str, user: str) -> str:
        """Render a minimal Qwen-style chat prompt for the Completions API."""
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    @staticmethod
    def _clean(text: str) -> str:
        if not isinstance(text, str):
            return ""
        # Strip any reasoning block and trailing chat markers.
        if "</think>" in text:
            text = text.split("</think>")[-1]
        text = text.replace("<|im_end|>", "")
        return text.strip()

    @staticmethod
    def _parse_score(text: str) -> float:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return 0.0
        try:
            val = float(match.group(0))
        except ValueError:
            return 0.0
        # Clamp to the 0-5 judge scale.
        return max(0.0, min(5.0, val))

    def log_trace(self, test_id: Optional[str], entry: dict) -> None:
        """Append a structured trace line for a single interception (best-effort)."""
        if not self.config.log_dir or not test_id:
            return
        try:
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            safe_id = re.sub(r"[^\w.-]", "_", str(test_id))
            with open(log_dir / f"{safe_id}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:  # noqa: BLE001 - logging must never break inference
            print(f"[MIG] trace logging failed: {e}")
