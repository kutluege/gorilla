"""
QwenHactHandler
===============

Action-side H_act instrumentation (H-Nav Stage 2) on top of
:class:`QwenGovHandler`. The two parents hook disjoint points of the ``@final``
multi-turn loop, so composition is clean:

  - ``QwenGovHandler``: ``decode_execute`` (govern the PRIMARY calls) +
    ``_add_execution_results_prompting`` (patch results / mirror observe).
  - This class adds ``_query_prompting`` only.

Per gated model step (memory prereq turns by default -- ``HACT_GATE_RECALL=0``),
it issues TWO requests against the same prompt:

  1. **primary** -- byte-identical parameter set to the baseline handler
     (handler temperature, same max_tokens / extra_body) plus
     ``logprobs=HACT_LOGPROBS`` and a deterministic ``seed``. In shadow mode
     this response object is returned unchanged: the agent loop sees exactly
     what the baseline would (modulo server nondeterminism, which the
     in-campaign baseline arm quantifies).
  2. **exploration** -- ``n=HACT_N-1`` at ``HACT_TEMP`` (SE precedent, 0.7),
     used ONLY to estimate the action distribution. Exploration samples are
     decoded with the UNGOVERNED grandparent parser -- they must never reach
     ``session.govern_calls`` (stateful step counter + mirror cache).

The assembled ``hact1`` record is stashed on the governance session
(``session.pending_hact``); ``govern_calls`` flushes it with the authoritative
``step_idx`` so hact rows join gov2 decision rows on ``(test_id, step_idx)``.
If the primary produced no governed calls the stash is flushed from
``decode_execute`` as an orphan (``step_idx=None``) -- orphans ARE data (the
parser-failure / no-call prediction target), not loss.

Non-memory categories, rec_sum, and ``HACT_ENABLED=0`` are byte-identical to
``QwenGovHandler``.
"""

import time

from bfcl_eval.model_handler.local_inference.qwen_gov import (
    GOVERNED_BACKENDS,
    QwenGovHandler,
)
from bfcl_eval.model_handler.middleware.action_space import canonicalize_sample
from bfcl_eval.model_handler.middleware.hact_sampler import (
    HactConfig,
    build_hact_record,
    log_hact_record,
    logprob_features,
    stable_seed,
)
from bfcl_eval.utils import (
    extract_memory_backend_type,
    extract_test_category_from_id,
    is_memory,
    is_memory_prereq,
)
from overrides import override

# Policies implemented so far. Stage 4 intervention policies land separately;
# failing loudly here prevents silently running an unimplemented arm.
_IMPLEMENTED_POLICIES = ("shadow",)


class QwenHactHandler(QwenGovHandler):
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(
            model_name, temperature, registry_name, is_fc_model, dtype=dtype, **kwargs
        )
        self.hact_config = HactConfig.from_env()
        if self.hact_config.enabled and self.hact_config.policy not in _IMPLEMENTED_POLICIES:
            raise ValueError(
                f"HACT_POLICY={self.hact_config.policy!r} is not implemented yet "
                f"(implemented: {_IMPLEMENTED_POLICIES})"
            )
        # None = unknown; True/False once the server's logprobs support is observed.
        self._hact_logprob_supported = None
        print(
            f"[HACT] QwenHactHandler active | enabled={self.hact_config.enabled} "
            f"n={self.hact_config.num_samples} temp={self.hact_config.temperature} "
            f"logprobs_k={self.hact_config.logprobs_k} "
            f"gate_recall={self.hact_config.gate_recall} "
            f"policy={self.hact_config.policy}"
        )

    # ------------------------------------------------------------- gating

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        inference_data = super()._pre_query_processing_prompting(test_entry)
        inference_data["_hact_test_id"] = test_entry["id"]
        inference_data["_hact_call_idx"] = 0
        return inference_data

    def _hact_backend(self, test_id: str) -> str:
        try:
            return extract_memory_backend_type(
                extract_test_category_from_id(test_id, remove_prereq=True)
            )
        except Exception:
            return ""

    def _hact_gate_active(self, test_id: str) -> bool:
        cfg = self.hact_config
        if not cfg.enabled or cfg.num_samples <= 1:
            return False
        if not test_id or not is_memory(test_id):
            return False
        if self._hact_backend(test_id) not in GOVERNED_BACKENDS:
            return False
        if not cfg.gate_recall and not is_memory_prereq(test_id):
            return False
        return True

    # ------------------------------------------------------------- sampling

    @override
    def _query_prompting(self, inference_data: dict):
        cfg = self.hact_config
        test_id = inference_data.get("_hact_test_id", "")
        if not self._hact_gate_active(test_id):
            return super()._query_prompting(inference_data)

        call_idx = inference_data.get("_hact_call_idx", 0)
        inference_data["_hact_call_idx"] = call_idx + 1
        backend = self._hact_backend(test_id)

        # --- replicate base_oss_handler._query_prompting request construction ---
        function: list[dict] = inference_data["function"]
        message: list[dict] = inference_data["message"]

        formatted_prompt: str = self._format_prompt(message, function)
        inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}

        input_token_count = len(self.tokenizer.tokenize(formatted_prompt))
        if self.max_context_length < input_token_count + 2:
            leftover_tokens_count = 1000
        else:
            leftover_tokens_count = min(
                4096,
                self.max_context_length - input_token_count - 2,
            )

        extra_body = {}
        if hasattr(self, "stop_token_ids"):
            extra_body["stop_token_ids"] = self.stop_token_ids
        if hasattr(self, "skip_special_tokens"):
            extra_body["skip_special_tokens"] = self.skip_special_tokens

        common = dict(
            model=self.model_path_or_id,
            prompt=formatted_prompt,
            max_tokens=leftover_tokens_count,
            timeout=72000,
            **({"extra_body": extra_body} if extra_body else {}),
        )
        seed_primary = stable_seed(cfg.seed_base, test_id, call_idx, salt=0)
        seed_explore = stable_seed(cfg.seed_base, test_id, call_idx, salt=1)

        start_time = time.time()

        # --- 1. PRIMARY: baseline params + logprobs + pinned seed ---
        want_logprobs = cfg.logprobs_k > 0 and self._hact_logprob_supported is not False
        try:
            primary_resp = self.client.completions.create(
                temperature=self.temperature,
                seed=seed_primary,
                **({"logprobs": cfg.logprobs_k} if want_logprobs else {}),
                **common,
            )
        except Exception as e:  # noqa: BLE001
            if want_logprobs:
                # Degrade once, loudly, and never fail the run for logprobs.
                print(f"[HACT] primary with logprobs failed ({e}); retrying without")
                self._hact_logprob_supported = False
                primary_resp = self.client.completions.create(
                    temperature=self.temperature, seed=seed_primary, **common
                )
            else:
                raise

        # --- 2. EXPLORATION: n-1 samples at exploration temperature ---
        explore_resp = None
        if cfg.num_samples > 1:
            explore_resp = self.client.completions.create(
                temperature=cfg.temperature,
                n=cfg.num_samples - 1,
                seed=seed_explore,
                **common,
            )
        end_time = time.time()

        # --- decode: UNGOVERNED grandparent parse for every sample ---
        def raw_decode(text: str):
            try:
                return super(QwenGovHandler, self).decode_execute(text, False)
            except Exception:
                return None

        primary_choice = primary_resp.choices[0]
        primary_text = primary_choice.text
        primary_calls = raw_decode(primary_text)

        sample_texts = [primary_text]
        sample_calls = [primary_calls]
        if explore_resp is not None:
            for ch in explore_resp.choices:
                sample_texts.append(ch.text)
                sample_calls.append(raw_decode(ch.text))
        actions_per_sample = [canonicalize_sample(c, backend) for c in sample_calls]

        # --- logprob features + support flag (only meaningful if requested) ---
        lp_obj = getattr(primary_choice, "logprobs", None)
        lp_feats = logprob_features(lp_obj, primary_text)
        if want_logprobs:
            got = any(v is not None for v in lp_feats.values())
            if self._hact_logprob_supported is None:
                self._hact_logprob_supported = got
                if not got:
                    print("[HACT] server returned no logprobs structure; degrading")

        # --- token accounting: full cost (primary + exploration) in the record;
        #     the harness sees only the primary response's usage, as baseline ---
        in_tok = getattr(primary_resp.usage, "prompt_tokens", None)
        out_tok = getattr(primary_resp.usage, "completion_tokens", None)
        if explore_resp is not None and getattr(explore_resp, "usage", None) is not None:
            in_tok = (in_tok or 0) + (explore_resp.usage.prompt_tokens or 0)
            out_tok = (out_tok or 0) + (explore_resp.usage.completion_tokens or 0)

        record = build_hact_record(
            test_id=test_id,
            is_prereq=is_memory_prereq(test_id),
            backend=backend,
            cfg=cfg,
            primary_text=primary_text,
            primary_calls=primary_calls,
            primary_lp_feats=lp_feats,
            logprobs_supported=self._hact_logprob_supported,
            sample_texts=sample_texts,
            sample_calls=sample_calls,
            actions_per_sample=actions_per_sample,
            seeds=[seed_primary, seed_explore],
            input_tokens=in_tok,
            output_tokens=out_tok,
            api_seconds=end_time - start_time,
        )
        record["call_idx_in_entry"] = call_idx
        record["_log_file"] = cfg.log_file

        session = getattr(self._gov_tls, "session", None)
        if session is not None:
            # Flushed by session.govern_calls with the authoritative step_idx,
            # or by our decode_execute override as an orphan.
            session.pending_hact = record
        else:
            # No governance session (GOV disabled): still keep the data.
            record.pop("_log_file", None)
            record["step_idx"] = None
            record["orphan"] = False
            record["session_missing"] = True
            log_hact_record(record, self.gov_config.log_dir or ".", cfg.log_file)

        if cfg.verbose:
            v = record["votes"]
            print(
                f"[HACT] {test_id} step~{call_idx} | H_r3={v['h_act_target']:.3f} "
                f"modal={v['modal_share_target']} margin={v['vote_margin_target']} "
                f"unique={v['n_unique_target']}"
            )

        # Shadow policy: the primary response object, untouched.
        return primary_resp, end_time - start_time

    # ------------------------------------------------------------- orphan flush

    @override
    def decode_execute(self, result, has_tool_call_tag):
        calls = super().decode_execute(result, has_tool_call_tag)
        session = getattr(self._gov_tls, "session", None)
        if session is not None:
            pending = getattr(session, "pending_hact", None)
            if pending is not None:
                # govern_calls was not reached (no calls decoded) -> orphan.
                session.pending_hact = None
                log_file = pending.pop("_log_file", "hact_log.jsonl")
                pending["step_idx"] = None
                pending["orphan"] = True
                log_hact_record(pending, session.cfg.log_dir or ".", log_file)
        return calls
