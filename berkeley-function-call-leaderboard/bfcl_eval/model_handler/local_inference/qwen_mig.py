"""
QwenMIGHandler
==============

A thin, non-invasive subclass of :class:`QwenFCHandler` that adds a Memory Information Gain
(MIG) reranker at the tool-result boundary for BFCL V4 Memory tasks.

It hooks exactly two overridable points of the (already ``@final``) multi-turn prompting loop:

  1. ``decode_execute``                  -> widen the retrieval pool (rewrite ``top_k`` / ``k``).
  2. ``_add_execution_results_prompting`` -> rerank/trim the retrieved candidates before they
                                            enter the chat history the agent sees next.

Plus ``_pre_query_processing_prompting`` to stash per-test state inside ``inference_data``
(which is created fresh per test entry and threaded through the loop, so this is naturally
thread-safe -- no shared mutable state on ``self``).

Everything else -- the agent loop, the memory backend, the scoring -- is untouched. For
non-memory categories the handler behaves byte-identically to :class:`QwenFCHandler`.

All behavior is driven by ``MIG_*`` environment variables (see :class:`MIGConfig`), so a
single registry entry covers every ablation arm.
"""

from typing import Any

from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from bfcl_eval.model_handler.middleware.mig_reranker import MIGConfig, MIGReranker
from bfcl_eval.model_handler.utils import convert_to_function_call
from bfcl_eval.utils import is_memory
from overrides import override


class QwenMIGHandler(QwenFCHandler):
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
        self.mig_config = MIGConfig.from_env()
        # Lazily constructed: the reranker needs ``self.model_path_or_id``, which is only
        # set during ``spin_up_local_server`` (well before any inference call).
        self._mig_reranker = None
        print(
            f"[MIG] QwenMIGHandler active | enabled={self.mig_config.enabled} "
            f"scorer={self.mig_config.scorer} mode={self.mig_config.mode} "
            f"pool_size={self.mig_config.pool_size} final_budget={self.mig_config.final_budget} "
            f"rerank={self.mig_config.rerank_enabled}"
        )

    @property
    def reranker(self) -> MIGReranker:
        if self._mig_reranker is None:
            self._mig_reranker = MIGReranker(
                client=self.client,
                model_id=self.model_path_or_id,
                config=self.mig_config,
                tokenizer=getattr(self, "tokenizer", None),
            )
        return self._mig_reranker

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        inference_data = super()._pre_query_processing_prompting(test_entry)
        # Activate MIG only for memory categories; everything else is a pass-through.
        if self.mig_config.enabled and is_memory(test_entry["id"]):
            inference_data["_mig_active"] = True
            inference_data["_mig_test_id"] = test_entry["id"]
            # Per-test state lives in inference_data (one dict per test entry -> thread-safe).
            inference_data["_mig_state"] = {"selected_texts": [], "interception": 0}
        return inference_data

    @override
    def decode_execute(self, result, has_tool_call_tag):
        """Same decoding as the parent, but widens the retrieval pool first so we have a
        larger candidate set to rerank from. Pool widening only touches memory retrieval
        functions; for any other call it is a no-op."""
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")

        if self.mig_config.enabled and self.mig_config.pool_size > 0:
            for item in tool_calls:
                MIGReranker.widen_pool(item, self.mig_config.pool_size)

        decoded_result = []
        for item in tool_calls:
            if type(item) == str:
                item = eval(item)
            decoded_result.append({item["name"]: item["arguments"]})
        return convert_to_function_call(decoded_result)

    @override
    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        """Rerank/trim retrieval results before they enter the chat history.

        We build a *new* list of results to hand to the parent (which appends them to the
        chat history). The original ``execution_results`` list is left intact, so the
        verbose inference log still records the full widened pool -- useful for offline
        gold-tracking / recall@k analysis."""
        if not inference_data.get("_mig_active") or not self.mig_config.enabled:
            return super()._add_execution_results_prompting(
                inference_data, execution_results, model_response_data
            )

        question = self.reranker.extract_question(inference_data.get("message", []))
        state = inference_data.setdefault("_mig_state", {"selected_texts": [], "interception": 0})
        test_id = inference_data.get("_mig_test_id")
        decoded_calls = model_response_data.get("model_responses_decoded", []) or []

        new_results: list[str] = []
        for idx, raw in enumerate(execution_results):
            fn_call = decoded_calls[idx] if idx < len(decoded_calls) else ""
            new_results.append(
                self._rerank_one(question, raw, fn_call, state, test_id)
            )

        return super()._add_execution_results_prompting(
            inference_data, new_results, model_response_data
        )

    def _rerank_one(
        self, question: str, raw: str, fn_call: str, state: dict, test_id
    ) -> str:
        kind, candidates = self.reranker.parse_candidates(raw)
        if kind is None or not candidates:
            # Not a rerankable retrieval result (exact-key value, whole blob, error, ...).
            return raw

        kept, trace = self.reranker.select(
            question, candidates, prior_selected_texts=state.get("selected_texts", [])
        )

        # Accumulate selected texts across interceptions so later retrievals can condition
        # their draft/scoring on what has already been surfaced (matters in greedy mode).
        state.setdefault("selected_texts", []).extend([c.text for c in kept])
        state["interception"] = state.get("interception", 0) + 1

        if self.mig_config.verbose:
            print(
                f"[MIG] {test_id} | {fn_call.split('(')[0] if fn_call else kind} | "
                f"pool={len(candidates)} -> kept={len(kept)} | scorer={self.mig_config.scorer} "
                f"mode={self.mig_config.mode}"
            )

        self.reranker.log_trace(
            test_id,
            {
                "interception": state["interception"],
                "question": question,
                "fn_call": fn_call,
                "kind": kind,
                "pool_size": len(candidates),
                "kept_size": len(kept),
                "trace": trace,
            },
        )

        return self.reranker.reserialize(kind, kept)
