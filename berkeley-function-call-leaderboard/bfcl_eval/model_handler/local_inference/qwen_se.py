"""
QwenSemanticEntropyHandler
==========================

A thin subclass of :class:`QwenFCHandler` that adds a sampling-based semantic-entropy
gate at the generation step for BFCL V4 Memory tasks (thesis "Idea 1").

For every gated step it samples ``SE_NUM_SAMPLES`` completions in a single vLLM call
(``n=N`` on the completions endpoint -- shared prefill, honest token accounting since
``usage.completion_tokens`` counts all N samples), clusters them semantically, and
commits the majority action. Destructive memory operations (clear/remove) additionally
require low cluster entropy and a strong majority; otherwise the gate falls back to the
largest non-destructive cluster. See ``middleware/semantic_entropy.py`` for the policy.

Only two overrides:

  1. ``_pre_query_processing_prompting`` -> stash the test id inside ``inference_data``
     (created fresh per test entry and threaded through the loop -> thread-safe).
  2. ``_query_prompting``               -> the gated N-sample query. The chosen candidate
     is moved to ``choices[0]``, which is all the downstream parsers ever read.

With ``SE_NUM_SAMPLES=1`` or for non-memory categories the handler defers to the parent
and is byte-identical to the ``-FC`` baseline.
"""

import time

from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from bfcl_eval.model_handler.middleware.semantic_entropy import (
    SEConfig,
    build_candidate,
    choose,
    log_record,
)
from bfcl_eval.utils import is_memory, is_memory_prereq
from overrides import override


class QwenSemanticEntropyHandler(QwenFCHandler):
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
        self.se_config = SEConfig.from_env()
        print(
            f"[SE] QwenSemanticEntropyHandler active | num_samples={self.se_config.num_samples} "
            f"temperature={self.se_config.temperature} "
            f"gate_destructive={self.se_config.gate_destructive} "
            f"gate_recall={self.se_config.gate_recall}"
        )

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        inference_data = super()._pre_query_processing_prompting(test_entry)
        # Dict-carried per-test state -> no self.* races across threads.
        inference_data["_se_test_id"] = test_entry["id"]
        return inference_data

    def _se_gate_active(self, test_id: str) -> bool:
        if self.se_config.num_samples <= 1:
            return False
        if not test_id or not is_memory(test_id):
            return False
        if not self.se_config.gate_recall and not is_memory_prereq(test_id):
            return False  # ablation E3: gate the memory-writing prereq entries only
        return True

    @override
    def _query_prompting(self, inference_data: dict):
        cfg = self.se_config
        test_id = inference_data.get("_se_test_id", "")
        if not self._se_gate_active(test_id):
            return super()._query_prompting(inference_data)

        # --- replicate base_oss_handler._query_prompting with n=N and gate temperature ---
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

        start_time = time.time()
        api_response = self.client.completions.create(
            model=self.model_path_or_id,
            temperature=cfg.temperature,
            prompt=formatted_prompt,
            max_tokens=leftover_tokens_count,
            n=cfg.num_samples,
            timeout=72000,
            **({"extra_body": extra_body} if extra_body else {}),
        )
        end_time = time.time()

        # --- decode every sample, gate, put the chosen one first ---
        candidates = []
        for choice in api_response.choices:
            try:
                calls = self.decode_execute(choice.text, False)
            except Exception:
                calls = None
            candidates.append(build_candidate(choice.text, calls))

        chosen, record = choose(candidates, cfg)
        if chosen != 0:
            choices = list(api_response.choices)
            choices.insert(0, choices.pop(chosen))
            api_response.choices = choices

        record["test_id"] = test_id
        record["is_prereq"] = is_memory_prereq(test_id)
        record["input_tokens"] = api_response.usage.prompt_tokens
        record["output_tokens"] = api_response.usage.completion_tokens
        record["api_seconds"] = round(end_time - start_time, 3)
        log_record(record, cfg)
        if cfg.verbose:
            print(
                f"[SE] {test_id} | H={record['entropy']} m={record['majority_fraction']} "
                f"clusters={record['cluster_sizes']} -> {record['decision']}"
            )

        return api_response, end_time - start_time
