"""
QwenGovHandler
==============

A thin, non-invasive subclass of :class:`QwenFCHandler` that adds the Stage 0
geometric governance filter (see ``middleware/governance_filter.py``) at the
tool-call boundary for BFCL V4 Memory tasks (KV and Vector backends only).

It hooks exactly two overridable points of the (already ``@final``) multi-turn
prompting loop:

  1. ``decode_execute``                   -> gate memory *write* calls; a NOOP'd
                                             write is rewritten in place to a
                                             read-only decoy call.
  2. ``_add_execution_results_prompting`` -> substitute the synthetic success for
                                             the decoy's output; observe genuine
                                             results to keep the mirror cache in
                                             sync.

Plus ``_pre_query_processing_prompting`` to build the per-test
:class:`GovernanceSession` (rehydrated from the backend's on-disk memory
snapshot). The session is stashed in ``inference_data`` (thread-safe: one dict
per test entry) AND in a ``threading.local`` slot, because ``decode_execute``
does not receive ``inference_data`` -- each test entry runs its whole loop on a
single worker thread, so thread-local state is safe.

Everything else -- the agent loop, the memory backend, the grader -- is untouched.
For non-memory categories (and for ``rec_sum``) the handler behaves byte-identically
to :class:`QwenFCHandler`.

All behavior is driven by ``GOV_*`` environment variables (see :class:`GovConfig`),
so a single registry entry covers every ablation arm, including ``GOV_DRY_RUN=1``
shadow mode.
"""

import json
import threading
from pathlib import Path

from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from bfcl_eval.model_handler.middleware.governance_filter import (
    GovConfig,
    GovernanceSession,
)
from bfcl_eval.utils import (
    extract_memory_backend_type,
    extract_test_category_from_id,
    get_directory_structure_by_id,
    is_first_memory_prereq_entry,
    is_memory,
)
from overrides import override

GOVERNED_BACKENDS = {"kv", "vector"}  # rec_sum is out of scope by design


class QwenGovHandler(QwenFCHandler):
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
        self.gov_config = GovConfig.from_env()
        self._gov_tls = threading.local()
        print(
            f"[GOV] QwenGovHandler active | enabled={self.gov_config.enabled} "
            f"dry_run={self.gov_config.dry_run} sim_high={self.gov_config.sim_high} "
            f"delta={self.gov_config.delta} "
            f"strict_thresholds={self.gov_config.strict_thresholds} "
            f"artifact={self.gov_config.artifact_path}"
        )
        print(
            f"[GOV] Stage1(NLI) enabled={self.gov_config.nli_enabled} "
            f"shadow={self.gov_config.nli_shadow} k={self.gov_config.nli_k} "
            f"tau_entail={self.gov_config.tau_entail} "
            f"tau_contra={self.gov_config.tau_contra} "
            f"delta_spec={self.gov_config.delta_spec} | "
            f"Stage2 enabled={self.gov_config.s2_enabled} "
            f"shadow={self.gov_config.s2_shadow} margin={self.gov_config.s2_margin} "
            f"canon_llm={self.gov_config.s2_canon_llm} "
            f"t_kv={self.gov_config.s2_t_kv} t_vec={self.gov_config.s2_t_vec}"
        )

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        inference_data = super()._pre_query_processing_prompting(test_entry)
        self._gov_tls.session = None  # reset any state from a previous entry

        session = self._build_session(test_entry)
        if session is not None:
            inference_data["_gov_session"] = session
            self._gov_tls.session = session
        return inference_data

    def _build_session(self, test_entry: dict):
        if not self.gov_config.enabled:
            return None
        test_id = test_entry.get("id", "")
        if not is_memory(test_id):
            return None
        backend = extract_memory_backend_type(
            extract_test_category_from_id(test_id, remove_prereq=True)
        )
        if backend not in GOVERNED_BACKENDS:
            return None

        snapshot, snapshot_path, snapshot_missing = self._load_snapshot(test_entry)
        try:
            return GovernanceSession(
                cfg=self.gov_config,
                backend=backend,
                test_id=test_id,
                snapshot=snapshot,
                snapshot_path=snapshot_path,
                snapshot_missing=snapshot_missing,
            )
        except RuntimeError:
            # Missing ABTT artifact / encoder must fail the run loudly, not silently
            # degrade into an ungoverned arm that pollutes the A/B comparison.
            raise

    def _load_snapshot(self, test_entry: dict):
        """Read (read-only) the same rolling snapshot the backend just loaded.

        Path construction mirrors MemoryAPI._prepare_snapshot exactly:
        <model_result_dir>/<agentic/memory/<backend>>/memory_snapshot/<scenario>_final.json
        The backend instance is created -- and the snapshot therefore already
        written by the previous entry in the depends_on chain -- before
        _pre_query_processing_prompting runs (base_handler.py:421 vs :465).
        """
        test_id = test_entry["id"]
        initial_config: dict = test_entry["initial_config"]
        backend_config = next(iter(initial_config.values()))
        model_result_dir = Path(backend_config["model_result_dir"])
        scenario = backend_config["scenario"]

        snapshot_file = (
            model_result_dir
            / get_directory_structure_by_id(test_id)
            / "memory_snapshot"
            / f"{scenario}_final.json"
        )

        if is_first_memory_prereq_entry(test_id):
            return None, str(snapshot_file), False  # clean start by design
        if not snapshot_file.exists():
            # Mirrors the backend's warn-and-start-empty behavior so both sides of
            # the mirror stay consistent even after a crashed prereq entry.
            print(
                f"[GOV] Warning: no snapshot at {snapshot_file}; "
                f"mirror cache starts empty for {test_id}."
            )
            return None, str(snapshot_file), True
        with open(snapshot_file, "r", encoding="utf-8") as f:
            return json.load(f), str(snapshot_file), False

    # -- Stage 2 probe source: capture the current user turn text BEFORE the
    # model ever produces a call (the SS4.1 anti-circularity guarantee). Both
    # hooks are non-@final in the base loop.

    def _stash_user_text(self, messages: list[dict]) -> None:
        session = getattr(self._gov_tls, "session", None)
        if session is None:
            return
        user_texts = [
            str(m.get("content", ""))
            for m in messages
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        if user_texts:
            session.user_text = user_texts[-1]

    @override
    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        self._stash_user_text(first_turn_message)
        return super().add_first_turn_message_prompting(
            inference_data, first_turn_message
        )

    @override
    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        self._stash_user_text(user_message)
        return super()._add_next_turn_user_message_prompting(
            inference_data, user_message
        )

    @override
    def decode_execute(self, result, has_tool_call_tag):
        calls = super().decode_execute(result, has_tool_call_tag)
        session = getattr(self._gov_tls, "session", None)
        if session is None or not calls:
            return calls
        return session.govern_calls(calls)

    @override
    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        session = inference_data.get("_gov_session")
        if session is None:
            return super()._add_execution_results_prompting(
                inference_data, execution_results, model_response_data
            )
        # patch_results also restores the original call strings inside
        # model_responses_decoded (in place), so the tool-message "name" and the
        # result-file log show the model's actual call, not the decoy.
        new_results = session.patch_results(
            execution_results, model_response_data.get("model_responses_decoded") or []
        )
        return super()._add_execution_results_prompting(
            inference_data, new_results, model_response_data
        )
