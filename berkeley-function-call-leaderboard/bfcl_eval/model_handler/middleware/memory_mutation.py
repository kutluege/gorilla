"""
Memory-mutation admission boundary types (Plan v2 Step 1, plan SS7.1).
=====================================================================

Producer-agnostic candidate/decision types for the two-stage
Geometry -> joint Margin-Entropy write-admission architecture
(``TWO_STAGE_GEOMETRY_MARGIN_ENTROPY_PLAN.md``).

The boundary's *input* is a :class:`MemoryMutationCandidate`, not a call string:
tool calls are one producer (the only live one in this benchmark), placement
expansions are the second (``source="placement_expansion"``), and offline audit
is the third (``source="audit"``). The concrete boundary implementation lives in
``admission_policy.py`` (Step 4); this module is pure types plus the adapter
from the existing :class:`~.governance_filter.WriteCandidate`.

Pure logic: no BFCL imports, no model calls, no I/O.
"""

from dataclasses import dataclass, field
from typing import Optional

# Action / stage vocabularies (plan SS7.1). Plain tuples rather than Literal so
# the module stays importable on older typing stacks and the closed vocabulary
# is testable at runtime.
ADMISSION_ACTIONS = ("ADD", "NOOP", "SAFE_REWRITE", "ABSTAIN")
ADMISSION_STAGES = ("GEOMETRY", "MARGIN_ENTROPY")

# Closed reason-code vocabulary (plan SS17.1). ``*_preflight_blocked`` /
# ``*_smallstore`` appear as suffixed variants of these bases.
REASON_CODES = (
    "s0_exact_dup",
    "s0_noop",
    "s0_confident_add",
    "s0_unretrievable_escalate",
    "s0_escalate",
    "s1_confident",
    "s1_bounded_interference",
    "s1_shadowed_duplicate",
    "s1_high_risk",
    "s1_ambiguous_default",
    "expansion_noop",
)
REASON_CODE_SUFFIXES = ("_preflight_blocked", "_smallstore")


def is_valid_reason_code(code: str) -> bool:
    """True iff ``code`` is a closed-vocabulary base, optionally suffixed."""
    stripped = code
    changed = True
    while changed:
        changed = False
        for suffix in REASON_CODE_SUFFIXES:
            if stripped.endswith(suffix):
                stripped = stripped[: -len(suffix)]
                changed = True
    return stripped in REASON_CODES


@dataclass(frozen=True)
class MemoryMutationCandidate:
    """One candidate memory mutation, normalized away from its producer.

    v1 normalization policy (plan SS11): ``normalized_value`` is byte-identical
    to ``raw_value`` (no generative rewriting of values/texts); only KV *keys*
    may carry a distinct ``normalized_key`` (deterministic sanitization).
    """

    operation: str  # "add" | "replace" | "update"
    backend: str  # "kv" | "vector"
    tier: str  # "core" | "archival"
    raw_key: Optional[str]  # KV key as emitted by the model (None for vector)
    raw_value: Optional[str]  # KV value / vector text as emitted
    normalized_key: Optional[str]  # v1: sanitized key (existing _sanitize_kv_key)
    normalized_value: Optional[str]  # v1: == raw_value (no value rewriting)
    source: str  # "tool_call" | "placement_expansion" | "audit"
    source_tool: Optional[str]  # original op name, e.g. "core_memory_add"
    raw_call: str  # verbatim call string (for _pending bookkeeping)
    metadata: dict = field(default_factory=dict)  # step_idx, call_idx, ...


@dataclass(frozen=True)
class AdmissionDecision:
    """The boundary's verdict on one candidate (plan SS7.1)."""

    action: str  # one of ADMISSION_ACTIONS
    stage: str  # one of ADMISSION_STAGES
    confidence: Optional[float]  # calibrated risk when policy=B, else None
    reason_code: str  # closed vocabulary, see REASON_CODES
    rewritten_call: Optional[str]  # only for SAFE_REWRITE
    diagnostics: dict = field(default_factory=dict)  # full signal record, SS17

    def __post_init__(self):
        if self.action not in ADMISSION_ACTIONS:
            raise ValueError(f"invalid admission action: {self.action!r}")
        if self.stage not in ADMISSION_STAGES:
            raise ValueError(f"invalid admission stage: {self.stage!r}")
        if self.action == "SAFE_REWRITE" and not self.rewritten_call:
            raise ValueError("SAFE_REWRITE requires rewritten_call")


def from_write_candidate(wc, source: str = "tool_call") -> MemoryMutationCandidate:
    """Adapt a ``governance_filter.WriteCandidate`` into the boundary's type.

    Import-free by design (duck-typed on the WriteCandidate fields) so this
    module never imports governance_filter — the dependency arrow points the
    other way (session -> boundary -> types).
    """
    if wc.backend == "kv":
        raw_key = str(wc.args["key"])
        raw_value = str(wc.args["value"])
        # Deterministic key normalization only (plan SS11): lowercase,
        # non-alphanumeric runs collapsed to underscores. Mirror of
        # governance_filter._sanitize_kv_key, inlined to keep this module pure.
        import re

        normalized_key = re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]+", "_", raw_key.lower())).strip("_")
    else:
        raw_key = None
        raw_value = str(wc.args["text"] if wc.kind == "add" else wc.args["new_text"])
        normalized_key = None
    return MemoryMutationCandidate(
        operation=wc.kind,
        backend=wc.backend,
        tier=wc.tier,
        raw_key=raw_key,
        raw_value=raw_value,
        normalized_key=normalized_key,
        normalized_value=raw_value,  # v1: byte-identical, never rewritten
        source=source,
        source_tool=wc.op,
        raw_call=wc.raw_call,
        metadata={"ref": wc.ref, "composite_text": wc.text},
    )
