"""Canonical action representation for action-side H_act (H-Nav Stage 2).

Maps decoded tool-call strings to a canonical action at four resolutions:

    R1  tool          -- raw function name
    R2  op            -- normalized operation family ("add:core", "read", ...)
    R3  op + target   -- adds the normalized target identity (KV key / vec_id)
    R4  op + target + content fingerprint

and computes vote-distribution features (Shannon entropy H_act, modal share,
top1-top2 vote margin, ...) over N sampled completions.

Design rules (mirrors the Stage 2 pre-registration):
- Parsing reuses ``governance_filter.parse_call`` / ``_bind_args`` and the
  KV/VECTOR op tables byte-identically -- semantically equivalent forms
  (positional vs keyword args, whitespace, quote style) collapse to the same
  canonical action; a parse failure maps to op="parse_fail" and NEVER raises.
- Superficial content differences (case, whitespace, trailing punctuation)
  do not create artificial disagreement at R4; nothing semantic is folded.
- This module is pure (no I/O, no model calls, no ground-truth imports) and
  is safe to import both at runtime (handler) and offline (analysis).
"""

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from bfcl_eval.model_handler.middleware.governance_filter import (
    KV_OBSERVED_OPS,
    KV_WRITE_OPS,
    VECTOR_OBSERVED_OPS,
    VECTOR_WRITE_OPS,
    _bind_args,
    kv_composite_text,
    parse_call,
)

RESOLUTIONS = (1, 2, 3, 4)
RESOLUTION_TAGS = {1: "tool", 2: "op", 3: "target", 4: "full"}

# Retrieve-family prefixes: memory reads are actions too (the model may choose
# to read instead of write); they carry no target/content canonicalization.
_MEMORY_PREFIXES = ("core_memory_", "archival_memory_")

_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class CanonicalAction:
    tool: str                    # R1: function name, "no_call" or "parse_fail"
    op: str                      # R2: e.g. "add:core", "replace:archival", "read",
                                 #     "remove:core", "clear:archival", "other",
                                 #     "no_call", "parse_fail"
    target: Optional[str]        # R3 component: KV key / vec_id / "" (keyless add) / None
    content_fp: Optional[str]    # R4 component: sha256(normalize_text)[:16] or None
    backend: str                 # "kv" | "vector" | "none"


NO_CALL = CanonicalAction(tool="no_call", op="no_call", target=None,
                          content_fp=None, backend="none")


def normalize_text(text: str) -> str:
    """Superficial normalization only: casefold, collapse whitespace, strip
    trailing punctuation. Never semantic."""
    t = _WS_RE.sub(" ", str(text)).strip().casefold()
    return t.rstrip(".,;: ")


def content_fingerprint(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()[:16]


def parse_fail(backend: str) -> CanonicalAction:
    return CanonicalAction(tool="parse_fail", op="parse_fail", target=None,
                           content_fp=None, backend=backend)


def canonicalize(call_str: str, backend: str) -> CanonicalAction:
    """Canonicalize one decoded call string. Never raises."""
    try:
        parsed = parse_call(call_str)
        if parsed is None:
            return parse_fail(backend)
        name, positional, kwargs = parsed

        write_table = KV_WRITE_OPS if backend == "kv" else VECTOR_WRITE_OPS
        observed_table = KV_OBSERVED_OPS if backend == "kv" else VECTOR_OBSERVED_OPS

        if name in write_table:
            kind, tier, params = write_table[name]
            args = _bind_args(positional, kwargs, params)
            if args is None:
                return parse_fail(backend)
            if backend == "kv":
                target = normalize_text(str(args["key"]))
                content = kv_composite_text(args["key"], args["value"])
            else:
                if kind == "add":
                    target = ""          # keyless: identity is the content itself
                    content = str(args["text"])
                else:
                    try:
                        target = str(int(args["vec_id"]))
                    except (TypeError, ValueError):
                        target = normalize_text(str(args["vec_id"]))
                    content = str(args["new_text"])
            return CanonicalAction(tool=name, op=f"{kind}:{tier}", target=target,
                                   content_fp=content_fingerprint(content),
                                   backend=backend)

        if name in observed_table:
            kind, tier, params = observed_table[name]
            args = _bind_args(positional, kwargs, params)
            target = None
            if args and params:
                target = normalize_text(str(args[params[0]]))
            return CanonicalAction(tool=name, op=f"{kind}:{tier}", target=target,
                                   content_fp=None, backend=backend)

        if name.startswith(_MEMORY_PREFIXES):
            return CanonicalAction(tool=name, op="read", target=None,
                                   content_fp=None, backend=backend)

        return CanonicalAction(tool=name, op="other", target=None,
                               content_fp=None, backend=backend)
    except Exception:
        return parse_fail(backend)


def canonicalize_sample(call_strs: Optional[Sequence[str]], backend: str) -> List[CanonicalAction]:
    """Canonicalize one sampled completion's decoded call list.
    ``None`` (decode raised) -> [parse_fail]; empty list -> [] (a no-call sample)."""
    if call_strs is None:
        return [parse_fail(backend)]
    return [canonicalize(c, backend) for c in call_strs]


def signature(action: CanonicalAction, resolution: int) -> str:
    if resolution == 1:
        return action.tool
    if resolution == 2:
        return action.op
    if resolution == 3:
        return f"{action.op}|{action.target if action.target is not None else '-'}"
    if resolution == 4:
        return (f"{action.op}|{action.target if action.target is not None else '-'}"
                f"|{action.content_fp if action.content_fp is not None else '-'}")
    raise ValueError(f"resolution must be in {RESOLUTIONS}, got {resolution}")


def sample_signature(actions: Sequence[CanonicalAction], resolution: int) -> str:
    """Signature of one whole sampled completion: order-insensitive join of its
    per-call signatures (argument/call ordering must not create disagreement)."""
    if not actions:
        return "no_call"
    return ";".join(sorted(signature(a, resolution) for a in actions))


def _entropy_nats(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


def vote_features(actions_per_sample: Sequence[Sequence[CanonicalAction]],
                  resolution: int) -> Dict[str, object]:
    """Vote-distribution features over N samples at one resolution.

    Sample 0 is by convention the PRIMARY completion (the one the agent loop
    actually uses); it is included in the vote.
    """
    n = len(actions_per_sample)
    sigs = [sample_signature(a, resolution) for a in actions_per_sample]
    counts: Dict[str, int] = {}
    for s in sigs:
        counts[s] = counts.get(s, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top1 = ordered[0][1] if ordered else 0
    top2 = ordered[1][1] if len(ordered) > 1 else 0
    n_unique = len(counts)
    h = _entropy_nats(list(counts.values()))
    h_norm = h / math.log(n_unique) if n_unique > 1 else 0.0
    modal_sig = ordered[0][0] if ordered else "no_call"
    return {
        "h_act": h,
        "h_act_norm": h_norm,
        "modal_share": top1 / n if n else None,
        "vote_margin": (top1 - top2) / n if n else None,
        "n_unique": n_unique,
        "n_samples": n,
        "n_no_call": sum(1 for s in sigs if s == "no_call"),
        "n_parse_fail": sum(
            1 for a in actions_per_sample if any(x.op == "parse_fail" for x in a)),
        "modal_signature": modal_sig,
        "primary_signature": sigs[0] if sigs else None,
        "primary_matches_modal": bool(sigs) and counts.get(sigs[0], 0) == top1,
    }


def vote_features_all(actions_per_sample: Sequence[Sequence[CanonicalAction]]
                      ) -> Dict[str, object]:
    """Features at all four resolutions, keys suffixed _tool/_op/_target/_full.
    Signature strings are kept only at R3 (the policy-relevant resolution)."""
    out: Dict[str, object] = {}
    for r in RESOLUTIONS:
        tag = RESOLUTION_TAGS[r]
        f = vote_features(actions_per_sample, r)
        for k, v in f.items():
            if k in ("modal_signature", "primary_signature") and r != 3:
                continue
            out[f"{k}_{tag}"] = v
    return out
