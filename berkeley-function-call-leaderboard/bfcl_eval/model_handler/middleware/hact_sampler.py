"""Action-side H_act sampling config, logprob features, and record building.

Companion to ``action_space.py`` (canonicalization / vote features) and the
``QwenHactHandler`` (the only runtime caller). Pure helpers -- no I/O here
except the thread-locked JSONL writer, mirroring semantic_entropy.log_record.

Env vars (HACT_* namespace):
    HACT_ENABLED       master switch (default 0)
    HACT_N             total samples incl. primary (default 8; <=1 disables)
    HACT_TEMP          exploration temperature (default 0.7, SE precedent)
    HACT_LOGPROBS      top-k logprobs on the PRIMARY request (default 5; 0 off)
    HACT_GATE_RECALL   1 = sample on every memory turn; 0 (default) = memory
                       prereq turns only (where writes happen; halves cost)
    HACT_POLICY        shadow | random_select | majority | vote_margin_gate |
                       hact_gate | factorized | full        (default shadow)
    HACT_SEED          base seed for per-request seeds (default 20260728)
    HACT_CALIB         path to frozen threshold JSON (Stage 4 policies only)
    HACT_LOG_FILE      hact log filename (default hact_log.jsonl, written
                       into GOV_LOG_DIR by the governance session flush)
    HACT_VERBOSE       print per-decision lines (default 0)
"""

import hashlib
import json
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from bfcl_eval.model_handler.middleware.action_space import (
    CanonicalAction,
    sample_signature,
    vote_features_all,
)

HACT_SCHEMA = "hact1"

_POLICIES = ("shadow", "random_select", "majority", "vote_margin_gate",
             "hact_gate", "factorized", "full")

_LOG_LOCK = threading.Lock()


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class HactConfig:
    enabled: bool = False
    num_samples: int = 8
    temperature: float = 0.7
    logprobs_k: int = 5
    gate_recall: bool = False
    policy: str = "shadow"
    seed_base: int = 20260728
    calib_path: str = ""
    log_file: str = "hact_log.jsonl"
    verbose: bool = False

    @classmethod
    def from_env(cls) -> "HactConfig":
        policy = os.getenv("HACT_POLICY", "shadow").strip().lower()
        if policy not in _POLICIES:
            raise ValueError(f"HACT_POLICY must be one of {_POLICIES}, got {policy!r}")
        return cls(
            enabled=_env_flag("HACT_ENABLED"),
            num_samples=int(os.getenv("HACT_N", "8")),
            temperature=float(os.getenv("HACT_TEMP", "0.7")),
            logprobs_k=int(os.getenv("HACT_LOGPROBS", "5")),
            gate_recall=_env_flag("HACT_GATE_RECALL"),
            policy=policy,
            seed_base=int(os.getenv("HACT_SEED", "20260728")),
            calib_path=os.getenv("HACT_CALIB", ""),
            log_file=os.getenv("HACT_LOG_FILE", "hact_log.jsonl"),
            verbose=_env_flag("HACT_VERBOSE"),
        )


def stable_seed(base: int, test_id: str, turn_counter: int, salt: int = 0) -> int:
    """Deterministic per-request seed: reproducible given (test_id, turn)."""
    h = hashlib.sha256(f"{base}|{test_id}|{turn_counter}|{salt}".encode()).hexdigest()
    return int(h[:8], 16)


# ------------------------------------------------------------------ logprobs

def _token_spans(tokens: Sequence[str]) -> List[tuple]:
    """Character [start, end) span of each token in the concatenated text."""
    spans, pos = [], 0
    for t in tokens:
        spans.append((pos, pos + len(t)))
        pos += len(t)
    return spans


def logprob_features(logprobs_obj, completion_text: str) -> Dict[str, Optional[float]]:
    """Features from a legacy-completions logprobs structure for ONE choice.

    Returns all-None dict if the structure is absent/empty (caller flips its
    degradation flag). Tool-span features locate the first
    ``<tool_call>...</tool_call>`` region by cumulative token offsets --
    deliberately not text_offset, whose base differs between echo modes.
    """
    keys = ("lp_n_tokens", "lp_sum", "lp_mean", "lp_min", "lp_ppl",
            "lp_topk_margin_mean", "lp_topk_margin_min",
            "lp_tool_sum", "lp_tool_mean", "lp_tool_min", "lp_tool_n_tokens")
    out: Dict[str, Optional[float]] = {k: None for k in keys}
    try:
        tokens = list(logprobs_obj.tokens or [])
        tlps = list(logprobs_obj.token_logprobs or [])
    except (AttributeError, TypeError):
        return out
    if not tokens or not tlps:
        return out

    vals = [v for v in tlps if v is not None]
    if not vals:
        return out
    out["lp_n_tokens"] = float(len(vals))
    out["lp_sum"] = float(sum(vals))
    out["lp_mean"] = out["lp_sum"] / len(vals)
    out["lp_min"] = float(min(vals))
    out["lp_ppl"] = math.exp(-out["lp_mean"])

    top = getattr(logprobs_obj, "top_logprobs", None) or []
    margins = []
    for d in top:
        if isinstance(d, dict) and len(d) >= 2:
            vs = sorted(d.values(), reverse=True)
            margins.append(float(vs[0] - vs[1]))
    if margins:
        out["lp_topk_margin_mean"] = sum(margins) / len(margins)
        out["lp_topk_margin_min"] = min(margins)

    start = completion_text.find("<tool_call>")
    if start != -1:
        end = completion_text.find("</tool_call>", start)
        end = end + len("</tool_call>") if end != -1 else len(completion_text)
        spans = _token_spans(tokens)
        tool_vals = [tlps[i] for i, (a, b) in enumerate(spans)
                     if i < len(tlps) and tlps[i] is not None and a < end and b > start]
        if tool_vals:
            out["lp_tool_sum"] = float(sum(tool_vals))
            out["lp_tool_mean"] = out["lp_tool_sum"] / len(tool_vals)
            out["lp_tool_min"] = float(min(tool_vals))
            out["lp_tool_n_tokens"] = float(len(tool_vals))
    return out


# ------------------------------------------------------------------ record

def build_hact_record(
    *,
    test_id: str,
    is_prereq: bool,
    backend: str,
    cfg: HactConfig,
    primary_text: str,
    primary_calls: Optional[Sequence[str]],
    primary_lp_feats: Dict[str, Optional[float]],
    logprobs_supported: Optional[bool],
    sample_texts: Sequence[str],
    sample_calls: Sequence[Optional[Sequence[str]]],
    actions_per_sample: Sequence[Sequence[CanonicalAction]],
    seeds: Sequence[int],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    api_seconds: float,
    policy_record: Optional[dict] = None,
) -> dict:
    """Assemble the full hact1 record. actions_per_sample[0] is the PRIMARY."""
    votes = vote_features_all(actions_per_sample)
    rec = {
        "schema": HACT_SCHEMA,
        "event": "hact",
        "test_id": test_id,
        "is_prereq": bool(is_prereq),
        "backend": backend,
        "sampling_config": {
            "n": cfg.num_samples,
            "exploration_temperature": cfg.temperature,
            "logprobs_k": cfg.logprobs_k,
            "gate_recall": cfg.gate_recall,
            "policy": cfg.policy,
            "seed_base": cfg.seed_base,
            "seeds": list(seeds),
        },
        "primary": {
            "text": primary_text,
            "calls": list(primary_calls) if primary_calls is not None else None,
            "signature_r3": sample_signature(actions_per_sample[0], 3)
            if actions_per_sample else None,
        },
        "samples": [
            {
                "text": (t[:400] if isinstance(t, str) else t),
                "calls": list(c) if c is not None else None,
            }
            for t, c in zip(sample_texts, sample_calls)
        ],
        "votes": votes,
        "logprobs_supported": logprobs_supported,
        "logprob_features": primary_lp_feats,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "api_seconds": round(api_seconds, 3),
    }
    if policy_record is not None:
        rec["policy_record"] = policy_record
    return rec


def log_hact_record(record: dict, log_dir: str, log_file: str) -> None:
    """Thread-locked append, mirroring semantic_entropy.log_record."""
    path = Path(log_dir) / log_file
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with _LOG_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
