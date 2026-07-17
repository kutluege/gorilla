"""
Deterministic probe generation (Stage 2 / offline replay).
==========================================================

Two probe channels (STAGE0 analysis SS8.1/SS8.3):

* **Item probes** (``generate_item_probes``): template probes derived from a
  *stored* memory item, generated at write time, cached on ``MemoryItem.probes``
  and persisted to the ``<scenario>_gov_state.json`` sidecar. Consumed by the
  dH_neighbor diagnostic (re-score a neighbor's own probes with vs without a
  provisional candidate) and by the offline replay instrument.

* **Decision probes** (``generate_decision_probes``): template probes derived
  from the *current user turn text* and its extraction fields ONLY. Hard
  anti-circularity invariant (SS4.1 / SS8.1): the model-generated candidate text /
  args never enter decision-probe text -- this function does not even accept
  them. Decision probes drive Stage 2's live min-margin rule.

The paraphrase channel (``GOV_PROBE_PARAPHRASE=1``, a separate small non-Qwen
CPU paraphraser) is deliberately a stub in Plan 1 (risk R5); with the default
``GOV_PROBE_PARAPHRASE=0`` generation is template-only and fully deterministic:
same input -> same probes, byte for byte.

Pure logic, no BFCL imports, no model calls.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

_STOPWORDS = frozenset(
    "a an the is are was were be been being am i you he she it we they your my mine "
    "their his her its our do does did don doesn didn not no yes to of in on at for "
    "with and or but that this these those there have has had as by from about so "
    "very just also user likes like s t re ve ll d m what know".split()
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def content_tokens_ordered(text: str, limit: Optional[int] = None) -> List[str]:
    """Content tokens in first-occurrence order (deterministic, deduped).
    Unlike governance_filter._content_tokens this preserves order -- probe text
    should read like a query, and salience follows source order."""
    seen = []
    for t in re.findall(r"[a-z0-9]+", text.lower().replace("_", " ")):
        if len(t) > 2 and t not in _STOPWORDS and t not in seen:
            seen.append(t)
            if limit is not None and len(seen) >= limit:
                break
    return seen


@dataclass
class Probe:
    text: str
    channel: str  # "template:<name>" | "paraphrase:<model>"
    source: str  # "item_text" | "user_text" | "stored_text"

    def to_dict(self) -> dict:
        return {"text": self.text, "channel": self.channel, "source": self.source}


@dataclass
class ProbeConfig:
    paraphrase: bool = False  # GOV_PROBE_PARAPHRASE; template-only when False
    probes_n: int = 4  # GOV_S2_PROBES_N: cap on decision probes

    @classmethod
    def from_env(cls) -> "ProbeConfig":
        return cls(
            paraphrase=_env_flag("GOV_PROBE_PARAPHRASE", False),
            probes_n=int(os.getenv("GOV_S2_PROBES_N", "4")),
        )


_PARAPHRASE_WARNED = False


def _paraphrase_stub(text: str, cfg: ProbeConfig) -> List[Probe]:
    """Plan-1 stub: the paraphrase channel needs a separate small CPU model of a
    non-Qwen family (SS8.1 channel b); wiring it is Plan-2-era work (risk R5).
    Returns no probes; warns once so an enabled flag is never silently a no-op."""
    global _PARAPHRASE_WARNED
    if cfg.paraphrase and not _PARAPHRASE_WARNED:
        _PARAPHRASE_WARNED = True
        print(
            "[probe_gen] WARNING: GOV_PROBE_PARAPHRASE=1 but the paraphrase "
            "channel is a Plan-1 stub; falling back to template-only probes."
        )
    return []


def generate_item_probes(
    backend: str,
    text: str,
    ref: Optional[str] = None,
    source: str = "item_text",
    cfg: Optional[ProbeConfig] = None,
) -> List[Probe]:
    """Template probes for a *stored* item, matched to the backend's retrieval
    reality: KV retrieval scores key names only, so KV probes are built from the
    key (``ref``); Vector retrieval scores the stored text.

    ``source`` is provenance: "item_text" when generated at write time from the
    live write, "stored_text" when regenerated later from a rehydrated snapshot
    (circularity-weaker: diagnostics only, never accept/reject -- SS8.3).
    """
    cfg = cfg or ProbeConfig()
    probes: List[Probe] = []
    if backend == "kv":
        key_words = str(ref or "").replace("_", " ").strip()
        if not key_words:
            # Fall back to the composite's key half ("key words: value").
            key_words = text.split(":", 1)[0].strip()
        value_part = text.split(":", 1)[1] if ":" in text else text
        probes.append(Probe(key_words, "template:key", source))
        probes.append(Probe(f"what is {key_words}", "template:question", source))
        value_toks = content_tokens_ordered(value_part, limit=3)
        if value_toks:
            probes.append(
                Probe(f"{key_words} {' '.join(value_toks)}", "template:key_value", source)
            )
    else:
        probes.append(Probe(text, "template:identity", source))
        focus = content_tokens_ordered(text, limit=4)
        if focus:
            probes.append(
                Probe(f"what do you know about {' '.join(focus)}", "template:question", source)
            )
        keywords = content_tokens_ordered(text, limit=8)
        if keywords:
            probes.append(Probe(" ".join(keywords), "template:keywords", source))
    probes.extend(_paraphrase_stub(text, cfg))
    return _dedupe(probes)


def generate_decision_probes(
    user_text: str,
    verbatim_values: Optional[List[str]] = None,
    cfg: Optional[ProbeConfig] = None,
) -> List[Probe]:
    """Template probes for the live Stage 2 decision, from the user's own turn
    text and its extracted values ONLY (anti-circularity: candidate text/args are
    not parameters of this function by design)."""
    cfg = cfg or ProbeConfig()
    user_text = (user_text or "").strip()
    if not user_text:
        return []
    probes: List[Probe] = [Probe(user_text, "template:user_identity", "user_text")]
    for val in (verbatim_values or [])[:2]:
        probes.append(Probe(f"what is {val}", "template:user_value_question", "user_text"))
    keywords = content_tokens_ordered(user_text, limit=8)
    if keywords:
        probes.append(Probe(" ".join(keywords), "template:user_keywords", "user_text"))
    probes.extend(_paraphrase_stub(user_text, cfg))
    return _dedupe(probes)[: cfg.probes_n]


def _dedupe(probes: List[Probe]) -> List[Probe]:
    seen = set()
    out = []
    for p in probes:
        key = p.text.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out
