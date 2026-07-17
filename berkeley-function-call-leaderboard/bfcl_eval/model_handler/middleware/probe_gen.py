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

The paraphrase channel (``GOV_PROBE_PARAPHRASE=1``, Plan 3 Step 3 / SS4.1b):
2-3 natural paraphrases of the *user's own sentence* from a small CPU seq2seq
model of a **different family than Qwen** (``GOV_PROBE_MODEL``, default
``google/flan-t5-small``) -- the structural loop-break: probes are never
derived from the model-generated key/value, and the paraphraser input is
``user_text`` only. Decoding is beam search (no sampling), so the channel is
deterministic for a fixed model. With the default ``GOV_PROBE_PARAPHRASE=0``,
or when the paraphraser cannot load, generation degrades to template-only,
byte-identical to Plan-2 behavior (warns once, never silently).

NOTE the ``GOV_S2_PROBES_N`` cap (default 4) is applied AFTER templates, so
templates always survive; enable the paraphrase channel together with
``GOV_S2_PROBES_N=5`` to get the brief's 3-5 probe budget.

Pure logic, no BFCL imports; the only model call is the lazily-loaded CPU
paraphraser behind the env flag.
"""

import os
import re
import threading
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
    paraphrase_model: str = "google/flan-t5-small"  # GOV_PROBE_MODEL (non-Qwen)
    paraphrase_n: int = 2  # GOV_PROBE_PARAPHRASE_N: 2-3 per the brief

    @classmethod
    def from_env(cls) -> "ProbeConfig":
        return cls(
            paraphrase=_env_flag("GOV_PROBE_PARAPHRASE", False),
            probes_n=int(os.getenv("GOV_S2_PROBES_N", "4")),
            paraphrase_model=os.getenv("GOV_PROBE_MODEL", "google/flan-t5-small"),
            paraphrase_n=max(1, min(3, int(os.getenv("GOV_PROBE_PARAPHRASE_N", "2")))),
        )


# ---------------------------------------------------------------------------
# Paraphrase channel (SS4.1b): lazy CPU seq2seq singleton, beam decoding.
# ---------------------------------------------------------------------------

_PARA_LOCK = threading.Lock()
_PARA_MODEL = None  # (tokenizer, model, name) once loaded; False after failure
_PARAPHRASE_WARNED = False


def _warn_once(msg: str) -> None:
    global _PARAPHRASE_WARNED
    if not _PARAPHRASE_WARNED:
        _PARAPHRASE_WARNED = True
        print(f"[probe_gen] WARNING: {msg}")


def _get_paraphraser(model_name: str):
    """Thread-locked lazy singleton, mirroring semantic_entropy._get_nli_model.
    Returns (tokenizer, model, name) or None when unavailable (warned once)."""
    global _PARA_MODEL
    if _PARA_MODEL is not None:
        return _PARA_MODEL or None
    with _PARA_LOCK:
        if _PARA_MODEL is not None:
            return _PARA_MODEL or None
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model.to("cpu").eval()
            _PARA_MODEL = (tokenizer, model, model_name)
        except Exception as exc:  # missing package, no cache + offline, etc.
            _PARA_MODEL = False
            _warn_once(
                f"GOV_PROBE_PARAPHRASE=1 but paraphraser '{model_name}' failed "
                f"to load ({exc}); falling back to template-only probes."
            )
            return None
    return _PARA_MODEL


def _hf_paraphrase(text: str, n: int, model_name: str) -> List[str]:
    """Deterministic beam-search paraphrases (no sampling). [] on any failure."""
    loaded = _get_paraphraser(model_name)
    if loaded is None:
        return []
    tokenizer, model, _ = loaded
    try:
        import torch

        inputs = tokenizer(
            f"Paraphrase the sentence: {text}",
            return_tensors="pt", truncation=True, max_length=128,
        )
        with torch.no_grad():
            out = model.generate(
                **inputs,
                num_beams=max(4, n + 2),
                num_return_sequences=n,
                do_sample=False,
                max_new_tokens=48,
            )
        return [tokenizer.decode(seq, skip_special_tokens=True) for seq in out]
    except Exception as exc:
        _warn_once(f"paraphrase generation failed ({exc}); template-only probes.")
        return []


def _paraphrase_probes(
    text: str, cfg: ProbeConfig, generate_fn=None
) -> List[Probe]:
    """Channel (b) probes from the USER sentence only (anti-circularity holds:
    callers pass user_text, never candidate text). Filters out empty/echo
    outputs; degrades to [] whenever the model is unavailable."""
    if not cfg.paraphrase or not text:
        return []
    generate_fn = generate_fn or _hf_paraphrase
    raw = generate_fn(text, cfg.paraphrase_n, cfg.paraphrase_model)
    short_name = cfg.paraphrase_model.rsplit("/", 1)[-1]
    norm_src = text.strip().lower()
    probes = []
    for cand in raw:
        cand = (cand or "").strip()
        if not cand or len(cand) > 300 or cand.lower() == norm_src:
            continue
        probes.append(Probe(cand, f"paraphrase:{short_name}", "user_text"))
        if len(probes) >= cfg.paraphrase_n:
            break
    return probes


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
    # Item probes stay template-only: SS4.1b's paraphrase channel is defined on
    # the USER sentence (decision probes); paraphrasing stored text would relax
    # the provenance story for no diagnostic gain.
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
    # Paraphrases append AFTER templates so the deterministic channel always
    # survives the probes_n cap (fallback safety, R12).
    probes.extend(_paraphrase_probes(user_text, cfg))
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
