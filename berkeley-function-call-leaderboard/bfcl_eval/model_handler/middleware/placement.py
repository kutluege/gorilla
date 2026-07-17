"""
SS5 placement / eviction / destructive-operation protection  --  Plan 3 Step 4.

Pure logic consumed by GovernanceSession (no BFCL imports, no model calls
except the injected NLI scorer for the last-copy check). Everything here is
gated by ``GOV_P_ENABLED`` (default 0) + ``GOV_P_SHADOW`` (default 1) in the
session -- with the flag off, the write path is byte-identical to Plan 2.

Mechanisms (brief G6/G7):

* **Category routing (G6b).** There is no fact-extraction prompt in this
  pipeline (the model emits tool calls directly), so the brief's "one added
  prompt field" is realized as a deterministic middleware classifier over the
  candidate text: ``identity`` facts route to core, ``event_detail`` facts
  route straight to archival. No fine-tuning, no model call.

* **Value formula (G6c).** ``value(m_i) = w1*(1 - max_{j!=i} sim(m'_i, m'_j))
  + w2*(1/(1+dt))`` in the WHITENED space (the same L2-normalized ABTT
  embeddings Stage 0 uses), dt = current step - turn_written. No
  access-frequency term. The lowest-value core item is the eviction victim.

* **Atomic move (G6b/R5).** Core-full moves expand ONE call into the pair
  ``archival_memory_add(victim)`` BEFORE ``core_memory_remove(victim)`` --
  archive-add strictly precedes removal, and the expansion is only planned
  when the mirror preflights the archive-add as accepting.

* **Verbatim final check (G6a).** An accepted write whose extracted verbatim
  values do not all appear in the outgoing key/value is rewritten to append
  the missing values (length-capped).

* **Last-copy protection (G6d).** Before an archival eviction/removal, an NLI
  check asks whether the victim's content is entailed by any remaining entry;
  if nowhere derivable, the deletion is refused and a
  ``critical_information_loss`` risk record is logged.

* **Destructive guard (G7).** ``*_memory_clear`` never passes: it is decoy-
  rewritten with a synthetic success. A bare ``core_memory_remove`` becomes
  the archive-then-remove pair; if archiving is impossible the remove is
  blocked (decoy + synthetic) -- content retention wins (Lyapunov rationale:
  while the archive stays reachable, S is monotone).
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

IDENTITY_PATTERNS = (
    "name", "age", "user", "prefer", "favorite", "favourite", "allerg",
    "birthday", "born", "pronoun", "email", "phone", "address", "lives",
    "live in", "occupation", "job", "diet", "language",
)


def classify_category(text: str) -> str:
    """'identity' -> core;  'event_detail' -> archival.  Deterministic."""
    low = str(text).lower().replace("_", " ")
    return "identity" if any(p in low for p in IDENTITY_PATTERNS) else "event_detail"


# ---------------------------------------------------------------------------
# G6c value formula (whitened space)
# ---------------------------------------------------------------------------

def value_scores(items: list, current_step: int, w1: float, w2: float) -> List[float]:
    """value(m_i) = w1*(1 - max_{j!=i} cos(m'_i, m'_j)) + w2*(1/(1+dt)).

    Embeddings are the L2-normalized whitened vectors already on MemoryItem,
    so cosine is a dot product. A single item has uniqueness 1.0. Items
    rehydrated from previous conversations carry turn_written=-1.
    """
    n = len(items)
    if n == 0:
        return []
    M = np.stack([it.emb_whitened for it in items])
    sims = M @ M.T
    np.fill_diagonal(sims, -np.inf)
    out = []
    for i, it in enumerate(items):
        max_sim = float(sims[i].max()) if n > 1 else 0.0
        uniqueness = 1.0 - max(0.0, max_sim)
        dt = max(0, current_step - it.turn_written)
        out.append(w1 * uniqueness + w2 * (1.0 / (1.0 + dt)))
    return out


def pick_eviction_victim(items: list, current_step: int, w1: float, w2: float):
    """Lowest-value item, deterministic tie-break by ref. None if empty."""
    if not items:
        return None
    scores = value_scores(items, current_step, w1, w2)
    return min(zip(scores, items), key=lambda t: (t[0], t[1].ref))[1]


# ---------------------------------------------------------------------------
# Call-string builders (KV composites carry 'key words: value'; the raw value
# is recovered as the substring after the first ': ').
# ---------------------------------------------------------------------------

def kv_value_of(item) -> str:
    return item.text.split(": ", 1)[1] if ": " in item.text else item.text


def archive_call_for(backend: str, item) -> str:
    if backend == "kv":
        return f"archival_memory_add(key={item.ref!r}, value={kv_value_of(item)!r})"
    return f"archival_memory_add(text={item.text!r})"


def remove_call_for(backend: str, tier: str, item) -> str:
    if backend == "kv":
        return f"{tier}_memory_remove(key={item.ref!r})"
    return f"{tier}_memory_remove(vec_id={int(item.ref)})"


def to_archival_call(candidate) -> Optional[str]:
    """Rewrite a core add to the archival tier (G6b event_detail routing)."""
    if candidate.kind != "add" or candidate.tier != "core":
        return None
    if candidate.backend == "kv":
        return (f"archival_memory_add(key={str(candidate.args['key'])!r}, "
                f"value={str(candidate.args['value'])!r})")
    return f"archival_memory_add(text={str(candidate.args['text'])!r})"


def archive_preflight_ok(backend: str, item, cache) -> bool:
    """Mirror-predicted acceptance of archive_call_for(item) (R5: the move is
    only planned when the archive-add will succeed)."""
    tier_items = cache.items["archival"]
    if len(tier_items) >= cache.capacity("archival"):
        return False
    if backend == "kv":
        if item.ref in tier_items:
            return False
        return len(kv_value_of(item)) <= cache.max_entry_length("archival")
    return len(item.text) <= cache.max_entry_length("archival")


# ---------------------------------------------------------------------------
# G6a verbatim final check
# ---------------------------------------------------------------------------

def verbatim_final_rewrite(candidate, verbatim_values: list, normalize_fn,
                           max_len: int) -> Optional[tuple]:
    """(rewritten_call, missing_values) if some extracted verbatim value is
    absent from the outgoing key+value text; None when nothing to fix or the
    appended form would exceed the tier's length cap."""
    if not verbatim_values:
        return None
    blob = normalize_fn(candidate.text)
    missing = [v for v in verbatim_values if normalize_fn(str(v)) not in blob]
    if not missing:
        return None
    suffix = " (" + ", ".join(str(m) for m in missing) + ")"
    if candidate.backend == "kv":
        value = str(candidate.args["value"]) + suffix
        if len(value) > max_len:
            return None
        op = f"{candidate.tier}_memory_{'add' if candidate.kind == 'add' else 'replace'}"
        return (f"{op}(key={str(candidate.args['key'])!r}, value={value!r})", missing)
    if candidate.kind == "add":
        text = str(candidate.args["text"]) + suffix
        if len(text) > max_len:
            return None
        return (f"{candidate.tier}_memory_add(text={text!r})", missing)
    text = str(candidate.args["new_text"]) + suffix
    if len(text) > max_len:
        return None
    return (f"{candidate.tier}_memory_update(vec_id={int(candidate.args['vec_id'])}, "
            f"new_text={text!r})", missing)


# ---------------------------------------------------------------------------
# G6d last-copy protection
# ---------------------------------------------------------------------------

def last_copy_entailed(victim_text: str, remaining_texts: list, nli_scorer,
                       tau_entail: float) -> bool:
    """True iff some remaining entry entails the victim's content (safe to
    delete). Empty remainder or no scorer -> False (NOT derivable -> protect)."""
    if not remaining_texts or nli_scorer is None:
        return False
    pairs = [(t, victim_text) for t in remaining_texts]
    probs = nli_scorer.probs_batch(pairs)
    return any(p[2] >= tau_entail for p in probs)


# ---------------------------------------------------------------------------
# G7 destructive-op planning (pure decision; the session applies it)
# ---------------------------------------------------------------------------

@dataclass
class DestructivePlan:
    action: str  # "block_clear" | "archive_then_remove" | "block_remove_no_archive"
    #             | "block_remove_last_copy" | "allow_remove_derivable" | "pass"
    expansion: Optional[list] = None  # calls to insert BEFORE the original
    synthetic: Optional[str] = None  # synthetic success when blocking (NOOP)
    risk: Optional[str] = None


def plan_destructive(backend: str, kind: str, tier: str, ref: Optional[str],
                     cache, nli_scorer, tau_entail: float,
                     synthetic_fn) -> DestructivePlan:
    """Decide what happens to a remove/clear. `synthetic_fn(kind, tier, ref)`
    builds the backend-exact success string for a blocked op."""
    if kind == "clear":
        return DestructivePlan("block_clear", synthetic=synthetic_fn("clear", tier, None))
    # kind == "remove"
    item = cache.items[tier].get(str(ref)) if ref is not None else None
    if item is None:
        return DestructivePlan("pass")  # unknown ref: let the backend error naturally
    if tier == "core":
        if archive_preflight_ok(backend, item, cache):
            return DestructivePlan(
                "archive_then_remove",
                expansion=[archive_call_for(backend, item)],
            )
        return DestructivePlan(
            "block_remove_no_archive",
            synthetic=synthetic_fn("remove", tier, ref),
            risk="archive_unavailable",
        )
    # archival remove: last-copy protection
    remaining = [it.text for it in cache.all_items() if it is not item]
    if last_copy_entailed(item.text, remaining, nli_scorer, tau_entail):
        return DestructivePlan("allow_remove_derivable")
    return DestructivePlan(
        "block_remove_last_copy",
        synthetic=synthetic_fn("remove", tier, ref),
        risk="critical_information_loss",
    )
