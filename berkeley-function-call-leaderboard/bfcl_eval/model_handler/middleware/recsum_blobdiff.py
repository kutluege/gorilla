"""
rec_sum NLI blob-diff surrogate  --  Plan 3 Step 4 (G8, SS5.6).

STANDALONE by design: rec_sum has no archive layer and no search, so retrieval
entropy does not apply, and the thesis explicitly declares rec_sum governance
OUT OF SCOPE. This module exists as the specified surrogate mechanism with
offline tests only -- it is NOT wired into any registry arm
(qwen_gov.GOVERNED_BACKENDS stays {"kv", "vector"}).

Mechanism: for a destructive blob operation (replace / clear / whole-blob
update), split the OLD blob into propositions (sentences) and ask the NLI
scorer whether each is still entailed by the NEW blob. Propositions that are
not entailed are "lost"; the verdict then rejects the operation and suggests
append/partial-update instead.
"""

import re
from dataclasses import dataclass, field
from typing import List

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_MIN_PROP_LEN = 15


def split_propositions(blob: str) -> List[str]:
    """Sentence-level propositions of a memory blob (>= 15 chars, deduped)."""
    seen, out = set(), []
    for raw in _SENT_SPLIT.split(str(blob or "")):
        s = raw.strip()
        key = s.lower()
        if len(s) >= _MIN_PROP_LEN and key not in seen:
            seen.add(key)
            out.append(s)
    return out


@dataclass
class BlobDiffVerdict:
    ok: bool  # True -> operation may proceed (nothing important lost)
    lost: List[str] = field(default_factory=list)
    checked: int = 0
    suggestion: str = ""

    def to_log(self) -> dict:
        return {"ok": self.ok, "lost": self.lost, "checked": self.checked,
                "suggestion": self.suggestion}


def blobdiff_verdict(old_blob: str, new_blob: str, nli_scorer,
                     tau_entail: float = 0.75) -> BlobDiffVerdict:
    """Are the old blob's propositions still entailed by the new blob?

    scorer.probs_batch(pairs) -> [(contradiction, neutral, entailment), ...],
    premise = new blob, hypothesis = old proposition. A clear (empty new blob)
    entails nothing, so every proposition of a non-empty old blob is lost.
    No scorer -> conservative: reject (nothing verifiable).
    """
    props = split_propositions(old_blob)
    if not props:
        return BlobDiffVerdict(ok=True, checked=0)
    if not str(new_blob or "").strip():
        return BlobDiffVerdict(
            ok=False, lost=props, checked=len(props),
            suggestion="clear rejected: old content is not preserved anywhere; "
                       "append or partial-update instead",
        )
    if nli_scorer is None:
        return BlobDiffVerdict(
            ok=False, lost=props, checked=len(props),
            suggestion="no NLI scorer available; cannot verify preservation",
        )
    probs = nli_scorer.probs_batch([(str(new_blob), p) for p in props])
    lost = [p for p, pr in zip(props, probs) if pr[2] < tau_entail]
    if lost:
        return BlobDiffVerdict(
            ok=False, lost=lost, checked=len(props),
            suggestion="destructive update loses propositions; append or "
                       "partial-update the missing content instead",
        )
    return BlobDiffVerdict(ok=True, checked=len(props))
