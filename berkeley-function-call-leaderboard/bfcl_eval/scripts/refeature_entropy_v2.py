"""
Entropy-v2 offline re-featurization (post-G1 redefinition, 2026-07-22).

Recomputes the redefined uncertainty signals for every ESCALATED gov2 decision
of a harvest log, from chain-carry decision-time store state plus the probe
texts logged in ``s1_me.per_probe`` (bit-faithful to what the live stage saw):

  dHz_self      -- mean over candidate-locating probes of the change in
                   z-scored score entropy (affine-invariant) on provisional
                   insertion.
  dHz_neighbor  -- same, over the decision-time neighbors' own item probes
                   (neighbors = the refs the live stage evaluated).
  svn_pre       -- von Neumann entropy of the candidate's top-m retrieval
                   neighborhood (embedding-layer Gram spectrum), pre-insertion.
  dSvn          -- S_vn(post, incl. candidate) - S_vn(pre): spectrum collapse
                   (duplicates) vs expansion (novelty).
  vendi_ratio   -- exp(S_post)/exp(S_pre): multiplicative effective-rank change.

Output: features_v2.jsonl keyed by candidate_id, consumed by
``calibrate_margin_entropy.py --extra-features`` for the pre-registered
nested-model test (dAUC(entropy_v2 | geometry+margin) + NC shuffle).

Run:
  python bfcl_eval/scripts/refeature_entropy_v2.py \
      --gov-logs "gov_logs/me_harvest/rep0*_v1_shadow" \
      --out gov_logs/me_harvest/features_v2.jsonl
"""

import argparse
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bfcl_eval.model_handler.middleware.entropy_metrics import (  # noqa: E402
    effective_rank,
    vn_entropy,
    zscore_entropy,
)
from bfcl_eval.model_handler.middleware.probe_gen import generate_item_probes  # noqa: E402
from bfcl_eval.model_handler.middleware.retrieval_sim import (  # noqa: E402
    default_encode,
    simulate_kv,
    simulate_vector,
)
from bfcl_eval.scripts.label_outcomes import walk_log  # noqa: E402

# Locating channels: mirror admission_policy exactly.
_KV_LOCATE = ("template:key", "template:key_value")
_VEC_LOCATE_PREFIXES = ("template:user_identity", "template:user_keywords", "paraphrase:")

TOP_M = 5


def scores_of(backend, corpus, probe, k=5):
    if backend == "kv":
        return [s for s, _ in simulate_kv(corpus, probe, k=k)]
    return [s for s, _ in simulate_vector(corpus, probe, k=k)]


def top_texts(backend, store, corpus, probe, cand_text=None, k=TOP_M):
    """Texts of the top-k retrieved items; for the post corpus the appended
    candidate entry maps to cand_text."""
    refs = list(store.keys())
    if backend == "kv":
        ranked = simulate_kv(corpus, probe, k=k)
        out = []
        for _, key in ranked:
            if key in store:
                out.append(store[key])
            elif cand_text is not None:
                out.append(cand_text)
        return out
    ranked = simulate_vector(corpus, probe, k=k)
    out = []
    for _, idx in ranked:
        if idx < len(refs):
            out.append(corpus[idx])
        elif cand_text is not None:
            out.append(cand_text)
    return out


def refeature_one(rec, tiers):
    backend, tier = rec["backend"], rec["tier"]
    store = tiers.get(tier, {})
    if not store:
        return None
    s1 = rec.get("s1_me") or {}
    per_probe = s1.get("per_probe") or []
    text = rec.get("candidate_text") or ""
    entry = str(rec.get("candidate_ref")) if backend == "kv" else text
    refs = list(store.keys())
    corpus_pre = refs if backend == "kv" else [store[r] for r in refs]
    corpus_post = list(corpus_pre) + [entry]

    # -- dHz_self over the logged locating probes ---------------------------
    dhz_terms = []
    for row in per_probe:
        ch = row.get("channel", "")
        locating = ch in _KV_LOCATE if backend == "kv" else ch.startswith(_VEC_LOCATE_PREFIXES)
        if not locating:
            continue
        p = row.get("probe", "")
        if not p:
            continue
        h_pre = zscore_entropy(scores_of(backend, corpus_pre, p))
        h_post = zscore_entropy(scores_of(backend, corpus_post, p))
        dhz_terms.append(h_post - h_pre)
    dhz_self = sum(dhz_terms) / len(dhz_terms) if dhz_terms else None

    # -- dHz_neighbor over the evaluated neighbors' own probes --------------
    n_terms = []
    for nb in s1.get("dH_neighbor") or []:
        nref = str(nb.get("ref"))
        ntext = store.get(nref)
        if ntext is None:
            continue
        probes = [p.text for p in generate_item_probes(backend, ntext, ref=nref,
                                                       source="stored_text")]
        terms = []
        for p in probes:
            h_pre = zscore_entropy(scores_of(backend, corpus_pre, p))
            h_post = zscore_entropy(scores_of(backend, corpus_post, p))
            terms.append(h_post - h_pre)
        if terms:
            n_terms.append(sum(terms) / len(terms))
    dhz_neighbor = sum(n_terms) / len(n_terms) if n_terms else None

    # -- von Neumann neighborhood spectrum ----------------------------------
    probe_self = entry.replace("_", " ") if backend == "kv" else text
    pre_texts = top_texts(backend, store, corpus_pre, probe_self)
    post_texts = top_texts(backend, store, corpus_post, probe_self, cand_text=text)
    svn_pre = svn_post = dsvn = vendi_ratio = None
    if len(pre_texts) >= 1 and len(post_texts) >= 1:
        svn_pre = vn_entropy(default_encode(pre_texts))
        svn_post = vn_entropy(default_encode(post_texts))
        dsvn = svn_post - svn_pre
        er_pre = effective_rank(svn_pre)
        vendi_ratio = effective_rank(svn_post) / er_pre if er_pre > 0 else None

    return {
        "candidate_id": rec.get("candidate_id"),
        "dHz_self": round(dhz_self, 6) if dhz_self is not None else None,
        "dHz_neighbor": round(dhz_neighbor, 6) if dhz_neighbor is not None else None,
        "svn_pre": round(svn_pre, 6) if svn_pre is not None else None,
        "dSvn": round(dsvn, 6) if dsvn is not None else None,
        "vendi_ratio": round(vendi_ratio, 6) if vendi_ratio is not None else None,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gov-logs", required=True, help="glob of governed arm dirs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dirs = sorted(glob.glob(args.gov_logs))
    if not dirs:
        sys.exit(f"no logs match {args.gov_logs!r}")
    rows, n_esc, n_skipped = [], 0, 0
    for d in dirs:
        log_path = Path(d)
        if log_path.is_dir():
            log_path = log_path / "governance_log.jsonl"
        for rec, tiers, synced in walk_log(log_path):
            if not rec.get("escalated"):
                continue
            n_esc += 1
            if not synced:
                n_skipped += 1
                continue
            row = refeature_one(rec, tiers)
            if row is None or not row["candidate_id"]:
                n_skipped += 1
                continue
            rows.append(row)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    covered = {k: sum(1 for r in rows if r[k] is not None)
               for k in ("dHz_self", "dHz_neighbor", "svn_pre", "dSvn", "vendi_ratio")}
    print(f"[refeature_v2] escalations={n_esc} featured={len(rows)} skipped={n_skipped}")
    print(f"[refeature_v2] coverage: {covered}")
    print(f"[refeature_v2] wrote {out}")


if __name__ == "__main__":
    main()
