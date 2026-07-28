"""
H-Nav Stage 1.1: suppressed-update-aware outcome labeling.

The existing ``label_outcomes.harmful_write`` label is structurally blind to
suppressed updates: a NOOPed candidate never reaches the final store, so no
label computed *from* the final store can ask whether writing it would have
helped. This module answers the counterfactual directly.

Per gov2 decision it reconstructs the write from the log alone --
``candidate_text`` is the new item text, ``label_outcomes.walk_log``'s chain
carry supplies the item text the write replaced -- then builds two worlds:

  S_with     the real final snapshot (the write applied)
  S_without  the same snapshot with this write reverted
             (add -> entry deleted; update/replace -> old text restored)

and evaluates two predicates over the scenario's benchmark questions using the
production grader (see ``hnav_answer_index``):

  necessity  some question whose answer THIS write introduced is answerable in
             S_with and not in S_without -- i.e. the write is the only
             retrievable carrier of a needed fact. Suppressing it costs.
  damage     some other question is answerable in S_without and not in S_with --
             the write destroyed a fact (dropped clause) or buried its carrier.
             Suppressing it would have helped.

"Answerable" = a top-3 retrieved item carries a gold answer. It is a strong
NECESSARY condition for correctness, not a sufficient one; ``hnav_answer_index
--validate`` measures the conversion factor (p_hat ~ 0.42 measured) that the
counterfactual sweep uses to express event counts as expected accuracy.

Two label columns are emitted:

A memory chain rewrites its own items: measured over the harvest, each store ref
receives 3.32 writes on average (max 52), and only the LAST one survives into
the snapshot the questions are answered against. A write whose content was later
overwritten therefore cannot change the final store at all, so to first order it
cannot be harmful to suppress. That is a property of the benchmark, not a
labeling failure, so it gets its own target value rather than being discarded:

``fate``
    terminal    the write's content IS the final content at its identity ->
                the counterfactual is well posed, full labeling applies
    superseded  identity still present, content replaced by a later write
    gone        identity absent from the final store (removed/cleared, or a
                text-identified vector add whose text no longer exists)

``hnav_target`` (arm-invariant, the calibration target)
    must_write        terminal, necessity and not damage
    must_suppress     terminal, damage and not necessity
    may_suppress      terminal, neither (inert write)
    inert_superseded  non-terminal: first-order, suppression changes nothing
    uncertain         not resolvable offline (see UNCERTAIN_REASONS)

For non-terminal writes a lineage diagnostic is still emitted
(``lineage_gold_survives`` / ``lineage_unique_carrier``): the gold this write
introduced may have been carried forward by the later write that replaced it,
in which case suppressing it *might* have broken the lineage. That assumption
cannot be tested offline, so it never enters ``must_write``; it defines the
sensitivity target ``must_write_lineage`` which upper-bounds the headroom.

``hnav_label`` (the audit view requested by the brief) = target x suppressed,
where ``suppressed`` is ``applied == "noop"`` -- NOT ``action``: the shadow
harvest logs ``action="NOOP", applied="none"``, meaning the candidate was
written anyway. That is exactly what makes the harvest a clean counterfactual.

Writes a NEW file; ``outcomes.jsonl`` and the frozen calibration are untouched.

Run (one arm at a time; append per replicate):
  python bfcl_eval/scripts/label_outcomes_hnav.py \
      --gov-log gov_logs/me_harvest/rep01_v1_shadow \
      --result-dir result_gov_me_harvest/rep01/v1_shadow/Qwen_Qwen3-4B-Instruct-2507-FC-GOV \
      --score-dir  score_gov_me_harvest/rep01/v1_shadow/Qwen_Qwen3-4B-Instruct-2507-FC-GOV \
      --arm v1_shadow --replicate 1 \
      --out gov_logs/me_harvest/outcomes_hnav.jsonl --append
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bfcl_eval.scripts.hnav_answer_index import (  # noqa: E402
    TIERS,
    TOP_N,
    carries,
    rank_corpus,
    scenario_questions,
    _tier_corpus,
)
from bfcl_eval.scripts.label_outcomes import (  # noqa: E402
    load_final_stores,
    scenario_of,
    walk_log,
)

UNCERTAIN_REASONS = (
    "chain_desync",       # decision-time store not reconstructible
    "store_empty",        # chain died: nothing in the final snapshot
    "no_gold_for_scenario",
    "old_text_missing",   # update/replace with no reconstructible predecessor
    "both_effects",       # necessity AND damage -- not first-order isolable
)

FATES = ("terminal", "superseded", "gone")

TARGETS = (
    "must_write",
    "must_suppress",
    "may_suppress",
    "inert_superseded",
    "uncertain",
)

# hnav_target x suppressed -> the five-value audit label of the brief.
# A non-terminal write cannot change the final store, so suppressing it is
# correct and writing it is (first-order) harmless.
_LABEL_MAP = {
    ("must_write", True): "harmful_noop",
    ("must_write", False): "correct_add_or_update",
    ("must_suppress", True): "correct_noop",
    ("must_suppress", False): "harmful_add_or_update",
    ("may_suppress", True): "correct_noop",
    ("may_suppress", False): "correct_add_or_update",
    ("inert_superseded", True): "correct_noop",
    ("inert_superseded", False): "correct_add_or_update",
    ("uncertain", True): "uncertain",
    ("uncertain", False): "uncertain",
}


# ---------------------------------------------------------------------------
# Answerability with per-tier reuse
# ---------------------------------------------------------------------------


def tier_carrier(backend: str, store: Dict[str, str], qtext: str, gold, top_n=TOP_N):
    """First top-``top_n`` item of ONE tier that carries ``gold`` (or None).

    Split out from ``hnav_answer_index.answerable`` so the counterfactual only
    re-ranks the tier it actually modified; the untouched tier's verdict is
    reused between S_with and S_without.
    """
    if not store or not qtext or not gold:
        return None
    corpus, refs = _tier_corpus(backend, store)
    for _, identity in rank_corpus(backend, corpus, qtext)[:top_n]:
        ref = identity if backend == "kv" else refs[int(identity)]
        if carries(store.get(ref), gold):
            return ref
    return None


def _answerable(backend, tiers, qtext, gold, cache: dict, tag: str) -> bool:
    """OR over tiers, memoized per (tag, tier, qid) inside one decision."""
    for tier in TIERS:
        key = (tag, tier)
        if key not in cache:
            cache[key] = tier_carrier(backend, tiers.get(tier) or {}, qtext, gold)
        if cache[key] is not None:
            return True
    return False


# ---------------------------------------------------------------------------
# Counterfactual store construction
# ---------------------------------------------------------------------------


def resolve_final_ref(backend, ref, x_new, final_tiers, tier) -> Tuple[Optional[str], str]:
    """Locate this write's entry in the final snapshot -> ``(fid, fate)``.

    KV writes and vector updates log a real ``candidate_ref`` (key name / vector
    id); vector adds log ``None`` and are matched by exact text, the same
    identity convention ``label_outcomes.label_own_probe`` uses.
    """
    store = final_tiers.get(tier) or {}
    if not store:
        return None, "gone"
    if ref is not None:
        fid = str(ref)
        if fid not in store:
            return None, "gone"
        return (fid, "terminal") if store[fid] == x_new else (fid, "superseded")
    matches = [r for r, t in store.items() if t == x_new]
    return (matches[0], "terminal") if matches else (None, "gone")


def build_without(final_tiers, tier, fid, is_add, x_old):
    """S_without: this write reverted, every other item untouched."""
    out = {t: dict(final_tiers.get(t) or {}) for t in TIERS}
    if is_add:
        out[tier].pop(fid, None)
    else:
        out[tier][fid] = x_old
    return out


# ---------------------------------------------------------------------------
# The label
# ---------------------------------------------------------------------------


def label_decision(rec, decision_tiers, synced, final_tiers, exact_damage=False) -> dict:
    """One gov2 decision -> label row (see module docstring)."""
    backend = rec["backend"]
    tier = rec["tier"]
    op = rec.get("op") or ""
    ref = rec.get("candidate_ref")
    x_new = rec.get("candidate_text") or ""
    is_add = op.endswith("_add")
    scenario = scenario_of(rec.get("test_id", ""))
    suppressed = rec.get("applied") == "noop"

    x_old = None
    if ref is not None:
        x_old = (decision_tiers.get(tier) or {}).get(str(ref))

    row = {
        "candidate_id": rec.get("candidate_id"),
        "test_id": rec.get("test_id"),
        "backend": backend,
        "scenario": scenario,
        "tier": tier,
        "op": op,
        "op_family": "add" if is_add else "update_replace",
        "suppressed": int(suppressed),
        "has_old": int(x_old is not None),
        "cand_len": len(x_new),
        # Stage-0 passthrough so the counterfactual sweep needs only this file
        "sim_max": rec.get("sim_max"),
        "r": rec.get("r"),
        "delta": rec.get("delta"),
        "sim_high": rec.get("sim_high"),
        "n_verbatim_misses": len(rec.get("verbatim_misses") or []),
        "n_items": rec.get("n_items"),
        "escalated": int(bool(rec.get("escalated"))),
        "reason_code": rec.get("reason_code"),
        "action": rec.get("action"),
        "applied": rec.get("applied"),
        "preflight_ok": int(bool(rec.get("preflight_ok"))),
    }

    def finish(target, reason=None, **extra):
        row["hnav_target"] = target
        row["uncertain_reason"] = reason
        row["hnav_label"] = _LABEL_MAP[(target, suppressed)]
        for t in TARGETS:
            row[t] = int(target == t)
        row["harmful_noop"] = int(row["hnav_label"] == "harmful_noop")
        row["correct_noop"] = int(row["hnav_label"] == "correct_noop")
        row["harmful_add_or_update"] = int(row["hnav_label"] == "harmful_add_or_update")
        row["correct_add_or_update"] = int(row["hnav_label"] == "correct_add_or_update")
        row["benign_write"] = int(target == "may_suppress" and not suppressed)
        row.update(extra)
        row.setdefault("lineage_gold_survives", 0)
        row.setdefault("lineage_unique_carrier", 0)
        # sensitivity target: strict headroom plus the untestable lineage claim
        row["must_write_lineage"] = int(
            row["must_write"] or row.get("lineage_unique_carrier", 0)
        )
        return row

    if not synced:
        return finish("uncertain", "chain_desync", fate=None)
    if not any(final_tiers.get(t) for t in TIERS):
        return finish("uncertain", "store_empty", fate=None)

    questions = scenario_questions(scenario)
    if not questions:
        return finish("uncertain", "no_gold_for_scenario", fate=None)

    fid, fate = resolve_final_ref(backend, ref, x_new, final_tiers, tier)
    row["fate"] = fate

    # M(d): answers THIS write introduced (old text did not already carry them)
    marginal, other = [], []
    for q in questions:
        if carries(x_new, q["gold"]) and not carries(x_old, q["gold"]):
            marginal.append(q)
        else:
            other.append(q)

    if fate != "terminal":
        # First-order inert: the final store holds someone else's content at
        # this identity, so suppressing this write cannot change any answer.
        # Lineage diagnostic only -- never promoted into must_write.
        survives, unique = 0, 0
        for q in marginal:
            all_carriers = [
                (t, r)
                for t in TIERS
                for r, txt in (final_tiers.get(t) or {}).items()
                if carries(txt, q["gold"])
            ]
            if all_carriers:
                survives = 1
                cache = {}
                if len(all_carriers) == 1 and _answerable(
                    backend, final_tiers, q["text"], q["gold"], cache, "with"
                ):
                    unique = 1
        return finish(
            "inert_superseded",
            None,
            n_marginal_q=len(marginal),
            necessity=0,
            damage=0,
            necessity_qids=[],
            damage_qids=[],
            lineage_gold_survives=survives,
            lineage_unique_carrier=unique,
        )

    if not is_add and x_old is None:
        return finish("uncertain", "old_text_missing")

    without = build_without(final_tiers, tier, fid, is_add, x_old)

    necessity, necessity_qids = False, []
    for q in marginal:
        cache = {}
        if _answerable(backend, final_tiers, q["text"], q["gold"], cache, "with") and not _answerable(
            backend, without, q["text"], q["gold"], cache, "without"
        ):
            necessity = True
            necessity_qids.append(q["qid"])

    # damage: only two mechanisms can flip a non-marginal question the other
    # way -- the write dropped a clause that carried the answer, or the written
    # item itself entered the question's top-3 and displaced the true carrier.
    damage, damage_qids = False, []
    for q in other:
        if not exact_damage:
            dropped = carries(x_old, q["gold"]) and not carries(x_new, q["gold"])
            if not dropped:
                store = final_tiers.get(tier) or {}
                corpus, refs = _tier_corpus(backend, store)
                top = [
                    (i if backend == "kv" else refs[int(i)])
                    for _, i in rank_corpus(backend, corpus, q["text"])[:TOP_N]
                ]
                if fid not in top:
                    continue
        cache = {}
        if _answerable(backend, without, q["text"], q["gold"], cache, "without") and not _answerable(
            backend, final_tiers, q["text"], q["gold"], cache, "with"
        ):
            damage = True
            damage_qids.append(q["qid"])

    extra = {
        "n_marginal_q": len(marginal),
        "necessity": int(necessity),
        "damage": int(damage),
        "necessity_qids": necessity_qids,
        "damage_qids": damage_qids,
    }
    if necessity and damage:
        return finish("uncertain", "both_effects", **extra)
    if necessity:
        return finish("must_write", None, **extra)
    if damage:
        return finish("must_suppress", None, **extra)
    return finish("may_suppress", None, **extra)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gov-log", required=True, help="gov2 arm log dir (or .jsonl)")
    ap.add_argument("--result-dir", required=True, help="governed arm result dir (model level)")
    ap.add_argument("--score-dir", default=None, help="unused by the label; recorded for provenance")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--replicate", default=None)
    ap.add_argument("--exact-damage", action="store_true",
                    help="skip the damage prefilter (slow; measures its cost)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    log_path = Path(args.gov_log)
    if log_path.is_dir():
        log_path = log_path / "governance_log.jsonl"
    arm = args.arm or log_path.parent.name

    finals = {b: load_final_stores(Path(args.result_dir), b) for b in ("kv", "vector")}

    rows = []
    targets, labels, fates, why = (defaultdict(int) for _ in range(4))
    for rec, decision_tiers, synced in walk_log(log_path):
        backend = rec["backend"]
        scenario = scenario_of(rec.get("test_id", ""))
        final_tiers = finals.get(backend, {}).get(scenario) or {"core": {}, "archival": {}}
        row = label_decision(rec, decision_tiers, synced, final_tiers,
                             exact_damage=args.exact_damage)
        row["arm"] = arm
        row["replicate"] = args.replicate
        rows.append(row)
        targets[row["hnav_target"]] += 1
        labels[row["hnav_label"]] += 1
        fates[row.get("fate")] += 1
        if row["uncertain_reason"]:
            why[row["uncertain_reason"]] += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a" if args.append else "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lineage = sum(r.get("lineage_unique_carrier", 0) for r in rows)
    print(f"[label_outcomes_hnav] {arm}: {len(rows)} decisions -> {out_path}")
    print("[label_outcomes_hnav]   fate:   " + " ".join(
        f"{f}={fates[f]}" for f in FATES if fates[f]) +
        (f" unresolved={fates[None]}" if fates[None] else ""))
    print("[label_outcomes_hnav]   target: " + " ".join(
        f"{t}={targets[t]}" for t in TARGETS))
    print("[label_outcomes_hnav]   label:  " + " ".join(
        f"{k}={labels[k]}" for k in
        ("harmful_noop", "correct_noop", "harmful_add_or_update",
         "correct_add_or_update", "uncertain")))
    print(f"[label_outcomes_hnav]   lineage_unique_carrier={lineage} "
          f"(sensitivity target must_write_lineage)")
    if why:
        print("[label_outcomes_hnav]   uncertain: " + " ".join(
            f"{k}={v}" for k, v in sorted(why.items())))


if __name__ == "__main__":
    main()
