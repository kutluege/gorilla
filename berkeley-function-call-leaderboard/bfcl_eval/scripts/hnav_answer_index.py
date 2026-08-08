"""
H-Nav Stage 1: gold-answer index + answerability oracle (plan T1).

The corrected outcome label (``label_outcomes_hnav.py``) has to answer one
question deterministically and offline: *did this memory write carry information
some benchmark question actually needs, and was that information retrievable?*

Both halves are available without a model:

* ``bfcl_eval/data/possible_answer/BFCL_v4_memory.json`` -- 155 rows of
  ``{id, ground_truth[], source}``, 1:1 with the question dataset.
* ``bfcl_eval.eval_checker.agentic_eval.agentic_checker.agentic_checker`` -- the
  **production grader** (word-boundary regex over a ``standardize_string``
  normalization). We reuse it verbatim rather than re-implementing substring
  matching, so "carries the answer" means exactly what the leaderboard means.

Retrievability reuses the backend-faithful simulators in ``retrieval_sim`` at
``top_n = 3`` -- the same constant as ``label_outcomes.TOP_N`` and
``geometry_gate.FLOOR_TOP_N``.

Answerability is a strong NECESSARY condition for a correct answer, not a
sufficient one. ``--validate`` measures the conversion factor empirically by
cross-tabulating answerability against real question-level correctness, and
writes ``p_hat`` for the counterfactual sweep to convert event counts into
expected accuracy. Never assume 1:1.

Run:
  python bfcl_eval/scripts/hnav_answer_index.py --validate \
      --arm harvest/rep01 \
      --result-dir result_gov_me_harvest/rep01/v1_shadow/Qwen_Qwen3-4B-Instruct-2507-FC-GOV \
      --score-dir  score_gov_me_harvest/rep01/v1_shadow/Qwen_Qwen3-4B-Instruct-2507-FC-GOV \
      --out gov_logs/hnav_proxy_validation.json --append
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bfcl_eval.eval_checker.agentic_eval.agentic_checker import (  # noqa: E402
    agentic_checker,
)
from bfcl_eval.model_handler.middleware.retrieval_sim import (  # noqa: E402
    simulate_kv,
    simulate_vector,
)
from bfcl_eval.scripts.label_outcomes import (  # noqa: E402
    correct_ids,
    load_final_stores,
    question_text_for,
    question_texts,
    scenario_of,
)

TOP_N = 3
TOP_K = 5

GOLD_PATH = REPO_ROOT / "bfcl_eval" / "data" / "possible_answer" / "BFCL_v4_memory.json"

TIERS = ("core", "archival")


# ---------------------------------------------------------------------------
# Gold index
# ---------------------------------------------------------------------------

_GOLD: Optional[Dict[str, List[str]]] = None
_BY_SCENARIO: Optional[Dict[str, List[dict]]] = None


def core_qid(qid: str) -> str:
    """memory_kv_12-customer-3 -> memory_12-customer-3 (dataset id form)."""
    return re.sub(r"^memory_(kv|vector)_", "memory_", qid or "")


def gold_index() -> Dict[str, List[str]]:
    """dataset question id -> list of acceptable answer strings."""
    global _GOLD
    if _GOLD is None:
        _GOLD = {}
        with open(GOLD_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                gt = e.get("ground_truth") or []
                if isinstance(gt, str):
                    gt = [gt]
                _GOLD[e["id"]] = [str(g) for g in gt]
    return _GOLD


def scenario_questions(scenario: str) -> List[dict]:
    """All questions of a scenario: ``[{qid, text, gold}]`` (dataset id form).

    Prereq entries carry no ground truth and are excluded -- only answerable
    questions define "necessary information".
    """
    global _BY_SCENARIO
    if _BY_SCENARIO is None:
        _BY_SCENARIO = defaultdict(list)
        texts = question_texts()
        for qid, gold in gold_index().items():
            if not gold:
                continue
            _BY_SCENARIO[scenario_of(qid)].append(
                {"qid": qid, "text": texts.get(qid), "gold": gold}
            )
        for rows in _BY_SCENARIO.values():
            rows.sort(key=lambda r: r["qid"])
    return _BY_SCENARIO.get(scenario, [])


# ---------------------------------------------------------------------------
# Carriage + retrievability
# ---------------------------------------------------------------------------


def carries(text: Optional[str], gold: Sequence[str]) -> bool:
    """Does ``text`` contain one of the acceptable answers, per the production
    grader? Word-boundary matched, so "380 seconds" does NOT carry "38"."""
    if not text or not gold:
        return False
    return bool(agentic_checker(str(text), list(gold)).get("valid"))


_RANK_CACHE: Dict[tuple, list] = {}


def rank_corpus(backend: str, corpus: Sequence[str], probe: str, k: int = TOP_K) -> list:
    """Memoized backend-faithful ranking. Key is the corpus itself (small,
    string-hash-cached by CPython) plus the probe -- the labeller re-ranks the
    same store for many questions and the same question for many decisions."""
    key = (backend, tuple(corpus), probe, k)
    hit = _RANK_CACHE.get(key)
    if hit is None:
        if backend == "kv":
            hit = simulate_kv(list(corpus), probe, k=k)
        else:
            hit = simulate_vector(list(corpus), probe, k=k)
        _RANK_CACHE[key] = hit
    return hit


def _tier_corpus(backend: str, store: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """(corpus fed to the simulator, parallel ref list).

    KV retrieval scores **key names only** (memory_kv._similarity_search); the
    vector backend scores stored texts. Refs stay aligned either way.
    """
    refs = list(store.keys())
    if backend == "kv":
        return refs, refs
    return [store[r] for r in refs], refs


def answerable(
    backend: str,
    tiers: Dict[str, Dict[str, str]],
    qtext: str,
    gold: Sequence[str],
    top_n: int = TOP_N,
    tier_filter: Optional[str] = None,
) -> Tuple[bool, List[Tuple[str, str]]]:
    """Is some item that carries ``gold`` inside the question's top-``top_n``
    retrieval result? Returns ``(answerable, [(tier, ref), ...])``.

    Primary variant ORs ``core`` and ``archival``: the agent can call both
    ``*_memory_key_search`` and ``archival_memory_search``. Per-tier variants
    (``tier_filter``) are reported as sensitivity columns.
    """
    carriers: List[Tuple[str, str]] = []
    if not qtext or not gold:
        return False, carriers
    for tier in TIERS:
        if tier_filter and tier != tier_filter:
            continue
        store = tiers.get(tier) or {}
        if not store:
            continue
        corpus, refs = _tier_corpus(backend, store)
        ranked = rank_corpus(backend, corpus, qtext)
        for _, identity in ranked[:top_n]:
            ref = identity if backend == "kv" else refs[int(identity)]
            if carries(store.get(ref), gold):
                carriers.append((tier, ref))
    return bool(carriers), carriers


def carried_anywhere(
    tiers: Dict[str, Dict[str, str]], gold: Sequence[str]
) -> List[Tuple[str, str]]:
    """Every stored item carrying ``gold``, ignoring retrievability."""
    out = []
    for tier in TIERS:
        for ref, text in (tiers.get(tier) or {}).items():
            if carries(text, gold):
                out.append((tier, ref))
    return out


# ---------------------------------------------------------------------------
# --validate: measure the proxy's conversion factor
# ---------------------------------------------------------------------------


def validate_arm(result_dir: Path, score_dir: Path, backends=("kv", "vector")) -> dict:
    """Cross-tabulate answerability against real correctness.

    Strata per question: ``carried_retrievable`` / ``carried_not_retrievable``
    / ``not_carried``. ``p_hat`` = P(correct | carried & retrievable) -
    P(correct | not carried) is the accuracy-conversion factor used by the
    counterfactual sweep.
    """
    out = {"backends": {}}
    for backend in backends:
        stores = load_final_stores(result_dir, backend)
        ok_ids, attempted = correct_ids(result_dir, score_dir, backend)
        strata = defaultdict(lambda: {"n": 0, "correct": 0})
        tier_strata = defaultdict(lambda: {"n": 0, "correct": 0})
        for qid in sorted(attempted):
            core = core_qid(qid)
            gold = gold_index().get(core)
            qtext = question_text_for(qid)
            if not gold or not qtext:
                continue
            tiers = stores.get(scenario_of(core)) or {"core": {}, "archival": {}}
            all_carriers = carried_anywhere(tiers, gold)
            if not all_carriers:
                stratum = "not_carried"
            else:
                reachable, _ = answerable(backend, tiers, qtext, gold)
                stratum = "carried_retrievable" if reachable else "carried_not_retrievable"
            strata[stratum]["n"] += 1
            strata[stratum]["correct"] += int(qid in ok_ids)
            # tier-conditional stratum: WHERE the gold is carried. Core is
            # auto-dumped into context by add_memory_instruction_system_prompt;
            # archival is only reachable through an explicit read call. The two
            # tiers therefore have radically different conversion factors, and
            # a tier-blind p_hat mis-prices any write landing in archival.
            carrier_tiers = {t for t, _ in all_carriers}
            if not carrier_tiers:
                tstratum = "not_carried"
            elif "core" in carrier_tiers:
                tstratum = "core_carried"
            else:
                tstratum = "archival_only"
            tier_strata[tstratum]["n"] += 1
            tier_strata[tstratum]["correct"] += int(qid in ok_ids)

        def _rows(d):
            return {
                name: {
                    "n": s["n"],
                    "correct": s["correct"],
                    "p_correct": round(s["correct"] / s["n"], 6) if s["n"] else None,
                }
                for name, s in d.items()
            }

        rows = _rows(strata)
        trows = _rows(tier_strata)
        p_hi = rows.get("carried_retrievable", {}).get("p_correct")
        p_lo = rows.get("not_carried", {}).get("p_correct")
        out["backends"][backend] = {
            "strata": rows,
            "by_tier": trows,
            "p_hat": round(p_hi - p_lo, 6) if (p_hi is not None and p_lo is not None) else None,
        }
    return out


def pooled_p_hat(validation: dict) -> Dict[str, Optional[float]]:
    """Per-backend p_hat pooled over every validated arm (count-weighted)."""
    acc = defaultdict(lambda: {"hi_n": 0, "hi_c": 0, "lo_n": 0, "lo_c": 0})
    for arm in validation.get("arms", {}).values():
        for backend, block in arm.get("backends", {}).items():
            s = block.get("strata", {})
            hi = s.get("carried_retrievable") or {}
            lo = s.get("not_carried") or {}
            acc[backend]["hi_n"] += hi.get("n", 0)
            acc[backend]["hi_c"] += hi.get("correct", 0)
            acc[backend]["lo_n"] += lo.get("n", 0)
            acc[backend]["lo_c"] += lo.get("correct", 0)
    out = {}
    for backend, d in acc.items():
        if d["hi_n"] and d["lo_n"]:
            out[backend] = round(d["hi_c"] / d["hi_n"] - d["lo_c"] / d["lo_n"], 6)
        else:
            out[backend] = None
    ns = sum(d["hi_n"] + d["lo_n"] for d in acc.values())
    if ns:
        hi_n = sum(d["hi_n"] for d in acc.values())
        hi_c = sum(d["hi_c"] for d in acc.values())
        lo_n = sum(d["lo_n"] for d in acc.values())
        lo_c = sum(d["lo_c"] for d in acc.values())
        out["pooled"] = (
            round(hi_c / hi_n - lo_c / lo_n, 6) if (hi_n and lo_n) else None
        )
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--validate", action="store_true", help="run proxy validation")
    ap.add_argument("--arm", default=None, help="label for this arm-replicate")
    ap.add_argument("--result-dir", default=None)
    ap.add_argument("--score-dir", default=None)
    ap.add_argument("--backends", default="kv,vector")
    ap.add_argument("--out", default="gov_logs/hnav_proxy_validation.json")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    if not args.validate:
        g = gold_index()
        print(f"[hnav_answer_index] {len(g)} gold rows; "
              f"{len(set(scenario_of(q) for q in g))} scenarios")
        return

    if not (args.result_dir and args.score_dir):
        ap.error("--validate requires --result-dir and --score-dir")

    arm = args.arm or Path(args.result_dir).as_posix()
    backends = tuple(b.strip() for b in args.backends.split(",") if b.strip())
    block = validate_arm(Path(args.result_dir), Path(args.score_dir), backends)

    out_path = Path(args.out)
    doc = {"arms": {}}
    if args.append and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        doc.setdefault("arms", {})
    doc["arms"][arm] = block
    doc["p_hat"] = pooled_p_hat(doc)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)

    for backend, b in block["backends"].items():
        s = b["strata"]
        parts = " ".join(
            f"{k}={v['correct']}/{v['n']}" for k, v in sorted(s.items())
        )
        print(f"[hnav_answer_index] {arm} {backend}: {parts} p_hat={b['p_hat']}")
    print(f"[hnav_answer_index] pooled p_hat={doc['p_hat']} -> {out_path}")


if __name__ == "__main__":
    main()
