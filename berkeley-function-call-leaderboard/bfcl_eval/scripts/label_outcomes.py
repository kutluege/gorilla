"""
Post-hoc outcome labeling for margin-entropy calibration (Plan v2 SS13.2).

Consumes a harvest arm's gov2 governance log + its result/score trees (and the
paired baseline replicate's score tree) and emits ``outcomes.jsonl`` -- one
line per gov2 decision, joined on ``candidate_id`` -- for
``calibrate_margin_entropy.py``. Benchmark questions and ground truth are read
HERE, offline, after the runs; they never touch the runtime feature path
(the calibration pipeline asserts the online-feature whitelist separately).

Labels (SS13.2):
  own_probe_fail      -- median rank of the candidate's own item probes in the
                         FINAL tier store > 3 (absent from the final store
                         counts as fail; ``own_probe_absent`` reported too).
  neighbor_probe_fail -- some neighbor evaluated at decision time (the
                         dH_neighbor list) was top-1 on its own probes in the
                         decision-time store but is not top-1 in the final
                         store (chain-carry replay supplies decision-time
                         state).
  harmful_write       -- chain survived AND >=1 question of the same
                         (backend, scenario) flipped correct->incorrect vs the
                         paired baseline replicate AND that question's
                         simulated retrieval over the final tier store ranks
                         the candidate top-3 (neighborhood touched). None when
                         no baseline pairing is supplied.
  useless_duplicate   -- admitted candidate whose normalized text duplicates a
                         decision-time store item and whose provisional
                         removal changes no other item's top-3 probe ranking.

Run (one governed arm at a time; append per replicate):
  python bfcl_eval/scripts/label_outcomes.py \
      --gov-log gov_logs/me_harvest/rep01_v1_shadow \
      --result-dir result_gov_me_harvest/rep01/v1_shadow/Qwen_Qwen3-4B-Instruct-2507-FC-GOV \
      --score-dir score_gov_me_harvest/rep01/v1_shadow/Qwen_Qwen3-4B-Instruct-2507-FC-GOV \
      --baseline-score-dir score_gov_me_harvest/rep01/baseline/Qwen_Qwen3-4B-Instruct-2507-FC \
      --baseline-result-dir result_gov_me_harvest/rep01/baseline/Qwen_Qwen3-4B-Instruct-2507-FC \
      --out gov_logs/me_harvest/outcomes.jsonl --append
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bfcl_eval.model_handler.middleware.entropy_metrics import (  # noqa: E402
    median,
    rank_self,
    top_identity_set,
)
from bfcl_eval.model_handler.middleware.governance_filter import (  # noqa: E402
    _normalize_for_match,
    kv_composite_text,
)
from bfcl_eval.model_handler.middleware.probe_gen import generate_item_probes  # noqa: E402
from bfcl_eval.model_handler.middleware.retrieval_sim import (  # noqa: E402
    simulate_kv,
    simulate_vector,
)

TOP_N = 3


def scenario_of(test_id: str) -> str:
    parts = (test_id or "").split("-")
    return parts[1] if len(parts) >= 2 else str(test_id)


def rank_in_corpus(backend, corpus, probe, target_identity, k=5):
    ranked = simulate_kv(corpus, probe, k=k) if backend == "kv" else simulate_vector(corpus, probe, k=k)
    return rank_self(ranked, target_identity, k=k), ranked


def item_probe_texts(backend, text, ref):
    return [p.text for p in generate_item_probes(backend, text, ref=ref)]


# ---------------------------------------------------------------------------
# Snapshots / scores / questions
# ---------------------------------------------------------------------------


def load_final_stores(result_dir: Path, backend: str):
    """scenario -> {tier: {ref: text}} from memory_snapshot/<scenario>_final.json."""
    snap_dir = result_dir / "agentic" / "memory" / backend / "memory_snapshot"
    stores = {}
    if not snap_dir.exists():
        return stores
    for f in snap_dir.glob("*_final.json"):
        scenario = f.name[: -len("_final.json")]
        with open(f, "r", encoding="utf-8") as fh:
            snap = json.load(fh)
        tiers = {"core": {}, "archival": {}}
        for tier, key in (("core", "core_memory"), ("archival", "archival_memory")):
            data = snap.get(key, {})
            if backend == "kv":
                for k, v in data.items():
                    tiers[tier][str(k)] = kv_composite_text(k, v)
            else:
                for vid, text in data.get("store", {}).items():
                    tiers[tier][str(int(vid))] = str(text)
        stores[scenario] = tiers
    return stores


def correct_ids(result_dir: Path, score_dir: Path, backend: str):
    """Question-level correctness: attempted result ids minus score-failure ids
    (score files hold header + failures only)."""
    cat = f"BFCL_v4_memory_{backend}"
    res = result_dir / "agentic" / "memory" / backend / f"{cat}_result.json"
    sco = score_dir / "agentic" / "memory" / backend / f"{cat}_score.json"
    attempted = set()
    if res.exists():
        with open(res, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    attempted.add(json.loads(line)["id"])
    failed = set()
    if sco.exists():
        with open(sco, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if i == 0 and "accuracy" in rec:
                    continue
                failed.add(rec["id"])
    return attempted - failed, attempted


_QUESTIONS = None


def question_texts():
    """(backend-agnostic) question id core -> user text; ids in the dataset are
    'memory_N-scenario-M'; per-backend ids prefix 'memory_kv'/'memory_vector'."""
    global _QUESTIONS
    if _QUESTIONS is None:
        _QUESTIONS = {}
        with open(REPO_ROOT / "bfcl_eval" / "data" / "BFCL_v4_memory.json", "r",
                  encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                try:
                    text = e["question"][0][0]["content"]
                except (KeyError, IndexError, TypeError):
                    continue
                _QUESTIONS[e["id"]] = text
    return _QUESTIONS


def question_text_for(qid: str):
    # memory_kv_12-customer-3 -> memory_12-customer-3
    core = re.sub(r"^memory_(kv|vector)_", "memory_", qid)
    return question_texts().get(core)


# ---------------------------------------------------------------------------
# Chain-carry over the gov log (decision-time stores)
# ---------------------------------------------------------------------------


def walk_log(log_path: Path):
    """Yield (record, decision_time_tiers) for every gov2 decision, maintaining
    per-(backend, scenario) chain state from rehydrate/observe events."""
    chains = defaultdict(lambda: {"core": {}, "archival": {}})
    synced = defaultdict(bool)
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = rec.get("event")
            key = (rec.get("backend"), scenario_of(rec.get("test_id", "")))
            if event == "rehydrate":
                n = int(rec.get("items_loaded", 0))
                if n == 0:
                    chains[key] = {"core": {}, "archival": {}}
                    synced[key] = True
                else:
                    total = sum(len(t) for t in chains[key].values())
                    synced[key] = synced[key] and total == n
            elif event == "observe":
                tier = rec.get("tier")
                if tier in ("core", "archival"):
                    chains[key][tier][str(rec.get("ref"))] = rec.get("text", "")
            elif event == "observe_remove":
                tier = rec.get("tier")
                if tier in ("core", "archival"):
                    chains[key][tier].pop(str(rec.get("ref")), None)
            elif event == "observe_clear":
                tier = rec.get("tier")
                if tier in ("core", "archival"):
                    chains[key][tier] = {}
            elif event == "decision" and rec.get("schema") == "gov2":
                snapshot = {t: dict(chains[key][t]) for t in ("core", "archival")}
                yield rec, snapshot, synced[key]


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def candidate_identity(rec):
    backend = rec["backend"]
    if backend == "kv":
        return str(rec.get("candidate_ref"))
    return rec.get("candidate_text") or ""


def label_own_probe(rec, final_tiers):
    backend, tier = rec["backend"], rec["tier"]
    store = final_tiers.get(tier, {})
    text = rec.get("candidate_text") or ""
    ref = rec.get("candidate_ref")
    if backend == "kv":
        present = str(ref) in store
        corpus = list(store.keys())
        target = str(ref)
    else:
        matches = [r for r, t in store.items() if t == text]
        present = bool(matches)
        corpus = [store[r] for r in store]
        target = list(store.keys()).index(matches[0]) if matches else None
    if not present or not corpus:
        return True, True  # fail, absent
    probes = item_probe_texts(backend, text, ref)
    if not probes:
        return False, False
    if backend == "kv":
        ranks = [rank_in_corpus("kv", corpus, p, target)[0] for p in probes]
    else:
        refs = list(store.keys())
        idx = refs.index(matches[0])
        ranks = [rank_in_corpus("vector", corpus, p, idx)[0] for p in probes]
    med = median(ranks)
    return (med is not None and med > TOP_N), False


def label_neighbor_probe(rec, decision_tiers, final_tiers):
    backend, tier = rec["backend"], rec["tier"]
    s1 = rec.get("s1_me") or {}
    neighbors = [d.get("ref") for d in (s1.get("dH_neighbor") or []) if d.get("ref") is not None]
    if not neighbors:
        return False
    dec_store = decision_tiers.get(tier, {})
    fin_store = final_tiers.get(tier, {})
    for nref in neighbors:
        nref = str(nref)
        ntext = dec_store.get(nref)
        if ntext is None:
            continue
        probes = item_probe_texts(backend, ntext, nref)
        if not probes:
            continue
        # top-1 pre-insertion (decision-time store)?
        if backend == "kv":
            dec_corpus = list(dec_store.keys())
            pre_top1 = all(
                rank_in_corpus("kv", dec_corpus, p, nref)[0] == 1 for p in probes
            )
        else:
            dec_refs = list(dec_store.keys())
            dec_corpus = [dec_store[r] for r in dec_refs]
            n_idx = dec_refs.index(nref)
            pre_top1 = all(
                rank_in_corpus("vector", dec_corpus, p, n_idx)[0] == 1 for p in probes
            )
        if not pre_top1:
            continue
        # still top-1 in the final store?
        if backend == "kv":
            fin_corpus = list(fin_store.keys())
            if nref not in fin_store:
                return True
            post_top1 = all(
                rank_in_corpus("kv", fin_corpus, p, nref)[0] == 1 for p in probes
            )
        else:
            fin_refs = list(fin_store.keys())
            fin_corpus = [fin_store[r] for r in fin_refs]
            fin_match = [r for r in fin_refs if fin_store[r] == ntext]
            if not fin_match:
                return True
            f_idx = fin_refs.index(fin_match[0])
            post_top1 = all(
                rank_in_corpus("vector", fin_corpus, p, f_idx)[0] == 1 for p in probes
            )
        if not post_top1:
            return True
    return False


def label_useless_duplicate(rec, decision_tiers):
    backend, tier = rec["backend"], rec["tier"]
    if rec.get("action") not in ("ADD", "ABSTAIN"):
        return False
    store = decision_tiers.get(tier, {})
    if not store:
        return False
    text = rec.get("candidate_text") or ""
    norm = _normalize_for_match(text)
    if not norm or not any(_normalize_for_match(t) == norm for t in store.values()):
        return False
    # duplicate: does removing (i.e. never adding) it change any ranking?
    refs = list(store.keys())
    corpus_wo = refs if backend == "kv" else [store[r] for r in refs]
    entry = str(rec.get("candidate_ref")) if backend == "kv" else text
    corpus_w = list(corpus_wo) + [entry]
    for nref in refs:
        probes = item_probe_texts(backend, store[nref], nref)
        for p in probes:
            if backend == "kv":
                top_wo = top_identity_set(simulate_kv(corpus_wo, p, k=5), TOP_N)
                top_w = [t for t in top_identity_set(simulate_kv(corpus_w, p, k=5), TOP_N)]
            else:
                top_wo = top_identity_set(simulate_vector(corpus_wo, p, k=5), TOP_N)
                top_w = [i for i in top_identity_set(simulate_vector(corpus_w, p, k=5), TOP_N)
                         if i < len(corpus_wo)]  # ignore the candidate itself
            if backend == "kv":
                top_w = [t for t in top_w if t != entry or t in corpus_wo]
            if [t for t in top_w] != [t for t in top_wo][: len(top_w)] and top_w != top_wo:
                return False  # ranking changed -> not useless
    return True


def build_flip_index(result_dir, score_dir, b_result_dir, b_score_dir, backend):
    """(scenario) -> [flipped question ids correct(base)->incorrect(governed)]."""
    gov_ok, gov_att = correct_ids(result_dir, score_dir, backend)
    base_ok, base_att = correct_ids(b_result_dir, b_score_dir, backend)
    flips = defaultdict(list)
    for qid in base_ok & gov_att:
        if qid not in gov_ok:
            flips[scenario_of(qid)].append(qid)
    return flips


def label_harmful(rec, final_tiers, flips):
    tier = rec["tier"]
    backend = rec["backend"]
    store = final_tiers.get(tier, {})
    chain_survived = any(final_tiers.get(t) for t in ("core", "archival"))
    if not chain_survived or not store:
        return False
    scenario = scenario_of(rec.get("test_id", ""))
    target = candidate_identity(rec)
    if backend == "kv":
        corpus = list(store.keys())
        if target not in store:
            return False
    else:
        refs = list(store.keys())
        corpus = [store[r] for r in refs]
        matches = [i for i, r in enumerate(refs) if store[r] == target]
        if not matches:
            return False
        target = matches[0]
    for qid in flips.get(scenario, []):
        qtext = question_text_for(qid)
        if not qtext:
            continue
        rank, _ = rank_in_corpus(backend, corpus, qtext, target)
        if rank <= TOP_N:
            return True
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gov-log", required=True, help="governed arm govlog dir (or .jsonl)")
    ap.add_argument("--result-dir", required=True, help="governed arm result dir (model level)")
    ap.add_argument("--score-dir", required=True)
    ap.add_argument("--baseline-result-dir", default=None)
    ap.add_argument("--baseline-score-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    log_path = Path(args.gov_log)
    if log_path.is_dir():
        log_path = log_path / "governance_log.jsonl"
    result_dir, score_dir = Path(args.result_dir), Path(args.score_dir)
    have_baseline = bool(args.baseline_result_dir and args.baseline_score_dir)

    finals = {b: load_final_stores(result_dir, b) for b in ("kv", "vector")}
    flips = {}
    if have_baseline:
        for b in ("kv", "vector"):
            flips[b] = build_flip_index(
                result_dir, score_dir,
                Path(args.baseline_result_dir), Path(args.baseline_score_dir), b,
            )

    out_rows, counters = [], defaultdict(int)
    for rec, decision_tiers, synced in walk_log(log_path):
        counters["decisions"] += 1
        if not synced:
            counters["desync_skipped"] += 1
            continue
        backend = rec["backend"]
        scenario = scenario_of(rec.get("test_id", ""))
        final_tiers = finals.get(backend, {}).get(scenario, {"core": {}, "archival": {}})
        own_fail, own_absent = label_own_probe(rec, final_tiers)
        row = {
            "candidate_id": rec.get("candidate_id"),
            "backend": backend,
            "scenario": scenario,
            "escalated": bool(rec.get("escalated")),
            "own_probe_fail": int(own_fail),
            "own_probe_absent": int(own_absent),
            "neighbor_probe_fail": int(
                label_neighbor_probe(rec, decision_tiers, final_tiers)
            ),
            "useless_duplicate": int(label_useless_duplicate(rec, decision_tiers)),
        }
        if have_baseline:
            row["harmful_write"] = int(
                label_harmful(rec, final_tiers, flips[backend])
            )
        for k in ("own_probe_fail", "neighbor_probe_fail", "useless_duplicate"):
            counters[k] += row[k]
        counters["harmful_write"] += row.get("harmful_write", 0)
        out_rows.append(row)

    mode = "a" if args.append else "w"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, mode, encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[label_outcomes] {log_path.parent.name}: {len(out_rows)} labeled "
          f"({counters['desync_skipped']} desync-skipped) -> {out_path}")
    print(f"[label_outcomes]   own_probe_fail={counters['own_probe_fail']} "
          f"neighbor_probe_fail={counters['neighbor_probe_fail']} "
          f"useless_duplicate={counters['useless_duplicate']} "
          f"harmful_write={counters['harmful_write'] if have_baseline else 'n/a'}")


if __name__ == "__main__":
    main()
