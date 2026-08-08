"""RAG-program offline falsifiers (methods M2, M3, M6, M7, M9, M10).

Cheapest-falsifier-first, per directive §9 item 10: every method gets a
zero-GPU falsifier against the prereq conversations, the gold index, the
backend-faithful retrieval simulators, and the existing campaign logs.
GO/NO-GO thresholds are stated in code BEFORE the numbers are computed and
frozen into the emitted JSON.

    python bfcl_eval/scripts/rag_falsifiers.py \
        --out gov_logs/hnav_rag/falsifiers.json

Analysis-side script: allowed to read gold (§3.2 -- offline evaluation only).
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from rank_bm25 import BM25Plus  # noqa: E402  (same lib as memory_kv)

from bfcl_eval.model_handler.middleware.retrieval_sim import (  # noqa: E402
    simulate_kv,
    simulate_vector,
)
from calibrate_margin_entropy import git_head  # noqa: E402
from falsifier_write_scaffold import prereq_turns, slug_key  # noqa: E402
from hnav_answer_index import carries, gold_index, scenario_questions  # noqa: E402

MAX_ENTRIES = 50
MAX_LEN = 2000
SCENARIOS = ("customer", "student", "finance", "healthcare", "notetaker")


# ---------------------------------------------------------------- builders

def build_per_turn(turns):
    """alt5R as written: one raw turn per entry, first 50, truncated."""
    return [t[:MAX_LEN] for t in turns[:MAX_ENTRIES]]


def build_packed(turns, max_len=MAX_LEN, cap=MAX_ENTRIES):
    """M2: greedy-pack consecutive turns into max_len-char entries."""
    entries, buf = [], ""
    for t in turns:
        t = str(t)
        if buf and len(buf) + 1 + len(t) > max_len:
            entries.append(buf)
            buf = ""
        while len(t) > max_len:          # single turn longer than an entry
            entries.append(t[:max_len])
            t = t[max_len:]
        buf = f"{buf} {t}".strip() if buf else t
    if buf:
        entries.append(buf)
    return entries[:cap], len(entries)   # capped view + uncapped need


def build_sentences(turns, cap=MAX_ENTRIES):
    """RAG-conventional fine chunking: one sentence per entry, first cap."""
    sents = []
    for t in turns:
        sents.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", str(t))
                     if s.strip())
    return sents[:cap]


def kv_keys(entries, n_tokens=6):
    keys, used = [], set()
    for e in entries:
        k = slug_key(e, n_tokens=n_tokens)
        n = 0
        while k in used:
            n += 1
            k = slug_key(e, n_tokens=n_tokens, salt=n)
        used.add(k)
        keys.append(k)
    return keys


# ---------------------------------------------------------------- metrics

def carried_and_recall(backend, entries, questions, ks=(5, 10, 20, 50),
                       keys=None):
    """(n_carried, {k: n_recall_at_k}) for one scenario's index."""
    n_car = 0
    rec = {k: 0 for k in ks}
    if backend == "kv":
        corpus = keys if keys is not None else kv_keys(entries)
    else:
        corpus = entries
    for q in questions:
        gold = q["gold"]
        carrier_idx = {i for i, e in enumerate(entries) if carries(e, gold)}
        if not carrier_idx:
            continue
        n_car += 1
        for k in ks:
            if backend == "kv":
                ranked = simulate_kv(corpus, q["qtext"], k=k)
                hits = {corpus.index(ident) for _, ident in ranked[:k]}
            else:
                ranked = simulate_vector(corpus, q["qtext"], k=k)
                hits = {int(ident) for _, ident in ranked[:k]}
            if hits & carrier_idx:
                rec[k] += 1
    return n_car, rec


def questions_of(scen):
    out = []
    for q in scenario_questions(scen):
        out.append({"qid": q["qid"], "qtext": q["text"], "gold": q["gold"]})
    return out


# ---------------------------------------------------------------- falsifiers

def falsifier_m2_m3(turns_by_scen, report):
    """M2 packed capture + M3 top_k curves, per backend."""
    go_m2 = {"criterion": "vector: packed carried-rate - per_turn carried-rate"
                          " >= +0.10 AND packed r@5 >= per_turn r@5",
             "frozen_before_compute": True}
    go_m3 = {"criterion": "vector packed r@20 - r@5 >= +0.05",
             "frozen_before_compute": True}
    out = {}
    for backend in ("vector", "kv"):
        agg = defaultdict(lambda: [0, 0, defaultdict(int)])  # nq, car, rec
        for scen in SCENARIOS:
            qs = questions_of(scen)
            turns = turns_by_scen[scen]
            builds = {
                "per_turn": (build_per_turn(turns), None),
                "packed": build_packed(turns),
                "sentences": (build_sentences(turns), None),
            }
            for name, built in builds.items():
                entries = built[0]
                n_car, rec = carried_and_recall(backend, entries, qs)
                agg[name][0] += len(qs)
                agg[name][1] += n_car
                for k, v in rec.items():
                    agg[name][2][k] += v
        rows = {}
        for name, (nq, car, rec) in agg.items():
            rows[name] = {
                "n_questions": nq,
                "carried": car, "carried_rate": round(car / nq, 4),
                **{f"r_at_{k}": round(v / nq, 4)
                   for k, v in sorted(rec.items())},
            }
        out[backend] = rows
    v = out["vector"]
    go_m2["value_carried_delta"] = round(
        v["packed"]["carried_rate"] - v["per_turn"]["carried_rate"], 4)
    go_m2["go"] = (go_m2["value_carried_delta"] >= 0.10
                   and v["packed"]["r_at_5"] >= v["per_turn"]["r_at_5"])
    go_m3["value_r20_minus_r5"] = round(
        v["packed"]["r_at_20"] - v["packed"]["r_at_5"], 4)
    go_m3["go"] = go_m3["value_r20_minus_r5"] >= 0.05
    report["m2_packed_capture"] = {"tables": out, "go": go_m2}
    report["m3_read_budget"] = {
        "note": "accuracy half of the tradeoff is unfalsifiable offline by "
                "construction; this GO covers only the recall half",
        "go": go_m3}


def falsifier_m6(turns_by_scen, report):
    """M6 verbatim-vs-abstractive: token-survival bound under deterministic
    extractive compression. Keeps the 'most informative' sentences (digits +
    capitalised tokens first) to a 1/ratio char budget."""
    def compress(text, ratio):
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
                 if s.strip()]
        scored = sorted(
            sents,
            key=lambda s: -(sum(c.isdigit() for c in s) * 2
                            + len(re.findall(r"\b[A-Z][a-z]+", s))))
        budget = max(1, len(text) // ratio)
        kept, used = [], 0
        for s in scored:
            if used >= budget:
                break
            kept.append(s)
            used += len(s)
        return " ".join(kept)

    rows = {}
    packed_car = 0
    nq_total = 0
    surv = {2: 0, 4: 0, 8: 0}
    for scen in SCENARIOS:
        qs = questions_of(scen)
        nq_total += len(qs)
        full = " ".join(str(t) for t in turns_by_scen[scen])
        packed_entries, _ = build_packed(turns_by_scen[scen])
        packed_text = " ".join(packed_entries)
        for q in qs:
            if carries(packed_text, q["gold"]):
                packed_car += 1
            for r in surv:
                if carries(compress(full, r), q["gold"]):
                    surv[r] += 1
    rows = {"n_questions": nq_total,
            "packed_verbatim_carried": packed_car,
            "carried_after_compression": {
                f"{r}x": {"n": n, "rate": round(n / nq_total, 4)}
                for r, n in surv.items()}}
    go = {"criterion": "carried-rate at 4x compression <= 0.75 x "
                       "packed-verbatim carried-rate -> abstractive-loses "
                       "hypothesis testable, run the live contrast",
          "frozen_before_compute": True,
          "value_4x_rate": rows["carried_after_compression"]["4x"]["rate"],
          "value_packed_rate": round(packed_car / nq_total, 4)}
    go["go"] = go["value_4x_rate"] <= 0.75 * go["value_packed_rate"]
    report["m6_abstractive_bound"] = {"table": rows, "go": go}


def falsifier_m7(turns_by_scen, report):
    """M7 capacity policies under the 50-entry cap (only student overflows)."""
    policies = {}
    for scen in SCENARIOS:
        qs = questions_of(scen)
        capped, need = build_packed(turns_by_scen[scen])
        if need <= MAX_ENTRIES:
            continue
        full_entries, _ = build_packed(turns_by_scen[scen], cap=10 ** 9)
        variants = {
            "fifo_drop_new": full_entries[:MAX_ENTRIES],
            "drop_oldest": full_entries[-MAX_ENTRIES:],
            "drop_longest_dupe": None,  # evict max-redundancy entries
        }
        # redundancy eviction: greedily drop the entry with the highest
        # token-overlap with any other until <= cap
        entries = list(full_entries)
        while len(entries) > MAX_ENTRIES:
            toks = [set(re.findall(r"[a-z0-9]+", e.lower())) for e in entries]
            worst, worst_i = -1.0, 0
            for i, ti in enumerate(toks):
                best = max((len(ti & tj) / max(1, len(ti | tj)))
                           for j, tj in enumerate(toks) if j != i)
                if best > worst:
                    worst, worst_i = best, i
            entries.pop(worst_i)
        variants["drop_longest_dupe"] = entries
        rows = {}
        for name, ents in variants.items():
            n_car, rec = carried_and_recall("vector", ents, qs, ks=(5,))
            rows[name] = {"carried_rate": round(n_car / len(qs), 4),
                          "r_at_5": round(rec[5] / len(qs), 4)}
        n_car_u, rec_u = carried_and_recall("vector", full_entries, qs,
                                            ks=(5,))
        rows["uncapped_ceiling"] = {
            "carried_rate": round(n_car_u / len(qs), 4),
            "r_at_5": round(rec_u[5] / len(qs), 4),
            "entries_needed": need}
        policies[scen] = rows
    go = {"criterion": "best policy carried-rate - fifo_drop_new "
                       ">= +0.03 on the overflowing scenario",
          "frozen_before_compute": True}
    if policies:
        scen, rows = next(iter(policies.items()))
        best = max(v["carried_rate"] for k, v in rows.items()
                   if k not in ("uncapped_ceiling",))
        go["value_best_minus_fifo"] = round(
            best - rows["fifo_drop_new"]["carried_rate"], 4)
        go["go"] = go["value_best_minus_fifo"] >= 0.03
        go["overflowing_scenario"] = scen
    else:
        go["go"] = False
        go["note"] = "no scenario overflows the cap"
    report["m7_capacity"] = {"per_scenario": policies, "go": go}


def falsifier_m9(turns_by_scen, report):
    """M9 rerank: simulate k=20 retrieve then BM25-rerank to 5 on the packed
    index; recall of the gold carrier in the reranked top-5."""
    nq = hit5 = hit20 = hit_rr = hit_rand = 0
    import random
    rng = random.Random(20260808)
    for scen in SCENARIOS:
        qs = questions_of(scen)
        entries, _ = build_packed(turns_by_scen[scen])
        for q in qs:
            carrier_idx = {i for i, e in enumerate(entries)
                           if carries(e, q["gold"])}
            if not carrier_idx:
                continue
            nq += 1
            r20 = [int(i) for _, i in
                   simulate_vector(entries, q["qtext"], k=20)][:20]
            r5 = set(r20[:5])
            hit5 += bool(r5 & carrier_idx)
            hit20 += bool(set(r20) & carrier_idx)
            if r20:
                toks = [re.findall(r"[a-z0-9]+", entries[i].lower())
                        for i in r20]
                bm = BM25Plus(toks)
                scores = bm.get_scores(
                    re.findall(r"[a-z0-9]+", q["qtext"].lower()))
                rr = [r20[i] for i in
                      sorted(range(len(r20)), key=lambda i: -scores[i])[:5]]
                hit_rr += bool(set(rr) & carrier_idx)
                hit_rand += bool(set(rng.sample(r20, min(5, len(r20))))
                                 & carrier_idx)
    rows = {"n_carried_questions": nq,
            "recall_top5": round(hit5 / nq, 4),
            "recall_top20": round(hit20 / nq, 4),
            "recall_rerank20to5_bm25": round(hit_rr / nq, 4),
            "recall_random5of20": round(hit_rand / nq, 4)}
    go = {"criterion": "reranked-top5 recall >= 0.90 x r@20 (recovers the "
                       "recall gain at k=5 context cost); else reject "
                       "without GPU",
          "frozen_before_compute": True,
          "value": rows["recall_rerank20to5_bm25"],
          "bar": round(0.90 * rows["recall_top20"], 4)}
    go["go"] = rows["recall_rerank20to5_bm25"] >= 0.90 * rows["recall_top20"]
    report["m9_rerank"] = {"table": rows, "go": go}


def falsifier_m10(report, baseline_result_dir, baseline_score_dir):
    """M10 abstention calibration: how many questions abstain while the gold
    sits in the store (esp. the always-visible core dump)?"""
    from label_outcomes import load_final_stores  # noqa: E402
    import json as _json
    ABSTAIN = re.compile(r"(do not|don't|cannot|can't|no information|"
                         r"not (?:have|stored|mentioned|available)|"
                         r"unable to)", re.I)
    rows = {}
    for backend in ("kv", "vector"):
        stores = load_final_stores(Path(baseline_result_dir), backend)
        f = (Path(baseline_result_dir) / "agentic" / "memory" / backend /
             f"BFCL_v4_memory_{backend}_result.json")
        n = n_abst = n_abst_carried = n_abst_core = 0
        gidx = gold_index()
        for line in open(f, encoding="utf-8"):
            rec = _json.loads(line)
            qid = rec["id"]
            core_id = qid.replace(f"memory_{backend}_", "memory_")
            gold = gidx.get(core_id)
            if not gold:
                continue
            scen = core_id.split("-")[1]
            tiers = stores.get(scen) or {"core": {}, "archival": {}}
            final = str((rec.get("result") or [[""]])[-1][-1])
            n += 1
            if not ABSTAIN.search(final):
                continue
            n_abst += 1
            carriers = [(t, r) for t in ("core", "archival")
                        for r, txt in (tiers.get(t) or {}).items()
                        if carries(txt, gold)]
            if carriers:
                n_abst_carried += 1
                if any(t == "core" for t, _ in carriers):
                    n_abst_core += 1
        rows[backend] = {"n": n, "abstain": n_abst,
                         "abstain_and_carried": n_abst_carried,
                         "abstain_and_core_carried": n_abst_core}
    go = {"criterion": "abstain-and-carried >= 5% of questions on vector",
          "frozen_before_compute": True,
          "value": round(rows["vector"]["abstain_and_carried"]
                         / max(1, rows["vector"]["n"]), 4)}
    go["go"] = go["value"] >= 0.05
    report["m10_abstention"] = {"table": rows, "go": go}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline-result-dir",
                    default="result_hnav_stage4/rep01/baseline/"
                            "Qwen_Qwen3-4B-Instruct-2507-FC")
    ap.add_argument("--baseline-score-dir",
                    default="score_hnav_stage4/rep01/baseline/"
                            "Qwen_Qwen3-4B-Instruct-2507-FC")
    args = ap.parse_args()

    turns_by_scen = prereq_turns()
    report = {"git_head": git_head(),
              "caps": {"max_entries": MAX_ENTRIES, "max_len": MAX_LEN}}
    falsifier_m2_m3(turns_by_scen, report)
    falsifier_m6(turns_by_scen, report)
    falsifier_m7(turns_by_scen, report)
    falsifier_m9(turns_by_scen, report)
    falsifier_m10(report, args.baseline_result_dir, args.baseline_score_dir)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for k, v in report.items():
        if isinstance(v, dict) and "go" in v:
            print(f"{k}: GO={v['go'].get('go')}  "
                  f"{ {a: b for a, b in v['go'].items() if a.startswith('value')} }")
    print(f"[rag-falsifiers] wrote {out}")


if __name__ == "__main__":
    main()
