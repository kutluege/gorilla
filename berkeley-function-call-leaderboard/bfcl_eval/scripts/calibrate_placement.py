"""
G10 calibration instruments  --  Plan 3 Step 6 (SS7, geometry-first).

Three offline instruments over the COMMITTED result trees (never accuracy-
grid-searched; accuracy stays validation-only, E11):

  wgrid       w1 in {0.4, 0.5, 0.6, 0.7} eviction simulation: rank snapshot
              entries by the SS5 value formula, evict the bottom 25%, measure
              the fraction of ANSWER-BEARING entries lost (lower = better).
              Note: snapshot items carry turn_written=-1, so the recency term
              is constant here -- the offline grid isolates the uniqueness
              weight; live calibration (3B) refines with real turn indices.

  ksens       k in {3, 5, 7, 10} Stage-1 candidate-set sensitivity: for each
              entry as a fresh candidate, related items = whitened cosine
              >= 0.4; coverage(k) = fraction of related items inside the
              top-k candidate set. The smallest k with near-max coverage is
              the economical choice.

  readmargin  SS6 threshold source: margins of the REAL questions (answer
              key) against surviving snapshot corpora under the backend's
              retrieval model, split by whether top-1 is answer-bearing.
              Suggested threshold = p25 of the answer-bearing margins
              (provisional; refined on live shadow logs in 3B).

Usage:
  python bfcl_eval/scripts/calibrate_placement.py \
      --model Qwen/Qwen3-4B-Instruct-2507-FC \
      --wrra gov_logs/wrra_baseline.jsonl \
      --out-json gov_logs/plan3_calibration.json \
      --out-md gov_logs/plan3_calibration.md
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

BFCL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BFCL_ROOT))

from bfcl_eval.model_handler.middleware.governance_filter import (  # noqa: E402
    GovConfig,
    GovernanceCache,
    load_abtt,
)
from bfcl_eval.model_handler.middleware.placement import value_scores  # noqa: E402
from bfcl_eval.model_handler.middleware.retrieval_sim import (  # noqa: E402
    margin as sim_margin,
    simulate_kv,
    simulate_vector,
)
from bfcl_eval.model_handler.middleware.semantic_entropy import (  # noqa: E402
    _get_encoder,
)

DATA = BFCL_ROOT / "bfcl_eval" / "data"
W_GRID = [0.4, 0.5, 0.6, 0.7]
K_GRID = [3, 5, 7, 10]
RELATED_SIM = 0.4
EVICT_FRACTION = 0.25


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_answer_key():
    """{scenario: [(question_text, [answers], source)]} from the answer key."""
    questions = {r["id"]: r for r in read_jsonl(DATA / "BFCL_v4_memory.json")}
    out = {}
    for row in read_jsonl(DATA / "possible_answer" / "BFCL_v4_memory.json"):
        q = questions.get(row["id"])
        if not q:
            continue
        try:
            text = q["question"][0][0]["content"]
        except (KeyError, IndexError, TypeError):
            continue
        out.setdefault(q["scenario"], []).append(
            (text, [str(a) for a in row.get("ground_truth", [])],
             str(row.get("source", "")))
        )
    return out


def surviving(wrra_records):
    return {
        (r["backend"], r["scenario"])
        for r in wrra_records
        if r.get("record") == "wrra_scenario" and not r.get("chain_dead")
    }


def build_cache(snapshot_path, backend, encoder, abtt):
    with open(snapshot_path, "r", encoding="utf-8") as f:
        snap = json.load(f)
    cache = GovernanceCache(backend)
    cache.rehydrate(
        snap, lambda texts: np.asarray(encoder.encode(texts,
                                                      normalize_embeddings=False)),
        abtt,
    )
    return cache


def answer_bearing(text, answers):
    low = text.lower()
    return any(str(a).lower() in low for a in answers)


def percentile(values, p):
    if not values:
        return None
    vs = sorted(values)
    return round(vs[min(len(vs) - 1, int(p / 100 * len(vs)))], 4)


def run_wgrid(caches, answers_by_scenario):
    grid = {w: {"evicted_answer_bearing": 0, "answer_bearing_total": 0}
            for w in W_GRID}
    for (backend, scenario), cache in caches.items():
        items = cache.all_items()
        if len(items) < 4:
            continue
        answers = [a for _, ans, _ in answers_by_scenario.get(scenario, [])
                   for a in ans]
        bearing = {it.ref for it in items if answer_bearing(it.text, answers)}
        if not bearing:
            continue
        n_evict = max(1, math.ceil(EVICT_FRACTION * len(items)))
        for w1 in W_GRID:
            scores = value_scores(items, current_step=0, w1=w1, w2=1 - w1)
            ranked = sorted(zip(scores, items), key=lambda t: (t[0], t[1].ref))
            evicted = {it.ref for _, it in ranked[:n_evict]}
            grid[w1]["evicted_answer_bearing"] += len(evicted & bearing)
            grid[w1]["answer_bearing_total"] += len(bearing)
    for w1, g in grid.items():
        g["answer_loss"] = (
            round(g["evicted_answer_bearing"] / g["answer_bearing_total"], 4)
            if g["answer_bearing_total"] else None
        )
    return {str(w): g for w, g in grid.items()}


def run_ksens(caches):
    cover = {k: [] for k in K_GRID}
    for cache in caches.values():
        items = cache.all_items()
        if len(items) < 3:
            continue
        M = np.stack([it.emb_whitened for it in items])
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        M = M / np.clip(norms, 1e-12, None)
        sims = M @ M.T
        np.fill_diagonal(sims, -np.inf)
        for i in range(len(items)):
            related = {j for j in range(len(items)) if sims[i, j] >= RELATED_SIM}
            if not related:
                continue
            order = np.argsort(-sims[i])
            for k in K_GRID:
                top = set(order[:k].tolist())
                cover[k].append(len(related & top) / len(related))
    return {
        str(k): {"mean_coverage": round(float(np.mean(v)), 4) if v else None,
                 "n_cases": len(v)}
        for k, v in cover.items()
    }


def run_readmargin(caches, answers_by_scenario):
    out = {}
    for backend in ("kv", "vector"):
        m_bearing, m_other = [], []
        for (b, scenario), cache in caches.items():
            if b != backend:
                continue
            items = cache.all_items()
            refs = [it.ref for it in items]
            texts = [it.text for it in items]
            for question, answers, _ in answers_by_scenario.get(scenario, []):
                if backend == "kv":
                    ranked = simulate_kv(refs, question, k=5)
                    top_text = texts[refs.index(ranked[0][1])] if ranked else ""
                else:
                    ranked = simulate_vector(texts, question, k=5)
                    top_text = texts[ranked[0][1]] if ranked else ""
                if not ranked:
                    continue
                m = sim_margin([s for s, _ in ranked])
                (m_bearing if answer_bearing(top_text, answers) else m_other
                 ).append(m)
        out[backend] = {
            "n_answer_bearing_top1": len(m_bearing),
            "n_other_top1": len(m_other),
            "answer_bearing_margin_p25": percentile(m_bearing, 25),
            "answer_bearing_margin_p50": percentile(m_bearing, 50),
            "other_margin_p50": percentile(m_other, 50),
            "suggested_read_margin": percentile(m_bearing, 25),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507-FC")
    ap.add_argument("--result-dir", default="result")
    ap.add_argument("--wrra", default="gov_logs/wrra_baseline.jsonl")
    ap.add_argument("--out-json", default="gov_logs/plan3_calibration.json")
    ap.add_argument("--out-md", default="gov_logs/plan3_calibration.md")
    args = ap.parse_args()

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else BFCL_ROOT / p

    slug = args.model.replace("/", "_")
    alive = surviving(read_jsonl(resolve(args.wrra)))
    encoder = _get_encoder()
    abtt = load_abtt(GovConfig().artifact_path)

    caches = {}
    for backend, scenario in sorted(alive):
        snap = (resolve(args.result_dir) / slug / "agentic" / "memory" / backend
                / "memory_snapshot" / f"{scenario}_final.json")
        if snap.exists():
            caches[(backend, scenario)] = build_cache(snap, backend, encoder, abtt)

    answers = load_answer_key()
    summary = {
        "instrument": "calibrate_placement",
        "model": slug,
        "surviving_chains": sorted(f"{b}/{s}" for b, s in alive),
        "wgrid": run_wgrid(caches, answers),
        "ksens": run_ksens(caches),
        "readmargin": run_readmargin(caches, answers),
        "notes": [
            "pooled calibration over surviving scenarios; report per-scenario in 3B",
            "wgrid isolates uniqueness (snapshot turn_written=-1 flattens recency)",
            "readmargin suggestion is provisional until the live shadow harvest",
        ],
    }

    out_json = resolve(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Plan 3 Step 6 -- G10 calibration note (offline instruments)",
        "",
        f"Source: `{slug}` committed trees, surviving chains "
        f"{summary['surviving_chains']}.",
        "",
        "## w1 grid (eviction simulation, answer-loss; lower = better)",
        "",
        "| w1 | answer-bearing evicted | total | answer_loss |",
        "|---|---|---|---|",
    ]
    for w, g in summary["wgrid"].items():
        lines.append(f"| {w} | {g['evicted_answer_bearing']} | "
                     f"{g['answer_bearing_total']} | {g['answer_loss']} |")
    lines += ["", "## k sensitivity (related-coverage of the Stage-1 candidate set)",
              "", "| k | mean coverage | cases |", "|---|---|---|"]
    for k, g in summary["ksens"].items():
        lines.append(f"| {k} | {g['mean_coverage']} | {g['n_cases']} |")
    lines += ["", "## read-time margin (real-question margins over snapshots)", ""]
    for backend, g in summary["readmargin"].items():
        lines.append(
            f"- **{backend}**: answer-bearing top-1 n={g['n_answer_bearing_top1']} "
            f"(p25={g['answer_bearing_margin_p25']}, p50={g['answer_bearing_margin_p50']}); "
            f"other top-1 n={g['n_other_top1']} (p50={g['other_margin_p50']}); "
            f"**suggested GOV_READ_MARGIN = {g['suggested_read_margin']}** (provisional)"
        )
    lines += ["", "Notes: " + "; ".join(summary["notes"]) + "."]
    out_md = resolve(args.out_md)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({k: summary[k] for k in ("wgrid", "ksens", "readmargin")},
                     indent=2))
    print(f"[calibrate] -> {out_json}\n[calibrate] -> {out_md}")


if __name__ == "__main__":
    main()
