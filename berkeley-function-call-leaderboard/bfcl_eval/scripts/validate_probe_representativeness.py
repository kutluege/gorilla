"""
Probe representativeness validation  --  Plan 3 Step 3 (G4, SS4.1).

Question: do the deterministic probes "stand in" for the real BFCL questions?
Operationalization (documented, since the master plan's exact protocol is not
in the repo): for each backend and each SURVIVING scenario, the corpus is the
committed final memory snapshot; for every real question of that scenario
(`bfcl_eval/data/BFCL_v4_memory.json`), find the entry the REAL question
retrieves top-1 (`simulate_kv` / `simulate_vector` -- the backend's actual
retrieval model), then generate that entry's item probes and measure the
fraction of probes that retrieve the SAME entry top-1.

High agreement = the probe channel targets what real questions target; a
thesis finding, reported separately -- never a gate.

Usage:
  python bfcl_eval/scripts/validate_probe_representativeness.py \
      --model Qwen/Qwen3-4B-Instruct-2507-FC \
      --out gov_logs/probe_representativeness.json
"""

import argparse
import json
import sys
from pathlib import Path

BFCL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BFCL_ROOT))

from bfcl_eval.model_handler.middleware.probe_gen import (  # noqa: E402
    generate_item_probes,
)
from bfcl_eval.model_handler.middleware.retrieval_sim import (  # noqa: E402
    simulate_kv,
    simulate_vector,
)

DATA_FILE = BFCL_ROOT / "bfcl_eval" / "data" / "BFCL_v4_memory.json"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_questions():
    """{scenario: [question_text, ...]} from the memory dataset."""
    out = {}
    for row in read_jsonl(DATA_FILE):
        try:
            text = row["question"][0][0]["content"]
        except (KeyError, IndexError, TypeError):
            continue
        out.setdefault(row["scenario"], []).append(text)
    return out


def load_corpus(snapshot_path, backend):
    """-> (refs, texts): KV refs are key names (the retrieval corpus), texts are
    'key: value' composites; Vector refs==texts are the stored strings."""
    with open(snapshot_path, "r", encoding="utf-8") as f:
        snap = json.load(f)
    refs, texts = [], []
    for tier in ("core_memory", "archival_memory"):
        data = snap.get(tier, {})
        store = data.get("store", data) if isinstance(data, dict) else {}
        for key, value in store.items():
            if backend == "kv":
                refs.append(key)
                texts.append(f"{key.replace('_', ' ')}: {value}")
            else:
                refs.append(str(value))
                texts.append(str(value))
    return refs, texts


def question_agreement(backend, refs, texts, question_text):
    """-> (real_top1_ref, probe_agreement_fraction) or None if corpus empty.

    real_top1 via the backend's retrieval model on the question itself; probes
    from the retrieved entry (source='stored_text', diagnostics-grade), scored
    over the same corpus.
    """
    if not refs:
        return None
    if backend == "kv":
        ranked = simulate_kv(refs, question_text, k=5)
        if not ranked:
            return None
        top_ref = ranked[0][1]
        entry_text = texts[refs.index(top_ref)]
        probes = generate_item_probes("kv", entry_text, ref=top_ref,
                                      source="stored_text")
        hits = sum(
            1 for p in probes
            if (r := simulate_kv(refs, p.text, k=5)) and r[0][1] == top_ref
        )
    else:
        ranked = simulate_vector(texts, question_text, k=5)
        if not ranked:
            return None
        top_idx = ranked[0][1]
        top_ref = texts[top_idx]
        probes = generate_item_probes("vector", top_ref, source="stored_text")
        hits = sum(
            1 for p in probes
            if (r := simulate_vector(texts, p.text, k=5)) and r[0][1] == top_idx
        )
    return top_ref, (hits / len(probes)) if probes else 0.0


def validate_arm(result_model_dir, backends, surviving_only, wrra_records):
    questions = load_questions()
    dead = {
        (r["backend"], r["scenario"])
        for r in wrra_records
        if r.get("record") == "wrra_scenario" and r.get("chain_dead")
    }
    report = {}
    for backend in backends:
        snap_dir = (Path(result_model_dir) / "agentic" / "memory" / backend
                    / "memory_snapshot")
        per_scenario = {}
        for scenario, qs in sorted(questions.items()):
            if surviving_only and (backend, scenario) in dead:
                per_scenario[scenario] = {"skipped": "dead_chain"}
                continue
            snap = snap_dir / f"{scenario}_final.json"
            if not snap.exists():
                per_scenario[scenario] = {"skipped": "no_snapshot"}
                continue
            refs, texts = load_corpus(snap, backend)
            agreements = []
            for q in qs:
                res = question_agreement(backend, refs, texts, q)
                if res is not None:
                    agreements.append(res[1])
            per_scenario[scenario] = {
                "n_questions": len(agreements),
                "mean_probe_agreement": (
                    round(sum(agreements) / len(agreements), 4)
                    if agreements else None
                ),
                "majority_agree_fraction": (
                    round(sum(1 for a in agreements if a > 0.5) / len(agreements), 4)
                    if agreements else None
                ),
            }
        scored = [v for v in per_scenario.values() if v.get("n_questions")]
        report[backend] = {
            "per_scenario": per_scenario,
            "mean_probe_agreement": (
                round(sum(v["mean_probe_agreement"] * v["n_questions"] for v in scored)
                      / sum(v["n_questions"] for v in scored), 4)
                if scored else None
            ),
        }
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507-FC")
    ap.add_argument("--result-dir", default="result")
    ap.add_argument("--backends", default="kv,vector")
    ap.add_argument("--wrra", default="gov_logs/wrra_baseline.jsonl",
                    help="Chain-survival source; dead chains are skipped")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else BFCL_ROOT / p

    slug = args.model.replace("/", "_")
    wrra = read_jsonl(resolve(args.wrra)) if resolve(args.wrra).exists() else []
    report = validate_arm(
        resolve(args.result_dir) / slug,
        [b.strip() for b in args.backends.split(",") if b.strip()],
        surviving_only=True,
        wrra_records=wrra,
    )
    summary = {"instrument": "validate_probe_representativeness",
               "model": slug, "report": report}
    for backend, r in report.items():
        print(f"[probe-rep] {backend}: mean probe agreement = "
              f"{r['mean_probe_agreement']}")
    if args.out:
        out = resolve(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[probe-rep] report -> {out}")


if __name__ == "__main__":
    main()
