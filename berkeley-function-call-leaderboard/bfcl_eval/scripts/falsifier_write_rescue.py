"""Cheapest falsifier for alt10_write_rescue (autonomous loop, ledger #10).

Question: on no-call prereq steps (orphans), do the EXPLORATION samples
contain writes whose content carries gold answers that the final store
missed? If yes, an A2-majority-style rescue policy has real headroom; if not,
the alternative dies here.

Offline only: shadow orphan records + final snapshots + the production-grader
gold index. Reports per (replicate, backend): questions with uncarried gold,
how many are rescued by ANY sampled write (ceiling) and by the MODAL sampled
write (what A2 would commit), plus expected accuracy via p_hat.

    python bfcl_eval/scripts/falsifier_write_rescue.py \
        --logs gov_logs/hnav_shadow/rep0*_hact_shadow \
        --result-root result_hnav_shadow \
        --out gov_logs/hnav_autonomous/falsifier_write_rescue.json
"""

import argparse
import glob as globmod
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from bfcl_eval.model_handler.middleware.governance_filter import (  # noqa: E402
    build_candidate,
)
from hnav_answer_index import carries, scenario_questions  # noqa: E402
from label_outcomes import load_final_stores  # noqa: E402
from calibrate_margin_entropy import git_head, sha_file  # noqa: E402

P_HAT = 0.408544   # gov_logs/hnav_proxy_validation.json (Stage 1, frozen)
SLUG = "Qwen_Qwen3-4B-Instruct-2507-FC-HACT"


def orphan_writes(rec, backend):
    """(any_write_texts, modal_write_text) among exploration samples."""
    texts, keys = [], Counter()
    by_key = {}
    for s in (rec.get("samples") or [])[1:]:
        for call in (s.get("calls") or []):
            cand = build_candidate(backend, call)
            if cand is None:
                continue
            texts.append(cand.text)
            k = (cand.op, cand.ref, cand.text.strip().casefold())
            keys[k] += 1
            by_key[k] = cand.text
    modal = by_key[keys.most_common(1)[0][0]] if keys else None
    return texts, modal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--result-root", default="result_hnav_shadow")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shuffle-assignment", action="store_true",
                    help="NEGATIVE CONTROL (pre-registered in the alt10 ledger "
                         "entry): permute which scenario each orphan step's "
                         "sample set is assigned to, within (replicate, "
                         "backend). A genuine content-specific rescue signal "
                         "must collapse; surviving counts measure generic/"
                         "boilerplate carriage.")
    ap.add_argument("--shuffle-seed", type=int, default=20260808)
    args = ap.parse_args()

    dirs = sorted(set(p for g in args.logs for p in globmod.glob(g)))
    report = {"git_head": git_head(), "p_hat": P_HAT, "per_arm": [],
              "shuffle_assignment": bool(args.shuffle_assignment),
              "shuffle_seed": args.shuffle_seed if args.shuffle_assignment else None,
              "logs": [{"dir": d, "sha": sha_file(Path(d) / "hact_log.jsonl")}
                       for d in dirs]}
    summary = defaultdict(list)
    for d in dirs:
        d = Path(d)
        rep = d.name.split("_")[0]          # rep01
        # orphan sampled writes per (backend, scenario)
        per_step = []          # (backend, scen, texts, modal)
        for line in open(d / "hact_log.jsonl", encoding="utf-8"):
            rec = json.loads(line)
            if not rec.get("orphan"):
                continue
            backend = rec.get("backend")
            scen = rec["test_id"].split("-")[1]
            if scen == "student":
                continue
            texts, modal = orphan_writes(rec, backend)
            per_step.append((backend, scen, texts, modal))

        if args.shuffle_assignment:
            # permute the scenario labels within each backend, deterministic
            rng = random.Random(f"{args.shuffle_seed}|{d.name}")
            by_backend = defaultdict(list)
            for i, (backend, scen, _, _) in enumerate(per_step):
                by_backend[backend].append(i)
            for backend, idxs in by_backend.items():
                scens = [per_step[i][1] for i in idxs]
                rng.shuffle(scens)
                for i, s in zip(idxs, scens):
                    b, _, texts, modal = per_step[i]
                    per_step[i] = (b, s, texts, modal)

        writes = defaultdict(lambda: {"any": [], "modal": []})
        for backend, scen, texts, modal in per_step:
            writes[(backend, scen)]["any"].extend(texts)
            if modal is not None:
                writes[(backend, scen)]["modal"].append(modal)

        for backend in ("kv", "vector"):
            result_dir = Path(args.result_root) / rep / "hact_shadow" / SLUG
            stores = load_final_stores(result_dir, backend)
            n_uncarried = n_resc_any = n_resc_modal = n_q = 0
            examples = []
            for scen, tiers in stores.items():
                if scen == "student":
                    continue
                store_texts = [t for tier in tiers.values()
                               for t in tier.values()]
                w = writes.get((backend, scen), {"any": [], "modal": []})
                for q in scenario_questions(scen):
                    n_q += 1
                    gold = q["gold"]
                    if any(carries(t, gold) for t in store_texts):
                        continue
                    n_uncarried += 1
                    if any(carries(t, gold) for t in w["any"]):
                        n_resc_any += 1
                    if any(carries(t, gold) for t in w["modal"]):
                        n_resc_modal += 1
                        if len(examples) < 5:
                            examples.append(q["qid"])
            arm = {
                "replicate": rep, "backend": backend, "n_questions": n_q,
                "n_gold_uncarried_by_store": n_uncarried,
                "n_rescued_any_sample": n_resc_any,
                "n_rescued_modal_sample": n_resc_modal,
                "expected_dacc_modal": round(n_resc_modal * P_HAT / n_q, 4)
                if n_q else None,
                "example_rescued_qids": examples,
            }
            report["per_arm"].append(arm)
            summary[backend].append(arm["expected_dacc_modal"] or 0.0)

    report["mean_expected_dacc"] = {
        b: round(sum(v) / len(v), 4) for b, v in summary.items() if v}
    # GO criterion pre-registered in the ledger entry
    report["go_criterion"] = "mean expected_dacc_modal >= 0.02 on >= 1 backend"
    report["go"] = any(v >= 0.02 for v in report["mean_expected_dacc"].values())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("mean_expected_dacc", "go")}, indent=2))
    for a in report["per_arm"]:
        print(f"{a['replicate']}/{a['backend']}: uncarried={a['n_gold_uncarried_by_store']}"
              f" rescued_any={a['n_rescued_any_sample']}"
              f" rescued_modal={a['n_rescued_modal_sample']}"
              f" -> dAcc~{a['expected_dacc_modal']}")
    print(f"[falsifier] wrote {out}")


if __name__ == "__main__":
    main()
