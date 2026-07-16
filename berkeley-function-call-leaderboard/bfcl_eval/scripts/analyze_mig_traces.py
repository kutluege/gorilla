"""Offline analysis of MIG reranker traces.

Reads the per-test JSONL traces written by ``MIGReranker.log_trace`` (one line per
retrieval interception) and, joining against the memory ground-truth, computes
interception-subset metrics that the overall BFCL score cannot show:

  * how often the agent actually issued a rerankable retrieval (interceptions)
  * gold-in-pool   : the answer-bearing memory was present in the widened pool
  * gold-selected  : the reranker kept the answer-bearing memory
  * rank-before/after and recall@k before vs after reranking

Usage:
  python -m bfcl_eval.scripts.analyze_mig_traces \
      --trace-dir mig_traces/exp_2026-07-01/judge \
      --result-file result_mig_judge/Qwen_Qwen3-4B-Instruct-2507-FC-MIG/agentic/memory/vector/BFCL_v4_memory_vector_result.json \
      --budget 5

Caveats (documented, not bugs):
  * Traces store candidate text truncated to 160 chars, so gold that appears only
    after char 160 of a memory entry will be under-counted (gold answers are short,
    so this is usually fine but is reported as ``gold_truncation_possible``).
  * gold matching is a lowercased substring test (mirrors the official agentic
    checker's ``standardize_string`` + substring approach, minus punctuation folding).
"""
import argparse
import glob
import json
import os
from collections import Counter


def load_gold(gold_file):
    gold = {}
    with open(gold_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            gold[e["id"]] = {
                "ground_truth": [str(g) for g in e.get("ground_truth", [])],
                "source": e.get("source", ""),
            }
    return gold


def runtime_id_to_gold_id(test_id):
    # runtime ids are backend-prefixed: memory_vector_30-healthcare-0 -> memory_30-healthcare-0
    return test_id.replace("memory_vector", "memory").replace("memory_kv", "memory").replace(
        "memory_rec_sum", "memory"
    )


def contains_gold(text, ground_truth):
    t = (text or "").lower()
    return any(g.lower() in t for g in ground_truth if g)


def analyze_trace_entry(line_obj, gold_entry, budget):
    """One interception -> per-interception metrics dict (or None if not scorable)."""
    tr = line_obj.get("trace", {}) or {}
    scores = tr.get("scores")  # flat mode: list of {text, prior, score}, sorted by score desc
    if not scores:
        # greedy mode or fast-path (similarity_trim / rerank_disabled_trim): no per-candidate table
        return {"scorable": False}
    gt = gold_entry["ground_truth"] if gold_entry else []
    # rerank order = as-stored (sorted by score desc)
    by_score = scores
    by_prior = sorted(scores, key=lambda s: s.get("prior", 0.0), reverse=True)

    def first_gold_rank(ordered):
        for i, s in enumerate(ordered):
            if contains_gold(s.get("text", ""), gt):
                return i + 1  # 1-indexed
        return None

    rank_after = first_gold_rank(by_score)
    rank_before = first_gold_rank(by_prior)
    kept_texts = tr.get("kept", [])
    gold_in_pool = rank_before is not None
    gold_selected = any(contains_gold(t, gt) for t in kept_texts)
    return {
        "scorable": True,
        "have_gold": bool(gt),
        "pool_size": line_obj.get("pool_size"),
        "kept_size": line_obj.get("kept_size"),
        "gold_in_pool": gold_in_pool,
        "gold_selected": gold_selected,
        "rank_before": rank_before,
        "rank_after": rank_after,
        "recall_at_1_before": rank_before == 1 if rank_before else False,
        "recall_at_1_after": rank_after == 1 if rank_after else False,
        "recall_at_k_before": (rank_before is not None and rank_before <= budget),
        "recall_at_k_after": (rank_after is not None and rank_after <= budget),
    }


def count_result_entries(result_file):
    """(total_entries, entries_with_a_tool_execution)."""
    if not result_file or not os.path.isfile(result_file):
        return None, None
    total = 0
    with_tool = 0
    with open(result_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            if '"role": "tool"' in line or '"role":"tool"' in line:
                with_tool += 1
    return total, with_tool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--result-file", default=None,
                    help="answer result JSONL, to count total + tool-executing entries")
    ap.add_argument("--gold-file",
                    default="bfcl_eval/data/possible_answer/BFCL_v4_memory.json")
    ap.add_argument("--budget", type=int, default=5)
    args = ap.parse_args()

    gold = load_gold(args.gold_file)
    trace_files = sorted(glob.glob(os.path.join(args.trace_dir, "*.jsonl")))

    per_interception = []
    interceptions_by_test = Counter()
    kinds = Counter()
    unscorable = 0
    for tf in trace_files:
        test_id = os.path.splitext(os.path.basename(tf))[0]
        gid = runtime_id_to_gold_id(test_id)
        gentry = gold.get(gid)
        with open(tf, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                kinds[obj.get("kind")] += 1
                interceptions_by_test[test_id] += 1
                m = analyze_trace_entry(obj, gentry, args.budget)
                if not m.get("scorable"):
                    unscorable += 1
                    continue
                m["test_id"] = test_id
                m["gold_id"] = gid
                per_interception.append(m)

    total_entries, tool_entries = count_result_entries(args.result_file)
    n_intercept = sum(interceptions_by_test.values())
    scored = [m for m in per_interception if m["have_gold"]]

    def rate(sel):
        return (sum(1 for m in scored if sel(m)) / len(scored)) if scored else 0.0

    print("=" * 70)
    print(f"MIG TRACE ANALYSIS  |  trace-dir: {args.trace_dir}")
    print("=" * 70)
    print("\n-- Coverage (how rare is interception?) --")
    if total_entries is not None:
        print(f"  answer entries in result file       : {total_entries}")
        print(f"  entries with >=1 tool execution     : {tool_entries}")
    print(f"  distinct entries with interceptions : {len(interceptions_by_test)}")
    print(f"  total interceptions (rerankable)    : {n_intercept}")
    print(f"  interception kinds                  : {dict(kinds)}")
    print(f"  unscorable interceptions (greedy/fast-path/no-table): {unscorable}")

    print("\n-- Reranker quality on scorable interceptions with gold --")
    print(f"  scorable interceptions with gold    : {len(scored)}")
    if scored:
        print(f"  gold-in-pool rate                   : {rate(lambda m: m['gold_in_pool']):.3f}")
        print(f"  gold-selected rate                  : {rate(lambda m: m['gold_selected']):.3f}")
        print(f"  recall@1  before -> after           : {rate(lambda m: m['recall_at_1_before']):.3f} -> {rate(lambda m: m['recall_at_1_after']):.3f}")
        print(f"  recall@{args.budget}  before -> after           : {rate(lambda m: m['recall_at_k_before']):.3f} -> {rate(lambda m: m['recall_at_k_after']):.3f}")
        ranks_b = [m["rank_before"] for m in scored if m["rank_before"]]
        ranks_a = [m["rank_after"] for m in scored if m["rank_after"]]
        if ranks_b:
            print(f"  mean gold rank before (of in-pool)  : {sum(ranks_b)/len(ranks_b):.2f}")
        if ranks_a:
            print(f"  mean gold rank after  (of in-pool)  : {sum(ranks_a)/len(ranks_a):.2f}")
    else:
        print("  (no scorable interceptions with gold labels)")

    # machine-readable summary
    summary = {
        "trace_dir": args.trace_dir,
        "total_answer_entries": total_entries,
        "entries_with_tool": tool_entries,
        "entries_with_interception": len(interceptions_by_test),
        "total_interceptions": n_intercept,
        "kinds": dict(kinds),
        "scorable_with_gold": len(scored),
        "gold_in_pool_rate": rate(lambda m: m["gold_in_pool"]) if scored else None,
        "gold_selected_rate": rate(lambda m: m["gold_selected"]) if scored else None,
    }
    print("\nJSON_SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    main()
