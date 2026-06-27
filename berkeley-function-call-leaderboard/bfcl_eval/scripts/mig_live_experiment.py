"""
MIG Reranker — controlled live experiment on BFCL V4 Memory data.
==================================================================

This is NOT the full `bfcl generate` pipeline (this env lacks faiss / sentence-transformers /
transformers / openai, so the agent-driven retrieval + snapshot generation can't run here).
Instead it is a *controlled retrieval experiment* that exercises:
  * the REAL `MIGReranker` code (parse_candidates / select / _judge / _mig_logprob / reserialize),
  * the REAL BFCL agentic scorer (`agentic_checker`, ground_truth substring match),
  * the REAL benchmark data (questions, gold `source` sentence, `ground_truth`),
  * the LIVE Qwen3-4B-Instruct-2507 served at localhost:8000 (completions + echo-logprobs).

Setup per question:
  Build a noisy candidate pool = the question's gold `source` sentence (which contains the
  answer) + (P-1) distractor `source` sentences drawn from OTHER questions in the same
  scenario (realistic same-user/same-domain distractors that do NOT contain this answer).
  Each candidate gets a lexical-overlap "similarity_score" with the question (a cheap stand-in
  for the MiniLM cosine the real backend uses).

Arms (all answer in BFCL agentic format; scored by ground_truth substring):
  * full_pool      : model answers from ALL P candidates (no trimming).
  * sim_topk       : keep top-k by lexical similarity   (MIGReranker scorer="similarity").
  * mig_judge_topk : keep top-k by LLM utility judge     (MIGReranker scorer="judge", flat).
  * mig_logprob_topk: keep top-k by exact logprob MIG    (MIGReranker scorer="logprob", flat).

Metrics:
  * gold_recall@k : fraction where the gold source survived into the kept top-k.
  * gold_rank     : mean rank the scorer assigned the gold candidate (1 = best).
  * answer_acc    : fraction where ground_truth appears in the model's final answer.
"""
import argparse
import hashlib
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# Repo root = .../berkeley-function-call-leaderboard (this file lives in bfcl_eval/scripts/).
DEFAULT_REPO = Path(__file__).resolve().parents[2]


def _stable_hash(s: str) -> int:
    """Process-independent hash (Python's built-in hash() is salted per run)."""
    return int.from_bytes(hashlib.sha1(s.encode("utf-8")).digest()[:4], "big")

# ----------------------------------------------------------------------------- #
# Minimal OpenAI-style client backed by requests, so the REAL MIGReranker runs   #
# unmodified against the live vLLM server.                                       #
# ----------------------------------------------------------------------------- #
class _Logprobs:
    def __init__(self, d):
        self.token_logprobs = d.get("token_logprobs") if d else None
        self.text_offset = d.get("text_offset") if d else None
        self.tokens = d.get("tokens") if d else None


class _Choice:
    def __init__(self, c):
        self.text = c.get("text", "")
        self.logprobs = _Logprobs(c.get("logprobs"))


class _Resp:
    def __init__(self, j):
        self.choices = [_Choice(c) for c in j.get("choices", [])]
        usage = j.get("usage", {}) or {}
        self.usage = usage


class _Completions:
    def __init__(self, base_url, model, stats):
        self.base_url = base_url
        self.model = model
        self.stats = stats

    def create(self, model=None, prompt="", max_tokens=16, temperature=0.0,
               echo=False, logprobs=None, timeout=600, **kw):
        body = {
            "model": model or self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if echo:
            body["echo"] = True
        if logprobs is not None:
            body["logprobs"] = logprobs
        t0 = time.time()
        r = requests.post(f"{self.base_url}/completions", json=body, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        self.stats["calls"] += 1
        self.stats["latency"] += time.time() - t0
        u = j.get("usage", {}) or {}
        self.stats["prompt_tokens"] += u.get("prompt_tokens", 0) or 0
        self.stats["completion_tokens"] += u.get("completion_tokens", 0) or 0
        return _Resp(j)


class HttpClient:
    def __init__(self, base_url, model, stats):
        self.completions = _Completions(base_url, model, stats)


# ----------------------------------------------------------------------------- #
# Data + helpers                                                                 #
# ----------------------------------------------------------------------------- #
def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


_WORD = re.compile(r"[a-z0-9]+")


def lexical_sim(question, text):
    q = set(_WORD.findall(question.lower()))
    t = set(_WORD.findall(text.lower()))
    if not q or not t:
        return 0.0
    return len(q & t) / len(q | t)


def build_pools(repo, scenario, n_questions, pool_size, seed, distractors_mode="random"):
    data = load_jsonl(repo / "bfcl_eval/data/BFCL_v4_memory.json")
    ans = {a["id"]: a for a in load_jsonl(repo / "bfcl_eval/data/possible_answer/BFCL_v4_memory.json")}

    items = []
    for entry in data:
        if entry.get("scenario") != scenario:
            continue
        a = ans.get(entry["id"])
        if not a or not a.get("source") or not a.get("ground_truth"):
            continue
        items.append({
            "id": entry["id"],
            "question": entry["question"][0][0]["content"],
            "ground_truth": a["ground_truth"],
            "source": a["source"].strip(),
        })

    rng = random.Random(seed)
    universe = [it["source"] for it in items]
    rng.shuffle(items)
    items = items[:n_questions]

    pools = []
    for it in items:
        distractor_universe = [s for s in universe if s != it["source"]]
        k = min(pool_size - 1, len(distractor_universe))
        if distractors_mode == "hard":
            # Hard negatives: the OTHER facts most lexically similar to the question. This
            # mirrors what a real retriever surfaces (similar-looking but wrong entries) and
            # stresses the reranker -- a lexical/similarity baseline will rank some of these
            # above the gold, so trimming drops the gold unless a semantic scorer recovers it.
            distractors = sorted(
                distractor_universe, key=lambda s: lexical_sim(it["question"], s), reverse=True
            )[:k]
        else:
            distractors = random.Random(_stable_hash(it["id"])).sample(distractor_universe, k)
        pool_texts = distractors + [it["source"]]
        random.Random(_stable_hash(it["id"] + "_shuffle")).shuffle(pool_texts)
        # Wrap in the vector-backend result schema so we exercise parse_candidates/reserialize.
        result = {"result": [
            {"id": i, "similarity_score": round(lexical_sim(it["question"], t), 6), "text": t}
            for i, t in enumerate(pool_texts)
        ]}
        # Sort by similarity descending, like the real backend returns.
        result["result"].sort(key=lambda c: c["similarity_score"], reverse=True)
        pools.append({**it, "pool_json": json.dumps(result)})
    return pools


def answer_prompt(scenario_setting, memory_entries, question, agentic_fmt):
    mem = "\n".join(f"{i+1}. {t}" for i, t in enumerate(memory_entries))
    system = (
        f"{scenario_setting}\n\n"
        "Here are the relevant memory entries retrieved about the user:\n"
        f"{mem}\n\n"
        f"{agentic_fmt}"
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def gold_rank(scored_candidates, gold_text):
    """scored_candidates: list of (Candidate, score) NOT pre-sorted. Return 1-based rank of gold."""
    ordered = sorted(scored_candidates, key=lambda x: x[1], reverse=True)
    for i, (c, _s) in enumerate(ordered):
        if c.text.strip() == gold_text.strip():
            return i + 1
    return None


# ----------------------------------------------------------------------------- #
# Main                                                                           #
# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(DEFAULT_REPO))
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--scenario", default="customer")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--pool-size", type=int, default=8)
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="full_pool,sim_topk,mig_judge_topk,mig_logprob_topk")
    ap.add_argument("--distractors", default="random", choices=["random", "hard"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    repo = Path(args.repo)
    sys.path.insert(0, str(repo))
    from bfcl_eval.eval_checker.agentic_eval.agentic_checker import agentic_checker
    from bfcl_eval.constants.default_prompts import (
        MEMORY_AGENT_SETTINGS,
        ADDITIONAL_SYSTEM_PROMPT_FOR_AGENTIC_RESPONSE_FORMAT,
    )
    from bfcl_eval.model_handler.middleware.mig_reranker import MIGReranker, MIGConfig

    arms = args.arms.split(",")
    scenario_setting = MEMORY_AGENT_SETTINGS[args.scenario]
    agentic_fmt = ADDITIONAL_SYSTEM_PROMPT_FOR_AGENTIC_RESPONSE_FORMAT

    pools = build_pools(repo, args.scenario, args.n, args.pool_size, args.seed, args.distractors)
    print(f"Loaded {len(pools)} questions from scenario='{args.scenario}', "
          f"pool_size={args.pool_size}, budget={args.budget}, distractors={args.distractors}, "
          f"model={args.model}")

    stats = {"calls": 0, "latency": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
    client = HttpClient(args.base_url, args.model, stats)

    def make_reranker(scorer):
        cfg = MIGConfig(scorer=scorer, mode="flat", final_budget=args.budget,
                        use_draft=True, verbose=False, rerank_enabled=True)
        return MIGReranker(client=client, model_id=args.model, config=cfg)

    rr_sim = make_reranker("similarity")
    rr_judge = make_reranker("judge")
    rr_logprob = make_reranker("logprob")

    def answer(memory_entries, question):
        prompt = answer_prompt(scenario_setting, memory_entries, question, agentic_fmt)
        resp = client.completions.create(model=args.model, prompt=prompt,
                                         max_tokens=160, temperature=0.0)
        return MIGReranker._clean(resp.choices[0].text)

    def run_one(item):
        kind, candidates = MIGReranker.parse_candidates(item["pool_json"])
        gold = item["source"]
        res = {"id": item["id"], "question": item["question"], "gt": item["ground_truth"]}

        # full pool
        if "full_pool" in arms:
            ans = answer([c.text for c in candidates], item["question"])
            res["full_pool_correct"] = agentic_checker(ans, item["ground_truth"])["valid"]

        def do_arm(name, reranker):
            kept, _trace = reranker.select(item["question"], list(candidates))
            kept_texts = [c.text for c in kept]
            gold_in = any(t.strip() == gold.strip() for t in kept_texts)
            ans = answer(kept_texts, item["question"])
            correct = agentic_checker(ans, item["ground_truth"])["valid"]
            res[f"{name}_gold_kept"] = gold_in
            res[f"{name}_correct"] = correct

        if "sim_topk" in arms:
            do_arm("sim", rr_sim)
        if "mig_judge_topk" in arms:
            do_arm("judge", rr_judge)
        if "mig_logprob_topk" in arms:
            do_arm("logprob", rr_logprob)
        return res

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, it): it for it in pools}
        done = 0
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"  question failed: {e}")
            done += 1
            if done % 5 == 0:
                print(f"  ... {done}/{len(pools)} done ({time.time()-t0:.0f}s)")
    wall = time.time() - t0

    # ---- aggregate ----
    n = len(results)
    def frac(key):
        vals = [r[key] for r in results if key in r]
        return (sum(1 for v in vals if v) / len(vals)) if vals else float("nan")

    print("\n" + "=" * 64)
    print(f"RESULTS  scenario={args.scenario}  N={n}  pool={args.pool_size}  budget={args.budget}")
    print("=" * 64)
    print(f"{'arm':<20} {'gold_recall@k':>14} {'answer_acc':>12}")
    print("-" * 48)
    if "full_pool" in arms:
        print(f"{'full_pool':<20} {'(all kept)':>14} {frac('full_pool_correct'):>12.3f}")
    if "sim_topk" in arms:
        print(f"{'sim_topk (lexical)':<20} {frac('sim_gold_kept'):>14.3f} {frac('sim_correct'):>12.3f}")
    if "mig_judge_topk" in arms:
        print(f"{'mig_judge_topk':<20} {frac('judge_gold_kept'):>14.3f} {frac('judge_correct'):>12.3f}")
    if "mig_logprob_topk" in arms:
        print(f"{'mig_logprob_topk':<20} {frac('logprob_gold_kept'):>14.3f} {frac('logprob_correct'):>12.3f}")
    print("-" * 48)
    print(f"wall={wall:.0f}s  model_calls={stats['calls']}  "
          f"prompt_tok={stats['prompt_tokens']}  gen_tok={stats['completion_tokens']}  "
          f"avg_call_latency={stats['latency']/max(1,stats['calls']):.2f}s")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "results": results, "stats": stats,
                       "wall_s": wall}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
