"""
Offline tests for the Stage 2 simulation core: retrieval_sim.py + probe_gen.py.

Run:  python bfcl_eval/scripts/test_retrieval_sim_offline.py

Parity contracts under test (Plan 1 Step 3 acceptance):
  (a) KV tokenization + ranking byte-identical to memory_kv._similarity_search
      on a shared corpus (the real backend function is imported and compared).
  (b) simulate_vector ranking matches a direct faiss.IndexFlatIP build for
      N <= 57 (cosine agreement <= 5e-4).
  (c) Probe provenance is complete and generation is deterministic.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bfcl_eval.model_handler.middleware.retrieval_sim import (  # noqa: E402
    default_encode,
    delta_h_for_probes,
    entropy,
    kv_tokenize,
    margin,
    n_eff,
    probe_scores,
    simulate_kv,
    simulate_vector,
)
from bfcl_eval.model_handler.middleware.probe_gen import (  # noqa: E402
    Probe,
    ProbeConfig,
    content_tokens_ordered,
    generate_decision_probes,
    generate_item_probes,
)

PASS, FAIL = 0, 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


KV_KEYS = [
    "user_name",
    "user_age",
    "favorite_drink",
    "meeting_notes_march",
    "q1_budget_deadline",
    "user_allergy_penicillin",
]

VEC_TEXTS = [
    "The user's name is Michael Rodriguez.",
    "The user is 35 years old.",
    "The user likes coffee.",
    "Allergic to penicillin, diagnosed 2019.",
    "Q1 budget report is due before Friday.",
]


def test_kv_parity():
    print("[kv parity vs memory_kv._similarity_search]")
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.memory_kv import (
        MemoryAPI_kv,
    )

    for q in ("user age", "budget deadline", "penicillin_allergy", "COFFEE drink", "zzz"):
        backend = MemoryAPI_kv._similarity_search(q, list(KV_KEYS), k=5)["ranked_results"]
        sim = simulate_kv(KV_KEYS, q, k=5)
        same = len(backend) == len(sim) and all(
            b[1] == s[1] and abs(float(b[0]) - s[0]) < 1e-12
            for b, s in zip(backend, sim)
        )
        check(f"ranked_results identical for {q!r}", same, f"{backend} vs {sim}")

    check(
        "tokenizer byte-identical",
        kv_tokenize("Meeting_Notes MARCH_2024") == "Meeting_Notes MARCH_2024".replace("_", " ").lower().split(),
    )
    check("k truncation", len(simulate_kv(KV_KEYS, "user", k=3)) == 3)
    check("empty corpus -> []", simulate_kv([], "user") == [])
    # Stable tie order: identical corpus entries keep corpus order (sorted is stable).
    tied = simulate_kv(["alpha_x", "beta_y"], "unrelated query", k=2)
    backend_tied = MemoryAPI_kv._similarity_search("unrelated query", ["alpha_x", "beta_y"], k=2)[
        "ranked_results"
    ]
    check(
        "tie order matches backend",
        [t[1] for t in tied] == [t[1] for t in backend_tied],
        f"{tied} vs {backend_tied}",
    )


def test_vector_parity():
    print("[vector parity vs faiss.IndexFlatIP]")
    import faiss

    corpus = VEC_TEXTS * 11  # 55 items, N <= 57 regime
    corpus = [f"{t} (v{i})" for i, t in enumerate(corpus)]
    embs = default_encode(corpus)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(embs.shape[1]))
    index.add_with_ids(embs, np.arange(len(corpus), dtype=np.int64))

    for q in ("how old is the user", "what allergies does the user have", "budget"):
        q_vec = default_encode([q])
        scores, ids = index.search(q_vec, 5)
        sim = simulate_vector(corpus, q, k=5)
        check(
            f"top-5 ids match faiss for {q!r}",
            [int(i) for i in ids[0]] == [i for _, i in sim],
            f"{ids[0]} vs {[i for _, i in sim]}",
        )
        max_diff = max(abs(float(s) - fs) for (s, _), fs in zip(sim, scores[0]))
        check(f"cosine agreement <= 5e-4 for {q!r}", max_diff <= 5e-4, f"diff={max_diff}")

    check("empty corpus -> []", simulate_vector([], "anything") == [])
    small = simulate_vector(VEC_TEXTS[:2], "user", k=5)
    check("k capped at corpus size", len(small) == 2)
    check(
        "scores sorted descending",
        all(small[i][0] >= small[i + 1][0] for i in range(len(small) - 1)),
    )


def test_metrics():
    print("[margin / entropy / n_eff]")
    check("margin top1-top2", abs(margin([3.0, 1.0, 0.5]) - 2.0) < 1e-12)
    check("margin single score = 0", margin([3.0]) == 0.0)
    check("margin empty = 0", margin([]) == 0.0)
    check("entropy empty = 0", entropy([]) == 0.0)
    check("entropy single = 0", abs(entropy([5.0])) < 1e-9)
    h_uniform = entropy([1.0, 1.0, 1.0, 1.0])
    check("uniform entropy = ln(4)", abs(h_uniform - np.log(4)) < 1e-9, str(h_uniform))
    check("n_eff(ln 4) = 4", abs(n_eff(np.log(4)) - 4.0) < 1e-9)
    h_peaked = entropy([10.0, 0.0, 0.0, 0.0])
    check("peaked < uniform entropy", h_peaked < h_uniform)
    check(
        "temperature only rescales logged H (monotone)",
        entropy([2.0, 1.0], temperature=5.0) > entropy([2.0, 1.0], temperature=0.5),
    )
    check("entropy top-k only", abs(entropy([1.0] * 10, k=5) - np.log(5)) < 1e-9)


def test_delta_h():
    print("[delta_h_for_probes]")
    corpus = ["user_name", "favorite_drink", "meeting_notes"]
    probes = ["user name", "what is user name"]
    # Duplicate key added: the neighbor's own probes now split mass over two
    # near-identical targets -> entropy goes up.
    dup = delta_h_for_probes("kv", corpus, "user_name_2", probes)
    check("kv duplicate raises neighbor entropy", dup["dH_mean"] > 0, str(dup["dH_mean"]))
    check("per-probe rows complete", len(dup["per_probe"]) == 2)
    check(
        "row fields present",
        all(set(r) == {"probe", "H_before", "H_after", "dH"} for r in dup["per_probe"]),
    )
    check("n_eff_after consistent", dup["n_eff_after_mean"] > 0)

    vec_corpus = list(VEC_TEXTS[:3])
    vec_probes = ["The user is 35 years old.", "what do you know about user age"]
    dup_v = delta_h_for_probes("vector", vec_corpus, "The user is 35 years old!", vec_probes)
    novel_v = delta_h_for_probes(
        "vector", vec_corpus, "The quarterly earnings were 4.2 million dollars.", vec_probes
    )
    check("vector duplicate dH > novel dH", dup_v["dH_mean"] > novel_v["dH_mean"],
          f"{dup_v['dH_mean']} vs {novel_v['dH_mean']}")
    check(
        "probe_scores kv/vector both rank",
        len(probe_scores("kv", corpus, "user name")) > 0
        and len(probe_scores("vector", vec_corpus, "user age")) > 0,
    )


def test_probe_gen():
    print("[probe_gen]")
    kv = generate_item_probes("kv", "user age: 35", ref="user_age")
    check("kv probes non-empty", len(kv) >= 2, str(kv))
    check("kv key probe from key words", kv[0].text == "user age", kv[0].text)
    check("kv probes never score values only",
          all("35" not in p.text or "user age" in p.text for p in kv))
    check("provenance complete",
          all(p.channel.startswith("template:") and p.source == "item_text" for p in kv))

    vec = generate_item_probes("vector", "The user likes strawberry matcha lattes.")
    check("vector identity probe first", vec[0].channel == "template:identity")
    check("vector probes >= 3 channels", len({p.channel for p in vec}) >= 3, str(vec))

    check("deterministic (same input -> same probes)",
          [p.to_dict() for p in vec]
          == [p.to_dict() for p in generate_item_probes("vector", "The user likes strawberry matcha lattes.")])

    sidecar = generate_item_probes("vector", "Some rehydrated fact.", source="stored_text")
    check("stored_text provenance flagged", all(p.source == "stored_text" for p in sidecar))

    dec = generate_decision_probes(
        "My allergy to penicillin was diagnosed in 2019.", ["2019"], ProbeConfig(probes_n=4)
    )
    check("decision probes from user text", dec[0].source == "user_text" and len(dec) <= 4)
    check("value question present",
          any(p.channel == "template:user_value_question" for p in dec), str(dec))
    check("empty user text -> no probes", generate_decision_probes("") == [])

    cfg = ProbeConfig(probes_n=2)
    check("probes_n cap respected",
          len(generate_decision_probes("A long sentence about many different things here.", [], cfg)) <= 2)

    toks = content_tokens_ordered("The User likes user_name and USER coffee coffee")
    check("ordered dedupe", toks == ["name", "coffee"], str(toks))
    check("limit respected", content_tokens_ordered("alpha beta gamma delta", limit=2) == ["alpha", "beta"])

    para = ProbeConfig(paraphrase=True)
    p_on = generate_item_probes("vector", "Paraphrase stub check.", cfg=para)
    check("paraphrase stub adds nothing (Plan 1)",
          all(p.channel.startswith("template:") for p in p_on))


def main():
    test_kv_parity()
    test_vector_parity()
    test_metrics()
    test_delta_h()
    test_probe_gen()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()


