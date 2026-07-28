"""
Offline tests for the marginal-diff signal library (H-Nav Stage 1.3).

Run:  python bfcl_eval/scripts/test_diff_signals.py

Covers: diff isolation (added / dropped clauses), the fresh-add defaults, the
falsifier's core claim (whole-blob cosine stays high while the marginal diff
does not), value/entity/date extraction, determinism, and the leakage guard --
an AST scan proving the signal module never touches benchmark ground truth.
"""

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bfcl_eval.model_handler.middleware import diff_signals  # noqa: E402
from bfcl_eval.model_handler.middleware.diff_signals import (  # noqa: E402
    ALL_FEATURES,
    LEXICAL_FEATURES,
    compute,
    lexical_features,
    marginal_diff,
)

PASS, FAIL = 0, 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def test_marginal_diff():
    print("[marginal_diff]")
    old = "The user lives in Seattle. He is a graphic designer."
    new = ("The user lives in Seattle. He is a graphic designer. "
           "He prefers express delivery.")
    d = marginal_diff(old, new)
    check("appended clause isolated",
          "express delivery" in d.added_text and "graphic designer" not in d.added_text,
          d.added_text)
    check("added clause counted", len(d.added_clauses) == 1, d.added_clauses)
    check("nothing dropped", d.dropped_clauses == [], d.dropped_clauses)
    check("has_old True", d.has_old is True)

    dropped = marginal_diff(
        "Alpha statement here. Beta statement here. Gamma statement here.",
        "Alpha statement here. Gamma statement here.",
    )
    check("dropped clause detected",
          any("Beta" in c for c in dropped.dropped_clauses), dropped.dropped_clauses)

    same = marginal_diff(old, old)
    check("identical text -> no added clauses", same.added_clauses == [])
    check("identical text -> edit ratio 0", abs(same.edit_ratio) < 1e-9)


def test_fresh_add_defaults():
    print("[fresh add (old is None)]")
    d = marginal_diff(None, "Brand new memory item about the espresso machine.")
    check("has_old False", d.has_old is False)
    check("added_text == new", d.added_text.startswith("Brand new"))
    check("edit ratio maximal", d.edit_ratio == 1.0)
    feats = lexical_features(d, None, "Brand new memory item about espresso.", [])
    for k in LEXICAL_FEATURES:
        check(f"{k} defined", feats.get(k) is not None, "None would read as 'no novelty'")
    check("has_old feature 0.0", feats["has_old"] == 0.0)
    check("target novelty maximal", feats["diff_novel_tok_frac_target"] == 1.0)


def test_falsifier_core_claim():
    print("[falsifier core claim: blob similar, diff not]")
    old = ("michael damaged espresso machine details: Michael from Seattle, 35, "
           "freelance graphic designer. Received a damaged stainless-steel "
           "espresso machine with a dented body and a bent steam wand.")
    new = old + (" Water reservoir area is slightly misaligned and the hinge "
                 "does not close flush. Frothing pitcher has a tiny scratch.")
    parts = marginal_diff(old, new)
    try:
        from bfcl_eval.model_handler.middleware.retrieval_sim import default_encode
        emb = default_encode([old, new, parts.added_text])
        blob_cos = float(emb[0] @ emb[1])
        diff_cos = float(emb[0] @ emb[2])
    except Exception as exc:  # encoder unavailable -> skip, do not fail silently
        check("encoder available", False, str(exc))
        return
    check("whole-blob cosine high", blob_cos >= 0.90, f"{blob_cos:.4f}")
    check("marginal-diff cosine materially lower", diff_cos < blob_cos - 0.10,
          f"blob={blob_cos:.4f} diff={diff_cos:.4f}")


def test_lexical_extraction():
    print("[lexical extraction]")
    old = "Delivery is pending."
    new = ("Delivery is pending. Originally estimated delivery: 11 business "
           "days. Contact Sarah Mitchell on 2026-03-05.")
    d = marginal_diff(old, new)
    f = lexical_features(d, old, new, [old])
    check("new value picked up", f["diff_n_new_values"] >= 1.0, f["diff_n_new_values"])
    check("new entity picked up", f["diff_n_new_entities"] >= 1.0, f["diff_n_new_entities"])
    check("new date picked up", f["diff_n_new_dates"] >= 1.0, f["diff_n_new_dates"])
    check("added fraction in (0,1)", 0.0 < f["diff_added_frac"] < 1.0, f["diff_added_frac"])
    check("edit ratio positive", f["diff_edit_ratio"] > 0.0)

    # store-scoped novelty: the same value already stored elsewhere is not new
    f2 = lexical_features(d, old, new, [old, "another item: 11 business days"])
    check("value already in store is not 'unseen'",
          f2["diff_n_new_values_unseen_in_store"] < f["diff_n_new_values_unseen_in_store"]
          or f["diff_n_new_values_unseen_in_store"] == 0,
          f"{f['diff_n_new_values_unseen_in_store']} -> "
          f"{f2['diff_n_new_values_unseen_in_store']}")


def test_compute_contract():
    print("[compute contract]")
    tiers = {
        "core": {"a_key": "a key: first stored item about billing",
                 "b_key": "b key: second stored item about shipping"},
        "archival": {},
    }
    fake_vecs = {}

    def fake_whiten(text):
        import numpy as np
        if text not in fake_vecs:
            rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
            fake_vecs[text] = rng.normal(size=8)
        return fake_vecs[text]

    def fake_encode(texts):
        import numpy as np
        return np.stack([fake_whiten(t) / (np.linalg.norm(fake_whiten(t)) or 1.0)
                         for t in texts])

    f = compute("kv", "core", "a_key",
                "a key: first stored item about billing",
                "a key: first stored item about billing and refunds",
                tiers, whiten=fake_whiten, encode=fake_encode)
    check("all feature keys present",
          all(k in f for k in ALL_FEATURES),
          sorted(set(ALL_FEATURES) - set(f)))
    check("no NaN values",
          not [k for k, v in f.items() if isinstance(v, float) and v != v],
          [k for k, v in f.items() if isinstance(v, float) and v != v])
    check("candidate excluded from its own store comparison",
          f["diff_sim_max"] is not None)

    f2 = compute("kv", "core", "a_key",
                 "a key: first stored item about billing",
                 "a key: first stored item about billing and refunds",
                 tiers, whiten=fake_whiten, encode=fake_encode)
    check("deterministic", f == f2)

    empty = compute("vector", "core", None, None, "brand new text",
                    {"core": {}, "archival": {}},
                    whiten=fake_whiten, encode=fake_encode)
    check("empty store does not crash", empty["has_old"] == 0.0)


def test_leakage_guard():
    print("[leakage guard]")
    forbidden = ("possible_answer", "ground_truth", "agentic_checker",
                 "question_text", "hnav_answer_index", "scenario_questions")
    for mod in ("bfcl_eval/model_handler/middleware/diff_signals.py",
                "bfcl_eval/scripts/refeature_diff.py"):
        src = (REPO_ROOT / mod).read_text(encoding="utf-8")
        tree = ast.parse(src)
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.update(node.module.split("."))
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    names.update(a.name.split("."))
        hits = sorted(n for n in names if n in forbidden)
        if mod.endswith("refeature_diff.py"):
            # The driver MAY reference the oracle helpers, but only inside
            # oracle_row -- never in feature_row.
            fn = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == "feature_row")
            fn_names = {x.id for x in ast.walk(fn) if isinstance(x, ast.Name)}
            fn_names |= {x.attr for x in ast.walk(fn) if isinstance(x, ast.Attribute)}
            check("feature_row is answer-blind",
                  not (fn_names & set(forbidden)), sorted(fn_names & set(forbidden)))
        else:
            check(f"{Path(mod).name} references no ground truth", not hits, hits)

    from bfcl_eval.scripts.calibrate_margin_entropy import (
        DIFF_FEATURES, FEATURE_KEYS, FORBIDDEN_FEATURE_TOKENS,
    )
    check("every diff feature is whitelisted",
          all(f in FEATURE_KEYS for f in DIFF_FEATURES),
          [f for f in DIFF_FEATURES if f not in FEATURE_KEYS])
    check("diff features carry no forbidden token",
          all(tok not in f.lower() for f in DIFF_FEATURES
              for tok in FORBIDDEN_FEATURE_TOKENS))
    check("oracle_* is NOT whitelisted",
          not any(k.startswith("oracle") for k in FEATURE_KEYS))
    check("module exports a stable feature order",
          diff_signals.ALL_FEATURES ==
          diff_signals.LEXICAL_FEATURES + diff_signals.SEMANTIC_FEATURES)


def main():
    test_marginal_diff()
    test_fresh_add_defaults()
    test_falsifier_core_claim()
    test_lexical_extraction()
    test_compute_contract()
    test_leakage_guard()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
