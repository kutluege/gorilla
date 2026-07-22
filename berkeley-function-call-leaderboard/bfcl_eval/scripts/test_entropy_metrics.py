"""
Offline tests for the margin-entropy signal functions (Plan v2 Step 3, SS17.2).

Run:  python bfcl_eval/scripts/test_entropy_metrics.py

Pure-function coverage: ties, zero margins, empty/singleton inputs, max/min
entropy configurations, dH sign in both directions, n_eff_norm monotonicity and
bounds, churn set semantics, rank_self sentinel behavior.
"""

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bfcl_eval.model_handler.middleware.entropy_metrics import (  # noqa: E402
    churn,
    delta_h_self,
    disp,
    effective_rank,
    median,
    n_eff_norm,
    nmargin,
    rank_self,
    top_identity_set,
    vn_entropy,
    zscore_entropy,
)
from bfcl_eval.model_handler.middleware.retrieval_sim import entropy  # noqa: E402

PASS, FAIL = 0, 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def test_nmargin():
    print("[nmargin]")
    check("kv scale-free", abs(nmargin([10.0, 5.0], "kv") - 0.5) < 1e-9)
    check("kv scale invariance", abs(nmargin([10.0, 5.0], "kv") - nmargin([100.0, 50.0], "kv")) < 1e-6)
    check("vector raw margin", abs(nmargin([0.9, 0.7], "vector") - 0.2) < 1e-9)
    check("tie -> 0 (kv)", nmargin([3.0, 3.0], "kv") == 0.0)
    check("tie -> 0 (vector)", nmargin([0.5, 0.5], "vector") == 0.0)
    check("singleton -> 0", nmargin([1.0], "kv") == 0.0)
    check("empty -> 0", nmargin([], "vector") == 0.0)
    check("kv zero top score no crash", nmargin([0.0, 0.0], "kv") == 0.0)


def test_n_eff_norm():
    print("[n_eff_norm]")
    h_uniform = entropy([1.0, 1.0, 1.0, 1.0, 1.0])  # log 5
    check("uniform 5-score store>=4 -> 1.0", abs(n_eff_norm(h_uniform, 10, k=5) - 1.0) < 1e-6)
    check("H=0 -> 1/denom", abs(n_eff_norm(0.0, 10, k=5) - 0.2) < 1e-9)
    check("small store denominator", abs(n_eff_norm(0.0, 1, k=5) - 0.5) < 1e-9)
    check("empty store denominator=1", abs(n_eff_norm(0.0, 0, k=5) - 1.0) < 1e-9)
    hs = [0.1, 0.5, 1.0, 1.5]
    vals = [n_eff_norm(h, 10, k=5) for h in hs]
    check("monotone in H", all(a < b for a, b in zip(vals, vals[1:])))
    check("bounded above by ~1 for valid H", n_eff_norm(math.log(5), 10, k=5) <= 1.0 + 1e-9)


def test_disp():
    print("[disp]")
    check("uniform scores -> 0", disp([2.0, 2.0, 2.0]) == 0.0)
    check("empty -> 0", disp([]) == 0.0)
    check("zero mean -> 0", disp([1.0, -1.0]) == 0.0)
    d1, d2 = disp([1.0, 0.9, 0.8]), disp([1.0, 0.5, 0.0])
    check("more spread -> larger disp", d2 > d1)
    check("positive on non-degenerate", disp([3.0, 1.0]) > 0)


def test_churn():
    print("[churn]")
    check("identical sets -> 0", churn(["a", "b", "c"], ["a", "b", "c"]) == 0.0)
    check("disjoint sets -> 1", churn(["a", "b"], ["c", "d"]) == 1.0)
    check("one displaced of three", abs(churn(["a", "b", "c"], ["a", "b", "d"]) - 0.5) < 1e-9)
    check("both empty -> 0", churn([], []) == 0.0)
    check("one empty -> 1", churn(["a"], []) == 1.0)
    check("order-insensitive", churn(["a", "b"], ["b", "a"]) == 0.0)
    check("int identities (vector indices)", churn([0, 1, 2], [0, 1, 3]) == 0.5)


def test_rank_self():
    print("[rank_self]")
    ranked = [(0.9, "k1"), (0.8, "k2"), (0.7, "k3")]
    check("top-1", rank_self(ranked, "k1") == 1)
    check("mid", rank_self(ranked, "k3") == 3)
    check("absent -> k+1 sentinel", rank_self(ranked, "kX", k=5) == 6)
    check("vector index identity", rank_self([(0.9, 2), (0.8, 0)], 0) == 2)
    check("empty ranking -> sentinel", rank_self([], "x", k=5) == 6)


def test_median():
    print("[median]")
    check("odd", median([3, 1, 2]) == 2)
    check("even (mean of middle)", median([1, 2, 3, 4]) == 2.5)
    check("singleton", median([7]) == 7)
    check("empty -> None", median([]) is None)


def test_top_identity_set():
    print("[top_identity_set]")
    ranked = [(0.9, "a"), (0.8, "b"), (0.7, "c"), (0.6, "d")]
    check("top-3", top_identity_set(ranked, 3) == ["a", "b", "c"])
    check("shorter ranking ok", top_identity_set(ranked[:2], 3) == ["a", "b"])


def test_delta_h_self():
    print("[delta_h_self]")
    # KV: candidate key that answers the probe reduces uncertainty is not
    # guaranteed; assert mechanics + both dH signs are reachable.
    corpus = ["user_name", "user_age", "favorite_drink"]
    res = delta_h_self("kv", corpus, "favorite_food", ["favorite food"], k=5)
    check("per-probe rows present", len(res["per_probe"]) == 1)
    check("dH_self_mean is finite", isinstance(res["dH_self_mean"], float))
    # A candidate duplicating an existing key's tokens raises ambiguity for that
    # key's probe (positive dH); a distinctive one on its own probe can lower it.
    dup = delta_h_self("kv", corpus, "user_name_extra", ["user name"], k=5)
    check("shadowing candidate -> dH >= 0", dup["dH_self_mean"] >= 0.0, str(dup["dH_self_mean"]))
    # Both signs reachable across configurations:
    distinct = delta_h_self("kv", ["alpha_one", "beta_two"], "gamma_three", ["gamma three"], k=5)
    check("some configuration yields dH <= 0", distinct["dH_self_mean"] <= 0.0, str(distinct["dH_self_mean"]))
    # Provisional isolation: corpus list is not mutated.
    before = list(corpus)
    delta_h_self("kv", corpus, "zzz_key", ["zzz"], k=5)
    check("corpus not mutated", corpus == before)


def test_zscore_entropy():
    print("[zscore_entropy]")
    import math as m

    check("all tied -> log n", abs(zscore_entropy([2.0, 2.0, 2.0]) - m.log(3)) < 1e-9)
    check("singleton -> 0", zscore_entropy([5.0]) == 0.0)
    check("empty -> 0", zscore_entropy([]) == 0.0)
    h_sep = zscore_entropy([10.0, 1.0, 0.9, 0.8, 0.7])
    h_flat = zscore_entropy([1.0, 0.99, 0.98, 0.97, 0.96])
    check("separated top-1 < near-flat", h_sep < h_flat, f"{h_sep} vs {h_flat}")
    check("affine invariance (scale)",
          abs(zscore_entropy([3.0, 2.0, 1.0]) - zscore_entropy([300.0, 200.0, 100.0])) < 1e-9)
    check("affine invariance (shift)",
          abs(zscore_entropy([3.0, 2.0, 1.0]) - zscore_entropy([3.5, 2.5, 1.5])) < 1e-9)
    # The cosine-compression case that killed v1: tiny differences on a high
    # base still separate after z-scoring.
    h_cos = zscore_entropy([0.83, 0.62, 0.61, 0.60, 0.59])
    check("compressed cosine scale still separates", h_cos < m.log(5) - 0.05, str(h_cos))


def test_vn_entropy():
    print("[vn_entropy]")
    import math as m

    import numpy as np

    check("single vector -> 0", vn_entropy([[1.0, 0.0]]) == 0.0)
    check("empty -> 0", vn_entropy(np.zeros((0, 3))) == 0.0)
    orth = np.eye(3)
    check("orthogonal set -> log m", abs(vn_entropy(orth) - m.log(3)) < 1e-9)
    dup = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    check("pure duplicates -> ~0", vn_entropy(dup) < 1e-9)
    mixed = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    v_mixed = vn_entropy(mixed)
    check("duplicate pair + distinct sits between", 0.1 < v_mixed < m.log(3), str(v_mixed))
    check("adding a duplicate LOWERS S_vn",
          vn_entropy(np.vstack([orth, [1.0, 0.0, 0.0]])) < vn_entropy(orth))
    check("effective_rank of orthogonal-3 = 3",
          abs(effective_rank(vn_entropy(orth)) - 3.0) < 1e-6)
    check("scale of rows irrelevant (normalized internally)",
          abs(vn_entropy(orth * 7.3) - vn_entropy(orth)) < 1e-9)


def main():
    test_zscore_entropy()
    test_vn_entropy()
    test_nmargin()
    test_n_eff_norm()
    test_disp()
    test_churn()
    test_rank_self()
    test_median()
    test_top_identity_set()
    test_delta_h_self()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
