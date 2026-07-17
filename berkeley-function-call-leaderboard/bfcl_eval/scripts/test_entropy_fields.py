"""
Offline tests for the SS4.4 entropy diagnostic fields (Plan 3 Step 1 / G1).

Covers:
  (a) H = -sum p log p, N_eff = exp(H), margin, and dH via delta_h_for_probes
      match hand-computed values on synthetic score lists;
  (b) every Stage 2 log record carries H / n_eff / min_margin / dH_neighbor /
      dH_mean AND the newly pinned `temperature` + `probes_n` metadata;
  (c) regression: the decision path is UNCHANGED -- same fixture input gives
      the same outcome/reason as pre-change Stage 2, temperature is metadata
      only (varying it changes logged H but never the decision).

Run:  python bfcl_eval/scripts/test_entropy_fields.py
"""

import json
import math
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import bfcl_eval.model_handler.middleware.governance_filter as gf  # noqa: E402
from bfcl_eval.model_handler.middleware.governance_filter import (  # noqa: E402
    GovConfig,
    GovDecision,
    GovernanceSession,
)
from bfcl_eval.model_handler.middleware.retrieval_sim import (  # noqa: E402
    delta_h_for_probes,
    entropy,
    margin,
    n_eff,
)

TMP = tempfile.mkdtemp(prefix="gov_entropy_test_")
PASS, FAIL = 0, 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def hand_entropy(scores, t=1.0):
    exps = [math.exp(s / t - max(scores) / t) for s in scores]
    z = sum(exps)
    ps = [e / z for e in exps]
    return -sum(p * math.log(p) for p in ps if p > 0)


class NeutralScorer:
    def probs_batch(self, pairs):
        return [(0.05, 0.90, 0.05) for _ in pairs]


def make_session(backend, snapshot, **cfg_overrides):
    log_file = cfg_overrides.pop("log_file", "entropy_test.jsonl")
    cfg = GovConfig(
        log_dir=TMP, log_file=log_file,
        nli_enabled=True, nli_shadow=False,
        s2_enabled=True, s2_shadow=False,
    )
    for k, v in cfg_overrides.items():
        setattr(cfg, k, v)
    return GovernanceSession(
        cfg=cfg,
        backend=backend,
        test_id=f"memory_{backend}_prereq_1-test-1",
        snapshot=snapshot,
        snapshot_path="<fabricated>",
        nli_scorer=NeutralScorer(),
        sidecar_path=str(Path(TMP) / f"{backend}_{log_file}_gov_state.json"),
    )


KV_SNAPSHOT = {
    "core_memory": {
        "user_name": "Michael Rodriguez",
        "user_age": "35",
        "favorite_drink": "coffee",
    },
    "archival_memory": {},
}


def force_escalate():
    original = gf.decide

    def fake_decide(candidate, signals, cache, cfg, preflight_ok):
        if signals.n_items == 0:
            return GovDecision.ADD, "empty_memory"
        return GovDecision.ESCALATE, "ambiguous"

    gf.decide = fake_decide
    return original


def read_decisions(log_file):
    path = Path(TMP) / log_file
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if r["event"] == "decision"]


def run_stage2_write(log_file, **cfg_overrides):
    """Drive one escalated KV write through the full session; return its record.

    The user text/candidate deliberately share tokens with the stored keys
    (favorite/drink/coffee) so probe BM25 scores are NON-uniform -- a uniform
    score list has H = ln(n) at every temperature and would make the
    temperature-sensitivity check vacuous.
    """
    original = force_escalate()
    try:
        s = make_session("kv", KV_SNAPSHOT, log_file=log_file, **cfg_overrides)
        s.user_text = "My favorite drink is espresso coffee these days."
        call = "core_memory_add(key='favorite_drink_espresso', value='espresso coffee')"
        governed = s.govern_calls([call])
    finally:
        gf.decide = original
    return governed, read_decisions(log_file)[-1]


def test_formulas():
    print("[SS4.4 formulas vs hand computation]")
    scores = [2.0, 1.0, 0.5]
    check("entropy matches hand softmax-H (T=1)",
          abs(entropy(scores, 1.0, k=5) - hand_entropy(scores)) < 1e-9)
    check("entropy matches hand softmax-H (T=2)",
          abs(entropy(scores, 2.0, k=5) - hand_entropy(scores, 2.0)) < 1e-9)
    h_uniform = entropy([1.0, 1.0, 1.0, 1.0], 1.0, k=5)
    check("uniform scores -> H = ln(4)",
          abs(h_uniform - math.log(4)) < 1e-9, str(h_uniform))
    check("N_eff = exp(H): uniform 4 -> 4.0",
          abs(n_eff(h_uniform) - 4.0) < 1e-9)
    check("peaked scores -> N_eff near 1",
          n_eff(entropy([50.0, 0.0, 0.0], 1.0, k=5)) < 1.01)
    check("margin = top1 - top2", abs(margin([2.0, 1.25, 0.5]) - 0.75) < 1e-12)
    check("margin of singleton = 0.0", margin([3.0]) == 0.0)

    # dH hand-check on the vector path: inserting a near-duplicate of the probe
    # target must raise that probe's retrieval entropy (dH > 0), inserting an
    # unrelated entry must move it far less.
    corpus = ["The user's name is Michael.", "The meeting is on March 3."]
    dup = delta_h_for_probes("vector", corpus, "The user's name is Mike.",
                             ["what is the user's name"], k=5, temperature=1.0)
    unrel = delta_h_for_probes("vector", corpus, "Budget report due Friday.",
                               ["what is the user's name"], k=5, temperature=1.0)
    check("dH: near-duplicate insertion raises probe entropy",
          dup["dH_mean"] > 0, str(dup["dH_mean"]))
    check("dH: duplicate perturbs more than unrelated entry",
          dup["dH_mean"] > unrel["dH_mean"],
          f"{dup['dH_mean']} vs {unrel['dH_mean']}")
    check("dH result carries n_eff_after_mean",
          dup["n_eff_after_mean"] is not None and dup["n_eff_after_mean"] > 0)


def test_log_fields():
    print("[Stage 2 record carries all SS4.4 fields]")
    _, rec = run_stage2_write("entropy_fields.jsonl")
    s2 = rec.get("stage2")
    check("stage2 block present", s2 is not None)
    if s2 is None:
        return
    for f in ("min_margin", "dH_neighbor", "dH_mean", "temperature",
              "probes_n", "per_probe"):
        check(f"stage2.{f} present", f in s2, str(sorted(s2)))
    check("temperature pinned to GOV_S2_T_KV default 1.0",
          s2["temperature"] == 1.0, str(s2["temperature"]))
    check("probes_n matches emitted probe list",
          s2["probes_n"] == len(s2["probes"]), f"{s2['probes_n']} vs {len(s2['probes'])}")
    per = s2["per_probe"]
    check("per_probe rows carry H and n_eff",
          bool(per) and all("H" in r and "n_eff" in r and "margin" in r for r in per))
    if per:
        check("per-probe n_eff == exp(H)",
              all(abs(r["n_eff"] - math.exp(r["H"])) < 1e-3 for r in per))
    check("dH_neighbor rows carry ref/dH/n_eff",
          all({"ref", "dH", "n_eff"} <= set(d) for d in s2["dH_neighbor"]))


def test_decision_regression():
    print("[regression: decision path unchanged, temperature is metadata only]")
    governed_a, rec_a = run_stage2_write("entropy_reg_a.jsonl")
    # Pre-change expectation for this fixture: Stage 2 resolves the escalation
    # (any stage2_* reason) and NEVER suppresses -- decision is ADD or REWRITE,
    # exactly as under Plan 1 Step 6 Stage 2.
    check("fixture decision matches pre-change expectation (never NOOP)",
          rec_a["decision"] in ("ADD", "REWRITE"), rec_a["decision"])
    check("fixture reason is a stage2 resolution",
          "stage2_" in rec_a["reason"], rec_a["reason"])
    check("write not suppressed (one call survives)",
          len(governed_a) == 1 and "core_memory" in governed_a[0])

    # Same input, wildly different softmax temperature: logged H must change,
    # decision/outcome/margin must not (margin is temperature-free).
    governed_b, rec_b = run_stage2_write("entropy_reg_b.jsonl", s2_t_kv=25.0)
    a2, b2 = rec_a["stage2"], rec_b["stage2"]
    check("temperature recorded per-record (1.0 vs 25.0)",
          a2["temperature"] == 1.0 and b2["temperature"] == 25.0)
    check("logged H responds to temperature",
          any(abs(ra["H"] - rb["H"]) > 1e-6
              for ra, rb in zip(a2["per_probe"], b2["per_probe"])),
          "H identical across T -- entropy not using T?")
    check("decision identical across temperatures",
          (rec_a["decision"], rec_a["reason"]) == (rec_b["decision"], rec_b["reason"]))
    check("min_margin identical across temperatures",
          a2["min_margin"] == b2["min_margin"])
    check("governed calls identical across temperatures", governed_a == governed_b)


if __name__ == "__main__":
    test_formulas()
    test_log_fields()
    test_decision_regression()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
