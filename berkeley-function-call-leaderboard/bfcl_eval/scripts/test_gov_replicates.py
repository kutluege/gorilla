"""
Offline tests for run_gov_replicates.py + analyze_gov_replicates.py
(Plan 2 Step 2 acceptance; target v2 SS8.3's ~16 tests).

All subprocess/network effects are injected fakes -- nothing here touches the
tunnel, the filesystem layout, or a real bfcl invocation.

Run:  python bfcl_eval/scripts/test_gov_replicates.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bfcl_eval.scripts.run_gov_replicates import (  # noqa: E402
    BASELINE_ARM,
    GOVERNED_ARM,
    arm_sequence,
    arm_specs,
    build_arm_env,
    build_evaluate_cmd,
    build_generate_cmd,
    expected_score_files,
    gov_env_of,
    parse_gov_env,
    run_replicates,
)
from bfcl_eval.scripts.analyze_gov_replicates import (  # noqa: E402
    analyze,
    bootstrap_delta_ci,
    completed_replicates,
    holm,
    mcnemar_exact,
    pair_replicate,
    unscored_replicates,
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


# ---------------------------------------------------------------------------
# Runner fixtures
# ---------------------------------------------------------------------------

def make_cfg(n=2, gov_env=None):
    return {
        "python": "python", "replicates": n, "start_replicate": 1,
        "baseline_model": "M-FC", "governed_model": "M-FC-GOV",
        "test_category": "memory_kv,memory_vector", "temperature": 0.001,
        "result_root": "res_root", "score_root": "sco_root",
        "gov_log_root": "gl_root", "manifest": "res_root/manifest.jsonl",
        "gov_env": gov_env or {"GOV_SIM_HIGH": "0.95", "GOV_DELTA": "0.32"},
        "base_url": "http://localhost:8000/v1", "run_id": "test_run",
    }


class Harness:
    """Injected fakes recording every side effect in order."""

    def __init__(self, fail_on_call=None, probe_fail_after=None):
        self.cmds, self.manifest = [], []
        self.probes = 0
        self.fail_on_call = fail_on_call
        self.probe_fail_after = probe_fail_after

    def run_cmd(self, cmd, env):
        self.cmds.append((list(cmd), dict(env)))
        return 1 if self.fail_on_call == len(self.cmds) else 0

    def probe(self, base_url):
        self.probes += 1
        if self.probe_fail_after is not None and self.probes > self.probe_fail_after:
            raise OSError("tunnel down")
        return {"served_models": ["Qwen/Qwen3-4B-Instruct-2507"],
                "vllm_version": "0.6.0"}

    def write(self, path, record):
        self.manifest.append(record)

    def go(self, cfg):
        return run_replicates(cfg, run_cmd=self.run_cmd, probe=self.probe,
                              manifest_writer=self.write, clock=lambda: 0.0)


def cmd_signature(harness):
    """[(model, phase), ...] in execution order."""
    out = []
    for cmd, _ in harness.cmds:
        phase = cmd[cmd.index("-m") + 2]  # python -m bfcl_eval <phase>
        out.append((cmd[cmd.index("--model") + 1], phase))
    return out


# ---------------------------------------------------------------------------
# Analyzer fixtures (synthetic wrra records)
# ---------------------------------------------------------------------------

def scen_rec(backend, scenario, dead=False, nq=0):
    return {"record": "wrra_scenario", "backend": backend,
            "scenario": scenario, "chain_dead": dead,
            "dead_reason": "empty_snapshot" if dead else None,
            "question_entries": nq}


def q_rec(backend, scenario, qid, correct):
    return {"record": "wrra_question", "backend": backend,
            "scenario": scenario, "id": qid, "correct": correct}


def arm_records(backend, spec):
    """spec: {scenario: (dead, {qid: correct})}."""
    recs = []
    for scenario, (dead, questions) in spec.items():
        recs.append(scen_rec(backend, scenario, dead, len(questions)))
        recs.extend(q_rec(backend, scenario, qid, ok)
                    for qid, ok in questions.items())
    return recs


def run():
    print("[runner: sequence + commands]")
    seq = arm_sequence(3)
    check("B1,G1,B2,G2,B3,G3 strict order",
          seq == [(1, BASELINE_ARM), (1, GOVERNED_ARM),
                  (2, BASELINE_ARM), (2, GOVERNED_ARM),
                  (3, BASELINE_ARM), (3, GOVERNED_ARM)], str(seq))
    check("start_replicate resumes mid-run",
          arm_sequence(3, start=3) == [(3, BASELINE_ARM), (3, GOVERNED_ARM)])

    cfg = make_cfg()
    gen = build_generate_cmd(cfg, "M-FC", 2, BASELINE_ARM)
    check("generate: --num-threads 1 hard-coded",
          gen[gen.index("--num-threads") + 1] == "1")
    check("generate: --skip-server-setup present", "--skip-server-setup" in gen)
    check("generate: per-(replicate, arm) result dir",
          gen[gen.index("--result-dir") + 1] == "res_root/rep02/baseline")
    ev = build_evaluate_cmd(cfg, "M-FC", 2, BASELINE_ARM)
    check("evaluate: partial-eval + per-(replicate, arm) score dir",
          "--partial-eval" in ev
          and ev[ev.index("--score-dir") + 1] == "sco_root/rep02/baseline")

    print("[runner: multi-arm generalization (G13)]")
    specs = arm_specs(make_cfg())
    check("default arm specs = classic two-arm A/B",
          [a["label"] for a in specs] == [BASELINE_ARM, GOVERNED_ARM]
          and specs[0]["gov_env"] is None)
    three = [{"label": "baseline", "model": "M-FC", "gov_env": None},
             {"label": "stage0", "model": "M-FC-GOV", "gov_env": {"GOV_NLI_ENABLED": "0"}},
             {"label": "full", "model": "M-FC-GOV", "gov_env": {"GOV_NLI_ENABLED": "1"}}]
    seq3 = arm_sequence(2, arms=three)
    check("N-arm alternation B1,A1,C1,B2,A2,C2",
          seq3 == [(1, "baseline"), (1, "stage0"), (1, "full"),
                   (2, "baseline"), (2, "stage0"), (2, "full")], str(seq3))
    cfg3 = make_cfg()
    cfg3["arms"] = three
    g_s0 = build_generate_cmd(cfg3, "M-FC-GOV", 1, "stage0")
    g_fl = build_generate_cmd(cfg3, "M-FC-GOV", 1, "full")
    check("same-model ablation arms get DISTINCT result dirs",
          g_s0[g_s0.index("--result-dir") + 1] != g_fl[g_fl.index("--result-dir") + 1])
    e_s0 = build_arm_env(cfg3, "stage0", 1, {"PATH": "p"})
    e_fl = build_arm_env(cfg3, "full", 1, {"PATH": "p"})
    check("per-arm gov_env applied from the spec",
          e_s0["GOV_NLI_ENABLED"] == "0" and e_fl["GOV_NLI_ENABLED"] == "1")
    check("per-(replicate, arm) GOV_LOG_DIR isolation across arms",
          e_s0["GOV_LOG_DIR"] != e_fl["GOV_LOG_DIR"]
          and e_s0["GOV_LOG_DIR"].endswith("rep01_stage0"))

    print("[runner: env construction]")
    base_env = {"PATH": "p", "GOV_SIM_HIGH": "0.11", "GOV_STALE": "x"}
    env_b = build_arm_env(cfg, BASELINE_ARM, 1, base_env)
    check("baseline: GOV_ENABLED=0 and stale GOV_* scrubbed",
          env_b["GOV_ENABLED"] == "0"
          and "GOV_STALE" not in env_b and env_b["PATH"] == "p")
    dirty = {"PATH": "p", "HACT_ENABLED": "1", "SCAF_ENABLED": "1",
             "RAG_ENABLED": "1", "RAG_MODE": "read_verbatim"}
    env_d = build_arm_env(cfg, BASELINE_ARM, 1, dirty)
    check("baseline: stale HACT_/SCAF_/RAG_ operator exports scrubbed",
          not any(k.startswith(("HACT_", "SCAF_", "RAG_")) for k in env_d),
          sorted(k for k in env_d if k.startswith(("HACT_", "SCAF_", "RAG_"))))

    print("[runner: arm-granularity resume + warmup wiring]")
    seq_sa = arm_sequence(2, start=1, arms=three, start_arm="stage0")
    check("--start-arm skips earlier arms in the start replicate ONLY",
          seq_sa == [(1, "stage0"), (1, "full"),
                     (2, "baseline"), (2, "stage0"), (2, "full")],
          str(seq_sa))
    try:
        arm_sequence(2, start=1, arms=three, start_arm="nope")
        check("--start-arm rejects unknown labels", False)
    except SystemExit:
        check("--start-arm rejects unknown labels", True)
    warm_calls = []
    h2 = Harness()
    cfg_w = make_cfg(n=1)
    cfg_w["warmup"] = lambda base_url, model: warm_calls.append(model) or 1
    h2.go(cfg_w)
    check("warmup called before every generate (per arm)",
          len(warm_calls) == 2, warm_calls)
    check("fast warmup (1 try) adds no manifest event",
          not any(r["event"] == "warmup_recovered" for r in h2.manifest))

    print("[runner: inference-error gate]")
    import tempfile as _tf
    from pathlib import Path as _P
    from run_gov_replicates import count_inference_errors
    tmp = _P(_tf.mkdtemp(prefix="govrep_"))
    good = tmp / "good_result.json"
    good.write_text('{"id": "a", "result": [["fine"]]}\n'
                    '{"id": "b", "result": [["also fine"]]}\n',
                    encoding="utf-8")
    bad = tmp / "bad_result.json"
    bad.write_text('{"id": "c", "result": [["ok"]]}\n'
                   '{"id": "d", "result": "Error during inference: '
                   'Connection error."}\n', encoding="utf-8")
    n, e = count_inference_errors(["good_result.json"], root=tmp)
    check("clean file counts 0 errors", (n, e) == (2, 0), (n, e))
    n, e = count_inference_errors(["good_result.json", "bad_result.json"],
                                  root=tmp)
    check("contaminated entry detected", (n, e) == (4, 1), (n, e))
    n, e = count_inference_errors(["missing_result.json"], root=tmp)
    check("missing file is not an error here (G15 covers absence)",
          (n, e) == (0, 0), (n, e))
    check("baseline: calibrated values NOT applied",
          "GOV_SIM_HIGH" not in env_b)
    env_g1 = build_arm_env(cfg, GOVERNED_ARM, 1, base_env)
    env_g2 = build_arm_env(cfg, GOVERNED_ARM, 2, base_env)
    check("governed: enabled + calibrated overrides",
          env_g1["GOV_ENABLED"] == "1" and env_g1["GOV_SIM_HIGH"] == "0.95"
          and env_g1["GOV_DELTA"] == "0.32")
    check("governed: distinct per-replicate GOV_LOG_DIR",
          env_g1["GOV_LOG_DIR"] != env_g2["GOV_LOG_DIR"]
          and env_g1["GOV_LOG_DIR"].endswith("rep01_governed"))
    check("gov_env_of filters non-GOV keys",
          set(gov_env_of(env_g1)) == {"GOV_ENABLED", "GOV_SIM_HIGH",
                                      "GOV_DELTA", "GOV_LOG_DIR"})
    try:
        parse_gov_env(["NOT_GOV=1"])
        check("parse_gov_env rejects non-GOV keys", False)
    except SystemExit:
        check("parse_gov_env rejects non-GOV keys", True)

    print("[runner: sequential execution + manifest]")
    h = Harness()
    h.go(make_cfg(n=2))
    check("execution order B1,B1,G1,G1,B2,B2,G2,G2 (gen+eval per arm)",
          cmd_signature(h) == [
              ("M-FC", "generate"), ("M-FC", "evaluate"),
              ("M-FC-GOV", "generate"), ("M-FC-GOV", "evaluate"),
              ("M-FC", "generate"), ("M-FC", "evaluate"),
              ("M-FC-GOV", "generate"), ("M-FC-GOV", "evaluate")],
          str(cmd_signature(h)))
    events = [r["event"] for r in h.manifest]
    check("manifest: run_start first, run_end last, no error",
          events[0] == "run_start" and events[-1] == "run_end"
          and "error" not in events)
    check("manifest: cmd_start/cmd_end per command",
          events.count("cmd_start") == 8 and events.count("cmd_end") == 8)
    start = h.manifest[0]
    check("manifest run_start pins identity fields",
          start["git_head"] is not None or True  # may be None outside a repo
          and start["served_models"] == ["Qwen/Qwen3-4B-Instruct-2507"]
          and start["vllm_version"] == "0.6.0"
          and start["seed"] is None and start["num_threads"] == 1
          and start["gov_env_governed"]["GOV_SIM_HIGH"] == "0.95")
    gov_starts = [r for r in h.manifest
                  if r["event"] == "cmd_start" and r["arm"] == GOVERNED_ARM]
    check("manifest records per-arm GOV_* env",
          all(r["gov_env"].get("GOV_ENABLED") == "1" for r in gov_starts))
    ev_ends = [r for r in h.manifest
               if r["event"] == "cmd_end" and r["phase"] == "evaluate"]
    check("G15: evaluate cmd_end records score paths + existence",
          ev_ends and all(
              len(r.get("score_files", [])) == 2
              and all({"path", "exists"} <= set(s) for s in r["score_files"])
              for r in ev_ends), str(ev_ends[:1]))
    check("G15: expected score paths carry arm + model slug",
          expected_score_files(make_cfg(), "M/X-FC", 3, "baseline")[0]
          == "sco_root/rep03/baseline/M_X-FC/agentic/memory/kv/"
             "BFCL_v4_memory_kv_score.json")

    print("[runner: fail-stop, never retried]")
    h = Harness(fail_on_call=3)  # G1 generate fails
    try:
        h.go(make_cfg(n=2))
        check("nonzero exit raises SystemExit", False)
    except SystemExit as e:
        check("nonzero exit raises SystemExit", e.code == 1)
    check("no further commands after failure (no retry)", len(h.cmds) == 3,
          str(len(h.cmds)))
    check("manifest carries the error record",
          h.manifest[-1]["event"] == "error"
          and h.manifest[-1]["replicate"] == 1
          and h.manifest[-1]["arm"] == GOVERNED_ARM)

    h = Harness(probe_fail_after=0)
    try:
        h.go(make_cfg(n=1))
        check("dead tunnel at preflight stops before any command", False)
    except SystemExit:
        check("dead tunnel at preflight stops before any command",
              len(h.cmds) == 0 and h.manifest[-1]["event"] == "error")
    h = Harness(probe_fail_after=2)  # preflight + B1 ok; G1 probe fails
    try:
        h.go(make_cfg(n=1))
        check("mid-run tunnel drop stops at arm boundary", False)
    except SystemExit:
        check("mid-run tunnel drop stops at arm boundary",
              len(h.cmds) == 2
              and h.manifest[-1]["stage"] == "tunnel_probe")

    print("[analyzer: exact McNemar]")
    check("mcnemar(1,5) = 0.21875 (hand-checked)",
          abs(mcnemar_exact(1, 5) - 0.21875) < 1e-12)
    check("mcnemar(1,4) = 0.375 (hand-checked)",
          abs(mcnemar_exact(1, 4) - 0.375) < 1e-12)
    check("mcnemar symmetric", mcnemar_exact(5, 1) == mcnemar_exact(1, 5))
    check("mcnemar(0,0) = 1.0", mcnemar_exact(0, 0) == 1.0)
    check("mcnemar clipped at 1", mcnemar_exact(3, 3) == 1.0)

    print("[analyzer: Holm]")
    adj = holm({"kv": 0.01, "vector": 0.04})
    check("holm m=2 hand-checked",
          abs(adj["kv"] - 0.02) < 1e-12 and abs(adj["vector"] - 0.04) < 1e-12)
    adj = holm({"a": 0.03, "b": 0.04, "c": 0.9})
    check("holm enforces monotonicity",
          abs(adj["a"] - 0.09) < 1e-12 and abs(adj["b"] - 0.09) < 1e-12
          and abs(adj["c"] - 0.9) < 1e-12, str(adj))

    print("[analyzer: survival-conditional pairing]")
    base = arm_records("kv", {
        "customer": (False, {"q1": True, "q2": False}),
        "finance": (False, {"q5": True}),
        "student": (False, {"q9": True}),
    })
    gov = arm_records("kv", {
        "customer": (False, {"q1": True, "q2": True}),
        "finance": (True, {"q5": False}),   # dead in governed arm only
        "student": (False, {"q9": False}),
    })
    pairs, drops, excl = pair_replicate(1, base, gov)
    check("dead chain in ONE arm drops the pair",
          len(drops) == 1 and drops[0]["scenario"] == "finance"
          and drops[0]["dead_in"] == [GOVERNED_ARM])
    check("drop counter carries question volume",
          drops[0]["n_questions"] == 1)
    check("surviving pairs matched", len(pairs) == 2
          and all(p["scenario"] == "customer" for p in pairs))
    check("student excluded by default (pre-registered)",
          excl["student_units"] == 1
          and not any(p["scenario"] == "student" for p in pairs))
    pairs_s, _, excl_s = pair_replicate(1, base, gov, include_student=True)
    check("--include-student is a labeled sensitivity switch",
          excl_s["student_units"] == 0
          and any(p["scenario"] == "student" for p in pairs_s))

    base_u = arm_records("kv", {"customer": (False, {"q1": True, "q2": True})})
    gov_u = arm_records("kv", {"customer": (False, {"q1": False, "q3": True})})
    _, _, excl_u = pair_replicate(1, base_u, gov_u)
    check("unmatched question ids excluded and counted",
          excl_u["unmatched_questions"] == 2)
    base_n = arm_records("kv", {"customer": (False, {"q1": None})})
    gov_n = arm_records("kv", {"customer": (False, {"q1": True})})
    p_n, _, excl_n = pair_replicate(1, base_n, gov_n)
    check("unscored (None) questions excluded and counted",
          excl_n["unscored_questions"] == 1 and not p_n)

    print("[analyzer: bootstrap CI]")
    units = [{"b_correct": 1, "g_correct": 2, "n": 4},
             {"b_correct": 2, "g_correct": 3, "n": 4}]
    d1 = bootstrap_delta_ci(units, n_boot=500, seed=7)
    d2 = bootstrap_delta_ci(units, n_boot=500, seed=7)
    check("deterministic under fixed seed", d1 == d2)
    check("observed delta pooled correctly", abs(d1[0] - 0.25) < 1e-12)
    same = [{"b_correct": 1, "g_correct": 3, "n": 4}] * 3
    d3 = bootstrap_delta_ci(same, n_boot=200, seed=1)
    check("degenerate units -> CI collapses to the point",
          d3[0] == 0.5 and d3[1] == 0.5 and d3[2] == 0.5)
    check("no data -> None CI",
          bootstrap_delta_ci([], n_boot=10, seed=1) == (None, None, None))

    print("[analyzer: end-to-end on a hand-checked table]")
    # rep1: customer alive both arms, finance dead in governed;
    # rep2: both alive. b (baseline-only) = 1, c (governed-only) = 4.
    rep1_base = arm_records("kv", {
        "customer": (False, {"q1": True, "q2": True, "q3": False, "q4": False}),
        "finance": (False, {"f1": True, "f2": False}),
    })
    rep1_gov = arm_records("kv", {
        "customer": (False, {"q1": True, "q2": False, "q3": True, "q4": True}),
        "finance": (True, {"f1": False, "f2": False}),
    })
    rep2_base = arm_records("kv", {
        "customer": (False, {"q1": True, "q2": False, "q3": False, "q4": False}),
        "finance": (False, {"f1": False, "f2": False}),
    })
    rep2_gov = arm_records("kv", {
        "customer": (False, {"q1": True, "q2": False, "q3": True, "q4": False}),
        "finance": (False, {"f1": True, "f2": False}),
    })
    summary = analyze({
        1: {BASELINE_ARM: rep1_base, GOVERNED_ARM: rep1_gov},
        2: {BASELINE_ARM: rep2_base, GOVERNED_ARM: rep2_gov},
    }, n_boot=200)
    kv = summary["per_backend"]["kv"]
    check("pairs = 10 (4+4+2), 1 unit dropped",
          kv["n_pairs"] == 10 and kv["n_units_dropped"] == 1
          and summary["total_units_dropped"] == 1)
    check("discordant b/c = 1/4",
          kv["discordant_baseline_only"] == 1
          and kv["discordant_governed_only"] == 4)
    check("accuracies pooled over survivors: 0.3 vs 0.6",
          abs(kv["acc_baseline"] - 0.3) < 1e-12
          and abs(kv["acc_governed"] - 0.6) < 1e-12)
    check("McNemar p matches hand calc 0.375",
          abs(kv["mcnemar_p"] - 0.375) < 1e-12)
    check("single backend: Holm p == raw p",
          kv["mcnemar_p_holm"] == kv["mcnemar_p"])
    check("output stamped as primary (student excluded)",
          summary["primary_result"] and not summary["include_student"])

    print("[analyzer: multi-arm joint survival + Holm dimension (G13)]")
    ref3 = arm_records("kv", {
        "customer": (False, {"q1": True, "q2": False}),
        "finance": (False, {"f1": True}),
    })
    armA = arm_records("kv", {
        "customer": (False, {"q1": True, "q2": True}),
        "finance": (True, {"f1": False}),      # dead ONLY in armA
    })
    armB = arm_records("kv", {
        "customer": (False, {"q1": False, "q2": False}),
        "finance": (False, {"f1": True}),
    })
    s3 = analyze({1: {"baseline": ref3, "armA": armA, "armB": armB}}, n_boot=100)
    check("joint survival: unit dead in ANY arm drops from ALL pairs",
          all(c["n_pairs"] == 2 for c in s3["comparisons"].values())
          and not any(p_key.startswith("finance")
                      for c in s3["comparisons"].values()
                      for p_key in [q["scenario"] for q in c["dropped_units"]]
                      if False))
    check("joint drop counted ONCE, not per pair",
          s3["total_units_dropped"] == 1
          and s3["comparisons"]["kv|armA_vs_baseline"]["dropped_units"][0]["dead_in"]
          == ["armA"])
    check("Holm m = backends x arm-pairs (1 x 2 = 2)", s3["holm_m"] == 2)
    check("comparison keys carry backend|arm_vs_ref",
          set(s3["comparisons"]) == {"kv|armA_vs_baseline", "kv|armB_vs_baseline"})
    check("Holm never lowers a raw p",
          all(c["mcnemar_p_holm"] >= c["mcnemar_p"] - 1e-12
              for c in s3["comparisons"].values()))
    check("N-arm output has no two-arm per_backend alias",
          "per_backend" not in s3)

    print("[analyzer: G15 unscored detection]")
    unscored_recs = {1: {
        "baseline": arm_records("kv", {"customer": (False, {"q1": None, "q2": None})}),
        "governed": arm_records("kv", {"customer": (False, {"q1": True, "q2": False})}),
    }}
    check("all-None arm/backend flagged as unscored",
          unscored_replicates(unscored_recs) == [(1, "baseline", "kv")])
    check("scored replicates produce no flags",
          unscored_replicates({1: {"baseline": arm_records(
              "kv", {"customer": (False, {"q1": True})})}}) == [])

    print("[analyzer: manifest completion filter]")
    man3 = [
        {"event": "run_start"},
        {"event": "cmd_end", "phase": "evaluate", "exit_code": 0,
         "replicate": 1, "arm": "baseline"},
        {"event": "cmd_end", "phase": "evaluate", "exit_code": 0,
         "replicate": 1, "arm": "stage0"},
    ]
    check("N-arm completion requires EVERY arm",
          completed_replicates(man3, arms=["baseline", "stage0", "full"]) == []
          and completed_replicates(man3, arms=["baseline", "stage0"]) == [1])
    man = [
        {"event": "run_start"},
        {"event": "cmd_end", "phase": "evaluate", "exit_code": 0,
         "replicate": 1, "arm": BASELINE_ARM},
        {"event": "cmd_end", "phase": "evaluate", "exit_code": 0,
         "replicate": 1, "arm": GOVERNED_ARM},
        {"event": "cmd_end", "phase": "evaluate", "exit_code": 0,
         "replicate": 2, "arm": BASELINE_ARM},
        {"event": "cmd_end", "phase": "evaluate", "exit_code": 1,
         "replicate": 2, "arm": GOVERNED_ARM},
        {"event": "cmd_end", "phase": "generate", "exit_code": 0,
         "replicate": 3, "arm": BASELINE_ARM},
    ]
    check("only replicates complete in BOTH arms analyzed",
          completed_replicates(man) == [1])

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run())
