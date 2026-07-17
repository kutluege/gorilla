"""
Geometry <-> dH correlation on LIVE Stage-2 logs  --  Plan 3 Step 2 (SS8.4 gate).

Computes, per backend, the Spearman correlation between Stage 0 geometry
(`sim_max`, `r`) and the Stage 2 decision-inert diagnostic `stage2.dH_mean`
read from live `governance_log.jsonl` decision records -- unlike
`replay_geometry_deltaH.py`, nothing is reconstructed: the joined fields come
from the SAME logged decision.

Chain-survival filter (binding, Plan 3 SS0.2): a dead chain's dH is
undefined/misleading, so records are joined against a `parse_wrra.py` output
and only writes belonging to SURVIVING (backend, scenario) chains enter the
correlation.

The rho >= 0.40 gate (SS8.4): per backend the verdict names the branch --
`geometry-first` (keep cascade order) or `nli-first-fallback` (geometry
dropped as pre-filter; Stage 1 sees that backend's traffic first). KV keeps
canonical-key-first regardless (zero decided NOOPs; Plan 1 Step 7).

The committed replay verdict remains the *provisional* answer; the *binding*
verdict must come from a live Stage-2-enabled shadow harvest (Plan 2 Step 3).
Stamp accordingly via --provisional.

Usage:
  python bfcl_eval/scripts/correlate_geometry_deltaH.py \
      --log gov_logs/<harvest>/governance_log.jsonl \
      --wrra gov_logs/wrra_governed.jsonl \
      --out gov_logs/geometry_dH_verdict.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

BFCL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BFCL_ROOT))

from bfcl_eval.scripts.replay_geometry_deltaH import (  # noqa: E402
    scenario_of,
    spearman,
)

GATE_RHO = 0.40


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def surviving_scenarios(wrra_records):
    """{(backend, scenario)} whose chain survived, from parse_wrra output."""
    return {
        (r["backend"], r["scenario"])
        for r in wrra_records
        if r.get("record") == "wrra_scenario" and not r.get("chain_dead")
    }


def collect_pairs(log_records, surviving):
    """Per-backend joined (sim_max, r, dH_mean) triples from live decisions.

    A record enters iff it is a `decision` event whose stage2 block carries a
    non-null dH_mean AND its (backend, scenario) chain survived. Counters make
    every exclusion auditable.
    """
    pairs = {}
    counters = {
        "decisions_total": 0, "no_stage2": 0, "no_dH": 0,
        "dead_chain": 0, "used": 0,
    }
    for rec in log_records:
        if rec.get("event") != "decision":
            continue
        counters["decisions_total"] += 1
        s2 = rec.get("stage2")
        if not s2:
            counters["no_stage2"] += 1
            continue
        if s2.get("dH_mean") is None:
            counters["no_dH"] += 1
            continue
        backend = rec["backend"]
        scenario = scenario_of(rec.get("test_id", ""))
        if (backend, scenario) not in surviving:
            counters["dead_chain"] += 1
            continue
        counters["used"] += 1
        pairs.setdefault(backend, []).append(
            (float(rec["sim_max"]), float(rec["r"]), float(s2["dH_mean"]))
        )
    return pairs, counters


def bootstrap_rho(xs, ys, n_boot=2000, seed=12345, alpha=0.05):
    """Percentile CI for Spearman rho, resampling joined records."""
    n = len(xs)
    if n < 3:
        return None, None
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        rho = spearman([xs[i] for i in idx], [ys[i] for i in idx])
        if rho is not None:
            draws.append(rho)
    if not draws:
        return None, None
    draws.sort()
    lo = draws[int((alpha / 2) * len(draws))]
    hi = draws[min(len(draws) - 1, int((1 - alpha / 2) * len(draws)))]
    return lo, hi


def correlate(pairs, gate=GATE_RHO, n_boot=2000, seed=12345):
    """Per-backend rho + CI + the SS8.4 branch decision."""
    out = {}
    for backend, triples in sorted(pairs.items()):
        sims = [t[0] for t in triples]
        rs = [t[1] for t in triples]
        dhs = [t[2] for t in triples]
        rho_sim = spearman(sims, dhs)
        rho_r = spearman(rs, dhs)
        ci_sim = bootstrap_rho(sims, dhs, n_boot=n_boot, seed=seed)
        ci_r = bootstrap_rho(rs, dhs, n_boot=n_boot, seed=seed + 1)
        gate_pass = rho_sim is not None and rho_sim >= gate
        out[backend] = {
            "n": len(triples),
            "rho_sim_max_vs_dH": rho_sim,
            "rho_sim_max_ci95": list(ci_sim),
            "rho_r_vs_dH": rho_r,
            "rho_r_ci95": list(ci_r),
            "gate": gate,
            "gate_pass": gate_pass,
            "branch": "geometry-first" if gate_pass else "nli-first-fallback",
            "note": (
                "KV keeps canonical-key-first regardless of this gate "
                "(Plan 1 Step 7: zero decided NOOPs, preflight catches dup keys)."
                if backend == "kv" else
                "Vector branch decision is binding for the cascade ordering."
            ),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--log", required=True,
                    help="Live governance_log.jsonl (must carry stage2 blocks)")
    ap.add_argument("--wrra", required=True,
                    help="parse_wrra JSONL of the SAME arm (chain-survival source)")
    ap.add_argument("--gate", type=float, default=GATE_RHO)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--provisional", action="store_true",
                    help="Stamp the verdict as provisional (not the binding "
                         "live-harvest verdict)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else BFCL_ROOT / p

    log_records = read_jsonl(resolve(args.log))
    wrra_records = read_jsonl(resolve(args.wrra))
    surviving = surviving_scenarios(wrra_records)
    pairs, counters = collect_pairs(log_records, surviving)
    verdict = correlate(pairs, gate=args.gate, n_boot=args.n_boot, seed=args.seed)

    summary = {
        "instrument": "correlate_geometry_deltaH",
        "binding": not args.provisional,
        "log": str(resolve(args.log)),
        "wrra": str(resolve(args.wrra)),
        "surviving_chains": sorted(f"{b}/{s}" for b, s in surviving),
        "counters": counters,
        "per_backend": verdict,
        "seed": args.seed,
        "n_boot": args.n_boot,
    }

    label = "PROVISIONAL" if args.provisional else "BINDING"
    print(f"[geo-dH] {label} | decisions used: {counters['used']}/"
          f"{counters['decisions_total']} (dead-chain excluded: "
          f"{counters['dead_chain']}, no stage2/dH: "
          f"{counters['no_stage2']}/{counters['no_dH']})")
    for backend, v in verdict.items():
        print(f"  {backend}: n={v['n']} rho(sim_max,dH)={v['rho_sim_max_vs_dH']} "
              f"CI95={v['rho_sim_max_ci95']} -> {v['branch']}")
    if not verdict:
        print("  no usable records -- is this a Stage-2-enabled log?")

    if args.out:
        out = resolve(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[geo-dH] verdict -> {out}")


if __name__ == "__main__":
    main()
