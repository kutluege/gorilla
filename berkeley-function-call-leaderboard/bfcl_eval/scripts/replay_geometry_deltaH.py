"""
Offline replay instrument: geometry vs retrieval-entropy (the SS8.4 go/no-go feed).

Walks a ``gov_logs/<arm>/governance_log.jsonl``, reconstructs per-scenario memory
state by chain-carry (rehydrate -> decisions -> observes), and for every
computable decision provisionally adds the logged candidate and measures
``dH_neighbor``: the change in the top-k retrieval entropy of the candidate's
tier neighbors on the neighbors' OWN template probes (STAGE0 analysis SS4.4 /
SS8.3), using the backend-faithful simulators in
``middleware/retrieval_sim.py``. The whitened (ABTT) space never enters the
simulation.

State reconstruction rules (Plan 1 Step 4):
  * Mutations come from ``observe`` / ``observe_remove`` / ``observe_clear``
    events only -- suppressed (NOOP) writes are never applied, in shadow arms
    the genuine execution produces its own observe event, so both arm types
    replay correctly with no special-casing.
  * Each ``rehydrate`` is a checkpoint: ``items_loaded`` must equal the carried
    state's size. ``items_loaded == 0`` resets the chain (first prereq entry or
    a fresh re-run appended to the same log). Any other mismatch retroactively
    invalidates the decisions since the previous checkpoint (we cannot know
    which step drifted -- typically an unlogged remove/clear in pre-Step-2a
    logs) and desyncs the chain until the next ``items_loaded == 0`` reset.
    Dropped decisions are counted and reported EXPLICITLY, never silently.
  * Pre-Step-2b logs truncate texts at 300 chars; rows whose reconstructed
    texts hit that boundary are flagged (``maybe_truncated``). Full repair from
    result files is possible but not implemented here -- the affected count is
    part of the report.

Run:
  python bfcl_eval/scripts/replay_geometry_deltaH.py --logs gov_logs/governed_calibrated
Outputs (under --out, default gov_logs/replay_<arm>/):
  replay_decisions.jsonl  -- one row per logged decision (computable or not)
  replay_summary.json     -- per-backend correlations + accounting
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bfcl_eval.model_handler.middleware.probe_gen import generate_item_probes  # noqa: E402
from bfcl_eval.model_handler.middleware.retrieval_sim import (  # noqa: E402
    delta_h_for_probes,
    simulate_kv,
    simulate_vector,
)

TRUNCATION_BOUNDARY = 300  # pre-Step-2b logs cut texts at this length


# ---------------------------------------------------------------------------
# Spearman rank correlation (no scipy dependency; average ranks for ties)
# ---------------------------------------------------------------------------


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx <= 0 or vy <= 0:
        return None
    return cov / (vx * vy) ** 0.5


def median(values):
    if not values:
        return None
    s = sorted(values)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


# ---------------------------------------------------------------------------
# Chain-carry state reconstruction
# ---------------------------------------------------------------------------


def scenario_of(test_id: str) -> str:
    parts = test_id.split("-")
    return parts[1] if len(parts) >= 2 else test_id


class ChainState:
    """Mirror of one (backend, scenario) memory chain."""

    def __init__(self):
        self.tiers = {"core": {}, "archival": {}}  # tier -> {ref: text}
        self.synced = False
        self.started = False
        # Decision rows since the last verified checkpoint; retroactively
        # invalidated if the next rehydrate contradicts the carried state.
        self.pending_rows = []

    def total(self):
        return len(self.tiers["core"]) + len(self.tiers["archival"])

    def reset(self):
        self.tiers = {"core": {}, "archival": {}}
        self.synced = True
        self.started = True


def replay_log(log_path: Path, neighbors_k: int, top_k: int, temperature: float):
    chains = defaultdict(ChainState)
    rows = []  # finalized decision rows
    counters = defaultdict(int)

    def flush(chain, valid: bool):
        for row in chain.pending_rows:
            row["validated"] = valid
            if not valid:
                row["computable"] = False
                row["drop_reason"] = "desync_retroactive"
                counters["dropped_desync"] += 1
        rows.extend(chain.pending_rows)
        chain.pending_rows = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                counters["unparseable_lines"] += 1
                continue
            event = rec.get("event")
            backend = rec.get("backend", "")
            key = (backend, scenario_of(rec.get("test_id", "")))
            chain = chains[key]

            if event == "rehydrate":
                n = int(rec.get("items_loaded", 0))
                if n == 0:
                    flush(chain, valid=chain.synced or not chain.started)
                    chain.reset()
                    counters["checkpoints_reset"] += 1
                elif chain.synced and chain.total() == n:
                    flush(chain, valid=True)  # checkpoint verified
                    counters["checkpoints_verified"] += 1
                else:
                    # Carried state contradicted (or never established): the
                    # decisions since the last checkpoint are unreliable.
                    flush(chain, valid=False)
                    chain.synced = False
                    chain.started = True
                    counters["checkpoints_desync"] += 1

            elif event == "observe":
                tier, ref, text = rec.get("tier"), str(rec.get("ref")), rec.get("text", "")
                if tier in ("core", "archival"):
                    chain.tiers[tier][ref] = text
                    if len(text) == TRUNCATION_BOUNDARY:
                        counters["maybe_truncated_observe"] += 1

            elif event == "observe_remove":
                tier, ref = rec.get("tier"), str(rec.get("ref"))
                if tier in ("core", "archival"):
                    chain.tiers[tier].pop(ref, None)

            elif event == "observe_clear":
                tier = rec.get("tier")
                if tier in ("core", "archival"):
                    chain.tiers[tier] = {}

            elif event == "decision":
                counters["decisions_total"] += 1
                row = {
                    "line_no": line_no,
                    "test_id": rec.get("test_id"),
                    "step_idx": rec.get("step_idx"),
                    "backend": backend,
                    "tier": rec.get("tier"),
                    "op": rec.get("op"),
                    "decision": rec.get("decision"),
                    "reason": rec.get("reason"),
                    "dry_run": rec.get("dry_run", False),
                    "sim_max": rec.get("sim_max"),
                    "r": rec.get("r"),
                    "sim_high": rec.get("sim_high"),
                    "delta": rec.get("delta"),
                    "n_items_logged": rec.get("n_items"),
                    "verbatim_misses": len(rec.get("verbatim_misses") or []),
                    "computable": False,
                    "validated": None,
                    "drop_reason": None,
                    "dH_mean": None,
                    "n_eff_after_mean": None,
                    "n_neighbors": 0,
                    "maybe_truncated": False,
                }
                if not chain.synced:
                    row["drop_reason"] = "desync"
                    counters["dropped_desync"] += 1
                    rows.append(row)
                    continue
                # Cross-check: the mirror size at decision time is logged.
                logged_n = rec.get("n_items")
                if logged_n is not None and int(logged_n) != chain.total():
                    row["drop_reason"] = "n_items_mismatch"
                    counters["dropped_n_items_mismatch"] += 1
                    chain.synced = False
                    rows.append(row)
                    continue
                compute_delta_h(rec, chain, row, neighbors_k, top_k, temperature, counters)
                chain.pending_rows.append(row)

    for chain in chains.values():
        flush(chain, valid=chain.synced or not chain.started)
    return rows, counters


def compute_delta_h(rec, chain, row, neighbors_k, top_k, temperature, counters):
    """Provisionally add the logged candidate to its tier and measure the mean
    dH over the candidate's nearest tier neighbors' own template probes."""
    backend = rec.get("backend")
    tier = rec.get("tier")
    tier_items = chain.tiers.get(tier, {})
    candidate_text = rec.get("candidate_text") or ""
    candidate_ref = rec.get("candidate_ref")
    row["computable"] = True
    counters["decisions_computable"] += 1
    if len(candidate_text) == TRUNCATION_BOUNDARY or any(
        len(t) == TRUNCATION_BOUNDARY for t in tier_items.values()
    ):
        row["maybe_truncated"] = True
        counters["maybe_truncated_rows"] += 1
    if not tier_items:
        row["drop_reason"] = "empty_tier"
        counters["computable_empty_tier"] += 1
        return

    refs = list(tier_items.keys())
    texts = [tier_items[r] for r in refs]
    if backend == "kv":
        corpus = refs  # KV retrieval scores key names only
        candidate_entry = str(candidate_ref)
        ranked = simulate_kv(corpus, candidate_entry, k=min(neighbors_k, len(corpus)))
        neighbor_refs = [key for _, key in ranked]
    else:
        corpus = texts
        candidate_entry = candidate_text
        ranked = simulate_vector(corpus, candidate_entry, k=min(neighbors_k, len(corpus)))
        neighbor_refs = [refs[i] for _, i in ranked]

    dhs, n_effs, per_neighbor = [], [], []
    for nref in neighbor_refs:
        ntext = tier_items[nref]
        probes = [p.text for p in generate_item_probes(backend, ntext, ref=nref)]
        if not probes:
            continue
        res = delta_h_for_probes(
            backend, corpus, candidate_entry, probes, k=top_k, temperature=temperature
        )
        dhs.append(res["dH_mean"])
        n_effs.append(res["n_eff_after_mean"])
        per_neighbor.append({"ref": nref, "dH": res["dH_mean"]})
    if not dhs:
        row["drop_reason"] = "no_probes"
        counters["computable_no_probes"] += 1
        return
    row["n_neighbors"] = len(dhs)
    row["dH_mean"] = round(sum(dhs) / len(dhs), 6)
    row["dH_max"] = round(max(dhs), 6)
    row["n_eff_after_mean"] = round(sum(n_effs) / len(n_effs), 6)
    row["per_neighbor"] = per_neighbor


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize(rows, counters):
    summary = {"accounting": dict(counters), "backends": {}}
    total = counters.get("decisions_total", 0)
    usable = [
        r for r in rows
        if r["computable"] and r.get("validated") is not False and r["dH_mean"] is not None
    ]
    summary["accounting"]["decisions_usable_for_correlation"] = len(usable)
    summary["accounting"]["usable_fraction"] = round(len(usable) / total, 4) if total else None

    for backend in ("kv", "vector"):
        b = [r for r in usable if r["backend"] == backend]
        sims = [r["sim_max"] for r in b]
        rs = [r["r"] for r in b]
        dhs = [r["dH_mean"] for r in b]
        dup = [r for r in b if r["decision"] == "NOOP"]
        nondup = [r for r in b if r["decision"] != "NOOP"]
        noop_region = [
            r for r in b
            if r["sim_max"] is not None and r["sim_high"] is not None
            and r["sim_max"] > r["sim_high"] and r["r"] < r["delta"]
            and r["verbatim_misses"] == 0
        ]
        summary["backends"][backend] = {
            "n_usable": len(b),
            "spearman_sim_max_vs_dH_mean": _round(spearman(sims, dhs)),
            "spearman_r_vs_dH_mean": _round(spearman(rs, dhs)),
            "duplicate_median_sim_max": _round(median([r["sim_max"] for r in dup])),
            "nonduplicate_median_sim_max": _round(median([r["sim_max"] for r in nondup])),
            "duplicate_median_dH": _round(median([r["dH_mean"] for r in dup])),
            "nonduplicate_median_dH": _round(median([r["dH_mean"] for r in nondup])),
            "noop_region_size": len(noop_region),
            "noop_region_purity_decided_noop": _round(
                (sum(1 for r in noop_region if r["decision"] == "NOOP") / len(noop_region))
                if noop_region else None
            ),
            "suppressed_noops": sum(1 for r in dup if not r["dry_run"]),
            "suppressed_noops_dH_ge_0": sum(
                1 for r in dup if not r["dry_run"] and r["dH_mean"] >= 0
            ),
        }
    return summary


def _round(v, nd=4):
    return round(v, nd) if isinstance(v, float) else v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", required=True, help="arm dir (or the .jsonl itself)")
    ap.add_argument("--out", default=None, help="output dir (default gov_logs/replay_<arm>)")
    ap.add_argument("--neighbors", type=int, default=5, help="tier neighbors per candidate")
    ap.add_argument("--top-k", type=int, default=5, help="retrieval top-k for entropy")
    ap.add_argument("--temperature", type=float, default=1.0, help="softmax T (logged H only)")
    args = ap.parse_args()

    log_path = Path(args.logs)
    if log_path.is_dir():
        log_path = log_path / "governance_log.jsonl"
    if not log_path.exists():
        sys.exit(f"log not found: {log_path}")
    arm = log_path.parent.name
    out_dir = Path(args.out) if args.out else log_path.parent.parent / f"replay_{arm}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, counters = replay_log(log_path, args.neighbors, args.top_k, args.temperature)
    summary = summarize(rows, counters)
    summary["arm"] = arm
    summary["log"] = str(log_path)
    summary["params"] = {
        "neighbors": args.neighbors, "top_k": args.top_k, "temperature": args.temperature,
    }

    with open(out_dir / "replay_decisions.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out_dir / "replay_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    total = counters.get("decisions_total", 0)
    usable = summary["accounting"]["decisions_usable_for_correlation"]
    dropped = counters.get("dropped_desync", 0) + counters.get("dropped_n_items_mismatch", 0)
    print(f"[replay] arm={arm} decisions={total} usable={usable} "
          f"({summary['accounting']['usable_fraction']}) dropped_desync={dropped} "
          f"empty_tier={counters.get('computable_empty_tier', 0)} "
          f"no_probes={counters.get('computable_no_probes', 0)}")
    for backend, s in summary["backends"].items():
        print(f"[replay]   {backend}: n={s['n_usable']} "
              f"rho(sim_max,dH)={s['spearman_sim_max_vs_dH_mean']} "
              f"rho(r,dH)={s['spearman_r_vs_dH_mean']} "
              f"dup/nondup median sim={s['duplicate_median_sim_max']}/{s['nonduplicate_median_sim_max']} "
              f"NOOP region={s['noop_region_size']} purity={s['noop_region_purity_decided_noop']}")
    print(f"[replay] wrote {out_dir / 'replay_decisions.jsonl'} and replay_summary.json")


if __name__ == "__main__":
    main()
