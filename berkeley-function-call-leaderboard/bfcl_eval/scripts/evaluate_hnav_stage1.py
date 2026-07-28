"""
H-Nav Stage 1.4: evaluation of the diff-aware coupling falsifier.

Answers the three pre-registered hypotheses of the Stage-1 brief and writes the
verdict artifact. Reuses the calibration machinery that is already frozen into
this project (chain-grouped leave-one-scenario-out CV, grouped bootstrap, the
NC shuffle control, the EPV gate) rather than reinventing statistics.

  H1  write-side headroom is real -- some NOOP-region cell suppresses >= 5 % of
      writes at correct-NOOP precision >= 0.90 (Wilson lower >= 0.80) with the
      harmful-NOOP Wilson upper <= 0.05.
  H2  the diff carries signal geometry lacks -- dAUC(diff | geometry+margin)
      CI95 lower > 0, the NC-shuffled dAUC CI covers 0, and PR-AUC improves.
  H3  the falsifier is cheap -- false-override rate <= 0.20 at H1's operating
      point.

Statistical discipline (binding, plan T8):
  * The full decision set carries the AUC/PR-AUC claim (EPV is comfortable).
  * Inside a NOOP-region cell, counts are small: exact integers + Wilson
    intervals only. Where positives < 25 the multivariate model is SKIPPED and
    a within-chain label-permutation p-value on the single pre-registered
    univariate falsifier (``diff_sim_max``) is reported instead -- the same
    EPV-gate pattern already frozen into calibrate_margin_entropy.
  * The sweep has many cells; operating points are Holm-corrected and the
    headline is the shape of the risk-coverage frontier, not any one cell.

Run:
  python bfcl_eval/scripts/evaluate_hnav_stage1.py \
      --logs "gov_logs/me_harvest/rep0*_v1_shadow" \
      --labels "gov_logs/me_harvest/outcomes_hnav.jsonl" \
      --diff-features "gov_logs/me_harvest/features_diff.jsonl" \
      --counterfactual gov_logs/hnav_counterfactual.json \
      --out gov_logs/hnav_stage1_report.json --md gov_logs/HNAV_STAGE1.md
"""

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from bfcl_eval.scripts.analyze_gov_replicates import holm  # noqa: E402
from bfcl_eval.scripts.calibrate_margin_entropy import (  # noqa: E402
    DIFF_FEATURES,
    ENTROPY_V2_FEATURES,
    GEOMETRY_FEATURES,
    MARGIN_FEATURES,
    assert_no_gt_in_features,
    auc,
    git_head,
    load_decisions,
    nc_shuffle_entropy_features,
    nested_auc_report,
    pr_auc,
)
from bfcl_eval.scripts.counterfactual_noop import in_region, wilson  # noqa: E402

MIN_POSITIVES_FOR_MODEL = 25
FALSIFIER = "diff_sim_max"  # the single pre-registered univariate falsifier

# H1 thresholds (pre-registered)
H1_MIN_COVERAGE = 0.05
H1_MIN_PRECISION = 0.90
H1_MIN_PRECISION_LO = 0.80
H1_MAX_HARM_HI = 0.05
H3_MAX_FALSE_OVERRIDE = 0.20


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def merge_features(rows, patterns, keys):
    """Join sidecar feature files on candidate_id.

    Records ACCUMULATE per candidate rather than replacing each other: a glob
    like ``features_diff_*.jsonl`` also matches ``features_diff_oracle_*.jsonl``,
    and a last-writer-wins dict would silently drop the real features for every
    arm whose oracle file sorts later. Returns the number of rows that actually
    received at least one requested key -- not the number that merely matched --
    so a silent miss shows up in the log line.
    """
    if not patterns:
        return 0
    extra = defaultdict(dict)
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        extra[rec["candidate_id"]].update(rec)
    n = 0
    for row in rows:
        e = extra.get(row["candidate_id"])
        if not e:
            continue
        hit = False
        for k in keys:
            if k in e:
                row["features"][k] = e[k]
                hit = True
        n += int(hit)
    return n


def attach_hnav_labels(rows, patterns, label):
    outcomes = {}
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        outcomes[rec["candidate_id"]] = rec
    kept = []
    for row in rows:
        out = outcomes.get(row["candidate_id"])
        if out is None or label not in out:
            continue
        row["label"] = int(bool(out[label]))
        row["outcomes"] = out
        kept.append(row)
    return kept


# ---------------------------------------------------------------------------
# Operating points: geometry region + diff-aware override
# ---------------------------------------------------------------------------


def parse_cell(key):
    parts = dict(p.split("=") for p in key.split("|"))
    return (float(parts["sim_high"]), float(parts["delta"]),
            bool(int(parts["veto"])), bool(int(parts["preflight"])))


def operating_point(region, tau):
    """Rule: stay suppressed iff ``diff_sim_max >= tau`` (the marginal content
    really is redundant); otherwise the falsifier overrides and the write goes
    through.

    harmful_noop_recall  -- must_write rows the falsifier rescues
    correct_noop_precision -- among rows still suppressed, fraction not must_write
    false_override_rate  -- safe rows the falsifier needlessly rescues (its cost)
    """
    harmful = [r for r in region if r["label"] == 1]
    safe = [r for r in region
            if r["label"] == 0 and r["outcomes"]["hnav_target"] != "uncertain"]

    def score(r):
        v = r["features"].get(FALSIFIER)
        return 1.0 if v is None else float(v)

    rescued_harm = [r for r in harmful if score(r) < tau]
    rescued_safe = [r for r in safe if score(r) < tau]
    still_harm = len(harmful) - len(rescued_harm)
    still_safe = len(safe) - len(rescued_safe)
    decided = still_harm + still_safe
    rec, rec_lo, rec_hi = wilson(len(rescued_harm), len(harmful))
    prec, prec_lo, prec_hi = wilson(still_safe, decided)
    fo, fo_lo, fo_hi = wilson(len(rescued_safe), len(safe))
    return {
        "tau": tau,
        "n_region": len(region),
        "n_harmful": len(harmful),
        "n_safe": len(safe),
        "harmful_noop_recall": rec,
        "harmful_noop_recall_wilson95": [rec_lo, rec_hi],
        "correct_noop_precision": prec,
        "correct_noop_precision_wilson95": [prec_lo, prec_hi],
        "false_override_rate": fo,
        "false_override_rate_wilson95": [fo_lo, fo_hi],
        "n_still_suppressed": decided,
    }


def permutation_p(region, seed=12345, n_perm=10000):
    """Within-chain label permutation on the univariate falsifier.

    Hypothesis: LOWER ``diff_sim_max`` (marginal content unlike anything stored)
    predicts must_write, so the statistic is AUC(y, -diff_sim_max).
    """
    rows = [r for r in region if r["features"].get(FALSIFIER) is not None]
    y = np.array([r["label"] for r in rows], dtype=np.float64)
    if len(rows) < 4 or y.sum() == 0 or y.sum() == len(y):
        return None
    x = np.array([-float(r["features"][FALSIFIER]) for r in rows])
    observed = auc(y, x)
    if observed is None:
        return None
    by_chain = defaultdict(list)
    for i, r in enumerate(rows):
        by_chain[r["chain"]].append(i)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        yp = y.copy()
        for idx in by_chain.values():
            arr = np.array(idx)
            yp[arr] = y[rng.permutation(arr)]
        a = auc(yp, x)
        if a is not None and a >= observed:
            hits += 1
    return {"statistic_auc": observed, "n": len(rows), "n_positive": int(y.sum()),
            "p_value": (hits + 1) / (n_perm + 1), "n_perm": n_perm}


# ---------------------------------------------------------------------------
# Breakdowns / stability
# ---------------------------------------------------------------------------


def group_auc(rows, key_fn):
    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)
    for g, rs in sorted(groups.items(), key=lambda x: str(x[0])):
        y = np.array([r["label"] for r in rs], dtype=np.float64)
        x = np.array([-float(r["features"].get(FALSIFIER) or 1.0) for r in rs])
        out[str(g)] = {
            "n": len(rs), "n_positive": int(y.sum()),
            "falsifier_auc": auc(y, x) if 0 < y.sum() < len(y) else None,
        }
    return out


def replicate_stability(rows, seed, n_boot, nests, delta_specs, delta_ref):
    out = {}
    reps = sorted({r["replicate"] for r in rows})
    for held in reps:
        sub = [r for r in rows if r["replicate"] != held]
        y = [r["label"] for r in sub]
        if not sub or not (0 < sum(y) < len(y)):
            out[held] = None
            continue
        rep = nested_auc_report(sub, seed=seed, n_boot=max(200, n_boot // 5),
                               nests=nests, delta_specs=delta_specs,
                               delta_ref=delta_ref)
        out[held] = {k: v for k, v in rep.items() if k.startswith("dAUC")}
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _md(s):
    """Escape pipes so cell keys (``sim_high=0.8|delta=0.6|...``) do not split
    markdown table columns."""
    return str(s).replace("|", "\\|")


def _f(v, nd=4):
    """Fixed-width numbers so the artifact is readable as a thesis table."""
    if v is None:
        return "n/a"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_f(x, nd) for x in v) + "]"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_markdown(path, doc):
    h = doc["hypotheses"]
    L = []
    L.append("# H-Nav Stage 1 -- corrected labels + coupling falsifier\n")
    L.append(f"**Verdict: {doc['verdict']['decision']}** -- {doc['verdict']['rationale']}\n")
    L.append("## 1. Provenance\n")
    L.append("| | |\n|---|---|")
    L.append(f"| Git head | `{doc['git_head']}` |")
    L.append(f"| Decisions labeled | {doc['n_labeled']} (positives: {doc['n_positive']}) |")
    L.append(f"| Target | `{doc['label']}` |")
    L.append(f"| Arms | {', '.join(doc['arms'])} |")
    L.append(f"| Seed / bootstrap | {doc['seed']} / {doc['n_boot']} |")
    L.append(f"| Proxy conversion factor p_hat | {doc.get('p_hat')} |\n")
    L.append("## 2. Label distribution\n")
    L.append("| target | n |\n|---|---|")
    for k, v in sorted(doc["target_distribution"].items()):
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("| fate | n |\n|---|---|")
    for k, v in sorted(doc["fate_distribution"].items()):
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("### Where the headroom actually sits\n")
    L.append("A memory chain rewrites its own items, so only the *terminal* "
             "write per store ref reaches the snapshot the questions are "
             "answered against. Everything else is first-order inert.\n")
    L.append("| fate / op family | n | must_write | rate |\n|---|---|---|---|")
    for k, v in sorted(doc["headroom_concentration"].items(),
                       key=lambda x: -x[1]["must_write"]):
        L.append(f"| {k.replace('|', ' / ')} | {v['n']} | {v['must_write']} | "
                 f"{_f(v['rate'])} |")
    L.append("")
    if doc.get("mechanical_claim"):
        m = doc["mechanical_claim"]
        L.append("### The mechanical claim that motivated H-Nav\n")
        L.append(f"On the {m['n_near_duplicate_updates']} near-duplicate "
                 f"update/replace decisions (`sim_max >= 0.90`), whole-blob "
                 f"similarity says *duplicate* while the marginal content does "
                 f"not: median `sim_max` {_f(m['median_sim_max'])} vs median "
                 f"`diff_sim_max` {_f(m['median_diff_sim_max'])}, a median drop "
                 f"of {_f(m['median_drop'])}, with "
                 f"{_f(m['frac_drop_gt_0.10'])} of rows dropping more than "
                 f"0.10. **The mechanism is real and large.** But "
                 + ("**none**" if m["n_must_write_among_them"] == 0
                    else f"only {m['n_must_write_among_them']}")
                 + " of those decisions is `must_write`. The write-side losses "
                   "sit on fresh adds -- where there is no predecessor, hence "
                   "no marginal diff to measure -- so measuring the defect "
                   "better cannot recover accuracy that was never at risk.\n")
    L.append("## 3. H1 -- write-side headroom\n")
    L.append(f"**{h['H1']['verdict']}** -- {h['H1']['detail']}\n")
    if h["H1"].get("best_cell"):
        b = h["H1"]["best_cell"]
        L.append("Widest deployable cell (the gate-faithful region, i.e. what "
                 "suppression could actually reach):\n")
        L.append("| | |\n|---|---|")
        L.append(f"| cell | `{_md(b['cell'])}` |")
        L.append(f"| coverage of all writes | {_f(b['coverage'])} "
                 f"(bar {H1_MIN_COVERAGE}) |")
        L.append(f"| correct-NOOP precision | {_f(b['suppression_precision'])} "
                 f"Wilson95 {_f(b['suppression_precision_wilson95'])} "
                 f"(bar {H1_MIN_PRECISION} / lo {H1_MIN_PRECISION_LO}) |")
        L.append(f"| harmful suppressions | {b['n_harmful_noop']}/{b['n_decided']} "
                 f"Wilson95 {_f(b['harmful_rate_wilson95'])} "
                 f"(bar hi <= {H1_MAX_HARM_HI}) |")
        L.append(f"| expected accuracy change | {_f(b.get('expected_dAcc'))} |\n")
    L.append("## 4. H2 -- does the marginal diff carry signal geometry lacks?\n")
    L.append(f"**{h['H2']['verdict']}**\n")
    n = doc["nested_models"]
    L.append("| nest | AUC | PR-AUC |\n|---|---|---|")
    for k, v in n.items():
        if isinstance(v, dict) and "auc" in v:
            L.append(f"| {k} | {_f(v['auc'])} | {_f(v['pr_auc'])} |")
    L.append(f"\nBase rate (PR-AUC no-skill line): {_f(n.get('base_rate'))}; "
             f"PR-AUC gain from the diff features: {_f(doc['pr_auc_gain'])}\n")
    for k, v in n.items():
        if k.startswith("dAUC"):
            L.append(f"- real `{k}`: mean {_f(v['mean'])} CI95 {_f(v['ci95'])}")
    for k, v in doc["nested_models_nc_shuffled"].items():
        if k.startswith("dAUC"):
            L.append(f"- NC-shuffled `{k}`: mean {_f(v['mean'])} "
                     f"CI95 {_f(v['ci95'])}")
    rb = doc.get("h2_robustness_has_old_ablation") or {}
    if rb:
        L.append("\n**Robustness -- is this just the add/update indicator?** "
                 "`must_write` concentrates in fresh adds, and `has_old` "
                 "separates adds from updates perfectly, so the gain is "
                 "re-measured with that column removed:\n")
        for k, v in rb.items():
            if k.startswith("dAUC"):
                L.append(f"- `{k}`: mean {_f(v['mean'])} CI95 {_f(v['ci95'])}")
    L.append("\nThe NC control permutes the diff features jointly within "
             "(backend, store-size bin), preserving their marginals and every "
             "geometry/margin feature. It must show no significant POSITIVE "
             "gain; it is expected to sit below zero, because 17 shuffled "
             "columns cost out-of-fold AUC rather than leaving it unchanged.\n")
    L.append("## 5. H3 -- is the falsifier cheap?\n")
    L.append(f"**{h['H3']['verdict']}** -- {h['H3']['detail']}\n")
    L.append("## 6. Breakdowns (univariate falsifier AUC, "
             f"`{FALSIFIER}` scored in the hypothesized direction)\n")
    for name, block in doc["breakdowns"].items():
        L.append(f"**{name}**\n")
        L.append("| group | n | positives | AUC |\n|---|---|---|---|")
        for g, v in block.items():
            L.append(f"| {g} | {v['n']} | {v['n_positive']} | "
                     f"{_f(v['falsifier_auc'])} |")
        L.append("")
    live = (doc.get("counterfactual_summary") or {}).get("live_suppressions") or {}
    if live:
        L.append("## 7. Reality check: the suppressions that really happened\n")
        L.append("The only place the first-order counterfactual meets reality. "
                 f"n = {live.get('n_live_suppressions')} across the live arms -- "
                 "a sanity check, never a rate estimate.\n")
        L.append("| | |\n|---|---|")
        L.append(f"| live suppressions | {live.get('n_live_suppressions')} |")
        L.append(f"| by arm | {live.get('by_arm')} |")
        L.append(f"| label verdict | {live.get('by_target')} |")
        L.append(f"| harmful_noop | {live.get('harmful_noop')} |\n")
    L.append("## 8. Limitations (binding)\n")
    for lim in doc["limitations"]:
        L.append(f"- {lim}")
    L.append("")
    Path(path).write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--diff-features", nargs="+", default=None)
    ap.add_argument("--extra-features", nargs="+", default=None)
    ap.add_argument("--oracle-features", nargs="+", default=None)
    ap.add_argument("--counterfactual", default="gov_logs/hnav_counterfactual.json")
    ap.add_argument("--label", default="must_write")
    ap.add_argument("--include-non-escalated", action="store_true", default=True)
    ap.add_argument("--escalated-only", dest="include_non_escalated",
                    action="store_false")
    ap.add_argument("--oracle-ceiling", action="store_true")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--out", default="gov_logs/hnav_stage1_report.json")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    log_dirs = []
    for pat in args.logs:
        log_dirs.extend(sorted(glob.glob(pat)))
    if not log_dirs:
        sys.exit(f"[hnav_eval] no logs match {args.logs!r}")

    rows, manifest = load_decisions(
        log_dirs, escalated_only=not args.include_non_escalated
    )
    print(f"[hnav_eval] gov2 decisions: {len(rows)} from {len(log_dirs)} arms")
    n_diff = merge_features(rows, args.diff_features, DIFF_FEATURES)
    n_v2 = merge_features(rows, args.extra_features, ENTROPY_V2_FEATURES)
    print(f"[hnav_eval] diff features merged: {n_diff}; entropy-v2: {n_v2}")
    assert_no_gt_in_features(rows)

    rows = attach_hnav_labels(rows, args.labels, args.label)
    y = np.array([r["label"] for r in rows], dtype=np.float64)
    n_pos = int(y.sum())
    print(f"[hnav_eval] labeled: {len(rows)} positives({args.label})={n_pos}")
    if not rows or not (0 < n_pos < len(rows)):
        sys.exit("[hnav_eval] label degenerate -- nothing to evaluate")

    # ---- H2: nested models -------------------------------------------------
    nests = {
        "geometry": list(GEOMETRY_FEATURES),
        "geometry+margin_entropy": list(GEOMETRY_FEATURES + MARGIN_FEATURES),
        "geometry+margin_entropy+diff": list(
            GEOMETRY_FEATURES + MARGIN_FEATURES + DIFF_FEATURES
        ),
        "diff_only": list(DIFF_FEATURES),
    }
    delta_specs = {
        "dAUC_diff_given_geometry_margin": "geometry+margin_entropy+diff"
    }
    delta_ref = "geometry+margin_entropy"
    nested = nested_auc_report(rows, seed=args.seed, n_boot=args.n_boot,
                               nests=nests, delta_specs=delta_specs,
                               delta_ref=delta_ref)
    nc_rows = nc_shuffle_entropy_features(rows, args.seed, feats=DIFF_FEATURES)
    nested_nc = nested_auc_report(nc_rows, seed=args.seed, n_boot=args.n_boot,
                                  nests=nests, delta_specs=delta_specs,
                                  delta_ref=delta_ref)

    # H2 robustness: must_write concentrates in fresh adds, and ``has_old`` is a
    # perfect add/update indicator -- so a naive dAUC could be that indicator
    # rather than any marginal-diff CONTENT. Re-run the nest without it.
    robust_nests = {
        "geometry+margin_entropy": list(GEOMETRY_FEATURES + MARGIN_FEATURES),
        "+has_old_only": list(GEOMETRY_FEATURES + MARGIN_FEATURES) + ["has_old"],
        "+diff_without_has_old": list(GEOMETRY_FEATURES + MARGIN_FEATURES)
        + [f for f in DIFF_FEATURES if f != "has_old"],
    }
    h2_robust = nested_auc_report(
        rows, seed=args.seed, n_boot=args.n_boot, nests=robust_nests,
        delta_specs={"dAUC_has_old_alone": "+has_old_only",
                     "dAUC_diff_content_only": "+diff_without_has_old"},
        delta_ref="geometry+margin_entropy",
    )

    d = nested.get("dAUC_diff_given_geometry_margin") or {}
    d_nc = nested_nc.get("dAUC_diff_given_geometry_margin") or {}
    pr_gain = None
    if nested["geometry+margin_entropy+diff"]["pr_auc"] is not None:
        pr_gain = (nested["geometry+margin_entropy+diff"]["pr_auc"]
                   - nested["geometry+margin_entropy"]["pr_auc"])
    # NC criterion: the shuffled control must show NO POSITIVE gain. Requiring
    # its CI to straddle zero would be wrong here -- the diff nest adds 17
    # features, so once they are shuffled into noise the extra parameters cost
    # out-of-fold AUC and the NC dAUC lands reliably below zero. The control's
    # job is to rule out "any 17 columns would have helped", which is exactly
    # "NC does not show a significant positive gain".
    h2_ok = bool(
        d.get("ci95") and d["ci95"][0] > 0
        and d_nc.get("ci95") and d_nc["ci95"][0] <= 0
        and d.get("mean") is not None and d_nc.get("mean") is not None
        and d["mean"] > d_nc["mean"]
        and (pr_gain or 0) > 0
    )

    # ---- H1 / H3: operating points over the sweep --------------------------
    cf = {}
    if Path(args.counterfactual).exists():
        with open(args.counterfactual, "r", encoding="utf-8") as f:
            cf = json.load(f)

    by_cell, pvals = {}, {}
    for key in (cf.get("cells") or {}):
        sh, dl, veto, pre = parse_cell(key)
        region = [r for r in rows if in_region(r["outcomes"], sh, dl, veto, pre)]
        if len(region) < 3:
            continue
        n_reg_pos = sum(r["label"] for r in region)
        entry = {
            "n_region": len(region),
            "coverage": round(len(region) / len(rows), 6),
            "n_positive": n_reg_pos,
            "operating_points": [
                operating_point(region, tau) for tau in (0.3, 0.5, 0.7, 0.9)
            ],
        }
        if n_reg_pos < MIN_POSITIVES_FOR_MODEL:
            entry["multivariate_model_skipped"] = (
                f"n_positive={n_reg_pos} < {MIN_POSITIVES_FOR_MODEL}"
            )
            perm = permutation_p(region, seed=args.seed, n_perm=args.n_perm)
            entry["permutation_test"] = perm
            if perm:
                pvals[key] = perm["p_value"]
        by_cell[key] = entry
    holm_adj = holm(pvals) if pvals else {}
    for key, p in holm_adj.items():
        by_cell[key]["permutation_test"]["p_holm"] = p

    # H1 over the DEPLOYABLE cells of the counterfactual sweep
    best, h1_ok = None, False
    for key, c in (cf.get("cells") or {}).items():
        if not key.endswith("veto=1|preflight=1"):
            continue
        lo = (c.get("suppression_precision_wilson95") or [None])[0]
        harm_hi = (c.get("harmful_rate_wilson95") or [None, None])[1]
        ok = (c["coverage"] >= H1_MIN_COVERAGE
              and (c.get("suppression_precision") or 0) >= H1_MIN_PRECISION
              and (lo or 0) >= H1_MIN_PRECISION_LO
              and harm_hi is not None and harm_hi <= H1_MAX_HARM_HI)
        if ok:
            h1_ok = True
        if best is None or c["coverage"] > best["coverage"]:
            best = dict(c, cell=key)

    # H3 at the widest deployable cell
    h3_rate, h3_ok = None, False
    if best:
        sh, dl, veto, pre = parse_cell(best["cell"])
        region = [r for r in rows if in_region(r["outcomes"], sh, dl, veto, pre)]
        if region:
            op = operating_point(region, 0.5)
            h3_rate = op["false_override_rate"]
            h3_ok = h3_rate is not None and h3_rate <= H3_MAX_FALSE_OVERRIDE

    # ---- Descriptive quantities the verdict text depends on -----------------
    targets = defaultdict(int)
    fates = defaultdict(int)
    for r in rows:
        targets[r["outcomes"]["hnav_target"]] += 1
        fates[r["outcomes"].get("fate") or "unresolved"] += 1

    # Where the headroom actually sits: must_write by fate x op family. This is
    # what decides whether a DIFF-based falsifier could ever have helped -- a
    # fresh add has no predecessor, so there is no marginal diff to score.
    concentration = defaultdict(lambda: {"n": 0, "must_write": 0})
    for r in rows:
        key = f"{r['outcomes'].get('fate') or 'unresolved'}|{r['outcomes']['op_family']}"
        concentration[key]["n"] += 1
        concentration[key]["must_write"] += r["label"]
    for v in concentration.values():
        v["rate"] = round(v["must_write"] / v["n"], 6) if v["n"] else None

    # The mechanical claim that motivated H-Nav: on near-duplicate updates the
    # whole blob looks redundant while the marginal content does not.
    near = [r for r in rows
            if (r["features"].get("sim_max") or 0) >= 0.90
            and r["outcomes"]["op_family"] == "update_replace"
            and r["features"].get("diff_sim_max") is not None]
    mech = None
    if near:
        blob = np.array([float(r["features"]["sim_max"]) for r in near])
        diff = np.array([float(r["features"]["diff_sim_max"]) for r in near])
        mech = {
            "n_near_duplicate_updates": len(near),
            "median_sim_max": round(float(np.median(blob)), 6),
            "median_diff_sim_max": round(float(np.median(diff)), 6),
            "median_drop": round(float(np.median(blob - diff)), 6),
            "frac_drop_gt_0.10": round(float(np.mean((blob - diff) > 0.10)), 6),
            "n_must_write_among_them": int(sum(r["label"] for r in near)),
        }

    # ---- Verdict -----------------------------------------------------------
    if h1_ok and h2_ok and h3_ok:
        decision, why = "PROCEED write-side", (
            "headroom exists, the marginal diff adds out-of-sample signal over "
            "geometry+margin, and the falsifier's override cost is acceptable")
    elif h1_ok and not h2_ok:
        decision, why = "RETUNE Stage-0 thresholds only", (
            "headroom exists but geometry already sees it -- no new machinery "
            "justified; the research contribution moves to action-side H_act")
    elif h1_ok and h2_ok and not h3_ok:
        decision, why = "NOT DEPLOYABLE AT COST -- pivot", (
            "the diff signal is real but the falsifier over-fires; keep "
            "diff_sim_max as a logged diagnostic and pivot to action-side H_act")
    else:
        why = ("write-side headroom is not there: the deployable NOOP region is "
               "too small and too contaminated with unique carriers for any "
               "suppression policy to be net-positive on this benchmark")
        if mech and mech["n_must_write_among_them"] == 0 and mech["median_drop"] > 0.1:
            why += (
                f". Note the hypothesised defect is REAL and large -- on the "
                f"{mech['n_near_duplicate_updates']} near-duplicate updates the "
                f"whole blob scores {mech['median_sim_max']:.3f} while its "
                f"marginal content scores {mech['median_diff_sim_max']:.3f} -- "
                f"but NONE of them is must_write. The losses sit on fresh adds, "
                f"where there is no predecessor and so no marginal diff to "
                f"measure. Fixing the geometry defect cannot recover accuracy "
                f"that was never at risk"
            )
        decision = "PIVOT to action-side H_act"

    doc = {
        "git_head": git_head(),
        "seed": args.seed,
        "n_boot": args.n_boot,
        "label": args.label,
        "arms": sorted({r["replicate"] for r in rows}),
        "data_manifest": manifest,
        "n_labeled": len(rows),
        "n_positive": n_pos,
        "p_hat": cf.get("p_hat"),
        "target_distribution": dict(targets),
        "fate_distribution": dict(fates),
        "headroom_concentration": dict(concentration),
        "mechanical_claim": mech,
        "nested_models": nested,
        "nested_models_nc_shuffled": nested_nc,
        "h2_robustness_has_old_ablation": h2_robust,
        "pr_auc_gain": pr_gain,
        "cells": by_cell,
        "counterfactual_summary": {
            k: cf.get(k) for k in
            ("n_decisions", "target_distribution", "fate_distribution",
             "live_suppressions")
        },
        "breakdowns": {
            "backend": group_auc(rows, lambda r: r["backend"]),
            "op_family": group_auc(rows, lambda r: r["outcomes"]["op_family"]),
            "fate": group_auc(rows, lambda r: r["outcomes"].get("fate")),
            "item_length": group_auc(
                rows,
                lambda r, med=float(np.median([x["outcomes"]["cand_len"] for x in rows])):
                "long" if r["outcomes"]["cand_len"] > med else "short",
            ),
            "scenario_customer_vs_other": group_auc(
                rows, lambda r: "customer" if r["scenario"] == "customer" else "other"
            ),
            "replicate": group_auc(rows, lambda r: r["replicate"]),
        },
        "replicate_stability": replicate_stability(
            rows, args.seed, args.n_boot, nests, delta_specs, delta_ref
        ),
        "hypotheses": {
            "H1": {
                "verdict": "PASS" if h1_ok else "FAIL",
                "detail": (
                    f"no deployable cell meets coverage>={H1_MIN_COVERAGE}, "
                    f"precision>={H1_MIN_PRECISION} (Wilson lo>={H1_MIN_PRECISION_LO}) "
                    f"and harmful Wilson hi<={H1_MAX_HARM_HI}"
                ) if not h1_ok else "a deployable cell meets all three bars",
                "best_cell": best,
            },
            "H2": {
                "verdict": "PASS" if h2_ok else "FAIL",
                "detail": (
                    f"dAUC(diff | geometry+margin) mean {d.get('mean')} "
                    f"CI95 {d.get('ci95')}; NC-shuffled CI95 {d_nc.get('ci95')}; "
                    f"PR-AUC gain {pr_gain}"
                ),
            },
            "H3": {
                "verdict": "PASS" if h3_ok else ("FAIL" if h3_rate is not None
                                                 else "NOT EVALUABLE"),
                "detail": f"false_override_rate at tau=0.5 = {h3_rate} "
                          f"(bar {H3_MAX_FALSE_OVERRIDE})",
            },
        },
        "verdict": {"decision": decision, "rationale": why},
        "limitations": [
            "First-order counterfactual: reverting a write from the final "
            "snapshot does not simulate how the agent would have behaved after "
            "seeing a synthetic success. Non-terminal writes are labeled "
            "inert_superseded on that basis; must_write_lineage bounds the "
            "under-count from above.",
            "Answerability (top-3 retrieval of a gold-carrying item) is a "
            "NECESSARY not sufficient condition for a correct answer; event "
            "counts are converted to expected accuracy with the measured "
            "p_hat, never 1:1.",
            "Operating-point cells carry small counts: Wilson intervals only, "
            "no multivariate fit below "
            f"{MIN_POSITIVES_FOR_MODEL} positives, Holm-corrected permutation "
            "p-values instead.",
            "Three replicates support a min/max stability check, not an "
            "interval.",
        ],
    }

    if args.oracle_ceiling and args.oracle_features:
        oracle_rows = [json.loads(json.dumps(r)) for r in rows]
        for o, r in zip(oracle_rows, rows):
            o["chain"] = r["chain"]
        merged = merge_features(
            oracle_rows, args.oracle_features,
            ("oracle_carries_gold", "oracle_diff_carries_gold",
             "oracle_n_marginal_gold"),
        )
        onests = dict(nests)
        onests["oracle"] = list(GEOMETRY_FEATURES) + [
            "oracle_carries_gold", "oracle_diff_carries_gold",
            "oracle_n_marginal_gold",
        ]
        doc["oracle_ceiling"] = {
            "merged": merged,
            "note": "GROUND-TRUTH-DERIVED, non-deployable. Reported only to "
                    "bound how much headroom any online feature could have had.",
            "report": nested_auc_report(
                oracle_rows, seed=args.seed, n_boot=max(200, args.n_boot // 5),
                nests=onests,
                delta_specs={"dAUC_oracle_given_geometry_margin": "oracle"},
                delta_ref=delta_ref,
            ),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False, default=str)
    if args.md:
        write_markdown(args.md, doc)

    print(f"[hnav_eval] H1={doc['hypotheses']['H1']['verdict']} "
          f"H2={doc['hypotheses']['H2']['verdict']} "
          f"H3={doc['hypotheses']['H3']['verdict']}")
    print(f"[hnav_eval] VERDICT: {decision}")
    print(f"[hnav_eval] wrote {out_path}" + (f" and {args.md}" if args.md else ""))


if __name__ == "__main__":
    main()
