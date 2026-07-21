"""
Margin-entropy calibration pipeline (Plan v2 Step 8, plan SS13 / SS18 / SS19).

Consumes gov2 harvest logs (shadow arms with full Stage-1 signal logging) plus
a post-hoc ``outcomes.jsonl`` (labels joined on ``candidate_id`` -- produced
offline, AFTER the runs, from score files and final snapshots; never read at
runtime), and produces:

  1. The assembled decision dataset (escalated decisions with features).
  2. Leakage checks: feature whitelist (no question/GT fields can enter),
     chain-grouped leave-one-scenario-out split integrity.
  3. Nested predictive models (gate G1): geometry < geometry+margin <
     geometry+entropy < geometry+margin+entropy, with grouped-bootstrap
     dAUC CIs (chain-level resampling).
  4. Option-A threshold selection by risk-coverage (selective risk at maximal
     coverage under --target-risk), per backend.
  5. Option-B monotone logistic fit (sign-constrained projected gradient),
     isotonic-calibrated, with risk thresholds from the risk-coverage curve.
  6. ``--freeze``: commits the calibration to
     gov_logs/margin_entropy_calibration.json + CALIBRATION_FROZEN.md with the
     git hash and data manifest (SS13.3 ceremony). Refuses to overwrite an
     existing frozen file (delete it deliberately if re-freezing).

Splits and labels follow SS13.1/SS13.2 verbatim; the exploratory-set rule is
enforced socially (gov_logs/margin_entropy_exploratory_set.md), not here.

Run:
  python bfcl_eval/scripts/calibrate_margin_entropy.py \
      --logs "gov_logs/me_harvest/rep0*" --outcomes gov_logs/me_harvest/outcomes.jsonl \
      [--label harmful_write] [--target-risk 0.15] [--seed 12345] [--freeze]
"""

import argparse
import glob
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

# Online feature whitelist (plan SS13.2 leakage control). Anything outside this
# list is REFUSED as a model feature; benchmark question/GT fields can never
# appear because they are not in the gov2 online record at all.
FEATURE_KEYS = (
    "sim_max",
    "r",
    "tau_t",
    "rho",
    "n_items",
    "rank_self_med",
    "nmargin_med",
    "dH_self",
    "dH_mean",
    "n_eff_norm",
    "disp",
    "churn",
    "smallstore",
    "backend_kv",
)
FORBIDDEN_FEATURE_TOKENS = ("question", "ground_truth", "answer", "score", "accuracy")

GEOMETRY_FEATURES = ("sim_max", "r", "tau_t", "rho", "n_items", "backend_kv")
MARGIN_FEATURES = ("rank_self_med", "nmargin_med")
ENTROPY_FEATURES = ("dH_self", "dH_mean", "n_eff_norm", "churn")

# Option-B monotone sign constraints (plan SS9.4): harm increases as margin
# decreases, rank increases, dH increases. +1 -> coef >= 0, -1 -> coef <= 0,
# 0 -> unconstrained.
OPTION_B_FEATURES = ("nmargin_med", "rank_self_med", "dH_mean", "n_eff_norm", "sim_max", "backend_kv")
OPTION_B_SIGNS = {"nmargin_med": -1, "rank_self_med": +1, "dH_mean": +1,
                  "n_eff_norm": +1, "sim_max": 0, "backend_kv": 0}

DEAD_SCENARIOS = ("student",)  # pre-registered dead chain (plan SS13.1)


def scenario_of(test_id: str) -> str:
    parts = (test_id or "").split("-")
    return parts[1] if len(parts) >= 2 else str(test_id)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------


def load_decisions(log_dirs):
    """Escalated gov2 decisions -> feature rows. Chain = (backend, scenario,
    replicate); replicate inferred from the arm directory name."""
    rows = []
    manifest = []
    for d in log_dirs:
        d = Path(d)
        log_path = d / "governance_log.jsonl" if d.is_dir() else d
        if not log_path.exists():
            sys.exit(f"[calibrate] log not found: {log_path}")
        manifest.append({"log": str(log_path), "sha": sha_file(log_path)})
        replicate = d.name if d.is_dir() else d.parent.name
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") != "decision" or rec.get("schema") != "gov2":
                    continue
                if not rec.get("escalated"):
                    continue  # Stage-1 calibration runs on escalations only
                scenario = scenario_of(rec.get("test_id", ""))
                if scenario in DEAD_SCENARIOS:
                    continue
                s1 = rec.get("s1_me") or {}
                row = {
                    "candidate_id": rec.get("candidate_id"),
                    "backend": rec.get("backend"),
                    "scenario": scenario,
                    "replicate": replicate,
                    "chain": (rec.get("backend"), scenario, replicate),
                    "action": rec.get("action"),
                    "reason_code": rec.get("reason_code"),
                    "features": {
                        "sim_max": rec.get("sim_max"),
                        "r": rec.get("r"),
                        "tau_t": rec.get("tau_t"),
                        "rho": rec.get("rho"),
                        "n_items": rec.get("n_items"),
                        "rank_self_med": s1.get("rank_self_med"),
                        "nmargin_med": s1.get("nmargin_med"),
                        "dH_self": s1.get("dH_self"),
                        "dH_mean": s1.get("dH_mean"),
                        "n_eff_norm": s1.get("n_eff_norm"),
                        "disp": s1.get("disp"),
                        "churn": s1.get("churn"),
                        "smallstore": 1.0 if s1.get("smallstore") else 0.0,
                        "backend_kv": 1.0 if rec.get("backend") == "kv" else 0.0,
                    },
                }
                rows.append(row)
    return rows, manifest


def assert_no_gt_in_features(rows):
    """SS13.2: the online feature vector may never contain benchmark question /
    ground-truth fields. Enforced structurally (whitelist) + lexically."""
    for row in rows:
        for key in row["features"]:
            if key not in FEATURE_KEYS:
                raise AssertionError(f"non-whitelisted feature key: {key!r}")
            low = key.lower()
            for tok in FORBIDDEN_FEATURE_TOKENS:
                if tok in low:
                    raise AssertionError(f"forbidden feature key: {key!r}")


def attach_outcomes(rows, outcomes_path, label):
    with open(outcomes_path, "r", encoding="utf-8") as f:
        outcomes = {}
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            outcomes[rec["candidate_id"]] = rec
    kept = []
    for row in rows:
        out = outcomes.get(row["candidate_id"])
        if out is None or label not in out:
            continue
        row["label"] = int(bool(out[label]))
        row["outcomes"] = {k: v for k, v in out.items() if k != "candidate_id"}
        kept.append(row)
    return kept


# ---------------------------------------------------------------------------
# Splits (SS13.1): leave-one-scenario-out, chain-grouped by construction
# ---------------------------------------------------------------------------


def nc_shuffle_entropy_features(rows, seed):
    """Offline negative control (plan SS18 NC / SS23 Phase 3): permute the
    ENTROPY features jointly across rows WITHIN (backend, store-size bin),
    fixed seed. Destroys the entropy<->outcome pairing while preserving the
    marginal distribution and the geometry/margin features. Returns new rows."""
    rng = np.random.default_rng(seed)
    bins = defaultdict(list)
    for i, r in enumerate(rows):
        n = r["features"].get("n_items") or 0
        bins[(r["backend"], min(int(n) // 5, 4))].append(i)
    out = [json.loads(json.dumps(r)) for r in rows]
    for o, r in zip(out, rows):
        o["chain"] = r["chain"]  # restore hashable tuple after the deep copy
    for idx_list in bins.values():
        perm = rng.permutation(len(idx_list))
        for slot, src in zip(idx_list, (idx_list[int(p)] for p in perm)):
            for feat in ENTROPY_FEATURES:
                out[slot]["features"][feat] = rows[src]["features"].get(feat)
    return out


def cv_splits(rows):
    scenarios = sorted({r["scenario"] for r in rows})
    for held_out in scenarios:
        train = [r for r in rows if r["scenario"] != held_out]
        test = [r for r in rows if r["scenario"] == held_out]
        train_chains = {r["chain"] for r in train}
        test_chains = {r["chain"] for r in test}
        assert not (train_chains & test_chains), "chain straddles folds"
        yield held_out, train, test


# ---------------------------------------------------------------------------
# Logistic machinery (numpy only)
# ---------------------------------------------------------------------------


def _design(rows, feature_names, stats=None):
    X = np.array(
        [[_fv(r["features"].get(f)) for f in feature_names] for r in rows],
        dtype=np.float64,
    )
    y = np.array([r["label"] for r in rows], dtype=np.float64)
    if stats is None:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-9] = 1.0
        stats = (mu, sd)
    X = (X - stats[0]) / stats[1]
    return X, y, stats


def _fv(v):
    if v is None:
        return 0.0
    return float(v)


def fit_logistic(X, y, signs=None, l2=1e-2, iters=3000, lr=0.1, seed=12345):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = rng.normal(0, 0.01, size=d)
    b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        gw = X.T @ (p - y) / n + l2 * w
        gb = float(np.mean(p - y))
        w -= lr * gw
        b -= lr * gb
        if signs is not None:
            for j, s in enumerate(signs):
                if s > 0:
                    w[j] = max(w[j], 0.0)
                elif s < 0:
                    w[j] = min(w[j], 0.0)
    return w, b


def predict(X, w, b):
    return 1.0 / (1.0 + np.exp(-(X @ w + b)))


def auc(y, p):
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p), dtype=np.float64)
    i = 0
    sorted_p = p[order]
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i: j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# Gate G1: nested models with chain-grouped bootstrap
# ---------------------------------------------------------------------------


def nested_auc_report(rows, seed=12345, n_boot=2000):
    nests = {
        "geometry": list(GEOMETRY_FEATURES),
        "geometry+margin": list(GEOMETRY_FEATURES + MARGIN_FEATURES),
        "geometry+entropy": list(GEOMETRY_FEATURES + ENTROPY_FEATURES),
        "geometry+margin+entropy": list(
            GEOMETRY_FEATURES + MARGIN_FEATURES + ENTROPY_FEATURES
        ),
    }
    # Out-of-fold predictions per nest (leave-one-scenario-out).
    preds = {name: np.zeros(len(rows)) for name in nests}
    index_of = {id(r): i for i, r in enumerate(rows)}
    for _held, train, test in cv_splits(rows):
        if not train or not test:
            continue
        for name, feats in nests.items():
            Xtr, ytr, stats = _design(train, feats)
            if len(set(ytr.tolist())) < 2:
                continue
            w, b = fit_logistic(Xtr, ytr, seed=seed)
            Xte, _, _ = _design(test, feats, stats=stats)
            p = predict(Xte, w, b)
            for r, pi in zip(test, p):
                preds[name][index_of[id(r)]] = pi
    y = np.array([r["label"] for r in rows], dtype=np.float64)
    report = {name: {"auc": auc(y, preds[name])} for name in nests}

    # Grouped bootstrap over chains for dAUC(full vs geometry+margin) -- the
    # pre-registered entropy claim (SS19).
    chains = sorted({r["chain"] for r in rows}, key=str)
    by_chain = defaultdict(list)
    for i, r in enumerate(rows):
        by_chain[r["chain"]].append(i)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        sample_idx = []
        for c in rng.choice(len(chains), size=len(chains), replace=True):
            sample_idx.extend(by_chain[chains[int(c)]])
        idx = np.array(sample_idx)
        yb = y[idx]
        a_full = auc(yb, preds["geometry+margin+entropy"][idx])
        a_gm = auc(yb, preds["geometry+margin"][idx])
        if a_full is None or a_gm is None:
            continue
        deltas.append(a_full - a_gm)
    if deltas:
        deltas = np.sort(np.array(deltas))
        report["dAUC_entropy_given_geometry_margin"] = {
            "mean": float(np.mean(deltas)),
            "ci95": [
                float(np.percentile(deltas, 2.5)),
                float(np.percentile(deltas, 97.5)),
            ],
            "n_boot_effective": len(deltas),
        }
    return report


# ---------------------------------------------------------------------------
# Option-A risk-coverage threshold selection
# ---------------------------------------------------------------------------


def option_a_grid(rows, backend, target_risk):
    """Grid over (R, M, D, C); 'accept' = the joint rule would answer
    s1_confident. Pick max coverage subject to selective risk <= target;
    tie-break lower risk. Returns (thresholds, table)."""
    b_rows = [r for r in rows if r["backend"] == backend]
    if not b_rows:
        return None, []
    vals = lambda key: sorted(  # noqa: E731
        {_fv(r["features"].get(key)) for r in b_rows}
    )
    m_grid = np.percentile(
        [_fv(r["features"].get("nmargin_med")) for r in b_rows], [10, 25, 50, 75]
    )
    d_grid = np.percentile(
        [_fv(r["features"].get("dH_mean")) for r in b_rows], [50, 75, 90]
    )
    c_grid = np.percentile(
        [_fv(r["features"].get("churn")) for r in b_rows], [50, 75, 90]
    )
    table = []
    best = None
    for R in (1, 2, 3):
        for M in m_grid:
            for D in d_grid:
                for C in c_grid:
                    accepted = []
                    for r in b_rows:
                        f = r["features"]
                        locate = _fv(f.get("rank_self_med")) <= R and f.get("rank_self_med") is not None
                        strong = f.get("nmargin_med") is not None and _fv(f.get("nmargin_med")) >= M
                        interf = _fv(f.get("dH_mean")) >= D or _fv(f.get("churn")) >= C
                        if f.get("smallstore"):
                            interf = False
                        if locate and strong and not interf:
                            accepted.append(r)
                    if not accepted:
                        continue
                    coverage = len(accepted) / len(b_rows)
                    risk = sum(r["label"] for r in accepted) / len(accepted)
                    entry = {
                        "R": R, "M": round(float(M), 6), "D": round(float(D), 6),
                        "C": round(float(C), 6),
                        "coverage": round(coverage, 4), "selective_risk": round(risk, 4),
                    }
                    table.append(entry)
                    if risk <= target_risk and (
                        best is None
                        or coverage > best["coverage"]
                        or (coverage == best["coverage"] and risk < best["selective_risk"])
                    ):
                        best = entry
    table.sort(key=lambda e: (-e["coverage"], e["selective_risk"]))
    return best, table[:50]


# ---------------------------------------------------------------------------
# Option-B fit + isotonic calibration + risk thresholds
# ---------------------------------------------------------------------------


def pav_isotonic(p, y):
    """Pool-adjacent-violators: returns (breakpoints, values) mapping raw
    scores to calibrated probabilities (step function)."""
    order = np.argsort(p, kind="stable")
    ps, ys = p[order], y[order]
    blocks = [[ys[i], 1.0, ps[i], ps[i]] for i in range(len(ys))]  # mean, w, lo, hi
    merged = []
    for blk in blocks:
        merged.append(blk)
        while len(merged) > 1 and merged[-2][0] >= merged[-1][0]:
            m2, m1 = merged[-2], merged[-1]
            w = m2[1] + m1[1]
            merged[-2] = [(m2[0] * m2[1] + m1[0] * m1[1]) / w, w, m2[2], m1[3]]
            merged.pop()
    return [b[3] for b in merged], [b[0] for b in merged]


def fit_option_b(rows, target_risk, seed):
    feats = list(OPTION_B_FEATURES)
    X, y, stats = _design(rows, feats)
    if len(set(y.tolist())) < 2:
        return None
    signs = [OPTION_B_SIGNS[f] for f in feats]
    w, b = fit_logistic(X, y, signs=signs, seed=seed)
    p_raw = predict(X, w, b)
    bp, bv = pav_isotonic(p_raw, y)

    def calibrated(p):
        i = np.searchsorted(np.array(bp), p, side="left")
        i = np.clip(i, 0, len(bv) - 1)
        return np.array(bv)[i]

    p_cal = calibrated(p_raw)
    # Risk-coverage curve over the calibrated risk; threshold_low = accept
    # boundary at target risk, threshold_high = ABSTAIN boundary at 2x target.
    cutoffs = np.unique(np.round(p_cal, 4))
    thr_low, thr_high = float(cutoffs[-1]), float(cutoffs[-1])
    for cut in cutoffs:
        accepted = p_cal <= cut
        if accepted.sum() == 0:
            continue
        risk = float(y[accepted].mean())
        if risk <= target_risk:
            thr_low = float(cut)
    for cut in cutoffs:
        flagged = p_cal >= cut
        if flagged.sum() == 0:
            continue
        if float(y[flagged].mean()) >= min(1.0, 2 * target_risk):
            thr_high = float(cut)
            break
    # De-standardized coefficients for the runtime model (raw feature space).
    mu, sd = stats
    w_raw = w / sd
    b_raw = float(b - np.sum(w * mu / sd))
    return {
        "features": feats,
        "intercept": b_raw,
        "coefs": {f: float(c) for f, c in zip(feats, w_raw)},
        "signs": OPTION_B_SIGNS,
        "isotonic": {"breakpoints": [float(x) for x in bp], "values": [float(v) for v in bv]},
        "threshold_low": thr_low,
        "threshold_high": thr_high,
        "train_auc": auc(y, p_raw),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--logs", required=True,
                    help="glob of gov2 harvest arm dirs (or .jsonl files)")
    ap.add_argument("--outcomes", default=None,
                    help="outcomes.jsonl joined on candidate_id (SS13.2)")
    ap.add_argument("--label", default="harmful_write")
    ap.add_argument("--target-risk", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="gov_logs/margin_entropy_calibration.json")
    ap.add_argument("--freeze", action="store_true",
                    help="write the frozen calibration + CALIBRATION_FROZEN.md")
    args = ap.parse_args()

    log_dirs = sorted(glob.glob(args.logs))
    if not log_dirs:
        sys.exit(f"[calibrate] no logs match {args.logs!r}")
    rows, manifest = load_decisions(log_dirs)
    print(f"[calibrate] escalated gov2 decisions: {len(rows)} from {len(log_dirs)} arms")
    assert_no_gt_in_features(rows)

    report = {
        "seed": args.seed,
        "label": args.label,
        "target_risk": args.target_risk,
        "git_head": git_head(),
        "data_manifest": manifest,
        "n_decisions": len(rows),
    }

    if args.outcomes:
        rows = attach_outcomes(rows, args.outcomes, args.label)
        print(f"[calibrate] labeled decisions: {len(rows)}")
        report["n_labeled"] = len(rows)
        pos = sum(r["label"] for r in rows)
        report["n_positive"] = pos
        if rows and pos and pos < len(rows):
            report["nested_models"] = nested_auc_report(rows, seed=args.seed)
            # Offline NC (SS23 Phase 3): the same nested analysis on rows whose
            # entropy features are permuted within (backend, size-bin). G1
            # requires the real dAUC CI to exclude 0 AND this one to sit at ~0.
            nc_rows = nc_shuffle_entropy_features(rows, args.seed)
            report["nested_models_nc_shuffled"] = nested_auc_report(nc_rows, seed=args.seed)
            report["option_a"] = {}
            for backend in ("kv", "vector"):
                best, table = option_a_grid(rows, backend, args.target_risk)
                report["option_a"][backend] = {"selected": best, "grid_top": table}
            epv = pos / max(1, len(OPTION_B_FEATURES))
            report["events_per_variable"] = round(epv, 2)
            if epv >= 20:
                report["option_b"] = fit_option_b(rows, args.target_risk, args.seed)
            else:
                report["option_b"] = None
                report["option_b_skipped"] = (
                    f"EPV {epv:.1f} < 20 (plan SS22): fall back to Option A"
                )
        else:
            print("[calibrate] label degenerate (all one class) -- report only")
    else:
        print("[calibrate] no --outcomes: dataset assembly + leakage checks only")

    out_path = Path(args.out)
    if args.freeze:
        if out_path.exists():
            sys.exit(f"[calibrate] {out_path} already exists -- refusing to "
                     f"overwrite a frozen calibration (SS13.3)")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        note = out_path.parent / "CALIBRATION_FROZEN.md"
        with open(note, "w", encoding="utf-8") as f:
            f.write(
                "# Margin-entropy calibration freeze (plan SS13.3)\n\n"
                f"- Calibration file: `{out_path.name}` (sha {sha_file(out_path)})\n"
                f"- Git head at freeze: `{report['git_head']}`\n"
                f"- Label: `{args.label}`; target risk: {args.target_risk}; "
                f"seed: {args.seed}\n"
                f"- Data manifest: {json.dumps(manifest)}\n\n"
                "Thresholds are FROZEN as of this commit. The confirmatory\n"
                "campaign must reference this file's git hash in its manifest\n"
                "(gate G3); no re-tuning after the first confirmatory replicate.\n"
            )
        print(f"[calibrate] FROZE {out_path} + {note}")
    else:
        preview = Path(str(out_path) + ".preview")
        preview.parent.mkdir(parents=True, exist_ok=True)
        with open(preview, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[calibrate] wrote {preview} (pass --freeze for the SS13.3 ceremony)")


if __name__ == "__main__":
    main()
