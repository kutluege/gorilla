"""H-Nav Stage 3: shadow-mode statistical validation of action-side H_act.

Pre-registered protocol: gov_logs/hnav_shadow/PREREGISTRATION.md (frozen and
committed BEFORE the shadow campaign ran; this docstring summarizes it).

Targets
    T1 (primary, candidate-level)  label = must_suppress from outcomes_hnav
        (the write action caused counterfactual retrieval damage and carried
        no necessity -> the action taken was wrong).
    T2 (step-level)                label = orphan (gated step whose PRIMARY
        produced no governed call: no-call or parse failure). Features are
        exploration-only votes + primary overall logprobs; primary-derived
        vote features and lp_tool_* are EXCLUDED (collinear with the target).
    T3 (exploratory, descriptive)  chain-level mean H_act vs official chain
        accuracy; reported, never gated on.

Nested ladder (leave-one-scenario-out OOF, chain-grouped bootstrap deltas)
    M0  op metadata (backend_kv, op_add, tier_core)
    M1  + geometry (sim_max, r, tau_t, rho, n_items)
    M2  + margin/entropy block from s1_me (None -> 0 for non-escalated)
    M3  + primary token-logprob features
    M4  + vote controls WITHOUT entropy (modal share, vote margin, n_unique,
          parse-fail / no-call counters, primary_matches_modal; R2+R3)
    M5  M4 + h_act_op          M6  M4 + h_act_target    M7  M4 + h_act_full
    M8  M4 + all H_act resolutions
    M9  M8 + normalized entropies
    M10 M9 + interactions (h_act_target x backend_kv, x op_add)

Pre-registered delta family (Holm-corrected):
    (M8-M4) HEADLINE: entropy beyond its own vote-margin controls
    (M5-M4), (M6-M4), (M7-M4), (M3-M2), (M4-M3), (M10-M8)

Gate (gov_logs/hnav_shadow/PREREGISTRATION.md, applied verbatim):
    GO       T1 dAUC(M8-M4) bootstrap CI95 excludes 0 AND Holm-adj within-chain
             permutation p < 0.05 AND NC2 shuffle collapses the gain
             (|NC dAUC mean| < 0.02 or NC CI covers 0) AND replicate-holdout
             sign-consistent AND n_positives >= 25.
    PARTIAL  same criteria pass for (M4-M0) but not (M8-M4)  -> vote-based
             arms only (A2/A3), no entropy-threshold arms.
    NO_GO    neither -> autonomous alternative loop.

Usage (CWD = berkeley-function-call-leaderboard):
    python bfcl_eval/scripts/evaluate_hact_shadow.py \
        --logs gov_logs/hnav_shadow/rep0*_hact_shadow \
        --outcomes gov_logs/hnav_shadow/outcomes_hnav_shadow.jsonl \
        --out gov_logs/hnav_shadow/stage3_gate.json \
        --md gov_logs/hnav_shadow/STAGE3_REPORT.md
"""

import argparse
import glob as globmod
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from bfcl_eval.model_handler.middleware.action_space import (  # noqa: E402
    canonicalize_sample,
    vote_features_all,
)
from calibrate_margin_entropy import (  # noqa: E402
    FEATURE_KEYS,
    FORBIDDEN_FEATURE_TOKENS,
    auc,
    brier,
    ece,
    git_head,
    load_decisions,
    nc_shuffle_entropy_features,
    nested_auc_report,
    pr_auc,
    sha_file,
)
from analyze_gov_replicates import holm  # noqa: E402
from evaluate_hnav_stage1 import attach_hnav_labels  # noqa: E402

# ---------------------------------------------------------------- features

META_FEATURES = ("backend_kv", "op_add", "tier_core")
GEOM_FEATURES = ("sim_max", "r", "tau_t", "rho", "n_items")
MARGIN_BLOCK = ("rank_self_med", "nmargin_med", "dH_self", "dH_mean",
                "n_eff_norm", "disp", "churn", "smallstore")
LP_FEATURES = ("lp_mean", "lp_min", "lp_ppl", "lp_topk_margin_mean",
               "lp_topk_margin_min", "lp_tool_mean", "lp_tool_min")
VOTE_CONTROL_FEATURES = (
    "modal_share_op", "vote_margin_op",
    "modal_share_target", "vote_margin_target", "n_unique_target",
    "n_parse_fail_target", "n_no_call_target", "primary_matches_modal_target",
)
ENTROPY_HACT = ("h_act_op",)
ENTROPY_HACT_TARGET = ("h_act_target",)
ENTROPY_HACT_FULL = ("h_act_full",)
ENTROPY_HACT_ALL = ("h_act_tool", "h_act_op", "h_act_target", "h_act_full")
ENTROPY_HACT_NORM = ("h_act_norm_op", "h_act_norm_target", "h_act_norm_full")
INTERACTIONS = ("hx_target_kv", "hx_target_add")

HACT_FEATURE_KEYS = tuple(sorted(set(
    META_FEATURES + LP_FEATURES + VOTE_CONTROL_FEATURES
    + ENTROPY_HACT_ALL + ENTROPY_HACT_NORM + INTERACTIONS
)))

NESTS = {
    "M0": list(META_FEATURES),
    "M1": list(META_FEATURES + GEOM_FEATURES),
    "M2": list(META_FEATURES + GEOM_FEATURES + MARGIN_BLOCK),
    "M3": list(META_FEATURES + GEOM_FEATURES + MARGIN_BLOCK + LP_FEATURES),
    "M4": list(META_FEATURES + GEOM_FEATURES + MARGIN_BLOCK + LP_FEATURES
               + VOTE_CONTROL_FEATURES),
}
NESTS["M5"] = NESTS["M4"] + list(ENTROPY_HACT)
NESTS["M6"] = NESTS["M4"] + list(ENTROPY_HACT_TARGET)
NESTS["M7"] = NESTS["M4"] + list(ENTROPY_HACT_FULL)
NESTS["M8"] = NESTS["M4"] + list(ENTROPY_HACT_ALL)
NESTS["M9"] = NESTS["M8"] + list(ENTROPY_HACT_NORM)
NESTS["M10"] = NESTS["M9"] + list(INTERACTIONS)

DELTA_SPECS = {
    "dAUC_M8_vs_M4": ("M8", "M4"),      # HEADLINE
    "dAUC_M5_vs_M4": ("M5", "M4"),
    "dAUC_M6_vs_M4": ("M6", "M4"),
    "dAUC_M7_vs_M4": ("M7", "M4"),
    "dAUC_M3_vs_M2": ("M3", "M2"),
    "dAUC_M4_vs_M3": ("M4", "M3"),
    "dAUC_M10_vs_M8": ("M10", "M8"),
    "dAUC_M4_vs_M0": ("M4", "M0"),      # PARTIAL-GO criterion
}
HEADLINE = "dAUC_M8_vs_M4"
PARTIAL_KEY = "dAUC_M4_vs_M0"
MIN_POSITIVES = 25
PRIMARY_LABEL = "must_suppress"


def leakage_guard(rows):
    """Whitelist + lexical guard over the merged feature vectors (NC8/NC13)."""
    allowed = set(FEATURE_KEYS) | set(HACT_FEATURE_KEYS)
    for row in rows:
        for key in row["features"]:
            if key not in allowed:
                raise AssertionError(f"non-whitelisted feature key: {key!r}")
            low = key.lower()
            if low.startswith("oracle"):
                raise AssertionError(f"oracle feature leaked: {key!r}")
            for tok in FORBIDDEN_FEATURE_TOKENS:
                if tok in low:
                    raise AssertionError(f"forbidden feature key: {key!r}")


# ---------------------------------------------------------------- loading

def load_hact_rows(log_dirs, log_name="hact_log.jsonl"):
    """All hact1 records per arm dir. Returns (joinable, orphans, manifest)."""
    joinable, orphans, manifest = {}, [], []
    for d in log_dirs:
        d = Path(d)
        p = d / log_name
        if not p.exists():
            sys.exit(f"[hact_shadow] hact log not found: {p}")
        manifest.append({"log": str(p), "sha": sha_file(p)})
        replicate = d.name
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("schema") != "hact1":
                    continue
                rec["_replicate"] = replicate
                if rec.get("orphan") or rec.get("step_idx") is None:
                    orphans.append(rec)
                else:
                    key = (replicate, rec["test_id"], rec["step_idx"])
                    if key in joinable:
                        raise AssertionError(f"duplicate hact row for {key} (NC14)")
                    joinable[key] = rec
    return joinable, orphans, manifest


def hact_features_from_record(rec):
    """Flatten a hact1 record into whitelisted runtime features."""
    v = rec.get("votes") or {}
    lp = rec.get("logprob_features") or {}
    feats = {}
    for k in LP_FEATURES:
        feats[k] = lp.get(k)
    for k in ("modal_share_op", "vote_margin_op", "modal_share_target",
              "vote_margin_target", "n_unique_target", "n_parse_fail_target",
              "n_no_call_target", "h_act_tool", "h_act_op", "h_act_target",
              "h_act_full", "h_act_norm_op", "h_act_norm_target",
              "h_act_norm_full"):
        feats[k] = v.get(k)
    feats["primary_matches_modal_target"] = (
        1.0 if v.get("primary_matches_modal_target") else 0.0)
    return feats


def join_hact(decision_rows, hact_joinable):
    """Merge hact features into decision rows on (replicate, test_id, step_idx).
    Returns (joined_rows, n_unjoined_decisions)."""
    joined, missing = [], 0
    for row in decision_rows:
        key = (row["replicate"], row.get("test_id"), row.get("step_idx"))
        rec = hact_joinable.get(key)
        if rec is None:
            missing += 1
            continue
        f = dict(row["features"])
        f.update(hact_features_from_record(rec))
        op = row.get("op") or ""
        f["op_add"] = 1.0 if op.endswith("memory_add") else 0.0
        f["tier_core"] = 1.0 if row.get("tier") == "core" else 0.0
        h = f.get("h_act_target")
        f["hx_target_kv"] = (h or 0.0) * (f.get("backend_kv") or 0.0)
        f["hx_target_add"] = (h or 0.0) * f["op_add"]
        row = dict(row)
        row["features"] = f
        joined.append(row)
    return joined, missing


def build_t2_rows(hact_joinable, orphans):
    """Step-level rows for T2 (orphan prediction). Exploration-only votes are
    recomputed from samples[1:]; primary-derived vote features and lp_tool_*
    are excluded (collinear with the target)."""
    rows = []
    all_recs = [(k, r, 0) for k, r in hact_joinable.items()]
    all_recs += [((r["_replicate"], r["test_id"], None), r, 1) for r in orphans]
    for (replicate, test_id, _), rec, is_orphan in all_recs:
        backend = rec.get("backend")
        scenario = test_id.split("-")[1] if "-" in str(test_id) else "unknown"
        if scenario == "student":
            continue
        samples = (rec.get("samples") or [])[1:]
        actions = [canonicalize_sample(s.get("calls"), backend) for s in samples]
        v = vote_features_all(actions) if actions else {}
        lp = rec.get("logprob_features") or {}
        feats = {
            "backend_kv": 1.0 if backend == "kv" else 0.0,
            "lp_mean": lp.get("lp_mean"), "lp_min": lp.get("lp_min"),
            "lp_ppl": lp.get("lp_ppl"),
            "lp_topk_margin_mean": lp.get("lp_topk_margin_mean"),
            "lp_topk_margin_min": lp.get("lp_topk_margin_min"),
            "h_act_op": v.get("h_act_op"), "h_act_target": v.get("h_act_target"),
            "modal_share_target": v.get("modal_share_target"),
            "vote_margin_target": v.get("vote_margin_target"),
            "n_no_call_target": v.get("n_no_call_target"),
            "n_parse_fail_target": v.get("n_parse_fail_target"),
        }
        rows.append({
            "candidate_id": f"t2|{replicate}|{test_id}|{rec.get('call_idx_in_entry')}",
            "backend": backend, "scenario": scenario, "replicate": replicate,
            "chain": (backend, scenario, replicate),
            "label": 1.0 if is_orphan else 0.0,
            "features": feats,
        })
    return rows


T2_NESTS = {
    "T2_base": ["backend_kv"],
    "T2_lp": ["backend_kv", "lp_mean", "lp_min", "lp_ppl",
              "lp_topk_margin_mean", "lp_topk_margin_min"],
    "T2_votes": ["backend_kv", "modal_share_target", "vote_margin_target",
                 "n_no_call_target", "n_parse_fail_target"],
    "T2_votes_entropy": ["backend_kv", "modal_share_target", "vote_margin_target",
                         "n_no_call_target", "n_parse_fail_target",
                         "h_act_op", "h_act_target"],
    "T2_all": ["backend_kv", "lp_mean", "lp_min", "lp_ppl",
               "lp_topk_margin_mean", "lp_topk_margin_min",
               "modal_share_target", "vote_margin_target",
               "n_no_call_target", "n_parse_fail_target",
               "h_act_op", "h_act_target"],
}
T2_DELTAS = {
    "dAUC_T2_entropy_vs_votes": ("T2_votes_entropy", "T2_votes"),
    "dAUC_T2_votes_vs_lp": ("T2_votes", "T2_lp"),
    "dAUC_T2_all_vs_base": ("T2_all", "T2_base"),
}


# ---------------------------------------------------------------- stats helpers

def within_chain_permutation_p(rows, score_key, seed=12345, n_perm=10000):
    """Permute labels within chains; statistic = AUC(label, feature)."""
    y = np.array([r["label"] for r in rows], dtype=np.float64)
    x = np.array([float(r["features"].get(score_key) or 0.0) for r in rows])
    stat = auc(y, x)
    if stat is None:
        return {"statistic_auc": None, "p_value": None, "n_perm": 0}
    by_chain = defaultdict(list)
    for i, r in enumerate(rows):
        by_chain[r["chain"]].append(i)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        yp = y.copy()
        for idx in by_chain.values():
            arr = np.array(idx)
            yp[arr] = yp[arr[rng.permutation(len(arr))]]
        s = auc(yp, x)
        if s is not None and abs(s - 0.5) >= abs(stat - 0.5):
            hits += 1
    return {"statistic_auc": float(stat),
            "p_value": (hits + 1) / (n_perm + 1), "n_perm": n_perm,
            "n": len(rows), "n_positive": int(y.sum())}


def gaussian_noise_features(rows, feats, seed):
    """NC3: replace the given features with seeded standard normal noise."""
    rng = np.random.default_rng(seed)
    out = [json.loads(json.dumps(r)) for r in rows]
    for o, r in zip(out, rows):
        o["chain"] = r["chain"]
        for f in feats:
            o["features"][f] = float(rng.normal())
    return out


def shift_labels_within_chain(rows):
    """NC5: attach each row's label to the chain's NEXT row (cyclic)."""
    out = [json.loads(json.dumps(r)) for r in rows]
    for o, r in zip(out, rows):
        o["chain"] = r["chain"]
    by_chain = defaultdict(list)
    for i, r in enumerate(rows):
        by_chain[r["chain"]].append(i)
    for idx in by_chain.values():
        labels = [rows[i]["label"] for i in idx]
        for j, i in enumerate(idx):
            out[i]["label"] = labels[(j + 1) % len(labels)]
    return out


def global_label_shuffle(rows, seed):
    """NC4: shuffle labels across ALL rows (ignores chain structure)."""
    rng = np.random.default_rng(seed)
    out = [json.loads(json.dumps(r)) for r in rows]
    labels = [r["label"] for r in rows]
    perm = rng.permutation(len(labels))
    for o, p in zip(out, perm):
        o["label"] = labels[int(p)]
    for o, r in zip(out, rows):
        o["chain"] = r["chain"]
    return out


def replicate_holdout_delta(rows, nest_a, nest_b, seed):
    """NC12: fit on all replicates but one, test on the held-out replicate;
    report the OOF AUC delta per held-out replicate (sign consistency)."""
    from calibrate_margin_entropy import _design, fit_logistic, predict
    reps = sorted({r["replicate"] for r in rows})
    out = {}
    for held in reps:
        train = [r for r in rows if r["replicate"] != held]
        test = [r for r in rows if r["replicate"] == held]
        if not train or not test:
            continue
        yte = np.array([r["label"] for r in test], dtype=np.float64)
        if len(set(yte.tolist())) < 2:
            out[held] = None
            continue
        aucs = {}
        for name, feats in ((("a"), NESTS[nest_a]), (("b"), NESTS[nest_b])):
            Xtr, ytr, stats = _design(train, feats)
            if len(set(ytr.tolist())) < 2:
                aucs[name] = None
                continue
            w, b = fit_logistic(Xtr, ytr, seed=seed)
            Xte, _, _ = _design(test, feats, stats=stats)
            aucs[name] = auc(yte, predict(Xte, w, b))
        out[held] = (None if aucs.get("a") is None or aucs.get("b") is None
                     else float(aucs["a"] - aucs["b"]))
    return out


def ladder_report(rows, seed, n_boot, nests=None, delta_specs=None, ref=None):
    nests = nests or NESTS
    delta_specs = delta_specs or DELTA_SPECS
    rep = nested_auc_report(rows, seed=seed, n_boot=n_boot, nests=nests,
                            delta_specs=delta_specs,
                            delta_ref=ref or "M4", return_preds=True)
    preds, y = rep.pop("_oof_preds"), rep.pop("_y")
    for name in nests:
        if isinstance(rep.get(name), dict):
            rep[name]["brier"] = brier(y, preds[name])
            rep[name]["ece"] = ece(y, preds[name])
    return rep


def ci_excludes_zero(block):
    return (block is not None and block.get("ci95")
            and (block["ci95"][0] > 0.0 or block["ci95"][1] < 0.0))


def nc_collapsed(block, tol=0.02):
    if block is None:
        return True   # no NC delta computable = nothing survived, treat as collapsed
    lo, hi = block.get("ci95", [None, None])
    covers_zero = lo is not None and lo <= 0.0 <= hi
    return covers_zero or abs(block.get("mean") or 0.0) < tol


def decide_gate(report):
    """Apply the pre-registered GO / PARTIAL / NO_GO rule mechanically."""
    t1 = report["t1"]
    head = t1["ladder"].get(HEADLINE)
    perm = t1["permutation"].get("h_act_target", {})
    p_holm = t1.get("holm_adjusted", {}).get("h_act_target")
    nc2 = t1["nc"]["nc2_shuffle_entropy"].get(HEADLINE)
    holdout = t1["nc"]["nc12_replicate_holdout"]
    signs = [v for v in holdout.values() if v is not None]
    sign_consistent = len(signs) > 0 and (all(v > 0 for v in signs)
                                          or all(v < 0 for v in signs))
    enough = t1["n_positive"] >= MIN_POSITIVES

    go = (ci_excludes_zero(head) and p_holm is not None and p_holm < 0.05
          and nc_collapsed(nc2) and sign_consistent
          and all(v > 0 for v in signs) and enough)

    part_block = t1["ladder"].get(PARTIAL_KEY)
    nc2_part = t1["nc"]["nc2_shuffle_entropy"].get(PARTIAL_KEY)
    partial = (not go and ci_excludes_zero(part_block)
               and enough and nc_collapsed(nc2_part))

    verdict = "GO" if go else ("PARTIAL" if partial else "NO_GO")
    return {
        "verdict": verdict,
        "criteria": {
            "headline_ci_excludes_zero": ci_excludes_zero(head),
            "holm_p_h_act_target": p_holm,
            "perm_p_raw": perm.get("p_value"),
            "nc2_collapsed": nc_collapsed(nc2),
            "replicate_holdout_signs": holdout,
            "sign_consistent_positive": sign_consistent and bool(signs)
            and all(v > 0 for v in signs),
            "n_positive": t1["n_positive"],
            "min_positives": MIN_POSITIVES,
            "partial_ci_excludes_zero": ci_excludes_zero(part_block),
        },
    }


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True,
                    help="hact_shadow arm gov-log dirs (explicit, may glob)")
    ap.add_argument("--outcomes", nargs="+", required=True)
    ap.add_argument("--label", default=PRIMARY_LABEL)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    dirs = sorted(set(p for g in args.logs for p in globmod.glob(g)))
    if not dirs:
        sys.exit("[hact_shadow] no log dirs matched")
    for d in dirs:
        if "oracle" in d:
            sys.exit(f"[hact_shadow] oracle path refused (NC13): {d}")

    decision_rows, gov_manifest = load_decisions(dirs, escalated_only=False)
    hact_joinable, orphans, hact_manifest = load_hact_rows(dirs)
    joined, n_unjoined = join_hact(decision_rows, hact_joinable)
    print(f"[hact_shadow] decisions={len(decision_rows)} "
          f"hact={len(hact_joinable)} orphans={len(orphans)} "
          f"joined={len(joined)} unjoined={n_unjoined}")

    labeled = attach_hnav_labels(joined, args.outcomes, args.label)
    seen = set()
    for r in labeled:
        if r["candidate_id"] in seen:
            raise AssertionError(f"duplicate candidate_id (NC14): {r['candidate_id']}")
        seen.add(r["candidate_id"])
    leakage_guard(labeled)
    y = np.array([r["label"] for r in labeled])
    n_pos = int(y.sum())
    print(f"[hact_shadow] labeled={len(labeled)} positives({args.label})={n_pos}")

    # ---- T1 ladder + permutation family + negative controls
    ladder = ladder_report(labeled, args.seed, args.n_boot)
    perm_family = {}
    for feat in ("h_act_target", "h_act_op", "vote_margin_target", "lp_tool_min"):
        perm_family[feat] = within_chain_permutation_p(
            labeled, feat, seed=args.seed, n_perm=args.n_perm)
    holm_adj = holm({k: v["p_value"] for k, v in perm_family.items()
                     if v.get("p_value") is not None})

    entropy_feats = list(ENTROPY_HACT_ALL + ENTROPY_HACT_NORM + INTERACTIONS)
    nc2_rows = nc_shuffle_entropy_features(
        labeled, args.seed, feats=entropy_feats,
        bin_fn=lambda r: (r["backend"], r["features"].get("op_add")))
    nc3_rows = gaussian_noise_features(labeled, entropy_feats, args.seed)
    nc4_rows = global_label_shuffle(labeled, args.seed)
    nc5_rows = shift_labels_within_chain(labeled)
    small = {"n_boot": max(200, args.n_boot // 4)}
    nc = {
        "nc2_shuffle_entropy": ladder_report(nc2_rows, args.seed, small["n_boot"]),
        "nc3_gaussian_noise": ladder_report(nc3_rows, args.seed, small["n_boot"]),
        "nc4_global_label_shuffle": ladder_report(nc4_rows, args.seed, small["n_boot"]),
        "nc5_step_shift": ladder_report(nc5_rows, args.seed, small["n_boot"]),
        "nc10_seed_stability": ladder_report(labeled, args.seed + 1, small["n_boot"]),
        "nc12_replicate_holdout": replicate_holdout_delta(
            labeled, "M8", "M4", args.seed),
    }

    t1 = {
        "label": args.label, "n": len(labeled), "n_positive": n_pos,
        "base_rate": float(np.mean(y)) if len(y) else None,
        "ladder": ladder, "permutation": perm_family, "holm_adjusted": holm_adj,
        "nc": nc,
    }

    # ---- T2: orphan prediction (step-level)
    t2_rows = build_t2_rows(hact_joinable, orphans)
    t2 = {"n": len(t2_rows),
          "n_positive": int(sum(r["label"] for r in t2_rows))}
    if t2["n_positive"] >= 5 and t2["n"] - t2["n_positive"] >= 5:
        t2["ladder"] = ladder_report(t2_rows, args.seed, small["n_boot"],
                                     nests=T2_NESTS, delta_specs=T2_DELTAS,
                                     ref="T2_votes")
        t2["nc11_perm"] = within_chain_permutation_p(
            t2_rows, "h_act_target", seed=args.seed, n_perm=args.n_perm)
    else:
        t2["skipped"] = "insufficient positives/negatives"

    # ---- T3 exploratory: chain-level mean H_act (descriptive only)
    by_chain = defaultdict(list)
    for r in labeled:
        by_chain[str(r["chain"])].append(r["features"].get("h_act_target") or 0.0)
    t3 = {"chain_mean_h_act": {c: float(np.mean(v)) for c, v in by_chain.items()},
          "note": "descriptive only; never gated on (pre-registration)"}

    report = {
        "git_head": git_head(), "seed": args.seed, "n_boot": args.n_boot,
        "n_perm": args.n_perm, "arms": dirs,
        "gov_manifest": gov_manifest, "hact_manifest": hact_manifest,
        "n_decisions": len(decision_rows), "n_hact": len(hact_joinable),
        "n_orphans": len(orphans), "n_joined": len(joined),
        "n_unjoined": n_unjoined,
        "nests": {k: v for k, v in NESTS.items()},
        "t1": t1, "t2": t2, "t3": t3,
    }
    report["gate"] = decide_gate(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[hact_shadow] gate: {report['gate']['verdict']} -> {out}")

    if args.md:
        md = ["# H-Nav Stage 3 -- shadow-mode H_act validation", "",
              f"Gate verdict: **{report['gate']['verdict']}**", "",
              f"T1 ({args.label}): n={t1['n']} positives={t1['n_positive']} "
              f"base_rate={t1['base_rate']:.5f}" if t1["base_rate"] is not None
              else "T1: empty", ""]
        md.append("| nest | AUC | PR-AUC | Brier | ECE |")
        md.append("|---|---|---|---|---|")
        for name in NESTS:
            b = ladder.get(name) or {}
            md.append(f"| {name} | {b.get('auc')} | {b.get('pr_auc')} | "
                      f"{b.get('brier')} | {b.get('ece')} |")
        md.append("")
        md.append("| delta | mean | CI95 |")
        md.append("|---|---|---|")
        for name in DELTA_SPECS:
            b = ladder.get(name) or {}
            md.append(f"| {name} | {b.get('mean')} | {b.get('ci95')} |")
        md.append("")
        md.append(f"Gate criteria: `{json.dumps(report['gate']['criteria'], default=str)}`")
        Path(args.md).write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"[hact_shadow] wrote {args.md}")

    return report


if __name__ == "__main__":
    main()
