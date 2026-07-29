"""Falsifier for alt5R_write_scaffold (autonomous loop, replaces #5).

Evidence that motivated it (gov_logs/hnav_autonomous/loss_decomposition.json):
~80% of questions have their gold fact absent from the final store, and ~88%
of those facts WERE stated verbatim in the prereq conversation. The agent had
the information and never wrote it: the benchmark's dominant loss is CAPTURE
failure, not gating (Stage 1), retrieval, or action uncertainty (Stage 3).

Policy under test (a runtime agent-design policy, MemGPT-style auto-archiving
-- no gold, no future turns, no evaluation labels): on a prereq user turn that
produced NO resolved write, the middleware archives the user turn text itself.

This script measures the offline ceiling under the REAL backend constraints:
    MAX_ARCHIVAL_SIZE       50 entries per tier
    MAX_ARCHIVAL_ENTRY_LENGTH 2000 chars (longer -> truncated)
    KV retrieval is BM25 over KEY NAMES only, so a KV scaffold entry needs a
    deterministic content-derived slug key (^[a-z]+(_[a-z0-9]+)*$), not an
    opaque id -- an opaque key makes the entry unretrievable by construction.

Variants reported:
    all_turns   scaffold every prereq user turn (upper bound)
    no_write    scaffold only turns that produced no write (the actual policy)

    python bfcl_eval/scripts/falsifier_write_scaffold.py \
        --logs gov_logs/hnav_shadow/rep0*_hact_shadow \
        --result-root result_hnav_shadow --score-root score_hnav_shadow \
        --out gov_logs/hnav_autonomous/falsifier_write_scaffold.json
"""

import argparse
import glob as globmod
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from calibrate_margin_entropy import git_head  # noqa: E402
from hnav_answer_index import (  # noqa: E402
    answerable,
    core_qid,
    gold_index,
    question_text_for,
)
from label_outcomes import correct_ids, load_final_stores, scenario_of  # noqa: E402

MAX_ARCHIVAL_SIZE = 50
MAX_ARCHIVAL_ENTRY_LENGTH = 2000
P_HAT = 0.408544          # gov_logs/hnav_proxy_validation.json (Stage 1, frozen)
PREREQ_DIR = Path("bfcl_eval/data/memory_prereq_conversation")

_STOP = {"the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for",
         "with", "is", "am", "are", "was", "were", "be", "been", "i", "my",
         "me", "you", "your", "it", "its", "this", "that", "so", "as", "at",
         "by", "we", "our", "us", "they", "them", "he", "she", "his", "her"}


def slug_key(text, n_tokens=6, salt=0):
    """Deterministic content-derived KV key: lowercase content tokens joined
    by underscores, matching memory_kv._is_valid_key_format."""
    toks = [t for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in _STOP and not t.isdigit()][:n_tokens]
    if not toks:
        toks = ["note"]
    key = "_".join(toks)
    if not re.match(r"^[a-z]+(_[a-z0-9]+)*$", key):
        key = "note_" + re.sub(r"[^a-z0-9_]", "", key) or "note"
    return f"{key}_{salt}" if salt else key


def prereq_turns():
    """scenario -> ordered list of user turn texts."""
    out = defaultdict(list)
    for f in sorted(PREREQ_DIR.glob("*.json")):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            scen = scenario_of(rec["id"])
            for turn in rec["question"]:
                for m in turn:
                    if m.get("content"):
                        out[scen].append(m["content"])
    return out


def written_user_texts(log_dirs):
    """(backend, scenario) -> set of user_text values that produced >=1 write
    decision. Everything else in the prereq conversation is a no-write turn."""
    seen = defaultdict(set)
    for d in log_dirs:
        p = Path(d) / "governance_log.jsonl"
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("event") != "decision":
                continue
            ut = rec.get("user_text")
            if ut:
                seen[(rec.get("backend"), scenario_of(rec.get("test_id", "")))].add(ut)
    return seen


def augment(tiers, backend, texts):
    """Apply the scaffold under real capacity limits; returns the new tiers."""
    aug = {"core": dict(tiers.get("core", {})),
           "archival": dict(tiers.get("archival", {}))}
    room = MAX_ARCHIVAL_SIZE - len(aug["archival"])
    for i, t in enumerate(texts):
        if room <= 0:
            break
        body = t[:MAX_ARCHIVAL_ENTRY_LENGTH]
        if backend == "kv":
            k = slug_key(body)
            n = 0
            while k in aug["archival"]:
                n += 1
                k = slug_key(body, salt=n)
            # store text mirrors kv_composite_text("key words: value")
            aug["archival"][k] = f"{k.replace('_', ' ')}: {body}"
        else:
            aug["archival"][f"9{i:03d}"] = body
        room -= 1
    return aug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--result-root", default="result_hnav_shadow")
    ap.add_argument("--score-root", default="score_hnav_shadow")
    ap.add_argument("--arm", default="hact_shadow")
    ap.add_argument("--slug", default="Qwen_Qwen3-4B-Instruct-2507-FC-HACT")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dirs = sorted(set(p for g in args.logs for p in globmod.glob(g)))
    turns = prereq_turns()
    written = written_user_texts(dirs)
    reps = sorted({Path(d).name.split("_")[0] for d in dirs})

    per_backend = {}
    for backend in ("kv", "vector"):
        counts = defaultdict(int)
        for rep in reps:
            r = Path(args.result_root) / rep / args.arm / args.slug
            s = Path(args.score_root) / rep / args.arm / args.slug
            if not r.exists():
                continue
            stores = load_final_stores(r, backend)
            ok, attempted = correct_ids(r, s, backend)
            for qid in sorted(attempted):
                core = core_qid(qid)
                gold = gold_index().get(core)
                qtext = question_text_for(qid)
                if not gold or not qtext:
                    continue
                scen = scenario_of(core)
                tiers = stores.get(scen) or {"core": {}, "archival": {}}
                counts["n"] += 1
                base = answerable(backend, tiers, qtext, gold)[0]
                counts["answerable_base"] += int(base)
                counts["correct_base"] += int(qid in ok)

                all_t = turns.get(scen, [])
                nw = [t for t in all_t
                      if t not in written.get((backend, scen), set())]
                counts["n_turns"] += len(all_t)
                counts["n_no_write_turns"] += len(nw)
                for name, texts in (("all_turns", all_t), ("no_write", nw)):
                    aug = augment(tiers, backend, texts)
                    counts[f"answerable_{name}"] += int(
                        answerable(backend, aug, qtext, gold)[0])
        n = counts["n"] or 1
        per_backend[backend] = {
            "n_questions": counts["n"],
            "n_prereq_turns": counts["n_turns"],
            "n_no_write_turns": counts["n_no_write_turns"],
            "answerable_base": counts["answerable_base"],
            "answerable_base_rate": round(counts["answerable_base"] / n, 4),
            "observed_correct": counts["correct_base"],
            "observed_accuracy": round(counts["correct_base"] / n, 4),
            "variants": {},
        }
        for name in ("all_turns", "no_write"):
            a = counts[f"answerable_{name}"]
            d = (a - counts["answerable_base"]) / n
            per_backend[backend]["variants"][name] = {
                "answerable": a,
                "answerable_rate": round(a / n, 4),
                "delta_answerable_rate": round(d, 4),
                "expected_dacc": round(d * P_HAT, 4),
            }

    report = {
        "git_head": git_head(), "p_hat": P_HAT,
        "capacity": {"max_archival_size": MAX_ARCHIVAL_SIZE,
                     "max_archival_entry_length": MAX_ARCHIVAL_ENTRY_LENGTH},
        "arms": dirs, "replicates": reps,
        "per_backend": per_backend,
        "go_criterion": "expected_dacc(no_write) >= 0.02 on >= 1 backend",
    }
    report["go"] = any(
        b["variants"]["no_write"]["expected_dacc"] >= 0.02
        for b in per_backend.values())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for b, d in per_backend.items():
        print(f"{b}: n={d['n_questions']} answerable_now={d['answerable_base']} "
              f"({d['answerable_base_rate']:.1%}) obs_acc={d['observed_accuracy']:.3f}")
        for name, v in d["variants"].items():
            print(f"   {name:10s} answerable={v['answerable']} "
                  f"({v['answerable_rate']:.1%}) d={v['delta_answerable_rate']:+.1%} "
                  f"-> expected dAcc {v['expected_dacc']:+.3f}")
    print(f"[falsifier] GO={report['go']} -> {out}")


if __name__ == "__main__":
    main()
