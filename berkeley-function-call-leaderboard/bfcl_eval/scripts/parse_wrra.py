"""
W/R/R/A snapshot parser  --  Plan 2 Step 1 (v2 SS8.1).

Parses, per (model arm, backend, scenario):

  W  Write count      -- memory-write tool calls decoded in the *prereq* phase,
                         split into attempted vs resolved (a call is *resolved*
                         when its paired tool result is a non-error status).
  R  Retrieve count   -- memory-read tool calls, prereq and question phase
                         counted separately.
  R  Recall           -- fraction of question entries that issued >= 1 resolved
                         memory-read call (retrieval-attempt recall; the
                         answer-level signal is Accuracy below).
  A  Accuracy         -- reconstructed from the score files, which store a
                         summary header plus FAILED entries only: an id absent
                         from the failure rows is correct.

plus the primary chain-survival statistic (v2 SS8.1): a scenario chain is
*dead* iff the final snapshot holds 0 entries (core + archival, store-aware
so both KV dicts and Vector {next_id, store} shapes count correctly) OR the
prereq phase resolved 0 content-writing calls. Content-writing ops are
add/replace/update; remove/clear are tracked separately as destructive ops
and do not count toward survival.

Emits tidy JSONL, three record types:
  {"record": "scenario", ...}   one line per (backend, scenario)
  {"record": "question", ...}   one line per question id (correct: true/false/null)
  {"record": "arm_summary", ...} one line per backend

Usage (paths relative to the berkeley-function-call-leaderboard root):
  python bfcl_eval/scripts/parse_wrra.py \
      --model Qwen/Qwen3-4B-Instruct-2507-FC \
      --result-dir result --score-dir score \
      --backends kv,vector --out wrra_baseline.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

BFCL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BFCL_ROOT))

VERSION_PREFIX = "BFCL_v4"

# Union of MemoryAPI_kv / MemoryAPI_vector op names (memory_kv.py, memory_vector.py).
# Both backends share the core_memory_* / archival_memory_* prefixes; the backend is
# disambiguated by directory, never by function name.
WRITE_OPS = {
    "core_memory_add", "core_memory_replace", "core_memory_update",
    "archival_memory_add", "archival_memory_replace", "archival_memory_update",
}
DESTRUCTIVE_OPS = {
    "core_memory_remove", "core_memory_clear",
    "archival_memory_remove", "archival_memory_clear",
}
READ_OPS = {
    "core_memory_retrieve", "core_memory_retrieve_all",
    "core_memory_list_keys", "core_memory_key_search",
    "archival_memory_retrieve", "archival_memory_retrieve_all",
    "archival_memory_list_keys", "archival_memory_key_search",
}


# ---------------------------------------------------------------------------
# Low-level parsing helpers
# ---------------------------------------------------------------------------

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def call_func_name(call_str):
    """'core_memory_add(key=..., value=...)' -> 'core_memory_add'."""
    return str(call_str).split("(", 1)[0].strip()


def tool_result_ok(content):
    """A tool result is a success iff it is not an error payload."""
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return "error" not in data
    except (TypeError, ValueError):
        pass
    return '"error"' not in str(content)


def iter_call_pairs(inference_log):
    """Yield (func_name, tool_ok_or_None) for every decoded call in an entry.

    Within each step_N message list, decoded calls (handler_log messages carrying
    model_response_decoded) and tool results appear in emission order; they are
    zipped positionally. A call with no paired tool message yields tool_ok=None
    (counted as unresolved).
    """
    for turn in inference_log or []:
        if not isinstance(turn, dict):
            continue
        for step_key in sorted(k for k in turn if k.startswith("step_")):
            calls, tools = [], []
            for msg in turn[step_key]:
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") == "handler_log":
                    calls.extend(msg.get("model_response_decoded") or [])
                elif msg.get("role") == "tool":
                    tools.append(msg.get("content"))
            for i, call in enumerate(calls):
                ok = tool_result_ok(tools[i]) if i < len(tools) else None
                yield call_func_name(call), ok


def scenario_from_id(test_id):
    """'memory_kv_prereq_0-customer-0' -> 'customer'."""
    return str(test_id).split(":")[0].rsplit("-", 2)[1]


def count_entry_calls(row):
    """Classify every decoded call of one result row into W/R/destructive."""
    counts = {
        "write_attempted": 0, "write_resolved": 0,
        "destructive_attempted": 0, "destructive_resolved": 0,
        "read_attempted": 0, "read_resolved": 0,
        "other_calls": 0, "unpaired_calls": 0,
    }
    for func, ok in iter_call_pairs(row.get("inference_log")):
        if ok is None:
            counts["unpaired_calls"] += 1
        if func in WRITE_OPS:
            counts["write_attempted"] += 1
            counts["write_resolved"] += bool(ok)
        elif func in DESTRUCTIVE_OPS:
            counts["destructive_attempted"] += 1
            counts["destructive_resolved"] += bool(ok)
        elif func in READ_OPS:
            counts["read_attempted"] += 1
            counts["read_resolved"] += bool(ok)
        else:
            counts["other_calls"] += 1
    return counts


def count_snapshot_entries(snapshot_path):
    """(core_entries, archival_entries) or None if the snapshot is missing.

    Store-aware: KV tiers are flat {key: value} dicts; Vector tiers are
    {"next_id": int, "store": {id: text}}.
    """
    path = Path(snapshot_path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def tier_len(tier):
        if not isinstance(tier, dict):
            return 0
        store = tier.get("store", tier)
        return len(store) if isinstance(store, dict) else 0

    return tier_len(data.get("core_memory", {})), tier_len(data.get("archival_memory", {}))


def parse_score_file(score_path):
    """-> (header dict, set of failed ids) or None if the file is missing.

    Score files hold one summary header line followed by FAILED entries only;
    correctness of an id is reconstructed as `id not in failed`.
    """
    path = Path(score_path)
    if not path.exists():
        return None
    rows = read_jsonl(path)
    if not rows:
        return None
    header, failures = rows[0], rows[1:]
    return header, {f["id"] for f in failures if "id" in f}


# ---------------------------------------------------------------------------
# Per-arm/backend aggregation
# ---------------------------------------------------------------------------

def parse_arm_backend(result_model_dir, score_model_dir, backend,
                      model=None, arm=None):
    """Parse one (model arm, backend) into (scenario_records, question_records).

    result_model_dir / score_model_dir are the per-model roots, i.e.
    <result_root>/<model_slug> and <score_root>/<model_slug>.
    """
    result_model_dir = Path(result_model_dir)
    backend_dir = result_model_dir / "agentic" / "memory" / backend
    prereq_file = backend_dir / f"{VERSION_PREFIX}_memory_{backend}_prereq_result.json"
    question_file = backend_dir / f"{VERSION_PREFIX}_memory_{backend}_result.json"
    snapshot_dir = backend_dir / "memory_snapshot"
    score_file = (
        Path(score_model_dir) / "agentic" / "memory" / backend
        / f"{VERSION_PREFIX}_memory_{backend}_score.json"
    )

    prereq_rows = read_jsonl(prereq_file) if prereq_file.exists() else []
    question_rows = read_jsonl(question_file) if question_file.exists() else []
    score = parse_score_file(score_file)
    failed_ids = score[1] if score else None

    scenarios = {}

    def scen(name):
        return scenarios.setdefault(name, {
            "record": "wrra_scenario", "model": model, "arm": arm,
            "backend": backend, "scenario": name,
            "prereq_entries": 0,
            "write_calls_attempted": 0, "write_calls_resolved": 0,
            "destructive_calls_resolved": 0,
            "retrieve_calls_prereq": 0,
            "question_entries": 0, "retrieve_calls_question": 0,
            "questions_with_retrieve": 0,
            "unpaired_calls": 0,
        })

    for row in prereq_rows:
        s = scen(scenario_from_id(row["id"]))
        c = count_entry_calls(row)
        s["prereq_entries"] += 1
        s["write_calls_attempted"] += c["write_attempted"]
        s["write_calls_resolved"] += c["write_resolved"]
        s["destructive_calls_resolved"] += c["destructive_resolved"]
        s["retrieve_calls_prereq"] += c["read_resolved"]
        s["unpaired_calls"] += c["unpaired_calls"]

    question_records = []
    for row in question_rows:
        name = scenario_from_id(row["id"])
        s = scen(name)
        c = count_entry_calls(row)
        s["question_entries"] += 1
        s["retrieve_calls_question"] += c["read_resolved"]
        s["questions_with_retrieve"] += c["read_resolved"] > 0
        s["unpaired_calls"] += c["unpaired_calls"]
        question_records.append({
            "record": "wrra_question", "model": model, "arm": arm,
            "backend": backend, "scenario": name, "id": row["id"],
            "correct": (row["id"] not in failed_ids) if failed_ids is not None else None,
            "retrieve_calls": c["read_resolved"],
        })

    for name, s in scenarios.items():
        snap = count_snapshot_entries(snapshot_dir / f"{name}_final.json")
        s["snapshot_missing"] = snap is None
        s["snapshot_core_entries"] = snap[0] if snap else 0
        s["snapshot_archival_entries"] = snap[1] if snap else 0
        snapshot_total = s["snapshot_core_entries"] + s["snapshot_archival_entries"]

        if snap is None:
            s["chain_dead"], s["dead_reason"] = True, "missing_snapshot"
        elif snapshot_total == 0:
            s["chain_dead"], s["dead_reason"] = True, "empty_snapshot"
        elif s["write_calls_resolved"] == 0:
            s["chain_dead"], s["dead_reason"] = True, "zero_resolved_writes"
        else:
            s["chain_dead"], s["dead_reason"] = False, None

        s["recall"] = (
            s["questions_with_retrieve"] / s["question_entries"]
            if s["question_entries"] else None
        )
        if failed_ids is not None and s["question_entries"]:
            correct = sum(
                1 for q in question_records
                if q["scenario"] == name and q["correct"]
            )
            s["questions_correct"] = correct
            s["accuracy"] = correct / s["question_entries"]
        else:
            s["questions_correct"] = None
            s["accuracy"] = None

    scenario_records = [scenarios[k] for k in sorted(scenarios)]
    return scenario_records, question_records


def summarize_backend(scenario_records, score_header=None):
    dead = [s["scenario"] for s in scenario_records if s["chain_dead"]]
    total_q = sum(s["question_entries"] for s in scenario_records)
    known = [s for s in scenario_records if s["questions_correct"] is not None]
    correct = sum(s["questions_correct"] for s in known) if known else None
    return {
        "record": "wrra_arm_summary",
        "model": scenario_records[0]["model"] if scenario_records else None,
        "arm": scenario_records[0]["arm"] if scenario_records else None,
        "backend": scenario_records[0]["backend"] if scenario_records else None,
        "chains_total": len(scenario_records),
        "chains_dead": len(dead),
        "dead_scenarios": dead,
        "write_calls_resolved": sum(s["write_calls_resolved"] for s in scenario_records),
        "questions_total": total_q,
        "questions_correct": correct,
        "accuracy": (correct / total_q) if (correct is not None and total_q) else None,
        "score_header": score_header,
    }


def parse_arm(result_root, score_root, model, backends, arm=None):
    """Full parse of one arm: -> flat list of JSONL-ready records."""
    slug = model.replace("/", "_")
    records = []
    for backend in backends:
        scen_recs, q_recs = parse_arm_backend(
            Path(result_root) / slug, Path(score_root) / slug, backend,
            model=slug, arm=arm or slug,
        )
        score = parse_score_file(
            Path(score_root) / slug / "agentic" / "memory" / backend
            / f"{VERSION_PREFIX}_memory_{backend}_score.json"
        )
        records.extend(scen_recs)
        records.extend(q_recs)
        records.append(summarize_backend(scen_recs, score[0] if score else None))
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--model", required=True,
                    help="Registry id or dir slug (slashes become underscores)")
    ap.add_argument("--result-dir", default="result",
                    help="Result root, relative to the BFCL root (or absolute)")
    ap.add_argument("--score-dir", default="score",
                    help="Score root, relative to the BFCL root (or absolute)")
    ap.add_argument("--backends", default="kv,vector")
    ap.add_argument("--arm", default=None,
                    help="Arm label stamped on every record (default: model slug)")
    ap.add_argument("--out", default=None,
                    help="Output JSONL path (default: stdout)")
    args = ap.parse_args()

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else BFCL_ROOT / p

    records = parse_arm(
        resolve(args.result_dir), resolve(args.score_dir),
        args.model, [b.strip() for b in args.backends.split(",") if b.strip()],
        arm=args.arm,
    )
    lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(lines + "\n")
        summaries = [r for r in records if r["record"] == "wrra_arm_summary"]
        for s in summaries:
            print(
                f"[wrra] {s['arm']}/{s['backend']}: "
                f"{s['chains_total'] - s['chains_dead']}/{s['chains_total']} chains alive"
                f" (dead: {', '.join(s['dead_scenarios']) or 'none'}), "
                f"accuracy={s['accuracy']}"
            )
        print(f"[wrra] {len(records)} records -> {args.out}")
    else:
        print(lines)


if __name__ == "__main__":
    main()
