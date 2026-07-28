"""Tests for hact_sampler.py -- config, logprob features, record schema.

Synthetic only (fake logprobs objects); no model, no logs. Run from BFCL root:
    python bfcl_eval/scripts/test_hact_sampler.py
"""

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bfcl_eval.model_handler.middleware.action_space import canonicalize  # noqa: E402
from bfcl_eval.model_handler.middleware.hact_sampler import (  # noqa: E402
    HACT_SCHEMA,
    HactConfig,
    build_hact_record,
    log_hact_record,
    logprob_features,
    stable_seed,
)

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name} {detail}")


def make_lp(tokens, tlps, top=None):
    return SimpleNamespace(tokens=tokens, token_logprobs=tlps,
                           top_logprobs=top, text_offset=None)


# ------------------------------------------------------------------ config

def test_config_from_env():
    saved = {k: os.environ.get(k) for k in list(os.environ) if k.startswith("HACT_")}
    for k in saved:
        del os.environ[k]
    try:
        cfg = HactConfig.from_env()
        check("defaults: disabled", cfg.enabled is False)
        check("defaults: n=8", cfg.num_samples == 8)
        check("defaults: temp 0.7", cfg.temperature == 0.7)
        check("defaults: policy shadow", cfg.policy == "shadow")
        check("defaults: prereq-only gating", cfg.gate_recall is False)
        os.environ.update({"HACT_ENABLED": "1", "HACT_N": "4", "HACT_TEMP": "0.9",
                           "HACT_LOGPROBS": "0", "HACT_POLICY": "majority",
                           "HACT_GATE_RECALL": "1", "HACT_SEED": "7"})
        cfg = HactConfig.from_env()
        check("env override n", cfg.num_samples == 4)
        check("env override policy", cfg.policy == "majority")
        check("env override gate_recall", cfg.gate_recall is True)
        check("env override seed", cfg.seed_base == 7)
        os.environ["HACT_POLICY"] = "bogus"
        try:
            HactConfig.from_env()
            check("bad policy raises", False)
        except ValueError:
            check("bad policy raises", True)
    finally:
        for k in list(os.environ):
            if k.startswith("HACT_"):
                del os.environ[k]
        os.environ.update({k: v for k, v in saved.items() if v is not None})


def test_stable_seed():
    a = stable_seed(1, "memory_kv_0-customer-0", 3)
    b = stable_seed(1, "memory_kv_0-customer-0", 3)
    c = stable_seed(1, "memory_kv_0-customer-0", 4)
    d = stable_seed(2, "memory_kv_0-customer-0", 3)
    check("seed deterministic", a == b)
    check("seed varies with turn", a != c)
    check("seed varies with base", a != d)
    check("seed is 32-bit int", 0 <= a < 2 ** 32)


# ------------------------------------------------------------------ logprobs

def test_logprob_features_basic():
    text = "hello <tool_call>\n{\"name\": \"f\"}\n</tool_call>"
    # tokens concatenate exactly to text
    tokens = ["hello ", "<tool_call>", "\n{\"name\": \"f\"}\n", "</tool_call>"]
    tlps = [-0.5, -1.0, -2.0, -0.25]
    top = [{"a": -0.5, "b": -1.5}, {"a": -1.0, "b": -1.2}, None, {"a": -0.25, "b": -3.0}]
    f = logprob_features(make_lp(tokens, tlps, top), text)
    check("lp_sum", abs(f["lp_sum"] - (-3.75)) < 1e-9, f["lp_sum"])
    check("lp_mean", abs(f["lp_mean"] - (-3.75 / 4)) < 1e-9)
    check("lp_min", f["lp_min"] == -2.0)
    check("lp_ppl", abs(f["lp_ppl"] - math.exp(3.75 / 4)) < 1e-9)
    check("topk margin mean", abs(f["lp_topk_margin_mean"] - ((1.0 + 0.2 + 2.75) / 3)) < 1e-9)
    check("topk margin min", abs(f["lp_topk_margin_min"] - 0.2) < 1e-9)
    # tool span = tokens 1..3 (indices overlapping the <tool_call> region)
    check("tool span n", f["lp_tool_n_tokens"] == 3.0, f)
    check("tool span sum", abs(f["lp_tool_sum"] - (-3.25)) < 1e-9)
    check("tool span min", f["lp_tool_min"] == -2.0)


def test_logprob_features_no_tool_call():
    f = logprob_features(make_lp(["just ", "text"], [-0.1, -0.2]), "just text")
    check("no tool span -> None", f["lp_tool_sum"] is None)
    check("overall still computed", f["lp_sum"] is not None)


def test_logprob_features_degraded():
    f = logprob_features(None, "x")
    check("None obj -> all None", all(v is None for v in f.values()))
    f = logprob_features(make_lp([], []), "x")
    check("empty obj -> all None", all(v is None for v in f.values()))
    f = logprob_features(make_lp(["a"], [None]), "a")
    check("all-None tlps -> all None", all(v is None for v in f.values()))
    f = logprob_features(make_lp(["a", "b"], [-1.0, None]), "ab")
    check("partial None tolerated", f["lp_n_tokens"] == 1.0)
    f = logprob_features(make_lp(["a"], [-1.0], top=None), "a")
    check("missing top_logprobs -> margin None, rest fine",
          f["lp_topk_margin_mean"] is None and f["lp_sum"] == -1.0)


# ------------------------------------------------------------------ record

def _mk_record(**over):
    cfg = HactConfig(enabled=True, num_samples=3)
    prim_calls = ["core_memory_add(key='age', value='35')"]
    samples_calls = [prim_calls, ["core_memory_add(key='age', value='35')"], None]
    actions = [[canonicalize(c, "kv") for c in (cs or [])] if cs is not None
               else [canonicalize("(", "kv")] for cs in samples_calls]
    kw = dict(
        test_id="memory_kv_prereq_0-customer-0", is_prereq=True, backend="kv",
        cfg=cfg, primary_text="<tool_call>...</tool_call>",
        primary_calls=prim_calls,
        primary_lp_feats={"lp_sum": -1.0}, logprobs_supported=True,
        sample_texts=["t0", "t1" * 300, "t2"], sample_calls=samples_calls,
        actions_per_sample=actions, seeds=[11, 12],
        input_tokens=100, output_tokens=50, api_seconds=1.23456,
    )
    kw.update(over)
    return build_hact_record(**kw)


def test_record_schema():
    rec = _mk_record()
    for key in ("schema", "event", "test_id", "is_prereq", "backend",
                "sampling_config", "primary", "samples", "votes",
                "logprobs_supported", "logprob_features",
                "input_tokens", "output_tokens", "api_seconds"):
        check(f"record key {key}", key in rec)
    check("schema tag", rec["schema"] == HACT_SCHEMA)
    check("votes carry all resolutions",
          all(f"h_act_{t}" in rec["votes"] for t in ("tool", "op", "target", "full")))
    check("primary sig r3", rec["primary"]["signature_r3"] == "add:core|age")
    check("sample text truncated to 400", len(rec["samples"][1]["text"]) == 400)
    check("None calls preserved", rec["samples"][2]["calls"] is None)
    check("api seconds rounded", rec["api_seconds"] == 1.235)
    check("no policy_record by default", "policy_record" not in rec)
    rec2 = _mk_record(policy_record={"decision": "keep_primary"})
    check("policy_record attached when given", rec2["policy_record"]["decision"] == "keep_primary")


def test_record_json_serializable_and_logged():
    rec = _mk_record()
    line = json.dumps(rec)
    check("record json-serializable", isinstance(line, str) and len(line) > 100)
    with tempfile.TemporaryDirectory() as d:
        log_hact_record(rec, d, "hact_log.jsonl")
        log_hact_record(rec, d, "hact_log.jsonl")
        rows = [json.loads(l) for l in
                (Path(d) / "hact_log.jsonl").read_text(encoding="utf-8").splitlines()]
        check("two appended rows", len(rows) == 2)
        check("roundtrip schema", rows[0]["schema"] == HACT_SCHEMA)
        check("nested dir created", True)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"[{fn.__name__}]")
        fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
