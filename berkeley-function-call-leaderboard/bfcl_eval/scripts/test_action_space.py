"""Tests for action_space.py -- canonical actions + H_act vote features.

Synthetic only; no logs, no model, no ground truth. Run from BFCL root:
    python bfcl_eval/scripts/test_action_space.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bfcl_eval.model_handler.middleware.action_space import (  # noqa: E402
    NO_CALL,
    CanonicalAction,
    canonicalize,
    canonicalize_sample,
    content_fingerprint,
    normalize_text,
    sample_signature,
    signature,
    vote_features,
    vote_features_all,
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


# ---------------------------------------------------------------- canonicalize

def test_kv_ops():
    a = canonicalize("core_memory_add(key='age', value='35')", "kv")
    check("kv add op", a.op == "add:core", a)
    check("kv add target is key", a.target == "age")
    check("kv add has content fp", a.content_fp is not None)

    b = canonicalize("core_memory_add('age', '35')", "kv")
    check("positional == keyword form", a == b, f"{a} vs {b}")

    c = canonicalize("core_memory_add(value='35', key='age')", "kv")
    check("kwarg order irrelevant", a == c)

    r = canonicalize("archival_memory_replace(key='diet', value='vegan')", "kv")
    check("kv archival replace", r.op == "replace:archival" and r.target == "diet")

    rm = canonicalize("core_memory_remove(key='age')", "kv")
    check("kv remove observed op", rm.op == "remove:core" and rm.target == "age")
    check("kv remove no content fp", rm.content_fp is None)

    cl = canonicalize("archival_memory_clear()", "kv")
    check("kv clear", cl.op == "clear:archival" and cl.target is None)


def test_vector_ops():
    a = canonicalize("core_memory_add(text='likes green tea')", "vector")
    check("vector add op", a.op == "add:core")
    check("vector add keyless target", a.target == "")
    u = canonicalize("core_memory_update(vec_id=3, new_text='likes coffee')", "vector")
    check("vector update target int-normalized", u.target == "3")
    u2 = canonicalize("core_memory_update(vec_id='3', new_text='likes coffee')", "vector")
    check("vec_id '3' == 3", u == u2, f"{u} vs {u2}")
    # KV-shaped call on the vector backend is NOT a valid write there
    x = canonicalize("core_memory_replace(key='a', value='b')", "vector")
    check("vector replace falls through to read/other bucket", x.op == "read", x)


def test_reads_other_failures():
    r = canonicalize("core_memory_retrieve(key='age')", "kv")
    check("retrieve is read", r.op == "read")
    o = canonicalize("send_email(to='x@y.z')", "kv")
    check("non-memory tool is other", o.op == "other" and o.tool == "send_email")
    f = canonicalize("core_memory_add(key='age'", "kv")   # unparseable
    check("unparseable -> parse_fail, no raise", f.op == "parse_fail")
    f2 = canonicalize("core_memory_add(key='age')", "kv")  # missing required param
    check("missing param -> parse_fail", f2.op == "parse_fail")
    f3 = canonicalize("", "kv")
    check("empty string -> parse_fail", f3.op == "parse_fail")


def test_content_normalization():
    a = canonicalize("core_memory_add(key='age', value='He is 35.')", "kv")
    b = canonicalize("core_memory_add(key='age', value='he  is 35')", "kv")
    check("case/whitespace/trailing-punct fold at R4", a.content_fp == b.content_fp)
    c = canonicalize("core_memory_add(key='age', value='he is 36')", "kv")
    check("semantic change NOT folded", a.content_fp != c.content_fp)
    check("normalize_text idempotent",
          normalize_text(normalize_text(" A  b. ")) == normalize_text(" A  b. "))
    check("fp is 16 hex", len(content_fingerprint("x")) == 16)


# ---------------------------------------------------------------- signatures

def test_signatures():
    a = canonicalize("core_memory_add(key='age', value='35')", "kv")
    check("R1 sig", signature(a, 1) == "core_memory_add")
    check("R2 sig", signature(a, 2) == "add:core")
    check("R3 sig", signature(a, 3) == "add:core|age")
    check("R4 sig extends R3", signature(a, 4).startswith(signature(a, 3) + "|"))

    read = canonicalize("core_memory_retrieve(key='age')", "kv")
    multi1 = sample_signature([a, read], 3)
    multi2 = sample_signature([read, a], 3)
    check("multi-call order-insensitive", multi1 == multi2)
    check("empty sample -> no_call", sample_signature([], 3) == "no_call")
    check("canonicalize_sample None -> parse_fail",
          canonicalize_sample(None, "kv")[0].op == "parse_fail")
    check("canonicalize_sample [] -> empty", canonicalize_sample([], "kv") == [])


# ---------------------------------------------------------------- vote features

def _kv(key, val):
    return [canonicalize(f"core_memory_add(key='{key}', value='{val}')", "kv")]


def test_entropy_unanimous():
    f = vote_features([_kv("age", "35")] * 8, 3)
    check("unanimous H=0", f["h_act"] == 0.0)
    check("unanimous modal_share=1", f["modal_share"] == 1.0)
    check("unanimous vote_margin=1", f["vote_margin"] == 1.0)
    check("unanimous n_unique=1", f["n_unique"] == 1)
    check("primary matches modal", f["primary_matches_modal"] is True)
    check("h_act_norm 0 when single cluster", f["h_act_norm"] == 0.0)


def test_entropy_all_distinct():
    samples = [_kv(f"k{i}", "v") for i in range(8)]
    f = vote_features(samples, 3)
    check("all-distinct H=ln N", abs(f["h_act"] - math.log(8)) < 1e-9, f["h_act"])
    check("all-distinct h_norm=1", abs(f["h_act_norm"] - 1.0) < 1e-9)
    check("all-distinct margin=0", f["vote_margin"] == 0.0)


def test_entropy_split():
    samples = [_kv("a", "v")] * 5 + [_kv("b", "v")] * 3
    f = vote_features(samples, 3)
    expect = -(5 / 8) * math.log(5 / 8) - (3 / 8) * math.log(3 / 8)
    check("5/3 split entropy", abs(f["h_act"] - expect) < 1e-9)
    check("5/3 modal share", abs(f["modal_share"] - 5 / 8) < 1e-9)
    check("5/3 vote margin", abs(f["vote_margin"] - 2 / 8) < 1e-9)
    f2 = vote_features(list(reversed(samples)), 3)
    check("primary in minority -> not modal", f2["primary_matches_modal"] is False)
    check("modal signature deterministic", f["modal_signature"] == "add:core|a")


def test_resolution_separation():
    # same op, different targets: agree at R2, disagree at R3
    samples = [_kv("a", "v"), _kv("b", "v"), _kv("c", "v"), _kv("a", "v")]
    f2 = vote_features(samples, 2)
    f3 = vote_features(samples, 3)
    check("R2 unanimous when only targets differ", f2["h_act"] == 0.0)
    check("R3 sees target disagreement", f3["h_act"] > 0.0)
    # same op+target, different content: agree at R3, disagree at R4
    samples = [_kv("a", "v1"), _kv("a", "v2")]
    f3 = vote_features(samples, 3)
    f4 = vote_features(samples, 4)
    check("R3 unanimous when only content differs", f3["h_act"] == 0.0)
    check("R4 sees content disagreement", f4["h_act"] > 0.0)


def test_no_call_and_parse_fail_counting():
    samples = [_kv("a", "v"), [], canonicalize_sample(None, "kv"),
               [CanonicalAction("parse_fail", "parse_fail", None, None, "kv")]]
    f = vote_features(samples, 3)
    check("n_no_call counted", f["n_no_call"] == 1)
    check("n_parse_fail counted", f["n_parse_fail"] == 2)
    check("NO_CALL constant is a no-op action", NO_CALL.op == "no_call")


def test_vote_features_all():
    out = vote_features_all([_kv("a", "v")] * 3 + [_kv("b", "v")])
    for tag in ("tool", "op", "target", "full"):
        check(f"all-resolutions key h_act_{tag}", f"h_act_{tag}" in out)
    check("signatures only at R3",
          "modal_signature_target" in out and "modal_signature_op" not in out)
    check("R1 unanimous here", out["h_act_tool"] == 0.0)
    check("R3 split here", out["h_act_target"] > 0.0)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        print(f"[{fn.__name__}]")
        fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
