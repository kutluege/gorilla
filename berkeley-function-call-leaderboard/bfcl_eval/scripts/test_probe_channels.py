"""
Offline tests for the SS4.1b paraphrase probe channel (Plan 3 Step 3 / G3)
and the G4 representativeness helper. The HF paraphraser is NEVER loaded here
-- generation is injected, so the suite is fast and offline.

Run:  python bfcl_eval/scripts/test_probe_channels.py
"""

import inspect
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bfcl_eval.model_handler.middleware.probe_gen as pg  # noqa: E402
from bfcl_eval.model_handler.middleware.probe_gen import (  # noqa: E402
    ProbeConfig,
    _paraphrase_probes,
    generate_decision_probes,
    generate_item_probes,
)
from bfcl_eval.scripts.validate_probe_representativeness import (  # noqa: E402
    question_agreement,
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


USER = "My favorite drink is espresso coffee these days."


def fake_gen(paraphrases):
    def gen(text, n, model_name):
        return paraphrases
    return gen


def run():
    print("[default off: byte-identical to Plan-2 behavior]")
    cfg_off = ProbeConfig()
    probes = generate_decision_probes(USER, ["espresso"], cfg_off)
    check("no paraphrase probes when disabled",
          all(p.channel.startswith("template:") for p in probes))
    check("_paraphrase_probes disabled -> []",
          _paraphrase_probes(USER, cfg_off, generate_fn=fake_gen(["x"])) == [])

    print("[enabled channel via injected generator]")
    cfg_on = ProbeConfig(paraphrase=True, probes_n=6, paraphrase_n=2)
    outs = ["These days I prefer drinking espresso coffee.",
            "Espresso coffee is my current favorite beverage.",
            "A third paraphrase that should be trimmed."]
    pp = _paraphrase_probes(USER, cfg_on, generate_fn=fake_gen(outs))
    check("capped at paraphrase_n", len(pp) == 2, str(len(pp)))
    check("channel tag carries model short-name",
          all(p.channel == "paraphrase:flan-t5-small" for p in pp),
          str([p.channel for p in pp]))
    check("provenance stays user_text", all(p.source == "user_text" for p in pp))
    check("echo of the source sentence filtered",
          _paraphrase_probes(USER, cfg_on, generate_fn=fake_gen([USER, " ", None]))
          == [])
    check("oversize output filtered",
          _paraphrase_probes(USER, cfg_on, generate_fn=fake_gen(["x" * 400])) == [])
    check("generator failure ([]) degrades to template-only",
          _paraphrase_probes(USER, cfg_on, generate_fn=fake_gen([])) == [])

    print("[cap ordering: templates always survive]")
    real_fn = pg._hf_paraphrase
    pg._hf_paraphrase = fake_gen(outs[:2])
    try:
        capped = generate_decision_probes(USER, ["espresso"],
                                          ProbeConfig(paraphrase=True, probes_n=3,
                                                      paraphrase_n=2))
        check("probes_n cap trims paraphrases, not templates",
              len(capped) == 3
              and all(p.channel.startswith("template:") for p in capped))
        full = generate_decision_probes(USER, ["espresso"],
                                        ProbeConfig(paraphrase=True, probes_n=5,
                                                    paraphrase_n=2))
        check("probes_n=5 admits the paraphrase channel",
              any(p.channel.startswith("paraphrase:") for p in full)
              and full[0].channel.startswith("template:"))
    finally:
        pg._hf_paraphrase = real_fn

    print("[anti-circularity invariants unchanged]")
    sig = inspect.signature(generate_decision_probes)
    check("decision probes still cannot receive candidate text",
          set(sig.parameters) == {"user_text", "verbatim_values", "cfg"},
          str(list(sig.parameters)))
    item = generate_item_probes("vector", "The user is 35 years old.")
    check("item probes stay template-only",
          all(p.channel.startswith("template:") for p in item))

    print("[ProbeConfig env plumbing]")
    old = {k: os.environ.get(k) for k in
           ("GOV_PROBE_PARAPHRASE", "GOV_PROBE_MODEL", "GOV_PROBE_PARAPHRASE_N")}
    try:
        os.environ["GOV_PROBE_PARAPHRASE"] = "1"
        os.environ["GOV_PROBE_MODEL"] = "org/tiny-para"
        os.environ["GOV_PROBE_PARAPHRASE_N"] = "9"
        cfg = ProbeConfig.from_env()
        check("env: model + flag read",
              cfg.paraphrase and cfg.paraphrase_model == "org/tiny-para")
        check("paraphrase_n clamped to [1,3]", cfg.paraphrase_n == 3)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    print("[G4 representativeness helper]")
    refs = ["user_name", "favorite_drink", "meeting_notes"]
    texts = ["user name: Michael", "favorite drink: espresso coffee",
             "meeting notes: budget review March 3"]
    res = question_agreement("kv", refs, texts, "what is my favorite drink")
    check("real question resolves top-1 entry",
          res is not None and res[0] == "favorite_drink", str(res))
    check("probes agree with the real question on a clean corpus",
          res[1] > 0.5, str(res))
    check("empty corpus -> None", question_agreement("kv", [], [], "q") is None)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run())
