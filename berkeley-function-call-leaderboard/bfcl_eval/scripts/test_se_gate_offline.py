"""
Offline validation of the semantic-entropy gate (no server, no BFCL run).

Feeds hand-crafted candidate sets straight into ``choose()`` and asserts the decision
policy, including the thesis failure-mode fix in miniature: an uncertain destructive
majority must fall back to a safe additive action.

Run:  python bfcl_eval/scripts/test_se_gate_offline.py
"""

import math
import sys

from bfcl_eval.model_handler.middleware.semantic_entropy import (
    SEConfig,
    build_candidate,
    choose,
)


def tool_candidate(*calls: str):
    text = "".join(
        "<tool_call>\n{{\"name\": \"{}\"}}\n</tool_call>".format(c.split("(", 1)[0])
        for c in calls
    )
    return build_candidate(text, list(calls))


def text_candidate(answer: str):
    return build_candidate(answer, None)


PASSED = []
FAILED = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}  {detail}")


def scenario_1_unanimous_add(cfg):
    print("\n[1] 5x paraphrased core_memory_add -> 1 cluster, H=0, commit")
    candidates = [
        tool_candidate("core_memory_add(key='user_age', value='35')"),
        tool_candidate("core_memory_add(key='age', value='35')"),
        tool_candidate("core_memory_add(key='age', value='35 years old')"),
        tool_candidate("core_memory_add(key='user_age', value='35 years')"),
        tool_candidate("core_memory_add(key='age', value='age 35')"),
    ]
    idx, rec = choose(candidates, cfg)
    check("1.single-cluster", rec["num_clusters"] == 1, f"clusters={rec['cluster_sizes']}")
    check("1.zero-entropy", rec["entropy"] == 0.0, f"H={rec['entropy']}")
    check("1.commit", rec["decision"] == "commit_majority", rec["decision"])


def scenario_2_split_safe(cfg):
    print("\n[2] 3x core_memory_add + 2x archival_memory_add -> 2 clusters, commit majority")
    candidates = [
        tool_candidate("core_memory_add(key='name', value='Alice')"),
        tool_candidate("core_memory_add(key='name', value='Alice Smith')"),
        tool_candidate("core_memory_add(key='user_name', value='Alice')"),
        tool_candidate("archival_memory_add(key='name', value='Alice')"),
        tool_candidate("archival_memory_add(key='user_name', value='Alice')"),
    ]
    idx, rec = choose(candidates, cfg)
    check("2.two-clusters", rec["num_clusters"] == 2, f"clusters={rec['cluster_sizes']}")
    check(
        "2.entropy",
        abs(rec["entropy"] - (-(0.6 * math.log2(0.6) + 0.4 * math.log2(0.4)))) < 0.01,
        f"H={rec['entropy']}",
    )
    check("2.commit-majority-safe", rec["decision"] == "commit_majority", rec["decision"])
    check(
        "2.chose-core-add",
        "core_memory_add" in candidates[idx].func_names,
        str(candidates[idx].func_names),
    )


def scenario_3_uncertain_destructive(cfg):
    print("\n[3] 2x core_memory_remove + 2x archival_memory_add + 1x core_memory_clear")
    print("    -> destructive majority uncertain -> fall back to archival_memory_add")
    candidates = [
        tool_candidate("core_memory_remove(key='name')"),
        tool_candidate("core_memory_remove(key='user_name')"),
        tool_candidate("archival_memory_add(key='name', value='Alice')"),
        tool_candidate("archival_memory_add(key='user_name', value='Alice')"),
        tool_candidate("core_memory_clear()"),
    ]
    idx, rec = choose(candidates, cfg)
    check("3.fallback", rec["fallback"] is True, rec["decision"])
    check(
        "3.chose-safe-add",
        candidates[idx].func_names == ("archival_memory_add",),
        str(candidates[idx].func_names),
    )
    check("3.not-forced", rec["forced_destructive"] is False, "")


def scenario_4_text_majority(cfg):
    print("\n[4] 5 free-text answers, 4x '35', 1x 'I don't know' -> majority medoid '35'")
    candidates = [
        text_candidate("The user is 35 years old."),
        text_candidate("You are 35 years old."),
        text_candidate("The age is 35 years old."),
        text_candidate("The user is 35 years of age."),
        text_candidate("I don't know the answer to that."),
    ]
    idx, rec = choose(candidates, cfg)
    check("4.majority-35", "35" in candidates[idx].text, candidates[idx].text)
    check("4.commit", rec["decision"] == "commit_majority", rec["decision"])
    check(
        "4.dontknow-alone",
        rec["cluster_sizes"][0] == 4 and rec["num_clusters"] == 2,
        f"clusters={rec['cluster_sizes']}",
    )


def scenario_4b_different_answers(cfg):
    print("\n[4b] same template, different answers -> never merge (coffee vs tea)")
    candidates = [
        text_candidate("Your favorite drink is coffee."),
        text_candidate("Your favorite drink is tea."),
        text_candidate("Your favorite drink is coffee."),
    ]
    idx, rec = choose(candidates, cfg)
    check("4b.two-clusters", rec["num_clusters"] == 2, f"clusters={rec['cluster_sizes']}")
    check("4b.majority-coffee", "coffee" in candidates[idx].text, candidates[idx].text)


def scenario_5_mixed_kinds(cfg):
    print("\n[5] tool-call and text candidates never share a cluster")
    candidates = [
        tool_candidate("core_memory_add(key='age', value='35')"),
        text_candidate("core memory add key age value 35"),  # same words, different kind
        tool_candidate("core_memory_add(key='age', value='35')"),
    ]
    idx, rec = choose(candidates, cfg)
    kinds_per_cluster = {}
    for c in rec["candidates"]:
        kinds_per_cluster.setdefault(c["cluster"], set()).add(c["kind"])
    check(
        "5.kinds-never-merge",
        all(len(k) == 1 for k in kinds_per_cluster.values()),
        str(kinds_per_cluster),
    )
    check("5.majority-toolcall", candidates[idx].kind == "tool_call", candidates[idx].kind)


def scenario_6_different_tool_names(cfg):
    print("\n[6] different tool names never merge, even with near-identical arguments")
    candidates = [
        tool_candidate("core_memory_add(key='age', value='35')"),
        tool_candidate("core_memory_remove(key='age', value='35')"),
        tool_candidate("core_memory_add(key='age', value='35')"),
    ]
    idx, rec = choose(candidates, cfg)
    check("6.two-clusters", rec["num_clusters"] == 2, f"clusters={rec['cluster_sizes']}")
    check(
        "6.majority-add",
        candidates[idx].func_names == ("core_memory_add",),
        str(candidates[idx].func_names),
    )


def scenario_7_forced_destructive(cfg):
    print("\n[7] every cluster destructive + uncertain -> forced majority, flagged")
    candidates = [
        tool_candidate("core_memory_remove(key='a')"),
        tool_candidate("core_memory_clear()"),
        tool_candidate("archival_memory_clear()"),
        tool_candidate("core_memory_remove(key='completely different thing')"),
        tool_candidate("archival_memory_remove(key='b')"),
    ]
    idx, rec = choose(candidates, cfg)
    check("7.forced", rec["forced_destructive"] is True, rec["decision"])


def scenario_8_confident_destructive(cfg):
    print("\n[8] unanimous destructive op -> allowed through the gate (H=0, m=1)")
    candidates = [
        tool_candidate("core_memory_remove(key='name')"),
        tool_candidate("core_memory_remove(key='name')"),
        tool_candidate("core_memory_remove(key='user_name')"),
        tool_candidate("core_memory_remove(key='name')"),
        tool_candidate("core_memory_remove(key='name')"),
    ]
    idx, rec = choose(candidates, cfg)
    check("8.commit-destructive", rec["decision"] == "commit_destructive", rec["decision"])
    check("8.no-fallback", rec["fallback"] is False, "")


def scenario_9_gate_off(cfg):
    print("\n[9] SE_GATE_DESTRUCTIVE off (arm E2): uncertain destructive majority commits")
    cfg_off = SEConfig(**{**cfg.__dict__, "gate_destructive": False})
    candidates = [
        tool_candidate("core_memory_remove(key='name')"),
        tool_candidate("core_memory_remove(key='user_name')"),
        tool_candidate("archival_memory_add(key='name', value='Alice')"),
        tool_candidate("archival_memory_add(key='user_name', value='Alice')"),
        tool_candidate("core_memory_clear()"),
    ]
    idx, rec = choose(candidates, cfg_off)
    check("9.pure-majority", rec["decision"] == "commit_majority", rec["decision"])


def main():
    cfg = SEConfig()  # library defaults, not env, so the test is deterministic
    print(f"Gate config: {cfg}")
    scenario_1_unanimous_add(cfg)
    scenario_2_split_safe(cfg)
    scenario_3_uncertain_destructive(cfg)
    scenario_4_text_majority(cfg)
    scenario_4b_different_answers(cfg)
    scenario_5_mixed_kinds(cfg)
    scenario_6_different_tool_names(cfg)
    scenario_7_forced_destructive(cfg)
    scenario_8_confident_destructive(cfg)
    scenario_9_gate_off(cfg)
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        sys.exit(1)


if __name__ == "__main__":
    main()
