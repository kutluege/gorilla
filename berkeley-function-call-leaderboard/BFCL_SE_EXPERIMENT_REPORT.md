# Semantic-Entropy Gate vs Baseline — BFCL V4 Memory

Baseline: `Qwen/Qwen3-4B-Instruct-2507-FC` (greedy, temperature 0.001). Gated: `Qwen/Qwen3-4B-Instruct-2507-FC-SE` (N samples per memory step, semantic clustering, majority commit, destructive ops gated on cluster entropy).

## Accuracy

| Backend | Baseline | Gated | Delta | Discordant (b/c) | McNemar p | 95% CI (delta) |
|---|---|---|---|---|---|---|
| kv | 15.48% (24/155) | 11.61% (18/155) | -3.87% | 12/6 | 0.2379 | [-9.03%, +1.29%] |
| vector | 17.42% (27/155) | 16.13% (25/155) | -1.29% | 17/15 | 0.8601 | [-8.39%, +5.81%] |
| rec_sum | 27.10% (42/155) | 26.45% (41/155) | -0.65% | 12/11 | 1.0000 | [-6.45%, +5.16%] |
| **overall** | 20.00% | 18.06% | -1.94% | 41/32 | 0.3492 | [-5.59%, +1.72%] |

Discordant b/c = baseline-correct-gated-wrong / baseline-wrong-gated-correct. McNemar is an exact two-sided binomial test on the discordant pairs.

## Gate audit

Total gated steps: **2024**
- **prereq**: 1502 steps | mean entropy 1.360 bits | unanimous 567 (38%)
- **recall**: 522 steps | mean entropy 0.517 bits | unanimous 311 (60%)

| Decision | Count |
|---|---|
| commit_majority | 1821 |
| commit_overwrite | 158 |
| commit_destructive | 22 |
| fallback_from_overwrite | 14 |
| fallback_from_destructive | 7 |
| forced_destructive | 2 |

Destructive/overwrite majority proposed: **31** destructive, **172** overwrite. Blocked with safe fallback: **21**. Forced through (no safe cluster): **2**.

### Blocked destructive/overwrite steps (fallback actions)

| Test id | H | m | Majority proposed | Fallback executed |
|---|---|---|---|---|
| memory_kv_prereq_0-customer-0 | 1.371 | 0.6 | core_memory_remove | (text) |
| memory_kv_prereq_1-customer-1 | 2.3219 | 0.2 | core_memory_remove | archival_memory_add(key='user_interest_steam_wand',value='Michael is interested  |
| memory_kv_prereq_5-customer-5 | 0.7219 | 0.8 | core_memory_remove | archival_memory_add(key='user_introduction_michael',value='Michael, 35, lives in |
| memory_kv_prereq_6-customer-6 | 1.371 | 0.6 | core_memory_remove | core_memory_retrieve_all() |
| memory_kv_prereq_17-finance-2 | 1.9219 | 0.4 | core_memory_remove | (text) |
| memory_kv_prereq_21-finance-6 | 1.371 | 0.6 | core_memory_remove | core_memory_list_keys() |
| memory_kv_prereq_36-notetaker-4 | 1.9219 | 0.4 | core_memory_remove | (text) |
| memory_rec_sum_prereq_2-customer-2 | 1.371 | 0.6 | memory_update | (text) |
| memory_rec_sum_prereq_2-customer-2 | 1.9219 | 0.4 | memory_replace | memory_retrieve() |
| memory_rec_sum_prereq_6-customer-6 | 1.9219 | 0.4 | memory_update | (text) |
| memory_rec_sum_prereq_6-customer-6 | 1.9219 | 0.4 | memory_update | (text) |
| memory_rec_sum_prereq_7-customer-7 | 1.9219 | 0.4 | memory_update | (text) |
| memory_rec_sum_prereq_7-customer-7 | 1.371 | 0.6 | memory_update | (text) |
| memory_rec_sum_prereq_8-customer-8 | 2.3219 | 0.2 | memory_update | (text) |
| memory_rec_sum_prereq_17-finance-2 | 1.9219 | 0.4 | memory_replace | (text) |
| memory_vector_prereq_0-customer-0 | 1.371 | 0.6 | core_memory_update | archival_memory_add(text='User Michael, inquires about warranty policy for coffe |
| memory_vector_prereq_3-customer-3 | 2.3219 | 0.2 | core_memory_update | (text) |
| memory_vector_prereq_5-customer-5 | 1.371 | 0.6 | core_memory_update | (text) |
| memory_vector_prereq_16-finance-1 | 1.9219 | 0.4 | core_memory_update | (text) |
| memory_vector_prereq_16-finance-1 | 1.371 | 0.6 | core_memory_update | (text) |
| memory_vector_prereq_36-notetaker-4 | 2.3219 | 0.2 | core_memory_update | (text) |
