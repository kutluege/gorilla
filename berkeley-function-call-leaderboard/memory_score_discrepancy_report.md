# BFCL Memory Score Discrepancy Report

## Version Alignment
- Local repo HEAD: `dac44e7ac9db5ff26a01ab0c1ec5de5a1e703b7a`
- Local git describe: `v1.3-44-gdac44e7-dirty`
- Official leaderboard checkpoint: `f7cf735`
- Official PyPI package: `bfcl-eval==2025.12.17`
- Official leaderboard page: https://gorilla.cs.berkeley.edu/leaderboard

The local checkout is not the published leaderboard checkpoint. The worktree is also dirty, so local generation behavior may not match the official run even before considering model-serving settings.

## Score Comparison

| Metric | Official screenshot | Local CSV | Score JSON header | Recomputed from header | Delta local-official |
| --- | ---: | ---: | ---: | ---: | ---: |
| Memory Summary | 17.63% | 21.94% | 21.94% | 21.94% | +4.31 pp |
| memory_kv | 16.13% | 13.55% | 13.55% | 13.55% | -2.58 pp |
| memory_vector | 12.26% | 19.35% | 19.35% | 19.35% | +7.09 pp |
| memory_rec_sum | 24.52% | 32.90% | 32.90% | 32.90% | +8.38 pp |

The local CSV, score JSON headers, and header recomputation agree. This means `data_agentic.csv` is not stale relative to the local memory score JSON files.

## Coverage

| Backend | Correct | Score total | Failure rows in score file | Result rows | Result unique IDs |
| --- | ---: | ---: | ---: | ---: | ---: |
| memory_kv | 21 | 155 | 134 | 155 | 155 |
| memory_vector | 30 | 155 | 125 | 155 | 155 |
| memory_rec_sum | 51 | 155 | 104 | 155 | 155 |

- Memory prompt rows: `155` from `bfcl_eval\data\BFCL_v4_memory.json`
- Memory possible-answer rows: `155` from `bfcl_eval\data\possible_answer\BFCL_v4_memory.json`
- Prereq conversation rows by scenario: {"memory_customer.json": 10, "memory_finance.json": 7, "memory_healthcare.json": 5, "memory_notetaker.json": 5, "memory_student.json": 10}

The three memory backends each scored 155 answer questions locally, matching the memory prompt and answer row counts. The score files contain a summary header plus only failed examples; they do not contain one row per successful example.

## Failure Profile

### memory_kv
- Score file: `score\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\kv\BFCL_v4_memory_kv_score.json`
- Error types: `agentic:answer_not_found`=134
- Sample `memory_kv_1-customer-1` expected `['35', 'thirty five']` but final response was `{'answer': 'I do not know', 'context': 'The query "How old am I?" refers to the user\'s age, but no information about the user\'s age is provided in the given context or memory records.'}`
- Sample `memory_kv_3-customer-3` expected `['strawberry matcha']` but final response was `{'answer': 'I do not know', 'context': 'The user\'s preferences or specific latte types are not mentioned in the provided information.'}`
- Sample `memory_kv_4-customer-4` expected `['38', 'thirty eight']` but final response was `{'answer': 'I do not know', 'context': 'I do not have access to any information about the user\'s kitchen or physical space dimensions.'}`

### memory_vector
- Score file: `score\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\vector\BFCL_v4_memory_vector_score.json`
- Error types: `agentic:answer_not_found`=125
- Sample `memory_vector_3-customer-3` expected `['strawberry matcha']` but final response was `{'answer': 'I do not know', 'context': 'The user\'s preferences regarding specific types of lattes are not mentioned in the provided information.'}`
- Sample `memory_vector_4-customer-4` expected `['38', 'thirty eight']` but final response was `{'answer': 'I do not know', 'context': 'The question asks for a specific measurement about the user\'s kitchen counter, which is not provided in the available information or memory.'}`
- Sample `memory_vector_5-customer-5` expected `['17', 'seventeen']` but final response was `{'answer': 'I do not know', 'context': 'The provided information does not include any details about discounts for new customers or the specific percent off.'}`

### memory_rec_sum
- Score file: `score\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\rec_sum\BFCL_v4_memory_rec_sum_score.json`
- Error types: `agentic:answer_not_found`=104
- Sample `memory_rec_sum_1-customer-1` expected `['35', 'thirty five']` but final response was `{'answer': 'I do not know', 'context': 'The question "How old am I?" cannot be answered based on the provided information or context. There is no personal age data available in the memory or query that would allow for a response.'}`
- Sample `memory_rec_sum_7-customer-7` expected `['11', 'eleven']` but final response was `{'answer': '3–5 business days', 'context': 'The memory content indicates that Michael specifically inquired about the estimated arrival timeline for the Aura Espresso Pro, noting a clear timeline of "3–5 business days" as an example. Thi...`
- Sample `memory_rec_sum_8-customer-8` expected `['family gathering']` but final response was `{'answer': 'A pop-up coffee corner for design clients.', 'context': 'Michael mentioned he is considering upgrading to a higher-end espresso machine to handle volume for events, family gatherings, and potential pop-up coffee corners for h...`

## Memory Snapshot Coverage
- `memory_kv` final snapshots: {"customer_final.json": {"archival_entries": 7, "core_entries": 7, "top_level_keys": 2, "type": "dict"}, "finance_final.json": {"archival_entries": 15, "core_entries": 6, "top_level_keys": 2, "type": "dict"}, "healthcare_final.json": {"archival_entries": 8, "core_entries": 0, "top_level_keys": 2, "type": "dict"}, "notetaker_final.json": {"archival_entries": 1, "core_entries": 7, "top_level_keys": 2, "type": "dict"}, "student_final.json": {"archival_entries": 0, "core_entries": 0, "top_level_keys": 2, "type": "dict"}}
- `memory_kv` prereq checkpoints: {"customer": {"checkpoints": 10, "empty_checkpoints": 0, "last_archival_entries": 7, "last_core_entries": 7}, "customer_memory_combined_with_turn_triggers": {"checkpoints": 1, "empty_checkpoints": 1, "last_archival_entries": 0, "last_core_entries": 0}, "finance": {"checkpoints": 7, "empty_checkpoints": 0, "last_archival_entries": 15, "last_core_entries": 6}, "healthcare": {"checkpoints": 5, "empty_checkpoints": 3, "last_archival_entries": 8, "last_core_entries": 0}, "notetaker": {"checkpoints": 5, "empty_checkpoints": 0, "last_archival_entries": 1, "last_core_entries": 7}, "student": {"checkpoints": 10, "empty_checkpoints": 10, "last_archival_entries": 0, "last_core_entries": 0}}
- `memory_vector` final snapshots: {"customer_final.json": {"archival_entries": 13, "core_entries": 3, "top_level_keys": 2, "type": "dict"}, "finance_final.json": {"archival_entries": 3, "core_entries": 7, "top_level_keys": 2, "type": "dict"}, "healthcare_final.json": {"archival_entries": 0, "core_entries": 7, "top_level_keys": 2, "type": "dict"}, "notetaker_final.json": {"archival_entries": 4, "core_entries": 7, "top_level_keys": 2, "type": "dict"}, "student_final.json": {"archival_entries": 0, "core_entries": 0, "top_level_keys": 2, "type": "dict"}}
- `memory_vector` prereq checkpoints: {"customer": {"checkpoints": 10, "empty_checkpoints": 0, "last_archival_entries": 13, "last_core_entries": 3}, "finance": {"checkpoints": 7, "empty_checkpoints": 0, "last_archival_entries": 3, "last_core_entries": 7}, "healthcare": {"checkpoints": 5, "empty_checkpoints": 4, "last_archival_entries": 0, "last_core_entries": 7}, "notetaker": {"checkpoints": 5, "empty_checkpoints": 0, "last_archival_entries": 4, "last_core_entries": 7}, "student": {"checkpoints": 10, "empty_checkpoints": 10, "last_archival_entries": 0, "last_core_entries": 0}}
- `memory_rec_sum` final snapshots: {"customer_final.json": {"memory_chars": 9939, "top_level_keys": 1, "type": "dict"}, "finance_final.json": {"memory_chars": 7386, "top_level_keys": 1, "type": "dict"}, "healthcare_final.json": {"memory_chars": 2603, "top_level_keys": 1, "type": "dict"}, "notetaker_final.json": {"memory_chars": 3484, "top_level_keys": 1, "type": "dict"}, "student_final.json": {"memory_chars": 0, "top_level_keys": 1, "type": "dict"}}
- `memory_rec_sum` prereq checkpoints: {"customer": {"checkpoints": 8, "empty_checkpoints": 8, "last_archival_entries": 0, "last_core_entries": 0}, "finance": {"checkpoints": 7, "empty_checkpoints": 7, "last_archival_entries": 0, "last_core_entries": 0}, "healthcare": {"checkpoints": 5, "empty_checkpoints": 5, "last_archival_entries": 0, "last_core_entries": 0}, "notetaker": {"checkpoints": 5, "empty_checkpoints": 5, "last_archival_entries": 0, "last_core_entries": 0}, "student": {"checkpoints": 10, "empty_checkpoints": 10, "last_archival_entries": 0, "last_core_entries": 0}}

## Conclusion

The discrepancy is not caused by local CSV aggregation or stale local score files. The local score artifacts are internally consistent.

However, the vector backend has a concrete local generation problem for the `student` scenario: its final snapshot and all ten student prereq checkpoint snapshots contain zero core and archival entries. This directly explains why all scored `memory_vector_*student*` answer cases fail with missing-memory responses.

The most likely cause is that the local outputs were produced under a different evaluation environment than the published leaderboard: different BFCL commit/package, dirty local inference code, and/or different serving/decoding settings. The official row should be reproduced from commit `f7cf735` or `bfcl-eval==2025.12.17`, then compared against these local result JSON files ID-by-ID.
