# BFCL Memory Pipeline Audit

## Version And Diff
- Local HEAD: `dac44e7ac9db5ff26a01ab0c1ec5de5a1e703b7a`
- Local describe: `v1.3-44-gdac44e7-dirty`
- Official target: `f7cf735` / `bfcl-eval==2025.12.17`
- Official leaderboard: https://gorilla.cs.berkeley.edu/leaderboard

Changed memory-relevant paths versus official target:

```text
M	berkeley-function-call-leaderboard/bfcl_eval/_llm_response_generation.py
M	berkeley-function-call-leaderboard/bfcl_eval/model_handler/local_inference/base_oss_handler.py
M	berkeley-function-call-leaderboard/bfcl_eval/utils.py
```

Diff stat:

```text
.../bfcl_eval/_llm_response_generation.py          | 24 ++++++++++
 .../local_inference/base_oss_handler.py            | 52 +++++++++++++++++++---
 .../bfcl_eval/utils.py                             |  2 +-
 3 files changed, 72 insertions(+), 6 deletions(-)
```

## Pipeline Trace
- `load_dataset_entry(memory_*)` expands `BFCL_v4_memory.json` with prereq conversations from `bfcl_eval/data/memory_prereq_conversation`.
- `populate_test_cases_with_predefined_functions` injects backend-specific function docs from `bfcl_eval/data/multi_turn_func_doc`.
- `_llm_response_generation.py` schedules prereq dependencies, writes result JSONL, and only flushes memory snapshots at the end of each memory prereq entry.
- `execute_multi_turn_func_call` keeps memory backend instances in module globals keyed by model, test id, and class; decoded calls mutate that instance.
- `_prepare_snapshot` loads `<scenario>_final.json` for non-first prereq and answer entries, while `_flush_memory_to_local_file` writes both prereq checkpoints and the final scenario snapshot.

Important stale-output risk: generation cleanup checks `model_result_dir / 'memory_snapshot' / test_category`, but actual memory snapshots are under `agentic/memory/<backend>/memory_snapshot`. A full overwrite may delete result files but leave old snapshots unless the current code path is corrected or snapshots are manually cleared in an isolated directory.

## Backend Scenario Health

### kv

| Scenario | Prereq Rows | Raw Tool Tags | Decoded Memory Calls | Final Snapshot | Score |
| --- | --- | --- | --- | --- | --- |
| customer | 10 | 99 | 82 | core=7 archival=7 | 4/30 (13.33%) |
| healthcare | 5 | 8 | 8 | core=0 archival=8 | 3/25 (12.00%) |
| finance | 7 | 59 | 55 | core=6 archival=15 | 2/25 (8.00%) |
| student | 10 | 0 | 0 | core=0 archival=0 | 1/50 (2.00%) |
| notetaker | 5 | 36 | 36 | core=7 archival=1 | 11/25 (44.00%) |

### vector

| Scenario | Prereq Rows | Raw Tool Tags | Decoded Memory Calls | Final Snapshot | Score |
| --- | --- | --- | --- | --- | --- |
| customer | 10 | 119 | 106 | core=3 archival=13 | 4/30 (13.33%) |
| healthcare | 5 | 12 | 12 | core=7 archival=0 | 10/25 (40.00%) |
| finance | 7 | 71 | 71 | core=7 archival=3 | 4/25 (16.00%) |
| student | 10 | 0 | 0 | core=0 archival=0 | 1/50 (2.00%) |
| notetaker | 5 | 25 | 25 | core=7 archival=4 | 11/25 (44.00%) |

### rec_sum

| Scenario | Prereq Rows | Raw Tool Tags | Decoded Memory Calls | Final Snapshot | Score |
| --- | --- | --- | --- | --- | --- |
| customer | 10 | 73 | 67 | memory_chars=9939 | 10/30 (33.33%) |
| healthcare | 5 | 17 | 17 | memory_chars=2603 | 9/25 (36.00%) |
| finance | 7 | 51 | 51 | memory_chars=7386 | 8/25 (32.00%) |
| student | 10 | 0 | 0 | memory_chars=0 | 1/50 (2.00%) |
| notetaker | 5 | 24 | 24 | memory_chars=3484 | 23/25 (92.00%) |

## Tool Availability Check

### kv
- Function docs loaded: `15`
- Memory write tools available: `8`
- Write tools: `archival_memory_add, archival_memory_clear, archival_memory_remove, archival_memory_replace, core_memory_add, core_memory_clear, core_memory_remove, core_memory_replace`
- First expanded student prereq id: `memory_kv_prereq_22-student-0`
- First student prompt preview: user
Oh, hey! Yeah, I’m a Computer Science major—fourth year now, so it’s basically crunch time for me. My schedule this semester is absolutely killer but in a good way, I guess. I’m finally taking those higher-level courses that seemed so far off when I was just a freshman. It’s kind of surreal to be at this stage, you know? It feels like only yesterday I was sitting in an intro class learning about simple data structures, but now I’m all in with complex algorithmic concepts, advanced system de
- First expanded customer prereq id: `memory_kv_prereq_0-customer-0`
- First customer prompt preview: user
Hey there! Thanks for getting back to me so quickly. I really appreciate having the chance to talk to someone who actually knows about your products. My name is Michael, and this is my first time interacting with your company in any way—so I'd love to start with a bit of an introduction and then dive into my questions.
user
I'm 35 years old, live in Seattle, and am pretty serious about both my work and my hobbies. I work as a freelance graphic designer, which means I'm home a lot, and I rea

### vector
- Function docs loaded: `12`
- Memory write tools available: `8`
- Write tools: `archival_memory_add, archival_memory_clear, archival_memory_remove, archival_memory_update, core_memory_add, core_memory_clear, core_memory_remove, core_memory_update`
- First expanded student prereq id: `memory_vector_prereq_22-student-0`
- First student prompt preview: user
Oh, hey! Yeah, I’m a Computer Science major—fourth year now, so it’s basically crunch time for me. My schedule this semester is absolutely killer but in a good way, I guess. I’m finally taking those higher-level courses that seemed so far off when I was just a freshman. It’s kind of surreal to be at this stage, you know? It feels like only yesterday I was sitting in an intro class learning about simple data structures, but now I’m all in with complex algorithmic concepts, advanced system de
- First expanded customer prereq id: `memory_vector_prereq_0-customer-0`
- First customer prompt preview: user
Hey there! Thanks for getting back to me so quickly. I really appreciate having the chance to talk to someone who actually knows about your products. My name is Michael, and this is my first time interacting with your company in any way—so I'd love to start with a bit of an introduction and then dive into my questions.
user
I'm 35 years old, live in Seattle, and am pretty serious about both my work and my hobbies. I work as a freelance graphic designer, which means I'm home a lot, and I rea

### rec_sum
- Function docs loaded: `5`
- Memory write tools available: `4`
- Write tools: `memory_append, memory_clear, memory_replace, memory_update`
- First expanded student prereq id: `memory_rec_sum_prereq_22-student-0`
- First student prompt preview: user
Oh, hey! Yeah, I’m a Computer Science major—fourth year now, so it’s basically crunch time for me. My schedule this semester is absolutely killer but in a good way, I guess. I’m finally taking those higher-level courses that seemed so far off when I was just a freshman. It’s kind of surreal to be at this stage, you know? It feels like only yesterday I was sitting in an intro class learning about simple data structures, but now I’m all in with complex algorithmic concepts, advanced system de
- First expanded customer prereq id: `memory_rec_sum_prereq_0-customer-0`
- First customer prompt preview: user
Hey there! Thanks for getting back to me so quickly. I really appreciate having the chance to talk to someone who actually knows about your products. My name is Michael, and this is my first time interacting with your company in any way—so I'd love to start with a bit of an introduction and then dive into my questions.
user
I'm 35 years old, live in Seattle, and am pretty serious about both my work and my hobbies. I work as a freelance graphic designer, which means I'm home a lot, and I rea

## Deep Failure Notes

### kv student
- Student prereq rows: `10`
- Raw `<tool_call>` tags: `0`
- Decoded memory calls: `0`
- First prereq result preview: That's an amazing journey—absolutely nothing short of inspiring! Transitioning from introductory data structures to advanced topics like quantum computing is a huge leap, and you're clearly embracing it with curiosity and resilience. It's not just about the technical depth; it's about how you're growing as a thinker and problem-solver.  

I can imagine how overwhelming it must feel to juggle a packed schedule, especially with higher-level courses that demand both deep conceptual understanding and practical application. But you're already ahead of the curve by recognizing the significance of this stage. It’s not just about surviving the workload—it’s about thriving in it.  

If you're open to
- Checkpoint progression: `memory_kv_prereq_22-student-0.json:c0/a0; memory_kv_prereq_23-student-1.json:c0/a0; memory_kv_prereq_24-student-2.json:c0/a0; memory_kv_prereq_25-student-3.json:c0/a0; memory_kv_prereq_26-student-4.json:c0/a0; memory_kv_prereq_27-student-5.json:c0/a0; memory_kv_prereq_28-student-6.json:c0/a0; memory_kv_prereq_29-student-7.json:c0/a0; memory_kv_prereq_30-student-8.json:c0/a0; memory_kv_prereq_31-student-9.json:c0/a0`

### vector student
- Student prereq rows: `10`
- Raw `<tool_call>` tags: `0`
- Decoded memory calls: `0`
- First prereq result preview: That's an incredible journey—what a transformation from intro classes to diving into quantum computing! It's so inspiring to hear how far you've come. The shift from basic data structures to advanced topics like system design and quantum computing really shows your growth and passion for the field.  

I'm curious—what’s one of the most challenging or exciting concepts you've encountered in your higher-level courses so far? And how do you balance such a packed schedule while still staying motivated and managing stress? I’d love to hear more about it! 😊
That’s absolutely awesome—Advanced Algorithms is one of the most intellectually rewarding courses you can take, especially when it bridges the
- Checkpoint progression: `memory_vector_prereq_22-student-0.json:c0/a0; memory_vector_prereq_23-student-1.json:c0/a0; memory_vector_prereq_24-student-2.json:c0/a0; memory_vector_prereq_25-student-3.json:c0/a0; memory_vector_prereq_26-student-4.json:c0/a0; memory_vector_prereq_27-student-5.json:c0/a0; memory_vector_prereq_28-student-6.json:c0/a0; memory_vector_prereq_29-student-7.json:c0/a0; memory_vector_prereq_30-student-8.json:c0/a0; memory_vector_prereq_31-student-9.json:c0/a0`

### rec_sum student
- Student prereq rows: `10`
- Raw `<tool_call>` tags: `0`
- Decoded memory calls: `0`
- First prereq result preview: That’s an amazing journey—absolutely nothing short of inspiring! Transitioning from introductory data structures to quantum computing basics is a huge leap, and it shows how much you’ve grown both academically and personally. It’s not just about the technical depth; it’s about the confidence and curiosity you’ve built along the way.  

I’m sure you’re feeling a mix of excitement and pressure, especially with the higher-level courses. But remember, even the most brilliant minds in CS faced similar moments—overwhelmed, unsure, and yet still pushing forward. The fact that you’re embracing these challenges now is a sign of real maturity in your academic path.  

If you’d like, I can help you bre
- Checkpoint progression: `memory_rec_sum_prereq_22-student-0.json:0 chars; memory_rec_sum_prereq_23-student-1.json:0 chars; memory_rec_sum_prereq_24-student-2.json:0 chars; memory_rec_sum_prereq_25-student-3.json:0 chars; memory_rec_sum_prereq_26-student-4.json:0 chars; memory_rec_sum_prereq_27-student-5.json:0 chars; memory_rec_sum_prereq_28-student-6.json:0 chars; memory_rec_sum_prereq_29-student-7.json:0 chars; memory_rec_sum_prereq_30-student-8.json:0 chars; memory_rec_sum_prereq_31-student-9.json:0 chars`

### vector healthcare
The final healthcare vector snapshot has populated core memory and empty archival memory. This is not equivalent to the student hard failure: answer-time prompts include core memory in context, while archival memory requires explicit retrieval. This needs score-level review to decide whether empty archival is acceptable or a retrieval weakness.
- Final snapshot: `{'exists': True, 'core_entries': 7, 'archival_entries': 0, 'memory_chars': 0, 'empty': False}`
- Decoded memory calls: `12`

## Artifact Timing
Memory artifact modification times can help decide whether result files were produced before or after local code edits, but they cannot prove the exact runtime code by themselves.

| Path | Modified | Bytes |
| --- | --- | --- |
| result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\kv\BFCL_v4_memory_kv_prereq_result.json | 2026-03-15T01:39:40 | 2233898 |
| result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\kv\BFCL_v4_memory_kv_result.json | 2026-03-15T01:39:40 | 539438 |
| result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\kv\memory_snapshot\customer_final.json | 2026-03-15T01:34:02 | 5934 |
| result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\kv\memory_snapshot\finance_final.json | 2026-03-15T01:29:11 | 9050 |
| result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\kv\memory_snapshot\healthcare_final.json | 2026-03-15T01:28:59 | 2396 |
| result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\kv\memory_snapshot\notetaker_final.json | 2026-03-15T01:24:23 | 1417 |
| result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\kv\memory_snapshot\student_final.json | 2026-03-15T01:32:46 | 55 |
| result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\rec_sum\BFCL_v4_memory_rec_sum_prereq_result.json | 2026-03-15T01:39:40 | 2912379 |
| result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\rec_sum\BFCL_v4_memory_rec_sum_result.json | 2026-03-15T01:39:40 | 1107472 |
| result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\rec_sum\memory_snapshot\customer_final.json | 2026-03-15T01:39:38 | 10135 |
| result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\rec_sum\memory_snapshot\finance_final.json | 2026-03-15T01:29:27 | 7483 |
| result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\rec_sum\memory_snapshot\healthcare_final.json | 2026-03-15T01:29:52 | 2694 |
| result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\rec_sum\memory_snapshot\notetaker_final.json | 2026-03-15T01:24:34 | 3731 |
| result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\rec_sum\memory_snapshot\student_final.json | 2026-03-15T01:33:34 | 22 |
| result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\vector\BFCL_v4_memory_vector_prereq_result.json | 2026-03-15T01:39:40 | 2487657 |
| result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\vector\BFCL_v4_memory_vector_result.json | 2026-03-15T01:39:40 | 579226 |
| result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\vector\memory_snapshot\customer_final.json | 2026-03-15T01:34:32 | 7584 |
| result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\vector\memory_snapshot\finance_final.json | 2026-03-15T01:29:58 | 5046 |
| result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\vector\memory_snapshot\healthcare_final.json | 2026-03-15T01:29:42 | 2182 |
| result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\vector\memory_snapshot\notetaker_final.json | 2026-03-15T01:25:02 | 2467 |
| result\Qwen_Qwen3-4B-Instruct-2507-FC\agentic\memory\vector\memory_snapshot\student_final.json | 2026-03-15T01:33:23 | 155 |

## Reproduction Commands
Use an isolated result directory, include input logs, and run single-threaded to make prompt/tool availability auditable. Do not point these commands at the existing `result/` tree.

```powershell
python -m bfcl_eval generate --model Qwen/Qwen3-4B-Instruct-2507-FC --test-category memory_vector --result-dir tmp\bfcl_memory_repro --allow-overwrite --include-input-log --num-threads 1
python -m bfcl_eval evaluate --model Qwen/Qwen3-4B-Instruct-2507-FC --test-category memory_vector --result-dir tmp\bfcl_memory_repro --score-dir tmp\bfcl_memory_repro_score
```

Repeat with `memory_kv` and `memory_rec_sum`, then compare student and customer prereq checkpoints. To match the official row, run from commit `f7cf735` or package `bfcl-eval==2025.12.17` with the same model-serving backend and decoding settings.

## Conclusion
The empty student memory snapshots are caused upstream of scoring: the student prereq generations produced zero raw tool calls and zero decoded memory calls across `kv`, `vector`, and `rec_sum`. Memory tools are present in the function docs, so the immediate failure is model-generation behavior or prompt/model-serving mismatch, not missing backend APIs.

The strongest next check is an isolated, single-threaded rerun with input logs at the official checkpoint. If student still emits no memory calls while customer does, compare the compiled prompts and model outputs. If official checkpoint reproduces the leaderboard values, regenerate all memory artifacts from that clean environment and discard the current affected local memory results.
