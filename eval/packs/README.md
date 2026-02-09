# Eval Packs

This directory contains bundled benchmark/eval datasets that can be loaded by:

- `/rlm import-evals pack=eval/packs/<file>`
- `/rlm bench preset=<name> pack=eval/packs/<file>`

Included sample packs:

- `pydantic_time_range_v1.yaml` (dataset-style `cases`)
- `google_adk_memory_eval.json` (Google ADK-style `eval_cases`)
- `superoptix_qa_pairs.json` (generic JSON records)
