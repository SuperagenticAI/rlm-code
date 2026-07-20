# Practical RLM codebase use cases

RLM is most useful when an answer requires structured exploration across many
files: architecture understanding, onboarding, cross-file audits, refactor
planning, dependency mapping, test-gap analysis, incident investigation, and
migration planning.

It is usually unnecessary for tiny one-file edits, simple syntax fixes, or work
where one search and one file read are sufficient.

## Suggested workflow

1. Install RLM Code and configure a secure execution backend.
2. Connect a local or hosted model.
3. Run one evidence-heavy repository question with `env=pure_rlm profile=lid`.
4. Inspect the persisted trajectory, including root/submodel usage.
5. Compare the result with a direct-model or retrieval-only baseline.

Example:

```text
/rlm run env=pure_rlm profile=lid context_profile=evidence steps=8 "Map this repository's architecture and tell a new maintainer which files to read first."
```

RAG and RLM answer different questions. Retrieval asks which chunks appear
relevant; RLM asks what program should run over the available environment to
understand the problem. A production harness can use both.
