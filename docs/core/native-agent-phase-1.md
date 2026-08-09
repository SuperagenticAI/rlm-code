# Native agent runtime: Phase 1 architecture

Status: accepted for the first vertical slice.

## Decision

The interactive coding agent is a new `rlm_code.agent` service. It does not
replace, wrap, or act as a backend for `RLMRunner`. Research runs keep their
deterministic environments, delegation, trajectories, replay, benchmarks, and
framework adapters.

Every model request made by the interactive runtime contains exactly one
executable tool definition: `python`. Repository and shell operations are
objects inside that persistent Python environment. They are never independent
model tool definitions.

The runtime is an asynchronous state machine. A compatibility adapter may move
one blocking call of the existing `LLMConnector` to a worker thread, but the
agent loop itself is not the synchronous research runner and is not wrapped in
`asyncio.to_thread()`.

## Phase 1 boundaries

| File or package | Responsibility |
| --- | --- |
| `rlm_code.agent.events` | Typed events, stable sequence numbers, async stream, append-only journal |
| `rlm_code.agent.model` | One-tool model contract and legacy connector compatibility |
| `rlm_code.agent.kernel` | Persistent restricted Python worker, interruption, checkpoint and restore |
| `rlm_code.agent.capabilities` | Repository and shell APIs with path, approval, audit, and sandbox enforcement |
| `rlm_code.agent.runtime` | Turns, cancellation, accounting, session lifecycle, and resume |
| `rlm_code.agent.terminal` | Minimal visible command-line event renderer |
| `rlm_code.execution.sandbox` | Backward-compatible execution in an explicitly bounded repository workdir |
| `rlm_code.main` | Narrow `agent` command dispatch while preserving the default Textual entry point |
| `tests.agent` | One-tool, persistence, effects, approval, resume, budget, and cancellation acceptance tests |
| `tests.test_sandbox_runtimes` | Regression coverage for explicit-workdir execution and wrapper cleanup |

## Effect path

```text
model -> python(code) -> kernel proxy -> capability broker
      -> approval gate + audit -> repository guard -> ExecutionSandbox/Superbox
      -> typed event + journal -> Python result -> model
```

The kernel has restricted builtins and no raw `open`, `exec`, `eval`, dynamic
imports, process, or network API. File access is rooted at the configured
repository, rejects traversal and `.git` mutation, applies output limits, and
executes through the configured RLM Code sandbox. Shell commands are argument
arrays, not shell strings, and use that same sandbox runtime in the repository
workdir. The local runtime is a development sandbox, while Docker or Apple
Container remains the isolation boundary when selected.

## Session and checkpoint contract

Each session owns an append-only `events.jsonl`, immutable creation metadata,
an approval audit JSONL, and a kernel snapshot under
`.rlm_code/agent/sessions/<session-id>/`.

After each completed Python cell, the kernel serializes user-created top-level
variables independently. Capability handles and runtime internals are always
recreated, and unpicklable or over-limit values are listed as skipped. Resume
replays model-visible messages from the event journal and restores exactly the
variables named by the snapshot manifest. It never claims that skipped values
were restored. Snapshot files are trusted local session artifacts and must not
be loaded from an untrusted source.

## Accounting and interruption

Model calls, input tokens, output tokens, cost, Python calls, effects, and
elapsed wall time are accumulated on the session result and emitted after each
turn. Cancellation sets a session signal, cancels an in-flight model request,
and sends an interrupt to the Python worker. A time or token budget produces an
interrupted result rather than a successful completion.

## Phase 1 verification

The acceptance test must prove an end-to-end task in which the model sees only
`python`, persists a variable between turns, searches and reads the repository,
writes a file, runs a test command through the selected sandbox, records approval
decisions and ordered events, and resumes the saved Python state. Separate
tests cover traversal rejection and active cancellation.

## Live recursive-agent extension

The next vertical slice adds an in-process supervisor beside the root runtime.
Root and child agents keep the same one-tool model contract: coordination is an
`rlm` object inside persistent Python, never a new model-facing tool. A bounded
supervisor schedules children concurrently and enforces global child capacity,
depth, per-child turn, token, and time budgets.

Every child has a stable ID, parent ID, model, sandbox, status, result, usage,
and assigned budgets. The root and every descendant append to the same ordered
session journal while keeping separate Python checkpoints. Spawn, messaging,
steering, follow-up, wait, cancellation, and deletion actions pass through the
same capability broker and approval audit as repository and shell effects.
Mailbox sends and their delivery acknowledgements are journaled so an
undelivered message can be restored with the session.

The terminal mirrors child events into the active root stream and renders
hierarchy activity as it happens. `rlm-code agent replay <session-id>` renders
the complete root-and-child trajectory in its original global event order.
Resuming the root restores its supported Python state and durable child
registry; a follow-up message can reopen a settled or interrupted child.

This extension is durable session recovery, not durable background execution.
Closing the process settles active in-process children. Daemon-owned workers,
detach/reattach while work continues, heartbeats, orphan recovery, fork, clone,
and automatic crash restart remain later supervision work.

## Deferred by design

Fork/clone, daemon supervision, background detach/reattach, heartbeats, orphan
recovery, skills, MCP, compaction, goals, schedules, continual harness
refinement, and a full terminal redesign are later phases. No competing runtime
adapter, TypeScript runtime, or additional model-facing tool is introduced.
