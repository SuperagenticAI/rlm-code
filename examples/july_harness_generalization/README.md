# July 2026 LID harness generalization demo

This is an API-key-free, executable proof of the harness ideas in Alex Zhang's
[July 2026 post](https://alexzhang13.github.io/blog/2026/harness/), implemented
using RLM Code's production `PureRLMEnvironment` rather than a standalone mock
harness.

Run it from the repository root:

```bash
uv run python examples/july_harness_generalization/demo.py
```

The demo generates two temporary, cross-domain task families:

- a four-unit “training-scale” commerce task;
- a 32-unit support task, giving an 8× length extrapolation.

Both use the same root policy. The policy discovers evidence units, sends one
focused programmatic subcall per unit, stores all semantic results in REPL
variables, aggregates without exposing those values to the root, and terminates
through `FINAL_VAR`. The report fails (non-zero exit) unless it proves all of
the following:

1. both cross-domain answers are correct;
2. the root prompts contain neither private context markers nor domain answer
   labels;
3. the bounded debug trace still contains the short submodel answers;
4. work is decomposed rather than delegated as one monolithic subcall;
5. the structural root trajectories are identical;
6. old structural history is offloaded into REPL history variables; and
7. the evaluation task is exactly 8× longer than the training-scale task.

The root policy is fixed in this offline proof so the result tests the harness,
not the quality or availability of a model provider. To exercise the same
policies with a connected model in the CLI, use:

```text
/rlm run env=pure_rlm profile=lid context_profile=evidence steps=12 <your task>
```

Useful overrides are `observe=raw|metadata|opaque`,
`history=full|structural|offload`, and `decompose=on|off`. The `lid` profile
defaults to `opaque`, `offload`, and the decomposition hint.

## What this demonstrates—and what it does not

It demonstrates the engineering mechanism behind locally in-distribution root
calls and gives reproducible trajectory-similarity measurements. It does not
claim that a particular model has learned the policy. For that claim, train and
evaluate models on disjoint families and length buckets, then compare their
persisted root trajectories and correctness using the same metrics.
