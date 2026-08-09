# Harness Generalization in RLM Code

**By Shashi Jagtap, Superagentic AI**

The recent article [Language model harnesses are compositional generalizers](https://alexzhang13.github.io/blog/2026/harness/) by Alex Zhang and Omar Khattab argues that an agent harness can improve how language models generalize across task length and domain. The key is to keep each model call focused on a familiar local problem, even when the complete task is large or unfamiliar.

[RLM Code](https://github.com/SuperagenticAI/rlm-code) now makes most of the engineering concepts in that article available to run and inspect. Version 0.1.11 adds explicit support for locally in-distribution harness experiments, repository context selection, structural history, bounded observations, and trajectory comparison. It also includes an API-free demonstration of the same orchestration policy operating across two domains and an eight-times difference in task length.

This work does not reproduce the reinforcement learning results reported in the article. It provides a practical implementation of the harness mechanisms behind those experiments.

## The role of the harness

A conventional agent often places the task, source material, tool output, and conversation history into one growing model context. That approach is simple, but it couples the model to the surface details and length of each task. As the context grows, the model sees prompts that look increasingly different from the shorter prompts on which it may have been trained.

The harness article proposes a different design. Large task state is kept outside the root model. The root model decides how to decompose the work, while smaller model calls handle selected pieces of the task. Intermediate results remain in program state instead of being copied back into the root prompt.

This can make two different tasks look structurally similar to the root model. A short commerce task and a longer support task may use different language and data, yet both can be solved by discovering evidence units, classifying each unit, and aggregating the results. The harness exposes that common structure and keeps domain-specific content in focused subcalls.

## How the article's core mechanisms map to RLM Code

### Context offloading

The article's first mechanism is to keep input-specific context out of the root model's prompt. The [`PureRLMEnvironment`](https://github.com/SuperagenticAI/rlm-code/blob/main/rlm_code/rlm/pure_rlm_environment.py) implements this by storing source material in a Python REPL variable named `context`. The root model receives a description of the available state rather than the complete source material.

For repository tasks, the public [`RepositoryContextBuilder`](https://github.com/SuperagenticAI/rlm-code/blob/main/rlm_code/rlm/repository_context.py) selects the content placed in that variable. It supports small repository views, task-ranked evidence, larger bounded contexts, and explicit caller-provided paths. Context selection is deterministic, while the harness profile independently controls what the root model can observe.

### Programmatic subcalls and REPL state

The second mechanism is programmatic submodel calling. Model-generated Python can invoke `llm_query()` or `llm_query_batched()` from the REPL. A program can select one evidence unit, ask a focused question, and store the answer in a variable. Later code can aggregate those variables and finish through `FINAL()` or `FINAL_VAR()`.

This avoids the common agent pattern in which every tool response and submodel answer is appended to the root conversation. Domain-specific information remains available to the program without changing the root prompt after each subcall.

### Locally in-distribution observations and history

RLM Code provides three harness profiles. The `reference` profile preserves standard Pure RLM behavior. The `repo_evidence` profile returns bounded evidence previews and structural history for repository analysis. The `lid` profile uses opaque observations, decomposition guidance, and history offloading so that the root model sees stable execution signals instead of task-specific values.

Structural history retains the form of earlier actions while removing most semantic content. When the configured history budget is reached, older entries move into versioned REPL variables. Recent actions remain visible, and complete bounded traces remain available for debugging and replay.

The decomposition hint addresses a failure mode discussed in the article. A short task can sometimes be solved by sending the entire problem to one subcall, but that strategy may fail on longer inputs. The hint encourages focused subcalls without forcing a particular algorithm.

### Trajectory measurement and controlled evaluation

The runner records root and submodel call counts, prompt sizes, hashes, structural actions, and bounded subcall previews. The [`trajectory similarity API`](https://github.com/SuperagenticAI/rlm-code/blob/main/rlm_code/rlm/trajectory_similarity.py) compares root trajectories using normalized edit similarity, trigram containment, and trigram overlap measures.

Benchmark records can include explicit context, expected answers, task family, domain, split, and length. Together, these fields support controlled comparisons between short and long tasks or between domains that share a decomposition. Trajectory similarity is only a proxy, so it should be interpreted alongside correctness, reward, and call counts.

## Run the harness generalization demo

The [harness generalization demo](https://github.com/SuperagenticAI/rlm-code/tree/main/examples/harness_generalization) is deterministic and does not require an API key. It creates a four-unit commerce task and a 32-unit support task. Both use the same root policy. The policy discovers evidence, issues one focused subcall per unit, stores the answers in the REPL, aggregates them in code, and returns the correct result.

The demo checks that private context and domain labels do not enter the root prompts, the longer task is exactly eight times the size of the shorter task, work is decomposed into focused subcalls, old history is offloaded, and the structural root trajectories remain identical.

Run it from the repository root:

```bash
git clone https://github.com/SuperagenticAI/rlm-code.git
cd rlm-code
uv sync
uv run --frozen python examples/harness_generalization/demo.py
```

The fixed root policy isolates the behavior of the harness from model quality and provider availability. It proves that the context and trajectory controls work as designed. It does not claim that a model learned the policy.

## Use the profiles with a model

Install and start RLM Code:

```bash
uv tool install "rlm-code[tui,llm-all]"
rlm-code
```

Then run a task with the LID profile:

```text
/rlm run env=pure_rlm profile=lid context_profile=evidence steps=12 <your task>
```

Observation, history, and decomposition behavior can be overridden with `observe=raw|metadata|opaque`, `history=full|structural|offload`, and `decompose=on|off`. The [environment documentation](https://superagenticai.github.io/rlm-code/core/environments/) and [configuration reference](https://superagenticai.github.io/rlm-code/getting-started/configuration/) describe the complete settings.

We also demonstrated RLM Code on a large repository at the AI Engineer World's Fair in San Francisco. The [talk and presentation repository](https://github.com/Shashikant86/rlm-codebase-aie-wf26-talk) contains the slides and event material, while the [maintained live probe](https://github.com/SuperagenticAI/rlm-code/tree/main/examples/aie_world_fair_2026) runs against the current RLM Code APIs.

## Scope

RLM Code v0.1.11 implements the principal harness mechanisms discussed in the article, but it does not reproduce the article's RL training runs or reported performance lift. Those results require trained checkpoints and controlled evaluation. Start with the [API-free demo](https://github.com/SuperagenticAI/rlm-code/tree/main/examples/harness_generalization), then connect a model and use the `lid` or `repo_evidence` profile to study your own workloads.
