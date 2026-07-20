# Reusable prompts from the conference demo

These prompts are designed for `env=pure_rlm profile=lid`.

## Architecture map

```text
Analyze this repository. Identify the main architectural components, entry points, and dependency flow. Cite the key files a new maintainer should read first. Use FINAL when done.
```

## New engineer onboarding

```text
Act like a staff engineer onboarding a new teammate. Explore the repository and produce a reading path: first five files, what each teaches, and what questions to ask next. Use file evidence.
```

## Refactor planning

```text
Explore this repository and identify one high-leverage refactor opportunity. Explain the affected modules, why the refactor matters, risks, and a staged migration plan with tests.
```

## Test strategy

```text
Analyze the test structure and source modules. Identify important behavior that appears under-tested, cite files, and propose five focused tests to add first.
```

## Security and authentication audit

```text
Search for authentication, authorization, secret handling, and network boundary code. Summarize the security-sensitive paths and list possible risks or missing tests with file evidence.
```

## Incident investigation

```text
Find the likely code paths involved in startup, configuration, or runtime errors. Map the execution path from entry point to error handling and cite files.
```
