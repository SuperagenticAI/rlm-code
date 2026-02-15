# Contributing to RLM Code

Thank you for considering contributing to RLM Code! 🎉

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to team@super-agentic.ai.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the [issue tracker](https://github.com/SuperagenticAI/rlm-code/issues) to avoid duplicates. When creating a bug report, include:

* Clear and descriptive title
* Exact steps to reproduce the problem
* Expected vs actual behavior
* Screenshots (if applicable)
* Environment details (OS, Python version, RLM Code version)

### Suggesting Enhancements

Enhancement suggestions are tracked as [GitHub issues](https://github.com/SuperagenticAI/rlm-code/issues). Please include:

* Clear and descriptive title
* Step-by-step description of the enhancement
* Specific examples
* Why this enhancement would be useful

### Pull Requests

We welcome pull requests! For major changes, please open an issue first to discuss what you would like to change.

### AI-generated contributions

See [AI-GENERATED-CODE-POLICY.md](AI-GENERATED-CODE-POLICY.md) for our rules about contributions that use AI tools. If you used AI, please disclose the model/agent and how it was used in your PR description.

## Development Setup

### Prerequisites

* Python 3.11 or higher (we support Python 3.11, 3.12, and 3.13)
* [uv](https://github.com/astral-sh/uv) (recommended) or pip
* Git

### Supported Platforms

We test on:
* Linux (Ubuntu)
* macOS
* Windows (limited testing)

If you encounter platform-specific issues, please report them with your OS details.

### Quick Setup

1. **Fork and clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/rlm-code.git
cd rlm-code
```

2. **Create virtual environment and install dependencies**

Using uv (recommended):
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev,test,docs]"
```

Or using pip:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev,test,docs]"
```

**Note:** The `-e` flag installs the package in "editable" or "development" mode. This means changes to the source code are immediately reflected without reinstalling. The `rlm-code` command will use your local development version.

3. **Install pre-commit hooks**

```bash
pre-commit install
```

4. **Verify setup**

```bash
# Run tests
pytest

# Run linter
ruff check .

# Run formatter
ruff format .
```

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bugfix-name
```

### 2. Make Your Changes

Follow our coding standards (see below).

### 3. Add Tests

Write tests for new features and bug fixes.

```bash
pytest tests/ -v --cov=rlm_code
```

### 4. Format and Lint

```bash
ruff check --fix .
ruff format .
```

### 4a. Type Checking (Optional but Recommended)

We use [mypy](https://mypy.readthedocs.io/) for static type checking:

```bash
mypy rlm_code
```

Note: Some third-party libraries (dspy, mcp, etc.) may not have complete type stubs, so some `ignore_missing_imports` exceptions are configured in `pyproject.toml`.

### 5. Commit

Use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git commit -m "feat: add new feature"
git commit -m "fix: resolve bug"
git commit -m "docs: update documentation"
```

Commit types:
* `feat:` - New features
* `fix:` - Bug fixes
* `docs:` - Documentation
* `style:` - Formatting
* `refactor:` - Code restructuring
* `test:` - Tests
* `chore:` - Maintenance

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

**CI/CD Expectations:**
* All PRs must pass CI checks (linting, formatting, tests)
* Tests must pass on all supported Python versions (3.11, 3.12, 3.13)
* Code must pass Ruff linting and formatting checks
* Coverage should not decrease significantly
* Fix any CI failures before requesting review

### 7. Code Review Process

All pull requests require review before merging. Here's what reviewers look for:

* **Functionality**: Does the code work as intended?
* **Tests**: Are there adequate tests covering the changes?
* **Documentation**: Is the code documented appropriately?
* **Style**: Does the code follow our coding standards?
* **Breaking changes**: Are any breaking changes properly documented?
* **Performance**: Are there any obvious performance issues?

**Review expectations:**
* Be respectful and constructive in feedback
* Respond to review comments promptly
* Address all requested changes before requesting re-review
* Keep PRs focused and reasonably sized (<500 lines when possible)

## Coding Standards

### Python Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

* Line length: 100 characters
* Use type hints for function signatures
* Write docstrings for public functions, classes, and modules
* Follow PEP 8 conventions
* Use meaningful variable and function names

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """Brief description of what the function does.

    Longer description if needed.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of the return value.

    Raises:
        ValueError: When parameter is invalid.
    """
    pass
```

### Import Organization

Organize imports in three groups (Ruff does this automatically):

1. Standard library imports
2. Third-party imports
3. Local application imports

## Testing

### Writing Tests

* Write tests for all new features and bug fixes
* Use descriptive test names: `test_feature_behavior_under_condition`
* Use pytest fixtures for setup/teardown
* Aim for >80% test coverage

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rlm_code --cov-report=html

# Run specific test
pytest tests/test_specific_file.py

# Run in parallel
pytest -n auto
```

## Documentation

Documentation is available at [https://superagenticai.github.io/rlm-code/](https://superagenticai.github.io/rlm-code/).

When updating documentation:
* Update relevant docstrings for API changes
* Update examples if behavior changes
* Add new examples for new features
* Update the main documentation site (in `docs/`) for user-facing changes
* Follow the existing documentation style

### Changelog

We follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

When making changes:
* Add entries to `CHANGELOG.md` under `[Unreleased]`
* Use appropriate categories: `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`
* For breaking changes, clearly mark them and explain migration steps
* Link to related issues/PRs when applicable

## Project Structure

```
rlm-code/
├── rlm_code/          # Main package
│   ├── commands/       # CLI commands
│   ├── core/          # Core functionality
│   ├── models/        # LLM integration
│   ├── validation/    # Code validation
│   └── ...
├── tests/             # Test suite
├── docs/              # Documentation
└── examples/          # Example scripts
```

## Package Development

### Building the Package

To build the package locally:

```bash
# Install build dependencies
uv pip install build hatchling twine

# Build wheel and source distribution
python -m build

# Check the built package
twine check dist/*
```

The built packages will be in the `dist/` directory:
* `rlm_code-X.Y.Z-py3-none-any.whl` - Wheel distribution
* `rlm_code-X.Y.Z.tar.gz` - Source distribution

### Testing Package Installation

Test installing the built package:

```bash
# Install from wheel
uv pip install dist/rlm_code-*.whl

# Or install from source
uv pip install dist/rlm_code-*.tar.gz

# Verify installation
rlm-code --version
```

### Package Structure

The package uses modern Python packaging standards:
* **`pyproject.toml`** - PEP 517/518 compliant build configuration
* **`hatchling`** - Modern build backend (no setup.py needed)
* **`MANIFEST.in`** - Controls which files are included in source distributions (README, LICENSE, CHANGELOG, etc.)
* **Entry point** - `rlm-code` command defined in `[project.scripts]` mapping to `rlm_code.main:main`

When adding files that should be included in distributions:
* Update `MANIFEST.in` for source distributions
* Update `[tool.hatch.build.targets.sdist]` in `pyproject.toml` if needed
* Test with `python -m build` to verify files are included

### Package Metadata

Package metadata is defined in `pyproject.toml`:
* Version is managed in `[project]` section
* Dependencies are listed in `[project.dependencies]`
* Optional dependencies in `[project.optional-dependencies]`
* Entry points in `[project.scripts]`
* URLs (homepage, docs, etc.) in `[project.urls]`

When updating metadata:
* Update version in `pyproject.toml` (we use semantic versioning)
* Update `CHANGELOG.md` with release notes
* Ensure classifiers match the current Python version support

## Release Process

We follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html):
* **MAJOR** (x.0.0): Breaking changes
* **MINOR** (0.x.0): New features, backwards compatible
* **PATCH** (0.0.x): Bug fixes, backwards compatible

### Version Management

* Version is stored in `pyproject.toml` under `[project]` → `version`
* Update version before each release
* Tag releases in git: `git tag v0.1.2`
* Version should match the release in CHANGELOG.md

### Breaking Changes

Breaking changes require:
1. Opening an issue for discussion first
2. Clear migration guide in the PR
3. Deprecation warnings (if applicable) before removal
4. Documentation updates
5. Entry in CHANGELOG.md under "Changed" with "BREAKING CHANGE:" prefix

### Deprecation Policy

* Deprecated features will be marked with `@deprecated` in docstrings
* Deprecation warnings will be shown for at least one minor version
* Breaking changes will be announced in advance when possible

## Security

### Reporting Security Vulnerabilities

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to **team@super-agentic.ai** with:
* Description of the vulnerability
* Steps to reproduce
* Potential impact
* Suggested fix (if available)

We will acknowledge receipt within 48 hours and provide a timeline for addressing the issue.

For more details, see our [Security documentation](https://superagenticai.github.io/rlm-code/security/).

## Dependency Management

* We use `uv` (recommended) or `pip` for dependency management
* Dependencies are managed in `pyproject.toml`
* When adding new dependencies:
  * Justify why it's needed
  * Check for license compatibility (Apache-2.0-compatible preferred)
  * Consider the dependency's maintenance status
  * Update version constraints appropriately
* Security updates are handled via Dependabot

### Optional Dependencies

The package supports optional dependencies for different LLM providers:

* `rlm-code[openai]` - OpenAI SDK support
* `rlm-code[anthropic]` - Anthropic SDK support
* `rlm-code[gemini]` - Google Gemini SDK support
* `rlm-code[llm-all]` - All LLM providers
* `rlm-code[mcp-ws]` - WebSocket support for MCP servers

When adding new optional dependencies:
* Add them to `[project.optional-dependencies]` in `pyproject.toml`
* Document their purpose in the README
* Ensure they're truly optional (the package should work without them)
* Add appropriate error messages if features require them

## Performance Considerations

* Consider performance impact of new features
* Use appropriate data structures and algorithms
* Profile code if making performance-critical changes
* Document any known performance trade-offs

## Getting Help

* 🐛 [Issue Tracker](https://github.com/SuperagenticAI/rlm-code/issues) - Report bugs
* 📚 [Documentation](https://superagenticai.github.io/rlm-code/) - Full documentation
* 📧 Email: team@super-agentic.ai

## Maintainers

* [@Shashikant86](https://github.com/Shashikant86) - Lead Maintainer

## Contributor Recognition

All contributors are recognized in:
* Git commit history
* Release notes (for significant contributions)
* Project documentation (when applicable)

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.

---

**Thank you for contributing to RLM Code!** 🚀
