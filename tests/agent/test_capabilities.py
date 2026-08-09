"""Boundary tests for model-accessible native-agent capabilities."""

from pathlib import Path

import pytest

from rlm_code.agent.capabilities import RepositoryCapability, ShellCapability
from rlm_code.execution.sandbox import ExecutionSandbox


def _local_sandbox() -> ExecutionSandbox:
    sandbox = ExecutionSandbox()
    sandbox.set_runtime("local")
    return sandbox


def test_repository_capability_executes_in_sandbox_and_blocks_traversal(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "source.py").write_text("answer = 41\n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    capability = RepositoryCapability(repository, _local_sandbox())

    assert capability.read("source.py") == "answer = 41"
    assert capability.search("answer") == [{"path": "source.py", "line": 1, "text": "answer = 41"}]
    assert capability.write("new.py", "answer = 42\n")["created"] is True
    assert (repository / "new.py").read_text(encoding="utf-8") == "answer = 42\n"

    with pytest.raises(PermissionError, match="escapes repository"):
        capability.read("../outside.txt")
    with pytest.raises(PermissionError, match="must be relative"):
        capability.read(str(outside))
    with pytest.raises(PermissionError, match=".git mutation"):
        capability.write(".git/config", "blocked")
    assert not list(repository.glob(".rlm_agent_*.py"))


def test_repository_capability_does_not_follow_file_symlinks(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret needle", encoding="utf-8")
    link = repository / "linked.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")

    capability = RepositoryCapability(repository, _local_sandbox())

    assert capability.list() == []
    assert capability.search("needle") == []
    with pytest.raises(PermissionError, match="escapes repository"):
        capability.read("linked.txt")


def test_shell_capability_requires_an_argument_array_and_bounded_cwd(tmp_path):
    capability = ShellCapability(Path(tmp_path), _local_sandbox())

    with pytest.raises(TypeError, match="argument list"):
        capability.run("echo unsafe")
    with pytest.raises(PermissionError, match="escapes repository"):
        capability.run(["echo", "unsafe"], cwd="..")
