"""Tests for PureRLMEnvironment security hardening."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from rlm_code.rlm.pure_rlm_environment import (
    SAFE_BUILTINS,
    PureRLMEnvironment,
    _check_code_safety,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path: Path) -> PureRLMEnvironment:
    """Create a PureRLMEnvironment with a temporary workdir."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        e = PureRLMEnvironment(workdir=tmp_path, allow_unsafe_exec=True)
    e.initialize_context("test context")
    return e


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    return tmp_path


# ---------------------------------------------------------------------------
# A1: __import__ removed from SAFE_BUILTINS
# ---------------------------------------------------------------------------


class TestImportBlocked:
    def test_import_not_in_safe_builtins(self):
        assert "__import__" not in SAFE_BUILTINS

    def test_import_os_blocked_by_scanner(self):
        result = _check_code_safety("x = __import__('os')")
        assert result is not None
        assert "__import__" in result

    def test_import_subprocess_blocked_by_scanner(self):
        result = _check_code_safety("import subprocess")
        assert result is not None
        assert "subprocess" in result

    def test_import_blocked_at_execution(self, env: PureRLMEnvironment):
        result = env._execute_code("x = __import__('os')")
        assert not result.success
        assert "Security check failed" in result.stderr

    def test_import_in_comment_allowed(self):
        """Comments containing blocked patterns should not be flagged."""
        result = _check_code_safety("# __import__('os') is dangerous")
        assert result is None

    def test_import_in_string_allowed(self):
        """String literals containing blocked patterns should not be flagged."""
        result = _check_code_safety("x = \"don't use __import__('os')\"")
        assert result is None


# ---------------------------------------------------------------------------
# A1: Safe stdlib modules pre-loaded
# ---------------------------------------------------------------------------


class TestSafeStdlib:
    def test_json_available(self, env: PureRLMEnvironment):
        result = env._execute_code('x = json.dumps({"a": 1})\nprint(x)')
        assert result.success
        assert '{"a": 1}' in result.stdout

    def test_math_available(self, env: PureRLMEnvironment):
        result = env._execute_code("x = math.sqrt(16)\nprint(x)")
        assert result.success
        assert "4.0" in result.stdout

    def test_re_available(self, env: PureRLMEnvironment):
        result = env._execute_code('m = re.match(r"(\\d+)", "42abc")\nprint(m.group(1))')
        assert result.success
        assert "42" in result.stdout

    def test_collections_available(self, env: PureRLMEnvironment):
        result = env._execute_code("c = collections.Counter([1,1,2,3])\nprint(c[1])")
        assert result.success
        assert "2" in result.stdout

    def test_datetime_available(self, env: PureRLMEnvironment):
        result = env._execute_code("d = datetime.date(2025, 1, 1)\nprint(d)")
        assert result.success
        assert "2025-01-01" in result.stdout

    def test_random_available(self, env: PureRLMEnvironment):
        result = env._execute_code("random.seed(42)\nx = random.randint(0, 100)\nprint(x)")
        assert result.success

    def test_copy_available(self, env: PureRLMEnvironment):
        result = env._execute_code(
            "x = [1, [2, 3]]\ny = copy.deepcopy(x)\ny[1].append(4)\nprint(len(x[1]), len(y[1]))"
        )
        assert result.success
        assert "2 3" in result.stdout


# ---------------------------------------------------------------------------
# A2: safe_open() restrictions
# ---------------------------------------------------------------------------


class TestSafeOpen:
    def test_read_within_workdir(self, env: PureRLMEnvironment, workdir: Path):
        """Reading a file inside workdir should succeed."""
        test_file = workdir / "test.txt"
        test_file.write_text("hello world")
        result = env._execute_code(f'f = open("{test_file}")\nprint(f.read())\nf.close()')
        assert result.success
        assert "hello world" in result.stdout

    def test_read_outside_workdir_blocked(self, env: PureRLMEnvironment):
        """Reading a file outside workdir should raise PermissionError."""
        result = env._execute_code('f = open("/etc/hosts")')
        assert not result.success
        assert (
            "outside the working directory" in result.stderr or "PermissionError" in result.stderr
        )

    def test_write_mode_blocked(self, env: PureRLMEnvironment, workdir: Path):
        """Write mode should be blocked even within workdir."""
        target = workdir / "output.txt"
        result = env._execute_code(f'f = open("{target}", "w")')
        assert not result.success
        assert "Write mode" in result.stderr or "PermissionError" in result.stderr

    def test_append_mode_blocked(self, env: PureRLMEnvironment, workdir: Path):
        """Append mode should also be blocked."""
        target = workdir / "output.txt"
        result = env._execute_code(f'f = open("{target}", "a")')
        assert not result.success

    def test_read_binary_within_workdir(self, env: PureRLMEnvironment, workdir: Path):
        """Binary read mode should work within workdir."""
        test_file = workdir / "data.bin"
        test_file.write_bytes(b"\x00\x01\x02")
        result = env._execute_code(
            f'f = open("{test_file}", "rb")\nprint(len(f.read()))\nf.close()'
        )
        assert result.success
        assert "3" in result.stdout


# ---------------------------------------------------------------------------
# A3: Pre-flight security scanner
# ---------------------------------------------------------------------------


class TestSecurityScanner:
    def test_eval_blocked(self):
        assert _check_code_safety("result = eval('1+1')") is not None

    def test_exec_blocked(self):
        assert _check_code_safety("exec('print(1)')") is not None

    def test_compile_blocked(self):
        assert _check_code_safety("c = compile('x=1', '<str>', 'exec')") is not None

    def test_globals_blocked(self):
        assert _check_code_safety("g = globals()") is not None

    def test_locals_blocked(self):
        assert _check_code_safety("l = locals()") is not None

    def test_os_system_blocked(self):
        assert _check_code_safety("os.system('ls')") is not None

    def test_os_popen_blocked(self):
        assert _check_code_safety("os.popen('ls')") is not None

    def test_subprocess_blocked(self):
        assert _check_code_safety("subprocess.run(['ls'])") is not None

    def test_subclasses_blocked(self):
        assert _check_code_safety("object.__subclasses__()") is not None

    def test_bases_blocked(self):
        assert _check_code_safety("str.__bases__") is not None

    def test_builtins_blocked(self):
        assert _check_code_safety("__builtins__['eval']('1')") is not None

    def test_safe_code_allowed(self):
        """Normal code should pass the scanner."""
        safe_code = """
x = [1, 2, 3]
y = sum(x)
result = json.dumps({"total": y})
print(result)
"""
        assert _check_code_safety(safe_code) is None

    def test_scanner_integration_with_execute(self, env: PureRLMEnvironment):
        """Blocked code should return a failed REPLResult, not raise."""
        result = env._execute_code("eval('1+1')")
        assert not result.success
        assert "Security check failed" in result.stderr


# ---------------------------------------------------------------------------
# A4: Deprecation warning
# ---------------------------------------------------------------------------


class TestDeprecationWarning:
    def test_constructor_rejects_unsafe_exec_by_default(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="allow_unsafe_exec=True"):
            PureRLMEnvironment(workdir=tmp_path)

    def test_warning_emitted_with_unsafe_exec_opt_in(self, tmp_path: Path):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            PureRLMEnvironment(workdir=tmp_path, allow_unsafe_exec=True)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) >= 1
            assert "exec()-based execution" in str(user_warnings[0].message)
            assert "MontyInterpreter" in str(user_warnings[0].message)

    def test_no_warning_with_interpreter(self, tmp_path: Path):
        """When an interpreter is provided, no warning should be emitted."""

        class FakeInterpreter:
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            PureRLMEnvironment(workdir=tmp_path, interpreter=FakeInterpreter())
            user_warnings = [
                x for x in w if issubclass(x.category, UserWarning) and "exec()" in str(x.message)
            ]
            assert len(user_warnings) == 0
