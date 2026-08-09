"""Persistent and restricted Python-kernel tests."""

import asyncio

import pytest

from rlm_code.agent.kernel import PersistentPythonKernel


async def _effect_handler(
    capability: str,
    method: str,
    args: list[object],
    kwargs: dict[str, object],
) -> object:
    if (capability, method) == ("repo", "read"):
        return {"args": args, "kwargs": kwargs}
    raise AttributeError(f"unsupported effect: {capability}.{method}")


@pytest.mark.asyncio
async def test_kernel_persists_variables_routes_effects_and_restores(tmp_path):
    snapshot = tmp_path / "kernel-state.pkl"
    kernel = PersistentPythonKernel(snapshot_path=snapshot, effect_handler=_effect_handler)
    try:
        assigned = await kernel.execute("answer = 41")
        calculated = await kernel.execute("answer + 1")
        effect = await kernel.execute("repo.read('sample.txt', start_line=2)")
        blocked_import = await kernel.execute("import os")
        blocked_open = await kernel.execute("open('sample.txt')")
        blocked_private_attribute = await kernel.execute("repo._connection")
        blocked_introspection = await kernel.execute("getattr(repo, '_connection')")
        await kernel.execute("nested_handle = [repo]")
        checkpoint = await kernel.checkpoint()
    finally:
        await kernel.close()

    assert assigned.status == "ok"
    assert calculated.status == "ok"
    assert calculated.result == "42"
    assert "'sample.txt'" in (effect.result or "")
    assert "start_line" in (effect.result or "")
    assert blocked_import.status == "error"
    assert "Import statements are blocked" in (blocked_import.error or "")
    assert blocked_open.status == "error"
    assert "open" in (blocked_open.error or "")
    assert blocked_private_attribute.status == "error"
    assert "Private attribute access" in (blocked_private_attribute.error or "")
    assert blocked_introspection.status == "error"
    assert "Restricted name" in (blocked_introspection.error or "")
    assert "answer" in checkpoint.saved
    assert any(item["name"] == "nested_handle" for item in checkpoint.skipped)

    restored_kernel = PersistentPythonKernel(
        snapshot_path=snapshot,
        effect_handler=_effect_handler,
    )
    try:
        restore = await restored_kernel.start()
        restored = await restored_kernel.execute("answer")
    finally:
        await restored_kernel.close()

    assert "answer" in restore.restored
    assert "nested_handle" not in restore.restored
    assert restored.result == "41"


@pytest.mark.asyncio
async def test_kernel_interrupts_active_python_without_discarding_worker(tmp_path):
    kernel = PersistentPythonKernel(
        snapshot_path=tmp_path / "kernel-state.pkl",
        effect_handler=_effect_handler,
    )
    try:
        execution = asyncio.create_task(kernel.execute("while True:\n    pass", timeout=10))
        await asyncio.sleep(0.2)
        kernel.interrupt()
        result = await asyncio.wait_for(execution, timeout=5)
        follow_up = await kernel.execute("6 * 7")
    finally:
        await kernel.close()

    assert result.status == "interrupted"
    assert follow_up.status == "ok"
    assert follow_up.result == "42"
