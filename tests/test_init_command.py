"""Tests for project init command behavior."""

from rlm_code.commands.init_command import (
    _copy_example_config,
    _create_full_project,
    _create_minimal_project,
)
from rlm_code.core.config import ConfigManager, ProjectConfig


def test_copy_example_config_includes_rlm_benchmark_pack(tmp_path):
    _copy_example_config(tmp_path)

    assert (tmp_path / "dspy_config_example.yaml").exists()
    assert (tmp_path / ".env.example").exists()
    assert (tmp_path / "rlm_benchmarks.yaml").exists()


def test_create_minimal_project_wires_default_rlm_benchmark_pack(tmp_path):
    config = ProjectConfig.create_default("demo")
    manager = ConfigManager(tmp_path)

    _create_minimal_project(tmp_path, config, manager)

    loaded = ProjectConfig.load_from_file(tmp_path / ConfigManager.CONFIG_FILENAME)
    assert "rlm_benchmarks.yaml" in loaded.rlm.benchmark_pack_paths
    assert (tmp_path / "rlm_benchmarks.yaml").exists()


def test_create_full_project_wires_default_rlm_benchmark_pack(tmp_path):
    config = ProjectConfig.create_default("demo")
    manager = ConfigManager(tmp_path)

    _create_full_project(tmp_path, config, manager)

    loaded = ProjectConfig.load_from_file(tmp_path / ConfigManager.CONFIG_FILENAME)
    assert "rlm_benchmarks.yaml" in loaded.rlm.benchmark_pack_paths
    assert (tmp_path / "rlm_benchmarks.yaml").exists()
