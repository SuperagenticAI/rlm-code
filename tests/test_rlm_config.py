"""Tests for RLM-related project config sections."""

from rlm__code.core.config import ConfigManager, ProjectConfig


def test_project_config_loads_rlm_reward_section(tmp_path):
    config_path = tmp_path / "dspy_config.yaml"
    config_path.write_text(
        """
name: demo
rlm:
  default_benchmark_preset: generic_smoke
  benchmark_pack_paths:
    - rlm_benchmarks.yaml
  reward:
    global_scale: 0.75
    run_python_success_bonus: 0.9
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = ProjectConfig.load_from_file(config_path)
    assert config.rlm.default_benchmark_preset == "generic_smoke"
    assert config.rlm.benchmark_pack_paths == ["rlm_benchmarks.yaml"]
    assert config.rlm.reward.global_scale == 0.75
    assert config.rlm.reward.run_python_success_bonus == 0.9


def test_project_config_saves_rlm_defaults(tmp_path):
    config = ProjectConfig.create_default("demo")
    config_path = tmp_path / "dspy_config.yaml"

    config.save_to_file(config_path)
    loaded = ProjectConfig.load_from_file(config_path)
    assert loaded.rlm.default_benchmark_preset == "dspy_quick"
    assert loaded.rlm.benchmark_pack_paths == []
    assert loaded.rlm.reward.global_scale == 1.0


def test_project_config_normalizes_single_benchmark_pack_path(tmp_path):
    config_path = tmp_path / "dspy_config.yaml"
    config_path.write_text(
        """
name: demo
rlm:
  benchmark_pack_paths: rlm_benchmarks.yaml
""".strip()
        + "\n",
        encoding="utf-8",
    )
    loaded = ProjectConfig.load_from_file(config_path)
    assert loaded.rlm.benchmark_pack_paths == ["rlm_benchmarks.yaml"]


def test_config_manager_uses_legacy_path_when_primary_missing(tmp_path):
    legacy_path = tmp_path / ConfigManager.LEGACY_CONFIG_FILENAME
    legacy_path.write_text("name: legacy-demo\n", encoding="utf-8")

    manager = ConfigManager(tmp_path)
    assert manager.config_path == legacy_path
    assert manager.is_project_initialized()
    assert manager.config.name == "legacy-demo"


def test_config_manager_prefers_primary_path_over_legacy(tmp_path):
    legacy_path = tmp_path / ConfigManager.LEGACY_CONFIG_FILENAME
    primary_path = tmp_path / ConfigManager.CONFIG_FILENAME
    legacy_path.write_text("name: legacy-demo\n", encoding="utf-8")
    primary_path.write_text("name: primary-demo\n", encoding="utf-8")

    manager = ConfigManager(tmp_path)
    assert manager.config_path == primary_path
    assert manager.config.name == "primary-demo"
