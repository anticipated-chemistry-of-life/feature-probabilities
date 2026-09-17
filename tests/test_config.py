"""Tests for the shared TOML config loader."""

from pathlib import Path

import pytest

from feature_probabilities.config import Config, ConfigError, load_config

SAMPLE_TOML = """
db_path = "db/database.duckdb"
required_sirius_version = "6.0.0"
massspecgym_revision = "abc123def456"

[sirius.analysis_params]
profile = "orbitrap"
ppm_max = 10.0

[sirius.import_params]
allow_ms1_only = false
"""


def write_config(tmp_path: Path, text: str = SAMPLE_TOML) -> Path:
    config_path = tmp_path / "config.toml"
    config_path.write_text(text)
    return config_path


def test_loading_sample_config_exposes_top_level_and_nested_sirius_sections(
    tmp_path: Path,
) -> None:
    config_path = write_config(tmp_path)

    config = load_config(config_path)

    assert isinstance(config, Config)
    assert config.db_path == "db/database.duckdb"
    assert config.required_sirius_version == "6.0.0"
    assert config.massspecgym_revision == "abc123def456"
    assert config.sirius.analysis_params == {"profile": "orbitrap", "ppm_max": 10.0}
    assert config.sirius.import_params == {"allow_ms1_only": False}


def test_override_wins_over_file_value_but_unset_override_leaves_file_value(
    tmp_path: Path,
) -> None:
    config_path = write_config(tmp_path)

    config = load_config(
        config_path,
        overrides={
            "db_path": "/override/path.duckdb",
            "required_sirius_version": None,
        },
    )

    assert config.db_path == "/override/path.duckdb"
    assert config.required_sirius_version == "6.0.0"
    assert config.massspecgym_revision == "abc123def456"


def test_override_can_replace_a_nested_sirius_param_key(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)

    config = load_config(
        config_path,
        overrides={"sirius": {"analysis_params": {"profile": "qtof"}}},
    )

    assert config.sirius.analysis_params == {"profile": "qtof", "ppm_max": 10.0}
    assert config.sirius.import_params == {"allow_ms1_only": False}


def test_missing_config_file_raises_config_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "does-not-exist.toml"

    with pytest.raises(ConfigError, match="not found"):
        load_config(missing_path)


def test_malformed_toml_raises_config_error(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, text="db_path = [unterminated")

    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(config_path)


def test_missing_required_key_raises_config_error(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, text='db_path = "db/database.duckdb"\n')

    with pytest.raises(ConfigError, match="required_sirius_version"):
        load_config(config_path)


def test_checked_in_repo_config_loads_successfully() -> None:
    repo_config_path = Path(__file__).parent.parent / "config.toml"

    config = load_config(repo_config_path)

    assert isinstance(config, Config)
    assert config.db_path
