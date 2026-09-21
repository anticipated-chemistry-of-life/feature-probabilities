"""Shared TOML config loader for the feature-probabilities CLI executables.

Every CLI (``generate-groundtruth``, ``fit-kde``, ``annotate``) loads its
settings through :func:`load_config`: a checked-in TOML file with CLI flags
overriding individual keys. Precedence, highest to lowest:

1. CLI flag values passed as ``overrides`` (a ``None`` entry means "flag not set"
   and is skipped, leaving lower layers untouched).
2. Values present in the TOML config file.
3. Hardcoded defaults baked into this module (currently: ``"models/kde_model.pkl"``
   for ``kde_output_path``, and empty ``{}`` dicts for the
   ``[sirius.analysis_params]``/``[sirius.import_params]`` tables, when a
   config file omits them; ``db_path``, ``required_sirius_version``, and
   ``massspecgym_revision`` have no such default and are required).

A missing or malformed config file raises :class:`ConfigError` with an
actionable message instead of letting a raw ``tomllib`` traceback surface.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

type TOMLValue = (
    str | int | float | bool | list[TOMLValue] | dict[str, TOMLValue] | None
)
type TOMLTable = dict[str, TOMLValue]

DEFAULT_CONFIG_PATH = Path("config.toml")

_REQUIRED_KEYS = ("db_path", "required_sirius_version", "massspecgym_revision")

_DEFAULTS: TOMLTable = {
    "kde_output_path": "models/kde_model.pkl",
    "sirius": {
        "analysis_params": {},
        "import_params": {},
    },
}


class ConfigError(Exception):
    """Raised when the config file is missing, malformed, or incomplete."""


@dataclass(frozen=True, slots=True)
class SiriusConfig:
    """Nested SIRIUS parameter defaults from the ``[sirius.*]`` TOML sections."""

    analysis_params: TOMLTable = field(default_factory=dict)
    import_params: TOMLTable = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Config:
    """Fully merged configuration: hardcoded defaults < config file < CLI flags."""

    db_path: str
    required_sirius_version: str
    massspecgym_revision: str
    kde_output_path: str
    sirius: SiriusConfig


def _deep_merge(base: TOMLTable, overlay: TOMLTable) -> TOMLTable:
    """Recursively merge ``overlay`` over ``base``; ``None`` values are skipped."""
    merged = dict(base)
    for key, value in overlay.items():
        if value is None:
            continue
        base_value = merged.get(key)
        if isinstance(value, dict) and isinstance(base_value, dict):
            merged[key] = _deep_merge(base_value, value)
        else:
            merged[key] = value
    return merged


def load_config(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    overrides: TOMLTable | None = None,
) -> Config:
    """Load and merge the TOML config file with CLI flag ``overrides``.

    Raises:
        ConfigError: the file is missing, isn't valid TOML, or is missing a
            required key after merging.
    """
    path = Path(config_path)
    if not path.is_file():
        raise ConfigError(
            f"Config file not found: {path}. Pass a valid --config path or "
            f"create {DEFAULT_CONFIG_PATH} in the working directory."
        )

    try:
        file_data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Config file {path} is not valid TOML: {exc}") from exc

    merged = _deep_merge(_deep_merge(_DEFAULTS, file_data), overrides or {})

    missing = [key for key in _REQUIRED_KEYS if not merged.get(key)]
    if missing:
        raise ConfigError(
            f"Config file {path} is missing required key(s): {', '.join(missing)}"
        )

    sirius_data = merged.get("sirius")
    if not isinstance(sirius_data, dict):
        raise ConfigError(f"Config file {path}: [sirius] section must be a table")

    return Config(
        db_path=str(merged["db_path"]),
        required_sirius_version=str(merged["required_sirius_version"]),
        massspecgym_revision=str(merged["massspecgym_revision"]),
        kde_output_path=str(merged["kde_output_path"]),
        sirius=SiriusConfig(
            analysis_params=dict(sirius_data.get("analysis_params") or {}),
            import_params=dict(sirius_data.get("import_params") or {}),
        ),
    )
