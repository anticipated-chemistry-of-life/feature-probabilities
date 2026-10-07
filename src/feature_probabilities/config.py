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
   config file omits them; ``db_path``, ``required_sirius_version``,
   ``massspecgym_revision``, and ``[sirius] top_k`` have no such default and
   are required).

The optional ``[smoke_test]`` section configures every CLI's ``--smoke-test``
mode (:class:`SmokeTestConfig`, :func:`for_smoke_test`); when present, every
one of its keys is required.

A missing or malformed config file raises :class:`ConfigError` with an
actionable message instead of letting a raw ``tomllib`` traceback surface.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

type TOMLValue = (
    str | int | float | bool | list[TOMLValue] | dict[str, TOMLValue] | None
)
type TOMLTable = dict[str, TOMLValue]

DEFAULT_CONFIG_PATH = Path("config.toml")

_REQUIRED_KEYS = ("db_path", "required_sirius_version", "massspecgym_revision")

_SMOKE_TEST_STRING_KEYS = (
    "db_path",
    "kde_output_path",
    "mzml_dir",
    "metadata_csv",
    "ionization_mode",
    "instrument_type",
    "export_path",
)

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
    """The ``[sirius]`` section: its ``top_k`` plus the nested ``[sirius.*]`` parameter tables.

    ``top_k`` is the Top-k (CONTEXT.md): how many best-ranked structure
    candidates per Feature a SIRIUS run retains. Part of every SIRIUS run's
    identity, so it is required with no default (ADR 0001).
    """

    top_k: int
    analysis_params: TOMLTable = field(default_factory=dict)
    import_params: TOMLTable = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SmokeTestConfig:
    """The ``[smoke_test]`` section: a small, isolated end-to-end pipeline run.

    ``db_path``/``kde_output_path`` replace the top-level ones (see
    :func:`for_smoke_test`); ``spectra_per_instrument_type`` caps
    ``generate-groundtruth``'s MassSpecGym input; the rest are ``annotate``'s
    defaults for its otherwise-required inputs and its export.
    """

    db_path: str
    kde_output_path: str
    spectra_per_instrument_type: int
    mzml_dir: str
    metadata_csv: str
    ionization_mode: str
    instrument_type: str
    export_path: str


@dataclass(frozen=True, slots=True)
class Config:
    """Fully merged configuration: hardcoded defaults < config file < CLI flags."""

    db_path: str
    required_sirius_version: str
    massspecgym_revision: str
    kde_output_path: str
    sirius: SiriusConfig
    smoke_test: SmokeTestConfig | None = None


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
            top_k=_top_k(sirius_data.get("top_k"), path),
            analysis_params=dict(sirius_data.get("analysis_params") or {}),
            import_params=dict(sirius_data.get("import_params") or {}),
        ),
        smoke_test=_smoke_test_config(merged.get("smoke_test"), path),
    )


def _top_k(value: TOMLValue, path: Path) -> int:
    """``[sirius] top_k``, which must be present and a positive integer."""
    if value is None:
        raise ConfigError(f"Config file {path} is missing required key: [sirius] top_k")
    return _positive_int(value, "[sirius] top_k", path)


def _positive_int(value: TOMLValue, key_label: str, path: Path) -> int:
    """`value` if it is a positive integer, else a `ConfigError` naming `key_label`."""
    # `bool` subclasses `int`: `key = true` must not read as 1.
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(
            f"Config file {path}: {key_label} must be a positive integer, got {value!r}"
        )
    return value


def _smoke_test_config(data: TOMLValue, path: Path) -> SmokeTestConfig | None:
    """Parse the optional ``[smoke_test]`` section; every key is required."""
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path}: [smoke_test] section must be a table")

    missing = [
        key
        for key in (*_SMOKE_TEST_STRING_KEYS, "spectra_per_instrument_type")
        if data.get(key) in (None, "")
    ]
    if missing:
        raise ConfigError(
            f"Config file {path}: [smoke_test] is missing required key(s): "
            f"{', '.join(missing)}"
        )

    spectra_per_instrument_type = _positive_int(
        data["spectra_per_instrument_type"],
        "[smoke_test] spectra_per_instrument_type",
        path,
    )

    return SmokeTestConfig(
        **{key: str(data[key]) for key in _SMOKE_TEST_STRING_KEYS},
        spectra_per_instrument_type=spectra_per_instrument_type,
    )


def for_smoke_test(config: Config) -> tuple[Config, SmokeTestConfig]:
    """`config` with its DB and KDE output paths swapped for ``[smoke_test]``'s.

    Returns the swapped config together with its (now guaranteed present)
    ``[smoke_test]`` section.

    Raises:
        ConfigError: the config has no ``[smoke_test]`` section, or its
            ``db_path`` resolves to the same file as the top-level one --
            ``generate-groundtruth --smoke-test`` deletes the smoke DB, so
            sharing it would delete the real one.
    """
    smoke_test = config.smoke_test
    if smoke_test is None:
        raise ConfigError(
            "--smoke-test needs a [smoke_test] section in the config file."
        )
    if Path(smoke_test.db_path).resolve() == Path(config.db_path).resolve():
        raise ConfigError(
            f"[smoke_test] db_path {smoke_test.db_path!r} is the same file as "
            f"db_path {config.db_path!r}; --smoke-test resets its database, so "
            "it must point elsewhere."
        )
    smoke_config = replace(
        config,
        db_path=smoke_test.db_path,
        kde_output_path=smoke_test.kde_output_path,
    )
    return smoke_config, smoke_test
