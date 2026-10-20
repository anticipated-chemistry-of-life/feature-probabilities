"""Raw instrument value -> Instrument type -> SIRIUS instrument profile.

The one place an instrument is mapped to a SIRIUS profile (GLOSSARY.md's
Instrument type). `resolve_instrument_type` turns a raw value (e.g. a ground
truth `instrument_type` or an instrument model such as
`LC-MS QExactive Plus`) into an Instrument type;
`analysis_params_for_instrument_type` builds the SIRIUS analysis parameters
for one. `generate-groundtruth` builds every chunk's parameters this way;
`annotate` does not yet, so field runs still get SIRIUS's default profile.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from PySirius.models.instrument_profile import InstrumentProfile

from feature_probabilities.config import ConfigError
from feature_probabilities.sirius import job_submission_from_config

if TYPE_CHECKING:
    from PySirius import JobSubmission

    from feature_probabilities.config import TOMLTable

#: The canonical Instrument types, as `generate-groundtruth` stores them on
#: `sirius_runs.instrument_type`.
type InstrumentType = Literal["Orbitrap", "QTOF"]

#: Normalized raw instrument value (`_normalize`) -> Instrument type, matched
#: exactly. Adding an instrument is a one-line, reviewed change here; there
#: is deliberately no fallback, since a guessed profile would silently skew
#: that run's results.
_INSTRUMENT_TYPE_BY_NORMALIZED_VALUE: dict[str, InstrumentType] = {
    "orbitrap": "Orbitrap",
    "lcmsqexactiveplus": "Orbitrap",  # LC-MS QExactive Plus
    "lcmsqexactivehfx": "Orbitrap",  # LC-MS QExactive HFX
    "lcmsorbitrapidx": "Orbitrap",  # LCMS Orbitrap ID-X
    "lcmssynaptxs": "QTOF",  # LCMS – Synapt XS
    "qtof": "QTOF",
}

#: One-to-one: SIRIUS has an instrument profile for exactly these two
#: Instrument types.
_PROFILE_BY_INSTRUMENT_TYPE: dict[InstrumentType, InstrumentProfile] = {
    "Orbitrap": InstrumentProfile.ORBITRAP,
    "QTOF": InstrumentProfile.QTOF,
}


class UnknownInstrumentError(Exception):
    """Raised when a raw instrument value matches no Instrument type."""

    def __init__(self, raw_value: str) -> None:
        super().__init__(
            f"Unknown instrument {raw_value!r}: no Instrument type is known for "
            f"its normalized value {_normalize(raw_value)!r}. If it is an "
            "Orbitrap or QTOF instrument, add that value to "
            "feature_probabilities.instrument._INSTRUMENT_TYPE_BY_NORMALIZED_VALUE."
        )
        self.raw_value = raw_value


def _normalize(raw_value: str) -> str:
    """`raw_value` lowercased, keeping only ASCII letters and digits.

    Mirrors `mascot-rs`'s `Instrument::from_metadata_value`, so spacing,
    case, and punctuation variants (`LCMS – Synapt XS`, `LCMS-Synapt XS`)
    normalize alike. Filtering before lowercasing keeps non-ASCII characters
    whose lowercase form is ASCII (e.g. the Kelvin sign) out of the key.
    """
    return "".join(
        char for char in raw_value if char.isascii() and char.isalnum()
    ).lower()


def resolve_instrument_type(raw_value: str) -> InstrumentType:
    """The Instrument type `raw_value` names.

    Raises:
        UnknownInstrumentError: `raw_value`'s normalized value has no
            Instrument type.
    """
    instrument_type = _INSTRUMENT_TYPE_BY_NORMALIZED_VALUE.get(_normalize(raw_value))
    if instrument_type is None:
        raise UnknownInstrumentError(raw_value)
    return instrument_type


def analysis_params_for_instrument_type(
    analysis_params: TOMLTable, instrument_type: InstrumentType, *, recompute: bool
) -> JobSubmission:
    """`analysis_params` with `instrument_type`'s SIRIUS profile injected.

    `config.toml` deliberately carries no `formula_id_params.profile`: one
    global value would analyse Orbitrap spectra under a QTOF mass-accuracy
    model (or vice versa). Injecting it per Instrument type also makes the
    two types' `analysis_params_checksum`s differ -- correct, since those
    runs are not interchangeable. `analysis_params` itself is not mutated.

    Raises:
        ConfigError: `analysis_params.formula_id_params` is not a table.
    """
    formula_id_params = analysis_params.get("formula_id_params") or {}
    if not isinstance(formula_id_params, dict):
        raise ConfigError(
            "[sirius.analysis_params] formula_id_params must be a table, got "
            f"{formula_id_params!r}"
        )
    profile = _PROFILE_BY_INSTRUMENT_TYPE[instrument_type]
    return job_submission_from_config(
        {
            **analysis_params,
            "formula_id_params": {**formula_id_params, "profile": profile.value},
        },
        recompute=recompute,
    )
