"""Tests for `sirius.job_submission_from_config`.

The real `Sirius` wrapper needs a licensed SIRIUS (see
`test_sirius_qtof_integration.py`); this helper is pure and runs anywhere.
"""

from __future__ import annotations

from feature_probabilities.sirius import job_submission_from_config


def test_heuristic_thresholds_are_sent_as_config_map_strings() -> None:
    """SIRIUS 6.5.4 runs a `useHeuristic` object as 0/0 whatever it carries.

    The thresholds must therefore reach SIRIUS only as `configMap` strings,
    alongside any entries already configured, with no `useHeuristic` object
    left in `formulaIdParams` to compete with them.
    """
    submission = job_submission_from_config(
        {
            "formula_id_params": {
                "enabled": True,
                "use_heuristic": {
                    "use_heuristic_above_mz": 300,
                    "use_only_heuristic_above_mz": 650,
                },
            },
            "config_map": {"NumberOfStructureCandidates": "5"},
        },
        recompute=False,
    )

    wire = submission.to_dict()
    assert "useHeuristic" not in wire["formulaIdParams"]
    assert wire["formulaIdParams"]["enabled"] is True
    assert wire["configMap"] == {
        "NumberOfStructureCandidates": "5",
        "UseHeuristic.useHeuristicAboveMz": "300",
        "UseHeuristic.useOnlyHeuristicAboveMz": "650",
    }
