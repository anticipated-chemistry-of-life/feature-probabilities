-- Rank of the correct assignment for every feature of every ground truth
-- (MSnLib) run.
--
-- One row per ground truth feature. `correct_rank` is the SIRIUS
-- CSI:FingerID rank (1 = best) of the structure candidate whose first-block
-- InChIKey matches the feature's `true_inchikey` (GLOSSARY.md's Correct
-- assignment). A feature has at most one such candidate: annotations are
-- unique per (feature, molecule) and molecules are unique per first-block
-- InChIKey. `correct_rank` is NULL when the true structure is not among the
-- feature's retained candidates -- it ranked below Top-k (ADR 0001), or
-- SIRIUS returned no candidate at all (`n_candidates` = 0).
--
-- Usage:
--   duckdb -readonly -csv db/database.duckdb \
--     < queries/ground_truth_correct_rank.sql > ground_truth_correct_rank.csv

SELECT
    r.run_id,
    r.ionization_mode,
    r.instrument_type,
    f.feature_id,
    f.external_feature_id AS msnlib_id,
    f.true_inchikey,
    count(a.annotation_id) AS n_candidates,
    min(a.rank) FILTER (WHERE m.inchikey = f.true_inchikey) AS correct_rank
FROM sirius_runs AS r
JOIN features AS f ON f.run_id = r.run_id
LEFT JOIN annotations AS a ON a.feature_id = f.feature_id
LEFT JOIN molecules AS m ON m.molecule_id = a.molecule_id
WHERE r.source_kind = 'ground_truth_msnlib'
GROUP BY ALL
ORDER BY r.ionization_mode, r.run_id, f.feature_id;
