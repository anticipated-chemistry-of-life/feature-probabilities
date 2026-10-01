-- Every molecule assigned to every feature of every field mzML run, per
-- species and extract, with its calibration score.
--
-- One row per annotation (feature-molecule pair). `inchikey_2d` is the
-- first-block (skeleton-level) InChIKey stored in `molecules`.
-- `calibration_score` comes from the most recently fitted KDE model; it is
-- NULL for an annotation that model never scored. Features without any
-- assigned molecule produce no row. Ground-truth (MassSpecGym) runs have no
-- extract and are excluded.
--
-- Usage:
--   duckdb -readonly -csv db/smoke_test/database.duckdb \
--     < queries/feature_molecule_scores.sql > feature_molecule_scores.csv

WITH latest_kde_model AS (
    SELECT kde_model_id
    FROM kde_models
    ORDER BY fitted_at DESC, kde_model_id DESC
    LIMIT 1
)
SELECT
    s.taxon_scientific_name AS species,
    e.extract_id,
    f.feature_id,
    m.inchikey AS inchikey_2d,
    cs.score AS calibration_score
FROM species AS s
JOIN extracts AS e ON e.species_id = s.species_id
JOIN sirius_runs AS r ON r.extract_id = e.extract_id
JOIN features AS f ON f.run_id = r.run_id
JOIN annotations AS a ON a.feature_id = f.feature_id
JOIN molecules AS m ON m.molecule_id = a.molecule_id
LEFT JOIN calibration_scores AS cs
    ON cs.annotation_id = a.annotation_id
    AND cs.kde_model_id = (SELECT kde_model_id FROM latest_kde_model)
WHERE r.source_kind = 'field_mzml'
ORDER BY species, e.extract_id, r.run_id, f.feature_id, a.rank;
