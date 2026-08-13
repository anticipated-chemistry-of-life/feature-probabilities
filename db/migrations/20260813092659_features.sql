-- migrate:up
CREATE TABLE features (
    feature_id   INTEGER PRIMARY KEY,
    extract_id   INTEGER NOT NULL REFERENCES extracts(extract_id),
    feature_code TEXT NOT NULL,
    precursor_mz REAL CHECK (precursor_mz > 0),
    UNIQUE (extract_id, feature_code)
) STRICT;


-- migrate:down
DROP TABLE features;
DROP INDEX IF EXISTS idx_features_extract;