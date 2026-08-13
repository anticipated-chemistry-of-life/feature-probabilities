-- migrate:up
CREATE TABLE annotations (
    feature_id   INTEGER NOT NULL REFERENCES features(feature_id) ON DELETE CASCADE,
    molecule_id  INTEGER NOT NULL REFERENCES molecules(molecule_id),
    tool         TEXT NOT NULL,
    tool_version TEXT NOT NULL,
    rank         INTEGER,
    csi_score    REAL,
    confidence   REAL CHECK (confidence BETWEEN 0 AND 1),
    PRIMARY KEY (feature_id, molecule_id, tool, tool_version)
) STRICT;
CREATE INDEX idx_annotations_molecule ON annotations(molecule_id);

-- migrate:down
DROP TABLE annotations;
DROP INDEX IF EXISTS idx_annotations_molecule;