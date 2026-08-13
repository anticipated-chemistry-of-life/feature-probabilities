CREATE TABLE IF NOT EXISTS "schema_migrations" (version varchar(128) primary key);
CREATE TABLE species (
    species_id INTEGER PRIMARY KEY,       -- rowid alias, auto-assigns
    taxon_name TEXT NOT NULL UNIQUE,
    ncbi_taxid INTEGER,
    family     TEXT
) STRICT;
CREATE TABLE extracts (
    extract_id  INTEGER PRIMARY KEY,
    sample_code TEXT NOT NULL UNIQUE,
    species_id  INTEGER NOT NULL REFERENCES species(species_id),
    organ       TEXT,
    polarity    TEXT NOT NULL CHECK (polarity IN ('pos','neg'))
) STRICT;
CREATE TABLE features (
    feature_id   INTEGER PRIMARY KEY,
    extract_id   INTEGER NOT NULL REFERENCES extracts(extract_id),
    feature_code TEXT NOT NULL,
    rt_seconds   REAL CHECK (rt_seconds >= 0),
    precursor_mz REAL CHECK (precursor_mz > 0),
    UNIQUE (extract_id, feature_code)
) STRICT;
CREATE INDEX idx_features_extract ON features(extract_id);
CREATE TABLE molecules (
    molecule_id       INTEGER PRIMARY KEY,
    inchikey          TEXT NOT NULL UNIQUE CHECK (length(inchikey) = 27),
    smiles            TEXT,
    molecular_formula TEXT,
    exact_mass        REAL
) STRICT;
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
-- Dbmate schema migrations
INSERT INTO "schema_migrations" (version) VALUES
  ('20260813074829'),
  ('20260813074850'),
  ('20260813074857'),
  ('20260813092659'),
  ('20260813092922'),
  ('20260813093115');
