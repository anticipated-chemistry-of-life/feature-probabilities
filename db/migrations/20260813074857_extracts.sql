-- migrate:up
CREATE TABLE extracts (
    extract_id  INTEGER PRIMARY KEY,
    sample_code TEXT NOT NULL UNIQUE,
    species_id  INTEGER NOT NULL REFERENCES species(species_id),
    organ       TEXT,
    polarity    TEXT NOT NULL CHECK (polarity IN ('pos','neg'))
) STRICT;

-- migrate:down
DROP TABLE extracts;