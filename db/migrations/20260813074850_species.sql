-- migrate:up
CREATE TABLE species (
    species_id INTEGER PRIMARY KEY,       -- rowid alias, auto-assigns
    taxon_name TEXT NOT NULL UNIQUE,
    ncbi_taxid INTEGER,
    family     TEXT
) STRICT;

-- migrate:down
DROP TABLE species;
