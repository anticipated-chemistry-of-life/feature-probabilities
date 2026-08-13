-- migrate:up
CREATE TABLE molecules (
    molecule_id       INTEGER PRIMARY KEY,
    inchikey          TEXT NOT NULL UNIQUE CHECK (length(inchikey) = 27),
    smiles            TEXT,
    molecular_formula TEXT,
    exact_mass        REAL
) STRICT;

-- migrate:down
DROP TABLE molecules;
