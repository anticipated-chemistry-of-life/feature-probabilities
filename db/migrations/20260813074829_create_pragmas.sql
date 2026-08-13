-- migrate:up transaction:false
PRAGMA journal_mode = WAL;

-- migrate:down transaction:false
PRAGMA journal_mode = DELETE;
