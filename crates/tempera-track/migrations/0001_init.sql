CREATE TABLE jobs (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    spec_yaml     TEXT NOT NULL,
    dataset_id    TEXT,
    split_id      TEXT,
    task          TEXT,
    metric        TEXT,
    status        TEXT NOT NULL,          -- created | auditing | blocked | running | succeeded | failed
    winner_run_id TEXT,
    test_json     TEXT,                   -- Evaluation of the winner on the test set
    report_path   TEXT,
    error         TEXT,
    created_at    TEXT NOT NULL,
    finished_at   TEXT
);

CREATE TABLE runs (
    id                 TEXT PRIMARY KEY,
    job_id             TEXT NOT NULL REFERENCES jobs(id),
    algorithm          TEXT NOT NULL,
    backend            TEXT NOT NULL,     -- native | python | builtin
    is_baseline        INTEGER NOT NULL DEFAULT 0,
    params_json        TEXT NOT NULL DEFAULT '{}',
    status             TEXT NOT NULL,     -- running | succeeded | failed | skipped
    valid_score        REAL,
    valid_metrics_json TEXT,
    model_dir          TEXT,
    error              TEXT,
    started_at         TEXT NOT NULL,
    finished_at        TEXT,
    duration_s         REAL
);
CREATE INDEX runs_job ON runs(job_id);

CREATE TABLE metrics (
    run_id  TEXT NOT NULL REFERENCES runs(id),
    name    TEXT NOT NULL,
    step    INTEGER,
    value   REAL NOT NULL,
    unix_ms INTEGER NOT NULL
);
CREATE INDEX metrics_run ON metrics(run_id, name);

CREATE TABLE artifacts (
    run_id TEXT NOT NULL REFERENCES runs(id),
    kind   TEXT NOT NULL,
    format TEXT NOT NULL,
    path   TEXT NOT NULL
);

-- The test set of a job is read exactly once (docs/wiki/Data-Audit.md).
CREATE TABLE test_access (
    job_id  TEXT PRIMARY KEY REFERENCES jobs(id),
    run_id  TEXT NOT NULL REFERENCES runs(id),
    at      TEXT NOT NULL
);
