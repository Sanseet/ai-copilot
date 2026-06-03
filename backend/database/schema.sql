-- =============================================================================
-- AI Copilot — SQLite Schema
-- =============================================================================
-- Tables:
--   1. sessions          → chat sessions per user
--   2. messages          → individual chat messages (history)
--   3. candidates        → candidate profiles (SQL RAG demo data)
--   4. skills            → skills linked to candidates
--   5. projects          → projects linked to candidates
--   6. document_metadata → tracks uploaded documents in ChromaDB
--   7. event_logs        → monitoring / observability (written by logger.py)
-- =============================================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- -----------------------------------------------------------------------------
-- 1. Sessions
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT    PRIMARY KEY,          -- UUID
    user_id      TEXT    NOT NULL,
    title        TEXT    NOT NULL DEFAULT 'New Chat',
    created_at   TEXT    NOT NULL,
    updated_at   TEXT    NOT NULL,
    is_active    INTEGER NOT NULL DEFAULT 1    -- 0 = deleted/archived
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

-- -----------------------------------------------------------------------------
-- 2. Messages  (chat history)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT    NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_id      TEXT    NOT NULL,
    role         TEXT    NOT NULL CHECK(role IN ('user','assistant','system')),
    content      TEXT    NOT NULL,
    tool_used    TEXT,                         -- which tool was invoked, if any
    sources      TEXT,                         -- JSON array of citation sources
    created_at   TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_user    ON messages(user_id);

-- -----------------------------------------------------------------------------
-- 3. Candidates  (SQL RAG demo database)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS candidates (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL,
    email        TEXT    UNIQUE NOT NULL,
    phone        TEXT,
    location     TEXT,
    experience_years INTEGER NOT NULL DEFAULT 0,
    education    TEXT,
    summary      TEXT,
    created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- -----------------------------------------------------------------------------
-- 4. Skills
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS skills (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    skill_name   TEXT    NOT NULL,
    proficiency  TEXT    CHECK(proficiency IN ('Beginner','Intermediate','Advanced','Expert')),
    years        INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_skills_candidate  ON skills(candidate_id);
CREATE INDEX IF NOT EXISTS idx_skills_name       ON skills(skill_name);

-- -----------------------------------------------------------------------------
-- 5. Projects
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS projects (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    title        TEXT    NOT NULL,
    description  TEXT,
    tech_stack   TEXT,                         -- comma-separated technologies
    github_url   TEXT,
    created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_projects_candidate ON projects(candidate_id);

-- -----------------------------------------------------------------------------
-- 6. Document Metadata  (tracks files uploaded to ChromaDB)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_metadata (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT    NOT NULL,
    file_name       TEXT    NOT NULL,
    file_type       TEXT    NOT NULL,          -- pdf / docx / txt
    file_size_bytes INTEGER,
    chunk_count     INTEGER,
    collection_name TEXT    NOT NULL,
    uploaded_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_docs_user ON document_metadata(user_id);

-- -----------------------------------------------------------------------------
-- 7. Event Logs  (monitoring — also written by logger.py)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS event_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    event_type  TEXT    NOT NULL,             -- tool_call | rag_retrieval | chat | upload | error
    user_id     TEXT,
    session_id  TEXT,
    tool_used   TEXT,
    latency_ms  REAL,
    detail      TEXT,
    extra       TEXT                          -- JSON blob for arbitrary fields
);

CREATE INDEX IF NOT EXISTS idx_logs_event_type ON event_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_logs_user       ON event_logs(user_id);
