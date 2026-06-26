-- auto-generated definition (updated for two-group weighted matching, Method A)
CREATE TABLE rag_lcia_semantic_index
(
    lcia_description_id uuid        NOT NULL
        PRIMARY KEY
        REFERENCES lcia_description,
    embedding_text      TEXT        NOT NULL,
    embedding           vector(384) NOT NULL,   -- legacy combined vector (all fields) — fallback path
    name_embedding      vector(384),            -- PRIMARY group: lcia_name + upr_exchange_name (+ cpc)
    context_embedding   vector(384)             -- SECONDARY group: stage + process (+ general comment)
);

ALTER TABLE rag_lcia_semantic_index
    OWNER TO postgres;

CREATE INDEX idx_rag_lcia_semantic_index_fts
    ON rag_lcia_semantic_index USING gin (TO_TSVECTOR('simple'::regconfig, embedding_text));

CREATE INDEX idx_rag_lcia_semantic_index_embedding
    ON rag_lcia_semantic_index USING hnsw (embedding public.vector_cosine_ops);

CREATE INDEX idx_rag_lcia_semantic_index_name_embedding
    ON rag_lcia_semantic_index USING hnsw (name_embedding public.vector_cosine_ops);

CREATE INDEX idx_rag_lcia_semantic_index_context_embedding
    ON rag_lcia_semantic_index USING hnsw (context_embedding public.vector_cosine_ops);


-- ============================================================================
-- MIGRATION for an existing, already-populated table (schema: lca)
-- Additive + nullable, so it is safe to run on a live table. After running,
-- (re)populate the two new columns with build_lcia_semantic_index.py.
-- Run these (uncommented) on the target DB BEFORE re-embedding:
-- ============================================================================
-- ALTER TABLE lca.rag_lcia_semantic_index ADD COLUMN IF NOT EXISTS name_embedding    vector(384);
-- ALTER TABLE lca.rag_lcia_semantic_index ADD COLUMN IF NOT EXISTS context_embedding vector(384);
--
-- CREATE INDEX IF NOT EXISTS idx_rag_lcia_semantic_index_name_embedding
--     ON lca.rag_lcia_semantic_index USING hnsw (name_embedding public.vector_cosine_ops);
-- CREATE INDEX IF NOT EXISTS idx_rag_lcia_semantic_index_context_embedding
--     ON lca.rag_lcia_semantic_index USING hnsw (context_embedding public.vector_cosine_ops);


-- ============================================================================
-- Deduplicated ACTIVITY-LEVEL index — one row per distinct lcia_name.
-- The recommend recall searches THIS table: with no per-geo/db vector
-- duplication, an exact seq-scan KNN (no HNSW index needed at ~10k rows) gives
-- correct AND diverse results, instead of collapsing to a single activity.
-- Built from the existing embeddings — no re-embedding required.
-- (build_lcia_semantic_index.py::refresh_activity_index recreates this on rebuild.)
-- ============================================================================
CREATE TABLE lca.rag_lcia_activity_index (
    lcia_name         text PRIMARY KEY,
    name_embedding    vector(384) NOT NULL,
    context_embedding vector(384) NOT NULL
);

INSERT INTO lca.rag_lcia_activity_index (lcia_name, name_embedding, context_embedding)
SELECT DISTINCT ON (ln.name)
       ln.name, r.name_embedding, r.context_embedding
FROM lca.rag_lcia_semantic_index r
JOIN lca.lcia_description ld ON ld.id = r.lcia_description_id
JOIN lca.lcia l             ON l.id = ld.lcia_id
JOIN lca.lcia_name ln       ON ln.id = l.lcia_name_id
WHERE r.name_embedding IS NOT NULL
  AND r.context_embedding IS NOT NULL
  AND ln.name IS NOT NULL
ORDER BY ln.name, r.lcia_description_id;

-- Speeds up the variant fetch (ln.name = ANY(...)); skip if it already exists.
CREATE INDEX IF NOT EXISTS idx_lcia_name_name ON lca.lcia_name (name);
