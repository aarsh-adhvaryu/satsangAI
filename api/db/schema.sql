-- SatsangAI V1 Postgres schema (pgvector). Mirrors the in-memory stores:
-- passages (retrieval + citation lookup), conversations (short-term), user_facts (long-term).
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS passages (
    id                      text PRIMARY KEY,
    source                  text,
    text_type               text,
    tradition               text,
    citation                text,
    ref                     text,
    lang_original           text,
    original                text,
    transliteration         text,
    translation             text,
    contextual_explanation  text,
    when_this_helps         text,
    core_principle          text,
    gujarati_explanation    text,
    embedding_source_text   text,
    verified                boolean,
    embedding               vector(1024)        -- BGE-M3, unit-norm
);
CREATE INDEX IF NOT EXISTS passages_emb_hnsw ON passages USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS passages_citation ON passages (lower(citation));
CREATE INDEX IF NOT EXISTS passages_tradition ON passages (tradition);

-- short-term conversation history (keeps everything)
CREATE TABLE IF NOT EXISTS conversations (
    id              bigserial PRIMARY KEY,
    conversation_id text NOT NULL,
    role            text NOT NULL,
    text            text NOT NULL,
    created_at      timestamptz DEFAULT now()
);
CREATE INDEX IF NOT EXISTS conversations_cid ON conversations (conversation_id, id);

-- long-term per-user facts (sensitive facts are gated out before insert)
CREATE TABLE IF NOT EXISTS user_facts (
    user_id    text NOT NULL,
    fact       text NOT NULL,
    created_at timestamptz DEFAULT now(),
    PRIMARY KEY (user_id, fact)
);
