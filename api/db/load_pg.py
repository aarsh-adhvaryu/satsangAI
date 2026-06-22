"""Create the schema and load the counseling-core passages (+ embeddings) into Postgres.

    docker run -d --name satsang-pg -e POSTGRES_PASSWORD=satsang -e POSTGRES_DB=satsang \
        -p 5433:5432 pgvector/pgvector:pg16
    SATSANG_DATABASE_URL=postgresql://postgres:satsang@localhost:5433/satsang \
        python -m api.db.load_pg
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from pgvector.psycopg import register_vector

from .. import config

COLS = ["id", "source", "text_type", "tradition", "citation", "ref", "lang_original",
        "original", "transliteration", "translation", "contextual_explanation",
        "when_this_helps", "core_principle", "gujarati_explanation",
        "embedding_source_text", "verified"]


def main() -> None:
    df = pd.read_parquet(config.INDEX_PATH)
    schema = (Path(__file__).parent / "schema.sql").read_text()
    with psycopg.connect(config.DATABASE_URL, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")  # before register_vector
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(schema)
            cur.execute("TRUNCATE passages")
            cols = COLS + ["embedding"]
            with cur.copy(f"COPY passages ({', '.join(cols)}) FROM STDIN") as cp:
                for _, r in df.iterrows():
                    vals = [r[c] if not pd.isna(r[c]) else None for c in COLS]
                    emb = np.asarray(r["embedding"], dtype="float32")
                    cp.write_row(vals + ["[" + ",".join(map(str, emb.tolist())) + "]"])
            cur.execute("SELECT count(*) FROM passages")
            n = cur.fetchone()[0]
    print(f"loaded {n} passages into Postgres ({config.DATABASE_URL.split('@')[-1]})")


if __name__ == "__main__":
    main()
