import os
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text
import asyncio
import urllib.parse

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

DB_USER = os.getenv("DB_USER")
DB_PWD = os.getenv("DB_PWD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME")

if DB_USER is None or DB_PWD is None or DB_NAME is None:
    raise ValueError("Please set DB_USER, DB_PWD, DB_NAME in environment variables")

encoded_pwd = urllib.parse.quote_plus(DB_PWD)
DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{encoded_pwd}@{DB_HOST}/{DB_NAME}"

engine = create_async_engine(DATABASE_URL)
SessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

MODEL_NAME = "intfloat/e5-small-v2"
print("🚀 Loading E5-small-v2 embedding model ...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)
print(f"✅ Model loaded on {device}")


def safe_join(items, sep="; "):
    if not items:
        return ""
    return sep.join([str(x) for x in items if x])


def normalize_meta_field(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.lower() in {"undefined", "not defined", "n/a", "na", "none"}:
        return None
    return s


def build_embedding_text(row) -> str:
    lcia_name = normalize_meta_field(row.get("lcia_name"))
    upr_name = normalize_meta_field(row.get("upr_exchange_name"))
    general_comment = normalize_meta_field(row.get("general_comment"))
    cpc_name = normalize_meta_field(row.get("cpc_name"))

    stage_names_raw = row.get("stage_names") or []
    process_names_raw = row.get("process_names") or []

    stage_names = [s for s in (stage_names_raw or []) if normalize_meta_field(s)]
    process_names = [s for s in (process_names_raw or []) if normalize_meta_field(s)]

    parts = []

    if lcia_name:
        parts.append(f"LCIA name: {lcia_name}")
    if upr_name:
        parts.append(f"Reference product: {upr_name}")
    if cpc_name:
        parts.append(f"CPC name: {cpc_name}")

    if stage_names:
        parts.append(f"Stage: {safe_join(stage_names)}")
    if process_names:
        parts.append(f"Process: {safe_join(process_names)}")
    if general_comment:
        parts.append(f"General comment: {general_comment}")

    if not parts:
        parts.append("LCIA description: (no metadata)")

    core_text = " | ".join(parts)
    return f"passage: {core_text.strip()}"


def build_name_text(row) -> str:
    """PRIMARY group embedding text: product identity (lcia_name + upr_exchange_name + cpc)."""
    lcia_name = normalize_meta_field(row.get("lcia_name"))
    upr_name = normalize_meta_field(row.get("upr_exchange_name"))
    cpc_name = normalize_meta_field(row.get("cpc_name"))

    parts = []
    if lcia_name:
        parts.append(f"LCIA name: {lcia_name}")
    if upr_name:
        parts.append(f"Reference product: {upr_name}")
    if cpc_name:
        parts.append(f"CPC name: {cpc_name}")
    if not parts:
        parts.append("LCIA product: (no name)")

    return f"passage: {' | '.join(parts).strip()}"


def build_context_text(row) -> str:
    """SECONDARY group embedding text: process context (stage + process + comment)."""
    stage_names = [s for s in (row.get("stage_names") or []) if normalize_meta_field(s)]
    process_names = [s for s in (row.get("process_names") or []) if normalize_meta_field(s)]
    general_comment = normalize_meta_field(row.get("general_comment"))

    parts = []
    if stage_names:
        parts.append(f"Stage: {safe_join(stage_names)}")
    if process_names:
        parts.append(f"Process: {safe_join(process_names)}")
    if general_comment:
        parts.append(f"General comment: {general_comment}")
    if not parts:
        parts.append("LCIA context: (no stage/process)")

    return f"passage: {' | '.join(parts).strip()}"


def vec_str(emb) -> str:
    """Format a numpy embedding row as a pgvector literal."""
    return "[" + ",".join(f"{x:.6f}" for x in emb.tolist()) + "]"


def embed_texts(texts):
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=64,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


async def build_lcia_semantic_index(fetch_size: int = 5000, batch_size: int = 500):
    async with SessionLocal() as session:
        count_sql = text("SELECT COUNT(*) FROM lca.lcia_description")
        count_result = await session.execute(count_sql)
        total_count = count_result.scalar()
        print(f"📊 Total lcia_description rows to process: {total_count}")

        offset = 0
        processed = 0
        pbar = tqdm(total=total_count, desc="Building LCIA semantic index")

        while True:
            select_sql = text("""
                SELECT *
                FROM (
                    SELECT
                        ld.id AS lcia_description_id,
                        ln.name AS lcia_name,
                        uen.name AS upr_exchange_name,
                        cpc.name AS cpc_name,
                        ld.general_comment AS general_comment,
                        array_remove(array_agg(DISTINCT us.name), NULL) AS stage_names,
                        array_remove(array_agg(DISTINCT up.name), NULL) AS process_names
                    FROM lca.lcia_description ld
                    JOIN lca.lcia l
                        ON l.id = ld.lcia_id
                    LEFT JOIN lca.lcia_name ln
                        ON ln.id = l.lcia_name_id
                    LEFT JOIN lca.upr_stage us
                        ON us.lcia_description_id = ld.id
                    LEFT JOIN lca.upr_process up
                        ON up.upr_stage_id = us.id
                    LEFT JOIN lca.upr_exchange_name uen
                        ON uen.id = ld.upr_exchange_name_id
                    LEFT JOIN lca.classification_cpc2_1_all cpc
                        ON cpc.id = uen.classification_cpc2_1_all_id
                    GROUP BY
                        ld.id,
                        ln.name,
                        uen.name,
                        cpc.name,
                        ld.general_comment
                    ORDER BY ld.id
                ) sub
                OFFSET :offset
                LIMIT :limit
            """)

            result = await session.execute(select_sql, {"offset": offset, "limit": fetch_size})
            rows = result.mappings().all()

            if not rows:
                break

            texts_combined = []
            texts_name = []
            texts_context = []
            rows_data = []

            for r in rows:
                row_dict = dict(r)
                combined_text = build_embedding_text(row_dict)
                texts_combined.append(combined_text)
                texts_name.append(build_name_text(row_dict))
                texts_context.append(build_context_text(row_dict))
                rows_data.append({
                    "lcia_description_id": str(row_dict["lcia_description_id"]),
                    "embedding_text": combined_text,
                })

            emb_combined = embed_texts(texts_combined)
            emb_name = embed_texts(texts_name)
            emb_context = embed_texts(texts_context)

            insert_sql = text("""
                INSERT INTO lca.rag_lcia_semantic_index
                    (lcia_description_id, embedding_text, embedding, name_embedding, context_embedding)
                VALUES (:lcia_description_id, :embedding_text, :embedding, :name_embedding, :context_embedding)
                ON CONFLICT (lcia_description_id)
                DO UPDATE SET
                    embedding_text    = EXCLUDED.embedding_text,
                    embedding         = EXCLUDED.embedding,
                    name_embedding    = EXCLUDED.name_embedding,
                    context_embedding = EXCLUDED.context_embedding
            """)

            for i in range(0, len(rows_data), batch_size):
                batch_rows = rows_data[i:i + batch_size]
                comb_slice = emb_combined[i:i + batch_size]
                name_slice = emb_name[i:i + batch_size]
                ctx_slice = emb_context[i:i + batch_size]

                for data, ce, ne, xe in zip(batch_rows, comb_slice, name_slice, ctx_slice):
                    data_with_emb = {
                        **data,
                        "embedding": vec_str(ce),
                        "name_embedding": vec_str(ne),
                        "context_embedding": vec_str(xe),
                    }
                    await session.execute(insert_sql, data_with_emb)

                await session.commit()
                processed += len(batch_rows)
                pbar.update(len(batch_rows))

            offset += fetch_size

        pbar.close()
        print(f"✅ Completed: {processed} LCIA descriptions processed.")

        await refresh_activity_index(session)


async def refresh_activity_index(session):
    """
    Rebuild the deduplicated activity-level index (one row per distinct lcia_name)
    from the per-description embeddings just written. This is what the recommend
    recall searches: one row per activity removes the vector duplication that
    otherwise collapses results to a single card. Derived from existing
    embeddings — no extra encoding.
    """
    print("🔄 Refreshing lca.rag_lcia_activity_index (dedup by lcia_name) ...")
    await session.execute(text("""
        CREATE TABLE IF NOT EXISTS lca.rag_lcia_activity_index (
            lcia_name         text PRIMARY KEY,
            name_embedding    vector(384) NOT NULL,
            context_embedding vector(384) NOT NULL
        )
    """))
    await session.execute(text("TRUNCATE lca.rag_lcia_activity_index"))
    await session.execute(text("""
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
        ORDER BY ln.name, r.lcia_description_id
    """))
    await session.commit()
    count = (await session.execute(text("SELECT COUNT(*) FROM lca.rag_lcia_activity_index"))).scalar()
    print(f"✅ Activity index refreshed: {count} distinct activities")


async def main():
    await build_lcia_semantic_index(fetch_size=5000, batch_size=500)

if __name__ == "__main__":
    asyncio.run(main())
