# file: build_lcia_semantic_index.py
import os
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text
import asyncio
import urllib.parse

# ============================
# 0. 数据库连接配置
# ============================
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

# ============================
# 1. 加载 E5-small-v2 模型
# ============================
MODEL_NAME = "intfloat/e5-small-v2"
print("🚀 Loading E5-small-v2 embedding model ...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)
print(f"✅ Model loaded on {device}")

# ============================
# 2. 辅助函数
# ============================

def safe_join(items, sep="; "):
    """把 list 安全拼接成字符串"""
    if not items:
        return ""
    return sep.join([str(x) for x in items if x])


def normalize_meta_field(value):
    """
    清洗 metadata 字段：
    - None / 空串 / 空格 → None
    - 'undefined' / 'not defined' / 'n/a' 等 → None
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.lower() in {"undefined", "not defined", "n/a", "na", "none"}:
        return None
    return s


def build_embedding_text(row) -> str:
    """
    按新模板构造 embedding_text：
    passage: LCIA name: ...
           | Reference product: ...
           | CPC name: ...
           | Stage: ...
           | Process: ...
           | General comment: ...
    """
    lcia_name = normalize_meta_field(row.get("lcia_name"))
    upr_name = normalize_meta_field(row.get("upr_exchange_name"))
    general_comment = normalize_meta_field(row.get("general_comment"))
    cpc_name = normalize_meta_field(row.get("cpc_name"))

    stage_names_raw = row.get("stage_names") or []
    process_names_raw = row.get("process_names") or []

    # 过滤掉 'undefined' 等垃圾值
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


def embed_texts(texts):
    """生成 L2 归一化 embedding（float32）"""
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=64,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


# ============================
# 3. 主逻辑：构建 rag_lcia_semantic_index
# ============================

async def build_lcia_semantic_index(fetch_size: int = 5000, batch_size: int = 500):
    """
    从 lcia_description 聚合相关信息，生成 embedding_text + embedding，
    写入 lca.rag_lcia_semantic_index。
    """

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
                break  # 没有更多行了

            texts_to_embed = []
            rows_data = []

            for r in rows:
                row_dict = dict(r)
                embedding_text = build_embedding_text(row_dict)
                texts_to_embed.append(embedding_text)
                rows_data.append({
                    "lcia_description_id": str(row_dict["lcia_description_id"]),
                    "embedding_text": embedding_text,
                })

            embeddings = embed_texts(texts_to_embed)

            insert_sql = text("""
                INSERT INTO lca.rag_lcia_semantic_index
                    (lcia_description_id, embedding_text, embedding)
                VALUES (:lcia_description_id, :embedding_text, :embedding)
                ON CONFLICT (lcia_description_id)
                DO UPDATE SET
                    embedding_text = EXCLUDED.embedding_text,
                    embedding      = EXCLUDED.embedding
            """)

            for i in range(0, len(rows_data), batch_size):
                batch_rows = rows_data[i:i + batch_size]
                emb_slice = embeddings[i:i + batch_size]

                for data, emb in zip(batch_rows, emb_slice):
                    vec_str = "[" + ",".join(f"{x:.6f}" for x in emb.tolist()) + "]"
                    data_with_emb = {
                        **data,
                        "embedding": vec_str,
                    }
                    await session.execute(insert_sql, data_with_emb)

                await session.commit()
                processed += len(batch_rows)
                pbar.update(len(batch_rows))

            offset += fetch_size

        pbar.close()
        print(f"✅ Completed: {processed} LCIA descriptions processed.")


# ============================
# 4. 脚本入口
# ============================

async def main():
    await build_lcia_semantic_index(fetch_size=5000, batch_size=500)

if __name__ == "__main__":
    asyncio.run(main())
