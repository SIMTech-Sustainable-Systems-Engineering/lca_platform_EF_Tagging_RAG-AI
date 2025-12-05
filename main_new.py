import os
import json
import asyncio
import traceback
from typing import List, Literal, Optional, Dict, Any
from urllib.parse import quote_plus

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text

from sentence_transformers import SentenceTransformer

EMBED_DIM = 384
_EMBED_MODEL: Optional[SentenceTransformer] = None


def get_embed_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer("intfloat/e5-small-v2")
    return _EMBED_MODEL


def none2str(x):
    return "" if x is None else str(x)


def normalize_text_field(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    lower = s.lower()
    if lower in {"undefined", "not defined", "n/a", "na", "none"}:
        return None
    return s


DB_USER = os.getenv("DB_USER")
DB_PWD = os.getenv("DB_PWD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME")

if not (DB_USER and DB_PWD and DB_NAME):
    raise RuntimeError("Please set DB_USER, DB_PWD, DB_NAME (and optionally DB_HOST).")

DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{quote_plus(DB_PWD)}@{DB_HOST}/{DB_NAME}"

engine = create_async_engine(DATABASE_URL, pool_pre_ping=True, echo=False)
SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)


class Filters(BaseModel):
    database_ids: Optional[List[str]] = None
    impact_method_ids: Optional[List[str]] = None


class RecommendRequest(BaseModel):
    query_text: Optional[str] = ""
    geography_id: Optional[str] = None
    ref_unit_id: Optional[str] = None

    query_lcia_description_id: Optional[str] = None
    query_lcia_name: Optional[str] = None
    query_upr_exchange_name: Optional[str] = None
    query_stage_name: Optional[str] = None
    query_process_name: Optional[str] = None
    query_cpc_name: Optional[str] = None

    filters: Optional[Filters] = None
    topk: int = 10


class LciaCard(BaseModel):
    type: Literal["EF"]
    lcia_description_id: str

    lcia_name: Optional[str] = None
    upr_exchange_name: Optional[str] = None
    lcia_system_model_name: Optional[str] = None
    geography_name: Optional[str] = None
    lcia_database_name: Optional[str] = None
    unit_name: Optional[str] = None
    stage_name: Optional[str] = None
    process_name: Optional[str] = None

    explain: Optional[str] = None


class RecommendResponse(BaseModel):
    items: List[dict]


_unit_conv_cache: Dict[tuple[str, str], bool] = {}
_geo_cache: Dict[tuple[str, str], float] = {}


async def has_unit_conversion(session: AsyncSession, from_unit_id: str, to_unit_id: str) -> bool:
    if not from_unit_id or not to_unit_id:
        return False
    key = (from_unit_id, to_unit_id)
    if key in _unit_conv_cache:
        return _unit_conv_cache[key]

    try:
        sql = text("""
            SELECT 1
            FROM lca.unit_conversion
            WHERE unit_id_from = :u_from AND unit_id_to = :u_to
            LIMIT 1;
        """)
        res = await session.execute(sql, {"u_from": from_unit_id, "u_to": to_unit_id})
        ok = res.scalar() is not None
        _unit_conv_cache[key] = ok
        return ok
    except Exception as e:
        return False


async def get_geography_distance(session: AsyncSession, geo_a: str, geo_b: str) -> float:
    if not geo_a or not geo_b:
        return 0.0

    key = (geo_a, geo_b)
    if key in _geo_cache:
        return _geo_cache[key]

    async def get_ancestry_path(geo_id: str) -> list[tuple[str, int]]:
        path = []
        current_id = geo_id
        visited = set()

        while current_id and current_id not in visited:
            visited.add(current_id)
            
            sql = text("""
                SELECT g.id, gt.level
                FROM lca.geography g
                LEFT JOIN lca.geography_type gt ON g.geography_type_id = gt.id
                WHERE g.id = :geo_id
            """)
            res = await session.execute(sql, {"geo_id": current_id})
            row = res.first()

            if not row:
                break

            level = row.level if row.level is not None else -1
            path.append((str(row.id), level))
            
            sql_parent = text("""
                SELECT parent_geography_id
                FROM lca.geography_parent
                WHERE geography_id = :geo_id
                LIMIT 1
            """)
            res_parent = await session.execute(sql_parent, {"geo_id": current_id})
            parent_row = res_parent.first()
            
            current_id = str(parent_row.parent_geography_id) if parent_row and parent_row.parent_geography_id else None

        return path

    try:
        path_a = await get_ancestry_path(geo_a)
        path_b = await get_ancestry_path(geo_b)

        if not path_a or not path_b:
            _geo_cache[key] = 0.0
            return 0.0

        ids_a = [p[0] for p in path_a]
        ids_b = [p[0] for p in path_b]

        if geo_a == geo_b:
            score = 1.00
            _geo_cache[key] = score
            return score

        is_a_global = path_a[0][1] == 0 if path_a[0][1] is not None else False
        is_b_global = path_b[0][1] == 0 if path_b[0][1] is not None else False

        if is_a_global or is_b_global:
            score = 0.10
            _geo_cache[key] = score
            return score

        if geo_a in ids_b:
            idx = ids_b.index(geo_a)
            if idx == 1:
                score = 0.75
            elif idx == 2:
                score = 0.55
            elif idx == 3:
                score = 0.40
            else:
                score = 0.25
            _geo_cache[key] = score
            return score

        if geo_b in ids_a:
            idx = ids_a.index(geo_b)
            if idx == 1:
                score = 0.75
            elif idx == 2:
                score = 0.55
            elif idx == 3:
                score = 0.40
            else:
                score = 0.25
            _geo_cache[key] = score
            return score

        common_ancestors = set(ids_a) & set(ids_b)

        if not common_ancestors:
            score = 0.00
            _geo_cache[key] = score
            return score

        nearest_common = None
        min_distance = float("inf")

        for ancestor in common_ancestors:
            dist_a = ids_a.index(ancestor)
            dist_b = ids_b.index(ancestor)
            total_dist = dist_a + dist_b

            if total_dist < min_distance:
                min_distance = total_dist
                nearest_common = ancestor

        if not nearest_common:
            score = 0.00
            _geo_cache[key] = score
            return score

        dist_a = ids_a.index(nearest_common)
        dist_b = ids_b.index(nearest_common)

        common_ancestor_level = None
        for p in path_a:
            if p[0] == nearest_common:
                common_ancestor_level = p[1]
                break
        if common_ancestor_level is None:
            for p in path_b:
                if p[0] == nearest_common:
                    common_ancestor_level = p[1]
                    break

        is_common_global = common_ancestor_level == 0 if common_ancestor_level is not None else False

        if dist_a == 1 and dist_b == 1:
            if is_common_global:
                score = 0.0
            else:
                score = 0.85
            _geo_cache[key] = score
            return score

        if min_distance == 2:
            score = 0.55
        elif min_distance == 3:
            score = 0.40
        elif min_distance >= 4:
            score = 0.25
        else:
            score = 0.25

        _geo_cache[key] = score
        return score

    except Exception as e:
        _geo_cache[key] = 0.0
        return 0.0


async def _fetch_cpc_name_for_query(
    session: AsyncSession,
    lcia_description_id: Optional[str],
) -> Optional[str]:
    if not lcia_description_id:
        return None

    try:
        sql = text("""
            SELECT cpc.name AS cpc_name
            FROM lca.lcia_description ld
            JOIN lca.lcia l
              ON l.id = ld.lcia_id
            JOIN lca.upr_exchange_name uen
              ON uen.id = ld.upr_exchange_name_id
             AND uen.lcia_database_id = l.lcia_database_id
            LEFT JOIN lca.classification_cpc2_1_all cpc
              ON cpc.id = uen.classification_cpc2_1_all_id
            WHERE ld.id = CAST(:desc_id AS uuid)
        """)
        res = await session.execute(sql, {"desc_id": lcia_description_id})
        row = res.first()
        if not row:
            return None
        return normalize_text_field(row.cpc_name)
    except Exception as e:
        return None


async def build_query_embedding_v2(
    session: AsyncSession,
    req: RecommendRequest,
) -> tuple[Optional[List[float]], Optional[List[float]], str]:
    db_cpc_name = await _fetch_cpc_name_for_query(
        session,
        req.query_lcia_description_id,
    )
    cpc_name = normalize_text_field(req.query_cpc_name) or db_cpc_name

    metadata_parts = []
    if normalize_text_field(req.query_lcia_name):
        metadata_parts.append(f"LCIA name: {normalize_text_field(req.query_lcia_name)}")
    if normalize_text_field(req.query_upr_exchange_name):
        metadata_parts.append(f"Reference product: {normalize_text_field(req.query_upr_exchange_name)}")
    if cpc_name:
        metadata_parts.append(f"CPC name: {cpc_name}")
    if normalize_text_field(req.query_stage_name):
        metadata_parts.append(f"Stage: {normalize_text_field(req.query_stage_name)}")
    if normalize_text_field(req.query_process_name):
        metadata_parts.append(f"Process: {normalize_text_field(req.query_process_name)}")

    metadata_vec = None
    if metadata_parts:
        metadata_text = " | ".join(metadata_parts)
        model = get_embed_model()
        formatted_metadata = f"query: {metadata_text}"
        metadata_vec = model.encode(formatted_metadata, normalize_embeddings=True).tolist()

    query_vec = None
    user_query = normalize_text_field(req.query_text or "")
    if user_query:
        model = get_embed_model()
        formatted_query = f"query: {user_query}"
        query_vec = model.encode(formatted_query, normalize_embeddings=True).tolist()

    all_parts = metadata_parts.copy()
    if user_query:
        all_parts.append(f"User query: {user_query}")
    if not all_parts:
        all_parts.append("LCIA dataset for life cycle impact assessment")
    
    semantic_text = " | ".join(all_parts)

    return metadata_vec, query_vec, semantic_text


async def hybrid_search_lcia_descriptions(
    session: AsyncSession,
    metadata_vec: Optional[List[float]],
    query_vec: Optional[List[float]],
    limit: int = 100,
    query_similarity_threshold: float = 0.0,
) -> List[dict]:
    if metadata_vec and not query_vec:
        print("  🔍 Using metadata-only search")
        return await _vector_search(session, metadata_vec, limit)
    
    if query_vec and not metadata_vec:
        print("  🔍 Using query-only search")
        return await _vector_search(session, query_vec, limit)
    
    print("  🔍 Using two-stage search: metadata → user query reranking")
    
    metadata_results = await _vector_search(session, metadata_vec, limit * 3)
    print(f"     • Stage 1 - Metadata results: {len(metadata_results)}")
    
    if not metadata_results:
        print("     ⚠️  No metadata results, falling back to query-only search")
        return await _vector_search(session, query_vec, limit)
    
    candidate_ids = [str(r["lcia_description_id"]) for r in metadata_results]
    
    query_results = await _vector_search_within_candidates(
        session, 
        query_vec, 
        candidate_ids, 
        limit * 2
    )
    print(f"     • Stage 2 - Query search results: {len(query_results)}")
    
    metadata_distance_map = {
        str(r["lcia_description_id"]): r["distance"] 
        for r in metadata_results
    }
    
    filtered_results = []
    for r in query_results:
        desc_id = str(r["lcia_description_id"])
        r["metadata_distance"] = metadata_distance_map.get(desc_id, 1.0)
        r["query_distance"] = r["distance"]
        
        metadata_sim = 1.0 - r["metadata_distance"]
        query_sim = 1.0 - r["query_distance"]
        
        metadata_weight = 0.4
        query_weight = 0.6
        
        r["combined_score"] = metadata_sim * metadata_weight + query_sim * query_weight
        
        if query_sim >= query_similarity_threshold:
            filtered_results.append(r)
        else:
            print(f"     ⚠️  Filtered out result (query_sim={query_sim:.4f}): {desc_id}")
    
    print(f"     • Stage 2 - After filtering: {len(filtered_results)} results")
    
    filtered_results.sort(key=lambda x: x["combined_score"], reverse=True)
    return filtered_results[:limit]


async def _vector_search_within_candidates(
    session: AsyncSession, 
    qvec: List[float], 
    candidate_ids: List[str],
    limit: int
) -> List[dict]:
    vec_literal = "[" + ",".join(str(x) for x in qvec) + "]"
    
    sql = text("""
        SELECT
            s.lcia_description_id,
            s.embedding_text,
            s.embedding <=> CAST(:qvec AS vector) AS distance
        FROM lca.rag_lcia_semantic_index s
        WHERE s.lcia_description_id = ANY(CAST(:candidate_ids AS uuid[]))
        ORDER BY distance ASC
        LIMIT :limit;
    """)
    
    res = await session.execute(sql, {
        "qvec": vec_literal, 
        "candidate_ids": candidate_ids,
        "limit": limit
    })
    return [dict(r) for r in res.mappings().all()]


async def _vector_search(session: AsyncSession, qvec: List[float], limit: int) -> List[dict]:
    vec_literal = "[" + ",".join(str(x) for x in qvec) + "]"
    
    sql = text("""
        SELECT
            s.lcia_description_id,
            s.embedding_text,
            s.embedding <=> CAST(:qvec AS vector) AS distance
        FROM lca.rag_lcia_semantic_index s
        ORDER BY distance ASC
        LIMIT :limit;
    """)
    
    res = await session.execute(sql, {"qvec": vec_literal, "limit": limit})
    return [dict(r) for r in res.mappings().all()]


async def fetch_lcia_metadata_for_descriptions(
    session: AsyncSession,
    lcia_rows: List[dict],
) -> List[dict]:
    if not lcia_rows:
        return []

    desc_ids = [str(r["lcia_description_id"]) for r in lcia_rows]
    
    distance_map: Dict[str, float] = {}
    fusion_score_map: Dict[str, float] = {}
    
    for r in lcia_rows:
        desc_id = str(r["lcia_description_id"])
        distance_map[desc_id] = float(r.get("distance", 0.5))
        if "fusion_score" in r:
            fusion_score_map[desc_id] = float(r["fusion_score"])

    sql = text("""
        WITH stage_agg AS (
            SELECT 
                s.lcia_description_id,
                array_agg(DISTINCT s.name ORDER BY s.name) AS stage_names
            FROM lca.upr_stage s
            GROUP BY s.lcia_description_id
        ),
        process_agg AS (
            SELECT 
                s.lcia_description_id,
                array_agg(DISTINCT p.name ORDER BY p.name) AS process_names
            FROM lca.upr_stage s
            JOIN lca.upr_process p ON p.upr_stage_id = s.id
            GROUP BY s.lcia_description_id
        )
        SELECT
            ld.id AS lcia_description_id,
            l.id AS lcia_id,
            ln.name AS lcia_name,
            db.name AS lcia_database_name,
            lsm.name AS lcia_system_model_name,
            g.name AS geography_name,
            l.geography_id,
            l.lcia_database_id,
            l.lcia_system_model_id,
            uen.id AS upr_exchange_name_id,
            uen.name AS upr_exchange_name,
            u.id AS unit_id,
            u.name AS unit_name,
            sa.stage_names,
            pa.process_names
        FROM lca.lcia_description ld
        JOIN lca.lcia l
            ON l.id = ld.lcia_id
        JOIN lca.lcia_database db
            ON db.id = l.lcia_database_id
        JOIN lca.lcia_name ln
            ON ln.id = l.lcia_name_id
        LEFT JOIN lca.lcia_system_model lsm
            ON lsm.id = l.lcia_system_model_id
        LEFT JOIN lca.geography g
            ON g.id = l.geography_id
        LEFT JOIN lca.upr_exchange_name uen
            ON uen.id = ld.upr_exchange_name_id
           AND uen.lcia_database_id = l.lcia_database_id
        LEFT JOIN lca.unit u
            ON u.id = uen.unit_id
        LEFT JOIN stage_agg sa
            ON sa.lcia_description_id = ld.id
        LEFT JOIN process_agg pa
            ON pa.lcia_description_id = ld.id
        WHERE ld.id = ANY(CAST(:desc_ids AS uuid[]))
    """)

    params = {"desc_ids": desc_ids}
    res = await session.execute(sql, params)
    rows = res.mappings().all()

    result: List[dict] = []
    for r in rows:
        row_dict = dict(r)
        did = str(row_dict["lcia_description_id"])
        
        row_dict["distance"] = distance_map.get(did, 0.5)
        if did in fusion_score_map:
            row_dict["fusion_score"] = fusion_score_map[did]
        
        if row_dict.get("stage_names"):
            row_dict["stage_name"] = ", ".join(row_dict["stage_names"])
        else:
            row_dict["stage_name"] = None
            
        if row_dict.get("process_names"):
            row_dict["process_name"] = ", ".join(row_dict["process_names"])
        else:
            row_dict["process_name"] = None
            
        result.append(row_dict)

    return result


async def batch_load_metadata(
    session: AsyncSession,
    rows: List[dict],
    ui: RecommendRequest
) -> Dict[str, Any]:
    all_unit_ids = set()
    all_geo_ids = set()

    for row in rows:
        if row.get("unit_id"):
            all_unit_ids.add(str(row["unit_id"]))
        if row.get("geography_id"):
            all_geo_ids.add(str(row["geography_id"]))

    if ui.ref_unit_id:
        all_unit_ids.add(str(ui.ref_unit_id))
    if ui.geography_id:
        all_geo_ids.add(str(ui.geography_id))
    
    metadata = {
        "unit_meta": {},
        "unit_conversions": {},
        "geo_scores": {},
    }

    if all_unit_ids:
        sql = text("""
            SELECT id, unit_type_id, unit_system_id
            FROM lca.unit
            WHERE id = ANY(CAST(:unit_ids AS uuid[]))
        """)
        res = await session.execute(sql, {"unit_ids": list(all_unit_ids)})
        unit_rows = res.all()
        for row in unit_rows:
            metadata["unit_meta"][str(row.id)] = {
                "type": str(row.unit_type_id) if row.unit_type_id else None,
                "system": str(row.unit_system_id) if row.unit_system_id else None,
            }

    if ui.ref_unit_id and all_unit_ids:
        sql = text("""
            SELECT unit_id_from, unit_id_to
            FROM lca.unit_conversion
            WHERE unit_id_from = ANY(CAST(:unit_ids AS uuid[]))
              AND unit_id_to = CAST(:target_unit AS uuid)
        """)
        res = await session.execute(sql, {
            "unit_ids": list(all_unit_ids),
            "target_unit": ui.ref_unit_id,
        })
        conv_rows = res.all()
        for row in conv_rows:
            key = (str(row.unit_id_from), str(row.unit_id_to))
            metadata["unit_conversions"][key] = True

    if ui.geography_id:
        for geo_id in all_geo_ids:
            if geo_id != ui.geography_id:
                score = await get_geography_distance(session, geo_id, ui.geography_id)
                metadata["geo_scores"][(geo_id, ui.geography_id)] = score
            else:
                metadata["geo_scores"][(geo_id, ui.geography_id)] = 1.0

    return metadata


def calc_score_lcia_fast(
    row: dict,
    ui: RecommendRequest,
    metadata: Dict[str, Any]
) -> Optional[tuple[float, dict]]:
    try:
        SEMANTIC_WEIGHT = 5.0
        
        if "combined_score" in row:
            semantic_score = row["combined_score"] * SEMANTIC_WEIGHT
            dist = row.get("query_distance", row.get("distance", 1.0))
        else:
            dist = float(row.get("distance", 1.0))
            semantic_score = (1.0 - dist) * SEMANTIC_WEIGHT
        
        score = semantic_score
        
        score_details = {
            "semantic": {
                "score": semantic_score,
                "weight": SEMANTIC_WEIGHT,
                "distance": dist,
                "description": "Semantic similarity"
            },
            "keyword_match": {
                "score": 0.0,
                "weight": 0.0,
                "match_type": "none",
                "description": "Keyword exact/fuzzy matching"
            },
            "unit": {
                "score": 0.0,
                "weight": 0.0,
                "match_type": "no_match",
                "description": "Unit matching"
            },
            "geography": {
                "score": 0.0,
                "weight": 0.0,
                "description": "Geography matching"
            },
            "total": semantic_score
        }
        
        if "metadata_distance" in row and "query_distance" in row:
            score_details["semantic"]["metadata_distance"] = row["metadata_distance"]
            score_details["semantic"]["query_distance"] = row["query_distance"]
            score_details["semantic"]["description"] = "Two-stage retrieval: metadata (40%) + query (60%)"
        
        if ui.query_text:
            query_keywords = set(normalize_text_field(ui.query_text).lower().split())
            
            lcia_name = row.get("lcia_name", "").lower()
            
            exact_match_bonus = 0.0
            fuzzy_match_penalty = 0.0
            
            for keyword in query_keywords:
                if keyword in lcia_name:
                    exact_match_bonus += 0.3
                    score_details["keyword_match"]["match_type"] = "exact"
                else:
                    conflicting_keywords = {
                        "alpine": ["tropical", "temperate"],
                        "tropical": ["alpine", "temperate"],
                        "temperate": ["alpine", "tropical"],
                    }
                    
                    if keyword in conflicting_keywords:
                        for conflict in conflicting_keywords[keyword]:
                            if conflict in lcia_name:
                                fuzzy_match_penalty -= 0.5
                                score_details["keyword_match"]["match_type"] = "conflict"
                                print(f"     ⚠️  Keyword conflict detected: query='{keyword}' vs result='{conflict}'")
            
            keyword_adjustment = exact_match_bonus + fuzzy_match_penalty
            if keyword_adjustment != 0:
                score += keyword_adjustment
                score_details["keyword_match"]["score"] = keyword_adjustment
                score_details["keyword_match"]["weight"] = abs(keyword_adjustment)

        row_unit_id = str(row.get("unit_id")) if row.get("unit_id") else None
        ui_unit_id = str(ui.ref_unit_id) if ui.ref_unit_id else None

        if row_unit_id and ui_unit_id:
            if row_unit_id == ui_unit_id:
                unit_boost = 1.0
                score += unit_boost
                score_details["unit"]["score"] = unit_boost
                score_details["unit"]["weight"] = 1.0
                score_details["unit"]["match_type"] = "exact_match"
                
            elif (row_unit_id, ui_unit_id) in metadata["unit_conversions"]:
                unit_boost = 0.7
                score += unit_boost
                score_details["unit"]["score"] = unit_boost
                score_details["unit"]["weight"] = 0.7
                score_details["unit"]["match_type"] = "convertible"
                
            else:
                row_meta = metadata["unit_meta"].get(row_unit_id, {})
                ui_meta = metadata["unit_meta"].get(ui_unit_id, {})
                partial_boost = 0.0

                if row_meta.get("type") and ui_meta.get("type"):
                    if row_meta["type"] == ui_meta["type"]:
                        partial_boost += 0.5
                        score_details["unit"]["match_type"] = "same_type"

                if row_meta.get("system") and ui_meta.get("system"):
                    if row_meta["system"] == ui_meta["system"]:
                        partial_boost += 0.3
                        if score_details["unit"]["match_type"] == "same_type":
                            score_details["unit"]["match_type"] = "same_type_and_system"
                        else:
                            score_details["unit"]["match_type"] = "same_system"
                
                if partial_boost > 0:
                    score += partial_boost
                    score_details["unit"]["score"] = partial_boost
                    score_details["unit"]["weight"] = partial_boost

        if ui.geography_id:
            row_geo = str(row.get("geography_id")) if row.get("geography_id") else None
            
            if row_geo:
                lookup_key = (row_geo, ui.geography_id)
                geo_score = metadata["geo_scores"].get(lookup_key, 0.2)
                
                if geo_score > 0:
                    score += geo_score
                    score_details["geography"]["score"] = geo_score
                    score_details["geography"]["weight"] = geo_score
                    
                    if geo_score >= 1.0:
                        score_details["geography"]["match_type"] = "exact_match"
                    elif geo_score >= 0.75:
                        score_details["geography"]["match_type"] = "parent_child_1_level"
                    elif geo_score >= 0.55:
                        score_details["geography"]["match_type"] = "parent_child_2_levels"
                    elif geo_score >= 0.40:
                        score_details["geography"]["match_type"] = "parent_child_3_levels"
                    elif geo_score > 0.2:
                        score_details["geography"]["match_type"] = "common_ancestor"
                    else:
                        score_details["geography"]["match_type"] = "distant_or_global"

        score_details["total"] = score
        return score, score_details
        
    except Exception as e:
        return None

    
def print_scoring_details(rank: int, row: dict, score: float, details: dict):
    print("\n" + "="*80)
    print(f"🏆 Rank #{rank} | Total Score: {score:.4f}")
    print("="*80)
    
    print(f"📋 LCIA Description ID: {row.get('lcia_description_id', 'N/A')}")
    print(f"📝 LCIA Name: {row.get('lcia_name', 'N/A')}")
    print(f"🏭 Reference Product: {row.get('upr_exchange_name', 'N/A')}")
    print(f"🌍 Geography: {row.get('geography_name', 'N/A')}")
    print(f"📦 Unit: {row.get('unit_name', 'N/A')}")
    print(f"🔄 Stage: {row.get('stage_name', 'N/A')}")
    print(f"⚙️  Process: {row.get('process_name', 'N/A')}")
    print(f"💾 Database: {row.get('lcia_database_name', 'N/A')}")
    
    print("\n📊 Scoring Breakdown:")
    print("-" * 80)
    
    sem = details.get("semantic", {})
    print(f"  🔤 Semantic Similarity:")
    print(f"     • Description: {sem.get('description', 'N/A')}")
    
    if "metadata_distance" in sem and "query_distance" in sem:
        print(f"     • Metadata Distance: {sem.get('metadata_distance', 0.0):.4f}")
        print(f"     • Query Distance: {sem.get('query_distance', 0.0):.4f}")
        print(f"     • Combined Score: {sem.get('score', 0.0):.4f} (40% metadata + 60% query) × {sem.get('weight', 1.0)}")
    else:
        print(f"     • Distance: {sem.get('distance', 0.0):.4f}")
        print(f"     • Score: {sem.get('score', 0.0):.4f} (weight: {sem.get('weight', 1.0):.2f})")
    
    keyword = details.get("keyword_match", {})
    if keyword.get("weight", 0.0) != 0:
        print(f"\n  🔑 Keyword Matching:")
        print(f"     • Match Type: {keyword.get('match_type', 'none')}")
        print(f"     • Score: {keyword.get('score', 0.0):.4f} (weight: {keyword.get('weight', 0.0):.2f})")
    
    unit = details.get("unit", {})
    print(f"\n  📏 Unit Matching:")
    print(f"     • Match Type: {unit.get('match_type', 'no_match')}")
    print(f"     • Score: {unit.get('score', 0.0):.4f} (weight: {unit.get('weight', 0.0):.2f})")
    
    geo = details.get("geography", {})
    if geo.get("weight", 0.0) > 0:
        print(f"\n  🌏 Geography Matching:")
        print(f"     • Match Type: {geo.get('match_type', 'N/A')}")
        print(f"     • Score: {geo.get('score', 0.0):.4f} (weight: {geo.get('weight', 0.0):.2f})")
    
    print("\n" + "-" * 80)
    print(f"  💯 Total Score: {details.get('total', 0.0):.4f}")
    print("=" * 80)


def to_lcia_card(row: dict) -> LciaCard:
    try:
        return LciaCard(
            type="EF",
            lcia_description_id=str(row["lcia_description_id"]),
            lcia_name=row.get("lcia_name"),
            upr_exchange_name=row.get("upr_exchange_name"),
            lcia_system_model_name=row.get("lcia_system_model_name"),
            geography_name=row.get("geography_name"),
            lcia_database_name=row.get("lcia_database_name"),
            unit_name=row.get("unit_name"),
            stage_name=row.get("stage_name"),
            process_name=row.get("process_name"),
            explain="Based on LCIA description semantic retrieval + weighted ranking of multi-dimensional rules such as units/regions",
        )
    except Exception as e:
        raise


app = FastAPI(title="LEAF RAG LCIA Recommendations", version="0.8.5")


@app.post("/lcia/recommendations", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    try:
        async with SessionLocal() as session:
            metadata_vec, query_vec, semantic_text = await build_query_embedding_v2(session, req)
            
            print(f"\n🔍 Query Analysis:")
            print(f"  📋 Metadata vector: {'✅ Present' if metadata_vec else '❌ None'}")
            print(f"  💬 User query vector: {'✅ Present' if query_vec else '❌ None'}")
            print(f"  📝 Semantic text: {semantic_text}\n")

            lcia_rows = await hybrid_search_lcia_descriptions(
                session,
                metadata_vec,
                query_vec,
                limit=max(50, req.topk * 5),
            )
            
            if not lcia_rows:
                return RecommendResponse(items=[])

            meta_rows = await fetch_lcia_metadata_for_descriptions(session, lcia_rows)
            if not meta_rows:
                return RecommendResponse(items=[])

            metadata = await batch_load_metadata(session, meta_rows, req)

        scored: List[tuple[float, dict, dict]] = []
        for r in meta_rows:
            result = calc_score_lcia_fast(r, req, metadata)
            if result is not None:
                score, details = result
                if "fusion_score" in r:
                    score += r["fusion_score"] * 0.5
                    details["fusion"] = {
                        "score": r["fusion_score"],
                        "weight": 0.5,
                        "description": "RRF fusion bonus"
                    }
                scored.append((score, r, details))

        scored.sort(key=lambda x: x[0], reverse=True)
        
        filtered_scored = scored
        if req.filters and req.filters.database_ids:
            db_ids = set(req.filters.database_ids)
            filtered_scored = [
                (s, r, d) for s, r, d in filtered_scored
                if r.get("lcia_database_id") and str(r["lcia_database_id"]) in db_ids
            ]

        print("\n" + "🎯" * 40)
        print(f"Top-{req.topk} Results with Detailed Scoring")
        print("🎯" * 40 + "\n")
        
        topk_results = filtered_scored[:req.topk]
        for rank, (score, row, details) in enumerate(topk_results, 1):
            print_scoring_details(rank, row, score, details)

        cards = [to_lcia_card(r) for (s, r, d) in topk_results]
        return RecommendResponse(items=[c.model_dump() for c in cards])

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "LEAF RAG LCIA API", "version": "0.8.5"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_new:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )