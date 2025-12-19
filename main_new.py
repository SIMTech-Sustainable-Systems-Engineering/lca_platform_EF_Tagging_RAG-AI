import os
import json
import asyncio
import traceback
from typing import List, Literal, Optional, Dict, Any
from urllib.parse import quote_plus
import re

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text

from sentence_transformers import SentenceTransformer
from groq import Groq

EMBED_DIM = 384
_EMBED_MODEL: Optional[SentenceTransformer] = None
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

ENABLE_LLM_INTENT = True
ENABLE_LLM_EXPLANATION = True
LLM_TIMEOUT = 30.0

_GROQ_CLIENT = None

def get_groq_client():
    global _GROQ_CLIENT
    if _GROQ_CLIENT is None and GROQ_API_KEY:
        _GROQ_CLIENT = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq API configured")
    return _GROQ_CLIENT

def get_embed_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        print("🔧 Loading embedding model: intfloat/e5-small-v2")
        _EMBED_MODEL = SentenceTransformer("intfloat/e5-small-v2")
        print("✅ Embedding model loaded")
    return _EMBED_MODEL


def llm_generate(prompt: str, max_new_tokens: int = 256) -> str:
    try:
        client = get_groq_client()
        
        if not client:
            raise Exception("Groq client not initialized - please set GROQ_API_KEY")
        
        print(f"   🌐 Calling Groq API...")
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs only JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=max_new_tokens,
        )
        
        result = response.choices[0].message.content
        
        if not result:
            raise Exception("Empty response from Groq API")
        
        print(f"   ✅ Groq API responded ({len(result)} chars)")
        return result
        
    except Exception as e:
        print(f"❌ Groq API error: {e}")
        raise


def extract_json_block(text: str) -> Optional[dict]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    json_str = text[start:end+1]
    try:
        return json.loads(json_str)
    except Exception:
        return None

def clean_llm_explanation(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value.strip()
    
    if isinstance(value, dict):
        if "reason_for_relevance" in value:
            return str(value["reason_for_relevance"]).strip()
        
        text_fields = [
            v for v in value.values() 
            if isinstance(v, str) and len(v) > 15
        ]
        
        if text_fields:
            return max(text_fields, key=len).strip()
    
    return None

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


class RecommendRequest(BaseModel):
    query_text: Optional[str] = ""
    chat_text: Optional[str] = ""
    geography_id: Optional[str] = None
    ref_unit_id: Optional[str] = None
    query_lcia_name: Optional[str] = None
    query_upr_exchange_name: Optional[str] = None
    query_stage_name: Optional[str] = None
    query_process_name: Optional[str] = None
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

class LlmParsedMetadata(BaseModel):
    mode: Optional[Literal["replace_all", "refine", "keep"]] = None

    query_lcia_name: Optional[str] = None
    query_upr_exchange_name: Optional[str] = None
    query_stage_name: Optional[str] = None
    query_process_name: Optional[str] = None

    geography_name: Optional[str] = None
    ref_unit_name: Optional[str] = None
    database_names: Optional[List[str]] = None
    topk: None = None

    class Config:
        extra = "ignore"


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

async def resolve_geography_id_by_name(session: AsyncSession, name: str) -> Optional[str]:
    if not name:
        return None
    name = name.strip()
    if not name:
        return None

    sql = text("""
        SELECT id FROM lca.geography
        WHERE LOWER(name) = LOWER(:name)
        LIMIT 1
    """)
    res = await session.execute(sql, {"name": name})
    row = res.first()
    if row:
        return str(row.id)

    sql2 = text("""
        SELECT id FROM lca.geography
        WHERE name ILIKE :pat
        ORDER BY name ASC
        LIMIT 1
    """)
    res2 = await session.execute(sql2, {"pat": f"%{name}%"})
    row2 = res2.first()
    return str(row2.id) if row2 else None


async def resolve_unit_id_by_name(session: AsyncSession, name: str) -> Optional[str]:
    if not name:
        return None
    name = name.strip()
    if not name:
        return None

    sql = text("""
        SELECT id FROM lca.unit
        WHERE LOWER(name) = LOWER(:name)
        LIMIT 1
    """)
    res = await session.execute(sql, {"name": name})
    row = res.first()
    if row:
        return str(row.id)

    sql2 = text("""
        SELECT id FROM lca.unit
        WHERE name ILIKE :pat
        ORDER BY name ASC
        LIMIT 1
    """)
    res2 = await session.execute(sql2, {"pat": f"%{name}%"})
    row2 = res2.first()
    return str(row2.id) if row2 else None


async def resolve_lcia_database_ids_by_names(
    session: AsyncSession,
    names: List[str],
) -> List[str]:
    if not names:
        return []
    clean_names = [n.strip() for n in names if n and n.strip()]
    if not clean_names:
        return []

    sql = text("""
        SELECT id, name
        FROM lca.lcia_database
        WHERE name ILIKE ANY(:patterns)
    """)
    patterns = [f"%{n}%" for n in clean_names]
    res = await session.execute(sql, {"patterns": patterns})
    rows = res.all()
    return [str(r.id) for r in rows]


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


async def enrich_request_with_llm_intent(
    session: AsyncSession,
    req: RecommendRequest,
) -> RecommendRequest:
    if not ENABLE_LLM_INTENT:
        print("ℹ️ LLM intent parsing is disabled")
        return req

    chat = normalize_text_field(req.chat_text)
    if not chat:
        return req

    try:
        async def _parse_intent():
            system_prompt = """
You are an assistant for a Life Cycle Assessment (LCA) EF/LCIA recommendation system.

Your job is to convert the user's free-text request ("chat_text") plus the current structured
query fields into **clean, consistent search metadata** that can be used to find suitable
LCIA/EF datasets (e.g. from ecoinvent).

You MUST follow this decision tree:

1) Decide the overall mode for this request:

- "replace_all": the user is clearly describing a NEW scenario that does NOT match the
  existing query_* fields (e.g. we were talking about aluminium smelting before, and now the
  user talks about tomato cultivation).
- "refine": the user is clearly refining the SAME scenario (e.g. adding geography, unit,
  stage, but still talking about the same product/system).
- "keep": the user input is too vague to safely change filters. In this case you should not
  change any field.

**Very important rules for mode:**
- If chat_text describes a different product or system than current_query_lcia_name and
  current_query_upr_exchange_name, you MUST set mode = "replace_all".
- If chat_text only adds geography / unit / stage / technology details for the same product,
  use mode = "refine".
- If chat_text is vague like "show me more options", "next", "similar ones", use mode = "keep".

2) For each metadata field, decide whether to:
- keep the existing value from current_*  → set the field to null
- override with a better value           → set the field to a new non-empty string
- explicitly clear a conflicting value   → set the field to the empty string ""

IMPORTANT semantics:
- null  = "keep whatever is in current_query_*"
- ""    = "clear the existing value (it conflicts with chat_text and you cannot give a better value)"

**Very important conflict rule:**
If chat_text clearly contradicts the existing value for a field (product, process, stage,
geography, unit, database), you MUST NOT return null for that field.
You must either give a new non-empty value, or set it to "".

3) Field-specific rules:

- query_lcia_name:
  * Treat this as a dataset / process / market name in ecoinvent style
    (e.g. "electricity production, natural gas, combined cycle power plant",
     "tomato production, fresh grade, open field").
  * DO NOT change it to generic impact categories like "climate change" or
    "agricultural soil occupation" UNLESS the user explicitly asks for a specific
    impact category (e.g. "climate change / GWP / land occupation").
  * If the existing query_lcia_name clearly refers to a different product or
    system than chat_text, either provide a new, consistent name or set it to "".

- query_upr_exchange_name:
  * This is the reference product / flow name, usually short (e.g. "electricity, high voltage",
    "tomato, fresh grade").
  * If chat_text specifies a different product, override it.

- query_stage_name:
  * Use this for the life cycle stage, e.g. "production", "cultivation", "import",
    "use phase", "waste disposal".
  * If chat_text mentions such a stage, set it accordingly. Otherwise you can keep it.
  * If chat_text clearly indicates a different stage than the current one, override or "".

- query_process_name:
  * This should be a more specific process name, close to what appears in the database
    (see the provided examples).
  * If chat_text adds details (technology, size, geography), refine or adjust the
    process name to reflect those details.

- geography_name:
  * ONLY set this when the user clearly mentions a geography (country, region, GLO, etc.).
  * If chat_text mentions a geography that conflicts with the existing one, either provide
    the new geography name or set it to "".

- ref_unit_name (very important for functional units):
  * ONLY change this if the user clearly refers to a unit or functional unit, e.g.:
    "per kWh", "per kWh of electricity", "per kg of tomato",
    "per tonne of waste", "per km", "per passenger.km", "per m2 of land", etc.
  * Map these to simple unit names where possible:
    - "per kg" → "kg"
    - "per t" / "per tonne" → "t"
    - "per kWh" → "kWh"
    - "per km" → "km"
    - "per passenger-km" → "passenger.km"
    - "per m2" → "m2"
  * If the user gives a clear unit, you SHOULD fill ref_unit_name with that unit string.
  * If no unit is mentioned, leave ref_unit_name as null (keep existing) unless you need
    to clear a conflicting unit.

- database_names:
  * ONLY output non-null if the user explicitly constrains the LCIA/EF database
    (e.g. mentions "ecoinvent 3.9.1", "local Singapore EF DB", "UK Defra", etc.).
  * If the user does not mention any database, you MUST set database_names to null.
  * NEVER guess or invent database names, and NEVER expand to multiple databases
    unless the user clearly requests that.

- topk:
  * ALWAYS set topk to null. Never infer a number of results.

4) Output **only ONE JSON object** with the exact schema:

{
  "mode": "replace_all | refine | keep",
  "query_lcia_name": string or null or "",
  "query_upr_exchange_name": string or null or "",
  "query_stage_name": string or null or "",
  "query_process_name": string or null or "",
  "geography_name": string or null or "",
  "ref_unit_name": string or null or "",
  "database_names": [string] or null,
  "topk": null
}

Do NOT output any text outside this single JSON object.
"""

            chat_trimmed = chat[:800] if len(chat) > 800 else chat
            payload = {
                "chat_text": chat_trimmed,
                "current_query_lcia_name": req.query_lcia_name,
                "current_query_upr_exchange_name": req.query_upr_exchange_name,
                "current_query_stage_name": req.query_stage_name,
                "current_query_process_name": req.query_process_name,
            }

            user_prompt = (
                "Here is the user input and current structured query fields:\n\n"
                f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
                "Based on this, produce ONE JSON object following the schema above.\n"
                "Remember: null = keep existing; \"\" = clear existing.\n"
                "JSON:"
            )

            full_prompt = system_prompt + "\n\n" + user_prompt

            raw_output = await asyncio.to_thread(llm_generate, full_prompt, 400)
            parsed_json = extract_json_block(raw_output)
            if not parsed_json:
                return None

            return LlmParsedMetadata.model_validate(parsed_json)

        parsed = await asyncio.wait_for(_parse_intent(), timeout=LLM_TIMEOUT)
        if not parsed:
            print("⚠️ LLM intent parse failed, keep original request.")
            return req

        mode = (parsed.mode or "refine").lower()
        if mode not in {"replace_all", "refine", "keep"}:
            mode = "refine"

        if mode == "keep":
            print("ℹ️ LLM mode=keep → do not change request.")
            return req

        def apply_field(current_value: Optional[str], new_value: Optional[str]) -> Optional[str]:
            """
            - new_value is None:
                refine      → keep current
                replace_all → clear (set None)
            - new_value == "" (empty string from LLM):
                → clear (set None)
            - else:
                → override with new_value
            """
            if new_value is None:
                return None if mode == "replace_all" else current_value
            if isinstance(new_value, str) and new_value.strip() == "":
                return None
            return new_value

        req.query_lcia_name = apply_field(req.query_lcia_name, parsed.query_lcia_name)
        req.query_upr_exchange_name = apply_field(req.query_upr_exchange_name, parsed.query_upr_exchange_name)
        req.query_stage_name = apply_field(req.query_stage_name, parsed.query_stage_name)
        req.query_process_name = apply_field(req.query_process_name, parsed.query_process_name)
        geo_name = apply_field(None, parsed.geography_name)
        if geo_name is not None:
            if geo_name == "":
                req.geography_id = None
                print("   🌍 Geography cleared by LLM")
            else:
                geo_id = await resolve_geography_id_by_name(session, geo_name)
                if geo_id:
                    req.geography_id = geo_id
                    print(f"   🌍 Geography updated to: {geo_name} (ID: {geo_id})")
                else:
                    print(f"   ⚠️ Geography '{geo_name}' not found in DB, keeping original")
        unit_name = apply_field(None, parsed.ref_unit_name)
        if unit_name is not None:
            if unit_name == "":
                req.ref_unit_id = None
                print("   📏 Unit cleared by LLM")
            else:
                unit_id = await resolve_unit_id_by_name(session, unit_name)
                if unit_id:
                    req.ref_unit_id = unit_id
                    print(f"   📏 Unit updated to: {unit_name} (ID: {unit_id})")
                else:
                    print(f"   ⚠️ Unit '{unit_name}' not found in DB, keeping original")

        if parsed.database_names is not None:
            if isinstance(parsed.database_names, list) and len(parsed.database_names) == 0:
                if req.filters is not None:
                    req.filters.database_ids = None
                print("   💾 Database filter cleared by LLM")
            else:
                db_ids: List[str] = await resolve_lcia_database_ids_by_names(session, parsed.database_names)
                if db_ids:
                    if req.filters is None:
                        req.filters = Filters()
                    req.filters.database_ids = db_ids
                    print(f"   💾 Database filter updated to: {parsed.database_names} (IDs: {db_ids})")
                else:
                    print(f"   ⚠️ Database names {parsed.database_names} not found in DB, keeping original filter")

        print(f"✅ LLM intent parsing applied (mode={mode})")
        return req

    except asyncio.TimeoutError:
        print(f"⏱️ LLM intent parsing timeout ({LLM_TIMEOUT}s), using original request")
        return req
    except Exception as e:
        print(f"⚠️ LLM intent parsing error: {e}, using original request")
        return req

def _build_lcia_context_sentence(
    lcia_name: Optional[str],
    upr_exchange_name: Optional[str],
    stage_name: Optional[str],
    process_name: Optional[str],
) -> str:
    parts = []

    lcia = (lcia_name or "").strip()
    upr = (upr_exchange_name or "").strip()
    stage = (stage_name or "").strip()
    proc = (process_name or "").strip()

    if upr:
        parts.append(f"reference product \"{upr}\"")
    if stage:
        parts.append(f"life cycle stage \"{stage}\"")
    if proc:
        parts.append(f"process \"{proc}\"")

    if lcia:
        head = f"LCIA dataset \"{lcia}\""
    else:
        head = "LCIA dataset"

    if parts:
        return head + " modelling " + ", ".join(parts)
    else:
        return head


def _normalize_product_name(name: Optional[str]) -> str:
    if not name:
        return ""
    s = str(name).lower()
    s = s.split(",")[0]
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


def _product_match_label(query_product: Optional[str], candidate_product: Optional[str]) -> str:
    q = _normalize_product_name(query_product)
    c = _normalize_product_name(candidate_product)

    if not q or not c:
        return "unknown"

    if q == c or q in c or c in q:
        return "exact"

    q_tokens = set(q.split())
    c_tokens = set(c.split())
    if q_tokens & c_tokens:
        return "overlap"

    return "different"

async def llm_explain_results(
    req: RecommendRequest,
    topk_scored: List[tuple[float, dict, dict]],
) -> Dict[str, str]:
    if not ENABLE_LLM_EXPLANATION:
        print("ℹ️ LLM explanation is disabled, using fallback")
        return _fallback_explanations(topk_scored)
    
    if not topk_scored:
        return {}
    fallback_technical = _fallback_technical_info(topk_scored)
    
    try:
        async def _generate_semantic_explanations():
            query_context = {
                "lcia_name": req.query_lcia_name or "",
                "upr_exchange_name": req.query_upr_exchange_name or "",
                "stage_name": req.query_stage_name or "",
                "process_name": req.query_process_name or "",
                "combined_text": _build_lcia_context_sentence(
                    req.query_lcia_name,
                    req.query_upr_exchange_name,
                    req.query_stage_name,
                    req.query_process_name,
                ),
            }
            
            items_for_llm = []
            expected_ids = set()

            for score, row, details in topk_scored:
                desc_id = str(row.get("lcia_description_id"))
                expected_ids.add(desc_id)

                product_match = _product_match_label(
                    req.query_upr_exchange_name,
                    row.get("upr_exchange_name"),
                )

                lcia_name = (row.get("lcia_name") or "")[:160]
                upr_name = (row.get("upr_exchange_name") or "")[:120]
                stage_name = (row.get("stage_name") or "")[:160]
                process_name = (row.get("process_name") or "")[:160]

                item = {
                    "id": desc_id,
                    "lcia_name": lcia_name,
                    "upr_exchange_name": upr_name,
                    "stage_name": stage_name,
                    "process_name": process_name,
                    "product_match_label": product_match,
                    "combined_text": _build_lcia_context_sentence(
                        lcia_name,
                        upr_name,
                        stage_name,
                        process_name,
                    ),
                }
                items_for_llm.append(item)


            prompt = f"""
You are an LCA/LCIA expert explanation assistant.

The system has already selected candidate EF/LCIA datasets using semantic retrieval and rule-based scoring.
Your task is ONLY to explain, from an LCA perspective, why each candidate is relevant (or how it can be used)
for the user's modelling intent, based on the naming of LCIA, reference product, stage and process.

USER INTENT (structured fields and combined_text summary):
{json.dumps(query_context, ensure_ascii=False, indent=2)}

CANDIDATE DATASETS (one item per LCIA description, with structured fields and combined_text):
{json.dumps(items_for_llm, ensure_ascii=False, indent=2)}


IMPORTANT FIELD:
- product_match_label:
  - "exact": same core product as the user intent (e.g. "tomato, fresh grade" vs "tomato production, fresh grade").
  - "overlap": closely related product wording with shared tokens (e.g. different variants of the same product).
  - "different": clearly different product/crop (e.g. tomato vs maize, grape, chickpea, bell pepper).
  - "unknown": insufficient information.

FOR EACH DATASET, write ONE independent English explanation in a short paragraph that:

1. Focuses on how lcia_name / upr_exchange_name / stage_name / process_name relate to the user's intent.
2. Clearly reflects product_match_label:
   - If "exact": emphasise that this is a direct match for the requested product/process within the life cycle.
   - If "overlap": describe it as a closely related variant of the requested product, explaining the relation.
   - If "different": explicitly state that it models a different product/crop, and can only serve as a proxy or
     comparison dataset when no exact data are available.
3. Indicate whether it represents a market, production process, or supporting process within the life cycle
   (based ONLY on the wording of the names, e.g. "market for...", "production", "voltage transformation").
4. Use positive, concise language that LCA practitioners can understand.

STRICT RULES:
- Each explanation is INDEPENDENT. Do NOT reference other datasets
  (avoid phrases like "similar to the previous one", "like the first match").
- Do NOT mention units or geography — these will be explained separately.
- Base your reasoning ONLY on the strings provided in the fields and product_match_label.
  Do NOT invent fuels, technologies, energy mixes or parameters that are not explicitly named.
- Avoid negative or apologetic language such as:
  "although not exactly", "not explicitly", "might be", "possibly".
- Do NOT talk about model versions, databases, or numerical scores.

GOOD EXAMPLES:
- (product_match_label = "exact")
  "Directly models tomato production, fresh grade in open-field systems, matching the requested crop and management stage."
- (product_match_label = "different")
  "Represents maize grain production in similar field-based systems and can serve as a proxy when tomato-specific data are unavailable."
- (electricity example)
  "Provides high-voltage electricity specifically for aluminium industry operations, supporting energy demand in smelting and related stages."

BAD EXAMPLES:
- "Similar to the first dataset, this one..."
- "While not explicitly matching the smelting stage..."
- "This might be relevant if the process uses hydro power..."

OUTPUT FORMAT:
Return ONLY a single JSON object mapping dataset IDs to explanation strings:

{{
  "dataset-id-1": "Explanation text here",
  "dataset-id-2": "Explanation text here"
}}

Do NOT output markdown. Do NOT add comments. Start directly with '{{'.
JSON:
"""

            print(f"\n🤖 Requesting LLM semantic explanations for {len(items_for_llm)} items")

            raw_output = await asyncio.to_thread(llm_generate, prompt, 1200)
            
            print(f"🤖 LLM output: {len(raw_output)} chars")
            print(f"   Preview: {raw_output[:300]}...")

            raw_output_clean = raw_output.replace("```json", "").replace("```", "").strip()
            parsed = extract_json_block(raw_output_clean)

            if not parsed or not isinstance(parsed, dict):
                print("❌ Failed to parse JSON from LLM output")
                return None
            
            cleaned_parsed: Dict[str, str] = {}
            for desc_id, value in parsed.items():
                text = clean_llm_explanation(value)
                if text:
                    cleaned_parsed[desc_id] = text

            matched = sum(1 for did in expected_ids if did in cleaned_parsed)
            total = len(expected_ids)
            print(f"   Matched explanations: {matched}/{total}")
            
            if total > 0 and matched / total < 0.5:
                print(f"   ⚠️ Low explanation coverage ({matched}/{total}), falling back to rule-only explanations")
                return None
            
            return cleaned_parsed

        llm_semantic = await asyncio.wait_for(_generate_semantic_explanations(), timeout=120.0)
        
        if llm_semantic and len(llm_semantic) > 0:
            print(f"✅ LLM semantic OK - Got {len(llm_semantic)} explanations")
            
            final_explanations: Dict[str, str] = {}
            for score, row, details in topk_scored:
                did = str(row["lcia_description_id"])
                semantic_part = llm_semantic.get(did, "")
                technical_part = fallback_technical.get(did, "").strip()
                
                if semantic_part and technical_part:
                    final_explanations[did] = f"{semantic_part} {technical_part}"
                elif semantic_part:
                    final_explanations[did] = semantic_part
                elif technical_part:
                    final_explanations[did] = f"Relevant LCIA dataset. {technical_part}"
                else:
                    final_explanations[did] = "Relevant LCIA dataset selected by semantic retrieval and rule-based scoring."
            
            return final_explanations
        else:
            print("⚠️ LLM returned empty or invalid explanations, using pure fallback")
            return _fallback_explanations(topk_scored)
    
    except asyncio.TimeoutError:
        print(f"⏱️ LLM timeout (120s), using pure fallback")
        return _fallback_explanations(topk_scored)
    except Exception as e:
        print(f"❌ LLM error: {e}")
        return _fallback_explanations(topk_scored)


def _fallback_technical_info(topk_scored: List[tuple[float, dict, dict]]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for _, row, details in topk_scored:
        desc_id = str(row.get("lcia_description_id"))
        parts: List[str] = []

        unit_info = details.get("unit", {})
        unit_match = unit_info.get("match_type", "no_match")
        unit_name = row.get("unit_name")

        if unit_match == "exact_match":
            if unit_name:
                parts.append(f"Unit exactly matches ({unit_name}).")
            else:
                parts.append("Unit exactly matches.")
        elif unit_match == "convertible":
            parts.append(f"Unit is directly convertible to ({unit_name}).")
        elif unit_match in ("same_type", "same_type_and_system"):
            parts.append("Unit measures the same physical quantity and is convertible.")
        elif unit_match == "same_system":
            parts.append("Unit is in the same unit system and convertible.")
        geo_info = details.get("geography", {})
        geo_match = geo_info.get("match_type")
        geo_score = geo_info.get("score", 0.0)
        geo_name = row.get("geography_name")

        if geo_match == "exact_match":
            if geo_name:
                parts.append(f"Geography exactly matches ({geo_name}).")
            else:
                parts.append("Geography exactly matches.")
        elif geo_match == "parent_child_1_level":
            parts.append(f"Geography is in a directly related parent/child region of ({geo_name}).")
        elif geo_match == "parent_child_2_levels":
            parts.append(f"Geography is within the same regional hierarchy, two levels away from ({geo_name}).")
        elif geo_match == "parent_child_3_levels":
            parts.append(f"Geography is within the same regional hierarchy, three levels away from ({geo_name}).")
        elif geo_match == "common_ancestor":
            parts.append(f"Geography shares a common ancestor region with ({geo_name}) in the hierarchy.")
        elif geo_match == "distant_or_global":
            parts.append(f"Geography is more generic or global compared to ({geo_name}).")

        if not parts and geo_score and geo_score > 0.0:
            parts.append("Geography is related to the requested region in the hierarchy.")

        result[desc_id] = " ".join(parts).strip()

    return result


def _fallback_explanations(topk_scored: List[tuple[float, dict, dict]]) -> Dict[str, str]:

    result: Dict[str, str] = {}
    for score, row, details in topk_scored:
        desc_id = str(row.get("lcia_description_id"))
        
        parts: List[str] = []
        
        sem_score = details.get("semantic", {}).get("score", 0)
        if sem_score > 4.5:
            parts.append("Strong semantic alignment with query requirements")
        elif sem_score > 4.0:
            parts.append("Good semantic relevance to query")
        else:
            parts.append("Moderate semantic match")
        
        unit = details.get("unit", {})
        unit_match = unit.get("match_type", "no_match")
        unit_name = row.get("unit_name")

        if unit_match == "exact_match":
            if unit_name:
                parts.append(f"exact unit match ({unit_name})")
            else:
                parts.append("exact unit match")
        elif unit_match == "convertible":
            parts.append("convertible units (compatible measurement)")
        elif unit_match == "same_type":
            parts.append("same unit type")
        elif unit_match == "same_type_and_system":
            parts.append("same unit type and system")
        elif unit_match == "same_system":
            parts.append("same unit system")
        
        geo = details.get("geography", {})
        geo_score = geo.get("score", 0)
        
        if geo_score >= 1.0:
            parts.append("exact geographic match")
        elif geo_score >= 0.75:
            parts.append("close geographic proximity (1 level in hierarchy)")
        elif geo_score >= 0.55:
            parts.append("related geography (2 levels apart)")
        elif geo_score >= 0.40:
            parts.append("broader geographic scope (3 levels)")
        elif geo_score > 0.2:
            parts.append("common geographic region")
        
        if len(parts) >= 2:
            explanation = f"{parts[0]}. {', '.join(parts[1:])}."
        else:
            explanation = f"Relevant match (total score: {score:.2f}). {parts[0] if parts else 'Meets search criteria'}."
        
        result[desc_id] = explanation
    
    return result

async def build_query_embedding_v2(
    session: AsyncSession,
    req: RecommendRequest,
) -> tuple[Optional[List[float]], Optional[List[float]], str]:

    metadata_parts = []
    if normalize_text_field(req.query_lcia_name):
        metadata_parts.append(f"LCIA name: {normalize_text_field(req.query_lcia_name)}")
    if normalize_text_field(req.query_upr_exchange_name):
        metadata_parts.append(f"Reference product: {normalize_text_field(req.query_upr_exchange_name)}")
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
    
    print(f"     • Stage 3 - After filtering: {len(filtered_results)} results")
    
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
    
    await session.execute(text("SET hnsw.ef_search = 400;"))
    
    sql = text("""
        SELECT
            s.lcia_description_id,
            s.embedding_text,
            s.embedding <=> CAST(:qvec AS vector) AS distance
        FROM lca.rag_lcia_semantic_index s
        WHERE s.embedding IS NOT NULL
        ORDER BY distance ASC
        LIMIT :limit;
    """)
    
    res = await session.execute(sql, {
        "qvec": vec_literal, 
        "limit": limit,
    })
    results = [dict(r) for r in res.mappings().all()]
    
    print(f"📊 _vector_search returned {len(results)} results (requested {limit})")
    if results:
        print(f"   Distance range: {results[0]['distance']:.4f} ~ {results[-1]['distance']:.4f}")
    
    return results


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

def to_lcia_card(row: dict, explain: Optional[str] = None) -> LciaCard:
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
            explain=explain or "Based on LCIA description semantic retrieval plus weighted ranking of unit and geography.",
        )
    except Exception as e:
        raise



app = FastAPI(title="LEAF RAG LCIA Recommendations", version="1.0.0-groq-only")

@app.on_event("startup")
async def startup_event():
    print("🚀 Initializing LCIA Recommendation API...")
    print(f"   • LLM Intent: {'✅ Enabled' if ENABLE_LLM_INTENT else '❌ Disabled'}")
    print(f"   • LLM Explanation: {'✅ Enabled' if ENABLE_LLM_EXPLANATION else '❌ Disabled'}")
    print(f"   • LLM Timeout: {LLM_TIMEOUT}s")
    print(f"   • LLM Provider: 🌐 Groq API (llama-3.3-70b-versatile)")
    
    if not GROQ_API_KEY:
        print("   ⚠️  WARNING: GROQ_API_KEY not set - LLM features will fail")
    else:
        get_groq_client()
    
    await asyncio.to_thread(get_embed_model)
    
    print("✅ API ready")


@app.post("/lcia/recommendations", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    try:
        async with SessionLocal() as session:
            req = await enrich_request_with_llm_intent(session, req)

            print("\n" + "="*80)
            print("📋 AFTER LLM Intent Parsing:")
            print("="*80)

            metadata_vec, query_vec, semantic_text = await build_query_embedding_v2(session, req)
            
            print(f"  📋 Enriched Metadata Fields:")
            print(f"     • query_lcia_name: {req.query_lcia_name or '(empty)'}")
            print(f"     • query_upr_exchange_name: {req.query_upr_exchange_name or '(empty)'}")
            print(f"     • query_stage_name: {req.query_stage_name or '(empty)'}")
            print(f"     • query_process_name: {req.query_process_name or '(empty)'}")
            print(f"     • ref_unit_id: {req.ref_unit_id or '(none)'}")
            print(f"     • geography_id: {req.geography_id or '(none)'}")
            print(f"     • database_ids: {req.filters.database_ids if req.filters and req.filters.database_ids else '(none)'}")
            print(f"     • topk: {req.topk}")
            print(f"\n  📝 Enriched Semantic Text:\n     {semantic_text}")
            

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

            if len(filtered_scored) < req.topk:
                print(f"\n⚠️  WARNING: Database filter reduced results from {len(scored)} to {len(filtered_scored)}")
                print(f"   Requested top-{req.topk}, but only {len(filtered_scored)} results match the database filter.")
                print(f"   Consider:")
                print(f"   1. Removing the database filter")
                print(f"   2. Adding more database IDs to the filter")
                print(f"   3. Reducing topk to {len(filtered_scored)}")

        print("\n" + "🎯" * 40)
        print(f"Top-{req.topk} Results with Detailed Scoring")
        print("🎯" * 40 + "\n")
        
        topk_results = filtered_scored[:req.topk]
        for rank, (score, row, details) in enumerate(topk_results, 1):
            print_scoring_details(rank, row, score, details)

        explanations_map: Dict[str, str] = await llm_explain_results(req, topk_results)

        print(f"\n📝 Explanations map has {len(explanations_map)} entries")
        if explanations_map:
            print(f"   Sample keys: {list(explanations_map.keys())[:3]}")

        cards = []
        for (score, row, details) in topk_results:
            did = str(row["lcia_description_id"])
            exp = explanations_map.get(did)
            if exp:
                print(f"✅ Found explanation for {did[:8]}...")
            else:
                print(f"⚠️ No explanation for {did[:8]}..., using default")
            cards.append(to_lcia_card(row, explain=exp))

        return RecommendResponse(items=[c.model_dump() for c in cards])

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "ok", 
        "service": "LEAF RAG LCIA API", 
        "version": "1.0.0-groq-only",
        "llm_provider": "Groq API (llama-3.3-70b-versatile)",
        "llm_intent_enabled": ENABLE_LLM_INTENT,
        "llm_explanation_enabled": ENABLE_LLM_EXPLANATION,
        "llm_timeout": LLM_TIMEOUT,
        "groq_configured": GROQ_API_KEY is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_new:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
