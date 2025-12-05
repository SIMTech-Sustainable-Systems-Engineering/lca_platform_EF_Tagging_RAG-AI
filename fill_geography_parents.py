
import os
import asyncio
import csv
from typing import Dict
from urllib.parse import quote_plus
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PWD = os.getenv("DB_PWD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME")

if not (DB_USER and DB_PWD and DB_NAME):
    raise RuntimeError("Please set DB_USER, DB_PWD, DB_NAME (and optionally DB_HOST).")

DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{quote_plus(DB_PWD)}@{DB_HOST}/{DB_NAME}"

engine = create_async_engine(DATABASE_URL, pool_pre_ping=True, echo=False)
SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)


class GeographyParentResult:
    def __init__(self):
        self.total_geographies = 0
        self.matched_by_iso3166_2 = 0
        self.matched_by_unsd_m49 = 0
        self.matched_by_country_code = 0
        self.matched_special_cases = 0
        self.matched_by_without_regions = 0
        self.matched_by_grid_regions = 0
        self.matched_by_iai_regions = 0
        self.matched_by_special_islands = 0
        self.matched_by_sovereignty = 0
        self.matched_by_complex_exclusions = 0
        self.matched_by_economic_regions = 0
        self.matched_by_fuzzy_name = 0
        self.matched_by_country_to_continent = 0
        self.matched_by_regional_groups = 0
        self.matched_by_continents_to_global = 0
        self.matched_by_disputed_territories = 0
        self.matched_by_australia_oceania = 0
        self.matched_by_missing_african_countries = 0
        self.matched_by_un_regions = 0
        self.matched_by_iai_areas_enhanced = 0
        self.matched_by_geographic_features = 0
        self.matched_by_un_subregions_asia = 0
        self.unmatched = 0
        self.errors = []
        
    def print_report(self):
        print("\n" + "="*80)
        print("📊 GEOGRAPHY PARENT FILLING REPORT - v2.0")
        print("="*80)
        print(f"📍 Total Geographies: {self.total_geographies}")
        print(f"✅ Successfully Matched: {self.total_matched}")
        print(f"\n🔹 Core Rules (High Impact):")
        print(f"   • Country → Continent (Rule 15): {self.matched_by_country_to_continent}")
        print(f"   • Name Prefix Match (Rule 2.5): {self.matched_by_country_code}")
        print(f"   • Sovereignty Field (Rule 10): {self.matched_by_sovereignty}")
        print(f"   • Regional Groups (Rule 16): {self.matched_by_regional_groups}")
        print(f"\n🔹 Specialized Rules:")
        print(f"   • ISO 3166-2 Subdivisions (Rule 1): {self.matched_by_iso3166_2}")
        print(f"   • UNSD M.49 (Rule 2): {self.matched_by_unsd_m49}")
        print(f"   • Without Regions (Rule 6): {self.matched_by_without_regions}")
        print(f"   • Grid Regions (Rule 7): {self.matched_by_grid_regions}")
        print(f"   • IAI Regions (Rule 8): {self.matched_by_iai_regions}")
        print(f"   • Special Islands (Rule 9): {self.matched_by_special_islands}")
        print(f"   • Complex Exclusions (Rule 12): {self.matched_by_complex_exclusions}")
        print(f"   • Economic Regions (Rule 13): {self.matched_by_economic_regions}")
        print(f"   • Fuzzy Matching (Rule 14): {self.matched_by_fuzzy_name}")
        print(f"   • Special Cases (Rule 4): {self.matched_special_cases}")
        print(f"\n🔹 Final Sweep Rules (New!):")
        print(f"   • Continents → Global (Rule 17): {self.matched_by_continents_to_global}")
        print(f"   • Disputed Territories (Rule 18): {self.matched_by_disputed_territories}")
        print(f"   • Australia/Oceania Fix (Rule 19): {self.matched_by_australia_oceania}")
        print(f"   • Missing African Countries (Rule 20): {self.matched_by_missing_african_countries}")
        print(f"   • UN Regions (Rule 21): {self.matched_by_un_regions}")
        print(f"   • IAI Areas Enhanced (Rule 22): {self.matched_by_iai_areas_enhanced}")
        print(f"   • Geographic Features (Rule 23): {self.matched_by_geographic_features}")
        print(f"   • UN Asia subregions (Rule 24): {self.matched_by_un_subregions_asia}")
        print(f"\n❌ Unmatched: {self.unmatched}")
        print(f"📈 Coverage: {self.coverage_percentage:.2f}%")
        
        if self.errors:
            print(f"\n⚠️  Errors encountered: {len(self.errors)}")
            for error in self.errors[:5]:
                print(f"   • {error}")
        
        print("="*80 + "\n")
    
    @property
    def total_matched(self):
        return (self.matched_by_iso3166_2 + self.matched_by_unsd_m49 + 
                self.matched_by_country_code + self.matched_special_cases +
                self.matched_by_without_regions + self.matched_by_grid_regions +
                self.matched_by_iai_regions + self.matched_by_special_islands +
                self.matched_by_sovereignty + self.matched_by_complex_exclusions +
                self.matched_by_economic_regions + self.matched_by_fuzzy_name +
                self.matched_by_country_to_continent + self.matched_by_regional_groups +
                self.matched_by_continents_to_global + self.matched_by_disputed_territories +
                self.matched_by_australia_oceania + self.matched_by_missing_african_countries +
                self.matched_by_un_regions + self.matched_by_iai_areas_enhanced +
                self.matched_by_geographic_features +
                self.matched_by_un_subregions_asia)
    
    @property
    def coverage_percentage(self):
        if self.total_geographies == 0:
            return 0.0
        return (self.total_matched / self.total_geographies) * 100


async def create_geography_parent_table(session: AsyncSession):
    print("🔧 Creating geography_parent table...")
    try:
        await session.execute(text("DROP TABLE IF EXISTS lca.geography_parent CASCADE;"))
        await session.execute(text("""
            CREATE TABLE lca.geography_parent (
                id UUID DEFAULT gen_random_uuid() NOT NULL PRIMARY KEY,
                geography_id UUID NOT NULL REFERENCES lca.geography(id) ON UPDATE CASCADE ON DELETE CASCADE,
                parent_geography_id UUID NOT NULL REFERENCES lca.geography(id) ON UPDATE CASCADE ON DELETE CASCADE,
                match_method VARCHAR(50),
                confidence VARCHAR(20) DEFAULT 'high',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                CONSTRAINT geography_parent_unique UNIQUE (geography_id)
            );
        """))
        await session.execute(text("COMMENT ON TABLE lca.geography_parent IS 'Parent-child relationships for geographical entities';"))
        
        for col, desc in [("geography_id", "Child geography ID"), ("parent_geography_id", "Parent geography ID"),
                         ("match_method", "Matching method used"), ("confidence", "Confidence level"), ("notes", "Additional notes")]:
            await session.execute(text(f"COMMENT ON COLUMN lca.geography_parent.{col} IS '{desc}';"))
        
        for idx_col in ["geography_id", "parent_geography_id", "match_method"]:
            await session.execute(text(f"CREATE INDEX idx_geography_parent_{idx_col} ON lca.geography_parent({idx_col});"))
        
        await session.commit()
        print("✅ Table created\n")
    except Exception as e:
        await session.rollback()
        print(f"❌ Error: {e}")
        raise

async def analyze_current_data(session: AsyncSession) -> Dict:
    print("="*80)
    print("🔍 ANALYZING CURRENT GEOGRAPHY DATA")
    print("="*80)
    stats = await session.execute(text("""
        SELECT COUNT(*) as total, COUNT("iso3166-1-alpha-2") as has_iso2,
               COUNT("iso3166-2") as has_iso_subdivision, COUNT("unsd-m49") as has_m49,
               COUNT(geography_type_id) as has_type
        FROM lca.geography;
    """))
    overall = stats.fetchone()
    print(f"\n📊 Overall Statistics:")
    print(f"   Total: {overall.total}")
    print(f"   With ISO 3166-1 Alpha-2: {overall.has_iso2} ({overall.has_iso2/overall.total*100:.1f}%)")
    print(f"   With UNSD M.49: {overall.has_m49} ({overall.has_m49/overall.total*100:.1f}%)")
    print("="*80 + "\n")
    return {"total": overall.total}


async def rule_1_iso3166_2_matching(session: AsyncSession, result: GeographyParentResult):
    print("🔄 Rule 1: Processing ISO 3166-2 subdivisions...")
    try:
        res = await session.execute(text("""
            INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
            SELECT child.id, parent.id, 'iso3166-2', 'high',
                'ISO 3166-2: ' || child."iso3166-2" || ' → ' || parent."iso3166-1-alpha-2"
            FROM lca.geography child
            JOIN lca.geography parent ON parent."iso3166-1-alpha-2" = LEFT(child."iso3166-2", 2)
            WHERE child."iso3166-2" IS NOT NULL AND LENGTH(child."iso3166-2") >= 2
              AND parent."iso3166-2" IS NULL
              AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
            ON CONFLICT (geography_id) DO NOTHING;
        """))
        result.matched_by_iso3166_2 = res.rowcount
        print(f"   ✅ Matched {res.rowcount} subdivisions")
        await session.commit()
    except Exception as e:
        await session.rollback()
        result.errors.append(f"Rule 1: {str(e)}")
        print(f"   ❌ Error: {e}")

async def rule_2_unsd_m49_matching(session: AsyncSession, result: GeographyParentResult):
    print("\n🔄 Rule 2: Processing UNSD M.49 country→region...")
    m49_country_to_region = {'012': '002', '818': '002', '434': '002'}
    try:
        matched_count = 0
        for child_code, parent_code in m49_country_to_region.items():
            res = await session.execute(text("""
                INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
                SELECT child.id, parent.id, 'm49', 'high', 'UNSD M.49: ' || :child_code || ' → ' || :parent_code
                FROM lca.geography child JOIN lca.geography parent ON parent."unsd-m49" = :parent_code
                WHERE child."unsd-m49" = :child_code
                  AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
                ON CONFLICT (geography_id) DO NOTHING;
            """), {"child_code": child_code, "parent_code": parent_code})
            matched_count += res.rowcount
        result.matched_by_unsd_m49 = matched_count
        print(f"   ✅ Matched {matched_count} countries to regions")
        await session.commit()
    except Exception as e:
        await session.rollback()
        result.errors.append(f"Rule 2: {str(e)}")
        print(f"   ❌ Error: {e}")

async def rule_25_country_code_in_name(session: AsyncSession, result: GeographyParentResult):
    print("\n🔄 Rule 2.5: Matching by country name in title...")
    try:
        res = await session.execute(text("""
            INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
            SELECT DISTINCT ON (child.id) child.id, parent.id, 'name_prefix', 'high',
                'Country prefix in name: ' || parent.name
            FROM lca.geography child JOIN lca.geography parent ON 
                child.name LIKE parent.name || ',%' OR child.name LIKE parent.name || ' %'
            WHERE parent."iso3166-1-alpha-2" IS NOT NULL AND child."iso3166-1-alpha-2" IS NULL
              AND parent.id != child.id
              AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
            ORDER BY child.id, LENGTH(parent.name) DESC;
        """))
        count1 = res.rowcount
        res = await session.execute(text("""
            INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
            SELECT DISTINCT ON (child.id) child.id, parent.id, 'alternate_code', 'high',
                'Matched via alternate_code array'
            FROM lca.geography child JOIN lca.geography parent ON parent."iso3166-1-alpha-2" IS NOT NULL
                AND EXISTS (SELECT 1 FROM unnest(child.alternate_code) AS code
                    WHERE code LIKE parent."iso3166-1-alpha-2" || '-%')
            WHERE child.alternate_code IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
            ORDER BY child.id;
        """))
        count2 = res.rowcount
        result.matched_by_country_code = count1 + count2
        print(f"   ✅ Matched {count1} by name prefix, {count2} by alternate code")
        await session.commit()
    except Exception as e:
        await session.rollback()
        result.errors.append(f"Rule 2.5: {str(e)}")
        print(f"   ❌ Error: {e}")

async def rule_4_special_cases(session: AsyncSession, result: GeographyParentResult):
    print("\n🔄 Rule 4: Handling special cases...")
    try:
        matched_count = 0
        res = await session.execute(text("""
            INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
            SELECT child.id, parent.id, 'special_row', 'high', 'Rest of World entity'
            FROM lca.geography child CROSS JOIN lca.geography parent
            WHERE (child.name ILIKE 'RoW%' OR child.name ILIKE 'Rest of%' OR child.name ILIKE 'ROW%')
              AND (parent.name IN ('GLO', 'Global', 'World') OR parent."unsd-m49" = '001')
              AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
            LIMIT 100;
        """))
        matched_count += res.rowcount
        city_states = {'Singapore': 'Asia', 'Monaco': 'Europe', 'Vatican City': 'Europe',
                      'San Marino': 'Europe', 'Hong Kong': 'Asia', 'Macao': 'Asia'}
        for city_state, continent in city_states.items():
            res = await session.execute(text("""
                INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
                SELECT child.id, parent.id, 'special_city_state', 'high', 'City-state → ' || :continent
                FROM lca.geography child CROSS JOIN lca.geography parent
                WHERE child.name = :city_state AND parent.name = :continent
                  AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
                LIMIT 1;
            """), {"city_state": city_state, "continent": continent})
            matched_count += res.rowcount
        result.matched_special_cases = matched_count
        print(f"   ✅ Handled {matched_count} special cases")
        await session.commit()
    except Exception as e:
        await session.rollback()
        result.errors.append(f"Rule 4: {str(e)}")
        print(f"   ❌ Error: {e}")

async def rule_6_without_regions(session: AsyncSession, result: GeographyParentResult):
    print("\n🔄 Rule 6: Matching 'without' regions...")
    try:
        res = await session.execute(text("""
            INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
            SELECT DISTINCT ON (child.id) child.id, parent.id, 'without_region', 'high',
                'Exclusion region based on: ' || parent.name
            FROM lca.geography child JOIN lca.geography parent ON 
                child.name ~ ('^' || parent.name || ' without ') OR child.name ~ ('^' || parent.name || ', without ')
            WHERE child.name ~ '.* without .*' AND parent.id != child.id
              AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
            ORDER BY child.id, LENGTH(parent.name) DESC;
        """))
        result.matched_by_without_regions = res.rowcount
        print(f"   ✅ Matched {res.rowcount} exclusion regions")
        await session.commit()
    except Exception as e:
        await session.rollback()
        result.errors.append(f"Rule 6: {str(e)}")
        print(f"   ❌ Error: {e}")

async def rule_7_grid_regions(session: AsyncSession, result: GeographyParentResult):
    print("\n🔄 Rule 7: Matching grid regions...")
    grid_mappings = {'BR-': 'Brazil', 'CN-': 'China', 'IN-': 'India', 'US-': 'United States',
                    'CA-': 'Canada', 'AU-': 'Australia', 'RU-': 'Russia', 'EU-': 'Europe'}
    try:
        matched_count = 0
        for prefix, country_name in grid_mappings.items():
            res = await session.execute(text("""
                INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
                SELECT child.id, parent.id, 'grid_region_prefix', 'high',
                    'Grid region: ' || :prefix || ' → ' || :country_name
                FROM lca.geography child JOIN lca.geography parent ON parent.name = :country_name
                WHERE child.name LIKE :prefix || '%'
                  AND (child.name ILIKE '%grid%' OR child.name ILIKE '%network%' OR child.name ~ '[A-Z]{2,}-[A-Z]+')
                  AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
                LIMIT 50;
            """), {"prefix": prefix, "country_name": country_name})
            matched_count += res.rowcount
        result.matched_by_grid_regions = matched_count
        print(f"   ✅ Matched {matched_count} grid regions")
        await session.commit()
    except Exception as e:
        await session.rollback()
        result.errors.append(f"Rule 7: {str(e)}")
        print(f"   ❌ Error: {e}")

async def rule_8_iai_regions(session: AsyncSession, result: GeographyParentResult):
    print("\n🔄 Rule 8: Matching IAI regions...")
    try:
        res = await session.execute(text("""
            INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
            SELECT DISTINCT ON (child.id) child.id, parent.id, 'iai_region', 'high',
                'IAI region → ' || parent.name
            FROM lca.geography child JOIN lca.geography parent ON 
                child.name ~ ('^IAI Area, ' || parent.name || '$') OR child.name ~ ('^IAI Area, ' || parent.name || ',')
            WHERE child.name LIKE 'IAI Area,%' AND parent.id != child.id
              AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
            ORDER BY child.id, LENGTH(parent.name) DESC;
        """))
        result.matched_by_iai_regions = res.rowcount
        print(f"   ✅ Matched {res.rowcount} IAI regions")
        await session.commit()
    except Exception as e:
        await session.rollback()
        result.errors.append(f"Rule 8: {str(e)}")
        print(f"   ❌ Error: {e}")

async def rule_9_special_islands(session: AsyncSession, result: GeographyParentResult):
    print("\n🔄 Rule 9: Matching special islands...")
    island_mappings = {
        'Canary Islands': 'Spain', 'British Virgin Islands': 'Europe',
        'Coral Sea Islands': 'Australia', 'Cayman Islands': 'North America',
        'Clipperton Island': 'France', 'French Polynesia': 'France',
        'New Caledonia': 'France', 'American Samoa': 'United States',
        'Guam': 'United States', 'Northern Mariana Islands': 'United States',
        'US Virgin Islands': 'United States', 'Christmas Island': 'Australia',
        'Cocos Islands': 'Australia', 'Norfolk Island': 'Australia',
        'Svalbard and Jan Mayen': 'Europe'
    }
    try:
        matched_count = 0
        for island, parent_name in island_mappings.items():
            res = await session.execute(text("""
                INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
                SELECT child.id, parent.id, 'special_island', 'high',
                    'Island territory: ' || :island || ' → ' || :parent_name
                FROM lca.geography child JOIN lca.geography parent ON parent.name = :parent_name
                WHERE child.name = :island
                  AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
                LIMIT 1;
            """), {"island": island, "parent_name": parent_name})
            matched_count += res.rowcount
        result.matched_by_special_islands = matched_count
        print(f"   ✅ Matched {matched_count} special islands")
        await session.commit()
    except Exception as e:
        await session.rollback()
        result.errors.append(f"Rule 9: {str(e)}")
        print(f"   ❌ Error: {e}")

async def rule_10_sovereignty_matching(session: AsyncSession, result: GeographyParentResult):
    print("\n🔄 Rule 10: Matching by sovereignty field...")
    try:
        res = await session.execute(text("""
            INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
            SELECT DISTINCT ON (child.id) child.id, parent.id, 'sovereignty', 'high',
                'Sovereignty: ' || child.sovereignty || ' → ' || parent.official_state_name
            FROM lca.geography child JOIN lca.geography parent ON 
                child.sovereignty IS NOT NULL AND parent.official_state_name IS NOT NULL
                AND (LOWER(TRIM(child.sovereignty)) = LOWER(TRIM(parent.official_state_name))
                     OR LOWER(TRIM(child.sovereignty)) = LOWER(TRIM(parent.name)))
            WHERE child.id != parent.id
              AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
            ORDER BY child.id, LENGTH(parent.official_state_name) DESC;
        """))
        result.matched_by_sovereignty = res.rowcount
        print(f"   ✅ Matched {res.rowcount} by sovereignty")
        await session.commit()
    except Exception as e:
        await session.rollback()
        result.errors.append(f"Rule 10: {str(e)}")
        print(f"   ❌ Error: {e}")

async def rule_12_complex_exclusion_regions(session: AsyncSession, result: GeographyParentResult):
    print("\n🔄 Rule 12: Matching complex exclusion regions...")
    complex_mappings = {
        'RER': 'Europe', 'RoW': 'GLO', 'WECC': 'United States',
        'NPCC': 'United States', 'RFC': 'United States', 'SERC': 'United States',
        'TRE': 'United States', 'MRO': 'United States', 'SPP': 'United States',
        'FRCC': 'United States', 'ENTSO-E': 'Europe', 'UCTE': 'Europe', 'NORDEL': 'Europe',
    }
    try:
        matched_count = 0
        for code, parent_name in complex_mappings.items():
            res = await session.execute(text("""
                INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
                SELECT child.id, parent.id, 'complex_exclusion', 'medium',
                       'Complex region: ' || :code || ' → ' || :parent_name
                FROM lca.geography child JOIN lca.geography parent ON parent.name = :parent_name
                WHERE (child.name LIKE :code || ',%' OR child.name LIKE :code || ' %' OR child.name = :code)
                  AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
                LIMIT 10;
            """), {"code": code, "parent_name": parent_name})
            matched_count += res.rowcount
        result.matched_by_complex_exclusions = matched_count
        print(f"   ✅ Matched {matched_count} complex exclusion regions")
        await session.commit()
    except Exception as e:
        await session.rollback()
        result.errors.append(f"Rule 12: {str(e)}")
        print(f"   ❌ Error: {e}")

async def rule_13_economic_regions(session: AsyncSession, result: GeographyParentResult):
    print("\n🔄 Rule 13: Matching economic regions...")
    economic_mappings = {
        'APEC': 'Asia', 'ASEAN': 'Asia', 'EU-27': 'Europe', 'EU-28': 'Europe',
        'EU27': 'Europe', 'EU28': 'Europe', 'NAFTA': 'North America',
        'MERCOSUR': 'South America', 'AU': 'Africa',
    }
    try:
        matched_count = 0
        for code, parent_name in economic_mappings.items():
            res = await session.execute(text("""
                INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
                SELECT child.id, parent.id, 'economic_region', 'medium',
                       'Economic region: ' || :code || ' → ' || :parent_name
                FROM lca.geography child JOIN lca.geography parent ON parent.name = :parent_name
                WHERE (child.name LIKE :code || ',%' OR child.name LIKE :code || ' %' OR child.name LIKE '%' || :code || '%')
                  AND LENGTH(child.name) < 50
                  AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
                LIMIT 5;
            """), {"code": code, "parent_name": parent_name})
            matched_count += res.rowcount
        result.matched_by_economic_regions = matched_count
        print(f"   ✅ Matched {matched_count} economic regions")
        await session.commit()
    except Exception as e:
        await session.rollback()
        result.errors.append(f"Rule 13: {str(e)}")
        print(f"   ❌ Error: {e}")

async def rule_14_fuzzy_name_enhancement(session: AsyncSession, result: GeographyParentResult):
    print("\n🔄 Rule 14: Enhanced fuzzy name matching...")
    try:
        res = await session.execute(text("""
            INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
            SELECT DISTINCT ON (child.id) child.id, parent.id, 'fuzzy_name_contains', 'low',
                'Fuzzy match: name contains country'
            FROM lca.geography child JOIN lca.geography parent ON 
                parent."iso3166-1-alpha-2" IS NOT NULL
                AND child.name ILIKE '%' || parent.name || '%'
                AND LENGTH(parent.name) >= 6
            WHERE child."iso3166-1-alpha-2" IS NULL AND parent.id != child.id
              AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
            ORDER BY child.id, LENGTH(parent.name) DESC
            LIMIT 50;
        """))
        count1 = res.rowcount
        res = await session.execute(text("""
            INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
            SELECT DISTINCT ON (child.id) child.id, parent.id, 'fuzzy_alternate_code', 'low',
                'Fuzzy match: via alternate codes'
            FROM lca.geography child JOIN lca.geography parent ON 
                child.alternate_code IS NOT NULL AND parent."iso3166-1-alpha-2" IS NOT NULL
                AND EXISTS (SELECT 1 FROM unnest(child.alternate_code) AS code
                    WHERE UPPER(code) LIKE UPPER(parent."iso3166-1-alpha-2") || '%')
            WHERE child."iso3166-1-alpha-2" IS NULL
              AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
            ORDER BY child.id
            LIMIT 30;
        """))
        count2 = res.rowcount
        result.matched_by_fuzzy_name = count1 + count2
        print(f"   ✅ Matched {count1} by name fuzzy, {count2} by alternate code fuzzy")
        await session.commit()
    except Exception as e:
        await session.rollback()
        result.errors.append(f"Rule 14: {str(e)}")
        print(f"   ❌ Error: {e}")

async def rule_15_country_to_continent_m49(session: AsyncSession, result: GeographyParentResult):
    print("\n🔄 Rule 15: Matching countries to continents (UNSD M.49) [FIXED]...")
    try:
        matched_count = 0
        
        res = await session.execute(text("""
            INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
            SELECT child.id, parent.id, 'country_to_continent_m49', 'high', 'M.49 Africa codes → Africa'
            FROM lca.geography child JOIN lca.geography parent ON LOWER(TRIM(parent.name)) = 'africa'
            WHERE child."unsd-m49" IS NOT NULL AND child."iso3166-1-alpha-2" IS NOT NULL
              AND (child."unsd-m49" IN ('002', '015', '011', '017', '014', '018')
                  OR (child."unsd-m49" >= '012' AND child."unsd-m49" < '019')
                  OR (child."unsd-m49" >= '024' AND child."unsd-m49" < '900' AND child."unsd-m49" ~ '^[0-9]{3}$'
                      AND child."unsd-m49" IN ('024','072','108','120','132','140','148','174','178','180','204','262','231','232','226','266','270','288','324','384','404','426','430','434','450','454','466','478','480','504','516','562','566','624','638','646','678','686','690','694','710','716','728','748','768','788','800','818','834','854','894')))
              AND child.id != parent.id
              AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
            ON CONFLICT (geography_id) DO NOTHING;
        """))
        matched_count += res.rowcount
        print(f"      • Africa: {res.rowcount}")
        
        res = await session.execute(text("""
            INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
            SELECT child.id, parent.id, 'country_to_continent_m49', 'high', 'M.49 Americas codes → Americas'
            FROM lca.geography child JOIN lca.geography parent ON LOWER(TRIM(parent.name)) = 'americas'
            WHERE child."unsd-m49" IS NOT NULL AND child."iso3166-1-alpha-2" IS NOT NULL
              AND (child."unsd-m49" IN ('005', '013', '019', '021', '029')
                  OR child."unsd-m49" IN ('028','032','044','052','060','068','076','084','092','124','136','152','170','188','192','212','214','218','222','238','254','308','312','320','328','332','340','388','474','484','500','533','534','535','558','591','600','604','630','652','659','660','662','663','670','740','780','796','840','850','858','862'))
              AND child.id != parent.id
              AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
            ON CONFLICT (geography_id) DO NOTHING;
        """))
        matched_count += res.rowcount
        print(f"      • Americas: {res.rowcount}")
        
        res = await session.execute(text("""
            INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
            SELECT child.id, parent.id, 'country_to_continent_m49', 'high', 'M.49 Asia codes → Asia'
            FROM lca.geography child JOIN lca.geography parent ON LOWER(TRIM(parent.name)) = 'asia'
            WHERE child."unsd-m49" IS NOT NULL AND child."iso3166-1-alpha-2" IS NOT NULL
              AND (child."unsd-m49" IN ('030', '034', '035', '142', '143', '145')
                  OR child."unsd-m49" IN ('004','008','050','051','031','048','064','090','096','104','116','144','156','196','242','268','296','316','356','360','364','368','376','392','398','400','408','410','414','417','418','422','446','458','462','496','512','524','554','586','626','634','682','699','702','704','760','762','764','784','792','795','860','882','886','887'))
              AND child.id != parent.id
              AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
            ON CONFLICT (geography_id) DO NOTHING;
        """))
        matched_count += res.rowcount
        print(f"      • Asia: {res.rowcount}")
        
        res = await session.execute(text("""
            INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
            SELECT child.id, parent.id, 'country_to_continent_m49', 'high', 'M.49 Europe codes → Europe'
            FROM lca.geography child JOIN lca.geography parent ON LOWER(TRIM(parent.name)) = 'europe'
            WHERE child."unsd-m49" IS NOT NULL AND child."iso3166-1-alpha-2" IS NOT NULL
              AND (child."unsd-m49" IN ('039', '150', '151', '154', '155')
                  OR child."unsd-m49" IN ('008','020','040','056','070','100','191','196','203','208','233','234','246','248','250','276','292','300','304','336','348','352','372','380','428','438','440','442','470','492','498','499','528','578','616','620','642','643','674','688','703','705','724','744','752','756','804','807','826','831','832','833','891'))
              AND child.id != parent.id
              AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
            ON CONFLICT (geography_id) DO NOTHING;
        """))
        matched_count += res.rowcount
        print(f"      • Europe: {res.rowcount}")
        
        res = await session.execute(text("""
            INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
            SELECT child.id, parent.id, 'country_to_continent_m49_digit', 'medium', 'M.49 1xx → Europe (fallback)'
            FROM lca.geography child JOIN lca.geography parent ON LOWER(TRIM(parent.name)) = 'europe'
            WHERE child."unsd-m49" ~ '^1[0-9]{2}$' AND child."iso3166-1-alpha-2" IS NOT NULL
              AND child.id != parent.id
              AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
            ON CONFLICT (geography_id) DO NOTHING;
        """))
        matched_count += res.rowcount
        if res.rowcount > 0:
            print(f"      • Europe (1xx fallback): {res.rowcount}")
        
        res = await session.execute(text("""
            INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
            SELECT child.id, parent.id, 'country_to_continent_m49_digit', 'medium', 'M.49 4xx/5xx → Asia (fallback)'
            FROM lca.geography child JOIN lca.geography parent ON LOWER(TRIM(parent.name)) = 'asia'
            WHERE child."unsd-m49" ~ '^[45][0-9]{2}$' AND child."iso3166-1-alpha-2" IS NOT NULL
              AND child.id != parent.id
              AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
            ON CONFLICT (geography_id) DO NOTHING;
        """))
        matched_count += res.rowcount
        if res.rowcount > 0:
            print(f"      • Asia (4xx/5xx fallback): {res.rowcount}")
        
        res = await session.execute(text("""
            INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
            SELECT sub.id, sub.parent_id, 'country_to_continent_m49_digit', 'low', 'M.49 6xx-8xx → Africa/Asia (fallback)'
            FROM (
                SELECT DISTINCT ON (child.id) child.id, parent.id as parent_id
                FROM lca.geography child JOIN lca.geography parent ON LOWER(TRIM(parent.name)) IN ('africa', 'asia')
                WHERE child."unsd-m49" ~ '^[678][0-9]{2}$' AND child."iso3166-1-alpha-2" IS NOT NULL
                  AND child.id != parent.id
                  AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
                ORDER BY child.id, parent.name
                LIMIT 100
            ) sub
            ON CONFLICT (geography_id) DO NOTHING;
        """))
        matched_count += res.rowcount
        if res.rowcount > 0:
            print(f"      • Africa/Asia (6xx-8xx fallback): {res.rowcount}")
        
        result.matched_by_country_to_continent = matched_count
        print(f"   ✅ Total matched {matched_count} countries to continents")
        await session.commit()
    except Exception as e:
        await session.rollback()
        result.errors.append(f"Rule 15: {str(e)}")
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def rule_16_regional_groups(session: AsyncSession, result: GeographyParentResult):
    print("\n🔄 Rule 16: Matching regional groups to continents...")
    regional_mappings = {
        'Caribbean': 'Americas', 'Central America': 'Americas', 'North America': 'Americas',
        'South America': 'Americas', 'Northern America': 'Americas', 'Latin America': 'Americas',
        'Latin America and the Caribbean': 'Americas',

        'Central Asia': 'Asia', 'East Asia': 'Asia',
        'Eastern Asia': 'Asia', 'South Asia': 'Asia', 'Southern Asia': 'Asia',
        'Southeast Asia': 'Asia', 'South-eastern Asia': 'Asia',
        'Western Asia': 'Asia', 'Middle East': 'Asia', 'Asia-Pacific': 'Asia',

        'Asia without China': 'Asia',
        'Russia (Asia)': 'Asia',

        'Eastern Europe': 'Europe', 'Northern Europe': 'Europe', 'Southern Europe': 'Europe',
        'Western Europe': 'Europe', 'Central Europe': 'Europe',

        'Northern Africa': 'Africa', 'Western Africa': 'Africa',
        'Eastern Africa': 'Africa', 'Southern Africa': 'Africa',
        'Middle Africa': 'Africa', 'Sub-Saharan Africa': 'Africa',

        'Commonwealth of Independent States': 'Europe',
        'Central European Power Association': 'Europe',
    }
    try:
        matched_count = 0
        for region_name, continent_name in regional_mappings.items():
            res = await session.execute(text("""
                INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
                SELECT child.id, parent.id, 'regional_group', 'high',
                    'Regional group: ' || :region_name || ' → ' || :continent_name
                FROM lca.geography child JOIN lca.geography parent ON LOWER(TRIM(parent.name)) = LOWER(:continent_name)
                WHERE LOWER(TRIM(child.name)) = LOWER(:region_name) AND child.id != parent.id
                  AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
                LIMIT 1;
            """), {"region_name": region_name, "continent_name": continent_name})
            matched_count += res.rowcount
        result.matched_by_regional_groups = matched_count
        print(f"   ✅ Matched {matched_count} regional groups")
        await session.commit()
    except Exception as e:
        await session.rollback()
        result.errors.append(f"Rule 16: {str(e)}")
        print(f"   ❌ Error: {e}")

async def rule_17_continents_to_global(session: AsyncSession, result: GeographyParentResult):
    print("\n🔄 Rule 17: Matching continents to Global...")
    try:
        res = await session.execute(text("""
            INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
            SELECT child.id, parent.id, 'continent_to_global', 'high',
                   'Continent → Global: ' || child.name
            FROM lca.geography child
            CROSS JOIN lca.geography parent
            WHERE parent.name IN ('Global', 'GLO', 'World')
              AND child.name IN ('Africa', 'Americas', 'Asia', 'Europe', 'Oceania', 'Antarctica')
              AND child."unsd-m49" IN ('002', '019', '142', '150', '009', '010')
              AND child.id != parent.id
              AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
            LIMIT 10;
        """))
        result.matched_by_continents_to_global = res.rowcount
        print(f"   ✅ Matched {res.rowcount} continents to Global")
        await session.commit()
    except Exception as e:
        await session.rollback()
        result.errors.append(f"Rule 17: {str(e)}")
        print(f"   ❌ Error: {e}")

async def rule_18_disputed_territories(session: AsyncSession, result: GeographyParentResult):
    print("\n🔄 Rule 18: Matching disputed territories...")
    disputed_mappings = {
        'Kosovo': 'Europe',
        'Palestinian Territory, Occupied': 'Asia',
        'Somaliland': 'Africa',
    }
    try:
        matched_count = 0
        for territory, continent in disputed_mappings.items():
            res = await session.execute(text("""
                INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
                SELECT child.id, parent.id, 'disputed_territory', 'medium',
                       'Disputed territory: ' || :territory || ' → ' || :continent
                FROM lca.geography child
                JOIN lca.geography parent ON parent.name = :continent
                WHERE child.name = :territory
                  AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
                LIMIT 1;
            """), {"territory": territory, "continent": continent})
            matched_count += res.rowcount
        result.matched_by_disputed_territories = matched_count
        print(f"   ✅ Matched {matched_count} disputed territories")
        await session.commit()
    except Exception as e:
        await session.rollback()
        result.errors.append(f"Rule 18: {str(e)}")
        print(f"   ❌ Error: {e}")

async def rule_19_australia_oceania(session: AsyncSession, result: GeographyParentResult):
    print("\n🔄 Rule 19: Matching Australia/Oceania countries...")
    try:
        res = await session.execute(text("""
            INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
            SELECT child.id, parent.id, 'oceania_country', 'high',
                   'Oceania country: ' || child.name || ' → Oceania'
            FROM lca.geography child
            JOIN lca.geography parent ON LOWER(TRIM(parent.name)) = 'oceania'
            WHERE child.name IN (
                'Australia', 'New Zealand', 'Fiji', 'Papua New Guinea',
                'Solomon Islands', 'Samoa', 'Tonga', 'Vanuatu'
            )
              AND child."iso3166-1-alpha-2" IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
            LIMIT 50;
        """))
        result.matched_by_australia_oceania = res.rowcount
        print(f"   ✅ Matched {res.rowcount} Oceania countries")
        await session.commit()
    except Exception as e:
        await session.rollback()
        result.errors.append(f"Rule 19: {str(e)}")
        print(f"   ❌ Error: {e}")


async def rule_20_missing_african_countries(session: AsyncSession, result: GeographyParentResult):
    print("\n🔄 Rule 20: Matching missing African countries...")
    african_countries = ['Djibouti', 'Equatorial Guinea', 'Eritrea', 'Ethiopia', 
                        'Gabon', 'Gambia', 'Ghana', 'Guinea']
    try:
        matched_count = 0
        for country in african_countries:
            res = await session.execute(text("""
                INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
                SELECT child.id, parent.id, 'african_country_manual', 'high',
                       'African country: ' || :country || ' → Africa'
                FROM lca.geography child
                JOIN lca.geography parent ON parent.name = 'Africa'
                WHERE child.name = :country
                  AND child."iso3166-1-alpha-2" IS NOT NULL
                  AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
                LIMIT 1;
            """), {"country": country})
            matched_count += res.rowcount
        result.matched_by_missing_african_countries = matched_count
        print(f"   ✅ Matched {matched_count} African countries")
        await session.commit()
    except Exception as e:
        await session.rollback()
        result.errors.append(f"Rule 20: {str(e)}")
        print(f"   ❌ Error: {e}")

async def rule_21_un_regions(session: AsyncSession, result: GeographyParentResult):
    print("\n🔄 Rule 21: Matching UN regions...")
    un_region_mappings = {
        'Europe, UN Region': 'Europe',
        'Melanesia': 'Oceania',
        'Polynesia': 'Oceania',
        'Micronesia': 'Oceania',  
    }
    try:
        matched_count = 0
        for region, continent in un_region_mappings.items():
            res = await session.execute(text("""
                INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
                SELECT child.id, parent.id, 'un_region', 'high',
                       'UN region: ' || :region || ' → ' || :continent
                FROM lca.geography child
                JOIN lca.geography parent ON parent.name = :continent
                WHERE child.name = :region
                  AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
                LIMIT 1;
            """), {"region": region, "continent": continent})
            matched_count += res.rowcount
        result.matched_by_un_regions = matched_count
        print(f"   ✅ Matched {matched_count} UN regions")
        await session.commit()
    except Exception as e:
        await session.rollback()
        result.errors.append(f"Rule 21: {str(e)}")
        print(f"   ❌ Error: {e}")

async def rule_22_iai_areas_enhanced(session: AsyncSession, result: GeographyParentResult):
    print("\n🔄 Rule 22: Matching IAI areas (enhanced)...")
    iai_mappings = {
        'IAI Area, Gulf Cooperation Council': 'Asia',
        'IAI Area, North America': 'Americas',
    }
    try:
        matched_count = 0
        for iai_area, continent in iai_mappings.items():
            res = await session.execute(text("""
                INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
                SELECT child.id, parent.id, 'iai_area_enhanced', 'high',
                       'IAI area: ' || :iai_area || ' → ' || :continent
                FROM lca.geography child
                JOIN lca.geography parent ON parent.name = :continent
                WHERE child.name = :iai_area
                  AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
                LIMIT 1;
            """), {"iai_area": iai_area, "continent": continent})
            matched_count += res.rowcount
        result.matched_by_iai_areas_enhanced = matched_count
        print(f"   ✅ Matched {matched_count} IAI areas")
        await session.commit()
    except Exception as e:
        await session.rollback()
        result.errors.append(f"Rule 22: {str(e)}")
        print(f"   ❌ Error: {e}")

async def rule_23_geographic_features(session: AsyncSession, result: GeographyParentResult):
    print("\n🔄 Rule 23: Matching geographic features and special zones...")
    special_mappings = {
        'Dhekelia Sovereign Base Area': 'Asia',
        'Siachen Glacier': 'Asia',
        'Scarborough Reef': 'Asia',
        'Serranilla Bank': 'Americas',
        'Spratly Islands': 'Asia',
        'European Network of Transmission Systems Operators for Electricity': 'Europe',
        'Union for the Co-ordination of Transmission of Electricity': 'Europe',
        'Western Electricity Coordinating Council': 'Americas',
        'North America without Quebec': 'Americas',
        'Québec, Hydro-Québec distribution network': 'Americas',
        'Oceania': 'Global',
    }
    try:
        matched_count = 0
        for feature, parent_name in special_mappings.items():
            res = await session.execute(text("""
                INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
                SELECT child.id, parent.id, 'geographic_feature', 'medium',
                       'Geographic feature: ' || :feature || ' → ' || :parent_name
                FROM lca.geography child
                JOIN lca.geography parent ON parent.name = :parent_name
                WHERE child.name = :feature
                  AND NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = child.id)
                LIMIT 1;
            """), {"feature": feature, "parent_name": parent_name})
            matched_count += res.rowcount
        result.matched_by_geographic_features = matched_count
        print(f"   ✅ Matched {matched_count} geographic features")
        await session.commit()
    except Exception as e:
        await session.rollback()
        result.errors.append(f"Rule 23: {str(e)}")
        print(f"   ❌ Error: {e}")

async def rule_24_country_to_un_subregion_asia(session: AsyncSession, result: GeographyParentResult):
    print("\n🔄 Rule 24: Matching Asian countries to UN subregions...")
    asia_subregions = {
        # Eastern Asia (030)
        'Eastern Asia': [
            'CN',  # China
            'HK',  # Hong Kong
            'MO',  # Macao
            'JP',  # Japan
            'MN',  # Mongolia
            'KP',  # DPRK
            'KR',  # Republic of Korea
            'TW',  # Taiwan 
        ],
        # South-Eastern Asia (035)
        'South-Eastern Asia': [
            'BN',  # Brunei Darussalam
            'KH',  # Cambodia
            'ID',  # Indonesia
            'LA',  # Lao PDR
            'MY',  # Malaysia
            'MM',  # Myanmar
            'PH',  # Philippines
            'SG',  # Singapore
            'TH',  # Thailand
            'TL',  # Timor-Leste
            'VN',  # Viet Nam
        ],
        # Southern Asia (034)
        'South Asia': [
            'AF',  # Afghanistan
            'BD',  # Bangladesh
            'BT',  # Bhutan
            'IN',  # India
            'IR',  # Iran (Islamic Republic of)
            'MV',  # Maldives
            'NP',  # Nepal
            'PK',  # Pakistan
            'LK',  # Sri Lanka
        ],
        # Central Asia (143)
        'Central Asia': [
            'KZ',  # Kazakhstan
            'KG',  # Kyrgyzstan
            'TJ',  # Tajikistan
            'TM',  # Turkmenistan
            'UZ',  # Uzbekistan
        ],
        # Western Asia (145)
        'Western Asia': [
            'AM',  # Armenia
            'AZ',  # Azerbaijan
            'BH',  # Bahrain
            'CY',  # Cyprus
            'GE',  # Georgia
            'IQ',  # Iraq
            'IL',  # Israel
            'JO',  # Jordan
            'KW',  # Kuwait
            'LB',  # Lebanon
            'OM',  # Oman
            'QA',  # Qatar
            'SA',  # Saudi Arabia
            'PS',  # State of Palestine
            'SY',  # Syrian Arab Republic
            'TR',  # Turkey
            'AE',  # United Arab Emirates
            'YE',  # Yemen
        ],
    }

    try:
        matched_count = 0
        for region_name, iso_list in asia_subregions.items():
            for iso2 in iso_list:
                res = await session.execute(text("""
                    INSERT INTO lca.geography_parent
                        (geography_id, parent_geography_id, match_method, confidence, notes)
                    SELECT child.id, parent.id,
                           'un_subregion_asia', 'high',
                           'UN Asia subregion: ' || :region_name
                    FROM lca.geography child
                    JOIN lca.geography parent
                      ON LOWER(TRIM(parent.name)) = LOWER(:region_name)
                    WHERE child."iso3166-1-alpha-2" = :iso2
                      AND child."unsd-m49" IS NOT NULL
                      AND NOT EXISTS (
                          SELECT 1 FROM lca.geography_parent gp
                          WHERE gp.geography_id = child.id
                      )
                    LIMIT 1;
                """), {"region_name": region_name, "iso2": iso2})
                matched_count += res.rowcount

        result.matched_by_un_subregions_asia = matched_count
        print(f"   ✅ Matched {matched_count} Asian countries to UN subregions")
        await session.commit()
    except Exception as e:
        await session.rollback()
        result.errors.append(f"Rule 24: {str(e)}")
        print(f"   ❌ Error: {e}")


async def export_unmatched_geographies(session: AsyncSession) -> int:
    print("\n📄 Exporting unmatched geographies with all columns...")
    try:
        rows = await session.execute(text("""
            SELECT g.*, gt.name as type_name, gp.id as parent_link_exists, parent_geo.name as suggested_parent_name
            FROM lca.geography g
            LEFT JOIN lca.geography_type gt ON g.geography_type_id = gt.id
            LEFT JOIN lca.geography_parent gp ON gp.geography_id = g.id
            LEFT JOIN lca.geography parent_geo ON parent_geo.id = g.geography_parent_id
            WHERE gp.id IS NULL
            ORDER BY g.name;
        """))
        unmatched = rows.fetchall()
        if not unmatched:
            print("   ✅ All matched!")
            return 0
        
        column_names = rows.keys()
        filename = f"unmatched_geographies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=column_names)
            writer.writeheader()
            for row in unmatched:
                writer.writerow(dict(row._mapping))
        
        print(f"   ✅ Exported {len(unmatched)} records to: {filename}")
        return len(unmatched)
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0

async def validate_results(session: AsyncSession):
    print("\n" + "="*80)
    print("🔍 VALIDATING RESULTS")
    print("="*80)
    try:
        print("\n1️⃣  Checking for circular references...")
        circular = await session.execute(text("""
            WITH RECURSIVE geo_path AS (
                SELECT gp.geography_id, gp.parent_geography_id, ARRAY[gp.geography_id] as path, 1 as depth
                FROM lca.geography_parent gp
                UNION ALL
                SELECT gp.geography_id, gp.parent_geography_id, geo_path.path || gp.geography_id, geo_path.depth + 1
                FROM lca.geography_parent gp JOIN geo_path ON gp.geography_id = geo_path.parent_geography_id
                WHERE NOT gp.geography_id = ANY(geo_path.path) AND geo_path.depth < 20
            )
            SELECT COUNT(*) FROM geo_path WHERE array_length(path, 1) > 10;
        """))
        circular_count = circular.scalar() or 0
        print(f"   {'⚠️  Found ' + str(circular_count) + ' potential circular references!' if circular_count > 0 else '✅ No circular references found'}")
    except Exception as e:
        print(f"   ⚠️  Circular reference check failed: {e}")
    
    try:
        print("\n2️⃣  Match method distribution:")
        methods = await session.execute(text("""
            SELECT match_method, confidence, COUNT(*) as count
            FROM lca.geography_parent GROUP BY match_method, confidence ORDER BY count DESC;
        """))
        for row in methods:
            print(f"   • {row.match_method} ({row.confidence}): {row.count} records")
    except Exception as e:
        print(f"   ⚠️  Method distribution check failed: {e}")
    
    print("="*80 + "\n")

async def rule_0_insert_unmatched(session: AsyncSession, result: GeographyParentResult):
    print("\n🔄 Rule 0: Inserting unmatched geographies...")
    try:
        res = await session.execute(text("""
            INSERT INTO lca.geography_parent (geography_id, parent_geography_id, match_method, confidence, notes)
            SELECT g.id,
                   (SELECT id FROM lca.geography WHERE name IN ('Global','World','GLO') LIMIT 1),
                   'unmatched', 'low', 'No parent found, fallback inserted'
            FROM lca.geography g
            WHERE NOT EXISTS (SELECT 1 FROM lca.geography_parent gp WHERE gp.geography_id = g.id);
        """))
        result.unmatched = res.rowcount
        print(f"   ✅ Inserted {res.rowcount} unmatched geographies")
        await session.commit()
    except Exception as e:
        await session.rollback()
        result.errors.append(f"Rule 0: {str(e)}")
        print(f"   ❌ Error: {e}")


async def main():
    print("\n" + "🌍"*40)
    print("GEOGRAPHY PARENT RELATIONSHIP FILLING SCRIPT - v2.0 FINAL OPTIMIZED")
    print("Target: 98%+ coverage (Current: 93.68% → Expected: 98%+)")
    print("🌍"*40 + "\n")
    
    result = GeographyParentResult()
    
    async with SessionLocal() as session:
        try:
            analysis = await analyze_current_data(session)
            result.total_geographies = analysis["total"]
            
            await create_geography_parent_table(session)
            await rule_24_country_to_un_subregion_asia(session, result)
            await rule_15_country_to_continent_m49(session, result)
            await rule_25_country_code_in_name(session, result)
            await rule_16_regional_groups(session, result)
            await rule_10_sovereignty_matching(session, result)
            
            await rule_1_iso3166_2_matching(session, result)
            await rule_2_unsd_m49_matching(session, result)
            await rule_4_special_cases(session, result)
            await rule_6_without_regions(session, result)
            await rule_7_grid_regions(session, result)
            await rule_8_iai_regions(session, result)
            await rule_9_special_islands(session, result)
            await rule_12_complex_exclusion_regions(session, result)
            await rule_13_economic_regions(session, result)
            await rule_14_fuzzy_name_enhancement(session, result)
            
            await rule_17_continents_to_global(session, result)
            await rule_18_disputed_territories(session, result)
            await rule_19_australia_oceania(session, result)
            await rule_20_missing_african_countries(session, result)
            await rule_21_un_regions(session, result)
            await rule_22_iai_areas_enhanced(session, result)
            await rule_23_geographic_features(session, result)
            
            result.unmatched = result.total_geographies - result.total_matched
            result.print_report()
            
            if result.unmatched > 0:
                await export_unmatched_geographies(session)
            
            await validate_results(session)
            await rule_0_insert_unmatched(session, result)

            print(f"\n✅ Completed! Coverage: {result.coverage_percentage:.2f}%")
            print(f"🎯 Target achieved: {result.coverage_percentage >= 98.0}")
            
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())