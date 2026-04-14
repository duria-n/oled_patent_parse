"""PostgreSQL 主数据存储 + RDKit + Apache AGE 图同步。"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from contextlib import contextmanager

from .config import PostgresConfig
from .models import StructuredPatentDocument

_MAT_ALIAS_KEY_RE = re.compile(r"[^A-Za-z0-9]+")


class PostgresPatentStore:
    def __init__(self, config: PostgresConfig):
        try:
            import psycopg
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("请先安装 psycopg: pip install psycopg[binary]") from exc
        self._psycopg = psycopg
        self.config = config

    @contextmanager
    def _connect(self):
        conn = self._psycopg.connect(self.config.dsn)
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _normalize_material_alias_key(text: str | None) -> str:
        if not text:
            return ""
        return _MAT_ALIAS_KEY_RE.sub("", str(text)).upper()

    @staticmethod
    def _fallback_material_canonical_id(text: str | None) -> str:
        raw = str(text or "").strip().lower()
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12].upper() if raw else "0" * 12
        return f"mat:RAW{digest}"

    @classmethod
    def _iter_material_alias_texts(cls, entity_row: dict) -> list[str]:
        values: list[str] = []
        for key in ("value_text", "normalized"):
            raw = entity_row.get(key)
            if not isinstance(raw, str):
                continue
            text = raw.strip()
            if not text:
                continue
            values.append(text)
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped

    @classmethod
    def _collect_material_alias_keys(cls, entity_rows: list[dict]) -> list[str]:
        keys: set[str] = set()
        for row in entity_rows:
            if row.get("entity_type") != "material":
                continue
            for alias_text in cls._iter_material_alias_texts(row):
                key = cls._normalize_material_alias_key(alias_text)
                if key:
                    keys.add(key)
        return sorted(keys)

    def _load_existing_material_alias_map(self, cur, schema: str, alias_keys: list[str]) -> dict[str, str]:
        if not alias_keys:
            return {}
        cur.execute(
            f"""
            SELECT alias_key, canonical_id
            FROM {schema}.material_alias
            WHERE alias_key = ANY(%s)
            """,
            (alias_keys,),
        )
        out: dict[str, str] = {}
        for alias_key, canonical_id in cur.fetchall():
            if not alias_key or not canonical_id:
                continue
            out[str(alias_key)] = str(canonical_id)
        return out

    @classmethod
    def _prepare_material_registry_rows(
        cls,
        doc_id: str,
        entity_rows: list[dict],
        existing_alias_map: dict[str, str] | None = None,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """准备跨文档 material registry 的批量 upsert 数据。

        返回：
        - canonical_rows: material_canonical 表行
        - alias_rows: material_alias 表行
        - usage_rows: material_doc_usage 表行
        """
        existing_alias_map = existing_alias_map or {}
        canonical_rows_map: dict[str, dict] = {}
        alias_rows_map: dict[str, dict] = {}
        usage_counts: dict[str, int] = defaultdict(int)
        usage_aliases: dict[str, set[str]] = defaultdict(set)

        for row in entity_rows:
            if row.get("entity_type") != "material":
                continue

            alias_texts = cls._iter_material_alias_texts(row)
            alias_pairs: list[tuple[str, str]] = []
            for alias_text in alias_texts:
                alias_key = cls._normalize_material_alias_key(alias_text)
                if alias_key:
                    alias_pairs.append((alias_text, alias_key))
            alias_keys = [alias_key for _, alias_key in alias_pairs]

            canonical_id = row.get("canonical_id")
            if not canonical_id or canonical_id == "PENDING_MAPPING":
                seed_text = alias_texts[0] if alias_texts else row.get("entity_id")
                canonical_id = cls._fallback_material_canonical_id(str(seed_text or ""))

            # 先使用已存在的 alias->canonical 映射，保证跨文档稳定映射。
            mapped_from_existing = None
            for alias_key in alias_keys:
                mapped = existing_alias_map.get(alias_key)
                if mapped:
                    mapped_from_existing = mapped
                    break
            if mapped_from_existing:
                canonical_id = mapped_from_existing

            canonical_id = str(canonical_id)
            row["canonical_id"] = canonical_id

            canonical_key = ""
            if canonical_id.startswith("mat:"):
                canonical_key = canonical_id.split(":", 1)[1]
            if not canonical_key:
                canonical_key = cls._normalize_material_alias_key(alias_texts[0] if alias_texts else canonical_id)

            preferred_name = alias_texts[0] if alias_texts else canonical_id
            existing_canonical = canonical_rows_map.get(canonical_id)
            if existing_canonical:
                old_name = existing_canonical.get("preferred_name") or ""
                if old_name and preferred_name:
                    if len(preferred_name) < len(old_name):
                        existing_canonical["preferred_name"] = preferred_name
                elif preferred_name:
                    existing_canonical["preferred_name"] = preferred_name
            else:
                canonical_rows_map[canonical_id] = {
                    "canonical_id": canonical_id,
                    "canonical_key": canonical_key,
                    "preferred_name": preferred_name,
                    "doc_id": doc_id,
                }

            usage_counts[canonical_id] += 1
            usage_aliases[canonical_id].update(alias_keys)

            for alias_text, alias_key in alias_pairs:
                existing_alias = alias_rows_map.get(alias_key)
                if existing_alias:
                    # 同一文档内同 alias 冲突时优先沿用已有映射，避免抖动。
                    if existing_alias["canonical_id"] != canonical_id:
                        continue
                    if alias_text and len(alias_text) > len(existing_alias["alias_text"]):
                        existing_alias["alias_text"] = alias_text
                    continue
                alias_rows_map[alias_key] = {
                    "alias_key": alias_key,
                    "alias_text": alias_text,
                    "canonical_id": canonical_id,
                    "doc_id": doc_id,
                }

        usage_rows = [
            {
                "doc_id": doc_id,
                "canonical_id": canonical_id,
                "mention_count": mention_count,
                "alias_keys": sorted(usage_aliases.get(canonical_id) or set()),
            }
            for canonical_id, mention_count in usage_counts.items()
        ]

        return (
            list(canonical_rows_map.values()),
            list(alias_rows_map.values()),
            usage_rows,
        )

    def _upsert_material_registry(self, cur, schema: str, doc_id: str, entity_rows: list[dict]) -> None:
        alias_keys = self._collect_material_alias_keys(entity_rows)
        existing_alias_map = self._load_existing_material_alias_map(cur, schema, alias_keys)
        canonical_rows, alias_rows, usage_rows = self._prepare_material_registry_rows(
            doc_id=doc_id,
            entity_rows=entity_rows,
            existing_alias_map=existing_alias_map,
        )

        # 幂等：先删除该文档旧 usage，再插入新 usage。
        cur.execute(f"DELETE FROM {schema}.material_doc_usage WHERE doc_id = %s", (doc_id,))

        if canonical_rows:
            cur.executemany(
                f"""
                INSERT INTO {schema}.material_canonical(
                    canonical_id, canonical_key, preferred_name,
                    first_seen_doc_id, last_seen_doc_id, updated_at
                ) VALUES (
                    %(canonical_id)s, %(canonical_key)s, %(preferred_name)s,
                    %(doc_id)s, %(doc_id)s, NOW()
                )
                ON CONFLICT (canonical_id) DO UPDATE SET
                    canonical_key = COALESCE(EXCLUDED.canonical_key, {schema}.material_canonical.canonical_key),
                    preferred_name = COALESCE(EXCLUDED.preferred_name, {schema}.material_canonical.preferred_name),
                    last_seen_doc_id = EXCLUDED.last_seen_doc_id,
                    updated_at = NOW();
                """,
                canonical_rows,
            )

        if alias_rows:
            cur.executemany(
                f"""
                INSERT INTO {schema}.material_alias(
                    alias_key, alias_text, canonical_id,
                    first_seen_doc_id, last_seen_doc_id, updated_at, source
                ) VALUES (
                    %(alias_key)s, %(alias_text)s, %(canonical_id)s,
                    %(doc_id)s, %(doc_id)s, NOW(), 'entity_mention'
                )
                ON CONFLICT (alias_key) DO UPDATE SET
                    alias_text = EXCLUDED.alias_text,
                    last_seen_doc_id = EXCLUDED.last_seen_doc_id,
                    updated_at = NOW();
                """,
                alias_rows,
            )

        if usage_rows:
            cur.executemany(
                f"""
                INSERT INTO {schema}.material_doc_usage(
                    doc_id, canonical_id, mention_count, alias_keys
                ) VALUES (
                    %(doc_id)s, %(canonical_id)s, %(mention_count)s, %(alias_keys)s
                )
                ON CONFLICT (doc_id, canonical_id) DO UPDATE SET
                    mention_count = EXCLUDED.mention_count,
                    alias_keys = EXCLUDED.alias_keys;
                """,
                usage_rows,
            )

    def init_schema(self) -> None:
        schema = self.config.schema
        age_graph = self.config.age_graph

        ddl = f"""
        CREATE SCHEMA IF NOT EXISTS {schema};
        CREATE EXTENSION IF NOT EXISTS rdkit;
        CREATE EXTENSION IF NOT EXISTS age;
        LOAD 'age';
        SET search_path = ag_catalog, "$user", public, {schema};

        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM ag_catalog.ag_graph WHERE name = '{age_graph}'
            ) THEN
                PERFORM ag_catalog.create_graph('{age_graph}');
            END IF;
        END
        $$;

        CREATE TABLE IF NOT EXISTS {schema}.document (
            doc_id TEXT PRIMARY KEY,
            publication_number TEXT,
            title TEXT,
            abstract TEXT,
            source_file TEXT,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            claim_tree JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            reference_numerals JSONB,
            structured_path TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS {schema}.block (
            block_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL REFERENCES {schema}.document(doc_id) ON DELETE CASCADE,
            block_type TEXT NOT NULL,
            text TEXT,
            section TEXT,
            subsection TEXT,
            example_id TEXT,
            claim_no INT,
            depends_on JSONB,
            table_id TEXT,
            table_no TEXT,
            figure_id TEXT,
            figure_no TEXT,
            char_offset INT[],
            provenance JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            raw JSONB NOT NULL DEFAULT '{{}}'::jsonb
        );

        CREATE TABLE IF NOT EXISTS {schema}.entity (
            entity_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL REFERENCES {schema}.document(doc_id) ON DELETE CASCADE,
            block_id TEXT NOT NULL REFERENCES {schema}.block(block_id) ON DELETE CASCADE,
            entity_type TEXT NOT NULL,
            value_text TEXT,
            value_num DOUBLE PRECISION,
            unit TEXT,
            value_pair DOUBLE PRECISION[],
            span INT[],
            normalized TEXT,
            canonical_id TEXT,
            smiles TEXT,
            raw JSONB NOT NULL DEFAULT '{{}}'::jsonb
        );

        CREATE TABLE IF NOT EXISTS {schema}.relation (
            relation_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL REFERENCES {schema}.document(doc_id) ON DELETE CASCADE,
            block_id TEXT NOT NULL REFERENCES {schema}.block(block_id) ON DELETE CASCADE,
            relation_type TEXT NOT NULL,
            source_entity_id TEXT REFERENCES {schema}.entity(entity_id) ON DELETE SET NULL,
            target_entity_id TEXT REFERENCES {schema}.entity(entity_id) ON DELETE SET NULL,
            confidence DOUBLE PRECISION,
            rule TEXT,
            distance INT,
            sentence_id TEXT,
            raw JSONB NOT NULL DEFAULT '{{}}'::jsonb
        );

        CREATE TABLE IF NOT EXISTS {schema}.experiment (
            experiment_id TEXT PRIMARY KEY,
            example_id TEXT NOT NULL,
            doc_id TEXT NOT NULL REFERENCES {schema}.document(doc_id) ON DELETE CASCADE,
            materials_used TEXT[] NOT NULL DEFAULT '{{}}',
            performance_relations TEXT[] NOT NULL DEFAULT '{{}}',
            role_relations TEXT[] NOT NULL DEFAULT '{{}}',
            source_block_ids TEXT[] NOT NULL DEFAULT '{{}}',
            raw JSONB NOT NULL DEFAULT '{{}}'::jsonb
        );

        CREATE TABLE IF NOT EXISTS {schema}.molecule (
            molecule_id BIGSERIAL PRIMARY KEY,
            doc_id TEXT NOT NULL REFERENCES {schema}.document(doc_id) ON DELETE CASCADE,
            entity_id TEXT REFERENCES {schema}.entity(entity_id) ON DELETE SET NULL,
            smiles TEXT NOT NULL,
            mol mol,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS {schema}.material_canonical (
            canonical_id TEXT PRIMARY KEY,
            canonical_key TEXT,
            preferred_name TEXT,
            first_seen_doc_id TEXT REFERENCES {schema}.document(doc_id) ON DELETE SET NULL,
            last_seen_doc_id TEXT REFERENCES {schema}.document(doc_id) ON DELETE SET NULL,
            source TEXT NOT NULL DEFAULT 'heuristic',
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS {schema}.material_alias (
            alias_key TEXT PRIMARY KEY,
            alias_text TEXT NOT NULL,
            canonical_id TEXT NOT NULL REFERENCES {schema}.material_canonical(canonical_id) ON DELETE CASCADE,
            first_seen_doc_id TEXT REFERENCES {schema}.document(doc_id) ON DELETE SET NULL,
            last_seen_doc_id TEXT REFERENCES {schema}.document(doc_id) ON DELETE SET NULL,
            source TEXT NOT NULL DEFAULT 'heuristic',
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS {schema}.material_doc_usage (
            doc_id TEXT NOT NULL REFERENCES {schema}.document(doc_id) ON DELETE CASCADE,
            canonical_id TEXT NOT NULL REFERENCES {schema}.material_canonical(canonical_id) ON DELETE CASCADE,
            mention_count INT NOT NULL DEFAULT 0,
            alias_keys TEXT[] NOT NULL DEFAULT '{{}}',
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (doc_id, canonical_id)
        );

        CREATE INDEX IF NOT EXISTS idx_document_publication ON {schema}.document(publication_number);
        CREATE INDEX IF NOT EXISTS idx_document_title_tsv ON {schema}.document USING GIN (to_tsvector('simple', coalesce(title, '')));
        CREATE INDEX IF NOT EXISTS idx_block_doc ON {schema}.block(doc_id);
        CREATE INDEX IF NOT EXISTS idx_block_type ON {schema}.block(block_type);
        CREATE INDEX IF NOT EXISTS idx_block_text_tsv ON {schema}.block USING GIN (to_tsvector('simple', coalesce(text, '')));
        CREATE INDEX IF NOT EXISTS idx_entity_doc ON {schema}.entity(doc_id);
        CREATE INDEX IF NOT EXISTS idx_entity_type ON {schema}.entity(entity_type);
        CREATE INDEX IF NOT EXISTS idx_relation_doc ON {schema}.relation(doc_id);
        CREATE INDEX IF NOT EXISTS idx_experiment_doc ON {schema}.experiment(doc_id);
        CREATE INDEX IF NOT EXISTS idx_experiment_example ON {schema}.experiment(example_id);
        CREATE INDEX IF NOT EXISTS idx_molecule_doc ON {schema}.molecule(doc_id);
        CREATE INDEX IF NOT EXISTS idx_molecule_mol ON {schema}.molecule USING GIST(mol);
        CREATE INDEX IF NOT EXISTS idx_material_canonical_key ON {schema}.material_canonical(canonical_key);
        CREATE INDEX IF NOT EXISTS idx_material_alias_canonical ON {schema}.material_alias(canonical_id);
        CREATE INDEX IF NOT EXISTS idx_material_usage_doc ON {schema}.material_doc_usage(doc_id);
        CREATE INDEX IF NOT EXISTS idx_material_usage_canonical ON {schema}.material_doc_usage(canonical_id);
        """

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(ddl)
            conn.commit()

    def migrate_experiment_primary_key(self, batch_size: int = 50000) -> None:
        """把旧版 experiment(example_id 主键) 迁移到 experiment_id 主键。

        说明：
        - 该方法用于升级已有生产库，避免把大体量 DML 放到 init_schema 热路径。
        - 回填采用分批更新，降低长事务和锁持有时长。
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        schema = self.config.schema
        table = f"{schema}.experiment"

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    experiment_id TEXT PRIMARY KEY,
                    example_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL REFERENCES {schema}.document(doc_id) ON DELETE CASCADE,
                    materials_used TEXT[] NOT NULL DEFAULT '{{}}',
                    performance_relations TEXT[] NOT NULL DEFAULT '{{}}',
                    role_relations TEXT[] NOT NULL DEFAULT '{{}}',
                    source_block_ids TEXT[] NOT NULL DEFAULT '{{}}',
                    raw JSONB NOT NULL DEFAULT '{{}}'::jsonb
                );
                ALTER TABLE {table} ADD COLUMN IF NOT EXISTS experiment_id TEXT;
                ALTER TABLE {table} ADD COLUMN IF NOT EXISTS example_id TEXT;
                """
            )
            conn.commit()

            while True:
                cur.execute(
                    f"""
                    WITH cte AS (
                        SELECT ctid
                        FROM {table}
                        WHERE experiment_id IS NULL
                        LIMIT %s
                    )
                    UPDATE {table} e
                    SET experiment_id = COALESCE(e.doc_id, 'unknown') || '::' || COALESCE(e.example_id, 'unknown')
                    FROM cte
                    WHERE e.ctid = cte.ctid;
                    """,
                    (batch_size,),
                )
                affected = cur.rowcount or 0
                conn.commit()
                if affected == 0:
                    break

            while True:
                cur.execute(
                    f"""
                    WITH cte AS (
                        SELECT ctid
                        FROM {table}
                        WHERE example_id IS NULL
                        LIMIT %s
                    )
                    UPDATE {table} e
                    SET example_id = NULLIF(split_part(e.experiment_id, '::', 2), '')
                    FROM cte
                    WHERE e.ctid = cte.ctid;
                    """,
                    (batch_size,),
                )
                affected = cur.rowcount or 0
                conn.commit()
                if affected == 0:
                    break

            cur.execute(
                f"""
                UPDATE {table}
                SET example_id = 'unknown'
                WHERE example_id IS NULL;
                """
            )
            cur.execute(f"ALTER TABLE {table} ALTER COLUMN experiment_id SET NOT NULL;")
            cur.execute(f"ALTER TABLE {table} ALTER COLUMN example_id SET NOT NULL;")

            cur.execute(
                """
                SELECT con.conname,
                       array_agg(att.attname ORDER BY arr.n) AS cols
                FROM pg_constraint con
                JOIN pg_class rel ON rel.oid = con.conrelid
                JOIN pg_namespace nsp ON nsp.oid = rel.relnamespace
                JOIN LATERAL unnest(con.conkey) WITH ORDINALITY arr(attnum, n) ON TRUE
                JOIN pg_attribute att ON att.attrelid = rel.oid AND att.attnum = arr.attnum
                WHERE nsp.nspname = %s
                  AND rel.relname = 'experiment'
                  AND con.contype = 'p'
                GROUP BY con.conname;
                """,
                (schema,),
            )
            row = cur.fetchone()
            if row:
                pk_name, pk_cols = row
                if pk_cols != ["experiment_id"]:
                    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", pk_name):
                        raise RuntimeError(f"unexpected constraint name: {pk_name}")
                    cur.execute(f"ALTER TABLE {table} DROP CONSTRAINT {pk_name};")
                    cur.execute(f"ALTER TABLE {table} ADD CONSTRAINT experiment_pkey PRIMARY KEY (experiment_id);")
            else:
                cur.execute(f"ALTER TABLE {table} ADD CONSTRAINT experiment_pkey PRIMARY KEY (experiment_id);")

            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_experiment_doc ON {table}(doc_id);")
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_experiment_example ON {table}(example_id);")
            conn.commit()

    def upsert_document(self, doc: StructuredPatentDocument, sync_graph: bool = True) -> None:
        schema = self.config.schema
        with self._connect() as conn, conn.cursor() as cur:
            # 文档主记录
            cur.execute(
                f"""
                INSERT INTO {schema}.document(
                    doc_id, publication_number, title, abstract, source_file,
                    metadata, claim_tree, reference_numerals, structured_path, updated_at
                ) VALUES (
                    %(doc_id)s, %(publication_number)s, %(title)s, %(abstract)s, %(source_file)s,
                    %(metadata)s::jsonb, %(claim_tree)s::jsonb, %(reference_numerals)s::jsonb,
                    %(structured_path)s, NOW()
                )
                ON CONFLICT (doc_id) DO UPDATE SET
                    publication_number = EXCLUDED.publication_number,
                    title = EXCLUDED.title,
                    abstract = EXCLUDED.abstract,
                    source_file = EXCLUDED.source_file,
                    metadata = EXCLUDED.metadata,
                    claim_tree = EXCLUDED.claim_tree,
                    reference_numerals = EXCLUDED.reference_numerals,
                    structured_path = EXCLUDED.structured_path,
                    updated_at = NOW();
                """,
                {
                    "doc_id": doc.doc_id,
                    "publication_number": doc.publication_number,
                    "title": doc.title,
                    "abstract": doc.abstract,
                    "source_file": doc.source_file,
                    "metadata": json.dumps(doc.metadata, ensure_ascii=False),
                    "claim_tree": json.dumps(doc.claim_tree, ensure_ascii=False),
                    "reference_numerals": json.dumps(doc.reference_numerals, ensure_ascii=False)
                    if doc.reference_numerals is not None
                    else "null",
                    "structured_path": str(doc.path),
                },
            )

            # 先删旧明细再插新明细，保持幂等
            cur.execute(f"DELETE FROM {schema}.experiment WHERE doc_id = %s", (doc.doc_id,))
            cur.execute(f"DELETE FROM {schema}.relation WHERE doc_id = %s", (doc.doc_id,))
            cur.execute(f"DELETE FROM {schema}.entity WHERE doc_id = %s", (doc.doc_id,))
            cur.execute(f"DELETE FROM {schema}.block WHERE doc_id = %s", (doc.doc_id,))
            cur.execute(f"DELETE FROM {schema}.molecule WHERE doc_id = %s", (doc.doc_id,))
            cur.execute(f"DELETE FROM {schema}.material_doc_usage WHERE doc_id = %s", (doc.doc_id,))

            block_rows: list[dict] = []
            entity_rows: list[dict] = []
            relation_rows: list[dict] = []
            experiment_rows: list[dict] = []
            molecule_rows: list[tuple[str, str, str, str]] = []

            for block in doc.blocks:
                block_rows.append(
                    {
                        "block_id": block.block_id,
                        "doc_id": doc.doc_id,
                        "block_type": block.block_type,
                        "text": block.text,
                        "section": block.section,
                        "subsection": block.subsection,
                        "example_id": block.example_id,
                        "claim_no": block.claim_no,
                        "depends_on": json.dumps(block.depends_on, ensure_ascii=False)
                        if block.depends_on is not None
                        else "null",
                        "table_id": block.table_id,
                        "table_no": block.table_no,
                        "figure_id": block.figure_id,
                        "figure_no": block.figure_no,
                        "char_offset": block.char_offset,
                        "provenance": json.dumps(block.provenance, ensure_ascii=False),
                        "raw": json.dumps(block.raw, ensure_ascii=False),
                    }
                )

                for entity in block.entities:
                    raw_json = json.dumps(entity.raw, ensure_ascii=False)
                    smiles = self._extract_smiles(entity)
                    entity_rows.append(
                        {
                            "entity_id": entity.entity_id,
                            "doc_id": doc.doc_id,
                            "block_id": block.block_id,
                            "entity_type": entity.entity_type,
                            "value_text": entity.value,
                            "value_num": entity.value_num,
                            "unit": entity.unit,
                            "value_pair": entity.value_pair,
                            "span": entity.span,
                            "normalized": entity.normalized,
                            "canonical_id": entity.canonical_id,
                            "smiles": smiles,
                            "raw": raw_json,
                        }
                    )

                    if smiles:
                        molecule_rows.append((doc.doc_id, entity.entity_id, smiles, smiles))

                for relation in block.relations:
                    relation_rows.append(
                        {
                            "relation_id": relation.relation_id,
                            "doc_id": doc.doc_id,
                            "block_id": block.block_id,
                            "relation_type": relation.relation_type,
                            "source_entity_id": relation.source_entity_id,
                            "target_entity_id": relation.target_entity_id,
                            "confidence": relation.confidence,
                            "rule": relation.rule,
                            "distance": relation.distance,
                            "sentence_id": relation.sentence_id,
                            "raw": json.dumps(relation.raw, ensure_ascii=False),
                        }
                    )

            for experiment in doc.experiments:
                experiment_rows.append(
                    {
                        "experiment_id": f"{doc.doc_id}::{experiment.example_id}",
                        "example_id": experiment.example_id,
                        "doc_id": doc.doc_id,
                        "materials_used": experiment.materials_used,
                        "performance_relations": experiment.performance_relations,
                        "role_relations": experiment.role_relations,
                        "source_block_ids": experiment.source_block_ids,
                        "raw": json.dumps(experiment.raw, ensure_ascii=False),
                    }
                )

            if entity_rows:
                # 先对齐历史 alias 映射并写入 registry，再插入实体，确保 canonical_id 跨文档稳定。
                self._upsert_material_registry(cur, schema, doc.doc_id, entity_rows)

            if block_rows:
                cur.executemany(
                    f"""
                    INSERT INTO {schema}.block(
                        block_id, doc_id, block_type, text, section, subsection, example_id,
                        claim_no, depends_on, table_id, table_no, figure_id, figure_no,
                        char_offset, provenance, raw
                    ) VALUES (
                        %(block_id)s, %(doc_id)s, %(block_type)s, %(text)s, %(section)s,
                        %(subsection)s, %(example_id)s, %(claim_no)s, %(depends_on)s::jsonb,
                        %(table_id)s, %(table_no)s, %(figure_id)s, %(figure_no)s,
                        %(char_offset)s, %(provenance)s::jsonb, %(raw)s::jsonb
                    );
                    """,
                    block_rows,
                )
            if entity_rows:
                cur.executemany(
                    f"""
                    INSERT INTO {schema}.entity(
                        entity_id, doc_id, block_id, entity_type, value_text, value_num,
                        unit, value_pair, span, normalized, canonical_id, smiles, raw
                    ) VALUES (
                        %(entity_id)s, %(doc_id)s, %(block_id)s, %(entity_type)s, %(value_text)s,
                        %(value_num)s, %(unit)s, %(value_pair)s, %(span)s, %(normalized)s,
                        %(canonical_id)s, %(smiles)s, %(raw)s::jsonb
                    );
                    """,
                    entity_rows,
                )
            if relation_rows:
                cur.executemany(
                    f"""
                    INSERT INTO {schema}.relation(
                        relation_id, doc_id, block_id, relation_type, source_entity_id,
                        target_entity_id, confidence, rule, distance, sentence_id, raw
                    ) VALUES (
                        %(relation_id)s, %(doc_id)s, %(block_id)s, %(relation_type)s,
                        %(source_entity_id)s, %(target_entity_id)s, %(confidence)s,
                        %(rule)s, %(distance)s, %(sentence_id)s, %(raw)s::jsonb
                    );
                    """,
                    relation_rows,
                )
            if experiment_rows:
                cur.executemany(
                    f"""
                    INSERT INTO {schema}.experiment(
                        experiment_id, example_id, doc_id, materials_used, performance_relations,
                        role_relations, source_block_ids, raw
                    ) VALUES (
                        %(experiment_id)s, %(example_id)s, %(doc_id)s, %(materials_used)s, %(performance_relations)s,
                        %(role_relations)s, %(source_block_ids)s, %(raw)s::jsonb
                    )
                    ON CONFLICT (experiment_id) DO UPDATE SET
                        example_id = EXCLUDED.example_id,
                        doc_id = EXCLUDED.doc_id,
                        materials_used = EXCLUDED.materials_used,
                        performance_relations = EXCLUDED.performance_relations,
                        role_relations = EXCLUDED.role_relations,
                        source_block_ids = EXCLUDED.source_block_ids,
                        raw = EXCLUDED.raw;
                    """,
                    experiment_rows,
                )
            if molecule_rows:
                self._insert_molecules_resilient(cur, schema, molecule_rows)

            if sync_graph:
                self._sync_age_graph(cur, doc)

            conn.commit()

    def _sync_age_graph(self, cur, doc: StructuredPatentDocument) -> None:
        graph = self.config.age_graph

        # AGE 查询前确保 search_path
        cur.execute(f"SET search_path = ag_catalog, \"$user\", public, {self.config.schema};")

        # 幂等同步：
        # 1) 删除旧的文档及其关联节点（避免重复导入时脏数据）
        # 2) 再批量插入本次文档图
        cur.execute(
            """
            SELECT * FROM cypher(%s, $$
                MATCH (d:Document {doc_id: $doc_id})
                OPTIONAL MATCH (d)-[*1..3]->(n)
                WITH collect(DISTINCT d) + collect(DISTINCT n) AS nodes
                UNWIND nodes AS x
                WITH x WHERE x IS NOT NULL
                DETACH DELETE x
                RETURN 1
            $$, %s) AS (v agtype);
            """,
            (graph, json.dumps({"doc_id": doc.doc_id})),
        )

        # 不在热路径做全图孤儿扫描；孤儿清理交给离线维护任务。

        # 创建文档节点
        cur.execute(
            """
            SELECT * FROM cypher(%s, $$
                CREATE (d:Document {
                    doc_id: $doc_id,
                    publication_number: $publication_number,
                    title: $title
                })
                RETURN d
            $$, %s) AS (d agtype);
            """,
            (
                graph,
                json.dumps(
                    {
                        "doc_id": doc.doc_id,
                        "publication_number": doc.publication_number,
                        "title": doc.title,
                    },
                    ensure_ascii=False,
                ),
            ),
        )

        # 批量写 block/entity/relation
        for block in doc.blocks:
            cur.execute(
                """
                SELECT * FROM cypher(%s, $$
                    MATCH (d:Document {doc_id: $doc_id})
                    CREATE (b:Block {
                        block_id: $block_id,
                        block_type: $block_type,
                        text: $text,
                        section: $section,
                        subsection: $subsection,
                        example_id: $example_id
                    })
                    CREATE (d)-[:HAS_BLOCK]->(b)
                    RETURN b
                $$, %s) AS (b agtype);
                """,
                (
                    graph,
                    json.dumps(
                        {
                            "doc_id": doc.doc_id,
                            "block_id": block.block_id,
                            "block_type": block.block_type,
                            "text": block.text,
                            "section": block.section,
                            "subsection": block.subsection,
                            "example_id": block.example_id,
                        },
                        ensure_ascii=False,
                    ),
                ),
            )

            for entity in block.entities:
                cur.execute(
                    """
                    SELECT * FROM cypher(%s, $$
                        MATCH (b:Block {block_id: $block_id})
                        CREATE (e:Entity {
                            entity_id: $entity_id,
                            entity_type: $entity_type,
                            value_text: $value_text,
                            canonical_id: $canonical_id,
                            normalized: $normalized
                        })
                        CREATE (b)-[:HAS_ENTITY]->(e)
                        RETURN e
                    $$, %s) AS (e agtype);
                    """,
                    (
                        graph,
                        json.dumps(
                            {
                                "block_id": block.block_id,
                                "entity_id": entity.entity_id,
                                "entity_type": entity.entity_type,
                                "value_text": entity.value,
                                "canonical_id": entity.canonical_id,
                                "normalized": entity.normalized,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )

            for relation in block.relations:
                if not relation.source_entity_id or not relation.target_entity_id:
                    continue
                cur.execute(
                    """
                    SELECT * FROM cypher(%s, $$
                        MATCH (s:Entity {entity_id: $source_entity_id})
                        MATCH (t:Entity {entity_id: $target_entity_id})
                        CREATE (s)-[:REL {
                            relation_id: $relation_id,
                            relation_type: $relation_type,
                            confidence: $confidence,
                            rule: $rule
                        }]->(t)
                        RETURN s, t
                    $$, %s) AS (s agtype, t agtype);
                    """,
                    (
                        graph,
                        json.dumps(
                            {
                                "source_entity_id": relation.source_entity_id,
                                "target_entity_id": relation.target_entity_id,
                                "relation_id": relation.relation_id,
                                "relation_type": relation.relation_type,
                                "confidence": relation.confidence,
                                "rule": relation.rule,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )

    @staticmethod
    def _extract_smiles(entity) -> str | None:
        # 约定：优先显式前缀，其次尝试把可能是 SMILES 的短串作为候选。
        # 注意：canonical_id 可能是 registry id（如 mat:CBP / mat:RAW...），
        # 因此不能只看 canonical_id。
        candidates: list[str] = []
        if entity.canonical_id and isinstance(entity.canonical_id, str):
            candidates.append(entity.canonical_id)
        if entity.value and isinstance(entity.value, str):
            candidates.append(entity.value)
        if entity.normalized and isinstance(entity.normalized, str):
            candidates.append(entity.normalized)

        for raw in candidates:
            text = raw.strip()
            if not text:
                continue
            if text.startswith("SMILES:"):
                value = text.split("SMILES:", 1)[1].strip()
                if value:
                    return value

        # Heuristic：过滤明显非结构标记，尽量减少误判。
        allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@+-[]=#$()\\/%.")
        banned_tokens = {"PENDING_MAPPING", "UNKNOWN", "N/A", "NA"}
        likely_smiles_markers = set("[]=#@()/\\")
        simple_organic_chain = re.compile(r"^(?:Br|Cl|[BCNOPSFIbcnops]){2,}$")
        for raw in candidates:
            text = raw.strip()
            if not text or text in banned_tokens:
                continue
            if " " in text:
                continue
            if len(text) < 3 or len(text) > 512:
                continue
            if any(ch not in allowed_chars for ch in text):
                continue
            # OLED 专利场景以有机分子为主；缺碳的简写代号误判风险高。
            carbon_count = text.count("C") + text.count("c")
            if carbon_count < 2:
                continue
            if not any(ch.isalpha() for ch in text):
                continue
            # 至少包含一个更像化学结构表达的特征，避免普通词/编号被误判。
            if not any(ch in likely_smiles_markers for ch in text):
                # 允许更典型的芳香原子小写写法或环编号模式作为例外
                if not (
                    re.search(r"[bcnops]|[0-9].*[A-Za-z]|[A-Za-z].*[0-9]", text)
                    or simple_organic_chain.fullmatch(text)
                ):
                    continue
            return text

        return None

    @staticmethod
    def _insert_molecules_resilient(cur, schema: str, molecule_rows: list[tuple[str, str, str, str]]) -> None:
        if not molecule_rows:
            return

        # 优先走批量写入，显著降低 savepoint/子事务开销。
        cur.execute("SAVEPOINT mol_bulk_insert_sp;")
        try:
            cur.executemany(
                f"""
                INSERT INTO {schema}.molecule(doc_id, entity_id, smiles, mol)
                VALUES (%s, %s, %s, mol_from_smiles(%s));
                """,
                molecule_rows,
            )
            cur.execute("RELEASE SAVEPOINT mol_bulk_insert_sp;")
            return
        except Exception:
            # 任一条无效时回退到逐条容错路径。
            cur.execute("ROLLBACK TO SAVEPOINT mol_bulk_insert_sp;")
            cur.execute("RELEASE SAVEPOINT mol_bulk_insert_sp;")

        for row in molecule_rows:
            # 单条无效 SMILES 不应导致整篇文档回滚。
            cur.execute("SAVEPOINT mol_insert_sp;")
            try:
                cur.execute(
                    f"""
                    INSERT INTO {schema}.molecule(doc_id, entity_id, smiles, mol)
                    VALUES (%s, %s, %s, mol_from_smiles(%s));
                    """,
                    row,
                )
            except Exception:
                cur.execute("ROLLBACK TO SAVEPOINT mol_insert_sp;")
            finally:
                cur.execute("RELEASE SAVEPOINT mol_insert_sp;")

    def cleanup_age_orphans(self) -> None:
        """离线维护：清理 AGE 图中的孤儿节点（避免在写入热路径全图扫描）。"""
        graph = self.config.age_graph
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SET search_path = ag_catalog, \"$user\", public, {self.config.schema};")
            cur.execute(
                """
                SELECT * FROM cypher(%s, $$
                    MATCH (b:Block)
                    WHERE NOT (:Document)-[:HAS_BLOCK]->(b)
                    DETACH DELETE b
                    RETURN 1
                $$) AS (v agtype);
                """,
                (graph,),
            )
            cur.execute(
                """
                SELECT * FROM cypher(%s, $$
                    MATCH (e:Entity)
                    WHERE NOT (:Block)-[:HAS_ENTITY]->(e)
                    DETACH DELETE e
                    RETURN 1
                $$) AS (v agtype);
                """,
                (graph,),
            )
            conn.commit()

    def cleanup_material_registry_orphans(self) -> None:
        """离线维护：删除不再被任何文档使用的 canonical 节点。"""
        schema = self.config.schema
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                DELETE FROM {schema}.material_canonical mc
                WHERE NOT EXISTS (
                    SELECT 1 FROM {schema}.material_doc_usage u WHERE u.canonical_id = mc.canonical_id
                )
                AND NOT EXISTS (
                    SELECT 1 FROM {schema}.material_alias a WHERE a.canonical_id = mc.canonical_id
                );
                """
            )
            conn.commit()

    def rebuild_material_registry(self) -> dict:
        """从 entity(material) 全量重建跨文档 material registry。"""
        schema = self.config.schema
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT doc_id, entity_id, value_text, normalized, canonical_id
                FROM {schema}.entity
                WHERE entity_type = 'material'
                ORDER BY doc_id, entity_id
                """
            )
            rows = cur.fetchall()

            cur.execute(f"DELETE FROM {schema}.material_doc_usage;")
            cur.execute(f"DELETE FROM {schema}.material_alias;")
            cur.execute(f"DELETE FROM {schema}.material_canonical;")

            grouped: dict[str, list[dict]] = defaultdict(list)
            for doc_id, entity_id, value_text, normalized, canonical_id in rows:
                grouped[str(doc_id)].append(
                    {
                        "entity_id": str(entity_id),
                        "entity_type": "material",
                        "value_text": value_text,
                        "normalized": normalized,
                        "canonical_id": canonical_id,
                    }
                )

            for doc_id, entity_rows in grouped.items():
                self._upsert_material_registry(cur, schema, doc_id, entity_rows)

            conn.commit()
            return {
                "documents": len(grouped),
                "material_entities": len(rows),
            }

    def rdkit_substructure_search(self, smarts: str, limit: int = 20) -> list[dict]:
        schema = self.config.schema
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT m.doc_id, m.entity_id, m.smiles
                FROM {schema}.molecule m
                WHERE m.mol @> qmol_from_smarts(%s)
                LIMIT %s
                """,
                (smarts, limit),
            )
            rows = cur.fetchall()
        return [
            {"doc_id": r[0], "entity_id": r[1], "smiles": r[2]}
            for r in rows
        ]

    def rdkit_similarity_search(self, smiles: str, threshold: float = 0.6, limit: int = 20) -> list[dict]:
        schema = self.config.schema
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT m.doc_id, m.entity_id, m.smiles,
                       tanimoto_sml(morganbv_fp(mol_from_smiles(%s), 2), morganbv_fp(m.mol, 2)) AS score
                FROM {schema}.molecule m
                WHERE tanimoto_sml(morganbv_fp(mol_from_smiles(%s), 2), morganbv_fp(m.mol, 2)) >= %s
                ORDER BY score DESC
                LIMIT %s
                """,
                (smiles, smiles, threshold, limit),
            )
            rows = cur.fetchall()
        return [
            {"doc_id": r[0], "entity_id": r[1], "smiles": r[2], "score": float(r[3])}
            for r in rows
        ]

    def healthcheck(self) -> dict:
        schema = self.config.schema
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.execute(
                "SELECT extname FROM pg_extension WHERE extname IN ('rdkit', 'age') ORDER BY extname"
            )
            exts = [row[0] for row in cur.fetchall()]
            doc_count = 0
            try:
                cur.execute(f"SELECT COUNT(*) FROM {schema}.document")
                doc_count = int(cur.fetchone()[0])
            except Exception:
                doc_count = 0
            experiment_pk_cols: list[str] = []
            migration_needed = False
            registry = {
                "canonical_count": 0,
                "alias_count": 0,
                "doc_usage_count": 0,
            }
            try:
                cur.execute(
                    """
                    SELECT array_agg(att.attname ORDER BY arr.n) AS cols
                    FROM pg_constraint con
                    JOIN pg_class rel ON rel.oid = con.conrelid
                    JOIN pg_namespace nsp ON nsp.oid = rel.relnamespace
                    JOIN LATERAL unnest(con.conkey) WITH ORDINALITY arr(attnum, n) ON TRUE
                    JOIN pg_attribute att ON att.attrelid = rel.oid AND att.attnum = arr.attnum
                    WHERE nsp.nspname = %s
                      AND rel.relname = 'experiment'
                      AND con.contype = 'p'
                    GROUP BY con.conname;
                    """,
                    (schema,),
                )
                row = cur.fetchone()
                if row and row[0]:
                    experiment_pk_cols = list(row[0])
                migration_needed = experiment_pk_cols != ["experiment_id"]
            except Exception:
                migration_needed = True
            try:
                cur.execute(f"SELECT COUNT(*) FROM {schema}.material_canonical")
                registry["canonical_count"] = int(cur.fetchone()[0])
                cur.execute(f"SELECT COUNT(*) FROM {schema}.material_alias")
                registry["alias_count"] = int(cur.fetchone()[0])
                cur.execute(f"SELECT COUNT(*) FROM {schema}.material_doc_usage")
                registry["doc_usage_count"] = int(cur.fetchone()[0])
            except Exception:
                pass
        return {
            "ok": True,
            "schema": schema,
            "extensions": exts,
            "document_count": doc_count,
            "experiment_pk_columns": experiment_pk_cols,
            "experiment_pk_migration_needed": migration_needed,
            "material_registry": registry,
        }
