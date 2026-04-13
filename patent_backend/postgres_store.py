"""PostgreSQL 主数据存储 + RDKit + Apache AGE 图同步。"""

from __future__ import annotations

import json
from contextlib import contextmanager

from .config import PostgresConfig
from .models import StructuredPatentDocument


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

        -- 兼容旧版本（example_id 作为主键）迁移到 experiment_id。
        ALTER TABLE {schema}.experiment ADD COLUMN IF NOT EXISTS experiment_id TEXT;
        ALTER TABLE {schema}.experiment ADD COLUMN IF NOT EXISTS example_id TEXT;
        UPDATE {schema}.experiment
        SET experiment_id = COALESCE(experiment_id, doc_id || '::' || COALESCE(example_id, 'unknown'))
        WHERE experiment_id IS NULL;
        UPDATE {schema}.experiment
        SET example_id = COALESCE(example_id, split_part(experiment_id, '::', 2))
        WHERE example_id IS NULL;
        ALTER TABLE {schema}.experiment ALTER COLUMN experiment_id SET NOT NULL;
        ALTER TABLE {schema}.experiment ALTER COLUMN example_id SET NOT NULL;
        DO $$
        DECLARE c RECORD;
        BEGIN
            FOR c IN
                SELECT conname
                FROM pg_constraint
                WHERE conrelid = '{schema}.experiment'::regclass
                  AND contype = 'p'
            LOOP
                EXECUTE 'ALTER TABLE {schema}.experiment DROP CONSTRAINT ' || quote_ident(c.conname);
            END LOOP;
            ALTER TABLE {schema}.experiment
            ADD CONSTRAINT experiment_pkey PRIMARY KEY (experiment_id);
        END
        $$;

        CREATE TABLE IF NOT EXISTS {schema}.molecule (
            molecule_id BIGSERIAL PRIMARY KEY,
            doc_id TEXT NOT NULL REFERENCES {schema}.document(doc_id) ON DELETE CASCADE,
            entity_id TEXT REFERENCES {schema}.entity(entity_id) ON DELETE SET NULL,
            smiles TEXT NOT NULL,
            mol mol,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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
        """

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(ddl)
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

            for block in doc.blocks:
                cur.execute(
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
                    },
                )

                for entity in block.entities:
                    raw_json = json.dumps(entity.raw, ensure_ascii=False)
                    smiles = self._extract_smiles(entity)
                    cur.execute(
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
                        },
                    )

                    if smiles:
                        # 单条无效 SMILES 不应导致整篇文档回滚。
                        cur.execute("SAVEPOINT mol_insert_sp;")
                        try:
                            cur.execute(
                                f"""
                                INSERT INTO {schema}.molecule(doc_id, entity_id, smiles, mol)
                                VALUES (%s, %s, %s, mol_from_smiles(%s));
                                """,
                                (doc.doc_id, entity.entity_id, smiles, smiles),
                            )
                        except Exception:
                            cur.execute("ROLLBACK TO SAVEPOINT mol_insert_sp;")
                        finally:
                            cur.execute("RELEASE SAVEPOINT mol_insert_sp;")

                for relation in block.relations:
                    cur.execute(
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
                        },
                    )

            for experiment in doc.experiments:
                cur.execute(
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
                    {
                        "experiment_id": f"{doc.doc_id}::{experiment.example_id}",
                        "example_id": experiment.example_id,
                        "doc_id": doc.doc_id,
                        "materials_used": experiment.materials_used,
                        "performance_relations": experiment.performance_relations,
                        "role_relations": experiment.role_relations,
                        "source_block_ids": experiment.source_block_ids,
                        "raw": json.dumps(experiment.raw, ensure_ascii=False),
                    },
                )

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

        # 兜底清理：把所有孤儿 Block/Entity 清掉，防止历史脏数据累积
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
        # 注意：postprocess 默认 material.canonical_id=PENDING_MAPPING，
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

        # Heuristic：过滤明显非结构标记，允许常见 SMILES 字符集。
        allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@+-[]=#$()\\/%.")
        banned_tokens = {"PENDING_MAPPING", "UNKNOWN", "N/A", "NA"}
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
            if not any(ch.isalpha() for ch in text):
                continue
            return text

        return None

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
        return {
            "ok": True,
            "schema": schema,
            "extensions": exts,
            "document_count": doc_count,
        }
