"""专利后端管道 CLI：PostgreSQL + RDKit + AGE + OpenSearch。"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from patent_backend import (
    OpenSearchConfig,
    OpenSearchPatentIndex,
    PostgresConfig,
    PostgresPatentStore,
    build_embedder,
    discover_structured_files,
    load_structured_documents,
)

logger = logging.getLogger("backend_pipeline")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="专利结构化数据后端导入/检索 CLI")

    ap.add_argument(
        "--inputs",
        nargs="+",
        default=["md", "output"],
        help="structured json 输入路径（文件或目录，可多个）",
    )
    ap.add_argument("--strict", action="store_true", help="遇到坏文件直接报错")

    ap.add_argument("--init-pg", action="store_true", help="初始化 PostgreSQL schema（含 RDKit/AGE）")
    ap.add_argument("--migrate-experiment-pk", action="store_true", help="迁移 experiment 表主键到 experiment_id")
    ap.add_argument("--migrate-batch-size", type=int, default=50000, help="迁移批次大小")
    ap.add_argument("--ingest-pg", action="store_true", help="导入 PostgreSQL 关系层")
    ap.add_argument("--sync-age", action="store_true", help="导入 PG 时同步 AGE 图")
    ap.add_argument("--age-clean-orphans", action="store_true", help="离线清理 AGE 图孤儿节点")
    ap.add_argument("--rebuild-material-registry", action="store_true", help="从 entity(material) 全量重建跨文档 material registry")
    ap.add_argument("--material-clean-orphans", action="store_true", help="清理不再被文档使用的 material canonical 节点")

    ap.add_argument("--init-os", action="store_true", help="初始化 OpenSearch 索引")
    ap.add_argument("--index-os", action="store_true", help="写入 OpenSearch 索引")

    ap.add_argument("--embedder", default="auto", choices=["auto", "hash", "openai"], help="向量模型")
    ap.add_argument("--embed-dim", type=int, default=384, help="hash 向量维度")
    ap.add_argument("--openai-embed-model", default="text-embedding-3-small", help="OpenAI 向量模型")

    ap.add_argument("--search-doc", default=None, help="执行文档级 hybrid 查询")
    ap.add_argument("--search-block", default=None, help="执行段落级 hybrid 查询")
    ap.add_argument("--search-doc-id", default=None, help="段落检索时按 doc_id 过滤")
    ap.add_argument("--top-k", type=int, default=10, help="检索 top k")

    ap.add_argument("--rdkit-substruct", default=None, help="执行 SMARTS 子结构检索")
    ap.add_argument("--rdkit-sim", default=None, help="执行 SMILES 相似度检索")
    ap.add_argument("--rdkit-threshold", type=float, default=0.6, help="RDKit 相似度阈值")
    ap.add_argument("--healthcheck", action="store_true", help="检查 PG/OS 连通性与扩展状态")
    ap.add_argument("--check-data", action="store_true", help="检查输入 JSON 的可导入质量摘要")

    ap.add_argument("--print-doc-limit", type=int, default=5, help="打印前 N 条导入文档")
    return ap


def _need_load_docs(args: argparse.Namespace) -> bool:
    return any(
        [
            args.ingest_pg,
            args.index_os,
            args.check_data,
        ]
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ap = _build_parser()
    args = ap.parse_args()

    if not any(
        [
            args.init_pg,
            args.migrate_experiment_pk,
            args.ingest_pg,
            args.age_clean_orphans,
            args.rebuild_material_registry,
            args.material_clean_orphans,
            args.init_os,
            args.index_os,
            args.search_doc,
            args.search_block,
            args.rdkit_substruct,
            args.rdkit_sim,
            args.healthcheck,
            args.check_data,
        ]
    ):
        ap.error("至少指定一个动作，例如 --init-pg / --ingest-pg / --index-os / --search-doc")

    pg_store: PostgresPatentStore | None = None
    os_index: OpenSearchPatentIndex | None = None
    pg_init_error: str | None = None
    os_init_error: str | None = None

    if (
        args.init_pg
        or args.migrate_experiment_pk
        or args.ingest_pg
        or args.age_clean_orphans
        or args.rebuild_material_registry
        or args.material_clean_orphans
        or args.rdkit_substruct
        or args.rdkit_sim
        or args.healthcheck
    ):
        try:
            pg_cfg = PostgresConfig.from_env()
            pg_store = PostgresPatentStore(pg_cfg)
            logger.info("PostgreSQL: schema=%s graph=%s", pg_cfg.schema, pg_cfg.age_graph)
        except Exception as exc:
            pg_init_error = str(exc)
            if not args.healthcheck:
                raise

    if args.init_os or args.index_os or args.search_doc or args.search_block or args.healthcheck:
        # healthcheck 仅用于连通性检查，不依赖真实语义向量模型。
        embedder_kind = args.embedder
        if args.healthcheck and not (args.init_os or args.index_os or args.search_doc or args.search_block):
            embedder_kind = "hash"
        try:
            embedder = build_embedder(
                embedder_kind,
                dimension=args.embed_dim,
                openai_model=args.openai_embed_model,
            )
            os_cfg = OpenSearchConfig.from_env()
            os_index = OpenSearchPatentIndex(os_cfg, embedder=embedder)
            logger.info("OpenSearch: index=%s block_index=%s", os_cfg.index_name, os_cfg.block_index_name)
        except Exception as exc:
            os_init_error = str(exc)
            if not args.healthcheck:
                raise

    docs = []
    if _need_load_docs(args):
        files = discover_structured_files(args.inputs)
        if not files:
            logger.warning("未发现 structured json，inputs=%s", args.inputs)
        else:
            logger.info("发现 structured 文件: %d", len(files))
        docs = list(load_structured_documents(files, strict=args.strict))
        logger.info("成功加载文档: %d", len(docs))
        for i, doc in enumerate(docs[: args.print_doc_limit], 1):
            logger.info("[%d] doc_id=%s title=%s blocks=%d", i, doc.doc_id, doc.title, len(doc.blocks))

    if args.check_data:
        total_blocks = 0
        total_entities = 0
        total_relations = 0
        candidate_smiles = 0
        for doc in docs:
            total_blocks += len(doc.blocks)
            for block in doc.blocks:
                total_entities += len(block.entities)
                total_relations += len(block.relations)
                for entity in block.entities:
                    for field in (entity.canonical_id, entity.value, entity.normalized):
                        if not isinstance(field, str):
                            continue
                        field = field.strip()
                        if not field:
                            continue
                        if field == "PENDING_MAPPING" or field.startswith("mat:"):
                            continue
                        if field.startswith("SMILES:") or (" " not in field and any(ch.isalpha() for ch in field)):
                            candidate_smiles += 1
                            break
        print(
            json.dumps(
                {
                    "documents": len(docs),
                    "blocks": total_blocks,
                    "entities": total_entities,
                    "relations": total_relations,
                    "smiles_candidates": candidate_smiles,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    if args.init_pg:
        assert pg_store is not None
        pg_store.init_schema()
        logger.info("PostgreSQL schema 初始化完成")

    if args.migrate_experiment_pk:
        assert pg_store is not None
        pg_store.migrate_experiment_primary_key(batch_size=args.migrate_batch_size)
        logger.info("experiment 主键迁移完成")

    if args.ingest_pg:
        assert pg_store is not None
        if not docs:
            logger.warning("没有可导入 PostgreSQL 的文档")
        for i, doc in enumerate(docs, 1):
            pg_store.upsert_document(doc, sync_graph=args.sync_age)
            logger.info("PG upsert [%d/%d] %s", i, len(docs), doc.doc_id)

    if args.age_clean_orphans:
        assert pg_store is not None
        pg_store.cleanup_age_orphans()
        logger.info("AGE 孤儿节点清理完成")

    if args.rebuild_material_registry:
        assert pg_store is not None
        result = pg_store.rebuild_material_registry()
        logger.info(
            "material registry 重建完成: docs=%d material_entities=%d",
            result.get("documents", 0),
            result.get("material_entities", 0),
        )

    if args.material_clean_orphans:
        assert pg_store is not None
        pg_store.cleanup_material_registry_orphans()
        logger.info("material registry 孤儿 canonical 清理完成")

    if args.init_os:
        assert os_index is not None
        os_index.ensure_indices()
        logger.info("OpenSearch 索引初始化完成")

    if args.index_os:
        assert os_index is not None
        if not docs:
            logger.warning("没有可写入 OpenSearch 的文档")
        else:
            os_index.ensure_indices()
            os_index.index_documents(docs)
            logger.info("OpenSearch 索引写入完成: %d 文档", len(docs))

    if args.healthcheck:
        result: dict[str, object] = {"ok": True}
        if pg_init_error:
            result["ok"] = False
            result["postgres"] = {"ok": False, "error": pg_init_error}
        if pg_store is not None:
            try:
                result["postgres"] = pg_store.healthcheck()
            except Exception as exc:
                result["ok"] = False
                result["postgres"] = {"ok": False, "error": str(exc)}
        if os_init_error:
            result["ok"] = False
            result["opensearch"] = {"ok": False, "error": os_init_error}
        if os_index is not None:
            try:
                result["opensearch"] = os_index.healthcheck()
            except Exception as exc:
                result["ok"] = False
                result["opensearch"] = {"ok": False, "error": str(exc)}
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.search_doc:
        assert os_index is not None
        result = os_index.hybrid_search_documents(args.search_doc, top_k=args.top_k)
        hits = (result.get("hits") or {}).get("hits") or []
        logger.info("文档检索命中: %d", len(hits))
        print(json.dumps(hits, ensure_ascii=False, indent=2))

    if args.search_block:
        assert os_index is not None
        result = os_index.hybrid_search_blocks(
            args.search_block,
            top_k=args.top_k,
            doc_id=args.search_doc_id,
        )
        hits = (result.get("hits") or {}).get("hits") or []
        logger.info("段落检索命中: %d", len(hits))
        print(json.dumps(hits, ensure_ascii=False, indent=2))

    if args.rdkit_substruct:
        assert pg_store is not None
        hits = pg_store.rdkit_substructure_search(args.rdkit_substruct, limit=args.top_k)
        logger.info("RDKit 子结构命中: %d", len(hits))
        print(json.dumps(hits, ensure_ascii=False, indent=2))

    if args.rdkit_sim:
        assert pg_store is not None
        hits = pg_store.rdkit_similarity_search(
            args.rdkit_sim,
            threshold=args.rdkit_threshold,
            limit=args.top_k,
        )
        logger.info("RDKit 相似检索命中: %d", len(hits))
        print(json.dumps(hits, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
