# 专利写作支撑后端（PostgreSQL + OpenSearch + RDKit + Apache AGE）

本仓库新增了 `patent_backend/` 与 `backend_pipeline.py`，用于把 `*_structured.json` 导入并提供四类能力：

- PostgreSQL：关系化主数据（doc/block/entity/relation/experiment）
- RDKit cartridge：结构化学检索（子结构、相似度）
- Apache AGE：在 PostgreSQL 上直接进行图查询（openCypher + SQL）
- OpenSearch：关键词 + 语义向量混合检索（hybrid search）

## 1. 安装依赖

```bash
pip install "psycopg[binary]" opensearch-py
# 如果要使用 OpenAI 向量
pip install openai
```

## 2. 配置环境变量

```bash
# PostgreSQL
export PATENT_PG_DSN='postgresql://postgres:postgres@localhost:5432/patent'
export PATENT_PG_SCHEMA='patent'
export PATENT_AGE_GRAPH='patent_graph'

# OpenSearch
export PATENT_OS_HOSTS='http://localhost:9200'
export PATENT_OS_INDEX='patent_docs'
export PATENT_OS_BLOCK_INDEX='patent_blocks'
# 如果有鉴权
export PATENT_OS_USERNAME='admin'
export PATENT_OS_PASSWORD='admin'

# OpenAI（可选）
export OPENAI_API_KEY='sk-...'
```

## 3. 从当前解析结果导入

默认会扫描 `md`、`output` 下的 `*_structured.json`。

```bash
# 初始化 PostgreSQL（含 rdkit/age extension + schema）
python backend_pipeline.py --init-pg

# 导入 PostgreSQL，并同步 AGE 图
python backend_pipeline.py --ingest-pg --sync-age --inputs md output

# （建议定期离线执行）清理 AGE 图孤儿节点
python backend_pipeline.py --age-clean-orphans

# 初始化 OpenSearch 索引（默认 embedder=auto：有 OPENAI_API_KEY 就用 OpenAI，否则回退 hash）
python backend_pipeline.py --init-os --embedder auto --embed-dim 384

# 导入 OpenSearch 索引
python backend_pipeline.py --index-os --embedder auto --embed-dim 384 --inputs md output
```

## 4. 检索示例

### 4.1 OpenSearch hybrid（关键词+语义）

```bash
# 文档级
python backend_pipeline.py --search-doc "蓝光OLED寿命提升" --top-k 10 --embedder auto

# 段落级（可选限定某个 doc_id）
python backend_pipeline.py --search-block "host dopant eqe" --search-doc-id WO2023123456A1 --top-k 10 --embedder auto
```

### 4.2 RDKit 检索

```bash
# 子结构（SMARTS）
python backend_pipeline.py --rdkit-substruct "c1ccccc1" --top-k 20

# 相似度（SMILES）
python backend_pipeline.py --rdkit-sim "CC1=CC=CC=C1" --rdkit-threshold 0.6 --top-k 20
```

## 5. AGE 图查询示例（在 PostgreSQL 中执行）

```sql
LOAD 'age';
SET search_path = ag_catalog, "$user", public, patent;

SELECT * FROM cypher('patent_graph', $$
  MATCH (d:Document)-[:HAS_BLOCK]->(b:Block)-[:HAS_ENTITY]->(e:Entity)
  WHERE e.entity_type = 'material'
  RETURN d.doc_id, b.block_id, e.entity_id, e.value_text
  LIMIT 20
$$) AS (doc_id agtype, block_id agtype, entity_id agtype, value_text agtype);
```

## 6. 说明

- `SMILES` 的入库规则：
  - 优先 `entity.canonical_id` / `entity.value` / `entity.normalized` 的 `SMILES:...` 前缀
  - 若无前缀，使用保守启发式识别潜在 SMILES（自动过滤 `PENDING_MAPPING` 等占位值）
- 如需更高质量语义检索，建议使用：
  - `--embedder openai --openai-embed-model text-embedding-3-small`
- 当前实现采用“文档级幂等 upsert + 明细重建”，适合离线批处理与增量重跑。

## 7. 健康检查与数据检查

```bash
# 连接、扩展与索引存在性检查
python backend_pipeline.py --healthcheck --init-pg --init-os

# 对输入结构化 JSON 做可导入质量摘要
python backend_pipeline.py --check-data --inputs md output
```
