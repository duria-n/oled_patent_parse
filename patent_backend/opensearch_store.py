"""OpenSearch 索引、批量写入与 hybrid 检索。"""

from __future__ import annotations

from typing import Any, Iterable

from .config import OpenSearchConfig
from .embedding import BaseEmbedder
from .models import StructuredPatentDocument


class OpenSearchPatentIndex:
    def __init__(self, config: OpenSearchConfig, embedder: BaseEmbedder):
        try:
            from opensearchpy import OpenSearch
            from opensearchpy.helpers import bulk
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("请先安装 opensearch-py: pip install opensearch-py") from exc

        self.config = config
        self.embedder = embedder
        self._bulk = bulk

        auth = None
        if config.username:
            auth = (config.username, config.password or "")

        self.client = OpenSearch(
            hosts=config.hosts,
            http_auth=auth,
            verify_certs=config.verify_certs,
            timeout=config.timeout_seconds,
        )

    def ensure_indices(self) -> None:
        doc_index = self.config.index_name
        block_index = self.config.block_index_name
        dim = self.embedder.dimension

        doc_mapping = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "patent_multilang": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "cjk_width"],
                        }
                    }
                },
                "index": {
                    "knn": True,
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                }
            },
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "publication_number": {"type": "keyword"},
                    "title": {"type": "text", "analyzer": "patent_multilang"},
                    "abstract": {"type": "text", "analyzer": "patent_multilang"},
                    "metadata": {"type": "object", "enabled": True},
                    "entity_types": {"type": "keyword"},
                    "relation_types": {"type": "keyword"},
                    "full_text": {"type": "text", "analyzer": "patent_multilang"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                        },
                    },
                }
            },
        }

        block_mapping = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "patent_multilang": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "cjk_width"],
                        }
                    }
                },
                "index": {
                    "knn": True,
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                }
            },
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "block_id": {"type": "keyword"},
                    "block_type": {"type": "keyword"},
                    "section": {"type": "keyword"},
                    "subsection": {"type": "keyword"},
                    "example_id": {"type": "keyword"},
                    "text": {"type": "text", "analyzer": "patent_multilang"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                        },
                    },
                }
            },
        }

        if not self.client.indices.exists(index=doc_index):
            self.client.indices.create(index=doc_index, body=doc_mapping)

        if not self.client.indices.exists(index=block_index):
            self.client.indices.create(index=block_index, body=block_mapping)

    def _iter_doc_actions(self, docs: list[StructuredPatentDocument]) -> Iterable[dict[str, Any]]:
        for doc in docs:
            full_text = doc.text_for_embedding()
            embedding = self.embedder.embed(full_text)
            entity_types = sorted({entity.entity_type for _, entity in doc.iter_entities()})
            relation_types = sorted({rel.relation_type for _, rel in doc.iter_relations()})

            yield {
                "_op_type": "index",
                "_index": self.config.index_name,
                "_id": doc.doc_id,
                "_source": {
                    "doc_id": doc.doc_id,
                    "publication_number": doc.publication_number,
                    "title": doc.title,
                    "abstract": doc.abstract,
                    "metadata": doc.metadata,
                    "entity_types": entity_types,
                    "relation_types": relation_types,
                    "full_text": full_text,
                    "embedding": embedding,
                },
            }

    def _iter_block_actions(self, docs: list[StructuredPatentDocument]) -> Iterable[dict[str, Any]]:
        for doc in docs:
            for block in doc.blocks:
                text = block.text or ""
                if not text.strip():
                    continue
                yield {
                    "_op_type": "index",
                    "_index": self.config.block_index_name,
                    "_id": block.block_id,
                    "_source": {
                        "doc_id": doc.doc_id,
                        "block_id": block.block_id,
                        "block_type": block.block_type,
                        "section": block.section,
                        "subsection": block.subsection,
                        "example_id": block.example_id,
                        "text": text,
                        "embedding": self.embedder.embed(text),
                    },
                }

    def _bulk_if_has_actions(self, actions: Iterable[dict[str, Any]], chunk_size: int) -> None:
        action_iter = iter(actions)
        try:
            first = next(action_iter)
        except StopIteration:
            return

        def _with_first() -> Iterable[dict[str, Any]]:
            yield first
            yield from action_iter

        self._bulk(self.client, _with_first(), chunk_size=chunk_size, refresh="wait_for")

    def index_documents(self, docs: list[StructuredPatentDocument], chunk_size: int = 200) -> None:
        self._bulk_if_has_actions(
            self._iter_doc_actions(docs),
            chunk_size=chunk_size,
        )
        self._bulk_if_has_actions(
            self._iter_block_actions(docs),
            chunk_size=chunk_size,
        )

    def hybrid_search_documents(
        self,
        query: str,
        top_k: int = 10,
        keyword_boost: float = 0.5,
        vector_boost: float = 0.5,
    ) -> dict[str, Any]:
        query_vec = self.embedder.embed(query)

        body = {
            "size": top_k,
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "bool": {
                                "should": [
                                    {"match": {"title": {"query": query, "boost": keyword_boost * 2}}},
                                    {"match": {"abstract": {"query": query, "boost": keyword_boost}}},
                                    {"match": {"full_text": {"query": query, "boost": keyword_boost}}},
                                ],
                                "minimum_should_match": 1,
                            }
                        },
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_vec,
                                    "k": top_k,
                                    "boost": vector_boost,
                                }
                            }
                        },
                    ]
                }
            },
        }
        return self.client.search(index=self.config.index_name, body=body)

    def hybrid_search_blocks(
        self,
        query: str,
        top_k: int = 10,
        keyword_boost: float = 0.5,
        vector_boost: float = 0.5,
        doc_id: str | None = None,
    ) -> dict[str, Any]:
        query_vec = self.embedder.embed(query)

        keyword_query: dict[str, Any] = {
            "bool": {
                "should": [
                    {"match": {"text": {"query": query, "boost": keyword_boost}}},
                    {"term": {"block_type": {"value": query, "boost": keyword_boost * 0.5}}},
                ],
                "minimum_should_match": 1,
            }
        }
        filter_clause = [{"term": {"doc_id": doc_id}}] if doc_id else []

        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "filter": filter_clause,
                    "must": [
                        {
                            "hybrid": {
                                "queries": [
                                    keyword_query,
                                    {
                                        "knn": {
                                            "embedding": {
                                                "vector": query_vec,
                                                "k": top_k,
                                                "boost": vector_boost,
                                            }
                                        }
                                    },
                                ]
                            }
                        }
                    ]
                }
            },
        }

        return self.client.search(index=self.config.block_index_name, body=body)

    def healthcheck(self) -> dict:
        info = self.client.info()
        return {
            "ok": True,
            "cluster_name": info.get("cluster_name"),
            "version": (info.get("version") or {}).get("number"),
            "doc_index_exists": self.client.indices.exists(index=self.config.index_name),
            "block_index_exists": self.client.indices.exists(index=self.config.block_index_name),
        }
