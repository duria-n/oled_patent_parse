"""专利写作支撑系统后端：PostgreSQL + OpenSearch + RDKit + AGE。"""

from .config import OpenSearchConfig, PostgresConfig
from .embedding import BaseEmbedder, HashEmbedder, OpenAIEmbedder, build_embedder
from .loader import discover_structured_files, load_structured_documents
from .models import StructuredPatentDocument
from .opensearch_store import OpenSearchPatentIndex
from .postgres_store import PostgresPatentStore

__all__ = [
    "PostgresConfig",
    "OpenSearchConfig",
    "BaseEmbedder",
    "HashEmbedder",
    "OpenAIEmbedder",
    "build_embedder",
    "discover_structured_files",
    "load_structured_documents",
    "StructuredPatentDocument",
    "OpenSearchPatentIndex",
    "PostgresPatentStore",
]
