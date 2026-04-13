"""后端连接配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _read_bool_env(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(slots=True)
class PostgresConfig:
    dsn: str = "postgresql://postgres:postgres@localhost:5432/patent"
    schema: str = "patent"
    age_graph: str = "patent_graph"

    @classmethod
    def from_env(cls) -> "PostgresConfig":
        return cls(
            dsn=os.environ.get("PATENT_PG_DSN", cls.dsn),
            schema=os.environ.get("PATENT_PG_SCHEMA", cls.schema),
            age_graph=os.environ.get("PATENT_AGE_GRAPH", cls.age_graph),
        )


@dataclass(slots=True)
class OpenSearchConfig:
    hosts: list[str]
    index_name: str = "patent_docs"
    block_index_name: str = "patent_blocks"
    username: str | None = None
    password: str | None = None
    verify_certs: bool = True
    timeout_seconds: int = 30

    @classmethod
    def from_env(cls) -> "OpenSearchConfig":
        hosts_raw = os.environ.get("PATENT_OS_HOSTS", "http://localhost:9200")
        hosts = [h.strip() for h in hosts_raw.split(",") if h.strip()]
        return cls(
            hosts=hosts,
            index_name=os.environ.get("PATENT_OS_INDEX", cls.index_name),
            block_index_name=os.environ.get("PATENT_OS_BLOCK_INDEX", cls.block_index_name),
            username=os.environ.get("PATENT_OS_USERNAME") or None,
            password=os.environ.get("PATENT_OS_PASSWORD") or None,
            verify_certs=_read_bool_env("PATENT_OS_VERIFY_CERTS", True),
            timeout_seconds=int(os.environ.get("PATENT_OS_TIMEOUT", "30")),
        )
