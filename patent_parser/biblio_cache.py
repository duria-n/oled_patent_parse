"""本地题录元数据缓存读取。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BiblioMetadata:
    publication_number: str | None = None
    publication_date: str | None = None
    application_number: str | None = None
    application_date: str | None = None
    priority: list[str] | None = None
    title: str | None = None
    applicants: list[str] | None = None
    inventors: list[str] | None = None
    ipc: list[str] | None = None
    cpc: list[str] | None = None
    source: str | None = None


class BiblioMetadataProvider:
    def __init__(self, cache_path: str | Path | None):
        self.cache_path = Path(cache_path).resolve() if cache_path else None
        self._cache = self._load_cache()

    def _load_cache(self) -> dict[str, dict]:
        if not self.cache_path or not self.cache_path.exists():
            return {}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def lookup(self, key: str) -> BiblioMetadata | None:
        raw = self._cache.get(key)
        if not raw:
            raw = self._cache.get(key.upper()) or self._cache.get(key.lower())
        if not isinstance(raw, dict):
            return None
        meta = raw.get("metadata", raw)
        return BiblioMetadata(
            publication_number=meta.get("publication_number"),
            publication_date=meta.get("publication_date"),
            application_number=meta.get("application_number"),
            application_date=meta.get("application_date"),
            priority=meta.get("priority"),
            title=meta.get("title"),
            applicants=meta.get("applicants"),
            inventors=meta.get("inventors"),
            ipc=meta.get("ipc"),
            cpc=meta.get("cpc"),
            source=meta.get("source"),
        )
