"""WIPO / PATENTSCOPE 元数据读取（本地缓存优先）。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WIPOMetadata:
    publication_language: str | None = None
    filing_language: str | None = None
    source: str | None = None


_WO_RE = re.compile(r"(WO\s*[-/]?\s*)?(\d{4})(\d{4,7})", re.I)


def normalize_wo_pubno(text: str) -> str | None:
    """将任意 WO 公开号格式标准化为 WOYYYYNNNNN..."""
    m = _WO_RE.search(text)
    if not m:
        return None
    year = m.group(2)
    serial = m.group(3)
    return f"WO{year}{serial}"


def _normalize_lang_code(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip().lower()
    if not v:
        return None
    # 常见语言全称
    name_map = {
        "english": "en",
        "french": "fr",
        "german": "de",
        "spanish": "es",
        "japanese": "ja",
        "korean": "ko",
        "chinese": "zh",
        "russian": "ru",
        "arabic": "ar",
        "portuguese": "pt",
        "italian": "it",
    }
    return name_map.get(v, v)


class WIPOMetadataProvider:
    """本地缓存优先的 WO 元数据提供器。"""

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

    def lookup(self, pub_no: str) -> WIPOMetadata | None:
        key = normalize_wo_pubno(pub_no) or pub_no
        raw = self._cache.get(key)
        if not raw:
            raw = self._cache.get(key.upper()) or self._cache.get(key.lower())
        if not isinstance(raw, dict):
            return None
        return WIPOMetadata(
            publication_language=_normalize_lang_code(raw.get("publication_language")),
            filing_language=_normalize_lang_code(raw.get("filing_language")),
            source=raw.get("source"),
        )

