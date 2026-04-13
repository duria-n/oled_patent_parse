"""结构化 JSON 扫描与加载。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Iterator

from .models import StructuredPatentDocument

logger = logging.getLogger("patent_backend.loader")


def discover_structured_files(inputs: Iterable[str | Path], pattern: str = "*_structured.json") -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if path.is_file() and path.name.endswith("_structured.json"):
            files.append(path)
            continue
        if path.is_dir():
            files.extend(sorted(path.rglob(pattern)))
    # 去重保序
    deduped: list[Path] = []
    seen: set[Path] = set()
    for p in files:
        if p in seen:
            continue
        seen.add(p)
        deduped.append(p)
    return deduped


def load_structured_documents(paths: Iterable[Path], strict: bool = False) -> Iterator[StructuredPatentDocument]:
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            if strict:
                raise
            logger.warning("跳过损坏 JSON: %s (%s)", path, exc)
            continue

        if not isinstance(payload, dict):
            if strict:
                raise ValueError(f"JSON 不是对象: {path}")
            logger.warning("跳过无效 JSON 对象: %s", path)
            continue
        try:
            yield StructuredPatentDocument.from_dict(payload, source_path=path)
        except Exception as exc:
            if strict:
                raise
            logger.warning("跳过模型转换失败文件: %s (%s)", path, exc)
            continue
