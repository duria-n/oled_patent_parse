"""向量化抽象层：支持 OpenAI 向量与本地哈希向量兜底。"""

from __future__ import annotations

import hashlib
import math
import os
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    dimension: int

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """返回固定维度向量。"""


class HashEmbedder(BaseEmbedder):
    """纯本地、零依赖向量器（用于开发/离线）。"""

    def __init__(self, dimension: int = 384):
        if dimension <= 0:
            raise ValueError("dimension must be > 0")
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        if not text:
            return vec
        for token in text.split():
            h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dimension
            sign = 1.0 if (h[4] & 1) == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


class OpenAIEmbedder(BaseEmbedder):
    """使用 OpenAI Embeddings API（需安装 openai 包）。"""

    def __init__(self, model: str = "text-embedding-3-small"):
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("请先安装 openai: pip install openai") from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("未设置 OPENAI_API_KEY")

        self._client = OpenAI(api_key=api_key)
        self._model = model

        # 用一个最短输入探测维度
        probe = self._client.embeddings.create(model=self._model, input="probe")
        self.dimension = len(probe.data[0].embedding)

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self._model, input=text or " ")
        return list(resp.data[0].embedding)


def build_embedder(kind: str, dimension: int = 384, openai_model: str = "text-embedding-3-small") -> BaseEmbedder:
    k = kind.strip().lower()
    if k in {"auto", ""}:
        # 默认优先生产可用语义向量；无 API Key 时自动回退本地哈希向量。
        if os.environ.get("OPENAI_API_KEY"):
            return OpenAIEmbedder(model=openai_model)
        return HashEmbedder(dimension=dimension)
    if k in {"hash", "local", "none"}:
        return HashEmbedder(dimension=dimension)
    if k in {"openai", "oa"}:
        return OpenAIEmbedder(model=openai_model)
    raise ValueError(f"未知 embedder: {kind}")
