"""结构化专利 JSON 的内部数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass(slots=True)
class EntityRecord:
    entity_id: str
    entity_type: str
    value: str | None = None
    value_num: float | None = None
    unit: str | None = None
    value_pair: list[float] | None = None
    span: list[int] | None = None
    normalized: str | None = None
    canonical_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EntityRecord":
        return cls(
            entity_id=str(payload.get("entity_id") or ""),
            entity_type=str(payload.get("type") or "unknown"),
            value=payload.get("value"),
            value_num=payload.get("value_num"),
            unit=payload.get("unit"),
            value_pair=payload.get("value_pair"),
            span=payload.get("span"),
            normalized=payload.get("normalized"),
            canonical_id=payload.get("canonical_id"),
            raw=payload,
        )


@dataclass(slots=True)
class RelationRecord:
    relation_id: str
    relation_type: str
    source_entity_id: str | None = None
    target_entity_id: str | None = None
    confidence: float | None = None
    rule: str | None = None
    distance: int | None = None
    sentence_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RelationRecord":
        return cls(
            relation_id=str(payload.get("relation_id") or ""),
            relation_type=str(payload.get("type") or "unknown"),
            source_entity_id=payload.get("source_entity_id"),
            target_entity_id=payload.get("target_entity_id"),
            confidence=payload.get("confidence"),
            rule=payload.get("rule"),
            distance=payload.get("distance"),
            sentence_id=payload.get("sentence_id"),
            raw=payload,
        )


@dataclass(slots=True)
class BlockRecord:
    block_id: str
    block_type: str
    text: str | None = None
    section: str | None = None
    subsection: str | None = None
    example_id: str | None = None
    depends_on: list[int] | None = None
    claim_no: int | None = None
    table_id: str | None = None
    table_no: str | None = None
    figure_id: str | None = None
    figure_no: str | None = None
    char_offset: list[int] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    entities: list[EntityRecord] = field(default_factory=list)
    relations: list[RelationRecord] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BlockRecord":
        entities_payload = payload.get("entities") or []
        relations_payload = payload.get("relations") or []
        return cls(
            block_id=str(payload.get("block_id") or ""),
            block_type=str(payload.get("type") or "unknown"),
            text=payload.get("text"),
            section=payload.get("section"),
            subsection=payload.get("subsection"),
            example_id=payload.get("example_id"),
            depends_on=payload.get("depends_on"),
            claim_no=payload.get("claim_no"),
            table_id=payload.get("table_id"),
            table_no=payload.get("table_no"),
            figure_id=payload.get("figure_id"),
            figure_no=payload.get("figure_no"),
            char_offset=payload.get("char_offset"),
            provenance=payload.get("provenance") or {},
            entities=[EntityRecord.from_dict(x) for x in entities_payload if isinstance(x, dict)],
            relations=[RelationRecord.from_dict(x) for x in relations_payload if isinstance(x, dict)],
            raw=payload,
        )


@dataclass(slots=True)
class ExperimentRecord:
    example_id: str
    materials_used: list[str] = field(default_factory=list)
    performance_relations: list[str] = field(default_factory=list)
    role_relations: list[str] = field(default_factory=list)
    source_block_ids: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentRecord":
        perf_rel = payload.get("performance_relations")
        if perf_rel is None:
            perf_rel = payload.get("performance") or []
        return cls(
            example_id=str(payload.get("example_id") or ""),
            materials_used=[str(x) for x in (payload.get("materials_used") or [])],
            performance_relations=[str(x) for x in perf_rel],
            role_relations=[str(x) for x in (payload.get("role_relations") or [])],
            source_block_ids=[str(x) for x in (payload.get("source_block_ids") or [])],
            raw=payload,
        )


@dataclass(slots=True)
class StructuredPatentDocument:
    doc_id: str
    source_file: str
    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    abstract: str | None = None
    claim_tree: dict[str, Any] = field(default_factory=dict)
    reference_numerals: dict[str, str] | None = None
    blocks: list[BlockRecord] = field(default_factory=list)
    experiments: list[ExperimentRecord] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], source_path: Path) -> "StructuredPatentDocument":
        blocks_payload = payload.get("blocks") or []
        experiments_payload = payload.get("experiments") or []
        doc_id = str(payload.get("doc_id") or source_path.stem.replace("_structured", ""))
        source_file = str(payload.get("source_file") or source_path)

        return cls(
            doc_id=doc_id,
            source_file=source_file,
            path=source_path,
            metadata=payload.get("metadata") or {},
            abstract=payload.get("abstract"),
            claim_tree=payload.get("claim_tree") or {},
            reference_numerals=payload.get("reference_numerals"),
            blocks=[BlockRecord.from_dict(x) for x in blocks_payload if isinstance(x, dict)],
            experiments=[ExperimentRecord.from_dict(x) for x in experiments_payload if isinstance(x, dict)],
            raw=payload,
        )

    @property
    def title(self) -> str | None:
        title = self.metadata.get("title") if isinstance(self.metadata, dict) else None
        if not title:
            return None
        return str(title)

    @property
    def publication_number(self) -> str | None:
        value = self.metadata.get("publication_number") if isinstance(self.metadata, dict) else None
        if not value:
            return None
        return str(value)

    def iter_entities(self):
        for block in self.blocks:
            for entity in block.entities:
                yield block, entity

    def iter_relations(self):
        for block in self.blocks:
            for relation in block.relations:
                yield block, relation

    def text_for_embedding(self, max_blocks: int = 120) -> str:
        """按语义优先级拼接文档向量文本。

        规则：标题/摘要 -> 权利要求1 -> 其他权利要求 -> 发明内容(summary) -> 实施例 -> 其余文本。
        """
        parts: list[str] = []
        seen_text: set[str] = set()
        selected_block_ids: set[str] = set()

        def _append_text(text: str | None) -> bool:
            if text is None:
                return False
            cleaned = text.strip()
            if not cleaned or cleaned in seen_text:
                return False
            seen_text.add(cleaned)
            parts.append(cleaned)
            return True

        _append_text(self.title)
        _append_text(self.abstract)

        def _is_claim(block: BlockRecord) -> bool:
            return block.block_type.startswith("claim") or block.section == "claims"

        def _is_claim_one(block: BlockRecord) -> bool:
            return _is_claim(block) and block.claim_no == 1

        def _is_summary(block: BlockRecord) -> bool:
            return block.subsection == "summary"

        def _is_example(block: BlockRecord) -> bool:
            return bool(block.example_id)

        def _collect(selector: Callable[[BlockRecord], bool]) -> None:
            if len(selected_block_ids) >= max_blocks:
                return
            for block in self.blocks:
                if len(selected_block_ids) >= max_blocks:
                    break
                if block.block_id in selected_block_ids:
                    continue
                if not selector(block):
                    continue
                if not _append_text(block.text):
                    continue
                selected_block_ids.add(block.block_id)

        _collect(_is_claim_one)
        _collect(_is_claim)
        _collect(_is_summary)
        _collect(_is_example)
        _collect(lambda _block: True)

        return "\n".join(parts)
