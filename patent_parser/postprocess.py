"""MinerU 解析结果后处理：生成生产级语义 JSON。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional
from html.parser import HTMLParser
from pathlib import Path

from .biblio_cache import BiblioMetadataProvider
from .config import logger

try:
    from pydantic import BaseModel, Field, ValidationError
    try:
        from pydantic import ConfigDict
    except Exception:
        ConfigDict = None  # type: ignore
except Exception:  # pragma: no cover
    BaseModel = None  # type: ignore
    Field = None  # type: ignore
    ConfigDict = None  # type: ignore


@dataclass
class PageInfo:
    width: int
    height: int


# ----------------------------
# Pydantic Data Contract
# ----------------------------
if BaseModel:
    class EntityModel(BaseModel):
        model_config = ConfigDict(extra="ignore") if ConfigDict else {}
        if ConfigDict is None:
            class Config:
                extra = "ignore"
        entity_id: str
        type: str
        value: Optional[str] = None
        value_num: Optional[float] = None
        unit: Optional[str] = None
        value_pair: Optional[list[float]] = None
        span: Optional[list[int]] = None
        normalized: Optional[str] = None
        canonical_id: Optional[str] = None

    class RelationModel(BaseModel):
        model_config = ConfigDict(extra="ignore") if ConfigDict else {}
        if ConfigDict is None:
            class Config:
                extra = "ignore"
        relation_id: str
        type: str
        source_entity_id: str
        target_entity_id: str
        confidence: float
        rule: str
        distance: int
        sentence_id: str

    class BlockModel(BaseModel):
        model_config = ConfigDict(extra="ignore") if ConfigDict else {}
        if ConfigDict is None:
            class Config:
                extra = "ignore"
        block_id: str
        type: str
        text: Optional[str] = None
        char_offset: Optional[list[int]] = None
        provenance: dict[str, Any]
        entities: list[EntityModel] = Field(default_factory=list)
        relations: list[RelationModel] = Field(default_factory=list)
        section: Optional[str] = None
        subsection: Optional[str] = None
        example_id: Optional[str] = None
        depends_on: Optional[list[int]] = None
        claim_no: Optional[int] = None

    class TableModel(BaseModel):
        model_config = ConfigDict(extra="ignore") if ConfigDict else {}
        if ConfigDict is None:
            class Config:
                extra = "ignore"
        table_id: str
        html: Optional[str] = None
        structure: dict[str, Any]
        units: list[str] = Field(default_factory=list)
        caption: list[str] = Field(default_factory=list)
        image_path: Optional[str] = None
        entities: list[EntityModel] = Field(default_factory=list)
        provenance: dict[str, Any]

    class FigureModel(BaseModel):
        model_config = ConfigDict(extra="ignore") if ConfigDict else {}
        if ConfigDict is None:
            class Config:
                extra = "ignore"
        figure_id: str
        figure_no: Optional[str] = None
        caption: list[str] = Field(default_factory=list)
        image_path: Optional[str] = None
        provenance: dict[str, Any]

    class DocumentModel(BaseModel):
        model_config = ConfigDict(extra="ignore") if ConfigDict else {}
        if ConfigDict is None:
            class Config:
                extra = "ignore"
        doc_id: str
        metadata: dict[str, Any]
        abstract: Optional[str] = None
        source_file: str
        blocks: list[BlockModel]
        figures: list[FigureModel]
        tables: list[TableModel]
        reference_numerals: Optional[dict[str, str]] = None
        claim_tree: dict[str, Any]
        experiments: list[dict[str, Any]] = Field(default_factory=list)


_FIG_RE = re.compile(r"\b(?:fig(?:ure)?|图)\s*\.?\s*(\d+[a-zA-Z]?)", re.I)
_TABLE_RE = re.compile(r"\b(?:table|表)\s*\.?\s*(\d+[a-zA-Z]?)", re.I)
_CLAIM_RE = re.compile(r"^\s*(\d+)\s*[\.\、:)]\s*(.+)$")
_CLAIM_DEP_RE = re.compile(
    r"(?:claim|claims|权利要求)\s*"
    r"(\d+(?:\s*(?:to|至|-|and|or|、|和)\s*\d+)*)",
    re.I,
)
_REF_NUM_RE = re.compile(r"([A-Za-z\u4e00-\u9fa5][A-Za-z\u4e00-\u9fa5\s\-]*)\s*\((\d+)\)")
_INID_RE = re.compile(r"^\s*[\[\(](\d{2})[\]\)]\s*(.*)$")


_SECTION_KEYWORDS = {
    "abstract": [r"\babstract\b", r"摘要"],
    "claims": [r"\bclaims?\b", r"权利要求书", r"权利要求"],
    "description": [r"\bdescription\b", r"说明书"],
    "background": [r"\bbackground\b", r"背景技术"],
    "summary": [r"\bsummary\b", r"发明内容"],
    "drawings_desc": [r"\bbrief description of the drawings\b", r"附图说明"],
    "detailed_desc": [r"\bdetailed description\b", r"具体实施方式"],
}

_EXAMPLE_KEYWORDS = {
    "synthesis_example": [r"\b(synthesis|preparation)\s+example\b", r"合成例", r"制备例"],
    "device_example": [r"\bdevice\s+example\b", r"器件例"],
    "comparative_example": [r"\bcomparative\s+example\b", r"对比例"],
    "effect_statement": [r"\beffect\b", r"效果", r"有益效果"],
    "example": [r"\bexample\b", r"实施例", r"例"],
}

_METRIC_KEYWORDS = [
    "eqe", "ce", "pe", "current efficiency", "power efficiency", "luminance",
    "t50", "lt50", "lifetime", "turn-on voltage", "voltage", "cie",
]
_METRIC_UNIT_HINTS = {
    "eqe": {"%"},
    "ce": {"cd/a", "cd/a."},
    "current efficiency": {"cd/a"},
    "power efficiency": {"lm/w"},
    "pe": {"lm/w"},
    "luminance": {"cd/m2", "cd*m-2", "cd m-2"},
    "t50": {"h", "hr", "hours"},
    "lt50": {"h", "hr", "hours"},
    "lifetime": {"h", "hr", "hours"},
    "turn-on voltage": {"v"},
    "voltage": {"v"},
    "cie": set(),  # 通常是坐标对，无单位
}
CONFIDENCE_SCORES = {
    "cie_coord_inline": 0.95,
    "metric_unit_match": 0.85,
    "nearest_metric": 0.65,
    "nearest_role": 0.70,
}
_CIE_RE = re.compile(
    r"cie(?:\s*1931)?\s*"
    r"(?:[\(\[]\s*([0-9.]+)\s*[,，]\s*([0-9.]+)\s*[\)\]]"
    r"|\s+([0-9.]+)\s*[,，]\s*([0-9.]+))",
    re.I,
)

_LAYER_SYNONYMS = {
    "htl": "hole_transport_layer",
    "etl": "electron_transport_layer",
    "eml": "emission_layer",
    "hil": "hole_injection_layer",
    "eil": "electron_injection_layer",
    "hbl": "hole_blocking_layer",
    "ebl": "electron_blocking_layer",
    "anode": "anode",
    "cathode": "cathode",
}

_ROLE_KEYWORDS = [
    "host", "dopant", "emitter", "acceptor", "donor",
    "hole transport", "electron transport", "injection",
]

_UNIT_RE = re.compile(
    r"(?P<value>[-+]?\d+(?:\.\d+)?(?:\s*×\s*10\^[-+]?\d+)?)\s*(?P<unit>"
    r"cd/m2|cd\s*m-2|mA/cm2|A/cm2|A/m2|V|nm|eV|%|h|hr|hours|K|°C|C|mW/cm2"
    r")",
    re.I,
)

_MATERIAL_RE = re.compile(
    r"\b("
    r"(?:[A-Z][a-z]?\d+)+"  # 化学式，如 Alq3
    r"|(?:[A-Z]{2,}\d+)"    # 大写缩写 + 数字，如 NPB1
    r"|(?:[A-Z]{2,}[A-Za-z]*\d+)"  # 大写缩写混合数字
    r"|(?:[A-Z][A-Za-z]{1,}\d+)"   # 可能的材料缩写 + 数字
    r"|(?:[A-Z]{2,}[A-Za-z0-9\-]{1,})"  # NPB, CBP 等
    r")\b"
)
_MATERIAL_STOP = {"FIG", "TABLE", "EXAMPLE", "OLED", "PCT", "WO", "US", "EP"}


class _SimpleHTMLTableParser(HTMLParser):
    """支持嵌套表格的轻量解析器（忽略嵌套表格内容）。"""

    def __init__(self):
        super().__init__()
        self.rows: list[list[dict]] = []
        self._current_row: list[dict] | None = None
        self._current_cell: dict | None = None
        self._in_cell = False
        self._table_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._table_depth += 1
            return
        if self._table_depth > 1:
            return
        if tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            attrs_dict = dict(attrs)
            self._current_cell = {
                "text": "",
                "rowspan": int(attrs_dict.get("rowspan", "1") or "1"),
                "colspan": int(attrs_dict.get("colspan", "1") or "1"),
                "is_header": tag == "th",
            }

    def handle_endtag(self, tag):
        if tag == "table":
            if self._table_depth > 0:
                self._table_depth -= 1
            return
        if self._table_depth > 1:
            return
        if tag in ("td", "th") and self._current_row is not None and self._current_cell:
            self._current_cell["text"] = self._current_cell["text"].strip()
            self._current_row.append(self._current_cell)
            self._current_cell = None
            self._in_cell = False
        elif tag == "tr" and self._current_row is not None:
            self.rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data):
        if self._table_depth > 1:
            return
        if self._in_cell and self._current_cell is not None:
            self._current_cell["text"] += data


def _parse_table_html(html: str) -> dict:
    parser = _SimpleHTMLTableParser()
    parser.feed(html)
    for row in parser.rows:
        for cell in row:
            if not isinstance(cell, dict):
                continue
            cell_entities = _extract_entities(cell.get("text", ""))
            if cell_entities:
                cell["entities"] = cell_entities
    return {"rows": parser.rows}


def _extract_table_units(rows: list[list[dict]]) -> list[str]:
    units: set[str] = set()
    for row in rows:
        for cell in row:
            if not isinstance(cell, dict):
                continue
            if not cell.get("is_header"):
                continue
            text = cell.get("text", "")
            for m in _UNIT_RE.finditer(text):
                units.add(m.group("unit"))
            # 括号内单位
            for m in re.findall(r"\(([^)]+)\)", text):
                if len(m) <= 12:
                    units.add(m)
    return sorted(units)


def _normalize_bbox(bbox_norm: list[int], page: PageInfo | None) -> dict:
    if not bbox_norm:
        return {}
    out = {"bbox_norm": bbox_norm}
    if page:
        x0, y0, x1, y1 = bbox_norm
        out["bbox_abs"] = [
            int(x0 * page.width / 1000),
            int(y0 * page.height / 1000),
            int(x1 * page.width / 1000),
            int(y1 * page.height / 1000),
        ]
    return out


def _match_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)


def _classify_text_block(text: str, context: dict) -> str:
    t = text.strip()
    if not t:
        return "text"
    for section, pats in _SECTION_KEYWORDS.items():
        if _match_any(t, pats):
            if section in {"background", "summary", "drawings_desc", "detailed_desc"}:
                context["section"] = "description"
                context["subsection"] = section
                return "title"
            context["section"] = section
            context["subsection"] = None
            return "title" if section != "claims" else "claims_title"
    for key, pats in _EXAMPLE_KEYWORDS.items():
        if _match_any(t, pats):
            context["section"] = key
            return key

    if context.get("section") == "abstract":
        return "abstract"
    if context.get("section") == "claims":
        m = _CLAIM_RE.match(t)
        if m:
            claim_text = m.group(2)
            dep = _CLAIM_DEP_RE.search(claim_text)
            return "claim_dependent" if dep else "claim_independent"
        return "claim"
    if context.get("section") == "description":
        return "description"
    if context.get("section") in _EXAMPLE_KEYWORDS:
        return context["section"]
    return "text"


def _extract_entities(text: str) -> list[dict]:
    entities: list[dict] = []
    if not text:
        return entities

    for m in _UNIT_RE.finditer(text):
        raw_val = m.group("value")
        unit = m.group("unit")
        span = [m.start(), m.end()]
        try:
            val = float(raw_val.replace("×", "x").replace("^", "").replace(" ", ""))
        except ValueError:
            val = None
        entities.append({
            "type": "value",
            "value": raw_val,
            "unit": unit,
            "value_num": val,
            "span": span,
        })

    lower = text.lower()
    for m in _CIE_RE.finditer(text):
        x = m.group(1) or m.group(3)
        y = m.group(2) or m.group(4)
        entities.append({
            "type": "metric",
            "value": "cie",
            "span": [m.start(), m.end()],
            "value_pair": [float(x), float(y)],
        })
        entities.append({
            "type": "value",
            "value": f"{x},{y}",
            "value_pair": [float(x), float(y)],
            "span": [m.start(), m.end()],
        })
    for metric in _METRIC_KEYWORDS:
        for m in re.finditer(re.escape(metric), lower):
            entities.append({
                "type": "metric",
                "value": metric,
                "span": [m.start(), m.end()],
            })

    for m in _MATERIAL_RE.finditer(text):
        token = m.group(0)
        if token.upper() in _MATERIAL_STOP:
            continue
        entities.append({
            "type": "material",
            "value": token,
            "normalized": token.lower(),
            "span": [m.start(), m.end()],
        })

    for key, canonical in _LAYER_SYNONYMS.items():
        for m in re.finditer(rf"\b{re.escape(key)}\b", lower):
            entities.append({
                "type": "device_layer",
                "value": key,
                "normalized": canonical,
                "span": [m.start(), m.end()],
            })

    for role in _ROLE_KEYWORDS:
        for m in re.finditer(re.escape(role), lower):
            entities.append({
                "type": "role",
                "value": role,
                "span": [m.start(), m.end()],
            })

    return entities


def _bind_metric_values(
    text: str,
    entities: list[dict],
    base_id: str,
) -> list[dict]:
    metrics = [e for e in entities if e.get("type") == "metric" and e.get("span")]
    values = [e for e in entities if e.get("type") == "value" and e.get("span")]
    relations: list[dict] = []
    if not metrics or not values:
        return relations

    sentence_bounds = []
    last = 0
    for m in re.finditer(r"[.;。；]\s*", text):
        sentence_bounds.append((last, m.end()))
        last = m.end()
    sentence_bounds.append((last, len(text)))

    def _in_sent(span, sent):
        return span[0] >= sent[0] and span[1] <= sent[1]

    for sid, sent in enumerate(sentence_bounds):
        sent_metrics = [m for m in metrics if _in_sent(m["span"], sent)]
        sent_values = [v for v in values if _in_sent(v["span"], sent)]
        if not sent_metrics or not sent_values:
            continue
        for val in sent_values:
            vpos = val["span"][0]
            # 优先找最近的前置 metric
            candidates = [m for m in sent_metrics if m["span"][0] <= vpos]
            if not candidates:
                candidates = sent_metrics
            best = None
            best_dist = None
            for met in candidates:
                mpos = met["span"][0]
                dist = abs(vpos - mpos)
                if best_dist is None or dist < best_dist:
                    best = met
                    best_dist = dist
            if best is None:
                continue
            # 避免跨越另一个 metric
            between = [
                m for m in sent_metrics
                if min(best["span"][0], vpos) < m["span"][0] < max(best["span"][0], vpos)
            ]
            if between:
                continue
            unit = val.get("unit")
            rule = "nearest_metric"
            confidence = CONFIDENCE_SCORES[rule]
            if best.get("value") == "cie" and val.get("value_pair"):
                rule = "cie_coord_inline"
                confidence = CONFIDENCE_SCORES[rule]
            if unit:
                unit_norm = unit.lower().replace(" ", "")
                allowed = _METRIC_UNIT_HINTS.get(best["value"], set())
                if allowed and unit_norm not in {u.replace(" ", "") for u in allowed}:
                    continue
                if allowed:
                    rule = "metric_unit_match"
                    confidence = CONFIDENCE_SCORES[rule]
            distance = abs(vpos - best["span"][0])
            src_id = best.get("entity_id")
            tgt_id = val.get("entity_id")
            if not src_id or not tgt_id:
                continue
            sent_id = f"{base_id}_s{sid:03d}"
            relations.append({
                "type": "has_value",
                "source_entity_id": src_id,
                "target_entity_id": tgt_id,
                "confidence": confidence,
                "rule": rule,
                "distance": distance,
                "sentence_id": sent_id,
            })
    return relations


def _bind_material_roles(text: str, entities: list[dict], base_id: str) -> list[dict]:
    materials = [e for e in entities if e.get("type") == "material" and e.get("span")]
    roles = [e for e in entities if e.get("type") == "role" and e.get("span")]
    relations: list[dict] = []
    if not materials or not roles:
        return relations

    sentence_bounds = []
    last = 0
    for m in re.finditer(r"[.;。；]\s*", text):
        sentence_bounds.append((last, m.end()))
        last = m.end()
    sentence_bounds.append((last, len(text)))

    def _in_sent(span, sent):
        return span[0] >= sent[0] and span[1] <= sent[1]

    for sid, sent in enumerate(sentence_bounds):
        sent_mats = [m for m in materials if _in_sent(m["span"], sent)]
        sent_roles = [r for r in roles if _in_sent(r["span"], sent)]
        if not sent_mats or not sent_roles:
            continue
        for role in sent_roles:
            rpos = role["span"][0]
            best = None
            best_dist = None
            for mat in sent_mats:
                mpos = mat["span"][0]
                dist = abs(rpos - mpos)
                if best_dist is None or dist < best_dist:
                    best = mat
                    best_dist = dist
            if not best:
                continue
            src_id = best.get("entity_id")
            tgt_id = role.get("entity_id")
            if not src_id or not tgt_id:
                continue
            relations.append({
                "type": "has_role",
                "source_entity_id": src_id,
                "target_entity_id": tgt_id,
                "confidence": CONFIDENCE_SCORES["nearest_role"],
                "rule": "nearest_role",
                "distance": int(best_dist or 0),
                "sentence_id": f"{base_id}_s{sid:03d}",
            })
    return relations


def _extract_example_id(text: str) -> str | None:
    m = re.search(r"(?:实施例|例|example)\s*([\dA-Za-z]+)", text, re.I)
    if not m:
        return None
    return m.group(1)


def _extract_claim_depends(text: str) -> list[int]:
    m = _CLAIM_DEP_RE.search(text)
    if not m:
        return []
    raw = m.group(1)
    nums: set[int] = set()
    # 支持范围（如 1-3 / 1至3）
    for rm in re.finditer(r"(\d+)\s*(?:to|至|-)\s*(\d+)", raw, re.I):
        start, end = int(rm.group(1)), int(rm.group(2))
        if start <= end and (end - start) < 50:
            nums.update(range(start, end + 1))
    # 处理散点数字（如 1,2 and 4-6）
    for num_str in re.findall(r"\d+", raw):
        try:
            nums.add(int(num_str))
        except ValueError:
            continue
    return sorted(nums)


def _build_claim_tree(blocks: list[dict]) -> dict:
    nodes: dict[int, dict] = {}
    for blk in blocks:
        if not isinstance(blk, dict):
            continue
        claim_no = blk.get("claim_no")
        if not claim_no:
            continue
        nodes[claim_no] = {
            "claim_no": claim_no,
            "text": blk.get("text"),
            "depends_on": blk.get("depends_on") or [],
            "children": [],
        }
    for node in nodes.values():
        for parent in node["depends_on"]:
            parent_node = nodes.get(parent)
            if parent_node:
                parent_node["children"].append(node["claim_no"])
    roots = [n for n in nodes.values() if not n["depends_on"] or all(p not in nodes for p in n["depends_on"])]

    def build_nested(claim_no: int) -> dict:
        node = nodes[claim_no]
        return {
            "claim_no": node["claim_no"],
            "text": node.get("text"),
            "depends_on": node.get("depends_on") or [],
            "children": [build_nested(c) for c in sorted(node.get("children", []))],
        }

    nested = [build_nested(r["claim_no"]) for r in sorted(roots, key=lambda x: x["claim_no"])]
    return {
        "roots": sorted([r["claim_no"] for r in roots]),
        "nodes": nodes,
        "nested": nested,
    }


def _load_page_info(middle_json_path: Path) -> dict[int, PageInfo]:
    if not middle_json_path.exists():
        return {}
    try:
        data = json.loads(middle_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    page_info: dict[int, PageInfo] = {}
    for page in data.get("pdf_info", []):
        try:
            idx = int(page.get("page_idx", 0))
            w, h = page.get("page_size", [0, 0])
            page_info[idx] = PageInfo(width=int(w), height=int(h))
        except Exception:
            continue
    return page_info


def _find_output_files(doc_dir: Path, stem: str) -> tuple[Path | None, Path | None]:
    content = None
    middle = None
    for path in doc_dir.rglob(f"{stem}_content_list.json"):
        content = path
        break
    for path in doc_dir.rglob(f"{stem}_middle.json"):
        middle = path
        break
    return content, middle


def _extract_inid_metadata(blocks: list[dict]) -> dict:
    # 仅看首页文本块
    lines: list[str] = []
    for blk in blocks:
        if blk.get("type") != "text":
            continue
        prov = blk.get("provenance", {})
        if prov.get("page_no") != 1:
            continue
        text = blk.get("text", "")
        if text:
            lines.extend([ln.strip() for ln in text.splitlines() if ln.strip()])

    if not lines:
        return {}

    fields: dict[str, list[str]] = {}
    current = None
    for ln in lines:
        m = _INID_RE.match(ln)
        if m:
            current = m.group(1)
            content = m.group(2).strip()
            fields[current] = [content] if content else []
            continue
        if current:
            fields[current].append(ln)

    def _join(code: str) -> str | None:
        vals = fields.get(code)
        if not vals:
            return None
        return " ".join(v for v in vals if v).strip() or None

    def _split_list(code: str) -> list[str] | None:
        text = _join(code)
        if not text:
            return None
        parts = [p.strip() for p in re.split(r";|/|,", text) if p.strip()]
        return parts or None

    return {
        "publication_number": _join("10") or _join("11"),
        "application_number": _join("21"),
        "application_date": _join("22"),
        "priority": _split_list("30"),
        "title": _join("54"),
        "applicants": _split_list("71") or _split_list("73"),
        "inventors": _split_list("72"),
    }


def build_structured_json(
    pdf_path: Path,
    output_dir: Path,
    parse_method: str | None = None,
    biblio_provider: BiblioMetadataProvider | None = None,
    keep_raw: bool = False,
) -> Path | None:
    stem = pdf_path.stem
    safe_stem = "doc_" + re.sub(r"[^a-zA-Z0-9]", "_", stem)
    doc_root = output_dir / stem
    if parse_method:
        doc_dir = doc_root / parse_method
    else:
        doc_dir = doc_root

    if not doc_dir.exists():
        doc_dir = doc_root

    content_path, middle_path = _find_output_files(doc_dir, stem)
    if not content_path:
        logger.warning("未找到 content_list.json，跳过后处理: %s", pdf_path.name)
        return None

    try:
        content_list = json.loads(content_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("content_list.json 读取失败: %s", content_path)
        return None

    page_info = _load_page_info(middle_path) if middle_path else {}
    blocks: list[dict] = []
    figures: list[dict] = []
    tables: list[dict] = []

    paragraph_index = 0
    doc_char_offset = 0
    context = {"section": None}
    reference_numerals: dict[str, str] = {}

    for idx, item in enumerate(content_list):
        btype = item.get("type")
        page_idx = item.get("page_idx")
        page_no = page_idx + 1 if isinstance(page_idx, int) else None
        bbox_norm = item.get("bbox")
        page = page_info.get(page_idx) if page_idx is not None else None

        block_id = f"{safe_stem}_p{page_no}_b{idx}"
        base_prov = {
            "page_no": page_no,
            "source_file": str(pdf_path),
            "block_id": block_id,
        }
        base_prov.update(_normalize_bbox(bbox_norm, page))

        if btype == "text":
            text = item.get("text", "").strip()
            sem_type = _classify_text_block(text, context)
            char_start = doc_char_offset
            char_end = char_start + len(text)
            doc_char_offset = char_end + 1
            entities = _extract_entities(text)
            for ei, ent in enumerate(entities, 1):
                ent["entity_id"] = f"{block_id}_e{ei:03d}"
                if ent.get("type") == "material" and not ent.get("canonical_id"):
                    ent["canonical_id"] = "PENDING_MAPPING"
            relations = _bind_metric_values(text, entities, base_id=block_id)
            relations.extend(_bind_material_roles(text, entities, base_id=block_id))
            for ri, rel in enumerate(relations, 1):
                rel["relation_id"] = f"{block_id}_r{ri:03d}"
            example_id = _extract_example_id(text) if sem_type in _EXAMPLE_KEYWORDS or sem_type.endswith("_example") else None
            if sem_type.startswith("claim"):
                depends_on = _extract_claim_depends(text)
            else:
                depends_on = []
            claim_no = None
            if sem_type.startswith("claim"):
                cm = _CLAIM_RE.match(text)
                if cm:
                    try:
                        claim_no = int(cm.group(1))
                    except ValueError:
                        claim_no = None

            if sem_type == "description" or context.get("subsection") in {"background", "summary", "drawings_desc", "detailed_desc"}:
                for m in _REF_NUM_RE.finditer(text):
                    label = m.group(1).strip().lower()
                    num = m.group(2)
                    reference_numerals.setdefault(num, label)

            blocks.append({
                "block_id": block_id,
                "type": sem_type,
                "text": text,
                "char_offset": [char_start, char_end],
                "provenance": base_prov | {"paragraph_index": paragraph_index},
                "entities": entities,
                "relations": relations,
                "section": context.get("section"),
                "subsection": context.get("subsection"),
                "example_id": example_id,
                "depends_on": depends_on if depends_on else None,
                "claim_no": claim_no,
            })
            paragraph_index += 1
            continue

        if btype == "table":
            html = item.get("table_body") or item.get("html") or ""
            caption_list = item.get("table_caption", []) or []
            table_id = f"{block_id}_table"
            table_struct = _parse_table_html(html) if html else {"rows": []}
            for r_idx, row in enumerate(table_struct.get("rows", []), 1):
                for c_idx, cell in enumerate(row, 1):
                    if not isinstance(cell, dict):
                        continue
                    ents = cell.get("entities") or []
                    for ei, ent in enumerate(ents, 1):
                        ent["entity_id"] = f"{table_id}_r{r_idx:03d}_c{c_idx:03d}_e{ei:03d}"
                        if ent.get("type") == "material" and not ent.get("canonical_id"):
                            ent["canonical_id"] = "PENDING_MAPPING"
                    if ents:
                        cell["entities"] = ents
            cell_text = " ".join(
                cell.get("text", "")
                for row in table_struct.get("rows", [])
                for cell in row
                if isinstance(cell, dict)
            )
            table_entities = _extract_entities(cell_text)
            for ei, ent in enumerate(table_entities, 1):
                ent["entity_id"] = f"{table_id}_e{ei:03d}"
                if ent.get("type") == "material" and not ent.get("canonical_id"):
                    ent["canonical_id"] = "PENDING_MAPPING"
            table_units = _extract_table_units(table_struct.get("rows", []))

            tables.append({
                "table_id": table_id,
                "html": html,
                "structure": table_struct,
                "units": table_units,
                "caption": caption_list,
                "image_path": item.get("img_path"),
                "entities": table_entities,
                "provenance": base_prov | {"paragraph_index": paragraph_index},
            })

            blocks.append({
                "block_id": block_id,
                "type": "table",
                "table_id": table_id,
                "provenance": base_prov | {"paragraph_index": paragraph_index},
            })
            paragraph_index += 1

            for cap_idx, cap in enumerate(caption_list, 1):
                table_no = None
                m = _TABLE_RE.search(cap)
                if m:
                    table_no = m.group(1)
                cap_entities = _extract_entities(cap)
                cap_block_id = f"{table_id}_cap{cap_idx:02d}"
                for ei, ent in enumerate(cap_entities, 1):
                    ent["entity_id"] = f"{cap_block_id}_e{ei:03d}"
                cap_relations = _bind_metric_values(cap, cap_entities, base_id=cap_block_id)
                cap_relations.extend(_bind_material_roles(cap, cap_entities, base_id=cap_block_id))
                for ri, rel in enumerate(cap_relations, 1):
                    rel["relation_id"] = f"{cap_block_id}_r{ri:03d}"
                blocks.append({
                    "block_id": cap_block_id,
                    "type": "table_caption",
                    "text": cap,
                    "table_no": table_no,
                    "provenance": base_prov | {"block_id": cap_block_id, "paragraph_index": paragraph_index},
                    "entities": cap_entities,
                    "relations": cap_relations,
                    "section": context.get("section"),
                    "subsection": context.get("subsection"),
                })
                paragraph_index += 1
            continue

        if btype == "image":
            caption_list = item.get("image_caption", []) or []
            fig_id = f"{block_id}_fig"
            figure_no = None
            for cap in caption_list:
                m = _FIG_RE.search(cap)
                if m:
                    figure_no = m.group(1)
                    break

            figures.append({
                "figure_id": fig_id,
                "figure_no": figure_no,
                "caption": caption_list,
                "image_path": item.get("img_path"),
                "provenance": base_prov | {"paragraph_index": paragraph_index},
            })

            blocks.append({
                "block_id": block_id,
                "type": "figure",
                "figure_id": fig_id,
                "provenance": base_prov | {"paragraph_index": paragraph_index},
            })
            paragraph_index += 1

            for cap_idx, cap in enumerate(caption_list, 1):
                fig_no = None
                m = _FIG_RE.search(cap)
                if m:
                    fig_no = m.group(1)
                cap_entities = _extract_entities(cap)
                cap_block_id = f"{fig_id}_cap{cap_idx:02d}"
                for ei, ent in enumerate(cap_entities, 1):
                    ent["entity_id"] = f"{cap_block_id}_e{ei:03d}"
                cap_relations = _bind_metric_values(cap, cap_entities, base_id=cap_block_id)
                cap_relations.extend(_bind_material_roles(cap, cap_entities, base_id=cap_block_id))
                for ri, rel in enumerate(cap_relations, 1):
                    rel["relation_id"] = f"{cap_block_id}_r{ri:03d}"
                blocks.append({
                    "block_id": cap_block_id,
                    "type": "figure_caption",
                    "text": cap,
                    "figure_no": fig_no,
                    "provenance": base_prov | {"block_id": cap_block_id, "paragraph_index": paragraph_index},
                    "entities": cap_entities,
                    "relations": cap_relations,
                    "section": context.get("section"),
                    "subsection": context.get("subsection"),
                })
                paragraph_index += 1
            continue

        # 其他类型直接落盘为 raw block
        blocks.append({
            "block_id": block_id,
            "type": btype or "unknown",
            "raw": item,
            "provenance": base_prov | {"paragraph_index": paragraph_index},
        })
        paragraph_index += 1

    output = {
        "doc_id": stem,
        "metadata": {},
        "abstract": None,
        "source_file": str(pdf_path),
        "blocks": blocks,
        "figures": figures,
        "tables": tables,
        "reference_numerals": reference_numerals or None,
    }
    output["claim_tree"] = _build_claim_tree(blocks)

    # 抽取 abstract（如果已分类）
    abstracts = [b.get("text") for b in blocks if b.get("type") == "abstract" and b.get("text")]
    if abstracts:
        output["abstract"] = "\n".join(abstracts)

    # 注入本地缓存题录元数据
    if biblio_provider:
        cached = biblio_provider.lookup(stem)
        if cached:
            output["metadata"] = {
                "publication_number": cached.publication_number,
                "publication_date": cached.publication_date,
                "application_number": cached.application_number,
                "application_date": cached.application_date,
                "priority": cached.priority,
                "title": cached.title,
                "applicants": cached.applicants,
                "inventors": cached.inventors,
                "ipc": cached.ipc,
                "cpc": cached.cpc,
                "source": cached.source or "cache",
            }

    # 回退：INID 扉页解析（仅在未命中缓存时）
    if not output["metadata"]:
        inid = _extract_inid_metadata(blocks)
        if inid:
            output["metadata"] = inid | {"source": "inid"}

    if not keep_raw:
        for blk in output.get("blocks", []):
            if isinstance(blk, dict) and "raw" in blk:
                blk.pop("raw", None)

    # experiments 聚合（示例/对比/合成等）
    experiments_map: dict[str, dict[str, Any]] = {}
    for blk in blocks:
        ex_id = blk.get("example_id")
        if not ex_id:
            continue
        exp = experiments_map.setdefault(
            ex_id,
            {
                "example_id": ex_id,
                "materials_used": [],
                "performance": [],
                "source_block_ids": [],
            },
        )
        exp["source_block_ids"].append(blk.get("provenance", {}).get("block_id"))
        for ent in blk.get("entities", []):
            if ent.get("type") == "material":
                exp["materials_used"].append(ent.get("entity_id"))
        for rel in blk.get("relations", []):
            exp["performance"].append(rel.get("relation_id"))
    output["experiments"] = list(experiments_map.values())

    # Pydantic 数据契约校验
    if BaseModel:
        try:
            doc = DocumentModel(**output)
            output = doc.model_dump() if hasattr(doc, "model_dump") else doc.dict()
        except ValidationError as ve:
            logger.error(
                "数据结构校验失败 [%s]:\n%s",
                stem,
                ve.json(indent=2) if hasattr(ve, "json") else str(ve),
            )
            return None
        except Exception as exc:
            logger.error("未知错误导致序列化失败 [%s]: %s", stem, str(exc))
            return None
    else:
        logger.warning("未安装 pydantic，跳过数据契约校验")

    out_path = doc_dir / f"{stem}_structured.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("结构化输出已生成: %s", out_path)
    return out_path
