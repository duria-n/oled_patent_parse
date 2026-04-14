"""MinerU 解析结果后处理：生成生产级语义 JSON。"""

from __future__ import annotations

import json
import hashlib
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
        raw: Optional[dict[str, Any]] = None
        table_id: Optional[str] = None
        table_no: Optional[str] = None
        figure_id: Optional[str] = None
        figure_no: Optional[str] = None

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
_CLAIM_RE = re.compile(r"^\s*(?:claim\s*|claims?\s*|权利要求\s*)?(\d+)\s*(?:[\.\、:：)]|\b)\s*(.+)$", re.I)
_CLAIM_LABEL_RE = re.compile(r"^\s*(?:claim|claims?|权利要求)\s*(\d+)\s*[\.\、:：)]?\s*(.+)$", re.I)
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
    # "example": [r"\bexample\b", r"实施例", r"例"],
    "example": [r"\bexample\s*\d+\b", r"实施例\s*[0-9A-Za-z\-]+"] ,
}

_METRIC_KEYWORDS = [
    "eqe", "ce", "pe", "current efficiency", "power efficiency", "luminance",
    "t50", "lt50", "lifetime", "turn-on voltage", "voltage", "cie",
]
_METRIC_PATTERNS = []
for _m in _METRIC_KEYWORDS:
    if _m in {"ce", "pe"}:
        _METRIC_PATTERNS.append((_m, re.compile(rf"\b{re.escape(_m)}\b", re.I)))
    else:
        pattern = re.escape(_m).replace(r"\ ", r"\s+")
        _METRIC_PATTERNS.append((_m, re.compile(rf"\b{pattern}\b", re.I)))
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
    "chunk_role_binding": 0.82,
    "pattern_material_as_role": 0.93,
    "pattern_role_material": 0.92,
    "pattern_material_in_layer": 0.9,
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
    "host", "host material",
    "dopant", "guest", "guest material",
    "emitter", "light-emitting material",
    "acceptor", "donor",
    "hole transport", "hole transporting", "htl",
    "electron transport", "electron transporting", "etl",
    "hole injection", "hil",
    "electron injection", "eil",
    "hole blocking", "hbl",
    "exciton blocking", "ebl",
    "injection",
]
_ROLE_PATTERNS = []
for _r in _ROLE_KEYWORDS:
    _role_pat = re.escape(_r).replace(r"\ ", r"\s+")
    _ROLE_PATTERNS.append((_r, re.compile(rf"\b{_role_pat}\b", re.I)))

_SUPERSCRIPT_TRANS = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁻": "-", "⁺": "+",
})

_UNIT_RE = re.compile(
    r"(?P<value>[-+]?\d+(?:\.\d+)?"
    r"(?:\s*(?:×|x|X)\s*10(?:\^)?\s*[-+−⁻⁺]?(?:\d+|[⁰¹²³⁴⁵⁶⁷⁸⁹]+)"
    r"|[eE][-+]?\d+)?)\s*(?P<unit>"
    r"cd\s*/\s*m(?:2|²)"
    r"|cd\s*[*·]?\s*m(?:-2|⁻²)"
    r"|mA\s*/\s*cm(?:2|²)"
    r"|A\s*/\s*cm(?:2|²)"
    r"|A\s*/\s*m(?:2|²)"
    r"|mW\s*/\s*cm(?:2|²)"
    r"|cd\s*/\s*A"
    r"|lm\s*/\s*W"
    r"|V|nm|eV|%|h|hr|hours|K|°C|C"
    r")",
    re.I,
)

_MATERIAL_PATTERNS = [
    re.compile(r"\b(?:[A-Z][a-z]?\d+)+\b"),  # Alq3
    re.compile(r"\b(?:[A-Z]{2,}[A-Za-z0-9\-]{1,}|[a-z][A-Z][A-Za-z0-9\-]{1,24})\b"),  # CBP/TPBi/mCP
    re.compile(r"\b(?:fac|mer)?-?[A-Z][a-z]?(?:\([A-Za-z0-9+\-]{1,24}\)){1,3}\d{0,3}\b", re.I),  # fac-Ir(ppy)3
    re.compile(r"\b(?:Compound|Cmpd\.?)\s*[A-Za-z0-9]+\b", re.I),  # Compound 1 / Cmpd. A
]
_MATERIAL_STOP = {"FIG", "TABLE", "EXAMPLE", "OLED", "PCT", "WO", "US", "EP"}
_MATERIAL_HINT_RE = re.compile(
    r"\b(host|dopant|emitter|acceptor|donor|compound|material|layer|"
    r"htl|etl|eml|hil|eil|hbl|ebl)\b|主体|客体|掺杂|化合物|材料|传输层|发光层|注入层",
    re.I,
)
_EXAMPLE_HEADING_RE = re.compile(
    r"^\s*(?:实施例|对比例|合成例|制备例|器件例|"
    r"example|comparative\s+example|synthesis\s+example|preparation\s+example|device\s+example)\b",
    re.I,
)
_EXAMPLE_HEAD_RE = re.compile(
    r"^\s*(?:实施例|合成例|制备例|器件例|对比例|example|comparative\s+example|"
    r"synthesis\s+example|preparation\s+example|device\s+example)\s*([0-9A-Za-z\-]+)\b",
    re.I,
)
_MAT_CANON_KEY_RE = re.compile(r"[^A-Za-z0-9]+")
_MAT_ALIAS_LEXICON = {
    "4,4'-bis(carbazol-9-yl)biphenyl": "CBP",
    "4,4-bis(carbazol-9-yl)biphenyl": "CBP",
    "N,N'-di(1-naphthyl)-N,N'-diphenylbenzidine": "NPB",
    "2,2',2''-(1,3,5-benzinetriyl)-tris(1-phenyl-1-h-benzimidazole)": "TPBi",
    "4,7-diphenyl-1,10-phenanthroline": "BPhen",
    "tris(2-phenylpyridine)iridium": "Ir(ppy)3",
    "fac-Ir(ppy)3": "Ir(ppy)3",
    "CBP": "CBP",
    "NPB": "NPB",
    "TPBi": "TPBi",
    "BPhen": "BPhen",
    "Ir(ppy)3": "Ir(ppy)3",
    "mCP": "mCP",
    "TCTA": "TCTA",
}
_MAT_ALIAS_KEY_MAP: dict[str, str] = {}
for _alias, _canonical in _MAT_ALIAS_LEXICON.items():
    _alias_key = _MAT_CANON_KEY_RE.sub("", _alias).upper()
    _canonical_key = _MAT_CANON_KEY_RE.sub("", _canonical).upper()
    if _alias_key and _canonical_key:
        _MAT_ALIAS_KEY_MAP[_alias_key] = _canonical_key
_MAT_ALIAS_PAIR_PATTERNS = [
    re.compile(
        r"\b([A-Za-z][A-Za-z0-9/\-]*(?:\s+[A-Za-z0-9/\-]+){0,5})\s*\(\s*([A-Z][A-Z0-9\-]{1,20})\s*\)"
    ),
    re.compile(
        r"\b([A-Z][A-Z0-9\-]{1,20})\s*\(\s*([A-Za-z][A-Za-z0-9/\-]*(?:\s+[A-Za-z0-9/\-]+){0,5})\s*\)"
    ),
]
_ROW_KEY_COL_HINT_RE = re.compile(
    r"\b(material|materials|compound|sample|name|structure|device|example)\b|"
    r"材料|化合物|样品|名称|结构|器件",
    re.I,
)
_ROLE_PREFIX_ONLY_RE = re.compile(r"^\s*(?:material|compound|is|was|:|：|=|-|\(|\)|\.)*\s*$", re.I)
_ROLE_AS_CUE_RE = re.compile(r"\b(?:was\s+used\s+as|used\s+as|served\s+as|functioned\s+as|acts?\s+as|as)\b", re.I)
_IN_LAYER_CUE_RE = re.compile(r"\bin\b", re.I)
_ROLE_CHUNK_SPLIT_RE = re.compile(r"[,，;；:：]\s*|\b(?:and|or|with|wherein)\b|以及|并且", re.I)


class _SimpleHTMLTableParser(HTMLParser):
    """支持嵌套表格的轻量解析器（忽略嵌套表格内容）。"""

    def __init__(self, block_id: str | None = None):
        super().__init__()
        self.rows: list[list[dict]] = []
        self._current_row: list[dict] | None = None
        self._current_cell: dict | None = None
        self._in_cell = False
        self._table_depth = 0
        self._nested_table_logged = False
        self._block_id = block_id

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._table_depth += 1
            if self._table_depth > 1 and not self._nested_table_logged:
                if self._block_id:
                    logger.debug("忽略嵌套表格内容: %s", self._block_id)
                else:
                    logger.debug("忽略嵌套表格内容")
                self._nested_table_logged = True
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


def _parse_table_html(html: str, block_id: str | None = None) -> dict:
    parser = _SimpleHTMLTableParser(block_id=block_id)
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
                units.add(_normalize_unit(m.group("unit")))
            # 括号内单位
            for m in re.findall(r"\(([^)]+)\)", text):
                if len(m) <= 12:
                    units.add(m)
    return sorted(units)


def _assign_table_cell_entity_ids(
    table_struct: dict,
    table_id: str,
    material_alias_map: dict[str, str] | None = None,
) -> None:
    rows = table_struct.get("rows", []) if isinstance(table_struct, dict) else []
    for r_idx, row in enumerate(rows, 1):
        if not isinstance(row, list):
            continue
        for c_idx, cell in enumerate(row, 1):
            if not isinstance(cell, dict):
                continue
            text = _cell_text(cell)
            if material_alias_map is not None and text:
                _update_material_alias_map(text, material_alias_map)
            ents = cell.get("entities")
            if not isinstance(ents, list):
                ents = _extract_entities(text)
            for ei, ent in enumerate(ents, 1):
                if not isinstance(ent, dict):
                    continue
                if not ent.get("entity_id"):
                    ent["entity_id"] = f"{table_id}_r{r_idx:03d}_c{c_idx:03d}_e{ei:03d}"
            if material_alias_map is not None:
                _assign_material_canonical_ids(ents, material_alias_map)
            if ents:
                cell["entities"] = ents


def _infer_table_schema(rows: list[list[dict]]) -> dict[int, dict]:
    expanded_rows = _expand_table_rows(rows)
    if not expanded_rows:
        return {}
    header_rows = _detect_header_row_count(expanded_rows)
    headers = _collect_col_headers(expanded_rows, header_rows)
    material_col = _detect_material_col(headers)

    schema: dict[int, dict] = {}
    has_metric_col = False
    for col_idx, header_text in headers.items():
        unit = _extract_header_unit(header_text)
        if col_idx == material_col:
            schema[col_idx] = {
                "kind": "row_key",
                "header_text": header_text,
                "header_rows": header_rows,
                "unit": unit,
            }
            continue
        metric = _extract_metric_from_header(header_text)
        role = _extract_role_from_header(header_text)
        if metric:
            has_metric_col = True
            schema[col_idx] = {
                "kind": "metric",
                "metric": metric,
                "header_text": header_text,
                "header_rows": header_rows,
                "unit": unit,
            }
            continue
        if role:
            schema[col_idx] = {
                "kind": "role",
                "role": role,
                "header_text": header_text,
                "header_rows": header_rows,
                "unit": unit,
            }
            continue
        schema[col_idx] = {
            "kind": "other",
            "header_text": header_text,
            "header_rows": header_rows,
            "unit": unit,
        }

    # 若没识别出 metric 列，则把非 row_key 且非 role 的列作为 metric 列兜底
    if not has_metric_col:
        for col_idx, meta in schema.items():
            if meta.get("kind") in {"row_key", "role"}:
                continue
            candidate = (meta.get("header_text") or f"col_{col_idx + 1}").strip()
            if not candidate:
                continue
            meta["kind"] = "metric"
            meta["metric"] = candidate
    return schema


def _infer_table_row_keys(rows: list[list[dict]], header_map: dict[int, dict]) -> dict[int, dict]:
    expanded_rows = _expand_table_rows(rows)
    if not expanded_rows:
        return {}
    if not header_map:
        return {}
    header_rows = max(int(meta.get("header_rows", 1)) for meta in header_map.values() if isinstance(meta, dict))
    material_col = 0
    for col_idx, meta in header_map.items():
        if meta.get("kind") == "row_key":
            material_col = col_idx
            break

    out: dict[int, dict] = {}
    for row_idx in range(header_rows, len(expanded_rows)):
        row = expanded_rows[row_idx]
        cells = [c for c in row if isinstance(c, dict)]
        if not cells or _is_header_row(cells):
            continue
        row_key_text = _cell_text(row[material_col]) if material_col < len(row) else ""
        if not row_key_text:
            continue
        out[row_idx] = {
            "row_idx": row_idx,
            "row_key_text": row_key_text,
            "material_col": material_col,
        }
    return out


def _collect_table_entities_from_cells(table_struct: dict) -> list[dict]:
    rows = table_struct.get("rows", []) if isinstance(table_struct, dict) else []
    out: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, list):
            continue
        for cell in row:
            if not isinstance(cell, dict):
                continue
            ents = cell.get("entities")
            if not isinstance(ents, list):
                continue
            for ent in ents:
                if not isinstance(ent, dict):
                    continue
                ent_id = ent.get("entity_id")
                if isinstance(ent_id, str) and ent_id and ent_id in seen:
                    continue
                if isinstance(ent_id, str) and ent_id:
                    seen.add(ent_id)
                out.append(ent)
    return out


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
                context["example_id"] = None
                return "title"
            context["section"] = section
            context["subsection"] = None
            context["example_id"] = None
            return "title" if section != "claims" else "claims_title"
    for key, pats in _EXAMPLE_KEYWORDS.items():
        if _match_any(t, pats):
            context["section"] = key
            context["example_id"] = None
            return key

    if context.get("section") == "abstract":
        return "abstract"
    if context.get("section") == "claims":
        _, claim_text = _parse_claim_line(t)
        if claim_text is not None:
            dep = _CLAIM_DEP_RE.search(claim_text)
            return "claim_dependent" if dep else "claim_independent"
        return "claim"
    if context.get("section") == "description":
        return "description"
    if context.get("section") in _EXAMPLE_KEYWORDS:
        return context["section"]
    return "text"


def _normalize_unit(unit: str) -> str:
    s = unit.strip().lower().replace(" ", "")
    s = s.replace("·", "*").replace("−", "-").replace("⁻", "-").replace("²", "2")
    if s in {"cd/m2", "cd*m-2"}:
        return "cd/m2"
    if s == "ma/cm2":
        return "mA/cm2"
    if s == "a/cm2":
        return "A/cm2"
    if s == "a/m2":
        return "A/m2"
    if s == "mw/cm2":
        return "mW/cm2"
    if s == "cd/a":
        return "cd/A"
    if s == "lm/w":
        return "lm/W"
    if s == "ev":
        return "eV"
    if s == "hr":
        return "hr"
    return unit.strip()


def _parse_claim_line(text: str) -> tuple[int | None, str | None]:
    t = text.strip()
    for pat in (_CLAIM_RE, _CLAIM_LABEL_RE):
        m = pat.match(t)
        if not m:
            continue
        claim_text = m.group(2).strip()
        try:
            claim_no = int(m.group(1))
        except ValueError:
            claim_no = None
        return claim_no, claim_text
    return None, None


def _has_material_context(text: str, span: list[int], token: str) -> bool:
    # 类似 US20230123456A1 的专利号片段，直接过滤
    if re.match(r"^[A-Z]{2}\d{4,}[A-Z0-9]*$", token):
        return False
    # 化学式类（含数字）优先保留，避免过度过滤 Alq3/Ir(ppy)3 风格名称
    if any(ch.isdigit() for ch in token):
        return True
    window = 48
    start = max(0, span[0] - window)
    end = min(len(text), span[1] + window)
    return _MATERIAL_HINT_RE.search(text[start:end]) is not None


def _dedup_entity_spans(entities: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple[str, int, int, str]] = set()
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        etype = str(ent.get("type") or "unknown")
        span = ent.get("span")
        if not isinstance(span, list) or len(span) != 2:
            continue
        try:
            start = int(span[0])
            end = int(span[1])
        except Exception:
            continue
        token = str(ent.get("value") or "").strip().lower()
        key = (etype, start, end, token)
        if key in seen:
            continue
        seen.add(key)
        out.append(ent)
    return out


def _extract_material_mentions(text: str) -> list[dict]:
    cands: list[dict] = []
    for pat in _MATERIAL_PATTERNS:
        for m in pat.finditer(text):
            token = m.group(0).strip()
            if not token:
                continue
            span = [m.start(), m.end()]
            if token.upper() in _MATERIAL_STOP:
                continue
            if not _has_material_context(text, span, token):
                continue
            cands.append(
                {
                    "type": "material",
                    "value": token,
                    "normalized": token.lower(),
                    "span": span,
                }
            )
    return _dedup_entity_spans(cands)


def _extract_entities(text: str) -> list[dict]:
    entities: list[dict] = []
    if not text:
        return entities

    def _parse_value(raw: str) -> float | None:
        raw_norm = raw.replace(" ", "")
        raw_norm = raw_norm.translate(_SUPERSCRIPT_TRANS)
        raw_norm = raw_norm.replace("×", "x").replace("−", "-").replace("⁻", "-")
        sci = re.match(r"^([+-]?\d+(?:\.\d+)?)[xX]10\^?([+-]?\d+)$", raw_norm)
        if sci:
            try:
                return float(sci.group(1)) * (10 ** int(sci.group(2)))
            except ValueError:
                return None
        try:
            return float(raw_norm)
        except ValueError:
            return None

    for m in _UNIT_RE.finditer(text):
        raw_val = m.group("value")
        unit = _normalize_unit(m.group("unit"))
        span = [m.start(), m.end()]
        val = _parse_value(raw_val)
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
    for metric, pat in _METRIC_PATTERNS:
        for m in pat.finditer(text):
            entities.append({
                "type": "metric",
                "value": metric,
                "span": [m.start(), m.end()],
            })

    entities.extend(_extract_material_mentions(text))

    for key, canonical in _LAYER_SYNONYMS.items():
        for m in re.finditer(rf"\b{re.escape(key)}\b", lower):
            entities.append({
                "type": "device_layer",
                "value": key,
                "normalized": canonical,
                "span": [m.start(), m.end()],
            })

    for role, pat in _ROLE_PATTERNS:
        for m in pat.finditer(text):
            entities.append({
                "type": "role",
                "value": role,
                "span": [m.start(), m.end()],
            })

    return entities


def _normalize_material_key(token: str) -> str:
    return _MAT_CANON_KEY_RE.sub("", token or "").upper()


def _material_canonical_id_from_key(key: str) -> str:
    return f"mat:{key}"


def _fallback_material_canonical_id(token: str) -> str:
    raw = (token or "").strip().lower()
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12].upper() if raw else "0" * 12
    return f"mat:RAW{digest}"


def _seed_material_alias_map(alias_map: dict[str, str]) -> None:
    for alias_key, canonical_key in _MAT_ALIAS_KEY_MAP.items():
        canonical_id = _material_canonical_id_from_key(canonical_key)
        alias_map.setdefault(canonical_key, canonical_id)
        alias_map.setdefault(alias_key, canonical_id)


def _looks_like_material_abbr(token: str) -> bool:
    t = token.strip()
    return bool(re.fullmatch(r"[A-Z][A-Z0-9\-]{1,20}", t))


def _extract_material_alias_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if not text:
        return pairs
    for pat in _MAT_ALIAS_PAIR_PATTERNS:
        for m in pat.finditer(text):
            left = m.group(1).strip()
            right = m.group(2).strip()
            left_is_abbr = _looks_like_material_abbr(left)
            right_is_abbr = _looks_like_material_abbr(right)
            if left_is_abbr and not right_is_abbr:
                canonical = left
            elif right_is_abbr and not left_is_abbr:
                canonical = right
            else:
                left_len = len(_normalize_material_key(left))
                right_len = len(_normalize_material_key(right))
                canonical = left if left_len <= right_len else right
            pairs.append((left, canonical))
            pairs.append((right, canonical))
    return pairs


def _update_material_alias_map(text: str, alias_map: dict[str, str]) -> None:
    for alias, canonical in _extract_material_alias_pairs(text):
        alias_key = _normalize_material_key(alias)
        canonical_key = _normalize_material_key(canonical)
        if canonical_key:
            canonical_key = _MAT_ALIAS_KEY_MAP.get(canonical_key, canonical_key)
        if alias_key:
            alias_key = _MAT_ALIAS_KEY_MAP.get(alias_key, alias_key)
        if not alias_key or not canonical_key:
            continue
        existing = alias_map.get(alias_key) or alias_map.get(canonical_key)
        canonical_id = existing or _material_canonical_id_from_key(canonical_key)
        alias_map[canonical_key] = canonical_id
        alias_map[alias_key] = canonical_id


def _assign_material_canonical_ids(entities: list[dict], alias_map: dict[str, str]) -> None:
    for ent in entities:
        if ent.get("type") != "material":
            continue
        raw_value = str(ent.get("value") or "")
        key = _normalize_material_key(raw_value)
        if not key:
            ent["canonical_id"] = _fallback_material_canonical_id(raw_value)
            continue
        mapped_key = _MAT_ALIAS_KEY_MAP.get(key, key)
        canonical_id = alias_map.get(key) or alias_map.get(mapped_key)
        if not canonical_id:
            canonical_id = _material_canonical_id_from_key(mapped_key)
            alias_map[mapped_key] = canonical_id
            alias_map[key] = canonical_id
        else:
            alias_map[mapped_key] = canonical_id
            alias_map[key] = canonical_id
        ent["canonical_id"] = canonical_id


def _is_example_heading(text: str) -> bool:
    return bool(_EXAMPLE_HEADING_RE.match(text.strip()))


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


def _split_role_chunks(text: str, sent: tuple[int, int]) -> list[tuple[int, int]]:
    start, end = sent
    if start >= end:
        return []
    chunks: list[tuple[int, int]] = []
    cursor = start
    for m in _ROLE_CHUNK_SPLIT_RE.finditer(text[start:end]):
        cut_start = start + m.start()
        cut_end = start + m.end()
        if cut_start > cursor:
            chunks.append((cursor, cut_start))
        cursor = max(cursor, cut_end)
    if cursor < end:
        chunks.append((cursor, end))
    return chunks or [sent]


def _bind_material_roles(
    text: str,
    entities: list[dict],
    base_id: str,
    max_distance: int = 40,
) -> list[dict]:
    materials = [e for e in entities if e.get("type") == "material" and e.get("span")]
    roles = [e for e in entities if e.get("type") == "role" and e.get("span")]
    layers = [e for e in entities if e.get("type") == "device_layer" and e.get("span")]
    relations: list[dict] = []
    if not materials or (not roles and not layers):
        return relations

    sentence_bounds = []
    last = 0
    for m in re.finditer(r"[.;。；]\s*", text):
        sentence_bounds.append((last, m.end()))
        last = m.end()
    sentence_bounds.append((last, len(text)))

    def _in_sent(span, sent):
        return span[0] >= sent[0] and span[1] <= sent[1]

    # P1: 模板优先绑定（方向敏感），最近邻仅作兜底
    pattern_bound_role_ids: set[str] = set()
    bound_pairs: set[tuple[str, str, str]] = set()
    for sid, sent in enumerate(sentence_bounds):
        sent_mats = [m for m in materials if _in_sent(m["span"], sent)]
        sent_roles = [r for r in roles if _in_sent(r["span"], sent)]
        sent_layers = [l for l in layers if _in_sent(l["span"], sent)]
        if not sent_mats:
            continue

        for role in sent_roles:
            role_id = role.get("entity_id")
            role_span = role.get("span")
            if not role_id or not isinstance(role_span, list):
                continue
            best = None
            best_dist = None
            best_rule = None
            for mat in sent_mats:
                mat_id = mat.get("entity_id")
                mat_span = mat.get("span")
                if not mat_id or not isinstance(mat_span, list):
                    continue
                if mat_span[1] <= role_span[0]:
                    between = text[mat_span[1]:role_span[0]]
                    if len(between) <= 64 and _ROLE_AS_CUE_RE.search(between):
                        dist = abs(role_span[0] - mat_span[0])
                        if best_dist is None or dist < best_dist:
                            best = mat
                            best_dist = dist
                            best_rule = "pattern_material_as_role"
                elif role_span[1] <= mat_span[0]:
                    between = text[role_span[1]:mat_span[0]]
                    if len(between) <= 32 and _ROLE_PREFIX_ONLY_RE.match(between):
                        dist = abs(role_span[0] - mat_span[0])
                        if best_dist is None or dist < best_dist:
                            best = mat
                            best_dist = dist
                            best_rule = "pattern_role_material"
            if not best or not best_rule:
                continue
            mat_id = best.get("entity_id")
            if not mat_id:
                continue
            key = (mat_id, role_id, "has_role")
            if key in bound_pairs:
                continue
            bound_pairs.add(key)
            pattern_bound_role_ids.add(role_id)
            relations.append({
                "type": "has_role",
                "source_entity_id": mat_id,
                "target_entity_id": role_id,
                "confidence": CONFIDENCE_SCORES[best_rule],
                "rule": best_rule,
                "distance": int(best_dist or 0),
                "sentence_id": f"{base_id}_s{sid:03d}",
            })

        # 方向模板：compound A in EML（device_layer 作为 role-like target）
        for layer in sent_layers:
            layer_id = layer.get("entity_id")
            layer_span = layer.get("span")
            if not layer_id or not isinstance(layer_span, list):
                continue
            best = None
            best_dist = None
            for mat in sent_mats:
                mat_id = mat.get("entity_id")
                mat_span = mat.get("span")
                if not mat_id or not isinstance(mat_span, list):
                    continue
                if mat_span[1] <= layer_span[0]:
                    between = text[mat_span[1]:layer_span[0]]
                    if len(between) <= 24 and _IN_LAYER_CUE_RE.search(between):
                        dist = abs(layer_span[0] - mat_span[0])
                        if best_dist is None or dist < best_dist:
                            best = mat
                            best_dist = dist
            if not best:
                continue
            mat_id = best.get("entity_id")
            if not mat_id:
                continue
            key = (mat_id, layer_id, "has_role")
            if key in bound_pairs:
                continue
            bound_pairs.add(key)
            relations.append({
                "type": "has_role",
                "source_entity_id": mat_id,
                "target_entity_id": layer_id,
                "confidence": CONFIDENCE_SCORES["pattern_material_in_layer"],
                "rule": "pattern_material_in_layer",
                "distance": int(best_dist or 0),
                "sentence_id": f"{base_id}_s{sid:03d}",
            })

    # clause/chunk 优先：例如 "CBP host, NPB dopant"
    bound_role_ids = set(pattern_bound_role_ids)
    for sid, sent in enumerate(sentence_bounds):
        sent_mats = [m for m in materials if _in_sent(m["span"], sent)]
        sent_roles = [r for r in roles if _in_sent(r["span"], sent)]
        if not sent_mats or not sent_roles:
            continue
        for chunk in _split_role_chunks(text, sent):
            chunk_mats = [m for m in sent_mats if _in_sent(m["span"], chunk)]
            if len(chunk_mats) != 1:
                continue
            mat = chunk_mats[0]
            mat_id = mat.get("entity_id")
            mat_span = mat.get("span")
            if not mat_id or not isinstance(mat_span, list):
                continue
            chunk_roles = [
                r for r in sent_roles
                if _in_sent(r["span"], chunk) and r.get("entity_id") not in bound_role_ids
            ]
            for role in chunk_roles:
                role_id = role.get("entity_id")
                role_span = role.get("span")
                if not role_id or not isinstance(role_span, list):
                    continue
                dist = abs(role_span[0] - mat_span[0])
                if max_distance > 0 and dist > max_distance:
                    continue
                pair_key = (mat_id, role_id, "has_role")
                if pair_key in bound_pairs:
                    continue
                bound_pairs.add(pair_key)
                bound_role_ids.add(role_id)
                relations.append({
                    "type": "has_role",
                    "source_entity_id": mat_id,
                    "target_entity_id": role_id,
                    "confidence": CONFIDENCE_SCORES["chunk_role_binding"],
                    "rule": "chunk_role_binding",
                    "distance": dist,
                    "sentence_id": f"{base_id}_s{sid:03d}",
                })

    # 兜底最近邻：仅在句内只有一个 material 时启用，避免多材料句错绑
    for sid, sent in enumerate(sentence_bounds):
        sent_mats = [m for m in materials if _in_sent(m["span"], sent)]
        sent_roles = [r for r in roles if _in_sent(r["span"], sent)]
        if len(sent_mats) != 1 or not sent_roles:
            continue
        best = sent_mats[0]
        src_id = best.get("entity_id")
        mspan = best.get("span")
        if not src_id or not isinstance(mspan, list):
            continue
        for role in sent_roles:
            role_id = role.get("entity_id")
            role_span = role.get("span")
            if not role_id or not isinstance(role_span, list):
                continue
            if role_id in bound_role_ids:
                continue
            dist = abs(role_span[0] - mspan[0])
            if max_distance > 0 and dist > max_distance:
                continue
            pair_key = (src_id, role_id, "has_role")
            if pair_key in bound_pairs:
                continue
            bound_pairs.add(pair_key)
            bound_role_ids.add(role_id)
            relations.append({
                "type": "has_role",
                "source_entity_id": src_id,
                "target_entity_id": role_id,
                "confidence": CONFIDENCE_SCORES["nearest_role"],
                "rule": "nearest_role",
                "distance": dist,
                "sentence_id": f"{base_id}_s{sid:03d}",
            })
    return relations


def _is_header_row(cells: list[dict]) -> bool:
    return bool(cells) and all(bool(c.get("is_header")) for c in cells)


def _row_text(cells: list[dict]) -> str:
    return " | ".join(
        str(c.get("text", "")).strip()
        for c in cells
        if isinstance(c, dict) and str(c.get("text", "")).strip()
    )


def _cell_text(cell: dict | None) -> str:
    if not isinstance(cell, dict):
        return ""
    return str(cell.get("text", "")).strip()


def _expand_table_rows(rows: list[list[dict]]) -> list[list[dict | None]]:
    expanded: list[list[dict | None]] = []
    carry: dict[int, tuple[dict, int]] = {}
    max_cols = 0

    for row in rows:
        if not isinstance(row, list):
            continue
        out: list[dict | None] = []
        col = 0

        def _flush_carry_contiguous() -> None:
            nonlocal col
            while col in carry:
                c, left = carry[col]
                out.append(c)
                if left <= 1:
                    del carry[col]
                else:
                    carry[col] = (c, left - 1)
                col += 1

        _flush_carry_contiguous()
        for raw_cell in row:
            if not isinstance(raw_cell, dict):
                continue
            _flush_carry_contiguous()
            try:
                rowspan = max(1, int(raw_cell.get("rowspan", 1) or 1))
            except Exception:
                rowspan = 1
            try:
                colspan = max(1, int(raw_cell.get("colspan", 1) or 1))
            except Exception:
                colspan = 1
            for off in range(colspan):
                out.append(raw_cell)
                if rowspan > 1:
                    carry[col + off] = (raw_cell, rowspan - 1)
            col += colspan
        _flush_carry_contiguous()
        expanded.append(out)
        max_cols = max(max_cols, len(out))

    if max_cols <= 0:
        return []
    for row in expanded:
        if len(row) < max_cols:
            row.extend([None] * (max_cols - len(row)))
    return expanded


def _detect_header_row_count(expanded_rows: list[list[dict | None]]) -> int:
    if not expanded_rows:
        return 0
    header_rows = 0
    limit = min(3, len(expanded_rows))
    for i in range(limit):
        row = expanded_rows[i]
        cells = [c for c in row if isinstance(c, dict)]
        if not cells:
            continue
        header_ratio = sum(1 for c in cells if c.get("is_header")) / max(len(cells), 1)
        numeric_cells = sum(1 for c in cells if re.search(r"\d", _cell_text(c)))
        if i == 0:
            if header_ratio >= 0.5 or numeric_cells <= max(1, len(cells) // 3):
                header_rows += 1
            else:
                header_rows = 1
                break
        else:
            if header_ratio >= 0.5 and numeric_cells <= max(1, len(cells) // 2):
                header_rows += 1
            else:
                break
    return max(header_rows, 1)


def _collect_col_headers(expanded_rows: list[list[dict | None]], header_rows: int) -> dict[int, str]:
    if not expanded_rows:
        return {}
    col_count = len(expanded_rows[0])
    headers: dict[int, str] = {}
    for col in range(col_count):
        parts: list[str] = []
        seen: set[str] = set()
        for r in range(min(header_rows, len(expanded_rows))):
            txt = _cell_text(expanded_rows[r][col])
            if not txt or txt in seen:
                continue
            seen.add(txt)
            parts.append(txt)
        headers[col] = " ".join(parts).strip()
    return headers


def _extract_metric_from_header(header_text: str) -> str | None:
    if not header_text:
        return None
    for metric, pat in _METRIC_PATTERNS:
        if pat.search(header_text):
            return metric
    return None


def _extract_role_from_header(header_text: str) -> str | None:
    if not header_text:
        return None
    for role, pat in _ROLE_PATTERNS:
        if pat.search(header_text):
            return role
    return None


def _extract_header_unit(header_text: str) -> str | None:
    if not header_text:
        return None
    m = _UNIT_RE.search(header_text)
    if m:
        return _normalize_unit(m.group("unit"))
    for inner in re.findall(r"\(([^)]+)\)", header_text):
        unit = inner.strip()
        if unit and len(unit) <= 12:
            return unit
    return None


def _detect_material_col(headers: dict[int, str]) -> int:
    for col, text in headers.items():
        if _ROW_KEY_COL_HINT_RE.search(text or ""):
            return col
    return 0


def _promote_row_material_canonical(material_alias_map: dict[str, str], row_key_text: str) -> str:
    key = _normalize_material_key(row_key_text)
    if not key:
        return _fallback_material_canonical_id(row_key_text)
    mapped_key = _MAT_ALIAS_KEY_MAP.get(key, key)
    canonical_id = _material_canonical_id_from_key(mapped_key)
    old = material_alias_map.get(key)
    if not old:
        old = material_alias_map.get(mapped_key)
    if old and old != canonical_id:
        for alias, mapped in list(material_alias_map.items()):
            if mapped == old:
                material_alias_map[alias] = canonical_id
    material_alias_map[mapped_key] = canonical_id
    material_alias_map[key] = canonical_id
    return canonical_id


def _extract_cell_value_entities(metric_name: str, cell_text: str, header_unit: str | None) -> list[dict]:
    bind_text = f"{metric_name}: {cell_text}".strip()
    entities = _extract_entities(bind_text)
    values = [e for e in entities if e.get("type") == "value"]
    if values:
        return values

    # CIE 常见为坐标对，单元格中可能没有 "cie" 字样
    if metric_name == "cie":
        m = re.search(r"([0-9]*\.?[0-9]+)\s*[,，/]\s*([0-9]*\.?[0-9]+)", cell_text)
        if m:
            x = m.group(1)
            y = m.group(2)
            try:
                pair = [float(x), float(y)]
            except ValueError:
                pair = None
            ent = {
                "type": "value",
                "value": f"{x},{y}",
                "span": [0, len(cell_text)],
            }
            if pair:
                ent["value_pair"] = pair
            return [ent]

    # 兜底：纯数字列（单位来自表头）
    values = []
    for m in re.finditer(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", cell_text):
        raw = m.group(0)
        try:
            value_num = float(raw)
        except ValueError:
            value_num = None
        ent = {
            "type": "value",
            "value": raw,
            "span": [m.start(), m.end()],
        }
        if value_num is not None:
            ent["value_num"] = value_num
        if header_unit:
            ent["unit"] = header_unit
        values.append(ent)
    return values


def _collect_table_entities_relations(
    rows: list[list[dict]],
    table_id: str,
    material_alias_map: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    expanded_rows = _expand_table_rows(rows)
    if not expanded_rows:
        return [], []

    table_struct = {"rows": rows}
    table_entities: list[dict] = _collect_table_entities_from_cells(table_struct)
    table_relations: list[dict] = []

    schema = _infer_table_schema(rows)
    row_keys = _infer_table_row_keys(rows, schema)
    ent_ids = {
        str(e.get("entity_id"))
        for e in table_entities
        if isinstance(e, dict) and isinstance(e.get("entity_id"), str) and e.get("entity_id")
    }
    ent_seq = max(len(ent_ids) + 1, 1)
    rel_seq = 1

    def _append_entity(ent: dict) -> str | None:
        nonlocal ent_seq
        if not isinstance(ent, dict):
            return None
        ent_id = ent.get("entity_id")
        if not isinstance(ent_id, str) or not ent_id:
            while True:
                candidate = f"{table_id}_e{ent_seq:03d}"
                ent_seq += 1
                if candidate not in ent_ids:
                    ent_id = candidate
                    ent["entity_id"] = ent_id
                    break
        if ent_id in ent_ids:
            return ent_id
        ent_ids.add(ent_id)
        table_entities.append(ent)
        return ent_id

    def _append_relation(rel: dict) -> None:
        nonlocal rel_seq
        rel["relation_id"] = f"{table_id}_r{rel_seq:03d}"
        rel_seq += 1
        table_relations.append(rel)

    metric_cols = {c: m for c, m in schema.items() if m.get("kind") == "metric"}
    role_cols = {c: m for c, m in schema.items() if m.get("kind") == "role"}

    for row_idx in sorted(row_keys):
        row_meta = row_keys[row_idx]
        row = expanded_rows[row_idx]
        material_col = int(row_meta.get("material_col", 0))
        row_key_text = str(row_meta.get("row_key_text") or "").strip()
        if not row_key_text:
            continue

        _update_material_alias_map(row_key_text, material_alias_map)
        row_cell = row[material_col] if material_col < len(row) else None
        row_cell_entities = row_cell.get("entities", []) if isinstance(row_cell, dict) else []
        row_mats = [e for e in row_cell_entities if isinstance(e, dict) and e.get("type") == "material"]
        if row_mats:
            row_material = row_mats[0]
            if not row_material.get("canonical_id"):
                row_material["canonical_id"] = _promote_row_material_canonical(
                    material_alias_map,
                    str(row_material.get("value") or row_key_text),
                )
        else:
            row_material = {
                "type": "material",
                "value": row_key_text,
                "normalized": row_key_text.lower(),
                "span": [0, max(1, len(row_key_text))],
                "canonical_id": _promote_row_material_canonical(material_alias_map, row_key_text),
            }
        row_material_id = _append_entity(row_material)
        if not row_material_id:
            continue

        for col_idx, meta in metric_cols.items():
            if col_idx >= len(row):
                continue
            metric_name = str(meta.get("metric") or "").strip()
            if not metric_name:
                continue
            cell = row[col_idx]
            cell_txt = _cell_text(cell)
            if not cell_txt:
                continue

            metric_ent = {
                "type": "metric",
                "value": metric_name,
                "normalized": metric_name.lower(),
                "span": [0, len(metric_name)],
            }
            metric_id = _append_entity(metric_ent)
            if not metric_id:
                continue

            _append_relation(
                {
                    "type": "row_has_metric",
                    "source_entity_id": row_material_id,
                    "target_entity_id": metric_id,
                    "confidence": 0.92,
                    "rule": "table_row_key_binding",
                    "distance": abs(col_idx - material_col),
                    "sentence_id": f"{table_id}_row{row_idx + 1:03d}",
                }
            )

            header_unit = meta.get("unit")
            cell_entities = cell.get("entities", []) if isinstance(cell, dict) else []
            value_entities = [
                e for e in cell_entities
                if isinstance(e, dict) and e.get("type") == "value"
            ]
            if not value_entities:
                value_entities = _extract_cell_value_entities(metric_name, cell_txt, header_unit)
            for val in value_entities:
                if header_unit and not val.get("unit"):
                    val["unit"] = header_unit
                val_id = _append_entity(val)
                if not val_id:
                    continue
                _append_relation(
                    {
                        "type": "has_value",
                        "source_entity_id": metric_id,
                        "target_entity_id": val_id,
                        "confidence": 0.95,
                        "rule": "table_col_binding",
                        "distance": 0,
                        "sentence_id": f"{table_id}_row{row_idx + 1:03d}",
                    }
                )

        for col_idx, meta in role_cols.items():
            if col_idx >= len(row):
                continue
            role_name = str(meta.get("role") or "").strip()
            if not role_name:
                continue
            cell = row[col_idx]
            cell_txt = _cell_text(cell)
            if not cell_txt:
                continue
            cell_entities = cell.get("entities", []) if isinstance(cell, dict) else []
            role_entities = [
                e for e in cell_entities
                if isinstance(e, dict) and e.get("type") == "role"
            ]
            if not role_entities:
                role_entities = [{"type": "role", "value": role_name, "span": [0, len(role_name)]}]
            for role_ent in role_entities:
                role_id = _append_entity(role_ent)
                if not role_id:
                    continue
                _append_relation(
                    {
                        "type": "has_role",
                        "source_entity_id": row_material_id,
                        "target_entity_id": role_id,
                        "confidence": 0.9,
                        "rule": "table_role_column",
                        "distance": abs(col_idx - material_col),
                        "sentence_id": f"{table_id}_row{row_idx + 1:03d}",
                    }
                )

    if table_entities and table_relations:
        return table_entities, table_relations

    # 兜底：极端坏表格结构，退回按行文本抽取，避免完全无结构产出
    for row_idx, row in enumerate(rows, 1):
        if not isinstance(row, list):
            continue
        bind_text = _row_text([c for c in row if isinstance(c, dict)])
        if not bind_text:
            continue
        _update_material_alias_map(bind_text, material_alias_map)
        row_entities = _extract_entities(bind_text)
        for ent in row_entities:
            _append_entity(ent)
        _assign_material_canonical_ids(row_entities, material_alias_map)
        row_relations = _bind_metric_values(bind_text, row_entities, base_id=f"{table_id}_row{row_idx:03d}")
        row_relations.extend(_bind_material_roles(bind_text, row_entities, base_id=f"{table_id}_row{row_idx:03d}"))
        for rel in row_relations:
            _append_relation(rel)
    return table_entities, table_relations


def _extract_example_id(text: str) -> str | None:
    m = _EXAMPLE_HEAD_RE.match(text.strip())
    if not m:
        return None
    token = m.group(1).strip()
    if not token:
        return None
    if re.search(r"\d", token):
        return token
    if re.fullmatch(r"[A-Za-z]", token):
        return token
    return None


def _extract_claim_depends(text: str, self_claim_no: int | None = None) -> list[int]:
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
    # 排除自引用，避免 claim tree 出现循环
    if self_claim_no is not None:
        nums.discard(self_claim_no)
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
    context = {"section": None, "subsection": None, "example_id": None}
    reference_numerals: dict[str, str] = {}
    material_alias_map: dict[str, str] = {}
    _seed_material_alias_map(material_alias_map)

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
            _update_material_alias_map(text, material_alias_map)
            entities = _extract_entities(text)
            for ei, ent in enumerate(entities, 1):
                ent["entity_id"] = f"{block_id}_e{ei:03d}"
            _assign_material_canonical_ids(entities, material_alias_map)
            relations = _bind_metric_values(text, entities, base_id=block_id)
            relations.extend(_bind_material_roles(text, entities, base_id=block_id))
            for ri, rel in enumerate(relations, 1):
                rel["relation_id"] = f"{block_id}_r{ri:03d}"
            is_example_sec = sem_type in _EXAMPLE_KEYWORDS or sem_type.endswith("_example")
            is_example_heading = is_example_sec and _is_example_heading(text)
            raw_ex_id = _extract_example_id(text) if is_example_heading else None
            if raw_ex_id and is_example_sec:
                context["example_id"] = f"{sem_type}_{raw_ex_id}"
            elif is_example_heading:
                # 新实施例标题未带编号时，清空旧编号，后续正文继承空值而不是串到上一个实施例
                context["example_id"] = None
            else:
                # 进入非实施例章节时清空上下文
                if context.get("section") in {"claims", "abstract"}:
                    context["example_id"] = None
                elif sem_type in {"claims_title", "claim"}:
                    context["example_id"] = None
                elif sem_type == "title" and context.get("subsection") in {
                    "background", "summary", "drawings_desc", "detailed_desc"
                }:
                    context["example_id"] = None
            example_id = context.get("example_id")
            claim_no = None
            if sem_type.startswith("claim"):
                claim_no, _ = _parse_claim_line(text)
            if sem_type.startswith("claim"):
                depends_on = _extract_claim_depends(text, self_claim_no=claim_no)
            else:
                depends_on = []

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
            table_struct = _parse_table_html(html, block_id=table_id) if html else {"rows": []}
            _assign_table_cell_entity_ids(
                table_struct=table_struct,
                table_id=table_id,
                material_alias_map=material_alias_map,
            )
            table_entities, table_relations = _collect_table_entities_relations(
                table_struct.get("rows", []),
                table_id=table_id,
                material_alias_map=material_alias_map,
            )
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
                "entities": table_entities,
                "relations": table_relations,
                "section": context.get("section"),
                "subsection": context.get("subsection"),
                "example_id": context.get("example_id"),
                "provenance": base_prov | {"paragraph_index": paragraph_index},
            })
            paragraph_index += 1

            for cap_idx, cap in enumerate(caption_list, 1):
                table_no = None
                m = _TABLE_RE.search(cap)
                if m:
                    table_no = m.group(1)
                _update_material_alias_map(cap, material_alias_map)
                cap_entities = _extract_entities(cap)
                cap_block_id = f"{table_id}_cap{cap_idx:02d}"
                for ei, ent in enumerate(cap_entities, 1):
                    ent["entity_id"] = f"{cap_block_id}_e{ei:03d}"
                _assign_material_canonical_ids(cap_entities, material_alias_map)
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
                    "example_id": context.get("example_id"),
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
                "section": context.get("section"),
                "subsection": context.get("subsection"),
                "example_id": context.get("example_id"),
                "provenance": base_prov | {"paragraph_index": paragraph_index},
            })
            paragraph_index += 1

            for cap_idx, cap in enumerate(caption_list, 1):
                fig_no = None
                m = _FIG_RE.search(cap)
                if m:
                    fig_no = m.group(1)
                _update_material_alias_map(cap, material_alias_map)
                cap_entities = _extract_entities(cap)
                cap_block_id = f"{fig_id}_cap{cap_idx:02d}"
                for ei, ent in enumerate(cap_entities, 1):
                    ent["entity_id"] = f"{cap_block_id}_e{ei:03d}"
                _assign_material_canonical_ids(cap_entities, material_alias_map)
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
                    "example_id": context.get("example_id"),
                })
                paragraph_index += 1
            continue

        # 其他类型直接落盘为 raw block
        blocks.append({
            "block_id": block_id,
            "type": btype or "unknown",
            "raw": item,
            "provenance": base_prov | {"paragraph_index": paragraph_index},
            "section": context.get("section"),
            "subsection": context.get("subsection"),
            "example_id": context.get("example_id"),
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
    for exp in output["experiments"]:
        perf = []
        role = []
        for rel_id in exp.get("performance", []):
            rel = next(
                (
                    r for b in blocks for r in b.get("relations", [])
                    if r.get("relation_id") == rel_id
                ),
                None,
            )
            if not rel:
                continue
            if rel.get("type") == "has_value":
                perf.append(rel_id)
            elif rel.get("type") == "has_role":
                role.append(rel_id)
        exp["performance_relations"] = perf
        exp["role_relations"] = role

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
