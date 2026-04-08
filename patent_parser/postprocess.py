"""MinerU 解析结果后处理：生成生产级语义 JSON。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from .config import logger


@dataclass
class PageInfo:
    width: int
    height: int


_FIG_RE = re.compile(r"\b(?:fig(?:ure)?|图)\s*\.?\s*(\d+[a-zA-Z]?)", re.I)
_TABLE_RE = re.compile(r"\b(?:table|表)\s*\.?\s*(\d+[a-zA-Z]?)", re.I)
_CLAIM_RE = re.compile(r"^\s*(\d+)\s*[\.\、:)]\s*(.+)$")
_CLAIM_DEP_RE = re.compile(r"(?:claim|权利要求)\s*(\d+)", re.I)


_SECTION_KEYWORDS = {
    "abstract": [r"\babstract\b", r"摘要"],
    "claims": [r"\bclaims?\b", r"权利要求书", r"权利要求"],
    "description": [r"\bdescription\b", r"说明书"],
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
    r"(?:[A-Z][a-z]?\\d+)+"  # 化学式，如 Alq3
    r"|(?:[A-Z]{2,}\\d+)"    # 大写缩写 + 数字，如 NPB1
    r"|(?:[A-Z]{2,}[A-Za-z]*\\d+)"  # 大写缩写混合数字
    r"|(?:[A-Z][A-Za-z]{1,}\\d+)"   # 可能的材料缩写 + 数字
    r"|(?:[A-Z]{2,}[A-Za-z0-9\\-]{1,})"  # NPB, CBP 等
    r")\\b"
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
            context["section"] = section
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
        try:
            val = float(raw_val.replace("×", "x").replace("^", "").replace(" ", ""))
        except ValueError:
            val = None
        entities.append({
            "type": "value",
            "value": raw_val,
            "unit": unit,
            "value_num": val,
        })

    lower = text.lower()
    for metric in _METRIC_KEYWORDS:
        if metric in lower:
            entities.append({"type": "metric", "value": metric})

    for token in _MATERIAL_RE.findall(text):
        if token.upper() in _MATERIAL_STOP:
            continue
        entities.append({
            "type": "material",
            "value": token,
            "normalized": token.lower(),
        })

    for key, canonical in _LAYER_SYNONYMS.items():
        if re.search(rf"\b{re.escape(key)}\b", lower):
            entities.append({"type": "device_layer", "value": key, "normalized": canonical})

    for role in _ROLE_KEYWORDS:
        if role in lower:
            entities.append({"type": "role", "value": role})

    return entities


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


def build_structured_json(
    pdf_path: Path,
    output_dir: Path,
    parse_method: str | None = None,
) -> Path | None:
    stem = pdf_path.stem
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

    for idx, item in enumerate(content_list):
        btype = item.get("type")
        page_idx = item.get("page_idx")
        page_no = page_idx + 1 if isinstance(page_idx, int) else None
        bbox_norm = item.get("bbox")
        page = page_info.get(page_idx) if page_idx is not None else None

        block_id = f"{stem}_p{page_no}_b{idx}"
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

            blocks.append({
                "type": sem_type,
                "text": text,
                "char_offset": [char_start, char_end],
                "provenance": base_prov | {"paragraph_index": paragraph_index},
                "entities": _extract_entities(text),
            })
            paragraph_index += 1
            continue

        if btype == "table":
            html = item.get("table_body") or item.get("html") or ""
            caption_list = item.get("table_caption", []) or []
            table_id = f"{block_id}_table"
            table_struct = _parse_table_html(html) if html else {"rows": []}
            cell_text = " ".join(
                cell.get("text", "")
                for row in table_struct.get("rows", [])
                for cell in row
                if isinstance(cell, dict)
            )
            table_entities = _extract_entities(cell_text)
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
                "type": "table",
                "table_id": table_id,
                "provenance": base_prov | {"paragraph_index": paragraph_index},
            })
            paragraph_index += 1

            for cap in caption_list:
                table_no = None
                m = _TABLE_RE.search(cap)
                if m:
                    table_no = m.group(1)
                cap_block_id = f"{table_id}_cap_{len(blocks)}"
                blocks.append({
                    "type": "table_caption",
                    "text": cap,
                    "table_no": table_no,
                    "provenance": base_prov | {"block_id": cap_block_id, "paragraph_index": paragraph_index},
                    "entities": _extract_entities(cap),
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
                "type": "figure",
                "figure_id": fig_id,
                "provenance": base_prov | {"paragraph_index": paragraph_index},
            })
            paragraph_index += 1

            for cap in caption_list:
                fig_no = None
                m = _FIG_RE.search(cap)
                if m:
                    fig_no = m.group(1)
                cap_block_id = f"{fig_id}_cap_{len(blocks)}"
                blocks.append({
                    "type": "figure_caption",
                    "text": cap,
                    "figure_no": fig_no,
                    "provenance": base_prov | {"block_id": cap_block_id, "paragraph_index": paragraph_index},
                    "entities": _extract_entities(cap),
                })
                paragraph_index += 1
            continue

        # 其他类型直接落盘为 raw block
        blocks.append({
            "type": btype or "unknown",
            "raw": item,
            "provenance": base_prov | {"paragraph_index": paragraph_index},
        })
        paragraph_index += 1

    output = {
        "doc_id": stem,
        "source_file": str(pdf_path),
        "blocks": blocks,
        "figures": figures,
        "tables": tables,
    }

    out_path = doc_dir / f"{stem}_structured.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("结构化输出已生成: %s", out_path)
    return out_path
