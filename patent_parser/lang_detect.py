"""PDF 语言检测（基于专利文件名前缀）。"""

import logging
import re
from pathlib import Path

from .config import PATENT_PREFIX_LANG, WIPO_LANG_MAP, logger
from .wipo_metadata import WIPOMetadataProvider, normalize_wo_pubno


def _detect_lang_by_filename(pdf_path: Path, allowed_langs: list[str] | None) -> str | None:
    """根据专利文件名前缀判断语言，命中则返回 MinerU 语言代码，否则返回 None。"""
    stem = pdf_path.stem
    m = re.match(r"^([A-Z]{2})", stem, re.I)
    if not m:
        return None
    prefix = m.group(1).upper()
    lang = PATENT_PREFIX_LANG.get(prefix)
    if lang is None:
        return None
    if allowed_langs and lang not in allowed_langs:
        return None
    return lang


def _is_scanned_pdf(pdf_path: Path) -> bool:
    """检查 PDF 是否为扫描件（无法提取文本）。"""
    from pypdf import PdfReader

    # 抑制 pypdf 及所有子 logger 的 ERROR（如日文 /90ms-RKSJ-H 编码警告）
    pypdf_loggers = [logging.getLogger(name) for name in logging.Logger.manager.loggerDict
                     if name.startswith("pypdf")]
    pypdf_loggers.append(logging.getLogger("pypdf"))
    old_levels = {lg: lg.level for lg in pypdf_loggers}
    for lg in pypdf_loggers:
        lg.setLevel(logging.CRITICAL)
    try:
        reader = PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages[:5]:
            text += page.extract_text() or ""
            if len(text) > 100:
                return False
        return len(text.strip()) == 0
    except Exception:
        # 保守：异常时按扫描件处理，避免漏 OCR
        return True
    finally:
        for lg, lvl in old_levels.items():
            lg.setLevel(lvl)


def detect_pdf_language(
    pdf_path: Path,
    allowed_langs: list[str] | None = None,
    wipo_provider: WIPOMetadataProvider | None = None,
) -> tuple[str, bool, str]:
    """根据专利文件名前缀判断语言，并检测是否为扫描件。

    Args:
        pdf_path: PDF 文件路径。
        allowed_langs: 限定的 MinerU 语言代码列表。
                       前缀不匹配时使用列表中的第一个作为回退。

    Returns:
        (lang, is_scanned): lang 为 MinerU 语言代码，is_scanned 为 True 时
        表示扫描件，应使用 OCR 方法解析。
    """
    fallback = allowed_langs[0] if allowed_langs else "en"

    prefix_lang = _detect_lang_by_filename(pdf_path, allowed_langs)

    # WO 优先使用官方 publication_language
    lang_source = "fallback"
    if pdf_path.stem.upper().startswith("WO") and wipo_provider:
        pub_no = normalize_wo_pubno(pdf_path.stem) or pdf_path.stem
        meta = wipo_provider.lookup(pub_no)
        if meta and meta.publication_language:
            mapped = WIPO_LANG_MAP.get(meta.publication_language)
            if mapped and (not allowed_langs or mapped in allowed_langs):
                lang = mapped
                lang_source = "wipo"
                logger.info("WO 官方语言 %s -> MinerU %s (%s)", meta.publication_language, mapped, pdf_path.name)
            else:
                lang = prefix_lang if prefix_lang else fallback
        else:
            lang = prefix_lang if prefix_lang else fallback
    else:
        lang = prefix_lang if prefix_lang else fallback

    if prefix_lang and lang == prefix_lang and lang_source != "wipo":
        lang_source = "prefix"
        logger.info("根据文件名前缀判断 %s 语言: %s", pdf_path.name, lang)
    else:
        logger.warning("无法从文件名 %s 判断语言，使用默认语言 %s", pdf_path.name, fallback)

    is_scanned = _is_scanned_pdf(pdf_path)
    if is_scanned:
        logger.info("%s 为扫描件，将使用 OCR 解析", pdf_path.name)

    return lang, is_scanned, lang_source
