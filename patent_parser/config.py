"""常量与配置。"""

import logging
from pathlib import Path

logger = logging.getLogger("patent_parser")
# 仅为 patent_parser logger 配置 handler，不调用 basicConfig 以避免污染根 logger
if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setLevel(logging.INFO)
    _console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_console_handler)
    logger.setLevel(logging.INFO)

def add_file_logger(log_path: str | Path, level: int = logging.INFO) -> Path:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Avoid duplicate handlers for the same file
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path:
            return path
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return path

DONE_FILENAME = "done.json"

# 专利文件名前缀 -> MinerU 语言代码
PATENT_PREFIX_LANG = {
    "CN": "ch",
    "CH": "ch",
    "JP": "japan",
    "KR": "korean",
    "US": "en",
    "EP": "en",
    "WO": "en",
    "GB": "en",
    "AU": "en",
    "CA": "en",
    "TW": "chinese_cht",
    "DE": "latin",
    "FR": "latin",
    "ES": "latin",
    "IT": "latin",
    "RU": "cyrillic",
}

# MinerU 支持的语言选项
SUPPORTED_LANGS = {
    "auto":        "自动检测",
    "ch":          "简体中文",
    "chinese_cht": "繁体中文",
    "en":          "英语",
    "japan":       "日语",
    "korean":      "韩语",
    "latin":       "拉丁语系（德/法/西/意/葡等）",
    "cyrillic":    "西里尔语系（俄语等）",
    "arabic":      "阿拉伯语",
}

# WIPO/PATENTSCOPE 语言码 -> MinerU 语言码
WIPO_LANG_MAP = {
    "en": "en",
    "fr": "latin",
    "de": "latin",
    "es": "latin",
    "it": "latin",
    "pt": "latin",
    "ru": "cyrillic",
    "zh": "ch",
    "ja": "japan",
    "ko": "korean",
    "ar": "arabic",
}
