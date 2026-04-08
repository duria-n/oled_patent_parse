"""done.json 落盘管理，记录已解析/失败的文件。"""

import json
from datetime import datetime
from pathlib import Path

from .config import DONE_FILENAME, logger


class DoneRecord:
    """管理 done.json 的读写。

    done.json 结构:
    {
      "file1.pdf": {"lang": "ch", "time": "2026-02-12 10:00:00", "status": "done"},
      "file2.pdf": {"lang": "en", "time": "2026-02-12 10:05:00", "status": "failed",
                    "error_msg": "IndexError: too many indices for array"}
    }
    """

    def __init__(self, output_dir: Path):
        self.path = output_dir / DONE_FILENAME
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("done.json 读取失败，将重新创建: %s", self.path)
        return {}

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def is_done(self, pdf_name: str) -> bool:
        entry = self._data.get(pdf_name)
        return entry is not None and entry.get("status") == "done"

    def is_failed(self, pdf_name: str) -> bool:
        entry = self._data.get(pdf_name)
        return entry is not None and entry.get("status") == "failed"

    def mark(self, pdf_name: str, lang: str, status: str = "done",
             error_msg: str | None = None, **extra) -> None:
        entry: dict = {
            "lang": lang,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
        }
        if error_msg:
            # 截断过长的错误信息，避免 done.json 膨胀
            entry["error_msg"] = error_msg[:500]
        for k, v in extra.items():
            entry[k] = v
        self._data[pdf_name] = entry
        self._save()

    @property
    def done_count(self) -> int:
        return sum(1 for v in self._data.values() if v.get("status") == "done")

    @property
    def failed_count(self) -> int:
        return sum(1 for v in self._data.values() if v.get("status") == "failed")

    @property
    def failed_list(self) -> list[str]:
        return [k for k, v in self._data.items() if v.get("status") == "failed"]
