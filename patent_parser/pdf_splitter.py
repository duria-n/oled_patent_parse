"""PDF 大文件切分与 Markdown 合并工具。

超过 PAGE_LIMIT 页的 PDF 会被切分为多个不超过 PAGE_LIMIT 页的子文件，
解析后再将各分片产生的 .md 合并为一个以原始文件名命名的 .md 文件。
"""

import re
import shutil
import tempfile
from pathlib import Path

from .config import logger

PAGE_LIMIT = 60  # 超过此页数则切分

_RESOURCE_DIR_NAMES = ("images", "figures", "equations")
_LOCAL_REF_PATTERN = re.compile(
    r"(?P<prefix>!\[[^\]]*\]\(|<img\b[^>]*?src=[\"'])"
    r"(?P<path>(?:\./)?(?:images|figures|equations)/[^)\"'>\s]+)"
    r"(?P<suffix>\)|[\"'])",
    flags=re.IGNORECASE,
)


def _natural_sort_key(path):
    """自然排序 key：将文件名中的数字段转为整数参与比较。

    例如 part_9 < part_10（字符串排序则 part_10 < part_9）。
    """
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', path.stem)]



def get_page_count(pdf_path: Path) -> int:
    """返回 PDF 页数。优先使用 PyMuPDF，回退到 pypdf。"""
    try:
        import fitz  # PyMuPDF
        with fitz.open(str(pdf_path)) as doc:
            return len(doc)
    except ImportError:
        pass
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(pdf_path)).pages)
    except ImportError:
        pass
    raise RuntimeError("需要 PyMuPDF 或 pypdf 才能获取 PDF 页数，请安装其中之一")


def split_pdf(pdf_path: Path, chunk_size: int = PAGE_LIMIT) -> list[Path]:
    """将 PDF 切分为多个不超过 chunk_size 页的子文件。

    返回临时目录中的分片路径列表，调用方负责在使用完毕后删除该临时目录
    （即 ``shutil.rmtree(parts[0].parent)``）。
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"patent_split_{pdf_path.stem}_"))
    try:
        try:
            import fitz
            return _split_with_fitz(pdf_path, tmp_dir, chunk_size)
        except ImportError:
            pass
        try:
            from pypdf import PdfReader, PdfWriter
            return _split_with_pypdf(pdf_path, tmp_dir, chunk_size)
        except ImportError:
            pass
        raise RuntimeError("需要 PyMuPDF 或 pypdf 才能切分 PDF，请安装其中之一")
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _split_with_fitz(pdf_path: Path, tmp_dir: Path, chunk_size: int) -> list[Path]:
    import fitz

    parts: list[Path] = []
    with fitz.open(str(pdf_path)) as doc:
        total = len(doc)
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            part_num = start // chunk_size + 1
            part_path = tmp_dir / f"{pdf_path.stem}_part{part_num:03d}.pdf"
            sub = fitz.open()
            sub.insert_pdf(doc, from_page=start, to_page=end - 1)
            sub.save(str(part_path))
            sub.close()
            parts.append(part_path)
    return parts


def _split_with_pypdf(pdf_path: Path, tmp_dir: Path, chunk_size: int) -> list[Path]:
    from pypdf import PdfReader, PdfWriter

    parts: list[Path] = []
    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        part_num = start // chunk_size + 1
        part_path = tmp_dir / f"{pdf_path.stem}_part{part_num:03d}.pdf"
        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])
        with open(part_path, "wb") as f:
            writer.write(f)
        parts.append(part_path)
    return parts


def _find_md_for_part(part_stem: str, output_dir: Path, merged_md: Path) -> Path | None:
    """在 output_dir 下定位某个分片的 .md 输出文件。"""
    part_out_dir = output_dir / part_stem

    candidate = part_out_dir / f"{part_stem}.md"
    if candidate.exists():
        return candidate

    if part_out_dir.is_dir():
        hits = [
            p for p in part_out_dir.rglob("*.md")
            if p.resolve() != merged_md.resolve()
        ]
        if hits:
            hits.sort(key=lambda p: (p.stem != part_stem, len(p.parts)))
            logger.debug("分片 %s 在子目录中找到 md: %s", part_stem, hits[0])
            return hits[0]

    hits = [
        p for p in output_dir.rglob("*.md")
        if p.stem == part_stem and p.resolve() != merged_md.resolve()
    ]
    if hits:
        logger.warning("分片 %s 未在预期子目录找到 md，使用兜底路径: %s", part_stem, hits[0])
        return hits[0]

    return None


def _replace_local_resource_refs(content: str, rename_map: dict[str, str]) -> str:
    """稳健替换 Markdown / HTML 中的本地资源引用。"""
    if not rename_map:
        return content

    def _repl(match: re.Match) -> str:
        raw_path = match.group("path")
        normalized = raw_path[2:] if raw_path.startswith("./") else raw_path
        new_path = rename_map.get(normalized)
        if not new_path:
            return match.group(0)
        return f"{match.group('prefix')}{new_path}{match.group('suffix')}"

    return _LOCAL_REF_PATTERN.sub(_repl, content)


def merge_markdown_parts(
    part_paths: list[Path],
    output_dir: Path,
    original_stem: str,
    *,
    require_all_parts: bool = True,
) -> Path | None:
    """合并各分片的 Markdown。

    当 require_all_parts=True 且任一分片缺失 md 时，不产出正式 merged md，返回 None。
    """
    merged_dir = output_dir / original_stem
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_md = merged_dir / f"{original_stem}.md"

    chunks: list[str] = []
    missing_parts: list[str] = []
    part_dirs_to_remove: list[Path] = []

    for part_path in sorted(part_paths, key=_natural_sort_key):
        part_stem = part_path.stem
        md_file = _find_md_for_part(part_stem, output_dir, merged_md)
        if md_file is None:
            missing_parts.append(part_stem)
            logger.warning("分片 %s 未找到任何 .md 输出", part_stem)
            continue

        content = md_file.read_text(encoding="utf-8")
        rename_map: dict[str, str] = {}

        for res_dir_name in _RESOURCE_DIR_NAMES:
            part_res_dir = md_file.parent / res_dir_name
            if not (part_res_dir.exists() and part_res_dir.is_dir()):
                continue

            target_res_dir = merged_dir / res_dir_name
            target_res_dir.mkdir(parents=True, exist_ok=True)

            for res_file in part_res_dir.iterdir():
                if not res_file.is_file():
                    continue
                new_name = f"{part_stem}_{res_file.name}"
                shutil.copy2(res_file, target_res_dir / new_name)
                rename_map[f"{res_dir_name}/{res_file.name}"] = f"{res_dir_name}/{new_name}"
                rename_map[f"./{res_dir_name}/{res_file.name}"] = f"{res_dir_name}/{new_name}"

        content = _replace_local_resource_refs(content, rename_map)
        chunks.append(content)

        part_top_dir = output_dir / part_stem
        if part_top_dir.is_dir() and part_top_dir != merged_dir and part_top_dir not in part_dirs_to_remove:
            part_dirs_to_remove.append(part_top_dir)

    if require_all_parts and missing_parts:
        logger.error(
            "存在 %d 个缺失分片 md，不生成正式合并文件: %s",
            len(missing_parts),
            ", ".join(missing_parts),
        )
        merged_md.unlink(missing_ok=True)
        for res_dir_name in _RESOURCE_DIR_NAMES:
            shutil.rmtree(merged_dir / res_dir_name, ignore_errors=True)
        return None

    if not chunks:
        logger.error("未找到任何可合并的分片 md: %s", original_stem)
        merged_md.unlink(missing_ok=True)
        return None

    merged_md.write_text("\n\n".join(chunks), encoding="utf-8")
    logger.info("合并 %d 片 → %s", len(chunks), merged_md)

    for part_dir in part_dirs_to_remove:
        shutil.rmtree(part_dir, ignore_errors=True)
        logger.debug("已清理分片目录: %s", part_dir)

    return merged_md
