"""PDF 大文件切分与 Markdown 合并工具。

超过 PAGE_LIMIT 页的 PDF 会被切分为多个不超过 PAGE_LIMIT 页的子文件，
解析后再将各分片产生的 .md 合并为一个以原始文件名命名的 .md 文件。
"""

import json
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


def split_pdf(
    pdf_path: Path,
    chunk_size: int = PAGE_LIMIT,
    parent_dir: Path | None = None,
) -> list[Path]:
    """将 PDF 切分为多个不超过 chunk_size 页的子文件。

    返回临时目录中的分片路径列表，调用方负责在使用完毕后删除该临时目录
    （即 ``shutil.rmtree(parts[0].parent)``）。

    ``parent_dir`` 可以指定切片所在父目录（默认使用系统 /tmp）。指定后
    切片写到 ``parent_dir / f"patent_split_{stem}"``，便于与输出目录一起
    受到启动清理扫描。
    """
    if parent_dir is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"patent_split_{pdf_path.stem}_"))
    else:
        parent_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = parent_dir / f"patent_split_{pdf_path.stem}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
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


def _find_part_file(part_stem: str, output_dir: Path, suffix: str) -> Path | None:
    part_out_dir = output_dir / part_stem
    candidate = part_out_dir / f"{part_stem}{suffix}"
    if candidate.exists():
        return candidate
    filename = f"{part_stem}{suffix}"
    if part_out_dir.is_dir():
        hits = [p for p in part_out_dir.rglob(filename) if p.is_file()]
        if hits:
            hits.sort(key=lambda p: (len(p.parts), len(str(p))))
            return hits[0]
    hits = [p for p in output_dir.rglob(filename) if p.is_file()]
    if hits:
        hits.sort(key=lambda p: (len(p.parts), len(str(p))))
        return hits[0]
    return None

def _rewrite_img_path(img_path: str, part_stem: str) -> str:
    if not img_path:
        return img_path
    p = img_path.replace("\\", "/")
    for prefix in ("images/", "figures/", "equations/"):
        idx = p.find(prefix)
        if idx >= 0:
            base = p[idx + len(prefix):]
            return f"{prefix}{part_stem}_{base}"
    return img_path


def _extract_part_number(part_stem: str) -> int | None:
    m = re.search(r"_part(\d+)$", part_stem, re.I)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None

def merge_content_list_parts(
    part_paths: list[Path],
    output_dir: Path,
    original_stem: str,
    *,
    require_all_parts: bool = True,
    chunk_size: int = PAGE_LIMIT,
) -> Path | None:
    merged_dir = output_dir / original_stem
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_json = merged_dir / f"{original_stem}_content_list.json"

    merged_list: list[dict] = []
    missing_parts: list[str] = []
    invalid_parts: list[str] = []
    sorted_parts = sorted(part_paths, key=_natural_sort_key)
    for i, part_path in enumerate(sorted_parts, 1):
        part_stem = part_path.stem
        part_no = _extract_part_number(part_stem) or i
        absolute_page_offset = (part_no - 1) * chunk_size
        part_json = _find_part_file(part_stem, output_dir, "_content_list.json")
        if part_json is None:
            missing_parts.append(part_stem)
            logger.warning("分片 %s 未找到 content_list 输出", part_stem)
            continue
        try:
            data = json.loads(part_json.read_text(encoding="utf-8"))
        except Exception as exc:
            invalid_parts.append(part_stem)
            logger.warning("分片 %s content_list 读取失败: %s", part_stem, exc)
            continue
        if not isinstance(data, list):
            invalid_parts.append(part_stem)
            logger.warning("分片 %s content_list 格式异常：根节点不是 list", part_stem)
            continue

        # 复制并重命名资源文件，使 _rewrite_img_path 生成的路径有效
        part_out_dir = output_dir / part_stem
        for res_dir_name in _RESOURCE_DIR_NAMES:
            part_res_dir = part_out_dir / res_dir_name
            if not (part_res_dir.exists() and part_res_dir.is_dir()):
                continue
            target_res_dir = merged_dir / res_dir_name
            target_res_dir.mkdir(parents=True, exist_ok=True)
            for res_file in part_res_dir.iterdir():
                if not res_file.is_file():
                    continue
                new_name = f"{part_stem}_{res_file.name}"
                dest = target_res_dir / new_name
                if not dest.exists():
                    shutil.copy2(res_file, dest)

        part_items: list[dict] = []
        part_invalid = False
        for item in data:
            if not isinstance(item, dict):
                logger.warning("分片 %s content_list 项格式异常：存在非 dict 项", part_stem)
                part_invalid = True
                break
            page_idx = item.get("page_idx")
            if page_idx is not None and not isinstance(page_idx, int):
                logger.warning("分片 %s content_list 项格式异常：page_idx 非 int", part_stem)
                part_invalid = True
                break
            item_copy = dict(item)
            if isinstance(page_idx, int):
                item_copy["page_idx"] = page_idx + absolute_page_offset
            if isinstance(item_copy.get("img_path"), str):
                item_copy["img_path"] = _rewrite_img_path(item_copy["img_path"], part_stem)
            part_items.append(item_copy)
        if part_invalid:
            invalid_parts.append(part_stem)
            continue
        merged_list.extend(part_items)

    if require_all_parts and (missing_parts or invalid_parts):
        missing_parts = list(dict.fromkeys(missing_parts))
        invalid_parts = list(dict.fromkeys(invalid_parts))
        detail = []
        if missing_parts:
            detail.append(f"缺失分片: {', '.join(missing_parts)}")
        if invalid_parts:
            detail.append(f"坏分片: {', '.join(invalid_parts)}")
        logger.error("content_list 合并失败，不生成文件: %s", " | ".join(detail))
        merged_json.unlink(missing_ok=True)
        return None

    merged_json.write_text(json.dumps(merged_list, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("合并 content_list -> %s", merged_json)
    return merged_json


def merge_middle_json_parts(
    part_paths: list[Path],
    output_dir: Path,
    original_stem: str,
    *,
    require_all_parts: bool = True,
    chunk_size: int = PAGE_LIMIT,
) -> Path | None:
    merged_dir = output_dir / original_stem
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_json = merged_dir / f"{original_stem}_middle.json"

    merged_pages: list[dict] = []
    missing_parts: list[str] = []
    invalid_parts: list[str] = []
    backend = None
    version = None

    sorted_parts = sorted(part_paths, key=_natural_sort_key)
    for i, part_path in enumerate(sorted_parts, 1):
        part_stem = part_path.stem
        part_no = _extract_part_number(part_stem) or i
        absolute_page_offset = (part_no - 1) * chunk_size
        part_json = _find_part_file(part_stem, output_dir, "_middle.json")
        if part_json is None:
            missing_parts.append(part_stem)
            logger.warning("分片 %s 未找到 middle.json 输出", part_stem)
            continue
        try:
            data = json.loads(part_json.read_text(encoding="utf-8"))
        except Exception as exc:
            invalid_parts.append(part_stem)
            logger.warning("分片 %s middle.json 读取失败: %s", part_stem, exc)
            continue
        if not isinstance(data, dict):
            invalid_parts.append(part_stem)
            logger.warning("分片 %s middle.json 格式异常：根节点不是 dict", part_stem)
            continue
        pages = data.get("pdf_info", [])
        if not isinstance(pages, list):
            invalid_parts.append(part_stem)
            logger.warning("分片 %s middle.json 格式异常：pdf_info 不是 list", part_stem)
            continue
        part_pages: list[dict] = []
        broken = False
        for page in pages:
            if not isinstance(page, dict):
                invalid_parts.append(part_stem)
                logger.warning("分片 %s middle.json 页项异常：存在非 dict 页", part_stem)
                broken = True
                break
            page_copy = dict(page)
            page_idx = page_copy.get("page_idx")
            if page_idx is not None and not isinstance(page_idx, int):
                invalid_parts.append(part_stem)
                logger.warning("分片 %s middle.json 页项异常：page_idx 非 int", part_stem)
                broken = True
                break
            if isinstance(page_idx, int):
                page_copy["page_idx"] = page_idx + absolute_page_offset
            part_pages.append(page_copy)
        if broken:
            continue
        merged_pages.extend(part_pages)
        backend = backend or data.get("_backend")
        version = version or data.get("_version_name")

    if require_all_parts and (missing_parts or invalid_parts):
        missing_parts = list(dict.fromkeys(missing_parts))
        invalid_parts = list(dict.fromkeys(invalid_parts))
        detail = []
        if missing_parts:
            detail.append(f"缺失分片: {', '.join(missing_parts)}")
        if invalid_parts:
            detail.append(f"坏分片: {', '.join(invalid_parts)}")
        logger.error("middle.json 合并失败，不生成文件: %s", " | ".join(detail))
        merged_json.unlink(missing_ok=True)
        return None

    merged = {"pdf_info": merged_pages}
    if backend:
        merged["_backend"] = backend
    if version:
        merged["_version_name"] = version
    merged_json.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("合并 middle.json -> %s", merged_json)
    return merged_json


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
    cleanup_part_dirs: bool = False,
) -> Path | None:
    """合并各分片的 Markdown。

    当 require_all_parts=True 且任一分片缺失 md 时，不产出正式 merged md，返回 None。
    """
    merged_dir = output_dir / original_stem
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_md = merged_dir / f"{original_stem}.md"

    chunks: list[str] = []
    missing_parts: list[str] = []
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

    if cleanup_part_dirs:
        cleanup_part_output_dirs(part_paths, output_dir)

    return merged_md


def cleanup_part_output_dirs(part_paths: list[Path], output_dir: Path) -> None:
    """清理每个分片在 output_dir 下的输出目录。"""
    for part_path in sorted(part_paths, key=_natural_sort_key):
        part_dir = output_dir / part_path.stem
        if not part_dir.exists() or not part_dir.is_dir():
            continue
        shutil.rmtree(part_dir, ignore_errors=True)
        logger.debug("已清理分片目录: %s", part_dir)


def cleanup_orphan_split_dirs(*roots: Path) -> int:
    """扫描并清理遗留的切分临时目录。

    遍历 ``roots`` 以及 ``/tmp``，将形如 ``patent_split_*`` 的空/孤目录删除。
    返回清理的目录数量。用于主进程启动时打扫上次 crash 残留。
    """
    removed = 0
    seen: set[Path] = set()

    candidates: list[Path] = []
    for root in roots:
        if root is None:
            continue
        try:
            root = Path(root)
        except TypeError:
            continue
        if not root.exists() or not root.is_dir():
            continue
        try:
            # 直接子目录 + 深一层 _splits 子目录
            for child in root.rglob("patent_split_*"):
                if child.is_dir():
                    candidates.append(child)
        except OSError:
            pass

    try:
        tmp = Path(tempfile.gettempdir())
        for child in tmp.glob("patent_split_*"):
            if child.is_dir():
                candidates.append(child)
    except OSError:
        pass

    for c in candidates:
        try:
            rc = c.resolve()
        except OSError:
            rc = c
        if rc in seen:
            continue
        seen.add(rc)
        try:
            shutil.rmtree(c, ignore_errors=True)
            removed += 1
            logger.info("清理残留切分目录: %s", c)
        except OSError:
            pass
    return removed
