"""子进程隔离解析，防止 segfault 杀死主进程。"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger("patent_parser")

_STDERR_TAIL = 500


def subprocess_parse_one(pdf_path_str: str, output_dir_str: str, lang: str,
                         backend: str, parse_method: str,
                         formula_enable: bool, table_enable: bool,
                         gpu_id: int | None = None) -> tuple[bool, str]:
    code = f"""\
import sys
from pathlib import Path
from mineru.cli.common import do_parse, read_fn

pdf_path = Path({pdf_path_str!r})
pdf_bytes = read_fn(pdf_path)
do_parse(
    output_dir={output_dir_str!r},
    pdf_file_names=[pdf_path.stem],
    pdf_bytes_list=[pdf_bytes],
    p_lang_list=[{lang!r}],
    backend={backend!r},
    parse_method={parse_method!r},
    formula_enable={formula_enable!r},
    table_enable={table_enable!r},
    f_dump_md=True,
    f_dump_middle_json=True,
    f_dump_model_output=False,
    f_dump_orig_pdf=False,
    f_dump_content_list=True,
    f_draw_layout_bbox=False,
    f_draw_span_bbox=False,
)
"""
    env = os.environ.copy()
    
    # 解析 gpu_id
    if gpu_id is None:
        gpu_id = int(os.environ.get("WORKER_GPU_ID", "0"))
        
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)

    stderr_file = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".stderr", delete=False
    )
    stderr_path = Path(stderr_file.name)
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
            env=env,
        )
        stderr_file.close()

        try:
            proc.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            logger.error("解析超时（30min），强制终止: %s", pdf_path_str)
            proc.kill()
            proc.wait()
            return False, "超时（> 30min）"
        except KeyboardInterrupt:
            logger.warning("收到中断，终止子进程: %s", pdf_path_str)
            proc.kill()
            proc.wait()
            raise

        if proc.returncode == 0:
            return True, ""

        error_msg = _read_tail(stderr_path, _STDERR_TAIL)
        if not error_msg:
            error_msg = f"非零退出码 {proc.returncode}"
        return False, error_msg

    finally:
        stderr_path.unlink(missing_ok=True)


def _read_tail(path: Path, max_bytes: int) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(-max_bytes, 2)
            raw = f.read(max_bytes)
        return raw.decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def _parse_one_with_fallback(
    pdf_path_str: str,
    output_dir_str: str,
    lang: str,
    backend: str,
    parse_method: str,
    formula_enable: bool,
    table_enable: bool,
    gpu_id: int | None,
) -> tuple[bool, str]:
    ok, err = subprocess_parse_one(
        pdf_path_str, output_dir_str, lang, backend,
        parse_method, formula_enable, table_enable, gpu_id,
    )
    if ok or not table_enable:
        return ok, err

    logger.warning(
        "解析失败（可能是表格模型崩溃），禁用表格后重试: %s",
        Path(pdf_path_str).name,
    )
    return subprocess_parse_one(
        pdf_path_str, output_dir_str, lang, backend,
        parse_method, formula_enable, table_enable=False, gpu_id=gpu_id,
    )


def subprocess_parse_one_smart(
    pdf_path_str: str,
    output_dir_str: str,
    lang: str,
    backend: str,
    parse_method: str,
    formula_enable: bool,
    table_enable: bool,
    gpu_id: int | None = None,
    page_limit: int = 60,
) -> tuple[bool, str]:
    import importlib
    _splitter = importlib.import_module("patent_parser.pdf_splitter")
    get_page_count = _splitter.get_page_count
    split_pdf = _splitter.split_pdf
    merge_markdown_parts = _splitter.merge_markdown_parts

    pdf_path = Path(pdf_path_str)
    output_dir = Path(output_dir_str)

    try:
        page_count = get_page_count(pdf_path)
    except Exception as exc:
        logger.warning("无法获取 %s 页数，直接解析: %s", pdf_path.name, exc)
        return _parse_one_with_fallback(
            pdf_path_str, output_dir_str, lang, backend,
            parse_method, formula_enable, table_enable, gpu_id,
        )

    if page_count <= page_limit:
        return _parse_one_with_fallback(
            pdf_path_str, output_dir_str, lang, backend,
            parse_method, formula_enable, table_enable, gpu_id,
        )

    logger.info(
        "PDF 页数 %d > %d，切分解析: %s", page_count, page_limit, pdf_path.name
    )
    part_paths = split_pdf(pdf_path, chunk_size=page_limit)

    if not part_paths:
        logger.error("切分结果为空，回退到直接解析: %s", pdf_path.name)
        return _parse_one_with_fallback(
            pdf_path_str, output_dir_str, lang, backend,
            parse_method, formula_enable, table_enable, gpu_id,
        )

    tmp_dir = part_paths[0].parent
    try:
        failed_parts: list[str] = []
        first_error: str = ""
        for j, part_path in enumerate(part_paths, 1):
            logger.info(
                "  解析分片 [%d/%d]: %s", j, len(part_paths), part_path.name
            )
            ok, err = _parse_one_with_fallback(
                str(part_path), output_dir_str, lang, backend,
                parse_method, formula_enable, table_enable, gpu_id,
            )
            if not ok:
                logger.error("  分片解析失败: %s — %s", part_path.name, err)
                failed_parts.append(part_path.name)
                if not first_error:
                    first_error = f"分片 {part_path.name} 失败: {err}"

        if failed_parts:
            logger.error("  存在失败分片，跳过正式合并: %s", ", ".join(failed_parts))
            merge_markdown_parts(part_paths, output_dir, pdf_path.stem, require_all_parts=True)
            return False, first_error or f"{len(failed_parts)} 个分片解析失败"

        merged = merge_markdown_parts(part_paths, output_dir, pdf_path.stem, require_all_parts=True)
        if merged is not None and merged.exists():
            logger.info("  合并完成 (%d/%d 片): %s",
                        len(part_paths) - len(failed_parts), len(part_paths), merged)
            return True, ""

        logger.error("  合并失败，无完整 md 可用: %s", pdf_path.name)
        return False, "分片合并失败或缺失主 md"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
