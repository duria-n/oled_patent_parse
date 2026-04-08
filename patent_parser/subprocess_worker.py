"""子进程隔离解析，防止 segfault 杀死主进程。"""

import json
import logging
import os
import shutil
import signal
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
    # 通过临时脚本 + JSON 配置传参，避免 -c 动态拼接导致的特殊字符/注入问题
    payload = {
        "pdf_path": pdf_path_str,
        "output_dir": output_dir_str,
        "lang": lang,
        "backend": backend,
        "parse_method": parse_method,
        "formula_enable": formula_enable,
        "table_enable": table_enable,
    }
    script = """\
import json
import sys
from pathlib import Path
from mineru.cli.common import do_parse, read_fn

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
pdf_path = Path(cfg["pdf_path"])
pdf_bytes = read_fn(pdf_path)
do_parse(
    output_dir=cfg["output_dir"],
    pdf_file_names=[pdf_path.stem],
    pdf_bytes_list=[pdf_bytes],
    p_lang_list=[cfg["lang"]],
    backend=cfg["backend"],
    parse_method=cfg["parse_method"],
    formula_enable=cfg["formula_enable"],
    table_enable=cfg["table_enable"],
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
    cfg_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    cfg_path = Path(cfg_file.name)
    cfg_file.write(json.dumps(payload, ensure_ascii=False))
    cfg_file.close()
    script_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    )
    script_path = Path(script_file.name)
    script_file.write(script)
    script_file.close()
    try:
        proc = subprocess.Popen(
            [sys.executable, str(script_path), str(cfg_path)],
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
            env=env,
            start_new_session=True,
        )
        stderr_file.close()

        try:
            proc.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            logger.error("解析超时（30min），强制终止: %s", pdf_path_str)
            # 杀整个进程组，避免 MinerU 子进程遗留
            try:
                if sys.platform != "win32":
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
            except Exception:
                proc.kill()
            proc.wait()
            return False, "超时（> 30min）"
        except KeyboardInterrupt:
            logger.warning("收到中断，终止子进程: %s", pdf_path_str)
            try:
                if sys.platform != "win32":
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
            except Exception:
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
        cfg_path.unlink(missing_ok=True)
        script_path.unlink(missing_ok=True)


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
