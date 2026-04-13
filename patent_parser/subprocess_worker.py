"""子进程隔离解析，防止 segfault 杀死主进程。"""

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("patent_parser")

_STDERR_TAIL = 500
_MAX_ERR_LOG_BYTES = 2 * 1024 * 1024
_TABLE_ERR_KEYWORDS = (
    "table", "tablemaster", "table transformer", "rapidtable", "wired table",
    "table_rec", "table_enable", "表格",
)
_ERROR_HINT_KEYWORDS = (
    "error", "exception", "traceback", "runtimeerror", "assert", "failed",
    "cuda", "cublas", "oom", "崩溃", "失败",
)


def subprocess_parse_one(pdf_path_str: str, output_dir_str: str, lang: str,
                         backend: str, parse_method: str,
                         formula_enable: bool, table_enable: bool,
                         gpu_id: int | None = None,
                         timeout_sec: int = 1800,
                         render_workers: int = 1) -> tuple[bool, str, dict]:
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
import os
from pathlib import Path

# === 资源硬限：在 import mineru / torch 之前完成所有 cap ===
# 这部分在模块级执行，因为 MinerU 内部的 ProcessPoolExecutor 使用 spawn
# 启动 worker 时会重新 import 本脚本——cap 必须在 worker 中也生效。
def _cap_process_pool_executor():
    raw = os.environ.get("MINERU_PDF_RENDER_WORKERS", "").strip()
    try:
        cap = int(raw)
    except Exception:
        cap = 0
    if cap <= 0:
        return
    import concurrent.futures
    _OrigPPE = concurrent.futures.ProcessPoolExecutor
    class _CappedPPE(_OrigPPE):
        def __init__(self, max_workers=None, *a, **kw):
            if max_workers is None or max_workers > cap:
                max_workers = cap
            super().__init__(max_workers=max_workers, *a, **kw)
    concurrent.futures.ProcessPoolExecutor = _CappedPPE
    try:
        import concurrent.futures.process as _cfp
        _cfp.ProcessPoolExecutor = _CappedPPE
    except Exception:
        pass

_cap_process_pool_executor()

# ====================================================================
# 以下全部在 __main__ 守卫内。
# MinerU 内部用 spawn 模式启动渲染 worker 时会重新 import 本脚本，
# 没有 __main__ 守卫会导致 worker 进程再次调用 do_parse() → 无限递归。
# ====================================================================
if __name__ == '__main__':
    def _cap_torch_threads():
        raw = os.environ.get("MINERU_PDF_RENDER_WORKERS", "1").strip()
        try:
            n = max(1, int(raw))
        except Exception:
            n = 1
        try:
            import torch
            torch.set_num_threads(n)
            try:
                torch.set_num_interop_threads(n)
            except Exception:
                pass
        except Exception:
            pass

    from mineru.cli.common import do_parse, read_fn
    _cap_torch_threads()

    if os.environ.get("MINERU_DEBUG") == "1":
        try:
            import onnxruntime as rt
            sys.stderr.write(f"[mineru-debug] onnxruntime={rt.__version__} device={rt.get_device()}\\n")
        except Exception as e:
            sys.stderr.write(f"[mineru-debug] onnxruntime import failed: {e}\\n")
        sys.stderr.write(f"[mineru-debug] python={sys.executable}\\n")

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
        try:
            gpu_id = int(os.environ.get("WORKER_GPU_ID", "0"))
        except ValueError:
            gpu_id = 0

    gpu_str = str(gpu_id)
    # 兼容 CUDA 与 ROCm/DCU 可见性变量
    env["CUDA_VISIBLE_DEVICES"] = gpu_str
    env["HIP_VISIBLE_DEVICES"] = gpu_str
    env["ROCR_VISIBLE_DEVICES"] = gpu_str
    env["HSA_VISIBLE_DEVICES"] = gpu_str
    # 限制 CPU 线程，避免每个外层任务再触发 CPU 线程风暴
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("BLIS_NUM_THREADS", "1")
    env.setdefault("MINERU_INTRA_OP_NUM_THREADS", "1")
    env.setdefault("MINERU_INTER_OP_NUM_THREADS", "1")
    env["MINERU_PDF_RENDER_WORKERS"] = str(max(1, int(render_workers)))

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
            proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timeout_min = max(timeout_sec // 60, 1)
            logger.error("解析超时（%dmin），强制终止: %s", timeout_min, pdf_path_str)
            timeout_detail = _read_tail(stderr_path, _STDERR_TAIL)
            failure_log = _write_failure_log(output_dir_str, pdf_path_str, stderr_path, "timeout")
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
            warn = {
                "timeout": True,
                "timeout_sec": timeout_sec,
            }
            if failure_log:
                warn["error_log"] = str(failure_log)
            if timeout_detail:
                return False, f"超时（> {timeout_min}min）; stderr_tail={timeout_detail}", warn
            return False, f"超时（> {timeout_min}min）", warn
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
            return True, "", {}

        error_msg = _read_tail(stderr_path, _STDERR_TAIL)
        if not error_msg:
            error_msg = f"非零退出码 {proc.returncode}"
        failure_log = _write_failure_log(
            output_dir_str,
            pdf_path_str,
            stderr_path,
            f"error_rc_{proc.returncode}",
        )
        warn = {"returncode": proc.returncode}
        if failure_log:
            warn["error_log"] = str(failure_log)
        return False, error_msg, warn
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        logger.exception("子进程执行异常: %s", pdf_path_str)
        failure_log = _write_failure_log(output_dir_str, pdf_path_str, stderr_path, "launcher_exception")
        warn = {"error_type": type(exc).__name__}
        if failure_log:
            warn["error_log"] = str(failure_log)
        return False, f"{type(exc).__name__}: {exc}", warn

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

def _write_failure_log(output_dir_str: str, pdf_path_str: str, stderr_path: Path, reason: str) -> Path | None:
    try:
        out_dir = Path(output_dir_str)
        pdf_path = Path(pdf_path_str)
        log_dir = out_dir / "failed_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        reason_safe = "".join(c if c.isalnum() or c in {"_", "-"} else "_" for c in reason)
        log_path = log_dir / f"{pdf_path.stem}.{ts}.{reason_safe}.stderr.txt"
        data = stderr_path.read_bytes() if stderr_path.exists() else b""
        truncated = False
        if len(data) > _MAX_ERR_LOG_BYTES:
            data = data[:_MAX_ERR_LOG_BYTES]
            truncated = True
        text = data.decode("utf-8", errors="replace")
        header = (
            f"[time={datetime.now().isoformat(timespec='seconds')}]\n"
            f"[reason={reason}] pdf={pdf_path.name}\n"
            f"[pdf_path={pdf_path_str}]\n"
        )
        if truncated:
            header += f"[truncated] bytes={_MAX_ERR_LOG_BYTES}\n"
        log_path.write_text(header + text, encoding="utf-8")
        logger.error("失败详情已写入: %s", log_path)
        return log_path
    except Exception as exc:
        logger.warning("写入失败日志失败: %s", exc)
        return None


def _looks_like_table_failure(err: str) -> bool:
    if not err:
        return False
    s = err.lower()
    has_table = any(k in s for k in _TABLE_ERR_KEYWORDS)
    has_err = any(k in s for k in _ERROR_HINT_KEYWORDS)
    return has_table and has_err


def _parse_one_with_fallback(
    pdf_path_str: str,
    output_dir_str: str,
    lang: str,
    backend: str,
    parse_method: str,
    formula_enable: bool,
    table_enable: bool,
    gpu_id: int | None,
    timeout_sec: int,
    render_workers: int,
) -> tuple[bool, str, dict]:
    ok, err, warnings = subprocess_parse_one(
        pdf_path_str, output_dir_str, lang, backend,
        parse_method,
        formula_enable,
        table_enable,
        gpu_id,
        timeout_sec=timeout_sec,
        render_workers=render_workers,
    )
    if ok or not table_enable:
        return ok, err, warnings
    if not _looks_like_table_failure(err):
        warnings["table_fallback_skipped"] = True
        return ok, err, warnings

    logger.warning(
        "解析失败（可能是表格模型崩溃），禁用表格后重试: %s",
        Path(pdf_path_str).name,
    )
    ok2, err2, retry_warn = subprocess_parse_one(
        pdf_path_str, output_dir_str, lang, backend,
        parse_method,
        formula_enable,
        table_enable=False,
        gpu_id=gpu_id,
        timeout_sec=timeout_sec,
        render_workers=render_workers,
    )
    merged_warn = {
        "table_fallback_used": True,
        "table_fallback_reason": "table_error_signature",
    }
    if warnings.get("error_log"):
        merged_warn["table_fallback_initial_error_log"] = warnings["error_log"]
    if not ok2 and retry_warn.get("error_log"):
        merged_warn["error_log"] = retry_warn["error_log"]
    if retry_warn.get("timeout"):
        merged_warn["timeout"] = True
        merged_warn["timeout_sec"] = retry_warn.get("timeout_sec")
    return ok2, err2, merged_warn


def _merge_part_warnings(merged: dict, part_warn: dict, part_name: str) -> None:
    if not part_warn:
        return
    if part_warn.get("table_fallback_used"):
        merged["table_fallback_used"] = True
    if part_warn.get("table_fallback_skipped"):
        merged["table_fallback_skipped_parts"] = int(merged.get("table_fallback_skipped_parts", 0)) + 1
    if part_warn.get("error_log"):
        logs = merged.setdefault("part_error_logs", [])
        logs.append({part_name: part_warn["error_log"]})
    if part_warn.get("table_fallback_initial_error_log"):
        logs = merged.setdefault("table_fallback_initial_error_logs", [])
        logs.append({part_name: part_warn["table_fallback_initial_error_log"]})
    if part_warn.get("timeout"):
        merged["timeout"] = True
        merged["timeout_sec"] = part_warn.get("timeout_sec")


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
    timeout_sec: int = 1800,
    render_workers: int = 1,
) -> tuple[bool, str, dict]:
    import importlib
    _splitter = importlib.import_module("patent_parser.pdf_splitter")
    get_page_count = _splitter.get_page_count
    split_pdf = _splitter.split_pdf
    merge_markdown_parts = _splitter.merge_markdown_parts
    merge_content_list_parts = _splitter.merge_content_list_parts
    merge_middle_json_parts = _splitter.merge_middle_json_parts
    cleanup_part_output_dirs = _splitter.cleanup_part_output_dirs

    pdf_path = Path(pdf_path_str)
    output_dir = Path(output_dir_str)

    try:
        page_count = get_page_count(pdf_path)
    except Exception as exc:
        logger.warning("无法获取 %s 页数，直接解析: %s", pdf_path.name, exc)
        return _parse_one_with_fallback(
            pdf_path_str, output_dir_str, lang, backend,
            parse_method, formula_enable, table_enable, gpu_id, timeout_sec, render_workers,
        )

    if page_count <= page_limit:
        return _parse_one_with_fallback(
            pdf_path_str, output_dir_str, lang, backend,
            parse_method, formula_enable, table_enable, gpu_id, timeout_sec, render_workers,
        )

    logger.info(
        "PDF 页数 %d > %d，切分解析: %s", page_count, page_limit, pdf_path.name
    )
    # 将切分目录落到 output_dir/_splits 下，而不是系统 /tmp，确保：
    # 1) 磁盘归属清晰，随输出目录一同被备份/清理；
    # 2) 即使 worker 被 SIGKILL，主进程启动时的孤儿扫描能兜底清理。
    splits_parent = output_dir / "_splits"
    part_paths = split_pdf(pdf_path, chunk_size=page_limit, parent_dir=splits_parent)

    if not part_paths:
        logger.error("切分结果为空，回退到直接解析: %s", pdf_path.name)
        return _parse_one_with_fallback(
            pdf_path_str, output_dir_str, lang, backend,
            parse_method, formula_enable, table_enable, gpu_id, timeout_sec, render_workers,
        )

    tmp_dir = part_paths[0].parent
    try:
        warnings: dict = {}
        failed_parts: list[str] = []
        first_error: str = ""
        use_tqdm = os.environ.get("MINERU_TQDM", "1") != "0"
        tqdm = None
        if use_tqdm:
            try:
                from tqdm import tqdm  # type: ignore
            except Exception:
                tqdm = None
        part_iter = part_paths
        pbar = None
        try:
            if tqdm and sys.stderr.isatty():
                pbar = tqdm(part_paths, total=len(part_paths), desc=f"{pdf_path.name} parts", unit="part")
                part_iter = pbar
            for j, part_path in enumerate(part_iter, 1):
                if not pbar:
                    logger.info(
                        "  解析分片 [%d/%d]: %s", j, len(part_paths), part_path.name
                    )
                ok, err, warn = _parse_one_with_fallback(
                    str(part_path), output_dir_str, lang, backend,
                    parse_method, formula_enable, table_enable, gpu_id, timeout_sec, render_workers,
                )
                _merge_part_warnings(warnings, warn, part_path.name)
                if not ok:
                    detail = ""
                    if warn.get("error_log"):
                        detail = f" | 失败详情: {warn['error_log']}"
                    logger.error("  分片解析失败: %s — %s%s", part_path.name, err, detail)
                    failed_parts.append(part_path.name)
                    if not first_error:
                        first_error = f"分片 {part_path.name} 失败: {err}"
        finally:
            if pbar:
                pbar.close()

        if failed_parts:
            logger.error("  存在失败分片，跳过正式合并: %s", ", ".join(failed_parts))
            try:
                merge_markdown_parts(part_paths, output_dir, pdf_path.stem, require_all_parts=True)
            except Exception:
                logger.exception("  分片 md 合并异常: %s", pdf_path.name)
            try:
                merge_content_list_parts(
                    part_paths, output_dir, pdf_path.stem,
                    require_all_parts=True, chunk_size=page_limit,
                )
            except Exception:
                logger.exception("  分片 content_list 合并异常: %s", pdf_path.name)
            try:
                merge_middle_json_parts(
                    part_paths, output_dir, pdf_path.stem,
                    require_all_parts=True, chunk_size=page_limit,
                )
            except Exception:
                logger.exception("  分片 middle.json 合并异常: %s", pdf_path.name)
            return False, first_error or f"{len(failed_parts)} 个分片解析失败", warnings

        merged = merge_markdown_parts(part_paths, output_dir, pdf_path.stem, require_all_parts=True)
        merged_content = merge_content_list_parts(
            part_paths, output_dir, pdf_path.stem,
            require_all_parts=True, chunk_size=page_limit,
        )
        merged_middle = merge_middle_json_parts(
            part_paths, output_dir, pdf_path.stem,
            require_all_parts=True, chunk_size=page_limit,
        )
        ok_md =  merged is not None and merged.exists()
        ok_content = merged_content is not None and merged_content.exists()
        ok_middle = merged_middle is not None and merged_middle.exists()
        if ok_md and ok_content and ok_middle:
            try:
                cleanup_part_output_dirs(part_paths, output_dir)
            except Exception:
                logger.exception("  清理分片目录失败: %s", pdf_path.name)
            logger.info("  合并完成 (%d/%d 片): %s",
                        len(part_paths) - len(failed_parts), len(part_paths), merged)
            return True, "", warnings
        missing = []
        if not ok_md:
            missing.append("md")
        if not ok_middle:
            missing.append("middle_json")
        if not ok_content:
            missing.append("content_list.json")           
        # if merged is not None and merged.exists() and merged_content is not None and merged_middle is not None:
        #     logger.info("  合并完成 (%d/%d 片): %s",
        #                 len(part_paths) - len(failed_parts), len(part_paths), merged)
        #     return True, "", warnings

        logger.error("%s 合并失败或不完整: %s", missing, pdf_path.name)
        warnings["merge_missing"] = missing
        return False, "分片合并失败或缺失 md/content_list/middle", warnings
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
