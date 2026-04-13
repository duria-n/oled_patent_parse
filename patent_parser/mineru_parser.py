"""MinerU 专利解析器具体实现。"""

import logging
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import (
    ProcessPoolExecutor,
    FIRST_COMPLETED,
    wait,
)
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from .base_parser import BasePDFParser
from .config import add_file_logger, logger
from .done_record import DoneRecord
from .lang_detect import detect_pdf_language
from .biblio_cache import BiblioMetadataProvider
from .pdf_splitter import cleanup_orphan_split_dirs
from .postprocess import build_structured_json
from .wipo_metadata import WIPOMetadataProvider
from .subprocess_worker import subprocess_parse_one_smart

PARSER_VERSION = "2026-04-08.dq3"

# BrokenProcessPool 发生后重建 executor 的最大次数，防止无限重试打爆日志
_MAX_POOL_REBUILDS = 3


def _format_warning_summary(warnings: dict) -> str:
    if not warnings:
        return ""
    parts: list[str] = []
    if warnings.get("table_fallback_used"):
        parts.append("table_fallback_used=True")
    if warnings.get("table_fallback_skipped"):
        parts.append("table_fallback_skipped=True")
    if warnings.get("table_fallback_skipped_parts"):
        parts.append(f"table_fallback_skipped_parts={warnings.get('table_fallback_skipped_parts')}")
    if warnings.get("timeout"):
        parts.append(f"timeout=True(timeout_sec={warnings.get('timeout_sec')})")
    if warnings.get("error_log"):
        parts.append(f"error_log={warnings.get('error_log')}")
    if warnings.get("part_error_logs"):
        parts.append(f"part_error_logs={len(warnings.get('part_error_logs', []))}")
    if warnings.get("merge_missing"):
        parts.append(f"merge_missing={warnings.get('merge_missing')}")
    return "; ".join(parts) if parts else str(warnings)

def _get_tqdm():
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm
    except Exception:
        return None


def _discover_log_path() -> Path | None:
    """从当前 logger 的 handlers 里反推主日志文件路径，供子 worker 复用。"""
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                return Path(h.baseFilename)
            except Exception:
                continue
    return None


def _parse_visible_devices(value: str | None) -> list[int] | None:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw in {"-1", "none", "None"}:
        return []
    if raw.lower() == "all":
        return None
    ids: list[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        if not p.isdigit():
            return None
        ids.append(int(p))
    return ids


def _get_gpu_count() -> int:
    """获取可用 GPU 数量（优先 torch，其次可见设备环境变量兜底）。"""
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            if count > 0:
                return count
    except Exception:
        pass
    # DCU/ROCm 环境兜底：当 torch 构建或驱动不一致时，仍可从可见设备变量推断
    for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HSA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        ids = _parse_visible_devices(os.environ.get(key))
        if ids is not None:
            return len(ids)
    return 0


def _init_worker(gpu_queue, fallback_gpu_ids, log_path):
    """Worker 初始化函数：绑定 GPU 并重建日志 handler。

    - 非阻塞地从 mp.Queue 拿一个 GPU 槽位，10 秒内取不到就退化为按 PID 取模，
      这样即使 ProcessPoolExecutor 意外重启 worker、或队列意外为空也不会死锁。
    - spawn 启动的子进程日志 handler 不会继承父进程，这里重新 attach 一个
      FileHandler 指向同一个日志文件，多进程 O_APPEND 写入在大多数 POSIX 内核
      下是行级原子的。
    """
    import os as _os

    try:
        gpu_id = gpu_queue.get(timeout=10)
    except Exception:
        if fallback_gpu_ids:
            gpu_id = fallback_gpu_ids[_os.getpid() % len(fallback_gpu_ids)]
        else:
            gpu_id = 0

    gpu_str = str(gpu_id)
    _os.environ["WORKER_GPU_ID"] = gpu_str
    for k in (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HSA_VISIBLE_DEVICES",
    ):
        _os.environ[k] = gpu_str

    if log_path:
        try:
            # spawn 下 handlers 会是空，直接 add 即可；fork 下避免重复挂载
            existing = [
                h for h in logger.handlers
                if isinstance(h, logging.FileHandler)
                and Path(h.baseFilename) == Path(log_path)
            ]
            if not existing:
                add_file_logger(log_path)
        except Exception:
            pass


class MinerUPatentParser(BasePDFParser):
    """使用 MinerU 解析专利 PDF 的具体实现。"""

    def __init__(
        self,
        input_root: str,
        output_root: str | None = None,
        langs: list[str] | None = None,
        backend: str = "pipeline",
        parse_method: str = "auto",
        formula_enable: bool = True,
        table_enable: bool = True,
        workers: int = 0,
        gpu_ids: list[int] | None = None,
        wipo_metadata_path: str | None = None,
        postprocess_enable: bool = True,
        biblio_metadata_path: str | None = None,
        keep_raw: bool = False,
        parser_version: str | None = None,
        parse_timeout_sec: int = 1800,
        render_workers: int = 0,
    ):
        super().__init__(input_root, output_root)
        self.langs = langs if langs else ["ch", "chinese_cht", "japan", "en", "korean"]
        self.backend = backend
        self.parse_method = parse_method
        self.formula_enable = formula_enable
        self.table_enable = table_enable
        self.postprocess_enable = postprocess_enable
        self.wipo_provider = WIPOMetadataProvider(wipo_metadata_path)
        self.biblio_provider = BiblioMetadataProvider(biblio_metadata_path)
        self.keep_raw = keep_raw
        self.parser_version = parser_version or PARSER_VERSION
        self.parse_timeout_sec = max(1, int(parse_timeout_sec))

        total_gpus = _get_gpu_count()
        if gpu_ids is not None:
            self.gpu_ids = gpu_ids
        else:
            self.gpu_ids = list(range(total_gpus)) if total_gpus > 0 else [0]

        # workers 默认等于指定 GPU 数量
        self.workers = workers if workers > 0 else len(self.gpu_ids)

        # render_workers 自动计算：
        # 之前默认 1 会让 MinerU 内部 ProcessPoolExecutor 强制串行渲染 PDF 各页面，
        # 单 PDF 解析时间因此被放大十倍以上，结果是大概率撞 30 分钟超时。
        # 自动模式按 cpu_count // workers 平均分配，给每个外层 worker 留够内层并发。
        raw_render = int(render_workers)
        if raw_render <= 0:
            cpu = os.cpu_count() or 4
            # 上限 16 防止机器 CPU 极多时反而内存爆炸；下限 2 保留基本并行度
            self.render_workers = max(2, min(16, cpu // max(self.workers, 1)))
            render_source = f"auto(cpu={cpu})"
        else:
            self.render_workers = raw_render
            render_source = "user"

        logger.info(
            "检测到 %d 张 GPU，使用 GPU %s，%d 个 worker，单任务超时 %ds，"
            "内层渲染并发 %d (%s)，进程总预算 = %d",
            total_gpus,
            self.gpu_ids,
            self.workers,
            self.parse_timeout_sec,
            self.render_workers,
            render_source,
            self.workers * self.render_workers,
        )
        # 跨子文件夹汇总失败记录
        self._all_failed: list[str] = []

    def run(self) -> None:
        """覆盖基类 run()，在所有子文件夹处理完毕后统一写 failed_files.txt。"""
        self._all_failed = []
        super().run()
        if self._all_failed:
            deduped_failed = list(dict.fromkeys(self._all_failed))
            failed_file = self.input_root / "failed_files.txt"
            failed_file.write_text("\n".join(deduped_failed) + "\n", encoding="utf-8")
            logger.info("失败文件列表已写入: %s (%d 个)", failed_file, len(deduped_failed))

    def _resolve_lang(self, pdf_path: Path) -> tuple[str, bool, str]:
        """返回 (语言代码, 是否为扫描件)。"""
        return detect_pdf_language(pdf_path, allowed_langs=self.langs, wipo_provider=self.wipo_provider)

    def prepare_output_dir(self, subdir: Path) -> Path:
        # 处理 subdir 就是 input_root 的情况
        if subdir == self.input_root:
            rel = Path("root_pdfs")
        else:
            rel = subdir.relative_to(self.input_root)

        if self.output_root:
            output_dir = self.output_root / rel
        else:
            output_dir = self.input_root / "output" / rel
        output_dir.mkdir(parents=True, exist_ok=True)

        # 清理上一轮 crash 残留的切分目录（output_dir/_splits/* 以及 /tmp/patent_split_*）
        try:
            removed = cleanup_orphan_split_dirs(output_dir / "_splits")
            if removed:
                logger.info("启动清理：共移除 %d 个残留切分目录", removed)
        except Exception:
            logger.debug("清理残留切分目录时忽略异常", exc_info=True)

        return output_dir

    def _parse_one(self, pdf_path: Path, lang: str, output_dir: Path,
                   is_scanned: bool = False, gpu_id: int | None = 0) -> tuple[bool, str, dict]:
        parse_method = "ocr" if is_scanned else self.parse_method
        return subprocess_parse_one_smart(
            pdf_path_str=str(pdf_path),
            output_dir_str=str(output_dir),
            lang=lang,
            backend=self.backend,
            parse_method=parse_method,
            formula_enable=self.formula_enable,
            table_enable=self.table_enable,
            gpu_id=gpu_id,
            timeout_sec=self.parse_timeout_sec,
            render_workers=self.render_workers,
        )

    def parse_pdfs(self, pdf_files: list[Path], output_dir: Path) -> None:
        done = DoneRecord(output_dir, parser_version=self.parser_version)
        total = len(pdf_files)

        pending: list[tuple[int, Path, str, bool, str]] = []
        skipped = 0
        for i, pdf_path in enumerate(pdf_files, 1):
            if done.is_done(pdf_path.name):
                logger.info("[%d/%d] 跳过（已解析）: %s", i, total, pdf_path.name)
                skipped += 1
                continue
            if done.is_failed(pdf_path.name):
                logger.info("[%d/%d] 重试（上次失败）: %s", i, total, pdf_path.name)
            lang, is_scanned, lang_source = self._resolve_lang(pdf_path)
            pending.append((i, pdf_path, lang, is_scanned, lang_source))

        if not pending:
            logger.info("所有 %d 个 PDF 均已解析，无需处理", total)
            return

        logger.info("待解析 %d 个，已跳过 %d 个，总计 %d 个", len(pending), skipped, total)

        if self.workers <= 1:
            self._parse_sequential(pending, output_dir, done)
        else:
            self._parse_parallel(pending, output_dir, done)

        logger.info(
            "本轮结束：累计完成 %d 个，失败 %d 个",
            done.done_count, done.failed_count,
        )
        self._all_failed.extend(done.failed_list)

    def _parse_sequential(self, pending, output_dir, done):
        pending_total = len(pending)
        tqdm = _get_tqdm()
        use_tqdm = os.environ.get("MINERU_TQDM", "1") != "0"
        seq_iter = pending
        if tqdm and use_tqdm and sys.stderr.isatty():
            seq_iter = tqdm(pending, total=pending_total, desc="PDFs", unit="file")
        for i, (idx, pdf_path, lang, is_scanned, lang_source) in enumerate(seq_iter, 1):
            gpu_id = self.gpu_ids[0]
            method_info = "OCR（扫描件）" if is_scanned else self.parse_method
            logger.info("[%d/%d] 正在解析: %s (语言: %s, 方法: %s, GPU: %d)",
                        i, pending_total, pdf_path.name, lang, method_info, gpu_id)
            t0 = time.time()
            success, err, warnings = self._parse_one(pdf_path, lang, output_dir, is_scanned, gpu_id)
            elapsed = time.time() - t0
            if success:
                post_ok = self._postprocess_if_needed(
                    pdf_path, output_dir, "ocr" if is_scanned else self.parse_method
                )
                if post_ok:
                    status = "done_without_tables" if warnings.get("table_fallback_used") else "done"
                    done.mark(
                        pdf_path.name, lang, status,
                        lang_source=lang_source,
                        table_fallback_used=bool(warnings.get("table_fallback_used")),
                        done_with_warnings=bool(warnings),
                        warnings=warnings or None,
                    )
                    logger.info("[%d/%d] 解析完成: %s (耗时 %.2fs)", i, pending_total, pdf_path.name, elapsed)
                    if warnings:
                        logger.warning(
                            "[%d/%d] 解析告警: %s — %s",
                            i,
                            pending_total,
                            pdf_path.name,
                            _format_warning_summary(warnings),
                        )
                else:
                    done.mark(
                        pdf_path.name, lang, "failed",
                        error_msg="后处理失败或结构化输出缺失",
                        lang_source=lang_source,
                        table_fallback_used=bool(warnings.get("table_fallback_used")),
                        done_with_warnings=bool(warnings),
                        warnings=warnings or None,
                    )
                    logger.error("[%d/%d] 后处理失败: %s (耗时 %.2fs)", i, pending_total, pdf_path.name, elapsed)
                    if warnings:
                        logger.error(
                            "[%d/%d] 失败上下文: %s — %s",
                            i,
                            pending_total,
                            pdf_path.name,
                            _format_warning_summary(warnings),
                        )
            else:
                done.mark(
                    pdf_path.name, lang, "failed",
                    error_msg=err or "非零退出码",
                    lang_source=lang_source,
                    table_fallback_used=bool(warnings.get("table_fallback_used")),
                    done_with_warnings=bool(warnings),
                    warnings=warnings or None,
                )
                logger.error(
                    "[%d/%d] 解析失败: %s (耗时 %.2fs) — %s",
                    i,
                    pending_total,
                    pdf_path.name,
                        elapsed,
                        err or "未知错误",
                    )
                if warnings:
                    logger.error(
                        "[%d/%d] 失败上下文: %s — %s",
                        i,
                        pending_total,
                        pdf_path.name,
                        _format_warning_summary(warnings),
                    )

    def _record_result(
        self,
        pdf_path: Path,
        lang: str,
        is_scanned: bool,
        lang_source: str,
        success: bool,
        err: str,
        warnings: dict,
        elapsed: float,
        completed: int,
        total: int,
        done: DoneRecord,
        output_dir: Path,
    ) -> None:
        """把 worker 的返回值写入 done.json 并打印日志。顺序与旧实现对齐。"""
        warnings = warnings or {}
        if success:
            post_ok = self._postprocess_if_needed(
                pdf_path, output_dir, "ocr" if is_scanned else self.parse_method
            )
            if post_ok:
                status = "done_without_tables" if warnings.get("table_fallback_used") else "done"
                done.mark(
                    pdf_path.name, lang, status,
                    lang_source=lang_source,
                    table_fallback_used=bool(warnings.get("table_fallback_used")),
                    done_with_warnings=bool(warnings),
                    warnings=warnings or None,
                )
                logger.info(
                    "[%d/%d] 解析完成: %s (总耗时含排队 %.2fs)",
                    completed, total, pdf_path.name, elapsed,
                )
                if warnings:
                    logger.warning(
                        "[%d/%d] 解析告警: %s — %s",
                        completed, total, pdf_path.name,
                        _format_warning_summary(warnings),
                    )
            else:
                done.mark(
                    pdf_path.name, lang, "failed",
                    error_msg="后处理失败或结构化输出缺失",
                    lang_source=lang_source,
                    table_fallback_used=bool(warnings.get("table_fallback_used")),
                    done_with_warnings=bool(warnings),
                    warnings=warnings or None,
                )
                logger.error(
                    "[%d/%d] 后处理失败: %s (总耗时含排队 %.2fs)",
                    completed, total, pdf_path.name, elapsed,
                )
                if warnings:
                    logger.error(
                        "[%d/%d] 失败上下文: %s — %s",
                        completed, total, pdf_path.name,
                        _format_warning_summary(warnings),
                    )
        else:
            done.mark(
                pdf_path.name, lang, "failed",
                error_msg=err or "非零退出码",
                lang_source=lang_source,
                table_fallback_used=bool(warnings.get("table_fallback_used")),
                done_with_warnings=bool(warnings),
                warnings=warnings or None,
            )
            logger.error(
                "[%d/%d] 解析失败: %s (总耗时含排队 %.2fs) — %s",
                completed, total, pdf_path.name, elapsed, err or "未知错误",
            )
            if warnings:
                logger.error(
                    "[%d/%d] 失败上下文: %s — %s",
                    completed, total, pdf_path.name,
                    _format_warning_summary(warnings),
                )

    def _parse_parallel(self, pending, output_dir, done):
        """并行解析入口：负责整体进度、BrokenProcessPool 重建与收尾。

        实际一轮跑在 ``_parse_parallel_batch`` 里，当外层 ProcessPool 整体
        broken 时返回未完成的 pending，这里再决定是否重建 executor 重试。
        """
        pending_total = len(pending)
        remaining = list(pending)
        progress = {"completed": 0}  # 跨 batch 共享完成计数
        attempts = 0

        while remaining:
            if attempts > 0:
                logger.warning(
                    "外层 ProcessPool 被判定 broken，第 %d 次重建 executor，剩余 %d 个文件",
                    attempts, len(remaining),
                )
                # broken 后稍等一下让 OS 把僵尸 worker 回收
                time.sleep(2)

            remaining = self._parse_parallel_batch(
                remaining, pending_total, progress, output_dir, done,
            )

            if not remaining:
                break

            attempts += 1
            if attempts >= _MAX_POOL_REBUILDS:
                logger.error(
                    "ProcessPool 连续 %d 次 broken，放弃剩余 %d 个文件",
                    _MAX_POOL_REBUILDS, len(remaining),
                )
                for item in remaining:
                    _, pdf_path, lang, _, lang_source = item
                    done.mark(
                        pdf_path.name, lang, "failed",
                        error_msg=f"ProcessPool 连续 {_MAX_POOL_REBUILDS} 次 broken，放弃",
                        lang_source=lang_source,
                    )
                break

    def _parse_parallel_batch(
        self,
        pending: list,
        pending_total: int,
        progress: dict,
        output_dir: Path,
        done: DoneRecord,
    ) -> list:
        """单次 executor 运行，滑动窗口提交 + 收集结果。

        Returns:
            broken 或中断时剩下的 pending 项；正常结束返回空列表。
        """
        logger.info(
            "启动 %d 个并发 worker，分布在 GPU %s 上，本轮待处理 %d 个",
            self.workers, self.gpu_ids, len(pending),
        )

        # spawn context 彻底切断 fork 带来的 fd / CUDA context 继承问题
        ctx = multiprocessing.get_context("spawn")

        gpu_queue = ctx.Queue()
        # 超量装填，防止 executor 重启 worker 时 get() 拿不到槽位
        slots = max(self.workers, len(self.gpu_ids)) * 4
        for i in range(slots):
            gpu_queue.put(self.gpu_ids[i % len(self.gpu_ids)])

        log_path = _discover_log_path()

        executor = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(gpu_queue, tuple(self.gpu_ids), str(log_path) if log_path else None),
        )

        pending_queue: list = list(pending)
        in_flight: dict = {}  # future -> (item, t_submit)
        window_size = max(self.workers * 2, 4)
        broken = False
        _interrupted = False

        tqdm = _get_tqdm()
        use_tqdm = os.environ.get("MINERU_TQDM", "1") != "0"
        pbar = (
            tqdm(
                total=pending_total,
                initial=min(progress["completed"], pending_total),
                desc="PDFs",
                unit="file",
            )
            if tqdm and use_tqdm and sys.stderr.isatty()
            else None
        )

        def _submit(item):
            _, pdf_path, lang, is_scanned, _lang_source = item
            parse_method = "ocr" if is_scanned else self.parse_method
            return executor.submit(
                subprocess_parse_one_smart,
                pdf_path_str=str(pdf_path),
                output_dir_str=str(output_dir),
                lang=lang,
                backend=self.backend,
                parse_method=parse_method,
                formula_enable=self.formula_enable,
                table_enable=self.table_enable,
                gpu_id=None,  # 交给 subprocess_worker 读 WORKER_GPU_ID
                timeout_sec=self.parse_timeout_sec,
                render_workers=self.render_workers,
            )

        def _fill_window():
            nonlocal broken
            while pending_queue and len(in_flight) < window_size and not broken:
                item = pending_queue[0]
                try:
                    fut = _submit(item)
                except BrokenProcessPool:
                    broken = True
                    break
                except Exception as exc:
                    # 提交阶段的普通异常：打失败后跳过
                    _, pdf_path, lang, _, lang_source = item
                    done.mark(
                        pdf_path.name, lang, "failed",
                        error_msg=f"submit 失败: {exc}",
                        lang_source=lang_source,
                    )
                    logger.exception("提交任务失败: %s", pdf_path.name)
                    pending_queue.pop(0)
                    progress["completed"] += 1
                    if pbar:
                        pbar.update(1)
                    continue
                pending_queue.pop(0)
                in_flight[fut] = (item, time.time())

        try:
            _fill_window()

            while in_flight and not broken:
                try:
                    done_set, _not_done = wait(
                        list(in_flight.keys()), return_when=FIRST_COMPLETED
                    )
                except KeyboardInterrupt:
                    _interrupted = True
                    raise

                for fut in done_set:
                    item, t0 = in_flight.pop(fut)
                    _, pdf_path, lang, is_scanned, lang_source = item
                    elapsed = time.time() - t0

                    try:
                        success, err, warnings = fut.result()
                    except BrokenProcessPool:
                        # 整个 pool 挂了：把这个条目放回 pending_queue，交给上层重建
                        pending_queue.insert(0, item)
                        broken = True
                        logger.error(
                            "检测到 BrokenProcessPool：%s 将在重建 executor 后重试",
                            pdf_path.name,
                        )
                        continue
                    except KeyboardInterrupt:
                        _interrupted = True
                        raise
                    except Exception as exc:
                        progress["completed"] += 1
                        if pbar:
                            pbar.update(1)
                        done.mark(
                            pdf_path.name, lang, "failed",
                            error_msg=str(exc) or type(exc).__name__,
                            lang_source=lang_source,
                        )
                        logger.exception(
                            "[%d/%d] worker 异常: %s (总耗时含排队 %.2fs)",
                            progress["completed"], pending_total, pdf_path.name, elapsed,
                        )
                        continue

                    progress["completed"] += 1
                    if pbar:
                        pbar.update(1)
                    self._record_result(
                        pdf_path, lang, is_scanned, lang_source,
                        success, err, warnings, elapsed,
                        progress["completed"], pending_total, done, output_dir,
                    )

                if broken:
                    break

                _fill_window()

        except KeyboardInterrupt:
            _interrupted = True
            logger.warning("并行模式收到中断，取消未开始的任务...")
            raise
        finally:
            if pbar:
                pbar.close()
            cancel = _interrupted or broken
            try:
                executor.shutdown(wait=not cancel, cancel_futures=cancel)
            except Exception:
                # broken 的 pool shutdown 可能也抛异常，忽略以保证 finally 走完
                logger.debug("executor.shutdown 期间忽略异常", exc_info=True)
            try:
                gpu_queue.close()
                gpu_queue.join_thread()
            except Exception:
                pass

        # 把还没来得及完成的条目交回上层决定是否重试
        unfinished = list(pending_queue) + [item for (item, _) in in_flight.values()]
        return unfinished if broken else []

    def collect_md_files(self, output_dir: Path, subdir_name: str) -> None:
        md_target = self.input_root / "md" / subdir_name
        md_target.mkdir(parents=True, exist_ok=True)

        md_files: list[Path] = []
        for top_dir in sorted(output_dir.iterdir()):
            if not top_dir.is_dir():
                continue
            stem = top_dir.name
            exact_hits = [p for p in top_dir.rglob(f"{stem}.md") if p.stem == stem]
            if exact_hits:
                exact_hits.sort(key=lambda p: (len(p.parts), len(str(p))))
                md_files.append(exact_hits[0])
            else:
                logger.warning("目录 %s 未找到主 md（%s.md），跳过收集，避免误收中间文件", top_dir, stem)

        if not md_files:
            return

        collected = 0
        for md_file in md_files:
            doc_target_dir = md_target / md_file.stem
            doc_target_dir.mkdir(parents=True, exist_ok=True)

            dest_md = doc_target_dir / md_file.name
            shutil.copy2(md_file, dest_md)

            md_parent = md_file.parent
            structured_json = md_parent / f"{md_file.stem}_structured.json"
            if structured_json.exists() and structured_json.is_file():
                shutil.copy2(structured_json, doc_target_dir / structured_json.name)
            for res_dir_name in ["images", "figures", "equations"]:
                res_dir = md_parent / res_dir_name
                if res_dir.exists() and res_dir.is_dir():
                    dest_res_dir = doc_target_dir / res_dir_name
                    if dest_res_dir.exists():
                        shutil.rmtree(dest_res_dir)
                    shutil.copytree(res_dir, dest_res_dir)

            collected += 1

        logger.info("共收集 %d 个新 md 项目（含图片）到 %s", collected, md_target)

    def _postprocess_if_needed(self, pdf_path: Path, output_dir: Path, parse_method: str) -> bool:
        if not self.postprocess_enable:
            return True
        try:
            result = build_structured_json(
                pdf_path,
                output_dir,
                parse_method=parse_method,
                biblio_provider=self.biblio_provider,
                keep_raw=self.keep_raw,
            )
            return result is not None and result.exists()
        except Exception as exc:
            logger.warning("后处理失败: %s (%s)", pdf_path.name, exc)
            return False
