"""MinerU 专利解析器具体实现。"""

import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .base_parser import BasePDFParser
from .config import logger
from .done_record import DoneRecord
from .lang_detect import detect_pdf_language
from .biblio_cache import BiblioMetadataProvider
from .postprocess import build_structured_json
from .wipo_metadata import WIPOMetadataProvider
from .subprocess_worker import subprocess_parse_one_smart


def _get_gpu_count() -> int:
    """获取可用 GPU 数量。"""
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _init_worker(gpu_queue):
    """Worker 初始化函数：从队列获取一个固定的 GPU ID 并绑定到环境变量。"""
    # 阻塞获取，确保每个 worker 进程启动时都能分配到专属的 GPU
    gpu_id = gpu_queue.get()
    os.environ["WORKER_GPU_ID"] = str(gpu_id)


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

        total_gpus = _get_gpu_count()
        if gpu_ids is not None:
            self.gpu_ids = gpu_ids
        else:
            self.gpu_ids = list(range(total_gpus)) if total_gpus > 0 else [0]

        # workers 默认等于指定 GPU 数量
        self.workers = workers if workers > 0 else len(self.gpu_ids)
        logger.info("检测到 %d 张 GPU，使用 GPU %s，%d 个 worker",
                     total_gpus, self.gpu_ids, self.workers)
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
        )

    def parse_pdfs(self, pdf_files: list[Path], output_dir: Path) -> None:
        done = DoneRecord(output_dir)
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
        for i, (idx, pdf_path, lang, is_scanned, lang_source) in enumerate(pending, 1):
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
                    done.mark(
                        pdf_path.name, lang, "done",
                        lang_source=lang_source,
                        table_fallback_used=bool(warnings.get("table_fallback_used")),
                        done_with_warnings=bool(warnings),
                        warnings=warnings or None,
                    )
                    logger.info("[%d/%d] 解析完成: %s (耗时 %.2fs)", i, pending_total, pdf_path.name, elapsed)
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
            else:
                done.mark(
                    pdf_path.name, lang, "failed",
                    error_msg=err or "非零退出码",
                    lang_source=lang_source,
                    table_fallback_used=bool(warnings.get("table_fallback_used")),
                    done_with_warnings=bool(warnings),
                    warnings=warnings or None,
                )
                logger.error("[%d/%d] 解析失败（segfault/超时）: %s (耗时 %.2fs)", i, pending_total, pdf_path.name, elapsed)

    def _parse_parallel(self, pending, output_dir, done):
        pending_total = len(pending)
        logger.info("启动 %d 个并发 worker，分布在 GPU %s 上", self.workers, self.gpu_ids)
        
        # 修复并发 OOM：使用队列将 GPU 资源绑定到固定的 worker 进程
        m = multiprocessing.Manager()
        gpu_queue = m.Queue()
        for i in range(self.workers):
            gpu_queue.put(self.gpu_ids[i % len(self.gpu_ids)])

        executor = ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=_init_worker,
            initargs=(gpu_queue,)
        )
        future_map = {}
        _interrupted = False
        try:
            for i, (idx, pdf_path, lang, is_scanned, lang_source) in enumerate(pending):
                parse_method = "ocr" if is_scanned else self.parse_method
                future = executor.submit(
                    subprocess_parse_one_smart,
                    pdf_path_str=str(pdf_path),
                    output_dir_str=str(output_dir),
                    lang=lang,
                    backend=self.backend,
                    parse_method=parse_method,
                    formula_enable=self.formula_enable,
                    table_enable=self.table_enable,
                    gpu_id=None,  # 让 subprocess_worker 去读取进程注入的环境变量
                )
                future_map[future] = (pdf_path, lang, is_scanned, lang_source, time.time())

            completed = 0
            for future in as_completed(future_map):
                completed += 1
                pdf_path, lang, is_scanned, lang_source, t0 = future_map[future]
                elapsed = time.time() - t0
                try:
                    success, err, warnings = future.result()
                    if success:
                        post_ok = self._postprocess_if_needed(
                            pdf_path, output_dir, "ocr" if is_scanned else self.parse_method
                        )
                        if post_ok:
                            done.mark(
                                pdf_path.name, lang, "done",
                                lang_source=lang_source,
                                table_fallback_used=bool(warnings.get("table_fallback_used")),
                                done_with_warnings=bool(warnings),
                                warnings=warnings or None,
                            )
                            logger.info("[%d/%d] 解析完成: %s (耗时 %.2fs)", completed, pending_total, pdf_path.name, elapsed)
                        else:
                            done.mark(
                                pdf_path.name, lang, "failed",
                                error_msg="后处理失败或结构化输出缺失",
                                lang_source=lang_source,
                                table_fallback_used=bool(warnings.get("table_fallback_used")),
                                done_with_warnings=bool(warnings),
                                warnings=warnings or None,
                            )
                            logger.error("[%d/%d] 后处理失败: %s (耗时 %.2fs)", completed, pending_total, pdf_path.name, elapsed)
                    else:
                        done.mark(
                            pdf_path.name, lang, "failed",
                            error_msg=err or "非零退出码",
                            lang_source=lang_source,
                            table_fallback_used=bool(warnings.get("table_fallback_used")),
                            done_with_warnings=bool(warnings),
                            warnings=warnings or None,
                        )
                        logger.error("[%d/%d] 解析失败: %s (耗时 %.2fs)", completed, pending_total, pdf_path.name, elapsed)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    done.mark(
                        pdf_path.name, lang, "failed",
                        error_msg=str(exc) or type(exc).__name__,
                        lang_source=lang_source,
                    )
                    logger.exception("[%d/%d] worker 异常: %s (耗时 %.2fs)", completed, pending_total, pdf_path.name, elapsed)

        except KeyboardInterrupt:
            _interrupted = True
            logger.warning("并行模式收到中断，取消未开始的任务...")
            raise
        finally:
            executor.shutdown(wait=not _interrupted, cancel_futures=_interrupted)

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
