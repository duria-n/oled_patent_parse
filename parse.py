"""专利 PDF 批量解析工具 — CLI 入口。"""

import argparse
import logging
import signal
import sys

from patent_parser.config import SUPPORTED_LANGS
from patent_parser.mineru_parser import MinerUPatentParser

logger = logging.getLogger("patent_parser")

DEFAULT_LANGS = "ch,chinese_cht,japan,en,korean,latin,cyrillic,arabic"


def _parse_langs(value: str) -> list[str]:
    """解析逗号分隔的语言列表并校验。"""
    langs = [l.strip() for l in value.split(",") if l.strip()]
    valid = set(SUPPORTED_LANGS.keys()) - {"auto"}
    for lang in langs:
        if lang not in valid:
            raise argparse.ArgumentTypeError(
                f"不支持的语言 '{lang}'。可选: {', '.join(sorted(valid))}"
            )
    return langs


def _parse_gpus(value: str) -> list[int]:
    """解析逗号分隔的 GPU 编号列表，如 '0,1,2,3'。"""
    try:
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"GPU 编号格式错误 '{value}'，应为逗号分隔的整数，如 0,1,2,3"
        )


def main():
    lang_help_lines = [f"  {k}: {v}" for k, v in SUPPORTED_LANGS.items() if k != "auto"]
    lang_help = (
        f"OCR 语言列表，逗号分隔，默认 {DEFAULT_LANGS}。\n"
        "根据专利文件名前缀（CN/JP/US/KR 等）自动判断语言，\n"
        "无法识别时使用列表中第一个作为默认。\n"
        "可选值:\n" + "\n".join(lang_help_lines)
    )

    ap = argparse.ArgumentParser(
        description="基于 MinerU 的专利 PDF 批量解析工具",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("-i", "--input", required=True,
                    help="输入主文件夹路径（包含多个子文件夹）")
    ap.add_argument("-o", "--output", default=None,
                    help="输出主文件夹路径（可选，默认在输入根目录下创建 output/）")
    ap.add_argument("-l", "--lang", default=DEFAULT_LANGS, type=_parse_langs,
                    help=lang_help)
    ap.add_argument("-b", "--backend", default="pipeline",
                    help="解析后端，默认 pipeline")
    ap.add_argument("-m", "--method", default="auto",
                    help="解析方法: auto/txt/ocr，默认 auto")
    ap.add_argument("-w", "--workers", type=int, default=0,
                    help="并发 worker 数（默认 0 = 自动匹配 GPU 数量）")
    ap.add_argument("-g", "--gpus", default=None, type=_parse_gpus,
                    help="指定使用的 GPU 编号，逗号分隔，如 0,1,2,3（默认使用全部）")
    ap.add_argument("--no-formula", action="store_true", help="禁用公式解析")
    ap.add_argument("--no-table", action="store_true", help="禁用表格解析")
    ap.add_argument("--wipo-metadata", default=None,
                    help="WO 官方语言元数据缓存 JSON 路径（优先使用 publication_language）")
    ap.add_argument("--biblio-metadata", default=None,
                    help="题录元数据缓存 JSON 路径（注入 metadata 字段）")
    ap.add_argument("--keep-raw", action="store_true",
                    help="保留原始 raw 字段（默认会丢弃以减小 JSON 体积）")
    ap.add_argument("--no-postprocess", action="store_true",
                    help="禁用结构化后处理 JSON 输出")
    args = ap.parse_args()

    lang_display = ",".join(args.lang)
    logger.info("语言: %s | 后端: %s | 方法: %s | workers: %d",
                lang_display, args.backend, args.method, args.workers)
    logger.info("公式解析: %s | 表格解析: %s",
                "开启" if not args.no_formula else "关闭",
                "开启" if not args.no_table else "关闭")
    if args.gpus:
        logger.info("指定 GPU: %s", args.gpus)

    parser = MinerUPatentParser(
        input_root=args.input,
        output_root=args.output,
        langs=args.lang,
        backend=args.backend,
        parse_method=args.method,
        formula_enable=not args.no_formula,
        table_enable=not args.no_table,
        workers=args.workers,
        gpu_ids=args.gpus,
        wipo_metadata_path=args.wipo_metadata,
        postprocess_enable=not args.no_postprocess,
        biblio_metadata_path=args.biblio_metadata,
        keep_raw=args.keep_raw,
    )

    # 捕获 Ctrl+C：打印提示后正常退出，让底层清理逻辑（finally 块）有机会执行
    def _sigint_handler(signum, frame):
        logger.warning("收到中断信号（Ctrl+C），正在退出，请稍候...")
        # 将信号重置为默认，再次 Ctrl+C 可强制杀死
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        # 抛出 KeyboardInterrupt 让 Python 正常展开调用栈（触发所有 finally）
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        parser.run()
    except KeyboardInterrupt:
        logger.warning("已中断，当前进度已保存到 done.json，下次运行将自动跳过已完成文件")
        sys.exit(130)  # 130 = 128 + SIGINT(2)，符合 Unix 惯例


if __name__ == "__main__":
    main()
