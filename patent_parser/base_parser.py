"""PDF 解析器抽象基类。"""

from abc import ABC, abstractmethod
from pathlib import Path

from .config import logger


class BasePDFParser(ABC):
    """PDF 解析器抽象基类，定义解析流程骨架（模板方法模式）。"""

    def __init__(self, input_root: str, output_root: str | None = None):
        self.input_root = Path(input_root).resolve()
        self.output_root = Path(output_root).resolve() if output_root else None

        # 这些目录不作为输入子文件夹
        self._SKIP_DIRS = {"output", "md"}

        # 如果指定了自定义输出目录，且它位于输入目录内部，将其加入忽略列表
        if self.output_root and self.input_root in self.output_root.parents:
            rel_parts = self.output_root.relative_to(self.input_root).parts
            if rel_parts:
                self._SKIP_DIRS.add(rel_parts[0])

    def run(self) -> None:
        subdirs = self.discover_subdirs()
        if not subdirs:
            logger.warning("未在 %s 下发现含有 PDF 的子文件夹", self.input_root)
            return
        for subdir in subdirs:
            # 处理根目录的情况
            if subdir == self.input_root:
                rel_str = "根目录"
            else:
                rel_str = str(subdir.relative_to(self.input_root))
                
            logger.info("===== 开始处理: %s =====", rel_str)
            pdf_files = self.collect_pdfs(subdir)
            if not pdf_files:
                logger.warning("目录 %s 中没有 PDF 文件，跳过", rel_str)
                continue
            output_dir = self.prepare_output_dir(subdir)
            self.parse_pdfs(pdf_files, output_dir)
            self.collect_md_files(output_dir, "root" if subdir == self.input_root else rel_str)
            logger.info("===== %s 处理完毕 =====\n", rel_str)

    def discover_subdirs(self) -> list[Path]:
        """递归收集所有直接包含 PDF 文件的目录（支持多级嵌套）。"""
        result: list[Path] = []
        
        # 修复：先检查根目录自身是否含有 PDF
        if any(self.input_root.glob("*.pdf")):
            result.append(self.input_root)
            
        self._collect_pdf_dirs(self.input_root, is_root=True, result=result)
        
        # 去重并排序
        unique_result = list(set(result))
        return sorted(unique_result, key=lambda p: str(p))

    def _collect_pdf_dirs(self, directory: Path, is_root: bool, result: list[Path]) -> None:
        """深度优先遍历，将含有 PDF 的目录加入 result。"""
        try:
            entries = sorted(directory.iterdir(), key=lambda p: p.name)
        except PermissionError:
            logger.warning("无权限访问目录: %s，跳过", directory)
            return

        for entry in entries:
            if not entry.is_dir():
                continue
            # 顶层目录中的跳过目录（output/、md/ 等）整棵跳过
            if is_root and entry.name in self._SKIP_DIRS:
                logger.debug("跳过保留目录: %s", entry.name)
                continue
                
            # 如果该目录直接含有 PDF，记录它
            if any(entry.glob("*.pdf")):
                result.append(entry)
                
            # 修复：移除 else，无论当前目录有没有 PDF，都强制向下递归查找更深层
            self._collect_pdf_dirs(entry, is_root=False, result=result)

    def collect_pdfs(self, subdir: Path) -> list[Path]:
        return sorted(subdir.glob("*.pdf"))

    @abstractmethod
    def prepare_output_dir(self, subdir: Path) -> Path:
        """子类必须实现：为子文件夹准备输出目录。"""

    @abstractmethod
    def parse_pdfs(self, pdf_files: list[Path], output_dir: Path) -> None:
        """子类必须实现：解析 PDF 列表并将结果写入 output_dir。"""

    @abstractmethod
    def collect_md_files(self, output_dir: Path, subdir_name: str) -> None:
        """子类必须实现：将解析产出的 .md 文件汇总到统一目录。"""