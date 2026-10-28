"""文档解析：把上传的文件变成纯文本。分块之前的最后一站。

设计要点：解析、分块、向量化各管一段，谁也不掺和谁——
这样以后换解析器（比如支持 Excel）或换分块策略，都不用动别的代码。
"""
import io
from pathlib import Path

from docx import Document
from pypdf import PdfReader

SUPPORTED_EXTS = (".md", ".txt", ".pdf", ".docx")


def parse_file(filename: str, raw: bytes) -> str:
    """根据扩展名选择解析器，返回纯文本。不支持的类型抛 ValueError。"""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"不支持的文件类型 {ext}，目前支持：{'/'.join(SUPPORTED_EXTS)}")

    if ext in (".md", ".txt"):
        # 文本类：直接解码（errors='ignore' 容忍个别坏字节）
        return raw.decode("utf-8", errors="ignore")

    if ext == ".pdf":
        # PDF：pypdf 逐页抽取文字再拼接（扫描件图片型 PDF 抽不出文字，这是已知局限）
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    # docx：本质是 zip 包，python-docx 帮我们解出段落
    doc = Document(io.BytesIO(raw))
    return "\n".join(p.text for p in doc.paragraphs)
