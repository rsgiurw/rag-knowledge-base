"""Pydantic 模型：接口的"数据合同"。

FastAPI 拿它们做三件事（讲义第三节）：
  1. 自动校验请求（字段缺失/类型错/超范围 → 自动返回 422，不用我们写 if）；
  2. 自动生成 /docs 交互文档；
  3. 响应序列化与类型标注。
"""
from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunks: int


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500, description="用户问题")
    top_k: int = Field(default=3, ge=1, le=10, description="检索返回的资料块数")


class Citation(BaseModel):
    """引用溯源：答案出自哪篇文档的哪个位置。"""
    doc_id: str
    source: str
    start: int
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
