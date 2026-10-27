"""rag/core.py 的单元测试——Day 2 你写实现，这份测试验收。

运行（在 backend 目录下）：
    .venv/Scripts/python -m pytest tests/ -v

注意用 `python -m pytest` 而不是直接 `pytest`：
`python -m` 会把当前目录加进模块搜索路径，`from rag.core import ...` 才能找到。
"""
import numpy as np

from rag.core import chunk_text, cosine_similarity


# ---------- chunk_text ----------

def test_chunk_empty():
    assert chunk_text("") == []


def test_chunk_basic():
    text = "甲乙" * 400  # 800 字
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    # 规格要求 1：块长不超过 chunk_size（最后一块可以短）
    assert all(len(c["text"]) <= 300 for c in chunks)
    # 规格 2：步进 = chunk_size - overlap = 250
    assert chunks[0]["start"] == 0
    assert chunks[1]["start"] == 250


def test_chunk_overlap_is_real():
    text = "甲乙" * 400
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    # 相邻两块真的重叠了 50 个字符：前块结尾 == 后块开头
    assert chunks[0]["text"][250:] == chunks[1]["text"][:50]


def test_chunk_covers_whole_text():
    text = "甲乙" * 375  # 750 字
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    # 规格 3：最后一块要顶到文末，一个字都不丢
    assert chunks[-1]["start"] + len(chunks[-1]["text"]) == len(text)


def test_chunk_short_text():
    # 短文本只出一块
    chunks = chunk_text("很短的文档", chunk_size=300, overlap=50)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "很短的文档"


# ---------- cosine_similarity ----------

def test_cosine_same_direction():
    v = cosine_similarity(np.array([1.0, 0.0]), np.array([3.0, 0.0]))
    assert abs(v - 1.0) < 1e-9


def test_cosine_orthogonal():
    v = cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 5.0]))
    assert abs(v) < 1e-9


def test_cosine_opposite():
    v = cosine_similarity(np.array([1.0, 0.0]), np.array([-2.0, 0.0]))
    assert abs(v - (-1.0)) < 1e-9
