"""RAG 核心两件套：分块与余弦相似度。

⚠️ 协作约定（AGENTS.md）：本文件的两个函数是 Day 2 上午【你手写】的部分。
我只给规格说明和单元测试，不给实现。写完之后运行：

    cd backend
    .venv/Scripts/python -m pytest tests/ -v

测试全绿，就说明你的实现是对的。
"""
import numpy as np


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[dict]:
    """把长文本切成带重叠的分块（固定长度策略，最朴素的起点）。

    返回格式：
        [{"text": "分块内容", "start": 该块首字符在原文中的下标}, ...]

    规格要求：
        1. 除最后一块外，每块长度 == chunk_size（最后一块可以更短）；
        2. 第 i+1 块的 start = 第 i 块的 start + chunk_size - overlap；
        3. 任何字符都不丢：相邻块恰好重叠 overlap 个字符，
           后一块的 start + len(text) 会推进到文末；
        4. 空字符串输入返回 []。

    提示：一个 while 循环 + 字符串切片就够，不需要任何第三方库。
    思考：为什么需要 overlap？（答案在 docs/day01-讲义.md，写完代码再看）
    """
    raise NotImplementedError("Day 2 上午任务：删掉这行，写你的实现")


def cosine_similarity(a, b) -> float:
    """余弦相似度：cos = (a·b) / (|a| * |b|)

    衡量两个向量"方向"的接近程度：
        1.0 = 方向相同（语义最接近）
        0.0 = 互相垂直（语义无关）

    要求：用 numpy 实现（两三行）；并且你要能在纸上手推这个公式——
    面试常问"为什么检索用余弦相似度而不是欧氏距离"。
    """
    raise NotImplementedError("Day 2 上午任务：删掉这行，写你的实现")
