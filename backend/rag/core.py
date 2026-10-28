"""RAG 核心两件套：分块与余弦相似度（Day 2 · 填空闯关版）

按 AGENTS.md 的新教学法分三步，**不需要你从零写**：

  第 1 步 ✅ 前置：先通关 scripts/00_python_warmup.py 的 10 关热身；
  第 2 步 👉 现在：把下面 4 个 TODO 填掉（每个都是一行），跑测试全绿；
  第 3 步 ⏭ 之后：回来找我做"逐行走读"，明天用"默写版"复现一遍。

验收（在 backend 目录）：
    .venv/Scripts/python -m pytest tests/ -v

8 个测试全绿 = 通关。测试就是需求文档，改完随时重跑。
"""
import numpy as np


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[dict]:
    """固定长度分块：返回 [{"text": 分块内容, "start": 起始下标}, ...]

    规格回顾（tests/test_core.py 会逐条验收）：
      1. 每块最长 chunk_size，最后一块可以更短；
      2. 步进（下一块 start 的增量）= chunk_size - overlap；
      3. 相邻块恰好重叠 overlap 个字符；
      4. 空字符串返回 []。
    """
    chunks = []
    start = 0                       # 第一块从下标 0 开始
    while start < len(text):        # 只要还没切到文末，就继续切
        piece = text[start:start+chunk_size]   # TODO-1: 应该取多长？改成正确的切片终点
        chunks.append({"text": piece, "start": start})
        start += chunk_size - overlap              # TODO-2: 删掉这行 break，替换成"把 start 推进到下一块起点"的语句（热身第 8 关）
    return chunks


def cosine_similarity(a, b) -> float:
    """余弦相似度 = 点积 / (模长 × 模长)。

    面试要求：能白板推出这个公式，并说清"为什么用余弦不用欧氏距离"
    （答案方向在 docs/day01-讲义.md 第四节，走读时我会抽查）。
    """
    dot = a@b    # TODO-3: 向量 a 与 b 的点积（第 10 关出现过）
    norm = np.linalg.norm(a)*np.linalg.norm(b)   # TODO-4: a 的模长 乘 b 的模长
    return float(dot / norm)
