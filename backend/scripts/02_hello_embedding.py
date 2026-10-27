"""Day 1 · Embedding：把文字变成向量

原理：embedding 模型把一段文本映射成一个固定长度的数值向量（本项目用 2048 维），
语义越相近的文本，向量方向越接近。这是整个 RAG"语义检索"的地基。

运行（在 backend 目录下）：
    .venv/Scripts/python scripts/02_hello_embedding.py
"""
import sys
from pathlib import Path

# 让脚本能找到 backend/ 下的 rag 包（Python 只会把"脚本所在目录"加进模块搜索路径，
# 不会自动加当前工作目录，所以需要手动补上）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rag.llm import embed

if __name__ == "__main__":
    words = ["猫", "小猫", "狗", "小猫喜欢抓老鼠"]
    vectors = np.array(embed(words))

    print(f"模型返回的向量维度：{vectors.shape[1]}")

    # 以第 0 个词"猫"为基准，算它和其余词的余弦相似度
    base = vectors[0]
    for i in range(1, len(words)):
        cos = float(base @ vectors[i] / (np.linalg.norm(base) * np.linalg.norm(vectors[i])))
        print(f"  '猫' vs '{words[i]}': {cos:.3f}")

    # 思考题（写进今晚复盘）：
    # 1. '猫'和'小猫喜欢抓老鼠'的相似度为什么远高于'猫'和'汽车'？
    # 2. 但'猫'和'狗'都是动物，相似度却可能不如带'小猫'字面的句子——
    #    这暴露了向量检索的什么短板？（提示：这正是 Week 2 要做混合检索的原因）
