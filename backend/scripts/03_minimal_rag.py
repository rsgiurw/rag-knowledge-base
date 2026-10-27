"""Day 2 主任务 · 最小可用 RAG（CLI 版）

完整链路（离线建库 + 在线问答）：
    data/raw_docs/*.md
      → chunk_text()      分块          【你手写】
      → embed()           向量化        【API】
    用户提问
      → embed(问题)
      → cosine_similarity() 找 top-k     【你手写】
      → 拼进 Prompt → GLM 生成带引用的回答

在你完成 rag/core.py 之前，本脚本会抛 NotImplementedError——这是设计好的，
先跑一次亲眼看报错，写完 core.py 再跑通它，你会对"链路"有体感。

运行（在 backend 目录下）：
    .venv/Scripts/python scripts/03_minimal_rag.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rag.core import chunk_text, cosine_similarity
from rag.llm import chat_once, embed

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw_docs"

PROMPT_TEMPLATE = """你是一个企业知识库助手。遵守以下规则：
1. 只根据【参考资料】回答，不允许使用资料之外的知识；
2. 资料里没有答案时，明确回答"知识库中没有相关内容"；
3. 回答末尾单独一行列出引用来源，格式：来源：文件名（可多个）。

【参考资料】
{context}

【问题】
{question}"""


def build_knowledge_base(chunk_size: int = 300, overlap: int = 50):
    """离线索引：读文档 → 分块（你写的）→ 向量化（API）。"""
    chunks = []
    for f in sorted(DATA_DIR.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        for c in chunk_text(text, chunk_size, overlap):
            chunks.append({"source": f.name, **c})
    vectors = np.array(embed([c["text"] for c in chunks]))
    print(f"知识库就绪：{len(DATA_DIR.glob('*.md'))} 篇文档 → {len(chunks)} 个分块")
    return chunks, vectors


def retrieve(question: str, chunks: list[dict], vectors, top_k: int = 3) -> list[dict]:
    """在线检索：问题向量化 → 逐块算相似度（你写的）→ 取前 k。"""
    q_vec = np.array(embed([question]))[0]
    scores = [cosine_similarity(q_vec, v) for v in vectors]
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{**chunks[i], "score": scores[i]} for i in order]


def main():
    chunks, vectors = build_knowledge_base()
    history = []
    while True:
        question = input("\n提问（输入 q 退出）> ").strip()
        if not question or question.lower() == "q":
            break

        hits = retrieve(question, chunks, vectors)

        print("\n-- 检索命中（看看检索准不准，这比答案本身更重要）--")
        for h in hits:
            print(f"  {h['source']} @第{h['start']}字  score={h['score']:.3f}")
            print(f"    {h['text'][:60]}...")

        context = "\n\n".join(
            f"[来源:{h['source']}] {h['text']}" for h in hits
        )
        messages = [
            {"role": "system", "content": "你是企业知识库助手，只根据给定资料回答。"},
            *history,
            {"role": "user", "content": PROMPT_TEMPLATE.format(context=context, question=question)},
        ]
        answer = chat_once(messages, temperature=0.1)

        print("\n-- 回答 --")
        print(answer)
        history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ][-2:]  # 保留最近一轮，控制上下文长度


if __name__ == "__main__":
    main()
