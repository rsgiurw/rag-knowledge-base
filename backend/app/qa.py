"""问答组装：检索 → 拼 prompt → 调 GLM → 返回带引用的结果。

和 scripts/03_minimal_rag.py 是同一套路，只是从"命令行脚本"搬进了"HTTP 服务"——
逻辑复用 rag/ 与 app/store.py，本文件只负责组装。
"""
from app import store
from rag.llm import chat_once

SYSTEM_PROMPT = (
    "你是企业知识库助手。只根据【参考资料】回答问题，不使用资料之外的知识；"
    "资料里没有答案就明确说'知识库中没有相关内容'。"
    "回答末尾单独一行列出引用来源，格式：来源：文件名（可多个）。"
)


def _build_context(hits: list[dict]) -> str:
    """把检索命中的块拼成资料文本，每块带上来源标签——模型引用溯源的全部秘密在这里。"""
    return "\n\n".join(
        f"[来源:{h['source']}·第{h['start']}字起] {h['text']}" for h in hits
    )


def answer_question(question: str, top_k: int = 3, history: list[dict] | None = None) -> dict:
    """完整问答链路。history 是多轮对话的上文（Day 4 接上）。"""
    hits = store.search(question, top_k)
    if not hits:
        return {"answer": "知识库是空的，请先上传文档。", "citations": []}

    user_content = f"【参考资料】\n{_build_context(hits)}\n\n【问题】\n{question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *(history or []),
        {"role": "user", "content": user_content},
    ]
    answer = chat_once(messages, temperature=0.1)
    return {"answer": answer, "citations": hits}
