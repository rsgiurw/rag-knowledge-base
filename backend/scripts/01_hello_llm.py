"""Day 1 · 裸调 GLM API（不用 SDK，直接发 HTTP 请求）

目的：看清一次 LLM 调用的真实面目，弄懂三件事——
    1. messages 的角色体系：system / user / assistant
    2. 参数的作用：temperature
    3. 流式输出（SSE）到底是怎么一回事

运行（在 backend 目录下）：
    .venv/Scripts/python scripts/01_hello_llm.py

运行前：把 .env.example 复制为 .env，填入你的 API Key。
"""
import json
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("ZHIPU_API_KEY", "")
URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL = "glm-4-flash"  # 免费模型


def chat_once(messages: list[dict], temperature: float = 0.7) -> str:
    """非流式：发一个请求，等模型全部生成完，一次性返回。"""
    resp = httpx.post(
        URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": MODEL, "messages": messages, "temperature": temperature},
        timeout=60,
    )
    resp.raise_for_status()  # 非 2xx 状态码直接抛错（429=频率超限，401=Key 错误）
    data = resp.json()

    # 动手：取消下面这行注释，看一眼 API 的原始返回结构——
    # 你会看到 usage 字段里的 prompt_tokens / completion_tokens，这就是计费单位
    # print(json.dumps(data, ensure_ascii=False, indent=2))

    return data["choices"][0]["message"]["content"]


def chat_stream(messages: list[dict]) -> str:
    """流式：服务器边生成边推，客户端逐行收——打字机效果的来源。"""
    pieces = []
    with httpx.stream(
        "POST",
        URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": MODEL, "messages": messages, "stream": True},
        timeout=60,
    ) as resp:
        for line in resp.iter_lines():
            # SSE 协议：每个数据帧以 "data: " 开头， "[DONE]" 表示结束
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            delta = json.loads(payload)["choices"][0]["delta"]
            piece = delta.get("content") or ""
            print(piece, end="", flush=True)  # flush 立刻上屏，才能看到逐字效果
            pieces.append(piece)
    print()
    return "".join(pieces)


if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit("缺少 ZHIPU_API_KEY：请复制 .env.example 为 .env 并填入 Key")

    messages = [
        # system：设定模型的行为，优先级高于用户指令
        {"role": "system", "content": "你是一个简洁的助手，回答不超过 3 句话。"},
        {"role": "user", "content": "用一句话解释什么是 RAG？"},
    ]

    print("== 1) 非流式调用 ==")
    print(chat_once(messages))

    print("\n== 2) 流式调用（观察打字机效果）==")
    chat_stream(messages)

    # 实验 1：temperature 对生成稳定性的影响
    # 把 temperature 分别设为 0.0 和 1.5，各跑 3 次，对比输出差异
    print("\n== 3) 实验：temperature=0.0 时同一问题连问 3 次 ==")
    for i in range(3):
        ans = chat_once(
            [{"role": "user", "content": "给企业知识库产品随机起一个中文名字"}],
            temperature=1.5,
        )
        print(f"  第{i + 1}次: {ans}")
    print("\n再把 0.0 改成 1.5 各跑 3 次，把观察结论写进今晚的复盘笔记。")
