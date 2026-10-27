"""GLM API 客户端（工程版，全项目共用）。

scripts/01_hello_llm.py 是"教学版"：裸写 HTTP 调用，让你看清每次请求的细节；
本模块是"工程版"：把同样的逻辑封装成可复用函数，Day 2 起统一从这里调用。
两者做的事完全一样——体会一下"封装"解决的是什么问题。
"""
import json
import os

import httpx
from dotenv import load_dotenv

load_dotenv()  # 读取项目根目录 .env 中的环境变量

API_KEY = os.environ.get("ZHIPU_API_KEY", "")
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
CHAT_MODEL = "glm-4-flash"       # 免费模型，开发学习够用；追求质量可换 glm-4-plus
EMBEDDING_MODEL = "embedding-3"  # 输出 2048 维向量，按量计费、价格极低


def _headers() -> dict:
    if not API_KEY:
        raise RuntimeError(
            "缺少 ZHIPU_API_KEY：请把 .env.example 复制为 .env 并填入你的 Key"
        )
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def chat_once(messages: list[dict], temperature: float = 0.3) -> str:
    """非流式调用：一次性拿到完整回答。

    temperature 越低回答越稳定，知识库问答场景用低值（默认 0.3）；
    创意生成场景才调高。
    """
    resp = httpx.post(
        f"{BASE_URL}/chat/completions",
        headers=_headers(),
        json={
            "model": CHAT_MODEL,
            "messages": messages,
            "temperature": temperature,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def chat_stream(messages: list[dict], temperature: float = 0.3):
    """流式调用：生成器，逐段 yield 回答片段。

    底层是 SSE（Server-Sent Events）：服务器把回答切成小块，
    每块用 `data: {...}` 帧推过来，直到 `data: [DONE]`。
    这就是网页版"打字机效果"的来源。
    """
    with httpx.stream(
        "POST",
        f"{BASE_URL}/chat/completions",
        headers=_headers(),
        json={
            "model": CHAT_MODEL,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        },
        timeout=120,
    ) as resp:
        for line in resp.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            delta = json.loads(payload)["choices"][0]["delta"]
            piece = delta.get("content")
            if piece:
                yield piece


def embed(texts: list[str]) -> list[list[float]]:
    """把一批文本变成向量。返回顺序与输入顺序一一对应。"""
    resp = httpx.post(
        f"{BASE_URL}/embeddings",
        headers=_headers(),
        json={"model": EMBEDDING_MODEL, "input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]
