"""Qdrant 向量库封装（本地嵌入式模式）：存向量、查向量。

【选型记录 2026-10-28，面试可讲】最初选 chromadb，本机踩坑后更换：
  - chromadb 1.5.9 在 Windows 上 collection.add() 段错误（原生库崩溃，服务静默退出）；
  - 0.5/0.6 系列依赖的 chroma-hnswlib 没有 Python 3.12 的 Windows 轮子，
    源码编译需要装 MSVC Build Tools，成本不可接受。
改用 Qdrant 本地模式：pip 装完即用、纯 Python 嵌入式、免编译免起服务。
生产升级路线：docker 起一个 qdrant 服务，客户端把 path= 换成 url=，其余代码零改动。

为什么从 numpy 换到向量数据库（面试考点，至少 3 个理由）：
  1. 持久化——服务重启知识库还在。numpy 版每次重启要重新 embedding，又慢又烧钱；
  2. payload 过滤——检索时按 doc_id/部门等条件限定范围，Week 3 权限隔离的基石；
  3. HNSW 索引——数据量大了以后不必全量暴力计算，为增长预留容量。

分工不变：embedding 仍然是我们自己调 API（rag.llm.embed），Qdrant 只管存储与检索。
"""
import hashlib
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models

from rag.core import chunk_text  # 你手写的分块，进入生产链路！
from rag.llm import embed

QDRANT_DIR = Path(__file__).resolve().parents[2] / "data" / "qdrant"
COLLECTION = "knowledge_base"
VECTOR_SIZE = 2048  # embedding-3 的输出维度，换 embedding 模型要跟着改

# path= 本地嵌入式模式：数据落盘 data/qdrant/（已 gitignore），不需要单独起服务
client = QdrantClient(path=str(QDRANT_DIR))

if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        # 距离度量：余弦，和我们手写版对齐（面试点：向量库的距离参数必须显式指定）
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
    )


def doc_id_of(raw: bytes) -> str:
    """内容指纹（md5 前 10 位）：同内容重复上传 = 同一文档；改一个字 = 新文档。"""
    return hashlib.md5(raw).hexdigest()[:10]


def ingest(filename: str, raw: bytes, chunk_size: int = 300, overlap: int = 50) -> dict:
    """上传入库全流程：解析 → 分块 → 向量化 → upsert 进 Qdrant。"""
    from app.parser import parse_file  # 函数内导入，避免模块间循环依赖

    text = parse_file(filename, raw)
    if not text.strip():
        raise ValueError("文档解析后没有文本内容（扫描版 PDF 或空文件？）")

    doc_id = doc_id_of(raw)
    chunks = chunk_text(text, chunk_size, overlap)
    # 一次 API 调用批量向量化（比逐块调用快且省钱；超大文件应分批，此处从简）
    vectors = embed([c["text"] for c in chunks])

    # 先删旧块再写入：同一文档重复上传不会越积越多（幂等）
    delete_document(doc_id)

    client.upsert(
        collection_name=COLLECTION,
        points=models.Batch(
            # Qdrant 的点 id 必须是 UUID 或整数：用 uuid5 从字符串确定性生成，
            # 同一 (doc_id, start) 永远得到同一 id——天然支持覆盖式更新
            ids=[str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}-{c['start']}")) for c in chunks],
            vectors=vectors,
            payloads=[
                {"doc_id": doc_id, "source": filename, "start": c["start"], "text": c["text"]}
                for c in chunks
            ],
        ),
    )
    return {"doc_id": doc_id, "filename": filename, "chunks": len(chunks)}


def _doc_filter(doc_id: str) -> models.Filter:
    """按 doc_id 过滤的条件。Week 3 权限隔离就是扩展这里（加部门/密级等字段）。"""
    return models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
    )


def search(question: str, top_k: int = 3) -> list[dict]:
    """问题向量化 → Qdrant 余弦检索 → 统一结果格式。

    注意：Qdrant 的 COSINE 返回的分数就是相似度（越大越像），
    不像 chroma 返回"距离"还要做 1-d 转换——对接不同库要核对口径，这是个真实踩坑点。
    做权限隔离时给 query_points 加 query_filter 参数即可限定检索范围。
    """
    q_vec = embed([question])[0]
    points = client.query_points(
        collection_name=COLLECTION, query=q_vec, limit=top_k, with_payload=True
    ).points
    return [
        {
            "doc_id": p.payload["doc_id"],
            "source": p.payload["source"],
            "start": p.payload["start"],
            "text": p.payload["text"],
            "score": round(p.score, 4),
        }
        for p in points
    ]


def list_documents() -> list[dict]:
    """列出知识库文档：翻全量 payload，按文档聚合分块数（scroll 分页避免一次拉爆内存）。"""
    stats: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION, limit=256, offset=offset, with_payload=True
        )
        for p in points:
            entry = stats.setdefault(
                p.payload["doc_id"],
                {"doc_id": p.payload["doc_id"], "filename": p.payload["source"], "chunks": 0},
            )
            entry["chunks"] += 1
        if offset is None:
            break
    return sorted(stats.values(), key=lambda d: d["filename"])


def delete_document(doc_id: str) -> None:
    """按 payload 过滤删除该文档的所有点。"""
    client.delete(
        collection_name=COLLECTION,
        points_selector=models.FilterSelector(filter=_doc_filter(doc_id)),
    )
