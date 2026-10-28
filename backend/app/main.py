"""FastAPI 入口：把 RAG 从命令行脚本变成一个 HTTP 服务。

启动（在 backend 目录）：
    .venv/Scripts/python -m uvicorn app.main:app --reload --port 8000

然后浏览器打开 http://127.0.0.1:8000/docs
—— FastAPI 自动生成的交互文档（Swagger UI），每个接口都能在网页上直接测试，
今天不需要写任何前端代码。

两个必须理解的细节：
  1. 必须用 `python -m uvicorn` 而不是直接敲 uvicorn：`-m` 会把当前目录加进
     模块搜索路径，`from rag.llm import ...` 才能被找到；
  2. 下面的业务接口故意用 def 而不是 async def：我们调 embedding/LLM 用的是
     同步阻塞的 httpx，写进 async def 会卡死整个事件循环；FastAPI 对普通 def
     路由会自动丢进线程池执行，不阻塞其他请求。Day 4 讲 SSE 时升级为真异步。
"""
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import qa, store
from app.schemas import AskRequest, AskResponse, DocumentInfo

app = FastAPI(title="企业知识库问答系统", version="0.1.0")

# 跨域放行：Day 5 的 React 开发服务器跑在 5173 端口，提前配好（浏览器同源策略）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    """健康检查。async def：函数体里没有任何等待（await），才允许这样写。"""
    return {"status": "ok"}


@app.post("/api/documents", response_model=DocumentInfo)
def upload_document(file: UploadFile):
    """上传文档：解析 → 分块（你手写的 chunk_text）→ 向量化 → 入库。"""
    raw = file.file.read()  # UploadFile.file 是标准文件对象，读出原始字节
    try:
        info = store.ingest(file.filename, raw)
    except ValueError as e:
        # 客户端的错（格式不对/空文档）返回 4xx；程序自己的 bug 会自动 500
        raise HTTPException(status_code=400, detail=str(e))
    return info


@app.get("/api/documents", response_model=list[DocumentInfo])
def list_documents():
    """知识库文档列表（Day 5 前端的文档管理页用）。"""
    return store.list_documents()


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    """【Day 3 你的填空任务】删除一篇文档的所有分块。

    照着提示写两行（写完删掉 raise 那行）：
      1. 调用 store.delete_document(doc_id)
      2. 返回字典 {"deleted": doc_id}
    测试：重启服务 → 打开 /docs → 找到 DELETE /api/documents/{doc_id} →
    Try it out → 填入任意文档的 doc_id → 执行后到 GET /api/documents 确认它没了。
    """
    raise NotImplementedError("Day 3 任务：删掉这行，按上面注释写两行代码")


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """检索 + 生成，返回答案和引用溯源。非流式版；Day 4 升级为 SSE 流式。"""
    return qa.answer_question(req.question, req.top_k)
