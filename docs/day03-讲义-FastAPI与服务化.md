# Day 3 讲义 · 从脚本到服务：FastAPI 与向量库

> 学习目标：搞懂"RAG 脚本"怎么变成"RAG 服务"——HTTP 服务模型、Web 框架、异步、向量数据库。
> 今天的代码在 `backend/app/`，启动后玩转 http://127.0.0.1:8000/docs。

## 一、为什么 03 脚本不算"产品"

`03_minimal_rag.py` 的问题：你问一句它答一句，关掉窗口知识库就没了，别人也没法用。要变成产品需要三样东西：
1. **常驻进程**：一个一直运行、随时等待请求的程序（服务器）；
2. **统一入口**：任何设备/程序都能通过 HTTP 协议访问它；
3. **持久化**：重启后知识库还在（今天用 Chroma 落盘解决）。

## 二、HTTP 服务模型（30 秒版，必须会画）

```
客户端（浏览器/前端/另一个程序）
    │  ① 发请求：方法 + 路径 + 数据
    ▼                    例：POST /api/documents  + 文件字节
服务器（uvicorn 跑着的 FastAPI）
    │  ② 路由匹配 → 执行你的函数
    │  ③ 返回响应：状态码 + JSON 数据
    ▼
客户端拿到 JSON 渲染/处理
```

- 常见方法语义：**GET 读、POST 建、PUT/PATCH 改、DELETE 删**（合称 REST 风格）。
- 常见状态码：**200 成功、400 客户端数据错、404 不存在、422 校验失败、500 服务器内部错**。
- 你今天调的智谱 API 就是同一套模型——`/chat/completions`、`/embeddings` 是别人的服务；今天我们把 RAG 也做成这样的服务。

## 三、Web 框架（FastAPI）替你干了什么

如果没有框架，你要自己写：解析 HTTP 报文、找路由、解析 JSON、校验字段、处理文件上传、序列化响应……框架把这些全包了：

1. **路由**：`@app.post("/api/documents")` 这个装饰器的本质——把"方法+路径"登记进一张路由表，请求来了按表分发到你的函数。装饰器就是"不改动函数本体，给函数登记/附加能力"的语法糖。
2. **请求校验（Pydantic）**：`AskRequest` 里 `top_k: int = Field(ge=1, le=10)`——传 99 会自动收到 422，我们一行 if 都不用写。校验发生在进函数体之前。
3. **自动文档**：/docs 页面（Swagger UI）从代码里的类型注解自动生成，还能直接发请求测试——今天你的"前端"就是它。

## 四、同步与异步：def vs async def（面试重点，慢慢读）

- **比喻**：同步 = 一个服务员点完单、站在厨房等菜、再端给客人，一次只能伺候一桌；异步 = 点完单把小票挂起，立刻去下一桌，菜好了再端——**等待时不占人**。
- 我们的场景是**网络 IO 密集**（等 LLM API 返回要几秒），最怕"干等"。
- `async def` + `await` 的机制：把"等待点"让出来，单线程事件循环去跑别的请求。**前提是等待点本身支持异步**。
- 那为什么今天接口全是 `def`？因为我们调 API 用的 httpx 是**同步**阻塞的——放进 `async def` 里会把事件循环卡死（所有人一起等）。FastAPI 对普通 `def` 路由自动丢进**线程池**，也能并发。**"async 里写阻塞调用"是新手最常见事故**，能讲清这一点面试直接加分。Day 4 做 SSE 时上真正的异步写法。

## 五、向量库 Qdrant：换掉 numpy 手写检索

**先讲一个今天真实发生的选型故事**（面试可以直接讲）：原计划用 chromadb，结果
1.x 版在 Windows 上 `add()` 直接段错误（原生库崩溃，服务静默退出）；退回 0.5/0.6
版本又因为 hnswlib 没有 Python 3.12 的 Windows 轮子、必须装 MSVC 编译器——
于是换成 Qdrant 本地嵌入式模式：pip 装完即用、免编译免起服务。教训：**依赖选型
要先跑通最小验证，"能 pip install" 和 "能在本机稳定跑" 是两回事。**

| | numpy 版（03 脚本） | Qdrant 版（今天） |
|---|---|---|
| 存哪 | 内存，进程退出就没 | 磁盘 data/qdrant/，重启还在 |
| 检索 | 手写循环余弦 | client.query_points（余弦已配置） |
| 过滤 | 不支持 | payload 过滤（按 doc_id 等字段） |
| 规模 | 几百块 | 百万级（HNSW 近似最近邻索引） |

- **注意分工**：embedding 还是我们自己调 API，Qdrant 只管"存 + 查"。向量库里一个点（point）= 向量 + payload（原文+来源等元数据）+ 唯一 id。
- **口径坑**：Qdrant 的 COSINE 返回的就是相似度分数（越大越像）；chroma 返回的是"距离"要 1-d 转换——对接第三方库先核对口径，真实踩坑点。
- **HNSW** 一句话：分层"跳表式"近邻图，查询不与全量向量比较，近似但快。当前数据量暴力更快——"为增长预留容量"是工程决策，不是过度设计。
- **payload 过滤是 Week 3 权限隔离的地基**：检索时限定 doc_id/部门范围，就是最朴素的"谁能看哪些文档"。
- **生产升级路线**：本地模式 `QdrantClient(path=...)` → Docker 起服务后改成 `QdrantClient(url=...)`，其余代码零改动。

## 六、今天的链路（对照代码走一遍）

```
上传：POST /api/documents
  file.file.read() 读字节 → parser.parse_file() 得纯文本
  → chunk_text()（你手写的！）→ embed() 向量化 → qdrant upsert()

提问：POST /api/ask
  question → embed → query_points → 命中块带来源标签拼 prompt
  → chat_once → {answer, citations}
```

## 七、今晚复盘：面试自测 5 题（写进 interview-log.md 的 Day 3 小节）

1. `@app.post("/api/documents")` 这行代码执行时发生了什么？（装饰器角度）
2. FastAPI 里 `def` 路由和 `async def` 路由有什么区别？什么时候 async 反而有害？
3. Pydantic 的校验发生在什么时机？相比手写 if 校验好在哪？
4. 已经能用 numpy 算余弦相似度了，为什么还要引入向量数据库？（至少 3 个理由）
5. 请求 `DELETE /api/documents/xxx` 从浏览器发出到文档被删，完整经过了几步？

## 附：今天的启动与测试命令（backend 目录）

```
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```
浏览器：http://127.0.0.1:8000/docs （--reload 改代码自动重启，开发期专用）
