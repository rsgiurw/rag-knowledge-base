# 企业知识库问答系统（RAG + 安全工程）

> 一个"不只是调 API"的 RAG 项目：手写核心检索链路、自建评测体系、内置企业级安全能力
> （提示注入防护 / PII 脱敏 / 输出合规过滤），用真实数据说话。
>
> 作者：张荀栋 ｜ 求职作品集项目 ｜ 当前进度：**Week 1 · Day 1（开发中）**

## 为什么做这个项目

企业私有大模型落地最常见的形态就是"知识库问答"：文档堆里精准找答案、答案可溯源、
敏感内容不出门。本项目按生产级思路完整实现这条链路，并叠加安全工程——
这也是作者一年大模型内容合规检测工作经验的延续。

## 架构

```
离线索引：  文档 → 解析 → 分块(chunking) → 向量化(embedding) → 向量库
在线问答：  问题 → 向量化 → 检索 top-k → 拼装 Prompt → LLM 流式生成 → 带引用回答
安全层  ：  输入侧提示注入检测 ｜ 文档/输出 PII 脱敏 ｜ 输出合规规则过滤
```

## 目录结构

```
backend/            Python 后端与 RAG 核心
  rag/              手写核心：分块、相似度、LLM/Embedding 客户端
  scripts/          教学与演示脚本（01 裸调 API → 03 最小 RAG）
  tests/            核心模块单元测试
frontend/           React 聊天界面（Week 1 后半引入）
data/raw_docs/      示例知识库文档
docs/               讲义、实验记录、面试拷问记录
AGENTS.md           项目全局约定
```

## 快速开始

```bash
# 1. 配置密钥
cp .env.example .env   # 填入你的智谱 API Key

# 2. 安装依赖
cd backend && python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows

# 3. Day1 演示脚本
.venv/Scripts/python scripts/01_hello_llm.py
```

## 路线图

- [x] Day 1：项目骨架 / 裸调 GLM API / embedding 实验
- [x] Day 2：手写分块 + 余弦相似度 → 最小 RAG CLI（带引用溯源）
- [ ] Week 1 后半：FastAPI + SSE 流式 + React 界面
- [ ] Week 2：混合检索(BM25+向量+RRF) / Rerank / Query 改写 + 自建评测集
- [ ] Week 3：安全模块（注入防护 / PII 脱敏 / 合规规则引擎）+ Docker 部署
