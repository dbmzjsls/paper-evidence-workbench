# Paper Evidence Workbench

面向科研论文的证据检索与筛选工作台。支持导入 PDF、DOCX、PPTX、XLSX、图片、TXT、Markdown、HTML、CSV 等多格式文档，通过 MinerU 进行结构化解析，提取文本、表格、图片、公式等证据块，并基于研究主题生成可追溯的论文筛选报告。

## 架构概览

```
data/                          # 原始论文文件（不入 git）
  └── _uploads/                # API 上传暂存
src/
  ├── config.py                # 全局配置（环境变量 + 默认值）
  ├── models.py                # Pydantic 数据模型
  ├── storage.py               # SQLite + FTS5 全文检索语料库
  ├── indexing.py              # 文档导入 → 解析 → 分块 → FAISS 向量索引
  ├── evidence.py              # 将解析元素构建为证据块
  ├── retrieval.py             # 混合检索 + LLM 问答 + 论文筛选
  ├── rag_chain.py             # LangChain 兼容适配层
  ├── parser.py                # MinerU 旧版兼容包装
  ├── bulid_db.py              # 旧版兼容 shim（注意：文件名拼写为历史遗留）
  └── parsers/
      ├── base.py              # 解析器基类 + 通用工具
      ├── mineru.py            # MinerU 结构化解析
      └── plain.py             # 纯文本/Markdown/HTML/CSV 解析
api.py                         # FastAPI Web 服务
main.py                        # CLI 入口
rag_eval/                      # RAGAS 评测
tests/                         # pytest 测试
web/                           # 前端静态文件
```

## 快速开始

```bash
# 安装依赖
uv sync --group dev

# 初始化本地环境配置
cp .env.example .env

# 导入 data/ 目录下所有论文
uv run python main.py ingest --path data/

# 查看语料库状态
uv run python main.py stats

# 基于证据问答
uv run python main.py query "新能源汽车负面口碑如何影响购买意愿？"

# 按研究主题筛选论文
uv run python main.py screen "新能源汽车负面口碑与购买意愿" --limit 10
```

### 启动 Web API

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

打开 http://localhost:8000 使用 Web 工作台。

### 主要 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/documents/upload` | 多文件上传，返回 `job_id` |
| `GET` | `/jobs/{job_id}` | 查询导入进度 |
| `GET` | `/documents` | 论文列表 |
| `GET` | `/documents/{id}` | 论文详情（含元素、资源、分块） |
| `POST` | `/query` | 证据问答，返回 `answer`、`citations` 及执行状态 |
| `POST` | `/query/stream` | SSE 问答，先返回执行状态，再返回回答 |
| `POST` | `/screen` | 按研究主题筛选论文 |
| `POST` | `/documents/rebuild` | 重建语料库 |
| `POST` | `/documents/reindex` | 从 SQLite 重建 FAISS 索引 |

### 执行状态与降级

`/query`、`/screen` 及 CLI `query` / `screen` JSON 的新结果始终包含以下顶层字段；筛选固定请求 `ensemble`，不调用回答 LLM。

| 字段 | 类型 | 含义 |
|------|------|------|
| `retrieval_mode_requested` | `str` | 原始请求模式，默认 `ensemble` |
| `retrieval_mode_used` | `str` | 实际成功执行的检索路径 |
| `degraded` | `bool` | 等价于原因数组非空 |
| `degradation_reason` | `list[DegradationReason]` | 按执行顺序累计并去重；正常为 `[]` |

每条原因包含 `stage`（`request` / `vector` / `rerank` / `llm`）、`code`、简短中文 `message` 和实际处理方式 `fallback`，四者均为字符串；这些原因不直接暴露异常字符串、堆栈或连接地址。

实际路径以 `vector`、`keyword`、`ensemble`（两路成功）或 `none`（无后端成功执行）为基础；实际执行超过一个查询变体追加 `+multiquery`，成功应用重排再追加 `+rerank`，例如 `ensemble+multiquery+rerank`。正常 `rerank` 请求可得到 `ensemble+rerank`，名称不同不代表降级。LLM 回退不改变检索路径。多查询中部分向量查询失败、部分成功时，路径仍包含向量，同时记录失败原因。

| 原因码 | 条件与回退 |
|--------|------------|
| `unsupported_mode` | 未知模式兼容替换为 `ensemble`，不改为 422；大小写和首尾空格规范化不算降级 |
| `vector_index_missing` | 向量索引不存在 |
| `vector_loading_disabled` | 需要加载索引但信任开关关闭；已加载且可用的对象不受影响 |
| `vector_load_failed` | 嵌入模型或索引加载失败 |
| `vector_search_failed` | 向量相似度查询失败，保留其他成功结果 |
| `reranker_not_configured` | 预期重排但模型名为空 |
| `reranker_failed` | 重排加载或预测失败（包括缓存的不可用状态） |
| `reranker_invalid_scores` | 重排分数数量错误或包含非有限值 |
| `llm_not_configured` | 有证据但无 API key，返回带引文的抽取式回答 |
| `llm_failed` | LLM 初始化、调用或解析失败，保留上下文与引文 |
| `llm_empty_response` | LLM 返回空白或不可用文本，使用抽取式回答 |

向量原因的 `fallback` 为 `keyword`（确实执行关键词路径）或 `none`（纯向量无可用结果）；纯 `vector` 保持既有行为，不新增关键词回退。重排原因使用 `original_order`，LLM 原因使用 `extractive`，未知模式使用 `ensemble`。

正常执行后零命中或命中被过滤不是故障，仍计入成功检索路径；首次惰性加载成功也不算降级。候选少于两个，或非 `rerank` 模式且 `ENABLE_RERANK=false` 时，正常跳过重排。不含证据时返回原有无证据文案，不调用 LLM，也不新增 LLM 原因，但保留先前原因。无 key 的本地问答有证据时会明确报告 `llm_not_configured`，便于区分生成式与抽取式回答。

`/query/stream` 在回答之前发送一个具名 `metadata` 事件，其 JSON 仅含上述四个字段；例如无索引、无 key 的混合问答：

```text
event: metadata
data: {"retrieval_mode_requested":"ensemble","retrieval_mode_used":"keyword","degraded":true,"degradation_reason":[{"stage":"vector","code":"vector_index_missing","message":"向量索引不存在，已使用关键词检索。","fallback":"keyword"},{"stage":"llm","code":"llm_not_configured","message":"未配置回答模型，已返回抽取式回答。","fallback":"extractive"}]}

data: 基于当前检索证据，可以先给出以下审慎结论：
data: - [C1] 证据摘录……

data: [DONE]
```

回答仍使用无事件名的 `data:` 消息，多行回答在同一消息中使用多个 `data:` 行；结束仍为 `[DONE]`。客户端应按 SSE 事件边界解析并区分 `metadata` 与回答，避免把 JSON 当成正文。网页同步展示实际路径与降级原因。旧 `RAGChainAdapter` 仍返回字符串，旧 CLI `--question` 文本输出不扩展；旧数据模型反序列化兼容默认值（实际路径为 `unknown`），不能将旧数据的默认值当作本次执行记录。评分权重、候选过滤及业务错误的 HTTP 语义保持不变。

## 配置

复制 `.env.example` 为 `.env` 后按需调整。无 API key 时，系统仍可执行导入、关键词/向量检索和抽取式回答。

### 必需配置

```env
DASHSCOPE_API_KEY=your_key    # LLM 生成式问答所需（可选，无 key 时回退到抽取式回答）
```

### 常用可选配置

```env
# 路径
PAPER_STORAGE_DIR=./data
SQLITE_PATH=./vectorstore/paper_corpus.sqlite3

# MinerU 解析
MINERU_BACKEND=pipeline        # pipeline | hybrid | vlm
MINERU_LANG=ch
MINERU_FORMULA=true
MINERU_TABLE=true

# 行为
AUTO_INGEST_ON_STARTUP=false
TRUST_LOCAL_FAISS_INDEX=false  # 启用向量检索前需设为 true
KEYWORD_CANDIDATE_LIMIT=500
ENABLE_RERANK=false
RERANK_MODEL=BAAI/bge-reranker-base
RERANK_CANDIDATE_LIMIT=50

# CORS
CORS_ALLOW_ORIGINS=*
```

## 运行测试

```bash
uv run --group dev python -m pytest tests -q
```

## 技术栈

- **文档解析**：MinerU 3.x (pipeline 后端)，支持 PDF/DOCX/PPTX/XLSX/图片
- **语料存储**：SQLite + FTS5 全文搜索
- **向量索引**：FAISS (HNSW) + sentence-transformers 嵌入
- **检索策略**：向量检索 + 关键词检索混合排序
- **LLM**：DashScope (通义千问) 兼容 OpenAI API
- **Web**：FastAPI + 静态前端
- **评测**：RAGAS

## 已知限制

- Rerank 模型已集成到混合检索候选集重排序；设置 `ENABLE_RERANK=true` 后会加载 `RERANK_MODEL`，如模型不可用会自动回退到原混合排序
- `Config` 中部分索引参数（HNSW 配置、语义分块等）已定义但未在 FAISS 构建时使用，FAISS 使用 langchain 默认参数
- 旧版兼容模块（`src/bulid_db.py`、`src/parser.py`、`src/rag_chain.py`）依赖内部私有 API，重构时需注意
- 缺少统一日志和全局异常脱敏；解析仍有静默回退，检索与回答模型的降级通过执行状态报告
