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
| `POST` | `/query` | 证据问答，返回 `answer` + `citations` |
| `POST` | `/screen` | 按研究主题筛选论文 |
| `POST` | `/documents/rebuild` | 重建语料库 |
| `POST` | `/documents/reindex` | 从 SQLite 重建 FAISS 索引 |

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
- 无日志系统，解析/检索异常静默处理，排查问题不便
