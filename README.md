# Research Paper Retrieval Workbench

带 Web 工作台的科研论文增强检索系统。V1 支持导入 PDF、DOCX、PPTX、XLSX、图片、TXT、Markdown、HTML、CSV，保留文本、表格、图片、公式等证据块，并基于研究主题生成可追溯的论文筛选报告。

## What Changed

- MinerU 3.x 输出适配：优先读取 `*_content_list_v2.json`，回退到 `*_content_list.json` 和 Markdown。
- SQLite 成为语料库事实层：保存 documents、elements、chunks、assets、jobs、reports。
- FAISS 只负责向量索引；旧 `docs.pkl` 仅保留兼容。
- 查询和筛选都会返回 citations 和 contexts，回答不再是无出处文本。
- FastAPI 直接服务 `web/` 静态工作台。

## Setup

```bash
uv sync --group dev
```

如需 LLM 生成式回答，配置：

```env
DASHSCOPE_API_KEY=your_key
```

无 API key 时，系统仍可执行导入、关键词/向量检索、抽取式回答和论文筛选。

## CLI

```bash
# 导入 data/ 目录下所有支持文件
uv run python main.py ingest --path data/

# 查看语料库状态
uv run python main.py stats

# 基于证据问答
uv run python main.py query "新能源汽车负面口碑如何影响购买意愿？"

# 按研究主题筛选论文
uv run python main.py screen "新能源汽车负面口碑与购买意愿" --limit 10
```

旧用法仍可用：

```bash
uv run python main.py --question "你的问题"
uv run python main.py --stats
```

## Web API

启动：

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

打开：

```text
http://localhost:8000
```

主要接口：

- `POST /documents/upload`：多文件上传，返回 `job_id`
- `GET /jobs/{job_id}`：查看解析/索引进度
- `GET /documents`、`GET /documents/{document_id}`：查看论文和证据
- `POST /query`：返回 `answer`、`citations`、`contexts`
- `POST /screen`：按研究主题生成论文筛选报告
- `POST /documents/rebuild`：重建 `data/` 语料库
- `POST /documents/reindex`：从 SQLite chunks 重建 FAISS

## Configuration

常用环境变量：

```env
PAPER_STORAGE_DIR=./data
PARSED_DIR=./data/parsed
ASSET_DIR=./data/assets
SQLITE_PATH=./vectorstore/paper_corpus.sqlite3
MINERU_BACKEND=pipeline
MINERU_METHOD=auto
MINERU_LANG=ch
MINERU_FORMULA=true
MINERU_TABLE=true
AUTO_INGEST_ON_STARTUP=false
CORS_ALLOW_ORIGINS=*
CORS_ALLOW_CREDENTIALS=false
TRUST_LOCAL_FAISS_INDEX=false
KEYWORD_CANDIDATE_LIMIT=500
```

默认 MinerU 后端为 CPU 友好的 `pipeline`。更高精度的 hybrid/VLM 后端可以通过环境变量切换。
默认不加载未经显式信任的本地 FAISS pickle 索引；需要启用向量索引检索时，请确认索引文件来自本机可信构建流程后设置 `TRUST_LOCAL_FAISS_INDEX=true`。

## Tests

```bash
uv run --group dev python -m pytest tests -q
```
