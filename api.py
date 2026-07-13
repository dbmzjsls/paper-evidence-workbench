from __future__ import annotations

import re
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import Config
from src.indexing import IndexingService, get_db_stats
from src.parsers import discover_files, is_supported
from src.retrieval import ResearchRAG
from src.storage import CorpusStorage


Config.ensure_dirs()


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage = CorpusStorage()
    if Config.AUTO_INGEST_ON_STARTUP and storage.stats()["document_count"] == 0:
        IndexingService(storage=storage).ingest_path(Config.DATA_DIR)
    yield


app = FastAPI(
    title="Paper Evidence Workbench",
    description="Evidence-grounded paper ingestion, retrieval, and screening.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ALLOW_ORIGINS,
    allow_credentials=Config.CORS_ALLOW_CREDENTIALS
    and "*" not in Config.CORS_ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=Config.WEB_DIR), name="static")
app.mount("/assets", StaticFiles(directory=Config.ASSET_DIR), name="assets")


class QueryRequest(BaseModel):
    question: str
    mode: str = "ensemble"


class ScreenRequest(BaseModel):
    topic: str
    limit: int = Field(default=10, ge=1, le=50)
    include_keywords: list[str] = Field(default_factory=list)
    exclude_keywords: list[str] = Field(default_factory=list)


class IngestPathRequest(BaseModel):
    path: str = Config.DATA_DIR
    recursive: bool = True
    rebuild: bool = False


@app.get("/")
async def index():
    index_path = Path(Config.WEB_DIR) / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"service": "paper-evidence-workbench", "web": "not_built"}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "paper-evidence-workbench"}


@app.get("/stats")
async def stats():
    return get_db_stats()


@app.get("/jobs/{job_id}")
async def job(job_id: str):
    item = CorpusStorage().get_job(job_id)
    if not item:
        raise HTTPException(status_code=404, detail="Job not found")
    return item


@app.get("/documents")
async def documents(limit: int = 200, offset: int = 0):
    storage = CorpusStorage()
    docs = storage.list_documents(limit=limit, offset=offset)
    return {"documents": docs, "count": len(docs)}


@app.get("/documents/{document_id}")
async def document_detail(document_id: str):
    storage = CorpusStorage()
    doc = storage.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    elements = storage.get_elements(document_id, limit=500)
    assets = storage.get_assets(document_id)
    chunks = storage.get_chunks(document_id)
    return {
        "document": doc,
        "elements": elements,
        "assets": assets,
        "chunks": chunks,
        "stats": {
            "elements": len(elements),
            "assets": len(assets),
            "chunks": len(chunks),
        },
    }


@app.get("/documents/{document_id}/elements")
async def document_elements(document_id: str, limit: int = 1000):
    storage = CorpusStorage()
    if not storage.get_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"elements": storage.get_elements(document_id, limit=limit)}


@app.post("/documents/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    upload_specs: list[tuple[UploadFile, Path]] = []
    for file in files:
        filename = _safe_filename(file.filename or "upload")
        if not is_supported(Path(filename)):
            suffix = Path(filename).suffix.lower()
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
        target = Path(Config.UPLOAD_DIR) / f"{uuid.uuid4().hex[:8]}_{filename}"
        upload_specs.append((file, target))

    saved_paths: list[Path] = []
    try:
        for file, target in upload_specs:
            with target.open("wb") as f:
                shutil.copyfileobj(file.file, f)
            saved_paths.append(target)
    except Exception as exc:
        for path in saved_paths:
            if path.exists():
                path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    background_tasks.add_task(IndexingService().ingest_files, saved_paths, False, job_id)
    return {"job_id": job_id, "status": "queued", "files": [p.name for p in saved_paths]}


@app.post("/documents/ingest")
async def ingest_path(req: IngestPathRequest, background_tasks: BackgroundTasks):
    target = _allowed_ingest_path(req.path)
    if target is None:
        raise HTTPException(status_code=404, detail="Path not found")
    if target.is_dir() and not discover_files(target, recursive=req.recursive):
        raise HTTPException(status_code=400, detail="No supported files found")
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    service = IndexingService()
    if req.rebuild:
        background_tasks.add_task(service.rebuild_corpus, target, req.recursive, job_id)
    else:
        background_tasks.add_task(service.ingest_path, target, req.recursive, False, job_id)
    return {"job_id": job_id, "status": "queued", "path": str(target)}


@app.post("/documents/reindex")
async def reindex():
    try:
        result = IndexingService().rebuild_vector_index()
        return {"status": "ok", **result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/documents/rebuild")
async def rebuild(background_tasks: BackgroundTasks):
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    background_tasks.add_task(IndexingService().rebuild_corpus, Config.DATA_DIR, True, job_id)
    return {"job_id": job_id, "status": "queued", "path": Config.DATA_DIR}


@app.post("/query")
async def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is required")
    return ResearchRAG().answer(req.question, mode=req.mode)


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    async def generate():
        try:
            result = ResearchRAG().answer(req.question, mode=req.mode)
            yield f"data: {result.answer}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            yield f"data: [ERROR] {exc}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/screen")
async def screen(req: ScreenRequest):
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic is required")
    return ResearchRAG().screen(
        topic=req.topic,
        limit=req.limit,
        include_keywords=req.include_keywords,
        exclude_keywords=req.exclude_keywords,
    )


def _safe_filename(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+", "_", name)
    return name[:180] or "upload"


def _allowed_ingest_path(raw_path: str) -> Path | None:
    try:
        target = Path(raw_path).expanduser().resolve(strict=False)
        allowed_roots = [
            Path(Config.DATA_DIR).expanduser().resolve(strict=False),
            Path(Config.UPLOAD_DIR).expanduser().resolve(strict=False),
        ]
    except (OSError, RuntimeError, ValueError):
        return None

    for root in allowed_roots:
        try:
            target.relative_to(root)
        except ValueError:
            continue
        return target if target.exists() else None
    raise HTTPException(
        status_code=403,
        detail="Path ingestion is restricted to the configured data and upload directories",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
