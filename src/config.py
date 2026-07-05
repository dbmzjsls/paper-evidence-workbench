import os
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_list(name: str, default: str) -> list[str]:
    value = os.getenv(name, default)
    return [item.strip() for item in value.split(",") if item.strip()]


class Config:
    # ========== 路径配置 ==========
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    VECTOR_DIR = os.path.join(BASE_DIR, "vectorstore", "faiss_index")
    DOCS_DIR = os.path.join(BASE_DIR, "vectorstore", "docs.pkl")
    # Source corpus directory. PDF_STORAGE_DIR remains supported for old deployments.
    DATA_DIR = os.getenv(
        "PAPER_STORAGE_DIR",
        os.getenv(
            "PDF_STORAGE_DIR",
            os.path.join(BASE_DIR, "data"),
        ),
    )
    UPLOAD_DIR = os.getenv(
        "UPLOAD_DIR",
        os.path.join(BASE_DIR, "data", "_uploads"),
    )
    PARSED_DIR = os.getenv(
        "PARSED_DIR",
        os.path.join(BASE_DIR, "data", "parsed"),
    )
    ASSET_DIR = os.getenv(
        "ASSET_DIR",
        os.path.join(BASE_DIR, "data", "assets"),
    )
    SQLITE_PATH = os.getenv(
        "SQLITE_PATH",
        os.path.join(BASE_DIR, "vectorstore", "paper_corpus.sqlite3"),
    )
    WEB_DIR = os.getenv(
        "WEB_DIR",
        os.path.join(BASE_DIR, "web"),
    )
    MINERU_OUTPUT_DIR = os.getenv(
        "MINERU_OUTPUT_DIR",
        PARSED_DIR,
    )
    EVAL_DATASETS_DIR = os.path.join(BASE_DIR, "evals", "datasets")

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
        ".txt",
        ".md",
        ".markdown",
        ".html",
        ".htm",
        ".csv",
    }
    MINERU_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }
    PLAIN_EXTENSIONS = {
        ".txt",
        ".md",
        ".markdown",
        ".html",
        ".htm",
        ".csv",
    }

    # ========== MinerU parsing ==========
    MINERU_BACKEND = os.getenv("MINERU_BACKEND", "pipeline")
    MINERU_METHOD = os.getenv("MINERU_METHOD", "auto")
    MINERU_LANG = os.getenv("MINERU_LANG", "ch")
    MINERU_FORMULA = _env_bool("MINERU_FORMULA", True)
    MINERU_TABLE = _env_bool("MINERU_TABLE", True)
    MINERU_IMAGE_ANALYSIS = _env_bool("MINERU_IMAGE_ANALYSIS", True)
    MINERU_TIMEOUT_SECONDS = _env_int("MINERU_TIMEOUT_SECONDS", 1200)

    AUTO_INGEST_ON_STARTUP = _env_bool("AUTO_INGEST_ON_STARTUP", False)
    RETRIEVAL_CONTEXTS = _env_int("RETRIEVAL_CONTEXTS", 8)
    SCREEN_DEFAULT_LIMIT = _env_int("SCREEN_DEFAULT_LIMIT", 10)
    SCREEN_MAX_LIMIT = _env_int("SCREEN_MAX_LIMIT", 50)
    CORS_ALLOW_ORIGINS = _env_list("CORS_ALLOW_ORIGINS", "*")
    CORS_ALLOW_CREDENTIALS = _env_bool("CORS_ALLOW_CREDENTIALS", False)

    # ========== FAISS 索引配置 ==========
    FAISS_INDEX_TYPE: str = "HNSW"  # "HNSW" | "Flat" | "IVF"
    HNSW_M: int = 32  # HNSW 每节点连接数 (越大越精确但越慢, 推荐 16-64)
    HNSW_EF_CONSTRUCTION: int = 200  # 构建时搜索宽度
    HNSW_EF_SEARCH: int = 50  # 检索时搜索宽度

    # ========== 分块配置 ==========
    CHUNK_SIZE: int = 500  # 字符数 (纯文本回退时使用)
    CHUNK_OVERLAP: int = 50
    USE_MINERU: bool = True  # 是否使用 MinerU 做结构化解析
    USE_SEMANTIC_CHUNKING: bool = True  # 是否使用语义分块

    # ========== 嵌入模型 ==========
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DEVICE: str = "cpu"  # "cpu" | "cuda"
    EMBEDDING_BATCH_SIZE: int = 32  # 批量嵌入大小

    # ========== 重排序模型 ==========
    RERANK_MODEL: str = "BAAI/bge-reranker-base"

    # ========== 检索配置 ==========
    RETRIEVER_K: int = 5  # 最终返回文档数
    BM25_K: int = 10  # BM25 检索数量
    VECTOR_K: int = 10  # 向量检索数量
    KEYWORD_CANDIDATE_LIMIT: int = _env_int("KEYWORD_CANDIDATE_LIMIT", 500)
    TRUST_LOCAL_FAISS_INDEX: bool = _env_bool("TRUST_LOCAL_FAISS_INDEX", False)
    ENSEMBLE_WEIGHTS: list = [0.4, 0.6]  # [BM25权重, 向量权重]
    TOP_N: int = 2  # Rerank 后保留

    # ========== LLM 配置 ==========
    LLM_MODEL: str = "glm-5"
    LLM_EVAL: str = "qwen3.6-plus"
    DASHSCOPE_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 2000

    @staticmethod
    def get_api_key():
        return os.getenv("DASHSCOPE_API_KEY")

    @classmethod
    def ensure_dirs(cls):
        """确保所有必要目录存在"""
        for d in [
            cls.VECTOR_DIR,
            cls.DATA_DIR,
            cls.UPLOAD_DIR,
            cls.PARSED_DIR,
            cls.ASSET_DIR,
            cls.MINERU_OUTPUT_DIR,
            cls.EVAL_DATASETS_DIR,
            cls.WEB_DIR,
        ]:
            os.makedirs(d, exist_ok=True)
        sqlite_dir = os.path.dirname(os.path.abspath(cls.SQLITE_PATH))
        if sqlite_dir:
            os.makedirs(sqlite_dir, exist_ok=True)
