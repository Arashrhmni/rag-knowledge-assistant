from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # LLM
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"  # local, no API key needed

    # Vector store
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "knowledge_base"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k: int = 5
    similarity_threshold: float = 0.3

    # App
    max_file_size_mb: int = 50
    allowed_extensions: list[str] = [".pdf", ".txt", ".md", ".csv"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
