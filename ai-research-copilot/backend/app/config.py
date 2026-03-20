from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    host: str = "0.0.0.0"
    port: int = 8000
    data_dir: Path = Path("./data")
    chunk_size: int = 900
    chunk_overlap: int = 120
    retrieval_top_k: int = 6


settings = Settings()
