from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    OPENAI_API_KEY: str
    PINECONE_API_KEY: str
    LANGSMITH_API_KEY: str


@lru_cache
def get_settings():
    return Settings() # type: ignore
