"""Process-level settings (environment). Runtime-tunable behaviour lives in runtime_config.py."""

from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Deployment settings. Changing these requires a restart.

    Secrets (API keys) are only ever read from the environment. Model provider keys are
    referenced by environment-variable *name* in the ``models`` table, never stored.
    """

    model_config = SettingsConfigDict(env_prefix="RC_", env_file=".env", extra="ignore")

    db_path: str = "ragcache.db"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    admin_api_key: SecretStr | None = None
    query_api_key: SecretStr | None = None
    log_level: str = "INFO"
    config_refresh_seconds: float = 5.0
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()
