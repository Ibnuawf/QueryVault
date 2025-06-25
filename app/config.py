"""
Configuration Module for RAG System
===================================
This module defines the application settings and configuration management for the Retrieval-Augmented Generation (RAG) system.
It leverages Pydantic's BaseSettings to load environment variables and .env files, ensuring secure and flexible configuration for all components.

Key Responsibilities:
- Centralizes all environment-based configuration for the app.
- Provides a cached settings instance for use throughout the project.
- Ensures robust error handling for missing or invalid configuration.

Usage:
Import `get_settings()` wherever configuration is needed.
"""

import logging
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application Settings
    -------------------
    Manages application configuration via environment variables and .env files.
    All critical parameters for model selection, data directories, and API limits are defined here.
    """
    GEMINI_API_KEY: str
    GEMINI_MODEL_NAME: str = 'gemini-1.5-flash-latest'
    EMBEDDING_MODEL_NAME: str = 'all-MiniLM-L6-v2'
    DATA_DIR: str = "data"
    CHROMA_PERSIST_DIR: str = "chroma_db_store"
    COLLECTION_NAME: str = "islamqa_collection_v1"
    N_RESULTS_RETRIEVAL: int = 5
    CHUNK_SIZE_WORDS: int = 300
    RATE_LIMIT_REQUESTS: int = 20
    RATE_LIMIT_TIMEFRAME_SECONDS: int = 60
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached instance of the application settings.
    Ensures that configuration is loaded only once and reused across the project.
    Raises a critical error and exits if configuration cannot be loaded.
    """
    try:
        return Settings()
    except Exception as e:
        logging.critical(f"FATAL: Could not load settings. Is .env file missing? Error: {e}")
        raise SystemExit(1)