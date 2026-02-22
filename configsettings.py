"""
Configuration management for ACDKN.
Uses Pydantic for validation and type safety.
"""
import os
from typing import Dict, List, Optional
from pydantic import BaseSettings, Field, validator
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Application settings with validation."""
    
    # Firebase Configuration
    firebase_project_id: str = Field(..., env="FIREBASE_PROJECT_ID")
    firebase_credentials_path: str = Field(..., env="FIREBASE_CREDENTIALS_PATH")
    
    # Model Configuration
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_knowledge_units: int = Field(default=10000, gt=0)
    
    # Processing Configuration
    batch_size: int = Field(default=100, gt=0)
    max_retries: int = Field(default=3, gt=0)
    retry_delay: float = Field(default=1.0, gt=0.0)
    
    # Domain Configuration
    supported_domains: List[str] = Field(default=["healthcare", "finance", "technology", "research"])
    
    # Performance Configuration
    cache_ttl_seconds: int = Field(default=3600, gt=0)
    max_concurrent_operations: int = Field(default=10, gt=0)
    
    # Validation
    @validator('firebase_credentials_path')
    def validate_credentials_path(cls, v: str) -> str:
        if not os.path.exists(v):
            raise ValueError(f"Firebase credentials file not found: {v}")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global settings instance
settings = Settings()