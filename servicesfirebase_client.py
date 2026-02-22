"""
Firebase Client for state management and real-time data synchronization.
Implements singleton pattern and robust error handling.
"""
import logging
from typing import Dict, List, Any, Optional, Generator
from dataclasses import dataclass, asdict
from datetime import datetime
import json

import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin.exceptions import FirebaseError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.settings import settings
from services.logger import get_logger

logger = get_logger(__name__)

@dataclass
class KnowledgeUnit:
    """Data model for cross-domain knowledge units."""
    id: str
    domain: str
    content: str
    metadata: Dict[str, Any]
    embeddings: Optional[List[float]] = None
    created_at: datetime = None
    updated_at: datetime = None
    confidence_score: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

class FirebaseClient:
    """Singleton Firebase client with retry logic and connection pooling."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FirebaseClient, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialize_firebase()
            self._initialized = True
    
    def _initialize_firebase(self) -> None:
        """Initialize Firebase connection with error handling."""
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(settings.firebase_credentials_path)
                firebase_admin.initialize_app(cred, {
                    'projectId': settings.firebase_project_id,
                })
                logger.info(f"Firebase initialized for project: {settings.firebase_project_id}")
            
            self.db = firestore.client()
            self.knowledge_collection = self.db.collection('knowledge_units')
            self.integration_collection = self.db.collection('integration_points')
            
            # Test connection
            self._test_connection()
            
        except FileNotFoundError as e:
            logger.error(f"Firebase credentials not found: {e}")
            raise
        except FirebaseError as e:
            logger.error(f"Firebase initialization failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during Firebase initialization: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=1, min=settings.retry_delay, max=10),
        retry=retry_if_exception