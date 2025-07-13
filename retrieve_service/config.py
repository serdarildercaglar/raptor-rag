#!/usr/bin/env python3
"""
Retrieve Service Configuration
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RetrieveServiceConfig:
    """Configuration for LlamaIndex Retrieve Service"""
    
    # VLLM Embedding Service (Query Mode)
    VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8008")
    VLLM_MODEL_NAME: str = os.getenv("VLLM_MODEL_NAME", "intfloat/multilingual-e5-large")
    
    # Qdrant Configuration
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "llamaindex_tree")
    
    # FastAPI Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    API_KEY: Optional[str] = os.getenv("API_KEY")
    
    # Retrieval Configuration
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    MAX_TOP_K: int = int(os.getenv("MAX_TOP_K", "50"))
    DEFAULT_SIMILARITY_CUTOFF: float = float(os.getenv("DEFAULT_SIMILARITY_CUTOFF", "0.0"))
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "100"))
    
    # Performance
    TIMEOUT_SECONDS: int = int(os.getenv("TIMEOUT_SECONDS", "30"))
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration"""
        # Test Qdrant connection
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=cls.QDRANT_URL, api_key=cls.QDRANT_API_KEY)
            collections = client.get_collections()
            
            # Check if collection exists
            collection_names = [col.name for col in collections.collections]
            if cls.QDRANT_COLLECTION_NAME not in collection_names:
                raise ValueError(f"Qdrant collection '{cls.QDRANT_COLLECTION_NAME}' not found. Available: {collection_names}")
                
        except Exception as e:
            raise ValueError(f"Qdrant connection failed: {e}")
        
        # Test VLLM connection
        try:
            import requests
            response = requests.get(f"{cls.VLLM_BASE_URL}/health", timeout=5)
            if response.status_code != 200:
                raise ValueError(f"VLLM service not healthy: {response.status_code}")
        except Exception as e:
            raise ValueError(f"VLLM connection failed: {e}")
        
        return True
    
    @classmethod
    def log_config(cls):
        """Log configuration (without sensitive data)"""
        print("🔧 Retrieve Service Configuration:")
        print(f"   Server: {cls.HOST}:{cls.PORT}")
        print(f"   VLLM URL: {cls.VLLM_BASE_URL}")
        print(f"   Qdrant URL: {cls.QDRANT_URL}")
        print(f"   Collection: {cls.QDRANT_COLLECTION_NAME}")
        print(f"   Default Top-K: {cls.DEFAULT_TOP_K}")
        print(f"   Max Batch Size: {cls.MAX_BATCH_SIZE}")
        print(f"   API Key: {'✅ Set' if cls.API_KEY else '❌ None'}")
        print("=" * 50)