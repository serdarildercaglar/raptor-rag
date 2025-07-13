#!/usr/bin/env python3
"""
Tree Builder Configuration
"""
import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TreeBuilderConfig:
    """Configuration for LlamaIndex Tree Builder"""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # VLLM Embedding Service
    VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8008")
    VLLM_MODEL_NAME: str = os.getenv("VLLM_MODEL_NAME", "intfloat/multilingual-e5-large")
    
    # Qdrant Configuration
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "llamaindex_tree")
    
    # Document Processing
    DOCUMENTS_FOLDER: Path = Path(os.getenv("DOCUMENTS_FOLDER", "./documents"))
    OUTPUT_PATH: Path = Path(os.getenv("OUTPUT_PATH", "./tree_data"))
    
    # Chunking Configuration
    BASE_CHUNK_SIZE: int = int(os.getenv("BASE_CHUNK_SIZE", "1024"))
    SUB_CHUNK_SIZES: List[int] = [
        int(x.strip()) for x in os.getenv("SUB_CHUNK_SIZES", "128,256,512").split(",")
    ]
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "20"))
    
    # Metadata Extraction
    NUM_QUESTIONS: int = int(os.getenv("NUM_QUESTIONS", "5"))
    ENABLE_SUMMARIES: bool = os.getenv("ENABLE_SUMMARIES", "true").lower() == "true"
    
    # Performance
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "10"))
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        
        if not cls.DOCUMENTS_FOLDER.exists():
            raise ValueError(f"Documents folder does not exist: {cls.DOCUMENTS_FOLDER}")
        
        # Create output directory if it doesn't exist
        cls.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        
        return True
    
    @classmethod
    def log_config(cls):
        """Log configuration (without sensitive data)"""
        print("🔧 Tree Builder Configuration:")
        print(f"   Documents Folder: {cls.DOCUMENTS_FOLDER}")
        print(f"   Output Path: {cls.OUTPUT_PATH}")
        print(f"   VLLM URL: {cls.VLLM_BASE_URL}")
        print(f"   Qdrant URL: {cls.QDRANT_URL}")
        print(f"   Base Chunk Size: {cls.BASE_CHUNK_SIZE}")
        print(f"   Sub Chunk Sizes: {cls.SUB_CHUNK_SIZES}")
        print(f"   Enable Summaries: {cls.ENABLE_SUMMARIES}")
        print(f"   Num Questions: {cls.NUM_QUESTIONS}")
        print("=" * 50)