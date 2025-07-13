# config.py - Tree Builder Configuration
"""
Tree Builder Configuration Module
Handles environment variables, validation, and configuration management
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class TreeBuilderConfig:
    """
    Configuration class for Tree Builder
    Loads settings from environment variables with sensible defaults
    """
    
    # ============================================================================
    # OpenAI Configuration
    # ============================================================================
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # ============================================================================
    # VLLM Embedding Service Configuration
    # ============================================================================
    VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8008")
    VLLM_MODEL_NAME: str = os.getenv("VLLM_MODEL_NAME", "intfloat/multilingual-e5-large")
    
    # ============================================================================
    # Qdrant Vector Database Configuration
    # ============================================================================
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY") or None
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "llamaindex_tree")
    
    # ============================================================================
    # File Path Configuration
    # ============================================================================
    DOCUMENTS_FOLDER: Path = Path(os.getenv("DOCUMENTS_FOLDER", "./documents"))
    OUTPUT_PATH: Path = Path(os.getenv("OUTPUT_PATH", "./tree_data"))
    
    # ============================================================================
    # Chunking Configuration
    # ============================================================================
    BASE_CHUNK_SIZE: int = int(os.getenv("BASE_CHUNK_SIZE", "1024"))
    SUB_CHUNK_SIZES: List[int] = [
        int(x.strip()) for x in os.getenv("SUB_CHUNK_SIZES", "128,256,512").split(",")
    ]
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "20"))
    
    # ============================================================================
    # Metadata Extraction Configuration
    # ============================================================================
    NUM_QUESTIONS: int = int(os.getenv("NUM_QUESTIONS", "3"))
    ENABLE_SUMMARIES: bool = os.getenv("ENABLE_SUMMARIES", "true").lower() == "true"
    
    # ============================================================================
    # Performance Configuration
    # ============================================================================
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "50"))
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "50"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "400"))
    
    # ============================================================================
    # Logging Configuration
    # ============================================================================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate required configuration parameters
        Raises ValueError if validation fails
        """
        errors = []
        
        # Check required OpenAI API key
        if not cls.OPENAI_API_KEY or cls.OPENAI_API_KEY == "your_openai_api_key_here":
            errors.append("OPENAI_API_KEY is required for metadata extraction")
        
        # Check documents folder exists
        if not cls.DOCUMENTS_FOLDER.exists():
            errors.append(f"Documents folder does not exist: {cls.DOCUMENTS_FOLDER}")
        
        # Check if documents folder has any files
        supported_extensions = ['.pdf', '.txt']
        document_files = []
        for ext in supported_extensions:
            document_files.extend(list(cls.DOCUMENTS_FOLDER.glob(f"*{ext}")))
        
        if not document_files:
            errors.append(f"No PDF or TXT files found in: {cls.DOCUMENTS_FOLDER}")
        
        # Create output directory if it doesn't exist
        try:
            cls.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory {cls.OUTPUT_PATH}: {e}")
        
        # Validate chunk sizes
        if cls.BASE_CHUNK_SIZE <= 0:
            errors.append("BASE_CHUNK_SIZE must be positive")
        
        if not cls.SUB_CHUNK_SIZES or any(size <= 0 for size in cls.SUB_CHUNK_SIZES):
            errors.append("SUB_CHUNK_SIZES must be positive integers")
        
        if any(size >= cls.BASE_CHUNK_SIZE for size in cls.SUB_CHUNK_SIZES):
            errors.append("SUB_CHUNK_SIZES must be smaller than BASE_CHUNK_SIZE")
        
        # Validate overlap
        if cls.CHUNK_OVERLAP < 0:
            errors.append("CHUNK_OVERLAP must be non-negative")
        
        # Validate performance settings
        if cls.MAX_WORKERS <= 0:
            errors.append("MAX_WORKERS must be positive")
        
        if cls.BATCH_SIZE <= 0:
            errors.append("BATCH_SIZE must be positive")
        
        if cls.EMBEDDING_BATCH_SIZE <= 0:
            errors.append("EMBEDDING_BATCH_SIZE must be positive")
        
        if cls.NUM_QUESTIONS < 0:
            errors.append("NUM_QUESTIONS must be non-negative")
        
        # If there are errors, raise them
        if errors:
            error_message = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ValueError(error_message)
        
        return True
    
    @classmethod
    def log_config(cls):
        """
        Log current configuration (without sensitive data)
        """
        print("🔧 Tree Builder Configuration:")
        print("=" * 60)
        
        print("📁 Paths:")
        print(f"   Documents Folder: {cls.DOCUMENTS_FOLDER}")
        print(f"   Output Path: {cls.OUTPUT_PATH}")
        
        print("\n🔗 Services:")
        print(f"   VLLM URL: {cls.VLLM_BASE_URL}")
        print(f"   VLLM Model: {cls.VLLM_MODEL_NAME}")
        print(f"   Qdrant URL: {cls.QDRANT_URL}")
        print(f"   Qdrant Collection: {cls.QDRANT_COLLECTION_NAME}")
        print(f"   OpenAI Model: {cls.OPENAI_MODEL}")
        
        print("\n📝 Chunking:")
        print(f"   Base Chunk Size: {cls.BASE_CHUNK_SIZE}")
        print(f"   Sub Chunk Sizes: {cls.SUB_CHUNK_SIZES}")
        print(f"   Chunk Overlap: {cls.CHUNK_OVERLAP}")
        
        print("\n🧠 Metadata Extraction:")
        print(f"   Enable Summaries: {cls.ENABLE_SUMMARIES}")
        print(f"   Number of Questions: {cls.NUM_QUESTIONS}")
        
        print("\n⚡ Performance:")
        print(f"   Max Workers: {cls.MAX_WORKERS}")
        print(f"   Batch Size: {cls.BATCH_SIZE}")
        print(f"   Embedding Batch Size: {cls.EMBEDDING_BATCH_SIZE}")
        print(f"   Max Tokens: {cls.MAX_TOKENS}")
        
        print("\n🔐 Security:")
        print(f"   OpenAI API Key: {'✅ Set' if cls.OPENAI_API_KEY else '❌ Missing'}")
        print(f"   Qdrant API Key: {'✅ Set' if cls.QDRANT_API_KEY else '❌ None'}")
        
        print("=" * 60)
    
    @classmethod
    def get_document_count(cls) -> dict:
        """
        Get count of documents by type in the documents folder
        """
        if not cls.DOCUMENTS_FOLDER.exists():
            return {"pdf": 0, "txt": 0, "total": 0}
        
        pdf_count = len(list(cls.DOCUMENTS_FOLDER.glob("*.pdf")))
        txt_count = len(list(cls.DOCUMENTS_FOLDER.glob("*.txt")))
        
        return {
            "pdf": pdf_count,
            "txt": txt_count,
            "total": pdf_count + txt_count
        }
    
    @classmethod
    def setup_logging(cls):
        """
        Setup logging configuration
        """
        log_level = getattr(logging, cls.LOG_LEVEL.upper(), logging.INFO)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(cls.OUTPUT_PATH / 'tree_builder.log')
            ]
        )
        
        # Reduce noise from external libraries
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('qdrant_client').setLevel(logging.WARNING)

def load_and_validate_config() -> TreeBuilderConfig:
    """
    Load and validate configuration
    Returns validated config instance
    """
    try:
        # Setup logging first
        TreeBuilderConfig.setup_logging()
        
        # Validate configuration
        TreeBuilderConfig.validate()
        
        # Log configuration
        TreeBuilderConfig.log_config()
        
        # Show document summary
        doc_count = TreeBuilderConfig.get_document_count()
        print(f"\n📄 Documents Found:")
        print(f"   PDF files: {doc_count['pdf']}")
        print(f"   TXT files: {doc_count['txt']}")
        print(f"   Total files: {doc_count['total']}")
        
        if doc_count['total'] == 0:
            print("\n⚠️  Warning: No documents found!")
            print(f"   Please add PDF or TXT files to: {TreeBuilderConfig.DOCUMENTS_FOLDER}")
        
        print()
        
        return TreeBuilderConfig
        
    except ValueError as e:
        print(f"❌ Configuration Error:")
        print(f"{e}")
        print(f"\n📋 Please check your .env file and fix the errors above.")
        raise
    except Exception as e:
        print(f"❌ Unexpected error loading configuration: {e}")
        raise

# Auto-validate when imported (optional)
if __name__ == "__main__":
    try:
        config = load_and_validate_config()
        print("✅ Configuration loaded and validated successfully!")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        exit(1)