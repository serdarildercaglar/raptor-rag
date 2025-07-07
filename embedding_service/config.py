# embedding_service/config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for VLLM Embedding Service"""
    
    # Model configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    
    # Server configuration
    EMBEDDING_PORT = int(os.getenv("EMBEDDING_PORT", 8008))
    
    # GPU configuration
    GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", 0.7))
    
    # Batch configuration
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 32))
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 64))
    
    # Model specific settings
    MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", 512))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    def __repr__(self):
        return f"""
        VLLMEmbeddingConfig:
            Model: {self.EMBEDDING_MODEL}
            Port: {self.EMBEDDING_PORT}
            GPU Memory: {self.GPU_MEMORY_UTILIZATION}
            Batch Size: {self.EMBEDDING_BATCH_SIZE}
            Max Model Length: {self.MAX_MODEL_LEN}
        """