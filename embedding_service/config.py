"""Production configuration with VLLM native parameters"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Config(BaseSettings):
    """Production VLLM Embedding Configuration"""
    
    # Model settings
    model_name: str = Field(
        default="intfloat/multilingual-e5-large",
        env="MODEL_NAME"
    )
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8008, env="PORT")
    
    # VLLM Native Parameters
    gpu_memory_utilization: float = Field(
        default=0.95, 
        env="GPU_MEMORY_UTILIZATION",
        description="Arctic Inference uses 0.95 for maximum performance"
    )
    max_model_len: int = Field(default=512, env="MAX_MODEL_LEN")
    max_num_seqs: int = Field(
        default=256, 
        env="MAX_NUM_SEQS",
        description="Higher concurrency for embeddings"
    )
    max_num_batched_tokens: int = Field(
        default=8192,
        env="MAX_NUM_BATCHED_TOKENS"
    )
    enable_prefix_caching: bool = Field(
        default=True,
        env="ENABLE_PREFIX_CACHING",
        description="Cache for repetitive queries"
    )
    enable_chunked_prefill: bool = Field(
        default=True,
        env="ENABLE_CHUNKED_PREFILL",
        description="Better handling of long sequences"
    )
    
    # Performance
    tensor_parallel_size: int = Field(
        default=1,
        description="Embeddings don't need TP"
    )
    swap_space: int = Field(
        default=4,
        description="GB of CPU swap space for OOM protection"
    )
    
    # Arctic Inference
    use_arctic_inference: bool = Field(
        default=True,
        env="USE_ARCTIC_INFERENCE"
    )
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Security
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    
    class Config:
        env_file = ".env"
        
    def get_vllm_args(self) -> dict:
        """Get VLLM initialization arguments"""
        return {
            "model": self.model_name,
            "task": "embed",
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "enable_prefix_caching": self.enable_prefix_caching,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "tensor_parallel_size": self.tensor_parallel_size,
            "swap_space": self.swap_space,
            "trust_remote_code": True,
            "enforce_eager": False,  # Keep graph compilation
        }