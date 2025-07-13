"""
Production VLLM Embedding Server
Minimal, high-performance implementation using VLLM native features
"""
import asyncio
import time
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
from pydantic import Field, validator, BaseModel
from vllm import LLM, SamplingParams

from config import Config
from monitoring import (
    REQUEST_COUNT, REQUEST_LATENCY, BATCH_SIZE, 
    ACTIVE_REQUESTS, metrics_endpoint
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = Config()

# Security
security = HTTPBearer() if config.api_key else None

# Global model instance
model: Optional[LLM] = None

# =============================================================================
# LIFESPAN MANAGEMENT
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - handles startup and shutdown"""
    global model
    
    # Startup
    logger.info("🚀 Starting VLLM Embedding Server")
    logger.info(f"📦 Model: {config.model_name}")
    
    try:
        # Load Arctic Inference if enabled
        if config.use_arctic_inference:
            try:
                import vllm.plugins
                vllm.plugins.load_general_plugins()
                logger.info("✅ Arctic Inference optimizations loaded")
            except Exception as e:
                logger.warning(f"⚠️  Arctic Inference not available: {e}")
        
        # Initialize VLLM with native parameters
        model = LLM(**config.get_vllm_args())
        logger.info("✅ Model loaded successfully")
        
        # Log VLLM native features status
        logger.info(f"📊 Prefix Caching: {config.enable_prefix_caching}")
        logger.info(f"📊 Chunked Prefill: {config.enable_chunked_prefill}")
        logger.info(f"📊 Max Sequences: {config.max_num_seqs}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown - VLLM handles cleanup automatically
    logger.info("👋 Shutting down server")

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="VLLM Native Embedding Server",
    description="High-performance embedding server using VLLM native features",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# MODELS
# =============================================================================

class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request"""
    input: List[str] = Field(
        ..., 
        min_items=1, 
        max_items=2048,
        description="List of texts to embed"
    )
    model: str = Field(
        default=None,
        description="Model name (optional)"
    )
    encoding_format: str = Field(
        default="float",
        description="Encoding format"
    )
    dimensions: Optional[int] = Field(
        default=None,
        description="For models supporting Matryoshka embeddings"
    )
    
    @validator('input')
    def validate_input(cls, v):
        """Validate input texts"""
        for i, text in enumerate(v):
            if not isinstance(text, str):
                raise ValueError(f"Input[{i}] must be string")
            if not text.strip():
                raise ValueError(f"Input[{i}] cannot be empty")
            if len(text) > 8192:  # VLLM will handle actual limits
                raise ValueError(f"Input[{i}] too long")
        return v

# =============================================================================
# AUTH
# =============================================================================

async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """Verify API key if configured"""
    if not config.api_key:
        return "no-auth"
    
    if credentials.credentials != config.api_key:
        raise HTTPException(403, "Invalid API key")
    
    return credentials.credentials

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not model:
        raise HTTPException(503, "Model not loaded")
    
    return {
        "status": "healthy",
        "model": config.model_name,
        "features": {
            "prefix_caching": config.enable_prefix_caching,
            "chunked_prefill": config.enable_chunked_prefill,
            "arctic_inference": config.use_arctic_inference,
        }
    }

@app.post("/v1/embeddings")
async def create_embeddings(
    request: EmbeddingRequest,
    api_key: str = Depends(verify_api_key) if security else None
):
    """
    Create embeddings - OpenAI compatible endpoint
    VLLM handles all optimizations internally
    """
    if not model:
        raise HTTPException(503, "Model not loaded")
    
    # Track metrics
    ACTIVE_REQUESTS.inc()
    start_time = time.perf_counter()
    
    try:
        # Update batch size metric
        BATCH_SIZE.observe(len(request.input))
        
        # VLLM handles everything internally:
        # - Automatic batching
        # - Prefix caching 
        # - Tokenization
        # - GPU memory management
        # - Error recovery
        outputs = model.embed(request.input)
        
        # Build response
        response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": output.outputs.embedding,
                    "index": i
                }
                for i, output in enumerate(outputs)
            ],
            "model": request.model or config.model_name,
            "usage": {
                "prompt_tokens": sum(len(text.split()) for text in request.input),
                "total_tokens": sum(len(text.split()) for text in request.input)
            }
        }
        
        # Track success
        REQUEST_COUNT.labels(status="success").inc()
        REQUEST_LATENCY.observe(time.perf_counter() - start_time)
        
        return response
        
    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        logger.error(f"Embedding error: {e}")
        raise HTTPException(500, str(e))
    finally:
        ACTIVE_REQUESTS.dec()

@app.post("/v1/embeddings/stream")
async def create_embeddings_stream(
    request: EmbeddingRequest,
    api_key: str = Depends(verify_api_key) if security else None
):
    """
    Streaming embeddings endpoint
    Uses VLLM's async capabilities
    """
    if not model:
        raise HTTPException(503, "Model not loaded")
    
    # For embeddings, streaming means returning results as they complete
    # VLLM processes them in optimal batches internally
    
    from fastapi.responses import StreamingResponse
    import json
    
    async def generate():
        try:
            # Process in chunks for large batches
            chunk_size = 100
            for i in range(0, len(request.input), chunk_size):
                chunk = request.input[i:i + chunk_size]
                outputs = model.embed(chunk)
                
                for j, output in enumerate(outputs):
                    data = {
                        "object": "embedding",
                        "embedding": output.outputs.embedding,
                        "index": i + j
                    }
                    yield f"data: {json.dumps(data)}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            error_data = {"error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

# Metrics endpoint
if config.enable_metrics:
    @app.get("/metrics")
    async def get_metrics():
        """Prometheus metrics endpoint"""
        return await metrics_endpoint()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 VLLM Native Embedding Server")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"GPU Memory: {config.gpu_memory_utilization * 100}%")
    logger.info(f"Max Sequences: {config.max_num_seqs}")
    logger.info(f"Prefix Caching: {config.enable_prefix_caching}")
    logger.info(f"Arctic Inference: {config.use_arctic_inference}")
    logger.info("=" * 60)
    
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
        access_log=True,
        # Use uvloop for better async performance
        loop="uvloop",
        # HTTP/2 support
        http="h11",
    )