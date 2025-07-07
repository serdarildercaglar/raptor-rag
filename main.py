#!/usr/bin/env python3
"""
RAPTOR Production Service - FastAPI
Agentic AI tool for high-performance document retrieval
"""
import os
import time
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, Union

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# RAPTOR imports
from raptor import (
    RetrievalAugmentation, 
    RetrievalAugmentationConfig, 
    VLLMEmbeddingModel
)

# Logging setup
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("raptor-service")

# Configuration
class ServiceConfig:
    # Service configuration
    SERVICE_PORT = int(os.getenv("SERVICE_PORT", 8000))
    SERVICE_HOST = os.getenv("SERVICE_HOST", "0.0.0.0")
    
    # RAPTOR configuration
    TREE_PATH = os.getenv("RAPTOR_TREE_PATH", "/path/to/raptor/tree.pkl")
    EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8008")
    
    # Performance configuration
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))
    DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", 3500))
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 20))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 30))

config = ServiceConfig()

# Request/Response Models
class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Query text")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Number of top results")
    max_tokens: Optional[int] = Field(default=None, ge=100, le=10000, description="Maximum tokens in response")
    collapse_tree: Optional[bool] = Field(default=True, description="Use collapsed tree search")
    include_metadata: Optional[bool] = Field(default=False, description="Include layer information")

class BatchRetrieveRequest(BaseModel):
    queries: List[str] = Field(..., min_items=1, max_items=20, description="List of queries")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Number of top results")
    max_tokens: Optional[int] = Field(default=None, ge=100, le=10000, description="Maximum tokens in response")
    collapse_tree: Optional[bool] = Field(default=True, description="Use collapsed tree search")
    include_metadata: Optional[bool] = Field(default=False, description="Include layer information")

class RetrieveResponse(BaseModel):
    query: str
    context: str
    context_length: int
    processing_time_ms: float
    metadata: Optional[Dict[str, Any]] = None

class BatchRetrieveResponse(BaseModel):
    results: List[RetrieveResponse]
    total_queries: int
    total_processing_time_ms: float
    average_time_ms: float

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    uptime_seconds: float
    embedding_service_status: str
    tree_loaded: bool
    total_requests: int
    average_response_time_ms: float

class StatusResponse(BaseModel):
    service_info: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    embedding_service: Dict[str, Any]

# Global service state
class ServiceState:
    def __init__(self):
        self.RA: Optional[RetrievalAugmentation] = None
        self.embedding_model: Optional[VLLMEmbeddingModel] = None
        self.start_time = time.time()
        self.request_count = 0
        self.total_response_time = 0.0
        self.is_healthy = False

service_state = ServiceState()

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifecycle"""
    # Startup
    logger.info("🚀 Starting RAPTOR Production Service...")
    
    try:
        # Initialize VLLM embedding model
        logger.info(f"Connecting to VLLM service: {config.EMBEDDING_SERVICE_URL}")
        service_state.embedding_model = VLLMEmbeddingModel(
            base_url=config.EMBEDDING_SERVICE_URL,
            timeout=config.REQUEST_TIMEOUT
        )
        
        # Initialize RAPTOR configuration
        raptor_config = RetrievalAugmentationConfig(
            embedding_model=service_state.embedding_model,
            tr_top_k=config.DEFAULT_TOP_K,
            tr_threshold=0.6,
            tr_selection_mode="top_k"
        )
        
        # Load RAPTOR tree
        logger.info(f"Loading RAPTOR tree: {config.TREE_PATH}")
        service_state.RA = RetrievalAugmentation(
            config=raptor_config, 
            tree=config.TREE_PATH
        )
        
        # Test connection
        await service_state.embedding_model.create_embedding_async("test")
        
        service_state.is_healthy = True
        logger.info("✅ RAPTOR Service initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAPTOR Service: {e}")
        service_state.is_healthy = False
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down RAPTOR Service...")
    if service_state.embedding_model:
        await service_state.embedding_model.close()
    logger.info("✅ Service shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="RAPTOR Production Service",
    description="High-performance document retrieval service for Agentic AI systems",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper functions
def update_metrics(response_time: float):
    """Update service metrics"""
    service_state.request_count += 1
    service_state.total_response_time += response_time

async def check_embedding_service() -> str:
    """Check embedding service health"""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{config.EMBEDDING_SERVICE_URL}/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    return "healthy"
                else:
                    return f"unhealthy (status: {response.status})"
    except Exception as e:
        return f"unreachable ({str(e)})"

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        embedding_status = await check_embedding_service()
        
        avg_response_time = (
            service_state.total_response_time / service_state.request_count
            if service_state.request_count > 0 else 0.0
        )
        
        return HealthResponse(
            status="healthy" if service_state.is_healthy else "unhealthy",
            service="RAPTOR Production Service",
            version="1.0.0",
            uptime_seconds=time.time() - service_state.start_time,
            embedding_service_status=embedding_status,
            tree_loaded=service_state.RA is not None,
            total_requests=service_state.request_count,
            average_response_time_ms=avg_response_time
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Detailed status information"""
    try:
        uptime = time.time() - service_state.start_time
        avg_response_time = (
            service_state.total_response_time / service_state.request_count
            if service_state.request_count > 0 else 0.0
        )
        
        return StatusResponse(
            service_info={
                "name": "RAPTOR Production Service",
                "version": "1.0.0",
                "uptime_seconds": uptime,
                "uptime_human": f"{uptime//3600:.0f}h {(uptime%3600)//60:.0f}m",
                "tree_path": config.TREE_PATH,
                "embedding_service_url": config.EMBEDDING_SERVICE_URL
            },
            performance_metrics={
                "total_requests": service_state.request_count,
                "average_response_time_ms": avg_response_time,
                "requests_per_second": service_state.request_count / uptime if uptime > 0 else 0,
                "default_top_k": config.DEFAULT_TOP_K,
                "default_max_tokens": config.DEFAULT_MAX_TOKENS,
                "max_batch_size": config.MAX_BATCH_SIZE
            },
            embedding_service={
                "url": config.EMBEDDING_SERVICE_URL,
                "status": await check_embedding_service(),
                "timeout": config.REQUEST_TIMEOUT
            }
        )
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail="Status check failed")

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_single(request: RetrieveRequest, background_tasks: BackgroundTasks):
    """Retrieve context for a single query"""
    if not service_state.is_healthy or not service_state.RA:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    start_time = time.time()
    
    try:
        # Set defaults
        top_k = request.top_k or config.DEFAULT_TOP_K
        max_tokens = request.max_tokens or config.DEFAULT_MAX_TOKENS
        
        # Retrieve context
        if request.include_metadata:
            context, metadata = await service_state.RA.retrieve_async(
                request.query,
                top_k=top_k,
                max_tokens=max_tokens,
                collapse_tree=request.collapse_tree,
                return_layer_information=True
            )
            metadata_dict = {"layer_information": metadata}
        else:
            context = await service_state.RA.retrieve_async(
                request.query,
                top_k=top_k,
                max_tokens=max_tokens,
                collapse_tree=request.collapse_tree,
                return_layer_information=False
            )
            metadata_dict = None
        
        processing_time = (time.time() - start_time) * 1000
        
        # Update metrics in background
        background_tasks.add_task(update_metrics, processing_time)
        
        return RetrieveResponse(
            query=request.query,
            context=context,
            context_length=len(context),
            processing_time_ms=round(processing_time, 2),
            metadata=metadata_dict
        )
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        background_tasks.add_task(update_metrics, processing_time)
        
        logger.error(f"Retrieve failed for query '{request.query}': {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

@app.post("/retrieve/batch", response_model=BatchRetrieveResponse)
async def retrieve_batch(request: BatchRetrieveRequest, background_tasks: BackgroundTasks):
    """Retrieve context for multiple queries (optimized for Agentic AI)"""
    if not service_state.is_healthy or not service_state.RA:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    if len(request.queries) > config.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch size {len(request.queries)} exceeds maximum {config.MAX_BATCH_SIZE}"
        )
    
    start_time = time.time()
    
    try:
        # Set defaults
        top_k = request.top_k or config.DEFAULT_TOP_K
        max_tokens = request.max_tokens or config.DEFAULT_MAX_TOKENS
        
        # Batch retrieve
        if request.include_metadata:
            results_with_metadata = await service_state.RA.retrieve_batch(
                request.queries,
                top_k=top_k,
                max_tokens=max_tokens,
                collapse_tree=request.collapse_tree,
                return_layer_information=True
            )
            results = []
            for i, (context, metadata) in enumerate(results_with_metadata):
                results.append(RetrieveResponse(
                    query=request.queries[i],
                    context=context,
                    context_length=len(context),
                    processing_time_ms=0,  # Will be set below
                    metadata={"layer_information": metadata}
                ))
        else:
            contexts = await service_state.RA.retrieve_batch(
                request.queries,
                top_k=top_k,
                max_tokens=max_tokens,
                collapse_tree=request.collapse_tree,
                return_layer_information=False
            )
            results = []
            for i, context in enumerate(contexts):
                results.append(RetrieveResponse(
                    query=request.queries[i],
                    context=context,
                    context_length=len(context),
                    processing_time_ms=0,  # Will be set below
                    metadata=None
                ))
        
        total_processing_time = (time.time() - start_time) * 1000
        avg_processing_time = total_processing_time / len(request.queries)
        
        # Update individual processing times
        for result in results:
            result.processing_time_ms = round(avg_processing_time, 2)
        
        # Update metrics in background
        background_tasks.add_task(update_metrics, total_processing_time)
        
        return BatchRetrieveResponse(
            results=results,
            total_queries=len(request.queries),
            total_processing_time_ms=round(total_processing_time, 2),
            average_time_ms=round(avg_processing_time, 2)
        )
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        background_tasks.add_task(update_metrics, processing_time)
        
        logger.error(f"Batch retrieve failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch retrieval failed: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "internal_error"}
    )

# Run the service
def main():
    """Main entry point"""
    logger.info(f"Starting RAPTOR Service on {config.SERVICE_HOST}:{config.SERVICE_PORT}")
    
    uvicorn.run(
        "main:app",
        host=config.SERVICE_HOST,
        port=config.SERVICE_PORT,
        workers=1,  # Single worker for async
        reload=False,
        access_log=True,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )

if __name__ == "__main__":
    main()