#!/usr/bin/env python3
"""
Production VLLM Embedding Server
Real production-grade server with actual optimizations
"""

import os
import sys
import time
import asyncio
import psutil
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import threading

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator, ValidationError
from vllm import LLM
import torch
try:
    import pynvml
    # Initialize NVML for GPU monitoring
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("production_server")

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ServerConfig:
    """Production server configuration"""
    
    # Model settings
    model_name: str = "intfloat/multilingual-e5-large"
    max_model_len: int = 512
    gpu_memory_utilization: float = 0.7
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8008
    workers: int = 1
    
    # Batch processing
    max_batch_size: int = 64
    optimal_batch_size: int = 32
    batch_timeout: float = 0.1
    
    # Performance settings
    max_concurrent_requests: int = 100
    request_timeout: float = 30.0
    
    # Memory limits
    max_memory_usage: float = 0.9  # 90% of available memory
    memory_check_interval: float = 5.0  # seconds
    
    # Error handling
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 30.0
    
    @classmethod
    def from_env(cls) -> 'ServerConfig':
        """Load configuration from environment variables"""
        return cls(
            model_name=os.getenv("MODEL_NAME", cls.model_name),
            max_model_len=int(os.getenv("MAX_MODEL_LEN", cls.max_model_len)),
            gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", cls.gpu_memory_utilization)),
            host=os.getenv("HOST", cls.host),
            port=int(os.getenv("PORT", cls.port)),
            max_batch_size=int(os.getenv("MAX_BATCH_SIZE", cls.max_batch_size)),
            optimal_batch_size=int(os.getenv("OPTIMAL_BATCH_SIZE", cls.optimal_batch_size)),
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", cls.max_concurrent_requests)),
            request_timeout=float(os.getenv("REQUEST_TIMEOUT", cls.request_timeout)),
        )

# =============================================================================
# MEMORY MONITORING
# =============================================================================

class MemoryMonitor:
    """Real memory monitoring for GPU and system"""
    
    def __init__(self):
        self.gpu_available = GPU_AVAILABLE
        self.gpu_count = 0
        if self.gpu_available:
            self.gpu_count = pynvml.nvmlDeviceGetCount()
    
    def get_gpu_memory(self) -> Dict[str, Any]:
        """Get actual GPU memory usage"""
        if not self.gpu_available:
            return {"available": False}
        
        gpu_info = []
        for i in range(self.gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_info.append({
                "gpu_id": i,
                "total_mb": mem_info.total // (1024 * 1024),
                "used_mb": mem_info.used // (1024 * 1024),
                "free_mb": mem_info.free // (1024 * 1024),
                "utilization": (mem_info.used / mem_info.total) * 100
            })
        
        return {
            "available": True,
            "gpu_count": self.gpu_count,
            "gpus": gpu_info
        }
    
    def get_system_memory(self) -> Dict[str, Any]:
        """Get actual system memory usage"""
        memory = psutil.virtual_memory()
        return {
            "total_mb": memory.total // (1024 * 1024),
            "used_mb": memory.used // (1024 * 1024),
            "available_mb": memory.available // (1024 * 1024),
            "utilization": memory.percent
        }
    
    def get_process_memory(self) -> Dict[str, Any]:
        """Get current process memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss // (1024 * 1024),
            "vms_mb": memory_info.vms // (1024 * 1024),
            "percent": process.memory_percent()
        }

# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

class PerformanceMetrics:
    """Real performance metrics collection"""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        self.total_tokens = 0
        self.batch_count = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        
        # Latency tracking
        self.latencies = []
        self.max_latency_samples = 1000
    
    def record_request(self, latency: float, token_count: int, success: bool):
        """Record actual request metrics"""
        with self.lock:
            self.request_count += 1
            
            if success:
                self.total_latency += latency
                self.total_tokens += token_count
                self.latencies.append(latency)
                
                # Keep only recent samples
                if len(self.latencies) > self.max_latency_samples:
                    self.latencies.pop(0)
            else:
                self.error_count += 1
    
    def record_batch(self, batch_size: int, latency: float):
        """Record batch processing metrics"""
        with self.lock:
            self.batch_count += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        with self.lock:
            uptime = time.time() - self.start_time
            successful_requests = self.request_count - self.error_count
            
            # Calculate latency percentiles
            latency_stats = {}
            if self.latencies:
                sorted_latencies = sorted(self.latencies)
                latency_stats = {
                    "min": min(sorted_latencies),
                    "max": max(sorted_latencies),
                    "avg": sum(sorted_latencies) / len(sorted_latencies),
                    "p50": sorted_latencies[len(sorted_latencies) // 2],
                    "p95": sorted_latencies[int(len(sorted_latencies) * 0.95)],
                    "p99": sorted_latencies[int(len(sorted_latencies) * 0.99)]
                }
            
            return {
                "uptime_seconds": uptime,
                "total_requests": self.request_count,
                "successful_requests": successful_requests,
                "error_count": self.error_count,
                "error_rate": (self.error_count / max(self.request_count, 1)) * 100,
                "requests_per_second": self.request_count / max(uptime, 1),
                "total_tokens": self.total_tokens,
                "tokens_per_second": self.total_tokens / max(uptime, 1),
                "batch_count": self.batch_count,
                "latency_ms": latency_stats
            }

# =============================================================================
# ERROR HANDLING
# =============================================================================

class CircuitBreaker:
    """Circuit breaker for error handling"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = asyncio.Lock()
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker"""
        async with self.lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

# =============================================================================
# BATCH PROCESSOR
# =============================================================================

class BatchProcessor:
    """Optimized batch processing"""
    
    def __init__(self, llm: LLM, config: ServerConfig):
        self.llm = llm
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.pending_requests = []
        self.lock = asyncio.Lock()
        self.processing = False
    
    async def add_request(self, texts: List[str]) -> List[List[float]]:
        """Add request to batch processing queue"""
        
        # Direct processing for small batches
        if len(texts) <= self.config.optimal_batch_size:
            return await self._process_direct(texts)
        
        # Chunk large batches
        results = []
        for i in range(0, len(texts), self.config.optimal_batch_size):
            chunk = texts[i:i + self.config.optimal_batch_size]
            chunk_results = await self._process_direct(chunk)
            results.extend(chunk_results)
        
        return results
    
    async def _process_direct(self, texts: List[str]) -> List[List[float]]:
        """Process texts directly through VLLM"""
        
        # Add query prefix for E5 model
        formatted_texts = []
        for text in texts:
            if not text.startswith(("query:")):
                text = f"passage: {text}"
            formatted_texts.append(text)
        
        # Check memory before processing
        memory_info = self.memory_monitor.get_gpu_memory()
        if memory_info["available"]:
            for gpu in memory_info["gpus"]:
                if gpu["utilization"] > 95:
                    raise Exception("GPU memory usage too high")
        
        # Process through VLLM (sync operation)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            outputs = await loop.run_in_executor(executor, self.llm.embed, formatted_texts)
        
        # Extract embeddings
        embeddings = []
        for output in outputs:
            embeddings.append(output.outputs.embedding)
        
        return embeddings
    
    def get_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on memory"""
        memory_info = self.memory_monitor.get_gpu_memory()
        if not memory_info["available"]:
            return self.config.optimal_batch_size
        
        # Adjust batch size based on available GPU memory
        for gpu in memory_info["gpus"]:
            if gpu["utilization"] > 80:
                return max(1, self.config.optimal_batch_size // 2)
        
        return self.config.optimal_batch_size

# =============================================================================
# MAIN SERVER
# =============================================================================

class ProductionEmbeddingServer:
    """Production-grade embedding server"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.llm = None
        self.batch_processor = None
        self.memory_monitor = MemoryMonitor()
        self.metrics = PerformanceMetrics()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold,
            timeout=config.circuit_breaker_timeout
        )
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
        # Memory monitoring task
        self.memory_task = None
        self.memory_alerts = []
    
    async def initialize(self):
        """Initialize the server"""
        try:
            # Set CUDA device order for multi-GPU systems
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            logger.info(f"Loading model: {self.config.model_name}")
            
            self.llm = LLM(
                model=self.config.model_name,
                task="embed",
                enforce_eager=True,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                trust_remote_code=True,
            )
            
            self.batch_processor = BatchProcessor(self.llm, self.config)
            
            # Start memory monitoring
            self.memory_task = asyncio.create_task(self._monitor_memory())
            
            logger.info("Server initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize server: {e}")
            raise
    
    async def _monitor_memory(self):
        """Background memory monitoring"""
        while True:
            try:
                # Check system memory
                sys_memory = self.memory_monitor.get_system_memory()
                if sys_memory["utilization"] > self.config.max_memory_usage * 100:
                    self.memory_alerts.append({
                        "type": "system_memory",
                        "utilization": sys_memory["utilization"],
                        "timestamp": time.time()
                    })
                
                # Check GPU memory
                gpu_memory = self.memory_monitor.get_gpu_memory()
                if gpu_memory["available"]:
                    for gpu in gpu_memory["gpus"]:
                        if gpu["utilization"] > self.config.max_memory_usage * 100:
                            self.memory_alerts.append({
                                "type": "gpu_memory",
                                "gpu_id": gpu["gpu_id"],
                                "utilization": gpu["utilization"],
                                "timestamp": time.time()
                            })
                
                # Keep only recent alerts
                current_time = time.time()
                self.memory_alerts = [
                    alert for alert in self.memory_alerts
                    if current_time - alert["timestamp"] < 300  # 5 minutes
                ]
                
                await asyncio.sleep(self.config.memory_check_interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(self.config.memory_check_interval)
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.memory_task:
            self.memory_task.cancel()
            try:
                await self.memory_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Server cleanup completed")

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class EmbeddingRequest(BaseModel):
    input: List[str] = Field(..., min_items=1, max_items=1000, description="List of texts to embed")
    model: str = Field(..., description="Model name for embedding")
    encoding_format: Optional[str] = Field(default="float", description="Encoding format")
    
    @validator('input')
    def validate_input_texts(cls, v):
        """Validate input texts"""
        if not v:
            raise ValueError("Input cannot be empty")
        
        # Check individual text lengths
        for i, text in enumerate(v):
            if not isinstance(text, str):
                raise ValueError(f"Input item {i} must be a string")
            if len(text.strip()) == 0:
                raise ValueError(f"Input item {i} cannot be empty or whitespace only")
            if len(text) > 10000:  # 10k chars max per text
                raise ValueError(f"Input item {i} too long (max 10000 characters)")
        
        return v
    
    @validator('model')
    def validate_model(cls, v):
        """Validate model name"""
        if not v or not isinstance(v, str):
            raise ValueError("Model name must be a non-empty string")
        return v

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

# Global server instance
server_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global server_instance
    
    # Startup
    config = ServerConfig.from_env()
    server_instance = ProductionEmbeddingServer(config)
    await server_instance.initialize()
    
    yield
    
    # Shutdown
    await server_instance.cleanup()

# Create FastAPI app
app = FastAPI(
    title="Production VLLM Embedding Server",
    description="Production-grade embedding server with real optimizations",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle pydantic validation errors"""
    errors = []
    for error in exc.errors():
        field = '.'.join(str(loc) for loc in error['loc'])
        message = error['msg']
        errors.append(f"{field}: {message}")
    
    return JSONResponse(
        status_code=400,
        content={
            "detail": "Input validation failed",
            "errors": errors,
            "type": "validation_error"
        }
    )

@app.exception_handler(ValidationError)
async def pydantic_validation_handler(request: Request, exc: ValidationError):
    """Handle pydantic validation errors"""
    return JSONResponse(
        status_code=400,
        content={
            "detail": "Request validation failed",
            "errors": [str(error) for error in exc.errors()],
            "type": "validation_error"
        }
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle value errors"""
    return JSONResponse(
        status_code=400,
        content={
            "detail": str(exc),
            "type": "value_error"
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": "internal_error"
        }
    )

# =============================================================================
# ROUTES
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not server_instance or not server_instance.llm:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    # Get real system status
    memory_info = server_instance.memory_monitor.get_system_memory()
    gpu_info = server_instance.memory_monitor.get_gpu_memory()
    process_info = server_instance.memory_monitor.get_process_memory()
    
    return {
        "status": "healthy",
        "model": server_instance.config.model_name,
        "system_memory": memory_info,
        "gpu_memory": gpu_info,
        "process_memory": process_info,
        "memory_alerts": len(server_instance.memory_alerts),
        "circuit_breaker": server_instance.circuit_breaker.state
    }

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics"""
    if not server_instance:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    return {
        "performance": server_instance.metrics.get_metrics(),
        "memory": {
            "system": server_instance.memory_monitor.get_system_memory(),
            "gpu": server_instance.memory_monitor.get_gpu_memory(),
            "process": server_instance.memory_monitor.get_process_memory()
        },
        "alerts": server_instance.memory_alerts[-10:],  # Last 10 alerts
        "config": {
            "max_batch_size": server_instance.config.max_batch_size,
            "optimal_batch_size": server_instance.config.optimal_batch_size,
            "max_concurrent_requests": server_instance.config.max_concurrent_requests
        }
    }

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Create embeddings with production optimizations"""
    if not server_instance or not server_instance.llm:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    # Validate model name matches server model
    if request.model != server_instance.config.model_name:
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{request.model}' not supported. Available model: '{server_instance.config.model_name}'"
        )
    
    # Check batch size limit
    if len(request.input) > server_instance.config.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(request.input)} exceeds maximum {server_instance.config.max_batch_size}"
        )
    
    async with server_instance.semaphore:
        start_time = time.perf_counter()
        
        try:
            # Process through circuit breaker
            embeddings = await server_instance.circuit_breaker.call(
                server_instance.batch_processor.add_request,
                request.input
            )
            
            # Build response
            data = []
            for i, embedding in enumerate(embeddings):
                data.append(EmbeddingData(
                    embedding=embedding,
                    index=i
                ))
            
            # Record metrics
            latency = time.perf_counter() - start_time
            token_count = sum(len(text.split()) for text in request.input)
            server_instance.metrics.record_request(latency, token_count, True)
            
            return EmbeddingResponse(
                data=data,
                model=request.model
            )
            
        except Exception as e:
            # Record error
            latency = time.perf_counter() - start_time
            server_instance.metrics.record_request(latency, 0, False)
            
            logger.error(f"Embedding request failed: {e}")
            
            # Handle specific error types
            error_msg = str(e)
            if "GPU memory" in error_msg:
                raise HTTPException(status_code=503, detail="Server temporarily overloaded. Please try again.")
            elif "Circuit breaker" in error_msg:
                raise HTTPException(status_code=503, detail="Service temporarily unavailable due to errors. Please try again later.")
            else:
                raise HTTPException(status_code=500, detail="Internal server error occurred during embedding generation.")


@app.post("/embeddings/batch")
async def create_embeddings_batch(texts: List[str]):
    """High-performance batch embeddings"""
    if not server_instance or not server_instance.llm:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    # Input validation
    if not texts:
        raise HTTPException(status_code=400, detail="Input cannot be empty")
    
    if len(texts) > server_instance.config.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(texts)} exceeds maximum {server_instance.config.max_batch_size}"
        )
    
    # Validate individual texts
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise HTTPException(status_code=400, detail=f"Input item {i} must be a string")
        if len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail=f"Input item {i} cannot be empty or whitespace only")
        if len(text) > 10000:  # 10k chars max per text
            raise HTTPException(status_code=400, detail=f"Input item {i} too long (max 10000 characters)")
    
    async with server_instance.semaphore:
        start_time = time.perf_counter()
        
        try:
            # Process batch
            embeddings = await server_instance.circuit_breaker.call(
                server_instance.batch_processor.add_request,
                texts
            )
            
            # Record metrics
            latency = time.perf_counter() - start_time
            token_count = sum(len(text.split()) for text in texts)
            server_instance.metrics.record_request(latency, token_count, True)
            server_instance.metrics.record_batch(len(texts), latency)
            
            return {
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": emb, "index": i}
                    for i, emb in enumerate(embeddings)
                ],
                "model": server_instance.config.model_name
            }
            
        except Exception as e:
            # Record error
            latency = time.perf_counter() - start_time
            server_instance.metrics.record_request(latency, 0, False)
            
            logger.error(f"Batch request failed: {e}")
            
            # Handle specific error types
            error_msg = str(e)
            if "GPU memory" in error_msg:
                raise HTTPException(status_code=503, detail="Server temporarily overloaded. Please try again.")
            elif "Circuit breaker" in error_msg:
                raise HTTPException(status_code=503, detail="Service temporarily unavailable due to errors. Please try again later.")
            else:
                raise HTTPException(status_code=500, detail="Internal server error occurred during batch processing.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point"""
    config = ServerConfig.from_env()
    
    logger.info(f"Starting server on {config.host}:{config.port}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Max batch size: {config.max_batch_size}")
    logger.info(f"Max concurrent requests: {config.max_concurrent_requests}")
    
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=config.workers,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()