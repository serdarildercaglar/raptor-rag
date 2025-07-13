"""Prometheus metrics for VLLM embedding service"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import psutil
import time

# Metrics definitions
REQUEST_COUNT = Counter(
    'embedding_requests_total', 
    'Total number of embedding requests',
    ['status']
)

REQUEST_LATENCY = Histogram(
    'embedding_request_duration_seconds',
    'Latency of embedding requests',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
)

BATCH_SIZE = Histogram(
    'embedding_batch_size',
    'Distribution of batch sizes',
    buckets=[1, 5, 10, 25, 50, 100, 200, 500]
)

GPU_MEMORY_USAGE = Gauge(
    'gpu_memory_usage_bytes',
    'Current GPU memory usage'
)

ACTIVE_REQUESTS = Gauge(
    'embedding_active_requests',
    'Number of active requests'
)

# VLLM native metrics (if available)
VLLM_CACHE_HIT_RATE = Gauge(
    'vllm_prefix_cache_hit_rate',
    'VLLM prefix caching hit rate'
)

def update_system_metrics():
    """Update system metrics"""
    try:
        # CPU and Memory
        memory = psutil.virtual_memory()
        
        # GPU metrics would come from VLLM's internal stats
        # This is a placeholder - VLLM provides these via its own metrics
        GPU_MEMORY_USAGE.set(0)  # VLLM handles this internally
        
    except Exception as e:
        print(f"Error updating metrics: {e}")

async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    update_system_metrics()
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )