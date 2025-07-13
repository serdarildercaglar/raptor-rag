#!/bin/bash
# Production startup script

echo "🚀 Starting VLLM Embedding Server..."

# Check GPU
nvidia-smi

# Set optimizations
export VLLM_ATTENTION_BACKEND=FLASHINFER  # Use FlashInfer if available
export VLLM_USE_MODELSCOPE=false
export TOKENIZERS_PARALLELISM=false

# Arctic Inference specifics
if [ "$USE_ARCTIC_INFERENCE" = "true" ]; then
    echo "❄️  Arctic Inference optimizations enabled"
    export VLLM_PLUGINS="arctic_inference"
fi

# Start server
exec python server.py