#!/bin/bash
# embedding_service/start_service.sh

echo "🚀 Starting VLLM Embedding Service..."

# # Install requirements
# echo "📚 Installing requirements..."
# pip install -r requirements.txt

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found, creating default..."
    cat > .env << EOF
EMBEDDING_MODEL=intfloat/multilingual-e5-large
EMBEDDING_PORT=8008
GPU_MEMORY_UTILIZATION=0.7
EMBEDDING_BATCH_SIZE=32
MAX_BATCH_SIZE=64
MAX_MODEL_LEN=512
LOG_LEVEL=INFO
EOF
fi

# 🔒 Use only RTX 3090 (GPU 1)
export CUDA_VISIBLE_DEVICES=1
echo "🎯 CUDA_VISIBLE_DEVICES set to 1 (using RTX 3090 only)"

# Start the service
echo "🔥 Starting embedding service on port 8008..."
python server.py
