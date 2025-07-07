# embedding_service/server.py
import os
import asyncio
import uvicorn
from typing import List, Dict, Any, Optional
from vllm import LLM
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from config import Config

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for OpenAI compatibility
class EmbeddingRequest(BaseModel):
    input: List[str]
    model: str
    encoding_format: Optional[str] = "float"

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Usage

class VLLMEmbeddingService:
    def __init__(self):
        self.config = Config()
        self.app = FastAPI(title="VLLM Embedding Service")
        self.setup_cors()
        self.setup_routes()
        self.llm = None
        
    def setup_cors(self):
        """Setup CORS middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    async def initialize_model(self):
        """Initialize VLLM model"""
        try:
            logger.info(f"Loading model: {self.config.EMBEDDING_MODEL}")
            
            # Set CUDA device order for multi-GPU systems
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
            self.llm = LLM(
                model=self.config.EMBEDDING_MODEL,
                task="embed",
                enforce_eager=True,
                gpu_memory_utilization=self.config.GPU_MEMORY_UTILIZATION,
                max_model_len=self.config.MAX_MODEL_LEN,
                trust_remote_code=True,
                tensor_parallel_size=1  # Single GPU usage
            )
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise e
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.on_event("startup")
        async def startup_event():
            await self.initialize_model()
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "model": self.config.EMBEDDING_MODEL}
        
        @self.app.post("/v1/embeddings", response_model=EmbeddingResponse)
        async def create_embeddings(request: EmbeddingRequest):
            """OpenAI compatible embeddings endpoint"""
            try:
                if self.llm is None:
                    raise HTTPException(status_code=503, detail="Model not loaded")
                
                # Add query prefix for E5 model if not already present
                formatted_texts = []
                for text in request.input:
                    if not text.startswith(("query:", "passage:")):
                        text = f"query: {text}"
                    formatted_texts.append(text)
                
                # Generate embeddings using VLLM
                outputs = self.llm.embed(formatted_texts)
                
                # Format response
                data = []
                for i, output in enumerate(outputs):
                    data.append(EmbeddingData(
                        embedding=output.outputs.embedding,
                        index=i
                    ))
                
                # Calculate usage
                total_tokens = sum(len(text.split()) for text in request.input)
                usage = Usage(prompt_tokens=total_tokens, total_tokens=total_tokens)
                
                return EmbeddingResponse(
                    data=data,
                    model=request.model,
                    usage=usage
                )
                
            except Exception as e:
                logger.error(f"Error creating embeddings: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/embeddings/batch")
        async def create_embeddings_batch(texts: List[str]):
            """Batch embeddings endpoint for high performance"""
            try:
                if self.llm is None:
                    raise HTTPException(status_code=503, detail="Model not loaded")
                
                # Add query prefix for E5 model
                formatted_texts = [f"query: {text}" for text in texts]
                
                # Generate embeddings using VLLM
                outputs = self.llm.embed(formatted_texts)
                
                embeddings = []
                for output in outputs:
                    embeddings.append(output.outputs.embedding)
                
                return {
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "embedding": emb,
                            "index": i
                        }
                        for i, emb in enumerate(embeddings)
                    ],
                    "model": self.config.EMBEDDING_MODEL,
                    "usage": {
                        "prompt_tokens": sum(len(text.split()) for text in texts),
                        "total_tokens": sum(len(text.split()) for text in texts)
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in batch embeddings: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

def run_server():
    """Run the embedding service"""
    config = Config()
    service = VLLMEmbeddingService()
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=config.EMBEDDING_PORT,
        workers=1  # VLLM requires single worker
    )

if __name__ == "__main__":
    run_server()