#!/usr/bin/env python3
"""
LlamaIndex Retrieve Service with DOĞRU RecursiveRetriever Pattern
TAMAMEN dokümantasyona dayalı implementation
"""
import asyncio
import logging
import time
import json
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager
from pathlib import Path

# FastAPI
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

# LlamaIndex Core
from llama_index.core.schema import NodeWithScore, BaseNode, IndexNode
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core import VectorStoreIndex

# Vector Store
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# VLLM Integration
import aiohttp

from config import RetrieveServiceConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
recursive_retriever: Optional[RecursiveRetriever] = None
security = HTTPBearer() if RetrieveServiceConfig.API_KEY else None

class VLLMQueryEmbedding(BaseEmbedding):
    """VLLM Embedding with query prefix for retrieval"""
    
    # Pydantic fields for LlamaIndex compatibility
    base_url: str = ""
    vllm_model_name: str = ""
    session: Optional[aiohttp.ClientSession] = None
    
    def __init__(self, base_url: str, model_name: str, **kwargs):
        super().__init__(**kwargs)
        # Use object.__setattr__ to bypass pydantic validation during init
        object.__setattr__(self, 'base_url', base_url.rstrip('/'))
        object.__setattr__(self, 'vllm_model_name', model_name)
        object.__setattr__(self, 'session', None)
        
    class Config:
        arbitrary_types_allowed = True
        
    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=RetrieveServiceConfig.TIMEOUT_SECONDS)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def _embed_batch(self, texts: List[str], prefix: str = "query") -> List[List[float]]:
        """Embed batch of texts with prefix and auto-truncation"""
        # Use context manager for safer session handling
        timeout = aiohttp.ClientTimeout(total=RetrieveServiceConfig.TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Add prefix and truncate if needed
            max_tokens = 400  # Leave margin for prefix
            import tiktoken
            tokenizer = tiktoken.get_encoding("o200k_base")
            
            prefixed_texts = []
            for text in texts:
                prefixed_text = f"{prefix}: {text}"
                
                # Truncate if too long
                tokens = tokenizer.encode(prefixed_text)
                if len(tokens) > max_tokens:
                    # Truncate tokens and decode back
                    truncated_tokens = tokens[:max_tokens]
                    prefixed_text = tokenizer.decode(truncated_tokens)
                    logger.warning(f"Query truncated from {len(tokens)} to {max_tokens} tokens")
                
                prefixed_texts.append(prefixed_text)
            
            payload = {
                "input": prefixed_texts,
                "model": self.vllm_model_name
            }
            
            async with session.post(f"{self.base_url}/v1/embeddings", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return [item['embedding'] for item in result['data']]
                else:
                    error_text = await response.text()
                    raise Exception(f"VLLM embedding error {response.status}: {error_text}")
    
    # Text embedding methods (use query prefix for retrieval service)
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get embedding for single text (query prefix)"""
        embeddings = await self._embed_batch([text], prefix="query")
        return embeddings[0]
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Sync version for text embedding"""
        return asyncio.run(self._aget_text_embedding(text))
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts (query prefix)"""
        return await self._embed_batch(texts, prefix="query")
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Sync version for text embeddings"""
        return asyncio.run(self._aget_text_embeddings(texts))
    
    # Query embedding methods (same as text for retrieval service)
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get embedding for single query (query prefix)"""
        embeddings = await self._embed_batch([query], prefix="query")
        return embeddings[0]
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Sync version for query embedding"""
        return asyncio.run(self._aget_query_embedding(query))
    
    async def close(self):
        """Close session"""
        if self.session and not self.session.closed:
            await self.session.close()

class AsyncRecursiveRetriever:
    """Async wrapper for RecursiveRetriever with batch processing"""
    
    def __init__(self, config: RetrieveServiceConfig):
        self.config = config
        self.embed_model = VLLMQueryEmbedding(
            base_url=config.VLLM_BASE_URL,
            model_name=config.VLLM_MODEL_NAME
        )
        
        # Initialize QdrantVectorStore (DOĞRU YÖNTEM)
        qdrant_client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
        )
        
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=config.QDRANT_COLLECTION_NAME,
        )
        
        # Load VectorStoreIndex from vector store (DOĞRU YÖNTEM)
        self.vector_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self.embed_model
        )
        
        # Load node mapping (CRITICAL for RecursiveRetriever)
        self.node_dict = self._load_node_mapping()
        
        # Create RecursiveRetriever (EXACTLY like documentation)
        self.recursive_retriever = self._create_recursive_retriever()
        
        logger.info(f"✅ Initialized AsyncRecursiveRetriever with {len(self.node_dict)} nodes")
    
    def _load_node_mapping(self) -> Dict[str, BaseNode]:
        """Load node mapping from JSON file (created by tree builder)"""
        node_mapping_path = Path("tree_data/node_mapping.json")
        
        if not node_mapping_path.exists():
            raise FileNotFoundError(f"Node mapping file not found: {node_mapping_path}")
        
        logger.info(f"📋 Loading node mapping from: {node_mapping_path}")
        
        with open(node_mapping_path, 'r', encoding='utf-8') as f:
            serialized_nodes = json.load(f)
        
        node_dict = {}
        for node_id, node_data in serialized_nodes.items():
            # Reconstruct nodes based on type
            if node_data.get("node_type") == "IndexNode" and node_data.get("index_id"):
                # This is an IndexNode (chunk or metadata reference)
                node = IndexNode(
                    id_=node_id,
                    text=node_data["text"],
                    index_id=node_data["index_id"],
                    metadata=node_data.get("metadata", {})
                )
            else:
                # This is a BaseNode
                node = BaseNode(
                    id_=node_id,
                    text=node_data["text"],
                    metadata=node_data.get("metadata", {})
                )
            
            node_dict[node_id] = node
        
        logger.info(f"📋 Loaded {len(node_dict)} nodes from mapping file")
        return node_dict
    
    def _create_recursive_retriever(self) -> RecursiveRetriever:
        """Create RecursiveRetriever (EXACTLY like documentation)"""
        vector_retriever = self.vector_index.as_retriever()
        
        recursive_retriever = RecursiveRetriever(
            root_id="vector",
            retriever_dict={"vector": vector_retriever},
            node_dict=self.node_dict,
            verbose=False
        )
        
        return recursive_retriever
    
    async def retrieve_single(self, query: str, top_k: int = None, similarity_cutoff: float = None) -> List[Dict[str, Any]]:
        """Retrieve for single query"""
        top_k = top_k or self.config.DEFAULT_TOP_K
        similarity_cutoff = similarity_cutoff or self.config.DEFAULT_SIMILARITY_CUTOFF
        
        # Update vector retriever parameters
        vector_retriever = self.vector_index.as_retriever(
            similarity_top_k=top_k,
            similarity_cutoff=similarity_cutoff
        )
        
        # Update recursive retriever
        self.recursive_retriever.retriever_dict["vector"] = vector_retriever
        
        # Perform recursive retrieval (LlamaIndex handles the magic!)
        nodes_with_scores = self.recursive_retriever.retrieve(query)
        
        # Format results
        results = []
        for node_with_score in nodes_with_scores:
            result = {
                "node_id": node_with_score.node.node_id,
                "text": node_with_score.node.get_content(),
                "score": node_with_score.score if hasattr(node_with_score, 'score') else 1.0,
                "metadata": node_with_score.node.metadata,
            }
            
            # Add reference information if it's an IndexNode
            if hasattr(node_with_score.node, 'index_id'):
                result["index_id"] = node_with_score.node.index_id
                result["node_type"] = "reference"
            else:
                result["node_type"] = "base"
            
            results.append(result)
        
        return results
    
    async def retrieve_batch(self, queries: List[str], top_k: int = None, similarity_cutoff: float = None) -> List[List[Dict[str, Any]]]:
        """Retrieve for multiple queries with async processing"""
        # Limit concurrent requests
        semaphore = asyncio.Semaphore(self.config.MAX_CONCURRENT_REQUESTS)
        
        async def retrieve_with_semaphore(query: str):
            async with semaphore:
                return await self.retrieve_single(query, top_k, similarity_cutoff)
        
        # Process all queries concurrently
        results = await asyncio.gather(*[
            retrieve_with_semaphore(query) for query in queries
        ])
        
        return results
    
    async def close(self):
        """Close resources"""
        await self.embed_model.close()

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class RetrieveRequest(BaseModel):
    """Single retrieve request"""
    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    top_k: Optional[int] = Field(None, ge=1, le=RetrieveServiceConfig.MAX_TOP_K, description="Number of results")
    similarity_cutoff: Optional[float] = Field(None, ge=0.0, le=1.0, description="Similarity cutoff")

class BatchRetrieveRequest(BaseModel):
    """Batch retrieve request"""
    queries: List[str] = Field(..., min_items=1, max_items=RetrieveServiceConfig.MAX_BATCH_SIZE, description="List of queries")
    top_k: Optional[int] = Field(None, ge=1, le=RetrieveServiceConfig.MAX_TOP_K, description="Number of results per query")
    similarity_cutoff: Optional[float] = Field(None, ge=0.0, le=1.0, description="Similarity cutoff")
    
    @validator('queries')
    def validate_queries(cls, v):
        """Validate query list"""
        for i, query in enumerate(v):
            if not query.strip():
                raise ValueError(f"Query {i} cannot be empty")
            if len(query) > 2000:
                raise ValueError(f"Query {i} too long (max 2000 chars)")
        return v

class RetrieveResult(BaseModel):
    """Single retrieve result"""
    node_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    node_type: str
    index_id: Optional[str] = None

class RetrieveResponse(BaseModel):
    """Single retrieve response"""
    query: str
    results: List[RetrieveResult]
    retrieval_time_ms: float
    total_results: int

class BatchRetrieveResponse(BaseModel):
    """Batch retrieve response"""
    queries: List[str]
    results: List[List[RetrieveResult]]
    retrieval_time_ms: float
    total_queries: int

# =============================================================================
# FASTAPI APP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global recursive_retriever
    
    # Startup
    logger.info("🚀 Starting LlamaIndex Retrieve Service...")
    try:
        RetrieveServiceConfig.validate()
        RetrieveServiceConfig.log_config()
        
        recursive_retriever = AsyncRecursiveRetriever(RetrieveServiceConfig)
        logger.info("✅ Recursive retrieve service initialized")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down retrieve service...")
    if recursive_retriever:
        await recursive_retriever.close()

app = FastAPI(
    title="LlamaIndex RecursiveRetriever Service",
    description="High-performance recursive retrieval with VLLM and Qdrant",
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
# AUTH
# =============================================================================

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """Verify API key if configured"""
    if not RetrieveServiceConfig.API_KEY:
        return "no-auth"
    
    if credentials.credentials != RetrieveServiceConfig.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    return credentials.credentials

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not recursive_retriever:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "status": "healthy",
        "service": "llamaindex_recursive_retriever",
        "qdrant_collection": RetrieveServiceConfig.QDRANT_COLLECTION_NAME,
        "node_count": len(recursive_retriever.node_dict),
        "vllm_url": RetrieveServiceConfig.VLLM_BASE_URL,
        "pattern": "RecursiveRetriever with IndexNode references"
    }

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_single(
    request: RetrieveRequest,
    api_key: str = Depends(verify_api_key) if security else None
):
    """Single query retrieval with RecursiveRetriever"""
    if not recursive_retriever:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    start_time = time.perf_counter()
    
    try:
        results = await recursive_retriever.retrieve_single(
            query=request.query,
            top_k=request.top_k,
            similarity_cutoff=request.similarity_cutoff
        )
        
        retrieval_time = (time.perf_counter() - start_time) * 1000
        
        return RetrieveResponse(
            query=request.query,
            results=[RetrieveResult(**result) for result in results],
            retrieval_time_ms=retrieval_time,
            total_results=len(results)
        )
        
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve/batch", response_model=BatchRetrieveResponse)
async def retrieve_batch(
    request: BatchRetrieveRequest,
    api_key: str = Depends(verify_api_key) if security else None
):
    """Batch query retrieval with async RecursiveRetriever"""
    if not recursive_retriever:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    start_time = time.perf_counter()
    
    try:
        batch_results = await recursive_retriever.retrieve_batch(
            queries=request.queries,
            top_k=request.top_k,
            similarity_cutoff=request.similarity_cutoff
        )
        
        retrieval_time = (time.perf_counter() - start_time) * 1000
        
        # Format batch results
        formatted_results = []
        for query_results in batch_results:
            formatted_results.append([
                RetrieveResult(**result) for result in query_results
            ])
        
        return BatchRetrieveResponse(
            queries=request.queries,
            results=formatted_results,
            retrieval_time_ms=retrieval_time,
            total_queries=len(request.queries)
        )
        
    except Exception as e:
        logger.error(f"Batch retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get service statistics"""
    if not recursive_retriever:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Count different node types
    node_types = {"IndexNode": 0, "BaseNode": 0}
    reference_types = {}
    
    for node in recursive_retriever.node_dict.values():
        if hasattr(node, 'index_id'):
            node_types["IndexNode"] += 1
            metadata_type = node.metadata.get('metadata_type', 'chunk_reference')
            reference_types[metadata_type] = reference_types.get(metadata_type, 0) + 1
        else:
            node_types["BaseNode"] += 1
    
    return {
        "total_nodes": len(recursive_retriever.node_dict),
        "node_types": node_types,
        "reference_types": reference_types,
        "collection_name": RetrieveServiceConfig.QDRANT_COLLECTION_NAME,
        "retriever_type": "RecursiveRetriever",
        "config": {
            "default_top_k": RetrieveServiceConfig.DEFAULT_TOP_K,
            "max_top_k": RetrieveServiceConfig.MAX_TOP_K,
            "max_batch_size": RetrieveServiceConfig.MAX_BATCH_SIZE,
            "max_concurrent_requests": RetrieveServiceConfig.MAX_CONCURRENT_REQUESTS
        }
    }

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 LlamaIndex RecursiveRetriever Service")
    logger.info("=" * 60)
    logger.info(f"Collection: {RetrieveServiceConfig.QDRANT_COLLECTION_NAME}")
    logger.info(f"VLLM URL: {RetrieveServiceConfig.VLLM_BASE_URL}")
    logger.info(f"Server: {RetrieveServiceConfig.HOST}:{RetrieveServiceConfig.PORT}")
    logger.info("🔗 Pattern: RecursiveRetriever with IndexNode references")
    logger.info("=" * 60)
    
    uvicorn.run(
        app,
        host=RetrieveServiceConfig.HOST,
        port=RetrieveServiceConfig.PORT,
        log_level=RetrieveServiceConfig.LOG_LEVEL.lower(),
        access_log=True,
        # Use uvloop for better async performance
        loop="uvloop",
    )