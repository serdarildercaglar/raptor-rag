#!/usr/bin/env python3
"""
Fixed LlamaIndex Retrieve Service with VLLM Embeddings
Pydantic sorunlarını çözdük ve RecursiveRetriever'ı düzelttik
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
from pydantic import BaseModel, Field, field_validator
import uvicorn

# LlamaIndex Core
from llama_index.core.schema import NodeWithScore, BaseNode, IndexNode

# Qdrant Client
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# VLLM Integration
import aiohttp
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RetrieveServiceConfig:
    """Configuration for Retrieve Service"""
    
    # VLLM Embedding Service (Query Mode)
    VLLM_BASE_URL: str = "http://localhost:8008"
    VLLM_MODEL_NAME: str = "intfloat/multilingual-e5-large"
    
    # Qdrant Configuration
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "llamaindex_tree"
    
    # Tree Data Path
    TREE_DATA_PATH: str = "./tree_data"  # Default path
    
    # FastAPI Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_KEY: Optional[str] = None
    
    # Retrieval Configuration
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 50
    DEFAULT_SIMILARITY_CUTOFF: float = 0.0
    MAX_BATCH_SIZE: int = 100
    
    # Performance
    TIMEOUT_SECONDS: int = 30
    MAX_CONCURRENT_REQUESTS: int = 10
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration"""
        # Test Qdrant connection
        try:
            client = QdrantClient(url=cls.QDRANT_URL, api_key=cls.QDRANT_API_KEY)
            collections = client.get_collections()
            
            # Check if collection exists
            collection_names = [col.name for col in collections.collections]
            if cls.QDRANT_COLLECTION_NAME not in collection_names:
                raise ValueError(f"Qdrant collection '{cls.QDRANT_COLLECTION_NAME}' not found. Available: {collection_names}")
                
        except Exception as e:
            raise ValueError(f"Qdrant connection failed: {e}")
        
        # Test VLLM connection
        try:
            import requests
            response = requests.get(f"{cls.VLLM_BASE_URL}/health", timeout=5)
            if response.status_code != 200:
                raise ValueError(f"VLLM service not healthy: {response.status_code}")
        except Exception as e:
            raise ValueError(f"VLLM connection failed: {e}")
        
        return True
    
    @classmethod
    def log_config(cls):
        """Log configuration (without sensitive data)"""
        print("🔧 Retrieve Service Configuration:")
        print(f"   Server: {cls.HOST}:{cls.PORT}")
        print(f"   VLLM URL: {cls.VLLM_BASE_URL}")
        print(f"   Qdrant URL: {cls.QDRANT_URL}")
        print(f"   Collection: {cls.QDRANT_COLLECTION_NAME}")
        print(f"   Tree Data Path: {cls.TREE_DATA_PATH}")
        print(f"   Default Top-K: {cls.DEFAULT_TOP_K}")
        print(f"   Max Batch Size: {cls.MAX_BATCH_SIZE}")
        print(f"   API Key: {'✅ Set' if cls.API_KEY else '❌ None'}")
        print("=" * 50)

class VLLMQueryEmbedding:
    """Fixed VLLM Embedding with query prefix for retrieval"""
    
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.session = None
        self._tokenizer = tiktoken.get_encoding("o200k_base")
        
    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=RetrieveServiceConfig.TIMEOUT_SECONDS)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def _embed_batch(self, texts: List[str], prefix: str = "query") -> List[List[float]]:
        """Embed batch of texts with prefix and auto-truncation"""
        await self._ensure_session()
        
        # Add prefix and truncate if needed
        max_tokens = 400  # Leave margin for prefix
        
        prefixed_texts = []
        for text in texts:
            prefixed_text = f"{prefix}: {text}"
            
            # Truncate if too long
            tokens = self._tokenizer.encode(prefixed_text)
            if len(tokens) > max_tokens:
                # Truncate tokens and decode back
                truncated_tokens = tokens[:max_tokens]
                prefixed_text = self._tokenizer.decode(truncated_tokens)
                logger.debug(f"Query truncated from {len(tokens)} to {max_tokens} tokens")
            
            prefixed_texts.append(prefixed_text)
        
        payload = {
            "input": prefixed_texts,
            "model": self.model_name
        }
        
        async with self.session.post(f"{self.base_url}/v1/embeddings", json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return [item['embedding'] for item in result['data']]
            else:
                error_text = await response.text()
                raise Exception(f"VLLM embedding error {response.status}: {error_text}")
    
    async def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for single query (query prefix)"""
        embeddings = await self._embed_batch([query], prefix="query")
        return embeddings[0]
    
    async def get_query_embeddings(self, queries: List[str]) -> List[List[float]]:
        """Get embeddings for multiple queries (query prefix)"""
        return await self._embed_batch(queries, prefix="query")
    
    async def close(self):
        """Close session"""
        if self.session and not self.session.closed:
            await self.session.close()

class FixedRecursiveRetriever:
    """Fixed implementation of RecursiveRetriever using direct Qdrant queries"""
    
    def __init__(self, config: RetrieveServiceConfig):
        self.config = config
        self.embed_model = VLLMQueryEmbedding(
            base_url=config.VLLM_BASE_URL,
            model_name=config.VLLM_MODEL_NAME
        )
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
        )
        
        # Load node mapping
        self.node_dict = self._load_node_mapping()
        
        logger.info(f"✅ Initialized FixedRecursiveRetriever with {len(self.node_dict)} nodes")
    
    def _load_node_mapping(self) -> Dict[str, BaseNode]:
        """Load node mapping from JSON file"""
        # Try different possible locations
        possible_paths = [
            Path(self.config.TREE_DATA_PATH) / "node_mapping.json",
            Path("tree_data/node_mapping.json"),
            Path("../tree_builder/tree_data/node_mapping.json"),
            Path("./tree_builder/tree_data/node_mapping.json")
        ]
        
        node_mapping_path = None
        for path in possible_paths:
            if path.exists():
                node_mapping_path = path
                break
        
        if not node_mapping_path:
            available_paths = [str(p) for p in possible_paths]
            raise FileNotFoundError(
                f"Node mapping file not found in any of these locations:\n" +
                "\n".join(f"  - {p}" for p in available_paths) +
                f"\n\nPlease run the tree builder first or set the correct TREE_DATA_PATH"
            )
        
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
    
    async def retrieve_single(self, query: str, top_k: int = None, similarity_cutoff: float = None) -> List[Dict[str, Any]]:
        """Retrieve for single query using direct Qdrant search"""
        top_k = top_k or self.config.DEFAULT_TOP_K
        similarity_cutoff = similarity_cutoff or self.config.DEFAULT_SIMILARITY_CUTOFF
        
        # Get query embedding
        query_embedding = await self.embed_model.get_query_embedding(query)
        
        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.config.QDRANT_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=similarity_cutoff if similarity_cutoff > 0 else None
        )
        
        # Process results and follow references
        final_results = []
        seen_nodes = set()
        
        for result in search_results:
            payload = result.payload
            node_id = payload.get("node_id")
            
            # Skip if we've already seen this node
            if node_id in seen_nodes:
                continue
            
            # Get the actual node (for reference following)
            if node_id in self.node_dict:
                node = self.node_dict[node_id]
                
                # If this is an IndexNode with reference, follow it
                if hasattr(node, 'index_id') and node.index_id:
                    # Get the referenced node
                    referenced_node_id = node.index_id
                    if referenced_node_id in self.node_dict:
                        referenced_node = self.node_dict[referenced_node_id]
                        
                        # Add the referenced node (the actual content)
                        if referenced_node_id not in seen_nodes:
                            final_result = {
                                "node_id": referenced_node_id,
                                "text": referenced_node.get_content(),
                                "score": result.score,
                                "metadata": referenced_node.metadata,
                                "node_type": "base",
                                "reference_path": f"{node_id} -> {referenced_node_id}"
                            }
                            final_results.append(final_result)
                            seen_nodes.add(referenced_node_id)
                
                # Also add the original node (the reference itself)
                result_data = {
                    "node_id": node_id,
                    "text": node.get_content(),
                    "score": result.score,
                    "metadata": node.metadata,
                }
                
                if hasattr(node, 'index_id') and node.index_id:
                    result_data["index_id"] = node.index_id
                    result_data["node_type"] = "reference"
                else:
                    result_data["node_type"] = "base"
                
                final_results.append(result_data)
                seen_nodes.add(node_id)
        
        return final_results
    
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

# Global variables
recursive_retriever: Optional[FixedRecursiveRetriever] = None
security = HTTPBearer() if RetrieveServiceConfig.API_KEY else None

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
    
    @field_validator('queries')
    @classmethod
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
    reference_path: Optional[str] = None

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
    logger.info("🚀 Starting Fixed Retrieve Service...")
    try:
        RetrieveServiceConfig.validate()
        RetrieveServiceConfig.log_config()
        
        recursive_retriever = FixedRecursiveRetriever(RetrieveServiceConfig)
        logger.info("✅ Fixed recursive retrieve service initialized")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down retrieve service...")
    if recursive_retriever:
        await recursive_retriever.close()

app = FastAPI(
    title="Fixed LlamaIndex RecursiveRetriever Service",
    description="High-performance recursive retrieval with VLLM and Qdrant",
    version="2.1.0",
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
        "service": "fixed_recursive_retriever",
        "qdrant_collection": RetrieveServiceConfig.QDRANT_COLLECTION_NAME,
        "node_count": len(recursive_retriever.node_dict),
        "vllm_url": RetrieveServiceConfig.VLLM_BASE_URL,
        "pattern": "Fixed RecursiveRetriever with direct Qdrant queries"
    }

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_single(
    request: RetrieveRequest,
    api_key: str = Depends(verify_api_key) if security else None
):
    """Single query retrieval with Fixed RecursiveRetriever"""
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
    """Batch query retrieval with async Fixed RecursiveRetriever"""
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
        "retriever_type": "FixedRecursiveRetriever",
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
    logger.info("🚀 Fixed LlamaIndex RecursiveRetriever Service")
    logger.info("=" * 60)
    logger.info(f"Collection: {RetrieveServiceConfig.QDRANT_COLLECTION_NAME}")
    logger.info(f"VLLM URL: {RetrieveServiceConfig.VLLM_BASE_URL}")
    logger.info(f"Server: {RetrieveServiceConfig.HOST}:{RetrieveServiceConfig.PORT}")
    logger.info("🔧 Pattern: Fixed RecursiveRetriever with direct Qdrant queries")
    logger.info("=" * 60)
    
    uvicorn.run(
        app,
        host=RetrieveServiceConfig.HOST,
        port=RetrieveServiceConfig.PORT,
        log_level="info",
        access_log=True,
    )