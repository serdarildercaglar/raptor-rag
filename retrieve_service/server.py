#!/usr/bin/env python3
"""
Official LlamaIndex RecursiveRetriever Pattern Implementation
Based on official LlamaIndex documentation and tutorials
"""
import asyncio
import logging
import time
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

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
    TREE_DATA_PATH: str = "./tree_data"
    
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

class VLLMQueryEmbedding:
    """VLLM Embedding with query prefix for retrieval"""
    
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

class OfficialRecursiveRetriever:
    """
    Official LlamaIndex RecursiveRetriever Pattern Implementation
    
    Based on: https://docs.llamaindex.ai/en/stable/examples/retrievers/recursive_retriever_nodes/
    
    Key Principles:
    1. Vector search returns IndexNode objects
    2. For IndexNode: follow index_id → get referenced node from node_dict
    3. Return referenced nodes (not reference nodes!)
    4. Simple deduplication by node_id
    5. Fallback: if node_dict missing, return original result
    """
    
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
        
        # Load node mapping (following official pattern)
        self.node_dict = self._load_node_mapping()
        
        logger.info(f"✅ Official RecursiveRetriever initialized with {len(self.node_dict)} nodes")
    
    def _load_node_mapping(self) -> Dict[str, BaseNode]:
        """Load node mapping from JSON file (official pattern)"""
        import json
        from pathlib import Path
        
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
                "\n".join(f"  - {p}" for p in available_paths)
            )
        
        logger.info(f"📋 Loading node mapping from: {node_mapping_path}")
        
        with open(node_mapping_path, 'r', encoding='utf-8') as f:
            serialized_nodes = json.load(f)
        
        node_dict = {}
        for node_id, node_data in serialized_nodes.items():
            # Reconstruct nodes based on type (official pattern)
            if node_data.get("node_type") == "IndexNode" and node_data.get("index_id"):
                # This is an IndexNode (reference)
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
    
    def _create_node_with_score(self, node: BaseNode, score: float) -> NodeWithScore:
        """Create NodeWithScore object (official pattern)"""
        return NodeWithScore(node=node, score=score)
    
    def _deduplicate_nodes(self, nodes_with_score: List[NodeWithScore]) -> List[NodeWithScore]:
        """Simple deduplication by node_id (official pattern)"""
        seen_ids = set()
        deduplicated = []
        
        for node_with_score in nodes_with_score:
            node_id = node_with_score.node.node_id
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                deduplicated.append(node_with_score)
        
        return deduplicated
    
    def _follow_node_references(self, nodes_with_score: List[NodeWithScore]) -> List[NodeWithScore]:
        """
        Follow node references (core official pattern)
        
        Logic:
        1. For IndexNode with index_id: get referenced node from node_dict
        2. For regular BaseNode: keep as is
        3. Fallback: if referenced node not found, keep original
        """
        result_nodes = []
        
        for node_with_score in nodes_with_score:
            node = node_with_score.node
            score = node_with_score.score
            
            # Check if this is an IndexNode with reference
            if isinstance(node, IndexNode) and hasattr(node, 'index_id') and node.index_id:
                # This is a reference node - follow the reference
                referenced_node_id = node.index_id
                
                if referenced_node_id in self.node_dict:
                    # Get the referenced node (this is the main content)
                    referenced_node = self.node_dict[referenced_node_id]
                    result_nodes.append(self._create_node_with_score(referenced_node, score))
                    
                    logger.debug(f"Followed reference: {node.node_id} → {referenced_node_id}")
                else:
                    # Fallback: referenced node not found, keep original
                    result_nodes.append(node_with_score)
                    logger.warning(f"Referenced node not found: {referenced_node_id}, keeping original")
            else:
                # Regular node or IndexNode without reference - keep as is
                result_nodes.append(node_with_score)
        
        return result_nodes
    
    async def retrieve_single(self, query: str, top_k: int = None, similarity_cutoff: float = None) -> List[Dict[str, Any]]:
        """Single query retrieval using official RecursiveRetriever pattern"""
        top_k = top_k or self.config.DEFAULT_TOP_K
        similarity_cutoff = similarity_cutoff or self.config.DEFAULT_SIMILARITY_CUTOFF
        
        # Step 1: Get query embedding
        query_embedding = await self.embed_model.get_query_embedding(query)
        
        # Step 2: Vector search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.config.QDRANT_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=similarity_cutoff if similarity_cutoff > 0 else None
        )
        
        # Step 3: Convert Qdrant results to NodeWithScore objects
        nodes_with_score = []
        for result in search_results:
            payload = result.payload
            node_id = payload.get("node_id")
            
            # Get node from node_dict (official pattern)
            if node_id in self.node_dict:
                node = self.node_dict[node_id]
                node_with_score = self._create_node_with_score(node, result.score)
                nodes_with_score.append(node_with_score)
            else:
                # Fallback: create node from payload if not in node_dict
                logger.warning(f"Node {node_id} not in node_dict, creating from payload")
                fallback_node = BaseNode(
                    id_=node_id,
                    text=payload.get("text", ""),
                    metadata=payload.get("metadata", {})
                )
                node_with_score = self._create_node_with_score(fallback_node, result.score)
                nodes_with_score.append(node_with_score)
        
        # Step 4: Follow references (official RecursiveRetriever core logic)
        retrieved_nodes = self._follow_node_references(nodes_with_score)
        
        # Step 5: Deduplicate (official pattern)
        final_nodes = self._deduplicate_nodes(retrieved_nodes)
        
        # Step 6: Convert to response format
        results = []
        for node_with_score in final_nodes:
            node = node_with_score.node
            
            result_data = {
                "node_id": node.node_id,
                "text": node.get_content(),
                "score": node_with_score.score,
                "metadata": node.metadata,
                "node_type": "IndexNode" if isinstance(node, IndexNode) else "BaseNode"
            }
            
            # Add reference info if it's an IndexNode
            if isinstance(node, IndexNode) and hasattr(node, 'index_id'):
                result_data["index_id"] = node.index_id
            
            results.append(result_data)
        
        logger.debug(f"Retrieved {len(results)} nodes for query: {query[:50]}...")
        return results
    
    async def retrieve_batch(self, queries: List[str], top_k: int = None, similarity_cutoff: float = None) -> List[List[Dict[str, Any]]]:
        """Batch query retrieval with async processing (official pattern)"""
        # Limit concurrent requests
        semaphore = asyncio.Semaphore(self.config.MAX_CONCURRENT_REQUESTS)
        
        async def retrieve_with_semaphore(query: str):
            async with semaphore:
                return await self.retrieve_single(query, top_k, similarity_cutoff)
        
        # Process all queries concurrently (official async pattern)
        results = await asyncio.gather(*[
            retrieve_with_semaphore(query) for query in queries
        ])
        
        return results
    
    async def close(self):
        """Close resources"""
        await self.embed_model.close()

# Global variables
recursive_retriever: Optional[OfficialRecursiveRetriever] = None
security = HTTPBearer() if RetrieveServiceConfig.API_KEY else None

# =============================================================================
# PYDANTIC MODELS (same as before)
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
    logger.info("🚀 Starting Official RecursiveRetriever Service...")
    try:
        recursive_retriever = OfficialRecursiveRetriever(RetrieveServiceConfig)
        logger.info("✅ Official RecursiveRetriever service initialized")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down retrieve service...")
    if recursive_retriever:
        await recursive_retriever.close()

app = FastAPI(
    title="Official LlamaIndex RecursiveRetriever Service",
    description="Official LlamaIndex RecursiveRetriever pattern with VLLM and Qdrant",
    version="3.0.0",
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
        "service": "official_recursive_retriever",
        "qdrant_collection": RetrieveServiceConfig.QDRANT_COLLECTION_NAME,
        "node_count": len(recursive_retriever.node_dict),
        "vllm_url": RetrieveServiceConfig.VLLM_BASE_URL,
        "pattern": "Official LlamaIndex RecursiveRetriever Pattern"
    }

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_single(
    request: RetrieveRequest,
    api_key: str = Depends(verify_api_key) if security else None
):
    """Single query retrieval with Official RecursiveRetriever"""
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
    """Batch query retrieval with async Official RecursiveRetriever"""
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
    reference_count = 0
    
    for node in recursive_retriever.node_dict.values():
        if isinstance(node, IndexNode):
            node_types["IndexNode"] += 1
            if hasattr(node, 'index_id') and node.index_id:
                reference_count += 1
        else:
            node_types["BaseNode"] += 1
    
    return {
        "total_nodes": len(recursive_retriever.node_dict),
        "node_types": node_types,
        "nodes_with_references": reference_count,
        "collection_name": RetrieveServiceConfig.QDRANT_COLLECTION_NAME,
        "retriever_type": "OfficialRecursiveRetriever",
        "pattern": "Official LlamaIndex RecursiveRetriever Pattern",
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
# =============================================================================
# MAIN WITH WORKERS
# =============================================================================

if __name__ == "__main__":
    import os
    
    logger.info("=" * 60)
    logger.info("🚀 Official LlamaIndex RecursiveRetriever Service")
    logger.info("=" * 60)
    logger.info(f"Collection: {RetrieveServiceConfig.QDRANT_COLLECTION_NAME}")
    logger.info(f"VLLM URL: {RetrieveServiceConfig.VLLM_BASE_URL}")
    logger.info(f"Server: {RetrieveServiceConfig.HOST}:{RetrieveServiceConfig.PORT}")
    logger.info("🎯 Pattern: Official LlamaIndex RecursiveRetriever")
    
    # Worker configuration
    workers = int(os.getenv("WORKERS", 2))  # Default 2 workers
    logger.info(f"Workers: {workers}")
    logger.info(f"Note: Each worker will load its own node_dict and connect to VLLM/Qdrant")
    logger.info("=" * 60)
    
    uvicorn.run(
        "server:app",  # Import string format for workers
        host=RetrieveServiceConfig.HOST,
        port=RetrieveServiceConfig.PORT,
        log_level="info",
        access_log=True,
        workers=workers,
        # Use uvloop for better async performance  
        loop="uvloop",
        # HTTP/2 support
        http="h11",
    )