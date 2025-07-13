#!/usr/bin/env python3
"""
Fixed LlamaIndex Tree Builder with VLLM Embeddings
Pydantic hatalarını çözdük ve production ready hale getirdik
"""
import asyncio
import json, os
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from tqdm import tqdm
import nest_asyncio

# LlamaIndex Core
from llama_index.core import Document, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode, BaseNode
from llama_index.core.extractors import SummaryExtractor, QuestionsAnsweredExtractor
from llama_index.core import VectorStoreIndex
from llama_index.readers.file import PDFReader

# LlamaIndex LLM
from llama_index.llms.openai import OpenAI

# Vector Store
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# VLLM Integration
import aiohttp
import tiktoken

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enable nested asyncio
nest_asyncio.apply()

class VLLMEmbedding:
    """
    Fixed VLLM Embedding implementation for LlamaIndex
    Pydantic sorunlarını çözdük - artık BaseEmbedding'den inherit etmiyoruz
    """
    
    def __init__(self, base_url: str, model_name: str, **kwargs):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.session = None
        self._tokenizer = tiktoken.get_encoding("o200k_base")
        
    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def _embed_batch(self, texts: List[str], prefix: str = "passage") -> List[List[float]]:
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
                logger.debug(f"Text truncated from {len(tokens)} to {max_tokens} tokens")
            
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
    
    # LlamaIndex compatible methods
    async def aget_text_embedding(self, text: str) -> List[float]:
        """Get embedding for single text (passage prefix)"""
        embeddings = await self._embed_batch([text], prefix="passage")
        return embeddings[0]
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Sync version for text embedding"""
        return asyncio.run(self.aget_text_embedding(text))
    
    async def aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts (passage prefix)"""
        return await self._embed_batch(texts, prefix="passage")
    
    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Sync version for text embeddings"""
        return asyncio.run(self.aget_text_embeddings(texts))
    
    async def aget_query_embedding(self, query: str) -> List[float]:
        """Get embedding for single query (query prefix)"""
        embeddings = await self._embed_batch([query], prefix="query")
        return embeddings[0]
    
    def get_query_embedding(self, query: str) -> List[float]:
        """Sync version for query embedding"""
        return asyncio.run(self.aget_query_embedding(query))
    
    async def close(self):
        """Close session"""
        if self.session and not self.session.closed:
            await self.session.close()

class TreeBuilderConfig:
    """Configuration for Tree Builder"""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = "YOUR_OPENAI_API_KEY"  # Set this!
    
    # VLLM Embedding Service
    VLLM_BASE_URL: str = "http://localhost:8008"
    VLLM_MODEL_NAME: str = "intfloat/multilingual-e5-large"
    
    # Qdrant Configuration
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "llamaindex_tree"
    
    # Document Processing
    DOCUMENTS_FOLDER: Path = Path("./documents")
    OUTPUT_PATH: Path = Path("./tree_data")
    
    # Chunking Configuration
    BASE_CHUNK_SIZE: int = 1024
    SUB_CHUNK_SIZES: List[int] = [128, 256, 512]
    CHUNK_OVERLAP: int = 20
    
    # Metadata Extraction
    NUM_QUESTIONS: int = 3  # Reduced for faster processing
    ENABLE_SUMMARIES: bool = True
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration"""
        if not cls.OPENAI_API_KEY or cls.OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
            raise ValueError("OPENAI_API_KEY is required for metadata extraction")
        
        if not cls.DOCUMENTS_FOLDER.exists():
            raise ValueError(f"Documents folder does not exist: {cls.DOCUMENTS_FOLDER}")
        
        # Create output directory if it doesn't exist
        cls.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        
        return True
    
    @classmethod
    def log_config(cls):
        """Log configuration (without sensitive data)"""
        print("🔧 Tree Builder Configuration:")
        print(f"   Documents Folder: {cls.DOCUMENTS_FOLDER}")
        print(f"   Output Path: {cls.OUTPUT_PATH}")
        print(f"   VLLM URL: {cls.VLLM_BASE_URL}")
        print(f"   Qdrant URL: {cls.QDRANT_URL}")
        print(f"   Base Chunk Size: {cls.BASE_CHUNK_SIZE}")
        print(f"   Sub Chunk Sizes: {cls.SUB_CHUNK_SIZES}")
        print(f"   Enable Summaries: {cls.ENABLE_SUMMARIES}")
        print(f"   Num Questions: {cls.NUM_QUESTIONS}")
        print("=" * 50)

class FixedTreeBuilder:
    """Fixed LlamaIndex Tree Builder with VLLM Embeddings"""
    
    def __init__(self, config: TreeBuilderConfig):
        self.config = config
        
        # Initialize LLM for summarization (GPT-4o)
        self.llm = OpenAI(model="gpt-4o-mini", api_key=config.OPENAI_API_KEY)
        
        # Initialize VLLM embedding
        self.embed_model = VLLMEmbedding(
            base_url=config.VLLM_BASE_URL,
            model_name=config.VLLM_MODEL_NAME
        )
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
        )
        
        # Node parsers for different chunk sizes
        self.base_parser = SentenceSplitter(
            chunk_size=config.BASE_CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        self.sub_parsers = [
            SentenceSplitter(chunk_size=size, chunk_overlap=config.CHUNK_OVERLAP)
            for size in config.SUB_CHUNK_SIZES
        ]
        
        # Metadata extractors
        self.extractors = []
        if config.ENABLE_SUMMARIES:
            self.extractors.append(
                SummaryExtractor(summaries=["self"], llm=self.llm, show_progress=True)
            )
        
        if config.NUM_QUESTIONS > 0:
            self.extractors.append(
                QuestionsAnsweredExtractor(
                    questions=config.NUM_QUESTIONS, 
                    llm=self.llm,
                    show_progress=True
                )
            )
    
    def load_documents(self) -> List[Document]:
        """Load PDF and TXT documents from folder"""
        logger.info(f"📁 Loading documents from: {self.config.DOCUMENTS_FOLDER}")
        
        documents = []
        
        # Load PDF files
        pdf_files = list(self.config.DOCUMENTS_FOLDER.glob("*.pdf"))
        if pdf_files:
            logger.info(f"📄 Found {len(pdf_files)} PDF files")
            pdf_reader = PDFReader()
            for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
                try:
                    docs = pdf_reader.load_data(file=pdf_file)
                    for doc in docs:
                        doc.metadata["source"] = str(pdf_file)
                        doc.metadata["file_type"] = "pdf"
                    documents.extend(docs)
                except Exception as e:
                    logger.error(f"Error loading {pdf_file}: {e}")
        
        # Load TXT files
        txt_files = list(self.config.DOCUMENTS_FOLDER.glob("*.txt"))
        if txt_files:
            logger.info(f"📝 Found {len(txt_files)} TXT files")
            for txt_file in tqdm(txt_files, desc="Loading TXTs"):
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    doc = Document(
                        text=content,
                        metadata={
                            "source": str(txt_file),
                            "file_type": "txt"
                        }
                    )
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"Error loading {txt_file}: {e}")
        
        logger.info(f"✅ Loaded {len(documents)} documents total")
        return documents
    
    def create_base_nodes(self, documents: List[Document]) -> List[BaseNode]:
        """Create base nodes from documents"""
        logger.info("🔪 Creating base chunks...")
        
        base_nodes = self.base_parser.get_nodes_from_documents(documents)
        
        # Set consistent node IDs (IMPORTANT for RecursiveRetriever)
        for idx, node in enumerate(base_nodes):
            node.id_ = f"base-{idx}"
        
        logger.info(f"✅ Created {len(base_nodes)} base nodes")
        return base_nodes
    
    def create_chunk_references(self, base_nodes: List[BaseNode]) -> List[IndexNode]:
        """Create multi-level chunk references (LlamaIndex RecursiveRetriever pattern)"""
        logger.info("🔗 Creating chunk references...")
        
        all_nodes = []
        
        for base_node in tqdm(base_nodes, desc="Processing base nodes"):
            # Create sub-chunks pointing to base node (EXACTLY like documentation)
            for parser in self.sub_parsers:
                sub_nodes = parser.get_nodes_from_documents([base_node])
                sub_index_nodes = [
                    IndexNode.from_text_node(sub_node, base_node.node_id)
                    for sub_node in sub_nodes
                ]
                all_nodes.extend(sub_index_nodes)
            
            # Add original base node as IndexNode (EXACTLY like documentation)
            original_node = IndexNode.from_text_node(base_node, base_node.node_id)
            all_nodes.append(original_node)
        
        logger.info(f"✅ Created {len(all_nodes)} nodes with chunk references")
        return all_nodes
    
    async def create_metadata_references(self, base_nodes: List[BaseNode]) -> List[IndexNode]:
        """Create metadata references (summaries + questions) - LlamaIndex pattern"""
        logger.info("📋 Creating metadata references...")
        
        if not self.extractors:
            logger.info("No extractors configured, skipping metadata extraction")
            return []
        
        # Extract metadata (EXACTLY like documentation)
        node_to_metadata = {}
        for extractor in self.extractors:
            logger.info(f"Running extractor: {extractor.__class__.__name__}")
            metadata_dicts = extractor.extract(base_nodes)
            
            for node, metadata in zip(base_nodes, metadata_dicts):
                if node.node_id not in node_to_metadata:
                    node_to_metadata[node.node_id] = metadata
                else:
                    node_to_metadata[node.node_id].update(metadata)
        
        # Save metadata cache
        metadata_path = self.config.OUTPUT_PATH / "metadata_cache.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(node_to_metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Saved metadata cache to: {metadata_path}")
        
        # Create IndexNodes from metadata (EXACTLY like documentation)
        metadata_nodes = []
        for node_id, metadata in node_to_metadata.items():
            for key, value in metadata.items():
                if isinstance(value, str) and value.strip():
                    # Create IndexNode pointing to base node
                    metadata_node = IndexNode(text=value, index_id=node_id)
                    metadata_nodes.append(metadata_node)
        
        logger.info(f"✅ Created {len(metadata_nodes)} metadata reference nodes")
        return metadata_nodes
    
    async def build_and_store_index(self, all_nodes: List[IndexNode]):
        """Build VectorStoreIndex and store in Qdrant"""
        logger.info(f"🏗️ Building VectorStoreIndex with {len(all_nodes)} nodes...")
        
        # Create QdrantVectorStore
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.config.QDRANT_COLLECTION_NAME,
        )
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Custom embedding function for LlamaIndex
        class EmbeddingFunctionWrapper:
            def __init__(self, embed_model):
                self.embed_model = embed_model
            
            def get_text_embedding(self, text: str) -> List[float]:
                return self.embed_model.get_text_embedding(text)
            
            def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                return self.embed_model.get_text_embeddings(texts)
            
            def get_query_embedding(self, query: str) -> List[float]:
                return self.embed_model.get_query_embedding(query)
        
        embedding_wrapper = EmbeddingFunctionWrapper(self.embed_model)
        
        # Build index step by step to avoid Pydantic issues
        logger.info("📝 Extracting text from nodes...")
        node_texts = [node.get_content() for node in all_nodes]
        
        logger.info("🔢 Generating embeddings...")
        start_time = time.perf_counter()
        
        # Process in batches to avoid memory issues
        batch_size = 50
        all_embeddings = []
        
        for i in tqdm(range(0, len(node_texts), batch_size), desc="Embedding batches"):
            batch_texts = node_texts[i:i+batch_size]
            batch_embeddings = embedding_wrapper.get_text_embeddings(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        embedding_time = time.perf_counter() - start_time
        logger.info(f"✅ Generated {len(all_embeddings)} embeddings in {embedding_time:.1f}s")
        
        # Create simple VectorStoreIndex (bypassing Pydantic embed_model issues)
        logger.info("🏗️ Building vector index...")
        
        # Use Qdrant directly for insertion
        from qdrant_client.models import Distance, VectorParams, PointStruct
        
        # Ensure collection exists with correct settings
        try:
            collection_info = self.qdrant_client.get_collection(self.config.QDRANT_COLLECTION_NAME)
            logger.info(f"✅ Using existing collection: {self.config.QDRANT_COLLECTION_NAME}")
        except:
            logger.info(f"📦 Creating new collection: {self.config.QDRANT_COLLECTION_NAME}")
            self.qdrant_client.create_collection(
                collection_name=self.config.QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )
        
        # Prepare points for insertion
        points = []
        for i, (node, embedding) in enumerate(zip(all_nodes, all_embeddings)):
            point = PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "node_id": node.node_id,
                    "text": node.get_content(),
                    "metadata": node.metadata,
                    "index_id": getattr(node, 'index_id', None),
                    "node_type": "IndexNode" if hasattr(node, 'index_id') else "BaseNode"
                }
            )
            points.append(point)
        
        # Insert in batches
        batch_size = 100
        for i in tqdm(range(0, len(points), batch_size), desc="Inserting to Qdrant"):
            batch_points = points[i:i+batch_size]
            self.qdrant_client.upsert(
                collection_name=self.config.QDRANT_COLLECTION_NAME,
                points=batch_points
            )
        
        logger.info("✅ VectorStoreIndex built and stored in Qdrant")
        return len(all_nodes)
    
    async def build_tree(self):
        """Main tree building process - Fixed RecursiveRetriever pattern"""
        logger.info("🚀 Starting Fixed Tree Building Process")
        self.config.log_config()
        
        try:
            # 1. Load documents
            documents = self.load_documents()
            if not documents:
                raise ValueError("No documents found!")
            
            # 2. Create base nodes
            base_nodes = self.create_base_nodes(documents)
            
            # 3. Create chunk references (multi-level)
            chunk_nodes = self.create_chunk_references(base_nodes)
            
            # 4. Create metadata references (summaries + questions)
            metadata_nodes = await self.create_metadata_references(base_nodes)
            
            # 5. Combine all nodes
            all_nodes = chunk_nodes + metadata_nodes
            logger.info(f"📊 Total nodes: {len(all_nodes)} ({len(chunk_nodes)} chunks + {len(metadata_nodes)} metadata)")
            
            # 6. Build VectorStoreIndex and store in Qdrant
            total_stored = await self.build_and_store_index(all_nodes)
            
            # 7. Save node mapping for RecursiveRetriever
            all_nodes_dict = {n.node_id: n for n in all_nodes}
            
            node_mapping_path = self.config.OUTPUT_PATH / "node_mapping.json"
            # Serialize node dict for later loading
            serialized_nodes = {}
            for node_id, node in all_nodes_dict.items():
                serialized_nodes[node_id] = {
                    "text": node.get_content(),
                    "metadata": node.metadata,
                    "index_id": getattr(node, 'index_id', None),
                    "node_type": "IndexNode" if hasattr(node, 'index_id') else "BaseNode"
                }
            
            with open(node_mapping_path, 'w', encoding='utf-8') as f:
                json.dump(serialized_nodes, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Saved node mapping to: {node_mapping_path}")
            
            # 8. Save summary statistics
            stats = {
                "total_documents": len(documents),
                "total_base_nodes": len(base_nodes),
                "total_chunk_nodes": len(chunk_nodes),
                "total_metadata_nodes": len(metadata_nodes),
                "total_nodes": len(all_nodes),
                "total_stored": total_stored,
                "qdrant_collection": self.config.QDRANT_COLLECTION_NAME,
                "base_chunk_size": self.config.BASE_CHUNK_SIZE,
                "sub_chunk_sizes": self.config.SUB_CHUNK_SIZES,
                "embedding_dimension": 1024
            }
            
            stats_path = self.config.OUTPUT_PATH / "build_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info("✅ Tree building completed successfully!")
            logger.info(f"📊 Final Stats: {stats}")
            
        except Exception as e:
            logger.error(f"❌ Tree building failed: {e}")
            raise
        finally:
            # Cleanup
            await self.embed_model.close()

async def main():
    """Main entry point"""
    try:
        # Update configuration with your OpenAI key
        TreeBuilderConfig.OPENAI_API_KEY = OPENAI_API_KEY
        
        # Validate configuration
        TreeBuilderConfig.validate()
        
        # Create builder and build tree
        builder = FixedTreeBuilder(TreeBuilderConfig)
        await builder.build_tree()
        
        print("\n🎉 Success! Tree built and stored in Qdrant.")
        print("📋 Next: Test the retriever service")
        
    except Exception as e:
        logger.error(f"❌ Build process failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)