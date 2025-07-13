# builder.py - Main Tree Builder Implementation
"""
LlamaIndex Tree Builder with VLLM Embeddings and LlamaIndex OpenAI LLM
Clean, modular implementation using configuration management
Uses LlamaIndex's built-in async OpenAI LLM for compatibility
"""

import asyncio
import json
import logging
import time
from typing import List, Dict, Optional, Any
from tqdm import tqdm
import nest_asyncio

# Configuration
from config import load_and_validate_config

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

# Enable nested asyncio
nest_asyncio.apply()

# Get logger
logger = logging.getLogger(__name__)

class VLLMEmbedding:
    """
    VLLM Embedding implementation with async batch processing
    """
    
    def __init__(self, base_url: str, model_name: str, max_tokens: int = 400):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.max_tokens = max_tokens
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
        
        prefixed_texts = []
        for text in texts:
            prefixed_text = f"{prefix}: {text}"
            
            # Truncate if too long
            tokens = self._tokenizer.encode(prefixed_text)
            if len(tokens) > self.max_tokens:
                truncated_tokens = tokens[:self.max_tokens]
                prefixed_text = self._tokenizer.decode(truncated_tokens)
                logger.debug(f"Text truncated from {len(tokens)} to {self.max_tokens} tokens")
            
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

class ModularTreeBuilder:
    """
    Modular LlamaIndex Tree Builder with configuration management
    Uses LlamaIndex's built-in OpenAI LLM for full compatibility with extractors
    """
    
    def __init__(self, config):
        self.config = config
        
        # Initialize LlamaIndex OpenAI LLM directly 
        # (Required for compatibility with SummaryExtractor and QuestionsAnsweredExtractor)
        self.llm = OpenAI(
            api_key=config.OPENAI_API_KEY,
            model=config.OPENAI_MODEL,
            temperature=0.1,
            max_tokens=1024
        )
        
        # Initialize VLLM embedding
        self.embed_model = VLLMEmbedding(
            base_url=config.VLLM_BASE_URL,
            model_name=config.VLLM_MODEL_NAME,
            max_tokens=config.MAX_TOKENS
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
        """Load PDF and TXT documents from configured folder"""
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
            # Create sub-chunks pointing to base node
            for parser in self.sub_parsers:
                sub_nodes = parser.get_nodes_from_documents([base_node])
                sub_index_nodes = [
                    IndexNode.from_text_node(sub_node, base_node.node_id)
                    for sub_node in sub_nodes
                ]
                all_nodes.extend(sub_index_nodes)
            
            # Add original base node as IndexNode
            original_node = IndexNode.from_text_node(base_node, base_node.node_id)
            all_nodes.append(original_node)
        
        logger.info(f"✅ Created {len(all_nodes)} nodes with chunk references")
        return all_nodes
    
    async def create_metadata_references(self, base_nodes: List[BaseNode]) -> List[IndexNode]:
        """Create metadata references (summaries + questions) - Async version"""
        logger.info("📋 Creating metadata references...")
        
        if not self.extractors:
            logger.info("No extractors configured, skipping metadata extraction")
            return []
        
        # Extract metadata using async extractors
        node_to_metadata = {}
        for extractor in self.extractors:
            logger.info(f"Running extractor: {extractor.__class__.__name__}")
            
            # Process in batches for better performance
            batch_size = self.config.BATCH_SIZE
            for i in tqdm(range(0, len(base_nodes), batch_size), desc=f"Extracting {extractor.__class__.__name__}"):
                batch_nodes = base_nodes[i:i+batch_size]
                
                try:
                    metadata_dicts = extractor.extract(batch_nodes)
                    
                    for node, metadata in zip(batch_nodes, metadata_dicts):
                        if node.node_id not in node_to_metadata:
                            node_to_metadata[node.node_id] = metadata
                        else:
                            node_to_metadata[node.node_id].update(metadata)
                            
                except Exception as e:
                    logger.error(f"Error in batch {i//batch_size + 1}: {e}")
                    # Continue with next batch
                    continue
        
        # Save metadata cache
        metadata_path = self.config.OUTPUT_PATH / "metadata_cache.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(node_to_metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Saved metadata cache to: {metadata_path}")
        
        # Create IndexNodes from metadata
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
        """Build VectorStoreIndex and store in Qdrant with batch processing"""
        logger.info(f"🏗️ Building VectorStoreIndex with {len(all_nodes)} nodes...")
        
        # Create QdrantVectorStore
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.config.QDRANT_COLLECTION_NAME,
        )
        
        # Custom embedding function for batch processing
        class EmbeddingFunctionWrapper:
            def __init__(self, embed_model, batch_size):
                self.embed_model = embed_model
                self.batch_size = batch_size
            
            def get_text_embedding(self, text: str) -> List[float]:
                return self.embed_model.get_text_embedding(text)
            
            def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                return self.embed_model.get_text_embeddings(texts)
            
            def get_query_embedding(self, query: str) -> List[float]:
                return self.embed_model.get_query_embedding(query)
        
        embedding_wrapper = EmbeddingFunctionWrapper(
            self.embed_model, 
            self.config.EMBEDDING_BATCH_SIZE
        )
        
        # Extract text from nodes
        logger.info("📝 Extracting text from nodes...")
        node_texts = [node.get_content() for node in all_nodes]
        
        # Generate embeddings in batches
        logger.info("🔢 Generating embeddings...")
        start_time = time.perf_counter()
        
        all_embeddings = []
        batch_size = self.config.EMBEDDING_BATCH_SIZE
        
        for i in tqdm(range(0, len(node_texts), batch_size), desc="Embedding batches"):
            batch_texts = node_texts[i:i+batch_size]
            try:
                batch_embeddings = embedding_wrapper.get_text_embeddings(batch_texts)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error in embedding batch {i//batch_size + 1}: {e}")
                # Create zero embeddings as fallback
                batch_embeddings = [[0.0] * 1024 for _ in batch_texts]
                all_embeddings.extend(batch_embeddings)
        
        embedding_time = time.perf_counter() - start_time
        logger.info(f"✅ Generated {len(all_embeddings)} embeddings in {embedding_time:.1f}s")
        
        # Store in Qdrant
        logger.info("🏗️ Storing in Qdrant...")
        
        from qdrant_client.models import Distance, VectorParams, PointStruct
        
        # Ensure collection exists
        try:
            collection_info = self.qdrant_client.get_collection(self.config.QDRANT_COLLECTION_NAME)
            logger.info(f"✅ Using existing collection: {self.config.QDRANT_COLLECTION_NAME}")
        except:
            logger.info(f"📦 Creating new collection: {self.config.QDRANT_COLLECTION_NAME}")
            self.qdrant_client.create_collection(
                collection_name=self.config.QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )
        
        # Prepare and insert points in batches
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
            try:
                self.qdrant_client.upsert(
                    collection_name=self.config.QDRANT_COLLECTION_NAME,
                    points=batch_points
                )
            except Exception as e:
                logger.error(f"Error inserting batch {i//batch_size + 1}: {e}")
                continue
        
        logger.info("✅ VectorStoreIndex built and stored in Qdrant")
        return len(all_nodes)
    
    async def build_tree(self):
        """Main tree building process - Async version with config"""
        logger.info("🚀 Starting Modular Tree Building Process")
        
        try:
            # 1. Load documents
            documents = self.load_documents()
            if not documents:
                raise ValueError("No documents found!")
            
            # 2. Create base nodes
            base_nodes = self.create_base_nodes(documents)
            
            # 3. Create chunk references (multi-level)
            chunk_nodes = self.create_chunk_references(base_nodes)
            
            # 4. Create metadata references (summaries + questions) - Async
            metadata_nodes = await self.create_metadata_references(base_nodes)
            
            # 5. Combine all nodes
            all_nodes = chunk_nodes + metadata_nodes
            logger.info(f"📊 Total nodes: {len(all_nodes)} ({len(chunk_nodes)} chunks + {len(metadata_nodes)} metadata)")
            
            # 6. Build VectorStoreIndex and store in Qdrant - Async
            total_stored = await self.build_and_store_index(all_nodes)
            
            # 7. Save node mapping for RecursiveRetriever
            all_nodes_dict = {n.node_id: n for n in all_nodes}
            
            node_mapping_path = self.config.OUTPUT_PATH / "node_mapping.json"
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
            
            # 8. Save comprehensive statistics
            stats = {
                "build_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": {
                    "base_chunk_size": self.config.BASE_CHUNK_SIZE,
                    "sub_chunk_sizes": self.config.SUB_CHUNK_SIZES,
                    "chunk_overlap": self.config.CHUNK_OVERLAP,
                    "num_questions": self.config.NUM_QUESTIONS,
                    "enable_summaries": self.config.ENABLE_SUMMARIES,
                    "openai_model": self.config.OPENAI_MODEL,
                    "vllm_model": self.config.VLLM_MODEL_NAME
                },
                "data": {
                    "total_documents": len(documents),
                    "total_base_nodes": len(base_nodes),
                    "total_chunk_nodes": len(chunk_nodes),
                    "total_metadata_nodes": len(metadata_nodes),
                    "total_nodes": len(all_nodes),
                    "total_stored": total_stored,
                    "embedding_dimension": 1024
                },
                "storage": {
                    "qdrant_collection": self.config.QDRANT_COLLECTION_NAME,
                    "qdrant_url": self.config.QDRANT_URL
                }
            }
            
            stats_path = self.config.OUTPUT_PATH / "build_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info("✅ Tree building completed successfully!")
            logger.info(f"📊 Final Stats: {stats['data']}")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Tree building failed: {e}")
            raise
        finally:
            # Cleanup
            await self.embed_model.close()

async def main():
    """
    Main entry point with configuration management
    Uses LlamaIndex's built-in OpenAI LLM for full compatibility
    """
    try:
        print("🚀 Modular Tree Builder Starting...")
        print("=" * 60)
        
        # Load and validate configuration
        config = load_and_validate_config()
        
        # Create builder and build tree
        builder = ModularTreeBuilder(config)
        stats = await builder.build_tree()
        
        print("\n🎉 Success! Tree built and stored in Qdrant.")
        print(f"📊 Built {stats['data']['total_nodes']} nodes from {stats['data']['total_documents']} documents")
        print(f"💾 Data stored in collection: {stats['storage']['qdrant_collection']}")
        print("📋 Next: Test the retriever service")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️  Build process interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Build process failed: {e}")
        print(f"\n❌ Build failed: {e}")
        print("📋 Please check the logs and configuration")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)