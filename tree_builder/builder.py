#!/usr/bin/env python3
"""
LlamaIndex Tree Builder with VLLM Embeddings and GPT-4.1 Summarization
DOĞRU RecursiveRetriever pattern implementasyonu
"""
import asyncio
import json
import logging
import copy
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import nest_asyncio

# LlamaIndex Core
from llama_index.core import Document, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode, BaseNode
from llama_index.core.extractors import SummaryExtractor, QuestionsAnsweredExtractor
from llama_index.core import VectorStoreIndex
from llama_index.readers.file import PDFReader

# LlamaIndex LLM and Embeddings
from llama_index.llms.openai import OpenAI
from llama_index.core.embeddings import BaseEmbedding

# Vector Store
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# VLLM Integration
import aiohttp
from typing import Optional

from config import TreeBuilderConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enable nested asyncio
nest_asyncio.apply()

class VLLMEmbedding(BaseEmbedding):
    """VLLM Embedding with intfloat prefix support"""
    
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
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def _embed_batch(self, texts: List[str], prefix: str = "passage") -> List[List[float]]:
        """Embed batch of texts with prefix and auto-truncation"""
        # Use context manager for safer session handling
        timeout = aiohttp.ClientTimeout(total=30)
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
                    logger.warning(f"Text truncated from {len(tokens)} to {max_tokens} tokens")
                
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
    
    # Text embedding methods (for documents/passages)
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get embedding for single text (passage prefix)"""
        embeddings = await self._embed_batch([text], prefix="passage")
        return embeddings[0]
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Sync version for text embedding"""
        return asyncio.run(self._aget_text_embedding(text))
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts (passage prefix)"""
        return await self._embed_batch(texts, prefix="passage")
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Sync version for text embeddings"""
        return asyncio.run(self._aget_text_embeddings(texts))
    
    # Query embedding methods (for search queries)
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

class LlamaIndexTreeBuilder:
    """LlamaIndex Tree Builder with DOĞRU RecursiveRetriever pattern"""
    
    def __init__(self, config: TreeBuilderConfig):
        self.config = config
        
        # Initialize LLM for summarization (GPT-4.1)
        self.llm = OpenAI(model="gpt-4o", api_key=config.OPENAI_API_KEY)
        
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
        """Build VectorStoreIndex and store in Qdrant (LlamaIndex way)"""
        logger.info(f"🏗️ Building VectorStoreIndex with {len(all_nodes)} nodes...")
        
        # Create QdrantVectorStore
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.config.QDRANT_COLLECTION_NAME,
        )
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Build index (LlamaIndex handles everything internally)
        index = VectorStoreIndex(
            nodes=all_nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=True
        )
        
        logger.info("✅ VectorStoreIndex built and stored in Qdrant")
        return index
    
    async def build_tree(self):
        """Main tree building process - DOĞRU LlamaIndex RecursiveRetriever pattern"""
        logger.info("🚀 Starting LlamaIndex Tree Building Process")
        self.config.log_config()
        
        try:
            # 1. Load documents
            documents = self.load_documents()
            if not documents:
                raise ValueError("No documents found!")
            
            # 2. Create base nodes
            base_nodes = self.create_base_nodes(documents)
            
            # 3. Create chunk references (multi-level) - EXACTLY like documentation
            chunk_nodes = self.create_chunk_references(base_nodes)
            
            # 4. Create metadata references (summaries + questions) - EXACTLY like documentation
            metadata_nodes = await self.create_metadata_references(base_nodes)
            
            # 5. Combine all nodes - EXACTLY like documentation
            all_nodes = chunk_nodes + metadata_nodes
            logger.info(f"📊 Total nodes: {len(all_nodes)} ({len(chunk_nodes)} chunks + {len(metadata_nodes)} metadata)")
            
            # 6. Build VectorStoreIndex and store in Qdrant
            index = await self.build_and_store_index(all_nodes)
            
            # 7. Save node mapping for RecursiveRetriever (CRITICAL!)
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
                "qdrant_collection": self.config.QDRANT_COLLECTION_NAME,
                "base_chunk_size": self.config.BASE_CHUNK_SIZE,
                "sub_chunk_sizes": self.config.SUB_CHUNK_SIZES,
                "embedding_dimension": len(await self.embed_model._aget_text_embedding("test"))
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
        # Validate configuration
        TreeBuilderConfig.validate()
        
        # Create builder and build tree
        builder = LlamaIndexTreeBuilder(TreeBuilderConfig)
        await builder.build_tree()
        
    except Exception as e:
        logger.error(f"❌ Build process failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)