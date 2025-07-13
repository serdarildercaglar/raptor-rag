# LlamaIndex RecursiveRetriever with VLLM & Qdrant

**Production-ready RecursiveRetriever** implementation with  **VLLM embeddings** ,  **GPT-4.1 summarization** ,  **Qdrant storage** , and  **async batch processing** .

## 🏗️ Architecture

```
Documents (PDF/TXT) → Multi-level Chunking → IndexNode References → Qdrant Storage
                                                                            ↓
Query → VLLM Query Embedding → RecursiveRetriever → Follow References → Results
```

### Key Features

✅  **DOĞRU RecursiveRetriever Pattern** : Exactly like LlamaIndex documentation

✅  **Multi-level Chunking** : 128, 256, 512 → 1024 token chunks with IndexNode references

✅  **Metadata References** : GPT-4.1 summaries + generated questions → base chunks

✅  **Prefix Support** : `passage:` for indexing, `query:` for retrieval

✅  **Async Batch Processing** : High-performance concurrent retrieval

✅  **Production Ready** : FastAPI + Qdrant + proper error handling

## 📊 How RecursiveRetriever Works

### 1. Tree Building (Offline)

```python
# Multi-level chunking with IndexNode references
for base_node in base_nodes:
    for parser in sub_parsers:  # 128, 256, 512
        sub_nodes = parser.get_nodes_from_documents([base_node])
        sub_index_nodes = [
            IndexNode.from_text_node(sub_node, base_node.node_id)  # ← Reference!
            for sub_node in sub_nodes
        ]
        all_nodes.extend(sub_index_nodes)

# Metadata references
for node_id, metadata in node_to_metadata.items():
    for value in metadata.values():
        metadata_node = IndexNode(text=value, index_id=node_id)  # ← Reference!
        all_nodes.append(metadata_node)

# Build VectorStoreIndex
index = VectorStoreIndex(nodes=all_nodes, storage_context=qdrant_storage)
```

### 2. Retrieval Process (Production)

```python
# Load from Qdrant
vector_store = QdrantVectorStore(client=client, collection_name="tree")
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# Create RecursiveRetriever
all_nodes_dict = {n.node_id: n for n in all_nodes}  # From node_mapping.json
recursive_retriever = RecursiveRetriever(
    root_id="vector",
    retriever_dict={"vector": vector_retriever},
    node_dict=all_nodes_dict,  # ← Critical!
    verbose=False
)

# Retrieval follows references automatically!
results = recursive_retriever.retrieve(query)
```

## 📁 Project Structure

```
project/
├── tree_builder/           # Offline tree building
│   ├── requirements.txt
│   ├── .env.example
│   ├── config.py
│   ├── builder.py          # ← FIXED: Uses correct RecursiveRetriever pattern
│   └── documents/          # Put your PDF/TXT files here
│
├── retrieve_service/       # Production API
│   ├── requirements.txt
│   ├── .env.example
│   ├── config.py
│   ├── server.py           # ← FIXED: VectorStoreIndex.from_vector_store
│   └── test_client.py      # ← UPDATED: Tests RecursiveRetriever
│
├── tree_data/              # Generated files
│   ├── node_mapping.json  # ← CRITICAL: Node dict for RecursiveRetriever
│   ├── metadata_cache.json
│   └── build_stats.json
│
└── README.md
```

## 🚀 Quick Start

### 1. Setup Services

```bash
# Start VLLM Embedding Service
# http://localhost:8008

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

### 2. Build Tree (Offline)

```bash
cd tree_builder
pip install -r requirements.txt
cp .env.example .env

# Edit .env file:
# OPENAI_API_KEY=your_key_here
# VLLM_BASE_URL=http://localhost:8008
# QDRANT_URL=http://localhost:6333

# Put documents in documents/ folder
mkdir documents
cp your_files.pdf documents/
cp your_files.txt documents/

# Build tree with RecursiveRetriever pattern
python builder.py
```

**Output:**

* Qdrant collection with embeddings
* `tree_data/node_mapping.json` (critical for RecursiveRetriever)
* `tree_data/build_stats.json`

### 3. Start Retrieve Service

```bash
cd retrieve_service
pip install -r requirements.txt
cp .env.example .env

# Edit .env file:
# VLLM_BASE_URL=http://localhost:8008
# QDRANT_URL=http://localhost:6333
# QDRANT_COLLECTION_NAME=llamaindex_tree

# Start RecursiveRetriever service
python server.py
```

### 4. Test RecursiveRetriever

```bash
# Test the RecursiveRetriever
python test_client.py

# Or use curl
curl -X POST "http://localhost:8000/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning nedir", "top_k": 5}'
```

## 📡 API Endpoints

### Single Query (RecursiveRetriever)

```bash
POST /retrieve
{
  "query": "your question here",
  "top_k": 5,
  "similarity_cutoff": 0.0
}
```

**Response includes:**

* **Reference nodes** : Smaller chunks pointing to larger ones
* **Base nodes** : Original large chunks
* **Metadata nodes** : Summaries and questions pointing to base chunks

### Batch Processing (Async)

```bash
POST /retrieve/batch
{
  "queries": ["question 1", "question 2", "..."],
  "top_k": 5,
  "similarity_cutoff": 0.0
}
```

### Health & Stats

```bash
GET /health      # Service health + RecursiveRetriever info
GET /stats       # Node statistics + reference types
```

## 🔬 What Makes This Special

### 1. True RecursiveRetriever Pattern

```json
{
  "node_id": "sub-chunk-123",
  "text": "Small chunk text...",
  "score": 0.95,
  "node_type": "reference",
  "index_id": "base-45"  // ← Points to larger chunk!
}
```

### 2. Multi-Level References

* **128-token chunks** → reference 1024-token base chunks
* **256-token chunks** → reference 1024-token base chunks
* **512-token chunks** → reference 1024-token base chunks
* **Summaries** → reference original base chunks
* **Questions** → reference original base chunks

### 3. Async Batch Processing

```python
# Process 100 queries concurrently
batch_results = await recursive_retriever.retrieve_batch(queries)
```

## ⚙️ Configuration

### Tree Builder (.env)

```env
OPENAI_API_KEY=your_openai_api_key
VLLM_BASE_URL=http://localhost:8008
QDRANT_URL=http://localhost:6333
BASE_CHUNK_SIZE=1024
SUB_CHUNK_SIZES=128,256,512    # ← Creates IndexNode references
NUM_QUESTIONS=5                 # ← Creates question references
ENABLE_SUMMARIES=true          # ← Creates summary references
```

### Retrieve Service (.env)

```env
VLLM_BASE_URL=http://localhost:8008
QDRANT_URL=http://localhost:6333
DEFAULT_TOP_K=5
MAX_BATCH_SIZE=100             # ← Async batch processing
MAX_CONCURRENT_REQUESTS=10     # ← Concurrent limit
```

## 🎯 Performance & Quality

### Accuracy Improvements

* **Better Context** : RecursiveRetriever follows references to get full context
* **Hierarchical Retrieval** : Finds small chunks but returns large chunks
* **Metadata Enhancement** : Summaries and questions improve retrieval

### Speed Optimizations

* **Async Batch Processing** : Up to 100 queries/batch
* **VLLM Native Batching** : Efficient embedding generation
* **Qdrant Vector Search** : Fast similarity search
* **Concurrent Requests** : Configurable concurrency limits

## 🔧 Troubleshooting

### Tree Builder Issues

```bash
# Check node mapping file
ls -la tree_data/node_mapping.json

# Verify Qdrant collection
curl http://localhost:6333/collections/llamaindex_tree
```

### Retrieve Service Issues

```bash
# Check RecursiveRetriever initialization
curl http://localhost:8000/health

# Verify node references
curl http://localhost:8000/stats
```

### Common Problems

1. **"Node mapping file not found"**
   * Run tree builder first: `python builder.py`
   * Check `tree_data/node_mapping.json` exists
2. **"No references found"**
   * Verify IndexNode creation in tree builder
   * Check node types in `/stats` endpoint
3. **"Collection not found"**
   * Qdrant collection doesn't exist
   * Run tree builder to create collection

## 🔄 Updates

To update the tree with new documents:

1. Add documents to `tree_builder/documents/`
2. Run `python builder.py` (overwrites collection)
3. Restart retrieve service

## 📈 Example Results

```json
{
  "query": "machine learning nedir",
  "results": [
    {
      "node_id": "chunk-128-45",
      "text": "Makine öğrenmesi...",
      "score": 0.95,
      "node_type": "reference",
      "index_id": "base-12"     // ← Points to full context
    },
    {
      "node_id": "summary-67", 
      "text": "ML özeti...",
      "score": 0.87,
      "node_type": "reference", 
      "index_id": "base-12"     // ← Same target, different reference
    }
  ]
}
```

---

**🎯 Ready for production with true RecursiveRetriever pattern!**
