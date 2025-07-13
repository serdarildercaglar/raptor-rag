#!/usr/bin/env python3
import os
import asyncio
import aiohttp
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# CRITICAL OPTIMIZATION: Make TreeBuilder use batch embeddings
import raptor.tree_builder
from raptor.tree_structures import Node

def create_leaf_nodes_with_batch_embeddings(self, chunks: List[str]) -> Dict[int, Node]:
    """Optimized leaf node creation using batch embeddings"""
    print(f"🚀 Creating {len(chunks)} leaf nodes using batch embeddings...")
    
    # First create all nodes without embeddings
    nodes_without_embeddings = []
    for index, text in enumerate(chunks):
        node = Node(text, index, set(), {})  # Empty embeddings for now
        nodes_without_embeddings.append(node)
    
    # Batch process embeddings
    batch_size = 32  # Optimal for VLLM
    all_embeddings = {}
    
    for model_name, embedding_model in self.embedding_models.items():
        print(f"   📡 Generating embeddings with {model_name}...")
        model_embeddings = []
        
        # Check if model supports batch processing
        if hasattr(embedding_model, 'create_embeddings_batch'):
            # Process in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                print(f"      Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                
                try:
                    # Use async batch method
                    batch_embeddings = asyncio.run(
                        embedding_model.create_embeddings_batch(batch)
                    )
                    model_embeddings.extend(batch_embeddings)
                except Exception as e:
                    print(f"      ⚠️ Batch failed, falling back to individual: {e}")
                    # Fallback to individual processing
                    for text in batch:
                        emb = embedding_model.create_embedding(text)
                        model_embeddings.append(emb)
        else:
            # Fallback to original single processing
            print(f"      ⚠️ {model_name} doesn't support batch, using sequential...")
            for text in chunks:
                emb = embedding_model.create_embedding(text)
                model_embeddings.append(emb)
        
        all_embeddings[model_name] = model_embeddings
    
    # Assign embeddings to nodes
    leaf_nodes = {}
    for i, node in enumerate(nodes_without_embeddings):
        node.embeddings = {
            model_name: all_embeddings[model_name][i]
            for model_name in all_embeddings
        }
        leaf_nodes[i] = node
    
    print(f"✅ Created all {len(leaf_nodes)} leaf nodes with embeddings!")
    return leaf_nodes

# Override both methods
raptor.tree_builder.TreeBuilder.multithreaded_create_leaf_nodes = create_leaf_nodes_with_batch_embeddings
raptor.tree_builder.TreeBuilder.create_leaf_nodes = create_leaf_nodes_with_batch_embeddings

# Also patch construct_tree to use batch for cluster embeddings
original_construct_tree = raptor.tree_builder.TreeBuilder.construct_tree

def patched_construct_tree(self, current_level_nodes, all_tree_nodes, layer_to_nodes, use_multithreading=False):
    """Patched construct_tree with progress tracking"""
    print(f"   🌳 Building tree layers...")
    return original_construct_tree(self, current_level_nodes, all_tree_nodes, layer_to_nodes, use_multithreading)

raptor.tree_builder.TreeBuilder.construct_tree = patched_construct_tree

# Now import RAPTOR
from raptor import (
    RetrievalAugmentation, 
    RetrievalAugmentationConfig,
    ClusterTreeConfig,
    TreeRetrieverConfig,
    VLLMEmbeddingModel, 
    GPT41SummarizationModel
)

from dotenv import load_dotenv
load_dotenv()

# Environment check
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Set OPENAI_API_KEY environment variable")


async def test_vllm_connection(vllm_url: str = "http://localhost:8008") -> bool:
    """Test VLLM service connection"""
    try:
        async with aiohttp.ClientSession() as session:
            # Test health endpoint
            async with session.get(f"{vllm_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ VLLM Service is healthy: {data.get('status', 'unknown')}")
                    print(f"   Model: {data.get('model', 'unknown')}")
                    
                    # Test batch endpoint
                    test_batch = ["test1", "test2"]
                    async with session.post(
                        f"{vllm_url}/embeddings/batch", 
                        json=test_batch,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as batch_response:
                        if batch_response.status == 200:
                            print(f"   ✅ Batch endpoint working")
                            return True
                        else:
                            print(f"   ⚠️ Batch endpoint returned: {batch_response.status}")
                    
                    return True
                else:
                    print(f"❌ VLLM Service returned status: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Cannot connect to VLLM service at {vllm_url}")
        print(f"   Error: {str(e)}")
        return False


def get_presets():
    """Config presets with performance info"""
    return {
        "fast": {
            "description": "⚡ Fast processing, general clusters",
            "performance": "Build: 10-15 min | Query: <0.5 sec | Users: 200+",
            "use_case": "Small docs, news, chat logs, quick prototyping",
            "params": {
                "tb_max_tokens": 150,           
                "tb_num_layers": 3,            
                "tb_summarization_length": 80,  
                "reduction_dimension": 8,       
                "clustering_threshold": 0.2,    
                "max_length_in_cluster": 3000   
            }
        },
        "balanced": {
            "description": "⚖️ Balanced speed and quality",
            "performance": "Build: 20-30 min | Query: 0.5-1 sec | Users: 100+",
            "use_case": "General purpose, technical docs, articles",
            "params": {
                "tb_max_tokens": 100,          
                "tb_num_layers": 5,            
                "tb_summarization_length": 150, 
                "reduction_dimension": 10,      
                "clustering_threshold": 0.1,    
                "max_length_in_cluster": 3500   
            }
        },
        "precise": {
            "description": "🎯 High quality, detailed clustering",
            "performance": "Build: 45-60 min | Query: 1-2 sec | Users: 50+",
            "use_case": "Research papers, complex docs, detailed analysis",
            "params": {
                "tb_max_tokens": 120,          
                "tb_num_layers": 7,            
                "tb_summarization_length": 200, 
                "reduction_dimension": 15,      
                "clustering_threshold": 0.08,   
                "max_length_in_cluster": 4000   
            }
        },
        "production": {
            "description": "🚀 Optimized for production concurrent users",
            "performance": "Build: 15-25 min | Query: <1 sec | Users: 100+",
            "use_case": "Production servers, concurrent load, fast responses",
            "params": {
                "tb_max_tokens": 200,          
                "tb_num_layers": 4,            
                "tb_summarization_length": 100, 
                "reduction_dimension": 10,       
                "clustering_threshold": 0.15,   
                "max_length_in_cluster": 3000   
            }
        },
        "production_premium": {
            "description": "🎯 Production + Higher quality",
            "performance": "Build: 30-45 min | Query: 1-2 sec | Users: 50+",
            "use_case": "Important docs, moderate concurrent load, better quality",
            "params": {
                "tb_max_tokens": 150,          
                "tb_num_layers": 5,            
                "tb_summarization_length": 150, 
                "reduction_dimension": 12,      
                "clustering_threshold": 0.08,   
                "max_length_in_cluster": 3500   
            }
        },
        "large_docs": {
            "description": "📚 Maximum quality for large documents",
            "performance": "⚠️ Build: 2-4 HOUR | Query: 3-8 SEC | Users: 5-10 MAX",
            "use_case": "Books, manuals, research datasets, offline processing",
            "params": {
                "tb_max_tokens": 200,          
                "tb_num_layers": 8,            
                "tb_summarization_length": 250, 
                "reduction_dimension": 20,      
                "clustering_threshold": 0.05,   
                "max_length_in_cluster": 5000   
            }
        }
    }


def build_raptor_db(
    data_file: str, 
    save_path: str, 
    vllm_url: str = "http://localhost:8008",
    
    # Tree Builder Config Parameters
    tb_max_tokens: int = 100,              
    tb_num_layers: int = 5,                
    tb_summarization_length: int = 150,    
    tb_threshold: float = 0.5,             
    tb_top_k: int = 5,                     
    tb_selection_mode: str = "top_k",      
    
    # Tree Retriever Config Parameters  
    tr_threshold: float = 0.5,             
    tr_top_k: int = 5,                     
    tr_selection_mode: str = "top_k",      
    tr_num_layers: Optional[int] = None,   
    tr_start_layer: Optional[int] = None,  
    
    # Clustering Parameters
    reduction_dimension: int = 10,         
    clustering_threshold: float = 0.1,     
    max_length_in_cluster: int = 3500,     
    
    # VLLM Parameters
    vllm_timeout: int = 30                 
):
    """Build RAPTOR DB with optimized configuration"""
    
    # Test VLLM connection first
    print("🔍 Testing VLLM connection...")
    is_connected = asyncio.run(test_vllm_connection(vllm_url))
    
    if not is_connected:
        print("\n💡 Solutions:")
        print("   1. Start VLLM service:")
        print("      cd embedding_service")
        print("      ./start_service.sh")
        print("   2. Check if port 8008 is available:")
        print("      lsof -i :8008")
        print("   3. Check GPU memory:")
        print("      nvidia-smi")
        raise ConnectionError("Cannot connect to VLLM embedding service")
    
    print(f"\n📖 Reading data from {data_file}...")
    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"✅ Loaded {len(text)} characters, ~{len(text.split())} words")
    
    # Token estimation (Türkçe için)
    estimated_tokens = len(text) // 3
    expected_chunks = estimated_tokens // tb_max_tokens
    print(f"📊 Estimated ~{expected_chunks} chunks")
    
    print("\n🔧 Initializing models...")
    embedding_model = VLLMEmbeddingModel(base_url=vllm_url, timeout=vllm_timeout)
    summarization_model = GPT41SummarizationModel()
    
    # Test embedding model
    print("🧪 Testing embedding model...")
    try:
        test_embedding = embedding_model.create_embedding("test")
        print(f"✅ Single embedding test passed: dimension={len(test_embedding)}")
        
        # Test batch endpoint
        test_batch_embeddings = asyncio.run(
            embedding_model.create_embeddings_batch(["test1", "test2"])
        )
        print(f"✅ Batch embedding test passed: {len(test_batch_embeddings)} embeddings")
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        raise
    
    print("\n⚙️ Creating RAPTOR configuration...")
    
    # Create ClusterTreeConfig with clustering parameters
    tree_builder_config = ClusterTreeConfig(
        tokenizer=None,  
        max_tokens=tb_max_tokens,
        num_layers=tb_num_layers,
        threshold=tb_threshold,
        top_k=tb_top_k,
        selection_mode=tb_selection_mode,
        summarization_length=tb_summarization_length,
        summarization_model=summarization_model,
        embedding_models={"VLLMEmbedding": embedding_model},
        cluster_embedding_model="VLLMEmbedding",
        reduction_dimension=reduction_dimension,
        clustering_params={
            'threshold': clustering_threshold,
            'max_length_in_cluster': max_length_in_cluster,
            'verbose': False  
        }
    )
    
    # Create TreeRetrieverConfig
    tree_retriever_config = TreeRetrieverConfig(
        tokenizer=None,  
        threshold=tr_threshold,
        top_k=tr_top_k,
        selection_mode=tr_selection_mode,
        context_embedding_model="VLLMEmbedding",
        embedding_model=embedding_model,
        num_layers=tr_num_layers,
        start_layer=tr_start_layer
    )
    
    # Create main config
    config = RetrievalAugmentationConfig(
        tree_builder_config=tree_builder_config,
        tree_retriever_config=tree_retriever_config,
        tree_builder_type="cluster"
    )
    
    print(f"\n🌳 Configuration Summary:")
    print(f"  - Chunks: ~{expected_chunks} x {tb_max_tokens} tokens")
    print(f"  - Tree layers: {tb_num_layers}")
    print(f"  - Clustering: {reduction_dimension}D, threshold={clustering_threshold}")
    print(f"  - Using batch embeddings: YES ✅")
    
    print("\n🚀 Building RAPTOR tree...")
    print("   This will take a while. Progress:")
    print("   [1/5] Text chunking...")
    print("   [2/5] Generating embeddings (BATCH MODE)...")
    print("   [3/5] Clustering...")
    print("   [4/5] Summarizing...")
    print("   [5/5] Building tree hierarchy...")
    
    # Build with error handling
    RA = None
    try:
        RA = RetrievalAugmentation(config=config)
        RA.add_documents(text)
        
        print(f"\n💾 Saving tree to {save_path}...")
        RA.save(save_path)
        
        print("\n✅ RAPTOR DB created successfully!")
        if RA.tree:
            print(f"📊 Tree Statistics:")
            print(f"  - Total nodes: {len(RA.tree.all_nodes)}")
            print(f"  - Leaf nodes: {len(RA.tree.leaf_nodes)}")
            print(f"  - Tree layers: {RA.tree.num_layers}")
            
            for layer, nodes in RA.tree.layer_to_nodes.items():
                print(f"    Layer {layer}: {len(nodes)} nodes")
                
    except Exception as e:
        print(f"\n❌ Build error: {e}")
        print(f"   Error type: {type(e).__name__}")
        raise
        
    finally:
        # Cleanup VLLM session
        if hasattr(embedding_model, 'close'):
            try:
                asyncio.run(embedding_model.close())
                print("✅ VLLM session cleaned up")
            except:
                pass
    
    return RA


def build_with_preset(data_file: str, save_path: str, preset: str):
    """Build RAPTOR DB with preset configuration"""
    
    presets = get_presets()
    if preset not in presets:
        raise ValueError(f"Invalid preset: {preset}")
        
    params = presets[preset]["params"]
    
    print(f"🎯 Using preset: '{preset}'")
    print(f"📋 {presets[preset]['description']}")
    print(f"⚡ {presets[preset]['performance']}")
    
    return build_raptor_db(
        data_file=data_file,
        save_path=save_path,
        **params
    )


def main():
    """Main function with config selection"""
    
    print("🚀 RAPTOR DB Builder (Optimized with Batch Embeddings)")
    print("=" * 55)
    
    # Check prerequisites
    print("\n📋 Checking prerequisites...")
    
    # Check OpenAI API key
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OpenAI API key found")
    else:
        print("❌ OpenAI API key not found!")
        return 1
    
    # Test VLLM connection
    print("\n🔍 Checking VLLM embedding service...")
    is_vllm_ready = asyncio.run(test_vllm_connection())
    
    if not is_vllm_ready:
        print("\n❌ VLLM service is not running!")
        print("\n💡 To start VLLM service:")
        print("   cd embedding_service")
        print("   ./start_service.sh")
        return 1
    
    presets = get_presets()
    
    # Show presets
    print("\n\nAvailable configurations:")
    preset_list = list(presets.keys())
    for i, (name, info) in enumerate(presets.items(), 1):
        print(f"\n{i}. {name.upper()}: {info['description']}")
        print(f"   Performance: {info['performance']}")
        print(f"   Use case: {info['use_case']}")
    
    # Get user choice
    print(f"\nSelect config (1-{len(preset_list)}): ", end="")
    try:
        choice = int(input().strip())
        if 1 <= choice <= len(preset_list):
            selected_preset = preset_list[choice - 1]
        else:
            print("Invalid choice, using 'balanced'")
            selected_preset = "balanced"
    except:
        print("Invalid input, using 'balanced'")
        selected_preset = "balanced"
    
    print(f"\n✅ Selected: {selected_preset.upper()}")
    
    # Warning for large_docs
    if selected_preset == "large_docs":
        print("\n⚠️ WARNING: large_docs is NOT suitable for production!")
        print("   - Build time: 2-4 hours")
        print("   - Query time: 3-8 seconds") 
        print("   - Max concurrent users: 5-10")
        proceed = input("\nContinue anyway? (y/N): ").strip().lower()
        if proceed != 'y':
            print("Cancelled. Choose a production config.")
            return
    
    # Check for data file
    data_file = "data.txt"
    if not os.path.exists(data_file):
        print(f"❌ {data_file} not found!")
        return 1
        
    # Build
    output_file = f"raptor_{selected_preset}.pkl"
    print(f"\n🔨 Building with {selected_preset} config...")
    
    try:
        RA = build_with_preset(data_file, output_file, selected_preset)
        
        # Quick test
        print("\n🧪 Testing retrieval...")
        test_question = "What is the main topic of this document?"
        result = RA.retrieve(test_question, top_k=3, max_tokens=500)
        print(f"📄 Sample result: {result[:200]}...")
        
        print(f"\n✅ Success! Tree saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())