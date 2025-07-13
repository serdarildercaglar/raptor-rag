#!/usr/bin/env python3
"""
RAPTOR Builder - Simple like original demo.py
Just uses VLLM embedding service with intfloat prefixes
"""
import os
import requests
from dotenv import load_dotenv

from raptor import (
    RetrievalAugmentation, 
    RetrievalAugmentationConfig,
    ClusterTreeConfig,
    GPT4OMiniSummarizationModel,
    BaseEmbeddingModel
)

load_dotenv()

class VLLMEmbedding(BaseEmbeddingModel):
    """Simple VLLM embedding with passage prefix"""
    
    def __init__(self, base_url: str = "http://localhost:8008"):
        self.base_url = base_url
        
    def create_embedding(self, text: str):
        # Always use passage prefix for building
        payload = {
            "input": [f"passage: {text}"],
            "model": "intfloat/multilingual-e5-large"
        }
        
        response = requests.post(f"{self.base_url}/v1/embeddings", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()['data'][0]['embedding']
        else:
            raise Exception(f"VLLM error: {response.status_code}")

def main():
    """Build RAPTOR tree - simple like demo.py"""
    
    # Check requirements
    if not os.path.exists("data.txt"):
        print("❌ data.txt not found!")
        return
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set!")
        return
    
    print("🚀 Building RAPTOR tree...")
    
    # Read data
    with open("data.txt", 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Initialize models
    embedding_model = VLLMEmbedding()
    summarization_model = GPT4OMiniSummarizationModel()
    
    # Test VLLM connection
    try:
        test_emb = embedding_model.create_embedding("test")
        print(f"✅ VLLM OK (dim: {len(test_emb)})")
    except Exception as e:
        print(f"❌ VLLM failed: {e}")
        return
    
    # Config with ClusterTreeConfig (like your example)
    tree_builder_config = ClusterTreeConfig(
        max_tokens=200,                    # tb_max_tokens
        num_layers=5,                      # tb_num_layers  
        summarization_length=100,          # tb_summarization_length
        summarization_model=summarization_model,
        embedding_models={"VLLM": embedding_model},
        cluster_embedding_model="VLLM",
        reduction_dimension=10,            # UMAP dimension reduction
        clustering_params={
            'threshold': 0.15,             # clustering_threshold
            'max_length_in_cluster': 3500, # max tokens per cluster
            'verbose': False
        }
    )
    
    # Main config with tree_builder_config
    config = RetrievalAugmentationConfig(
        tree_builder_config=tree_builder_config,
        tree_builder_type="cluster",
        # TreeRetriever parameters
        tr_top_k=5,
        tr_selection_mode="top_k", 
        tr_context_embedding_model="VLLM",
        tr_embedding_model=embedding_model
    )
    
    # Build tree
    RA = RetrievalAugmentation(config=config)
    RA.add_documents(text)
    
    # Save
    save_path = "raptor_tree.pkl"
    RA.save(save_path)
    
    print(f"✅ Success! Saved to {save_path}")
    if RA.tree:
        print(f"📊 Tree: {len(RA.tree.all_nodes)} nodes, {RA.tree.num_layers} layers")

if __name__ == "__main__":
    main()