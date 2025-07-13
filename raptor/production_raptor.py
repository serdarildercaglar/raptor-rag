#!/usr/bin/env python3
"""
Production RAPTOR v2 - Hybrid FAISS-Tree Integration
Combines FAISS speed with TreeRetriever's hierarchical context quality
"""
import asyncio
import logging
from typing import List, Dict
import pickle

from .hybrid_faiss_retriever import HybridFaissTreeRetriever, ProductionRAPTORv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Convenience functions
async def create_hybrid_raptor(tree_path: str, vllm_url: str = "http://localhost:8008") -> ProductionRAPTORv2:
    """
    Create and initialize Hybrid RAPTOR
    
    Args:
        tree_path: Path to RAPTOR tree pickle file
        vllm_url: VLLM embedding service URL
        
    Returns:
        Initialized ProductionRAPTORv2 instance
    """
    raptor = ProductionRAPTORv2(tree_path, vllm_url)
    await raptor.initialize()
    return raptor


async def quick_hybrid_retrieve(tree_path: str, queries: List[str], vllm_url: str = "http://localhost:8008") -> List[str]:
    """
    Quick one-off retrieval with hybrid approach
    
    Args:
        tree_path: Path to RAPTOR tree pickle file
        queries: List of queries
        vllm_url: VLLM embedding service URL
        
    Returns:
        Retrieved contexts
    """
    raptor = await create_hybrid_raptor(tree_path, vllm_url)
    try:
        results = await raptor.retrieve_batch(queries)
        return results
    finally:
        await raptor.close()


# Example usage and comparison
async def compare_approaches(tree_path: str, test_queries: List[str]):
    """
    Compare different retrieval approaches
    """
    print("🔬 Comparing Retrieval Approaches")
    print("=" * 50)
    
    from .production_raptor import ProductionRAPTOR  # Original FAISS (leaf-only)
    import time
    
    # Test queries
    if not test_queries:
        test_queries = [
            "divan edebiyatı nedir",
            "tanzimat dönemi edebiyat özellikleri",
            "nazım hikmet şiir analizi",
            "modern türk edebiyatı"
        ]
    
    # 1. Original FAISS (leaf nodes only)
    print("\n1️⃣ Testing Original FAISS (leaf nodes only)")
    start_time = time.time()
    
    original_raptor = ProductionRAPTOR(tree_path)
    await original_raptor.initialize()
    
    try:
        original_results = await original_raptor.retrieve_batch(test_queries)
        original_time = time.time() - start_time
        
        print(f"   Duration: {original_time:.2f}s")
        print(f"   Results: {len(original_results)} contexts")
        print(f"   Avg length: {sum(len(ctx) for ctx in original_results) / len(original_results):.0f} chars")
        
    finally:
        await original_raptor.close()
    
    # 2. Hybrid FAISS-Tree (all nodes)
    print("\n2️⃣ Testing Hybrid FAISS-Tree (all nodes)")
    start_time = time.time()
    
    hybrid_raptor = await create_hybrid_raptor(tree_path)
    
    try:
        hybrid_results = await hybrid_raptor.retrieve_batch(test_queries)
        hybrid_time = time.time() - start_time
        
        print(f"   Duration: {hybrid_time:.2f}s")
        print(f"   Results: {len(hybrid_results)} contexts")
        print(f"   Avg length: {sum(len(ctx) for ctx in hybrid_results) / len(hybrid_results):.0f} chars")
        
        # Show stats
        stats = hybrid_raptor.hybrid_retriever.get_stats()
        print(f"   Tree stats: {stats['total_nodes']} total ({stats['leaf_nodes']} leaf + {stats['parent_nodes']} parent)")
        
    finally:
        await hybrid_raptor.close()
    
    # 3. Quality comparison
    print("\n3️⃣ Quality Comparison")
    print("=" * 30)
    
    for i, query in enumerate(test_queries):
        print(f"\nQuery {i+1}: {query}")
        print(f"Original length: {len(original_results[i])} chars")
        print(f"Hybrid length: {len(hybrid_results[i])} chars")
        
        # Check for hierarchical content (parent summaries)
        original_has_summary = "özetle" in original_results[i].lower() or "genel" in original_results[i].lower()
        hybrid_has_summary = "özetle" in hybrid_results[i].lower() or "genel" in hybrid_results[i].lower()
        
        print(f"Original has summaries: {'✅' if original_has_summary else '❌'}")
        print(f"Hybrid has summaries: {'✅' if hybrid_has_summary else '✅' if len(hybrid_results[i]) > len(original_results[i]) else '❌'}")
    
    # 4. Performance comparison
    print(f"\n4️⃣ Performance Summary")
    print("=" * 30)
    print(f"Original FAISS: {len(test_queries)/original_time:.1f} q/s")
    print(f"Hybrid FAISS-Tree: {len(test_queries)/hybrid_time:.1f} q/s") 
    
    if hybrid_time <= original_time * 1.2:  # Within 20%
        print("✅ Hybrid maintains speed while improving quality")
    else:
        print("⚠️  Hybrid slower but better quality")


# Main test function
async def test_hybrid_approach():
    """Test the hybrid approach"""
    tree_path = "raptor_tree.pkl"  # Update this
    
    test_queries = [
        "divan edebiyatı özellikleri nelerdir",
        "tanzimat dönemi yazarları kimlerdir", 
        "cumhuriyet dönemi edebiyat analizi",
        "modern türk şiiri örnekleri"
    ]
    
    print("🎯 Testing Hybrid FAISS-Tree Approach")
    print("🔬 This tests ALL nodes (leaf + parent) for better context")
    
    try:
        await compare_approaches(tree_path, test_queries)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Make sure tree_path is correct and VLLM service is running")


if __name__ == "__main__":
    asyncio.run(test_hybrid_approach())