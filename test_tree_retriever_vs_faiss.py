#!/usr/bin/env python3
"""
Full Text Content Comparison
TreeRetriever vs FAISS - Complete content analysis
"""
import asyncio
import time

from raptor import RetrievalAugmentation, RetrievalAugmentationConfig, VLLMEmbeddingModel
from raptor.production_raptor import ProductionRAPTOR


def print_section(title, text, max_length=None):
    """Print a section with title and formatted text"""
    print(f"\n📄 {title}")
    print("=" * 60)
    
    if max_length and len(text) > max_length:
        print(text[:max_length])
        print(f"\n... [TRUNCATED - Total length: {len(text)} chars] ...")
    else:
        print(text)
    print("=" * 60)


def analyze_content_differences(tree_text, faiss_text, query):
    """Analyze differences between tree and faiss results"""
    print(f"\n🔍 CONTENT ANALYSIS FOR: {query}")
    print("-" * 40)
    
    # Basic stats
    print(f"📊 Length comparison:")
    print(f"   TreeRetriever: {len(tree_text)} chars, {len(tree_text.split())} words")
    print(f"   FAISS:         {len(faiss_text)} chars, {len(faiss_text.split())} words")
    print(f"   Difference:    {len(tree_text) - len(faiss_text)} chars")
    
    # Content overlap analysis
    tree_words = set(tree_text.lower().split())
    faiss_words = set(faiss_text.lower().split())
    
    if tree_words and faiss_words:
        common_words = tree_words.intersection(faiss_words)
        unique_tree = tree_words - faiss_words
        unique_faiss = faiss_words - tree_words
        
        overlap_pct = len(common_words) / len(tree_words.union(faiss_words)) * 100
        
        print(f"\n📝 Content overlap:")
        print(f"   Common words: {len(common_words)} ({overlap_pct:.1f}%)")
        print(f"   Unique to TreeRetriever: {len(unique_tree)}")
        print(f"   Unique to FAISS: {len(unique_faiss)}")
        
        # Show some unique words
        if unique_tree:
            unique_tree_sample = list(unique_tree)[:10]
            print(f"   Sample TreeRetriever-only: {', '.join(unique_tree_sample)}")
        
        if unique_faiss:
            unique_faiss_sample = list(unique_faiss)[:10]
            print(f"   Sample FAISS-only: {', '.join(unique_faiss_sample)}")


async def detailed_content_comparison():
    """Compare full content of retrieval results"""
    
    # Single focused query for detailed analysis
    test_query = "divan edebiyatı nedir ve özellikleri nelerdir"
    
    print("🔍 DETAILED CONTENT COMPARISON")
    print(f"📝 Query: {test_query}")
    print("=" * 80)
    
    # Initialize systems
    print("🔧 Initializing systems...")
    embedding_model = VLLMEmbeddingModel("http://localhost:8008")
    config = RetrievalAugmentationConfig(embedding_model=embedding_model)
    tree_RA = RetrievalAugmentation(config=config, tree="/home/serdar/Documents/raptor-rag/vectordb/raptor-db")
    
    faiss_raptor = ProductionRAPTOR(
        tree_path="/home/serdar/Documents/raptor-rag/vectordb/raptor-db",
        vllm_url="http://localhost:8008"
    )
    await faiss_raptor.initialize()
    
    try:
        # Get TreeRetriever result
        print("\n⏱️  Getting TreeRetriever result...")
        tree_start = time.time()
        tree_result = await tree_RA.retrieve_async(test_query)
        tree_time = time.time() - tree_start
        
        # Get FAISS result
        print("⏱️  Getting FAISS result...")
        faiss_start = time.time()
        faiss_result = await faiss_raptor.retrieve(test_query)
        faiss_time = time.time() - faiss_start
        
        # Performance summary
        print(f"\n⚡ PERFORMANCE:")
        print(f"   TreeRetriever: {tree_time:.2f}s")
        print(f"   FAISS:         {faiss_time:.2f}s")
        print(f"   Speed ratio:   {tree_time/faiss_time:.1f}x faster (FAISS)")
        
        # Show full content
        print_section("TREERETRIEVER FULL CONTENT", tree_result)
        print_section("FAISS FULL CONTENT", faiss_result)
        
        # Detailed analysis
        analyze_content_differences(tree_result, faiss_result, test_query)
        
    finally:
        await embedding_model.close()
        await faiss_raptor.close()


async def side_by_side_comparison():
    """Side by side comparison of multiple queries"""
    
    queries = [
        "tanzimat dönemi edebiyatı",
        "nazım hikmet şiiri",
        "yaşar kemal romanları"
    ]
    
    print("\n🔍 SIDE-BY-SIDE COMPARISON")
    print("=" * 80)
    
    # Initialize systems
    embedding_model = VLLMEmbeddingModel("http://localhost:8008")
    config = RetrievalAugmentationConfig(embedding_model=embedding_model)
    tree_RA = RetrievalAugmentation(config=config, tree="/home/serdar/Documents/raptor-rag/vectordb/raptor-db")
    
    faiss_raptor = ProductionRAPTOR(
        tree_path="/home/serdar/Documents/raptor-rag/vectordb/raptor-db",
        vllm_url="http://localhost:8008"
    )
    await faiss_raptor.initialize()
    
    try:
        for i, query in enumerate(queries, 1):
            print(f"\n" + "="*80)
            print(f"📝 QUERY {i}: {query}")
            print("="*80)
            
            # Get both results
            tree_result = await tree_RA.retrieve_async(query)
            faiss_result = await faiss_raptor.retrieve(query)
            
            # Show results side by side (truncated for readability)
            print(f"\n🌳 TREERETRIEVER ({len(tree_result)} chars):")
            print("-" * 50)
            print(tree_result[:800])  # First 800 chars
            if len(tree_result) > 800:
                print("... [CONTINUES] ...")
            
            print(f"\n🚀 FAISS ({len(faiss_result)} chars):")
            print("-" * 50)
            print(faiss_result[:800])  # First 800 chars
            if len(faiss_result) > 800:
                print("... [CONTINUES] ...")
            
            # Quick analysis
            print(f"\n📊 QUICK ANALYSIS:")
            print(f"   Length ratio: {len(tree_result)/len(faiss_result):.1f}x (TreeRetriever/FAISS)")
            
            # Check if FAISS result seems complete
            if len(faiss_result) < 500:
                print("   ⚠️  FAISS result seems short")
            elif len(faiss_result) > 2000:
                print("   ✅ FAISS result has good length")
            else:
                print("   ✅ FAISS result has reasonable length")
    
    finally:
        await embedding_model.close()
        await faiss_raptor.close()


async def quality_assessment():
    """Assess content quality for specific use cases"""
    
    print("\n🎯 QUALITY ASSESSMENT FOR USE CASES")
    print("=" * 50)
    
    # Different types of queries
    quality_queries = [
        ("factual", "divan edebiyatı ne zaman başladı"),
        ("analytical", "ahmet hamdi tanpınar eserlerinde zaman teması"),
        ("comparative", "klasik ve modern türk şiiri farkları"),
        ("educational", "edebiyat dersi için şiir analizi yöntemleri")
    ]
    
    # Initialize systems
    embedding_model = VLLMEmbeddingModel("http://localhost:8008")
    config = RetrievalAugmentationConfig(embedding_model=embedding_model)
    tree_RA = RetrievalAugmentation(config=config, tree="/home/serdar/Documents/raptor-rag/vectordb/raptor-db")
    
    faiss_raptor = ProductionRAPTOR(
        tree_path="/home/serdar/Documents/raptor-rag/vectordb/raptor-db",
        vllm_url="http://localhost:8008"
    )
    await faiss_raptor.initialize()
    
    try:
        for query_type, query in quality_queries:
            print(f"\n📊 {query_type.upper()} QUERY: {query}")
            print("-" * 60)
            
            # Get results
            tree_result = await tree_RA.retrieve_async(query)
            faiss_result = await faiss_raptor.retrieve(query)
            
            print(f"🌳 TreeRetriever: {len(tree_result)} chars")
            print(f"🚀 FAISS: {len(faiss_result)} chars")
            
            # Show first paragraph of each
            tree_para = tree_result.split('\n\n')[0] if '\n\n' in tree_result else tree_result[:300]
            faiss_para = faiss_result.split('\n\n')[0] if '\n\n' in faiss_result else faiss_result[:300]
            
            print(f"\n📝 TreeRetriever first section:")
            print(f"   {tree_para}")
            
            print(f"\n📝 FAISS first section:")
            print(f"   {faiss_para}")
            
            # Quality indicators
            print(f"\n🎯 Quality indicators:")
            tree_sentences = len([s for s in tree_result.split('.') if s.strip()])
            faiss_sentences = len([s for s in faiss_result.split('.') if s.strip()])
            
            print(f"   Sentences: TreeRetriever={tree_sentences}, FAISS={faiss_sentences}")
            print(f"   Depth: {'High' if len(tree_result) > 2000 else 'Medium' if len(tree_result) > 1000 else 'Low'} (Tree) vs {'High' if len(faiss_result) > 2000 else 'Medium' if len(faiss_result) > 1000 else 'Low'} (FAISS)")
    
    finally:
        await embedding_model.close()
        await faiss_raptor.close()


async def main():
    """Run full content comparison suite"""
    
    print("🎯 FULL TEXT CONTENT COMPARISON SUITE")
    print("TreeRetriever vs FAISS RAPTOR")
    print("=" * 80)
    
    choice = input("""
Choose test type:
1. Detailed single query analysis (full text)
2. Side-by-side multiple queries  
3. Quality assessment by query type
4. All tests

Enter choice (1-4): """).strip()
    
    try:
        if choice == "1":
            await detailed_content_comparison()
        elif choice == "2":
            await side_by_side_comparison()
        elif choice == "3":
            await quality_assessment()
        elif choice == "4":
            await detailed_content_comparison()
            await side_by_side_comparison()
            await quality_assessment()
        else:
            print("Invalid choice, running detailed analysis...")
            await detailed_content_comparison()
        
        print("\n🎉 CONTENT COMPARISON COMPLETE!")
        print("\nKey observations to check:")
        print("• Content completeness")
        print("• Information accuracy") 
        print("• Contextual relevance")
        print("• Length vs quality tradeoff")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())