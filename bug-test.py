#!/usr/bin/env python3
"""
Critical Bug Detection Test
Single vs Multi-Query Batch Issue
"""

import requests
import json

def test_batch_size_bug():
    """Test if batch size affects results"""
    
    working_query = "Zulficore protokolü nedir?"
    
    print("🚨 BATCH SIZE BUG DETECTION")
    print("="*60)
    
    # Test different batch sizes with SAME working query
    batch_sizes = [1, 2, 3, 5]
    
    for size in batch_sizes:
        print(f"\n📦 Batch Size: {size}")
        
        # Create batch with same query repeated
        queries = [f"query: {working_query}"] * size
        
        batch_payload = {
            "queries": queries,
            "top_k": 5,
            "similarity_cutoff": 0.0
        }
        
        response = requests.post("http://localhost:8000/retrieve/batch", json=batch_payload)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"   Total queries: {result['total_queries']}")
            print(f"   Time: {result['retrieval_time_ms']:.1f}ms")
            
            for i, query_results in enumerate(result['results']):
                print(f"   Query {i+1}: {len(query_results)} results")
                if len(query_results) > 0:
                    print(f"      Best score: {query_results[0]['score']:.3f}")
                else:
                    print(f"      ❌ ZERO RESULTS!")
        else:
            print(f"   ❌ Batch error: {response.status_code}")

def test_query_interference():
    """Test if different queries interfere with each other"""
    
    print("\n🔍 QUERY INTERFERENCE TEST")
    print("="*50)
    
    working_query = "Zulficore protokolü nedir?"
    
    # Test with working query + different second queries
    interference_tests = [
        ["valid query only", [working_query]],
        ["valid + invalid", [working_query, "nonexistent term xyz"]],
        ["valid + valid context", [working_query, "Zulficore sistemi nedir"]],
        ["valid + keyword", [working_query, "kuantum"]],
        ["valid + empty-like", [working_query, "test"]]
    ]
    
    for test_name, queries in interference_tests:
        print(f"\n{test_name}:")
        
        batch_payload = {
            "queries": [f"query: {q}" for q in queries],
            "top_k": 3,
            "similarity_cutoff": 0.0
        }
        
        response = requests.post("http://localhost:8000/retrieve/batch", json=batch_payload)
        
        if response.status_code == 200:
            result = response.json()
            
            for i, (original_query, query_results) in enumerate(zip(queries, result['results'])):
                status = "✅" if len(query_results) > 0 else "❌"
                print(f"   {status} Query {i+1} ({original_query[:20]}...): {len(query_results)} results")
        else:
            print(f"   ❌ Error: {response.status_code}")

def test_embedding_service_directly():
    """Test VLLM embedding service directly"""
    
    print("\n🔧 EMBEDDING SERVICE DIRECT TEST")
    print("="*50)
    
    # Test if embedding service works properly
    test_texts = [
        "query: Zulficore protokolü nedir?",
        "query: kuantum",
        "query: yapay zeka"
    ]
    
    embedding_payload = {
        "input": test_texts,
        "model": "intfloat/multilingual-e5-large"
    }
    
    try:
        response = requests.post("http://localhost:8008/v1/embeddings", json=embedding_payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ VLLM Embedding Service Works:")
            
            for i, item in enumerate(result['data']):
                embedding = item['embedding']
                print(f"   Text {i+1}: {len(embedding)} dimensions")
                print(f"   First 5 values: {embedding[:5]}")
                
                # Check if embedding is not all zeros
                non_zero = sum(1 for x in embedding if abs(x) > 0.001)
                print(f"   Non-zero values: {non_zero}/{len(embedding)}")
                
        else:
            print(f"❌ VLLM Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ VLLM Connection Error: {e}")

def test_qdrant_directly():
    """Test Qdrant connection directly"""
    
    print("\n🗄️  QDRANT DIRECT TEST")
    print("="*40)
    
    try:
        # Test Qdrant health
        response = requests.get("http://localhost:6333", timeout=5)
        if response.status_code == 200:
            print("✅ Qdrant connection works")
            
            # Test collection info
            collections_response = requests.get("http://localhost:6333/collections", timeout=5)
            if collections_response.status_code == 200:
                collections = collections_response.json()
                print(f"✅ Collections available: {[c['name'] for c in collections['result']['collections']]}")
                
                # Test specific collection
                collection_response = requests.get("http://localhost:6333/collections/llamaindex_tree", timeout=5)
                if collection_response.status_code == 200:
                    collection_info = collection_response.json()
                    print(f"✅ Collection 'llamaindex_tree':")
                    print(f"   Points: {collection_info['result']['points_count']}")
                    print(f"   Vectors: {collection_info['result']['vectors_count']}")
                else:
                    print(f"❌ Collection error: {collection_response.status_code}")
            else:
                print(f"❌ Collections error: {collections_response.status_code}")
        else:
            print(f"❌ Qdrant error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Qdrant connection error: {e}")

def test_manual_qdrant_search():
    """Manually test Qdrant search with known embedding"""
    
    print("\n🎯 MANUAL QDRANT SEARCH TEST")
    print("="*50)
    
    # Get embedding for working query
    embedding_payload = {
        "input": ["query: Zulficore protokolü nedir?"],
        "model": "intfloat/multilingual-e5-large"
    }
    
    try:
        embed_response = requests.post("http://localhost:8008/v1/embeddings", json=embedding_payload, timeout=10)
        
        if embed_response.status_code == 200:
            embed_result = embed_response.json()
            query_vector = embed_result['data'][0]['embedding']
            
            print(f"✅ Got embedding: {len(query_vector)} dimensions")
            
            # Search Qdrant directly
            search_payload = {
                "vector": query_vector,
                "limit": 5,
                "with_payload": True
            }
            
            qdrant_response = requests.post(
                "http://localhost:6333/collections/llamaindex_tree/points/search",
                json=search_payload,
                timeout=10
            )
            
            if qdrant_response.status_code == 200:
                search_result = qdrant_response.json()
                results = search_result['result']
                
                print(f"✅ Qdrant direct search: {len(results)} results")
                for i, result in enumerate(results[:3]):
                    print(f"   {i+1}. Score: {result['score']:.3f}")
                    print(f"      Node ID: {result['payload'].get('node_id', 'N/A')}")
                    print(f"      Text: {result['payload'].get('text', 'N/A')[:60]}...")
            else:
                print(f"❌ Qdrant search error: {qdrant_response.status_code}")
                print(qdrant_response.text)
        else:
            print(f"❌ Embedding error: {embed_response.status_code}")
            
    except Exception as e:
        print(f"❌ Manual search error: {e}")

if __name__ == "__main__":
    test_batch_size_bug()
    test_query_interference()
    test_embedding_service_directly()
    test_qdrant_directly()
    test_manual_qdrant_search()
    
    print("\n" + "="*80)
    print("🔧 DIAGNOSIS SUMMARY:")
    print("1. If batch size affects results → RecursiveRetriever batch bug")
    print("2. If embeddings work but Qdrant search fails → Retriever bug") 
    print("3. If manual Qdrant works but service doesn't → Service layer bug")
    print("4. If VLLM embeddings are zeros → Embedding service bug")
    print("="*80)