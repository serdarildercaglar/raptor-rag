#!/usr/bin/env python3
"""
Root Cause Test - Query Processing Bug
"""

import requests
import json

def test_exact_query_formats():
    """Test exact query format differences"""
    
    print("🔬 EXACT QUERY FORMAT TEST")
    print("="*50)
    
    base_query = "Zulficore protokolü nedir?"
    
    # Test different exact formats
    format_tests = [
        ("Direct single", {"query": f"query: {base_query}", "top_k": 5}),
        ("Batch single", {"queries": [f"query: {base_query}"], "top_k": 5}),
        ("Batch mixed A", {"queries": [f"query: {base_query}", "query: test"], "top_k": 5}),
        ("Batch mixed B", {"queries": ["query: test", f"query: {base_query}"], "top_k": 5}),
        ("Batch similar", {"queries": [f"query: {base_query}", "query: Zulficore sistem nedir"], "top_k": 5})
    ]
    
    for test_name, payload in format_tests:
        print(f"\n{test_name}:")
        
        if "queries" in payload:
            # Batch endpoint
            response = requests.post("http://localhost:8000/retrieve/batch", json=payload)
            if response.status_code == 200:
                result = response.json()
                
                for i, query_results in enumerate(result['results']):
                    original_query = payload["queries"][i].replace("query: ", "")[:30]
                    status = "✅" if len(query_results) > 0 else "❌"
                    score = query_results[0]['score'] if query_results else 0.0
                    print(f"   {status} Query {i+1} ({original_query}): {len(query_results)} results (score: {score:.3f})")
            else:
                print(f"   ❌ Error: {response.status_code}")
        else:
            # Single endpoint  
            response = requests.post("http://localhost:8000/retrieve", json=payload)
            if response.status_code == 200:
                result = response.json()
                score = result['results'][0]['score'] if result['results'] else 0.0
                print(f"   ✅ Results: {result['total_results']} (score: {score:.3f})")
            else:
                print(f"   ❌ Error: {response.status_code}")

def test_query_order_effects():
    """Test if query order affects results"""
    
    print("\n🔄 QUERY ORDER EFFECTS TEST")
    print("="*50)
    
    working_query = "query: Zulficore protokolü nedir?"
    other_query = "query: sistem nedir"
    
    order_tests = [
        ("Working first", [working_query, other_query]),
        ("Working second", [other_query, working_query]),
        ("Working alone", [working_query]),
        ("Working + similar", [working_query, "query: Zulficore sistem"])
    ]
    
    for test_name, queries in order_tests:
        print(f"\n{test_name}:")
        
        payload = {"queries": queries, "top_k": 5, "similarity_cutoff": 0.0}
        response = requests.post("http://localhost:8000/retrieve/batch", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            for i, (query, query_results) in enumerate(zip(queries, result['results'])):
                is_working_query = "Zulficore protokolü" in query
                expected = "📍" if is_working_query else "📝"
                status = "✅" if len(query_results) > 0 else "❌"
                score = query_results[0]['score'] if query_results else 0.0
                
                print(f"   {expected} {status} Query {i+1}: {len(query_results)} results (score: {score:.3f})")
                if is_working_query and len(query_results) == 0:
                    print(f"       🚨 WORKING QUERY FAILED!")

def test_async_batch_implementation():
    """Test if async batch has issues"""
    
    print("\n⚡ ASYNC BATCH IMPLEMENTATION TEST")
    print("="*50)
    
    # Create identical queries to test async handling
    working_query = "query: Zulficore protokolü nedir?"
    
    async_tests = [
        ("2 identical", [working_query, working_query]),
        ("3 identical", [working_query, working_query, working_query]),
        ("2 different", [working_query, "query: test different"]),
        ("3 mixed", [working_query, "query: test", working_query])
    ]
    
    for test_name, queries in async_tests:
        print(f"\n{test_name}:")
        
        payload = {"queries": queries, "top_k": 5, "similarity_cutoff": 0.0}
        response = requests.post("http://localhost:8000/retrieve/batch", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            # Check if all identical queries return same results
            working_results = []
            for i, query_results in enumerate(result['results']):
                status = "✅" if len(query_results) > 0 else "❌"
                score = query_results[0]['score'] if query_results else 0.0
                print(f"   {status} Query {i+1}: {len(query_results)} results (score: {score:.3f})")
                
                if queries[i] == working_query:
                    working_results.append(len(query_results))
            
            # Check consistency
            if len(set(working_results)) > 1:
                print(f"   🚨 INCONSISTENCY: Same query returned different results: {working_results}")
            elif all(r == 0 for r in working_results):
                print(f"   🚨 ALL WORKING QUERIES FAILED!")

def test_embedding_vs_retrieval():
    """Compare embedding generation vs retrieval results"""
    
    print("\n🧮 EMBEDDING vs RETRIEVAL COMPARISON")
    print("="*50)
    
    test_query = "query: Zulficore protokolü nedir?"
    
    # 1. Get embedding directly
    embed_payload = {"input": [test_query], "model": "intfloat/multilingual-e5-large"}
    embed_response = requests.post("http://localhost:8008/v1/embeddings", json=embed_payload)
    
    if embed_response.status_code == 200:
        embed_result = embed_response.json()
        embedding = embed_result['data'][0]['embedding']
        print(f"✅ Direct embedding: {len(embedding)} dims")
        
        # 2. Manual Qdrant search with this embedding
        search_payload = {
            "vector": embedding,
            "limit": 5,
            "with_payload": True,
            "score_threshold": 0.0
        }
        
        qdrant_response = requests.post(
            "http://localhost:6333/collections/llamaindex_tree/points/search",
            json=search_payload
        )
        
        if qdrant_response.status_code == 200:
            qdrant_result = qdrant_response.json()
            manual_results = qdrant_result['result']
            print(f"✅ Manual Qdrant: {len(manual_results)} results")
            if manual_results:
                print(f"   Best score: {manual_results[0]['score']:.3f}")
        
        # 3. Service single retrieval
        single_payload = {"query": test_query, "top_k": 5, "similarity_cutoff": 0.0}
        single_response = requests.post("http://localhost:8000/retrieve", json=single_payload)
        
        if single_response.status_code == 200:
            single_result = single_response.json()
            print(f"✅ Service single: {single_result['total_results']} results")
            if single_result['results']:
                print(f"   Best score: {single_result['results'][0]['score']:.3f}")
        
        # 4. Service batch retrieval  
        batch_payload = {"queries": [test_query], "top_k": 5, "similarity_cutoff": 0.0}
        batch_response = requests.post("http://localhost:8000/retrieve/batch", json=batch_payload)
        
        if batch_response.status_code == 200:
            batch_result = batch_response.json()
            batch_results = batch_result['results'][0] if batch_result['results'] else []
            print(f"✅ Service batch: {len(batch_results)} results")
            if batch_results:
                print(f"   Best score: {batch_results[0]['score']:.3f}")
        
        # 5. Service batch with interference
        interference_payload = {"queries": [test_query, "query: test"], "top_k": 5, "similarity_cutoff": 0.0}
        interference_response = requests.post("http://localhost:8000/retrieve/batch", json=interference_payload)
        
        if interference_response.status_code == 200:
            interference_result = interference_response.json()
            first_query_results = interference_result['results'][0] if interference_result['results'] else []
            print(f"❓ Service batch (interference): {len(first_query_results)} results")
            if first_query_results:
                print(f"   Best score: {first_query_results[0]['score']:.3f}")
            else:
                print(f"   🚨 INTERFERENCE DETECTED!")

if __name__ == "__main__":
    test_exact_query_formats()
    test_query_order_effects()
    test_async_batch_implementation()
    test_embedding_vs_retrieval()
    
    print("\n" + "="*80)
    print("🔧 ROOT CAUSE ANALYSIS:")
    print("1. If batch single works but batch mixed fails → Query interference bug")
    print("2. If async identical queries fail → Async implementation bug") 
    print("3. If manual Qdrant works but service fails → Service layer bug")
    print("4. If embedding path works but retrieval fails → RecursiveRetriever bug")
    print("="*80)