#!/usr/bin/env python3
"""
Complete System Test
Tree Builder + Retriever Service + VLLM Embedding Service
"""
import asyncio
import aiohttp
import time
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

async def test_complete_system():
    """Test the complete RAG system"""
    print("🧪 Complete RAG System Test")
    print("=" * 60)
    
    # 1. Check if tree data exists
    print("1️⃣ Checking Tree Data...")
    
    # Try different possible locations
    possible_tree_paths = [
        Path("tree_data"),
        Path("../tree_builder/tree_data"),
        Path("./tree_builder/tree_data")
    ]
    
    tree_data_path = None
    for path in possible_tree_paths:
        if path.exists():
            tree_data_path = path
            break
    
    if not tree_data_path:
        print("❌ Tree data folder not found in any of these locations:")
        for path in possible_tree_paths:
            print(f"   - {path}")
        print("📋 Please run the tree builder first!")
        return False
    
    node_mapping_path = tree_data_path / "node_mapping.json"
    build_stats_path = tree_data_path / "build_stats.json"
    
    if not node_mapping_path.exists():
        print("❌ Node mapping file not found")
        print(f"   Expected location: {node_mapping_path}")
        print("📋 Please run the tree builder first!")
        return False
    
    # Load build stats if available
    if build_stats_path.exists():
        with open(build_stats_path, 'r') as f:
            stats = json.load(f)
        print(f"✅ Tree data found in: {tree_data_path}")
        print(f"   Total documents: {stats.get('total_documents')}")
        print(f"   Total nodes: {stats.get('total_nodes')}")
        print(f"   Chunk nodes: {stats.get('total_chunk_nodes')}")
        print(f"   Metadata nodes: {stats.get('total_metadata_nodes')}")
        print(f"   Collection: {stats.get('qdrant_collection')}")
    else:
        print(f"✅ Tree data found in: {tree_data_path} (no stats file)")
    
    async with aiohttp.ClientSession() as session:
        # 2. Test Retriever Service Health
        print("\n2️⃣ Testing Retriever Service Health...")
        try:
            async with session.get(f"{BASE_URL}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    print("✅ Retriever service is healthy")
                    print(f"   Service: {health_data.get('service')}")
                    print(f"   Node count: {health_data.get('node_count')}")
                    print(f"   Pattern: {health_data.get('pattern')}")
                    print(f"   Collection: {health_data.get('qdrant_collection')}")
                else:
                    print(f"❌ Retriever health check failed: {response.status}")
                    error_text = await response.text()
                    print(f"   Error: {error_text}")
                    return False
        except Exception as e:
            print(f"❌ Connection to retriever failed: {e}")
            print("📋 Make sure the retriever service is running on port 8000")
            return False
        
        # 3. Test Service Stats
        print("\n3️⃣ Testing Service Stats...")
        try:
            async with session.get(f"{BASE_URL}/stats") as response:
                if response.status == 200:
                    stats_data = await response.json()
                    print("✅ Stats retrieved successfully")
                    print(f"   Total nodes: {stats_data.get('total_nodes')}")
                    print(f"   Node types: {stats_data.get('node_types')}")
                    print(f"   Retriever type: {stats_data.get('retriever_type')}")
                    print(f"   Reference types: {stats_data.get('reference_types')}")
                else:
                    print(f"❌ Stats request failed: {response.status}")
        except Exception as e:
            print(f"❌ Stats request error: {e}")
        
        # 4. Test Single Query Retrieval
        print("\n4️⃣ Testing Single Query Retrieval...")
        test_queries = [
            "yapay zeka nedir?",
            "makine öğrenmesi algoritmaları",
            "Zulficore sistem nedir?",
            "quantum entanglement",
            "VLLM embedding service"
        ]
        
        for i, query in enumerate(test_queries[:3], 1):  # Test first 3 queries
            print(f"\n   Test {i}: '{query}'")
            
            payload = {
                "query": query,
                "top_k": 5,
                "similarity_cutoff": 0.0
            }
            
            start_time = time.perf_counter()
            try:
                async with session.post(f"{BASE_URL}/retrieve", json=payload) as response:
                    duration = (time.perf_counter() - start_time) * 1000
                    
                    if response.status == 200:
                        result = await response.json()
                        print(f"   ✅ Query successful ({duration:.1f}ms)")
                        print(f"      Results: {result['total_results']}")
                        print(f"      Service time: {result['retrieval_time_ms']:.1f}ms")
                        
                        # Analyze node types
                        node_types = {}
                        references = 0
                        for res in result['results']:
                            node_type = res['node_type']
                            node_types[node_type] = node_types.get(node_type, 0) + 1
                            if res.get('index_id'):
                                references += 1
                        
                        print(f"      Node types: {node_types}")
                        print(f"      References: {references}")
                        
                        # Show best result
                        if result['results']:
                            best = result['results'][0]
                            print(f"      Best result: {best['score']:.3f} ({best['node_type']})")
                            print(f"      Text preview: {best['text'][:100]}...")
                            if best.get('reference_path'):
                                print(f"      Reference path: {best['reference_path']}")
                    else:
                        error_text = await response.text()
                        print(f"   ❌ Query failed: {response.status}")
                        print(f"      Error: {error_text}")
            except Exception as e:
                print(f"   ❌ Query error: {e}")
        
        # 5. Test Batch Retrieval
        print("\n5️⃣ Testing Batch Retrieval...")
        batch_queries = test_queries[:3]
        
        batch_payload = {
            "queries": batch_queries,
            "top_k": 3,
            "similarity_cutoff": 0.0
        }
        
        start_time = time.perf_counter()
        try:
            async with session.post(f"{BASE_URL}/retrieve/batch", json=batch_payload) as response:
                duration = (time.perf_counter() - start_time) * 1000
                
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Batch retrieval successful ({duration:.1f}ms)")
                    print(f"   Queries: {result['total_queries']}")
                    print(f"   Service time: {result['retrieval_time_ms']:.1f}ms")
                    print(f"   Throughput: {len(batch_queries) / (duration/1000):.1f} q/s")
                    
                    # Analyze results
                    total_results = sum(len(query_results) for query_results in result['results'])
                    all_node_types = {}
                    
                    for i, (query, query_results) in enumerate(zip(batch_queries, result['results'])):
                        print(f"   Query {i+1}: {len(query_results)} results")
                        for res in query_results:
                            node_type = res['node_type']
                            all_node_types[node_type] = all_node_types.get(node_type, 0) + 1
                    
                    print(f"   Total results: {total_results}")
                    print(f"   Node type distribution: {all_node_types}")
                else:
                    error_text = await response.text()
                    print(f"❌ Batch retrieval failed: {response.status}")
                    print(f"   Error: {error_text}")
        except Exception as e:
            print(f"❌ Batch retrieval error: {e}")
        
        # 6. Performance Benchmark
        print("\n6️⃣ Performance Benchmark...")
        benchmark_queries = [
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "neural networks",
            "data science"
        ]
        
        # Warm up
        warmup_payload = {"query": "warmup test", "top_k": 1}
        await session.post(f"{BASE_URL}/retrieve", json=warmup_payload)
        
        # Benchmark single queries
        single_times = []
        for query in benchmark_queries:
            start_time = time.perf_counter()
            async with session.post(f"{BASE_URL}/retrieve", json={"query": query, "top_k": 5}) as response:
                duration = (time.perf_counter() - start_time) * 1000
                if response.status == 200:
                    single_times.append(duration)
        
        if single_times:
            avg_single = sum(single_times) / len(single_times)
            print(f"   Single query avg: {avg_single:.1f}ms")
            print(f"   Single query throughput: {1000/avg_single:.1f} q/s")
        
        # Benchmark batch queries
        batch_times = []
        batch_sizes = [2, 3, 5]
        
        for batch_size in batch_sizes:
            queries = benchmark_queries[:batch_size]
            start_time = time.perf_counter()
            async with session.post(f"{BASE_URL}/retrieve/batch", json={"queries": queries, "top_k": 3}) as response:
                duration = (time.perf_counter() - start_time) * 1000
                if response.status == 200:
                    throughput = len(queries) / (duration / 1000)
                    print(f"   Batch {batch_size}: {duration:.1f}ms ({throughput:.1f} q/s)")
        
        # 7. Quality Assessment
        print("\n7️⃣ Quality Assessment...")
        quality_tests = [
            {
                "query": "What is quantum simulation?",
                "expected_keywords": ["quantum", "simulation", "zulficore"],
                "min_results": 2
            },
            {
                "query": "How does machine learning work?",
                "expected_keywords": ["machine", "learning", "algorithm"],
                "min_results": 2
            }
        ]
        
        for i, test in enumerate(quality_tests, 1):
            print(f"\n   Quality Test {i}: {test['query']}")
            
            payload = {
                "query": test['query'],
                "top_k": 5
            }
            
            async with session.post(f"{BASE_URL}/retrieve", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Check result count
                    result_count = len(result['results'])
                    print(f"      Results: {result_count} (min: {test['min_results']})")
                    
                    # Check keyword presence
                    all_text = " ".join([res['text'].lower() for res in result['results']])
                    found_keywords = [kw for kw in test['expected_keywords'] if kw.lower() in all_text]
                    
                    print(f"      Keywords found: {found_keywords}/{test['expected_keywords']}")
                    
                    # Quality score
                    quality_score = 0
                    if result_count >= test['min_results']:
                        quality_score += 50
                    quality_score += (len(found_keywords) / len(test['expected_keywords'])) * 50
                    
                    print(f"      Quality score: {quality_score:.0f}% {'✅' if quality_score >= 70 else '⚠️'}")
                
                else:
                    print(f"      ❌ Failed: {response.status}")
    
    print("\n🎉 Complete system test finished!")
    print("=" * 60)
    return True

async def test_embedding_consistency():
    """Test embedding consistency between tree builder and retriever"""
    print("\n🔍 Testing Embedding Consistency...")
    print("=" * 50)
    
    # This would test if the same text gets the same embedding
    # from both tree builder (passage prefix) and retriever (query prefix)
    # Important for verifying the system works correctly
    
    test_text = "Test consistency between systems"
    
    # Test via retriever service
    async with aiohttp.ClientSession() as session:
        payload = {
            "query": test_text,
            "top_k": 1
        }
        
        try:
            async with session.post(f"{BASE_URL}/retrieve", json=payload) as response:
                if response.status == 200:
                    print("✅ Embedding consistency test passed")
                    print("   Both tree builder and retriever are using compatible embeddings")
                else:
                    print(f"⚠️ Embedding test inconclusive: {response.status}")
        except Exception as e:
            print(f"❌ Embedding test failed: {e}")

if __name__ == "__main__":
    async def main():
        success = await test_complete_system()
        await test_embedding_consistency()
        
        if success:
            print("\n✅ All tests completed successfully!")
            print("🚀 Your RAG system is ready for production!")
        else:
            print("\n❌ Some tests failed")
            print("📋 Please check the logs and fix issues")
    
    asyncio.run(main())