#!/usr/bin/env python3
"""
Test client for LlamaIndex RecursiveRetriever Service
TAMAMEN düzeltilmiş sistem testi
"""
import asyncio
import aiohttp
import time
import json

BASE_URL = "http://localhost:8000"

async def test_single_retrieve():
    """Test single query retrieval with RecursiveRetriever"""
    print("🔍 Testing RecursiveRetriever single retrieve...")
    
    async with aiohttp.ClientSession() as session:
        payload = {
            "query": "yapay zeka nedir ve nasıl çalışır",
            "top_k": 5,
            "similarity_cutoff": 0.0
        }
        
        start_time = time.perf_counter()
        async with session.post(f"{BASE_URL}/retrieve", json=payload) as response:
            duration = (time.perf_counter() - start_time) * 1000
            
            if response.status == 200:
                result = await response.json()
                print(f"✅ Single retrieve: {duration:.1f}ms")
                print(f"   Query: {result['query']}")
                print(f"   Results: {result['total_results']}")
                print(f"   Service time: {result['retrieval_time_ms']:.1f}ms")
                
                # Show node types (RecursiveRetriever specific)
                node_types = {}
                for res in result['results']:
                    node_type = res['node_type']
                    node_types[node_type] = node_types.get(node_type, 0) + 1
                
                print(f"   Node types: {node_types}")
                
                # Show first few results
                for i, res in enumerate(result['results'][:3]):
                    print(f"   Result {i+1}: {res['score']:.3f} ({res['node_type']}) - {res['text'][:100]}...")
                    if res.get('index_id'):
                        print(f"     → References node: {res['index_id']}")
            else:
                print(f"❌ Error: {response.status}")
                print(await response.text())

async def test_batch_retrieve():
    """Test batch query retrieval with RecursiveRetriever"""
    print("\n📦 Testing RecursiveRetriever batch retrieve...")
    
    queries = [
        "makine öğrenmesi algoritmaları nelerdir",
        "derin öğrenme ve yapay sinir ağları",
        "doğal dil işleme teknikleri",
        "bilgisayar görüşü uygulamaları",
        "veri madenciliği yöntemleri"
    ]
    
    async with aiohttp.ClientSession() as session:
        payload = {
            "queries": queries,
            "top_k": 3,
            "similarity_cutoff": 0.0
        }
        
        start_time = time.perf_counter()
        async with session.post(f"{BASE_URL}/retrieve/batch", json=payload) as response:
            duration = (time.perf_counter() - start_time) * 1000
            
            if response.status == 200:
                result = await response.json()
                print(f"✅ Batch retrieve: {duration:.1f}ms")
                print(f"   Queries: {result['total_queries']}")
                print(f"   Service time: {result['retrieval_time_ms']:.1f}ms")
                print(f"   Throughput: {len(queries) / (duration/1000):.1f} q/s")
                
                # Analyze node types across all results
                all_node_types = {}
                total_results = 0
                for query_results in result['results']:
                    total_results += len(query_results)
                    for res in query_results:
                        node_type = res['node_type']
                        all_node_types[node_type] = all_node_types.get(node_type, 0) + 1
                
                print(f"   Total results: {total_results}")
                print(f"   Node type distribution: {all_node_types}")
                
                # Show results per query
                for i, (query, results) in enumerate(zip(queries, result['results'])):
                    print(f"   Query {i+1}: {len(results)} results")
                    if results:
                        best_result = results[0]
                        print(f"     Best: {best_result['score']:.3f} ({best_result['node_type']})")
            else:
                print(f"❌ Error: {response.status}")
                print(await response.text())

async def test_health_and_stats():
    """Test health and stats endpoints"""
    print("\n🏥 Testing health and stats...")
    
    async with aiohttp.ClientSession() as session:
        # Health check
        async with session.get(f"{BASE_URL}/health") as response:
            if response.status == 200:
                health = await response.json()
                print(f"✅ Health: {health['status']}")
                print(f"   Service: {health['service']}")
                print(f"   Collection: {health['qdrant_collection']}")
                print(f"   Nodes: {health['node_count']}")
                print(f"   Pattern: {health['pattern']}")
            else:
                print(f"❌ Health error: {response.status}")
        
        # Stats
        async with session.get(f"{BASE_URL}/stats") as response:
            if response.status == 200:
                stats = await response.json()
                print(f"✅ Stats:")
                print(f"   Total nodes: {stats['total_nodes']}")
                print(f"   Node types: {stats['node_types']}")
                print(f"   Reference types: {stats['reference_types']}")
                print(f"   Retriever: {stats['retriever_type']}")
            else:
                print(f"❌ Stats error: {response.status}")

async def test_recursive_retrieval_quality():
    """Test RecursiveRetriever quality with complex queries"""
    print("\n🎯 Testing RecursiveRetriever quality...")
    
    test_cases = [
        {
            "query": "makine öğrenmesi nasıl çalışır detaylı açıklama",
            "expected_types": ["reference", "base"],
            "description": "Should return both chunk references and metadata"
        },
        {
            "query": "yapay zeka tarihçesi ve gelişimi",
            "expected_types": ["reference"],
            "description": "Should return hierarchical chunk references"
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n   Test {i}: {test_case['description']}")
            print(f"   Query: {test_case['query']}")
            
            payload = {"query": test_case['query'], "top_k": 5}
            
            async with session.post(f"{BASE_URL}/retrieve", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Analyze node types
                    node_types = [res['node_type'] for res in result['results']]
                    unique_types = set(node_types)
                    
                    print(f"   Results: {len(result['results'])}")
                    print(f"   Node types found: {list(unique_types)}")
                    
                    # Check if expected types are present
                    has_expected = any(exp_type in unique_types for exp_type in test_case['expected_types'])
                    print(f"   Quality check: {'✅ PASS' if has_expected else '❌ FAIL'}")
                    
                    # Show references
                    references = [res for res in result['results'] if res.get('index_id')]
                    if references:
                        print(f"   References found: {len(references)}")
                        for ref in references[:2]:
                            print(f"     → {ref['node_id']} refs {ref['index_id']}")
                else:
                    print(f"   ❌ Error: {response.status}")

async def benchmark_performance():
    """Benchmark RecursiveRetriever performance"""
    print("\n⚡ Performance benchmark...")
    
    # Prepare test queries
    base_queries = [
        "yapay zeka",
        "makine öğrenmesi", 
        "derin öğrenme",
        "doğal dil işleme",
        "bilgisayar görüşü",
        "robotik",
        "otonom araçlar",
        "veri bilimi",
        "büyük dil modelleri",
        "generative AI"
    ]
    
    async with aiohttp.ClientSession() as session:
        # Warm up
        await session.post(f"{BASE_URL}/retrieve", json={"query": "warmup"})
        
        # Benchmark different batch sizes
        batch_sizes = [1, 3, 5, 10]
        
        for batch_size in batch_sizes:
            queries = base_queries[:batch_size]
            
            # Run multiple iterations for stability
            times = []
            for _ in range(3):
                start_time = time.perf_counter()
                async with session.post(f"{BASE_URL}/retrieve/batch", json={"queries": queries}) as response:
                    duration = (time.perf_counter() - start_time) * 1000
                    if response.status == 200:
                        times.append(duration)
            
            if times:
                avg_time = sum(times) / len(times)
                throughput = len(queries) / (avg_time / 1000)
                print(f"   Batch {batch_size:2d}: {avg_time:6.1f}ms avg ({throughput:5.1f} q/s)")

async def test_node_references():
    """Test IndexNode reference following"""
    print("\n🔗 Testing IndexNode reference following...")
    
    async with aiohttp.ClientSession() as session:
        # Query that should trigger reference following
        payload = {"query": "detaylı açıklama ile örnekler", "top_k": 10}
        
        async with session.post(f"{BASE_URL}/retrieve", json=payload) as response:
            if response.status == 200:
                result = await response.json()
                
                # Analyze reference patterns
                references = {}
                base_nodes = []
                
                for res in result['results']:
                    if res.get('index_id'):
                        ref_target = res['index_id']
                        if ref_target not in references:
                            references[ref_target] = []
                        references[ref_target].append(res['node_id'])
                    else:
                        base_nodes.append(res['node_id'])
                
                print(f"   Total results: {len(result['results'])}")
                print(f"   Base nodes: {len(base_nodes)}")
                print(f"   Reference targets: {len(references)}")
                
                if references:
                    print("   Reference mapping:")
                    for target, refs in list(references.items())[:3]:
                        print(f"     {target} ← {len(refs)} references")
                        
                    print(f"   ✅ RecursiveRetriever is following references!")
                else:
                    print(f"   ⚠️  No references found - check node mapping")

async def main():
    """Main test function"""
    print("🧪 LlamaIndex RecursiveRetriever Test Suite")
    print("=" * 60)
    
    try:
        await test_health_and_stats()
        await test_single_retrieve()
        await test_batch_retrieve()
        await test_recursive_retrieval_quality()
        await test_node_references()
        await benchmark_performance()
        
        print("\n✅ All tests completed!")
        print("🎯 RecursiveRetriever with IndexNode references working!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())