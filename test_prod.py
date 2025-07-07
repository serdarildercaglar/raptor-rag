#!/usr/bin/env python3
"""
Production Performance Test - Simulates real agentic AI usage
"""
import asyncio
import time
from typing import List
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig, VLLMEmbeddingModel

class ProductionTester:
    def __init__(self, tree_path: str, vllm_url: str = "http://localhost:8008"):
        self.tree_path = tree_path
        self.vllm_url = vllm_url
        self.embedding_model = None
        self.RA = None

    async def setup(self):
        """Initialize RAPTOR with VLLM"""
        self.embedding_model = VLLMEmbeddingModel(self.vllm_url)
        config = RetrievalAugmentationConfig(embedding_model=self.embedding_model)
        self.RA = RetrievalAugmentation(config=config, tree=self.tree_path)
        print("✅ RAPTOR Production Setup Complete")

    async def cleanup(self):
        """Clean up resources"""
        if self.embedding_model:
            await self.embedding_model.close()
        print("✅ Resources cleaned up")

    async def test_single_user_queries(self):
        """Test single user with multiple queries (Agentic AI scenario)"""
        print("\n1️⃣ Single User - Multiple Queries Test")
        
        # Simulate agentic AI generating 3-5 queries per user
        user_queries = [
            "edebiyat nedir?",
            "şiir türleri nelerdir?", 
            "roman ve hikaye arasındaki fark nedir?",
            "modern edebiyatın özellikleri nelerdir?"
        ]
        
        # Test sync version
        print("  📊 Sync Version:")
        start = time.time()
        for i, query in enumerate(user_queries):
            context = self.RA.retrieve(query)
            print(f"    Q{i+1}: {len(context)} chars")
        sync_time = time.time() - start
        print(f"    ⏱️ Sync Total: {sync_time:.2f}s")
        
        # Test async batch version  
        print("  📊 Async Batch Version:")
        start = time.time()
        contexts = await self.RA.retrieve_batch(user_queries)
        async_time = time.time() - start
        print(f"    ⏱️ Async Batch: {async_time:.2f}s")
        print(f"    🚀 Speedup: {sync_time/async_time:.1f}x faster")
        
        for i, (query, context) in enumerate(zip(user_queries, contexts)):
            print(f"    Q{i+1}: {len(context)} chars")

    async def test_concurrent_users(self, num_users: int = 10):
        """Test concurrent users (Production scenario)"""
        print(f"\n2️⃣ Concurrent Users Test ({num_users} users)")
        
        # Generate different queries for each user
        base_queries = [
            ["edebiyat nedir?", "şiir nedir?"],
            ["roman türleri nelerdir?", "hikaye nasıl yazılır?"],
            ["modern edebiyat nedir?", "klasik edebiyat nedir?"],
            ["destan nedir?", "masal nedir?"],
            ["tiyatro nedir?", "komedi nedir?"]
        ]
        
        async def simulate_user(user_id: int):
            """Simulate one user's agentic AI queries"""
            queries = base_queries[user_id % len(base_queries)]
            queries = [f"{q} (kullanıcı {user_id})" for q in queries]
            
            start = time.time()
            try:
                contexts = await self.RA.retrieve_batch(queries)
                end = time.time()
                return {
                    "user_id": user_id,
                    "success": True,
                    "time": end - start,
                    "queries": len(queries),
                    "total_chars": sum(len(ctx) for ctx in contexts)
                }
            except Exception as e:
                return {
                    "user_id": user_id,
                    "success": False,
                    "error": str(e)
                }
        
        start_time = time.time()
        results = await asyncio.gather(*[simulate_user(i) for i in range(num_users)])
        total_time = time.time() - start_time
        
        # Analyze results
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        total_queries = sum(r['queries'] for r in successful)
        avg_time = sum(r['time'] for r in successful) / len(successful) if successful else 0
        
        print(f"  📊 Results:")
        print(f"    ✅ Successful users: {len(successful)}/{num_users}")
        print(f"    ❌ Failed users: {len(failed)}")
        print(f"    ⏱️ Total time: {total_time:.2f}s")
        print(f"    📈 Avg time per user: {avg_time:.2f}s")
        print(f"    🔥 Total queries: {total_queries}")
        print(f"    🚀 Throughput: {total_queries/total_time:.1f} queries/sec")
        print(f"    👥 User throughput: {len(successful)/total_time:.1f} users/sec")
        
        if failed:
            print(f"  ❌ Failures:")
            for f in failed[:3]:  # Show first 3 failures
                print(f"    User {f['user_id']}: {f['error']}")

    async def test_sustained_load(self, duration_seconds: int = 60):
        """Test sustained load for a duration"""
        print(f"\n3️⃣ Sustained Load Test ({duration_seconds}s)")
        
        query_templates = [
            "edebiyat hakkında bilgi ver",
            "şiir türleri nelerdir", 
            "roman analizi yap",
            "modern edebiyat nedir",
            "klasik eserler hakkında"
        ]
        
        completed_requests = 0
        start_time = time.time()
        
        async def continuous_requests():
            nonlocal completed_requests
            query_counter = 0
            
            while time.time() - start_time < duration_seconds:
                try:
                    # Generate batch of queries
                    queries = [
                        f"{query_templates[i % len(query_templates)]} #{query_counter + i}"
                        for i in range(3)
                    ]
                    
                    await self.RA.retrieve_batch(queries)
                    completed_requests += len(queries)
                    query_counter += len(queries)
                    
                    # Small delay to prevent overwhelming
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    print(f"    ⚠️ Error: {e}")
                    await asyncio.sleep(1)
        
        # Run continuous requests
        await continuous_requests()
        
        actual_duration = time.time() - start_time
        throughput = completed_requests / actual_duration
        
        print(f"  📊 Sustained Load Results:")
        print(f"    ⏱️ Duration: {actual_duration:.1f}s")
        print(f"    🔥 Total requests: {completed_requests}")
        print(f"    🚀 Average throughput: {throughput:.1f} requests/sec")

    async def run_all_tests(self):
        """Run complete production test suite"""
        print("🏭 Production Performance Test Suite")
        print("=" * 50)
        
        try:
            await self.setup()
            
            # Test 1: Single user scenario
            await self.test_single_user_queries()
            
            # Test 2: Concurrent users
            await self.test_concurrent_users(num_users=10)
            
            # Test 3: Sustained load
            await self.test_sustained_load(duration_seconds=30)
            
            print("\n" + "=" * 50)
            print("🎉 All production tests completed!")
            
        finally:
            await self.cleanup()

async def main():
    # Configuration
    TREE_PATH = "/home/serdar/Documents/raptor-rag/vectordb/raptor-db"
    VLLM_URL = "http://localhost:8008"
    
    tester = ProductionTester(TREE_PATH, VLLM_URL)
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())