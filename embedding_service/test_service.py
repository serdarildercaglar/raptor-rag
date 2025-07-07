# embedding_service/test_service.py
import asyncio
import aiohttp
import time
import json
from typing import List

class EmbeddingServiceTester:
    def __init__(self, base_url: str = "http://localhost:8008"):
        self.base_url = base_url
        
    async def test_health(self):
        """Test health endpoint"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/health") as response:
                    result = await response.json()
                    print(f"✅ Health check: {result}")
                    return True
            except Exception as e:
                print(f"❌ Health check failed: {e}")
                return False
    
    async def test_single_embedding(self):
        """Test single embedding via OpenAI compatible endpoint"""
        payload = {
            "input": ["How much protein should a female eat"],
            "model": "intfloat/multilingual-e5-large"
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                start_time = time.time()
                async with session.post(
                    f"{self.base_url}/v1/embeddings",
                    json=payload
                ) as response:
                    result = await response.json()
                    end_time = time.time()
                    
                    if response.status == 200:
                        embedding_size = len(result['data'][0]['embedding'])
                        print(f"✅ Single embedding: {embedding_size}D vector")
                        print(f"⏱️  Time: {end_time - start_time:.2f}s")
                        return True
                    else:
                        print(f"❌ Single embedding failed: {result}")
                        return False
                        
            except Exception as e:
                print(f"❌ Single embedding error: {e}")
                return False
    
    async def test_batch_embedding(self):
        """Test batch embedding endpoint"""
        test_texts = [
            "How much protein should a female eat",
            "What is the capital of France",
            "Explain machine learning",
            "Best practices for Python programming",
            "Climate change effects"
        ]
        
        async with aiohttp.ClientSession() as session:
            try:
                start_time = time.time()
                async with session.post(
                    f"{self.base_url}/embeddings/batch",
                    json=test_texts
                ) as response:
                    result = await response.json()
                    end_time = time.time()
                    
                    if response.status == 200:
                        num_embeddings = len(result['data'])
                        embedding_size = len(result['data'][0]['embedding'])
                        print(f"✅ Batch embedding: {num_embeddings} texts, {embedding_size}D vectors")
                        print(f"⏱️  Time: {end_time - start_time:.2f}s")
                        print(f"📊 Throughput: {num_embeddings/(end_time - start_time):.2f} texts/sec")
                        return True
                    else:
                        print(f"❌ Batch embedding failed: {result}")
                        return False
                        
            except Exception as e:
                print(f"❌ Batch embedding error: {e}")
                return False
    
    async def test_concurrent_requests(self, num_concurrent: int = 5):
        """Test concurrent requests"""
        print(f"\n🔄 Testing {num_concurrent} concurrent requests...")
        
        async def single_request(session, request_id):
            payload = {
                "input": [f"Test request {request_id}"],
                "model": "intfloat/multilingual-e5-large"
            }
            
            start_time = time.time()
            async with session.post(
                f"{self.base_url}/v1/embeddings",
                json=payload
            ) as response:
                result = await response.json()
                end_time = time.time()
                
                return {
                    "request_id": request_id,
                    "status": response.status,
                    "time": end_time - start_time,
                    "success": response.status == 200
                }
        
        async with aiohttp.ClientSession() as session:
            try:
                start_time = time.time()
                tasks = [single_request(session, i) for i in range(num_concurrent)]
                results = await asyncio.gather(*tasks)
                end_time = time.time()
                
                successful = sum(1 for r in results if r['success'])
                avg_time = sum(r['time'] for r in results) / len(results)
                total_time = end_time - start_time
                
                print(f"✅ Concurrent test: {successful}/{num_concurrent} successful")
                print(f"⏱️  Total time: {total_time:.2f}s")
                print(f"📊 Average per request: {avg_time:.2f}s")
                print(f"🚀 Throughput: {num_concurrent/total_time:.2f} requests/sec")
                
                return successful == num_concurrent
                
            except Exception as e:
                print(f"❌ Concurrent test error: {e}")
                return False
    
    async def run_all_tests(self):
        """Run all tests"""
        print("🧪 Starting Embedding Service Tests...\n")
        
        # Test 1: Health check
        print("1️⃣ Health Check")
        health_ok = await self.test_health()
        if not health_ok:
            print("❌ Health check failed, stopping tests")
            return
        
        # Test 2: Single embedding
        print("\n2️⃣ Single Embedding Test")
        single_ok = await self.test_single_embedding()
        
        # Test 3: Batch embedding
        print("\n3️⃣ Batch Embedding Test")
        batch_ok = await self.test_batch_embedding()
        
        # Test 4: Concurrent requests
        print("\n4️⃣ Concurrent Requests Test")
        concurrent_ok = await self.test_concurrent_requests()
        
        # Summary
        print("\n" + "="*50)
        print("📋 Test Summary:")
        print(f"   Health Check: {'✅' if health_ok else '❌'}")
        print(f"   Single Embedding: {'✅' if single_ok else '❌'}")
        print(f"   Batch Embedding: {'✅' if batch_ok else '❌'}")
        print(f"   Concurrent Requests: {'✅' if concurrent_ok else '❌'}")
        
        if all([health_ok, single_ok, batch_ok, concurrent_ok]):
            print("\n🎉 All tests passed! Service is ready for production.")
        else:
            print("\n⚠️  Some tests failed. Check the service configuration.")

async def main():
    tester = EmbeddingServiceTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())