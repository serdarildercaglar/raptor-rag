#!/usr/bin/env python3
"""
RAPTOR Service Client - Agentic AI Integration Example
"""
import asyncio
import aiohttp
import time
from typing import List, Dict, Any

class RaptorClient:
    """
    Agentic AI client for RAPTOR service
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        async with self.session.get(f"{self.base_url}/health") as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Health check failed: {response.status}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get detailed service status"""
        async with self.session.get(f"{self.base_url}/status") as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Status check failed: {response.status}")
    
    async def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        max_tokens: int = 3500,
        include_metadata: bool = False
    ) -> Dict[str, Any]:
        """Retrieve context for single query"""
        payload = {
            "query": query,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "include_metadata": include_metadata
        }
        
        async with self.session.post(
            f"{self.base_url}/retrieve",
            json=payload
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Retrieve failed: {response.status} - {error_text}")
    
    async def retrieve_batch(
        self, 
        queries: List[str], 
        top_k: int = 5, 
        max_tokens: int = 3500,
        include_metadata: bool = False
    ) -> Dict[str, Any]:
        """Retrieve context for multiple queries (Agentic AI optimized)"""
        payload = {
            "queries": queries,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "include_metadata": include_metadata
        }
        
        async with self.session.post(
            f"{self.base_url}/retrieve/batch",
            json=payload
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Batch retrieve failed: {response.status} - {error_text}")

# Agentic AI Tool Integration Example
class AgenticAIRAGTool:
    """
    Example tool for Agentic AI systems
    """
    
    def __init__(self, raptor_service_url: str = "http://localhost:8000"):
        self.raptor_service_url = raptor_service_url
    
    async def generate_research_queries(self, user_question: str) -> List[str]:
        """Generate multiple research queries from user question"""
        # In real implementation, this would use LLM to generate queries
        base_queries = [
            user_question,
            f"{user_question} nedir?",
            f"{user_question} hakkında detaylı bilgi",
            f"{user_question} örnekleri",
            f"{user_question} türleri"
        ]
        return base_queries[:4]  # Limit to 4 queries
    
    async def research_topic(self, user_question: str) -> Dict[str, Any]:
        """
        Complete research flow for Agentic AI:
        1. Generate multiple research queries
        2. Batch retrieve contexts
        3. Return structured results
        """
        async with RaptorClient(self.raptor_service_url) as client:
            # Check service health
            health = await client.health_check()
            if health['status'] != 'healthy':
                raise Exception(f"RAPTOR service unhealthy: {health}")
            
            # Generate research queries
            queries = await self.generate_research_queries(user_question)
            
            # Batch retrieve contexts
            result = await client.retrieve_batch(
                queries=queries,
                top_k=5,
                max_tokens=3500,
                include_metadata=False
            )
            
            # Structure results for Agentic AI
            return {
                "original_question": user_question,
                "research_queries": queries,
                "total_contexts": len(result['results']),
                "total_processing_time_ms": result['total_processing_time_ms'],
                "contexts": [
                    {
                        "query": r['query'],
                        "context": r['context'],
                        "context_length": r['context_length']
                    }
                    for r in result['results']
                ],
                "summary": {
                    "total_characters": sum(r['context_length'] for r in result['results']),
                    "average_time_ms": result['average_time_ms'],
                    "queries_per_second": len(queries) / (result['total_processing_time_ms'] / 1000)
                }
            }

# Test Functions
async def test_basic_functionality():
    """Test basic service functionality"""
    print("🧪 Testing RAPTOR Service Basic Functionality")
    
    async with RaptorClient() as client:
        # Health check
        print("\n1️⃣ Health Check:")
        health = await client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Uptime: {health['uptime_seconds']:.1f}s")
        print(f"   Embedding Service: {health['embedding_service_status']}")
        
        # Single retrieve
        print("\n2️⃣ Single Retrieve:")
        start = time.time()
        result = await client.retrieve("edebiyat nedir?")
        end = time.time()
        
        print(f"   Query: {result['query']}")
        print(f"   Context Length: {result['context_length']} chars")
        print(f"   Processing Time: {result['processing_time_ms']:.1f}ms")
        print(f"   Client Time: {(end-start)*1000:.1f}ms")
        
        # Batch retrieve
        print("\n3️⃣ Batch Retrieve:")
        queries = [
            "edebiyat nedir?",
            "şiir türleri nelerdir?",
            "roman nedir?"
        ]
        
        start = time.time()
        batch_result = await client.retrieve_batch(queries)
        end = time.time()
        
        print(f"   Total Queries: {batch_result['total_queries']}")
        print(f"   Total Processing Time: {batch_result['total_processing_time_ms']:.1f}ms")
        print(f"   Average Time: {batch_result['average_time_ms']:.1f}ms")
        print(f"   Client Time: {(end-start)*1000:.1f}ms")
        print(f"   Throughput: {batch_result['total_queries']/(batch_result['total_processing_time_ms']/1000):.1f} queries/sec")

async def test_agentic_ai_integration():
    """Test Agentic AI integration"""
    print("\n🤖 Testing Agentic AI Integration")
    
    tool = AgenticAIRAGTool()
    
    test_questions = [
        "modern edebiyat",
        "şiir analizi",
        "roman türleri"
    ]
    
    for question in test_questions:
        print(f"\n🔍 Researching: {question}")
        
        start = time.time()
        research_result = await tool.research_topic(question)
        end = time.time()
        
        print(f"   Generated Queries: {len(research_result['research_queries'])}")
        print(f"   Total Contexts: {research_result['total_contexts']}")
        print(f"   Total Characters: {research_result['summary']['total_characters']:,}")
        print(f"   Processing Time: {research_result['total_processing_time_ms']:.1f}ms")
        print(f"   End-to-end Time: {(end-start)*1000:.1f}ms")
        print(f"   Throughput: {research_result['summary']['queries_per_second']:.1f} queries/sec")

async def test_performance_load():
    """Test performance under load"""
    print("\n⚡ Performance Load Test")
    
    async def single_user_simulation(user_id: int):
        """Simulate single agentic AI user"""
        async with RaptorClient() as client:
            queries = [
                f"User {user_id}: edebiyat nedir?",
                f"User {user_id}: şiir analizi",
                f"User {user_id}: roman türleri"
            ]
            
            start = time.time()
            result = await client.retrieve_batch(queries)
            end = time.time()
            
            return {
                "user_id": user_id,
                "queries": len(queries),
                "processing_time_ms": result['total_processing_time_ms'],
                "total_time_ms": (end - start) * 1000,
                "success": True
            }
    
    # Simulate 5 concurrent agentic AI users
    num_users = 5
    print(f"   Simulating {num_users} concurrent Agentic AI users...")
    
    start_time = time.time()
    results = await asyncio.gather(*[
        single_user_simulation(i) for i in range(num_users)
    ])
    total_time = time.time() - start_time
    
    total_queries = sum(r['queries'] for r in results)
    avg_processing_time = sum(r['processing_time_ms'] for r in results) / len(results)
    
    print(f"   Results:")
    print(f"     Total Users: {len(results)}")
    print(f"     Total Queries: {total_queries}")
    print(f"     Total Time: {total_time:.2f}s")
    print(f"     Average Processing Time: {avg_processing_time:.1f}ms")
    print(f"     System Throughput: {total_queries/total_time:.1f} queries/sec")
    print(f"     User Throughput: {len(results)/total_time:.1f} users/sec")

async def main():
    """Run all tests"""
    print("🔬 RAPTOR Service Client Tests")
    print("=" * 50)
    
    try:
        await test_basic_functionality()
        await test_agentic_ai_integration()
        await test_performance_load()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())