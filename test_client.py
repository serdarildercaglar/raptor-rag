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

def format_query_context(query: str, context: str, max_preview: int = 200):
    """Format query and context for display"""
    print(f"\n📝 Query: {query}")
    print(f"📄 Context ({len(context)} chars):")
    print("=" * 60)
    if len(context) > max_preview:
        print(f"{context[:max_preview]}...")
        print(f"[... {len(context) - max_preview} more characters]")
    else:
        print(context)
    print("=" * 60)

def format_service_info(health_data: Dict[str, Any]):
    """Format service information"""
    print("🏥 Service Health:")
    print(f"   Status: {'🟢' if health_data['status'] == 'healthy' else '🔴'} {health_data['status']}")
    print(f"   Uptime: {health_data.get('uptime_seconds', 0):.1f}s")
    print(f"   Total Requests: {health_data.get('total_requests', 0)}")
    print(f"   Avg Response Time: {health_data.get('average_response_time_ms', 0):.1f}ms")
    print(f"   Embedding Service: {'🟢' if 'healthy' in health_data.get('embedding_service_status', '') else '🔴'}")

def format_performance_metrics(processing_time: float, context_length: int, query_count: int = 1):
    """Format performance metrics"""
    print(f"⚡ Performance:")
    print(f"   Processing Time: {processing_time:.1f}ms")
    print(f"   Context Length: {context_length:,} chars")
    if query_count > 1:
        print(f"   Queries: {query_count}")
        print(f"   Avg per Query: {processing_time/query_count:.1f}ms")
        print(f"   Throughput: {query_count/(processing_time/1000):.1f} queries/sec")

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
        format_service_info(health)
        
        # Single retrieve
        print("\n2️⃣ Single Retrieve:")
        query = "edebiyat nedir?"
        start = time.time()
        result = await client.retrieve(query)
        end = time.time()
        
        format_query_context(result['query'], result['context'])
        format_performance_metrics(
            result['processing_time_ms'], 
            result['context_length']
        )
        print(f"   Client Round-trip: {(end-start)*1000:.1f}ms")
        
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
        
        print(f"\n📊 Batch Results:")
        for i, result in enumerate(batch_result['results']):
            print(f"\n   Query {i+1}: {result['query']}")
            print(f"   Context: {result['context_length']} chars")
            print(f"   Preview: {result['context'][:100]}...")
        
        format_performance_metrics(
            batch_result['total_processing_time_ms'],
            sum(r['context_length'] for r in batch_result['results']),
            batch_result['total_queries']
        )
        print(f"   Client Round-trip: {(end-start)*1000:.1f}ms")

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
        print("=" * 50)
        
        start = time.time()
        research_result = await tool.research_topic(question)
        end = time.time()
        
        print(f"🎯 Original Question: {research_result['original_question']}")
        print(f"📝 Generated Queries: {len(research_result['research_queries'])}")
        
        # Show each query and its context
        for i, context_data in enumerate(research_result['contexts']):
            format_query_context(
                context_data['query'], 
                context_data['context'],
                max_preview=150
            )
        
        # Summary metrics
        print(f"\n📊 Research Summary:")
        summary = research_result['summary']
        print(f"   Total Characters: {summary['total_characters']:,}")
        print(f"   Processing Time: {research_result['total_processing_time_ms']:.1f}ms")
        print(f"   End-to-end Time: {(end-start)*1000:.1f}ms")
        print(f"   Throughput: {summary['queries_per_second']:.1f} queries/sec")

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
                "success": True,
                "total_chars": sum(r['context_length'] for r in result['results'])
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
    total_chars = sum(r['total_chars'] for r in results)
    
    print(f"\n📊 Load Test Results:")
    print(f"   Total Users: {len(results)}")
    print(f"   Total Queries: {total_queries}")
    print(f"   Total Characters: {total_chars:,}")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Average Processing Time: {avg_processing_time:.1f}ms")
    print(f"   System Throughput: {total_queries/total_time:.1f} queries/sec")
    print(f"   User Throughput: {len(results)/total_time:.1f} users/sec")
    
    # Show individual user results
    print(f"\n👥 Individual User Results:")
    for result in results:
        print(f"   User {result['user_id']}: {result['queries']} queries, "
              f"{result['processing_time_ms']:.1f}ms processing, "
              f"{result['total_chars']:,} chars")

async def main():
    """Run all tests"""
    print("🔬 RAPTOR Service Client Tests")
    print("=" * 60)
    
    try:
        await test_basic_functionality()
        await test_agentic_ai_integration()
        await test_performance_load()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("🚀 RAPTOR Service is ready for Agentic AI integration!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())