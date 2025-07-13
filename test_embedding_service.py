#!/usr/bin/env python3
"""
VLLM Embedding Service Test
Mevcut servisin çalışıp çalışmadığını test eder
"""
import asyncio
import aiohttp
import time

EMBEDDING_SERVICE_URL = "http://localhost:8008"

async def test_embedding_service():
    """Test VLLM embedding service"""
    print("🧪 Testing VLLM Embedding Service...")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        # 1. Health Check
        print("1️⃣ Health Check:")
        try:
            async with session.get(f"{EMBEDDING_SERVICE_URL}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    print("✅ Service is healthy")
                    print(f"   Model: {health_data.get('model')}")
                    print(f"   Status: {health_data.get('status')}")
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
        
        # 2. Single Embedding Test
        print("\n2️⃣ Single Embedding Test:")
        payload = {
            "input": ["passage: Bu bir test metnidir. Türkçe embedding test ediyoruz."],
            "model": "intfloat/multilingual-e5-large"
        }
        
        start_time = time.perf_counter()
        try:
            async with session.post(f"{EMBEDDING_SERVICE_URL}/v1/embeddings", json=payload) as response:
                duration = (time.perf_counter() - start_time) * 1000
                
                if response.status == 200:
                    result = await response.json()
                    embedding = result['data'][0]['embedding']
                    
                    print("✅ Single embedding successful")
                    print(f"   Latency: {duration:.1f}ms")
                    print(f"   Dimension: {len(embedding)}")
                    print(f"   Model: {result.get('model')}")
                else:
                    error_text = await response.text()
                    print(f"❌ Single embedding failed: {response.status}")
                    print(f"   Error: {error_text}")
                    return False
        except Exception as e:
            print(f"❌ Single embedding error: {e}")
            return False
        
        # 3. Batch Embedding Test
        print("\n3️⃣ Batch Embedding Test:")
        batch_texts = [
            "passage: Bu birinci test metnidir.",
            "passage: Bu ikinci test metnidir.",
            "passage: Bu üçüncü test metnidir.",
            "query: Test sorgusu nedir?",
            "query: Başka bir test sorgusu."
        ]
        
        batch_payload = {
            "input": batch_texts,
            "model": "intfloat/multilingual-e5-large"
        }
        
        start_time = time.perf_counter()
        try:
            async with session.post(f"{EMBEDDING_SERVICE_URL}/v1/embeddings", json=batch_payload) as response:
                duration = (time.perf_counter() - start_time) * 1000
                
                if response.status == 200:
                    result = await response.json()
                    embeddings = [item['embedding'] for item in result['data']]
                    
                    print("✅ Batch embedding successful")
                    print(f"   Latency: {duration:.1f}ms")
                    print(f"   Texts: {len(batch_texts)}")
                    print(f"   Embeddings: {len(embeddings)}")
                    print(f"   Throughput: {len(batch_texts) / (duration/1000):.1f} texts/sec")
                    
                    # Test dimension consistency
                    dimensions = [len(emb) for emb in embeddings]
                    if len(set(dimensions)) == 1:
                        print(f"   ✅ All embeddings have same dimension: {dimensions[0]}")
                    else:
                        print(f"   ❌ Dimension mismatch: {dimensions}")
                        return False
                        
                else:
                    error_text = await response.text()
                    print(f"❌ Batch embedding failed: {response.status}")
                    print(f"   Error: {error_text}")
                    return False
        except Exception as e:
            print(f"❌ Batch embedding error: {e}")
            return False
        
        # 4. Prefix Test (query vs passage)
        print("\n4️⃣ Prefix Test (query vs passage):")
        query_text = "query: makine öğrenmesi nedir?"
        passage_text = "passage: Makine öğrenmesi, bilgisayarların deneyim yoluyla öğrenmesini sağlayan bir yapay zeka dalıdır."
        
        prefix_payload = {
            "input": [query_text, passage_text],
            "model": "intfloat/multilingual-e5-large"
        }
        
        try:
            async with session.post(f"{EMBEDDING_SERVICE_URL}/v1/embeddings", json=prefix_payload) as response:
                if response.status == 200:
                    result = await response.json()
                    query_emb = result['data'][0]['embedding']
                    passage_emb = result['data'][1]['embedding']
                    
                    # Calculate cosine similarity
                    import numpy as np
                    query_emb = np.array(query_emb)
                    passage_emb = np.array(passage_emb)
                    
                    # Normalize
                    query_emb = query_emb / np.linalg.norm(query_emb)
                    passage_emb = passage_emb / np.linalg.norm(passage_emb)
                    
                    similarity = np.dot(query_emb, passage_emb)
                    
                    print("✅ Prefix test successful")
                    print(f"   Query embedding dim: {len(query_emb)}")
                    print(f"   Passage embedding dim: {len(passage_emb)}")
                    print(f"   Cosine similarity: {similarity:.4f}")
                    
                    if similarity > 0.3:  # Reasonable threshold for related content
                        print("   ✅ Similarity looks good for related content")
                    else:
                        print("   ⚠️  Low similarity - might be normal for different prefixes")
                        
                else:
                    print(f"❌ Prefix test failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Prefix test error: {e}")
            return False
    
    print("\n🎉 All tests passed! Embedding service is working correctly.")
    return True

async def test_qdrant_connection():
    """Test Qdrant connection"""
    print("\n🔍 Testing Qdrant Connection...")
    print("=" * 50)
    
    try:
        from qdrant_client import QdrantClient
        
        # Test connection
        client = QdrantClient(url="http://localhost:6333")
        
        # Get collections
        collections = client.get_collections()
        
        print("✅ Qdrant connection successful")
        print(f"   Collections: {[col.name for col in collections.collections]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        print("   Make sure Qdrant is running on localhost:6333")
        return False

if __name__ == "__main__":
    async def main():
        # Test embedding service
        embedding_ok = await test_embedding_service()
        
        # Test Qdrant
        qdrant_ok = await test_qdrant_connection()
        
        if embedding_ok and qdrant_ok:
            print("\n✅ All services are ready!")
            print("📋 Next: Fix tree builder bugs")
        else:
            print("\n❌ Some services need attention")
    
    asyncio.run(main())