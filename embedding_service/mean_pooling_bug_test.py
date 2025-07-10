#!/usr/bin/env python3
"""
VLLM vs Sentence-Transformers Pooling Bug Test
intfloat/multilingual-e5-large için
"""
import subprocess
# run this export CUDA_VISIBLE_DEVICES=1
subprocess.run(["export", "CUDA_VISIBLE_DEVICES=1"], shell=True, check=True)
import numpy as np
from sentence_transformers import SentenceTransformer
from vllm import LLM
from sklearn.metrics.pairwise import cosine_similarity
import time

def test_pooling_bug():
    """VLLM pooling bug'ını test et"""
    
    # Test metinleri
    test_texts = [
        "query: What is machine learning?",
        "query: How does artificial intelligence work?",
        "query: Deep learning algorithms",
        "passage: Machine learning is a subset of AI",
        "passage: Neural networks process data"
    ]
    
    model_name = "intfloat/multilingual-e5-large"
    
    print("🔍 Testing VLLM vs Sentence-Transformers Pooling")
    print("=" * 60)
    
    # 1. Sentence-Transformers (Ground Truth)
    print("\n1️⃣ Loading Sentence-Transformers...")
    st_model = SentenceTransformer(model_name)
    
    start_time = time.time()
    st_embeddings = st_model.encode(test_texts, normalize_embeddings=True)
    st_time = time.time() - start_time
    
    print(f"✅ ST Embeddings: {st_embeddings.shape}")
    print(f"⏱️ ST Time: {st_time:.2f}s")
    
    # 2. VLLM
    print("\n2️⃣ Loading VLLM...")
    vllm_model = LLM(model=model_name, task="embed", enforce_eager=True)
    
    start_time = time.time()
    vllm_outputs = vllm_model.embed(test_texts)
    vllm_embeddings = np.array([output.outputs.embedding for output in vllm_outputs])
    # Normalize embeddings
    vllm_embeddings = vllm_embeddings / np.linalg.norm(vllm_embeddings, axis=1, keepdims=True)
    vllm_time = time.time() - start_time
    
    print(f"✅ VLLM Embeddings: {vllm_embeddings.shape}")
    print(f"⏱️ VLLM Time: {vllm_time:.2f}s")
    
    # 3. Karşılaştırma
    print("\n3️⃣ Comparison Results:")
    print("=" * 60)
    
    # Cosine similarity hesapla
    similarities = []
    for i in range(len(test_texts)):
        sim = cosine_similarity(
            st_embeddings[i].reshape(1, -1),
            vllm_embeddings[i].reshape(1, -1)
        )[0][0]
        similarities.append(sim)
        print(f"Text {i+1}: {sim:.4f}")
    
    avg_similarity = np.mean(similarities)
    print(f"\n📊 Average Similarity: {avg_similarity:.4f}")
    
    # 4. Bug değerlendirmesi
    print("\n4️⃣ Bug Assessment:")
    print("=" * 60)
    
    if avg_similarity > 0.95:
        print("🟢 EXCELLENT: No pooling bug detected!")
        print("   Embeddings are nearly identical")
    elif avg_similarity > 0.85:
        print("🟡 GOOD: Minor differences (probably OK)")
        print("   Small variations, likely due to implementation")
    elif avg_similarity > 0.70:
        print("🟠 WARNING: Significant differences")
        print("   Possible pooling strategy mismatch")
    else:
        print("🔴 CRITICAL: Major pooling bug detected!")
        print("   VLLM embeddings are very different from ST")
    
    # 5. Detaylı analiz
    print("\n5️⃣ Detailed Analysis:")
    print("=" * 60)
    
    # Embedding statistics
    st_mean = np.mean(st_embeddings, axis=0)
    vllm_mean = np.mean(vllm_embeddings, axis=0)
    
    print(f"ST Mean Embedding Norm: {np.linalg.norm(st_mean):.4f}")
    print(f"VLLM Mean Embedding Norm: {np.linalg.norm(vllm_mean):.4f}")
    
    # Variance analizi
    st_var = np.var(st_embeddings, axis=0).mean()
    vllm_var = np.var(vllm_embeddings, axis=0).mean()
    
    print(f"ST Embedding Variance: {st_var:.6f}")
    print(f"VLLM Embedding Variance: {vllm_var:.6f}")
    
    # Performance karşılaştırması
    print(f"\n⚡ Performance:")
    print(f"ST: {len(test_texts)/st_time:.1f} embeddings/sec")
    print(f"VLLM: {len(test_texts)/vllm_time:.1f} embeddings/sec")
    
    return avg_similarity > 0.85

def test_semantic_search():
    """Semantic search kalitesini test et"""
    
    print("\n🔍 Semantic Search Quality Test")
    print("=" * 60)
    
    queries = ["query: What is AI?", "query: Machine learning basics"]
    passages = [
        "passage: Artificial intelligence (AI) is computer science",
        "passage: Machine learning is a subset of AI",
        "passage: Deep learning uses neural networks",
        "passage: Today's weather is sunny"
    ]
    
    model_name = "intfloat/multilingual-e5-large"
    
    # ST
    st_model = SentenceTransformer(model_name)
    st_query_emb = st_model.encode(queries, normalize_embeddings=True)
    st_passage_emb = st_model.encode(passages, normalize_embeddings=True)
    st_scores = np.dot(st_query_emb, st_passage_emb.T)
    
    # VLLM
    vllm_model = LLM(model=model_name, task="embed", enforce_eager=True)
    vllm_query_outputs = vllm_model.embed(queries)
    vllm_passage_outputs = vllm_model.embed(passages)
    
    vllm_query_emb = np.array([out.outputs.embedding for out in vllm_query_outputs])
    vllm_passage_emb = np.array([out.outputs.embedding for out in vllm_passage_outputs])
    
    # Normalize
    vllm_query_emb = vllm_query_emb / np.linalg.norm(vllm_query_emb, axis=1, keepdims=True)
    vllm_passage_emb = vllm_passage_emb / np.linalg.norm(vllm_passage_emb, axis=1, keepdims=True)
    
    vllm_scores = np.dot(vllm_query_emb, vllm_passage_emb.T)
    
    print("ST Scores:")
    print(st_scores)
    print("\nVLLM Scores:")
    print(vllm_scores)
    
    # Ranking karşılaştırması
    st_rankings = np.argsort(-st_scores, axis=1)
    vllm_rankings = np.argsort(-vllm_scores, axis=1)
    
    print(f"\nST Rankings: {st_rankings}")
    print(f"VLLM Rankings: {vllm_rankings}")
    
    # Ranking similarity
    ranking_match = np.mean(st_rankings == vllm_rankings)
    print(f"\nRanking Match: {ranking_match:.2f}")
    
    return ranking_match > 0.7

if __name__ == "__main__":
    print("🧪 VLLM Pooling Bug Test Suite")
    print("Testing intfloat/multilingual-e5-large")
    print("=" * 60)
    
    try:
        # Test 1: Embedding similarity
        embedding_ok = test_pooling_bug()
        
        # Test 2: Semantic search quality
        search_ok = test_semantic_search()
        
        # Final verdict
        print("\n🎯 FINAL VERDICT:")
        print("=" * 60)
        
        if embedding_ok and search_ok:
            print("✅ PASS: No significant pooling bug detected")
            print("   Safe to use VLLM for production")
        else:
            print("❌ FAIL: Pooling bug detected")
            print("   Consider using sentence-transformers instead")
            print("   Or wait for VLLM fix")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("Check your VLLM installation and model availability")