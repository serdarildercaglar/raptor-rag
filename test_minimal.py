#!/usr/bin/env python3
"""
Minimal RAPTOR Retrieve Test - Proper session management
"""
import asyncio
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig, VLLMEmbeddingModel

async def main():
    # 1. VLLMEmbeddingModel oluştur
    embedding_model = VLLMEmbeddingModel("http://localhost:8008")
    
    try:
        # 2. Config oluştur  
        config = RetrievalAugmentationConfig(embedding_model=embedding_model)

        # 3. RAPTOR tree'yi yükle
        tree_path = "/home/serdar/Documents/raptor-rag/vectordb/raptor-db"
        RA = RetrievalAugmentation(config=config, tree=tree_path)

        # 4. Query gönder (async version kullan)
        question = "edebiyat nedir?"
        context = await RA.retrieve_async(question)

        # 5. Sonucu göster
        print(f"Question: {question}")
        print(f"Context ({len(context)} chars):")
        print("=" * 60)
        print(context[:200])
        print("=" * 60)
        
        # 6. Batch test de yapalım
        questions = [
            "edebiyat nedir?",
            "şiir nedir?", 
            "roman nedir?"
        ]
        
        print("\n🚀 Batch Test:")
        contexts = await RA.retrieve_batch(questions)
        
        for i, (q, ctx) in enumerate(zip(questions, contexts)):
            print(f"Q{i+1}: {q} -> {len(ctx)} chars")
            
    finally:
        # 7. Session'ı temizle
        await embedding_model.close()
        print("\n✅ Session cleaned up properly")

# Sync wrapper
def run_sync():
    """Sync version for backward compatibility"""
    embedding_model = VLLMEmbeddingModel("http://localhost:8008")
    
    # Config oluştur  
    config = RetrievalAugmentationConfig(embedding_model=embedding_model)

    # RAPTOR tree'yi yükle
    tree_path = "/home/serdar/Documents/raptor-rag/vectordb/raptor-db"
    RA = RetrievalAugmentation(config=config, tree=tree_path)

    # Query gönder (sync version)
    question = "edebiyat nedir?"
    context = RA.retrieve(question)

    # Sonucu göster
    print(f"Question: {question}")
    print(f"Context ({len(context)} chars):")
    print("=" * 60)
    print(context[:200])
    print("=" * 60)
    
    print("\n✅ Sync test completed (session cleanup will happen automatically)")

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Async test (recommended, proper cleanup)")
    print("2. Sync test (simple, may have cleanup warnings)")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "1":
        asyncio.run(main())
    else:
        run_sync()