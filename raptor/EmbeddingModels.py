"""
Embedding Models - Minimal version aligned with VLLM Native Server
"""
import logging
import asyncio
import aiohttp
from abc import ABC, abstractmethod
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from sentence_transformers import SentenceTransformer

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        pass
    
    async def create_embedding_async(self, text):
        """Async version - default implementation runs sync in thread pool"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self.create_embedding, text)
    
    async def create_embeddings_batch(self, texts: List[str]):
        """Batch embedding creation - default implementation"""
        tasks = [self.create_embedding_async(text) for text in texts]
        return await asyncio.gather(*tasks)


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="text-embedding-ada-002"):
        self.client = OpenAI()
        self.model = model

    def create_embedding(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )


class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text).tolist()


class VLLMEmbeddingModel(BaseEmbeddingModel):
    """VLLM Native Embedding Model - Minimal version"""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8008",
        model_name: str = "intfloat/multilingual-e5-large",
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.session = None
        self._closed = False
        
    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self._closed:
            raise RuntimeError("VLLMEmbeddingModel has been closed")
            
        if self.session is None or self.session.closed:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
    
    async def close(self):
        """Close the session"""
        if self.session and not self.session.closed:
            await self.session.close()
        self._closed = True
    
    def create_embedding(self, text: str):
        """Sync version - for backward compatibility"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context
                raise RuntimeError(
                    "Cannot call sync method from async context. "
                    "Use create_embedding_async instead."
                )
            return loop.run_until_complete(self.create_embedding_async(text))
        except RuntimeError:
            # No event loop
            return asyncio.run(self.create_embedding_async(text))
    
    async def create_embedding_async(self, text: str):
        """Create single embedding"""
        await self._ensure_session()
        
        payload = {
            "input": [text],
            "model": self.model_name
        }
        
        async with self.session.post(
            f"{self.base_url}/v1/embeddings",
            json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result['data'][0]['embedding']
            else:
                error_text = await response.text()
                raise Exception(f"VLLM error {response.status}: {error_text}")
    
    async def create_embeddings_batch(self, texts: List[str]):
        """
        Batch embeddings - VLLM handles batching internally
        Just send all texts in one request
        """
        await self._ensure_session()
        
        payload = {
            "input": texts,  # VLLM handles internal batching
            "model": self.model_name
        }
        
        async with self.session.post(
            f"{self.base_url}/v1/embeddings",
            json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                return [item['embedding'] for item in result['data']]
            else:
                error_text = await response.text()
                raise Exception(f"VLLM error {response.status}: {error_text}")