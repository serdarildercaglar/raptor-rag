import logging
import asyncio
import aiohttp
from abc import ABC, abstractmethod
from typing import List
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential

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

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
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
        return self.model.encode(text)


class VLLMEmbeddingModel(BaseEmbeddingModel):
    """VLLM Embedding Model that connects to VLLM service via HTTP"""
    
    def __init__(self, base_url: str = "http://localhost:8008", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = None
        self._closed = False
        
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self._closed:
            raise RuntimeError("VLLMEmbeddingModel has been closed")
            
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def _close_session(self):
        """Close aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
        self._closed = True
    
    async def close(self):
        """Explicitly close the session"""
        await self._close_session()
    
    def create_embedding(self, text: str):
        """Sync version - runs async version in event loop"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.create_embedding_async(text))
    
    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
    async def create_embedding_async(self, text: str):
        """Async embedding creation via VLLM service"""
        session = await self._get_session()
        
        payload = {
            "input": [text],
            "model": "intfloat/multilingual-e5-large"
        }
        
        try:
            async with session.post(
                f"{self.base_url}/v1/embeddings",
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['data'][0]['embedding']
                else:
                    error_text = await response.text()
                    raise Exception(f"VLLM service error {response.status}: {error_text}")
                    
        except aiohttp.ClientError as e:
            raise Exception(f"Connection error to VLLM service: {str(e)}")
    
    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
    async def create_embeddings_batch(self, texts: List[str]):
        """Optimized batch embedding via VLLM service"""
        session = await self._get_session()
        
        # Use batch endpoint for better performance
        try:
            async with session.post(
                f"{self.base_url}/embeddings/batch",
                json=texts
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return [item['embedding'] for item in result['data']]
                else:
                    error_text = await response.text()
                    raise Exception(f"VLLM batch service error {response.status}: {error_text}")
                    
        except aiohttp.ClientError as e:
            raise Exception(f"Connection error to VLLM batch service: {str(e)}")