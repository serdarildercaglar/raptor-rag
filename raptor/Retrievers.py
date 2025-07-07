import asyncio
from abc import ABC, abstractmethod
from typing import List, Union
from concurrent.futures import ThreadPoolExecutor


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str) -> str:
        """Synchronous retrieve method"""
        pass
    
    async def retrieve_async(self, query: str) -> str:
        """Async retrieve method - default implementation runs sync in thread pool"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self.retrieve, query)
    
    async def retrieve_batch(self, queries: List[str]) -> List[str]:
        """Batch retrieve method - default implementation"""
        tasks = [self.retrieve_async(query) for query in queries]
        return await asyncio.gather(*tasks)