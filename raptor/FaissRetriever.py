import random
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Union

import faiss
import numpy as np
import tiktoken
from tqdm import tqdm

from .EmbeddingModels import BaseEmbeddingModel, OpenAIEmbeddingModel, VLLMEmbeddingModel
from .Retrievers import BaseRetriever
from .utils import split_text


class FaissRetrieverConfig:
    def __init__(
        self,
        max_tokens=100,
        max_context_tokens=3500,
        use_top_k=False,
        embedding_model=None,
        question_embedding_model=None,
        top_k=5,
        tokenizer=tiktoken.get_encoding("o200k_base"),
        embedding_model_string=None,
        use_gpu=True,  # New parameter for GPU usage
    ):
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if max_context_tokens is not None and max_context_tokens < 1:
            raise ValueError("max_context_tokens must be at least 1 or None")

        if embedding_model is not None and not isinstance(
            embedding_model, BaseEmbeddingModel
        ):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel or None"
            )

        if question_embedding_model is not None and not isinstance(
            question_embedding_model, BaseEmbeddingModel
        ):
            raise ValueError(
                "question_embedding_model must be an instance of BaseEmbeddingModel or None"
            )

        self.top_k = top_k
        self.max_tokens = max_tokens
        self.max_context_tokens = max_context_tokens
        self.use_top_k = use_top_k
        self.embedding_model = embedding_model or OpenAIEmbeddingModel()
        self.question_embedding_model = question_embedding_model or self.embedding_model
        self.tokenizer = tokenizer
        self.embedding_model_string = embedding_model_string or "OpenAI"
        self.use_gpu = use_gpu

    def log_config(self):
        config_summary = """
		FaissRetrieverConfig:
			Max Tokens: {max_tokens}
			Max Context Tokens: {max_context_tokens}
			Use Top K: {use_top_k}
			Embedding Model: {embedding_model}
			Question Embedding Model: {question_embedding_model}
			Top K: {top_k}
			Tokenizer: {tokenizer}
			Embedding Model String: {embedding_model_string}
			Use GPU: {use_gpu}
		""".format(
            max_tokens=self.max_tokens,
            max_context_tokens=self.max_context_tokens,
            use_top_k=self.use_top_k,
            embedding_model=self.embedding_model,
            question_embedding_model=self.question_embedding_model,
            top_k=self.top_k,
            tokenizer=self.tokenizer,
            embedding_model_string=self.embedding_model_string,
            use_gpu=self.use_gpu,
        )
        return config_summary


class FaissRetriever(BaseRetriever):
    """
    FaissRetriever is a class that retrieves similar context chunks for a given query using Faiss.
    Optimized for async operations and GPU acceleration.
    """

    def __init__(self, config):
        self.embedding_model = config.embedding_model
        self.question_embedding_model = config.question_embedding_model
        self.index = None
        self.context_chunks = None
        self.max_tokens = config.max_tokens
        self.max_context_tokens = config.max_context_tokens
        self.use_top_k = config.use_top_k
        self.tokenizer = config.tokenizer
        self.top_k = config.top_k
        self.embedding_model_string = config.embedding_model_string
        self.use_gpu = config.use_gpu
        
        # Check if embedding models support async
        self.supports_async = hasattr(self.embedding_model, 'create_embedding_async')
        self.supports_batch = hasattr(self.embedding_model, 'create_embeddings_batch')
        self.question_supports_async = hasattr(self.question_embedding_model, 'create_embedding_async')
        self.question_supports_batch = hasattr(self.question_embedding_model, 'create_embeddings_batch')

    def build_from_text(self, doc_text):
        """
        Builds the index from a given text.

        :param doc_text: A string containing the document text.
        """
        self.context_chunks = np.array(
            split_text(doc_text, self.tokenizer, self.max_tokens)
        )

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self.embedding_model.create_embedding, context_chunk)
                for context_chunk in self.context_chunks
            ]

        self.embeddings = []
        for future in tqdm(futures, total=len(futures), desc="Building embeddings"):
            self.embeddings.append(future.result())

        self.embeddings = np.array(self.embeddings, dtype=np.float32)
        self._build_index()

    async def build_from_text_async(self, doc_text):
        """
        Async version of build_from_text.
        
        :param doc_text: A string containing the document text.
        """
        self.context_chunks = np.array(
            split_text(doc_text, self.tokenizer, self.max_tokens)
        )

        if self.supports_batch:
            # Use batch embedding for better performance
            self.embeddings = await self.embedding_model.create_embeddings_batch(
                self.context_chunks.tolist()
            )
        elif self.supports_async:
            # Use individual async calls
            tasks = [
                self.embedding_model.create_embedding_async(chunk)
                for chunk in self.context_chunks
            ]
            self.embeddings = await asyncio.gather(*tasks)
        else:
            # Fallback to sync version in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                tasks = [
                    loop.run_in_executor(
                        executor, self.embedding_model.create_embedding, chunk
                    )
                    for chunk in self.context_chunks
                ]
                self.embeddings = await asyncio.gather(*tasks)

        self.embeddings = np.array(self.embeddings, dtype=np.float32)
        self._build_index()

    def build_from_leaf_nodes(self, leaf_nodes):
        """
        Builds the index from leaf nodes.

        :param leaf_nodes: List of leaf nodes with embeddings.
        """
        self.context_chunks = [node.text for node in leaf_nodes]

        self.embeddings = np.array(
            [node.embeddings[self.embedding_model_string] for node in leaf_nodes],
            dtype=np.float32,
        )
        self._build_index()

    def _build_index(self):
        """Build Faiss index from embeddings"""
        try:
            if self.use_gpu and faiss.get_num_gpus() > 0:
                # Use GPU index for better performance
                cpu_index = faiss.IndexFlatIP(self.embeddings.shape[1])
                self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index)
            else:
                # Use CPU index
                self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        except Exception as e:
            # Fallback to CPU if GPU fails
            print(f"GPU index creation failed, falling back to CPU: {e}")
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        
        self.index.add(self.embeddings)

    def sanity_check(self, num_samples=4):
        """
        Perform a sanity check by recomputing embeddings of a few randomly-selected chunks.

        :param num_samples: The number of samples to test.
        """
        indices = random.sample(range(len(self.context_chunks)), num_samples)

        for i in indices:
            original_embedding = self.embeddings[i]
            recomputed_embedding = self.embedding_model.create_embedding(
                self.context_chunks[i]
            )
            assert np.allclose(
                original_embedding, recomputed_embedding
            ), f"Embeddings do not match for index {i}!"

        print(f"Sanity check passed for {num_samples} random samples.")

    def retrieve(self, query: str) -> str:
        """
        Synchronous retrieve method (backwards compatibility).
        
        :param query: A string containing the query.
        :return: A string containing the retrieved context chunks.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.retrieve_async(query))

    async def retrieve_async(self, query: str) -> str:
        """
        Async version of retrieve method.
        
        :param query: A string containing the query.
        :return: A string containing the retrieved context chunks.
        """
        if self.question_supports_async:
            query_embedding = await self.question_embedding_model.create_embedding_async(query)
        else:
            # Fallback to sync version in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                query_embedding = await loop.run_in_executor(
                    executor, self.question_embedding_model.create_embedding, query
                )

        query_embedding = np.array([np.array(query_embedding, dtype=np.float32).squeeze()])

        context = ""

        if self.use_top_k:
            _, indices = self.index.search(query_embedding, self.top_k)
            for i in range(self.top_k):
                if i < len(indices[0]):  # Check bounds
                    context += self.context_chunks[indices[0][i]]
        else:
            range_ = int(self.max_context_tokens / self.max_tokens)
            _, indices = self.index.search(query_embedding, range_)
            total_tokens = 0
            for i in range(min(range_, len(indices[0]))):  # Check bounds
                tokens = len(self.tokenizer.encode(self.context_chunks[indices[0][i]]))
                if total_tokens + tokens > self.max_context_tokens:
                    break
                context += self.context_chunks[indices[0][i]]
                total_tokens += tokens

        return context

    async def retrieve_batch(self, queries: List[str]) -> List[str]:
        """
        Batch retrieve method for multiple queries.
        
        :param queries: List of query strings.
        :return: List of retrieved context chunks.
        """
        if not queries:
            return []

        if len(queries) == 1:
            # Single query
            result = await self.retrieve_async(queries[0])
            return [result]

        # Batch generate embeddings for all queries
        if self.question_supports_batch:
            query_embeddings = await self.question_embedding_model.create_embeddings_batch(queries)
        elif self.question_supports_async:
            tasks = [
                self.question_embedding_model.create_embedding_async(query)
                for query in queries
            ]
            query_embeddings = await asyncio.gather(*tasks)
        else:
            # Fallback to sync version in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                tasks = [
                    loop.run_in_executor(
                        executor, self.question_embedding_model.create_embedding, query
                    )
                    for query in queries
                ]
                query_embeddings = await asyncio.gather(*tasks)

        # Convert to numpy array
        query_embeddings = np.array(query_embeddings, dtype=np.float32)

        results = []

        for query_embedding in query_embeddings:
            query_embedding = query_embedding.reshape(1, -1)  # Ensure 2D shape
            context = ""

            if self.use_top_k:
                _, indices = self.index.search(query_embedding, self.top_k)
                for i in range(min(self.top_k, len(indices[0]))):  # Check bounds
                    context += self.context_chunks[indices[0][i]]
            else:
                range_ = int(self.max_context_tokens / self.max_tokens)
                _, indices = self.index.search(query_embedding, range_)
                total_tokens = 0
                for i in range(min(range_, len(indices[0]))):  # Check bounds
                    tokens = len(self.tokenizer.encode(self.context_chunks[indices[0][i]]))
                    if total_tokens + tokens > self.max_context_tokens:
                        break
                    context += self.context_chunks[indices[0][i]]
                    total_tokens += tokens

            results.append(context)

        return results