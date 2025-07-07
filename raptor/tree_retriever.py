import logging
import os
import asyncio
from typing import Dict, List, Set, Union, Tuple
from concurrent.futures import ThreadPoolExecutor

import tiktoken
import numpy as np
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .EmbeddingModels import BaseEmbeddingModel, OpenAIEmbeddingModel
from .Retrievers import BaseRetriever
from .tree_structures import Node, Tree
from .utils import (distances_from_embeddings, get_children, get_embeddings,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances,
                    reverse_mapping)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class TreeRetrieverConfig:
    def __init__(
        self,
        tokenizer=None,
        threshold=None,
        top_k=None,
        selection_mode=None,
        context_embedding_model=None,
        embedding_model=None,
        num_layers=None,
        start_layer=None,
    ):
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = tokenizer

        if threshold is None:
            threshold = 0.5
        if not isinstance(threshold, float) or not (0 <= threshold <= 1):
            raise ValueError("threshold must be a float between 0 and 1")
        self.threshold = threshold

        if top_k is None:
            top_k = 5
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be an integer and at least 1")
        self.top_k = top_k

        if selection_mode is None:
            selection_mode = "top_k"
        if not isinstance(selection_mode, str) or selection_mode not in [
            "top_k",
            "threshold",
        ]:
            raise ValueError(
                "selection_mode must be a string and either 'top_k' or 'threshold'"
            )
        self.selection_mode = selection_mode

        if context_embedding_model is None:
            context_embedding_model = "OpenAI"
        if not isinstance(context_embedding_model, str):
            raise ValueError("context_embedding_model must be a string")
        self.context_embedding_model = context_embedding_model

        if embedding_model is None:
            embedding_model = OpenAIEmbeddingModel()
        if not isinstance(embedding_model, BaseEmbeddingModel):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel"
            )
        self.embedding_model = embedding_model

        if num_layers is not None:
            if not isinstance(num_layers, int) or num_layers < 0:
                raise ValueError("num_layers must be an integer and at least 0")
        self.num_layers = num_layers

        if start_layer is not None:
            if not isinstance(start_layer, int) or start_layer < 0:
                raise ValueError("start_layer must be an integer and at least 0")
        self.start_layer = start_layer

    def log_config(self):
        config_log = """
        TreeRetrieverConfig:
            Tokenizer: {tokenizer}
            Threshold: {threshold}
            Top K: {top_k}
            Selection Mode: {selection_mode}
            Context Embedding Model: {context_embedding_model}
            Embedding Model: {embedding_model}
            Num Layers: {num_layers}
            Start Layer: {start_layer}
        """.format(
            tokenizer=self.tokenizer,
            threshold=self.threshold,
            top_k=self.top_k,
            selection_mode=self.selection_mode,
            context_embedding_model=self.context_embedding_model,
            embedding_model=self.embedding_model,
            num_layers=self.num_layers,
            start_layer=self.start_layer,
        )
        return config_log


class TreeRetriever(BaseRetriever):

    def __init__(self, config, tree) -> None:
        if not isinstance(tree, Tree):
            raise ValueError("tree must be an instance of Tree")

        if config.num_layers is not None and config.num_layers > tree.num_layers + 1:
            raise ValueError(
                "num_layers in config must be less than or equal to tree.num_layers + 1"
            )

        if config.start_layer is not None and config.start_layer > tree.num_layers:
            raise ValueError(
                "start_layer in config must be less than or equal to tree.num_layers"
            )

        self.tree = tree
        self.num_layers = (
            config.num_layers if config.num_layers is not None else tree.num_layers + 1
        )
        self.start_layer = (
            config.start_layer if config.start_layer is not None else tree.num_layers
        )

        if self.num_layers > self.start_layer + 1:
            raise ValueError("num_layers must be less than or equal to start_layer + 1")

        self.tokenizer = config.tokenizer
        self.top_k = config.top_k
        self.threshold = config.threshold
        self.selection_mode = config.selection_mode
        self.embedding_model = config.embedding_model
        self.context_embedding_model = config.context_embedding_model

        self.tree_node_index_to_layer = reverse_mapping(self.tree.layer_to_nodes)

        # Check if embedding model supports async
        self.supports_async = hasattr(self.embedding_model, 'create_embedding_async')
        self.supports_batch = hasattr(self.embedding_model, 'create_embeddings_batch')

        logging.info(
            f"Successfully initialized TreeRetriever with Config {config.log_config()}"
        )
        logging.info(f"Async support: {self.supports_async}, Batch support: {self.supports_batch}")

    def create_embedding(self, text: str) -> List[float]:
        """
        Generates embeddings for the given text using the specified embedding model.

        Args:
            text (str): The text for which to generate embeddings.

        Returns:
            List[float]: The generated embeddings.
        """
        return self.embedding_model.create_embedding(text)
    
    async def create_embedding_async(self, text: str) -> List[float]:
        """Async version of create_embedding"""
        if self.supports_async:
            return await self.embedding_model.create_embedding_async(text)
        else:
            # Fallback to sync version in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                return await loop.run_in_executor(executor, self.create_embedding, text)
    
    async def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding creation"""
        if self.supports_batch:
            embeddings = await self.embedding_model.create_embeddings_batch(texts)
            # Ensure all embeddings are lists (VLLMEmbeddingModel returns List[List[float]])
            return [emb if isinstance(emb, list) else emb.tolist() for emb in embeddings]
        else:
            # Fallback to individual async calls
            tasks = [self.create_embedding_async(text) for text in texts]
            return await asyncio.gather(*tasks)

    def retrieve_information_collapse_tree(self, query: str, top_k: int, max_tokens: int) -> str:
        """
        Retrieves the most relevant information from the tree based on the query.

        Args:
            query (str): The query text.
            max_tokens (int): The maximum number of tokens.

        Returns:
            str: The context created using the most relevant nodes.
        """

        query_embedding = self.create_embedding(query)

        selected_nodes = []

        node_list = get_node_list(self.tree.all_nodes)

        embeddings = get_embeddings(node_list, self.context_embedding_model)

        distances = distances_from_embeddings(query_embedding, embeddings)

        indices = indices_of_nearest_neighbors_from_distances(distances)

        total_tokens = 0
        for idx in indices[:top_k]:

            node = node_list[idx]
            node_tokens = len(self.tokenizer.encode(node.text))

            if total_tokens + node_tokens > max_tokens:
                break

            selected_nodes.append(node)
            total_tokens += node_tokens

        context = get_text(selected_nodes)
        return selected_nodes, context

    async def retrieve_information_collapse_tree_async(
        self, query: str, top_k: int, max_tokens: int
    ) -> Tuple[List[Node], str]:
        """
        Async version of retrieve_information_collapse_tree.
        
        Args:
            query (str): The query text.
            top_k (int): Number of top results to return.
            max_tokens (int): The maximum number of tokens.

        Returns:
            Tuple[List[Node], str]: Selected nodes and context.
        """

        query_embedding = await self.create_embedding_async(query)

        selected_nodes = []
        node_list = get_node_list(self.tree.all_nodes)
        embeddings = get_embeddings(node_list, self.context_embedding_model)

        distances = distances_from_embeddings(query_embedding, embeddings)
        indices = indices_of_nearest_neighbors_from_distances(distances)

        total_tokens = 0
        for idx in indices[:top_k]:
            node = node_list[idx]
            node_tokens = len(self.tokenizer.encode(node.text))

            if total_tokens + node_tokens > max_tokens:
                break

            selected_nodes.append(node)
            total_tokens += node_tokens

        context = get_text(selected_nodes)
        return selected_nodes, context

    def retrieve_information(
        self, current_nodes: List[Node], query: str, num_layers: int
    ) -> str:
        """
        Retrieves the most relevant information from the tree based on the query.

        Args:
            current_nodes (List[Node]): A List of the current nodes.
            query (str): The query text.
            num_layers (int): The number of layers to traverse.

        Returns:
            str: The context created using the most relevant nodes.
        """

        query_embedding = self.create_embedding(query)

        selected_nodes = []

        node_list = current_nodes

        for layer in range(num_layers):

            embeddings = get_embeddings(node_list, self.context_embedding_model)

            distances = distances_from_embeddings(query_embedding, embeddings)

            indices = indices_of_nearest_neighbors_from_distances(distances)

            if self.selection_mode == "threshold":
                best_indices = [
                    index for index in indices if distances[index] > self.threshold
                ]

            elif self.selection_mode == "top_k":
                best_indices = indices[: self.top_k]

            nodes_to_add = [node_list[idx] for idx in best_indices]

            selected_nodes.extend(nodes_to_add)

            if layer != num_layers - 1:

                child_nodes = []

                for index in best_indices:
                    child_nodes.extend(node_list[index].children)

                # take the unique values
                child_nodes = list(dict.fromkeys(child_nodes))
                node_list = [self.tree.all_nodes[i] for i in child_nodes]

        context = get_text(selected_nodes)
        return selected_nodes, context

    async def retrieve_information_async(
        self, current_nodes: List[Node], query: str, num_layers: int
    ) -> Tuple[List[Node], str]:
        """
        Async version of retrieve_information.

        Args:
            current_nodes (List[Node]): A List of the current nodes.
            query (str): The query text.
            num_layers (int): The number of layers to traverse.

        Returns:
            Tuple[List[Node], str]: Selected nodes and context.
        """

        query_embedding = await self.create_embedding_async(query)
        selected_nodes = []
        node_list = current_nodes

        for layer in range(num_layers):
            embeddings = get_embeddings(node_list, self.context_embedding_model)
            distances = distances_from_embeddings(query_embedding, embeddings)
            indices = indices_of_nearest_neighbors_from_distances(distances)

            if self.selection_mode == "threshold":
                best_indices = [
                    index for index in indices if distances[index] > self.threshold
                ]
            elif self.selection_mode == "top_k":
                best_indices = indices[: self.top_k]

            nodes_to_add = [node_list[idx] for idx in best_indices]
            selected_nodes.extend(nodes_to_add)

            if layer != num_layers - 1:
                child_nodes = []
                for index in best_indices:
                    child_nodes.extend(node_list[index].children)

                # take the unique values
                child_nodes = list(dict.fromkeys(child_nodes))
                node_list = [self.tree.all_nodes[i] for i in child_nodes]

        context = get_text(selected_nodes)
        return selected_nodes, context

    async def retrieve_async(
        self,
        query: str,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10, 
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = False,
    ) -> Union[str, Tuple[str, List[Dict]]]:
        """
        Async version of retrieve method.
        
        Args:
            query (str): The query text.
            start_layer (int): The layer to start from. Defaults to self.start_layer.
            num_layers (int): The number of layers to traverse. Defaults to self.num_layers.
            max_tokens (int): The maximum number of tokens. Defaults to 3500.
            collapse_tree (bool): Whether to retrieve information from all nodes. Defaults to True.
            return_layer_information (bool): Whether to return layer information. Defaults to False.

        Returns:
            str or Tuple[str, List[Dict]]: The result of the query.
        """
        
        if not isinstance(query, str):
            raise ValueError("query must be a string")

        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be an integer and at least 1")

        if not isinstance(collapse_tree, bool):
            raise ValueError("collapse_tree must be a boolean")

        # Set defaults
        start_layer = self.start_layer if start_layer is None else start_layer
        num_layers = self.num_layers if num_layers is None else num_layers

        if not isinstance(start_layer, int) or not (
            0 <= start_layer <= self.tree.num_layers
        ):
            raise ValueError(
                "start_layer must be an integer between 0 and tree.num_layers"
            )

        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be an integer and at least 1")

        if num_layers > (start_layer + 1):
            raise ValueError("num_layers must be less than or equal to start_layer + 1")

        if collapse_tree:
            logging.info(f"Using collapsed_tree")
            selected_nodes, context = await self.retrieve_information_collapse_tree_async(
                query, top_k, max_tokens
            )
        else:
            layer_nodes = self.tree.layer_to_nodes[start_layer]
            selected_nodes, context = await self.retrieve_information_async(
                layer_nodes, query, num_layers
            )

        if return_layer_information:
            layer_information = []
            for node in selected_nodes:
                layer_information.append(
                    {
                        "node_index": node.index,
                        "layer_number": self.tree_node_index_to_layer[node.index],
                    }
                )
            return context, layer_information

        return context
    
    async def retrieve_batch(
        self, 
        queries: List[str],
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10, 
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = False,
    ) -> List[Union[str, Tuple[str, List[Dict]]]]:
        """
        Batch retrieve method for multiple queries.
        
        Args:
            queries (List[str]): List of query texts.
            Other args: Same as retrieve_async
            
        Returns:
            List of retrieve results
        """
        if not queries:
            return []
            
        if len(queries) == 1:
            # Single query - use regular async method
            result = await self.retrieve_async(
                queries[0], start_layer, num_layers, top_k, 
                max_tokens, collapse_tree, return_layer_information
            )
            return [result]
        
        # For multiple queries, we can optimize by batching embeddings
        if collapse_tree:
            return await self._retrieve_batch_collapsed(
                queries, top_k, max_tokens, return_layer_information
            )
        else:
            # For non-collapsed, process individually (can be optimized further)
            tasks = [
                self.retrieve_async(
                    query, start_layer, num_layers, top_k,
                    max_tokens, collapse_tree, return_layer_information
                )
                for query in queries
            ]
            return await asyncio.gather(*tasks)
    
    async def _retrieve_batch_collapsed(
        self, 
        queries: List[str], 
        top_k: int, 
        max_tokens: int, 
        return_layer_information: bool
    ) -> List[Union[str, Tuple[str, List[Dict]]]]:
        """
        Optimized batch retrieval for collapsed tree mode.
        Uses batch embedding generation for better performance.
        """
        
        # Batch generate embeddings for all queries
        query_embeddings = await self.create_embeddings_batch(queries)
        
        # Pre-compute node embeddings and list (same for all queries)
        node_list = get_node_list(self.tree.all_nodes)
        node_embeddings = get_embeddings(node_list, self.context_embedding_model)
        
        results = []
        
        for query_embedding in query_embeddings:
            # Compute distances for this query
            distances = distances_from_embeddings(query_embedding, node_embeddings)
            indices = indices_of_nearest_neighbors_from_distances(distances)
            
            # Select nodes based on top_k and max_tokens
            selected_nodes = []
            total_tokens = 0
            
            for idx in indices[:top_k]:
                node = node_list[idx]
                node_tokens = len(self.tokenizer.encode(node.text))

                if total_tokens + node_tokens > max_tokens:
                    break

                selected_nodes.append(node)
                total_tokens += node_tokens

            context = get_text(selected_nodes)
            
            if return_layer_information:
                layer_information = []
                for node in selected_nodes:
                    layer_information.append({
                        "node_index": node.index,
                        "layer_number": self.tree_node_index_to_layer[node.index],
                    })
                results.append((context, layer_information))
            else:
                results.append(context)
        
        return results

    def retrieve(
        self,
        query: str,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10, 
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = False,
    ) -> Union[str, Tuple[str, List[Dict]]]:
        """
        Synchronous retrieve method (backwards compatibility).
        Internally uses async method with asyncio.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.retrieve_async(
                query, start_layer, num_layers, top_k,
                max_tokens, collapse_tree, return_layer_information
            )
        )