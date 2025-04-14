from typing import (
    List,
    Dict,
    Union,
    Optional,
    Tuple,
    Any,
)

import time
import json
import os

from llm_traders.finagent.memory.base import VectorStore, BaseMemory, Image


class BasicMemory(BaseMemory):
    def __init__(
            self,
            memory_path: str,
            vectorstore: VectorStore,
            memory: Optional[Dict] = None,
    ) -> None:
        if memory is None:
            self.memory = {}
        else:
            self.memory = memory
        self.memory_path = memory_path
        self.vectorstore = vectorstore

    def add(
            self,
            data: Dict,
            embedding_key: str,
            **kwargs,
    ) -> None:
        """
        Add data to memory.
        """
        name = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())  # the unique id of the added unit.
        self.memory[name] = data

        assert embedding_key in data, f"embedding_key {embedding_key} not in data"
        embeddings = data[embedding_key]

        self.vectorstore.add_embeddings([name], [embeddings])

    def similarity_search(
            self,
            data: Dict,
            embedding_query: str,
            top_k: int = 3,
            **kwargs) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Retrieve the keys from the vectorstores.
        """
        assert embedding_query in data, f"embedding_query {embedding_query} not in data"

        query_embedding = data[embedding_query]

        try:
            key_and_score = self.vectorstore.similarity_search(query_embedding, top_k)
            items = [self.memory[k] for k, score in key_and_score]
            scores = [score for k, score in key_and_score]
        except:
            items = []
            scores = []

        return items, scores

    def query(self,
              data: Dict,
              embedding_query: str,
              top_k: int = 3,
              **kwargs) -> Tuple[List[Dict[str, Any]], List[float]]:
        items, scores = self.similarity_search(data, embedding_query, top_k=top_k, **kwargs)
        return items, scores

    def load_local(
            self,
            memory_path: str = None,
            vectorstore: VectorStore = None,
    ):

        if memory_path is None:
            memory_path = self.memory_path

        """Load the memory from the local file."""
        with open(os.path.join(memory_path, "memory.json"), "r") as rf:
            memory = json.load(rf)

        self.memory_path = memory_path
        self.vectorstore = vectorstore
        self.memory = memory

    def save_local(self, memory_path = None) -> None:

        if memory_path is None:
            memory_path = self.memory_path

        """Save the memory to the local file."""
        with open(os.path.join(memory_path, "memory.json"), "w") as f:
            json.dump(self.memory, f, indent=2)
        self.vectorstore.save_local(memory_path)
