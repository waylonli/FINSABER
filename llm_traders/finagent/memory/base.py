import abc
from typing import (
    Any,
    Iterable,
    List,
    Dict,
    Union,
    Tuple,
    Optional,
)

Image = Any

class VectorStore(abc.ABC):
    """Interface for vector store."""

    @abc.abstractmethod
    def add_embeddings(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Add embeddings to the vectorstore.

        Args:
            keys: list of metadatas associated with the embedding.
            embeddings: Iterable of embeddings to add to the vectorstore.
            kwargs: vectorstore specific parameters
        """

    @abc.abstractmethod
    def delete(self,
               *args: Any,
               **kwargs: Any,) -> bool:
        """Delete by keys.

        Args:
            keys: List of keys to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            bool: True if deletion is successful,
            False otherwise, None if not implemented.
        """

    @abc.abstractmethod
    def similarity_search(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> List[Tuple[str, float]]:
        """Return keys most similar to query."""

    def save_local(self, memory_path = None) -> None:
        """Save FAISS index and index_to_key to disk."""


class BaseMemory:
    """Base class for all memories."""

    @abc.abstractmethod
    def add(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Add data to memory.

        Args:
            **kwargs: Other keyword arguments that subclasses might use.
        """
        pass

    @abc.abstractmethod
    def similarity_search(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> List[Union[str, Image]]:
        """Retrieve the keys from the vectorstores.

        Args:
            data: the query data.
            top_k: the number of results to return.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            the corresponding values from the memory.
        """

    @abc.abstractmethod
    def query(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> List[Union[str, Image]]:
        """Retrieve the keys from the vectorstores.

        Args:
            data: the query data.
            top_k: the number of results to return.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            the corresponding values from the memory.
        """
