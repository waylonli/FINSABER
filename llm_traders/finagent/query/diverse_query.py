from llm_traders.finagent.memory import MemoryInterface
from llm_traders.finagent.provider import EmbeddingProvider
from llm_traders.finagent.query import QUERY_TYPES
from typing import Dict, Any, List
class DiverseQuery():
    def __init__(self,
                 memory: MemoryInterface,
                 provider: EmbeddingProvider,
                 top_k: int = 5):
        self.memory = memory
        self.provider = provider
        self.top_k = top_k

    def query(self,
              params: Dict= None,
              query_types: List[str] = ["plain", "short_term", "long_term"],
              top_k: int = None):

        return self.diverse_query(params, query_types=query_types, top_k=top_k)

    def diverse_query(self,
                      params: Dict,
                      query_types: List[str] = ["plain", "short_term",  "long_term"],
                      top_k: int = None):

        top_k = top_k if top_k is not None else self.top_k

        type = params["type"]
        symbol = params["symbol"]

        res = {}

        for query_type in query_types:

            query_text = QUERY_TYPES[query_type](params)
            embedding = self.provider.embed_query(query_text)
            query_items, _ = self.memory.query_memory(type=type,
                                                      symbol=symbol,
                                                      data={"embedding": embedding},
                                                      embedding_query="embedding",
                                                      top_k=top_k)

            pre_query_items = query_items

            if len(pre_query_items) == 0:
                post_query_items = []
            else:
                post_query_items = pre_query_items

            res[query_type] = {
                "query_text": query_text,
                "query_items": post_query_items
            }

        return res