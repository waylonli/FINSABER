import os
from typing import (
    Any,
    List,
    Dict,
    Union,
    Tuple,
)
from collections import deque

from llm_traders.finagent.memory.base import VectorStore, Image
from llm_traders.finagent.memory.faiss import FAISS
from llm_traders.finagent.memory.basic_memory import BasicMemory
from llm_traders.finagent.registry import MEMORY

@MEMORY.register_module(force=True)
class MemoryInterface():
    def __init__(
        self,
        root,
        symbols: List[str],
        memory_path: str,
        embedding_dim: int,
        max_recent_steps = 5,
        workdir = None,
        tag = None,
    ) -> None:

        self.root = root
        self.symbols = symbols
        self.embedding_dim = embedding_dim
        self.max_recent_steps = max_recent_steps
        self.workdir = workdir
        self.tag = tag
        self.memory_path = os.path.join(self.root, self.workdir, self.tag, memory_path)
        os.makedirs(self.memory_path, exist_ok=True)

        self.market_intelligence_memorys = dict()
        self.low_level_reflection_memorys = dict()
        self.high_level_reflection_memorys = dict()
        self._init_memorys()

        self.market_intelligence_recent_histories = dict()
        self.low_level_reflection_recent_histories = dict()
        self.high_level_reflection_recent_histories = dict()
        self._init_recent_histories()

    def _init_memorys(self):
        for symbol in self.symbols:
            if symbol not in self.market_intelligence_memorys:
                memory_path = os.path.join(self.memory_path, symbol, "market_intelligence")
                os.makedirs(memory_path, exist_ok=True)
                vecstore = FAISS(memory_path = memory_path, embedding_dim = self.embedding_dim)
                self.market_intelligence_memorys[symbol] = BasicMemory(memory_path = memory_path,
                                                         vectorstore = vecstore)
            if symbol not in self.low_level_reflection_memorys:
                memory_path = os.path.join(self.memory_path, symbol, "low_level_reflection")
                os.makedirs(memory_path, exist_ok=True)
                vecstore = FAISS(memory_path = memory_path, embedding_dim=self.embedding_dim)
                self.low_level_reflection_memorys[symbol] = BasicMemory(memory_path = memory_path,
                                                         vectorstore = vecstore)
            if symbol not in self.high_level_reflection_memorys:
                memory_path = os.path.join(self.memory_path, symbol, "high_level_reflection")
                os.makedirs(memory_path, exist_ok=True)
                vecstore = FAISS(memory_path = memory_path, embedding_dim=self.embedding_dim)
                self.high_level_reflection_memorys[symbol] = BasicMemory(memory_path = memory_path,
                                                         vectorstore = vecstore)
    def _init_recent_histories(self):
        for symbol in self.symbols:
            if symbol not in self.market_intelligence_recent_histories:
                self.market_intelligence_recent_histories[symbol] = deque(maxlen=self.max_recent_steps)
            if symbol not in self.low_level_reflection_recent_histories:
                self.low_level_reflection_recent_histories[symbol] = deque(maxlen=self.max_recent_steps)
            if symbol not in self.high_level_reflection_recent_histories:
                self.high_level_reflection_recent_histories[symbol] = deque(maxlen=self.max_recent_steps)

    def get_memory(self, type: str, symbol: str):
        return self._get_memory(type, symbol)

    def _get_memory(self, type: str, symbol: str):

        assert type in ["market_intelligence", "low_level_reflection", "high_level_reflection"],\
            f"type = {type} should be one of ['market_intelligence', 'low_level_reflection', 'high_level_reflection']."

        if type == "market_intelligence":
            return self.market_intelligence_memorys[symbol]
        elif type == "low_level_reflection":
            return self.low_level_reflection_memorys[symbol]
        elif type == "high_level_reflection":
            return self.high_level_reflection_memorys[symbol]

    def _get_recent_history(self, type: str, symbol: str):

        assert type in ["market_intelligence", "low_level_reflection", "high_level_reflection"], \
            f"type = {type} should be one of ['market_intelligence', 'low_level_reflection', 'high_level_reflection']."

        if type == "market_intelligence":
            return self.market_intelligence_recent_histories[symbol]
        elif type == "low_level_reflection":
            return self.low_level_reflection_recent_histories[symbol]
        elif type == "high_level_reflection":
            return self.high_level_reflection_recent_histories[symbol]

    def add_memory(
        self,
        type: str,
        symbol: str,
        data: Dict,
        embedding_key: str,
    ) -> None:
        memory = self._get_memory(type, symbol)
        memory.add(data = data, embedding_key = embedding_key)
        # print(f"Add memory for {type} {symbol}.")

    def query_memory(
        self,
        type: str,
        symbol: str,
        data: Dict,
        embedding_query: str,
        top_k: int = 3)-> Tuple[List[Dict[str, Any]], List[float]]:
        memory = self._get_memory(type, symbol)
        res = memory.query(
            data = data,
            embedding_query = embedding_query,
            top_k = top_k,
        )
        # print(f"Query memory for {type} {symbol}.")
        return res

    def add_recent_history(
        self,
        type: str,
        symbol: str,
        data: Dict,
    ) -> None:
        recent_history = self._get_recent_history(type, symbol)
        recent_history.append(data)
        print(f"Add recent history for {type} {symbol}.")

    def get_recent_history(
        self,
        type: str,
        symbol: str,
        k: int = 1,
    ) -> List[Any]:

        assert k <= self.max_recent_steps, f"k = {k} should be less than max_recent_steps = {self.max_recent_steps}."

        recent_history = []
        recent_history_ = self._get_recent_history(type, symbol)
        for item in recent_history_:
            recent_history.append(item)

        if len(recent_history) < k:
            res = recent_history
        else:
            res = recent_history[-k:]

        print(f"Get recent history for {type} {symbol}.")
        return res

    def load_local(
        self,
        memory_path: str = None,
    ) -> None:

        if memory_path is None:
            memory_path = self.memory_path

        """Load the memory from the local file."""
        for symbol in self.symbols:

            try:
                path = os.path.join(memory_path, symbol, "market_intelligence")
                os.makedirs(path, exist_ok=True)
                vecstore = FAISS(memory_path=path, embedding_dim=self.embedding_dim)
                vecstore.load_local(memory_path=path, embedding_dim=self.embedding_dim)

                # print length of index
                print(f"symbols: {symbol}, memory_path: {path}, vecstore length: {vecstore.index.ntotal}")

                self.market_intelligence_memorys[symbol].load_local(
                    memory_path=path,
                    vectorstore=vecstore,
                )
            except Exception as e:
                print(f"Failed to load market_intelligence_memorys: {e}")

            try:
                path = os.path.join(memory_path, symbol, "low_level_reflection")
                os.makedirs(path, exist_ok=True)
                vecstore = FAISS(memory_path=path, embedding_dim=self.embedding_dim)
                vecstore.load_local(memory_path=path, embedding_dim=self.embedding_dim)

                # print length of index
                print(f"symbols: {symbol}, memory_path: {path}, vecstore length: {vecstore.index.ntotal}")

                self.low_level_reflection_memorys[symbol].load_local(
                    memory_path=path,
                    vectorstore=vecstore,
                )
            except Exception as e:
                print(f"Failed to load low_level_reflection_memorys: {e}")

            try:
                path = os.path.join(memory_path, symbol, "high_level_reflection")
                os.makedirs(path, exist_ok=True)
                vecstore = FAISS(memory_path=path, embedding_dim=self.embedding_dim)
                vecstore.load_local(memory_path=path, embedding_dim=self.embedding_dim)

                # print length of index
                print(f"symbols: {symbol}, memory_path: {path}, vecstore length: {vecstore.index.ntotal}")

                self.high_level_reflection_memorys[symbol].load_local(
                    memory_path=path,
                    vectorstore=vecstore,
                )
            except Exception as e:
                print(f"Failed to load high_level_reflection_memorys: {e}")

    def save_local(self, memory_path = None) -> None:
        """Save the memory to the local file."""
        if memory_path is None:
            memory_path = self.memory_path

        for symbol in self.symbols:
            path = os.path.join(memory_path, symbol, "market_intelligence")
            os.makedirs(path, exist_ok=True)
            self.market_intelligence_memorys[symbol].save_local(path)
            path = os.path.join(memory_path, symbol, "low_level_reflection")
            os.makedirs(path, exist_ok=True)
            self.low_level_reflection_memorys[symbol].save_local(path)
            path = os.path.join(memory_path, symbol, "high_level_reflection")
            os.makedirs(path, exist_ok=True)
            self.high_level_reflection_memorys[symbol].save_local(path)
