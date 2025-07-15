"""
Base agent class for FINCON system.
All agents in the system inherit from this class.
"""

import os
import json
from abc import ABC, abstractmethod
from datetime import datetime
import logging

import openai
from sentence_transformers import SentenceTransformer

from llm_traders.fincon_selector.memory.working_memory import WorkingMemory
from llm_traders.fincon_selector.memory.procedural_memory import ProceduralMemory
from llm_traders.fincon_selector.memory.episodic_memory import EpisodicMemory
from llm_traders.fincon_selector.modules.general_config import GeneralConfigModule
from llm_traders.fincon_selector.modules.profiling import ProfilingModule
from llm_traders.fincon_selector.modules.perception import PerceptionModule
from llm_traders.fincon_selector.modules.action import ActionModule
import llm_traders.fincon_selector.fincon_config as config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class BaseAgent(ABC):
    """Base agent class for FINCON system."""
    
    def __init__(self, agent_id, agent_type, target_symbols=None):
        """
        Initialize the base agent.
        
        Args:
            agent_id (str): Unique identifier for the agent
            agent_type (str): Type of agent (manager, analyst, risk_control)
            target_symbols (list): List of stock symbols the agent focuses on
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.target_symbols = target_symbols or []
        self.logger = logging.getLogger(f"{agent_type}_{agent_id}")
        
        # Initialize LLM client
        openai.api_key = config.LLM_API_KEY
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize modules
        self.general_config = GeneralConfigModule()
        self.profiling = ProfilingModule(agent_type)
        self.perception = PerceptionModule(agent_type)
        self.action = ActionModule(agent_type)
        
        # Initialize memory components
        self.working_memory = WorkingMemory()
        self.procedural_memory = self._initialize_procedural_memory()
        
        # Only manager agent has episodic memory
        self.episodic_memory = None
        if agent_type == "manager":
            self.episodic_memory = EpisodicMemory()
            
        self.logger.info(f"Initialized {agent_type} agent with ID: {agent_id}")
    
    def _initialize_procedural_memory(self):
        """Initialize procedural memory with appropriate decay rate."""
        decay_rate = 0.95  # Default decay rate
        
        # Set decay rate based on agent type
        if self.agent_type in config.MEMORY_DECAY_RATES:
            decay_rate = config.MEMORY_DECAY_RATES[self.agent_type]
            
        return ProceduralMemory(decay_rate=decay_rate)
    
    def _get_llm_response(self, prompt, temperature=None):
        """
        Get response from LLM.
        
        Args:
            prompt (str): Prompt to send to LLM
            temperature (float, optional): Temperature for LLM sampling
            
        Returns:
            str: LLM response
        """
        if temperature is None:
            temperature = config.LLM_TEMPERATURE
            
        try:
            response = openai.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "system", "content": prompt}],
                temperature=temperature,
                max_tokens=config.LLM_MAX_TOKENS
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error getting LLM response: {str(e)}")
            return ""
    
    def _generate_embedding(self, text):
        """
        Generate embedding for text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        return self.embedding_model.encode(text)
    
    def add_to_procedural_memory(self, content, event_type, importance=1.0):
        """
        Add event to procedural memory.
        
        Args:
            content (str): Content of the memory event
            event_type (str): Type of event
            importance (float): Initial importance score
            
        Returns:
            str: ID of the memory event
        """
        embedding = self._generate_embedding(content)
        return self.procedural_memory.add_event(content, event_type, embedding, importance)
    
    def retrieve_from_procedural_memory(self, query, top_k=None):
        """
        Retrieve top-k events from procedural memory based on query.
        
        Args:
            query (str): Query to search for
            top_k (int, optional): Number of events to retrieve
            
        Returns:
            list: List of memory events
        """
        if top_k is None:
            top_k = config.MAX_MEMORY_EVENTS
            
        query_embedding = self._generate_embedding(query)
        return self.procedural_memory.retrieve_events(query_embedding, top_k)
    
    def update_event_importance(self, event_id, importance_change):
        """
        Update importance score of a memory event.
        
        Args:
            event_id (str): ID of the memory event
            importance_change (float): Change in importance score
        """
        self.procedural_memory.update_importance(event_id, importance_change)
    
    @abstractmethod
    def process(self, observation):
        """
        Process an observation and produce output.
        
        Args:
            observation: Input observation for the agent to process
            
        Returns:
            dict: Agent's output
        """
        pass
    
    def save_state(self, path):
        """
        Save agent state to disk.
        
        Args:
            path (str): Path to save state to
        """
        state = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "target_symbols": self.target_symbols,
            "procedural_memory": self.procedural_memory.to_dict()
        }
        
        if self.episodic_memory:
            state["episodic_memory"] = self.episodic_memory.to_dict()
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f)
            
        self.logger.info(f"Saved agent state to {path}")
    
    def load_state(self, path):
        """
        Load agent state from disk.
        
        Args:
            path (str): Path to load state from
        """
        with open(path, 'r') as f:
            state = json.load(f)
            
        self.agent_id = state["agent_id"]
        self.agent_type = state["agent_type"]
        self.target_symbols = state["target_symbols"]
        
        self.procedural_memory.from_dict(state["procedural_memory"])
        
        if self.episodic_memory and "episodic_memory" in state:
            self.episodic_memory.from_dict(state["episodic_memory"])
            
        self.logger.info(f"Loaded agent state from {path}")