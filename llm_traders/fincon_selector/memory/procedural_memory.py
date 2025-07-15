"""
Procedural memory module for FINCON agents.
Stores historical actions, outcomes, and reflections during sequential decision-making.
"""

import logging
from datetime import datetime
import uuid
import json
import os
import numpy as np
from scipy.spatial.distance import cosine

class ProceduralMemory:
    """
    Procedural memory for FINCON agents.
    
    Stores historical actions, outcomes, and reflections during sequential decision-making.
    Implements the memory event ranking using relevancy, recency, and importance.
    """
    
    def __init__(self, decay_rate=0.95, max_events=1000):
        """
        Initialize procedural memory.
        
        Args:
            decay_rate (float): Rate at which importance decays over time (0-1)
            max_events (int): Maximum number of events to store
        """
        self.events = {}
        self.decay_rate = decay_rate
        self.max_events = max_events
        self.logger = logging.getLogger("procedural_memory")
        
    def add_event(self, content, event_type, embedding, importance=1.0):
        """
        Add a new memory event.
        
        Args:
            content (str): Textual content of the memory event
            event_type (str): Type of event (e.g., "news", "trading_decision")
            embedding (np.ndarray): Vector embedding of content
            importance (float): Initial importance of the event
            
        Returns:
            str: ID of the added event
        """
        event_id = str(uuid.uuid4())
        
        event = {
            "id": event_id,
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "content": content,
            "importance": importance,
            "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            "access_count": 0
        }
        
        self.events[event_id] = event
        
        # If we've exceeded the maximum number of events, remove the least important events
        if len(self.events) > self.max_events:
            self._prune_events()
            
        self.logger.debug(f"Added event to procedural memory: {event_id}")
        return event_id
        
    def retrieve_events(self, query_embedding, top_k=5, min_similarity=0.1):
        """
        Retrieve the top-k events based on query similarity and importance.
        
        Args:
            query_embedding (np.ndarray): Vector embedding of query
            top_k (int): Number of events to retrieve
            min_similarity (float): Minimum similarity threshold
            
        Returns:
            list: Top-k events sorted by relevance score
        """
        if not self.events:
            return []
            
        query_embedding = np.array(query_embedding)
        current_time = datetime.now()
        
        # Calculate relevance scores for all events
        scored_events = []
        
        for event_id, event in self.events.items():
            # Convert embedding to numpy array if needed
            event_embedding = np.array(event["embedding"])
            
            # Calculate cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(query_embedding, event_embedding)
            
            # Skip events with similarity below threshold
            if similarity < min_similarity:
                continue
                
            # Calculate time decay based on age of event
            event_time = datetime.fromisoformat(event["timestamp"])
            time_diff = (current_time - event_time).total_seconds() / 86400.0  # Convert to days
            time_decay = self.decay_rate ** time_diff
            
            # Calculate relevance score using custom formula from the paper
            relevance_score = similarity + event["importance"] * time_decay
            
            scored_events.append((event_id, relevance_score))
            
        # Sort events by relevance score (descending)
        scored_events.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k events
        top_events = []
        for event_id, score in scored_events[:top_k]:
            event = self.events[event_id].copy()
            event["relevance_score"] = score
            
            # Increment access count
            self.events[event_id]["access_count"] += 1
            
            top_events.append(event)
            
        self.logger.debug(f"Retrieved {len(top_events)} events from procedural memory")
        return top_events
        
    def update_importance(self, event_id, importance_change):
        """
        Update the importance of a memory event.
        
        Args:
            event_id (str): ID of the memory event
            importance_change (float): Change in importance
            
        Returns:
            bool: True if event was found and updated, False otherwise
        """
        if event_id not in self.events:
            self.logger.warning(f"Event not found in procedural memory: {event_id}")
            return False
            
        self.events[event_id]["importance"] += importance_change
        
        # Ensure importance stays within reasonable bounds
        self.events[event_id]["importance"] = max(0.0, min(5.0, self.events[event_id]["importance"]))
        
        self.logger.debug(f"Updated importance of event {event_id} by {importance_change}")
        return True
        
    def _prune_events(self):
        """
        Remove least important events to stay within capacity.
        """
        # Calculate effective importance (importance * time decay)
        current_time = datetime.now()
        effective_importance = {}
        
        for event_id, event in self.events.items():
            event_time = datetime.fromisoformat(event["timestamp"])
            time_diff = (current_time - event_time).total_seconds() / 86400.0  # Convert to days
            time_decay = self.decay_rate ** time_diff
            
            # Consider both importance and access count
            effective_importance[event_id] = event["importance"] * time_decay * (1 + 0.1 * event["access_count"])
            
        # Sort events by effective importance
        sorted_events = sorted(effective_importance.items(), key=lambda x: x[1])
        
        # Remove least important events
        num_to_remove = len(self.events) - self.max_events
        for event_id, _ in sorted_events[:num_to_remove]:
            self.events.pop(event_id)
            
        self.logger.debug(f"Pruned {num_to_remove} events from procedural memory")
        
    def apply_decay(self):
        """
        Apply time-based decay to all events' importance.
        """
        current_time = datetime.now()
        
        for event_id, event in self.events.items():
            event_time = datetime.fromisoformat(event["timestamp"])
            time_diff = (current_time - event_time).total_seconds() / 86400.0  # Convert to days
            
            # Apply decay
            decay_factor = self.decay_rate ** time_diff
            self.events[event_id]["importance"] *= decay_factor
            
        self.logger.debug("Applied decay to all events in procedural memory")
        
    def get_event_by_id(self, event_id):
        """
        Get event by ID.
        
        Args:
            event_id (str): ID of the event
            
        Returns:
            dict: Event data or None if not found
        """
        return self.events.get(event_id)
        
    def get_events_by_type(self, event_type):
        """
        Get all events of a specific type.
        
        Args:
            event_type (str): Type of events to retrieve
            
        Returns:
            list: Events of the specified type
        """
        return [event for event in self.events.values() if event["type"] == event_type]
    
    def clear(self):
        """Clear all events from procedural memory."""
        self.events = {}
        self.logger.debug("Cleared procedural memory")
        
    def save_to_file(self, filepath):
        """
        Save procedural memory to a file.
        
        Args:
            filepath (str): Path to save the memory
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(self.events, f)
                
            self.logger.info(f"Saved procedural memory to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving procedural memory: {str(e)}")
            return False
            
    def load_from_file(self, filepath):
        """
        Load procedural memory from a file.
        
        Args:
            filepath (str): Path to load the memory from
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                self.events = json.load(f)
                
            self.logger.info(f"Loaded procedural memory from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading procedural memory: {str(e)}")
            return False
            
    def to_dict(self):
        """
        Convert procedural memory to a dictionary.
        
        Returns:
            dict: Dictionary representation of procedural memory
        """
        return {
            "events": self.events,
            "decay_rate": self.decay_rate,
            "max_events": self.max_events
        }
        
    def from_dict(self, data):
        """
        Load procedural memory from a dictionary.
        
        Args:
            data (dict): Dictionary representation of procedural memory
        """
        self.events = data.get("events", {})
        self.decay_rate = data.get("decay_rate", self.decay_rate)
        self.max_events = data.get("max_events", self.max_events)