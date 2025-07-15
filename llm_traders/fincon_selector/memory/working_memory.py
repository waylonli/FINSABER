"""
Working memory module for FINCON agents.
Handles short-term memory processing tasks.
"""

import logging
from datetime import datetime
import uuid
import numpy as np

class WorkingMemory:
    """
    Working memory module for FINCON agents.
    
    Responsible for observation, distillation, and refinement of available memory events.
    """
    
    def __init__(self, capacity=10):
        """
        Initialize working memory.
        
        Args:
            capacity (int): Maximum number of items in working memory
        """
        self.capacity = capacity
        self.items = []
        self.logger = logging.getLogger("working_memory")
    
    def add_item(self, item, item_type):
        """
        Add item to working memory.
        
        Args:
            item: Item to add to working memory
            item_type (str): Type of item
            
        Returns:
            dict: Memory item with metadata
        """
        memory_item = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "type": item_type,
            "content": item,
            "processed": False
        }
        
        # Add to memory and maintain capacity limit
        self.items.append(memory_item)
        if len(self.items) > self.capacity:
            removed_item = self.items.pop(0)
            self.logger.debug(f"Removed oldest item from working memory: {removed_item['id']}")
        
        self.logger.debug(f"Added item to working memory: {memory_item['id']}")
        return memory_item
    
    def get_items(self, item_type=None, processed=None):
        """
        Get items from working memory, optionally filtered.
        
        Args:
            item_type (str, optional): Filter by item type
            processed (bool, optional): Filter by processed status
            
        Returns:
            list: Filtered memory items
        """
        filtered_items = self.items
        
        if item_type is not None:
            filtered_items = [item for item in filtered_items if item["type"] == item_type]
            
        if processed is not None:
            filtered_items = [item for item in filtered_items if item["processed"] == processed]
            
        return filtered_items
    
    def mark_processed(self, item_id):
        """
        Mark item as processed.
        
        Args:
            item_id (str): ID of item to mark as processed
            
        Returns:
            bool: True if item was found and marked, False otherwise
        """
        for item in self.items:
            if item["id"] == item_id:
                item["processed"] = True
                self.logger.debug(f"Marked item as processed: {item_id}")
                return True
                
        self.logger.warning(f"Item not found in working memory: {item_id}")
        return False
    
    def clear(self):
        """Clear all items from working memory."""
        self.items = []
        self.logger.debug("Cleared working memory")
    
    def summarize(self, items, max_length=500):
        """
        Summarize a list of memory items.
        
        Args:
            items (list): List of memory items to summarize
            max_length (int): Maximum length of summary
            
        Returns:
            str: Summary of memory items
        """
        if not items:
            return "No items to summarize."
        
        # Group items by type
        items_by_type = {}
        for item in items:
            item_type = item["type"]
            if item_type not in items_by_type:
                items_by_type[item_type] = []
            items_by_type[item_type].append(item)
        
        # Create summary
        summary = "Working Memory Summary:\n\n"
        
        for item_type, type_items in items_by_type.items():
            summary += f"{item_type.upper()} ({len(type_items)} items):\n"
            
            # Summarize the most recent 3 items of each type
            for item in type_items[-3:]:
                content = item["content"]
                
                # If content is a string, take first 100 characters
                if isinstance(content, str):
                    content_summary = content[:100] + "..." if len(content) > 100 else content
                # If content is a dict, take keys and first few values
                elif isinstance(content, dict):
                    content_summary = ", ".join([f"{k}: {str(v)[:20]}" for k, v in list(content.items())[:3]])
                # Otherwise, convert to string
                else:
                    content_summary = str(content)[:100] + "..."
                    
                summary += f"- {content_summary}\n"
            
            summary += "\n"
        
        # Trim to max length if needed
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
            
        return summary
    
    def process_observation(self, observation, observation_type):
        """
        Process an observation and store in working memory.
        
        Args:
            observation: Observation to process
            observation_type (str): Type of observation
            
        Returns:
            dict: Processed observation
        """
        # Add to working memory
        memory_item = self.add_item(observation, observation_type)
        
        # Process observation based on type
        processed_observation = {
            "id": memory_item["id"],
            "type": observation_type,
            "timestamp": memory_item["timestamp"],
            "content": observation
        }
        
        # Mark as processed
        self.mark_processed(memory_item["id"])
        
        self.logger.info(f"Processed observation of type {observation_type}")
        return processed_observation
    
    def distill_insights(self, observations, max_insights=5):
        """
        Distill key insights from a set of observations.
        
        Args:
            observations (list): List of observations
            max_insights (int): Maximum number of insights to extract
            
        Returns:
            list: Distilled insights
        """
        if not observations:
            return []
            
        # Example logic for distilling insights
        insights = []
        
        # Group observations by type
        observations_by_type = {}
        for obs in observations:
            obs_type = obs["type"]
            if obs_type not in observations_by_type:
                observations_by_type[obs_type] = []
            observations_by_type[obs_type].append(obs)
        
        # Extract insights from each type
        for obs_type, type_obs in observations_by_type.items():
            # Sort by timestamp (newest first)
            type_obs.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Take most recent observations
            recent_obs = type_obs[:min(3, len(type_obs))]
            
            # Extract content from observations
            for obs in recent_obs:
                content = obs["content"]
                
                # Extract key points based on observation type
                if obs_type == "news":
                    if "key_points" in content:
                        insights.extend(content["key_points"][:2])
                elif obs_type == "market_data":
                    if "technical_indicators" in content:
                        for indicator, value in content["technical_indicators"].items():
                            insights.append(f"{indicator}: {value}")
                elif obs_type == "filing":
                    if "key_insights" in content:
                        insights.extend(content["key_insights"][:2])
        
        # Limit number of insights
        insights = insights[:max_insights]
        
        self.logger.info(f"Distilled {len(insights)} insights from {len(observations)} observations")
        return insights