"""
Episodic memory module for FINCON agents.
Stores actions, PnL series from previous episodes, and updated conceptual investment beliefs.
"""

import logging
from datetime import datetime
import json
import os

class EpisodicMemory:
    """
    Episodic memory for FINCON agents.
    
    Exclusive to the manager agent, this memory stores actions, PnL series from 
    previous episodes, and updated conceptual investment beliefs from the risk control component.
    """
    
    def __init__(self, max_episodes=10):
        """
        Initialize episodic memory.
        
        Args:
            max_episodes (int): Maximum number of episodes to store
        """
        self.episodes = {}
        self.beliefs = {}
        self.max_episodes = max_episodes
        self.logger = logging.getLogger("episodic_memory")
        
    def add_episode(self, episode_id, start_date, end_date, actions, pnl_series, metrics):
        """
        Add a new episode to memory.
        
        Args:
            episode_id (str): Identifier for the episode
            start_date (str): Start date of the episode
            end_date (str): End date of the episode
            actions (list): List of trading actions taken during the episode
            pnl_series (list): Series of daily PnL values
            metrics (dict): Performance metrics for the episode
            
        Returns:
            bool: True if successful, False otherwise
        """
        if episode_id in self.episodes:
            self.logger.warning(f"Episode {episode_id} already exists in episodic memory")
            return False
            
        episode = {
            "id": episode_id,
            "start_date": start_date,
            "end_date": end_date,
            "actions": actions,
            "pnl_series": pnl_series,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        self.episodes[episode_id] = episode
        
        # If we've exceeded the maximum number of episodes, remove the oldest ones
        if len(self.episodes) > self.max_episodes:
            self._prune_episodes()
            
        self.logger.info(f"Added episode {episode_id} to episodic memory")
        return True
        
    def update_beliefs(self, belief_set_id, beliefs, source_episodes=None, reasoning=None):
        """
        Update investment beliefs.
        
        Args:
            belief_set_id (str): Identifier for the belief set
            beliefs (dict): Updated investment beliefs
            source_episodes (list, optional): Episodes used to derive these beliefs
            reasoning (str, optional): Reasoning behind the belief updates
            
        Returns:
            bool: True if successful, False otherwise
        """
        belief_set = {
            "id": belief_set_id,
            "beliefs": beliefs,
            "source_episodes": source_episodes,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        }
        
        self.beliefs[belief_set_id] = belief_set
        self.logger.info(f"Updated investment beliefs {belief_set_id} in episodic memory")
        return True
        
    def get_episode(self, episode_id):
        """
        Get episode by ID.
        
        Args:
            episode_id (str): ID of the episode
            
        Returns:
            dict: Episode data or None if not found
        """
        return self.episodes.get(episode_id)
        
    def get_episodes(self, limit=None):
        """
        Get all episodes, optionally limited.
        
        Args:
            limit (int, optional): Maximum number of episodes to return
            
        Returns:
            list: Episodes sorted by timestamp (newest first)
        """
        # Sort episodes by timestamp (newest first)
        sorted_episodes = sorted(
            self.episodes.values(), 
            key=lambda x: x["timestamp"], 
            reverse=True
        )
        
        if limit is not None:
            sorted_episodes = sorted_episodes[:limit]
            
        return sorted_episodes
        
    def get_beliefs(self, belief_set_id=None):
        """
        Get investment beliefs.
        
        Args:
            belief_set_id (str, optional): ID of specific belief set to retrieve
            
        Returns:
            dict or list: Specific belief set or all belief sets sorted by timestamp
        """
        if belief_set_id is not None:
            return self.beliefs.get(belief_set_id)
            
        # Sort beliefs by timestamp (newest first)
        sorted_beliefs = sorted(
            self.beliefs.values(), 
            key=lambda x: x["timestamp"], 
            reverse=True
        )
        
        return sorted_beliefs
        
    def get_latest_beliefs(self):
        """
        Get the most recent set of investment beliefs.
        
        Returns:
            dict: Latest belief set or None if no beliefs exist
        """
        sorted_beliefs = self.get_beliefs()
        return sorted_beliefs[0] if sorted_beliefs else None
        
    def compare_episodes(self, episode_id1, episode_id2):
        """
        Compare two episodes based on performance metrics.
        
        Args:
            episode_id1 (str): ID of first episode
            episode_id2 (str): ID of second episode
            
        Returns:
            dict: Comparison results
        """
        episode1 = self.get_episode(episode_id1)
        episode2 = self.get_episode(episode_id2)
        
        if not episode1 or not episode2:
            missing_episodes = []
            if not episode1:
                missing_episodes.append(episode_id1)
            if not episode2:
                missing_episodes.append(episode_id2)
                
            self.logger.warning(f"Episodes not found in episodic memory: {missing_episodes}")
            return {"error": f"Episodes not found: {missing_episodes}"}
            
        comparison = {
            "episodes": [episode_id1, episode_id2],
            "metrics_comparison": {},
            "action_overlap": self._calculate_action_overlap(episode1, episode2),
            "better_performer": None
        }
        
        # Compare metrics
        metrics1 = episode1["metrics"]
        metrics2 = episode2["metrics"]
        
        for metric in set(metrics1.keys()).union(metrics2.keys()):
            if metric in metrics1 and metric in metrics2:
                value1 = metrics1[metric]
                value2 = metrics2[metric]
                
                # Determine which episode performed better for this metric
                # For metrics like Sharpe Ratio and Cumulative Return, higher is better
                if metric in ["cumulative_return", "sharpe_ratio"]:
                    better = episode_id1 if value1 > value2 else episode_id2
                # For metrics like Max Drawdown, lower is better
                elif metric in ["max_drawdown"]:
                    better = episode_id1 if value1 < value2 else episode_id2
                else:
                    better = None
                    
                comparison["metrics_comparison"][metric] = {
                    episode_id1: value1,
                    episode_id2: value2,
                    "difference": value1 - value2,
                    "better": better
                }
                
        # Determine overall better performer based on Sharpe Ratio
        if "sharpe_ratio" in comparison["metrics_comparison"]:
            sharpe_comp = comparison["metrics_comparison"]["sharpe_ratio"]
            comparison["better_performer"] = sharpe_comp["better"]
            
        return comparison
        
    def _calculate_action_overlap(self, episode1, episode2):
        """
        Calculate the overlap in trading actions between two episodes.
        
        Args:
            episode1 (dict): First episode
            episode2 (dict): Second episode
            
        Returns:
            float: Percentage of overlapping actions (0-1)
        """
        actions1 = episode1["actions"]
        actions2 = episode2["actions"]
        
        # Ensure actions are of the same length
        min_length = min(len(actions1), len(actions2))
        
        if min_length == 0:
            return 0.0
            
        # Count matching actions
        matches = 0
        for i in range(min_length):
            action1 = actions1[i]
            action2 = actions2[i]
            
            # Check if the trading decisions match
            if "decision" in action1 and "decision" in action2:
                if action1["decision"] == action2["decision"]:
                    matches += 1
                    
        return matches / min_length
        
    def _prune_episodes(self):
        """
        Remove oldest episodes to stay within capacity.
        """
        # Sort episodes by timestamp
        sorted_episodes = sorted(
            self.episodes.items(), 
            key=lambda x: x[1]["timestamp"]
        )
        
        # Remove oldest episodes
        num_to_remove = len(self.episodes) - self.max_episodes
        for episode_id, _ in sorted_episodes[:num_to_remove]:
            self.episodes.pop(episode_id)
            
        self.logger.debug(f"Pruned {num_to_remove} episodes from episodic memory")
        
    def clear(self):
        """Clear all episodes and beliefs from episodic memory."""
        self.episodes = {}
        self.beliefs = {}
        self.logger.debug("Cleared episodic memory")
        
    def save_to_file(self, filepath):
        """
        Save episodic memory to a file.
        
        Args:
            filepath (str): Path to save the memory
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            data = {
                "episodes": self.episodes,
                "beliefs": self.beliefs
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f)
                
            self.logger.info(f"Saved episodic memory to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving episodic memory: {str(e)}")
            return False
            
    def load_from_file(self, filepath):
        """
        Load episodic memory from a file.
        
        Args:
            filepath (str): Path to load the memory from
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            self.episodes = data.get("episodes", {})
            self.beliefs = data.get("beliefs", {})
                
            self.logger.info(f"Loaded episodic memory from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading episodic memory: {str(e)}")
            return False
            
    def to_dict(self):
        """
        Convert episodic memory to a dictionary.
        
        Returns:
            dict: Dictionary representation of episodic memory
        """
        return {
            "episodes": self.episodes,
            "beliefs": self.beliefs,
            "max_episodes": self.max_episodes
        }
        
    def from_dict(self, data):
        """
        Load episodic memory from a dictionary.
        
        Args:
            data (dict): Dictionary representation of episodic memory
        """
        self.episodes = data.get("episodes", {})
        self.beliefs = data.get("beliefs", {})
        self.max_episodes = data.get("max_episodes", self.max_episodes)