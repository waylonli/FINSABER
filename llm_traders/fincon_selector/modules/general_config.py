"""
General configuration module for FINCON agents.
Defines task types and trading targets.
"""

class GeneralConfigModule:
    """
    General configuration module for FINCON agents.
    
    This module defines the task types (e.g., stock trading, portfolio management)
    and specifies trading targets including sector and performance details.
    """
    
    def __init__(self):
        """Initialize the general configuration module."""
        self.task_types = {
            "single_stock_trading": "Trading of a single stock asset",
            "portfolio_management": "Management of a portfolio of multiple stock assets"
        }
        
        self.trading_sectors = {
            "technology": "Technology sector companies",
            "automotive": "Automotive industry companies",
            "healthcare": "Healthcare and pharmaceutical companies",
            "finance": "Financial services companies",
            "consumer": "Consumer goods and services companies",
            "energy": "Energy sector companies",
            "communication": "Communication services companies"
        }
        
        # Mapping of stock symbols to sectors
        self.symbol_to_sector = {
            "TSLA": "automotive",
            "AMZN": "consumer",
            "NIO": "automotive",
            "MSFT": "technology",
            "AAPL": "technology",
            "GOOG": "technology",
            "NFLX": "communication",
            "COIN": "finance",
            "PFE": "healthcare",
            "GM": "automotive",
            "LLY": "healthcare"
        }
    
    def get_task_description(self, task_type):
        """
        Get description for a task type.
        
        Args:
            task_type (str): Type of task
            
        Returns:
            str: Description of the task
        """
        return self.task_types.get(task_type, "Unknown task type")
    
    def get_sector_description(self, sector):
        """
        Get description for a sector.
        
        Args:
            sector (str): Sector name
            
        Returns:
            str: Description of the sector
        """
        return self.trading_sectors.get(sector, "Unknown sector")
    
    def get_symbol_sector(self, symbol):
        """
        Get sector for a stock symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            str: Sector of the stock
        """
        return self.symbol_to_sector.get(symbol, "unknown")
    
    def generate_task_config(self, task_type, trading_targets):
        """
        Generate task configuration text.
        
        Args:
            task_type (str): Type of task
            trading_targets (list): List of trading targets
            
        Returns:
            str: Formatted task configuration text
        """
        task_desc = self.get_task_description(task_type)
        
        config_text = f"Investment Task: {task_type}\n"
        config_text += f"Task Description: {task_desc}\n\n"
        
        config_text += "Trading Targets:\n"
        for target in trading_targets:
            sector = self.get_symbol_sector(target)
            sector_desc = self.get_sector_description(sector)
            config_text += f"- {target}: {sector_desc}\n"
        
        return config_text