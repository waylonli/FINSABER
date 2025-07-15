"""
Configuration file for FINCON system.
Contains parameters for training, testing, LLM settings, and agent configuration.
"""

import os
from datetime import datetime
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# LLM settings
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0
LLM_BELIEF_TEMPERATURE = 0.0  # For consistent belief generation
LLM_MAX_TOKENS = 1024
LLM_API_KEY = os.getenv("OPENAI_API_KEY")

# Memory settings
MAX_MEMORY_EVENTS = 5  # Number of top memory events retrieved for each agent
MEMORY_DECAY_RATES = {
    "news_analyst": 0.9,     # Daily decay rate for news (fast decay)
    "filing_analyst": 0.99,  # Decay rate for SEC filings (slow decay)
    "ecc_analyst": 0.97,     # Decay rate for earnings calls (medium decay)
    "data_analyst": 0.95,    # Decay rate for market data (medium-fast decay)
}
