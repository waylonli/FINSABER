import pathlib
import sys
from typing import List, Dict
import os

ROOT  = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(ROOT)

from llm_traders.finagent.registry import DATASET

@DATASET.register_module(force=True)
class BaseDataset():
    def __init__(self):
        pass

    def _init_stocks(self):
        pass

    def _load_prices(self):
        pass

    def _load_news(self):
        pass