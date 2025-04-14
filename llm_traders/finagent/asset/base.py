import os
from typing import Dict, List
from bs4 import BeautifulSoup
from glob import glob
from llm_traders.finagent.utils import assemble_project_path
from llm_traders.finagent.utils import Singleton
import json
from copy import deepcopy

class Asset(metaclass=Singleton):
    def __init__(self):
        self.assets = self._load_assets()

    def _load_assets(self):
        assets = {}
        assets["asset_infos"] = self._load_asset_infos()
        assets["traders"] = self._load_traders()
        assets["modules"] = self._load_modules()
        return assets

    def _load_traders(self)->Dict[str, str]:
        traders_dir_path = assemble_project_path("finagent/res/prompts/trader")

        traders_paths = glob(os.path.join(traders_dir_path, "**", "*.txt"), recursive=True)
        traders = {}
        for trader_path in traders_paths:
            name = os.path.basename(trader_path).replace(".txt", "")
            with open(trader_path, "r") as f:
                text = f.read().strip()
                traders[name] = text
        return traders

    def _load_asset_infos(self)->Dict[str, str]:
        asset_infos_dir_path = assemble_project_path("finagent/res/prompts/asset_infos")

        asset_infos = {}
        asset_infos_paths = glob(os.path.join(asset_infos_dir_path, "**", "*.json"), recursive=True)
        for asset_info_path in asset_infos_paths:
            with open(asset_info_path, "r") as f:
                asset_info = json.load(f)
                for k, v in asset_info.items():
                    if k not in asset_infos:
                        asset_infos[k] = v

        return asset_infos

    def _load_modules(self)->Dict[str, BeautifulSoup]:
        modules_dir_path = assemble_project_path("finagent/res/prompts/module")
        modules = {}

        modules_paths = glob(os.path.join(modules_dir_path, "**", "*.html"), recursive=True)
        for module_path in modules_paths:
            with open(module_path, "r") as f:
                module = BeautifulSoup(f.read(), "html.parser")
                name = os.path.basename(module_path).replace(".html", "")
                modules[name] = module
        return modules

    def check_asset_info(self, symbol: str = None)-> bool:
        return symbol in self.assets["asset_infos"]

    def get_asset_info(self, symbol: str = None)-> str:
        return deepcopy(self.assets["asset_infos"][symbol]) if self.check_asset_info(symbol) else \
            {"companyName": symbol, "symbol": symbol, "exchange": symbol, "sector": symbol, "industry": symbol, "description": symbol}

    def check_trader(self, name: str = None)-> bool:
        return name in self.assets["traders"]

    def get_trader(self, name: str = None)-> str:
        return deepcopy(self.assets["traders"][name])

    def check_module(self, name: str = None)-> bool:
        return name in self.assets["modules"]

    def get_module(self, name: str = None)-> BeautifulSoup:
        return deepcopy(self.assets["modules"][name])

ASSET = Asset()