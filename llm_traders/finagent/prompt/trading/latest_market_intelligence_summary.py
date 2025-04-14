import math
import os
import backoff
from typing import Dict, List, Any
from copy import deepcopy

from llm_traders.finagent.registry import PROMPT
from llm_traders.finagent.prompt import Prompt
from llm_traders.finagent.asset import ASSET
from llm_traders.finagent.memory import MemoryInterface
from llm_traders.finagent.provider import EmbeddingProvider
from llm_traders.finagent.query import DiverseQuery
from llm_traders.finagent.utils import init_path
from llm_traders.finagent.utils import save_html
from llm_traders.finagent.utils import save_json, load_json

@PROMPT.register_module(force=True)
class LatestMarketIntelligenceSummaryTrading(Prompt):
    def __init__(self,
                 *args,
                 model: Any = None,
                 **kwargs):
        self.model = model
        super(LatestMarketIntelligenceSummaryTrading, self).__init__()

    def convert_to_params(self,
                            state: Dict,
                            info: Dict,
                            params: Dict,
                            memory: MemoryInterface,
                            provider: EmbeddingProvider,
                            diverse_query: DiverseQuery = None) -> Dict:
        res_params = deepcopy(params)

        asset_info = ASSET.get_asset_info(info["symbol"])

        asset_name = asset_info["companyName"]
        asset_symbol = asset_info["symbol"]
        asset_exchange = asset_info["exchange"]
        asset_sector = asset_info["sector"]
        asset_industry = asset_info["industry"]
        asset_description = asset_info["description"]
        asset_type = info["asset_type"]
        current_date = info["date"]

        price = deepcopy(state["price"])
        news = deepcopy(state["news"])

        price = price[price.index == current_date]
        news = news[news.index == current_date]

        if len(news) > 20:
            news = news.sample(n=20)

        latest_market_intelligence_text = f"Date: Today is {current_date}.\n"

        if len(price) > 0:
            open = price["open"].values[0]
            high = price["high"].values[0]
            low = price["low"].values[0]
            close = price["close"].values[0]
            adj_close = price["adj_close"].values[0]
            latest_market_intelligence_text += f"Prices: Open: ({open}), High: ({high}), Low: ({low}), Close: ({close}), Adj Close: ({adj_close})\n"
        else:
            latest_market_intelligence_text += f"Prices: Today is closed for trading.\n"

        if len(news) == 0:
            latest_market_intelligence_text = "There is no latest market_intelligence.\n"
        else:
            latest_market_intelligence_list = []

            for row in news.iterrows():
                row = row[1]
                id = row["id"]
                title = row["title"]
                text = row["text"]

                latest_market_intelligence_item = f"ID: {id}\n" + \
                                                  f"Headline: {title}\n" + \
                                                  f"Content: {text}\n"

                latest_market_intelligence_list.append(latest_market_intelligence_item)

            if len(latest_market_intelligence_list) == 0:
                latest_market_intelligence_text = "There is no latest market_intelligence.\n"
            else:
                latest_market_intelligence_text = "\n".join(latest_market_intelligence_list)

        res_params.update({
            "date": current_date,
            "asset_name": asset_name,
            "asset_type": asset_type,
            "asset_symbol": asset_symbol,
            "asset_exchange": asset_exchange,
            "asset_sector": asset_sector,
            "asset_industry": asset_industry,
            "asset_description": asset_description,
            "latest_market_intelligence": latest_market_intelligence_text,
        })

        return res_params


    @backoff.on_exception(backoff.constant, (KeyError), max_tries=3, interval=10)
    def get_response_dict(self,
                          provider,
                          model,
                          messages,
                          check_keys: List[str] = None):

        check_keys = [
            "query",
            "summary"
        ]

        response_dict, res_html = super(LatestMarketIntelligenceSummaryTrading, self).get_response_dict(provider=provider,
                                                                                        model=model,
                                                                                        messages=messages,
                                                                                        check_keys=check_keys)

        return response_dict, res_html

    def add_to_memory(self,
                      state: Dict,
                      info: Dict,
                      res: Dict,
                      memory: MemoryInterface = None,
                      provider: EmbeddingProvider = None) -> None:

        response_dict = deepcopy(res["response_dict"])

        current_date = info["date"]
        stock_symbol = info["symbol"]
        
        price = deepcopy(state["price"])
        news = deepcopy(state["news"])

        price = price[price.index == current_date]
        news = news[news.index == current_date]

        if len(price) > 0:
            open = price["open"].values[0]
            high = price["high"].values[0]
            low = price["low"].values[0]
            close = price["close"].values[0]
            adj_close = price["adj_close"].values[0]
        else:
            open = math.nan
            high = math.nan
            low = math.nan
            close = math.nan
            adj_close = math.nan

        for row in news.iterrows():
            date = row[0] if isinstance(row[0], str) else row[0].strftime("%Y-%m-%d")
            row = row[1]

            id = row["id"]
            title = row["title"]
            text = row["text"]

            embedding_text = f"Heading: {title}\n" + \
                             f"Content: {text}\n"

            embedding = provider.embed_query(embedding_text)

            data = {
                "date": date,
                "id": id,
                "title": title,
                "text": text,
                "open": open,
                "high": high,
                "low": low,
                "close": close,
                "adj_close": adj_close,
                "query": response_dict["query"],
                "summary": response_dict["summary"],
                "embedding_text": embedding_text,
                "embedding": embedding,
            }

            memory.add_memory(type="market_intelligence",
                              symbol=stock_symbol,
                              data=data,
                              embedding_key="embedding")

    def run(self,
            state: Dict,
            info: Dict,
            params: Dict,
            template: Any = None,
            memory: MemoryInterface = None,
            provider: EmbeddingProvider = None,
            diverse_query: DiverseQuery = None,
            exp_path: str = None,
            save_dir: str = None,
            call_provider = True,
            **kwargs):

        # print(">" * 50 + f"{info['date']} - Running Latest Market Intelligence Summary Trading Prompt" + ">" * 50)

        # init path
        res_json_path = init_path(os.path.join(exp_path, "json", save_dir, "latest_market_intelligence"))
        html_path = init_path(os.path.join(exp_path, "html", save_dir, "latest_market_intelligence"))

        if call_provider:
            # summary latest market intelligence
            task_params = self.convert_to_params(state=state,
                                                 info=info,
                                                 params=params,
                                                 memory=memory,
                                                 provider=provider,
                                                 diverse_query=diverse_query)
            message, html = self.to_message(params=task_params, template=template)
            response_dict, res_html = self.get_response_dict(provider = provider,
                                                          model=self.model,
                                                          messages=message)

            query = response_dict["query"]
            summary = response_dict["summary"]

            html = html.prettify()
            res_html = res_html.prettify()

            res = {
                "params": task_params,
                "message": message,
                "html": html,
                "res_html": res_html,
                "response_dict": response_dict,
            }

        else:

            res = load_json(os.path.join(res_json_path, f"res_{info['date']}.json"))

            task_params= res["params"]

            html = res["html"]
            res_html = res["res_html"]
            query = res["response_dict"]["query"]
            summary = res["response_dict"]["summary"]

        params.update(task_params)

        params.update({
            "latest_market_intelligence_query": query,
            "latest_market_intelligence_summary": summary,
        })

        # save_html(html, os.path.join(html_path, f"prompt_{info['date']}.html"))
        # save_html(res_html, os.path.join(html_path, f"res_{info['date']}.html"))
        # save_json(res, os.path.join(res_json_path, f"res_{info['date']}.json"))

        # print("<" * 50 + f"{info['date']} - Finish Running Latest Market Intelligence Summary Trading Prompt" + "<" * 50)

        return res
