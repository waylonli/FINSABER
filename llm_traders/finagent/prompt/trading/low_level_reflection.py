import os
from typing import Dict, List, Any
import pandas as pd
from copy import deepcopy
import backoff

from llm_traders.finagent.registry import PROMPT
from llm_traders.finagent.prompt import Prompt
from llm_traders.finagent.memory import MemoryInterface
from llm_traders.finagent.provider import EmbeddingProvider
from llm_traders.finagent.query import DiverseQuery
from llm_traders.finagent.utils import init_path, save_html, save_json, load_json

@PROMPT.register_module(force=True)
class LowLevelReflectionTrading(Prompt):
    def __init__(self,
                 *args,
                 model: Any = None,
                 short_term_past_date_range: int = 1,
                 medium_term_past_date_range: int = 7,
                 long_term_past_date_range: int = 14,
                 short_term_next_date_range: int = 1,
                 medium_term_next_date_range: int = 7,
                 long_term_next_date_range: int = 14,
                 look_back_days: int = 14,
                 look_forward_days: int = 14,
                 **kwargs):

        self.model = model
        self.short_term_past_date_range = short_term_past_date_range
        self.medium_term_past_date_range = medium_term_past_date_range
        self.long_term_past_date_range = long_term_past_date_range
        self.short_term_next_date_range = short_term_next_date_range
        self.medium_term_next_date_range = medium_term_next_date_range
        self.long_term_next_date_range = long_term_next_date_range
        self.look_back_days = look_back_days
        self.look_forward_days = look_forward_days

        super(LowLevelReflectionTrading, self).__init__()

    def _convert_to_price_movement(self, state: Dict, current_date: str = None):

        def price_movement_to_text(x):
            if x > 0:
                return "an increase of {:.2f}%".format(abs(x * 100))
            elif x < 0:
                return "a decrease of {:.2f}%".format(abs(x * 100))
            elif x == 0:
                return "no change"
            else:
                return "unknown"
            
        price = state["price"]
        price = deepcopy(price)
        price = price.reset_index(drop=False)
        price = price[["timestamp", "adj_close"]]
        price = price.dropna(axis=0, how="any")
        price = price.drop_duplicates(subset=["timestamp"], keep="first")


        past_price = price[price["timestamp"] <= pd.to_datetime(current_date).date()]
        next_price = price[price["timestamp"] >= pd.to_datetime(current_date).date()]

        short_term_past_price_movement = past_price["adj_close"].pct_change(periods=self.short_term_past_date_range).iloc[-1]
        medium_term_past_price_movement = past_price["adj_close"].pct_change(periods=self.medium_term_past_date_range).iloc[-1]
        long_term_past_price_movement = past_price["adj_close"].pct_change(periods=self.long_term_past_date_range).iloc[-1]
        short_term_past_price_movement_text = price_movement_to_text(short_term_past_price_movement)
        medium_term_past_price_movement_text = price_movement_to_text(medium_term_past_price_movement)
        long_term_past_price_movement_text = price_movement_to_text(long_term_past_price_movement)

        short_term_next_price_movement = next_price["adj_close"].pct_change(periods=self.short_term_next_date_range).shift(-self.short_term_next_date_range).iloc[0]
        medium_term_next_price_movement = next_price["adj_close"].pct_change(periods=self.medium_term_next_date_range).shift(-self.medium_term_next_date_range).iloc[0]
        long_term_next_price_movement = next_price["adj_close"].pct_change(periods=self.long_term_next_date_range).shift(-self.long_term_next_date_range).iloc[0]
        short_term_next_price_movement_text = price_movement_to_text(short_term_next_price_movement)
        medium_term_next_price_movement_text = price_movement_to_text(medium_term_next_price_movement)
        long_term_next_price_movement_text = price_movement_to_text(long_term_next_price_movement)

        res = {
            "short_term_past_price_movement": str(short_term_past_price_movement_text),
            "medium_term_past_price_movement": str(medium_term_past_price_movement_text),
            "long_term_past_price_movement": str(long_term_past_price_movement_text),
            "short_term_past_date_range": int(self.short_term_past_date_range),
            "medium_term_past_date_range": int(self.medium_term_past_date_range),
            "long_term_past_date_range": int(self.long_term_past_date_range),
            "short_term_next_price_movement": str(short_term_next_price_movement_text),
            "medium_term_next_price_movement": str(medium_term_next_price_movement_text),
            "long_term_next_price_movement": str(long_term_next_price_movement_text),
            "short_term_next_date_range": int(self.short_term_next_date_range),
            "medium_term_next_date_range": int(self.medium_term_next_date_range),
            "long_term_next_date_range": int(self.long_term_next_date_range),
        }

        return res

    def convert_to_params(self,
                         state: Dict,
                         info: Dict,
                         params: Dict,
                         memory: MemoryInterface = None,
                         provider: EmbeddingProvider = None,
                         diverse_query: DiverseQuery = None) -> Dict:

        res_params = deepcopy(params)

        current_date = info["date"]
        price_movement = self._convert_to_price_movement(state, current_date=current_date)

        res_params.update(price_movement)

        return res_params

    @backoff.on_exception(backoff.constant, (KeyError), max_tries=3, interval=10)
    def get_response_dict(self,
                          provider,
                          model,
                          messages,
                          check_keys: List[str] = None):

        check_keys = [
            "reasoning",
            "query"
        ]

        response_dict, res_html = super(LowLevelReflectionTrading, self).get_response_dict(provider = provider,
                                                                                 model = model,
                                                                                 messages = messages,
                                                                                 check_keys = check_keys)

        return response_dict, res_html

    def add_to_memory(self,
                     state: pd.DataFrame = None,
                     info: Dict = None,
                     res: Dict = None,
                     memory: MemoryInterface = None,
                     provider: EmbeddingProvider = None) -> None:
        symbol = info["symbol"]

        data = deepcopy(res["params"])
        response_dict = deepcopy(res["response_dict"])

        embedding_text = response_dict["query"]
        embedding = provider.embed_query(embedding_text)

        data.update({
            "embedding_text": embedding_text,
            "embedding": embedding,
            "reasoning": response_dict["reasoning"],
            "query": response_dict["query"],
        })

        memory.add_memory(type="low_level_reflection",
                          symbol=symbol,
                          data=data,
                          embedding_key="embedding")

    def run(self,
            state: Dict,
            info: Dict,
            template: Any = None,
            params: Dict = None,
            memory: MemoryInterface = None,
            provider: EmbeddingProvider = None,
            diverse_query: DiverseQuery = None,
            exp_path: str = None,
            save_dir: str = None,
            call_provider = True,
            **kwargs):

        # print(">" * 50 + f"{info['date']} - Running Low Level Reflection Trading Prompt" + ">" * 50)

        # init path
        res_json_path = init_path(os.path.join(exp_path, "json", save_dir, "low_level_reflection"))
        html_path = init_path(os.path.join(exp_path, "html", save_dir, "low_level_reflection"))

        if call_provider:
            # low level reflection
            task_params = self.convert_to_params(state=state,
                                                 info=info,
                                                 params=params,
                                                 memory=memory,
                                                 provider=provider,
                                                 diverse_query=diverse_query,)
            message, html = self.to_message(params=task_params, template=template)
            response_dict, res_html = self.get_response_dict(provider = provider,
                                                   model = self.model,
                                                   messages = message)

            reasoning = response_dict["reasoning"]
            query = response_dict["query"]

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
            reasoning = res["response_dict"]["reasoning"]
            query = res["response_dict"]["query"]

        params.update(task_params)

        params.update({
            "low_level_reflection_reasoning": reasoning,
            "low_level_reflection_query": query,
        })

        # save_html(html, os.path.join(html_path, f"prompt_{info['date']}.html"))
        # save_html(res_html, os.path.join(html_path, f"res_{info['date']}.html"))
        # save_json(res, os.path.join(res_json_path, f"res_{info['date']}.json"))

        # print("<" * 50 + f"{info['date']} - Finish Running Low Level Reflection Trading Prompt" + "<" * 50)

        return res