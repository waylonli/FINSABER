import os
from typing import Dict, List, Any
import pandas as pd
from copy import deepcopy
import backoff

from llm_traders.finagent.registry import PROMPT
from llm_traders.finagent.prompt import Prompt
from llm_traders.finagent.memory import MemoryInterface
from llm_traders.finagent.provider import EmbeddingProvider
from llm_traders.finagent.utils import init_path, save_html, save_json, load_json
from llm_traders.finagent.query import DiverseQuery

@PROMPT.register_module(force=True)
class DecisionTrading(Prompt):
    def __init__(self,
                 *args,
                 model: Any = None,
                 **kwargs):
        self.model = model
        super(DecisionTrading, self).__init__()

    def convert_to_params(self,
                         state: Dict,
                         info: Dict,
                         params: Dict,
                         memory: MemoryInterface = None,
                         provider: EmbeddingProvider = None,
                         diverse_query: DiverseQuery=None) -> Dict:

        def return_to_text(x):
            if x > 0:
                return "an increase of {:.2f}%".format(abs(x * 100))
            elif x < 0:
                return "a decrease of {:.2f}%".format(abs(x * 100))
            elif x == 0:
                return "0%"


        res_params = deepcopy(params)

        asset_price = info["price"]
        asset_cash = info["cash"]
        asset_position = info["position"]
        asset_profit = info["total_profit"]
        asset_return = info["total_return"]

        asset_price = "{:.2f}".format(asset_price)
        asset_cash = "{:.2f}".format(asset_cash)
        asset_position = "{}".format(int(asset_position))
        asset_profit = "{:.2f}%".format(asset_profit)
        asset_return = return_to_text(asset_return)

        trader_preference = params["trader_preference"]

        res_params.update({
            "asset_price": asset_price,
            "asset_cash": asset_cash,
            "asset_position": asset_position,
            "asset_profit": asset_profit,
            "asset_return": asset_return,
            "trader_preference": trader_preference,
        })

        return res_params

    @backoff.on_exception(backoff.constant, (KeyError), max_tries=3, interval=10)
    def get_response_dict(self, provider, model, messages, check_keys: List[str] = None):

        check_keys = [
            "action",
            "reasoning",
        ]

        response_dict, res_html = super(DecisionTrading, self).get_response_dict(provider = provider,
                                                                       messages = messages,
                                                                       model = model,
                                                                       check_keys=check_keys)
        response_dict["action"] = response_dict["action"].replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")

        return response_dict, res_html

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
            call_provider: bool = True,
            **kwargs):

        # print(">" * 50 + f"{info['date']} - Running Decision Trading Prompt" + ">" * 50)

        # init path
        res_json_path = init_path(os.path.join(exp_path, "json", save_dir, "decision"))
        html_path = init_path(os.path.join(exp_path, "html", save_dir, "decision"))

        if call_provider:
            # decision
            task_params = self.convert_to_params(state=state,
                                                   info=info,
                                                   params=params,
                                                   memory=memory,
                                                   provider=provider,
                                                   diverse_query=diverse_query)
            message, html = self.to_message(params=task_params, template=template)
            response_dict, res_html = self.get_response_dict(provider = provider,
                                                            model = self.model,
                                                            messages = message)

            reasoning = response_dict["reasoning"]
            action = response_dict["action"]

            html = html.prettify()
            res_html = res_html.prettify()

            res = {
                "params": params,
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
            action = res["response_dict"]["action"]

        params.update(task_params)

        params.update({
            "decision_reasoning": reasoning,
            "decision_action": action,
        })

        # save_html(html, os.path.join(html_path, f"prompt_{info['date']}.html"))
        # save_html(res_html, os.path.join(html_path, f"res_{info['date']}.html"))
        # save_json(res, os.path.join(res_json_path, f"res_{info['date']}.json"))

        # print("<" * 50 + f"{info['date']} - Finish Running Decision Trading Prompt" + "<" * 50)

        return res