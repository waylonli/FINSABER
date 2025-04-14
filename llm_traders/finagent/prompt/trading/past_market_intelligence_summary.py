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
class PastMarketIntelligenceSummaryTrading(Prompt):
    def __init__(self,
                 *args,
                 model: Any = None,
                 **kwargs):
        self.model = model
        super(PastMarketIntelligenceSummaryTrading, self).__init__()
    def convert_to_params(self,
                            state: Dict,
                            info: Dict,
                            params: Dict,
                            memory: MemoryInterface,
                            provider: EmbeddingProvider,
                            diverse_query: DiverseQuery = None) -> Dict:
        res_params = deepcopy(params)
        return res_params


    @backoff.on_exception(backoff.constant, (KeyError), max_tries=3, interval=10)
    def get_response_dict(self,
                          provider,
                          model,
                          messages,
                          check_keys: List[str] = None):

        check_keys = [
            "summary"
        ]

        response_dict, res_html = super(PastMarketIntelligenceSummaryTrading, self).get_response_dict(provider=provider,
                                                                                        model=model,
                                                                                        messages=messages,
                                                                                        check_keys=check_keys)

        return response_dict, res_html

    def add_to_memory(self,
                      state: Dict,
                      info: Dict,
                      params: Dict,
                      memory: MemoryInterface = None,
                      provider: EmbeddingProvider = None) -> None:
        raise NotImplementedError("PastMarketIntelligenceSummaryTrading does not support add_to_memory")

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

        # print(">" * 50 + f"{info['date']} - Running Past Market Intelligence Summary Trading Prompt" + ">" * 50)

        # init path
        res_json_path = init_path(os.path.join(exp_path, "json", save_dir, "past_market_intelligence"))
        html_path = init_path(os.path.join(exp_path, "html", save_dir, "past_market_intelligence"))

        if call_provider:
            task_params = self.convert_to_params(state=state,
                                                 info=info,
                                                 params=params,
                                                 memory=memory,
                                                 provider=provider,
                                                 diverse_query=diverse_query,
                                                 )
            message, html = self.to_message(params=task_params, template=template)
            response_dict, res_html = self.get_response_dict(provider = provider,
                                                        model = self.model,
                                                        messages = message)

            summary = response_dict["summary"]

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
            summary = res["response_dict"]["summary"]

        params.update(task_params)

        params.update({
            "past_market_intelligence_summary": summary,
        })

        save_html(html, os.path.join(html_path, f"prompt_{info['date']}.html"))
        save_html(res_html, os.path.join(html_path, f"res_{info['date']}.html"))
        save_json(res, os.path.join(res_json_path, f"res_{info['date']}.json"))

        # print("<" * 50 + f"{info['date']} - Finish Running Past Market Intelligence Summary Trading Prompt" + "<" * 50)

        return res
