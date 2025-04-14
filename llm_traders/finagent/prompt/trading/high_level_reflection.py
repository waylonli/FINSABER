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
class HighLevelReflectionTrading(Prompt):
    def __init__(self,
                 *args,
                 model: Any = None,
                 previous_action_look_back_days: int = 14,
                 **kwargs):
        self.model = model
        self.previous_action_look_back_days = previous_action_look_back_days
        super(HighLevelReflectionTrading, self).__init__()

    def convert_to_params(self,
                         state: Dict,
                         info: Dict,
                         params: Dict,
                         memory: MemoryInterface = None,
                         provider: EmbeddingProvider = None,
                         diverse_query: DiverseQuery = None) -> Dict:

        res_params = deepcopy(params)

        date = info["date"]
        previous_date = params["previous_date"]
        previous_action = params["previous_action"]
        previous_reasoning = params["previous_reasoning"]

        if len(previous_date) == 0:
            previous_action_and_reasoning = "There is no previous action and reasoning as it is trading initialised."
        else:
            assert len(previous_date) == len(previous_action) == len(previous_reasoning), "length of previous_date, previous_action and previous_reasoning should be the same"

            previous_date = previous_date[-min(len(previous_date), self.previous_action_look_back_days):]
            previous_action = previous_action[-min(len(previous_action), self.previous_action_look_back_days):]
            previous_reasoning = previous_reasoning[-min(len(previous_reasoning), self.previous_action_look_back_days):]

            previous_action_and_reasoning = []
            for item in zip(previous_date, previous_action, previous_reasoning):
                text = "Date: {}\nAction: {}\nReasoning: {}".format(item[0], item[1], item[2])
                previous_action_and_reasoning.append(text)
            previous_action_and_reasoning = "\n".join(previous_action_and_reasoning)

        res_params.update({
            "previous_action_look_back_days": self.previous_action_look_back_days,
            "previous_action_and_reasoning": previous_action_and_reasoning,
        })

        return res_params

    @backoff.on_exception(backoff.constant, (KeyError), max_tries=3, interval=10)
    def get_response_dict(self, provider, model, messages, check_keys: List[str] = None):

        check_keys = [
            "reasoning",
            "improvement",
            "summary",
            "query"
        ]

        response_dict, res_html = super(HighLevelReflectionTrading, self).get_response_dict(provider=provider,
                                                                                  model=model,
                                                                                  messages=messages,
                                                                                  check_keys=check_keys)

        return response_dict, res_html

    def add_to_memory(self,
                      state: Dict= None,
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
            "improvement": response_dict["improvement"],
            "summary": response_dict["summary"],
            "query": response_dict["query"],
        })

        memory.add_memory(type="high_level_reflection",
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
            call_provider: bool = True,
            **kwargs):

        # print(">" * 50 + f"{info['date']} - Running High Level Reflection Trading Prompt" + ">" * 50)

        # init path
        res_json_path = init_path(os.path.join(exp_path, "json", save_dir, "high_level_reflection"))
        html_path = init_path(os.path.join(exp_path, "html", save_dir, "high_level_reflection"))

        if call_provider:
            # high level reflection
            task_params = self.convert_to_params(state=state,
                                                   info=info,
                                                   params=params,
                                                   memory=memory,
                                                   provider=provider,
                                                   diverse_query=diverse_query)
            message, html = self.to_message(params=task_params, template=template)
            response_dict, res_html = self.get_response_dict(provider=provider,
                                                   model=self.model,
                                                   messages=message)

            reasoning = response_dict["reasoning"]
            improvement = response_dict["improvement"]
            summary = response_dict["summary"]
            query = response_dict["query"]

            html = html.prettify()
            res_html = res_html.prettify()

            res = {
                "params": task_params,
                "message":message,
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
            improvement = res["response_dict"]["improvement"]
            summary = res["response_dict"]["summary"]
            query = res["response_dict"]["query"]

        params.update(task_params)

        params.update({
            "high_level_reasoning": reasoning,
            "high_level_improvement": improvement,
            "high_level_summary": summary,
            "high_level_query": query,
        })

        # save_html(html, os.path.join(html_path, f"prompt_{info['date']}.html"))
        # save_html(res_html, os.path.join(html_path, f"res_{info['date']}.html"))
        # save_json(res, os.path.join(res_json_path, f"res_{info['date']}.json"))

        # print("<" * 50 + f"{info['date']} - Finish Running High Level Reflection Trading Prompt" + "<" * 50)

        return res