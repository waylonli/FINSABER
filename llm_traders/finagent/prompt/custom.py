import json
import os

import backoff
from typing import Dict, Any
from llm_traders.finagent.registry import PROMPT
from llm_traders.finagent.provider.provider import encode_image
from llm_traders.finagent.prompt.helper import generate_prompt_html, content_replace
from llm_traders.finagent.utils import parse_semi_formatted_xml, parse_semi_formatted_json
from backtest.toolkit.llm_cost_monitor import add_openai_cost_from_tokens_count

@PROMPT.register_module(force=True)
class Prompt():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _replace_keys(self, text: str, params: Dict):
        keys = params.keys()
        for key in keys:
            if key in text:
                if isinstance(params[key], list) or isinstance(params[key], dict):
                    text = text.replace(f'<${key}$>', json.dumps(params[key], indent=4))
                else:
                    text = text.replace(f'<${key}$>', str(params[key]))
        return text

    def convert_state_info_to_parmas(self, *args, **kwargs):
        raise NotImplementedError

    def to_message(self, *args,
                   params: Dict = None,
                   template: Any = None,
                   **kwargs):

        assert params is not None, "params is None"
        assert template is not None, "template is None"

        html = generate_prompt_html(params, template)

        system_message = None
        system_div_tag = html.find("div", class_="message", role="system")
        if system_div_tag is not None:
            p_tag = system_div_tag.find("p")
            content = p_tag.text

            content = content_replace(content)

            message = {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": content
                    }
                ]
            }
            system_message = message

        assert system_message is not None, "system_message is None"

        user_messages = []
        user_div_tags = html.find_all("div", class_="message", role="user")

        for user_div_tag in user_div_tags:
            tags = user_div_tag.find_all(["p", "img"])

            message = {
                "role": "user",
                "content": []
            }

            for tag in tags:

                if tag.name == "p":
                    content = tag.get_text(separator='\n')
                    content = content_replace(content)
                    message["content"].append({
                        "type": "text",
                        "text": content
                    })

                elif tag.name == "img":
                    image_path = tag.attrs["src"] if "src" in tag.attrs else None

                    if image_path is None or not os.path.exists(image_path):
                        message["content"].append({
                            "type": "text",
                            "text": "There is no figure as it is trading initialised."
                        })
                    else:
                        image_base64 = encode_image(image_path)
                        message["content"].append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        })
            user_messages.append(message)

        assert len(user_messages) > 0, "user_messages is None"

        messages = [system_message] + user_messages

        return messages, html

    def get_response_dict(self,
                          provider,
                          model,
                          messages,
                          check_keys=["decision", "reasoning"]):

        response, info = provider.create_completion(messages, model=model)
        # print("response from llm model {}: \ninfo: {}\nresponse: \n{}".format(model, info, response))
        add_openai_cost_from_tokens_count(model=model,
                                          prompt_tokens=info["prompt_tokens"],
                                          generated_tokens=info["completion_tokens"])

        response_dict, soup = parse_semi_formatted_xml(response)

        # print("response_dict: \n{}\n".format(response_dict))

        for key in check_keys:
            if key not in response_dict:
                print(messages)
                # raise KeyError(f"Key {key} not in response: {response_dict}")
                response_dict[key] = f"No {key} in response"
        return response_dict, soup