import os
import base64
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import backoff
import tiktoken
import numpy as np
from openai import OpenAI, AzureOpenAI, APIError, RateLimitError, BadRequestError, APITimeoutError

from llm_traders.finagent.provider import LLMProvider, EmbeddingProvider
from llm_traders.finagent.registry import PROVIDER
from llm_traders.finagent.utils import assemble_project_path, load_json

MAX_TOKENS = {
    "gpt-3.5-turbo-0301": 4097,
    "gpt-3.5-turbo-0613": 4097,
    "gpt-3.5-turbo-16k-0613": 16385,
}

PROVIDER_SETTING_KEY_VAR = "key_var"
PROVIDER_SETTING_EMB_MODEL = "emb_model"
PROVIDER_SETTING_COMP_MODEL = "comp_model"
PROVIDER_SETTING_IS_AZURE = "is_azure"
PROVIDER_SETTING_BASE_VAR = "base_var"       # Azure-speficic setting
PROVIDER_SETTING_API_VERSION = "api_version" # Azure-speficic setting
PROVIDER_SETTING_DEPLOYMENT_MAP = "models"   # Azure-speficic setting

@PROVIDER.register_module(force=True)
class OpenAIProvider(LLMProvider, EmbeddingProvider):
    """A class that wraps a given model"""

    client: Any = None
    llm_model: str = ""
    embedding_model: str = ""

    allowed_special: Union[Literal["all"], Set[str]] = set()
    disallowed_special: Union[Literal["all"], Set[str], Sequence[str]] = "all"
    chunk_size: int = 1000
    embedding_ctx_length: int = 8191
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    tiktoken_model_name: Optional[str] = None

    """Whether to skip empty strings when embedding or raise an error."""
    skip_empty: bool = False


    def __init__(self, provider_cfg_path) -> None:
        """Initialize a class instance

        Args:
            cfg: Config object

        Returns:
            None
        """
        self.retries = 5
        provider_cfg_path = assemble_project_path(provider_cfg_path)
        provider_cfg = load_json(provider_cfg_path)
        self.init_provider(provider_cfg)


    def init_provider(self, provider_cfg ) -> None:
        self.provider_cfg = self._parse_config(provider_cfg)

    def _parse_config(self, provider_cfg) -> dict:
        """Parse the config object"""

        conf_dict = dict()

        if isinstance(provider_cfg, dict):
            conf_dict = provider_cfg

        key_var_name = conf_dict[PROVIDER_SETTING_KEY_VAR]

        if conf_dict[PROVIDER_SETTING_IS_AZURE]:
            
            key = os.getenv(key_var_name)  
            endpoint_var_name = conf_dict[PROVIDER_SETTING_BASE_VAR]
            endpoint = os.getenv(endpoint_var_name)

            self.client = AzureOpenAI(
                api_key = key,
                api_version = conf_dict[PROVIDER_SETTING_API_VERSION],
                azure_endpoint = endpoint
            )
        else:
            key = os.getenv(key_var_name)
            self.client = OpenAI(api_key=key)
            # self.client = OpenAI(
            #     base_url="http://192.168.1.169:1234/v1/",
            # )

        self.embedding_model = conf_dict[PROVIDER_SETTING_EMB_MODEL]
        self.llm_model = conf_dict[PROVIDER_SETTING_COMP_MODEL]

        try:
            self.encoding = tiktoken.encoding_for_model(self.llm_model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

        return conf_dict

    @property
    def _emb_invocation_params(self) -> Dict:

        openai_args = {
            "model": self.embedding_model,
        }

        if self.provider_cfg[PROVIDER_SETTING_IS_AZURE]:
            engine = self._get_azure_deployment_id_for_model(self.embedding_model)
            openai_args = {
                "model": self.embedding_model,
                "engine": engine,
            }

        return openai_args

    def embed_with_retry(self, **kwargs: Any) -> Any:
        """Use backoff to retry the embedding call."""

        @backoff.on_exception(
            backoff.expo,
            (
                APIError,
                RateLimitError,
                APITimeoutError,
            ),
            max_tries=self.retries,
            max_value=10,
            jitter=None,
        )
        def _embed_with_retry(**kwargs: Any) -> Any:
            response = self.client.embeddings.create(**kwargs)
            if any(len(d.embedding) == 1 for d in response.data):
                raise RuntimeError("OpenAI API returned an empty embedding")
            return response

        return _embed_with_retry(**kwargs)


    def _get_len_safe_embeddings(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        embeddings: List[List[float]] = [[] for _ in range(len(texts))]
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to for OpenAIEmbeddings. "
                "Please install it with `pip install tiktoken`."
            )

        tokens = []
        indices = []
        model_name = self.tiktoken_model_name or self.embedding_model
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # print("Warning: model not found. Using cl100k_base encoding to count token numbers.")
            model = "cl100k_base"
            encoding = tiktoken.get_encoding(model)
        for i, text in enumerate(texts):
            token = encoding.encode(
                text,
                allowed_special=self.allowed_special,
                disallowed_special=self.disallowed_special,
            )
            for j in range(0, len(token), self.embedding_ctx_length):
                tokens.append(token[j : j + self.embedding_ctx_length])
                indices.append(i)

        batched_embeddings: List[List[float]] = []
        _chunk_size = self.chunk_size
        _iter = range(0, len(tokens), _chunk_size)

        for i in _iter:
            response = self.embed_with_retry(
                input=tokens[i : i + self.chunk_size],
                **self._emb_invocation_params,
            )
            batched_embeddings.extend(r.embedding for r in response.data)

        results: List[List[List[float]]] = [[] for _ in range(len(texts))]
        num_tokens_in_batch: List[List[int]] = [[] for _ in range(len(texts))]
        for i in range(len(indices)):
            if self.skip_empty and len(batched_embeddings[i]) == 1:
                continue
            results[indices[i]].append(batched_embeddings[i])
            num_tokens_in_batch[indices[i]].append(len(tokens[i]))

        for i in range(len(texts)):
            _result = results[i]
            if len(_result) == 0:
                average = self.embed_with_retry(
                    input="",
                    **self._emb_invocation_params,
                ).data[0].embedding
            else:
                average = np.average(_result, axis=0, weights=num_tokens_in_batch[i])
            embeddings[i] = (average / np.linalg.norm(average)).tolist()

        return embeddings

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """Call out to OpenAI's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        # NOTE: to keep things simple, we assume the list may contain texts longer
        #       than the maximum context and use length-safe embedding function.
        return self._get_len_safe_embeddings(texts)


    def embed_query(self, text: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        return self.embed_documents([text])[0]


    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        if self.embedding_model == "text-embedding-ada-002":
            embedding_dim = 1536
        elif self.embedding_model == "text-embedding-3-small":
            embedding_dim = 1536
        elif self.embedding_model == "text-embedding-3-large":
            embedding_dim = 3072
        else:
            raise ValueError(f"Unknown embedding model: {self.embedding_model}")
        return embedding_dim


    def create_completion(
        self,
        messages: List[Dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        seed: int | None = 42,
        max_tokens: int = 512,
    ) -> Tuple[str, Dict[str, int]]:
        """Create a chat completion using the OpenAI API

        Supports both GPT-4 and GPT-4V).

        Example Usage:
        image_path = "path_to_your_image.jpg"
        base64_image = encode_image(image_path)
        response, info = self.create_completion(
            model="gpt-4o-mini",
            messages=[
              {
                "role": "user",
                "content": [
                  {
                    "type": "text",
                    "text": "What’s in this image?"
                  },
                  {
                    "type": "image_url",
                    "image_url": {
                      "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                  }
                ]
              }
            ],
        )
        """

        if model is None:
            model = self.llm_model

        # print(f"Creating chat completion with model {model}, temperature {temperature}, max_tokens {max_tokens}")

        @backoff.on_exception(
            backoff.constant,
            (
                APIError, 
                RateLimitError, 
                APITimeoutError),
            max_tries=self.retries,
            interval=10,
        )
        def _generate_response_with_retry(
            messages: List[Dict[str, str]],
            model: str,
            temperature: float,
            seed: int | None,
            max_tokens: int = 512,
        ) -> Tuple[str, Dict[str, int]]:
            
            """Send a request to the OpenAI API."""
            MAX_RETRIES = 5

            for i in range(MAX_RETRIES):
                if self.provider_cfg[PROVIDER_SETTING_IS_AZURE]:
                    response = self.client.chat.completions.create(deployment_id=self.get_azure_deployment_id_for_model(model),
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    seed=seed,
                    max_tokens=max_tokens,)
                else:
                    response = self.client.chat.completions.create(model=model,
                    messages=messages,
                    temperature=temperature,
                    seed=seed,
                    max_tokens=max_tokens,)

                if response is None:
                    print("Failed to get a response from OpenAI. Try again.")
                    continue

                message = response.choices[0].message.content

                info = {
                    "prompt_tokens" : response.usage.prompt_tokens,
                    "completion_tokens" : response.usage.completion_tokens,
                    "total_tokens" : response.usage.total_tokens,
                }

                return message, info

            print(f"Failed to get a response from OpenAI after {MAX_RETRIES} retries.")
            return "", {}

        return _generate_response_with_retry(
            messages,
            model,
            temperature,
            seed,
            max_tokens,
        )

    def num_tokens_from_messages(self, messages, model = None) -> int:
        """Return the number of tokens used by a list of messages.
        Borrowed from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """

        model = self.provider_cfg[PROVIDER_SETTING_COMP_MODEL] if model is None else model

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        if model in {
            "gpt-4-1106-vision-preview",
        }:
            raise ValueError("We don't support counting tokens of GPT-4V yet.")

        if model in {
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            "gpt-4o-mini",
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = (
                4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            )
            tokens_per_name = -1  # if there's a name, the role is omitted
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if key == "content":
                    for content in value:
                        if content["type"] == "text":
                            num_tokens += len(encoding.encode(content["text"]))
                if key == "name":
                    num_tokens += tokens_per_name

        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

        return num_tokens


    def _get_azure_deployment_id_for_model(self, model_label) -> list:
        return self.provider_cfg[PROVIDER_SETTING_DEPLOYMENT_MAP][model_label]

    def assemble_prompt(self, system_prompts: List[str], user_inputs: List[str], image_filenames: List[str]) -> List[str]:
        encoded_images = [encode_image(image_path) for image_path in image_filenames]

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"{system_prompts[0]}"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{user_inputs[0]}"
                    },
                    # {
                    #     "type": "image_url",
                    #     "image_url":
                    #         {
                    #             "url": f"data:image/jpeg;base64,{encoded_images[0]}"
                    #         }
                    # }
                ]
            }
        ]

        for image in encoded_images:
            messages[1]["content"].append(
                {
                    "type": "image_url",
                    "image_url":
                        {
                            "url": f"data:image/jpeg;base64,{image}"
                        }
                },
            )

        return messages

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')