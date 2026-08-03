import json
from pathlib import Path

import toml

from examples.experiments import run_finmem_finsaber2 as finmem_runner
from llm_traders.finmem.puppy.chat import ChatOpenAICompatible


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    REPO_ROOT
    / "examples"
    / "experiments"
    / "manifests"
    / "finmem_finsaber2_2024_2026.json"
)
CONFIG_PATHS = (
    REPO_ROOT / "strats_configs" / "finmem_gpt_config.toml",
    REPO_ROOT / "llm_traders" / "finmem" / "config" / "finmem_gpt_config.toml",
)


def test_finmem_sampling_config_reaches_chat_payload(monkeypatch):
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    configs = [toml.load(path) for path in CONFIG_PATHS]
    assert configs[0] == configs[1]

    chat_config = dict(configs[0]["chat"])
    assert manifest["llm_sampling"]["temperature"] == chat_config["temperature"]
    assert manifest["llm_sampling"]["request_seed"] == chat_config["seed"]
    endpoint = chat_config.pop("end_point")
    model = chat_config.pop("model")
    system_message = chat_config.pop("system_message")
    client = ChatOpenAICompatible(
        end_point=endpoint,
        model=model,
        system_message=system_message,
        other_parameters=chat_config,
    )
    payload = {}

    class Response:
        def raise_for_status(self):
            return None

    def fake_post(url, *, headers, data, timeout):
        payload.update(json.loads(data))
        return Response()

    monkeypatch.setattr("llm_traders.finmem.puppy.chat.httpx.post", fake_post)
    monkeypatch.setattr(client, "parse_response", lambda response: "{}")
    assert client.guardrail_endpoint()("test") == "{}"
    assert payload["temperature"] == 0.0
    assert payload["seed"] == 42


def test_finmem_runner_materializes_stable_artifact_run_key(tmp_path):
    manifest = finmem_runner.load_manifest(MANIFEST_PATH)
    config = finmem_runner._materialized_artifact_config(
        manifest,
        output_root=tmp_path,
        setup="random_sp500_5",
        window="2024-01-01_2025-01-01",
    )

    assert config["enabled"] is True
    assert config["run_key"] == "run_2024-01-01_2025-01-01"
