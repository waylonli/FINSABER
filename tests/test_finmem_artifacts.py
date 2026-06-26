import json
from datetime import date
from pathlib import Path

from llm_traders.finsaber_strategies.finmem_artifacts import (
    ArtifactWindow,
    FinMemArtifactWriter,
)


class _DummyCheckpointObject:
    reflection_result_series_dict = {}
    query_trace_series_dict = {}
    llm_trace_series_dict = {}

    def save_checkpoint(self, path: str, force: bool = False) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        Path(path, "checkpoint.txt").write_text("ok", encoding="utf-8")


def _artifact_writer(tmp_path: Path) -> FinMemArtifactWriter:
    train_window = ArtifactWindow(
        requested_start=date(2024, 1, 2),
        requested_end=date(2024, 1, 5),
        effective_start=date(2024, 1, 2),
        effective_end=date(2024, 1, 5),
    )
    test_window = ArtifactWindow(
        requested_start=date(2024, 1, 8),
        requested_end=date(2024, 1, 10),
        effective_start=date(2024, 1, 8),
        effective_end=date(2024, 1, 10),
    )
    return FinMemArtifactWriter(
        artifact_config={
            "enabled": True,
            "root": str(tmp_path),
            "save_agent_checkpoint": False,
            "save_environment_checkpoint": False,
            "save_reflections": False,
            "save_query_trace": True,
            "save_llm_trace": True,
        },
        symbol="AAA",
        config_path="strats_configs/finmem_gpt_config.toml",
        resolved_strategy_params={"symbol": "AAA"},
        finmem_config={"general": {"trading_symbol": "AAA"}},
        requested_train_window=train_window,
        requested_test_window=test_window,
        input_data_loader=None,
        runtime_market_data=None,
        filing_options={},
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_streamed_traces_survive_phase_snapshots(tmp_path):
    writer = _artifact_writer(tmp_path)
    train_payload = {
        "run_mode": "train",
        "date": "2024-01-03",
        "symbol": "AAA",
        "payload_kind": "query",
    }
    test_payload = {
        "run_mode": "test",
        "date": "2024-01-09",
        "symbol": "AAA",
        "payload_kind": "llm",
    }

    writer.append_query_trace(train_payload)
    writer.append_llm_trace(test_payload)

    dummy_agent = _DummyCheckpointObject()
    dummy_env = _DummyCheckpointObject()
    writer.save_post_train(agent=dummy_agent, environment=dummy_env)
    writer.save_test_state(
        agent=dummy_agent,
        environment=dummy_env,
        capture_stage="strategy_done_pre_finalization",
        snapshot_reason="strategy_done",
        framework_status=True,
    )

    ticker_dir = Path(writer._ticker_dir())
    train_query_rows = _read_jsonl(ticker_dir / "train_query_trace.jsonl")
    test_llm_rows = _read_jsonl(ticker_dir / "test_llm_trace.jsonl")

    assert len(train_query_rows) == 1
    assert train_query_rows[0]["phase"] == "train"
    assert train_query_rows[0]["date"] == "2024-01-03"
    assert train_query_rows[0]["payload"]["payload_kind"] == "query"

    assert len(test_llm_rows) == 1
    assert test_llm_rows[0]["phase"] == "test"
    assert test_llm_rows[0]["date"] == "2024-01-09"
    assert test_llm_rows[0]["payload"]["payload_kind"] == "llm"
