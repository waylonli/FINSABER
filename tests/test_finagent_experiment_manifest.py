import json
from pathlib import Path


MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "experiments"
    / "manifests"
    / "finagent_finsaber2_2024_2026.json"
)


def test_finagent_manifest_has_reproducible_magnificent_7_snapshot():
    config = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {"AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"}
    windows = set(config["windows"])
    selections = config["selections"]

    assert config["model"] == "gpt-4o-mini"
    assert config["seed"] == 2026
    assert config["evaluation"]["execution_timing"] == "next_open"
    assert config["evaluation"]["training_years"] == 3
    assert set(selections["magnificent_7"]) == windows

    for tickers in selections["magnificent_7"].values():
        assert set(tickers) == expected
        assert len(tickers) == len(expected)

    expected_jobs = sum(
        len(tickers)
        for setup in selections.values()
        for tickers in setup.values()
    )
    assert expected_jobs == 54
