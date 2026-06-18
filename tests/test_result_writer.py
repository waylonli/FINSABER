import json

import pandas as pd

from backtest.toolkit.result_writer import summarize_results, write_result_artifacts


def _sample_results():
    return {
        "2024-01-01_2024-02-01": {
            "AAA": {
                "total_return": 0.1,
                "annual_return": 0.2,
                "sharpe_ratio": 1.5,
                "total_commission": 1.23,
                "total_external_cost": 0.01,
                "total_trading_cost": 1.23,
                "equity_with_time": pd.DataFrame(
                    {"datetime": [pd.Timestamp("2024-01-02")], "equity": [101000.0]}
                ),
                "trades": pd.DataFrame({"ticker": ["AAA"], "quantity": [10]}),
                "external_costs": pd.DataFrame({"reason": ["llm_inference_cost"], "cost": [0.01]}),
                "llm_cost_records": pd.DataFrame({"model": ["gpt-4o-mini"], "cost": [0.01]}),
            }
        }
    }


def test_summarize_results_flattens_metric_leaves():
    summary = summarize_results(_sample_results())

    assert list(summary["path"]) == ["2024-01-01_2024-02-01/AAA"]
    assert summary.iloc[0]["window"] == "2024-01-01_2024-02-01"
    assert summary.iloc[0]["ticker"] == "AAA"
    assert summary.iloc[0]["total_return"] == 0.1


def test_write_result_artifacts_uses_stable_output_schema(tmp_path):
    write_result_artifacts(tmp_path, {"setup_name": "unit"}, _sample_results())

    assert (tmp_path / "run_config.json").exists()
    assert (tmp_path / "run_manifest.json").exists()
    assert (tmp_path / "run_summary.csv").exists()

    leaf = tmp_path / "2024-01-01_2024-02-01" / "AAA"
    assert json.loads((leaf / "metrics.json").read_text())["total_return"] == 0.1
    assert json.loads((leaf / "metrics.json").read_text())["total_external_cost"] == 0.01
    assert (leaf / "equity_curve.csv").exists()
    assert (leaf / "trades.csv").exists()
    assert (leaf / "llm_costs.csv").exists()
    assert (leaf / "external_costs.csv").exists()
