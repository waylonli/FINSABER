import json
import os
from dataclasses import asdict, is_dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd


METRIC_KEYS = {
    "final_value",
    "total_return",
    "annual_return",
    "annual_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "total_commission",
    "total_slippage",
    "total_llm_cost",
    "total_trading_cost",
}


DATAFRAME_FILENAMES = {
    "equity_with_time": "equity_curve.csv",
    "trades": "trades.csv",
    "executed_orders": "orders.csv",
    "rejected_orders": "rejected_orders.csv",
    "llm_cost_records": "llm_costs.csv",
}


def _json_safe(value):
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return value.__class__.__name__


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(_json_safe(data), file, indent=2)


def write_result_artifacts(output_dir, config, results):
    os.makedirs(output_dir, exist_ok=True)
    _write_json(os.path.join(output_dir, "run_config.json"), config)
    _write_json(
        os.path.join(output_dir, "run_manifest.json"),
        {
            "schema_version": 1,
            "generated_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "summary_file": "run_summary.csv",
            "leaf_metrics_file": "metrics.json",
            "dataframe_files": DATAFRAME_FILENAMES,
        },
    )
    summary = summarize_results(results)
    if not summary.empty:
        summary.to_csv(os.path.join(output_dir, "run_summary.csv"), index=False)
    _write_result_node(output_dir, results)


def summarize_results(results):
    rows = []
    _collect_summary_rows(results, path=(), rows=rows)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["path"]).reset_index(drop=True)


def _collect_summary_rows(node, path, rows):
    if not isinstance(node, dict):
        return
    if any(key in node for key in METRIC_KEYS):
        scalar_metrics = {
            key: _json_safe(value)
            for key, value in node.items()
            if key in METRIC_KEYS and not isinstance(value, pd.DataFrame)
        }
        row = {"path": "/".join(path)}
        if len(path) >= 1:
            row["ticker"] = path[-1]
        if len(path) >= 2:
            row["window"] = path[-2]
        row.update(scalar_metrics)
        rows.append(row)
        return
    for key, value in node.items():
        _collect_summary_rows(value, path + (str(key),), rows)


def _write_result_node(output_dir, node):
    if not isinstance(node, dict):
        return
    if any(key in node for key in METRIC_KEYS):
        _write_metric_leaf(output_dir, node)
        return
    for key, value in node.items():
        child_dir = os.path.join(output_dir, str(key).replace(":", "_"))
        os.makedirs(child_dir, exist_ok=True)
        _write_result_node(child_dir, value)


def _write_metric_leaf(output_dir, metrics):
    scalar_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, pd.DataFrame):
            filename = DATAFRAME_FILENAMES.get(key, f"{key}.csv")
            value.to_csv(os.path.join(output_dir, filename), index=False)
        elif key in METRIC_KEYS:
            scalar_metrics[key] = value
    _write_json(os.path.join(output_dir, "metrics.json"), scalar_metrics)
