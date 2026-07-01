import pandas as pd

from llm_traders.finagent.plots import kline, trading


def test_default_plot_renderer_avoids_selenium(monkeypatch, tmp_path):
    monkeypatch.delenv("FINAGENT_PLOT_RENDERER", raising=False)

    def reject_pyecharts(*args, **kwargs):
        raise AssertionError("Pyecharts/Selenium renderer should not be called")

    monkeypatch.setattr(kline, "_plot_kline_pyecharts", reject_pyecharts)
    monkeypatch.setattr(trading, "_plot_trading_pyecharts", reject_pyecharts)

    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    prices = pd.DataFrame(
        {
            "open": [100, 101, 102, 101, 103],
            "high": [102, 103, 104, 104, 105],
            "low": [99, 100, 100, 100, 102],
            "close": [101, 102, 101, 103, 104],
            "volume": [1_000, 1_100, 1_200, 1_150, 1_300],
        },
        index=dates,
    )
    kline_path = tmp_path / "kline.jpeg"
    kline.plot_kline(
        prices,
        "Test K-line",
        str(kline_path),
        now_date="2024-01-08",
        mode="valid",
    )

    trading_path = tmp_path / "trading.jpeg"
    trading.plot_trading(
        {
            "date": dates,
            "price": [101, 102, 101, 103, 104],
            "total_profit": [0.0, 0.01, 0.0, 0.02, 0.03],
            "action": ["HOLD", "BUY", "HOLD", "SELL", "HOLD"],
        },
        str(trading_path),
    )

    assert kline_path.stat().st_size > 0
    assert trading_path.stat().st_size > 0
