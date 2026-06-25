from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from snapshot_selenium import snapshot as driver
import pyecharts.options as opts
from pyecharts.charts import Line, Grid
from pyecharts.render import make_snapshot
import os

from pyecharts.globals import CurrentConfig
CurrentConfig.ONLINE_HOST = ""


def plot_trading(data,
                 save_path,
                 now_date=None,
                 width=3.5,
                 opacity=0.8,
                 path=None):
    try:
        _plot_trading_pyecharts(
            data,
            save_path,
            now_date=now_date,
            width=width,
            opacity=opacity,
            path=path,
        )
    except Exception as exc:
        warnings.warn(
            f"Pyecharts trading-chart rendering failed ({exc}); using matplotlib fallback.",
            RuntimeWarning,
        )
        _plot_trading_matplotlib(data, save_path, now_date=now_date)


def _plot_trading_pyecharts(data,
                            save_path,
                            now_date=None,
                            width=3.5,
                            opacity=0.8,
                            path=None):
    dates = data['date'][:-1]
    closing_prices = data['price'][:-1]
    returns = data['total_profit'][1:]
    actions = data['action'][:-1]

    min_y = min(closing_prices)
    max_y = max(closing_prices)
    delta = max_y - min_y
    lowerbound = round(min_y - delta * 0.1, 2)
    upperbound = round(max_y + delta * 0.1, 2)
    if delta > 5:
        lowerbound = int(lowerbound)
        upperbound = int(upperbound)

    markers = [
        opts.MarkPointItem(
            coord=[date, price - (delta * 0.08 if action == 'BUY' else 0)],
            value=action,
            symbol_size=45 if action == 'BUY' else 60,
            symbol="diamond" if action == 'BUY' else "pin",
            itemstyle_opts=opts.ItemStyleOpts(color="green" if action == 'BUY' else "red"),
        ) for date, price, action in zip(dates, closing_prices, actions) if action in ['BUY', 'SELL']
    ]
    if now_date:
        index = dates.index(now_date)
        closing_price_at_now_date = closing_prices[index]
        markers.append(
            opts.MarkPointItem(
                coord=[now_date, closing_price_at_now_date],
                value=now_date,
                symbol_size=120,
                symbol="pin",
                itemstyle_opts=opts.ItemStyleOpts(color="grey"),
            )
        )

    signal_line = (
        Line()
        .set_global_opts(
            tooltip_opts=opts.TooltipOpts(is_show=False),
            xaxis_opts=opts.AxisOpts(type_="category"),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                min_=lowerbound,
                max_=upperbound,
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
            legend_opts=opts.LegendOpts(orient="horizontal", pos_top="2%"),
        )
        .add_xaxis(xaxis_data=dates)
        .add_yaxis(
            series_name="Adj Close Prices",
            y_axis=closing_prices,
            symbol="emptyCircle",
            is_symbol_show=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=width, opacity=opacity),
            markpoint_opts=opts.MarkPointOpts(data=markers)
        )
    )

    return_line = (
        Line()
        .set_global_opts(
            tooltip_opts=opts.TooltipOpts(is_show=False),
            xaxis_opts=opts.AxisOpts(type_="category"),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(is_show=True),
                axislabel_opts=opts.LabelOpts(formatter="{value}%"),
            ),
            legend_opts=opts.LegendOpts(orient="horizontal", pos_top="55%"),
        )
        .add_xaxis(xaxis_data=dates)
        .add_yaxis(
            series_name="Cumulative Returns",
            y_axis=returns,
            symbol="emptyCircle",
            is_symbol_show=True,
            linestyle_opts=opts.LineStyleOpts(width=width, opacity=opacity),
            label_opts=opts.LabelOpts(is_show=False),
        )
    )

    grid_chart = Grid(
        init_opts=opts.InitOpts(
            width="1000px",
            height="800px",
            animation_opts=opts.AnimationOpts(animation=False),
            bg_color="white",
        )
    )
    grid_chart.add(
        signal_line,
        grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", height="40%"),
    )
    grid_chart.add(
        return_line,
        grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", pos_top="60%", height="35%"),
    )

    if not path:
        path = os.path.join(os.path.dirname(save_path), 'trading.html')

    make_snapshot(driver, grid_chart.render(path=path), save_path, is_remove_html=True)


def _plot_trading_matplotlib(data, save_path, now_date=None):
    dates = list(data.get("date", []))[:-1]
    prices = list(data.get("price", []))[:-1]
    returns = list(data.get("total_profit", []))[1:]
    actions = list(data.get("action", []))[:-1]
    length = min(len(dates), len(prices), len(actions))
    if length == 0:
        raise ValueError("No trading records available for trading chart.")

    dates = pd.to_datetime(dates[:length])
    prices = prices[:length]
    actions = actions[:length]
    returns = returns[:length] if returns else [0.0] * length

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, (price_ax, return_ax) = plt.subplots(2, 1, figsize=(10, 8), dpi=150, sharex=True)
    price_ax.plot(dates, prices, label="Adj Close", color="#1f77b4", linewidth=1.5)

    for date, price, action in zip(dates, prices, actions):
        if action == "BUY":
            price_ax.scatter(date, price, marker="D", color="green", s=45, zorder=3, label="BUY")
        elif action == "SELL":
            price_ax.scatter(date, price, marker="v", color="red", s=65, zorder=3, label="SELL")

    if now_date is not None:
        now_ts = pd.to_datetime(now_date)
        nearest = min(dates, key=lambda value: abs(value - now_ts))
        idx = list(dates).index(nearest)
        price_ax.scatter(nearest, prices[idx], marker="v", color="grey", s=90, zorder=4, label="Current")

    handles, labels = price_ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    price_ax.legend(unique.values(), unique.keys(), loc="best")
    price_ax.set_ylabel("Adjusted Price")
    price_ax.grid(alpha=0.25)

    return_ax.plot(dates[:len(returns)], returns, label="Cumulative Return", color="#ff7f0e", linewidth=1.5)
    return_ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    return_ax.set_ylabel("Return (%)")
    return_ax.set_xlabel("Date")
    return_ax.grid(alpha=0.25)
    return_ax.legend(loc="best")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
