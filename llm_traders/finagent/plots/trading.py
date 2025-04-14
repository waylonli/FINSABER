from datetime import datetime
from snapshot_selenium import snapshot as driver
import pyecharts.options as opts
from pyecharts.charts import Line, Grid
from pyecharts.render import make_snapshot
import os

from pyecharts.globals import CurrentConfig
CurrentConfig.ONLINE_HOST = ""

def plot_trading(data,
                 save_path,
                 now_date = None,
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
        ))

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
