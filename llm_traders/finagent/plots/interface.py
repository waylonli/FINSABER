import os

import pandas as pd

from llm_traders.finagent.registry import PLOTS
from llm_traders.finagent.plots import plot_kline
from llm_traders.finagent.plots import plot_trading
from llm_traders.finagent.utils import init_path
import shutil

@PLOTS.register_module(force=True)
class PlotsInterface():
    def __init__(self,
                 root = None,
                 workdir = None,
                 tag = None,
                 suffix = 'jpeg') -> None:
        super(PlotsInterface, self).__init__()
        self.root = root
        self.workdir = workdir
        self.tag = tag
        self.suffix = suffix

        self.exp_path = init_path(os.path.join(self.root, self.workdir, self.tag))
        self.plot_path = init_path(os.path.join(self.exp_path, "plots"))
        self.kline_plot_path = init_path(os.path.join(self.plot_path, "kline"))
        self.trading_plot_path = init_path(os.path.join(self.plot_path, "trading"))

        self.echarts_js_path = os.path.join(self.root, "tools", "echarts-5.4.3" , "dist", "echarts.min.js")

    def plot_kline(self, state, info, save_dir, mode = "train"):

        try:
            price = state["price"]

            kline_dir = init_path(os.path.join(self.kline_plot_path, save_dir))

            if not os.path.exists(os.path.join(kline_dir, "echarts.min.js")):
                shutil.copy(self.echarts_js_path, kline_dir)

            price = price[["open", "high", "low", "close", "volume"]]
            price = price.reset_index(drop=False)
            price = price.dropna(axis=0, how="any")
            price = price.drop_duplicates(subset=["timestamp"], keep="first")
            price = price.set_index("timestamp")

            title = "{} kline of {}".format(info["date"], info["symbol"])
            kline_path = os.path.join(kline_dir, "kline_{}.{}".format(info["date"], self.suffix))

            now_date = pd.to_datetime(info["date"])
            now_date = min(price.index, key=lambda x: abs(x - now_date)) # find the nearest date before now_date
            now_date = now_date.strftime("%Y-%m-%d")

            plot_kline(price,
                       title,
                       kline_path,
                       now_date=now_date,
                       path=os.path.join(kline_dir, f"{info['date']}_{self.suffix}_kline_render.html"),
                       mode=mode)

        except Exception as e:
            # print(e)
            kline_path = None
        return kline_path

    def plot_trading(self, records, info, save_dir):
        try:

            trading_dir = init_path(os.path.join(self.trading_plot_path, save_dir))

            if not os.path.exists(os.path.join(trading_dir, "echarts.min.js")):
                shutil.copy(self.echarts_js_path, trading_dir)

            trading_path = os.path.join(trading_dir, "trading_{}.{}".format(info['date'], self.suffix))

            plot_trading(records, trading_path, path=os.path.join(trading_dir, f"{info['date']}_{self.suffix}_trading_render.html"))

        except Exception as e:
            # print(e)
            trading_path = None

        return trading_path