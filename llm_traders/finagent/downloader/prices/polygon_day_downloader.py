import os
import pandas as pd
from llm_traders.finagent.registry import DOWNLOADER
from llm_traders.finagent.downloader.custom import Downloader
from llm_traders.finagent.utils.misc import generate_intervals
from tqdm.auto import tqdm
from datetime import datetime
import time
from polygon import RESTClient
import signal
from pandas_market_calendars import get_calendar
from dotenv import load_dotenv

load_dotenv(verbose=True)

class TimeoutException(Exception):
    pass
def timeout_handler(signum, frame):
    raise TimeoutException("Time out")

@DOWNLOADER.register_module(force=True)
class PolygonDayPriceDownloader(Downloader):
    def __init__(self,
                 root: str = "",
                 token: str = None,
                 delay: int = 1,
                 start_date: str = "2023-04-01",
                 end_date: str = "2023-04-01",
                 interval: str = "day",
                 stocks_path: str = None,
                 workdir: str = "",
                 tag: str = "",
                 **kwargs):

        self.root = root
        self.token = token if token is not None else os.environ.get("OA_POLYGON_KEY")
        self.delay = delay
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.stocks_path = os.path.join(root, stocks_path)
        self.tag = tag
        self.workdir = os.path.join(root, workdir, tag)

        self.log_path = os.path.join(self.workdir, "{}.txt".format(tag))

        with open(self.log_path, "w") as op:
            op.write("")

        self.stocks = self._init_stocks()

        self.client = None

        self.nyse = get_calendar('XNYS')

        super().__init__(**kwargs)

    def _init_stocks(self):
        with open(self.stocks_path) as op:
            stocks = [line.strip() for line in op.readlines()]
        return stocks

    def check_download(self,
                       stocks = None,
                       start_date = None,
                       end_date = None):
        start_date = datetime.strptime(start_date if start_date else self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date if end_date else self.end_date, "%Y-%m-%d")
        stocks = stocks if stocks else self.stocks

        intervals = generate_intervals(start_date, end_date, "year")

        failed_stocks = []

        total_count = 0
        total_stock_count = 0

        for stock in stocks:
            count = 0
            stock_count = 0

            for (start, end) in intervals:
                if os.path.exists(os.path.join(self.workdir, stock, "{}.csv".format(start.strftime("%Y")))):
                    count += 1
                    total_count += 1
                stock_count += 1
                total_stock_count += 1

            if count != stock_count:
                failed_stocks.append(stock)

            print("{}: {}/{}".format(stock, count, stock_count))

        print("Total: {}/{}, failed {}/{}".format(total_count, total_stock_count, total_stock_count - total_count, total_stock_count))

        return failed_stocks

    def download(self,
                 stocks = None,
                 start_date = None,
                 end_date = None):

        start_date = datetime.strptime(start_date if start_date else self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date if end_date else self.end_date, "%Y-%m-%d")

        stocks = stocks if stocks else self.stocks

        intervals = generate_intervals(start_date, end_date, "year")

        for stock in stocks:

            os.makedirs(os.path.join(self.workdir, stock), exist_ok=True)

            self.client = RESTClient(self.token)

            df = pd.DataFrame()

            for (start, end) in tqdm(intervals, bar_format="Download {} Prices:".format(stock) + "{bar:50}{percentage:3.0f}%|{elapsed}/{remaining}{postfix}"):

                is_trading_day = self.nyse.valid_days(start_date=start, end_date=end).size > 0
                if is_trading_day:
                    if os.path.exists(os.path.join(self.workdir, stock, "{}.csv".format(start.strftime("%Y-%m-%d")))):
                        chunk_df = pd.read_csv(os.path.join(self.workdir, stock, "{}.csv".format(start.strftime("%Y-%m-%d"))))
                    else:
                        chunk_df = {
                            "open": [],
                            "high": [],
                            "low": [],
                            "close": [],
                            "volume": [],
                            "timestamp": [],
                            "vwap": [],
                            "transactions": [],
                        }

                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(60)

                        try:
                            aggs = self.client.get_aggs(
                                stock,
                                1,
                                self.interval,
                                start.strftime("%Y-%m-%d"),
                                end.strftime("%Y-%m-%d"),
                                limit=50000,
                            )
                            signal.alarm(0)
                        except TimeoutException:
                            print("Time out")
                            aggs = []

                        if len(aggs) == 0:
                            with open(self.log_path, "a") as op:
                                op.write("{},{}\n".format(stock, start.strftime("%Y-%m-%d")))
                            continue

                        for a in aggs:
                            chunk_df["open"].append(a.open)
                            chunk_df["high"].append(a.high)
                            chunk_df["low"].append(a.low)
                            chunk_df["close"].append(a.close)
                            chunk_df["volume"].append(a.volume)
                            chunk_df["timestamp"].append(a.timestamp)
                            chunk_df["vwap"].append(a.vwap)
                            chunk_df["transactions"].append(a.transactions)

                        chunk_df = pd.DataFrame(chunk_df,index=range(len(chunk_df["timestamp"])))
                        chunk_df["timestamp"] = pd.to_datetime(chunk_df["timestamp"], unit='ms').apply(
                            lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

                        chunk_df.to_csv(os.path.join(self.workdir, stock, "{}.csv".format(start.strftime("%Y-%m-%d"))), index=False)
                        time.sleep(self.delay)

                    df = pd.concat([df, chunk_df], axis=0)

            df.to_csv(os.path.join(self.workdir, "{}.csv".format(stock)), index=False)
