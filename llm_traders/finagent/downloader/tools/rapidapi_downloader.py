import os.path
import pandas as pd
from datetime import datetime, timedelta
from tqdm.auto import tqdm
import time
from pandas_market_calendars import get_calendar
from dotenv import load_dotenv
import pytz

load_dotenv(verbose=True)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

from llm_traders.finagent.registry import DOWNLOADER
from llm_traders.finagent.downloader.custom import Downloader
from llm_traders.finagent.utils.misc import generate_intervals
from llm_traders.finagent.tools import RapidAPIs

NYSE = get_calendar('XNYS')
timezone_edt = pytz.timezone("America/New_York")

@DOWNLOADER.register_module()
class RapidAPIDownloader(Downloader):
    def __init__(self,
                 *args,
                 root: str = "",
                 token: str = "",
                 use_proxy: str = None,
                 max_retry: int = 5,
                 proxy_pages: int = 5,
                 tunnel: str = None,
                 username: str = None,
                 password: str = None,
                 start_date: str = "2022-04-01",
                 end_date: str = "2023-04-01",
                 interval: int = 30,
                 stocks_path: str = None,
                 workdir: str = "",
                 tag: str = "",
                 **kwargs) -> None:

        super(RapidAPIDownloader, self).__init__(use_proxy=use_proxy,
                                                     max_retry=max_retry,
                                                     proxy_pages=proxy_pages,
                                                     tunnel = tunnel,
                                                     username = username,
                                                     password = password)

        self.root = root
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.stocks_path = os.path.join(root, stocks_path)
        self.tag = tag

        self.stocks = self._init_stocks()
        self.workdir=os.path.join(root, workdir, tag)

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

        intervals = generate_intervals(start_date, end_date, "day")

        failed_stocks = []

        total_count = 0
        total_stock_count = 0

        for stock in stocks:
            count = 0
            stock_count = 0

            for (start, end) in intervals:
                if os.path.exists(os.path.join(self.workdir, stock, "{}.csv".format(start.strftime("%Y-%m-%d")))):
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
                 stocks: str = None,
                 start_date: str = None,
                 end_date: str = None,
                 ) -> None:

        print(f"Downloading expert opinions from via RapidAPIs with tools...")
        try:
            api= RapidAPIs()
        except:
            print("Failed to initialize RapidAPIs")


        start_date = datetime.strptime(start_date if start_date else self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date if end_date else self.end_date, "%Y-%m-%d")

        stocks = stocks if stocks else self.stocks

        intervals = generate_intervals(start_date, end_date, "day")

        for stock in stocks:
            os.makedirs(os.path.join(self.workdir, stock), exist_ok=True)

            df = pd.DataFrame()

            for (start, end) in tqdm(intervals, bar_format="Download {}:".format(stock) +"{bar:50}{percentage:3.0f}%|{elapsed}/{remaining}{postfix}"):
                if os.path.exists(os.path.join(self.workdir, stock, "{}.csv".format(start.strftime("%Y-%m-%d")))):
                    chunk_df = pd.read_csv(os.path.join(self.workdir, stock, "{}.csv".format(start.strftime("%Y-%m-%d"))))
                else:
                    #convert datetime to timestamp as EDT time
                    start = timezone_edt.localize(start)
                    start_timestamp = start.timestamp()
                    end = timezone_edt.localize(end)
                    end_timestamp = end.timestamp()

                    respond = api.get_seekingAlpha_analysis(stock, start_timestamp, end_timestamp)

                    chunk_df = pd.DataFrame(respond)

                    time.sleep(3)

                    chunk_df["timestamp"] = pd.to_datetime(chunk_df["timestamp"]).apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
                    chunk_df.to_csv(os.path.join(self.workdir, stock, "{}.csv".format(start.strftime("%Y-%m-%d"))), index=False)

                df = pd.concat([df, chunk_df])

            df.to_csv(os.path.join(self.workdir, "{}.csv".format(stock)), index = False)
