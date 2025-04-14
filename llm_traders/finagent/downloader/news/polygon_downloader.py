import os
import pandas as pd
from llm_traders.finagent.registry import DOWNLOADER
from llm_traders.finagent.downloader.custom import Downloader
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

NYSE = get_calendar('XNYS')

@DOWNLOADER.register_module(force=True)
class PolygonNewsDownloader(Downloader):
    def __init__(self,
                 root: str = "",
                 token: str = None,
                 delay: int = 1,
                 start_date: str = "2023-04-01",
                 end_date: str = "2023-04-01",
                 stocks_path: str = None,
                 workdir: str = "",
                 tag: str = "",
                 **kwargs):

        self.root = root
        self.token = token if token is not None else os.environ.get("OA_POLYGON_KEY")
        self.delay = delay
        self.start_date = start_date
        self.end_date = end_date
        self.stocks_path = os.path.join(root, stocks_path)
        self.tag = tag
        self.workdir = os.path.join(root, workdir, tag)

        self.log_path = os.path.join(self.workdir, "{}.txt".format(tag))

        with open(self.log_path, "w") as op:
            op.write("")

        self.stocks = self._init_stocks()

        self.client = None

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

        total_count = 0
        total_days = 0

        failed_stocks = []

        for stock in stocks:
            count = 0
            stock_count = 0
            for date in pd.date_range(start_date, end_date, freq="D"):
                is_trading_day = NYSE.valid_days(start_date=date, end_date=date).size > 0

                if is_trading_day:
                    stock_count += 1
                    if os.path.exists(os.path.join(self.workdir, stock, "{}.csv".format(date.strftime("%Y-%m-%d")))):
                        count += 1
                else:
                    continue

            total_count += count
            total_days += stock_count

            if count != stock_count:
                failed_stocks.append(stock)

            # print("{}: {}/{}".format(stock, count, stock_count))

        print("Total: {}/{}, failed {}/{}".format(total_count, total_days, total_days- total_count, total_days))

        return failed_stocks

    def download(self,
                 stocks = None,
                 start_date = None,
                 end_date = None):

        start_date = datetime.strptime(start_date if start_date else self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date if end_date else self.end_date, "%Y-%m-%d")

        stocks = stocks if stocks else self.stocks

        for stock in stocks:

            os.makedirs(os.path.join(self.workdir, stock), exist_ok=True)

            self.client = RESTClient(self.token)

            df = pd.DataFrame()

            for date in tqdm(pd.date_range(start_date, end_date, freq="D"), bar_format="Download {} News:".format(stock) +
                                                                                       "{bar:50}{percentage:3.0f}%|{elapsed}/{remaining}{postfix}"):

                is_trading_day = NYSE.valid_days(start_date=date, end_date=date).size > 0

                if is_trading_day:
                    if os.path.exists(os.path.join(self.workdir, stock, "{}.csv".format(date.strftime("%Y-%m-%d")))):
                        chunk_df = pd.read_csv(os.path.join(self.workdir, stock, "{}.csv".format(date.strftime("%Y-%m-%d"))))
                    else:
                        chunk_df = {
                            "amp_url": [],
                            "article_url": [],
                            "author": [],
                            "description": [],
                            "id": [],
                            "image_url": [],
                            "keywords": [],
                            "timestamp": [],
                            "publisher_favicon_url":[],
                            "publisher_logo_url": [],
                            "publisher_homepage_url": [],
                            "publisher_name": [],
                            "tickers": [],
                            "title": [],
                        }

                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(60)

                        try:

                            time.sleep(self.delay)

                            aggs = self.client.list_ticker_news(
                                stock,
                                published_utc_lte=date.strftime("%Y-%m-%d"),
                                published_utc_gte=date.strftime("%Y-%m-%d"),
                            )

                            aggs = [item for item in aggs]

                            signal.alarm(0)
                        except TimeoutException:
                            print("Time out")
                            aggs = []
                        except Exception as e:
                            print(e)
                            aggs = []

                        if len(aggs) == 0:
                            with open(self.log_path, "a") as op:
                                op.write("{},{}\n".format(stock, date.strftime("%Y-%m-%d")))
                            continue

                        for a in aggs:

                            chunk_df["amp_url"].append(a.amp_url)
                            chunk_df["article_url"].append(a.article_url)
                            chunk_df["author"].append(a.author)
                            chunk_df["description"].append(a.description)
                            chunk_df["id"].append(a.id)
                            chunk_df["image_url"].append(a.image_url)
                            chunk_df["keywords"].append(a.keywords)
                            chunk_df["timestamp"].append(a.published_utc)
                            chunk_df["publisher_favicon_url"].append(a.publisher.favicon_url)
                            chunk_df["publisher_logo_url"].append(a.publisher.logo_url)
                            chunk_df["publisher_homepage_url"].append(a.publisher.homepage_url)
                            chunk_df["publisher_name"].append(a.publisher.name)
                            chunk_df["tickers"].append(a.tickers)
                            chunk_df["title"].append(a.title)

                        chunk_df = pd.DataFrame(chunk_df,index=range(len(chunk_df["timestamp"])))
                        chunk_df["timestamp"] = pd.to_datetime(chunk_df["timestamp"], utc=True).apply(
                            lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

                        chunk_df.to_csv(os.path.join(self.workdir, stock, "{}.csv".format(date.strftime("%Y-%m-%d"))), index=False)
                        time.sleep(self.delay)

                    df = pd.concat([df, chunk_df], axis=0)

            df.to_csv(os.path.join(self.workdir, "{}.csv".format(stock)), index=False)
