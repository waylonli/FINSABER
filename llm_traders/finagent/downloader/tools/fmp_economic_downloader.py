import os
import pandas as pd
from llm_traders.finagent.registry import DOWNLOADER
from llm_traders.finagent.downloader.custom import Downloader
from tqdm.auto import tqdm
import time
import signal
from pandas_market_calendars import get_calendar
from urllib.request import urlopen
import certifi
import json
from dotenv import load_dotenv

load_dotenv(verbose=True)

class TimeoutException(Exception):
    pass
def timeout_handler(signum, frame):
    raise TimeoutException("Time out")

def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

NYSE = get_calendar('XNYS')

@DOWNLOADER.register_module(force=True)
class FMPEconomicDownloader(Downloader):
    def __init__(self,
                 root: str = "",
                 token: str = None,
                 delay: int = 1,
                 start_date: str = "2023-04-01",
                 end_date: str = "2023-04-01",
                 interval: str = "minute",
                 stocks_path: str = None,
                 indicators: list = None,
                 workdir: str = "",
                 tag: str = "",
                 **kwargs):

        self.root = root
        self.token = token if token is not None else os.environ.get("OA_FMP_KEY")
        print(self.token)
        self.delay = delay
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.stocks_path = os.path.join(root, stocks_path)
        self.tag = tag
        self.workdir = os.path.join(root, workdir, tag)
        self.indicators = indicators

        self.log_path = os.path.join(self.workdir, "{}.txt".format(tag))

        with open(self.log_path, "w") as op:
            op.write("")

        self.stocks = self._init_stocks()

        # https://financialmodelingprep.com/api/v4/historical/social-sentiment?symbol=AAPL&page=0&apikey=[token]
        self.request_url = "https://financialmodelingprep.com/api/v4/economic?name={}&apikey={}"

        super().__init__(**kwargs)

    def _init_stocks(self):
        with open(self.stocks_path) as op:
            stocks = [line.strip() for line in op.readlines()]
        return stocks

    def check_download(self):
        failed_stocks = self.stocks
        return failed_stocks

    def download(self,
                 stocks = None,
                 start_date = None,
                 end_date = None):

            os.makedirs(os.path.join(self.workdir), exist_ok=True)

            for indicator in tqdm(self.indicators, bar_format="Download Economic:" + "{bar:50}{percentage:3.0f}%|{elapsed}/{remaining}{postfix}"):
                if os.path.exists(os.path.join(self.workdir, "{}.csv".format(indicator))):
                    chunk_df = pd.read_csv(os.path.join(self.workdir, "{}.csv".format(indicator)))
                else:

                    """
                    [
                        {
                            "date": "2022-04-01",
                            "value": "24882.878"
                        },
                        {
                            "date": "2022-01-01",
                            "value": "24386.734"
                        }
                    ]
                    """

                    chunk_df = {
                        "timestamp": [],
                        indicator: []
                    }

                    request_url = self.request_url.format(indicator, self.token)

                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(60)

                    try:
                        time.sleep(self.delay)
                        aggs = get_jsonparsed_data(request_url)
                        signal.alarm(0)
                    except TimeoutException:
                        print("Time out")
                        aggs = []

                    if len(aggs) == 0:
                        with open(self.log_path, "a") as op:
                            op.write("{},{}\n".format(stock, indicator))
                        continue

                    for a in aggs:
                        chunk_df["timestamp"].append(a["date"])
                        chunk_df[indicator].append(a["value"])

                    chunk_df = pd.DataFrame(chunk_df, index=range(len(chunk_df["timestamp"])))
                    chunk_df["timestamp"] = pd.to_datetime(chunk_df["timestamp"]).apply(lambda x: x.strftime("%Y-%m-%d"))

                    chunk_df.to_csv(os.path.join(self.workdir, "{}.csv".format(indicator)), index=False)