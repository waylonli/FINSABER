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
class FMPSentimentDownloader(Downloader):
    def __init__(self,
                 root: str = "",
                 token: str = None,
                 delay: int = 1,
                 start_date: str = "2023-04-01",
                 end_date: str = "2023-04-01",
                 interval: str = "minute",
                 stocks_path: str = None,
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

        self.max_pages = 250

        self.log_path = os.path.join(self.workdir, "{}.txt".format(tag))

        with open(self.log_path, "w") as op:
            op.write("")

        self.stocks = self._init_stocks()

        # https://financialmodelingprep.com/api/v4/historical/social-sentiment?symbol=AAPL&page=0&apikey=[token]
        self.request_url = "https://financialmodelingprep.com/api/v4/historical/social-sentiment?symbol={}&page={}&apikey={}"

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

        for stock in stocks:

            os.makedirs(os.path.join(self.workdir, stock), exist_ok=True)

            df = pd.DataFrame()

            for page in tqdm(range(self.max_pages), bar_format="Download {} Sentiment:".format(stock) + "{bar:50}{percentage:3.0f}%|{elapsed}/{remaining}{postfix}"):
                if os.path.exists(os.path.join(self.workdir, stock, "page{:04d}.csv".format(page))):
                    chunk_df = pd.read_csv(os.path.join(self.workdir, stock, "page{:04d}.csv".format(page)))
                else:

                    """
                    [
                        {
                            "date": "2022-06-30 23:00:00",
                            "symbol": "AAPL",
                            "stocktwitsPosts": 13,
                            "twitterPosts": 163,
                            "stocktwitsComments": 9,
                            "twitterComments": 7769,
                            "stocktwitsLikes": 16,
                            "twitterLikes": 40957,
                            "stocktwitsImpressions": 15141,
                            "twitterImpressions": 1576854,
                            "stocktwitsSentiment": 0.5411,
                            "twitterSentiment": 0.5888
                        }
                    ]
                    """

                    chunk_df = {
                        "timestamp": [],
                        "stocktwits_posts": [],
                        "twitter_posts": [],
                        "stocktwits_comments": [],
                        "twitter_comments": [],
                        "stocktwits_likes": [],
                        "twitter_likes": [],
                        "stocktwits_impressions": [],
                        "twitter_impressions": [],
                        "stocktwits_sentiment": [],
                        "twitter_sentiment": []
                    }

                    request_url = self.request_url.format(stock, page, self.token)

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
                            op.write("{},{}\n".format(stock, page))
                        continue

                    for a in aggs:
                        chunk_df["timestamp"].append(a["date"])
                        chunk_df["stocktwits_posts"].append(a["stocktwitsPosts"])
                        chunk_df["twitter_posts"].append(a["twitterPosts"])
                        chunk_df["stocktwits_comments"].append(a["stocktwitsComments"])
                        chunk_df["twitter_comments"].append(a["twitterComments"])
                        chunk_df["stocktwits_likes"].append(a["stocktwitsLikes"])
                        chunk_df["twitter_likes"].append(a["twitterLikes"])
                        chunk_df["stocktwits_impressions"].append(a["stocktwitsImpressions"])
                        chunk_df["twitter_impressions"].append(a["twitterImpressions"])
                        chunk_df["stocktwits_sentiment"].append(a["stocktwitsSentiment"])
                        chunk_df["twitter_sentiment"].append(a["twitterSentiment"])

                    chunk_df = pd.DataFrame(chunk_df,index=range(len(chunk_df["timestamp"])))
                    chunk_df["timestamp"] = pd.to_datetime(chunk_df["timestamp"]).apply(
                        lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

                    chunk_df.to_csv(os.path.join(self.workdir, stock, "page{:04d}.csv".format(page)), index=False)

                df = pd.concat([df, chunk_df], axis=0)

            df.to_csv(os.path.join(self.workdir, "{}.csv".format(stock)), index=False)