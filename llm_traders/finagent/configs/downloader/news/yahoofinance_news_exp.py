root = None
workdir = "workdir"
tag = "yahoofinance_news_exp"
batch_size = 5

downloader = dict(
    type = "YahooFinanceNewsDownloader",
    root = root,
    token = None, # finnhub api key
    use_proxy = "",
    max_retry = 5,
    proxy_pages = 5,
    tunnel = None,
    username = None,
    password = None,
    start_date = "2022-01-01",
    end_date = "2024-01-01",
    interval = 30,
    delay = 1,
    stocks_path = "configs/_stock_list_/exp_stocks.txt",
    workdir = workdir,
    tag = tag
)