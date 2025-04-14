root = None
workdir = "workdir"
tag = "fmp_day_prices_sp500"
batch_size = 100

downloader = dict(
    type = "FMPDayPriceDownloader",
    root = root,
    token = None,
    start_date = "1993-12-01",
    end_date = "2023-12-01",
    interval = "1d",
    delay = 1,
    stocks_path = "configs/_stock_list_/sp500.txt",
    workdir = workdir,
    tag = tag
)