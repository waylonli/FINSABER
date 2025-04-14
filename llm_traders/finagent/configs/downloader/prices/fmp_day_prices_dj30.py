root = None
workdir = "workdir"
tag = "fmp_day_prices_dj30"
batch_size = 5

downloader = dict(
    type = "FMPDayPriceDownloader",
    root = root,
    token = None,
    start_date = "1993-12-01",
    end_date = "2024-01-01",
    interval = "1d",
    delay = 1,
    stocks_path = "configs/_stock_list_/dj30.txt",
    workdir = workdir,
    tag = tag
)