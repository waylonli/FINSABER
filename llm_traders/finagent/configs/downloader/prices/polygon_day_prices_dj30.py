root = None
workdir = "workdir"
tag = "polygon_day_prices_dj30"
batch_size = 5

downloader = dict(
    type = "PolygonDayPriceDownloader",
    root = root,
    token = None,
    start_date = "2018-10-25",
    end_date = "2023-08-24",
    interval = "day",
    delay = 2,
    stocks_path = "configs/_stock_list_/dj30.txt",
    workdir = workdir,
    tag = tag
)