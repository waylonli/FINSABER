root = None
workdir = "workdir"
tag = "polygon_news_dj30"
batch_size = 4

downloader = dict(
    type = "PolygonNewsDownloader",
    root = root,
    token = "",
    start_date = "2023-01-01",
    end_date = "2023-12-01",
    stocks_path = "configs/_stock_list_/dj30.txt",
    workdir = workdir,
    tag = tag
)