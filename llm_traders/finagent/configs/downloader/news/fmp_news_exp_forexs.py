root = None
workdir = "workdir"
tag = "fmp_news_exp_forexs"
batch_size = 1

downloader = dict(
    type = "FMPForexNewsDownloader",
    root = root,
    token = None,
    start_date = "1993-12-01",
    end_date = "2024-01-01",
    interval = "1d",
    delay = 1,
    stocks_path = "configs/_stock_list_/exp_forexs.txt",
    workdir = workdir,
    tag = tag
)