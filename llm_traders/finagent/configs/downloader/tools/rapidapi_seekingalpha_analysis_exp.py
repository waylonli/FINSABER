root = None
workdir = "workdir"
tag = "rapidapi_seekingalpha_analysis_exp"
batch_size = 1

downloader = dict(
    type = "RapidAPIDownloader",
    root = root,
    token = None,
    start_date = "2022-01-01",
    end_date = "2024-01-01",
    interval = "day",
    delay = 2,
    stocks_path = "configs/_stock_list_/exp_stocks.txt",
    workdir = workdir,
    tag = tag,
)