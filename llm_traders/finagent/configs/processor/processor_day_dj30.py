root = None
workdir = "workdir"
tag = "processd_day_dj30"
batch_size = 5

processor = dict(
    type = "Processor",
    root = root,
    path_params = {
        "prices": [
            {
                "type": "fmp",
                "path":"workdir/fmp_day_prices_dj30",
            }
        ],
        "news": [
            {
                "type": "fmp",
                "path":"workdir/fmp_news_dj30",
            }
        ],
        "tools": [
            {
                "type": "seekingalpha_analysis",
                "path":"workdir/rapidapi_day_seekingalpha_analysis_dj30",
            }
        ]
    },
    start_date = "2020-01-01",
    end_date = "2024-01-01",
    interval = "1d",
    stocks_path = "configs/_stock_list_/dj30.txt",
    workdir = workdir,
    tag = tag
)