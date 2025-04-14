root = None
workdir = "workdir"
tag = "processd_day_exp_stocks"
batch_size = 5

processor = dict(
    type = "Processor",
    root = root,
    path_params = {
        "prices": [
            {
                "type": "fmp",
                "path":"workdir/fmp_day_prices_exp",
            }
        ],
        "news": [
            {
                "type": "fmp",
                "path":"workdir/fmp_news_exp",
            },
            {
                "type": "yahoofinance",
                "path":"workdir/yahoofinance_news_exp",
            }
        ],
        "guidance": [
            {
                "type": "rapidapi_seekingalpha",
                "path":"workdir/rapidapi_seekingalpha_analysis_exp",
            }
        ],
        "sentiment": [
            {
                "type": "fmp",
                "path":"workdir/fmp_sentiment_exp",
            }
        ],
        "economic": [
            {
                "type": "fmp",
                "path":"workdir/fmp_economic_exp",
            }
        ],
    },
    start_date = "2020-01-01",
    end_date = "2024-01-01",
    interval = "1d",
    if_parse_url = False,
    stocks_path = "configs/_asset_list_/exp_stocks.txt",
    workdir = workdir,
    tag = tag
)