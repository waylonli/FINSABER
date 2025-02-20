import datetime
import os
import backtrader as bt
import datasets as ds
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import pandas_datareader.data as web
import pickle
load_dotenv()
HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")
import pwb_toolbox.datasets as pwb_ds


def get_tickers_price(
    tickers: list[str] | str,
    date_from: str = "2000-01-01",
    date_to: str = "2024-01-01",
    return_original: bool = False
) -> pd.DataFrame:
    """
    Get daily price and technical indicators for specified tickers within the given date range.
    Technical indicators added:
      - EMA_9
      - SMA_5, SMA_10, SMA_15, SMA_30
      - RSI
      - MACD (plus signal and histogram)

    Note: To obtain SMA_30 correctly from the first day, we load 30 days before `date_from` and then filter them out.
    """
    # Extend the start date to fetch enough data for 30-day calculations
    extended_date_from = (pd.to_datetime(date_from) - datetime.timedelta(days=30)).strftime("%Y-%m-%d")

    # Load data from your source
    if isinstance(tickers, list):
        df = pwb_ds.load_dataset("Stocks-Daily-Price", tickers, adjust=True)
    elif tickers == "all":
        df = pwb_ds.load_dataset("Stocks-Daily-Price", adjust=True)
    else:
        df = pwb_ds.load_dataset("Stocks-Daily-Price", [tickers], adjust=True)

    # Filter by extended date range
    df = df[(df["date"] >= pd.to_datetime(extended_date_from)) & (df["date"] < pd.to_datetime(date_to))]
    df = df.sort_values(["symbol", "date"])

    # Function to compute indicators on a per-symbol basis
    def compute_indicators(g):
        g = g.sort_values("date")

        # Simple moving averages
        g["SMA_5"] = g["close"].rolling(5).mean()
        g["SMA_10"] = g["close"].rolling(10).mean()
        g["SMA_15"] = g["close"].rolling(15).mean()
        g["SMA_30"] = g["close"].rolling(30).mean()

        # Exponential moving average
        g["EMA_9"] = g["close"].ewm(span=9, adjust=False).mean()

        # RSI
        delta = g["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        g["RSI"] = 100 - (100 / (1 + rs))

        # MACD
        short_ema = g["close"].ewm(span=12, adjust=False).mean()
        long_ema = g["close"].ewm(span=26, adjust=False).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        g["MACD"] = macd_line
        g["MACD_signal"] = signal_line
        g["MACD_hist"] = macd_line - signal_line

        return g

    # Apply indicators per symbol
    df = df.groupby("symbol", group_keys=False).apply(compute_indicators)

    # Filter out rows earlier than original date_from
    df = df[df["date"] >= pd.to_datetime(date_from)]

    # if df becomes empty, return None
    if df.empty:
        return None

    # Pivot to have open, high, low, close, and indicators in columns
    pivot_df = pd.pivot_table(
        df,
        index="date",
        columns="symbol",
        values=[
            "open", "high", "low", "close",
            "SMA_5", "SMA_10", "SMA_15", "SMA_30",
            "EMA_9", "RSI",
            "MACD", "MACD_signal", "MACD_hist"
        ],
        aggfunc="first",
    )

    # Reindex to ensure all dates are represented
    try:
        full_date_range = pd.date_range(
            start=df["date"].min(),
            end=df["date"].max(),
            freq="D"
        )
        pivot_df = pivot_df.reindex(full_date_range)
    except Exception:
        return None

    return pivot_df if not return_original else df


def add_tickers_data(cerebro: bt.Cerebro, pivot_df: pd.DataFrame):
    datas = []
    # Process data and add to Cerebro
    for symbol in pivot_df.columns.levels[1]:
        symbol_df = pivot_df.xs(symbol, axis=1, level=1, drop_level=False).copy()
        symbol_df.columns = symbol_df.columns.droplevel(1)
        symbol_df.reset_index(inplace=True)
        symbol_df.rename(columns={"index": "date"}, inplace=True)
        symbol_df.set_index("date", inplace=True)
        symbol_df.ffill(inplace=True)
        symbol_df.bfill(inplace=True)

        data = bt.feeds.PandasData(dataname=symbol_df)
        datas.append((symbol, data))

        if cerebro is not None:
            cerebro.adddata(data, name=symbol)

    return datas

def process_for_ff(df: pd.DataFrame):
    """
    Process the DataFrame to include the Fama-French Five-Factor data
    :param df: The DataFrame containing the price of the specified tickers within the specified date range
    :return: The DataFrame with the Fama-French Five-Factor data included
    """
    # set date as the index if it is not
    if "date" in df.columns:
        df.set_index("date", inplace=True)
        # parse the date
        df.index = pd.to_datetime(df.index)

    # Calculate daily returns based on 'Adj Close' prices
    df['return'] = df['adj_close'].pct_change()

    # Get the start and end dates
    start_date = df.index.min()
    end_date = df.index.max()

    # Convert to datetime if necessary
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Fetch the Five-Factor data
    ff_factors = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench', start=start_date, end=end_date)[
        0]

    # Adjust the index to datetime format and align with your data
    ff_factors.index = pd.to_datetime(ff_factors.index)

    # Convert the percentage returns to decimal format
    ff_factors = ff_factors / 100

    # Merge the Fama-French factors into your DataFrame
    df = df.merge(ff_factors[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']], left_index=True, right_index=True, how='left')

    # Handle any missing values (if any)
    df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']] = df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].fillna(method='ffill')

    return df


def get_indices_data():
    dataset = ds.load_dataset(
        "paperswithbacktest/Indices-Daily-Price", token=HF_ACCESS_TOKEN
    )
    df = dataset["train"].to_pandas()
    # print(df["symbol"].unique())
    # symbols_to_keep = [
    #     "SPX",  # United States
    # ]
    # df = df[df["symbol"].isin(symbols_to_keep)]
    adj_factor = df["adj_close"] / df["close"]
    df["adj_open"] = df["open"] * adj_factor
    df["adj_high"] = df["high"] * adj_factor
    df["adj_low"] = df["low"] * adj_factor
    df = df[["date", "symbol", "adj_open", "adj_high", "adj_low", "adj_close"]]
    df.rename(
        columns={
            "adj_open": "open",
            "adj_high": "high",
            "adj_low": "low",
            "adj_close": "close",
        },
        inplace=True,
    )
    columns_to_keep = [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
    ]
    df = df[columns_to_keep]
    return df


def get_etfs_data():
    dataset = ds.load_dataset(
        "paperswithbacktest/ETFs-Daily-Price", token=HF_ACCESS_TOKEN, cache_dir="data/hf/"
    )
    df = dataset["train"].to_pandas()
    symbols_to_keep = ["SPY", "IEF"]
    df = df[df["symbol"].isin(symbols_to_keep)].copy()
    adj_factor = df["adj_close"] / df["close"]
    df["adj_open"] = df["open"] * adj_factor
    df["adj_high"] = df["high"] * adj_factor
    df["adj_low"] = df["low"] * adj_factor
    df = df[["date", "symbol", "adj_open", "adj_high", "adj_low", "adj_close"]]
    df.rename(
        columns={
            "adj_open": "open",
            "adj_high": "high",
            "adj_low": "low",
            "adj_close": "close",
        },
        inplace=True,
    )
    columns_to_keep = [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
    ]
    df = df[columns_to_keep]
    return df


def replace_index_with_etfs(df):
    mapping_etfs = {
        "SPY": "SPX",
    }
    frames = []
    for etf, index in mapping_etfs.items():
        # Get the ETF & Index data
        etf_data = df[df["symbol"] == etf]
        if etf_data.empty:
            raise ValueError(f"Data not found for {etf}")

        index_data = df[df["symbol"] == index]
        if index_data.empty:
            raise ValueError(f"Data not found for {index}")

        # Find the first overlapping date
        common_dates = etf_data["date"].isin(index_data["date"])
        first_common_date = etf_data.loc[common_dates, "date"].min()

        if pd.isnull(first_common_date):
            raise ValueError(f"No common date found for {etf} and {index}")

        etf_first_common = etf_data[etf_data["date"] == first_common_date]
        index_first_common = index_data[index_data["date"] == first_common_date]

        # Compute the adjustment factor (using closing prices for simplicity)
        adjustment_factor = (
            etf_first_common["close"].values[0] / index_first_common["close"].values[0]
        )

        # Adjust index data before the first common date
        index_data_before_common = index_data[
            index_data["date"] < first_common_date
        ].copy()
        for column in ["open", "high", "low", "close"]:
            index_data_before_common.loc[:, column] *= adjustment_factor
        index_data_before_common.loc[:, "symbol"] = etf

        # Combine adjusted index data with ETF data
        combined_data = pd.concat([index_data_before_common, etf_data])
        frames.append(combined_data)

    # Concatenate all frames to form the final dataframe
    result_df = (
        pd.concat(frames).sort_values(by=["date", "symbol"]).reset_index(drop=True)
    )

    return result_df


def aggregate_results(selection_strategy:str):
    # read all the folders in the selection strategy
    strategy_names = os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", selection_strategy))
    loop = tqdm(strategy_names)
    for strategy_name in loop:
        if "." in strategy_name:
            continue

        loop.set_description(f"Processing {strategy_name}")
        # automatically check the filename xxx.pkl under the directory
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", selection_strategy, strategy_name)
        output_file = os.path.join(output_dir, os.listdir(output_dir)[0])

        with open(output_file, "rb") as f:
            all_results = pickle.load(f)

        # import ipdb; ipdb.set_trace()
        # level 0 keys
        rolling_windows = all_results.keys()
        # level 1 keys
        tickers = all_results[list(rolling_windows)[-1]].keys()

        results_df_by_tickers = pd.DataFrame(
            columns=["Period", "ticker", "total_return (%)", "annual_return (%)", "annual_volatility (%)", "sharpe_ratio", "sortino_ratio",
                     "max_drawdown"])

        all_ticker_avg_total_return = 0
        all_ticker_avg_annual_return = 0
        all_ticker_avg_annual_volatility = 0
        all_ticker_avg_sharpe_ratio = 0
        all_ticker_avg_sortino_ratio = 0
        all_ticker_avg_max_drawdown = 0
        all_ticker_valid_window = 0

        try:
            for ticker in tickers:
                valid_window = 0
                # calculate the average return
                avg_total_return = 0
                avg_annual_return = 0
                avg_annual_volatility = 0
                avg_sharpe_ratio = 0
                avg_sortino_ratio = 0
                avg_max_drawdown = 0

                for window in rolling_windows:
                    if ticker not in all_results[window]:
                        continue

                    avg_total_return += all_results[window][ticker]["total_return"]
                    avg_annual_return += all_results[window][ticker]["annual_return"]
                    avg_annual_volatility += all_results[window][ticker]["annual_volatility"]
                    avg_sharpe_ratio += all_results[window][ticker]["sharpe_ratio"]
                    avg_sortino_ratio += all_results[window][ticker]["sortino_ratio"]
                    avg_max_drawdown += all_results[window][ticker]["max_drawdown"]

                    all_ticker_avg_total_return += all_results[window][ticker]["total_return"]
                    all_ticker_avg_annual_return += all_results[window][ticker]["annual_return"]
                    all_ticker_avg_annual_volatility += all_results[window][ticker]["annual_volatility"]
                    all_ticker_avg_sharpe_ratio += all_results[window][ticker]["sharpe_ratio"]
                    all_ticker_avg_sortino_ratio += all_results[window][ticker]["sortino_ratio"]
                    all_ticker_avg_max_drawdown += all_results[window][ticker]["max_drawdown"]

                    # print("="*10)
                    # print(all_ticker_avg_sharpe_ratio)
                    # print(all_results[window][ticker]["sharpe_ratio"])

                    valid_window += 1
                    all_ticker_valid_window += 1

                    results_df_by_tickers = results_df_by_tickers._append(
                        {
                            "Period": window,
                            "ticker": ticker,
                            "total_return (%)": "{:.3f}".format(all_results[window][ticker]["total_return"] * 100),
                            "annual_return (%)": "{:.3f}".format(all_results[window][ticker]["annual_return"] * 100),
                            "annual_volatility (%)": "{:.3f}".format(all_results[window][ticker]["annual_volatility"] * 100),
                            "sharpe_ratio": "{:.3f}".format(all_results[window][ticker]["sharpe_ratio"]),
                            "sortino_ratio": "{:.3f}".format(all_results[window][ticker]["sortino_ratio"]),
                            "max_drawdown": "{:.3f}".format(-all_results[window][ticker]["max_drawdown"]),
                        },
                        ignore_index=True)

                avg_total_return /= valid_window
                avg_annual_return /= valid_window
                avg_annual_volatility /= valid_window
                avg_sharpe_ratio /= valid_window
                avg_sortino_ratio /= valid_window
                avg_max_drawdown /= valid_window

                results_df_by_tickers = results_df_by_tickers._append(
                    {
                        "Period": "Average",
                        "ticker": ticker,
                        "total_return (%)": "{:.3f}".format(avg_total_return * 100),
                        "annual_return (%)": "{:.3f}".format(avg_annual_return * 100),
                        "annual_volatility (%)": "{:.3f}".format(avg_annual_volatility * 100),
                        "sharpe_ratio": "{:.3f}".format(avg_sharpe_ratio),
                        "sortino_ratio": "{:.3f}".format(avg_sortino_ratio),
                        "max_drawdown": "{:.3f}".format(-avg_max_drawdown),
                    },
                    ignore_index=True)

            all_ticker_avg_total_return /= all_ticker_valid_window
            all_ticker_avg_annual_return /= all_ticker_valid_window
            all_ticker_avg_annual_volatility /= all_ticker_valid_window
            all_ticker_avg_sharpe_ratio /= all_ticker_valid_window
            all_ticker_avg_sortino_ratio /= all_ticker_valid_window
            all_ticker_avg_max_drawdown /= all_ticker_valid_window

            results_df_by_tickers = results_df_by_tickers._append(
                {
                    "Period": "Average",
                    "ticker": "All",
                    "total_return (%)": "{:.3f}".format(all_ticker_avg_total_return * 100),
                    "annual_return (%)": "{:.3f}".format(all_ticker_avg_annual_return * 100),
                    "annual_volatility (%)": "{:.3f}".format(all_ticker_avg_annual_volatility * 100),
                    "sharpe_ratio": "{:.3f}".format(all_ticker_avg_sharpe_ratio),
                    "sortino_ratio": "{:.3f}".format(all_ticker_avg_sortino_ratio),
                    "max_drawdown": "{:.3f}".format(-all_ticker_avg_max_drawdown),
                },
                ignore_index=True)

            results_df_by_tickers.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", selection_strategy, strategy_name, "results.csv"),
                                         index=False)
        except Exception as e:
            print(f"Error processing {strategy_name}: {e}")
            continue


def aggregate_results_one_strategy(selection_strategy: str, trading_strategy: str):

    # automatically check the filename xxx.pkl under the directory
    # root dir is the grandparent directory of this file
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", selection_strategy, trading_strategy)
    output_file = os.path.join(output_dir, [file for file in os.listdir(output_dir) if file.endswith(".pkl")][0])

    with open(output_file, "rb") as f:
        all_results = pickle.load(f)

    # import ipdb; ipdb.set_trace()
    # level 0 keys
    rolling_windows = all_results.keys()
    # level 1 keys
    tickers = all_results[list(rolling_windows)[-1]].keys()

    results_df_by_tickers = pd.DataFrame(
        columns=["Period", "ticker", "total_return (%)", "annual_return (%)", "annual_volatility (%)", "sharpe_ratio", "sortino_ratio",
                 "max_drawdown"])

    all_ticker_avg_total_return = 0
    all_ticker_avg_annual_return = 0
    all_ticker_avg_annual_volatility = 0
    all_ticker_avg_sharpe_ratio = 0
    all_ticker_avg_sortino_ratio = 0
    all_ticker_avg_max_drawdown = 0
    all_ticker_valid_window = 0

    try:
        for ticker in tickers:
            valid_window = 0
            # calculate the average return
            avg_total_return = 0
            avg_annual_return = 0
            avg_annual_volatility = 0
            avg_sharpe_ratio = 0
            avg_sortino_ratio = 0
            avg_max_drawdown = 0

            for window in rolling_windows:
                if ticker not in all_results[window]:
                    continue

                avg_total_return += all_results[window][ticker]["total_return"]
                avg_annual_return += all_results[window][ticker]["annual_return"]
                avg_annual_volatility += all_results[window][ticker]["annual_volatility"]
                avg_sharpe_ratio += all_results[window][ticker]["sharpe_ratio"]
                avg_sortino_ratio += all_results[window][ticker]["sortino_ratio"]
                avg_max_drawdown += all_results[window][ticker]["max_drawdown"]

                all_ticker_avg_total_return += all_results[window][ticker]["total_return"]
                all_ticker_avg_annual_return += all_results[window][ticker]["annual_return"]
                all_ticker_avg_annual_volatility += all_results[window][ticker]["annual_volatility"]
                all_ticker_avg_sharpe_ratio += all_results[window][ticker]["sharpe_ratio"]
                all_ticker_avg_sortino_ratio += all_results[window][ticker]["sortino_ratio"]
                all_ticker_avg_max_drawdown += all_results[window][ticker]["max_drawdown"]

                # print("="*10)
                # print(all_ticker_avg_sharpe_ratio)
                # print(all_results[window][ticker]["sharpe_ratio"])

                valid_window += 1
                all_ticker_valid_window += 1

                results_df_by_tickers = results_df_by_tickers._append(
                    {
                        "Period": window,
                        "ticker": ticker,
                        "total_return (%)": "{:.3f}".format(all_results[window][ticker]["total_return"] * 100),
                        "annual_return (%)": "{:.3f}".format(all_results[window][ticker]["annual_return"] * 100),
                        "annual_volatility (%)": "{:.3f}".format(all_results[window][ticker]["annual_volatility"] * 100),
                        "sharpe_ratio": "{:.3f}".format(all_results[window][ticker]["sharpe_ratio"]),
                        "sortino_ratio": "{:.3f}".format(all_results[window][ticker]["sortino_ratio"]),
                        "max_drawdown": "{:.3f}".format(-all_results[window][ticker]["max_drawdown"]),
                    },
                    ignore_index=True)

            avg_total_return /= valid_window
            avg_annual_return /= valid_window
            avg_annual_volatility /= valid_window
            avg_sharpe_ratio /= valid_window
            avg_sortino_ratio /= valid_window
            avg_max_drawdown /= valid_window

            results_df_by_tickers = results_df_by_tickers._append(
                {
                    "Period": "Average",
                    "ticker": ticker,
                    "total_return (%)": "{:.3f}".format(avg_total_return * 100),
                    "annual_return (%)": "{:.3f}".format(avg_annual_return * 100),
                    "annual_volatility (%)": "{:.3f}".format(avg_annual_volatility * 100),
                    "sharpe_ratio": "{:.3f}".format(avg_sharpe_ratio),
                    "sortino_ratio": "{:.3f}".format(avg_sortino_ratio),
                    "max_drawdown": "{:.3f}".format(-avg_max_drawdown),
                },
                ignore_index=True)

        all_ticker_avg_total_return /= all_ticker_valid_window
        all_ticker_avg_annual_return /= all_ticker_valid_window
        all_ticker_avg_annual_volatility /= all_ticker_valid_window
        all_ticker_avg_sharpe_ratio /= all_ticker_valid_window
        all_ticker_avg_sortino_ratio /= all_ticker_valid_window
        all_ticker_avg_max_drawdown /= all_ticker_valid_window

        results_df_by_tickers = results_df_by_tickers._append(
            {
                "Period": "Average",
                "ticker": "All",
                "total_return (%)": "{:.3f}".format(all_ticker_avg_total_return * 100),
                "annual_return (%)": "{:.3f}".format(all_ticker_avg_annual_return * 100),
                "annual_volatility (%)": "{:.3f}".format(all_ticker_avg_annual_volatility * 100),
                "sharpe_ratio": "{:.3f}".format(all_ticker_avg_sharpe_ratio),
                "sortino_ratio": "{:.3f}".format(all_ticker_avg_sortino_ratio),
                "max_drawdown": "{:.3f}".format(-all_ticker_avg_max_drawdown),
            },
            ignore_index=True)

        results_df_by_tickers.to_csv(os.path.join(output_dir, "results.csv"),
                                     index=False)
    except Exception as e:
        print(f"Error processing {trading_strategy}: {e}")

if __name__ == "__main__":
    pass


