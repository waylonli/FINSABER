import os
import backtrader as bt
import datasets as ds
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import pandas_datareader.data as web
load_dotenv()
HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")
import pwb_toolbox.datasets as pwb_ds


def get_tickers_price(tickers: list[str] or str, date_from: str = "2000-01-01", date_to: str = "2024-01-01", return_original: bool = False) -> pd.DataFrame:
    """
    Get the price of the specified tickers within the specified date range
    :param tickers: A list of tickers, or "all" to get the price of all tickers
    :param date_from: Start date, format "YYYY-MM-DD"
    :param date_to: End date, format "YYYY-MM-DD"
    :return: A pandas DataFrame containing the price of the specified tickers within the specified date range
    """
    # df = pd.read_csv("data/stocks_daily.csv")
    # # df = pd.read_csv("data/etfs_daily.csv")
    #
    # df = df[(df["date"] >= date_from) & (df["date"] <= date_to) & df["symbol"].isin(tickers)] if tickers != "all" else df[
    #     (df["date"] >= date_from) & (df["date"] <= date_to)]
    #
    # if df.empty:
    #     raise ValueError("No data available for the specified tickers and date range")
    #
    # df["date"] = pd.to_datetime(df["date"])
    # # date as index
    # df.set_index("date", inplace=True)
    if isinstance(tickers, list):
        df = pwb_ds.load_dataset("Stocks-Daily-Price", tickers, adjust=True)
    elif tickers == "all":
        df = pwb_ds.load_dataset("Stocks-Daily-Price", adjust=True)
    else:
        df = pwb_ds.load_dataset("Stocks-Daily-Price", [tickers], adjust=True)

    df = df[df["date"] >= pd.to_datetime(date_from)]
    df = df[df["date"] < pd.to_datetime(date_to)]

    pivot_df = pd.pivot_table(
        df,
        index="date",
        columns="symbol",
        values=["open", "high", "low", "close"],
        aggfunc="first",
    )

    try:
        full_date_range = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="D")
        pivot_df = pivot_df.reindex(full_date_range)
    except:
        return None

    return pivot_df if not return_original else df


def add_tickers_data(cerebro: bt.Cerebro, pivot_df: pd.DataFrame):
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
        cerebro.adddata(data, name=symbol)

    return

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


if __name__ == "__main__":
    pass


