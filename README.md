# FINSABER

FINSABER is a comprehensive framework for evaluating trading strategies with a specific focus on comparing traditional technical analysis approaches with modern machine learning and large language model (LLM) based strategies. 

<img src="https://github.com/waylonli/FINSABER/blob/main/figs/framework.png" width="600">

## 1. Environment Setup

```bash
conda create -n finsaber python=3.10
conda activate finsaber
pip install -r requirements.txt
```

Rename `.env.example` to `.env` and set the environment variables. 
- `OPENAI_API_KEY` is required to run LLM-based strategies. 
- `HF_ACCESS_TOKEN` is optional.

## 2. Data

The aggregated S&P500 sample data can be downloaded from [here](https://drive.google.com/file/d/1g9GTNr1av2b9-HphssRrQsLSnoyW0lCF/view?usp=sharing) (10.23GB).

The data is organised as a dictionary with the following structure:
```python
{
    datetime.date(2024,1,1): {
        "price": {
            "AAPL": ...,
            "MSFT": ...,
            ...
        },
        "news": {
            "AAPL": ...,
            "MSFT": ...,
            ...
        },
        "filing_k": {
            "AAPL": ...,
            "MSFT": ...,
            ...
        },
        "filing_q": {
            "AAPL": ...,
            "MSFT": ...,
            ...
        },
        ...
    }
}
```

To plug in your own data, simply inherit the `backtest.data_util.backtest_dataset.BacktestDataset` class and implement the necessary methods.
An example for processing the data format above is provided in `backtest/data_util/finmem_dataset.py`.


## 3. Reproduce the results in the paper

The paper contains three experimental setup: *selective (cherry picking) setup*, *selected-4 setup*, and *composite setup*.
These three experiments can be reproduced by running the following commands.

Baselines (non-LLM):
```bash
python backtest/run_baselines_exp.py \
    --setup <setup_name> \ # can be "cherry_pick_both_finmem", "cherry_pick_both_fincon", "selected_4", "random_sp500_5", "momentum_sp500_5", "lowvol_sp500_5"
    --include <strategy_name> \ # can be one of the class name under backtest/strategy/timing
    --date_from 2004-01-01 \
    --date_to 2024-01-01 \
    --rolling_window_size <window_size> \ # 1 or 2 in the paper
    --rolling_window_step 1
```

LLM Strategies:
```bash
python backtest/run_llm_traders_exp.py \
    --setup <setup_name> \ # can be "cherry_pick_both_finmem", "cherry_pick_both_fincon", "selected_4", "random_sp500_5", "momentum_sp500_5", "lowvol_sp500_5"
    --strategy <strategy_name> \ # can be one of the class name under backtest/strategy/timing_llm
    --strat_config_path <config_path> \ # path to the config file for the LLM strategy, examples under strats_configs folder
    --date_from 2004-01-01 \
    --date_to 2024-01-01 \
    --rolling_window_size <window_size> \ # 1 or 2 in the paper
    --rolling_window_step 1
```
