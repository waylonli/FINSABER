# Installation

## Branch And Dataset

Use the branch that matches your workflow and dataset:

| Branch | Use case | Dataset |
| --- | --- | --- |
| `main` | Current FINSABER-2 package workflow and active development | [https://huggingface.co/datasets/finsaber-team/FINSABER-V2-Data](https://huggingface.co/datasets/finsaber-team/FINSABER-V2-Data) |
| `reproduce` | Reproduce FINSABER-1 experimental results with the v1.0 framework | [https://huggingface.co/datasets/finsaber-team/FINSABER-reproduce](https://huggingface.co/datasets/finsaber-team/FINSABER-reproduce) |

## Package Install

```bash
pip install finsaber
```

The package installs the reusable backtesting framework only.

## Local Development

```bash
git clone https://github.com/waylonli/FINSABER
cd FINSABER
conda create -n trading python=3.10 -y
conda activate trading
python -m pip install -U pip setuptools wheel
pip install -e ".[dev,research]"
```

The `research` extra includes optional dependencies for baseline research strategies, such as `statsmodels`, `xgboost`, `datasets`, and `pandas-datareader`.

The documentation dependencies are separate:

```bash
pip install -e ".[docs]"
```

For full local development, install all extras:

```bash
pip install -e ".[dev,research,docs]"
```

## Build A Wheel

```bash
python -m build --wheel
```

The generated wheel should include only `backtest*` packages. It should not include `examples/`, `llm_traders/`, `rl_traders/`, `tmp/`, `build/`, or `dist/`.

## Environment Variables

LLM integrations in the repository may require:

```text
OPENAI_API_KEY=...
HF_ACCESS_TOKEN=...
```

The core backtesting package does not require API keys.
