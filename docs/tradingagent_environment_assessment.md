# TradingAgents Environment Assessment

## Scope

This note covers the current pre-integration assessment for bringing
`TauricResearch/TradingAgents` into the FINSABER benchmark on the
`tradingagent` branch.

Current branch state:

- Working branch: `tradingagent`
- Upstream TradingAgents cloned into `llm_traders/tradingagent`
- Inner `.git` removed so the code is now managed by this repo

This note intentionally does **not** perform installation yet. The goal is to
define a safe environment baseline and a realistic integration path first.

## Status Update

Current validated status on the `tradingagent` branch:

- TradingAgents source is vendored at `llm_traders/tradingagent`
- A new conda env `finsaber-ta310` was created
- `Phase 0` completed
- `Phase 1` completed
- `Phase 2` completed
- `Phase 2.5` completed
- `Phase 3` completed on the upstream-style runtime path
- `Phase 3.5` analysis completed at the code-review level

### Phase 3 empirical run

Validated run profile:

- provider: `openai`
- model: `gpt-4o-mini`
- selected analysts: `["market", "news"]`
- ticker: `TSLA`
- trade date: `2023-06-01`
- debate rounds: `1`
- risk rounds: `1`

Observed result:

- run completed successfully
- elapsed time: about `97.6s`
- final decision signal: `Underweight`
- `final_trade_decision` was produced successfully

What this proves:

- the Python 3.10 environment can execute the real TradingAgents graph
- the OpenAI path is operational in this environment
- the reduced analyst path (`market + news`) is viable for first-pass integration

What this does **not** prove:

- benchmark readiness
- FINSABER local-data integration
- strict historical reproducibility suitable for backtesting

### Problems encountered during validation

#### 1. `conda` was not on PATH inside the Codex shell

Cause:

- the machine had Anaconda installed at `/opt/anaconda3`
- the Codex subprocess PATH did not include `/opt/anaconda3/bin`

Resolution:

- used `/opt/anaconda3/bin/conda` explicitly for all environment actions

#### 2. `conda run` with heredoc swallowed output

Cause:

- command execution via `conda run ... python - <<'PY'` returned exit codes
  correctly but did not reliably surface stdout in this environment

Resolution:

- switched validation commands to `python -c ...`

#### 3. First real OpenAI run failed with `Missing credentials`

Cause:

- a retry command prefixed `OPENAI_API_KEY="$OPENAI_API_KEY"`
- in the parent shell, `OPENAI_API_KEY` was unset
- this created an explicit empty env var in the child process
- TradingAgents loads `.env` with `override=False`, so the empty env var
  shadowed the real `.env` value instead of being replaced

Resolution:

- removed the explicit shell-level `OPENAI_API_KEY=...` prefix
- relied on TradingAgents package import to load `.env` from the repo root

Important implication:

- for future runs, do not pre-set `OPENAI_API_KEY` to an empty shell value
  before invoking the process

#### 4. First `Phase 2.5` test subset failed on Anthropic/Google cases

Cause:

- `test_temperature_config.py` includes provider-path tests for Anthropic and Google
- the current environment intentionally excludes `langchain-anthropic` and
  `langchain-google-genai`

Resolution:

- re-ran the intended OpenAI-slim subset with:
  `-k "not anthropic and not google"`

Interpretation:

- this was not a defect in the OpenAI-slim environment
- it confirmed that the provider surface was intentionally narrowed

## Key Findings

### 1. Python baseline should be 3.10

- FINSABER root requires `>=3.10`.
- TradingAgents upstream requires `>=3.10` and tests 3.10 through 3.13.
- FinMem is pinned to `>=3.10,<3.11`.

Conclusion: the final integration baseline should be **Python 3.10**.

Python 3.12 is useful only as an upstream sanity reference because TradingAgents
README and Dockerfile currently center on 3.12. It should not be our long-term
benchmark baseline.

### 2. TradingAgents full install is wider than we need

Upstream package metadata includes both core runtime and product-facing extras:

- Core runtime candidates:
  - `langchain-core`
  - `langchain-openai`
  - `langgraph`
  - `langgraph-checkpoint-sqlite`
  - `pandas`
  - `pydantic`
  - `python-dotenv`
  - `requests`
  - `stockstats`
  - `yfinance`
- Full-package extras we do not need for the benchmark first pass:
  - `questionary`
  - `typer`
  - `rich` CLI surface
  - `langchain-anthropic`
  - `langchain-google-genai`
  - `redis`
  - `parsel`
  - `langchain-experimental`

Conclusion: for FINSABER integration we should target a **slim TradingAgents
runtime**, not the full upstream CLI environment.

### 3. The main compatibility risk is not FINSABER, but old `FinMem`

`FinMem` still carries an older stack:

- `langchain-community==0.0.15`
- `langchain-core==0.1.22`
- `openai==1.12.0`
- Python `<3.11`

TradingAgents uses a much newer stack:

- `langchain-core>=0.3.81`
- `langchain-openai>=0.3.23`
- `langgraph>=0.4.8`

Conclusion: we should **not** try to build one environment that treats
`FinMem`, `FinAgent`, and `TradingAgents` as co-equal first-class install
targets. That would create unnecessary dependency pressure before we even start
integration.

### 4. `FinAgent` is less problematic than `FinMem`

`FinAgent` in this repo is mostly source-driven and uses:

- local modules inside `llm_traders/finagent`
- direct `openai` access
- some `langchain_community` utilities for processing

It does not expose a separate package manifest at the same level as `FinMem` or
TradingAgents. This means we can treat it as existing repo code that runs inside
the selected environment, instead of forcing a separate packaging story first.

### 5. TradingAgents mainline logic is separable from the CLI

The benchmark integration target is the programmatic graph, not the user CLI.

Primary programmatic entry:

- `tradingagents.graph.trading_graph.TradingAgentsGraph`

Important properties:

- `selected_analysts` is configurable
- provider client creation is lazy by provider
- the graph can run without the CLI layer

Conclusion: we should integrate around `TradingAgentsGraph`, not around
`tradingagents` CLI commands.

### 6. First-pass benchmark integration should use a reduced analyst set

TradingAgents default analysts are:

- `market`
- `social`
- `news`
- `fundamentals`

FINSABER local benchmark data currently maps most naturally to:

- `market`
- `news`

The weakest matches today are:

- `social`: depends on StockTwits/Reddit style sources
- `fundamentals`: expects structured financials, while our current benchmark
  side is closer to filings/news text

Conclusion: first environment validation should target a reduced core path such
as:

```python
selected_analysts = ["market", "news"]
```

Only after that is stable should we expand to `fundamentals` or social signals.

## Recommended Environment Strategy

### Baseline decision

Use **one new Python 3.10 conda environment** as the integration baseline for:

- FINSABER root package
- TradingAgents core runtime
- repo-local `FinAgent` code as needed

Do **not** include `FinMem` as an installed target in this first environment.

### Why this is the right cut

- It preserves the repo's 3.10 mainline.
- It avoids forcing old `FinMem` LangChain pins into the same resolver path.
- It keeps TradingAgents focused on the graph logic we actually need.
- It gives us a clean place to verify imports, one dry run, and later adapter
  work.

## Recommended Install Shape

### Step A: create the base conda env

Recommended baseline:

```bash
conda create -n finsaber-ta310 python=3.10 pip
conda activate finsaber-ta310
```

### Step B: install FINSABER root first

```bash
pip install -e .
```

This installs the benchmark framework without pulling `llm_traders*` as package
discovery targets, which is consistent with the current root `pyproject.toml`.

### Step C: install a curated TradingAgents core dependency set

Initial target set:

```text
langchain-core>=0.3.81
langchain-openai>=0.3.23
langgraph>=0.4.8
langgraph-checkpoint-sqlite>=2.0.0
pandas>=2.3.0
pydantic>=2,<3
python-dotenv>=1.0.0
requests>=2.32.4
stockstats>=0.6.5
typing-extensions>=4.14.0
yfinance>=1.4.1
```

Notes:

- `openai` will be pulled transitively by `langchain-openai`.
- `backtrader`, `rich`, `tqdm`, `python-dotenv`, `pandas` are already aligned
  or shared with FINSABER.
- We intentionally do not pull extra provider packages or CLI-only packages in
  the first pass.

### Step D: install TradingAgents source in editable mode without auto-deps

Recommended shape:

```bash
pip install -e llm_traders/tradingagent --no-deps
```

Reason:

- Upstream `pyproject.toml` declares the full product dependency set.
- We want the code import path, but we do not want pip to force in unused CLI
  and multi-provider extras at the first stage.

## Why `--no-deps` is justified here

`pip install -e llm_traders/tradingagent --no-deps` does one very specific
thing:

- it registers the local TradingAgents source as an importable editable package
- it does **not** ask pip to solve and install everything declared in upstream
  `pyproject.toml`

This is useful only because the code structure supports it.

### What we confirmed from the code

#### 1. CLI dependencies are isolated from the graph runtime

The package metadata includes `cli*`, but CLI imports are concentrated under
`llm_traders/tradingagent/cli/`.

Examples:

- `cli/main.py` imports `typer`, `questionary`, and many `rich` components
- the graph runtime entry is `tradingagents.graph.trading_graph.TradingAgentsGraph`

The graph import path does not require CLI modules.

#### 2. Provider imports are lazy

`tradingagents.llm_clients.factory.create_llm_client()` lazily imports the
provider client only when that provider is selected.

Implication:

- if we run only `llm_provider="openai"`, we need the OpenAI path
- we do **not** need `langchain-anthropic` or `langchain-google-genai` just to
  import the graph or construct the object on the OpenAI route

#### 3. Some declared dependencies appear unused on the first-pass path

From the current code scan:

- `redis`: no active runtime import found in the core graph path
- `parsel`: no active runtime import found in the core graph path
- `langchain-experimental`: no active runtime import found in the core graph
  path

This does not prove they are never useful. It does show they are not obvious
first-pass requirements for the benchmark integration route we want.

### What `--no-deps` does **not** mean

It does **not** mean we skip dependency management.

It means:

1. we install dependencies ourselves in a controlled order
2. we install only the subset needed for the validated runtime path
3. we let verification reveal any missing dependency before integration work

### Do we need to split the package?

Not immediately.

There are two possible approaches:

#### Option A: no code split yet

- keep upstream layout unchanged
- use curated dependency installation
- install TradingAgents editable with `--no-deps`
- validate against the actual graph import path

This is the recommended first move because it is low-risk and reversible.

#### Option B: later dependency split

If validation shows repeated friction, we can later refactor the local
TradingAgents copy into clearer dependency groups, for example:

- core graph runtime
- CLI extras
- optional provider extras

That would mean changing local package metadata, possibly adding extras such as
`[project.optional-dependencies] core/openai/cli`.

I do **not** recommend doing this before first validation, because that would
mean editing packaging before we have evidence it is necessary.

### Do we need to change the code now?

My current answer is: **probably not for the first validation pass**.

The first validation target is only:

- import `tradingagents`
- import `TradingAgentsGraph`
- construct a graph with a reduced analyst set
- verify the OpenAI path loads

If any missing import appears during these steps, then we decide whether the
fix belongs in:

- the curated dependency list, or
- a small local code change

We should not pre-emptively rewrite code before the verification tells us where
the real boundary is.

### How we will verify that `--no-deps` is safe enough

We verify in increasing order of strength:

1. package import
2. graph import
3. graph construction
4. reduced-analyst dry run
5. selected upstream unit tests on the paths we rely on

If any step fails due to a missing dependency, we add that dependency
explicitly. That is the whole point of this strategy: make the real dependency
surface visible instead of accepting the full upstream bundle blindly.

## What We Should Not Do First

- Do not install the old `FinMem` dependency set into this environment.
- Do not start from the historical `finsaber_env.yml`; it is Windows-heavy and
  not a reliable base for this macOS setup.
- Do not optimize for TradingAgents CLI parity before verifying
  `TradingAgentsGraph` import and a minimal run.
- Do not solve filings tooling now; that belongs to a later adapter phase.

## Validation Plan After Environment Creation

The environment should be validated in this order.

### Phase 0: environment creation integrity

Goal:

- confirm the interpreter is really Python 3.10
- confirm installs come from the intended conda env

Checks:

```bash
python --version
which python
python -m pip --version
python -c "import sys; print(sys.executable); print(sys.version)"
```

### Phase 1: import validation

Target checks:

```bash
python -c "import finsaber; print('finsaber ok')"
python -c "import tradingagents; print('tradingagents pkg ok')"
python -c "from tradingagents.graph.trading_graph import TradingAgentsGraph; print('ta ok')"
```

Purpose:

- verifies editable package registration works
- verifies `python-dotenv` and base package import path work
- verifies graph runtime imports do not pull CLI-only dependencies

Failure interpretation:

- if `tradingagents` import fails, the curated dependency set is incomplete
- if `TradingAgentsGraph` import fails, a graph-path dependency is missing
- if `cli.main` fails but graph import succeeds, that is acceptable at this
  stage because CLI is not our current target

### Phase 2: object construction

Construct only:

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"

ta = TradingAgentsGraph(
    selected_analysts=["market", "news"],
    debug=True,
    config=config,
)
```

This verifies that the graph, prompts, tool nodes, and provider stack load
correctly on Python 3.10 before any benchmark wiring.

Purpose:

- verifies lazy provider loading works on the OpenAI path
- verifies the reduced analyst set is accepted
- verifies graph compilation succeeds without CLI participation

Expected outcome:

- object construction succeeds without hitting Anthropic/Google/CLI imports

### Phase 2.5: selected offline upstream tests

Before any live API call, run a curated subset of upstream tests that match the
runtime surface we plan to depend on.

Recommended subset:

```bash
pytest -q \
  llm_traders/tradingagent/tests/test_analyst_execution.py \
  llm_traders/tradingagent/tests/test_dataflows_config.py \
  llm_traders/tradingagent/tests/test_vendor_routing.py \
  llm_traders/tradingagent/tests/test_temperature_config.py \
  llm_traders/tradingagent/tests/test_structured_agents.py
```

Why this subset:

- `test_analyst_execution.py`: validates analyst selection/order logic
- `test_dataflows_config.py`: validates config behavior
- `test_vendor_routing.py`: validates routing behavior for tool vendors
- `test_temperature_config.py`: validates provider-kwargs path and graph helper
- `test_structured_agents.py`: validates structured output behavior for key
  downstream agents

Important refinement for the current **OpenAI-slim** environment:

- `test_temperature_config.py` includes Anthropic and Google provider cases
- if those provider packages are intentionally not installed, run the subset as:

```bash
pytest -q \
  llm_traders/tradingagent/tests/test_analyst_execution.py \
  llm_traders/tradingagent/tests/test_dataflows_config.py \
  llm_traders/tradingagent/tests/test_vendor_routing.py \
  llm_traders/tradingagent/tests/test_temperature_config.py \
  llm_traders/tradingagent/tests/test_structured_agents.py \
  -k "not anthropic and not google"
```

This is not hiding a real failure. It aligns the offline test surface with the
provider surface we intentionally installed for the first-pass integration.

Purpose:

- catches missing runtime deps before we spend API money
- validates code paths that are core to graph execution

### Phase 2.6: optional graph-construction micro-check without provider call

If needed, we can add a local micro-check that monkeypatches client creation so
we can test graph construction without any external API dependency.

This is useful when we want to separate:

- dependency/import correctness
- actual provider/network correctness

This check is optional because upstream already unit-tests many local paths, but
it may still be helpful in our repo context.

### Phase 3: single-run sanity check

Only after object construction succeeds:

- supply one provider key
- run one narrow ticker/date sanity check
- confirm that the decision path completes

This is still an upstream-style sanity run, not yet a FINSABER backtest.

Recommended constraints:

- `selected_analysts=["market", "news"]`
- one US ticker first, e.g. `AAPL` or `TSLA`
- one fixed historical date
- one provider only: OpenAI
- low debate rounds to reduce cost and noise

Validation goals:

- the graph reaches a final decision
- the OpenAI path works under Python 3.10
- no accidental CLI dependency is needed
- no unexpected provider package is required

### Phase 3.5: benchmark-readiness review

Only after the upstream-style sanity run succeeds, perform a short review of
what still assumes online fetching.

Checklist:

- which graph tools still call `yfinance`
- which paths assume online news retrieval
- whether `fundamentals` is still unfit for first-pass benchmark use
- whether the social/sentiment path should stay disabled

Purpose:

- freeze the exact scope of the first adapter implementation
- avoid mixing environment debugging with data-adapter debugging

Current findings from the code review after `Phase 3`:

- `market` analyst still depends on online vendor-routed tools:
  - `get_stock_data`
  - `get_indicators`
  - `get_verified_market_snapshot`
- with current defaults, all three resolve to `yfinance` / `stockstats` data
- `news` analyst still depends on:
  - ticker news from `yfinance`
  - macro news from `yfinance.Search(...)`
- `sentiment` / `social` is still unsuitable for first-pass benchmark use:
  - it pre-fetches Reddit and StockTwits data directly
  - this does not map to the current FINSABER benchmark dataset
- `fundamentals` is still unsuitable for first-pass benchmark use:
  - it expects structured financial statement tools
  - current benchmark-side local data is closer to raw filings/news text than
    to ready-made balance sheet / cashflow / income-statement tables

### Code-level online-path audit

This section is the main design answer for `Phase 4`.

#### Role-by-role data map

| Role | Reads | Default source in TA | Phase 4 status |
|---|---|---|---|
| Market Analyst | OHLCV, indicators, verified snapshot | `yfinance` + `stockstats` | Must be replaced for local benchmark runs |
| News Analyst | ticker news, macro news, insider transactions | `yfinance` news / search | Must be replaced or trimmed |
| Sentiment Analyst | news, StockTwits, Reddit | `yfinance` + public web | Disable in the first pass |
| Fundamentals Analyst | fundamentals, balance sheet, cashflow, income statement | `yfinance` financials | Disable in the first pass |
| Research Manager | debate history | local LLM state | Fine as-is |
| Trader | investment plan | local LLM state | Fine as-is |
| Portfolio Manager | risk debate + past context + trader proposal | local LLM state + memory log | Fine, but memory log needs an offline decision |

#### Hidden online paths

- `TradingAgentsGraph.resolve_instrument_context()` calls `resolve_instrument_identity()`, which does a live `yfinance.Ticker(...).info` lookup before the graph starts.
- `TradingAgentsGraph._resolve_pending_entries()` calls `_fetch_returns()`, which uses `yfinance.Ticker(...).history(...)` to resolve prior decisions and write reflections.
- `load_ohlcv()` in `stockstats_utils.py` downloads market data with `yfinance.download(...)` and caches it locally. `get_stock_data`, `get_indicators`, and `get_verified_market_snapshot` all inherit that path.
- `get_news_yfinance()` and `get_global_news_yfinance()` use `yfinance` news/search APIs.
- `get_fundamentals()`, `get_balance_sheet()`, `get_cashflow()`, `get_income_statement()`, and `get_insider_transactions()` all use `yfinance`.
- `fetch_stocktwits_messages()` and `fetch_reddit_posts()` call public web endpoints directly, even though they do not require API keys.
- `cli/announcements.py` fetches `https://api.tauric.ai/v1/announcements` on CLI startup.
- `cli/utils.py` fetches `https://openrouter.ai/api/v1/models` when the user selects OpenRouter in the CLI.
- `alpha_vantage_common.py` uses `requests.get(...)` against Alpha Vantage if that vendor is selected.
- `create_llm_client()` is lazy, so Anthropic / Google are only loaded when selected, but they are still live network providers if used.

#### What can be closed without code changes

- Do not use the CLI in the benchmark loop. That avoids announcements and OpenRouter model discovery entirely.
- Keep `llm_provider=openai` and do not select Anthropic, Google, OpenRouter, or other provider routes.
- Select only `["market", "news"]` for the first graph runs; that already disables the `social` and `fundamentals` analyst nodes.
- If we want to remove cross-window memory coupling for the first benchmark pass, set `memory_log_path` to an empty value so `TradingMemoryLog` becomes a no-op.

#### What still requires code changes

- `resolve_instrument_identity()` must stop calling `yfinance` and instead read identity metadata from the local dataset or return `{}`.
- `load_ohlcv()` must be backed by local benchmark prices instead of Yahoo Finance if we want a strict offline run.
- `_fetch_returns()` must use the same local price source as the benchmark, otherwise memory reflection keeps calling Yahoo Finance in the background.
- `get_news_yfinance()` and `get_global_news_yfinance()` need local news adapters if the news analyst is to remain enabled.
- `get_fundamentals()` and its statement helpers need a filings-backed adapter if we ever want the fundamentals analyst back in play.
- `fetch_stocktwits_messages()` and `fetch_reddit_posts()` should stay disabled unless we intentionally add social data to the local dataset.

#### What breaks if each path is closed

- Closing `resolve_instrument_identity()` only removes resolved company metadata; the graph still works with ticker-only context.
- Closing the memory log removes past reflections and cross-window lessons; the first run still works, but later runs stop accumulating history.
- Closing the market/news yfinance paths without replacements breaks the first two analysts, so the graph can no longer generate a meaningful first-pass report.
- Closing `social` and `fundamentals` is acceptable for the first benchmark version because they are already outside the planned initial scope.
- Closing CLI-only online calls has no effect on the programmatic benchmark path.

#### How to re-enable later

- Keep the current graph layout, but switch the data backend back to the remote path or vendor route once local adapters are ready.
- Re-enable `social` only after we have a real local social dataset, otherwise it will keep being a web-crawl dependency.
- Re-enable `fundamentals` only after filings are split into a structured tool-friendly format.
- Re-enable memory only when we want cross-window accumulation and are prepared to accept the shared-log semantics again.

#### Recommendation for Phase 4

- Treat `market + news` as the only first-pass analysts.
- Add a local-data adapter layer for prices, news, and benchmark returns before trying to restore any of the social or fundamentals chains.
- Keep CLI paths out of the benchmark loop.
- Keep provider selection on OpenAI only until the benchmark adapter is stable.

#### Code change checklist

This is the concrete change list for the hidden online paths.

- `tradingagents/graph/trading_graph.py`
  - Replace the hard `yfinance.Ticker(...).info` identity lookup in `resolve_instrument_context()` with an injected identity source or a pure ticker-only fallback for benchmark mode.
  - Replace the hard `yfinance.Ticker(...).history(...)` lookup in `_fetch_returns()` with a pluggable price provider backed by the local `TradingData` loader.
  - Keep the current yfinance fallback only as a live-mode compatibility path, not as the benchmark default.
- `tradingagents/agents/utils/agent_utils.py`
  - Split identity resolution from prompt formatting so `build_instrument_context()` stays pure and `resolve_instrument_identity()` becomes optional.
  - Add a local identity resolver hook that can read company metadata from the benchmark dataset when it exists.
- `tradingagents/dataflows/stockstats_utils.py`
  - Make `load_ohlcv()` accept a local data loader or runtime provider instead of always downloading from Yahoo Finance.
  - Keep the cache only as a live fallback, not as the benchmark source of truth.
- `tradingagents/dataflows/y_finance.py`
  - Add local/backtest implementations for stock data and indicators, or route these calls to a new local vendor.
  - Stop relying on direct yfinance calls inside the benchmark path.
- `tradingagents/dataflows/yfinance_news.py`
  - Add a local news provider backed by FINSABER news items.
  - Keep the yfinance path only for live runs.
- `tradingagents/dataflows/interface.py`
  - Add a `local` or `finsaber` vendor branch for `get_stock_data`, `get_indicators`, `get_news`, `get_global_news`, and `get_verified_market_snapshot`.
  - Leave `alpha_vantage` as an optional remote fallback, not as the benchmark default.
- `tradingagents/agents/analysts/sentiment_analyst.py`
  - Leave disabled in the first benchmark pass.
  - If we ever enable it, replace StockTwits/Reddit web fetches with a local social-data adapter.
- `tradingagents/agents/analysts/fundamentals_analyst.py`
  - Leave disabled in the first benchmark pass.
  - If we ever enable it, back it with a filings adapter built from the local dataset or an edgartools-based preprocessing step.
- `tradingagents/cli/announcements.py` and `tradingagents/cli/utils.py`
  - No code change is needed if we never use CLI in the benchmark loop.
  - If we want CLI-safe offline mode, guard the network calls behind an explicit opt-in.

#### FINSABER 2.0 data-flow contract

The benchmark side already has a clean local-data contract:

1. `TradeConfig` resolves the run scope and injects `data_loader`.
2. `resolve_trading_data()` normalizes that into a `TradingData` object.
3. `trading_data_to_env_dict()` materializes the date-window dict shape expected by strategy code.
4. `FINSABERFrameworkHelper.load_backtest_data_single_ticker()` slices the per-ticker window.
5. `strategy.on_data(date, today_data, framework)` receives one day at a time.
6. `framework.buy()/sell()` turn strategy intent into executable orders.
7. `framework.run()` handles pending orders, liquidation at the end, and equity tracking.
8. `framework.evaluate()` computes final metrics.
9. `write_result_artifacts()` persists summary JSON/CSV outputs.

The important consequence is that TA should not invent a second data pipeline. It should sit inside the existing FINSABER strategy slot and consume the same `TradingData` / `today_data` / `framework` objects that FinMem and FinAgent already use.

#### TA integration shape

The lowest-risk integration path is:

- build a TA-specific strategy class under `llm_traders/finsaber_strategies`
- give that strategy a FINSABER `data_loader` and run dates
- create a small local adapter that serves TA tools from the same `TradingData`
- instantiate `TradingAgentsGraph` once per ticker/run
- call `graph.propagate(ticker, date)` inside `on_data`
- fold the final `Buy / Overweight / Hold / Underweight / Sell` rating into FINSABER execution as `buy / hold / sell`
- let the FINSABER framework own sizing, slippage, liquidity, commissions, and performance statistics

That keeps TA's internal 5-tier reasoning intact while letting FINSABER remain the only execution and evaluation engine.

Implication for `Phase 4`:

- first adapter target should remain `market + news`
- the adapter work should focus on replacing online vendor calls with
  FINSABER-backed local data sources
- `social` and `fundamentals` should remain disabled until the local data layer
  can support them cleanly

### Phase 4: benchmark adapter work

Only after the runtime is stable:

- map FINSABER `TradingData` into TradingAgents data tools
- replace online fetch assumptions where needed
- design a benchmark strategy wrapper similar to the existing `FinAgent`
  strategy path

Success criteria for Phase 4 entry:

- Python 3.10 env stable
- reduced TradingAgents graph runs successfully
- selected offline tests pass
- no unresolved packaging ambiguity remains for the OpenAI path

## Integration Direction After Environment Validation

Recommended first integration target:

1. Keep TradingAgents source under `llm_traders/tradingagent`
2. Build a new FINSABER-facing strategy wrapper under
   `llm_traders/finsaber_strategies`
3. Start with `market + news`
4. Keep vendor calls local or adapter-backed where possible
5. Treat `fundamentals` and social channels as second-stage work

## Final Recommendation

The right next move is:

1. Build a **Python 3.10 slim integration environment**
2. Install **FINSABER root + curated TradingAgents core runtime**
3. Install **TradingAgents editable with `--no-deps`**
4. Verify `TradingAgentsGraph` import and construction
5. Only then start dataset and backtest adapter work

This keeps the branch clean, matches the repo's real compatibility center, and
avoids solving unnecessary dependency conflicts too early.
