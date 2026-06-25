# TradingAgents -> FINSABER 改造计划

## 目标

把 `TauricResearch/TradingAgents` 作为一个可插拔策略接入 FINSABER，
并保持下面这几个边界不变：

- FINSABER 继续负责数据切片、撮合执行、手续费、滑点和绩效统计
- TradingAgents 继续负责多角色分析、辩论、记忆和最终评级
- TA 的内部 5 档语义保持不变，外层执行先折叠为 `buy / hold / sell`
- 第一阶段只接本地数据，不走公网数据源

## 当前已经确认的代码事实

### 1. FINSABER 的统一数据面是 `TradingData`

`TradingData` 是最小抽象接口，要求实现：

- `get_data_by_date`
- `get_ticker_price_by_date`
- `get_ticker_data_by_date`
- `get_tickers_list`
- `get_subset_by_time_range`
- `get_ticker_subset_by_time_range`
- `get_date_range`

当前有两个实现：

- `FinsaberDataset`：旧版 dict / pickle 兼容层
- `FinsaberParquetDataset`：FINSABER-2 的主力 parquet 数据层

### 2. FinMem / FinAgent 已经能吃 `FinsaberParquetDataset`

FinMem：

- `resolve_trading_data(...)` 先接收外部 `data_loader`
- 没有外部 loader 时，`create_finsaber2_data_loader()` 返回 `FinsaberParquetDataset`
- `trading_data_to_env_dict()` 把 `TradingData` 展成 `MarketEnvironment` 要的日级字典

FinAgent：

- `FinsaberTradingDataDataset` 直接包一层 `TradingData`
- `EnvironmentTrading` 吃的是 adapter 产出的 `prices/news` DataFrame
- `guidances/sentiments/economics` 现在是可选空位，不是强依赖

结论：

- 当前 main 的目标数据面已经是 `FinsaberParquetDataset`
- `FinsaberDataset` 只应保留为 legacy pickle 兼容，不应作为新 benchmark 主线

### 3. TA 目前的在线依赖主要分四类

- 价格 / 技术指标：`yfinance` + `load_ohlcv`
- 新闻：`yfinance_news`
- 身份识别：`resolve_instrument_identity()` 的 `yfinance.Ticker().info`
- 反思 outcome：`_fetch_returns()` 的 `yfinance.Ticker().history`

此外还有：

- `sentiment_analyst` 的 StockTwits / Reddit
- `fundamentals_analyst` 的财务链
- CLI 的公告 / model discovery

## TA 接入 FINSABER 的目标形态

### 外层

新增一个 FINSABER 策略类，和 `FinMemStrategy` / `FinAgentStrategy` 同级：

- 构造时接收 `data_loader / symbol / date_from / date_to`
- 在 `on_data(date, today_data, framework)` 里调用 `TradingAgentsGraph`
- 将 TA 的最终 5 档 rating 映射成 FINSABER 的交易动作

### 内层

TA graph 不改主干，只改数据供给：

- `market` analyst 从本地价格和本地技术指标拿数据
- `news` analyst 从本地新闻拿数据
- `social` / `fundamentals` 第一阶段默认关闭
- `global_news` 第一阶段默认关闭
- `insider_transactions` 第一阶段默认关闭

## 推荐分阶段计划

### Phase 1

- 只保留 `market + news`
- local vendor 接入 `TradingData`
- 关掉所有公网数据路径
- `position_sizing` 只保留为文本字段，不进入执行层

### Phase 2

- 接 filings 切分工具
- 先定义一个统一的本地 filings provider，再由 FINSABER / FinMem / TA 分别消费
- 再考虑 `fundamentals`
- 再考虑 `get_insider_transactions`

### Filings Tool 设计要点

- 工具输入不应该只是“ticker + date”，而应该显式带上：
  - `ticker`
  - `as_of_date`
  - `form_type`，例如 `10-K` / `10-Q`
  - `item_codes`，例如 `7`、`part1item2`、`8`
  - `mode`，例如 `raw` / `items` / `summary`
- 工具输出需要保留元数据，而不是只返回一大段字符串：
  - 原始文件标识
  - item code
  - filing date / accession
  - 解析后的 item text
- 对 FINSABER 来说，最稳妥的接法是把它当成 `TradingData` 的一层上游处理，而不是直接把 SEC 逻辑塞进策略里。
  - `TradingData` 继续负责“日级数据切片”
  - filings tool 负责“文档切分与结构化”
  - `trading_data_to_env_dict()` 可以继续输出旧的 `filing_k` / `filing_q`，同时可选增加 item 级字段，保持向后兼容
- 对 FinMem 来说，item 级 filings 后续是有价值的，因为它的 memory 入口本来就是字符串写入，完全可以把不同 item 作为不同 memory 文本进入短中长记忆。
- 对 FinAgent 来说，先不接 filings。它当前的适配层只需要 `price + news`。

### 重要结论

- 只做 item 切分，不能满足 TA 的完整 fundamentals 链。
- TA 现在的 fundamentals 逻辑除了叙述性财报文本，还依赖：
  - `get_fundamentals`：公司概览 / ratio / 基本面摘要
  - `get_balance_sheet`
  - `get_cashflow`
  - `get_income_statement`
- 如果 filings tool 只切 `MD&A` 类 item，那么它只能补充叙述性上下文，不能替代 statement tools。
- 如果以后要让 filings tool 真正支撑 TA 的 fundamentals，至少要覆盖：
  - 10-K 的 `item 7` 和 `item 8`
  - 10-Q 的 `part1item2` 和财务报表相关部分
  - 必要时再补 `1A`、`7A` 等风险/市场讨论项
- 也就是说，item 划分是入口，不是终点；TA 还需要“结构化财务数字 + 文本 narrative”两条腿一起走。

### Phase 3

- 再讨论 `global_news` 的本地 surrogate
- 再讨论是否保留 memory 的跨 ticker lessons
- 再讨论是否把 TA 内部状态输出落到 FINSABER 的结果目录

## 需要改的代码区域

- `tradingagents/dataflows/interface.py`
- `tradingagents/dataflows/stockstats_utils.py`
- `tradingagents/dataflows/y_finance.py`
- `tradingagents/dataflows/yfinance_news.py`
- `tradingagents/agents/utils/agent_utils.py`
- `tradingagents/graph/trading_graph.py`
- `tradingagents/graph/setup.py`
- `tradingagents/agents/analysts/sentiment_analyst.py`
- `tradingagents/agents/analysts/fundamentals_analyst.py`
- 新增 `llm_traders/finsaber_strategies/tradingagents_strategy.py`

## 当前保守决策

- 不改 TA 的 5 档 rating 语义
- 不让 TA 直接控制 FINSABER 仓位大小
- 不把 `TradingData` 塞进 `DEFAULT_CONFIG`
- 不在第一阶段保留 `global_news`
- 不在第一阶段依赖公网社交数据

## 这份计划要验证什么

1. TA 是否能在没有任何公网数据的情况下跑完整个 graph
2. TA 是否能在 FINSABER 的单 ticker 单窗口策略里稳定工作
3. TA 的结果是否能被 FINSABER 正确统计成收益、回撤、成本和交易日志
