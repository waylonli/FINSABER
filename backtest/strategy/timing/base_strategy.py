"""Base strategy with composable execution cost handling."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import backtrader as bt
import pandas as pd

from backtest.toolkit.execution_helper import ExecutionContext, PreparedOrder, ExecutionFill


class BaseStrategy(bt.Strategy):
    params = (
        ("execution_context", None),
        ("setup_name", ""),
        ("strategy_name", ""),
    )

    def __init__(self):
        super().__init__()
        self.execution: ExecutionContext = self._resolve_execution_context()

        # Core bookkeeping
        self.trades = []
        self.trade_returns = []
        self.buys: list[pd.Timestamp] = []
        self.sells: list[pd.Timestamp] = []
        self.equity: list[float] = []
        self.equity_date: list[pd.Timestamp] = []

        self.trade_records: Optional[list[dict]] = [] if self.execution.config.record_trades else None
        self.sizing_debug_records: list[dict] = [] if self.execution.config.record_sizing_debug else []
        if self.execution.config.record_sizing_debug:
            self.execution.enable_debug_logging(self.sizing_debug_records)

        self.logger = logging.getLogger(f"{self.__class__.__name__}.{self.p.strategy_name or self.__class__.__name__}")
        self.logger.propagate = False

        self._prepared_orders: Dict[Tuple[str, str], PreparedOrder] = {}

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def _resolve_execution_context(self) -> ExecutionContext:
        ctx = getattr(self.p, "execution_context", None)
        if ctx is None:
            raise ValueError(
                "BaseStrategy requires an 'execution_context' parameter. "
                "Construct one via backtest.toolkit.execution.ExecutionContext and pass it through FINSABERBt."
            )
        return ctx

    def post_next_actions(self):
        equity_value = self.broker.getvalue()
        self.equity.append(equity_value)
        self.equity_date.append(pd.Timestamp(self.data.datetime.date(0)))

    # ------------------------------------------------------------------
    # Execution helpers used by child strategies
    # ------------------------------------------------------------------

    def should_continue_trading(self, symbol: str, is_buy_signal: bool) -> bool:
        side = "buy" if is_buy_signal else "sell"
        return self.execution.should_continue(symbol, side)

    def apply_liquidity_cap_to_size(self, intended_size: int, data: Optional[bt.LineSeries] = None) -> int:
        return self._prepare_size(intended_size, side="sell", data=data)

    def _prepare_size(
        self,
        intended_size: int,
        side: str = "buy",
        data: Optional[bt.LineSeries] = None,
    ) -> int:
        data_feed = data or self.data
        symbol = getattr(data_feed, "_name", "UNKNOWN")
        price = float(data_feed.close[0])
        date = pd.Timestamp(data_feed.datetime.date(0))
        cash = float(self.broker.get_cash()) if side == "buy" else float("inf")

        prepared = self.execution.prepare_order(symbol, date, price, int(intended_size), side, cash)
        self._prepared_orders[(symbol, side)] = prepared
        return prepared.size

    def _adjust_size_for_commission(
        self,
        intended_size: int,
        side: str = "buy",
        data: Optional[bt.LineSeries] = None,
    ) -> int:
        return self._prepare_size(intended_size, side=side, data=data)

    # ------------------------------------------------------------------
    # Order submission overrides to attach execution metadata
    # ------------------------------------------------------------------

    def buy(self, data=None, size=None, **kwargs):  # noqa: D401 - signature aligns with bt.Strategy
        if size is None or size <= 0:
            return None
        data_feed = data or self.data
        symbol = getattr(data_feed, "_name", "UNKNOWN")
        prepared = self._prepared_orders.pop((symbol, "buy"), None)
        if prepared is None:
            prepared_size = self._prepare_size(size, side="buy", data=data_feed)
            if prepared_size <= 0:
                return None
            prepared = self._prepared_orders.pop((symbol, "buy"))
            size = prepared_size
        else:
            size = prepared.size
        if size <= 0:
            return None
        order = super().buy(data=data, size=size, **kwargs)
        if order is not None:
            self.execution.register_order(order, prepared)
        return order

    def sell(self, data=None, size=None, **kwargs):  # noqa: D401 - signature aligns with bt.Strategy
        if size is None or size <= 0:
            return None
        data_feed = data or self.data
        symbol = getattr(data_feed, "_name", "UNKNOWN")
        prepared = self._prepared_orders.pop((symbol, "sell"), None)
        if prepared is None:
            prepared_size = self._prepare_size(size, side="sell", data=data_feed)
            if prepared_size <= 0:
                return None
            prepared = self._prepared_orders.pop((symbol, "sell"))
            size = prepared_size
        else:
            size = prepared.size
        if size <= 0:
            return None
        order = super().sell(data=data, size=size, **kwargs)
        if order is not None:
            self.execution.register_order(order, prepared)
        return order

    # ------------------------------------------------------------------
    # Backtrader callbacks
    # ------------------------------------------------------------------

    def notify_order(self, order: bt.Order):
        if order.status in {order.Submitted, order.Accepted}:
            return

        if order.status == order.Completed:
            executed_date = pd.Timestamp(order.data.datetime.date(0))
            executed_price = float(order.executed.price)
            executed_size = abs(order.executed.size)
            fill = self.execution.finalize_fill(order.ref, executed_size, executed_price, executed_date)

            if fill:
                if fill.slippage_cost:
                    self.broker.add_cash(-fill.slippage_cost)
                self._record_fill(order, fill)

        elif order.status in {order.Canceled, order.Margin, order.Rejected}:
            self.logger.debug("Order %s canceled/margin/rejected", order.ref)

    def notify_trade(self, trade: bt.Trade):
        if trade.isclosed:
            self.trades.append(trade)
            if trade.price:
                pnl = trade.pnlcomm / trade.price
            else:
                pnl = 0.0
            self.trade_returns.append(pnl)

    # ------------------------------------------------------------------
    # Logging / persistence
    # ------------------------------------------------------------------

    def _record_fill(self, order: bt.Order, fill: ExecutionFill) -> None:
        timestamp = pd.Timestamp(fill.date)
        if fill.side == "buy":
            self.buys.append(timestamp)
        else:
            self.sells.append(timestamp)

        if self.trade_records is None:
            return

        current_equity = self.broker.getvalue()
        current_cash = self.broker.get_cash()
        invested = current_equity - current_cash

        record = {
            "date": timestamp,
            "setup": self.p.setup_name,
            "strategy": self.p.strategy_name or self.__class__.__name__,
            "symbol": fill.symbol,
            "side": fill.side,
            "shares": fill.size,
            "price": fill.price,
            "adv": fill.snapshot.adv,
            "vix": fill.snapshot.vix,
            "idiosync_vol": fill.snapshot.idiosyncratic_vol,
            "log_market_cap": fill.snapshot.log_market_cap,
            "beta": fill.snapshot.beta,
            "sp500_return": fill.snapshot.sp500_return,
            "pct_dtv": fill.pct_dtv,
            "signed_sqrt_dtv": fill.signed_sqrt_dtv,
            "liquidity_capped": fill.was_liquidity_capped,
            "slippage": fill.slippage_cost,
            "total_balance": current_equity,
            "cash_balance": current_cash,
            "invested_balance": invested,
        }
        self.trade_records.append(record)

    def save_trading_records(self):
        if not self.trade_records:
            return
        log_dir = "trading_logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{log_dir}/trading_log_{self.p.setup_name}_{self.p.strategy_name}_{timestamp}.csv"
        df = pd.DataFrame(self.trade_records)
        df.to_csv(filename, index=False)
        self.logger.info("Trading records saved to %s", filename)

    def save_sizing_debug_records(self):
        if not self.sizing_debug_records:
            return
        log_dir = "sizing_debug_logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{log_dir}/sizing_debug_{self.p.setup_name}_{self.p.strategy_name}_{timestamp}.csv"
        df = pd.DataFrame(self.sizing_debug_records)
        df.to_csv(filename, index=False)
        self.logger.info("Sizing debug records saved to %s", filename)

    def stop(self):
        self.save_trading_records()
        self.save_sizing_debug_records()
        super().stop()

    # ------------------------------------------------------------------
    # Legacy helper retained for compatibility
    # ------------------------------------------------------------------

    def log(self, txt: str, print_log: bool = False, dt: Optional[pd.Timestamp] = None):
        if print_log:
            dt = dt or pd.Timestamp(self.data.datetime.date(0))
            self.logger.info("%s - %s", dt.date(), txt)
