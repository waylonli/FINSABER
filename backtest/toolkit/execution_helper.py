"""Execution utilities: liquidity capping, slippage sizing, and fill instrumentation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from backtest.data_util.market_data_provider import MarketDataProvider, MarketSnapshot
from backtest.toolkit.trade_config import ExecutionConfig


@dataclass(slots=True)
class PreparedOrder:
    symbol: str
    side: str  # "buy" or "sell"
    size: int
    was_liquidity_capped: bool
    snapshot: MarketSnapshot


@dataclass(slots=True)
class ExecutionFill:
    symbol: str
    side: str
    date: pd.Timestamp
    size: int
    price: float
    slippage_cost: float
    snapshot: MarketSnapshot
    pct_dtv: float
    signed_sqrt_dtv: float
    was_liquidity_capped: bool


@dataclass(slots=True)
class _SymbolState:
    continue_buying: bool = True
    recent_sell: bool = False


class ExecutionContext:
    """Coordinates execution cost modelling for a single strategy run."""

    def __init__(
        self,
        config: ExecutionConfig,
        market_data: MarketDataProvider,
    ) -> None:
        self.config = config
        self.market_data = market_data
        self._states: Dict[str, _SymbolState] = {}
        self._pending: Dict[int, PreparedOrder] = {}
        self._debug_records: Optional[list] = None

    def should_continue(self, symbol: str, side: str) -> bool:
        if side != "buy":
            return True
        if not (self.config.liquidity.enabled and self.config.liquidity.continue_after_cap):
            return True
        state = self._state(symbol)
        allow = state.continue_buying or state.recent_sell
        if allow:
            state.recent_sell = False
        return allow

    def enable_debug_logging(self, storage: list) -> None:
        self._debug_records = storage

    def prepare_order(
        self,
        symbol: str,
        date: pd.Timestamp,
        price: float,
        intended_size: int,
        side: str,
        cash_available: float,
    ) -> PreparedOrder:
        if intended_size <= 0 or price <= 0:
            return PreparedOrder(symbol, side, 0, False, MarketSnapshot())

        snapshot = self.market_data.snapshot(symbol, date)
        size = min(int(intended_size), 10 ** 9)
        was_capped = False

        if self.config.liquidity.enabled:
            size, was_capped = self._apply_liquidity_cap(snapshot, size, price)

        if side == "buy":
            size = self._fit_to_cash(symbol, date, snapshot, size, price, cash_available)
        else:
            size = max(size, 0)

        return PreparedOrder(symbol, side, int(size), was_capped, snapshot)

    def register_order(self, order, prepared: PreparedOrder) -> None:
        order.addinfo(execution_symbol=prepared.symbol, execution_side=prepared.side)
        self._pending[order.ref] = prepared

    def finalize_fill(
        self,
        order_ref: int,
        executed_size: float,
        executed_price: float,
        execution_date: pd.Timestamp,
    ) -> Optional[ExecutionFill]:
        prepared = self._pending.pop(order_ref, None)
        if prepared is None:
            return None
        return self._build_fill(prepared, executed_size, executed_price, execution_date)

    def manual_fill(
        self,
        prepared: PreparedOrder,
        executed_size: float,
        executed_price: float,
        execution_date: pd.Timestamp,
    ) -> Optional[ExecutionFill]:
        """Compute execution costs for manually managed orders (non-Backtrader flows)."""

        if prepared is None:
            return None
        return self._build_fill(prepared, executed_size, executed_price, execution_date)

    def preview_slippage(
        self,
        prepared: PreparedOrder,
        executed_size: float,
        executed_price: float,
    ) -> tuple[float, float, float]:
        """Estimate slippage components without mutating internal state."""

        if (
            prepared is None
            or not self.config.slippage.enabled
            or executed_size <= 0
            or executed_price <= 0
        ):
            return 0.0, 0.0, 0.0

        snapshot = prepared.snapshot
        if snapshot is None or not isinstance(snapshot, MarketSnapshot):
            snapshot = self.market_data.snapshot(prepared.symbol, pd.Timestamp.utcnow())

        return self._slippage_cost(snapshot, executed_size, executed_price, prepared.side)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _state(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()
        return self._states[symbol]

    def _build_fill(
        self,
        prepared: PreparedOrder,
        executed_size: float,
        executed_price: float,
        execution_date: pd.Timestamp,
    ) -> Optional[ExecutionFill]:
        if executed_size == 0:
            return None

        side = prepared.side
        signed_size = executed_size if side == "buy" else abs(executed_size)
        snapshot = self.market_data.snapshot(prepared.symbol, execution_date)
        slippage_cost = 0.0
        pct_dtv = 0.0
        signed_sqrt_dtv = 0.0

        if self.config.slippage.enabled and signed_size > 0 and executed_price > 0:
            slippage_cost, pct_dtv, signed_sqrt_dtv = self._slippage_cost(
                snapshot,
                signed_size,
                executed_price,
                side,
            )

        self._update_state(prepared.symbol, side, prepared.was_liquidity_capped)

        return ExecutionFill(
            symbol=prepared.symbol,
            side=side,
            date=execution_date,
            size=int(round(signed_size)),
            price=float(executed_price),
            slippage_cost=float(slippage_cost),
            snapshot=snapshot,
            pct_dtv=float(pct_dtv),
            signed_sqrt_dtv=float(signed_sqrt_dtv),
            was_liquidity_capped=prepared.was_liquidity_capped,
        )

    def _apply_liquidity_cap(
        self,
        snapshot: MarketSnapshot,
        intended_size: int,
        price: float,
    ) -> tuple[int, bool]:
        fraction = max(self.config.liquidity.cap_fraction_of_adv, 0.0)
        median_dv = max(snapshot.median_dollar_volume, 0.0)
        max_usd_cap = max(self.config.liquidity.max_usd_cap, 0.0)

        usd_caps = []
        if fraction > 0 and median_dv > 0:
            usd_caps.append(fraction * median_dv)
        if max_usd_cap > 0:
            usd_caps.append(max_usd_cap)

        if not usd_caps:
            return intended_size, False

        usd_limit = min(usd_caps)
        if usd_limit <= 0:
            return 0, True

        max_shares = int(usd_limit / price)
        if max_shares <= 0:
            return 0, True
        if intended_size > max_shares:
            return max_shares, True
        return intended_size, False

    def _fit_to_cash(
        self,
        symbol: str,
        date: pd.Timestamp,
        snapshot: MarketSnapshot,
        size: int,
        price: float,
        cash_available: float,
    ) -> int:
        if size <= 0:
            return 0
        affordable = min(size, int(cash_available // price))
        if affordable <= 0:
            return 0
        if not self.config.slippage.enabled:
            return affordable

        current = affordable
        iteration = 0
        for _ in range(100):
            if current <= 0:
                return 0
            slippage_cost, _, _ = self._slippage_cost(snapshot, current, price, "buy")
            total_cost = current * price + slippage_cost
            if self._debug_records is not None:
                self._debug_records.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "iteration": iteration,
                        "intended_size": size,
                        "candidate_size": current,
                        "price": price,
                        "slippage_cost": slippage_cost,
                        "total_cost": total_cost,
                        "cash_available": cash_available,
                    }
                )
            if total_cost <= cash_available + 1e-6:
                return current
            deficit = total_cost - cash_available
            step = max(1, int(math.ceil(deficit / price)))
            current -= step
            iteration += 1
        return 0

    def _slippage_cost(
        self,
        snapshot: MarketSnapshot,
        size: float,
        price: float,
        side: str,
    ) -> tuple[float, float, float]:
        dollar_size = size * price
        adv = max(snapshot.adv, 1.0)
        pct_dtv = (dollar_size / adv) * 100.0
        signed_sqrt_dtv = math.copysign(math.sqrt(abs(pct_dtv)), pct_dtv)
        buysell_dummy = 1.0 if side == "buy" else -1.0
        log_market_cap = snapshot.log_market_cap
        if math.isnan(log_market_cap):
            log_market_cap = self.config.slippage.fallback_log_market_cap

        slippage_bps = (
            self.config.slippage.coef_beta_indexret * snapshot.beta * snapshot.sp500_return * buysell_dummy
            + self.config.slippage.coef_log_market_cap * log_market_cap
            + self.config.slippage.coef_pct_dtv * pct_dtv
            + self.config.slippage.coef_signed_sqrt_dtv * signed_sqrt_dtv
            + self.config.slippage.coef_idiosync_vol * snapshot.idiosyncratic_vol
            + self.config.slippage.coef_vix * snapshot.vix
        )
        slippage_pct = slippage_bps / 10_000.0
        slippage_cost = size * price * slippage_pct
        return slippage_cost, pct_dtv, signed_sqrt_dtv

    def _update_state(self, symbol: str, side: str, was_capped: bool) -> None:
        if not (self.config.liquidity.enabled and self.config.liquidity.continue_after_cap):
            return
        state = self._state(symbol)
        if side == "buy":
            state.continue_buying = bool(was_capped)
            state.recent_sell = False
        else:
            state.recent_sell = True
            state.continue_buying = True
