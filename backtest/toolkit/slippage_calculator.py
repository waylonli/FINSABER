"""
Shared slippage calculator module for financial backtesting strategies.

This module provides a comprehensive slippage calculation based on academic research 
that models market impact as a function of multiple factors including:
- Beta × Index Return × Buy/Sell Direction
- Log Market Capitalization
- Percentage of Daily Trading Volume
- Signed Square Root of Trading Volume
- Idiosyncratic Volatility
- Market Volatility (VIX)

The calculation returns slippage in basis points, which is converted to a percentage
for applying to trade execution prices.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


class SlippageCalculator:
    """
    Academic-based slippage calculator for market impact modeling.
    
    Uses regression coefficients from academic literature to predict
    market impact in basis points based on multiple factors.
    """
    
    def __init__(self, 
                 coef_beta_indexret: float = 0.3,
                 coef_log_market_cap: float = -0.2,
                 coef_pct_dtv: float = 0.35,
                 coef_signed_sqrt_dtv: float = 9.32,
                 coef_idiosync_vol: float = 0.32,
                 coef_vix: float = 0.13,
                 adv_lookup: Optional[Dict[pd.Timestamp, float]] = None,
                 vix_lookup: Optional[Dict[pd.Timestamp, float]] = None,
                 idiosync_vol_lookup: Optional[Dict[pd.Timestamp, float]] = None,
                 market_cap_lookup: Optional[Dict[pd.Timestamp, float]] = None,
                 beta_lookup: Optional[Dict[pd.Timestamp, float]] = None,
                 sp500_returns_lookup: Optional[Dict[pd.Timestamp, float]] = None):
        """
        Initialize the slippage calculator with coefficients and lookup tables.
        
        Args:
            coef_beta_indexret: Coefficient for Beta × SP500 return × buysell
            coef_log_market_cap: Coefficient for log market cap (negative = larger stocks have lower slippage)
            coef_pct_dtv: Coefficient for fraction of daily volume (linear term)
            coef_signed_sqrt_dtv: Coefficient for signed square root of fraction (concave term)
            coef_idiosync_vol: Coefficient for idiosyncratic volatility
            coef_vix: Coefficient for market volatility (VIX)
            adv_lookup: Dictionary mapping timestamps to average daily dollar volume
            vix_lookup: Dictionary mapping timestamps to VIX values
            idiosync_vol_lookup: Dictionary mapping timestamps to idiosyncratic volatility
            market_cap_lookup: Dictionary mapping timestamps to log market cap
            beta_lookup: Dictionary mapping timestamps to beta values
            sp500_returns_lookup: Dictionary mapping timestamps to SP500 daily returns
        """
        self.coef_beta_indexret = coef_beta_indexret
        self.coef_log_market_cap = coef_log_market_cap
        self.coef_pct_dtv = coef_pct_dtv
        self.coef_signed_sqrt_dtv = coef_signed_sqrt_dtv
        self.coef_idiosync_vol = coef_idiosync_vol
        self.coef_vix = coef_vix
        
        # Store lookup tables
        self.adv_lookup = adv_lookup or {}
        self.vix_lookup = vix_lookup or {}
        self.idiosync_vol_lookup = idiosync_vol_lookup or {}
        self.market_cap_lookup = market_cap_lookup or {}
        self.beta_lookup = beta_lookup or {}
        self.sp500_returns_lookup = sp500_returns_lookup or {}
    
    def calculate_slippage(self, 
                          trade_date: pd.Timestamp,
                          dollar_size: float,
                          is_buy: bool,
                          symbol: str = "UNKNOWN") -> Tuple[float, float, Dict[str, float]]:
        """
        Calculate slippage for a trade based on multiple factors.
        
        Args:
            trade_date: Date of the trade
            dollar_size: Dollar value of the trade
            is_buy: True if buying, False if selling
            symbol: Stock symbol for logging
            
        Returns:
            Tuple of (slippage_bps, slippage_pct, components_dict)
            - slippage_bps: Slippage in basis points
            - slippage_pct: Slippage as decimal percentage
            - components_dict: Dictionary with all calculated components
        """
        # Get lookup values for this date
        adv = max(self.adv_lookup.get(trade_date, 1.0), 1.0)
        vix = self.vix_lookup.get(trade_date, 0.0)
        idiosync_vol = self.idiosync_vol_lookup.get(trade_date, 0.0)
        log_market_cap = self.market_cap_lookup.get(trade_date, 0.0)
        beta = self.beta_lookup.get(trade_date, 0.0)
        sp500_return = self.sp500_returns_lookup.get(trade_date, 0.0)
        
        # Calculate derived variables
        buysell_dummy = 1.0 if is_buy else -1.0
        beta_indexret_buysell = beta * sp500_return * buysell_dummy
        
        # Calculate volume-based variables
        pct_dtv = (dollar_size / adv) * 100  # Fraction of daily dollar volume as percentage
        signed_sqrt_dtv = np.sign(pct_dtv) * np.sqrt(abs(pct_dtv))  # Signed square root
        
        # Calculate slippage using academic formula
        slippage_bps = (self.coef_beta_indexret * beta_indexret_buysell + 
                       self.coef_log_market_cap * log_market_cap + 
                       self.coef_pct_dtv * pct_dtv + 
                       self.coef_signed_sqrt_dtv * signed_sqrt_dtv + 
                       self.coef_idiosync_vol * idiosync_vol + 
                       self.coef_vix * vix)
        
        slippage_pct = slippage_bps / 10000  # Convert basis points to decimal percentage
        
        # Store all components for debugging/logging
        components = {
            'date': trade_date,
            'symbol': symbol,
            'dollar_size': dollar_size,
            'adv': adv,
            'vix': vix,
            'idiosync_vol': idiosync_vol,
            'log_market_cap': log_market_cap,
            'beta': beta,
            'sp500_return': sp500_return,
            'buysell_dummy': buysell_dummy,
            'beta_indexret_buysell': beta_indexret_buysell,
            'pct_dtv': pct_dtv,
            'signed_sqrt_dtv': signed_sqrt_dtv,
            'slippage_bps': slippage_bps,
            'slippage_pct': slippage_pct
        }
        
        return slippage_bps, slippage_pct, components
    
    def calculate_slippage_cost(self, 
                               trade_date: pd.Timestamp,
                               shares: float,
                               price: float,
                               is_buy: bool,
                               symbol: str = "UNKNOWN") -> Tuple[float, Dict[str, float]]:
        """
        Calculate the dollar cost of slippage for a trade.
        
        Args:
            trade_date: Date of the trade
            shares: Number of shares traded
            price: Price per share
            is_buy: True if buying, False if selling
            symbol: Stock symbol for logging
            
        Returns:
            Tuple of (slippage_cost, components_dict)
            - slippage_cost: Dollar cost of slippage
            - components_dict: Dictionary with all calculated components
        """
        dollar_size = abs(shares) * price
        slippage_bps, slippage_pct, components = self.calculate_slippage(
            trade_date, dollar_size, is_buy, symbol
        )
        
        slippage_cost = dollar_size * slippage_pct
        components['shares'] = shares
        components['price'] = price
        components['slippage_cost'] = slippage_cost
        
        return slippage_cost, components
    
    def format_slippage_log(self, components: Dict[str, float]) -> str:
        """
        Format slippage calculation components for logging.
        
        Args:
            components: Dictionary returned by calculate_slippage methods
            
        Returns:
            Formatted string for logging
        """
        return (
            f"[SLIPPAGE] {components['date']} | {components['symbol']} "
            f"shares={components.get('shares', 'N/A')} (${components['dollar_size']:.2f}) "
            f"@ {components.get('price', 'N/A'):.2f} ADV=${components['adv']:.0f} "
            f"VIX={components['vix']:.4f} IdioVol={components['idiosync_vol']:.4f} "
            f"LogMC={components['log_market_cap']:.4f} Beta={components['beta']:.4f} "
            f"SP500Ret={components['sp500_return']:.6f} BiB={components['beta_indexret_buysell']:.6f} "
            f"pct_dtv={components['pct_dtv']:.6f} signed_sqrt_dtv={components['signed_sqrt_dtv']:.6f} "
            f"slip={components['slippage_bps']:.2f}bps({components['slippage_pct']*100:.4f}%) "
            f"cost=${components.get('slippage_cost', 'N/A'):.2f}"
        )
    
    def update_lookups(self, 
                      adv_lookup: Optional[Dict[pd.Timestamp, float]] = None,
                      vix_lookup: Optional[Dict[pd.Timestamp, float]] = None,
                      idiosync_vol_lookup: Optional[Dict[pd.Timestamp, float]] = None,
                      market_cap_lookup: Optional[Dict[pd.Timestamp, float]] = None,
                      beta_lookup: Optional[Dict[pd.Timestamp, float]] = None,
                      sp500_returns_lookup: Optional[Dict[pd.Timestamp, float]] = None):
        """
        Update the lookup tables with new data.
        
        Args:
            adv_lookup: New average daily dollar volume lookup
            vix_lookup: New VIX lookup
            idiosync_vol_lookup: New idiosyncratic volatility lookup
            market_cap_lookup: New market cap lookup
            beta_lookup: New beta lookup
            sp500_returns_lookup: New SP500 returns lookup
        """
        if adv_lookup is not None:
            self.adv_lookup = adv_lookup
        if vix_lookup is not None:
            self.vix_lookup = vix_lookup
        if idiosync_vol_lookup is not None:
            self.idiosync_vol_lookup = idiosync_vol_lookup
        if market_cap_lookup is not None:
            self.market_cap_lookup = market_cap_lookup
        if beta_lookup is not None:
            self.beta_lookup = beta_lookup
        if sp500_returns_lookup is not None:
            self.sp500_returns_lookup = sp500_returns_lookup


def create_slippage_calculator_from_params(params) -> SlippageCalculator:
    """
    Create a SlippageCalculator from strategy parameters.
    
    Args:
        params: Strategy parameters object with slippage coefficients and lookups
        
    Returns:
        Initialized SlippageCalculator instance
    """
    return SlippageCalculator(
        coef_beta_indexret=getattr(params, 'coef_beta_indexret', 0.3),
        coef_log_market_cap=getattr(params, 'coef_log_market_cap', -0.2),
        coef_pct_dtv=getattr(params, 'coef_pct_dtv', 0.35),
        coef_signed_sqrt_dtv=getattr(params, 'coef_signed_sqrt_dtv', 9.32),
        coef_idiosync_vol=getattr(params, 'coef_idiosync_vol', 0.32),
        coef_vix=getattr(params, 'coef_vix', 0.13),
        adv_lookup=getattr(params, 'adv_lookup', None),
        vix_lookup=getattr(params, 'vix_lookup', None),
        idiosync_vol_lookup=getattr(params, 'idiosync_vol_lookup', None),
        market_cap_lookup=getattr(params, 'market_cap_lookup', None),
        beta_lookup=getattr(params, 'beta_lookup', None),
        sp500_returns_lookup=getattr(params, 'sp500_returns_lookup', None)
    ) 