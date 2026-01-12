"""
Risk analysis and solvency calculations.
Separated from UI logic for reusability.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config import ExchangeLimits, safe_divide


class RiskCalculator:
    """Calculates risk metrics and leverage limits"""
    
    def __init__(self):
        self.hl_limit = ExchangeLimits.HYPERLIQUID_MAX_LEVERAGE
        self.ibkr_limit = ExchangeLimits.IBKR_MAX_LEVERAGE
        self.vol_factor = ExchangeLimits.VOL_SAFETY_FACTOR
    
    def calculate_drawdown_series(self, equity_series: pd.Series) -> pd.DataFrame:
        """
        Calculate drawdown metrics from equity curve.
        
        Returns:
            DataFrame with ['cum_max', 'drawdown', 'drawdown_pct']
        """
        result = pd.DataFrame(index=equity_series.index)
        result['equity'] = equity_series
        result['cum_max'] = equity_series.cummax()
        result['drawdown'] = equity_series - result['cum_max']
        result['drawdown_pct'] = safe_divide(result['drawdown'], result['cum_max']) * 100
        
        return result
    
    def calculate_volatility_limit(self, returns: pd.Series, window: int = 720) -> float:
        """
        Calculate safe leverage based on volatility (3-sigma rule).
        
        Args:
            returns: Price returns series
            window: Rolling window in hours (default: 30 days)
            
        Returns:
            Maximum safe leverage multiplier
        """
        if returns.empty or len(returns) < window:
            return 1.0
        
        # Use active trading hours only
        vol = returns.tail(window).std() * np.sqrt(24)
        
        if vol == 0:
            return 20.0  # Fallback for zero volatility
        
        limit = safe_divide(1.0, self.vol_factor * vol, default=20.0)
        return min(limit, 50.0)  # Cap at 50x for sanity
    
    def calculate_rolling_vol_limit(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate time-series of volatility-based safe leverage.
        
        Args:
            df: Enhanced dataframe with 'returns' column
            
        Returns:
            Series of safe leverage limits aligned with df index
        """
        if 'returns' not in df.columns:
            return pd.Series(20.0, index=df.index)
        
        # Filter to active days only
        active_mask = df['day_of_week'] < 5
        active_returns = df.loc[active_mask, 'returns']
        
        # Calculate rolling vol on active data
        rolling_vol = active_returns.rolling(window=720).std() * np.sqrt(24)
        
        # Calculate safe leverage
        safe_lev = safe_divide(1.0, self.vol_factor * rolling_vol, default=20.0)
        
        # Reindex to full timeline and forward-fill weekends
        safe_lev = safe_lev.reindex(df.index).ffill().fillna(20.0)
        
        return safe_lev.clip(upper=50.0)
    
    def calculate_leg_leverage(self, res_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate per-exchange leverage from strategy results.
        
        Args:
            res_df: Strategy results with position and equity columns
            
        Returns:
            Dict with 'hl' and 'ibkr' leverage series
        """
        hl_lev = safe_divide(
            res_df['position_usd'],
            res_df['hl_equity'],
            default=0.0
        )
        
        ibkr_lev = safe_divide(
            res_df['position_usd'],
            res_df['ibkr_equity'],
            default=0.0
        )
        
        return {'hl': hl_lev, 'ibkr': ibkr_lev}
    
    def get_risk_status(self, df: pd.DataFrame, res_df: pd.DataFrame) -> Dict:
        """
        Calculate current risk metrics for dashboard display.
        
        Args:
            df: Enhanced price dataframe
            res_df: Strategy results
            
        Returns:
            Dict with current risk metrics
        """
        # Drawdown metrics
        dd_data = self.calculate_drawdown_series(res_df['total_equity'])
        curr_dd = dd_data['drawdown_pct'].iloc[-1]
        max_dd = dd_data['drawdown_pct'].min()
        
        # Volatility metrics (active days only)
        active_returns = df[df['day_of_week'] < 5]['returns']
        
        if len(active_returns) > 720:
            vol_30d = active_returns.tail(720).std() * np.sqrt(24)
        else:
            vol_30d = active_returns.std() * np.sqrt(24) if not active_returns.empty else 0.05
        
        if len(active_returns) > 120:
            vol_5d = active_returns.tail(120).std() * np.sqrt(24)
        else:
            vol_5d = vol_30d
        
        # Calculate effective limits
        vol_limit = self.calculate_volatility_limit(active_returns)
        safe_hl = min(self.hl_limit, vol_limit)
        safe_ibkr = min(self.ibkr_limit, vol_limit)
        
        return {
            'current_dd': curr_dd,
            'max_dd': max_dd,
            'vol_30d': vol_30d,
            'vol_5d': vol_5d,
            'vol_limit': vol_limit,
            'safe_hl': safe_hl,
            'safe_ibkr': safe_ibkr
        }
    
    def run_leverage_stress_test(
        self,
        df: pd.DataFrame,
        capital: float,
        hl_split: float,
        benchmark: float,
        strategy_class
    ) -> pd.DataFrame:
        """
        Simulate strategy across different leverage levels.
        
        Args:
            df: Enhanced price dataframe
            capital: Total capital
            hl_split: HL allocation percentage
            benchmark: Benchmark rate
            strategy_class: FundingStrategy class (not instance)
            
        Returns:
            DataFrame with stress test results
        """
        scenarios = np.arange(1.0, 4.5, 0.5)
        results = []
        
        for lev in scenarios:
            try:
                strat = strategy_class(capital, lev, hl_split, benchmark)
                res = strat.run(df)
                metrics = strat.get_metrics(res)
                
                if metrics:
                    # Calculate peak stress
                    leg_lev = self.calculate_leg_leverage(res)
                    
                    # Calculate equity volatility
                    temp_df = res.set_index('datetime')
                    daily_equity = temp_df['total_equity'].resample('D').last()
                    daily_rets = daily_equity.pct_change().dropna()
                    equity_vol = daily_rets.std() * np.sqrt(365)
                    
                    results.append({
                        "System Lev": f"{lev}x",
                        "HL Max": leg_lev['hl'].max(),
                        "IBKR Max": leg_lev['ibkr'].max(),
                        "CAGR": metrics['CAGR'] * 100,
                        "Max DD": metrics['MaxDD'] * 100,
                        "Sharpe": metrics['Sharpe'],
                        "Eq Vol": equity_vol * 100
                    })
                else:
                    raise ValueError("Metrics failed")
                    
            except Exception:
                # Mark failed scenarios
                results.append({
                    "System Lev": f"{lev}x",
                    "HL Max": np.nan,
                    "IBKR Max": np.nan,
                    "CAGR": 0.0,
                    "Max DD": 100.0,
                    "Sharpe": 0.0,
                    "Eq Vol": 0.0
                })
        
        return pd.DataFrame(results)