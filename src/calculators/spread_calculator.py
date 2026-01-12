"""
Net spread and yield calculations.
"""
import pandas as pd
import numpy as np
from typing import Dict


class SpreadCalculator:
    """Calculates funding spread and yield metrics"""
    
    @staticmethod
    def calculate_kpi_metrics(res_df: pd.DataFrame, metrics: Dict) -> Dict:
        """
        Calculate key performance indicators for dashboard.
        
        Args:
            res_df: Strategy results dataframe
            metrics: Strategy metrics from get_metrics()
            
        Returns:
            Dict with KPI values
        """
        if res_df.empty:
            return {
                'avg_yield': 0.0,
                'curr_yield': 0.0,
                'cagr': 0.0,
                'max_dd': 0.0,
                'neg_freq': 0.0
            }
        
        # Average yield
        avg_yield = res_df['net_spread_apr'].mean()
        
        # Current yield (last 24h smoothed)
        if len(res_df) >= 24:
            curr_yield = res_df['net_spread_apr'].iloc[-24:].mean()
        else:
            curr_yield = res_df['net_spread_apr'].iloc[-1]
        
        # Negative spread frequency
        neg_hours = (res_df['net_spread_apr'] < 0).sum()
        total_hours = len(res_df)
        neg_freq = (neg_hours / total_hours) * 100 if total_hours > 0 else 0.0
        
        return {
            'avg_yield': avg_yield,
            'curr_yield': curr_yield,
            'cagr': metrics['CAGR'] * 100,
            'max_dd': metrics['MaxDD'] * 100,
            'neg_freq': neg_freq
        }
    
    @staticmethod
    def prepare_spread_data(res_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Separate positive and negative spread for area charts.
        
        Args:
            res_df: Strategy results with 'net_spread_apr' column
            
        Returns:
            Dict with 'positive' and 'negative' series
        """
        spread = res_df['net_spread_apr']
        
        return {
            'positive': spread.apply(lambda x: x if x > 0 else 0),
            'negative': spread.apply(lambda x: x if x < 0 else 0)
        }
    
    @staticmethod
    def calculate_capital_allocation(res_df: pd.DataFrame) -> Dict:
        """
        Calculate final capital allocation metrics.
        
        Args:
            res_df: Strategy results dataframe
            
        Returns:
            Dict with allocation metrics
        """
        if res_df.empty:
            return {}
        
        # Final snapshot
        hl_eq = res_df['hl_equity'].iloc[-1]
        ib_eq = res_df['ibkr_equity'].iloc[-1]
        total_eq = res_df['total_equity'].iloc[-1]
        pos_val = res_df['position_usd'].iloc[-1]
        
        # Percentages
        hl_pct = (hl_eq / total_eq) * 100 if total_eq > 0 else 0
        ib_pct = (ib_eq / total_eq) * 100 if total_eq > 0 else 0
        
        # Leg-specific leverage
        hl_lev = pos_val / hl_eq if hl_eq > 0 else 0
        ib_lev = pos_val / ib_eq if ib_eq > 0 else 0
        
        return {
            'hl_equity': hl_eq,
            'hl_pct': hl_pct,
            'hl_lev': hl_lev,
            'ibkr_equity': ib_eq,
            'ibkr_pct': ib_pct,
            'ibkr_lev': ib_lev,
            'total_equity': total_eq
        }