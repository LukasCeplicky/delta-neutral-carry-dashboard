"""
Statistical analysis calculations for market dynamics.
Handles correlations, autocorrelations, and temporal patterns.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config import DataLimits, DAYS_ORDER


class StatsCalculator:
    """Calculates correlations, autocorrelations, and market dynamics"""
    
    @staticmethod
    def calculate_rolling_correlation(
        series1: pd.Series,
        series2: pd.Series,
        window: int = DataLimits.ROLLING_WINDOW_7D
    ) -> pd.Series:
        """
        Calculate rolling correlation between two series.
        
        Args:
            series1: First time series
            series2: Second time series
            window: Rolling window size (default: 7 days = 168 hours)
            
        Returns:
            Rolling correlation series
        """
        return series1.rolling(window).corr(series2)
    
    @staticmethod
    def separate_positive_negative_corr(corr: pd.Series) -> Dict[str, pd.Series]:
        """
        Split correlation series into positive and negative components for area charts.
        
        Args:
            corr: Correlation series
            
        Returns:
            Dict with 'positive' and 'negative' series
        """
        return {
            'positive': corr.apply(lambda x: x if x > 0 else 0),
            'negative': corr.apply(lambda x: x if x < 0 else 0)
        }
    
    @staticmethod
    def calculate_autocorrelation(series: pd.Series, max_lag: int = 24) -> List[float]:
        """
        Calculate autocorrelation function (ACF) for time series.
        
        Args:
            series: Time series to analyze
            max_lag: Maximum lag to calculate (default: 24 hours)
            
        Returns:
            List of autocorrelation values from lag 1 to max_lag
        """
        return [series.autocorr(lag=lag) for lag in range(1, max_lag + 1)]
    
    @staticmethod
    def calculate_market_dynamics(df: pd.DataFrame, window: int = DataLimits.ROLLING_WINDOW_7D) -> Dict:
        """
        Calculate all market dynamics metrics in one pass.
        
        This includes:
        - Funding vs Volatility correlation (do vol spikes help or hurt?)
        - Funding vs Price Direction correlation (directional bias?)
        - Autocorrelation (is funding predictable?)
        
        Args:
            df: Enhanced dataframe with funding and returns columns
            window: Rolling window for correlations (default: 7 days)
            
        Returns:
            Dict with all correlation metrics
        """
        if df.empty or 'funding' not in df.columns or 'returns' not in df.columns:
            return {
                'corr_vol': pd.Series(),
                'corr_price': pd.Series(),
                'cur_vol': 0.0,
                'cur_price': 0.0,
                'vol_positive': pd.Series(),
                'vol_negative': pd.Series(),
                'autocorr': [0.0] * 24
            }
        
        # Funding vs Volatility (magnitude)
        corr_vol = df['funding'].rolling(window).corr(df['abs_returns'])
        
        # Funding vs Price Direction
        corr_price = df['funding'].rolling(window).corr(df['returns'])
        
        # Current values (latest non-NaN)
        cur_vol = corr_vol.iloc[-1] if not pd.isna(corr_vol.iloc[-1]) else 0.0
        cur_price = corr_price.iloc[-1] if not pd.isna(corr_price.iloc[-1]) else 0.0
        
        # Separate positive/negative for vol correlation (for green/red area chart)
        vol_split = StatsCalculator.separate_positive_negative_corr(corr_vol)
        
        # Autocorrelation (stickiness)
        acf = StatsCalculator.calculate_autocorrelation(df['funding'])
        
        return {
            'corr_vol': corr_vol,
            'corr_price': corr_price,
            'cur_vol': cur_vol,
            'cur_price': cur_price,
            'vol_positive': vol_split['positive'],
            'vol_negative': vol_split['negative'],
            'autocorr': acf
        }
    
    @staticmethod
    def calculate_heatmap_data(df: pd.DataFrame, column: str = 'funding') -> Tuple[pd.DataFrame, float]:
        """
        Prepare data for hour-by-day heatmap visualization.
        
        Args:
            df: Enhanced dataframe with temporal features (day_name, hour)
            column: Column to aggregate (default: 'funding')
            
        Returns:
            Tuple of (pivoted_heatmap_data, max_abs_value_for_colorscale)
        """
        if df.empty or 'day_name' not in df.columns or 'hour' not in df.columns:
            return pd.DataFrame(), 0.0
        
        # Group by day and hour, calculate mean, pivot to matrix
        heatmap = df.groupby(['day_name', 'hour'])[column].mean().unstack()
        
        # Reorder days (Monday first)
        heatmap = heatmap.reindex(DAYS_ORDER)
        
        # Convert to APR percentage
        heatmap = heatmap * 24 * 365 * 100
        
        # Calculate symmetric color scale max (for centered diverging colormap)
        max_val = max(abs(heatmap.min().min()), abs(heatmap.max().max()))
        
        return heatmap, max_val
    
    @staticmethod
    def calculate_streak_statistics(series: pd.Series) -> Dict:
        """
        Calculate winning/losing streak statistics.
        
        Args:
            series: Pandas Series (typically funding rates or returns)
            
        Returns:
            Dict with:
                - max_win_streak: Longest positive streak
                - max_lose_streak: Longest negative streak
                - avg_win_streak: Average positive streak length
                - current_streak_len: Current streak length
        """
        if series.empty:
            return {
                'max_win_streak': 0,
                'max_lose_streak': 0,
                'avg_win_streak': 0.0,
                'current_streak_len': 0
            }
        
        # Identify positive vs negative periods
        is_positive = series > 0
        
        # Create streak IDs (changes when sign flips)
        streak_id = (is_positive != is_positive.shift()).cumsum()
        
        # Group by streak and count length
        streaks = series.to_frame().copy()
        streaks['streak_id'] = streak_id
        streaks['is_positive'] = is_positive
        streak_counts = streaks.groupby(['streak_id', 'is_positive']).size().reset_index(name='length')
        
        # Extract metrics
        win_streaks = streak_counts[streak_counts['is_positive'] == True]['length']
        lose_streaks = streak_counts[streak_counts['is_positive'] == False]['length']
        
        # Current streak length (from last change to end)
        last_streak_id = streak_id.iloc[-1]
        current_streak_len = (streak_id == last_streak_id).sum()
        
        return {
            'max_win_streak': int(win_streaks.max()) if not win_streaks.empty else 0,
            'max_lose_streak': int(lose_streaks.max()) if not lose_streaks.empty else 0,
            'avg_win_streak': float(win_streaks.mean()) if not win_streaks.empty else 0.0,
            'current_streak_len': int(current_streak_len)
        }