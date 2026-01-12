"""
Data preparation and enrichment utilities.
All derived metrics are calculated here in a single pass.
"""
import pandas as pd
import numpy as np
from typing import Tuple
from datetime import date


def prepare_enhanced_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single-pass calculation of all derived metrics.
    Prevents redundant calculations across tabs.
    
    Args:
        df: Raw dataframe with ['datetime', 'price', 'funding']
        
    Returns:
        Enhanced dataframe with all derived columns
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # === PRICE DERIVATIVES ===
    df['returns'] = df['price'].pct_change().fillna(0)
    df['abs_returns'] = df['returns'].abs()
    
    # === VOLATILITY METRICS ===
    df['vol_24h'] = df['returns'].rolling(24).std() * np.sqrt(24) * 100
    df['vol_7d'] = df['returns'].rolling(168).std() * np.sqrt(24) * 100
    
    # === FUNDING DERIVATIVES ===
    df['funding_apr'] = df['funding'] * 24 * 365 * 100
    
    # === TEMPORAL FEATURES ===
    df['hour'] = df['datetime'].dt.hour
    df['day_name'] = df['datetime'].dt.day_name()
    df['day_of_week'] = df['datetime'].dt.dayofweek
    
    return df


def filter_data_by_date(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """
    Filter dataframe by date range with validation.
    
    Args:
        df: Dataframe with 'datetime' column
        start: Start date (inclusive)
        end: End date (inclusive)
        
    Returns:
        Filtered dataframe
    """
    if df.empty:
        return df
        
    mask = (df['datetime'].dt.date >= start) & (df['datetime'].dt.date <= end)
    return df.loc[mask].copy()


def get_active_trading_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to weekdays only (Monday-Friday).
    Useful for volatility calculations that should exclude weekends.
    
    Args:
        df: Enhanced dataframe with 'day_of_week' column
        
    Returns:
        Filtered dataframe (weekdays only)
    """
    if df.empty or 'day_of_week' not in df.columns:
        return df
    
    return df[df['day_of_week'] < 5].copy()


def calculate_streak_statistics(series: pd.Series) -> dict:
    """
    Calculate winning/losing streak statistics.
    
    Args:
        series: Pandas Series (typically funding rates)
        
    Returns:
        Dict with streak stats
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


def calculate_heatmap_data(df: pd.DataFrame, column: str = 'funding') -> Tuple[pd.DataFrame, float]:
    """
    Prepare data for hour-by-day heatmap visualization.
    
    Args:
        df: Enhanced dataframe with temporal features
        column: Column to aggregate (default: 'funding')
        
    Returns:
        Tuple of (pivoted_data, max_abs_value)
    """
    if df.empty:
        return pd.DataFrame(), 0.0
    
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Group by day and hour, convert to APR
    heatmap = df.groupby(['day_name', 'hour'])[column].mean().unstack()
    heatmap = heatmap.reindex(days_order) * 24 * 365 * 100
    
    # Calculate symmetric color scale max
    max_val = max(abs(heatmap.min().min()), abs(heatmap.max().max()))
    
    return heatmap, max_val


def validate_dataframe(df: pd.DataFrame, min_rows: int = 24) -> Tuple[bool, str]:
    """
    Validate that dataframe meets minimum requirements.
    
    Args:
        df: Dataframe to validate
        min_rows: Minimum number of rows required
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "Dataframe is empty"
    
    required_cols = ['datetime', 'price', 'funding']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    
    if len(df) < min_rows:
        return False, f"Insufficient data: {len(df)} rows (minimum: {min_rows})"
    
    return True, ""