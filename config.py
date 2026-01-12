"""
Configuration constants for the Delta Neutral Carry Trade application.
All magic numbers and system constraints are defined here.
"""
import pandas as pd
import numpy as np

# === EXCHANGE LIMITS ===
class ExchangeLimits:
    """Hard leverage limits imposed by exchanges"""
    HYPERLIQUID_MAX_LEVERAGE = 20.0   # Conservative altcoin cap
    HYPERLIQUID_MAINTENANCE_MARGIN = 0.05  # 5% maintenance requirement
    
    IBKR_MAX_LEVERAGE = 6.6           # Portfolio Margin cap (~15% req)
    
    # Volatility-based safety factor (3-sigma rule)
    VOL_SAFETY_FACTOR = 3.0


# === STRATEGY DEFAULTS ===
class StrategyDefaults:
    """Default parameter values for the strategy"""
    CAPITAL = 1_000_000
    LEVERAGE = 1.0
    HL_SPLIT_PCT = 0.30               # 30% to Hyperliquid
    BENCHMARK_RATE = 0.0364           # 3.64% (Jan 2026 Fed Rate)
    EXECUTION_COST_BPS = 12.0         # Taker fees in basis points


# === IBKR INTEREST RATE TIERS ===
class IBKRTiers:
    """Interactive Brokers Pro margin loan tiers (USD)"""
    TIER_1_MAX = 100_000
    TIER_1_SPREAD = 0.015             # BM + 1.5%
    
    TIER_2_MAX = 1_000_000
    TIER_2_SPREAD = 0.010             # BM + 1.0%
    
    TIER_3_SPREAD = 0.0075            # BM + 0.75% (above $1M)


# === CHART STYLING ===
class ChartColors:
    """Consistent color scheme across all visualizations"""
    POSITIVE = '#00E396'              # Green (profit)
    NEGATIVE = '#FF4560'              # Red (loss)
    WARNING = '#FEB019'               # Yellow/Orange (caution)
    PRIMARY = '#F9A825'               # Gold (equity curve)
    NEUTRAL = '#2E93fA'               # Blue
    PURPLE = '#AB63FA'                # Purple (distributions)
    ORANGE = '#FFA500'                # Orange (volatility)


# === DATA CONSTRAINTS ===
class DataLimits:
    """Minimum data requirements for analysis"""
    MIN_HOURS_FOR_ANALYSIS = 24
    MIN_HOURS_FOR_METRICS = 24
    ROLLING_WINDOW_7D = 24 * 7        # 168 hours
    ROLLING_WINDOW_30D = 24 * 30      # 720 hours
    
    # Percentile bounds for outlier filtering
    OUTLIER_LOWER = 0.01
    OUTLIER_UPPER = 0.99


# === OPTIMIZER SETTINGS ===
class OptimizerConfig:
    """Grid search parameter ranges"""
    LEVERAGE_MIN = 1.0
    LEVERAGE_MAX = 5.5
    LEVERAGE_STEP = 0.5
    
    SPLIT_MIN = 0.10                  # 10% minimum to HL
    SPLIT_MAX = 0.60                  # 60% maximum to HL
    SPLIT_STEP = 0.10


# === ASSET RANKER THRESHOLDS ===
class RankerThresholds:
    """Yield classification thresholds"""
    HIGH_YIELD = 15.0                 # Green tier
    MODERATE_YIELD = 5.0              # Yellow tier
    # Below MODERATE_YIELD = Red tier


# === DAYS OF WEEK ===
DAYS_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
TRADING_DAYS = [0, 1, 2, 3, 4]        # Monday=0 to Friday=4


# === VALIDATION HELPERS ===
def validate_leverage(lev: float) -> bool:
    """Check if leverage is within safe bounds"""
    return 0.1 <= lev <= 10.0

def validate_split(split: float) -> bool:
    """Check if capital split is valid"""
    return 0.10 <= split <= 0.90

def safe_divide(numerator, denominator, default: float = 0.0):
    """
    Prevent division by zero errors.
    Works with both scalars and pandas Series/arrays.
    
    Args:
        numerator: Value to divide (scalar, Series, or array)
        denominator: Divisor (scalar, Series, or array)
        default: Default value when denominator is zero
        
    Returns:
        Result of division with zeros handled
    """
    # Handle pandas Series
    if isinstance(denominator, pd.Series):
        result = numerator / denominator.replace(0, np.nan)
        return result.fillna(default)
    
    # Handle numpy arrays
    if isinstance(denominator, np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(denominator != 0, numerator / denominator, default)
        return result
    
    # Handle scalars
    return numerator / denominator if denominator != 0 else default