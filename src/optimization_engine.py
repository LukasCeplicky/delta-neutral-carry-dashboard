import numpy as np
import pandas as pd
from strategy import FundingStrategy 

class OptimizationEngine:
    def __init__(self, capital, benchmark_rate):
        # The engine only needs these two for the math
        self.capital = capital
        self.benchmark_rate = benchmark_rate

    def run_grid_search(self, df):
        """
        Runs a Grid Search to find the optimal Leverage and Capital Split.
        """
        # Define Search Space: 1.0x to 5.0x leverage in 0.5 increments
        lev_range = np.arange(1.0, 5.5, 0.5)        
        # 10% to 50% Hyperliquid Split
        split_range = np.arange(0.10, 0.60, 0.10)   
        
        results = []

        for lev in lev_range:
            for split in split_range:
                try:
                    # Instantiate Strategy for this specific combo
                    s = FundingStrategy(self.capital, lev, split, self.benchmark_rate)
                    r = s.run(df)
                    m = s.get_metrics(r)
                    
                    if m:
                        # Calculate peak stress on each leg for safety check
                        hl_lev_series = r['position_usd'] / r['hl_equity'].replace(0, np.nan)
                        ibkr_lev_series = r['position_usd'] / r['ibkr_equity'].replace(0, np.nan)
                        
                        max_hl = hl_lev_series.max()
                        max_ibkr = ibkr_lev_series.max()
                        
                        # Safety Boolean: Stay within exchange hard limits
                        # HL: 20x, IBKR: 6.6x
                        is_safe = (max_hl < 20.0) and (max_ibkr < 6.6)
                        
                        results.append({
                            "Leverage": lev,
                            "Split": split,
                            "APR": m['CAGR'] * 100, # Using CAGR as the yield metric
                            "Safe": is_safe,
                            "Max_HL_Lev": max_hl,
                            "Max_IBKR_Lev": max_ibkr
                        })
                    else:
                        raise ValueError("Simulation metrics failed")
                except Exception:
                    # Mark as failed/liquidated
                    results.append({
                        "Leverage": lev,
                        "Split": split,
                        "APR": 0.0,
                        "Safe": False,
                        "Max_HL_Lev": 999.0,
                        "Max_IBKR_Lev": 999.0
                    })
                    
        return pd.DataFrame(results)