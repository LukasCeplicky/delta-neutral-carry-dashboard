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
        Runs a Grid Search to find the optimal Safety Factor and Capital Split.
        """
        # Define Search Space: 50% to 100% of max safe leverage
        safety_range = np.arange(0.50, 1.05, 0.05)
        # 10% to 60% Hyperliquid Split
        split_range = np.arange(0.10, 0.65, 0.05)

        results = []

        for safety in safety_range:
            for split in split_range:
                try:
                    # Instantiate Strategy for this specific combo
                    s = FundingStrategy(self.capital, split, self.benchmark_rate, safety)
                    r = s.run(df)
                    m = s.get_metrics(r)

                    if m:
                        # Calculate peak leverage on each leg
                        hl_lev_series = r['position_usd'] / r['hl_equity'].replace(0, np.nan)
                        ibkr_lev_series = r['position_usd'] / r['ibkr_equity'].replace(0, np.nan)

                        max_hl = hl_lev_series.max()
                        max_ibkr = ibkr_lev_series.max()

                        # Safety Boolean: Stay within exchange hard limits
                        # HL: 20x, IBKR: 6.6x
                        is_safe = (max_hl < 20.0) and (max_ibkr < 6.6)

                        results.append({
                            "Safety": safety,
                            "Split": split,
                            "APR": m['CAGR'] * 100,
                            "Safe": is_safe,
                            "Max_HL_Lev": max_hl,
                            "Max_IBKR_Lev": max_ibkr
                        })
                    else:
                        raise ValueError("Simulation metrics failed")
                except Exception:
                    # Mark as failed/liquidated
                    results.append({
                        "Safety": safety,
                        "Split": split,
                        "APR": 0.0,
                        "Safe": False,
                        "Max_HL_Lev": 999.0,
                        "Max_IBKR_Lev": 999.0
                    })

        return pd.DataFrame(results)