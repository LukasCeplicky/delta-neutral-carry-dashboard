import pandas as pd
import numpy as np

class AssetRanker:
    def __init__(self, capital, benchmark_rate):
        self.capital = capital
        self.benchmark_rate = benchmark_rate

    def run_ranking(self, ticker_list, engine, filter_func, start_date, end_date):
        """
        Loops through all tickers and ranks them by their Raw Average APR.
        """
        results = []
        
        for ticker in ticker_list:
            try:
                # 1. Fetch and filter data
                raw_df = engine.get_data(ticker)
                df = filter_func(raw_df, start_date, end_date)
                
                if not df.empty and len(df) > 24:
                    # 2. Calculate the average net funding yield
                    # Net APR = (Funding Rate * 24 * 365) - Benchmark Interest
                    raw_funding_apr = df['funding'].mean() * 24 * 365
                    net_apr = (raw_funding_apr - self.benchmark_rate) * 100
                    
                    results.append({
                        "Symbol": ticker,
                        "Avg Net APR": net_apr,
                        "Sample Hours": len(df)
                    })
            except Exception:
                continue # Skip assets with data errors
                
        # 3. Sort by the only metric that matters
        return pd.DataFrame(results).sort_values(by="Avg Net APR", ascending=False)