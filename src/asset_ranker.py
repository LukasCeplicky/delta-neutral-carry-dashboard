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
        errors = []

        for ticker in ticker_list:
            try:
                # 1. Fetch and filter data
                raw_df = engine.get_data(ticker)
                df = filter_func(raw_df, start_date, end_date)

                if df.empty:
                    errors.append(f"{ticker}: Empty after filtering")
                    continue

                if len(df) <= 24:
                    errors.append(f"{ticker}: Only {len(df)} hours (need >24)")
                    continue

                # 2. Calculate the average net funding yield
                # Net APR = (Funding Rate * 24 * 365) - Benchmark Interest
                raw_funding_apr = df['funding'].mean() * 24 * 365
                net_apr = (raw_funding_apr - self.benchmark_rate) * 100

                results.append({
                    "Symbol": ticker,
                    "Avg Net APR": net_apr,
                    "Sample Hours": len(df)
                })
            except Exception as e:
                errors.append(f"{ticker}: {str(e)}")
                continue

        # Debug: Print why assets were skipped
        if errors and not results:
            print(f"\n⚠️ All {len(ticker_list)} assets were filtered out:")
            for err in errors[:5]:  # Show first 5
                print(f"  - {err}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more")

        # 3. Sort by the only metric that matters
        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results).sort_values(by="Avg Net APR", ascending=False)