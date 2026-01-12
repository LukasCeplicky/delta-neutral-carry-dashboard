import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.data_engine import DataEngine
from src.strategy import FundingStrategy

# --- CONFIG ---
STRAT_CONFIG = {
    "leverage": 1.0,
    "hl_split_pct": 0.30,  # Added (required)
    "benchmark_rate": 0.0364  # Renamed from ibkr_rate
}

def main():
    print("--- 1. SYSTEM INIT ---")
    engine = DataEngine()
    
    # 2. DATA SYNC
    target_universe = engine.get_universe()
    
    if not target_universe:
        print("CRITICAL: Failed to load universe. Check data_engine.py.")
        return

    print(f"-> Syncing {len(target_universe)} assets...")
    engine.update_data(target_universe)
    
    # 3. STRATEGY EXECUTION
    print(f"\n{'='*85}")
    print(f"{'TICKER':<15} | {'CAGR':<10} | {'SHARPE':<8} | {'MAX DD':<10} | {'STATUS'}")
    print(f"{'-'*85}")
    
    for ticker in target_universe:
        df = engine.get_data(ticker)
        
        if df.empty or len(df) < 24:
            continue
            
        strat = FundingStrategy(
            capital=200_000, 
            leverage=STRAT_CONFIG['leverage'],
            hl_split_pct=STRAT_CONFIG['hl_split_pct'],  # Added
            benchmark_rate=STRAT_CONFIG['benchmark_rate']  # Fixed
        )
        
        try:
            res_df = strat.run(df)
            metrics = strat.get_metrics(res_df)
            
            if metrics:
                status = "Alive" if metrics['Final'] > 0 else "LIQUIDATED"
                print(f"{ticker:<15} | "
                      f"{metrics['CAGR']*100:>6.1f}%   | "
                      f"{metrics['Sharpe']:>6.2f}   | "
                      f"{metrics['MaxDD']*100:>7.2f}%  | "
                      f"{status}")
        except Exception as e:
             print(f"{ticker:<15} | ERR: {str(e)[:30]}")

    print(f"{'='*85}")
    print("Done.")

if __name__ == "__main__":
    main()