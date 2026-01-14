import pandas as pd
import numpy as np

class FundingStrategy:
    def __init__(self, capital=1_000_000, hl_split_pct=0.30, benchmark_rate=0.0533, safety_factor=0.80):
        """
        Args:
            capital: Total Equity (USD).
            hl_split_pct: % of Total Equity allocated to Hyperliquid (remaining to IBKR).
            benchmark_rate: Fed Funds Rate or Benchmark (default ~5.33%).
            safety_factor: Use X% of maximum safe leverage (0.80 = 80% of max).
        """
        self.capital = capital
        self.hl_split_pct = hl_split_pct
        self.benchmark_rate = benchmark_rate
        self.safety_factor = safety_factor
        
        # Constraints & Costs
        self.cost_bps = 12.0            # Execution fees (bps)
        self.ibkr_max_lev = 6.6         # Portfolio Margin Hard Cap (~15% req)
        self.hl_max_lev = 20.0          # Conservative Altcoin Cap (5% req)
        self.hl_maint_margin = 0.05     # Maintenance Margin

    def _calculate_max_safe_notional(self, hl_allocation, ibkr_allocation):
        """
        Calculate maximum notional we can safely trade on each leg.

        Args:
            hl_allocation: Capital allocated to Hyperliquid
            ibkr_allocation: Capital allocated to IBKR

        Returns:
            Maximum notional (position size on each leg for delta neutral)
        """
        # Max notional HL can support at its leverage limit
        max_hl_notional = hl_allocation * self.hl_max_lev

        # Max notional IBKR can support at its leverage limit
        max_ibkr_notional = ibkr_allocation * self.ibkr_max_lev

        # Delta neutral constraint: use the smaller of the two (limiting factor)
        return min(max_hl_notional, max_ibkr_notional)

    def _calc_tiered_interest(self, loan_amount):
        """
        Calculates hourly interest cost based on IBKR Pro tiered schedule (USD).
        """
        if loan_amount <= 0: return 0.0

        cost = 0.0
        # Tier 1: 0 - 100k (BM + 1.5%)
        t1_amt = min(loan_amount, 100_000)
        cost += t1_amt * (self.benchmark_rate + 0.015)

        # Tier 2: 100k - 1M (BM + 1.0%)
        if loan_amount > 100_000:
            t2_amt = min(loan_amount - 100_000, 900_000)
            cost += t2_amt * (self.benchmark_rate + 0.010)

        # Tier 3: > 1M (BM + 0.75%)
        if loan_amount > 1_000_000:
            t3_amt = loan_amount - 1_000_000
            cost += t3_amt * (self.benchmark_rate + 0.0075)

        return cost / 365 / 24  # Hourly Interest Cost

    def run(self, df):
        if df.empty: return pd.DataFrame()
        df = df.copy()
        
        if 'datetime' not in df.columns and 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Init State
        total_equity = self.capital
        shares = 0.0
        results = []
        
        for i, row in df.iterrows():
            price = row['price']
            funding = row['funding']

            # --- 1. CALCULATE ALLOCATION ---
            hl_allocation = total_equity * self.hl_split_pct
            ibkr_allocation = total_equity * (1 - self.hl_split_pct)

            # --- 2. SET POSITION (Max Safe Notional) ---
            max_safe_notional = self._calculate_max_safe_notional(hl_allocation, ibkr_allocation)
            target_notional = max_safe_notional * self.safety_factor
            
            # --- 3. SOLVENCY CHECKS (Pre-Trade) ---
            # Check IBKR Leg Leverage
            if ibkr_allocation > 0:
                ibkr_lev = target_notional / ibkr_allocation
                if ibkr_lev > self.ibkr_max_lev:
                    results.append(self._fail_row(row, "IBKR Margin Call (Lev > 6.6x)"))
                    break

            # Check HL Leg Leverage
            if hl_allocation > 0:
                hl_lev = target_notional / hl_allocation
                if hl_lev > self.hl_max_lev:
                    results.append(self._fail_row(row, "HL Max Leverage Exceeded (>20x)"))
                    break

            # --- 4. EXECUTION ---
            # Check if rebalancing is needed (±10% tolerance)
            current_notional = shares * price if shares > 0 else 0
            drift_pct = abs(current_notional - target_notional) / target_notional if target_notional > 0 else 999

            if drift_pct > 0.10:  # Only rebalance if drift exceeds 10%
                target_shares = target_notional / price
                diff_shares = abs(target_shares - shares)
                trade_cost = (diff_shares * price) * (self.cost_bps / 10000)

                total_equity -= trade_cost
                shares = target_shares
            # else: keep current position, no trading

            position_val = shares * price

            # --- 5. FINANCING COSTS ---
            ibkr_loan = max(0, position_val - ibkr_allocation)
            hourly_interest = self._calc_tiered_interest(ibkr_loan)
            
            # --- 6. FUNDING INCOME ---
            fund_income = position_val * funding

            # --- 7. NET PnL UPDATE ---
            net_pnl = fund_income - hourly_interest
            total_equity += net_pnl

            # Maintain fixed allocation split (unified portfolio approach)
            # (hl_allocation and ibkr_allocation already calculated at top of loop)

            # --- 8. MAINTENANCE CHECK (Post-PnL) ---
            # HL maintenance margin check
            if position_val > 0 and (hl_allocation / position_val) < self.hl_maint_margin:
                results.append(self._fail_row(row, "HL Liquidation (Equity < 5%)"))
                break

            # IBKR maintenance margin check (10% requirement)
            if position_val > 0 and (ibkr_allocation / position_val) < 0.10:
                results.append(self._fail_row(row, "IBKR Margin Call (Equity < 10%)"))
                break
                
            # --- 9. RECORD ---
            results.append({
                'datetime': row['datetime'],
                'price': price,
                'total_equity': total_equity,
                'position_usd': position_val,
                'hl_equity': hl_allocation,
                'ibkr_equity': ibkr_allocation,
                'ibkr_loan': ibkr_loan,
                'funding_income': fund_income,
                'interest_cost': hourly_interest,
                'funding_apr': funding * 24 * 365 * 100,
                'net_spread_apr': ((fund_income - hourly_interest) * 8760) / (position_val if position_val > 0 else 1) * 100,
                'breakeven_funding': (hourly_interest / position_val) if position_val > 0 else 0
            })
            
            if total_equity <= 0: break

        return pd.DataFrame(results)

    def _fail_row(self, row, reason):
        return {
            'datetime': row['datetime'], 'price': row['price'], 
            'total_equity': 0, 'hl_equity': 0, 'ibkr_equity': 0, 
            'position_usd': 0, 'ibkr_loan': 0,
            'net_spread_apr': 0, 'funding_apr': 0, 'fail_reason': reason
        }

    def get_metrics(self, df):
        if df.empty or len(df) < 24: return None
        final = df['total_equity'].iloc[-1]
        days = (df['datetime'].iloc[-1] - df['datetime'].iloc[0]).total_seconds() / 86400
        cagr = ((final / self.capital) ** (365/max(1, days))) - 1
        peak = df['total_equity'].cummax()
        max_dd = ((df['total_equity'] - peak) / peak).min()
        hourly_ret = df['total_equity'].pct_change().fillna(0)
        sharpe = (hourly_ret.mean() / hourly_ret.std()) * np.sqrt(8760) if hourly_ret.std() > 0 else 0
        return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd, "Final": final}

    def get_spread_stats(self, df):
        if df.empty: return None
        s = df['net_spread_apr']
        pos_hours = (s > 0).sum()
        total = len(s)
        gross_p = s[s>0].sum()
        gross_l = abs(s[s<0].sum())
        
        is_neg = s < 0
        groups = (is_neg != is_neg.shift()).cumsum()
        max_streak = int(is_neg.groupby(groups).sum().max())
        
        return {
            "WinRate": pos_hours / total if total > 0 else 0,
            "MaxLosingStreak": max_streak,
            "ProfitFactor": gross_p/gross_l if gross_l > 0 else 0,
            "Vol": s.std()
        }
    
    # Removed get_risk_analysis as it's now handled by the dashboard gauges directly