"""
Delta Neutral Carry Trade Analysis Dashboard
Refactored: UI logic only, calculations in separate modules
"""
import streamlit as st
import pandas as pd
import sys
import os
from datetime import date

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Core modules
from src.data_engine import DataEngine
from src.strategy import FundingStrategy
from src.optimization_engine import OptimizationEngine

# New utility modules
from src.utils.data_prep import (
    prepare_enhanced_dataframe,
    filter_data_by_date,
    validate_dataframe,
    calculate_streak_statistics,
    calculate_heatmap_data
)
from src.calculators.spread_calculator import SpreadCalculator
from src.calculators.risk_calculator import RiskCalculator
from src.calculators.stats_calculator import StatsCalculator
from src.visualizations.charts import ChartBuilder

# Import config
from config import StrategyDefaults, ExchangeLimits, DataLimits, ChartColors

# === PAGE CONFIG ===
st.set_page_config(
    page_title="Delta Neutral Carry Trade",
    layout="wide",
    page_icon="⚡"
)

# === INITIALIZATION ===
@st.cache_resource
def get_engine():
    return DataEngine()

@st.cache_data(show_spinner=False)
def get_optimization_results(_df, capital, benchmark):
    engine = OptimizationEngine(capital, benchmark)
    return engine.run_grid_search(_df)

# Initialize
engine = get_engine()
charts = ChartBuilder()

with st.sidebar:
    st.header("⚡ Delta Neutral Carry")
    
    # ADD THIS NEW SECTION
    if st.button("🔄 Sync Latest Data", use_container_width=True):
        with st.spinner("Fetching latest funding rates..."):
            try:
                universe = engine.get_universe()
                if universe:
                    engine.update_data(universe)
                    st.success(f"✅ Synced {len(universe)} assets!")
                    st.rerun()
                else:
                    st.error("Failed to fetch asset universe")
            except Exception as e:
                st.error(f"Sync failed: {str(e)}")
    
    st.markdown("---")
    # ... rest of your existing sidebar code continues here

    # Get universe
    try:
        with engine.conn:
            universe = [r[0] for r in engine.conn.execute("SELECT DISTINCT ticker FROM hourly_data")]
            min_ts = engine.conn.execute("SELECT MIN(timestamp) FROM hourly_data").fetchone()[0]
            max_ts = engine.conn.execute("SELECT MAX(timestamp) FROM hourly_data").fetchone()[0]
        g_min = pd.to_datetime(min_ts, unit='ms').date()
        g_max = pd.to_datetime(max_ts, unit='ms').date()
    except Exception as e:
        st.error(f"Database error: {e}")
        universe, g_min, g_max = [], date.today(), date.today()

    if not universe:
        st.error("Database empty. Run main.py to initialize.")
        st.stop()

    # === SIDEBAR INPUTS ===
    selected_asset = st.selectbox("Select Asset", universe, index=0)
    
    st.subheader("Strategy Parameters")
    capital = st.number_input("Total Capital ($)", 100_000, 50_000_000, 1_000_000, step=100_000)

    st.subheader("Risk & Cost")
    hl_split = st.slider("HL Collateral Allocation (%)", 10, 90, 30, 5)
    hl_split_dec = hl_split / 100

    safety_factor = st.slider("Safety Factor (%)", 50, 100, 80, 5) / 100
    st.caption("💡 Uses X% of maximum safe leverage based on exchange limits")

    benchmark = st.number_input("Benchmark Rate (%)", 0.0, 15.0, 3.64, 0.01) / 100
    
    st.subheader("Timeframe")
    date_range = st.date_input("Date Range", (g_min, g_max), min_value=g_min, max_value=g_max)
    
    if not isinstance(date_range, tuple) or len(date_range) != 2:
        st.error("Please select both start and end dates")
        st.stop()
    
    start_date, end_date = date_range

# --- MAIN LOGIC ---
if selected_asset:
    # 1. Load and prepare data
    df_raw = engine.get_data(selected_asset)
    df = filter_data_by_date(df_raw, start_date, end_date)
    
    # Validate data
    is_valid, error_msg = validate_dataframe(df, min_rows=DataLimits.MIN_HOURS_FOR_ANALYSIS)
    if not is_valid:
        st.info(f"Insufficient data: {error_msg}")
        st.stop()
    
    # Enhance dataframe with all derived metrics (SINGLE PASS)
    df = prepare_enhanced_dataframe(df)
    
    # Run strategy simulation
    strat = FundingStrategy(capital, hl_split_dec, benchmark, safety_factor)
    res = strat.run(df)
    metrics = strat.get_metrics(res)
    
    # Validate strategy succeeded
    if metrics is None:
        st.error("🚨 **Strategy Failed Immediately!**")
        st.warning(
            f"At {safety_factor*100:.0f}% safety factor, the account was liquidated in less than 24 hours.\n\n"
            "**Action:** Reduce safety factor or adjust your Capital Split."
        )
        st.stop()

# --- CALCULATE ALL METRICS ONCE (Before Tabs) ---
spread_calc = SpreadCalculator()
risk_calc = RiskCalculator()
stats_calc = StatsCalculator()

kpis = spread_calc.calculate_kpi_metrics(res, metrics)
capital_alloc = spread_calc.calculate_capital_allocation(res)
risk_status = risk_calc.get_risk_status(df, res)
market_dynamics = stats_calc.calculate_market_dynamics(df)

# === MAIN DASHBOARD ===
st.subheader(f"Analysis: {selected_asset}")

# KPI Summary
with st.container(border=True):
    k1, k2, k3, k4, k5 = st.columns(5)
    
    k1.metric("Avg Net Yield (APR)", f"{kpis['avg_yield']:.2f}%",
              help="Historical average of realized Net Spread (Funding - Interest).")
    
    k2.metric("Current Yield (24h)", f"{kpis['curr_yield']:.2f}%",
              delta=f"{kpis['curr_yield'] - kpis['avg_yield']:.2f}% vs Avg",
              help="Average Net Spread over the last 24 hours.")
    
    k3.metric("Proj. CAGR", f"{kpis['cagr']:.2f}%",
              help="Compound Annual Growth Rate based on simulation period.")
    
    k4.metric("Max Drawdown", f"{kpis['max_dd']:.2f}%",
              help="Real account drawdown (net of all costs).")
    
    k5.metric("Neg. Spread Freq", f"{kpis['neg_freq']:.1f}%",
              help="% of time the net spread was negative.")

# 2. DETAILED ANALYSIS TABS
tab_backtest, tab_stats, tab_risk, tab_optimizer = st.tabs([
    "📈 Backtest & Performance",
    "📊 Statistical Analysis",
    "🛡️ Risk & Solvency",
    "🤖 Optimizer"
])

# --- TAB 1: BACKTEST & PERFORMANCE ---
with tab_backtest:
    st.markdown("### 📈 Performance Analysis")
    
    # Equity Curve
    with st.container(border=True):
        st.markdown("**Portfolio Equity Curve**")
        st.caption(f"ℹ️ Initial drop reflects execution costs ({strat.cost_bps} bps taker fee)")
        
        fig_eq = charts.create_equity_curve(res, ChartColors.PRIMARY)
        st.plotly_chart(fig_eq, use_container_width=True)
    
    # Spread Chart
    with st.container(border=True):
        col_head, col_tog = st.columns([3, 1])
        with col_head:
            st.markdown("**Net Spread Yield**")
            st.caption("Green = Profitable Carry | Red = Negative Carry")
        with col_tog:
            filter_outliers = st.checkbox("Filter Outliers", value=False)
        
        spread_data = SpreadCalculator.prepare_spread_data(res)
        fig_spread = charts.create_spread_chart(res, spread_data, filter_outliers)
        st.plotly_chart(fig_spread, use_container_width=True)
    
    # Capital Allocation
    st.markdown("### 🏦 Capital Allocation & Leverage")
    alloc = SpreadCalculator.calculate_capital_allocation(res)

    # Top-level metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Portfolio", f"${alloc['total_equity']:,.0f}")
    with col2:
        position_notional = res['position_usd'].iloc[-1]
        st.metric("Position Notional/Leg", f"${position_notional:,.0f}")
    with col3:
        system_lev = position_notional / alloc['total_equity'] if alloc['total_equity'] > 0 else 0
        st.metric("System Leverage", f"{system_lev:.2f}x",
                 help="Emergent leverage = Position / Total Equity")

    st.markdown("---")

    # Leg details
    col_hl, col_ib = st.columns(2)
    with col_hl:
        with st.container(border=True):
            st.markdown("**⚡ Hyperliquid (Short Perp)**")
            m1, m2 = st.columns(2)
            m1.metric("Allocation", f"${alloc['hl_equity']:,.0f}",
                     f"{alloc['hl_pct']:.1f}% of total")
            m2.metric("Leg Leverage", f"{alloc['hl_lev']:.2f}x",
                     f"Limit: 20x")

    with col_ib:
        with st.container(border=True):
            st.markdown("**🏛️ IBKR (Long Spot)**")
            n1, n2, n3 = st.columns(3)
            n1.metric("Allocation", f"${alloc['ibkr_equity']:,.0f}",
                     f"{alloc['ibkr_pct']:.1f}% of total")
            n2.metric("Loan", f"${res['ibkr_loan'].iloc[-1]:,.0f}")
            n3.metric("Leg Leverage", f"{alloc['ibkr_lev']:.2f}x",
                     f"Limit: 6.6x")

# === TAB 2: STATISTICS ===
with tab_stats:
    st.markdown("### 🧬 Funding DNA & Market Regimes")
    
    # Distribution Analysis
    st.markdown("**1. Yield Distribution Profile**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📊 Frequency (Count)")
        fig_hist = charts.create_histogram(df['funding_apr'])
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.markdown("##### 🎻 Probability Density")
        fig_vio = charts.create_violin_plot(df['funding_apr'])
        st.plotly_chart(fig_vio, use_container_width=True)
    
    # Heatmap
    st.markdown("**2. Temporal Seasonality**")
    heatmap_data, max_val = calculate_heatmap_data(df)
    fig_heat = charts.create_heatmap(heatmap_data, max_val)
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # Streak Stats
    streak_stats = calculate_streak_statistics(df['funding'])
    st.markdown("##### ⏱️ Duration Analysis")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Longest Win", f"{streak_stats['max_win_streak']} Hours")
    k2.metric("Longest Loss", f"{streak_stats['max_lose_streak']} Hours")
    k3.metric("Avg Win", f"{streak_stats['avg_win_streak']:.1f} Hours")
    k4.metric("Current", 
              "Paying" if df['funding'].iloc[-1] < 0 else "Earning",
              delta=f"Last {streak_stats['current_streak_len']}h")
    
    # Market Dynamics
    st.markdown("**3. Market Dynamics & Predictability**")
    dynamics = StatsCalculator.calculate_market_dynamics(df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 🌊 Funding vs. Volatility")
        fig_corr = charts.create_correlation_chart(
            df['datetime'], 
            dynamics['vol_positive'], 
            dynamics['vol_negative']
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        st.markdown("##### 🔮 Yield Stickiness (ACF)")
        fig_acf = charts.create_acf_chart(dynamics['autocorr'])
        st.plotly_chart(fig_acf, use_container_width=True)
    
    # Correlation Structure
    st.markdown("**4. Asset Correlation Structure**")
    fig_comp = charts.create_comparison_chart(
        df['datetime'],
        dynamics['corr_vol'],
        dynamics['corr_price'],
        dynamics['cur_vol'],
        dynamics['cur_price']
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# === TAB 3: RISK ===
with tab_risk:
    st.markdown("### 🛡️ Risk & Solvency Analysis")
    
    # Risk Status
    risk_calc = RiskCalculator()
    risk_status = risk_calc.get_risk_status(df, res)
    
    st.markdown("##### 🚨 Current Risk Status")
    r1, r2, r3 = st.columns([1, 1, 1.5])
    
    with r1:
        st.metric("Equity vs Peak", f"{risk_status['current_dd']:.2f}%",
                 f"Worst: {risk_status['max_dd']:.2f}%", delta_color="inverse")
    
    with r2:
        st.metric("Volatility (30d)", f"{risk_status['vol_30d']*100:.2f}%",
                 f"Last 5d: {risk_status['vol_5d']*100:.2f}%", delta_color="off")
    
    with r3:
        c_hl, c_ibkr = st.columns(2)
        c_hl.metric("HL Safe Max", f"{risk_status['safe_hl']:.1f}x",
                   f"Vol Lim: {risk_status['vol_limit']:.1f}x", delta_color="off")
        c_ibkr.metric("IBKR Safe Max", f"{risk_status['safe_ibkr']:.1f}x",
                     f"Vol Lim: {risk_status['vol_limit']:.1f}x", delta_color="off")
    
    # Drawdown Plot
    st.markdown("##### 🌊 Underwater Plot")
    dd_data = risk_calc.calculate_drawdown_series(res['total_equity'])
    fig_dd = charts.create_drawdown_chart(res['datetime'], dd_data['drawdown_pct'])
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # Leverage Monitor
    st.markdown("##### 🏗️ Venue-Specific Leverage Monitor")
    leg_lev = risk_calc.calculate_leg_leverage(res)
    vol_limit = risk_calc.calculate_rolling_vol_limit(df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Hyperliquid (Perp Leg)**")
        fig_hl = charts.create_leverage_chart(
            res['datetime'], leg_lev['hl'], vol_limit,
            ExchangeLimits.HYPERLIQUID_MAX_LEVERAGE, "HL", ChartColors.POSITIVE
        )
        st.plotly_chart(fig_hl, use_container_width=True)
    
    with col2:
        st.markdown("**IBKR (Spot Leg)**")
        fig_ibkr = charts.create_leverage_chart(
            res['datetime'], leg_lev['ibkr'], vol_limit,
            ExchangeLimits.IBKR_MAX_LEVERAGE, "IBKR", ChartColors.NEUTRAL
        )
        st.plotly_chart(fig_ibkr, use_container_width=True)
    
    # Stress Test
    st.markdown("##### 🎚️ Leverage Sensitivity (Stress Test)")
    df_stress = risk_calc.run_leverage_stress_test(
        df, capital, hl_split_dec, benchmark, FundingStrategy
    )
    
    if not df_stress.empty:
        st.dataframe(df_stress, use_container_width=True, hide_index=True)

# === TAB 4: OPTIMIZER ===
with tab_optimizer:
    st.markdown("### 🤖 Strategy Optimizer")
    
    if st.button("🚀 Run Optimization Grid"):
        with st.spinner("Simulating scenarios..."):
            df_opt = get_optimization_results(df, capital, benchmark)
            
            safe_runs = df_opt[df_opt['Safe'] == True]
            if not safe_runs.empty:
                best = safe_runs.loc[safe_runs['APR'].idxmax()]
                st.success(f"**Optimal:** {best['Safety']:.0%} Safety @ {best['Split']:.0%} HL Split | "
                          f"CAGR: {best['APR']:.2f}%")
            else:
                st.error("No safe configuration found")

            # Heatmap
            pivot_apr = df_opt.pivot(index="Split", columns="Safety", values="APR")
            pivot_safe = df_opt.pivot(index="Split", columns="Safety", values="Safe")
            
            fig_opt = charts.create_optimization_heatmap(pivot_apr, pivot_safe)
            st.plotly_chart(fig_opt, use_container_width=True)