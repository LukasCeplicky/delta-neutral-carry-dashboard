"""
Reusable chart building functions.
All Plotly/Altair chart logic centralized here.
"""
import plotly.graph_objects as go
import plotly.figure_factory as ff
import altair as alt
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config import ChartColors


class ChartBuilder:
    """Factory for creating consistent, reusable charts"""
    
    @staticmethod
    def create_equity_curve(res_df: pd.DataFrame, color: str = ChartColors.PRIMARY) -> go.Figure:
        """Create portfolio equity curve chart"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=res_df['datetime'],
            y=res_df['total_equity'],
            mode='lines',
            name='Total Equity',
            line=dict(color=color, width=3)
        ))
        
        fig.update_layout(
            height=350,
            template="plotly_dark",
            margin=dict(l=0, r=0, t=10, b=0),
            hovermode="x unified",
            yaxis=dict(title="Total Equity ($)"),
            xaxis_title=None
        )
        return fig
    
    @staticmethod
    def create_spread_chart(
        res_df: pd.DataFrame,
        spread_data: dict,
        filter_outliers: bool = False
    ) -> go.Figure:
        """Create positive/negative spread area chart"""
        spread = res_df['net_spread_apr']
        
        # Calculate y-range if filtering
        y_range = None
        if filter_outliers:
            p05, p95 = spread.quantile([0.05, 0.95])
            y_range = [p05 * 1.2, p95 * 1.2]
        
        fig = go.Figure()
        
        # Positive carry (green)
        fig.add_trace(go.Scatter(
            x=res_df['datetime'],
            y=spread_data['positive'],
            mode='lines',
            name='Positive Carry',
            line=dict(color=ChartColors.POSITIVE, width=0),
            fill='tozeroy',
            fillcolor='rgba(0, 227, 150, 0.5)',
            hovertemplate='%{y:.2f}%'
        ))
        
        # Negative carry (red)
        fig.add_trace(go.Scatter(
            x=res_df['datetime'],
            y=spread_data['negative'],
            mode='lines',
            name='Negative Carry',
            line=dict(color=ChartColors.NEGATIVE, width=0),
            fill='tozeroy',
            fillcolor='rgba(255, 69, 96, 0.5)',
            hovertemplate='%{y:.2f}%'
        ))
        
        fig.update_layout(
            height=300,
            template="plotly_dark",
            margin=dict(l=0, r=0, t=10, b=0),
            hovermode="x unified",
            yaxis_title="APR (%)",
            yaxis=dict(range=y_range) if y_range else dict(autorange=True),
            showlegend=True,
            legend=dict(orientation="h", y=1.05, x=1, xanchor="right")
        )
        return fig
    
    @staticmethod
    def create_histogram(data: pd.Series, bins: int = 100) -> go.Figure:
        """Create log-scale histogram"""
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=bins,
            name='Frequency',
            marker_color=ChartColors.PURPLE,
            opacity=0.8,
            hovertemplate='APR: %{x:.2f}%<br>Count: %{y}<extra></extra>'
        ))
        
        # Zero line
        fig.add_vline(x=0, line_width=3, line_dash="dash", line_color=ChartColors.NEGATIVE)
        
        fig.update_layout(
            xaxis_title="APR (%)",
            yaxis_title="Hours (Log Scale)",
            yaxis_type="log",
            height=350,
            template="plotly_dark",
            margin=dict(t=10, b=20),
            hovermode="x unified"
        )
        return fig
    
    @staticmethod
    def create_violin_plot(data: pd.Series) -> go.Figure:
        """Create violin plot for distribution"""
        # Filter outliers for display
        p01, p99 = data.quantile([0.01, 0.99])
        
        fig = go.Figure()
        fig.add_trace(go.Violin(
            x=data,
            name='Distribution',
            line_color=ChartColors.POSITIVE,
            side='positive',
            orientation='h',
            points=False,
            meanline_visible=True,
            hovertemplate='APR: %{x:.2f}%<extra></extra>'
        ))
        
        # Zero line
        fig.add_vline(x=0, line_width=3, line_dash="dash", line_color=ChartColors.NEGATIVE)
        
        fig.update_layout(
            xaxis_title="APR (%)",
            xaxis=dict(range=[p01 * 1.2, p99 * 1.2]),
            height=350,
            template="plotly_dark",
            margin=dict(t=10, b=20),
            showlegend=False,
            hovermode="closest"
        )
        return fig
    
    @staticmethod
    def create_heatmap(heatmap_data: pd.DataFrame, max_val: float) -> go.Figure:
        """Create hour-by-day heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlGn',
            zmid=0,
            zmin=-max_val,
            zmax=max_val,
            hovertemplate='<b>%{y} @ %{x}:00 UTC</b><br>Avg APR: %{z:.2f}%<extra></extra>',
            colorbar=dict(title='Avg APR', ticksuffix='%')
        ))
        
        fig.update_layout(
            height=350,
            template="plotly_dark",
            margin=dict(t=0, b=20),
            xaxis=dict(title="Hour of Day (UTC)", tickmode='linear', tick0=0, dtick=1, showgrid=False),
            yaxis=dict(showgrid=False)
        )
        return fig
    
    @staticmethod
    def create_correlation_chart(
        datetime: pd.Series,
        pos_corr: pd.Series,
        neg_corr: pd.Series
    ) -> go.Figure:
        """Create positive/negative correlation area chart"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=datetime, y=pos_corr,
            mode='lines', name='Vol Boosts Yield',
            line=dict(color=ChartColors.POSITIVE, width=0),
            fill='tozeroy', fillcolor='rgba(0, 227, 150, 0.5)',
            hovertemplate='Corr: %{y:.2f}'
        ))
        
        fig.add_trace(go.Scatter(
            x=datetime, y=neg_corr,
            mode='lines', name='Vol Kills Yield',
            line=dict(color=ChartColors.NEGATIVE, width=0),
            fill='tozeroy', fillcolor='rgba(255, 69, 96, 0.5)',
            hovertemplate='Corr: %{y:.2f}'
        ))
        
        fig.update_layout(
            height=300,
            template="plotly_dark",
            margin=dict(t=10, b=20),
            yaxis_title="Correlation",
            yaxis=dict(range=[-1, 1]),
            showlegend=False
        )
        return fig
    
    @staticmethod
    def create_acf_chart(acf_values: list) -> go.Figure:
        """Create autocorrelation function bar chart"""
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(1, len(acf_values) + 1)),
            y=acf_values,
            name='Autocorrelation',
            marker_color=ChartColors.NEUTRAL,
            hovertemplate='Lag %{x}h: %{y:.2f}<extra></extra>'
        ))
        
        fig.add_hline(y=0.2, line_width=2, line_dash="dash", line_color="white", annotation_text="Noise (0.2)")
        
        fig.update_layout(
            height=300,
            template="plotly_dark",
            margin=dict(t=10, b=20),
            xaxis_title="Lag (Hours)",
            yaxis_title="Correlation",
            yaxis=dict(range=[0, 1]),
            showlegend=False
        )
        return fig
    
    @staticmethod
    def create_comparison_chart(
        datetime: pd.Series,
        corr_vol: pd.Series,
        corr_price: pd.Series,
        cur_vol: float,
        cur_price: float
    ) -> go.Figure:
        """Create dual correlation comparison chart"""
        fig = go.Figure()
        
        # Volatility correlation (orange)
        fig.add_trace(go.Scatter(
            x=datetime, y=corr_vol,
            mode='lines',
            name=f'vs Volatility ({cur_vol:.2f})',
            line=dict(color=ChartColors.ORANGE, width=2),
            hovertemplate='Vol Corr: %{y:.2f}'
        ))
        
        # Price correlation (purple)
        fig.add_trace(go.Scatter(
            x=datetime, y=corr_price,
            mode='lines',
            name=f'vs Price Action ({cur_price:.2f})',
            line=dict(color=ChartColors.PURPLE, width=2),
            hovertemplate='Price Corr: %{y:.2f}'
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="white")
        
        fig.update_layout(
            height=350,
            template="plotly_dark",
            margin=dict(t=10, b=20),
            yaxis_title="Correlation (-1 to 1)",
            yaxis=dict(range=[-1, 1]),
            legend=dict(orientation="h", y=1.05, x=0)
        )
        return fig
    
    @staticmethod
    def create_drawdown_chart(datetime: pd.Series, drawdown_pct: pd.Series) -> go.Figure:
        """Create underwater (drawdown) chart"""
        max_dd = drawdown_pct.min()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=datetime, y=drawdown_pct,
            mode='lines', name='Drawdown',
            fill='tozeroy',
            line=dict(color=ChartColors.NEGATIVE, width=1),
            hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            height=250,
            template="plotly_dark",
            margin=dict(t=10, b=20),
            yaxis=dict(range=[min(max_dd * 1.1, -0.05), 0.05], ticksuffix="%")
        )
        return fig
    
    @staticmethod
    def create_leverage_chart(
        datetime: pd.Series,
        leverage: pd.Series,
        vol_limit: pd.Series,
        hard_limit: float,
        venue: str,
        color: str
    ) -> go.Figure:
        """Create venue-specific leverage monitoring chart"""
        curr_val = leverage.iloc[-1]
        y_max = 25 if venue == "HL" else 15
        
        fig = go.Figure()
        
        # Actual leverage
        fig.add_trace(go.Scatter(
            x=datetime, y=leverage,
            mode='lines',
            name=f'Actual Lev ({curr_val:.2f}x)',
            line=dict(color=color, width=2),
            fill='tozeroy',
            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)',
            hovertemplate='%{y:.2f}x<extra></extra>'
        ))
        
        # Hard limit
        fig.add_hline(y=hard_limit, line_dash="dash", line_color=ChartColors.NEGATIVE, 
                     annotation_text=f"Limit ({hard_limit}x)")
        
        # Volatility limit
        fig.add_trace(go.Scatter(
            x=datetime, y=vol_limit.clip(upper=y_max),
            mode='lines',
            name='Vol Limit (3σ)',
            line=dict(color=ChartColors.WARNING, width=2, dash='dot'),
            hovertemplate='Vol Limit: %{y:.1f}x<extra></extra>'
        ))
        
        fig.update_layout(
            height=300,
            template="plotly_dark",
            margin=dict(t=30, b=10, l=10, r=10),
            yaxis=dict(title="Leverage (x)", range=[0, y_max], ticksuffix="x"),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig
    
    @staticmethod
    def create_optimization_heatmap(pivot_apr: pd.DataFrame, pivot_safe: pd.DataFrame) -> go.Figure:
        """Create optimization grid heatmap"""
        x_labels = [f"{c:.1f}x" for c in pivot_apr.columns]
        y_labels = [f"{int(r*100)}%" for r in pivot_apr.index]
        
        # Create annotation text
        z_text = []
        for split_val in pivot_apr.index:
            row_text = []
            for lev_val in pivot_apr.columns:
                is_safe = pivot_safe.loc[split_val, lev_val]
                apr_val = pivot_apr.loc[split_val, lev_val]
                row_text.append(f"{apr_val:.1f}%" if is_safe else "💀")
            z_text.append(row_text)
        
        fig = ff.create_annotated_heatmap(
            z=pivot_apr.values,
            x=x_labels,
            y=y_labels,
            annotation_text=z_text,
            colorscale='Viridis',
            showscale=True
        )
        
        fig.update_layout(
            title="Strategy Yield Heatmap (CAGR %)",
            xaxis_title="System Leverage",
            yaxis_title="Hyperliquid Capital Split (%)",
            template="plotly_dark",
            height=500
        )
        return fig
    
    @staticmethod
    def create_ranking_chart(df_ranked: pd.DataFrame):
        """Create Altair bar chart for asset ranking"""
        color_scale = alt.Scale(
            domain=['🟢 High Yield', '🟡 Moderate', '🔴 Low Yield'],
            range=[ChartColors.POSITIVE, ChartColors.WARNING, ChartColors.NEGATIVE]
        )
        
        chart = alt.Chart(df_ranked).mark_bar().encode(
            x=alt.X('Symbol:N', sort='-y', title="Asset"),
            y=alt.Y('Avg Net APR:Q', title="Net APR (%)", axis=alt.Axis(format=".2f")),
            color=alt.Color('Status:N', scale=color_scale),
            tooltip=[
                alt.Tooltip('Symbol:N'),
                alt.Tooltip('Avg Net APR:Q', format=".2f", title="Net APR (%)"),
                alt.Tooltip('Status:N')
            ]
        ).properties(height=400)
        
        return chart