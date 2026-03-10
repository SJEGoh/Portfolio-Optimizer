import streamlit as st
from helper import get_matrices, get_ticker_expected, plot_efficient_frontier, get_market_caps, get_black_litterman, get_full_portfolio_df
from helper import run_hrp_optimization, run_cvar_optimization, get_returns, run_nrp_optimization, run_erp_optimization
from helper import fit_model
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
import pandas as pd

def main():
    st.set_page_config(page_title="Portfolio Optimizer", layout="wide", initial_sidebar_state="collapsed")
    st.title("Portfolio Optimizer")
    c1, c2 = st.columns([0.4, 0.6])

    with c1:
        st.header("Tickers")
        st.divider()
        tickers = st.text_input("Enter tickers (eg. SPY, TLT, AAPL)")
        lookback_period = st.number_input("Lookback Days",
                                          value = 365)
        models = st.multiselect(
            "Select Models",
            options = ["Markowitz", "Black-Litterman", "Naive Risk Parity", "ERC Risk Parity", "Hierarchal Risk Parity", "CVAR"]
            )
    with c2:
        st.header("Further params")
        st.divider()
        start_date = (date.today() - timedelta(days=lookback_period)).strftime("%Y-%m-%d")
        if not tickers or not start_date:
            st.stop()
        basket = [x.strip() for x in tickers.strip().split(",")]
        price_data = get_full_portfolio_df(basket, start_date = start_date)
        cov_matrix, _ = get_matrices(basket, start_date = start_date)
        expected = get_ticker_expected(basket, start_date = start_date)
        models_to_run, traces = fit_model(models, basket, expected, cov_matrix, price_data)
    if not traces:
        traces, weights = plot_efficient_frontier(expected, cov_matrix, basket)
    fig = go.Figure()
    if isinstance(traces, dict):
        for group, (trace_list, _) in traces.items():
            for i, trace in enumerate(trace_list):
                # 1. Access the Scatter object INSIDE the list
                trace.legendgroup = group
                
                # 2. Assign the legend name only to the first trace (the Frontier line)
                if i == 0:
                    trace.name = group
                    trace.showlegend = True
                else:
                    # Hide 'Max Sharpe' and 'Individual Assets' from the legend
                    trace.showlegend = False
                    
                fig.add_trace(trace)
    else: 
        fig = go.Figure(data = traces)

    color_map = {
        "Markowitz": "#ff7f0e",           # Muted Blue
        "Black-Litterman": "#1f77b4",      # Safety Orange
        "CVAR": "#2ca02c",                  # Success Green
        "Naive Risk Parity": "#9467bd",     # Amethyst Purple
        "ERC Risk Parity": "#8c564b",       # Terracotta Brown
        "Hierarchal Risk Parity": "#d62728" # Institutional Red
    }
    MPT_COLOR = "#7f7f7f"  # Neutral Gray for historical/actual
    BL_COLOR = "#00BFFF"   # Deep Sky Blue for AI-Adjusted views
# 3. UPDATE COLORS GLOBALLY
    for trace in fig.data:
        # Match the color to the legend group name
        if trace.legendgroup in color_map:
            model_color = color_map[trace.legendgroup]
            
            # Apply to lines (The Frontier)
            if hasattr(trace, 'line') and trace.line:
                trace.line.color = model_color
                
            # Apply to markers (The Max Sharpe Star)
            # We check name so we don't accidentally color the 'Individual Assets' dots
            if hasattr(trace, 'marker') and trace.marker and "Individual Assets" not in trace.name:
                trace.marker.color = model_color
        if trace.name == "Individual Assets":
        
            if trace.legendgroup == "Markowitz":
                trace.marker.color = MPT_COLOR
                trace.name = "Actual Assets (MPT)"
                
            elif trace.legendgroup == "Black-Litterman":
                trace.marker.color = BL_COLOR
                trace.name = "Adjusted Assets (BL View)"
                # Let the BL dots show up in the legend for clarity
                trace.showlegend = True

    # 4. FINAL STYLING
    fig.update_layout(template="plotly_white", hovermode="closest")
    

    results = []
    for name, s in zip(models, models_to_run):
        # Logic: model_function(**all_parameters_in_dict)
        weights = s['model'](**s['model_params'])
        ret, vol, sharpe = get_returns(weights, price_data, expected, cov_matrix)
        results.append({"name": name, "weights": weights, "metrics": (ret, vol, sharpe)})
        if name == "Markowitz" or name == "Black-Litterman":
            continue

        current_port_trace = go.Scatter(
            x=[vol], 
            y=[ret],
            mode='markers',
            name=f'{name}',
            marker=dict(
                color=color_map[name], 
                size=12, 
                symbol='diamond',
                line=dict(width=2, color='white')
            ),
            # Add the Sharpe Ratio to the hover text for the judges
            hovertemplate=f"Volatility: %{{x:.2%}}<br>Return: %{{y:.2%}}<br>Sharpe: {sharpe:.2f}<extra></extra>"
        )

        # Add it to your figure
        fig.add_trace(current_port_trace)

    st.plotly_chart(fig, theme = None)

if __name__ == "__main__":
    main()
