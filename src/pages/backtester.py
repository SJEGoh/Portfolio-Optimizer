import streamlit as st
from pathlib import Path
import sys
import pandas as pd
import plotly.graph_objects as go

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from helper import run_10yr_backtest, run_mpt_optimization, extract_portfolio_metrics, calculate_average_turnover, create_ticker_color_map, draw_custom_header
from helper import get_full_portfolio_df, fit_model, run_10yr_backtest, plot_backtest_results, plot_turnover_analysis, plot_strategy_performance, run_spy_benchmark

def main():
    draw_custom_header()
    st.title("Backtest Portfolio Optimizers")
    st.set_page_config(page_title="Seasonality Analysis", layout="wide", initial_sidebar_state="collapsed")

    c1, c2 = st.columns([0.4, 0.6])

    with c1:
        st.header("Tickers")
        st.divider()
        tickers = st.text_input("Enter backtest tickers (eg. SPY, TLT, AAPL)")
        lookback_period = st.number_input("Lookback Days",
                                          value = 365)
        rebalance_days = st.number_input("Days between rebalancing",
                                    value = 365)
        backtest_length = st.text_input("Backtest Start Date (eg. 2015-01-01)",
                                        value = "2015-01-01")
        models = st.multiselect(
            "Select Models to Backtest",
            options = ["Markowitz", "Naive Risk Parity", "ERC Risk Parity", "Hierarchal Risk Parity", "CVAR"]
            )
    with c2:
        st.header("Further params")
        st.divider()
        if not tickers or not models:
            st.stop()
        

        basket = [x.strip().upper() for x in tickers.strip().split(",")]
        full_price_data = get_full_portfolio_df(basket, start_date = backtest_length)
        models_to_run, _ = fit_model(models)

        backtest_data = {}
        for name, model in zip(models, models_to_run):
            backtest_data[name] = run_10yr_backtest(full_price_data, lambda data: model['model'](data, **model['model_params']),
                                              rebalance_days = rebalance_days, lookback_days = lookback_period)


    spy_ec = run_spy_benchmark(get_full_portfolio_df(["SPY"], start_date = backtest_length))
    sample_weights = backtest_data['Hierarchal Risk Parity'][1]
    rebalance_dates = sample_weights.index
    universe = sample_weights.columns

# 2. Create the 100% SPY weights DataFrame
    spy_weights = pd.DataFrame(0.0, index=rebalance_dates, columns=universe)
    spy_weights['SPY'] = 1.0
    backtest_data['SPY Benchmark'] = (spy_ec, spy_weights)
    fig, metrics = plot_backtest_results(backtest_data)
    st.plotly_chart(fig, width = "stretch")

    tab_summary, *strategy_tabs = st.tabs(["📊 Executive Summary"] + [f"🎯 {m}" for m in metrics.keys()])
    with tab_summary:
        st.subheader("Summary Statistics")
        
        df_metrics = pd.DataFrame(metrics).T
        df_metrics.index.name = 'Strategy'
        df_metrics = df_metrics.reset_index()

        # 3. Display with Styling
        st.subheader("🏆 Strategy Performance Leaderboard")
        st.dataframe(
            df_metrics.style.format({"Sharpe Ratio": "{:.2f}"}),
            width = "stretch"
        )
    ticker_color_map = create_ticker_color_map(basket+["SPY Benchmark"])
    mod_spy = models + ["SPY Benchmark"]
    for i, tab in enumerate(strategy_tabs):
        with tab:
            curr = mod_spy[i]
            curr_metrics = metrics[curr]
            curr_ec, curr_weight = backtest_data[curr]
            col1, col2 = st.columns([1, 1])
            with col1:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("CAGR", curr_metrics["CAGR"])
                with c2:
                    st.metric("Volatility", curr_metrics["Volatility"])
                with c3:
                    st.metric("Sharpe Ratio", f"{curr_metrics["Sharpe Ratio"]:.2f}")
                st.plotly_chart(plot_strategy_performance(curr_ec, model_name = curr))
            with col2:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Max Drawdown", curr_metrics["Max Drawdown"])
                with c2:
                    st.metric("Average Turnover", f"{calculate_average_turnover(curr_weight):.2%}") 
                st.plotly_chart(plot_turnover_analysis(curr_weight,ticker_color_map, model_name = curr))
            
                

if __name__ == "__main__":
    main()
