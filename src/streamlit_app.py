import streamlit as st
from helper import get_matrices, get_ticker_expected, plot_efficient_frontier, get_market_caps, get_black_litterman, get_full_portfolio_df
from helper import run_hrp_optimization, run_cvar_optimization, get_returns, run_nrp_optimization, run_erp_optimization
from helper import fit_model

def main():
    st.set_page_config(page_title="Portfolio Optimizer", layout="wide", initial_sidebar_state="collapsed")
    st.title("Portfolio Optimizer")
    c1, c2 = st.columns([0.4, 0.6])

    with c1:
        tickers = st.text_input("Enter tickers (eg. SPY, TLT, AAPL)")
        lookback_period = st.number_input("Lookback Days",
                                          value = 252)
        models = st.multiselect(
            "Select Models",
            options = ["Markowitz", "Black-Litterman", "Naive Risk Parity", "ERC Risk Parity", "Hierarchal Risk Parity", "CVAR"]
            )
    with c2:
        st.write("Further params")
        st.divider()
        models_to_run = fit_model(models)
        print(models_to_run)




    basket = [x.strip() for x in tickers.split(",") if x.strip()]

if __name__ == "__main__":
    main()
