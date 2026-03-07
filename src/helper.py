import streamlit as st
from massive import RESTClient
import pandas as pd
from datetime import datetime, date, timedelta
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numpy.linalg import inv

client = RESTClient(st.secrets["POLYGON_API_KEY"])


def get_polygon_data(ticker, frm = "2015-01-01", to = date.today(), timespan = "day"):
    aggs = client.get_aggs(
        ticker=ticker, 
        multiplier=1, 
        timespan=timespan, 
        from_=frm,
        to = to,
        adjusted = True
    )

    df = pd.DataFrame(aggs)[["close", "timestamp"]]
    df["Date"] = pd.to_datetime(df["timestamp"], unit = "ms")
    df = df[["Date", "close"]]
    df.columns = ["Date", "Close"]
    df.set_index("Date", inplace = True)

    return df

def get_matrices(tickers, start_date = "2020-01-01", end_date = date.today()):
    price_df = pd.DataFrame()
    for ticker in tickers:
        price_df[ticker] = get_polygon_data(ticker, frm = start_date, to = end_date)["Close"]
    returns = np.log(price_df / price_df.shift(1)).dropna()

    # Annualize by 252 trading days
    cov_matrix = returns.cov() * 252
    corr_matrix = returns.corr()

    return cov_matrix, corr_matrix

def get_ticker_expected_returns(tickers, start_date = "2020-01-01"):
    price_df = pd.DataFrame()
    
    for ticker in tickers:
        full_history = get_polygon_data(ticker, frm=start_date)
        price_df[ticker] = full_history["Close"]
    
    returns_df = np.log(price_df / price_df.shift(1)).dropna()

    ticker_returns = returns_df.mean() * 252
    
    return ticker_returns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def plot_efficient_frontier(mu, cov_matrix, tickers, target_vol=None, find_max_sharpe=False, risk_free_rate=0.02):
    """
    Plots the Efficient Frontier and returns weights with crosshairs for the selected strategy.
    """
    num_assets = len(tickers)
    sigma = cov_matrix.values
    mu_values = mu.values
    
    # 1. Helper Stats Function
    def get_port_stats(weights):
        p_ret = np.dot(weights, mu_values)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        p_sharpe = (p_ret - risk_free_rate) / p_vol
        return p_ret, p_vol, p_sharpe

    def neg_sharpe(weights): return -get_port_stats(weights)[2]
    def min_vol_func(weights): return get_port_stats(weights)[1]

    # 2. Optimization: Constraints and Bounds
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = [1/num_assets] * num_assets

    # 3. Calculate Frontier Line (Background Curve)
    target_returns = np.linspace(mu_values.min(), mu_values.max(), 50)
    efficient_vols = []
    for target in target_returns:
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.dot(w, mu_values) - target})
        res = minimize(min_vol_func, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        efficient_vols.append(res.fun if res.success else np.nan)

    # 4. Logical Branching for Output & Crosshairs
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(efficient_vols, target_returns, color='#1f77b4', linestyle='--', alpha=0.6, label='Efficient Frontier')

    opt_ret, opt_vol = None, None

    if find_max_sharpe:
        # Optimization for Max Sharpe
        res = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        opt_weights = res.x
        opt_ret, opt_vol, opt_sharpe = get_port_stats(opt_weights)
        
        # Gold Star and Crosshairs for Max Sharpe
        ax.axvline(x=opt_vol, color='gold', linestyle=':', alpha=0.8, linewidth=1.5)
        ax.axhline(y=opt_ret, color='gold', linestyle=':', alpha=0.8, linewidth=1.5)
        ax.scatter(opt_vol, opt_ret, color='gold', marker='*', s=300, edgecolors='black', zorder=15, 
                   label=f'Max Sharpe (SR: {opt_sharpe:.2f})')
        title_suffix = "Max Sharpe Optimization"
    
    elif target_vol is not None:
        # Optimization for specific Risk Budget
        valid_vols = np.array(efficient_vols)
        idx = np.nanargmin(np.abs(valid_vols - target_vol))
        opt_ret = target_returns[idx]
        opt_vol = target_vol # Use the exact target for the line
        
        # Re-run specific optimization for that return to get exact weights
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.dot(w, mu_values) - opt_ret})
        res = minimize(min_vol_func, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        opt_weights = res.x
        
        # Red Crosshairs for Target Vol
        ax.axvline(x=opt_vol, color='red', linestyle=':', alpha=0.8, linewidth=1.5, label=f'Target Risk ({opt_vol:.1%})')
        ax.axhline(y=opt_ret, color='red', linestyle=':', alpha=0.8, linewidth=1.5, label=f'Exp. Return ({opt_ret:.1%})')
        ax.scatter(opt_vol, opt_ret, color='red', marker='X', s=150, zorder=10, label='Target Portfolio')
        title_suffix = f"Risk-Constrained Optimization ({target_vol:.1%})"
    else:
        opt_weights = None
        title_suffix = "Strategic Asset Allocation"

    # 5. Plot Individual Assets
    asset_vols = np.sqrt(np.diag(sigma))
    for i, ticker in enumerate(tickers):
        ax.scatter(asset_vols[i], mu_values[i], s=100, edgecolors='black', alpha=0.8)
        ax.annotate(f" {ticker}", (asset_vols[i], mu_values[i]), fontsize=9, fontweight='bold')

    # Styling
    ax.set_title(f"HSBC Case: {title_suffix}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Annualized Volatility (Risk)")
    ax.set_ylabel("Annualized Expected Return")
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.legend(loc='best', frameon=True)
    ax.set_xlim(left=0)
    
    # Output Weights as pd.Series
    weights_series = pd.Series(opt_weights, index=tickers, name="Optimal weights") if opt_weights is not None else None
    return fig, weights_series

def get_market_caps(tickers):
    mcaps = {}
    for ticker in tickers:
        try:
            # Fetch from Polygon
            details = client.get_ticker_details(ticker)
            
            # 1. Try explicit market_cap first (usually for Stocks)
            m_cap = getattr(details, 'market_cap', None)
            if m_cap:
                mcaps[ticker] = m_cap
            else:
                shares = getattr(details, 'share_class_shares_outstanding', None)
                
                if shares is None:
                    raise AttributeError(f"'{ticker}' has no 'market_cap' or 'weighted_shares_outstanding'")
                
                price_df = get_polygon_data(ticker, frm=(date.today() - timedelta(days=7)).strftime("%Y-%m-%d"))
                if price_df.empty:
                    raise ValueError(f"No price data found for {ticker} to calculate MCap")
                
                last_price = price_df["Close"].iloc[-1]
                mcaps[ticker] = shares * last_price
                
        except Exception as e:
            # This will alert you in the console/app exactly which ticker failed and why
            print(f"CRITICAL ERROR for {ticker}: {str(e)}")
            mcaps[ticker] = None
            
    return pd.Series(mcaps, name="Market_Cap")


def get_black_litterman(cov_matrix, mcaps, views_dict, conf_dict, delta=3.0, tau=0.05):
    tickers = cov_matrix.index
    sigma = cov_matrix.values
    
    # --- Step A: Market Equilibrium (The Prior) ---
    w_mkt = mcaps / mcaps.sum()
    # Pi is what the world thinks. delta=3.0 is Mr. Seng's risk profile.
    pi = delta * sigma.dot(w_mkt.values)
    
    # --- Step B: Quantifying Views (P, Q, Omega) ---
    if not views_dict:
        return pd.Series(pi, index=tickers, name="Implied Returns")

    num_views = len(views_dict)
    P = np.zeros((num_views, len(tickers)))
    Q = np.zeros(num_views)
    omega_diag = []

    for i, (ticker, view_val) in enumerate(views_dict.items()):
        t_idx = list(tickers).index(ticker)
        P[i, t_idx] = 1
        Q[i] = view_val
        
        # Omega (Uncertainty): Scaling variance by (1-conf)/conf
        conf = conf_dict.get(ticker, 0.5)
        conf = max(0.01, min(0.99, conf)) # Boundary safety
        
        # Calculate the variance of the asset scaled by tau
        v_view = P[i].dot(tau * sigma).dot(P[i].T)
        omega_diag.append(v_view * ((1 - conf) / conf))

    Omega = np.diag(omega_diag)

    # --- Step C: The Bayesian Blend (The Master Formula) ---
    # We solve for the Adjusted Expected Returns (mu_bl)
    tau_sigma_inv = inv(tau * sigma)
    p_omega_p = P.T.dot(inv(Omega)).dot(P)
    
    term1 = inv(tau_sigma_inv + p_omega_p)
    term2 = tau_sigma_inv.dot(pi) + P.T.dot(inv(Omega)).dot(Q)
    
    mu_bl = term1.dot(term2)
    
    return pd.Series(mu_bl, index=tickers, name="BL Adjusted Returns")

if __name__ == "__main__":
    basket = ["SPY", "TLT", "GLD", "IVOL"]
    weights = get_market_caps(basket)
    print(weights)
