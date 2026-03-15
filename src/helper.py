import streamlit as st
from massive import RESTClient
import pandas as pd
from datetime import datetime, date, timedelta
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numpy.linalg import inv
from pypfopt import EfficientCVaR, expected_returns, HRPOpt
import plotly.graph_objects as go
from finbert import ai_bl_params
import plotly.colors as pc
from plotly.subplots import make_subplots

client = RESTClient(st.secrets["POLYGON_API_KEY"])

@st.cache_data(ttl=86400)
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

@st.cache_data(ttl=86400)
def get_full_portfolio_df(tickers, start_date="2015-01-01"):
    all_dfs = []
    
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        try:
            # Call your custom function
            df = get_polygon_data(ticker, frm=start_date)
            
            # Rename 'Close' to the Ticker name for the final join
            df.columns = [ticker]
            all_dfs.append(df)
                
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    # Combine all individual ticker DataFrames into one Wide Format DataFrame
    portfolio_df = pd.concat(all_dfs, axis=1)
    
    # Drop rows where ANY ticker has a NaN (important for HRP/BL math)
    return portfolio_df

@st.cache_data(ttl=86400)
def get_matrices(tickers, start_date = "2020-01-01", end_date = date.today()):
    price_df = pd.DataFrame()
    for ticker in tickers:
        price_df[ticker] = get_polygon_data(ticker, frm = start_date, to = end_date)["Close"]
    returns = np.log(price_df / price_df.shift(1)).dropna()

    # Annualize by 252 trading days
    cov_matrix = returns.cov() * 252
    corr_matrix = returns.corr()

    return cov_matrix, corr_matrix

@st.cache_data(ttl=86400)
def get_ticker_expected(tickers, start_date = "2020-01-01"):
    price_df = pd.DataFrame()
    
    for ticker in tickers:
        full_history = get_polygon_data(ticker, frm=start_date)
        price_df[ticker] = full_history["Close"]
    
    returns_df = np.log(price_df / price_df.shift(1)).dropna()

    ticker_returns = returns_df.mean() * 252
    ticker_volatility = returns_df.std() * np.sqrt(252)
    stats_df = pd.DataFrame({
        'Expected Return': ticker_returns,
        'Volatility': ticker_volatility
    })
    
    return stats_df

def plot_efficient_frontier(stats, cov_matrix, tickers, target_vol=None, find_max_sharpe=False, risk_free_rate=0.02):
    num_assets = len(tickers)
    sigma = cov_matrix.values
    mu_values = stats["Expected Return"].values
    
    def get_port_stats(weights):
        p_ret = np.dot(weights, mu_values)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        p_sharpe = (p_ret - risk_free_rate) / p_vol
        return p_ret, p_vol, p_sharpe

    def min_vol_func(weights): return get_port_stats(weights)[1]
    def neg_sharpe(weights): return -get_port_stats(weights)[2]
    def neg_ret(weights): return -get_port_stats(weights)[0]

    sum_cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = [1/num_assets] * num_assets

    # 1. Background Curve Logic
    res_gmv = minimize(min_vol_func, init_guess, method='SLSQP', bounds=bounds, constraints=sum_cons)
    min_possible_vol = res_gmv.fun
    
    target_returns = np.linspace(get_port_stats(res_gmv.x)[0], mu_values.max(), 50)
    efficient_vols = []
    for target in target_returns:
        cons = [sum_cons, {'type': 'eq', 'fun': lambda w: np.dot(w, mu_values) - target}]
        res = minimize(min_vol_func, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        efficient_vols.append(res.fun if res.success else np.nan)

    traces = []

    # Trace 1: The Frontier Line
    traces.append(go.Scatter(
        x=efficient_vols, y=target_returns,
        mode='lines', name='Efficient Frontier',
        line=dict(dash='dash', width=2),
        opacity=0.6,
        hovertemplate="Frontier - Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>"
    ))

    # 2. Optimization Logic & Points
    opt_weights = None
    if find_max_sharpe:
        res = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=sum_cons)
        opt_weights = res.x
        opt_ret, opt_vol, opt_sharpe = get_port_stats(opt_weights)
        
        # MATCHING DIAMOND TEMPLATE
        traces.append(go.Scatter(
            x=[opt_vol], y=[opt_ret], 
            mode='markers',
            name='Max Sharpe',
            marker=dict(size=14, symbol='star', line=dict(width=2, color='white')),
            hovertemplate=f"Volatility: %{{x:.2%}}<br>Return: %{{y:.2%}}<br>Sharpe: {opt_sharpe:.2f}<extra></extra>"
        ))
    
    if target_vol is not None:
        if target_vol >= min_possible_vol:
            vol_cons = {'type': 'eq', 'fun': lambda w: np.sqrt(np.dot(w.T, np.dot(sigma, w))) - target_vol}
            res = minimize(neg_ret, init_guess, method='SLSQP', bounds=bounds, constraints=[sum_cons, vol_cons])
            opt_weights = res.x
        else:
            opt_weights = res_gmv.x
        
        opt_ret, opt_vol, opt_sharpe = get_port_stats(opt_weights)
        
        # MATCHING DIAMOND TEMPLATE
        traces.append(go.Scatter(
            x=[opt_vol], y=[opt_ret], 
            mode='markers',
            name='Target Risk',
            marker=dict(size=12, symbol='diamond', line=dict(width=2, color='white')),
            hovertemplate=f"Volatility: %{{x:.2%}}<br>Return: %{{y:.2%}}<br>Sharpe: {opt_sharpe:.2f}<extra></extra>"
        ))

    # Trace 3: Individual Assets
    traces.append(go.Scatter(
        x=stats["Volatility"].values, y=mu_values,
        mode='markers+text', text=tickers, textposition="top right",
        name='Individual Assets',
        marker=dict(size=10, color='gray', opacity=0.8, line=dict(width=1, color='black')),
        hovertemplate="<b>%{text}</b><br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>"
    ))
    
    weights_series = pd.Series(opt_weights, index=tickers, name="Optimal weights") if opt_weights is not None else None
    return traces, weights_series

@st.cache_data(ttl=86400)
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

@st.cache_data(ttl=86400)
def get_black_litterman(cov_matrix, mcaps, views_dict, conf_dict, delta=3.0, tau=0.05):
    tickers = cov_matrix.index
    sigma = cov_matrix.values
    
    # --- Step A: Market Equilibrium (The Prior) ---
    w_mkt = np.log(mcaps) / np.log(mcaps).sum()
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

@st.cache_data(ttl=86400)
def run_hrp_optimization(price_data):
    # 1. Calculate historical returns for the correlation matrix
    returns = price_data.pct_change().dropna()
    
    # 2. Initialize HRP
    if price_data.shape[1] < 2:
        return pd.Series(1.0, index=price_data.columns)
        
    # 2. Initialize HRP
    try:
        hrp = HRPOpt(returns)
        weights = hrp.optimize()
        return pd.Series(hrp.clean_weights())
    except Exception as e:
        # Fallback to Equal Weight if the clustering still fails
        n = price_data.shape[1]
        return pd.Series([1/n]*n, index=price_data.columns)

@st.cache_data(ttl=86400)
def run_nrp_optimization(price_data):
    returns = price_data.pct_change().dropna()

    vols = returns.std()

    inv_vols = 1/vols
    weights = inv_vols/inv_vols.sum()
    return pd.Series(weights)

@st.cache_data(ttl=86400)
def run_erp_optimization(price_data):
    # 1. Raw Sample Covariance Matrix (No shrinking)
    returns = price_data.pct_change().dropna()
    cov = returns.cov().values * 252
    n = len(price_data.columns)
    
    # 2. The ERC Goal: Each asset's Risk Contribution = Total Risk / N
    def objective(w):
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        # Risk Contribution = Weight * (Covariance * Weights) / Portfolio Vol
        rc = w * (np.dot(cov, w)) / p_vol
        target_rc = p_vol / n
        return np.sum(np.square(rc - target_rc))

    # 3. Standard constraints: Weights sum to 100%, Long-only (0 to 1)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    bounds = [(0, 1) for _ in range(n)]
    
    # Start with equal weights
    res = minimize(objective, [1/n]*n, bounds=bounds, constraints=constraints)
    
    return pd.Series(res.x, index=price_data.columns)

@st.cache_data(ttl=86400)
def run_mpt_optimization(price_data, target_vol=None, find_max_sharpe=False, risk_free_rate=0.02, is_bl=False, stats = None):
    """
    Standard MPT Solver for Backtesting.
    Supports Black-Litterman by accepting pre-calculated 'stats' (expected returns).
    """
    # 1. Prepare Data (Stats & Covariance)
    tickers = price_data.columns
    returns = price_data.pct_change().dropna()
    num_assets = len(tickers)
    
    # Calculate Annualized Covariance (Sigma remains the same for both)
    sigma = (returns.cov().values) * 252
    
    # LOGIC SWITCH: Use BL adjusted returns if provided, otherwise use historical
    if is_bl is True:
        # Pull the 'Expected Return' column we built in fit_model
        mu_values = stats["Expected Return"].values
    else:
        # Standard historical mean
        mu_values = returns.mean().values * 252
    
    def get_port_stats(weights):
        p_ret = np.dot(weights, mu_values)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        p_sharpe = (p_ret - risk_free_rate) / p_vol
        return p_ret, p_vol, p_sharpe

    # Optimization Functions
    def neg_sharpe(weights): return -get_port_stats(weights)[2]
    def neg_ret(weights): return -get_port_stats(weights)[0]
    def min_vol_func(weights): return get_port_stats(weights)[1]

    # Constraints & Bounds
    sum_cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = [1/num_assets] * num_assets

    # 2. Optimization Logic
    opt_weights = None
    
    if find_max_sharpe:
        res = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=sum_cons)
        opt_weights = res.x
    
    elif target_vol is not None:
        res_gmv = minimize(min_vol_func, init_guess, method='SLSQP', bounds=bounds, constraints=sum_cons)
        
        if target_vol < res_gmv.fun:
            opt_weights = res_gmv.x 
        else:
            vol_cons = {'type': 'eq', 'fun': lambda w: np.sqrt(np.dot(w.T, np.dot(sigma, w))) - target_vol}
            res = minimize(neg_ret, init_guess, method='SLSQP', bounds=bounds, constraints=[sum_cons, vol_cons])
            opt_weights = res.x

    return pd.Series(opt_weights, index=tickers)

@st.cache_data(ttl=86400)
def run_cvar_optimization(price_data, alpha=0.95):
    """
    Optimizes for the Minimum CVaR (Expected Shortfall).
    alpha=0.95 means we are looking at the average of the worst 5% of losses.
    """
    # 1. Calculate daily returns (CVaR needs the raw distribution, not a matrix)
    returns = price_data.pct_change().dropna()
    
    # 2. Initialize EfficientCVaR 
    # We pass 'None' for expected returns to focus purely on risk minimization
    # beta is the confidence level (0.95 corresponds to the 5% tail)
    ec = EfficientCVaR(expected_returns=None, returns=returns, beta=alpha)
    
    # 3. Optimize for the minimum CVaR
    weights = ec.min_cvar()
    
    # 4. Clean and return as a Series
    cleaned_weights = ec.clean_weights()
    return pd.Series(cleaned_weights.values(), index=returns.columns)

def get_returns(cleaned_optimization, price_data, stats, cov_matrix):
    weights = np.array([cleaned_optimization[t] for t in price_data.columns])
    ret = np.dot(weights, stats["Expected Return"].values)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
    sharpe = (ret-0.02)/vol
    return ret, vol, sharpe

def run_10yr_backtest(price_data, strategy_func, rebalance_days=126, lookback_days=252):
    initial_cash = 100.0
    current_value = initial_cash
    equity_curve = []
    equity_dates = []
    weight_history = {}
    
    # Identify Rebalance Dates
    rebalance_indices = np.arange(lookback_days, len(price_data), rebalance_days)
    
    for start_idx in rebalance_indices:
        # A. Look-back & DYNAMIC FILTER
        # We slice the training window
        raw_train_data = price_data.iloc[start_idx - lookback_days : start_idx]
        
        # Only keep tickers that have 100% valid data in this specific window
        # This is where IVOL gets ignored in 2015 but picked up in 2020
        active_tickers = raw_train_data.dropna(axis=1).columns
        
        if len(active_tickers) == 0:
            continue
            
        train_data = raw_train_data[active_tickers]
        
        # B. Optimize (Returns weights ONLY for active tickers)
        weights_active = strategy_func(train_data)
        
        # RE-ALIGN: Map back to the full universe so we don't get indexing errors
        full_weights = pd.Series(0.0, index=price_data.columns)
        full_weights.update(weights_active)
        
        rebalance_date = price_data.index[start_idx]
        weight_history[rebalance_date] = full_weights
        
        # C. Apply (Calculate performance for the next rebalance period)
        end_idx = min(start_idx + rebalance_days, len(price_data))
        
        # We use ffill() here to handle any temporary halts during the forward period
        period_prices = price_data.iloc[start_idx : end_idx].ffill()
        period_returns = period_prices.ffill().pct_change(fill_method=None).fillna(0)
        
        # Calculate daily growth using the FULL weights series
        portfolio_daily_rets = (period_returns * full_weights).sum(axis=1)
        
        for date, ret in portfolio_daily_rets.items():
            current_value *= (1 + ret)
            equity_curve.append(current_value)
            equity_dates.append(date)

    weights_df = pd.DataFrame(weight_history).T
    return pd.Series(equity_curve, index=equity_dates), weights_df

def extract_portfolio_metrics(equity_curve, risk_free_rate=0.02):
    """
    Calculates institutional metrics from an equity curve series.
    risk_free_rate: Annualized (e.g., 0.04 for 4% in 2026).
    """
    # 1. Convert Equity Curve to Daily Returns
    daily_rets = equity_curve.pct_change().dropna()
    
    # 2. Annualized Return (CAGR)
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    cagr = (1 + total_return)**(1/years) - 1
    
    # 3. Annualized Volatility
    vol = daily_rets.std() * np.sqrt(252)
    
    # 4. Sharpe Ratio (The 'Hurdle' metric)
    sharpe = (cagr - risk_free_rate) / vol
    
    # 5. Maximum Drawdown (The 'Safety' metric)
    rolling_max = equity_curve.cummax()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    return {
        "CAGR": f"{cagr:.2%}",
        "Volatility": f"{vol:.2%}",
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown": f"{max_drawdown:.2%}"
    }

def calculate_average_turnover(weights_df):
    """
    Calculates the average portfolio turnover per rebalance period.
    weights_df: Index = Dates, Columns = Tickers, Values = Weights (0 to 1)
    """
    # 1. Calculate the absolute difference between each rebalance
    # .diff() compares current row to previous row
    absolute_diff = weights_df.diff().abs()
    
    # 2. Sum the differences across all assets for each period
    # We divide by 2 because selling 10% of A to buy 10% of B is 10% turnover, not 20%.
    period_turnover = absolute_diff.sum(axis=1) / 2
    
    # 3. Calculate the mean turnover across all periods (ignoring the first NaN row)
    avg_turnover = period_turnover.mean()
    
    return avg_turnover

def get_model(model):
    d = {
        "Markowitz": run_mpt_optimization,
        "Black-Litterman": run_mpt_optimization,
        "Hierarchal Risk Parity": run_hrp_optimization, 
        "CVAR": run_cvar_optimization, 
        "Naive Risk Parity": run_nrp_optimization, 
        "ERC Risk Parity": run_erp_optimization
    }

    return d[model]

def fit_model(models, price_data = None, basket = None, expected = None, cov_matrix = None):
    to_run = []
    # markowitz target vol or max sharpe
    # black-litterman view + confidence (make dropdown for portfolio?)
    # CVAR alpha
    # Risk parities nothing (yay)
    traces = {}
    for model in models:
        params = {}
        temp = {}
        params["model"] = get_model(model)
        if isinstance(price_data, pd.DataFrame):
            temp["price_data"] = price_data
        if model == "Markowitz":
            with st.expander("Markowitz Params"):
                st.write("Markowitz Model Params")
                max_sharpe = st.radio("Max Sharpe",
                        [True, False], key = f"Mark_sharpe_{model}",
                        horizontal = True)
                if max_sharpe == False:
                    target_vol = st.number_input(
                        "Enter Volatility", key = f"Mark_vol_{model}",
                        step = 0.001,
                        min_value = 0.0,
                    )
                temp["find_max_sharpe"] = max_sharpe
                temp["target_vol"] = target_vol if not max_sharpe else None
            if basket:
                traces["Markowitz"] = plot_efficient_frontier(expected, cov_matrix, basket, target_vol = temp["target_vol"], find_max_sharpe = max_sharpe)
        if model == "Black-Litterman":
            with st.expander("Black-Litterman Params"):
                delta = st.number_input(
                "Enter Risk Aversion (1.0 - 10.0)",
                step = 0.001,
                min_value = 0.001,
                value = 5.00
                )
                tau = st.number_input(
                "Enter Market Efficiency (0.01 - 1.00)",
                step = 0.001,
                min_value = 0.0,
                value = 0.05
                )
                # make it expected instead
                view_dict = {}
                conf_dict = {}
                default_views = expected["Expected Return"].to_dict()
                default_conf = {t: 0.0 for t in basket}
                if st.button("Populate with Finbert"):
                    delta_views, default_conf = ai_bl_params(basket)
                    for t in basket:
                        default_views[t] = expected.loc[t]["Expected Return"] + delta_views[t]
                with st.expander("Stocks Views and Confidence", expanded = True):
                    for ticker in basket:
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.write(ticker)
                        with c2:
                            curr_view = st.number_input(
                                f"Enter expected return for {ticker}",
                                step = 0.001,
                                value = default_views[ticker]
                            )
                            default_views[ticker] = curr_view
                        with c3:
                            curr_conf = st.number_input(
                                f"Enter confidence for {ticker}",
                                step = 0.001,
                                min_value = 0.0,
                                value = default_conf[ticker]
                            )
                            default_conf[ticker] = curr_conf
                    max_sharpe = st.radio("Max Sharpe",
                            [True, False], key = f"BL_sharpe_{ticker}",
                            horizontal = True)
                    if max_sharpe == False:
                        target_vol = st.number_input(
                            "Enter Volatility",
                            step = 0.001, key = f"BL_vol_{ticker}",
                            min_value = 0.0,
                        )
                
                temp["find_max_sharpe"] = max_sharpe
                temp["target_vol"] = target_vol if not max_sharpe else None
                mcaps = get_market_caps(basket)
                bl_mu = get_black_litterman(cov_matrix = cov_matrix, mcaps = mcaps, views_dict = default_views,
                                            conf_dict = default_conf, delta = delta, tau = tau)
                t = expected.copy()
                t["Expected Return"] = bl_mu
                temp["price_data"] = price_data
                temp["is_bl"] = True
                temp["stats"] = t
          
            traces["Black-Litterman"] = plot_efficient_frontier(t, cov_matrix, basket, target_vol = temp["target_vol"], find_max_sharpe = max_sharpe)

        if model == "CVAR":
            with st.expander("CVAR params"):
                st.write("CVAR Model Params")
                alpha = st.number_input(
                    "Enter alpha",
                    step = 0.001,
                    min_value = 0.0,
                    value = 0.95
                )
                temp["alpha"] = alpha

        params["model_params"] = temp
        to_run.append(params)
    return to_run, traces

def create_ticker_color_map(tickers):
    """
    Creates a deterministic mapping of tickers to colors.
    """
    # 1. Sort tickers alphabetically so 'AAPL' always gets the same index
    sorted_tickers = sorted(list(set(tickers)))
    
    # 2. Use a high-contrast professional palette (e.g., Plotly's D3 or G10)
    # D3 is the industry standard for financial dashboards
    palette = pc.qualitative.D3 
    
    # 3. Create the dictionary (using modulo % to prevent index errors if basket > 10)
    color_map = {
        ticker: palette[i % len(palette)] 
        for i, ticker in enumerate(sorted_tickers)
    }
    
    return color_map

def generate_comparison_bar(results):
    names = [r['name'] for r in results]
    returns = [r['metrics'][0] for r in results]
    volatilities = [r['metrics'][1] for r in results]
    sharpes = [r['metrics'][2] for r in results]
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. Add Expected Return Bar
    fig.add_trace(
        go.Bar(x=names, y=returns, name='Expected Return', marker_color='#2ca02c',
               hovertemplate='Return: %{y:.2%}<extra></extra>'),
        secondary_y=False,
    )

    # 2. Add Volatility Bar
    fig.add_trace(
        go.Bar(x=names, y=volatilities, name='Annualized Volatility', marker_color='#d62728',
               hovertemplate='Volatility: %{y:.2%}<extra></extra>'),
        secondary_y=False,
    )

    # 3. Add Sharpe Ratio Line (Secondary Axis)
    fig.add_trace(
        go.Scatter(x=names, y=sharpes, name='Sharpe Ratio', mode='lines+markers',
                   line=dict(color='#1f77b4', width=3),
                   marker=dict(size=10, symbol='diamond'),
                   hovertemplate='Sharpe: %{y:.2f}<extra></extra>'),
        secondary_y=True,
    )

    # Styling
    fig.update_layout(
        title="<b>Portfolio Performance: Risk, Return, and Efficiency</b>",
        barmode='group',
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(tickangle=45)
    )

    # Set y-axis titles
    fig.update_yaxes(title_text="Annualized %", secondary_y=False, tickformat=".0%")
    fig.update_yaxes(title_text="Sharpe Ratio (Score)", secondary_y=True)

    return fig

def plot_backtest_results(backtest_results):
    fig = go.Figure()
    
    metrics = {}
    # Define your standard color map for consistency
    for name, data in backtest_results.items():
        # data[0] is the Equity Series (the first element in your tuple)
        equity_series = data[0]
        curr_metrics = extract_portfolio_metrics(equity_series)
        metrics[name] = curr_metrics

        fig.add_trace(go.Scatter(
            x=equity_series.index,
            y=equity_series.values,
            mode='lines',
            name=name,
            hovertemplate=f"<b>{name}</b><br>Value: $%{{y:,.2f}}<extra></extra>"
        ))

    fig.update_layout(
        title=f"<b>Portfolio Equity Growth ({equity_series.index[0].year} - {equity_series.index[-1].year})</b>",
        xaxis_title="Timeline",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Optional: Log scale for long-term growth comparison
    # fig.update_yaxes(type="log") 

    return fig, metrics

def plot_turnover_analysis(weights_df, ticker_color_map, model_name="Strategy"):
    """
    Creates a professional stacked area chart using plotly.graph_objects.
    """
    fig = go.Figure()

    # We iterate through tickers so each one is its own 'layer' in the stack
    # Standardizing the order (alphabetical) ensures consistency across tabs
    for ticker in sorted(weights_df.columns):
        fig.add_trace(go.Scatter(
            x=weights_df.index,
            y=weights_df[ticker],
            name=ticker,
            mode='lines',
            line=dict(width=0.5, color='white'), # Thin border between layers
            stackgroup='one', # This is what makes it a stacked area chart
            fillcolor=ticker_color_map.get(ticker, '#7f7f7f'),
            hovertemplate=f"<b>{ticker}</b>: %{{y:.2%}}<extra></extra>"
        ))

    fig.update_layout(
        title=f"<b>{model_name}: Historical Capital Allocation</b>",
        xaxis_title="Rebalance Timeline",
        yaxis_title="Portfolio Weight (%)",
        yaxis_tickformat='.0%',
        template="plotly_white",
        height = 600,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=40, l=60, r=40),
        # Ensure the y-axis is locked to 100%
        yaxis=dict(range=[0, 1], fixedrange=True)
    )

    return fig

def plot_strategy_performance(equity_series, model_name="Strategy"):
    # 1. Calculate Drawdown
    rolling_max = equity_series.cummax()
    
    fig = go.Figure()

    # 2. Add Equity Curve (Primary Y-Axis)
    fig.add_trace(go.Scatter(
        x=equity_series.index, 
        y=equity_series.values,
        name="Portfolio Value",
        fill='tozeroy',
        fillcolor='rgba(44, 160, 44, 0.1)',
        yaxis="y1"
    ))

    # 4. Mirror Layout Styling
    fig.update_layout(
        title=f"<b>{model_name}: Equity Curve</b>",
        template="plotly_white",
        hovermode="x unified",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        
        # Primary Axis (Value)
        yaxis=dict(
            title="Portfolio Value ($)",
            side="left",
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.2)'
        ),
        margin=dict(t=100, b=40, l=60, r=60)
    )

    return fig

def run_spy_benchmark(price_data, start_idx=252):
    """
    Simulates a $100 investment in SPY starting from the 
    first rebalance date of the other models.
    """
    # 1. Isolate SPY returns
    spy_returns = price_data['SPY'].pct_change().dropna()
    
    # 2. Align the start date with your other backtests (after the 1yr lookback)
    spy_test_returns = spy_returns.iloc[start_idx:]
    
    # 3. Calculate the cumulative growth of $100
    spy_equity_curve = 100 * (1 + spy_test_returns).cumprod()
    
    return spy_equity_curve

if __name__ == "__main__":
    basket = ["SPY", "TLT", "GLD", "IVOL"]
    weights = get_market_caps(basket)
    print(weights)
