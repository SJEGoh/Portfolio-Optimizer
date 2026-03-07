import streamlit as st
from massive import RESTClient
import pandas as pd
from datetime import datetime, date
import numpy as np

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

