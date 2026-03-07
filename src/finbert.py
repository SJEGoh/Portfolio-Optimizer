from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import streamlit as st
from datetime import datetime, date, timedelta, timezone
from massive import RESTClient
import pandas as pd
import numpy as np

client = RESTClient(st.secrets["POLYGON_API_KEY"])
test_sentences = [
    "S&P 500 futures climb as easing inflation data suggests a pause in rate hikes.",
    "Equities retreat as underwhelming tech earnings spark fears of a broader valuation reset.",
    "Markets remain flat despite a surprise beat in payroll data, as investors weigh the potential for a hawkish Fed response.",
    "The regional bank's net interest margin exceeded expectations, but rising loan-loss provisions suggest a darkening credit outlook.",
    "Gold prices surge to record highs as geopolitical tensions in the Middle East drive a flight to safety.",
    "GLD sees significant outflows as a stabilizing dollar reduces the appeal of non-yielding assets.",
    "Bullion holds steady even as real yields rise, indicating strong underlying physical demand.",
    "Treasury yields spike to 15-year highs following a lackluster 30-year bond auction.",
    "The VIX jumps 20% as market participants rush to buy protection against a potential systemic credit event.",
    "Yield curve inversion deepens, further signaling an impending recessionary environment.",
    "The FOMC maintains a restrictive stance, signaling that 'higher for longer' remains the primary policy path.",
    "Central bank officials hint at a pivot, noting that the balance of risks is shifting toward economic support.",
    "The latest CPI print came in hotter than expected, diminishing hopes for an early summer rate cut.",
    "The semiconductor sector outperformed today following reports of a major supply chain breakthrough.",
    "Retail sales fell short of analyst estimates as consumer spending shows signs of fatigue amid high borrowing costs.",
    "The company's restructuring plan aims to improve margins, though execution risks remain elevated."
]

@st.cache_data(ttl=3600)
def fetch_benzinga_news(tickers, period = 20):
    news_data = {}
    # Calculate the timestamp for 20 days ago
    # We use ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ) which Polygon expects
    start_date = (datetime.now(timezone.utc) - timedelta(days=period)).strftime('%Y-%m-%dT%H:%M:%SZ')
    
    for ticker in tickers:
        try:
            # We use published_utc_gte to filter for news 'greater than or equal to' our date
            # We set a high limit (e.g., 1000) to ensure we get everything in that 20-day window
            news_generator = client.list_ticker_news(
                ticker, 
                published_utc_gte=start_date, 
                order="desc", 
                limit=1000 
            )
            
            articles = []
            for art in news_generator:
                articles.append({
                    "title": art.title,
                    "published_at": pd.to_datetime(art.published_utc),
                    "source": getattr(art, 'source', 'Unknown')
                })
            news_data[ticker] = articles
            
        except Exception as e:
            st.error(f"News fetch error for {ticker}: {e}")
            news_data[ticker] = []
            
    return news_data

@st.cache_resource
def load_finbert():
    finbert = BertForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels = 3)
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")

    nlp = pipeline("sentiment-analysis", model = finbert, tokenizer = tokenizer)
    return nlp

@st.cache_resource
def load_fls_finbert():
    finbert = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-fls", num_labels = 3)
    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-fls")

    nlp = pipeline("sentiment-analysis", model = finbert, tokenizer = tokenizer)
    return nlp


def get_sentiments(nlp, tickers, period = 20):
    sentiments = {}
    news = fetch_benzinga_news(tickers, period)
    for k, v in news.items():
        if not v:
            sentiments[k] = []
            continue
        titles = [t["title"] for t in v]
        sentiments[k] = nlp(titles)
    return sentiments

def get_bl_parameters(tickers, news_data, tone_model, fls_model):
    views = {}
    confs = {}

    for t in tickers:
        articles = news_data.get(t, [])
        if not articles:
            views[t], confs[t] = 0, 0.01 # Default to Neutral
            continue
        
        titles = [a['title'] for a in articles]
        tones = tone_model(titles)
        fls_labels = fls_model(titles)

        # 1. Calculate Average Sentiment (Returns)
        sentiment_scores = []
        for r in tones:
            val = 1 if r['label'] == 'positive' else -1 if r['label'] == 'negative' else 0
            sentiment_scores.append(val * r['score'])
        
        # Map avg sentiment (-1 to 1) to a return view (-5% to +5%)
        views[t] = np.mean(sentiment_scores) * 0.05

        # 2. Calculate Confidence (Quality Filter)
        fls_weights = []
        for f in fls_labels:
            if f['label'] == 'Specific FLS': weight = 0.9
            elif f['label'] == 'Non-specific FLS': weight = 0.5
            else: weight = 0.1 # Not FLS
            fls_weights.append(weight)
        
        # Final Confidence = Average FLS weight scaled by model certainty
        confs[t] = np.mean(fls_weights)
        
    return views, confs

def main():
    tickers = ["AAPL", "STX", "ASML","NVDA"]
    nlp = load_finbert()
    fls = load_fls_finbert()
    news_data = fetch_benzinga_news(tickers)
    bl_params = get_bl_parameters(tickers, news_data, nlp, fls)
    print(bl_params)

if __name__ == "__main__":
    main()
