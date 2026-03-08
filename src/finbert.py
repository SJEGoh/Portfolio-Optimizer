from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import streamlit as st
from datetime import datetime, date, timedelta, timezone
from massive import RESTClient
import pandas as pd
import numpy as np

client = RESTClient(st.secrets["POLYGON_API_KEY"])
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

def get_bl_parameters(tickers, news_data, tone_model, fls_model):
    views = {}
    confs = {}

    for t in tickers:
        articles = news_data.get(t, [])
        if not articles:
            views[t], confs[t] = 0.0, 0.01 
            continue
        
        titles = [a['title'] for a in articles]
        
        # Run models separately to avoid 'Zip' crashes
        try:
            tones = tone_model(titles)
            fls = fls_model(titles)
        except Exception as e:
            # Fallback: If FLS fails, just use Tones with neutral weights
            print(f"Model error for {t}: {e}")
            continue

        weighted_scores = []
        # Use range to ensure we stay within bounds
        for i in range(len(tones)):
            # TONE LOGIC
            t_res = tones[i]
            direction = 1 if t_res['label'] == 'positive' else -1 if t_res['label'] == 'negative' else 0
            score = direction * t_res['score']
            
            # FLS WEIGHTING (The extra logic)
            # We add a 'get' or check to make sure the FLS result exists for this index
            f_label = fls[i]['label'] if i < len(fls) else 'Not FLS'
            
            if f_label == 'Specific FLS': weight = 1.5
            elif f_label == 'Non-specific FLS': weight = 1.0
            else: weight = 0.5 
                
            weighted_scores.append(score * weight)

        # 1. Update Mu (Q)
        views[t] = np.mean(weighted_scores) * 0.2
        
        # 2. Update Confidence (The Consensus Math)
        std_dev = np.std(weighted_scores)
        agreement = 1 / (1 + std_dev)
        volume = min(np.log10(len(weighted_scores) + 1) / np.log10(31), 1.0)
        
        confs[t] = agreement * volume
        
    return views, confs

def ai_bl_params(tickers, period = 20):
    nlp = load_finbert()
    fls = load_fls_finbert()
    news_data = fetch_benzinga_news(tickers, period)
    bl_params = get_bl_parameters(tickers, news_data, nlp, fls)
    return bl_params

def main():
    tickers = ["AAPL", "STX", "ASML","NVDA"]
    nlp = load_finbert()
    fls = load_fls_finbert()
    news_data = fetch_benzinga_news(tickers)
    bl_params = get_bl_parameters(tickers, news_data, nlp, fls)
    print(bl_params)

if __name__ == "__main__":
    main()
