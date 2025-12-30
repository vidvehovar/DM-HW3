from __future__ import annotations
import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from pathlib import Path

# Nastavitve poti
DATA_DIR = Path("data")

# 1. Funkcije za nalaganje podatkov
@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    p = DATA_DIR / name
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    return df

@st.cache_resource
def get_sentiment_pipe():
    # Model DistilBERT za analizo sentimenta [cite: 29]
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def month_options_2023():
    return pd.period_range("2023-01", "2023-12", freq="M")

# 2. Glavna aplikacija
def main():
    st.set_page_config(page_title="HW3 Brand Reputation Monitor (2023)", layout="wide")
    st.title("Brand Reputation Monitor — 2023")

    # Sidebar Navigacija [cite: 21]
    section = st.sidebar.radio("Navigacija", ["Izdelki", "Pričevanja", "Mnenja (Reviews)"])

    if section == "Izdelki":
        df = load_csv("products.csv")
        st.subheader("Seznam izdelkov")
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("Podatki o izdelkih niso na voljo. Najprej zaženite scrape.py.")

    elif section == "Pričevanja":
        df = load_csv("testimonials.csv")
        st.subheader("Pričevanja strank")
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("Podatki o pričevanjih niso na voljo.")

    else:
        df = load_csv("reviews.csv")
        st.subheader("Analiza sentimenta mnenj (2023)")

        if df.empty:
            st.error("Datoteka reviews.csv ne obstaja ali je prazna!")
            return

        # Priprava datumov [cite: 14]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        
        # Izbira meseca z drsnikom [cite: 25]
        months = month_options_2023()
        month_labels = [m.strftime("%b %Y") for m in months]
        selected_label = st.select_slider("Izberite mesec v letu 2023:", options=month_labels)

        # Filtriranje [cite: 26]
        selected_period = pd.Period(selected_label, freq="M")
        month_df = df[df["date"].dt.to_period("M") == selected_period].copy()

        st.write(f"Število mnenj za **{selected_label}**: {len(month_df)}")

        if month_df.empty:
            st.info("V tem mesecu ni mnenj. Izberite drug mesec.")
            return

        # Sentiment analiza s Transformers [cite: 27, 29]
        pipe = get_sentiment_pipe()
        with st.spinner("Analiziram sentiment..."):
            texts = month_df["text"].fillna("").astype(str).tolist()
            preds = pipe(texts)

        month_df["sentiment"] = [p["label"] for p in preds]
        month_df["confidence"] = [float(p["score"]) for p in preds]

        # Prikaz tabele [cite: 23, 31]
        st.dataframe(month_df[["date", "text", "sentiment", "confidence"]], use_container_width=True)

        # Vizualizacija [cite: 33, 35]
        st.markdown("### Statistika sentimenta")
        summary = month_df.groupby("sentiment").agg(
            count=("sentiment", "size"),
            avg_conf=("confidence", "mean")
        ).reset_index()

        fig, ax = plt.subplots()
        bars = ax.bar(summary["sentiment"], summary["count"], color=['#ff9999','#66b3ff'])
        ax.set_ylabel("Število mnenj")
        
        # Dodajanje confidence score v graf [cite: 35]
        for i, row in summary.iterrows():
            ax.text(i, row["count"], f"Zaupanje: {row['avg_conf']:.2%}", ha='center', va='bottom')
        
        st.pyplot(fig)

if __name__ == "__main__":
    main()
    