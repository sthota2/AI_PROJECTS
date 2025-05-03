import streamlit as st
import pandas as pd
from utils import load_csv_data, create_embeddings, build_faiss_index
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt


# --- Global Config ---
CURRENCY_SYMBOL = "$"

# --- Functions ---

def detect_special_query(query):
    query = query.lower()
    if "highest" in query or "most expensive" in query:
        return "highest"
    if "lowest" in query or "cheapest" in query:
        return "lowest"
    if "average" in query:
        return "average"
    return None

def handle_special_query(special_case, df):
    if special_case == "highest":
        highest_row = df.loc[df['Amount'].idxmax()]
        return f"🤑 Your highest transaction was {CURRENCY_SYMBOL}{highest_row['Amount']} on {highest_row['Date']} for {highest_row['Category']}"
    elif special_case == "lowest":
        lowest_row = df.loc[df['Amount'].idxmin()]
        return f"💸 Your lowest transaction was {CURRENCY_SYMBOL}{lowest_row['Amount']} on {lowest_row['Date']} for {lowest_row['Category']}"
    elif special_case == "average":
        avg_spent = df['Amount'].mean()
        return f"📊 Your average spending is {CURRENCY_SYMBOL}{avg_spent:.2f}"
    return None

def plot_category_spending(df):
    """Pie chart of spending per category"""
    category_sums = df.groupby('Category')['Amount'].sum()
    fig, ax = plt.subplots()
    ax.pie(category_sums, labels=category_sums.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

def plot_monthly_spending(df):
    """Line chart of monthly spending"""
    df['Month'] = pd.to_datetime(df['Date']).dt.to_period('M')
    monthly_sums = df.groupby('Month')['Amount'].sum()
    fig, ax = plt.subplots()
    monthly_sums.plot(kind='line', marker='o', ax=ax)
    ax.set_title('Monthly Spending Trend')
    ax.set_ylabel(f'Spending ({CURRENCY_SYMBOL})')
    ax.set_xlabel('Month')
    st.pyplot(fig)

# --- App Starts Here ---

st.set_page_config(page_title="💸 Deepni Finance Chat", layout="wide")
st.title("💸 Deepni Personal Finance Dashboard & Chat")

# Load data
df = load_csv_data("/app/data/transactions.csv")

with st.spinner('Loading Model & Building Index... ⏳'):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    if os.path.exists("data/embeddings.pkl") and os.path.exists("data/texts.pkl"):
        with open("data/embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
        with open("data/texts.pkl", "rb") as f:
            texts = pickle.load(f)
    else:
        embeddings, texts = create_embeddings(df, model)
        with open("data/embeddings.pkl", "wb") as f:
            pickle.dump(embeddings, f)
        with open("data/texts.pkl", "wb") as f:
            pickle.dump(texts, f)

    index = build_faiss_index(embeddings)

st.success('Model and index loaded! 🚀')

# --- Tabs: Chat | Dashboard ---
tab1, tab2 = st.tabs(["💬 Chat", "📊 Dashboard"])

with tab1:
    query = st.text_input("Ask about your spending:")
    if query:
        special_case = detect_special_query(query)

        if special_case:
            response = handle_special_query(special_case, df)
            if response:
                st.success(response)
            else:
                st.info("🤔 No matching result found.")
        else:
            query_vec = model.encode([query])
            D, I = index.search(query_vec, k=3)
            st.write("Relevant Transactions:")
            for i in I[0]:
                st.write(texts[i])

with tab2:
    st.subheader("📊 Spending by Category")
    plot_category_spending(df)

    st.subheader("📈 Monthly Spending Trend")
    plot_monthly_spending(df)

    st.subheader("📄 Download Full Transactions Data")
    csv_data = df.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv_data, file_name="transactions_summary.csv", mime="text/csv")

