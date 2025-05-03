import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import os

def load_csv_data(file_path):
    return pd.read_csv(file_path)

def create_embeddings(df, model):
    texts = df.apply(lambda row: f"{row['Date']} {row['Description']} {row['Original Description']} {row['Amount']} {row['Transaction Type']} {row['Category']} {row['Account Name']} {row['Labels']} {row['Notes']}", axis=1)
    embeddings = model.encode(texts.tolist(), show_progress_bar=True)
    return embeddings, texts.tolist()

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index
