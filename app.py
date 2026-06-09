import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    df = pd.read_csv("detik_hukum_50_artikel.csv", sep=",")
    df = df.dropna(subset=['isi'])
    return df

@st.cache_resource
def build_tfidf(_df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(_df['isi'])
    return vectorizer, tfidf_matrix

st.set_page_config(page_title="Sistem Temu Kembali Berita Hukum", page_icon="⚖️")
st.title("⚖️ Sistem Temu Kembali Berita Hukum")
st.caption("Pencarian artikel berita hukum dari Detik.com menggunakan TF-IDF & Cosine Similarity")

df = load_data()
vectorizer, tfidf_matrix = build_tfidf(df)

query = st.text_input("🔍 Masukkan kata kunci:", placeholder="contoh: korupsi bupati KPK")

if query:
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = scores.argsort()[::-1][:10]
    hasil = [(idx, scores[idx]) for idx in top_idx if scores[idx] > 0]
    st.markdown(f"### 📄 Ditemukan {len(hasil)} artikel relevan")
    if not hasil:
        st.warning("Tidak ada artikel yang cocok dengan kata kunci tersebut.")
    else:
        for rank, (idx, score) in enumerate(hasil, 1):
            with st.expander(f"#{rank} — {df.iloc[idx]['judul']} | Skor: {score:.4f}"):
                st.write(f"📅 {df.iloc[idx]['tanggal']}")
                st.write(df.iloc[idx]['isi'][:400] + "...")
                st.markdown(f"[🔗 Baca selengkapnya]({df.iloc[idx]['url']})")