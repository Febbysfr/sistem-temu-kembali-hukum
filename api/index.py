from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__, template_folder='../templates', static_folder='../static')

def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'detik_hukum_50_artikel.csv')
    df = pd.read_csv(csv_path, sep=',')
    df = df.dropna(subset=['isi'])
    return df

def build_tfidf(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['isi'])
    return vectorizer, tfidf_matrix

df = load_data()
vectorizer, tfidf_matrix = build_tfidf(df)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'results': [], 'total': 0})
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = scores.argsort()[::-1][:10]
    hasil = [(int(idx), float(scores[idx])) for idx in top_idx if scores[idx] > 0]
    results = []
    for idx, score in hasil:
        row = df.iloc[idx]
        results.append({
            'judul': str(row['judul']),
            'tanggal': str(row['tanggal']),
            'isi': str(row['isi'])[:400] + '...',
            'url': str(row['url']),
            'skor': round(score, 4)
        })
    return jsonify({'results': results, 'total': len(results)})

if __name__ == '__main__':
    app.run(debug=True)