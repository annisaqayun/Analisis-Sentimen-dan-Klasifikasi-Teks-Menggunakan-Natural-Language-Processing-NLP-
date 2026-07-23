# 🍷 AI Wine Quality Prediction Dashboard

Dashboard prediksi kualitas wine berbasis NLP menggunakan Machine Learning.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLTK Data (otomatis saat pertama kali dijalankan)

```bash
python -c "import nltk; nltk.download('stopwords')"
```

### 3. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## 📁 Struktur Project

```
NLP/
├── app.py                          # Aplikasi utama Streamlit
├── utils/
│   ├── __init__.py                 # Package init
│   ├── preprocessing.py            # NLP preprocessing pipeline
│   ├── prediction.py               # Model loading & prediksi
│   ├── visualization.py            # Plotly chart builders
│   └── helpers.py                  # Utility functions
├── assets/
│   └── custom.css                  # Custom CSS (glassmorphism)
├── decision_tree_model.joblib      # Pre-trained model
├── tfidf_vectorizer.joblib         # Pre-trained vectorizer
├── model_specification.json        # Spesifikasi model
├── requirements.txt                # Dependencies
└── README.md                       # Dokumentasi
```

## 🛠️ Teknologi

- **Python** — Bahasa pemrograman
- **Streamlit** — Web framework
- **scikit-learn** — Machine Learning (Decision Tree + TF-IDF)
- **NLTK** — Natural Language Processing
- **Plotly** — Interactive charts
- **fpdf2** — PDF report generation

## 📊 Fitur

1. **Prediksi Kualitas Wine** dari deskripsi teks
2. **Visualisasi Probabilitas** (Bar, Pie, Gauge chart)
3. **NLP Processing Viewer** — lihat setiap langkah preprocessing
4. **Feature Importance** — kata-kata paling berpengaruh
5. **Prediction History** — riwayat prediksi
6. **Export** — Download PDF report & CSV history
7. **Analytics Dashboard** — model metrics & confusion matrix

## 📝 Model Info

| Parameter | Value |
|-----------|-------|
| Algorithm | Decision Tree Classifier |
| Vectorizer | TF-IDF (max 5000 features) |
| Accuracy | ~61% |
| Classes | Low, Medium, High |
