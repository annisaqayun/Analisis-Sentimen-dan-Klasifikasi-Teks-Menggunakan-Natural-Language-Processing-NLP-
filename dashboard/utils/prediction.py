# =========================================================
# MODEL LOADING & PREDICTION MODULE (REFACTORED FOR NAIVE BAYES)
# =========================================================

import json
import numpy as np
import streamlit as st
import joblib
import os

# =========================================================
# PATH CONFIGURATION
# =========================================================

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Diperbarui untuk membaca pipeline tunggal (.pkl)
_MODEL_PATH = os.path.join(_BASE_DIR, 'best_model_pipeline.pkl')
_SPEC_PATH = os.path.join(_BASE_DIR, 'model_specification.json')


# =========================================================
# MODEL LOADING (cached)
# =========================================================

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load the pre-trained GridSearchCV Pipeline dari .pkl.
    Mengembalikan best_estimator_ yang berisi vectorizer dan model NB.
    """
    try:
        grid = joblib.load(_MODEL_PATH)
        return grid.best_estimator_
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_vectorizer():
    """
    Mock fungsi untuk kompabilitas mundur dengan app.py lama.
    Mengembalikan objek vectorizer dari dalam pipeline.
    """
    pipeline = load_model()
    if pipeline:
        return pipeline.named_steps['vectorizer']
    return None

@st.cache_data(show_spinner=False)
def load_model_spec() -> dict:
    with open(_SPEC_PATH, 'r') as f:
        return json.load(f)


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_quality(processed_text: str, model, vectorizer=None) -> dict:
    """
    Predict wine quality from preprocessed text menggunakan pipeline utuh.
    """
    # model di sini adalah pipeline utuh
    prediction = model.predict([processed_text])[0]
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([processed_text])[0]
        classes = model.classes_.tolist()
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, proba)}
        confidence = float(np.max(proba))
        proba_array = proba.tolist()
    else:
        # Fallback jika model tidak punya probabilitas
        prob_dict = {prediction: 1.0}
        classes = [prediction]
        confidence = 1.0
        proba_array = [1.0]

    return {
        'prediction': prediction,
        'confidence': confidence,
        'probabilities': prob_dict,
        'classes': classes,
        'probabilities_array': proba_array
    }


# =========================================================
# FEATURE IMPORTANCE EXTRACTION
# =========================================================

def get_feature_importance(model, vectorizer=None, top_n: int = 20) -> dict:
    """
    Extract the top N most important features (words) dari Naive Bayes
    menggunakan variansi log probability antar kelas.
    """
    try:
        nb_model = model.named_steps['model']
        vec = model.named_steps['vectorizer']
        
        feature_names = vec.get_feature_names_out()
        
        # Ekstrak feature_log_prob_ (n_classes, n_features)
        if hasattr(nb_model, 'feature_log_prob_'):
            log_probs = nb_model.feature_log_prob_
            # Menghitung seberapa diskriminatif sebuah kata (selisih max-min antar kelas)
            importance = np.ptp(log_probs, axis=0)
            
            top_indices = np.argsort(importance)[-top_n:][::-1]
            
            top_features = [feature_names[idx] for idx in top_indices]
            top_importances = [float(importance[idx]) for idx in top_indices]
            
            # Normalisasi agar terlihat bagus di chart (0 sampai 1)
            if len(top_importances) > 0:
                max_imp = max(top_importances)
                top_importances = [imp / max_imp for imp in top_importances]
                
            return {
                'features': top_features,
                'importances': top_importances
            }
        else:
            return {'features': [], 'importances': []}
    except Exception as e:
        print("Feature importance error:", e)
        return {'features': [], 'importances': []}
