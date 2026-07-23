# =========================================================
# NLP TEXT PREPROCESSING MODULE
# =========================================================
# Implements the full preprocessing pipeline matching the
# model specification: lowercase → punctuation removal →
# numeric removal → stopwords removal → stemming.
# =========================================================

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =========================================================
# NLTK RESOURCE DOWNLOAD (one-time, cached)
# =========================================================

nltk.download('stopwords', quiet=True)

# =========================================================
# INITIALIZE NLP COMPONENTS
# =========================================================

_stop_words = set(stopwords.words('english'))
_lemmatizer = WordNetLemmatizer()


# =========================================================
# MAIN PREPROCESSING FUNCTION
# =========================================================

def preprocess_text(text: str) -> str:
    """
    Apply the full NLP preprocessing pipeline to input text.
    
    Pipeline:
        1. Lowercase conversion
        2. Punctuation removal
        3. Numeric removal
        4. Whitespace normalization
        5. Stopwords removal (NLTK English)
        6. Lemmatization (WordNetLemmatizer)
    
    Args:
        text: Raw input text string.
    
    Returns:
        Fully preprocessed text string.
    """
    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Step 3: Remove numbers
    text = re.sub(r'\d+', '', text)

    # Step 4: Normalize whitespace
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    # Step 5: Tokenize
    tokens = text.split()

    # Step 6: Remove stopwords
    tokens = [word for word in tokens if word not in _stop_words]

    # Step 7: Stemming
    tokens = [_lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)


# =========================================================
# DETAILED PREPROCESSING STEPS (for NLP Viewer)
# =========================================================

def get_preprocessing_steps(text: str) -> dict:
    """
    Apply preprocessing step-by-step and return intermediate
    results at each stage. Used to display the NLP Processing
    Viewer so users can see what happens at each step.
    
    Args:
        text: Raw input text string.
    
    Returns:
        Dictionary with keys for each preprocessing stage:
        - raw_text
        - after_lowercase
        - after_punctuation_removal
        - after_numeric_removal
        - after_whitespace_normalization
        - tokens_before_stopwords (list)
        - tokens_after_stopwords (list)
        - stopwords_removed (list)
        - tokens_after_stemming (list)
        - final_text
        - token_count
        - unique_token_count
    """
    steps = {}

    # Raw text
    steps['raw_text'] = text

    # Step 1: Lowercase
    text_lower = text.lower()
    steps['after_lowercase'] = text_lower

    # Step 2: Remove punctuation
    text_no_punct = text_lower.translate(
        str.maketrans('', '', string.punctuation)
    )
    steps['after_punctuation_removal'] = text_no_punct

    # Step 3: Remove numbers
    text_no_nums = re.sub(r'\d+', '', text_no_punct)
    steps['after_numeric_removal'] = text_no_nums

    # Step 4: Normalize whitespace
    text_clean = text_no_nums.strip()
    text_clean = re.sub(r'\s+', ' ', text_clean)
    steps['after_whitespace_normalization'] = text_clean

    # Step 5: Tokenize
    tokens = text_clean.split()
    steps['tokens_before_stopwords'] = tokens.copy()

    # Step 6: Remove stopwords
    tokens_filtered = [w for w in tokens if w not in _stop_words]
    removed_stopwords = [w for w in tokens if w in _stop_words]
    steps['tokens_after_stopwords'] = tokens_filtered.copy()
    steps['stopwords_removed'] = removed_stopwords

    # Step 7: Stemming
    tokens_stemmed = [_lemmatizer.lemmatize(w) for w in tokens_filtered]
    steps['tokens_after_stemming'] = tokens_stemmed

    # Final output
    steps['final_text'] = ' '.join(tokens_stemmed)
    steps['token_count'] = len(tokens_stemmed)
    steps['unique_token_count'] = len(set(tokens_stemmed))

    return steps
