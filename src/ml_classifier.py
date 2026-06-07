"""
ml_classifier.py
Charge le modèle ML sauvegardé et expose une fonction predict().
Le modèle est entraîné dans notebooks/02_model_training.ipynb.
"""

import os
import numpy as np
from src.analysis_utils import preprocess_pipeline

# Chemins vers les artefacts sauvegardés par le notebook 02
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
ENCODER_PATH    = os.path.join(MODELS_DIR, "label_encoder.joblib")
MODEL_PATH      = os.path.join(MODELS_DIR, "sentiment_classifier.joblib")

# ---------------------------------------------------------------------------
# Chargement paresseux (une seule fois par session)
# ---------------------------------------------------------------------------

_vectorizer = None
_encoder    = None
_model      = None


def _load_artifacts():
    """Charge les artefacts ML si ce n'est pas déjà fait."""
    global _vectorizer, _encoder, _model

    if _model is not None:
        return  # déjà chargés

    try:
        import joblib
    except ImportError:
        raise RuntimeError("Package 'joblib' manquant. Lancez : pip install joblib")

    missing = [
        p for p in [VECTORIZER_PATH, ENCODER_PATH, MODEL_PATH]
        if not os.path.exists(p)
    ]
    if missing:
        raise FileNotFoundError(
            f"Modèles introuvables : {missing}\n"
            "Lancez d'abord le notebook 02_model_training.ipynb pour entraîner et sauvegarder les modèles."
        )

    _vectorizer = joblib.load(VECTORIZER_PATH)
    _encoder    = joblib.load(ENCODER_PATH)
    _model      = joblib.load(MODEL_PATH)


def is_model_available() -> bool:
    """Retourne True si les fichiers modèles existent sur le disque."""
    return all(
        os.path.exists(p)
        for p in [VECTORIZER_PATH, ENCODER_PATH, MODEL_PATH]
    )


# ---------------------------------------------------------------------------
# Prédiction
# ---------------------------------------------------------------------------

def predict(processed_text: str) -> dict:
    """
    Prédit le sentiment d'un texte déjà prétraité (processed_avis).

    Args:
        processed_text: Texte nettoyé et lemmatisé (sortie du pipeline notebook 01).

    Returns:
        {
            "sentiment":     str,           # classe prédite
            "confiance":     str,           # "élevée" | "moyenne" | "faible"
            "score":         float,         # probabilité max (0–1)
            "probabilites":  dict[str,float] # score par classe
        }
    """
    _load_artifacts()

    X = _vectorizer.transform([processed_text])

    predicted_index = _model.predict(X)[0]
    sentiment = _encoder.inverse_transform([predicted_index])[0]

    # Probabilités par classe (si le modèle les supporte)
    if hasattr(_model, "predict_proba"):
        proba_array = _model.predict_proba(X)[0]
        classes = _encoder.inverse_transform(np.arange(len(proba_array)))
        probas = {cls: round(float(p), 4) for cls, p in zip(classes, proba_array)}
        score = float(max(proba_array))
    else:
        # LinearSVC : utilise decision_function comme proxy
        decision = _model.decision_function(X)[0]
        if decision.ndim == 0:
            score = float(abs(decision))
        else:
            score = float(max(decision))
        classes = _encoder.inverse_transform(np.arange(len(decision)))
        probas = {cls: round(float(v), 4) for cls, v in zip(classes, decision)}

    # Seuils de confiance
    if score >= 0.70:
        confiance = "élevée"
    elif score >= 0.45:
        confiance = "moyenne"
    else:
        confiance = "faible"

    return {
        "sentiment":    sentiment,
        "confiance":    confiance,
        "score":        round(score, 4),
        "probabilites": probas,
    }


def predict_raw_text(raw_text: str) -> dict:
    """
    Variante qui accepte un avis brut et applique le prétraitement minimal.
    Utile dans l'app quand on n'a pas déjà le processed_avis.

    Args:
        raw_text: Texte brut de l'avis client.

    Returns:
        Même structure que predict().
    """
    processed = preprocess_pipeline(raw_text)
    return predict(processed)