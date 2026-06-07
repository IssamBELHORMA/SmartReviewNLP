"""
analysis_utils.py
Utilitaires NLP : nettoyage, tokenisation, stop words, lemmatisation.
Réutilisés depuis le notebook 01_exploration_preprocessing.
"""

import re
import unicodedata

# Stop words français — liste prudente (évite de supprimer les négations)

FRENCH_STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "du", "au", "aux",
    "ce", "cet", "cette", "ces", "mon", "ton", "son", "ma", "ta",
    "sa", "notre", "votre", "leur", "mes", "tes", "ses", "nos",
    "vos", "leurs", "je", "tu", "il", "elle", "nous", "vous",
    "ils", "elles", "me", "te", "se", "lui", "y", "en",
    "que", "qui", "quoi", "dont", "où",
    "et", "ou", "mais", "donc", "or", "ni", "car",
    "de", "à", "par", "pour", "sur", "sous", "avec", "sans",
    "dans", "entre", "vers", "chez", "lors",
    "est", "sont", "était", "être", "avoir", "a", "ont",
    "très", "aussi", "plus", "moins", "bien", "tout",
    "comme", "si", "même",
}

# Dictionnaire de lemmatisation simple (français courant)

_LEMMA_MAP = {
    "arrivée": "arriver", "arrivé": "arriver",
    "reçu": "recevoir", "reçue": "recevoir",
    "cassée": "casser", "cassé": "casser",
    "déçu": "décevoir", "déçue": "décevoir",
    "livré": "livrer", "livrée": "livrer",
    "commandé": "commander", "commandée": "commander",
    "remboursé": "rembourser", "remboursée": "rembourser",
    "refusé": "refuser", "refusée": "refuser",
    "enregistrée": "enregistrer", "enregistré": "enregistrer",
    "satisfait": "satisfaire", "satisfaite": "satisfaire",
    "rapide": "rapide", "rapides": "rapide",
    "longue": "long", "long": "long", "longs": "long",
    "professionnels": "professionnel",
}


# Fonctions publiques

def normalize_text(text: str) -> str:
    """Retire les accents et met en minuscules."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def clean_text(text: str) -> str:
    """
    Nettoyage minimal : minuscules, suppression ponctuation, espaces multiples.
    Conserve les accents pour lisibilité.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)   # ponctuation → espace (garde apostrophe)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_tokenize(text: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ'\s-]", " ", text)
    tokens = text.split()
    return tokens


def tokenize_text(text: str) -> list[str]:
    """Alias explicite pour le notebook."""
    return simple_tokenize(text)


def remove_stop_words(tokens: list[str]) -> list[str]:
    """Retire les stop words de la liste de tokens."""
    return [t for t in tokens if t not in FRENCH_STOPWORDS and len(t) > 1]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Lemmatisation simple via dictionnaire de règles."""
    result = []
    for token in tokens:
        # Vérifie le dictionnaire, sinon retire suffixes courants
        if token in _LEMMA_MAP:
            result.append(_LEMMA_MAP[token])
        elif token.endswith("ées") or token.endswith("ée"):
            result.append(token[:-2] + "er")
        elif token.endswith("és") or token.endswith("é"):
            result.append(token[:-1] + "er")
        elif token.endswith("tion"):
            result.append(token)
        elif token.endswith("ment") and len(token) > 6:
            result.append(token[:-4])
        else:
            result.append(token)
    return result


def preprocess_pipeline(text: str) -> str:
    """Pipeline complet : clean → tokenize → stop words → lemmatize → rejoin."""
    tokens = tokenize_text(clean_text(text))
    tokens = remove_stop_words(tokens)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)


# Évaluation des résultats

def compute_accuracy(results: list[dict], labels: list[str]) -> dict:
    """
    Calcule accuracy et distribution des erreurs.

    Args:
        results: Liste de dicts contenant 'sentiment'.
        labels: Labels attendus correspondants.

    Returns:
        Dict avec accuracy, nb_correct, nb_total, errors.
    """
    correct = 0
    errors = []
    for i, (res, label) in enumerate(zip(results, labels)):
        predicted = res.get("sentiment", "")
        if predicted == label:
            correct += 1
        else:
            errors.append({"index": i, "attendu": label, "obtenu": predicted})

    return {
        "accuracy": correct / len(labels) if labels else 0,
        "nb_correct": correct,
        "nb_total": len(labels),
        "errors": errors,
    }


def format_result_for_display(result: dict) -> str:
    """Formate un résultat d'analyse pour affichage terminal."""
    lines = [
        f"  Sentiment    : {result.get('sentiment', '?')}",
        f"  Confiance    : {result.get('confiance', '?')}",
        f"  Catégorie    : {result.get('categorie', '?')}",
        f"  Justification: {result.get('justification', '?')}",
        f"  Points +     : {', '.join(result.get('points_positifs', [])) or '—'}",
        f"  Points -     : {', '.join(result.get('points_negatifs', [])) or '—'}",
        f"  Action       : {result.get('action_recommandee', '?')}",
    ]
    return "\n".join(lines)