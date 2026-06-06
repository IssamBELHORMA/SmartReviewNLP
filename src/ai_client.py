import json
import os
import re

# Appel API réel — Anthropic / Claude

def analyze_with_api(review_text: str) -> dict:
    """
    Envoie l'avis à Claude via l'API Anthropic et retourne le JSON d'analyse.

    Nécessite :
        pip install anthropic
        Variable d'environnement ANTHROPIC_API_KEY (ou fichier .env)

    Args:
        review_text: Texte de l'avis client.

    Returns:
        Dictionnaire structuré avec sentiment, confiance, etc.

    Raises:
        RuntimeError si l'API n'est pas disponible.
    """
    try:
        import anthropic
        from src.prompt_builder import build_prompt, build_system_prompt
    except ImportError:
        raise RuntimeError("Package 'anthropic' manquant. Lancez : pip install anthropic")

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY non définie. Ajoutez-la dans votre fichier .env ou "
            "vos variables d'environnement."
        )

    client = anthropic.Anthropic(api_key=api_key)

    user_prompt = build_prompt(review_text)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=build_system_prompt(),
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = message.content[0].text.strip()

    # Extraction robuste du JSON (retire éventuels blocs markdown)
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"Réponse API non parseable : {raw}")

    return json.loads(json_match.group())


# Simulation pédagogique (fallback sans clé API)

_POSITIVE_WORDS = [
    "rapide", "professionnel", "agréable", "clair", "bonne qualité",
    "excellent", "satisfait", "super", "bravo", "parfait", "top",
    "efficace", "impressionné", "ravi", "content", "recommande",
]

_NEGATIVE_WORDS = [
    "déçu", "cassée", "panne", "refusé", "retard", "trop longue",
    "problème", "impossible", "nul", "mauvais", "perdu", "abîmé",
    "manquant", "erreur", "aucune réponse", "attente", "jamais",
]

_CATEGORY_KEYWORDS = {
    "livraison": ["livraison", "colis", "transport", "livreur", "expédié"],
    "paiement": ["paiement", "carte", "prélèvement", "facture", "remise"],
    "support": ["service client", "support", "conseiller", "opérateur", "chat"],
    "application": ["application", "appli", "site", "bug", "interface"],
    "remboursement": ["remboursement", "remboursé", "retour", "avoir"],
    "produit": ["produit", "article", "commande", "qualité"],
    "administratif": ["dossier", "formulaire", "compte", "email", "confirmation"],
}


def _detect_category(text: str) -> str:
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return cat
    return "general"


def analyze_with_simulated_ai(review_text: str) -> dict:
    """
    Simulation pédagogique par mots-clés (ne nécessite pas d'API).

    Args:
        review_text: Texte de l'avis client.

    Returns:
        Dictionnaire structuré.
    """
    text = review_text.lower()

    has_positive = any(word in text for word in _POSITIVE_WORDS)
    has_negative = any(word in text for word in _NEGATIVE_WORDS)

    if has_positive and has_negative:
        sentiment = "mitigé"
        confidence = "moyenne"
    elif has_positive:
        sentiment = "positif"
        confidence = "élevée"
    elif has_negative:
        sentiment = "négatif"
        confidence = "élevée"
    else:
        sentiment = "neutre"
        confidence = "moyenne"

    category = _detect_category(text)

    action_map = {
        "positif": "Archiver l'avis positif et remercier le client.",
        "négatif": "Transmettre au service concerné pour prise en charge rapide.",
        "neutre": "Aucune action requise — avis informatif.",
        "mitigé": "Relire l'avis et contacter le client pour clarification.",
    }

    return {
        "sentiment": sentiment,
        "confiance": confidence,
        "justification": "Analyse simulée à partir de mots-clés présents dans l'avis.",
        "points_positifs": ["Éléments positifs détectés"] if has_positive else [],
        "points_negatifs": ["Éléments négatifs détectés"] if has_negative else [],
        "categorie": category,
        "action_recommandee": action_map[sentiment],
    }


# Fonction principale : essaie l'API, bascule sur la simulation

def analyze_review(review_text: str, force_simulation: bool = False) -> tuple[dict, str]:
    """
    Analyse un avis client.

    Tente d'abord l'API Anthropic, bascule sur la simulation en cas d'échec.

    Args:
        review_text: Texte de l'avis.
        force_simulation: Si True, bypass l'API.

    Returns:
        (résultat_dict, source) où source vaut "api" ou "simulation".
    """
    if not force_simulation:
        try:
            result = analyze_with_api(review_text)
            return result, "api"
        except Exception:
            pass

    result = analyze_with_simulated_ai(review_text)
    return result, "simulation"