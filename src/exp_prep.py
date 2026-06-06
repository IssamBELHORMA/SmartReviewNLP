import pandas as pd
import re




BASIC_STOP_WORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de",
    "ce", "cet", "cette", "ces",
    "et", "ou",
    "à", "au", "aux", "en", "dans", "sur", "pour", "par",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
    "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses",
    "est", "suis", "sont", "été", "être",
    "a", "ai", "as", "avons", "avez", "ont",
    "avec", "sansplus"
}

NEGATION_WORDS = {
    "ne", "n'", "pas", "jamais", "aucun", "aucune", "rien", "ni", "sans"
}




SIMPLE_LEMMAS = {
    # --- Vos lemmes initiaux ---
    "recommande": "recommander",
    "recommandé": "recommander",
    "recommander": "recommander",
    "satisfait": "satisfaire",
    "satisfaite": "satisfaire",
    "déçu": "décevoir",
    "déçue": "décevoir",
    "reçue": "recevoir",
    "reçu": "recevoir",
    "arrivé": "arriver",
    "arrivée": "arriver",
    "plante": "planter",
    "plantes": "planter",
    "répond": "répondre",
    "résolu": "résoudre",
    "utiliser": "utiliser",
    "utile": "utile",
    "rapide": "rapide",
    "lent": "lent",
    "lente": "lent",
    "planté": "planter",
    "plantent": "planter",
    "bug": "bug",
    "bugs": "bug",
    "panne": "panne",
    "pannes": "panne",
    "connecter": "connecter",
    "connecté": "connecter",
    "consomme": "consommer",
    "corrige": "corriger",
    "corrigé": "corriger",
    "ferme": "fermer",
    "refuse": "refuser",
    "refusé": "refuser",

    # --- Extensions E-commerce & Logistique ---
    "commande": "commande",
    "commandes": "commande",     
    "commandé": "commander",
    "commanderai": "commander", 
    "produit": "produit",
    "produits": "produit",      
    "article": "article",
    "articles": "article",
    "colis": "colis",
    "livraison": "livraison",
    "livraisons": "livraison",
    "livré": "livrer",
    "livrée": "livrer",
    "payé": "payer",
    "payer": "payer",
    "remboursement": "remboursement",
    "remboursements": "remboursement",

    # --- Évaluations, Sentiments & Support ---
    "déçus": "décevoir",
    "déçoivent": "décevoir",
    "cassé": "casser",
    "cassée": "casser",
    "excellent": "excellent",
    "excellente": "excellent",
    "irréprochable": "irréprochable",
    "correct": "correct",
    "correcte": "correct",
    "fiable": "fiable",
    "agressif": "agressif",
    "condescendant": "condescendant",
    "étonné": "étonner",
    "étonnée": "étonner"
}

















def simple_tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ'\s-]", " ", text)
    tokens = text.split()
    return tokens


def clean_text(text):
    text = str(text)

    # Passage en minuscules
    text = text.lower()

    # Suppression des URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remplacement des apostrophes typographiques
    text = text.replace("’", "'")

    # Suppression de certains caractères spéciaux
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ0-9'\s-]", " ", text)

    # Normalisation des espaces multiples
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize_text(text):
    return str(text).split()


def remove_stop_words(tokens):
    filtered_tokens = []

    for token in tokens:
        if token in NEGATION_WORDS:
            filtered_tokens.append(token)
        elif token not in BASIC_STOP_WORDS:
            filtered_tokens.append(token)

    return filtered_tokens


def lemmatize_tokens(tokens):
    lemmatized = []

    for token in tokens:
        lemma = SIMPLE_LEMMAS.get(token, token)
        lemmatized.append(lemma)

    return lemmatized