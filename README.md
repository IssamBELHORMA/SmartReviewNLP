# SmartReview AI

**Auteur :** Issam BELHORMA
**Formation :** Master 1 Data & IA — NLP et Text Mining — Coda Orléans  
**Projet :** Analyse d'avis clients par IA générative et modèle de classification ML

---

## Présentation

SmartReview AI est une application d'analyse de sentiment d'avis clients francophones.

Elle combine deux approches complémentaires :

- un **modèle ML classique** (TF-IDF + classifieur) entraîné sur un corpus de 1 000 avis labelisés
- un **modèle d'IA générative** (Claude d'Anthropic) appelé via un prompt structuré

Les deux résultats sont affichés côte à côte dans l'interface, avec indication d'accord ou de désaccord entre les deux approches.

Un **mode simulation pédagogique** (sans clé API) est disponible comme fallback.

---

## Structure du projet

```
smartreview-ai/
│
├── app.py                                    # Application Streamlit principale
├── requirements.txt                          # Dépendances Python
├── .env.example                              # Template de configuration
├── README.md
│
├── data/
│   ├── raw/
│   │   └── reviews_nlp.csv                  # Corpus brut — 1 000 avis labelisés
│   └── processed/
│       └── Avis_client_processed.csv         # Corpus prétraité (généré par notebook 01)
│
├── notebooks/
│   ├── 01_exploration_preprocessing.ipynb   # EDA + nettoyage + prétraitement
│   └── 02_model_training.ipynb              # Entraînement + évaluation + sauvegarde du modèle
│
├── models/                                   # Artefacts ML (générés par notebook 02)
│   ├── tfidf_vectorizer.joblib
│   ├── label_encoder.joblib
│   └── sentiment_classifier.joblib
│
├── prompts/
│   └── prompt_analyse_sentiment.md           # Prompt principal (template avec placeholder)
│
├── src/
│   ├── ai_client.py                          # Appel API Claude + simulation + analyse combinée
│   ├── prompt_builder.py                     # Injection de l'avis dans le template de prompt
│   ├── ml_classifier.py                      # Chargement du modèle ML et prédiction
│   └── analysis_utils.py                     # Pipeline NLP : nettoyage, tokenisation, lemmatisation
│
└── reports/
    ├── rapport_final.md                      # Rapport de projet
    ├── label_distribution.png                # Graphique généré par notebook 02
    ├── model_comparison.png                  # Comparaison des classifieurs
    ├── confusion_matrix.png                  # Matrice de confusion du meilleur modèle
    └── feature_importance.png                # Features TF-IDF les plus discriminantes
```

---

## Installation

```bash
# 1. Cloner le dépôt
git clone <url_du_repo>
cd smartreview-ai

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## Configuration

Créez un fichier `.env` à la racine du projet :

```
ANTHROPIC_API_KEY=sk-ant-...
```

Sans clé API, l'application bascule automatiquement en **mode simulation pédagogique** (analyse par mots-clés, aucune connexion externe requise).

---

## Ordre d'exécution

Les notebooks doivent être exécutés dans l'ordre avant de lancer l'application.

**Étape 1 — Prétraitement**

```bash
# Ouvrir et exécuter toutes les cellules
notebooks/01_exploration_preprocessing.ipynb
```

Produit : `data/processed/Avis_client_processed.csv`

**Étape 2 — Entraînement du modèle**

```bash
# Ouvrir et exécuter toutes les cellules
notebooks/02_model_training.ipynb
```

Produit : les trois fichiers `.joblib` dans `models/` et les figures dans `reports/`

**Étape 3 — Lancer l'application**

```bash
streamlit run app.py
```

L'interface s'ouvre sur `http://localhost:8501`.

---

## Fonctionnalités de l'application

| Fonctionnalité | Description |
|---|---|
| Analyse manuelle | Saisie libre d'un avis, résultats ML et IA côte à côte |
| Indicateur d'accord | Signal visuel si ML et IA convergent ou divergent |
| Probabilités par classe | Score de confiance par sentiment (modèle ML) |
| Justification IA | Explication textuelle générée par Claude |
| Analyse en lot | Chargement d'un CSV, analyse de tous les avis en séquence |
| Calcul d'accuracy | Comparaison aux labels attendus si présents dans le CSV |
| Historique de session | Liste des analyses réalisées, exportable en CSV |
| Mode simulation | Fallback sans API, basé sur mots-clés |

---

## Format du CSV d'entrée (analyse en lot)

Le fichier CSV uploadé dans l'onglet **Analyse en lot** doit contenir au minimum une colonne `avis`. La colonne `label` est optionnelle et permet le calcul automatique de l'accuracy.

```
id,avis,label
1,"Le service client a été très rapide.",positif
2,"Ma commande est arrivée cassée.",négatif
3,"Commande reçue aujourd'hui.",neutre
```

---

## Format de sortie (IA générative)

```json
{
  "sentiment": "positif | négatif | neutre | mitigé",
  "confiance": "faible | moyenne | élevée",
  "justification": "explication courte du sentiment détecté",
  "points_positifs": ["..."],
  "points_negatifs": ["..."],
  "categorie": "livraison | paiement | support | application | produit | remboursement | administratif | general | autre",
  "action_recommandee": "action suggérée pour le service concerné"
}
```

---

## Format de sortie (modèle ML)

```json
{
  "sentiment": "positif | négatif | neutre | mitigé",
  "confiance": "faible | moyenne | élevée",
  "score": 0.87,
  "probabilites": {
    "positif": 0.87,
    "négatif": 0.05,
    "neutre": 0.04,
    "mitigé": 0.04
  }
}
```

---

## Pipeline NLP (`src/analysis_utils.py`)

Le prétraitement appliqué à chaque avis avant vectorisation :

1. **Nettoyage** — minuscules, suppression de la ponctuation, espaces multiples
2. **Tokenisation** — découpage par espaces
3. **Suppression des stop words** — liste française personnalisée (les négations sont conservées)
4. **Lemmatisation** — dictionnaire de règles pour les formes verbales et adjectivales courantes

---

## Données

Le corpus `data/raw/reviews_nlp.csv` contient **1 000 avis clients** en français avec les colonnes suivantes :

| Colonne | Description |
|---|---|
| `id` | Identifiant unique |
| `avis` | Texte brut de l'avis |
| `label` | Sentiment : `positif`, `négatif`, `neutre`, `mitigé` |
| `categorie` | Domaine : `livraison`, `produit`, `support`, `application`, `paiement`, `remboursement`, `administratif`, `general` |
| `difficulte` | Complexité linguistique : `simple`, `factuel`, `problème`, `opinion_mixte`, `sarcasme`, `négation_positive` |

---

## Dépendances principales

| Package | Usage |
|---|---|
| `streamlit` | Interface web |
| `anthropic` | Appel API Claude |
| `scikit-learn` | TF-IDF, classifieurs, évaluation |
| `joblib` | Sauvegarde et chargement des modèles |
| `pandas` | Manipulation des données |
| `matplotlib` / `seaborn` | Visualisations dans les notebooks |
| `python-dotenv` | Chargement de la clé API depuis `.env` |