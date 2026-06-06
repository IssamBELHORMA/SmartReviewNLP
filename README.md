# SmartReview AI

**Auteur :** [Votre Prénom Nom]  
**Formation :** Master 1 Data & IA — NLP et Text Mining — Coda Orléans  
**Projet final :** Application simple d'analyse d'avis clients avec IA générative et prompt engineering

---

## Présentation

SmartReview AI est une application d'analyse de sentiment d'avis clients francophones.  
Elle utilise un prompt structuré envoyé au modèle **Claude (Anthropic)** via l'API, et retourne une analyse JSON exploitable directement dans l'interface.

Un **mode simulation pédagogique** (sans API) est disponible si vous n'avez pas de clé Anthropic.

---

## Architecture

```
smartreview-ai/
├── README.md
├── app.py                            # Application Streamlit
├── requirements.txt
├── .env.example
├── data
│   ├── raw            
│       ├── avis_test.csv                 # 10 avis de test labelisés
│       └── reviews_nlp.csv               # Corpus complet 1000 avis
│   └── processed
│       └── Acis_client_processed.csv              
├── prompts/
│   └── prompt_analyse_sentiment.md   # Prompt principal
├── src/
│   ├── ai_client.py                  # Appel API Anthropic + simulation
│   ├── prompt_builder.py             # Construction du prompt
│   └── analysis_utils.py             # NLP utils (clean, tokenize, etc.)
├── notebooks/
│   └── 01_exploration_preprocessing.ipynb
├── models/
└── reports/
    └── rapport_final.md
```

---

## Installation

```bash
# Cloner le dépôt
git clone <votre_repo_url>
cd smartreview-ai

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate   # Windows : venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

---

## Configuration (optionnelle — API Anthropic)

Créez un fichier `.env` à la racine :

```
ANTHROPIC_API_KEY=sk-ant-...
```

Sans clé, l'application bascule automatiquement en **mode simulation**.

---

## Lancement

```bash
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`.

---

## Fonctionnalités

| Fonctionnalité | Disponible |
|---|---|
| Analyse manuelle d'un avis | ✅ |
| Analyse en lot (CSV) | ✅ |
| Appel API Claude réel | ✅ |
| Simulation pédagogique (fallback) | ✅ |
| Réponse JSON structurée | ✅ |
| Historique des analyses | ✅ |
| Export CSV des résultats | ✅ |
| Visualisation des sentiments | ✅ |
| Tests sur 10 avis labelisés | ✅ |

---

## Format de sortie

```json
{
  "sentiment": "positif|négatif|neutre|mitigé",
  "confiance": "faible|moyenne|élevée",
  "justification": "...",
  "points_positifs": ["..."],
  "points_negatifs": ["..."],
  "categorie": "livraison|paiement|support|application|produit|remboursement|administratif|general|autre",
  "action_recommandee": "..."
}
```