# Rapport final — SmartReview AI

**Auteur :** Issam BELHORMA 
**Formation :** Master 1 Data & IA — NLP et Text Mining — Coda Orléans  
**Date :** Juin 2026

---

## 1. Présentation du projet

SmartReview AI est une application d'analyse automatique d'avis clients francophones. Elle combine deux approches complémentaires : un **modèle ML classique** (TF-IDF + classifieur) entraîné sur un corpus de 1 000 avis labelisés, et un **modèle d'IA générative** (Claude d'Anthropic) interrogé via un prompt structuré. Les deux résultats sont produits en parallèle et affichés côte à côte dans une interface Streamlit, avec un indicateur visuel d'accord ou de désaccord.

---

## 2. Architecture du projet

```
smartreview-ai/
│
├── app.py                                    # Interface Streamlit principale
├── requirements.txt
├── .env.example
│
├── data/
│   ├── raw/
│   │   ├── avis_test.csv                    # Echantillon données test
│   │   └── reviews_nlp.csv                  # Corpus brut — 1 000 avis
│   └── processed/
│       └── Avis_client_processed.csv        # Corpus prétraité (notebook 01)
│
├── notebooks/
│   ├── 01_exploration_preprocessing.ipynb
│   └── 02_model_training.ipynb
│
├── models/
│   ├── tfidf_vectorizer.joblib
│   ├── label_encoder.joblib
│   └── sentiment_classifier.joblib
│
├── prompts/
│   └── prompt_analyse_sentiment.md
│
├── src/
│   ├── ai_client.py
│   ├── prompt_builder.py
│   ├── ml_classifier.py
│   └── analysis_utils.py
│
└── reports/
    └── rapport_final.md
```

**Flux de traitement :**

```
[Avis brut]
    → analysis_utils.py     (prétraitement : clean → tokenize → stop words → lemmatize)
    → ml_classifier.py      (TF-IDF + LinearSVC → sentiment + probabilités)
    → ai_client.py          (Claude API → JSON structuré)
    → app.py                (affichage côte à côte + indicateur d'accord)
```

---

## 3. Données — Exploration (Notebook 01)

### 3.1 Corpus brut

Le corpus `reviews_nlp.csv` contient **1 000 avis clients** en français, sans aucune valeur manquante ni doublon.

| Colonne | Type | Description |
|---|---|---|
| `id` | int64 | Identifiant unique |
| `avis` | str | Texte brut de l'avis |
| `label` | str | Sentiment cible |
| `categorie` | str | Domaine métier concerné |
| `difficulte` | str | Complexité linguistique |

### 3.2 Distribution des labels

| Label | Effectif | % |
|---|---|---|
| positif | 674 | 67,4% |
| négatif | 185 | 18,5% |
| mitigé | 85 | 8,5% |
| neutre | 56 | 5,6% |

Le corpus est **fortement déséquilibré** en faveur de la classe `positif` (67,4%). Ce déséquilibre explique en partie les difficultés du modèle sur les classes minoritaires `mitigé` et `neutre`.

### 3.3 Distribution des catégories

| Catégorie | Effectif |
|---|---|
| general | 206 |
| produit | 196 |
| support | 182 |
| livraison | 152 |
| application | 142 |
| remboursement | 48 |
| administratif | 46 |
| paiement | 28 |

### 3.4 Distribution des niveaux de difficulté

| Difficulté | Effectif | Description |
|---|---|---|
| simple | 497 | Avis direct, sans ambiguïté |
| négation_positive | 191 | Ex : "je ne pensais pas être déçu" |
| problème | 155 | Description d'un incident |
| opinion_mixte | 82 | Éléments positifs et négatifs mêlés |
| factuel | 55 | Constat neutre sans opinion |
| sarcasme | 20 | Ironie ou antiphrase |

### 3.5 Longueur des avis

| Stat | Caractères | Mots |
|---|---|---|
| Moyenne | 45,7 | 7,6 |
| Médiane | 45 | 7 |
| Min | 15 | 2 |
| Max | 92 | 15 |
| Écart-type | 10,2 | 2,1 |

Les avis `mitigé` sont les plus longs en moyenne (50,99 caractères, 8,25 mots), ce qui est cohérent : exprimer une opinion nuancée nécessite plus de mots. Les avis `neutre` sont les plus courts (34,80 caractères, 5,54 mots), car ils correspondent souvent à des constats factuels brefs.

### 3.6 Prétraitement

Le pipeline appliqué à chaque avis (colonne `processed_avis`) :

1. **Nettoyage** — minuscules, suppression de la ponctuation, normalisation des espaces
2. **Tokenisation** — découpage par espaces
3. **Suppression des stop words** — liste française personnalisée, les négations (`pas`, `ne`, `jamais`) sont conservées intentionnellement
4. **Lemmatisation** — dictionnaire de règles couvrant les formes verbales et adjectivales courantes

Le fichier exporté `Avis_client_processed.csv` contient 1 000 lignes et 9 colonnes, sans aucune valeur manquante après traitement.

---

## 4. Modélisation (Notebook 02)

### 4.1 Split et vectorisation

Le corpus prétraité (colonne `processed_avis`) est divisé en **800 avis pour l'entraînement** et **200 pour le test**, avec stratification sur les labels pour respecter la distribution du corpus.

La vectorisation TF-IDF est configurée avec :
- unigrammes et bigrammes (`ngram_range=(1,2)`)
- maximum 5 000 features
- `min_df=2` pour ignorer les termes très rares
- `sublinear_tf=True` pour atténuer les termes très fréquents

### 4.2 Comparaison des classifieurs

Trois classifieurs ont été entraînés et évalués :

| Modèle | Accuracy test | CV mean (5-fold) | CV std |
|---|---|---|---|
| Logistic Regression | 69,50% | 73,62% | ±1,08% |
| Naive Bayes | 72,50% | 78,50% | ±2,42% |
| **LinearSVC (calibré)** | **78,50%** | **80,00%** | **±2,40%** |

Le **LinearSVC calibré** est retenu comme meilleur modèle sur les deux critères : accuracy test et score de cross-validation. Il est sauvegardé avec le vectorizer et le label encoder dans le dossier `models/`.

### 4.3 Rapport de classification — LinearSVC

| Classe | Précision | Rappel | F1-score | Support |
|---|---|---|---|---|
| mitigé | 0,57 | 0,24 | 0,33 | 17 |
| neutre | 0,83 | 0,45 | 0,59 | 11 |
| négatif | 0,57 | 0,62 | 0,60 | 37 |
| positif | 0,85 | 0,93 | 0,89 | 135 |
| **accuracy** | | | **0,79** | **200** |
| macro avg | 0,71 | 0,56 | 0,60 | 200 |
| weighted avg | 0,77 | 0,79 | 0,77 | 200 |

### 4.4 Analyse des performances par classe

**`positif` (F1 = 0,89)** — Très bonne performance, cohérente avec le poids de cette classe dans le corpus (67,4%). Le modèle a eu suffisamment d'exemples pour apprendre des patterns robustes.

**`négatif` (F1 = 0,60)** — Performance correcte mais limitée par la précision (0,57) : le modèle produit des faux positifs négatifs, notamment sur des avis mitigés ou sarcastiques.

**`neutre` (F1 = 0,59)** — Le rappel faible (0,45) indique que beaucoup d'avis neutres sont mal classés. La classe est très petite (11 exemples en test) et les avis factuels courts manquent de features discriminantes après TF-IDF.

**`mitigé` (F1 = 0,33)** — La performance la plus faible, avec un rappel de seulement 0,24. La classe est petite (17 exemples en test) et intrinsèquement ambiguë : un avis mitigé contient à la fois des mots positifs et négatifs, ce qui rend la vectorisation TF-IDF peu discriminante pour cette classe.

---

## 5. Tests sur les 10 avis de référence

Les 10 avis de `data/avis_test.csv` ont été soumis à l'application en mode **simulation** (sans appel API réel).

| id | Avis | Label attendu | IA prédit | IA ✓ | ML prédit | ML score | ML ✓ | Accord |
|---|---|---|---|---|---|---|---|---|
| 1 | Le service client a été très rapide… | positif | positif | ✅ | positif | 91% | ✅ | ✅ |
| 2 | Je suis très déçu, ma commande est arrivée cassée | négatif | négatif | ✅ | négatif | 79% | ✅ | ✅ |
| 3 | Votre demande a été enregistrée | neutre | neutre | ✅ | positif | 53% | ❌ | ❌ |
| 4 | Le produit est bon mais la livraison… | mitigé | négatif | ❌ | mitigé | 74% | ✅ | ❌ |
| 5 | Super, encore une panne de l'application | négatif | mitigé | ❌ | négatif | 71% | ✅ | ❌ |
| 6 | Commande reçue aujourd'hui | neutre | neutre | ✅ | négatif | 48% | ❌ | ❌ |
| 7 | Paiement refusé alors que ma carte fonctionne ailleurs | négatif | négatif | ✅ | négatif | 50% | ✅ | ✅ |
| 8 | Pas mal du tout, je pensais être déçu | positif | négatif | ❌ | positif | 80% | ✅ | ❌ |
| 9 | Le remboursement est enfin arrivé après trois semaines | mitigé | neutre | ❌ | négatif | 78% | ❌ | ❌ |
| 10 | Application claire, rapide et agréable | positif | positif | ✅ | positif | 94% | ✅ | ✅ |

**Résultats synthétiques :**

| Métrique | Valeur |
|---|---|
| Accuracy IA (simulation) | 60% — 6/10 |
| Accuracy ML | 70% — 7/10 |
| Taux d'accord ML / IA | 40% — 4/10 |

---

## 6. Analyse des erreurs

### Erreurs communes aux deux approches

**Avis 9 — "Le remboursement est enfin arrivé après trois semaines"**  
Label attendu : `mitigé`. L'IA prédit `neutre`, le ML prédit `négatif`. L'adverbe *enfin* porte implicitement une frustration, mais il n'est pas un marqueur de sentiment fort dans un vocabulaire TF-IDF. La nuance temporelle ("après trois semaines") est difficile à capturer sans contexte sémantique profond.

### Erreurs spécifiques à l'IA (simulation)

**Avis 4 — opinion mixte**  
"Le produit est bon mais la livraison a été beaucoup trop longue" → prédit `négatif` au lieu de `mitigé`. La conjonction adversative *mais* crée une balance positive/négative que la simulation par mots-clés ne gère pas : elle détecte le terme négatif dominant et bascule.

**Avis 5 — sarcasme**  
"Super, encore une panne de l'application" → prédit `mitigé` au lieu de `négatif`. Le mot *super* est détecté comme positif par la simulation. C'est un cas typique de sarcasme où le sens est inversé par le contexte, ce que seul un LLM réel avec un prompt adapté peut traiter correctement.

**Avis 8 — négation positive**  
"Pas mal du tout, je pensais être déçu" → prédit `négatif` au lieu de `positif`. La présence du mot *déçu* et de la négation *pas* déclenche une mauvaise classification. La double négation crée un sens positif que la simulation par mots-clés ne peut pas résoudre.

### Erreurs spécifiques au ML

**Avis 3 — avis factuel court**  
"Votre demande a été enregistrée" → prédit `positif` au lieu de `neutre`. Après prétraitement, il ne reste que très peu de tokens ("demand enregistr"). Le modèle n'a pas suffisamment de signal TF-IDF et bascule vers la classe majoritaire `positif`.

**Avis 6 — avis factuel court**  
"Commande reçue aujourd'hui" → prédit `négatif` au lieu de `neutre`. Même problème : avis très court, peu de features discriminantes après suppression des stop words, score bas (48%).

---

## 7. Limites

**Déséquilibre des classes.** La classe `positif` représente 67,4% du corpus. Cela biaise le modèle vers cette classe majoritaire, au détriment des classes `mitigé` et `neutre` très minoritaires. Une technique de rééchantillonnage (SMOTE, class_weight='balanced') améliorerait les F1-scores sur ces classes.

**Avis courts et factuels.** Les avis de 2 à 4 mots offrent très peu de features après prétraitement. Le modèle TF-IDF est mis en difficulté et tend à prédire la classe majoritaire par défaut.

**Sarcasme et négation positive.** Ces deux phénomènes linguistiques sont invisibles pour un classifieur bag-of-words. TF-IDF ne capture pas le sens contextuel. Le mode simulation de l'IA souffre du même problème ; seul un appel API réel à Claude avec un prompt incluant des consignes explicites sur le sarcasme permet de les traiter.

**Mode simulation vs API réelle.** Les résultats de test ci-dessus utilisent le mode simulation. Avec un appel API Claude réel, l'accuracy IA sur les cas de sarcasme et de négation positive serait significativement meilleure, notamment sur les avis 5 et 8.

**Taille du corpus.** 1 000 avis est suffisant pour un prototype, mais insuffisant pour entraîner un modèle robuste sur 4 classes déséquilibrées. Les classes `mitigé` (85 exemples) et `neutre` (56 exemples) sont trop peu représentées pour que le modèle généralise bien.

---

## 8. Améliorations possibles

**Rééquilibrage des classes.** Appliquer `class_weight='balanced'` dans le LinearSVC ou utiliser SMOTE sur les représentations TF-IDF pour augmenter artificiellement les classes minoritaires.

**Enrichissement du corpus.** Collecter ou générer davantage d'avis `mitigé` et `neutre`, notamment des cas de sarcasme et de négation positive qui sont les plus difficiles à classifier.

**Few-shot prompting.** Ajouter des exemples concrets dans le prompt Claude (un exemple par classe difficile) pour améliorer la détection du sarcasme et des opinions mixtes.

**Embeddings.** Remplacer TF-IDF par des embeddings multilingues (CamemBERT, sentence-transformers) pour capturer la sémantique et le contexte, ce que TF-IDF ne peut pas faire.

**Fine-tuning.** Sur un corpus plus large et bien équilibré, fine-tuner un modèle de langue pré-entraîné sur cette tâche de classification spécifique.

**Persistance des analyses.** Stocker les résultats dans une base SQLite pour un historique permanent entre sessions, avec possibilité de correction manuelle des prédictions incorrectes.

---

## 9. Conclusion

Ce projet démontre la complémentarité des deux approches NLP. Le modèle ML TF-IDF + LinearSVC atteint **78,5% d'accuracy** sur le corpus de test global, avec d'excellentes performances sur la classe `positif` (F1 = 0,89) mais des difficultés sur les classes minoritaires. Sur les 10 avis de référence, il obtient **70%** avec des erreurs concentrées sur les avis très courts et factuels.

Le modèle IA en mode simulation atteint **60%** sur les mêmes avis, avec des erreurs sur le sarcasme, les négations positives et les opinions mixtes — cas qui seraient mieux traités par un appel API réel à Claude grâce au prompt structuré incluant des consignes explicites sur ces phénomènes.

Le faible **taux d'accord de 40%** entre les deux approches illustre que ML et IA font des erreurs sur des cas différents, ce qui renforce l'intérêt de les combiner plutôt que d'en choisir une seule. La principale leçon de ce projet est que la qualité du prétraitement et l'équilibre des classes sont aussi déterminants que le choix du modèle pour une tâche de classification de sentiment sur corpus court et déséquilibré.