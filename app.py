"""
app.py — SmartReview AI
Application Streamlit d'analyse d'avis clients avec IA générative (Claude)
et fallback simulation pédagogique.

Lancement :
    streamlit run app.py
"""

import csv
import io
import json
import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from src.ai_client import analyze_review

# Config

st.set_page_config(
    page_title="SmartReview AI",
    page_icon="🔍",
    layout="wide",
)

# Session state

if "history" not in st.session_state:
    st.session_state.history = []

# Helpers

SENTIMENT_COLORS = {
    "positif": "🟢",
    "négatif": "🔴",
    "neutre": "🔵",
    "mitigé": "🟡",
}

CONFIDENCE_COLORS = {
    "élevée": "✅",
    "moyenne": "⚠️",
    "faible": "❓",
}


def display_result(result: dict, source: str, avis_text: str):
    sentiment = result.get("sentiment", "?")
    confiance = result.get("confiance", "?")

    emoji = SENTIMENT_COLORS.get(sentiment, "⚪")
    conf_icon = CONFIDENCE_COLORS.get(confiance, "")

    col1, col2, col3 = st.columns(3)
    col1.metric("Sentiment", f"{emoji} {sentiment.capitalize()}")
    col2.metric("Confiance", f"{conf_icon} {confiance.capitalize()}")
    col3.metric("Catégorie", result.get("categorie", "?").capitalize())

    st.markdown(f"**Justification :** {result.get('justification', '—')}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Points positifs :**")
        pts = result.get("points_positifs", [])
        if pts:
            for p in pts:
                st.markdown(f"- ✅ {p}")
        else:
            st.markdown("*Aucun*")

    with c2:
        st.markdown("**Points négatifs :**")
        pts = result.get("points_negatifs", [])
        if pts:
            for p in pts:
                st.markdown(f"- ❌ {p}")
        else:
            st.markdown("*Aucun*")

    st.info(f"**Action recommandée :** {result.get('action_recommandee', '—')}")
    st.caption(f"Source : {source} | {datetime.now().strftime('%H:%M:%S')}")


def export_history_csv(history: list[dict]) -> str:
    """Génère un CSV de l'historique des analyses."""
    output = io.StringIO()
    if not history:
        return ""
    fields = ["timestamp", "avis", "sentiment", "confiance", "categorie",
              "justification", "points_positifs", "points_negatifs",
              "action_recommandee", "source"]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(history)
    return output.getvalue()


# UI

st.title("🔍 SmartReview AI")
st.markdown(
    "Application d'analyse d'avis clients par IA générative (**Claude**) "
    "avec prompt engineering structuré."
)

# Sidebar — configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    force_sim = st.checkbox(
        "Mode simulation (sans API)",
        value=not bool(os.getenv("ANTHROPIC_API_KEY")),
        help="Cocher si vous n'avez pas de clé API Anthropic.",
    )
    st.markdown("---")
    st.markdown("**Modèle :** `claude-sonnet-4-6`")
    st.markdown("**Prompt :** `prompt_analyse_sentiment.md`")
    st.markdown("---")

    if st.session_state.history:
        csv_data = export_history_csv(st.session_state.history)
        st.download_button(
            "⬇️ Exporter l'historique CSV",
            data=csv_data,
            file_name=f"smartreview_resultats_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
        if st.button("🗑️ Vider l'historique"):
            st.session_state.history = []
            st.rerun()

# Tabs
tab_single, tab_batch, tab_history = st.tabs(
    ["✍️ Analyse manuelle", "📂 Analyse en lot", "📋 Historique"]
)

# ---- Tab 1 : Analyse manuelle ----
with tab_single:
    review_text = st.text_area(
        "Saisissez un avis client :",
        height=120,
        placeholder="Exemple : Le produit est bon mais la livraison a été beaucoup trop longue.",
    )

    if st.button("🔍 Analyser l'avis", type="primary"):
        if not review_text.strip():
            st.warning("Veuillez saisir un avis avant de lancer l'analyse.")
        else:
            with st.spinner("Analyse en cours…"):
                try:
                    result, source = analyze_review(review_text, force_simulation=force_sim)
                    st.success("Analyse terminée !")
                    display_result(result, source, review_text)

                    # Ajout à l'historique
                    st.session_state.history.append({
                        "timestamp": datetime.now().isoformat(),
                        "avis": review_text,
                        "source": source,
                        **result,
                        "points_positifs": "; ".join(result.get("points_positifs", [])),
                        "points_negatifs": "; ".join(result.get("points_negatifs", [])),
                    })

                    with st.expander("🔎 JSON brut"):
                        st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")

                except Exception as e:
                    st.error(f"Erreur lors de l'analyse : {e}")

# ---- Tab 2 : Analyse en lot ----
with tab_batch:
    st.markdown(
        "Chargez un fichier CSV avec une colonne `avis` (et optionnellement `label`)."
    )
    uploaded = st.file_uploader("📎 Fichier CSV", type=["csv"])

    # Bouton pour charger le fichier de test par défaut
    use_default = st.button("Utiliser avis_test.csv (données de test)")

    df_batch = None
    if use_default:
        df_batch = pd.read_csv("data/raw/avis_test.csv")
        st.success("Fichier de test chargé !")
    elif uploaded:
        df_batch = pd.read_csv(uploaded)

    if df_batch is not None:
        st.dataframe(df_batch, use_container_width=True)

        if st.button("🚀 Lancer l'analyse sur tout le fichier"):
            results_rows = []
            progress = st.progress(0)
            n = len(df_batch)

            for i, row in df_batch.iterrows():
                avis = str(row.get("avis", ""))
                label = str(row.get("label", ""))

                with st.spinner(f"Analyse {i+1}/{n}…"):
                    try:
                        result, source = analyze_review(avis, force_simulation=force_sim)
                    except Exception:
                        result = {"sentiment": "erreur", "confiance": "faible",
                                  "justification": "Erreur API", "points_positifs": [],
                                  "points_negatifs": [], "categorie": "autre",
                                  "action_recommandee": "—"}
                        source = "erreur"

                correct = (
                    "✅" if result.get("sentiment") == label else "❌"
                ) if label else "—"

                results_rows.append({
                    "id": row.get("id", i + 1),
                    "avis": avis[:60] + "…" if len(avis) > 60 else avis,
                    "label": label,
                    "sentiment_obtenu": result.get("sentiment", "?"),
                    "confiance": result.get("confiance", "?"),
                    "categorie": result.get("categorie", "?"),
                    "correct": correct,
                    "source": source,
                })
                progress.progress((i + 1) / n)

            df_results = pd.DataFrame(results_rows)
            st.markdown("### Résultats")
            st.dataframe(df_results, use_container_width=True)

            if "label" in df_results.columns:
                nb_valid = df_results[df_results["label"] != ""]["correct"]
                nb_correct = (nb_valid == "✅").sum()
                nb_total = (nb_valid != "—").sum()
                if nb_total > 0:
                    acc = nb_correct / nb_total
                    st.metric("Accuracy", f"{acc:.0%}", f"{nb_correct}/{nb_total} corrects")

            csv_out = df_results.to_csv(index=False)
            st.download_button(
                "⬇️ Télécharger les résultats",
                data=csv_out,
                file_name="resultats_analyses.csv",
                mime="text/csv",
            )

# ---- Tab 3 : Historique ----
with tab_history:
    if not st.session_state.history:
        st.info("Aucune analyse réalisée dans cette session.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True)

        # Répartition des sentiments
        if len(df_hist) >= 3:
            st.markdown("### Répartition des sentiments")
            counts = df_hist["sentiment"].value_counts()
            st.bar_chart(counts)