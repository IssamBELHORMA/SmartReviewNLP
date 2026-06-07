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
from src.ai_client import analyze_review, analyze_review_full
from src.ml_classifier import is_model_available

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SmartReview AI",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

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
    st.markdown("**Modèle IA :** `claude-sonnet-4-6`")
    st.markdown("**Prompt :** `prompt_analyse_sentiment.md`")

    st.markdown("---")
    st.markdown("**Modèle ML :**")
    if is_model_available():
        st.success("✅ Modèle chargé")
    else:
        st.warning("⚠️ Modèle absent — lancez `02_model_training.ipynb`")

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
                    full = analyze_review_full(review_text, force_simulation=force_sim)
                    ia_result = full["ia"]
                    ml_result = full["ml"]
                    ia_source = full["ia_source"]
                    accord    = full["accord"]

                    st.success("Analyse terminée !")

                    # ── Bandeau accord / désaccord ──────────────────────
                    if accord is True:
                        st.success("🤝 ML et IA sont **d'accord** sur le sentiment.")
                    elif accord is False:
                        ml_sent = ml_result.get("sentiment", "?")
                        ia_sent = ia_result.get("sentiment", "?")
                        st.warning(
                            f"⚡ **Désaccord** — ML prédit `{ml_sent}`, "
                            f"IA prédit `{ia_sent}`."
                        )
                    else:
                        st.info("ℹ️ Modèle ML non disponible — résultat IA uniquement.")

                    # ── Deux colonnes côte à côte ────────────────────────
                    col_ml, col_ia = st.columns(2)

                    with col_ml:
                        st.markdown("### 🤖 Modèle ML (TF-IDF + classifieur)")
                        if ml_result and "sentiment" in ml_result:
                            emoji = SENTIMENT_COLORS.get(ml_result["sentiment"], "⚪")
                            conf_icon = CONFIDENCE_COLORS.get(ml_result["confiance"], "")
                            st.metric("Sentiment", f"{emoji} {ml_result['sentiment'].capitalize()}")
                            st.metric("Confiance", f"{conf_icon} {ml_result['confiance'].capitalize()}")
                            st.metric("Score (proba max)", f"{ml_result['score']:.2%}")
                            with st.expander("Probabilités par classe"):
                                probas = ml_result.get("probabilites", {})
                                for cls, p in sorted(probas.items(), key=lambda x: -x[1]):
                                    bar = "█" * int(p * 20)
                                    st.text(f"{cls:<12} {bar:<20} {p:.2%}")
                        elif ml_result and "erreur" in ml_result:
                            st.error(f"Erreur ML : {ml_result['erreur']}")
                        else:
                            st.info("Modèle ML non entraîné.\nLancez `02_model_training.ipynb`.")

                    with col_ia:
                        st.markdown(f"### 🧠 IA générative (Claude — {ia_source})")
                        display_result(ia_result, ia_source, review_text)

                    # ── Historique ──────────────────────────────────────
                    st.session_state.history.append({
                        "timestamp":       datetime.now().isoformat(),
                        "avis":            review_text,
                        "ml_sentiment":    ml_result.get("sentiment", "—") if ml_result else "—",
                        "ml_confiance":    ml_result.get("confiance", "—") if ml_result else "—",
                        "ml_score":        ml_result.get("score", "") if ml_result else "",
                        "ia_sentiment":    ia_result.get("sentiment", "—"),
                        "ia_confiance":    ia_result.get("confiance", "—"),
                        "ia_categorie":    ia_result.get("categorie", "—"),
                        "ia_justification": ia_result.get("justification", "—"),
                        "ia_source":       ia_source,
                        "accord":          accord,
                    })

                    with st.expander("🔎 JSON complet"):
                        st.code(json.dumps(full, ensure_ascii=False, indent=2), language="json")

                except Exception as e:
                    st.error(f"Erreur lors de l'analyse : {e}")

# ---- Tab 2 : Analyse en lot ----
with tab_batch:
    st.markdown(
        "Chargez un fichier CSV avec une colonne `avis`"
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
                        full   = analyze_review_full(avis, force_simulation=force_sim)
                        result = full["ia"]
                        source = full["ia_source"]
                        ml_res = full["ml"] or {}
                        accord = full["accord"]
                    except Exception:
                        result = {"sentiment": "erreur", "confiance": "faible",
                                  "justification": "Erreur", "points_positifs": [],
                                  "points_negatifs": [], "categorie": "autre",
                                  "action_recommandee": "—"}
                        source = "erreur"
                        ml_res = {}
                        accord = None

                ia_correct = (
                    "✅" if result.get("sentiment") == label else "❌"
                ) if label else "—"

                ml_correct = (
                    "✅" if ml_res.get("sentiment") == label else "❌"
                ) if (label and ml_res.get("sentiment")) else "—"

                results_rows.append({
                    "id":               row.get("id", i + 1),
                    "avis":             avis[:60] + "…" if len(avis) > 60 else avis,
                    "label":    label,
                    "ia_sentiment":     result.get("sentiment", "?"),
                    "ia_confiance":     result.get("confiance", "?"),
                    "ia_correct":       ia_correct,
                    "ml_sentiment":     ml_res.get("sentiment", "—"),
                    "ml_score":         f"{ml_res.get('score', 0):.0%}" if ml_res.get("score") else "—",
                    "ml_correct":       ml_correct,
                    "accord":           "✅" if accord else ("❌" if accord is False else "—"),
                    "source":           source,
                })
                progress.progress((i + 1) / n)

            df_results = pd.DataFrame(results_rows)
            st.markdown("### Résultats")
            st.dataframe(df_results, use_container_width=True)

            if "label" in df_results.columns:
                has_label = df_results["label"].astype(str).str.strip() != ""
                sub = df_results[has_label]
                if len(sub) > 0:
                    ia_acc = (sub["ia_correct"] == "✅").sum() / len(sub)
                    ml_acc = (sub["ml_correct"] == "✅").sum() / len(sub) if "ml_correct" in sub else None
                    acc_cols = st.columns(3)
                    acc_cols[0].metric("Accuracy IA", f"{ia_acc:.0%}",
                                       f"{(sub['ia_correct']=='✅').sum()}/{len(sub)}")
                    if ml_acc is not None:
                        acc_cols[1].metric("Accuracy ML", f"{ml_acc:.0%}",
                                           f"{(sub['ml_correct']=='✅').sum()}/{len(sub)}")
                    accord_rate = (sub["accord"] == "✅").sum() / len(sub)
                    acc_cols[2].metric("Taux d'accord ML/IA", f"{accord_rate:.0%}")

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