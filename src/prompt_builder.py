import os

PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "prompts", "prompt_analyse_sentiment.md"
)

PLACEHOLDER = "{{AVIS_CLIENT}}"


def load_prompt_template(path: str = PROMPT_PATH) -> str:
    """Charge le fichier de prompt Markdown."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(avis: str, template: str | None = None) -> str:
    """
    Injecte l'avis client dans le template de prompt.

    Args:
        avis: Texte de l'avis client.
        template: Template optionnel (charge le fichier par défaut si None).

    Returns:
        Prompt final prêt à être envoyé au modèle.
    """
    if template is None:
        template = load_prompt_template()
    return template.replace(PLACEHOLDER, avis.strip())


def build_system_prompt() -> str:
    """Retourne le system prompt court pour l'API."""
    return (
        "Tu es un assistant expert en analyse de sentiment d'avis clients francophones. "
        "Tu réponds uniquement en JSON valide, sans texte supplémentaire."
    )