"""
Gradio Web Application for Multilingual Healthcare Misinformation RAG
Allows users to enter health-related or general claims in English, Hindi, or Hinglish
and receive automated verification, evidence citations, and risk assessment.
"""

import os
import sys

# Prevent macOS OpenMP / Rust tokenizers thread conflict
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings

# Suppress urllib3 and SSL warnings on macOS
warnings.filterwarnings("ignore")

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gradio as gr

try:
    from src.preprocessing import preprocess_text
    from src.language_detector import detect_language
    from src.health_classifier import is_health_related
    from src.rag_engine import HealthcareRAGEngine
    from src.verdict_engine import assess_verdict_and_risk, generate_explanation
except ModuleNotFoundError:
    from preprocessing import preprocess_text
    from language_detector import detect_language
    from health_classifier import is_health_related
    from rag_engine import HealthcareRAGEngine
    from verdict_engine import assess_verdict_and_risk, generate_explanation

# Default dataset path
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "evidence_dataset.csv")

# Global RAG Engine instance (lazy-loaded on first use)
_rag_engine = None


def get_rag_engine():
    global _rag_engine
    if _rag_engine is None:
        print("Initializing Multilingual RAG Engine...")
        _rag_engine = HealthcareRAGEngine(csv_path=DATASET_PATH)
        print("RAG Engine ready.")
    return _rag_engine


def verify_claim(claim: str):
    """
    Orchestrates the 10-step verification workflow:
    1. Preprocess query
    2. Detect language
    3. Check if health-related
    4. If health-related: retrieve top-3 evidence via FAISS
    5. Evaluate verdict and risk
    6. Generate structured response
    """
    if not claim or not claim.strip():
        return (
            "N/A",
            "N/A",
            "Please enter a claim to verify.",
            "N/A",
            "N/A",
            "",
            ""
        )

    # Step 4: Preprocess
    cleaned_claim = preprocess_text(claim)

    # Step 5: Detect Language
    lang = detect_language(claim)

    # Step 6: Health Classification
    health_flag = is_health_related(cleaned_claim)

    if not health_flag:
        verdict_data = {
            "verdict": "Non-Health Claim",
            "evidence_status": "Filtered before retrieval",
            "risk": "None",
            "confidence": 0.0,
            "stance": "none"
        }
        explanation = generate_explanation(claim, lang, False, verdict_data, [])
        return (
            lang,
            "No (Filtered Out)",
            "Non-Health Claim",
            "None",
            explanation["summary_text"],
            "No retrieval performed for non-health query.",
            explanation["disclaimer"]
        )

    # Step 7: Retrieve Evidence via FAISS
    engine = get_rag_engine()
    retrieved = engine.retrieve_evidence(cleaned_claim, top_k=3)

    # Step 8: Verdict & Risk
    verdict_data = assess_verdict_and_risk(cleaned_claim, retrieved)

    # Step 9: Generate Explanation
    explanation = generate_explanation(claim, lang, True, verdict_data, retrieved)

    # Format Evidence Table / Markdown
    evidence_md = "### Top Retrieved Clinical Evidence\n\n"
    for item in retrieved:
        evidence_md += (
            f"**Rank {item['rank']}** (Similarity: `{item['similarity_score']}` | Stance: `{item['stance'].upper()}`)\n"
            f"> *\"{item['evidence_text']}\"*\n"
            f"**Source**: {item['source']} — [Verified URL Link]({item['url']})\n\n"
            f"---\n"
        )

    return (
        lang,
        "Yes (Proceeded to Retrieval)",
        verdict_data["verdict"],
        verdict_data["risk"],
        explanation["summary_text"],
        evidence_md,
        explanation["disclaimer"]
    )


def create_demo():
    """Builds the Gradio interface."""
    custom_css = """
    .gradio-container { max-width: 950px !important; margin: auto; }
    .risk-badge { font-weight: bold; padding: 4px 8px; border-radius: 4px; }
    """

    with gr.Blocks(title="Multilingual Healthcare Misinformation RAG", css=custom_css) as demo:
        gr.Markdown(
            """
            # 🩺 Multilingual Healthcare Misinformation Verifier (RAG)
            Verify medical claims in **English**, **Hindi (हिंदी)**, and **Hinglish** using cross-lingual dense retrieval and trusted sources (*WHO, CDC, NHS, Cancer.org*).
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                claim_input = gr.Textbox(
                    label="Enter Health Claim",
                    placeholder="e.g., Haldi doodh cancer ko theek karta hai / Vaccines cause autism",
                    lines=2,
                )
                verify_btn = gr.Button("🔍 Verify Claim", variant="primary")

                gr.Markdown("### Demo Test Inputs")
                gr.Examples(
                    examples=[
                        ["Haldi doodh cancer ko theek karta hai"],
                        ["Vaccines autism cause karte hain"],
                        ["Smoking causes lung cancer"],
                        ["Antibiotics cold ko cure karte hain"],
                        ["India won the cricket match"],
                    ],
                    inputs=[claim_input],
                )

            with gr.Column(scale=2):
                with gr.Row():
                    lang_output = gr.Textbox(label="Detected Language", interactive=False)
                    health_output = gr.Textbox(label="Health-Related?", interactive=False)
                with gr.Row():
                    verdict_output = gr.Textbox(label="Scientific Verdict", interactive=False)
                    risk_output = gr.Textbox(label="Risk Assessment", interactive=False)

        with gr.Row():
            with gr.Column():
                explanation_output = gr.Markdown(label="Explanation")
                evidence_output = gr.Markdown(label="Retrieved Evidence")
                disclaimer_output = gr.Markdown(label="Disclaimer")

        verify_btn.click(
            fn=verify_claim,
            inputs=[claim_input],
            outputs=[
                lang_output,
                health_output,
                verdict_output,
                risk_output,
                explanation_output,
                evidence_output,
                disclaimer_output,
            ],
        )

    return demo


if __name__ == "__main__":
    print("Pre-loading Multilingual RAG Engine on main thread...")
    get_rag_engine()
    print("RAG Engine ready! Launching web interface...")
    demo = create_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
