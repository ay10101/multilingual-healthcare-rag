"""
Verdict and Explanation Engine Module
Determines scientific verification verdict, assesses health risk levels,
and formats standardized educational explanations with clickable citations.
"""

from typing import List, Dict, Any

# Dangerous medical advice indicators that elevate risk from High to Very High
DANGEROUS_PATTERNS = [
    "stop chemo", "skip chemo", "stop insulin", "no insulin", "avoid vaccine",
    "do not vaccinate", "don't vaccinate", "dawai band", "chemo mat lo",
    "insulin mat lo", "teeka mat lagao", "toxic", "bleach", "gomutra cure"
]

HIGH_SIMILARITY_THRESHOLD = 0.35
MODERATE_SIMILARITY_THRESHOLD = 0.22


def assess_verdict_and_risk(claim: str, retrieved_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates verdict, evidence status, and risk level based on retrieved evidence.
    """
    if not retrieved_evidence or len(retrieved_evidence) == 0:
        return {
            "verdict": "No relevant evidence",
            "evidence_status": "No matching clinical documents found",
            "risk": "Medium",
            "confidence": 0.0,
            "stance": "none"
        }

    top_evidence = retrieved_evidence[0]
    top_score = top_evidence.get("similarity_score", 0.0)
    top_stance = top_evidence.get("stance", "").strip().lower()

    top_topic = top_evidence.get("claim_topic", "").strip().lower()

    # A true conflict only occurs if a contradictory stance is on the same topic and competitively close
    has_conflict = any(
        item.get("claim_topic", "").strip().lower() == top_topic
        and item.get("stance", "").strip().lower() != top_stance
        and (top_score - item.get("similarity_score", 0.0)) <= 0.12
        and item.get("similarity_score", 0.0) >= MODERATE_SIMILARITY_THRESHOLD
        for item in retrieved_evidence[1:]
    )

    if top_score < MODERATE_SIMILARITY_THRESHOLD:
        verdict = "No relevant evidence"
        evidence_status = "No relevant clinical evidence found in trusted repository"
        risk = "Medium"
    elif has_conflict:
        verdict = "Uncertain / Insufficient evidence"
        evidence_status = "Conflicting clinical evidence found among top sources"
        risk = "Medium"
    elif top_score >= HIGH_SIMILARITY_THRESHOLD:
        if top_stance == "refutes":
            verdict = "Refuted"
            evidence_status = "Directly refuted by reputable clinical evidence"
            # Check for dangerous medical advice
            claim_lower = claim.lower()
            is_dangerous = any(pat in claim_lower for pat in DANGEROUS_PATTERNS)
            risk = "Very High" if is_dangerous else "High"
        elif top_stance == "supports":
            verdict = "Supported"
            evidence_status = "Corroborated by verified medical sources"
            risk = "Low"
        else:
            verdict = "Uncertain / Insufficient evidence"
            evidence_status = "Evidence stance inconclusive"
            risk = "Medium"
    else:
        # Moderate similarity
        verdict = "Uncertain / Insufficient evidence"
        evidence_status = "Moderate similarity; insufficient clinical evidence"
        risk = "Medium"

    return {
        "verdict": verdict,
        "evidence_status": evidence_status,
        "risk": risk,
        "confidence": top_score,
        "stance": top_stance
    }


def generate_explanation(
    original_claim: str,
    detected_language: str,
    is_health: bool,
    verdict_data: Dict[str, Any],
    retrieved_evidence: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generates standardized educational explanation following strict compliance templates.
    """
    verdict = verdict_data["verdict"]
    risk = verdict_data["risk"]
    evidence_status = verdict_data["evidence_status"]

    if not is_health:
        summary_text = (
            "The claim was classified as NOT health-related. "
            "Our medical verification engine only processes health, medical, or clinical statements. "
            "No clinical retrieval was performed."
        )
        return {
            "original_claim": original_claim,
            "detected_language": detected_language,
            "is_health_related": False,
            "verdict": "Non-Health Claim",
            "evidence_status": "Not Applicable",
            "risk": "None",
            "summary_text": summary_text,
            "retrieved_evidence": [],
            "disclaimer": "This tool is for educational purposes only and does not constitute medical advice."
        }

    # Stance phrasing in standard template
    if verdict == "Supported":
        action_phrase = "supports"
    elif verdict == "Refuted":
        action_phrase = "refutes"
    else:
        action_phrase = "does not sufficiently address"

    summary_text = (
        f"The claim was classified as **{verdict}** based on the most relevant retrieved medical evidence. "
        f"The evidence **{action_phrase}** the claim.\n\n"
        f"**Risk Level**: `{risk}`\n"
        f"**Evidence Status**: {evidence_status}\n\n"
        f"⚠️ *This tool is for educational purposes and is not medical advice.*"
    )

    return {
        "original_claim": original_claim,
        "detected_language": detected_language,
        "is_health_related": True,
        "verdict": verdict,
        "evidence_status": evidence_status,
        "risk": risk,
        "summary_text": summary_text,
        "retrieved_evidence": retrieved_evidence,
        "disclaimer": "This tool is for educational purposes and is not medical advice. Consult a certified medical professional for diagnosis and treatment."
    }
