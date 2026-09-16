"""
Health Claim Classifier Module
Rule-based / keyword-driven domain classifier to ensure only health-related
queries proceed to the RAG retrieval pipeline.

Note: In future production iterations, this rule-based classifier can be replaced
with a fine-tuned Clinical BioBERT or MuRIL text classifier.
"""

import re

# Comprehensive multi-lingual health terms (English, Hinglish, and Devanagari Hindi)
HEALTH_KEYWORDS = {
    # Core English keywords specified in requirements
    "cancer", "vaccine", "vaccines", "vaccination", "medicine", "medicines",
    "doctor", "doctors", "hospital", "hospitals", "disease", "diseases",
    "diabetes", "fever", "antibiotic", "antibiotics", "covid", "covid-19",
    "health", "treatment", "cure", "cures", "curing", "virus", "viral",
    "cough", "infection", "infections", "pregnancy", "pregnant", "blood", "heart",
    # Additional common clinical / lifestyle terms
    "smoking", "tobacco", "cigarette", "lungs", "lung", "hypertension",
    "stroke", "tumor", "chemo", "chemotherapy", "dosage", "drug", "drugs",
    "pain", "autism", "immune", "immunity", "handwash", "handwashing", "soap",
    "hygiene", "sanitize", "sanitizer",
    # Hinglish terms
    "bimari", "bimaari", "ilaj", "ilaaj", "dawa", "dawai", "haldi", "doodh",
    "dard", "sehat", "aspatal", "khansi", "bukhar", "sugar", "chot",
    # Devanagari Hindi terms
    "कैंसर", "वैक्सीन", "टीका", "टीकाकरण", "दवा", "दवाई", "डॉक्टर", "अस्पताल",
    "बीमारी", "मधुमेह", "डायबिटीज", "बुखार", "एंटीबायोटिक", "कोविड", "स्वास्थ्य",
    "इलाज", "उपचार", "वायरस", "खांसी", "संक्रमण", "गर्भावस्था", "रक्त", "दिल"
}


def is_health_related(text: str) -> bool:
    """
    Evaluates whether the input claim is related to health or medicine.
    Returns True if at least one health keyword is present, False otherwise.
    """
    if not text or not isinstance(text, str):
        return False

    # Extract alphanumeric and Devanagari tokens
    tokens = set(re.findall(r"[\w\u0900-\u097F]+", text.lower()))

    # Check for exact token matches
    matched_keywords = tokens.intersection(HEALTH_KEYWORDS)
    if matched_keywords:
        return True

    # Check for substring matches for compound words (e.g. 'antibiotic-resistant', 'handwashing')
    text_lower = text.lower()
    for kw in HEALTH_KEYWORDS:
        if kw in text_lower:
            return True

    return False
