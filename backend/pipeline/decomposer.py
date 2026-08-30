"""
Stage 1 — Claim Decomposition

Two strategies:
  • rule_based  — regex sentence splitting (fast, free, always available)
  • llm         — Google Gemini via google-genai SDK (better quality, needs API key)

See PROJECT_SPEC.md Section 3, Stage 1 for the validated logic.
"""

import re
import logging
import time

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Rule-Based Decomposition
# ═══════════════════════════════════════════════════════════════════════

def _extract_subject(sentence: str) -> str:
    """Extract leading subject/noun phrase from a sentence (e.g. 'The Eiffel Tower', 'The company')."""
    match = re.match(r'^(?:The|A|An|This|That|These|Those|[A-Z][a-z]+)(?:\s+[A-Z\ba-z]+){0,4}?\s+(?=(?:is|was|were|are|had|has|have|announced|reported|operates|built|completed)\b)', sentence.strip())
    if match:
        return match.group(0).strip()
    # Fallback: take first 2-3 words before verb
    words = sentence.strip().split()
    if len(words) >= 2:
        return " ".join(words[:2])
    return ""


def rule_based_decompose(text: str) -> list[str]:
    """
    Split *text* into atomic claims using regex heuristics with Subject Propagation.
    Ensures each clause fragment is a complete sentence with a subject.
    """
    if not text or not text.strip():
        return [text] if text else []

    # Step 1 — sentence-level split
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())

    claims: list[str] = []
    for sentence in sentences:
        sentence_clean = sentence.strip()
        subject = _extract_subject(sentence_clean)
        parts = re.split(r',?\s+(?:with|and|which|while)\s+|;\s+', sentence_clean)

        for i, part in enumerate(parts):
            cleaned = part.strip()
            if len(cleaned) <= 3:
                continue

            # If this fragment doesn't start with its own subject/capital letter or starts with verb, prepend subject
            starts_with_verb = re.match(r'^(?:is|was|were|are|has|have|had|located|built)\b', cleaned, re.IGNORECASE)
            if i > 0 and subject and starts_with_verb:
                cleaned = f"{subject} {cleaned}"

            # Capitalize first letter and ensure ending punctuation
            cleaned = cleaned[0].upper() + cleaned[1:]
            if not cleaned.endswith(('.', '!', '?')):
                cleaned += '.'

            claims.append(cleaned)

    return claims if claims else [text.strip()]


# ═══════════════════════════════════════════════════════════════════════
#  LLM-Based Decomposition (Google Gemini)
# ═══════════════════════════════════════════════════════════════════════

_DECOMPOSE_PROMPT = """Break the following text into a list of independent, atomic factual claims.

Rules:
- Each claim must be a complete, grammatically correct sentence on its own
  (it must make sense with no other context).
- Each claim should express exactly ONE fact.
- Do not add any information that isn't in the text.
- Output ONLY the claims, one per line, with no numbering, bullets, or
  extra commentary.

Text: "{text}"
"""

# Retry settings
_MAX_RETRIES = 3
_BASE_DELAY = 2.0  # seconds


def llm_decompose(text: str, api_key: str, model_name: str) -> list[str]:
    """
    Use Google Gemini to decompose *text* into atomic claims.

    Handles:
    - 429 RESOURCE_EXHAUSTED / 503 UNAVAILABLE with exponential backoff
    - response.text being None (safety-filter blocks, empty candidates)
    - Falls back to [text] on persistent failure rather than crashing.
    """
    if not api_key:
        logger.warning("No Gemini API key configured — falling back to single claim.")
        return [text.strip()]

    try:
        from google import genai
    except ImportError:
        logger.error("google-genai package not installed. Falling back to single claim.")
        return [text.strip()]

    client = genai.Client(api_key=api_key)
    prompt = _DECOMPOSE_PROMPT.format(text=text)

    last_error = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )

            # Guard against None response text (safety blocks, empty candidates)
            if response.text is None:
                logger.warning(
                    "Gemini returned None text (safety filter or empty candidates). "
                    "Falling back to single claim."
                )
                return [text.strip()]

            # Parse: one claim per line, skip empty lines
            claims = [
                line.strip()
                for line in response.text.strip().split("\n")
                if line.strip() and len(line.strip()) > 3
            ]

            if claims:
                logger.info(f"LLM decomposed text into {len(claims)} claims.")
                return claims
            else:
                logger.warning("LLM returned empty claims list. Falling back.")
                return [text.strip()]

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Retry on rate-limit or transient errors
            if "resource_exhausted" in error_str or "429" in error_str or \
               "unavailable" in error_str or "503" in error_str:
                delay = _BASE_DELAY * (2 ** attempt)
                logger.warning(
                    f"Gemini API error (attempt {attempt + 1}/{_MAX_RETRIES}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                # Non-retryable error
                logger.error(f"Gemini API non-retryable error: {e}. Falling back.")
                return [text.strip()]

    logger.error(
        f"All {_MAX_RETRIES} Gemini API attempts failed. Last error: {last_error}. "
        "Falling back to treating entire answer as single claim."
    )
    return [text.strip()]
