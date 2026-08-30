"""
Stage 3 — Entailment Classification

Uses DeBERTa-v3-base-mnli-fever-anli to classify the relationship between
evidence (premise) and a claim (hypothesis) as one of:
    entailment   → supported
    neutral      → unsupported
    contradiction → contradicted

See PROJECT_SPEC.md Section 3, Stage 3 for the validated logic.
"""

import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

# DeBERTa MNLI label mapping — order matches model's output logits
_LABEL_MAP = {
    0: "contradicted",   # contradiction
    1: "unsupported",    # neutral
    2: "supported",      # entailment
}


class EntailmentClassifier:
    """
    Wraps a HuggingFace NLI model for three-way claim verification.
    Model and tokenizer are loaded once and kept in memory.
    """

    def __init__(self, model_name: str, device: str, max_length: int = 256):
        logger.info(f"Loading NLI model '{model_name}' on {device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # CRITICAL: explicitly move model to GPU — this is NOT automatic
        # for plain transformers models (unlike SentenceTransformer).
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        self.max_length = max_length
        logger.info("NLI model loaded and set to eval mode.")

    def classify(self, claim: str, evidence: str) -> tuple[str, float]:
        """
        Classify the entailment relationship.

        Args:
            claim: The hypothesis (the factual claim to verify).
            evidence: The premise (the aligned passage text).

        Returns:
            (label, confidence) where label ∈ {supported, unsupported, contradicted}
            and confidence ∈ [0.0, 1.0].
        """
        # Tokenize: evidence = premise (arg 1), claim = hypothesis (arg 2)
        inputs = self.tokenizer(
            evidence,
            claim,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # CRITICAL: move tokenized inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[0]
        probabilities = torch.softmax(logits, dim=-1)

        p_supported = probabilities[0].item()      # 0: entailment
        p_unsupported = probabilities[1].item()    # 1: neutral
        p_contradicted = probabilities[2].item()   # 2: contradiction

        # Thresholds / argmax selection
        if p_contradicted >= 0.50:
            label = "contradicted"
            confidence = p_contradicted
        elif p_supported >= p_unsupported:
            label = "supported"
            confidence = p_supported
        else:
            label = "unsupported"
            confidence = p_unsupported

        return label, round(confidence, 4)
