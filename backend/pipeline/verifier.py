"""
Pipeline Orchestrator — HallucinationVerifier

Runs all 4 stages in sequence:
  Stage 1: Claim Decomposition (rule-based or LLM)
  Stage 2: Retrieval Alignment
  Stage 3: Entailment Classification
  Stage 4: Aggregation

This class holds references to the loaded ML models (aligner, classifier)
and coordinates a single verification request end-to-end.
"""

import time
import logging

from backend.config import Settings
from backend.models import ClaimResult, VerifyResponse
from backend.pipeline.decomposer import rule_based_decompose, llm_decompose
from backend.pipeline.aligner import RetrievalAligner
from backend.pipeline.classifier import EntailmentClassifier
from backend.pipeline.aggregator import aggregate

logger = logging.getLogger(__name__)


class HallucinationVerifier:
    """
    End-to-end hallucination verification pipeline.

    Models (aligner, classifier) are injected at construction — they should
    be loaded once at app startup, not per-request.
    """

    def __init__(self, aligner: RetrievalAligner, classifier: EntailmentClassifier):
        self.aligner = aligner
        self.classifier = classifier

    def verify(
        self,
        answer: str,
        passages: list[str],
        decomposition_mode: str,
        config: Settings,
    ) -> VerifyResponse:
        """
        Run the full 4-stage verification pipeline.

        Args:
            answer: The RAG-generated answer to verify.
            passages: Raw retrieved passages (will be chunked internally).
            decomposition_mode: 'rule_based' or 'llm'.
            config: App settings (for API keys, model names, top_k, etc.).

        Returns:
            VerifyResponse with per-claim results, decision, and latency.
        """
        start_time = time.time()

        # ── Stage 1: Claim Decomposition ────────────────────────────────
        if decomposition_mode == "llm":
            logger.info("Using LLM-based claim decomposition.")
            claims = llm_decompose(
                text=answer,
                api_key=config.gemini_api_key,
                model_name=config.gemini_model,
            )
        else:
            logger.info("Using rule-based claim decomposition.")
            claims = rule_based_decompose(answer)

        logger.info(f"Decomposed into {len(claims)} claim(s).")

        # ── Stages 2 & 3: Align + Classify each claim ──────────────────
        claim_results: list[ClaimResult] = []

        for i, claim_text in enumerate(claims):
            # Stage 2: Retrieval Alignment
            evidence = self.aligner.align(
                claim=claim_text,
                passages=passages,
                top_k=config.top_k,
                sentences_per_chunk=config.sentences_per_chunk,
            )

            # Stage 3: Entailment Classification
            label, confidence = self.classifier.classify(
                claim=claim_text,
                evidence=evidence,
            )

            logger.info(
                f"  Claim {i + 1}/{len(claims)}: "
                f"'{claim_text[:60]}...' → {label} ({confidence:.3f})"
            )

            claim_results.append(
                ClaimResult(
                    claim=claim_text,
                    label=label,
                    confidence=confidence,
                    evidence=evidence,
                )
            )

        # ── Stage 4: Aggregation ────────────────────────────────────────
        all_labels = [cr.label for cr in claim_results]
        decision = aggregate(all_labels)

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Verification complete: {decision} ({elapsed_ms}ms)")

        return VerifyResponse(
            claims=claim_results,
            decision=decision,
            latency_ms=elapsed_ms,
        )
