"""
Application configuration — loads settings from environment / .env file.
All model names, API keys, and tuning parameters are centralized here.
"""

import torch
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Central configuration for the Hallucination Verification Module.
    Values are loaded from environment variables or a .env file.
    """

    # ── Gemini (LLM decomposition) ──────────────────────────────────────
    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API key. Required only for LLM decomposition mode.",
    )
    gemini_model: str = Field(
        default="gemini-3.5-flash-lite",
        description="Gemini model name. Changes frequently — update when retired.",
    )

    # ── Embedding model (Stage 2 — Retrieval Alignment) ─────────────────
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="SentenceTransformer model for claim-passage alignment.",
    )

    # ── NLI model (Stage 3 — Entailment Classification) ─────────────────
    nli_model: str = Field(
        default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        description="HuggingFace NLI model for entailment classification.",
    )

    # ── Pipeline tuning ─────────────────────────────────────────────────
    top_k: int = Field(
        default=2,
        description="Number of top aligned passages to use per claim.",
    )
    sentences_per_chunk: int = Field(
        default=2,
        description="Sentences per chunk when splitting long passages.",
    )
    nli_max_length: int = Field(
        default=256,
        description="Max token length for the NLI model tokenizer.",
    )

    # ── Device ──────────────────────────────────────────────────────────
    device: str = Field(
        default="auto",
        description="Compute device: 'auto' (detect), 'cuda', or 'cpu'.",
    )

    @property
    def resolved_device(self) -> str:
        """Return the actual device string after auto-detection."""
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


# Singleton instance — import this throughout the app
settings = Settings()
