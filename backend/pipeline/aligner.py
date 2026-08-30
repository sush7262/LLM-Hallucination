"""
Stage 2 — Retrieval Alignment

Chunks raw passages into small windows (~2 sentences each), embeds them with
a SentenceTransformer model, and retrieves the top-k most relevant chunks for
each claim via cosine similarity.

See PROJECT_SPEC.md Section 3, Stage 2 for the validated logic.
"""

import re
import logging
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


def chunk_context(context: str, sentences_per_chunk: int = 2) -> list[str]:
    """
    Split a long passage into small overlapping windows of
    *sentences_per_chunk* sentences each.

    This is CRITICAL — without chunking, long passages get truncated by the
    downstream NLI model's 256-token limit, silently losing evidence and
    causing a systematic bias toward 'unsupported' verdicts.

    Validated function from PROJECT_SPEC.md.
    """
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', context.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 3]

    chunks: list[str] = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i : i + sentences_per_chunk])
        if chunk:
            chunks.append(chunk)

    # Fallback: if splitting produced nothing, use first 1000 chars
    return chunks if chunks else [context[:1000]]


class RetrievalAligner:
    """
    Embeds claims and passage chunks with SentenceTransformer,
    then ranks chunks by cosine similarity to retrieve top-k.
    """

    def __init__(self, model_name: str, device: str):
        logger.info(f"Loading embedding model '{model_name}' on {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        logger.info("Embedding model loaded.")

    def align(
        self,
        claim: str,
        passages: list[str],
        top_k: int = 2,
        sentences_per_chunk: int = 1,
    ) -> str:
        """
        Return the most relevant passage chunks for *claim*,
        filtered by similarity score to eliminate low-relevance noise.
        """
        # Chunk all passages
        all_chunks: list[str] = []
        for passage in passages:
            all_chunks.extend(chunk_context(passage, sentences_per_chunk))

        if not all_chunks:
            return " ".join(passages)  # fallback

        # Embed claim and chunks
        claim_embedding = self.model.encode(claim, convert_to_tensor=True)
        chunk_embeddings = self.model.encode(all_chunks, convert_to_tensor=True)

        # Cosine similarity
        similarities = util.cos_sim(claim_embedding, chunk_embeddings)[0]

        # Top indices sorted descending (highest similarity first)
        top_indices = similarities.argsort(descending=True)
        top_score = similarities[top_indices[0]].item()

        # Keep highest scoring chunk, plus any secondary top-k chunk that meets relevance threshold
        selected_chunks = [all_chunks[top_indices[0]]]
        for idx_tensor in top_indices[1:min(top_k, len(all_chunks))]:
            idx = idx_tensor.item()
            score = similarities[idx].item()
            # Only include secondary chunk if its similarity is reasonably high relative to top chunk
            if score >= max(0.35, top_score * 0.65):
                selected_chunks.append(all_chunks[idx])

        return " ".join(selected_chunks)
