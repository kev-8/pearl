"""Shared Cohere embedding helpers with lazy client initialization."""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache

import cohere


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _cohere_client():
    return cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))


def generate_text_embeddings(body, model_id="embed-v4.0", region_name="us-east-1"):
    """Generate Cohere embeddings from the repository's legacy byte payload."""
    del region_name  # retained for compatibility with the former Bedrock helper
    body_dict = json.loads(body)
    response = _cohere_client().embed(
        texts=body_dict.get("texts", []),
        model=model_id,
        input_type=body_dict.get("input_type", "search_document"),
        embedding_types=["float"],
    )
    return {"embeddings": response.embeddings.float_}


def normalize_embeddings(embeddings, expected_n=None):
    """Normalize supported embedding response shapes to a list of vectors."""
    if isinstance(embeddings, dict):
        try:
            items = sorted(embeddings.items(), key=lambda item: int(item[0]))
            flat = [value for _, value in items]
        except Exception:
            flat = list(embeddings.values())
    elif isinstance(embeddings, list):
        flat = embeddings
    else:
        raise ValueError(f"Unexpected embeddings type: {type(embeddings)}")

    if expected_n is not None and len(flat) != expected_n:
        logger.warning(
            "embeddings length != expected_n: %d != %s", len(flat), expected_n
        )
    return flat
