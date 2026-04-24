from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LeannSettings:
    enabled: bool
    storage_path: Path
    backend_name: str
    complexity: int
    beam_width: int
    candidate_multiplier: int
    distance_metric: str
    use_precomputed_vectors: bool = True
    embedding_model: str = "text-embedding-3-small"
    embedding_mode: str = "openai"
    embedding_options: dict[str, Any] = field(default_factory=dict)


@lru_cache(maxsize=1)
def get_leann_settings() -> LeannSettings:
    backend_name = os.getenv("LEANN_BACKEND_NAME", "hnsw").strip().lower() or "hnsw"
    storage_path = Path(
        os.getenv("LEANN_STORAGE_PATH", "data/leann_indexes").strip()
        or "data/leann_indexes"
    )
    embedding_options: dict[str, Any] = {}
    vectorize_base_url = os.getenv("VECTORIZE_BASE_URL", "").strip()
    vectorize_api_key = os.getenv("VECTORIZE_API_KEY", "").strip()
    if vectorize_base_url:
        embedding_options["base_url"] = vectorize_base_url
    if vectorize_api_key:
        embedding_options["api_key"] = vectorize_api_key

    return LeannSettings(
        enabled=os.getenv("MEMORY_SEARCH_BACKEND", "").strip().lower() == "leann",
        storage_path=storage_path,
        backend_name=backend_name,
        complexity=max(8, int(os.getenv("LEANN_SEARCH_COMPLEXITY", "64"))),
        beam_width=max(1, int(os.getenv("LEANN_SEARCH_BEAM_WIDTH", "1"))),
        candidate_multiplier=max(
            1, int(os.getenv("LEANN_SEARCH_CANDIDATE_MULTIPLIER", "5"))
        ),
        distance_metric=os.getenv("LEANN_DISTANCE_METRIC", "cosine").strip().lower()
        or "cosine",
        use_precomputed_vectors=_to_bool(
            os.getenv("LEANN_USE_PRECOMPUTED_VECTORS"), True
        ),
        embedding_model=os.getenv("VECTORIZE_MODEL", "text-embedding-3-small").strip()
        or "text-embedding-3-small",
        embedding_mode=os.getenv("LEANN_EMBEDDING_MODE", "openai").strip().lower()
        or "openai",
        embedding_options=embedding_options,
    )
