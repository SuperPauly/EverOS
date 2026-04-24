from __future__ import annotations

from api_specs.memory_models import MemoryType
from infra_layer.adapters.out.search.leann.config import get_leann_settings
from infra_layer.adapters.out.search.repository.atomic_fact_leann_repository import (
    AtomicFactLeannRepository,
)
from infra_layer.adapters.out.search.repository.episodic_memory_leann_repository import (
    EpisodicMemoryLeannRepository,
)
from infra_layer.adapters.out.search.repository.foresight_leann_repository import (
    ForesightLeannRepository,
)

LEANN_KEYWORD_REPOS = {
    MemoryType.FORESIGHT: ForesightLeannRepository,
    MemoryType.ATOMIC_FACT: AtomicFactLeannRepository,
    MemoryType.EPISODIC_MEMORY: EpisodicMemoryLeannRepository,
}

LEANN_VECTOR_REPOS = {
    MemoryType.FORESIGHT: ForesightLeannRepository,
    MemoryType.ATOMIC_FACT: AtomicFactLeannRepository,
    MemoryType.EPISODIC_MEMORY: EpisodicMemoryLeannRepository,
}


def leann_backend_enabled() -> bool:
    return get_leann_settings().enabled


def get_keyword_repository_class(memory_type: MemoryType):
    if not leann_backend_enabled():
        return None
    return LEANN_KEYWORD_REPOS.get(memory_type)


def get_vector_repository_class(memory_type: MemoryType):
    if not leann_backend_enabled():
        return None
    return LEANN_VECTOR_REPOS.get(memory_type)
