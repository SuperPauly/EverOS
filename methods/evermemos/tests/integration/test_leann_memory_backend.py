from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from core.tenants.tenant_contextvar import clear_current_tenant, set_current_tenant
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


def _ts() -> datetime:
    return datetime(2026, 4, 24, 7, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_SEARCH_BACKEND", "leann")
    monkeypatch.setenv("LEANN_STORAGE_PATH", str(tmp_path))
    monkeypatch.setenv("LEANN_BACKEND_NAME", "hnsw")
    get_leann_settings.cache_clear()
    clear_current_tenant()
    yield
    clear_current_tenant()
    get_leann_settings.cache_clear()


@pytest.mark.asyncio
async def test_realistic_memory_flow_across_memory_types():
    set_current_tenant(SimpleNamespace(tenant_id="tenant_product"))

    episodic_repo = EpisodicMemoryLeannRepository()
    atomic_repo = AtomicFactLeannRepository()
    foresight_repo = ForesightLeannRepository()

    await episodic_repo.create_and_save_episodic_memory(
        id="episode-1",
        user_id="u1",
        timestamp=_ts(),
        episode="The agent learned that Alice likes jasmine tea.",
        search_content=["alice", "jasmine tea"],
        vector=[1.0, 0.0, 0.0],
        group_id="g1",
        parent_id="memcell-1",
        parent_type="memcell",
    )
    await atomic_repo.create_and_save_atomic_fact(
        id="fact-1",
        user_id="u1",
        atomic_fact="Alice likes jasmine tea",
        parent_id="episode-1",
        parent_type="episode",
        timestamp=_ts(),
        vector=[1.0, 0.0, 0.0],
        group_id="g1",
        search_content=["alice", "jasmine tea"],
    )
    await foresight_repo.create_and_save_foresight_mem(
        id="foresight-1",
        user_id="u1",
        content="Offer jasmine tea recommendations in future chats.",
        parent_id="episode-1",
        parent_type="episode",
        vector=[1.0, 0.0, 0.0],
        group_id="g1",
        search_content=["jasmine tea", "recommendations"],
    )

    episode_hits = await episodic_repo.vector_search(
        query_vector=[1.0, 0.0, 0.0], user_id="u1", limit=3
    )
    fact_hits = await atomic_repo.multi_search(query=["jasmine"], user_id="u1", size=3)
    foresight_hits = await foresight_repo.vector_search(
        query_vector=[1.0, 0.0, 0.0], user_id="u1", limit=3
    )

    assert [hit["id"] for hit in episode_hits] == ["episode-1"]
    assert [hit["_id"] for hit in fact_hits] == ["fact-1"]
    assert [hit["id"] for hit in foresight_hits] == ["foresight-1"]

    delete_count = await episodic_repo.delete_by_filters(user_id="u1", group_id="g1")
    assert delete_count == 1
    assert await episodic_repo.get_by_id("episode-1") is None
    assert await atomic_repo.get_by_id("fact-1") is not None


@pytest.mark.asyncio
async def test_concurrent_tenant_operations_stay_isolated():
    async def store_and_fetch(tenant_id: str, record_id: str, vector: list[float]):
        set_current_tenant(SimpleNamespace(tenant_id=tenant_id))
        repo = EpisodicMemoryLeannRepository()
        await repo.create_and_save_episodic_memory(
            id=record_id,
            user_id="u1",
            timestamp=_ts(),
            episode=f"Memory for {tenant_id}",
            search_content=[tenant_id],
            vector=vector,
            group_id="g1",
        )
        hits = await repo.vector_search(query_vector=vector, user_id="u1", limit=3)
        storage_dir = await repo.adapter.tenant_storage_path()
        clear_current_tenant()
        return hits, storage_dir

    (hits_a, dir_a), (hits_b, dir_b) = await asyncio.gather(
        store_and_fetch("tenant_a", "episode-a", [1.0, 0.0]),
        store_and_fetch("tenant_b", "episode-b", [0.0, 1.0]),
    )

    assert [hit["id"] for hit in hits_a] == ["episode-a"]
    assert [hit["id"] for hit in hits_b] == ["episode-b"]
    assert dir_a != dir_b
