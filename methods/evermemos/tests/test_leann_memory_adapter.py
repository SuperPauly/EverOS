from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from core.tenants.tenant_contextvar import clear_current_tenant, set_current_tenant
from infra_layer.adapters.out.search.leann.config import get_leann_settings
from infra_layer.adapters.out.search.repository.atomic_fact_leann_repository import (
    AtomicFactLeannRepository,
)
from infra_layer.adapters.out.search.repository.backend_selector import (
    get_keyword_repository_class,
    get_vector_repository_class,
)
from infra_layer.adapters.out.search.repository.episodic_memory_leann_repository import (
    EpisodicMemoryLeannRepository,
)
from infra_layer.adapters.out.search.repository.foresight_leann_repository import (
    ForesightLeannRepository,
)


@pytest.fixture(autouse=True)
def _leann_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_SEARCH_BACKEND", "leann")
    monkeypatch.setenv("LEANN_STORAGE_PATH", str(tmp_path))
    monkeypatch.setenv("LEANN_BACKEND_NAME", "hnsw")
    get_leann_settings.cache_clear()
    clear_current_tenant()
    yield tmp_path
    clear_current_tenant()
    get_leann_settings.cache_clear()


def _now() -> datetime:
    return datetime(2026, 4, 24, 6, 30, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_selector_uses_leann_repositories():
    from api_specs.memory_models import MemoryType

    assert (
        get_keyword_repository_class(MemoryType.EPISODIC_MEMORY)
        is EpisodicMemoryLeannRepository
    )
    assert (
        get_vector_repository_class(MemoryType.ATOMIC_FACT) is AtomicFactLeannRepository
    )
    assert get_vector_repository_class(MemoryType.FORESIGHT) is ForesightLeannRepository


@pytest.mark.asyncio
async def test_tenant_specific_storage_paths_are_isolated():
    repo = EpisodicMemoryLeannRepository()

    set_current_tenant(SimpleNamespace(tenant_id="tenant_alpha"))
    alpha_path = await repo.adapter.tenant_storage_path()

    set_current_tenant(SimpleNamespace(tenant_id="tenant_beta"))
    beta_path = await repo.adapter.tenant_storage_path()

    assert alpha_path != beta_path
    assert alpha_path.name == "tenant_alpha"
    assert beta_path.name == "tenant_beta"


@pytest.mark.asyncio
async def test_async_thread_offloading_is_used(monkeypatch):
    calls: list[str] = []

    async def fake_to_thread(func, *args, **kwargs):
        calls.append(getattr(func, "__name__", repr(func)))
        return func(*args, **kwargs)

    from infra_layer.adapters.out.search.leann import adapter as leann_adapter_module

    monkeypatch.setattr(leann_adapter_module.asyncio, "to_thread", fake_to_thread)

    repo = EpisodicMemoryLeannRepository()
    await repo.create_and_save_episodic_memory(
        id="evt-async",
        user_id="u1",
        timestamp=_now(),
        episode="Async adapter test memory",
        search_content=["async", "adapter"],
        vector=[1.0, 0.0],
    )
    await repo.vector_search(query_vector=[1.0, 0.0], user_id="u1", limit=1)

    assert "_write_manifest" in calls
    assert "_rebuild_index" in calls
    assert "_create_searcher" in calls


@pytest.mark.asyncio
async def test_invalid_backend_config_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("LEANN_BACKEND_NAME", "unknown-backend")
    get_leann_settings.cache_clear()

    repo = EpisodicMemoryLeannRepository()
    with pytest.raises(ValueError, match="Backend 'unknown-backend'"):
        await repo.create_and_save_episodic_memory(
            id="evt-invalid",
            user_id="u1",
            timestamp=_now(),
            episode="Invalid backend",
            search_content=["invalid"],
            vector=[1.0, 0.0],
        )


@pytest.mark.asyncio
async def test_episodic_leann_crud_search_update_delete():
    repo = EpisodicMemoryLeannRepository()

    await repo.create_and_save_episodic_memory(
        id="evt-1",
        user_id="u1",
        timestamp=_now(),
        episode="Alice discussed pasta recipes",
        search_content=["alice", "pasta", "recipes"],
        vector=[1.0, 0.0],
        group_id="g1",
    )
    await repo.create_and_save_episodic_memory(
        id="evt-2",
        user_id="u1",
        timestamp=_now(),
        episode="Bob planned a mountain trip",
        search_content=["bob", "mountain", "trip"],
        vector=[0.0, 1.0],
        group_id="g1",
    )

    stored = await repo.get_by_id("evt-1")
    assert stored is not None
    assert stored["episode"] == "Alice discussed pasta recipes"

    keyword_hits = await repo.multi_search(query=["pasta"], user_id="u1", size=5)
    assert [hit["_id"] for hit in keyword_hits] == ["evt-1"]

    vector_hits = await repo.vector_search(
        query_vector=[1.0, 0.0], user_id="u1", limit=2
    )
    assert vector_hits[0]["id"] == "evt-1"

    await repo.create_and_save_episodic_memory(
        id="evt-1",
        user_id="u1",
        timestamp=_now(),
        episode="Alice switched to sushi plans",
        search_content=["alice", "sushi", "plans"],
        vector=[0.0, 1.0],
        group_id="g1",
    )

    updated = await repo.get_by_id("evt-1")
    assert updated["episode"] == "Alice switched to sushi plans"
    updated_keyword_hits = await repo.multi_search(
        query=["sushi"], user_id="u1", size=5
    )
    assert [hit["_id"] for hit in updated_keyword_hits] == ["evt-1"]

    deleted = await repo.delete_by_event_id("evt-2")
    assert deleted is True
    assert await repo.get_by_id("evt-2") is None

    deleted_count = await repo.delete_by_filters(user_id="u1", group_id="g1")
    assert deleted_count == 1
    assert await repo.get_by_id("evt-1") is None
