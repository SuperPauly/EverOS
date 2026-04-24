# LEANN Memory Backend Design

## Summary

EverMemOS now supports LEANN as an optional local search backend for episodic memories, atomic facts, and foresights. The integration preserves the existing MongoDB raw-memory flow and swaps the secondary search index layer from Elasticsearch + Milvus to LEANN when `MEMORY_SEARCH_BACKEND=leann`.

## Integration Points

- `infra_layer/adapters/out/search/leann/adapter.py`
  - Async wrapper around synchronous LEANN index rebuild/search operations.
  - Owns tenant-aware on-disk routing, manifest persistence, and vector/keyword search execution.
- `infra_layer/adapters/out/search/repository/*_leann_repository.py`
  - EverMemOS-shaped repositories for episodic, atomic fact, and foresight search.
- `infra_layer/adapters/out/search/repository/backend_selector.py`
  - Runtime switch for LEANN-backed repositories.
- `biz_layer/mem_memorize.py`
  - Writes episodic memories into LEANN instead of Elasticsearch + Milvus when enabled.
- `biz_layer/mem_sync.py`
  - Synchronizes atomic facts and foresights into LEANN when enabled.
- `service/memcell_delete_service.py`
  - Deletes LEANN-backed search records alongside MongoDB source records.
- `agentic_layer/memory_manager.py` and `agentic_layer/search_mem_service.py`
  - Read-path selection for keyword/vector retrieval.

## Data Mapping

Raw MongoDB memory documents remain the source of truth. LEANN stores a tenant-scoped sidecar manifest plus a local LEANN index:

- Manifest: full searchable record + precomputed EverMemOS vector.
- LEANN passages: concatenated searchable text fields and a lightweight `document` payload for result mapping.
- Index: HNSW graph built from the stored vectors.

Per memory type:

- Episodic memory text: `episode`, `title`, `summary`, `subject`, `search_content`
- Atomic fact text: `atomic_fact`, `search_content`
- Foresight text: `content`, `evidence`, `search_content`

## Tenant Isolation

LEANN storage uses the same tenant identity source as the rest of EverMemOS:

- Current tenant context from `core.tenants.tenant_contextvar.get_current_tenant_id()`
- Fallback base prefix from `core.tenants.tenant_constants.get_base_resource_prefix()`

Effective layout:

```text
LEANN_STORAGE_PATH/
  <tenant_id>/
    episodic_memory.leann
    episodic_memory.manifest.json
    atomic_fact.leann
    foresight.leann
```

This keeps each tenant in a separate filesystem namespace and avoids cross-tenant index reuse.

## Async / Sync Boundary

LEANN’s build/search operations are synchronous. The adapter uses `asyncio.to_thread(...)` for:

- manifest reads/writes,
- full index rebuilds,
- searcher construction,
- vector search execution.

That keeps FastAPI and agent workflows off the blocking path.

## Index / Update Strategy

### Chosen strategy

- Backend: `hnsw`
- Input vectors: precomputed EverMemOS vectors
- Mutation handling: rebuild-on-write

### Why

The published LEANN package (`leann==0.3.7`) currently exposes `hnsw` and `diskann` backends, but not the IVF backend from the reference repository. Because the packaged HNSW path does not offer safe in-place delete/update semantics for EverMemOS’s mutation-heavy lifecycle, the adapter rebuilds the tenant-local index after each change. This keeps behavior correct while still using LEANN as the retrieval engine.

### Tradeoff

- Pros: no extra embedding worker service, deterministic tests, direct reuse of EverMemOS vectors.
- Cons: update/delete operations are more expensive than the IVF design from the reference repository.

## Operational Notes

- No extra Docker service is required for the default LEANN integration.
- LEANN uses EverMemOS precomputed vectors instead of recomputing embeddings at search time.
- `LEANN_STORAGE_PATH` should point to persistent storage in production if index durability matters.

## Risks / Follow-ups

- If packaged LEANN publishes IVF support, the adapter can switch from rebuild-on-write to incremental update/delete.
- Agent case / agent skill search still uses the existing Elasticsearch + Milvus path.
- The adapter currently favors correctness and isolation over large-scale mutation throughput.
