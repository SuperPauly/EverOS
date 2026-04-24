# EverMemOS Agent Notes

## Scope

- Main app code lives in `methods/evermemos/src/`
- Tests live in `methods/evermemos/tests/`
- Docs live in `methods/evermemos/docs/`

## Useful Commands

```bash
cd methods/evermemos
docker-compose up -d
python -m uv sync --dev
PYTHONPATH=src python -m uv run pytest tests/
python -m uv run black src/ tests/
```

## Architecture Reminders

- All application I/O is async-first.
- MongoDB stores raw memory records.
- Elasticsearch handles keyword retrieval by default.
- Milvus handles vector retrieval by default.
- Tenant isolation is enforced through `core/tenants/` and per-backend storage routing.

## LEANN Backend

- Enable with `MEMORY_SEARCH_BACKEND=leann`.
- LEANN indexes live under `LEANN_STORAGE_PATH/<tenant_id>/`.
- Current LEANN integration covers episodic memories, atomic facts, and foresights.
- EverMemOS writes precomputed vectors into LEANN, so no extra embedding worker container is required for the default setup.
- Published LEANN packages currently expose `hnsw` and `diskann`; the EverMemOS adapter defaults to `hnsw` and rebuilds indexes on mutation to keep update/delete behavior correct.
