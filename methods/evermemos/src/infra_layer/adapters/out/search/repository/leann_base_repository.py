from __future__ import annotations

from typing import Any

from infra_layer.adapters.out.search.leann import (
    LeannIndexConfig,
    LeannIndexRecord,
    LeannMemoryAdapter,
)


class BaseLeannRepository:
    def __init__(self, config: LeannIndexConfig):
        self.adapter = LeannMemoryAdapter(config=config)

    async def get_by_id(self, entity_id: str) -> dict[str, Any] | None:
        return await self.adapter.get(entity_id)

    async def delete_by_id(self, entity_id: str) -> bool:
        return await self.adapter.delete(entity_id)

    async def upsert_record(
        self, *, record_id: str, vector: list[float], document: dict[str, Any]
    ) -> dict[str, Any]:
        return await self.adapter.upsert(
            LeannIndexRecord(id=record_id, vector=vector, document=document)
        )
