from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from core.di.decorators import repository
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.search.leann import LeannIndexConfig
from infra_layer.adapters.out.search.repository.leann_base_repository import (
    BaseLeannRepository,
)


def _atomic_fact_result(document: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "id": document.get("id"),
        "score": float(score),
        "user_id": document.get("user_id", ""),
        "group_id": document.get("group_id", ""),
        "session_id": document.get("session_id", ""),
        "participants": document.get("participants") or [],
        "timestamp": document.get("timestamp"),
        "parent_type": document.get("parent_type", ""),
        "parent_id": document.get("parent_id", ""),
        "atomic_fact": document.get("atomic_fact", ""),
        "search_content": document.get("search_content") or [],
    }


@repository("atomic_fact_leann_repository", primary=False)
class AtomicFactLeannRepository(BaseLeannRepository):
    def __init__(self):
        super().__init__(
            LeannIndexConfig(
                collection_name="atomic_fact",
                text_fields=("atomic_fact", "search_content"),
                default_result_mapper=_atomic_fact_result,
            )
        )

    async def create_and_save_atomic_fact(
        self,
        id: str,
        user_id: Optional[str],
        atomic_fact: str,
        parent_id: str,
        parent_type: str,
        timestamp: datetime,
        vector: List[float],
        group_id: Optional[str] = None,
        participants: Optional[List[str]] = None,
        sender_ids: Optional[List[str]] = None,
        event_type: Optional[str] = None,
        search_content: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        document = {
            "id": id,
            "user_id": user_id or "",
            "group_id": group_id or "",
            "participants": participants or [],
            "sender_ids": sender_ids or [],
            "parent_type": parent_type or "",
            "parent_id": parent_id or "",
            "type": event_type or "",
            "timestamp": int(timestamp.timestamp() * 1000),
            "atomic_fact": atomic_fact,
            "search_content": list(search_content or [atomic_fact]),
        }
        return await self.upsert_record(record_id=id, vector=vector, document=document)

    async def multi_search(
        self,
        query: List[str],
        user_id: Optional[str] = None,
        group_ids: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        parent_type: Optional[str] = None,
        parent_id: Optional[str] = None,
        date_range: Optional[Dict[str, Any]] = None,
        size: int = 10,
        from_: int = 0,
    ) -> List[Dict[str, Any]]:
        filters = {
            "user_id": user_id,
            "group_ids": group_ids,
            "parent_type": parent_type,
            "parent_id": parent_id,
            "start_time": (date_range or {}).get("gte"),
            "end_time": (date_range or {}).get("lte"),
        }
        results = await self.adapter.keyword_search(
            query, limit=size + from_, filters=filters
        )
        sliced = results[from_ : from_ + size]
        return [
            {
                "_id": item["id"],
                "_score": item["score"],
                "_source": {k: v for k, v in item.items() if k not in {"id", "score"}},
            }
            for item in sliced
        ]

    async def vector_search(
        self,
        query_vector: List[float],
        user_id: Optional[str] = None,
        group_ids: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        parent_type: Optional[str] = None,
        parent_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        radius: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        return await self.adapter.vector_search(
            query_vector,
            limit=limit,
            filters={
                "user_id": user_id,
                "group_ids": group_ids,
                "parent_type": parent_type,
                "parent_id": parent_id,
                "start_time": start_time,
                "end_time": end_time,
            },
            score_threshold=max(score_threshold, radius or 0.0),
        )

    async def delete_by_filters(
        self,
        user_id: Optional[str] = MAGIC_ALL,
        group_id: Optional[str] = MAGIC_ALL,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        return await self.adapter.delete_by_filters(
            user_id=user_id, group_id=group_id, start_time=start_time, end_time=end_time
        )
