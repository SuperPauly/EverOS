from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from core.di.decorators import repository
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.search.leann import LeannIndexConfig
from infra_layer.adapters.out.search.repository.leann_base_repository import (
    BaseLeannRepository,
)


def _foresight_result(document: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "id": document.get("id"),
        "score": float(score),
        "user_id": document.get("user_id", ""),
        "group_id": document.get("group_id", ""),
        "participants": document.get("participants") or [],
        "sender_ids": document.get("sender_ids") or [],
        "parent_type": document.get("parent_type", ""),
        "parent_id": document.get("parent_id", ""),
        "start_time": document.get("start_time"),
        "end_time": document.get("end_time"),
        "duration_days": document.get("duration_days", 0),
        "content": document.get("content", ""),
        "evidence": document.get("evidence", ""),
        "type": document.get("type", ""),
        "search_content": document.get("search_content") or [],
        "timestamp": document.get("start_time"),
    }


@repository("foresight_leann_repository", primary=False)
class ForesightLeannRepository(BaseLeannRepository):
    def __init__(self):
        super().__init__(
            LeannIndexConfig(
                collection_name="foresight",
                text_fields=("content", "evidence", "search_content"),
                default_result_mapper=_foresight_result,
            )
        )

    async def create_and_save_foresight_mem(
        self,
        id: str,
        user_id: Optional[str],
        content: str,
        parent_id: str,
        parent_type: str,
        vector: List[float],
        group_id: Optional[str] = None,
        event_type: Optional[str] = None,
        participants: Optional[List[str]] = None,
        sender_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        duration_days: Optional[int] = None,
        evidence: Optional[str] = None,
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
            "start_time": int(start_time.timestamp() * 1000) if start_time else 0,
            "end_time": int(end_time.timestamp() * 1000) if end_time else 0,
            "duration_days": duration_days or 0,
            "content": content,
            "evidence": evidence or "",
            "type": event_type or "",
            "search_content": list(search_content or [content, evidence or ""]),
        }
        return await self.upsert_record(record_id=id, vector=vector, document=document)

    async def create_and_save_foresight(
        self,
        id: str,
        user_id: str,
        content: str,
        search_content: List[str],
        parent_id: str,
        parent_type: str,
        event_type: Optional[str] = None,
        group_id: Optional[str] = None,
        participants: Optional[List[str]] = None,
        sender_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        duration_days: Optional[int] = None,
        evidence: Optional[str] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        return await self.create_and_save_foresight_mem(
            id=id,
            user_id=user_id,
            content=content,
            parent_id=parent_id,
            parent_type=parent_type,
            vector=[],
            group_id=group_id,
            event_type=event_type,
            participants=participants,
            sender_ids=sender_ids,
            start_time=start_time,
            end_time=end_time,
            duration_days=duration_days,
            evidence=evidence,
            search_content=search_content,
        )

    async def multi_search(
        self,
        query: List[str],
        user_id: Optional[str] = None,
        group_ids: Optional[List[str]] = None,
        sender_id: Optional[str] = None,
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
            "sender_id": sender_id,
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
        sender_id: Optional[str] = None,
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
                "sender_id": sender_id,
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
