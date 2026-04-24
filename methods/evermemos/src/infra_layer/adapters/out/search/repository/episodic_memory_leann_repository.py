from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from core.di.decorators import repository
from core.oxm.constants import MAGIC_ALL
from infra_layer.adapters.out.search.leann import LeannIndexConfig
from infra_layer.adapters.out.search.repository.leann_base_repository import (
    BaseLeannRepository,
)


def _episodic_result(document: dict[str, Any], score: float) -> dict[str, Any]:
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
        "type": document.get("type", ""),
        "episode": document.get("episode", ""),
        "search_content": document.get("search_content") or [],
    }


@repository("episodic_memory_leann_repository", primary=False)
class EpisodicMemoryLeannRepository(BaseLeannRepository):
    def __init__(self):
        super().__init__(
            LeannIndexConfig(
                collection_name="episodic_memory",
                text_fields=(
                    "episode",
                    "title",
                    "summary",
                    "subject",
                    "search_content",
                ),
                default_result_mapper=_episodic_result,
            )
        )

    async def create_and_save_episodic_memory(
        self,
        id: str,
        user_id: str,
        timestamp: datetime,
        episode: str,
        search_content: List[str],
        vector: List[float],
        title: Optional[str] = None,
        summary: Optional[str] = None,
        group_id: Optional[str] = None,
        participants: Optional[List[str]] = None,
        sender_ids: Optional[List[str]] = None,
        event_type: Optional[str] = None,
        subject: Optional[str] = None,
        parent_type: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        document = {
            "id": id,
            "user_id": user_id or "",
            "timestamp": int(timestamp.timestamp() * 1000),
            "episode": episode,
            "title": title or "",
            "summary": summary or "",
            "group_id": group_id or "",
            "participants": participants or [],
            "sender_ids": sender_ids or [],
            "type": event_type or "",
            "subject": subject or "",
            "parent_type": parent_type or "",
            "parent_id": parent_id or "",
            "search_content": search_content or [],
        }
        return await self.upsert_record(record_id=id, vector=vector, document=document)

    async def append_episodic_memory(self, document: Any) -> Dict[str, Any]:
        return await self.create_and_save_episodic_memory(
            id=str(document.id),
            user_id=document.user_id or "",
            timestamp=document.timestamp,
            episode=document.episode,
            search_content=list(document.search_content or []),
            vector=list(document.vector or []),
            title=getattr(document, "title", "") or "",
            summary=getattr(document, "summary", "") or "",
            group_id=document.group_id or "",
            participants=list(document.participants or []),
            sender_ids=list(document.sender_ids or []),
            event_type=getattr(document, "type", "") or "",
            subject=getattr(document, "subject", "") or "",
            parent_type=getattr(document, "parent_type", "") or "",
            parent_id=getattr(document, "parent_id", "") or "",
        )

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
        filters = {
            "user_id": user_id,
            "group_ids": group_ids,
            "parent_type": parent_type,
            "parent_id": parent_id,
            "start_time": start_time,
            "end_time": end_time,
        }
        results = await self.adapter.vector_search(
            query_vector,
            limit=limit,
            filters=filters,
            score_threshold=max(score_threshold, radius or 0.0),
        )
        return results

    async def delete_by_event_id(self, event_id: str) -> bool:
        return await self.delete_by_id(event_id)

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
