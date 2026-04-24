from __future__ import annotations

import asyncio
import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from leann.api import LeannBuilder, LeannSearcher

from core.observation.logger import get_logger
from core.oxm.constants import MAGIC_ALL
from core.tenants.tenant_constants import get_base_resource_prefix
from core.tenants.tenant_contextvar import get_current_tenant_id
from infra_layer.adapters.out.search.leann.config import (
    LeannSettings,
    get_leann_settings,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class LeannIndexConfig:
    collection_name: str
    text_fields: Sequence[str]
    default_result_mapper: Callable[[dict[str, Any], float], dict[str, Any]]


@dataclass(frozen=True)
class LeannIndexRecord:
    id: str
    vector: Sequence[float]
    document: dict[str, Any]


class LeannMemoryAdapter:
    _locks: dict[str, asyncio.Lock] = {}

    def __init__(
        self, config: LeannIndexConfig, settings: LeannSettings | None = None
    ) -> None:
        self.config = config
        self.settings = settings or get_leann_settings()

    async def upsert(self, record: LeannIndexRecord) -> dict[str, Any]:
        if not record.id:
            raise ValueError("record.id is required")
        records = await self._load_records()
        records[record.id] = self._normalize_record(record)
        await self._write_records(records)
        return records[record.id]["document"]

    async def get(self, record_id: str) -> dict[str, Any] | None:
        records = await self._load_records()
        record = records.get(record_id)
        return None if record is None else dict(record["document"])

    async def delete(self, record_id: str) -> bool:
        records = await self._load_records()
        deleted = records.pop(record_id, None) is not None
        if deleted:
            await self._write_records(records)
        return deleted

    async def delete_by_filters(self, **filters: Any) -> int:
        records = await self._load_records()
        matched_ids = [
            record_id
            for record_id, record in records.items()
            if self._matches_filters(record["document"], filters)
        ]
        if not matched_ids:
            return 0
        for record_id in matched_ids:
            records.pop(record_id, None)
        await self._write_records(records)
        return len(matched_ids)

    async def keyword_search(
        self,
        query_terms: Sequence[str],
        *,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        terms = [term.strip().lower() for term in query_terms if term and term.strip()]
        if not terms:
            return []
        records = await self._load_records()
        scored_results: list[tuple[float, dict[str, Any]]] = []
        for record in records.values():
            document = record["document"]
            if not self._matches_filters(document, filters or {}):
                continue
            haystack = record["text"].lower()
            score = float(sum(haystack.count(term) for term in terms))
            if score <= 0:
                continue
            scored_results.append((score, document))

        scored_results.sort(key=lambda item: item[0], reverse=True)
        return [
            self.config.default_result_mapper(document, score)
            for score, document in scored_results[:limit]
        ]

    async def vector_search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int,
        filters: dict[str, Any] | None = None,
        score_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        index_path = self._index_path()
        if not await asyncio.to_thread(
            index_path.with_name(f"{index_path.name}.meta.json").exists
        ):
            return []

        query_np = np.ascontiguousarray([query_vector], dtype=np.float32)
        if self.settings.distance_metric == "cosine":
            norms = np.linalg.norm(query_np, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            query_np = query_np / norms

        overscan = max(limit * self.settings.candidate_multiplier, limit)
        searcher = await asyncio.to_thread(self._create_searcher)
        try:
            raw_results = await asyncio.to_thread(
                searcher.backend_impl.search,
                query_np,
                overscan,
                complexity=self.settings.complexity,
                beam_width=self.settings.beam_width,
                recompute_embeddings=False,
            )
            labels = raw_results.get("labels", [[]])[0]
            distances = raw_results.get("distances", [[]])[0]
            search_hits: list[dict[str, Any]] = []
            for label, score in zip(labels, distances):
                try:
                    passage = searcher.passage_manager.get_passage(str(label))
                except KeyError:
                    passage = self._get_passage_via_id_map(searcher, label)
                    if passage is None:
                        continue
                document = dict(passage.get("metadata", {}).get("document", {}))
                if not document:
                    continue
                score_value = float(score)
                if score_value < score_threshold:
                    continue
                if not self._matches_filters(document, filters or {}):
                    continue
                search_hits.append(
                    self.config.default_result_mapper(document, score_value)
                )
                if len(search_hits) >= limit:
                    break
            return search_hits
        finally:
            await asyncio.to_thread(searcher.cleanup)

    async def tenant_storage_path(self) -> Path:
        return self._tenant_dir()

    def _normalize_record(self, record: LeannIndexRecord) -> dict[str, Any]:
        vector = [float(value) for value in record.vector]
        text = self._build_text(record.document)
        document = dict(record.document)
        document["id"] = record.id
        return {"id": record.id, "text": text, "vector": vector, "document": document}

    async def _load_records(self) -> dict[str, dict[str, Any]]:
        manifest = self._manifest_path()
        if not await asyncio.to_thread(manifest.exists):
            return {}
        return await asyncio.to_thread(self._read_manifest, manifest)

    async def _write_records(self, records: dict[str, dict[str, Any]]) -> None:
        lock = self._get_lock()
        async with lock:
            await asyncio.to_thread(
                self._write_manifest, self._manifest_path(), records
            )
            await asyncio.to_thread(self._rebuild_index, records)

    def _read_manifest(self, manifest: Path) -> dict[str, dict[str, Any]]:
        with open(manifest, encoding="utf-8") as fh:
            raw = json.load(fh)
        return {record["id"]: record for record in raw.get("records", [])}

    def _write_manifest(
        self, manifest: Path, records: dict[str, dict[str, Any]]
    ) -> None:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": sorted(records.values(), key=lambda item: item["id"])}
        with open(manifest, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    def _rebuild_index(self, records: dict[str, dict[str, Any]]) -> None:
        index_path = self._index_path()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self._cleanup_index_files(index_path)
        if not records:
            return

        builder = LeannBuilder(
            backend_name=self.settings.backend_name,
            embedding_model=self.settings.embedding_model,
            embedding_mode=self.settings.embedding_mode,
            embedding_options=self.settings.embedding_options,
            dimensions=len(next(iter(records.values()))["vector"]),
            is_recompute=False,
            is_compact=False,
            distance_metric=self.settings.distance_metric,
        )
        ids: list[str] = []
        vectors: list[list[float]] = []
        for record in sorted(records.values(), key=lambda item: item["id"]):
            ids.append(record["id"])
            vectors.append(record["vector"])
            builder.add_text(
                record["text"],
                metadata={"id": record["id"], "document": record["document"]},
            )
        embeddings = np.ascontiguousarray(vectors, dtype=np.float32)
        if self.settings.distance_metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms

        self._build_index_from_arrays(builder, index_path, ids, embeddings)

    def _cleanup_index_files(self, index_path: Path) -> None:
        for suffix in (".meta.json", ".passages.jsonl", ".passages.idx", ".ids.txt"):
            target = index_path.with_name(f"{index_path.name}{suffix}")
            if target.exists():
                target.unlink()
        index_file = index_path.with_suffix(".index")
        if index_file.exists():
            index_file.unlink()

    def _build_index_from_arrays(
        self,
        builder: LeannBuilder,
        index_path: Path,
        ids: Sequence[str],
        embeddings: np.ndarray,
    ) -> None:
        index_dir = index_path.parent
        index_name = index_path.name
        passages_file = index_dir / f"{index_name}.passages.jsonl"
        offset_file = index_dir / f"{index_name}.passages.idx"
        offset_map: dict[str, int] = {}

        with open(passages_file, "w", encoding="utf-8") as fh:
            for chunk in builder.chunks:
                offset = fh.tell()
                json.dump(
                    {
                        "id": chunk["id"],
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                    },
                    fh,
                    ensure_ascii=False,
                )
                fh.write("\n")
                offset_map[chunk["id"]] = offset

        with open(offset_file, "wb") as fh:
            pickle.dump(offset_map, fh)

        idmap_file = index_dir / f"{index_path.stem}.ids.txt"
        with open(idmap_file, "w", encoding="utf-8") as fh:
            for record_id in ids:
                fh.write(f"{record_id}\n")

        current_backend_kwargs = {
            **builder.backend_kwargs,
            "dimensions": embeddings.shape[1],
        }
        builder_instance = builder.backend_factory.builder(**current_backend_kwargs)
        builder_instance.build(embeddings, list(ids), str(index_path))

        meta_path = index_dir / f"{index_name}.meta.json"
        meta_data = {
            "version": "1.0",
            "backend_name": builder.backend_name,
            "embedding_model": builder.embedding_model,
            "dimensions": int(embeddings.shape[1]),
            "backend_kwargs": builder.backend_kwargs,
            "embedding_mode": builder.embedding_mode,
            "passage_sources": [
                {
                    "type": "jsonl",
                    "path": passages_file.name,
                    "index_path": offset_file.name,
                    "path_relative": passages_file.name,
                    "index_path_relative": offset_file.name,
                }
            ],
        }
        if builder.embedding_options:
            meta_data["embedding_options"] = builder.embedding_options
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta_data, fh, ensure_ascii=False, indent=2)

    def _create_searcher(self) -> LeannSearcher:
        return LeannSearcher(
            index_path=str(self._index_path()),
            enable_warmup=False,
            recompute_embeddings=False,
        )

    def _get_passage_via_id_map(
        self, searcher: LeannSearcher, label: Any
    ) -> dict[str, Any] | None:
        try:
            index = int(label)
        except (TypeError, ValueError):
            return None
        ids = self._read_id_map()
        if index < 0 or index >= len(ids):
            return None
        try:
            return searcher.passage_manager.get_passage(ids[index])
        except KeyError:
            return None

    def _build_text(self, document: dict[str, Any]) -> str:
        parts: list[str] = []
        for field_name in self.config.text_fields:
            value = document.get(field_name)
            if value is None:
                continue
            if isinstance(value, list):
                parts.extend(str(item) for item in value if item not in {None, ""})
            elif value not in {"", None}:
                parts.append(str(value))
        return "\n".join(parts).strip()

    def _matches_filters(
        self, document: dict[str, Any], filters: dict[str, Any]
    ) -> bool:
        for key, value in (filters or {}).items():
            if value is None:
                continue
            if key == "user_id":
                doc_value = document.get("user_id", "")
                if value == MAGIC_ALL:
                    continue
                if (value or "") != (doc_value or ""):
                    return False
            elif key == "group_ids":
                if value is None:
                    continue
                group_id = document.get("group_id", "")
                if len(value) > 0 and group_id not in value:
                    return False
            elif key == "group_id":
                if value == MAGIC_ALL:
                    continue
                if (value or "") != (document.get("group_id", "") or ""):
                    return False
            elif key == "parent_id" and value and document.get("parent_id") != value:
                return False
            elif (
                key == "parent_type" and value and document.get("parent_type") != value
            ):
                return False
            elif key == "sender_id" and value:
                sender_ids = document.get("sender_ids") or []
                if value not in sender_ids:
                    return False
            elif key == "start_time":
                timestamp = self._coerce_timestamp(document.get("timestamp"))
                start_ts = self._coerce_timestamp(value)
                if timestamp is None or start_ts is None or timestamp < start_ts:
                    return False
            elif key == "end_time":
                timestamp = self._coerce_timestamp(document.get("timestamp"))
                end_ts = self._coerce_timestamp(value)
                if timestamp is None or end_ts is None or timestamp > end_ts:
                    return False
        return True

    def _coerce_timestamp(self, value: Any) -> int | None:
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return int(value.timestamp() * 1000)
        if isinstance(value, (int, float)):
            return int(value if value > 10_000_000_000 else value * 1000)
        if isinstance(value, str):
            try:
                return int(datetime.fromisoformat(value).timestamp() * 1000)
            except ValueError:
                return None
        return None

    def _manifest_path(self) -> Path:
        return self._tenant_dir() / f"{self.config.collection_name}.manifest.json"

    def _index_path(self) -> Path:
        return self._tenant_dir() / f"{self.config.collection_name}.leann"

    def _id_map_path(self) -> Path:
        return self._tenant_dir() / f"{self.config.collection_name}.ids.txt"

    def _tenant_dir(self) -> Path:
        tenant_prefix = get_current_tenant_id() or get_base_resource_prefix()
        return self.settings.storage_path / tenant_prefix

    def _read_id_map(self) -> list[str]:
        path = self._id_map_path()
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip()]

    def _get_lock(self) -> asyncio.Lock:
        key = str(self._index_path())
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock
