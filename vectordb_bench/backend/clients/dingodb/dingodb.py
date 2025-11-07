"""VectorDBBench client wrapper for DingoDB."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

import numpy as np
from dingodb import SDKClient, SDKVectorDingoDB
from dingodb.common.vector_rep import ScalarColumn, ScalarSchema, ScalarType

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import DingoDBCaseConfig, DingoDBConfigDict

log = logging.getLogger(__name__)


class DingoDB(VectorDB):
    """DingoDB implementation of the VectorDB interface."""

    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: DingoDBConfigDict,
        db_case_config: DingoDBCaseConfig | None,
        collection_name: str = "",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs: Any,
    ):
        self.name = "DingoDB"
        self.dim = dim
        self.db_config = db_config
        self.index_name = db_config["index_name"]
        self.addrs = db_config["addrs"]
        self.case_config = db_case_config or DingoDBCaseConfig()
        self.with_scalar_labels = with_scalar_labels

        self._sdk_client: SDKClient | None = None
        self._vector_client: SDKVectorDingoDB | None = None
        self._search_params: dict[str, Any] | None = None
        self._pre_filter: bool = False
        self._logged_search_schema: bool = False
        self._sample_logs_remaining: int = 5

        if drop_old:
            self._recreate_index()
        else:
            self._ensure_index()

    def need_normalize_cosine(self) -> bool:
        """Normalize embeddings for COSINE to align with IP/L2 backends.

        Normalizing ensures ranking equivalence for COSINE when the backend
        uses IP or L2 by default, improving recall without requiring backend
        metric configuration.
        """
        return True

    def _recreate_index(self) -> None:
        client, vector_client = self._create_connection()
        try:
            try:
                vector_client.delete_index(self.index_name)
            except Exception as exc:  # noqa: BLE001
                log.debug("Delete index failed during recreation, ignoring error: %s", exc)
            # Always create with scalar schema to enable filtering on fields
            schema = self._build_scalar_schema()
            try:
                vector_client.create_index_with_schema(
                    self.index_name,
                    self.dim,
                    schema,
                    index_type="hnsw",
                    auto_id=False,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("create_index_with_schema failed, falling back to create_index: %s", exc)
                vector_client.create_index(self.index_name, self.dim, index_type="hnsw", auto_id=False)
        finally:
            self._close_connection(client)

    def _ensure_index(self) -> None:
        client, vector_client = self._create_connection()
        try:
            try:
                # Prefer creating index with scalar schema so that meta_expr and range filters work
                schema = self._build_scalar_schema()
                vector_client.create_index_with_schema(
                    self.index_name,
                    self.dim,
                    schema,
                    index_type="hnsw",
                    auto_id=False,
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("Ensure index with schema failed (%s), trying basic create_index", exc)
                try:
                    vector_client.create_index(self.index_name, self.dim, index_type="hnsw", auto_id=False)
                except Exception as exc2:  # noqa: BLE001
                    log.debug("Ensure index skipped, create_index raised %s", exc2)
        finally:
            self._close_connection(client)

    def _build_scalar_schema(self) -> ScalarSchema:
        """Build scalar schema for DingoDB index.

        - Always include an INT64 'id' column to support numeric filters (e.g., ge('id', ...)).
        - Optionally include a STRING 'labels' column when scalar-label cases are enabled.
        """
        schema = ScalarSchema()
        # Numeric id column for range/int filters and mapping to dataset id
        schema.add_scalar_column(ScalarColumn("id", ScalarType.INT64, True))
        # String label column for equality filters
        if self.with_scalar_labels:
            schema.add_scalar_column(ScalarColumn("labels", ScalarType.STRING, True))
        return schema

    @staticmethod
    def _close_connection(client: SDKClient | None) -> None:
        if client is None:
            return
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                log.debug("Ignoring exception while closing DingoDB client", exc_info=True)

    def _create_connection(self) -> tuple[SDKClient, SDKVectorDingoDB]:
        client = SDKClient(self.addrs)
        vector_client = SDKVectorDingoDB(client)
        return client, vector_client

    @contextmanager
    def init(self) -> Iterable[None]:
        client, vector_client = self._create_connection()
        self._sdk_client = client
        self._vector_client = vector_client
        try:
            yield
        finally:
            self._vector_client = None
            self._search_params = None
            self._close_connection(client)
            self._sdk_client = None

    def _ensure_client(self) -> SDKVectorDingoDB:
        if self._vector_client is None:
            msg = "DingoDB client has not been initialised. Use the context manager returned by init()."
            raise RuntimeError(msg)
        return self._vector_client

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        client = self._ensure_client()

        if self.with_scalar_labels and labels_data is None:
            msg = "labels_data must be provided when with_scalar_labels is True"
            raise ValueError(msg)

        if len(embeddings) != len(metadata):
            msg = "Embeddings and metadata length mismatch"
            raise ValueError(msg)

        scalar_payloads = []
        for idx, meta in enumerate(metadata):
            payload: dict[str, Any] = {"id": int(meta)}
            if self.with_scalar_labels:
                payload["labels"] = labels_data[idx]
            scalar_payloads.append(payload)

        try:
            vector_payloads = np.asarray(embeddings, dtype=np.float32).tolist()
            # DingoDB requires primary keys > 0; shift by +1 for PKs
            ids = [int(m) + 1 for m in metadata]
            client.vector_add(self.index_name, scalar_payloads, vector_payloads, ids)
            return len(metadata), None
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to insert data into DingoDB index %s: %s", self.index_name, exc)
            return 0, exc

    def _escape_value(self, value: str) -> str:
        return value.replace("'", "\\'")

    def prepare_filter(self, filters: Filter) -> None:
        log.debug("Preparing filter: %s", filters)
        if filters.type == FilterOp.NonFilter:
            self._search_params = None
            self._pre_filter = False
            return
        if filters.type == FilterOp.NumGE:
            field = getattr(filters, "int_field", "id")
            expr = f"ge('{field}',{int(filters.int_value)})"
            self._search_params = {"langchain_expr": expr}
            self._pre_filter = True
            log.debug("Using numeric filter (langchain_expr): %s", self._search_params)
            return
        if filters.type == FilterOp.StrEqual:
            # Prefer structured meta_expr for equality filtering
            label = filters.label_value
            field = getattr(filters, "label_field", "labels")
            self._search_params = {"meta_expr": {field: label}}
            self._pre_filter = True
            log.debug("Using equality filter (meta_expr): %s", self._search_params)
            return
        msg = f"Unsupported filter type for DingoDB: {filters.type}"
        raise ValueError(msg)

    def _extract_hits(self, raw_result: Any) -> list[Any]:
        if raw_result is None:
            return []
        if isinstance(raw_result, list):
            if not raw_result:
                return []
            first = raw_result[0]
            if isinstance(first, list):
                return first
            if isinstance(first, dict):
                # Many SDKs return a list per query where each item is a dict
                # containing the actual hits under a nested key (e.g., 'hits', 'vectors').
                inner = self._extract_hits(first)
                if inner:
                    return inner
                return raw_result
            if isinstance(first, (int, np.integer)):
                return raw_result
        if isinstance(raw_result, dict):
            # DingoDB SDK SearchResult to_dict() uses 'vectorWithDistances'
            if "vectorWithDistances" in raw_result:
                v = raw_result.get("vectorWithDistances")
                if isinstance(v, list):
                    return v
            for key in ("result", "results", "vectors", "hits", "data"):
                if key in raw_result:
                    return self._extract_hits(raw_result[key])
            # Fallback: if dict contains a direct list of ints under common names
            for key in ("ids", "id_list", "keys"):
                val = raw_result.get(key)
                if isinstance(val, list):
                    return val
        return []

    def _extract_id(self, hit: Any) -> int | None:
        if hit is None:
            return None
        if isinstance(hit, (int, np.integer)):
            # Assume primary key (shift -1 to dataset id)
            return int(hit) - 1
        if isinstance(hit, dict):
            # Prefer scalar_data mapping to original dataset id (no offset)
            def _parse_scalar_mapping(container: dict[str, Any]) -> dict[str, Any]:
                # direct mapping or per-key value dict with 'fields'
                mapping: dict[str, Any] = {}
                for k in ("id", "key", "labels", "label"):
                    if k in container:
                        v = container[k]
                        if isinstance(v, dict):
                            # DingoDB Python adapter emits {key: {fieldType: ..., fields: [val]}}
                            fields_list = v.get("fields")
                            if isinstance(fields_list, list) and fields_list:
                                v = fields_list[0]
                        mapping[k] = v
                # case 2: fields list form: {fields: [{key: 'id', value: {...}}]}
                fields = container.get("fields") or container.get("scalar_fields")
                if isinstance(fields, list):
                    for item in fields:
                        if not isinstance(item, dict):
                            continue
                        k = item.get("key") or item.get("name") or item.get("field")
                        v = item.get("value")
                        # unwrap typed values
                        if isinstance(v, dict):
                            for t in (
                                "intValue",
                                "longValue",
                                "int",
                                "long",
                                "int32Value",
                                "int64Value",
                                "stringValue",
                                "string",
                                "boolValue",
                                "bool",
                                "doubleValue",
                                "floatValue",
                                "double",
                                "float",
                            ):
                                if t in v:
                                    v = v[t]
                                    break
                        if k:
                            mapping[str(k)] = v
                return mapping

            for nested_key in (
                "scalar_data",
                "scalarData",
                "vector_with_scalar_data",
                "vectorWithScalarData",
            ):
                nested = hit.get(nested_key)
                if isinstance(nested, dict):
                    m = _parse_scalar_mapping(nested)
                    for key in ("id", "key"):
                        if key in m and m[key] is not None:
                            try:
                                return int(m[key])
                            except Exception:  # noqa: BLE001
                                continue

            # Treat remaining id-like fields as primary keys; adjust -1 to dataset id
            for key in (
                "id",
                "key",
                "vector_id",
                "vectorId",
                "primary_key",
                "primaryKey",
            ):
                if key in hit:
                    try:
                        return int(hit[key]) - 1
                    except Exception:  # noqa: BLE001
                        continue
        return None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        client = self._ensure_client()

        # Normalize query for cosine if needed (stored vectors are normalized by runner)
        q = query
        if self.need_normalize_cosine():
            q_np = np.asarray(query, dtype=np.float32)
            norm = float(np.linalg.norm(q_np))
            if norm > 0:
                q = (q_np / norm).tolist()
            else:
                q = q_np.tolist()

        search_kwargs: dict[str, Any] = {}
        # Merge base search params from case config (e.g., efSearch) with filter params
        base_params = {}
        try:
            base_params = self.case_config.search_param() or {}
        except Exception:  # noqa: BLE001
            base_params = {}
        merged_params: dict[str, Any] = {**base_params}
        if self._search_params:
            merged_params.update(self._search_params)
        if merged_params:
            search_kwargs["search_params"] = merged_params
        if self._pre_filter:
            search_kwargs["pre_filter"] = True

        try:
            response = client.vector_search(self.index_name, [q], k, **search_kwargs)
        except Exception as exc:  # noqa: BLE001
            log.warning("Vector search failed on index %s: %s", self.index_name, exc)
            raise

        hits = self._extract_hits(response)
        if not self._logged_search_schema:
            log.debug(
                "DingoDB search response schema sample: type=%s, size=%s, first=%s",
                type(response).__name__,
                len(response) if hasattr(response, "__len__") else "n/a",
                hits[0] if hits else None,
            )
            log.debug("Search kwargs: %s", search_kwargs)
            self._logged_search_schema = True
        ids: list[int] = []
        for hit in hits:
            candidate = self._extract_id(hit)
            if candidate is not None:
                ids.append(candidate)
        if self._sample_logs_remaining > 0:
            # log a small sample to aid debugging recall issues (limited times)
            log.debug("Extracted top ids sample: %s", ids[:10])
            self._sample_logs_remaining -= 1
        return ids

    def optimize(self, data_size: int | None = None) -> None:
        # DingoDB SDK does not expose a separate optimize/build step.
        return
